# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Body,
    SpiceManager,
    StateArray,
    Frame,
    CelestialBody
)

import scarabaeus.utils.NumpyWrapper as np
from math import factorial
import json
import warnings


#######################################################################
class SphericalHarmonicsGravity(DynamicModel):
    """ Models the gravitational acceleration due to a body's non-spherical mass distribution
    using spherical harmonics.

    The gravitational potential is expanded as:

    .. math::

        U = \\frac{\\mu}{r} \\sum_{n=0}^{N} \\sum_{m=0}^{n}
            \\left(\\frac{R_e}{r}\\right)^n
            \\bar{P}_{nm}(\\sin\\phi)
            \\left[
                \\bar{C}_{nm} \\cos(m\\lambda)
                + \\bar{S}_{nm} \\sin(m\\lambda)
            \\right]

    where :math:`\\mu` is the gravitational parameter, :math:`R_e` is the
    reference radius, :math:`r`, :math:`\\phi`, :math:`\\lambda` are the
    spherical coordinates of the spacecraft in the body-fixed frame,
    :math:`\\bar{P}_{nm}` are the normalized associated Legendre polynomials,
    and :math:`\\bar{C}_{nm}`, :math:`\\bar{S}_{nm}` are the normalized
    spherical harmonic coefficients.  The acceleration is
    :math:`\\mathbf{a} = \\nabla U`, computed in the body-fixed frame via
    the Pines/Eckman recurrence and rotated to the propagation frame.

    The body-fixed frame orientation is defined by the IAU pole model:

    .. math::

        \\alpha_0,\\quad \\delta_0,\\quad W

    where :math:`\\alpha_0` (right ascension of the pole), :math:`\\delta_0`
    (declination of the pole), and :math:`W` (prime meridian angle) are
    secular polynomials in time whose coefficients come from the PCK kernel.
    All three angles can be estimated as solve-for parameters by including
    ``("pole_ra", spice_id)``, ``("pole_dec", spice_id)``, or
    ``("pole_pm", spice_id)`` entries in the state vector.  When estimated,
    the rotation matrix :math:`\\mathbf{R}` (body-fixed → inertial) is rebuilt
    analytically from the estimated angles rather than from SPICE, so partial
    derivatives with respect to the pole parameters are available.

    The coefficients :math:`\\bar{C}_{nm}` and :math:`\\bar{S}_{nm}` as well
    as :math:`\\mu` itself may also be estimated via ``("cnm_n_m", spice_id)``,
    ``("snm_n_m", spice_id)``, and ``("gm", spice_id)`` state entries
    respectively.

    Parameters
    ----------
    sph_harm_order : int
        The maximum order of spherical harmonics to use.

    sph_harm_cs_file : str
        Path to the JSON file containing spherical harmonic coefficients.

    sph_harm_body : dict
        Dictionary containing body-specific parameters, including:
        * ``'spice_name'`` (str): The SPICE name of the body.
        * ``'ref_name'`` (str): The reference frame in which harmonics are computed.

    state_vector : StateArray
        The state vector of the main spacecraft, containing:
        * All dynamic and static parameters considered in the simulation (e.g., position, velocity, estimated parameters).
        * The reference frame in which the state is defined.
        * The origin body relative to which the state is expressed.
        * The epoch corresponding to the state.

    sph_harm_norm_flag : bool, optional
        Flag indicating whether to normalize coefficients. Defaults to ``True``.

    n_degree_max : int, optional
        Maximum degree for harmonics. Defaults to ``100``.

    base_frame : Frame, optional
        Frame of the propagator. If ``base_frame = None``, defaults to the J2000 frame.

    See Also
    --------
    scarabaeus.StateArray : Represents state vectors with origins and reference frames.

    Notes
    -----
    The spherical harmonic model accounts for deviations from a perfect sphere, ypically used for planetary bodies.

    Coefficients can be loaded in normalized or unnormalized form.

    Uses the gravitational parameter and reference radius for calculations.

    References
    ----------
    Kaula, W. M. (1966). Theory of Satellite Geodesy.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        sph_harm_order: int,
        sph_harm_cs_file: str,
        sph_harm_body,
        state_vector: StateArray,
        sph_harm_norm_flag: bool = True,
        n_degree_max: int = 100,
        base_frame: Frame = None,
    ):
        # Initialize the class attributes
        self._n_degree_max = n_degree_max
        self._order = sph_harm_order
        self._cs_file = sph_harm_cs_file
        self._body = sph_harm_body
        self._origin = state_vector.origin
        self._normalized = sph_harm_norm_flag
        self._origin_name = state_vector.origin.name
        self.ref_frame = base_frame

        # Cache the integer SPICE id for Cnm/Snm/pole state lookups.
        # Prefer the "SPICE_ID" key in the body dict; fall back to a SPICE name lookup.
        if isinstance(self._body, dict):
            spice_id_direct = self._body.get("SPICE_ID", None)
            if spice_id_direct is not None:
                self._sph_body_spice_id = int(spice_id_direct)
            else:
                spice_name_lookup = self._body.get("spice_name", None)
                try:
                    self._sph_body_spice_id = (
                        SpiceManager.name_to_ID(spice_name_lookup)
                        if spice_name_lookup
                        else None
                    )
                except Exception:
                    self._sph_body_spice_id = None
        else:
            # CelestialBody object
            try:
                self._sph_body_spice_id = getattr(self._body, "spice_id", None)
            except Exception:
                self._sph_body_spice_id = None

        # Check that the SPH body has a valid SPICE name and warn if it differs from the propagation origin
        # sph_name = self._body.get("spice_name", None)
        if isinstance(sph_harm_body, dict):
            sph_name = self._body.get("spice_name", None)
        else:
            sph_name = self._body.name
        if sph_name is None:
            raise ValueError("sph_harm_body must contain 'spice_name'.")
        if self._origin.name != sph_name:
            warnings.warn(
                f"SPH gravity body is '{sph_name}', but propagation origin is "
                f"'{self._origin.name}'. The model will compute the relative vector "
                f"r_eval/sph = r_eval/origin - r_sph/origin using SPICE.",
                RuntimeWarning,
            )

        # Cache IAU secular pole constants (deg / deg·century / deg·day) from SPICE PCK.
        # Used to rebuild R when pole_ra / pole_dec / pole_pm are estimated parameters.
        # Only the secular (zeroth- and first-order) terms are stored; higher-order nutation
        # terms are small and cancel in the correction computation.
        self._iau_ra_coeffs = None
        self._iau_dec_coeffs = None
        self._iau_pm_coeffs = None
        try:
            import spiceypy as _spice

            _sn = sph_name
            self._iau_ra_coeffs = _spice.bodvrd(
                _sn, "POLE_RA", 3
            )  # [RA0, RA1, RA2]  deg & deg/century
            self._iau_dec_coeffs = _spice.bodvrd(
                _sn, "POLE_DEC", 3
            )  # [DEC0, DEC1, DEC2]
            self._iau_pm_coeffs = _spice.bodvrd(
                _sn, "PM", 3
            )  # [W0, Wdot, Wddot] deg & deg/day
        except Exception:
            pass

        # Load spherical harmonics coefficients from the JSON file
        self._load_coefficients()

        # Normalize coefficients if needed
        if not self._input_normalized and self._normalized:
            self._normalize_coeffs()

    def _load_coefficients(self):
        """
        Loads spherical harmonics coefficients from the JSON file.
        """
        with open(self._cs_file, mode="r") as file:
            data = json.load(file)

            # Parse reference radius and gravitational parameter
            self._r_ref = data["ReferenceRadius"]  # Assumed in km
            self._mu = data["GravitationalParameter"]  # Assumed in km^3/sec^2

            # Determine if coefficients are normalized in the file
            self._input_normalized = data["Normalized"]

            # Determine maximum degree in the file and adjust order if needed
            max_degree_file = data["MaxDegree"]
            if self._order > max_degree_file:
                warnings.warn(
                    f"Spherical harmonics data file for {data['BodyName']} only goes to degree {max_degree_file}."
                )
                self._order = max_degree_file

            # Initialize coefficient arrays
            self._Cnm = np.zeros((self._order + 1, self._order + 1))
            self._Snm = np.zeros((self._order + 1, self._order + 1))

            # Load coefficients into arrays
            for coeff in data["Coefficients"]:
                if coeff["degree"] <= self._order:
                    indices = coeff["degree"], coeff["order"]
                    np.set(self._Cnm, indices, coeff["Cnm"])
                    np.set(self._Snm, indices, coeff["Snm"])

            # Ensure Cnm[0,0] is non-zero
            if self._Cnm[0, 0] == 0.0:
                indices = 0, 0
                np.set(self._Cnm, indices, 1.0)
                warnings.warn(
                    "C_nm(0,0) has been forced to 1 with respect to specified value in the JSON file."
                )

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def order(self) -> int:
        """
        The maximum order of the spherical harmonic expansion.
        """
        return self._order

    @property
    def n_degree_max(self) -> int:
        """
        The maximum degree for spherical harmonics calculations.
        """
        return self._n_degree_max

    @property
    def normalized(self) -> bool:
        """
        Indicates whether the spherical harmonic coefficients are normalized.
        """
        return self._normalized

    @property
    def cs_file(self) -> str:
        """
        The file path of the spherical harmonic coefficient JSON file.
        """
        return self._cs_file

    @property
    def SPH_body(self) -> Body:
        """
        The body object representing the primary body for spherical harmonics calculations.
        """
        return self._body

    @property
    def ref_frame(self) -> str:
        """
        The reference frame associated with the input state vector, used for acceleration
        and partial derivative computations.
        """
        return self._ref_frame

    @ref_frame.setter
    def ref_frame(self, input_val):
        if input_val is None:
            input_val = Frame("J2000")

        self._ref_frame = input_val

    @property
    def Cnm(self) -> np.ndarray:
        """
        The matrix of cosine spherical harmonic coefficients.
        """
        return self._Cnm

    @Cnm.setter
    def Cnm(self, input_val):

        self._Cnm = input_val

    @property
    def Snm(self) -> np.ndarray:
        """
        The matrix of sine spherical harmonic coefficients.
        """
        return self._Snm

    @Snm.setter
    def Snm(self, input_val):

        self._Snm = input_val

    @property
    def bnm_ext_real(self) -> np.ndarray:
        """
        The real part of the B_nm exterior coefficients. Unitless.
        """
        return self._bnm_ext_real

    @property
    def bnm_ext_imag(self) -> np.ndarray:
        """
        The imaginary part of the B_nm exterior coefficients. Unitless.
        """
        return self._bnm_ext_imag

    @property
    def mu(self) -> float:
        """
        The gravitational parameter of the spherical harmonics body. Assumed
        in units of :math:`\\frac{km^3}{s^2}`.
        """
        return self._mu

    @property
    def r_ref(self) -> float:
        """
        The reference radius used in the spherical harmonics model. Assumed
        in units of :math:`km`.
        """
        return self._r_ref

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _acceleration_transform_rf(
        self, acceleration_bf: np.ndarray, epoch: float, R_b2rf=None
    ) -> np.ndarray:
        """
        Transforms an acceleration vector from the body-fixed frame to the reference frame.

        Parameters
        ----------
        acceleration_bf : np.ndarray
            3x1 acceleration vector in the body-fixed frame (in km/s^2).

        epoch : float
            Epoch at which the transformation is computed.

        R_b2rf : np.ndarray, optional
            Pre-computed body-fixed -> reference-frame DCM. When supplied (e.g. when
            pole parameters are being estimated) this overrides the SPICE lookup.

        Returns
        -------
        acceleration_rf : np.ndarray
            3x1 acceleration vector in the reference frame (in km/s^2).
        """
        body_fixed_to_ext_DCM = (
            R_b2rf
            if R_b2rf is not None
            else SpiceManager.get_xfrm(
                self._sph_body_fixed_frame(), self._ref_frame.name, epoch
            )
        )
        acceleration_rf = body_fixed_to_ext_DCM @ acceleration_bf.reshape((3, 1))
        return acceleration_rf.flatten()

    def _sph_body_fixed_frame(self):
        """Body-fixed (rotating) frame name of the SPH gravity body.

        ``self._body`` may be either a ``dict`` (frame under the ``"ref_name"``
        key) or a ``CelestialBody`` (frame is the ``.base_frame`` attribute).
        """
        if isinstance(self._body, dict):
            return self._body["ref_name"]
        return self._body.base_frame

    def _jacobian_C_transform_rf(
        self, jacobian_C_bf: np.ndarray, epoch: float, R_b2rf=None
    ) -> np.ndarray:
        """
        Transforms the Jacobian matrix from the body-fixed frame to the reference frame.

        Parameters
        ----------
        jacobian_C_bf : np.ndarray
            The Jacobian matrix in the body-fixed frame with size (order+1)x(order+1).
            Expressed in units of :math:`\\frac{km}{s^2}`.

        epoch : float
            The epoch time at which the transformation is computed.

        R_b2rf : np.ndarray, optional
            Pre-computed body-fixed -> reference-frame DCM (overrides SPICE when supplied).

        Returns
        -------
        jacobian_C_rf: np.ndarray
            The Jacobian matrix in the reference frame
            (size: (order+1)x(order+1), typical units: km/s^2).
        """
        bodyFixed_to_ext_DCM = (
            R_b2rf
            if R_b2rf is not None
            else SpiceManager.get_xfrm(
                self._sph_body_fixed_frame(), self._ref_frame.name, epoch
            )
        )
        jacobian_C_rf = np.tensordot(bodyFixed_to_ext_DCM, jacobian_C_bf, axes=(1, 0))
        return jacobian_C_rf

    def _jacobian_pos_transform_rf(
        self, jacobian_bf: np.ndarray, epoch: float, R_b2rf=None
    ) -> np.ndarray:
        """
        Transforms the Jacobian matrix from the body-fixed frame to the reference frame (unit-free).

        Parameters
        ----------
        jacobian_bf : np.ndarray
            The 3x3 Jacobian matrix in the body-fixed frame (float only, no units).

        epoch : float
            The epoch at which the transformation is computed.

        R_b2rf : np.ndarray, optional
            Pre-computed body-fixed -> reference-frame DCM (overrides SPICE when supplied).

        Returns
        -------
        np.ndarray
            The 3x3 Jacobian matrix in the reference frame (unit-free).
        """
        bodyFixed_to_ext_DCM = (
            R_b2rf
            if R_b2rf is not None
            else SpiceManager.get_xfrm(
                self._sph_body_fixed_frame(), self._ref_frame.name, epoch
            )
        )
        jacobian_rf = bodyFixed_to_ext_DCM @ jacobian_bf @ bodyFixed_to_ext_DCM.T
        return jacobian_rf

    def _rf_transform(
        self,
        position: np.ndarray,
        epoch: float,
        R_b2rf=None,
    ) -> np.ndarray:
        """
        position: 3x1 vector of the evaluation body (self._eval_body)
                w.r.t. the propagation origin (self._origin),
                expressed in self._ref_frame, in km.

        R_b2rf : np.ndarray, optional
            Pre-computed body-fixed -> reference-frame DCM. When supplied the
            inverse (R_rf2b = R_b2rf.T) is used instead of the SPICE lookup.
        """

        # Position of the SPH body w.r.t. the same origin and frame
        if isinstance(self.SPH_body, dict): spice_name = self.SPH_body["spice_name"]
        else                              : spice_name = self.SPH_body.name
        r_sphBody_origin = SpiceManager.get_pos(
            spice_name,
            epoch,
            self._ref_frame,
            self._origin.name,
        ).values  # km

        r_eval_sph = position - r_sphBody_origin

        # Rotate into SPH body-fixed frame
        if R_b2rf is not None:
            ext_to_bodyFixed_DCM = R_b2rf.T  # R_b2rf is orthogonal
        else:
            ext_to_bodyFixed_DCM = SpiceManager.get_xfrm(
                frame_from=self._ref_frame.name,
                frame_to=self._sph_body_fixed_frame(),
                epoch=epoch,
            )

        r_eval_sph_bf = ext_to_bodyFixed_DCM @ r_eval_sph
        return r_eval_sph_bf

    def _compute_bnm(
        self, position: np.ndarray, epoch: float, R_b2rf=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the B_nm coefficients required for spherical harmonics acceleration computation - using raw ndarray

        Evaluates the real and imaginary components of the B_nm terms based on
        the spacecraft's position relative to the SPH body, considering whether the coefficients
        are normalized or not.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector in the original reference frame (in km).

        epoch : float
            Epoch at which the transformation is performed.

        Returns
        -------
        B_nm coeffs : tuple
            A tuple with indices containing the following:
            tuple[np.ndarray, np.ndarray]
            Real and imaginary B_nm coefficient matrices, shape (n_degree_max+3, n_degree_max+3).
        """
        # Position in body-fixed frame (uses estimated rotation when provided)
        r_sc_sph = self._rf_transform(position, epoch, R_b2rf)
        x, y, z = r_sc_sph[0], r_sc_sph[1], r_sc_sph[2]
        r_sat = np.linalg.norm(r_sc_sph)

        # Precompute terms
        inv_r_sat = 1.0 / r_sat
        x_by_r_sat = x * inv_r_sat
        y_by_r_sat = y * inv_r_sat
        z_by_r_sat = z * inv_r_sat
        r_ref_by_r_sat = self.r_ref * inv_r_sat
        r_ref_sq_by_r_sat_sq = r_ref_by_r_sat * r_ref_by_r_sat

        # Allocate N
        N = self.order + 3
        bnm_ext_real = []
        bnm_ext_imag = []

        for i in range(N):
            bnm_ext_real.append([])
            bnm_ext_imag.append([])

        if not self.normalized:
            # This branch indexes with tuple subscripts, so it needs (N, N) arrays
            # rather than the lists-of-lists allocated above.
            bnm_ext_real = np.zeros((N, N))
            bnm_ext_imag = np.zeros((N, N))
            for m in range(self.order + 3):
                for n in range(m, self.order + 3):
                    if m == n:
                        if m == 0:
                            bnm_ext_real[0, 0] = r_ref_by_r_sat
                            bnm_ext_imag[0, 0] = 0.0
                        else:
                            real_val = (
                                (2 * n - 1)
                                * r_ref_by_r_sat
                                * (
                                    x_by_r_sat * bnm_ext_real[n - 1, n - 1]
                                    - y_by_r_sat * bnm_ext_imag[n - 1, n - 1]
                                )
                            )
                            imag_val = (
                                (2 * n - 1)
                                * r_ref_by_r_sat
                                * (
                                    y_by_r_sat * bnm_ext_real[n - 1, n - 1]
                                    + x_by_r_sat * bnm_ext_imag[n - 1, n - 1]
                                )
                            )
                            bnm_ext_real[n, n] = real_val
                            bnm_ext_imag[n, n] = imag_val
                    else:
                        if n >= 2:
                            real_val = (
                                (2 * n - 1) / (n - m)
                            ) * r_ref_by_r_sat * z_by_r_sat * bnm_ext_real[n - 1, m] - (
                                (n + m - 1) / (n - m)
                            ) * r_ref_sq_by_r_sat_sq * bnm_ext_real[
                                n - 2, m
                            ]
                            imag_val = (
                                (2 * n - 1) / (n - m)
                            ) * r_ref_by_r_sat * z_by_r_sat * bnm_ext_imag[n - 1, m] - (
                                (n + m - 1) / (n - m)
                            ) * r_ref_sq_by_r_sat_sq * bnm_ext_imag[
                                n - 2, m
                            ]
                        else:
                            real_val = (
                                ((2 * n - 1) / (n - m))
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_real[n - 1, m]
                            )
                            imag_val = (
                                ((2 * n - 1) / (n - m))
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_imag[n - 1, m]
                            )

                        bnm_ext_real[n, m] = real_val
                        bnm_ext_imag[n, m] = imag_val

        else:
            for m in range(self.order + 3):
                # bnm_ext_real.append([])
                # bnm_ext_imag.append([])

                for n in range(0, m):
                    bnm_ext_real[n].append(0.0)
                    bnm_ext_imag[n].append(0.0)

                for n in range(m, self.order + 3):
                    if m == n:
                        if m == 0:
                            bnm_ext_real[n].append(r_ref_by_r_sat)
                            bnm_ext_imag[n].append(0.0)
                            # np.set(bnm_ext_real,[0, 0],r_ref_by_r_sat)
                            # np.set(bnm_ext_imag,[0, 0],0.0)
                        else:
                            delta_1_n = SphericalHarmonicsGravity._kronecker_delta(1, n)
                            norm_factor = np.sqrt(
                                (1.0 + delta_1_n) * (2.0 * n + 1.0) / (2.0 * n)
                            )
                            real_val = (
                                norm_factor
                                * r_ref_by_r_sat
                                * (
                                    x_by_r_sat * bnm_ext_real[n - 1][n - 1]
                                    - y_by_r_sat * bnm_ext_imag[n - 1][n - 1]
                                )
                            )
                            imag_val = (
                                norm_factor
                                * r_ref_by_r_sat
                                * (
                                    y_by_r_sat * bnm_ext_real[n - 1][n - 1]
                                    + x_by_r_sat * bnm_ext_imag[n - 1][n - 1]
                                )
                            )
                            bnm_ext_real[n].append(real_val)
                            bnm_ext_imag[n].append(imag_val)
                            # np.set(bnm_ext_real,[n, n],real_val)
                            # bnm_ext_real[n, n] = real_val
                            # np.set(bnm_ext_imag,[n, n],imag_val)
                    else:
                        sqrt_term_1 = np.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
                        if n >= 2:
                            sqrt_term_2 = np.sqrt(
                                (2.0 * n + 1.0)
                                * ((n - 1.0) * (n - 1.0) - m * m)
                                / ((2.0 * n - 3.0) * (n * n - m * m))
                            )
                            real_val = (
                                sqrt_term_1
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_real[n - 1][m]
                                - sqrt_term_2
                                * r_ref_sq_by_r_sat_sq
                                * bnm_ext_real[n - 2][m]
                            )
                            imag_val = (
                                sqrt_term_1
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_imag[n - 1][m]
                                - sqrt_term_2
                                * r_ref_sq_by_r_sat_sq
                                * bnm_ext_imag[n - 2][m]
                            )
                        else:
                            real_val = (
                                sqrt_term_1
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_real[n - 1][m]
                            )
                            imag_val = (
                                sqrt_term_1
                                * r_ref_by_r_sat
                                * z_by_r_sat
                                * bnm_ext_imag[n - 1][m]
                            )

                        bnm_ext_real[n].append(real_val)
                        bnm_ext_imag[n].append(imag_val)
                        # np.set(bnm_ext_real,[n, m],real_val)
                        # np.set(bnm_ext_imag,[n, m],imag_val)

        bnm_ext_real = np.array(bnm_ext_real)
        bnm_ext_imag = np.array(bnm_ext_imag)

        return bnm_ext_real, bnm_ext_imag

    def _apply_cs_from_state(self, current_state: dict):
        """
        Return (Cnm, Snm) arrays updated with any estimated corrections present
        in ``current_state``. Returns the originals unchanged when no corrections
        are found (no copy overhead).
        """
        if self._sph_body_spice_id is None or current_state is None:
            return self._Cnm, self._Snm

        Cnm, Snm = self._Cnm, self._Snm
        modified = False
        for nn in range(self._order + 1):
            for mm in range(nn + 1):
                c_key = (f"cnm_{nn}_{mm}", self._sph_body_spice_id)
                s_key = (f"snm_{nn}_{mm}", self._sph_body_spice_id)
                has_c = c_key in current_state
                has_s = s_key in current_state and mm > 0
                if has_c or has_s:
                    if not modified:
                        Cnm = self._Cnm.copy()
                        Snm = self._Snm.copy()
                        modified = True
                    if has_c:
                        val = current_state[c_key]
                        Cnm[nn, mm] = float(
                            val.values if hasattr(val, "values") else val
                        )
                    if has_s:
                        val = current_state[s_key]
                        Snm[nn, mm] = float(
                            val.values if hasattr(val, "values") else val
                        )
        return Cnm, Snm

    @staticmethod
    def _build_R_b2j_iau(ra_deg: float, dec_deg: float, w_deg: float):
        """
        Body-fixed -> J2000 rotation from IAU Euler angles (all in degrees).

        Uses the standard NAIF PCK 3-1-3 sequence:
          M = Rz(-(90+RA)) @ Rx(-(90-DEC)) @ Rz(-W)
        which matches SPICE ``pxform("IAU_<BODY>", "J2000", epoch)``.
        """
        import math as _m
        import numpy as _np

        phi = _m.radians(90.0 + ra_deg)  # = 90 + RA
        theta = _m.radians(90.0 - dec_deg)  # = 90 - DEC
        psi = _m.radians(w_deg)  # = W

        cp, sp = _m.cos(phi), _m.sin(phi)
        ct, st = _m.cos(theta), _m.sin(theta)
        cw, sw = _m.cos(psi), _m.sin(psi)

        # passive rotation matrices Ri(-angle)
        Rz_phi = _np.array([[cp, sp, 0.0], [-sp, cp, 0.0], [0.0, 0.0, 1.0]])
        Rx_theta = _np.array([[1.0, 0.0, 0.0], [0.0, ct, st], [0.0, -st, ct]])
        Rz_psi = _np.array([[cw, sw, 0.0], [-sw, cw, 0.0], [0.0, 0.0, 1.0]])
        return Rz_phi @ Rx_theta @ Rz_psi

    def _get_R_b2rf_effective(self, current_state: dict, epoch: float):
        """
        Return the body-fixed -> reference-frame rotation matrix.

        When no pole parameters are estimated the rotation comes entirely from
        SPICE pxform (full nutation/precession model).

        When ``pole_ra``, ``pole_dec``, or ``pole_pm`` are present in
        ``current_state`` the rotation is built in two steps:
          1. Body-fixed -> J2000  from the IAU 3-1-3 Euler formula at the
             estimated pole angles.  The secular PCK constants (bodvrd) fill in
             any angle that is not being estimated.
          2. J2000 -> reference frame  via SPICE pxform, so no hardcoded frame
             assumptions are made and the full precession model is used.

        Returns
        -------
        np.ndarray  (3, 3)  R such that v_rf = R @ v_body.
        """
        if isinstance(self._body, dict):
            body_frame = self._body["ref_name"]
        else:
            body_frame = self._body.base_frame
        ref_frame = self._ref_frame.name

        # No estimation — delegate entirely to SPICE pxform.
        if (
            self._iau_ra_coeffs is None
            or self._sph_body_spice_id is None
            or current_state is None
        ):
            return SpiceManager.get_xfrm(body_frame, ref_frame, epoch)

        ra_key = ("pole_ra", self._sph_body_spice_id)
        dec_key = ("pole_dec", self._sph_body_spice_id)
        pm_key = ("pole_pm", self._sph_body_spice_id)

        if not any(k in current_state for k in (ra_key, dec_key, pm_key)):
            return SpiceManager.get_xfrm(body_frame, ref_frame, epoch)

        # At least one pole parameter is estimated.
        # Secular IAU angles at this epoch (PCK bodvrd coefficients).
        # epoch is SPICE ET: seconds past J2000 TDB (verified by callers).
        import spiceypy as _spice
        import math as _m

        d = epoch / _spice.spd()  # Julian days from J2000 (spd() = 86400 s/day)
        T = d / 36525.0           # Julian centuries from J2000

        ra_deg = (self._iau_ra_coeffs[0]
                  + self._iau_ra_coeffs[1] * T
                  + self._iau_ra_coeffs[2] * T**2)
        dec_deg = (self._iau_dec_coeffs[0]
                   + self._iau_dec_coeffs[1] * T
                   + self._iau_dec_coeffs[2] * T**2)
        w_deg = (self._iau_pm_coeffs[0]
                 + self._iau_pm_coeffs[1] * d
                 + self._iau_pm_coeffs[2] * d**2)

        # Override with estimated state values where present.
        if ra_key in current_state:
            val = current_state[ra_key]
            ra_deg = _m.degrees(float(val.values if hasattr(val, "values") else val))
        if dec_key in current_state:
            val = current_state[dec_key]
            dec_deg = _m.degrees(float(val.values if hasattr(val, "values") else val))
        if pm_key in current_state:
            val = current_state[pm_key]
            pm_rate_deg_day = float(val.values if hasattr(val, "values") else val)
            w_deg = self._iau_pm_coeffs[0] + pm_rate_deg_day * d

        # Step 1: body-fixed -> J2000 from IAU Euler angles.
        R_b2j2000 = SphericalHarmonicsGravity._build_R_b2j_iau(ra_deg, dec_deg, w_deg)

        # Step 2: J2000 -> reference frame via SPICE pxform.
        if ref_frame.upper() == "J2000":
            return R_b2j2000
        return SpiceManager.get_xfrm("J2000", ref_frame, epoch) @ R_b2j2000

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch, body
    ) -> np.ndarray:
        """
        Computes the acceleration due to spherical harmonics gravity.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector of the spacecraft in the original reference frame, in kilometers.

        current_state : dict
            Dictionary containing the current state information.

        epoch : float
            Epoch at which the transformation is performed.

        body : None
            Not explicitly used but included for interface consistency.

        Returns
        -------
        acc_vec : np.ndarray
            The 3x1 acceleration vector due to spherical harmonics in the reference frame, in km/s^2. Expressed in the input reference frame
        """
        # Apply estimated GM from the state vector (if any)
        _mu_save = self._mu
        if self._sph_body_spice_id is not None and current_state is not None:
            _mu_key = ("gm", self._sph_body_spice_id)
            if _mu_key in current_state:
                _mu_val = current_state[_mu_key]
                self._mu = float(
                    _mu_val.values if hasattr(_mu_val, "values") else _mu_val
                )

        # Apply estimated Cnm/Snm corrections from the state vector (if any)
        _Cnm_save, _Snm_save = self._Cnm, self._Snm
        self._Cnm, self._Snm = self._apply_cs_from_state(current_state)

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)

        # Initialization acceleration
        x_ddot = 0.0
        y_ddot = 0.0
        z_ddot = 0.0

        # Compute external terms (body-fixed position uses estimated rotation)
        self._bnm_ext_real, self._bnm_ext_imag = self._compute_bnm(
            position, epoch, R_b2rf
        )

        # Calculate accelerations
        for nn in range(0, self.order + 1):
            n = nn

            for mm in range(0, nn + 1):
                m = mm
                delta_0_m = SphericalHarmonicsGravity._kronecker_delta(0, mm)
                if not self.normalized:
                    g_ext_1 = 0.5 * (1.0 + delta_0_m)
                    g_ext_2 = 0.5 * (n - m + 2.0) * (n - m + 1.0)
                    g_ext_3 = -(n - m + 1.0)
                else:
                    delta_1_m = SphericalHarmonicsGravity._kronecker_delta(1, mm)
                    g_ext_1 = (
                        0.5
                        * (1.0 + delta_0_m)
                        * np.sqrt(
                            (2.0 - delta_0_m)
                            * (2.0 * n + 1.0)
                            * (n + m + 2.0)
                            * (n + m + 1.0)
                            / (2.0 * (2.0 * n + 3.0))
                        )
                    )
                    g_ext_2 = 0.5 * np.sqrt(
                        (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / ((2.0 - delta_1_m) * (2.0 * n + 3.0))
                    )
                    g_ext_3 = -np.sqrt(
                        (2.0 * n + 1.0)
                        * (n + m + 1.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 3.0)
                    )

                mu_by_r2 = self.mu / (self.r_ref**2)

                if mm == 0:
                    x_ddot += (
                        mu_by_r2
                        * g_ext_1
                        * (-self._Cnm[nn, mm] * self.bnm_ext_real[nn + 1, mm + 1])
                    )
                    y_ddot += (
                        mu_by_r2
                        * g_ext_1
                        * (-self._Cnm[nn, mm] * self.bnm_ext_imag[nn + 1, mm + 1])
                    )
                else:
                    x_ddot += mu_by_r2 * (
                        g_ext_1
                        * (
                            -self._Cnm[nn, mm] * self.bnm_ext_real[nn + 1, mm + 1]
                            - self._Snm[nn, mm] * self.bnm_ext_imag[nn + 1, mm + 1]
                        )
                        + g_ext_2
                        * (
                            self._Cnm[nn, mm] * self.bnm_ext_real[nn + 1, mm - 1]
                            + self._Snm[nn, mm] * self.bnm_ext_imag[nn + 1, mm - 1]
                        )
                    )
                    y_ddot += mu_by_r2 * (
                        g_ext_1
                        * (
                            self._Snm[nn, mm] * self.bnm_ext_real[nn + 1, mm + 1]
                            - self._Cnm[nn, mm] * self.bnm_ext_imag[nn + 1, mm + 1]
                        )
                        + g_ext_2
                        * (
                            self._Snm[nn, mm] * self.bnm_ext_real[nn + 1, mm - 1]
                            - self._Cnm[nn, mm] * self.bnm_ext_imag[nn + 1, mm - 1]
                        )
                    )

                z_ddot += (
                    mu_by_r2
                    * g_ext_3
                    * (
                        self._Cnm[nn, mm] * self.bnm_ext_real[nn + 1, mm]
                        + self._Snm[nn, mm] * self.bnm_ext_imag[nn + 1, mm]
                    )
                )

        # Pack acceleration vector
        sph_acceleration_bf = np.array([x_ddot, y_ddot, z_ddot])

        # Map back to input reference frame using estimated rotation
        sph_acceleration = self._acceleration_transform_rf(
            sph_acceleration_bf, epoch, R_b2rf
        )

        # Restore original coefficients and GM
        self._Cnm, self._Snm = _Cnm_save, _Snm_save
        self._mu = _mu_save

        return sph_acceleration.transpose()

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the spherical harmonic acceleration with respect to position as ndarray.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector.

        current_state : dict
            Dictionary containing the current state information.

        epoch : float
            Epoch at which the transformation is performed.

        body : None
            Not explicitly used but included for interface consistency.

        Returns
        -------
        partial_a_by_pos : np.ndarray
            The 3x3 partial derivative (Jacobian) of acceleration with respect to
            position. Assumed in units of :math:`\\frac{1}{s^2}`. Expressed in the input reference frame
        """

        # Initialize the total partial derivative matrix (by elements)
        dxddot_dx = 0.0
        dyddot_dy = 0.0
        dzddot_dz = 0.0
        dxddot_dy = 0.0
        dxddot_dz = 0.0
        dyddot_dz = 0.0

        # Initialize the total partial derivative matrix (by elements) for output
        partial_a_by_pos_bf = np.zeros(
            (3, 3)
        )  # Jacobian matrix in 1/s^2, body-fixed frame
        partial_a_by_pos = np.zeros((3, 3))  # Jacobian matrix in 1/s^2, input frame

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)

        # Compute raw Bnm arrays (body-fixed position uses estimated rotation)
        bnm_real, bnm_imag = self._compute_bnm(position, epoch, R_b2rf)
        self._bnm_ext_real = bnm_real
        self._bnm_ext_imag = bnm_imag

        mu_r3 = self.mu / (self.r_ref**3)
        r = self._bnm_ext_real
        i = self._bnm_ext_imag

        for nn in range(0, self.order + 1):
            n = nn
            for mm in range(0, nn + 1):
                m = mm

                C = float(self._Cnm[nn, mm])
                S = float(self._Snm[nn, mm])

                if self.normalized:
                    delta_0_m = SphericalHarmonicsGravity._kronecker_delta(0, mm)
                    delta_1_m = SphericalHarmonicsGravity._kronecker_delta(1, mm)
                    delta_2_m = SphericalHarmonicsGravity._kronecker_delta(2, mm)

                    s1 = np.sqrt(
                        0.5
                        * (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n + m + 4.0)
                        * (n + m + 3.0)
                        * (n + m + 2.0)
                        * (n + m + 1.0)
                        / (2.0 * n + 5.0)
                    )
                    s2 = np.sqrt(
                        (2.0 * n + 1.0)
                        * (n + m + 2.0)
                        * (n + m + 1.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 5.0)
                    )
                    s3 = np.sqrt(
                        (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n - m + 4.0)
                        * (n - m + 3.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / ((2.0 - delta_2_m) * (2.0 * n + 5.0))
                    )
                    s4 = np.sqrt(
                        0.5
                        * (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n + m + 3.0)
                        * (n + m + 2.0)
                        * (n + m + 1.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 5.0)
                    )
                    s5 = np.sqrt(
                        (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n + m + 1.0)
                        * (n - m + 3.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / ((2.0 - delta_1_m) * (2.0 * n + 5.0))
                    )
                    s6 = np.sqrt(
                        (2.0 * n + 1.0)
                        * (n + m + 2.0)
                        * (n + m + 1.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 5.0)
                    )
                    if m == 0:
                        dxddot_dx += (
                            mu_r3 * 0.5 * C * (s1 * r[nn + 2, 2] - s2 * r[nn + 2, 0])
                        )
                        dxddot_dy += mu_r3 * 0.5 * C * s1 * i[nn + 2, 2]
                    elif m == 1:
                        dxddot_dx += (
                            mu_r3
                            * 0.25
                            * (
                                C * (s1 * r[nn + 2, 3] - 3.0 * s2 * r[nn + 2, 1])
                                + S * (s1 * i[nn + 2, 3] - s2 * i[nn + 2, 1])
                            )
                        )
                        dxddot_dy += (
                            mu_r3
                            * 0.25
                            * (
                                -S * (s1 * r[nn + 2, 3] + s2 * r[nn + 2, 1])
                                + C * (s1 * i[nn + 2, 3] - s2 * i[nn + 2, 1])
                            )
                        )
                    else:
                        dxddot_dx += (
                            mu_r3
                            * 0.25
                            * (
                                C
                                * (
                                    s1 * r[nn + 2, m + 2]
                                    - 2.0 * s2 * r[nn + 2, m]
                                    + s3 * r[nn + 2, m - 2]
                                )
                                + S
                                * (
                                    s1 * i[nn + 2, m + 2]
                                    - 2.0 * s2 * i[nn + 2, m]
                                    + s3 * i[nn + 2, m - 2]
                                )
                            )
                        )
                        dxddot_dy += (
                            mu_r3
                            * 0.25
                            * (
                                -S * (s1 * r[nn + 2, m + 2] - s3 * r[nn + 2, m - 2])
                                + C * (s1 * i[nn + 2, m + 2] - s3 * i[nn + 2, m - 2])
                            )
                        )

                    if m == 0:
                        dxddot_dz += mu_r3 * s4 * C * r[nn + 2, 1]
                        dyddot_dz += mu_r3 * s4 * C * i[nn + 2, 1]
                    else:
                        dxddot_dz += (
                            mu_r3
                            * 0.5
                            * (
                                C * (s4 * r[nn + 2, m + 1] - s5 * r[nn + 2, m - 1])
                                + S * (s4 * i[nn + 2, m + 1] - s5 * i[nn + 2, m - 1])
                            )
                        )
                        dyddot_dz += (
                            mu_r3
                            * 0.5
                            * (
                                -S * (s4 * r[nn + 2, m + 1] + s5 * r[nn + 2, m - 1])
                                + C * (s4 * i[nn + 2, m + 1] + s5 * i[nn + 2, m - 1])
                            )
                        )

                    dzddot_dz += mu_r3 * s6 * (C * r[nn + 2, m] + S * i[nn + 2, m])

                else:
                    dxddot_dz += (
                        self.mu
                        / (self.r_ref**3)
                        * 0.5
                        * (
                            (
                                self._Cnm[nn, mm]
                                * (
                                    (n - m + 1.0) * self.bnm_ext_real[nn + 2, mm + 1]
                                    - (n - m + 3.0)
                                    * (n - m + 2.0)
                                    * (n - m + 1.0)
                                    * self.bnm_ext_real[nn + 2, mm - 1]
                                )
                            )
                            + (
                                self._Snm[nn, mm]
                                * (
                                    (n - m + 1.0) * self.bnm_ext_imag[nn + 2, mm + 1]
                                    - (n - m + 3.0)
                                    * (n - m + 2.0)
                                    * (n - m + 1.0)
                                    * self.bnm_ext_imag[nn + 2, mm - 1]
                                )
                            )
                        )
                    )
                    dyddot_dz += (
                        self.mu
                        / (self.r_ref**3)
                        * 0.5
                        * (
                            (
                                -self._Snm[nn, mm]
                                * (
                                    (n - m + 1.0) * self.bnm_ext_real[nn + 2, mm + 1]
                                    + (n - m + 3.0)
                                    * (n - m + 2.0)
                                    * (n - m + 1.0)
                                    * self.bnm_ext_real[nn + 2, mm - 1]
                                )
                            )
                            + (
                                self._Cnm[nn, mm]
                                * (
                                    (n - m + 1.0) * self.bnm_ext_imag[nn + 2, mm + 1]
                                    + (n - m + 3.0)
                                    * (n - m + 2.0)
                                    * (n - m + 1.0)
                                    * self.bnm_ext_imag[nn + 2, mm - 1]
                                )
                            )
                        )
                    )

        dyddot_dy = -dxddot_dx - dzddot_dz

        # Pack Jacobian
        partial_a_by_pos_bf[0, 0] = dxddot_dx
        partial_a_by_pos_bf[0, 1] = dxddot_dy
        partial_a_by_pos_bf[0, 2] = dxddot_dz
        partial_a_by_pos_bf[1, 0] = dxddot_dy
        partial_a_by_pos_bf[1, 1] = dyddot_dy
        partial_a_by_pos_bf[1, 2] = dyddot_dz
        partial_a_by_pos_bf[2, 0] = dxddot_dz
        partial_a_by_pos_bf[2, 1] = dyddot_dz
        partial_a_by_pos_bf[2, 2] = dzddot_dz

        # Map back to input reference frame using estimated rotation
        partial_a_by_pos = self._jacobian_pos_transform_rf(
            partial_a_by_pos_bf, epoch, R_b2rf
        )

        return partial_a_by_pos

    def _compute_partial_by_C(
        self,
        position: np.ndarray,
        epoch,
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the spherical harmonic acceleration with respect to the C coefficients.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector of the spacecraft in the original reference frame.
            Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state information.

        epoch : float
            Epoch at which the transformation is performed.

        body : CelestialBody
            The celestial body for which spherical harmonics are computed.

        Returns
        -------
        partials_a_by_C : np.ndarray
            Partial derivatives of acceleration with respect to the C coefficients. Size of
            (order+1)x(order+1). Expressed in units of :math:`\\frac{km}{s^2}`.
        """
        # Initialization
        dxddot_dC = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2
        dyddot_dC = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2
        dzddot_dC = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2

        # Compute external terms
        self._bnm_ext_real, self._bnm_ext_imag = self._compute_bnm(position, epoch)

        # Calculate partials
        for nn in range(0, self.order + 1):
            n = nn
            for mm in range(0, nn + 1):
                m = mm
                delta_0_m = SphericalHarmonicsGravity._kronecker_delta(0, mm)
                if self.normalized is False:
                    g_ext_1 = 0.5 * (1.0 + delta_0_m)
                    g_ext_2 = 0.5 * (n - m + 2.0) * (n - m + 1.0)
                    g_ext_3 = -(n - m + 1.0)
                elif self.normalized is True:
                    delta_1_m = SphericalHarmonicsGravity._kronecker_delta(1, mm)
                    g_ext_1 = (
                        0.5
                        * (1.0 + delta_0_m)
                        * np.sqrt(
                            (2.0 - delta_0_m)
                            * (2.0 * n + 1.0)
                            * (n + m + 2.0)
                            * (n + m + 1.0)
                            / (2.0 * (2.0 * n + 3.0))
                        )
                    )
                    g_ext_2 = 0.5 * np.sqrt(
                        (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / ((2.0 - delta_1_m) * (2.0 * n + 3.0))
                    )
                    g_ext_3 = -np.sqrt(
                        (2.0 * n + 1.0)
                        * (n + m + 1.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 3.0)
                    )

                if mm == 0:
                    dxddot_dC[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * g_ext_1
                        * (-self.bnm_ext_real[nn + 1, mm + 1])
                    )
                    dyddot_dC[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * g_ext_1
                        * (-self.bnm_ext_imag[nn + 1, mm + 1])
                    )
                else:
                    dxddot_dC[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * (
                            g_ext_1 * (-self.bnm_ext_real[nn + 1, mm + 1])
                            + g_ext_2 * (self.bnm_ext_real[nn + 1, mm - 1])
                        )
                    )
                    dyddot_dC[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * (
                            g_ext_1 * (-self.bnm_ext_imag[nn + 1, mm + 1])
                            + g_ext_2 * (-self.bnm_ext_imag[nn + 1, mm - 1])
                        )
                    )

                dzddot_dC[nn, mm] = (
                    self.mu / (self.r_ref**2) * g_ext_3 * self.bnm_ext_real[nn + 1, mm]
                )

        # Pack partials wrt C in body-fixed frame
        partials_a_by_C_bf = np.array([dxddot_dC, dyddot_dC, dzddot_dC])

        # Map back to input reference frame
        partials_a_by_C = self._jacobian_C_transform_rf(partials_a_by_C_bf, epoch)

        return partials_a_by_C

    def _compute_partial_by_S(
        self,
        position: np.ndarray,
        epoch,
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the spherical harmonic acceleration with respect to the S coefficients.

        Parameters
        ----------
        position : np.ndarray
            3x1 position vector of the spacecraft in the original reference frame.
            Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state information.

        epoch : float
            Epoch at which the transformation is performed.

        body : CelestialBody
            The celestial body for which spherical harmonics are computed.

        Returns
        -------
        partials_a_by_S : np.ndarray
            Partial derivatives of acceleration with respect to the S coefficients. Size of (order+1)x(order+1).
            Expressed in units :math:`\\frac{km}{s^2}`.
        """
        # Initialization
        dxddot_dS = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2
        dyddot_dS = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2
        dzddot_dS = np.zeros((self.order + 1, self.order + 1))  # Assumed in km/sec^2

        # Compute external terms
        self._bnm_ext_real, self._bnm_ext_imag = self._compute_bnm(position, epoch)

        # Calculate partials
        for nn in range(0, self.order + 1):
            n = nn
            for mm in range(0, nn + 1):
                m = mm
                delta_0_m = SphericalHarmonicsGravity._kronecker_delta(0, mm)
                if self.normalized is False:
                    g_ext_1 = 0.5 * (1.0 + delta_0_m)
                    g_ext_2 = 0.5 * (n - m + 2.0) * (n - m + 1.0)
                    g_ext_3 = -(n - m + 1.0)
                elif self.normalized is True:
                    delta_1_m = SphericalHarmonicsGravity._kronecker_delta(1, mm)
                    g_ext_1 = (
                        0.5
                        * (1.0 + delta_0_m)
                        * np.sqrt(
                            (2.0 - delta_0_m)
                            * (2.0 * n + 1.0)
                            * (n + m + 2.0)
                            * (n + m + 1.0)
                            / (2.0 * (2.0 * n + 3.0))
                        )
                    )
                    g_ext_2 = 0.5 * np.sqrt(
                        (2.0 - delta_0_m)
                        * (2.0 * n + 1.0)
                        * (n - m + 2.0)
                        * (n - m + 1.0)
                        / ((2.0 - delta_1_m) * (2.0 * n + 3.0))
                    )
                    g_ext_3 = -np.sqrt(
                        (2.0 * n + 1.0)
                        * (n + m + 1.0)
                        * (n - m + 1.0)
                        / (2.0 * n + 3.0)
                    )

                if mm == 0:
                    continue

                else:
                    dxddot_dS[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * (
                            g_ext_1 * (-self.bnm_ext_imag[nn + 1, mm + 1])
                            + g_ext_2 * (self.bnm_ext_imag[nn + 1, mm - 1])
                        )
                    )
                    dyddot_dS[nn, mm] = (
                        self.mu
                        / (self.r_ref**2)
                        * (
                            g_ext_1 * (self.bnm_ext_real[nn + 1, mm + 1])
                            + g_ext_2 * (self.bnm_ext_real[nn + 1, mm - 1])
                        )
                    )

                dzddot_dS[nn, mm] = (
                    self.mu / (self.r_ref**2) * g_ext_3 * self.bnm_ext_imag[nn + 1, mm]
                )

        # Pack partials wrt S
        partials_a_by_S_bf = np.array([dxddot_dS, dyddot_dS, dzddot_dS])

        # Map back to input reference frame
        partials_a_by_S = self._jacobian_C_transform_rf(partials_a_by_S_bf, epoch)

        return partials_a_by_S

    def _normalize_coeffs(self) -> None:
        """
        Normalizes the spherical harmonic coefficients if the loaded coefficients are unnormalized.

        Normalization is performed using the formula:

        .. math::

            N = \\sqrt{\\frac{(2 - \\delta_{0_m}) * (2(n + 1)) * (n - m)!}{(n + m)!}}

        where :math:`\\delta_{0_m}` is 1 if m = 0, otherwise 0.

        This ensures the coefficients comply with the normalization convention.

        Returns
        -------
        None
        """
        # Normalize the coefficients
        for nn in range(0, self.order + 1):
            for mm in range(0, nn + 1):
                delta_0_m = SphericalHarmonicsGravity._kronecker_delta(0, mm)
                N = np.sqrt(
                    (2.0 - delta_0_m)
                    * (2.0 * nn + 1.0)
                    * factorial(nn - mm)
                    / factorial(nn + mm)
                )

                self._Cnm[nn, mm] = self._Cnm[nn, mm] / N
                self._Snm[nn, mm] = self._Snm[nn, mm] / N

    def _kronecker_delta(i: int, j: int):
        """Kronecker delta function. See https://en.wikipedia.org/wiki/Kronecker_delta

        Parameters
        ----------
        i : int
            First value
        j : int
            Second value

        Returns
        -------
        int
            1 if i == j, or 0 if i ~= j
        """
        return int(i == j)

    # ------------------------------------------------------------------
    # Pole-parameter partials
    # ------------------------------------------------------------------
    # All three use the identity (derived from the IAU pole rotation):
    #
    #   da_rf/d(param) = A(param) @ a_rf  -  J_rf @ (A(param) @ r_rf)
    #
    # where:
    #   a_rf   = SH acceleration in reference frame
    #   J_rf   = da_rf/dr_rf  (3x3 position Jacobian in reference frame)
    #   A(RA)  = G_z                   (generator of z-rotations, body-fixed z = pole)
    #   A(DEC) = -G̃_x(Ω)              (rotated x-generator; Ω = RA + 90°)
    #   A(W)   = G̃_z = R @ G_z @ R^T  (G_z expressed in reference frame)
    #   A(PM_rate) = A(W) * t_days * (π/180)
    #
    # Derivation: the body-fixed-to-RF DCM is R = R_z(Ω) * R_x(i) * R_z(W).
    # dR/dRA = G_z * R,  dR/dDEC = -G̃_x(Ω) * R,  dR/dW = R * G_z.
    # ------------------------------------------------------------------

    def _compute_partial_by_pole_RA(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Partial of SH acceleration wrt the north-pole right ascension RA [rad].

        Returns
        -------
        np.ndarray
            (3, 1) array in km/s^2 per radian.
        """
        a_rf = self.compute_acceleration(position, current_state, epoch, body)
        J_rf = self._compute_partial_by_position(position, current_state, epoch, body)

        G_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # G_z acts in the reference frame (RA rotates about the inertial z-axis)
        partial = G_z @ a_rf - J_rf @ (G_z @ position)
        return partial.reshape(3, 1)

    def _compute_partial_by_pole_DEC(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Partial of SH acceleration wrt the north-pole declination DEC [rad].

        Returns
        -------
        np.ndarray
            (3, 1) array in km/s^2 per radian.
        """
        a_rf = self.compute_acceleration(position, current_state, epoch, body)
        J_rf = self._compute_partial_by_position(position, current_state, epoch, body)

        # Build G̃_x(Ω) from the third column of R_b2rf (north-pole direction in RF).
        # Use estimated rotation if pole parameters are in the state.
        R = self._get_R_b2rf_effective(current_state, epoch)
        r3 = R[:, 2]
        cd = float(np.sqrt(r3[0] ** 2 + r3[1] ** 2))  # = cos(DEC)
        if cd > 1e-10:
            sO = r3[0] / cd  # sin(Ω) = cos(RA)
            neg_cO = r3[1] / cd  # -cos(Ω) = sin(RA)
        else:
            sO, neg_cO = 0.0, 0.0

        G_x_tilde = np.array([[0.0, 0.0, sO], [0.0, 0.0, neg_cO], [-sO, -neg_cO, 0.0]])

        partial = -G_x_tilde @ a_rf + J_rf @ (G_x_tilde @ position)
        return partial.reshape(3, 1)

    def _compute_partial_by_pole_PM(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Partial of SH acceleration wrt the prime-meridian rate PM [deg/day].

        ``epoch`` is SPICE ephemeris time (seconds past J2000 TDB).

        Returns
        -------
        np.ndarray
            (3, 1) array in km/s^2 per (deg/day).
        """
        a_rf = self.compute_acceleration(position, current_state, epoch, body)
        J_rf = self._compute_partial_by_position(position, current_state, epoch, body)

        # G̃_z = R @ G_z @ R^T: G_z expressed in the reference frame.
        # Use estimated rotation if pole parameters are in the state.
        R = self._get_R_b2rf_effective(current_state, epoch)

        G_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        G_z_tilde = R @ G_z @ R.T

        # da/dW [km/s^2/rad]:  A(W) @ a_rf - J_rf @ (A(W) @ r_rf)
        da_dW = G_z_tilde @ a_rf - J_rf @ (G_z_tilde @ position)

        # Convert from rad to deg/day: dW/d(PM_deg_per_day) = t_days * (π/180)
        import spiceypy as _spice
        t_days = epoch / _spice.spd()  # epoch is SPICE ET (seconds past J2000 TDB)
        partial = da_dW * t_days * (np.pi / 180.0)
        return partial.reshape(3, 1)

    def _compute_partial_by_mu(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Partial of SH acceleration wrt the body's gravitational parameter GM.

        Because every term in the SH expansion is linear in GM, this is simply
        ``a_sh / GM``.

        Returns
        -------
        np.ndarray
            (3, 1) array in km^-2  (i.e., km/s^2 per km^3/s^2).
        """
        # Determine the active GM (estimated or nominal) — must match what
        # compute_acceleration used, since a_SH is linear in GM and da/dGM = a/GM.
        mu_active = self._mu
        if self._sph_body_spice_id is not None and current_state is not None:
            _mu_key = ("gm", self._sph_body_spice_id)
            if _mu_key in current_state:
                _mu_val = current_state[_mu_key]
                mu_active = float(
                    _mu_val.values if hasattr(_mu_val, "values") else _mu_val
                )
        a_rf = self.compute_acceleration(position, current_state, epoch, body)
        partial = a_rf / mu_active
        return partial.reshape(3, 1)
