# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Body,
    SpiceManager,
    StateArray,
    Frame,
    CelestialBody,
)

from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation

import scarabaeus.utils.NumpyWrapper as np
from math import factorial
import json
import warnings

try:
    from scarabaeus.rust import SphericalHarmonicsCore as _SphHarmRustCore
except ImportError:
    _SphHarmRustCore = None


#######################################################################
class SphericalHarmonicsGravity(DynamicModel):
    """Models the gravitational acceleration due to a body's non-spherical mass distribution
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
    The angles can be estimated as solve-for parameters by including
    ``("pole_ra", spice_id)``, ``("pole_dec", spice_id)``,
    ``("pole_pm", spice_id)`` or ``("pole_w0", spice_id)`` entries in the
    state vector.  When estimated,
    the rotation matrix :math:`\\mathbf{R}` (body-fixed to inertial) is rebuilt
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

    origin : Body
        Body the propagated position is measured from. Only its identity is
        needed: it fixes the vector from the propagation origin to the harmonics
        body, so the model can be built once — in an ops configuration file, for
        instance — and reused across propagations.

        When the field body is not the origin, the acceleration returned is the
        perturbation felt in the origin-centred frame: the field at the
        evaluation point less the pull the same body exerts on the origin. That
        second piece is the indirect term of the classical third-body
        perturbation, so such a body must not also be listed in
        ``third_bodies`` — :class:`~scarabaeus.ForceModelTranslation` drops it
        from the point-mass list for exactly this reason. When the field body
        *is* the origin the expansion is returned as-is.

        A :class:`~scarabaeus.StateArray` is also accepted, in which case its
        ``origin`` is used. That is the historical form of this argument and is
        kept so existing call sites keep working.

    sph_harm_norm_flag : bool, optional
        Retained for backward compatibility. Defaults to ``True``. Coefficients
        are always converted to normalized form on load (required by the
        ``rust`` core), so both normalized and unnormalized
        coefficient files are supported; the file's own ``Normalized`` field
        determines whether a conversion is performed.

    n_degree_max : int, optional
        Maximum degree for harmonics. Defaults to ``100``.

    base_frame : Frame, optional
        Frame of the propagator. If ``base_frame = None``, defaults to the J2000 frame.

    See Also
    --------
    scarabaeus.StateArray : Represents state vectors with origins and reference frames.
    scarabaeus.ForceModelTranslation : Aggregator that builds this model for a propagation.

    Notes
    -----
    The spherical harmonic model accounts for deviations from a perfect sphere, ypically used for planetary bodies.

    Coefficients can be loaded in normalized or unnormalized form.

    Uses the gravitational parameter and reference radius for calculations.

    References
    ----------
    .. [1] Kaula, W. M. (1966). Theory of Satellite Geodesy.

    Examples
    --------
    Build the model on its own — no state vector needed — so it can live in a
    configuration file and be reused across propagations.

    >>> import scarabaeus as scb

    >>> earth = scb.CelestialBody.from_constants("EARTH")
    >>> sh = scb.SphericalHarmonicsGravity(
    ...     8,
    ...     "data/dynamic_setup/sph_coefficients/earth_ggm03.json",
    ...     earth,
    ...     origin = earth,
    ... )
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError.
    class RustCoreUnavailableError(Exception):
        """The rust spherical-harmonics core is unavailable.

        Raised by the compute and partial methods when ``_rust_core`` could not
        be built because the extension is not importable. The Rust kernels are
        the only implementation of the body-fixed acceleration and its
        analytical partials.
        """

    class AutogradTracingError(Exception):
        """Autograd is tracing the acceleration through the Rust kernels.

        The autograd NumPy backend (``UseAutoGrad=TRUE``) is fine on its own —
        the model runs on plain arrays under it. Only a traced position cannot
        pass into Rust. The partials are verified by the Rust dual-number
        differentiation instead (``SphericalHarmonicsCore.compute_jacobian_ad_bf``
        and the GM, C_nm/S_nm and pole counterparts), which is what
        ``check_force_model_partials=TRUE`` runs.
        """

    class MissingSpiceNameError(Exception):
        """The spherical-harmonics body has no SPICE name to resolve its frame."""

    class MissingBodyFixedFrameError(Exception):
        """The spherical-harmonics body carries no body-fixed (rotating) frame."""

    class InvalidOriginError(TypeError):
        """The supplied origin is neither a body nor a state vector holding one."""

    class PropagatedBodyStateMissingError(Exception):
        """The harmonics body is integrated in the state vector, but its
        position is absent from the state handed to this model.

        Falling back to the ephemeris here would silently evaluate the field
        about the a priori position while the filter estimates another, so the
        absence is reported instead.
        """

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        sph_harm_order: int,
        sph_harm_cs_file: str,
        sph_harm_body,
        origin,
        sph_harm_norm_flag: bool = True,
        n_degree_max: int = 100,
        base_frame: Frame = None,
        propagated_bodies: list | None = None,
    ):
        # Initialize the class attributes
        self._n_degree_max = n_degree_max
        self._order = sph_harm_order
        self._cs_file = sph_harm_cs_file
        self._body = sph_harm_body
        self._origin = self._resolve_origin(origin)
        self._normalized = sph_harm_norm_flag
        self._origin_name = self._origin.name
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
            raise self.MissingSpiceNameError("sph_harm_body must contain 'spice_name'.")
        # When the SPH body IS the propagation origin, r_sph/origin is always
        # zero and the position lookup in _rf_transform can be skipped. SPICE
        # names are case-insensitive ("MARS" in a field spec, "Mars" on the
        # body), and ForceModelTranslation._names_match compares them that way
        # to decide whether the origin still needs its monopole; the two must
        # agree.
        self._sph_body_is_origin = (
            str(self._origin.name).upper() == str(sph_name).upper()
        )

        # Whether the harmonics body is integrated in the state vector rather
        # than read from the ephemeris. When it is, its position has to come
        # from the state: an estimator moves that body away from its a priori
        # ephemeris, and evaluating the field about the ephemeris value instead
        # would hold the field fixed while the estimate moves.
        self._sph_body_is_propagated = any(
            str(getattr(candidate, "name", "")).upper() == str(sph_name).upper()
            for candidate in (propagated_bodies or [])
        )

        if not self._sph_body_is_origin:
            source = (
                "the state vector, which carries it"
                if self._sph_body_is_propagated
                else "SPICE"
            )
            warnings.warn(
                f"SPH gravity body is '{sph_name}', but propagation origin is "
                f"'{self._origin.name}'. The model will compute the relative vector "
                f"r_eval/sph = r_eval/origin - r_sph/origin, taking r_sph/origin "
                f"from {source}.",
                RuntimeWarning,
            )

        # Cache IAU secular pole constants (deg / deg·century / deg·day) from SPICE PCK.
        # Used to rebuild R when pole_ra / pole_dec / pole_pm / pole_w0 are estimated.
        # Only the secular (zeroth- and first-order) terms are stored; higher-order nutation
        # terms are small and cancel in the correction computation.
        self._iau_ra_coeffs = None
        self._iau_dec_coeffs = None
        self._iau_pm_coeffs = None
        try:
            import spiceypy as _spice

            # bodvrd returns (dim, values); keep the values array so the
            # secular coefficients index as [0], [1], [2].
            _sn = sph_name
            self._iau_ra_coeffs = _spice.bodvrd(_sn, "POLE_RA", 3)[
                1
            ]  # [RA0, RA1, RA2]  deg & deg/century
            self._iau_dec_coeffs = _spice.bodvrd(_sn, "POLE_DEC", 3)[
                1
            ]  # [DEC0, DEC1, DEC2]
            self._iau_pm_coeffs = _spice.bodvrd(_sn, "PM", 3)[
                1
            ]  # [W0, Wdot, Wddot] deg & deg/day
        except Exception:
            pass

        # Load spherical harmonics coefficients from the JSON file
        self._load_coefficients()

        # The Rust core computes with normalized coefficients only. Convert
        # unnormalized input files here (Python-side) so both normalized and
        # unnormalized coefficient sets are supported through the same Rust
        # path. After this the stored coefficients are always normalized.
        if not self._input_normalized:
            self._normalize_coeffs()
        self._normalized = True

        # Optional solid-tide perturbation of the field; see solid_tide.
        self._solid_tide = None

        # Body-fixed kernels run in rust — the only compute path.
        # Unavailable only when the extension is missing; the compute methods
        # then raise (_require_rust_core). Built under the autograd backend too:
        # the propagation evaluates the model on plain arrays.
        self._rust_core = None
        if _SphHarmRustCore is not None:
            self._rust_core = _SphHarmRustCore(self._order, self._r_ref)

    def __getstate__(self):
        # Compiled extension objects are not picklable; rebuilt in __setstate__.
        state = self.__dict__.copy()
        state["_rust_core"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if _SphHarmRustCore is not None:
            self._rust_core = _SphHarmRustCore(self._order, self._r_ref)

    def _require_rust_core(self, position):
        """
        Guard for the compute methods: the rust core is the only
        implementation of the body-fixed kernels.

        Coefficients (normalized or unnormalized) are always converted to
        normalized form in the constructor, so ``_rust_core`` is available for
        both conventions. It is ``None`` only when the extension is missing.

        The autograd NumPy backend alone is not an error — only a position
        that autograd is actually tracing, since it cannot enter Rust.

        Parameters
        ----------
        position : np.ndarray
            The position handed to the compute method.

        Raises
        ------
        RustCoreUnavailableError
            If the rust core was not built.
        AutogradTracingError
            If autograd is tracing ``position``.
        """
        if self._rust_core is None:
            raise self.RustCoreUnavailableError(
                "SphericalHarmonicsGravity requires the rust "
                "extension. Build it with `maturin develop --release` from the "
                "repository root."
            )
        if isinstance(position, np.autograd_types):
            raise self.AutogradTracingError(
                "SphericalHarmonicsGravity evaluates its body-fixed kernels in "
                "Rust, which autograd cannot trace. Verify its partials with "
                "check_force_model_partials=TRUE, which uses the Rust "
                "dual-number differentiation (compute_jacobian_ad_bf and the "
                "GM, C_nm/S_nm and pole counterparts)."
            )

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
    def solid_tide(self):
        """Optional :class:`~scarabaeus.SolidTide` perturbing this field.

        The same tide that displaces a surface station also redistributes
        mass, which changes the external gravity field. Attaching a tide model
        here adds its degree-2 coefficient perturbation at every evaluation.

        The tide is a known perturbation rather than an estimated one, so it
        rides on the coefficients passed to the kernel and leaves the
        Jacobian on the nominal field, exactly as the estimated Cnm/Snm
        corrections do.

        Its ``central_body`` must be the body this field belongs to, and the
        base coefficients' own tide convention has to be consistent: a
        *zero-tide* field already contains the permanent deformation, so
        adding a tide on top of one counts it twice. That difference is about
        1e-8 in normalized C20, comparable to the whole tidal signal, and
        nothing here can detect it for you.
        """
        return self._solid_tide

    @solid_tide.setter
    def solid_tide(self, value):
        if value is not None and not hasattr(value, "geopotential_coefficients"):
            raise TypeError("solid_tide needs a geopotential_coefficients(epoch, frame) method")
        self._solid_tide = value
        self._jac_rf_cache = None

    def _estimated_love_k(self, current_state: dict) -> dict:
        """Estimated ``love_k_<n>_<m>`` entries for this body.

        Parameters
        ----------
        current_state : dict or None
            Filter state keyed ``(name, spice_id)``; only ``love_k_<n>_<m>``
            entries owned by this field's body are read.

        Returns
        -------
        dict
            ``(n, m)`` mapped to the estimated :math:`k_{nm}`. Empty when none
            are solved for, which is the ordinary case, when the state is
            None, or when the field has no body.
        """
        if self._sph_body_spice_id is None or current_state is None:
            return {}
        out = {}
        for key, value in current_state.items():
            if not (isinstance(key, tuple) and len(key) == 2
                    and key[1] == self._sph_body_spice_id
                    and isinstance(key[0], str) and key[0].startswith("love_k_")):
                continue
            parts = key[0].split("_")
            out[(int(parts[2]), int(parts[3]))] = float(
                value.values if hasattr(value, "values") else value)
        return out

    def _tide_delta(self, epoch: float, current_state: dict, unit_love: bool = False):
        """Tidal coefficient perturbation, honouring estimated Love numbers.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        current_state : dict
            Filter state, read for estimated ``love_k_<n>_<m>`` entries.
        unit_love : bool, optional
            Return the derivative with respect to each :math:`k_{nm}` instead
            of the perturbation itself. Defaults to ``False``.

        Returns
        -------
        dict
            ``(degree, order)`` mapped to ``(dC, dS)``.
        """
        tide = self._solid_tide
        if unit_love:
            return tide.geopotential_partials(float(epoch), self._sph_body_fixed_frame())
        estimated = self._estimated_love_k(current_state)
        if estimated:
            from dataclasses import replace
            tide = replace(tide, love_k={**tide.love_k, **estimated})
        return tide.geopotential_coefficients(float(epoch), self._sph_body_fixed_frame())

    def _coefficients_with_tide(self, epoch: float, current_state: dict = None):
        """Return ``(Cnm, Snm)`` with the solid-tide perturbation added.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        current_state : dict, optional
            Filter state; estimated ``love_k_<n>_<m>`` entries replace the
            tide model's own values, so the forward acceleration follows the
            iterate. Defaults to ``None``, the tide model's values.

        Returns
        -------
        tuple of numpy.ndarray
            Copies of the current coefficients with the tidal delta applied.
            Degrees above the field's own order are dropped rather than
            silently resized.
        """
        Cnm = np.array(self._Cnm, dtype=float, copy=True)
        Snm = np.array(self._Snm, dtype=float, copy=True)
        for (degree, order), (dC, dS) in self._tide_delta(epoch, current_state).items():
            if degree <= self._order and order <= self._order:
                Cnm[degree, order] += dC
                Snm[degree, order] += dS
        return Cnm, Snm

    def _compute_partial_by_love_k(self, degree, order, position, current_state, epoch):
        r"""Acceleration sensitivity to one potential Love number, km/s^2.

        Parameters
        ----------
        degree : int
            Degree :math:`n` of the Love number :math:`k_{nm}`.
        order : int
            Order :math:`m` of the Love number.
        position : numpy.ndarray
            Spacecraft position in the propagation frame, km.
        current_state : dict
            Filter state.
        epoch : float
            TDB seconds past J2000.

        Returns
        -------
        numpy.ndarray
            ``(3, 1)`` partial, or zeros when no tide model is attached, the
            tide has no :math:`k_{nm}`, or ``(n, m)`` is beyond the field's
            order.

        Notes
        -----
        Each :math:`k_{nm}` drives only its own :math:`(n, m)` coefficients,
        and linearly, so

        .. math::

            \frac{\partial a}{\partial k_{nm}} =
                \frac{\partial a}{\partial \bar C_{nm}}
                \frac{\partial \Delta\bar C_{nm}}{\partial k_{nm}}
              + \frac{\partial a}{\partial \bar S_{nm}}
                \frac{\partial \Delta\bar S_{nm}}{\partial k_{nm}}

        contracting partials the field already computes, with the
        :math:`\bar S_{nm}` term absent at order zero. Nothing is
        differentiated numerically.
        """
        key = (int(degree), int(order))
        if self._solid_tide is None or key not in self._solid_tide.love_k:
            return np.zeros((3, 1))
        if key[0] > self._order or key[1] > self._order:
            return np.zeros((3, 1))
        # Diagonal: this k_nm drives its own (n, m) and nothing else, so the
        # chain rule is a single term rather than a sum over the field.
        delta_C, delta_S = self._tide_delta(epoch, current_state, unit_love=True)[key]
        total = self._compute_partial_by_C(
            position, current_state, epoch)[:, key[0], key[1]] * delta_C
        if key[1] > 0:
            total = total + self._compute_partial_by_S(
                position, current_state, epoch)[:, key[0], key[1]] * delta_S
        return np.asarray(total, dtype=float).reshape(3, 1)

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
        self._jac_rf_cache = None  # cached jacobian depends on the coefficients

    @property
    def Snm(self) -> np.ndarray:
        """
        The matrix of sine spherical harmonic coefficients.
        """
        return self._Snm

    @Snm.setter
    def Snm(self, input_val):

        self._Snm = input_val
        self._jac_rf_cache = None  # cached jacobian depends on the coefficients

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

    def _spice_xfrm_cached(self, frame_from: str, frame_to: str, epoch: float):
        """
        Return a SPICE frame transform, memoized per epoch.

        :meth:`compute_acceleration` and the partial methods query the same epoch
        within one right-hand-side evaluation, so each ``(frame_from, frame_to)``
        pair is looked up once per epoch. The cache is cleared whenever the epoch
        changes.

        Parameters
        ----------
        frame_from : str
            Name of the frame the vector is expressed in.
        frame_to : str
            Name of the frame to rotate into.
        epoch : float
            Epoch, in TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` rotation matrix from ``frame_from`` to ``frame_to``.
        """
        if self.__dict__.get("_xfrm_cache") is None or self._xfrm_cache_epoch != epoch:
            self._xfrm_cache = {}
            self._xfrm_cache_epoch = epoch
        key = (frame_from, frame_to)
        R = self._xfrm_cache.get(key)
        if R is None:
            R = self._xfrm_cache[key] = SpiceManager.get_xfrm(
                frame_from, frame_to, epoch
            )
        return R

    def _sph_body_position(self, current_state, epoch: float) -> np.ndarray:
        """
        Position of the harmonics body w.r.t. the propagation origin.

        Parameters
        ----------
        current_state : dict or None
            Current full-state representation from the integrator, keyed by
            ``(name, spice_id)``.
        epoch : float
            Epoch [TDB seconds].

        Returns
        -------
        np.ndarray
            3-element position vector [km], origin to harmonics body.

        Raises
        ------
        PropagatedBodyStateMissingError
            If the body is integrated in the state but its position is absent.

        Notes
        -----
        Mirrors :meth:`~scarabaeus.ThreeBodyGravity._resolve_point_mass_position`:
        a body carried in the state vector is read from the state, and only an
        ephemeris body falls through to SPICE.
        """
        if self._sph_body_is_propagated:
            key = ("position", self._sph_body_spice_id)
            if current_state is None or key not in current_state:
                raise self.PropagatedBodyStateMissingError(
                    f"spherical-harmonics body '{self._sph_body_name()}' is "
                    f"integrated in the state vector, but its position entry "
                    f"{key!r} is missing from the state handed to this model."
                )
            value = current_state[key]
            return np.asarray(
                value.values if hasattr(value, "values") else value, dtype=float
            )
        return self._sph_body_pos_cached(self._sph_body_name(), epoch)

    def _sph_body_name(self) -> str:
        """Return the SPICE name of the harmonics body.

        Returns
        -------
        str
            ``SPH_body["spice_name"]`` when the body is given as a dict, otherwise
            ``SPH_body.name``.
        """
        if isinstance(self.SPH_body, dict):
            return self.SPH_body["spice_name"]
        return self.SPH_body.name

    def _sph_body_pos_cached(self, spice_name: str, epoch: float):
        """
        Return the SPICE position of the harmonics body, memoized per epoch.

        Only queried when the harmonics body is neither the propagation origin nor
        integrated in the state. The cache holds a single epoch.

        Parameters
        ----------
        spice_name : str
            SPICE name of the harmonics body.
        epoch : float
            Epoch, in TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            ``(3,)`` position from the propagation origin to the harmonics body,
            in km, expressed in the reference frame.
        """
        cache = self.__dict__.get("_sph_pos_cache")
        if cache is not None and cache[0] == epoch:
            return cache[1]
        r_sphBody_origin = SpiceManager.get_pos_values(
            spice_name,
            epoch,
            self._ref_frame,
            self._origin.name,
        )  # km, raw (skips ArrayWUnits wrap/unwrap)
        self._sph_pos_cache = (epoch, r_sphBody_origin)
        return r_sphBody_origin

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
            else self._spice_xfrm_cached(
                self._sph_body_fixed_frame(), self._ref_frame.name, epoch
            )
        )
        acceleration_rf = body_fixed_to_ext_DCM @ acceleration_bf.reshape((3, 1))
        return acceleration_rf.flatten()

    @classmethod
    def _resolve_origin(cls, origin):
        """
        Return the propagation-origin body from *origin*.

        The model only ever needs the origin body — its name resolves the
        ``r_sph/origin`` vector and decides whether that lookup can be skipped —
        so a body may be supplied directly and the model built independently of
        any state vector. A :class:`~scarabaeus.StateArray` is still accepted
        and unwrapped, keeping older call sites working.

        Parameters
        ----------
        origin : Body or StateArray
            Origin body, or a state vector whose ``origin`` is that body.

        Returns
        -------
        Body
            The propagation-origin body.

        Raises
        ------
        SphericalHarmonicsGravity.InvalidOriginError
            If *origin* is neither a named body nor a state vector holding one.
        """
        # StateArray (or anything else exposing .origin) — historical form.
        candidate = getattr(origin, "origin", origin)

        if getattr(candidate, "name", None) is None:
            raise cls.InvalidOriginError(
                "'origin' must be the propagation-origin body (e.g. "
                "CelestialBody.from_constants('EARTH')) or a StateArray whose "
                f"origin is that body. Received {type(origin).__name__}."
            )

        return candidate

    def _sph_body_fixed_frame(self) -> str:
        """Body-fixed (rotating) frame **name** of the SPH gravity body.

        ``self._body`` may be either a ``dict`` (frame under the ``"ref_name"``
        key) or a ``CelestialBody`` (``base_frame_name``, which returns the name
        without needing SPICE). A ``Frame`` object or a plain string is accepted
        from either source; SPICE needs the name, hence the normalization here.

        Returns
        -------
        str
            Frame name accepted by ``SpiceManager.get_xfrm``.

        Raises
        ------
        SphericalHarmonicsGravity.MissingBodyFixedFrameError
            If no body-fixed frame can be resolved from the configured body.
        """
        if isinstance(self._body, dict):
            frame = self._body.get("ref_name")
        else:
            frame = getattr(self._body, "base_frame_name", None)
            if frame is None:
                # Bodies that are not CelestialBody instances may still carry a
                # plain 'base_frame' attribute.
                frame = getattr(self._body, "base_frame", None)

        if frame is None:
            raise self.MissingBodyFixedFrameError(
                "Spherical harmonics needs the body-fixed frame of "
                f"{getattr(self._body, 'name', self._body)!r}. Supply it as "
                "the 'ref_name' key when passing a dict, or set 'base_frame' "
                "on the CelestialBody."
            )

        return frame.name if hasattr(frame, "name") else str(frame)

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
            else self._spice_xfrm_cached(
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
            else self._spice_xfrm_cached(
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
        current_state=None,
    ) -> np.ndarray:
        """
        Express an evaluation position relative to the harmonics body, in its
        body-fixed frame.

        Parameters
        ----------
        position : np.ndarray
            ``(3,)`` position of the evaluation body relative to the propagation
            origin, in km, expressed in the reference frame.
        epoch : float
            Epoch, in TDB seconds past J2000.
        R_b2rf : np.ndarray, optional
            ``(3, 3)`` body-fixed to reference-frame rotation. When given, its
            transpose is used instead of the SPICE lookup. Defaults to ``None``.
        current_state : dict, optional
            The propagated state, keyed by ``(name, spice_id)``; needed when the
            harmonics body is integrated in the state. Defaults to ``None``.

        Returns
        -------
        np.ndarray
            ``(3,)`` position relative to the harmonics body, in km, in its
            body-fixed frame.

        Raises
        ------
        PropagatedBodyStateMissingError
            If the harmonics body is integrated in the state but ``current_state``
            holds no position for it.
        """

        # Position of the SPH body w.r.t. the same origin and frame.
        # When the SPH body IS the origin the vector is identically zero,
        # so the SPICE lookup is skipped entirely.
        if self._sph_body_is_origin:
            r_eval_sph = position
        else:
            r_eval_sph = position - self._sph_body_position(current_state, epoch)

        # Rotate into SPH body-fixed frame
        if R_b2rf is not None:
            ext_to_bodyFixed_DCM = R_b2rf.T  # R_b2rf is orthogonal
        else:
            ext_to_bodyFixed_DCM = self._spice_xfrm_cached(
                self._ref_frame.name,
                self._sph_body_fixed_frame(),
                epoch,
            )

        r_eval_sph_bf = ext_to_bodyFixed_DCM @ r_eval_sph
        return r_eval_sph_bf

    def _apply_cs_from_state(self, current_state: dict):
        """
        Return the harmonic coefficients with any estimated values applied.

        An entry ``("cnm_<n>_<m>", spice_id)`` or ``("snm_<n>_<m>", spice_id)`` of
        the harmonics body replaces the nominal coefficient, for
        :math:`0 \\le m \\le n \\le` ``order`` (and :math:`m \\ge 1` for
        :math:`\\bar{S}_{nm}`). Only the canonical spelling is matched: ``cnm_2_0``,
        not ``cnm_02_0``.

        Parameters
        ----------
        current_state : dict or None
            The propagated state, keyed by ``(name, spice_id)``.

        Returns
        -------
        Cnm : np.ndarray
            Normalized cosine coefficients.
        Snm : np.ndarray
            Normalized sine coefficients.

        Notes
        -----
        The nominal arrays are copied only when at least one estimated coefficient
        is found; otherwise they are returned as they are.
        """
        if self._sph_body_spice_id is None or current_state is None:
            return self._Cnm, self._Snm

        # Iterate over the state's few keys instead of probing the dict for
        # every (n, m) pair — O(len(state)) instead of O(order^2) per call.
        Cnm, Snm = self._Cnm, self._Snm
        modified = False
        sid = self._sph_body_spice_id
        for key in current_state:
            if not (isinstance(key, tuple) and len(key) == 2 and key[1] == sid):
                continue
            name = key[0]
            if not isinstance(name, str):
                continue
            if name.startswith("cnm_"):
                is_c = True
            elif name.startswith("snm_"):
                is_c = False
            else:
                continue
            try:
                nn_str, mm_str = name[4:].split("_")
                nn, mm = int(nn_str), int(mm_str)
            except ValueError:
                continue
            # Match only the canonical key spelling and index range the
            # original per-(n, m) probing would have generated.
            if name[4:] != f"{nn}_{mm}" or not 0 <= mm <= nn <= self._order:
                continue
            if not is_c and mm == 0:
                continue
            if not modified:
                Cnm = self._Cnm.copy()
                Snm = self._Snm.copy()
                modified = True
            val = current_state[key]
            target = Cnm if is_c else Snm
            target[nn, mm] = float(val.values if hasattr(val, "values") else val)
        return Cnm, Snm

    @staticmethod
    def _build_R_b2j_iau(ra_deg: float, dec_deg: float, w_deg: float):
        """
        Build the body-fixed to J2000 rotation from IAU pole angles.

        The matrix is the transpose of the NAIF PCK J2000 to body-fixed
        rotation, a 3-1-3 sequence of frame rotations through the angles
        :math:`\\phi = 90^\\circ + \\alpha`,
        :math:`\\theta = 90^\\circ - \\delta` and :math:`\\psi = W`:

        .. math::

            \\mathbf{M} = \\left[\\mathbf{R}_3(\\psi)\\,\\mathbf{R}_1(\\theta)\\,
            \\mathbf{R}_3(\\phi)\\right]^{\\mathsf{T}},
            \\qquad
            \\mathbf{R}_3(x) =
            \\begin{bmatrix}
            \\cos x & \\sin x & 0 \\\\
            -\\sin x & \\cos x & 0 \\\\
            0 & 0 & 1
            \\end{bmatrix},

        where :math:`\\mathbf{R}_i(x)` is the passive (frame) rotation by :math:`x`
        about axis :math:`i`.

        Parameters
        ----------
        ra_deg : float
            Pole right ascension :math:`\\alpha`, in deg.
        dec_deg : float
            Pole declination :math:`\\delta`, in deg.
        w_deg : float
            Prime-meridian angle :math:`W`, in deg.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` matrix :math:`\\mathbf{M}`.
        """
        import math as _m
        import numpy as _np

        phi = _m.radians(90.0 + ra_deg)  # = 90 + RA
        theta = _m.radians(90.0 - dec_deg)  # = 90 - DEC
        psi = _m.radians(w_deg)  # = W

        cp, sp = _m.cos(phi), _m.sin(phi)
        ct, st = _m.cos(theta), _m.sin(theta)
        cw, sw = _m.cos(psi), _m.sin(psi)

        # passive rotation matrices Ri(angle); J2000 -> body is
        # Rz_psi @ Rx_theta @ Rz_phi, so body -> J2000 is its transpose
        Rz_phi = _np.array([[cp, sp, 0.0], [-sp, cp, 0.0], [0.0, 0.0, 1.0]])
        Rx_theta = _np.array([[1.0, 0.0, 0.0], [0.0, ct, st], [0.0, -st, ct]])
        Rz_psi = _np.array([[cw, sw, 0.0], [-sw, cw, 0.0], [0.0, 0.0, 1.0]])
        return (Rz_psi @ Rx_theta @ Rz_phi).T

    def _get_R_b2rf_effective(self, current_state: dict, epoch: float):
        """
        Return the body-fixed to reference-frame rotation of the harmonics body.

        When no pole parameter is in ``current_state``, or no PCK pole
        coefficients were loaded, the rotation is SPICE's (full PCK model).

        When ``("pole_ra", spice_id)``, ``("pole_dec", spice_id)``,
        ``("pole_pm", spice_id)`` or ``("pole_w0", spice_id)`` is present, the
        rotation is built in two steps:

        1. Body-fixed to J2000 with :meth:`_build_R_b2j_iau` from the angles of
           :meth:`~scarabaeus.BodyOrientation.iau_angles`. The secular PCK
           angles :math:`\\alpha = \\alpha_0 + \\alpha_1 T + \\alpha_2 T^2`,
           :math:`\\delta = \\delta_0 + \\delta_1 T + \\delta_2 T^2` and
           :math:`W = W_0 + W_1 d + W_2 d^2` (:math:`d` in days and :math:`T` in
           Julian centuries past J2000) are used for any angle not estimated. An
           estimated ``pole_ra`` or ``pole_dec`` (rad) replaces :math:`\\alpha` or
           :math:`\\delta`; an estimated ``pole_pm`` (deg/day) replaces :math:`W`
           with :math:`W_0 + \\mathrm{PM}\\,d`; an estimated ``pole_w0`` (rad)
           replaces :math:`W_0`. The optical measurement models use the same
           angles.
        2. J2000 to the reference frame with SPICE, so no frame is hard-coded.

        Parameters
        ----------
        current_state : dict or None
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Epoch, in TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` rotation :math:`\\mathbf{R}` with
            :math:`\\mathbf{v}_{rf} = \\mathbf{R}\\,\\mathbf{v}_{body}`.
        """
        body_frame = self._sph_body_fixed_frame()
        ref_frame = self._ref_frame.name

        # No estimation — delegate entirely to SPICE pxform.
        if (
            self._iau_ra_coeffs is None
            or self._sph_body_spice_id is None
            or current_state is None
        ):
            return self._spice_xfrm_cached(body_frame, ref_frame, epoch)

        values = {}
        for name in BodyOrientation.IAU_POLE_PARAMETERS:
            key = (name, self._sph_body_spice_id)
            if key in current_state:
                val = current_state[key]
                values[name] = float(val.values if hasattr(val, "values") else val)

        if not values:
            return self._spice_xfrm_cached(body_frame, ref_frame, epoch)

        # At least one pole parameter is estimated: the secular IAU angles at
        # this epoch (PCK bodvrd coefficients) with the estimated ones in place.
        # epoch is SPICE ET: seconds past J2000 TDB (verified by callers).
        angles, _ = BodyOrientation.iau_angles(
            epoch,
            None,
            (self._iau_ra_coeffs, self._iau_dec_coeffs, self._iau_pm_coeffs),
            **values,
        )
        ra_deg, dec_deg, w_deg = np.rad2deg(angles)

        # Step 1: body-fixed -> J2000 from IAU Euler angles.
        R_b2j2000 = SphericalHarmonicsGravity._build_R_b2j_iau(ra_deg, dec_deg, w_deg)

        # Step 2: J2000 -> reference frame via SPICE pxform.
        if ref_frame.upper() == "J2000":
            return R_b2j2000
        return self._spice_xfrm_cached("J2000", ref_frame, epoch) @ R_b2j2000

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
        # Body-fixed kernels + frame rotation run in rust; the SPICE lookups
        # and estimated-parameter handling stay in Python. There is no Python
        # implementation of the kernels, so the guard runs *first*: the state
        # reads below cast with float(), which raises an opaque TypeError on an
        # autograd ArrayBox rather than reporting that tracing is unsupported.
        self._require_rust_core(position)

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

        # A solid tide deforms the body, and the deformed mass changes the
        # field. This is the other half of the tide that moves a station on
        # the surface; both are driven by the same geometry, but this one by
        # the potential Love number k2 rather than by h2 and l2. It is a known
        # perturbation, not an estimated one, so it rides on the coefficients
        # here and the Jacobian below stays on the nominal field.
        if self._solid_tide is not None:
            self._Cnm, self._Snm = self._coefficients_with_tide(epoch, current_state)

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)

        r_bf = self._rf_transform(position, epoch, R_b2rf, current_state)
        if self.__dict__.get("_fuse_jacobian", False):
            # A position partial has been requested before (STM run):
            # compute it on the same Bnm pass — with the NOMINAL
            # coefficients, matching _compute_partial_by_position — and
            # cache the reference-frame result for the same-step partial call.
            acc_rf, jac_flat, bnm_real, bnm_imag = (
                self._rust_core.compute_accel_and_jacobian_rf(
                    r_bf,
                    R_b2rf,
                    self._mu,
                    self._Cnm,
                    self._Snm,
                    mu_jac=_mu_save,
                    cnm_jac=_Cnm_save,
                    snm_jac=_Snm_save,
                )
            )
            self._jac_rf_cache = (r_bf.tobytes(), jac_flat)
        else:
            acc_rf, bnm_real, bnm_imag = self._rust_core.compute_acceleration_rf(
                r_bf, R_b2rf, self._mu, self._Cnm, self._Snm
            )
        self._bnm_ext_real = bnm_real
        self._bnm_ext_imag = bnm_imag
        sph_acceleration = np.asarray(acc_rf)

        # A field centred anywhere but the origin also drags the origin, and the
        # propagation frame rides with the origin. What is felt is therefore the
        # difference of the two pulls, so the origin's share is subtracted here
        # — the indirect term of the classical third-body perturbation. The
        # origin sits far outside the field body, where the shape terms have
        # long died away, so its share is the monopole.
        #
        # ThreeBodyGravity used to supply this, back when the field body was
        # also left in the point-mass list and its attraction was consequently
        # counted twice. It is carried here now, alongside the direct term it
        # belongs with.
        if not self._sph_body_is_origin:
            r_origin_sph = self._sph_body_position(current_state, epoch)
            r_norm = float(np.linalg.norm(r_origin_sph))
            if r_norm > 0.0:
                sph_acceleration = sph_acceleration - (
                    self._mu * r_origin_sph / r_norm**3
                )

        self._Cnm, self._Snm = _Cnm_save, _Snm_save
        self._mu = _mu_save
        return sph_acceleration

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
        self._require_rust_core(position)

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)

        # From now on compute_acceleration fuses the jacobian into its Bnm pass,
        # so the same-step call lands in the cache below (already rotated into
        # the reference frame).
        self._fuse_jacobian = True
        r_bf = self._rf_transform(position, epoch, R_b2rf, current_state)
        cached = self.__dict__.get("_jac_rf_cache")
        if cached is not None and cached[0] == r_bf.tobytes():
            # Bnm side effects were already set by the fused accel call
            # on this same body-fixed position.
            jac_flat = cached[1]
        else:
            jac_flat, bnm_real, bnm_imag = self._rust_core.compute_jacobian_rf(
                r_bf, R_b2rf, self._mu, self._Cnm, self._Snm
            )
            self._bnm_ext_real = bnm_real
            self._bnm_ext_imag = bnm_imag
        return np.asarray(jac_flat).reshape((3, 3))

    def _compute_partial_by_sph_body_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray | None:
        """
        Compute ``d a_sph / d r_origin->sph_body``.

        Parameters
        ----------
        position : np.ndarray
            3x1 position of the evaluation body w.r.t. the propagation origin.
        current_state : dict
            Current full-state representation from the integrator.
        epoch : float
            Epoch at which the partial is evaluated.
        body : None
            Not explicitly used but included for interface consistency.

        Returns
        -------
        np.ndarray or None
            The 3x3 Jacobian in units of :math:`\\frac{1}{s^2}`, or None when
            the harmonics body is not integrated in the state — its position is
            then read from the ephemeris and is not a solve-for.

        Notes
        -----
        The direct term is evaluated at ``position - r_sph_body``, so it depends
        on the two positions only through that difference and contributes
        exactly the negative of :meth:`_compute_partial_by_position`. Without
        it an estimator moving the harmonics body sees no change in the
        acceleration it causes — the term the flyby geometry is most
        sensitive to.

        The indirect term subtracted in :meth:`compute_acceleration` depends on
        ``r_sph_body`` alone, so it adds its own monopole Jacobian here. It has
        no counterpart in :meth:`_compute_partial_by_position`, which is
        untouched by it.
        """
        if not self._sph_body_is_propagated:
            return None

        partial = -self._compute_partial_by_position(
            position, current_state, epoch, body
        )

        if not self._sph_body_is_origin:
            r_origin_sph = self._sph_body_position(current_state, epoch)
            r_norm = float(np.linalg.norm(r_origin_sph))
            if r_norm > 0.0:
                partial = partial - self._mu * (
                    np.eye(3) / r_norm**3
                    - 3.0 * np.outer(r_origin_sph, r_origin_sph) / r_norm**5
                )

        return partial

    def _compute_partial_by_C(
        self,
        position: np.ndarray,
        current_state: dict,
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
            Dictionary containing the current state information. Needed to
            locate the harmonics body when it is integrated in the state
            vector rather than read from the ephemeris, and for the rotation
            when the pole is estimated.

        epoch : float
            Epoch at which the transformation is performed.

        Returns
        -------
        partials_a_by_C : np.ndarray
            Partial derivatives of acceleration with respect to the C coefficients. Size of
            (order+1)x(order+1). Expressed in units of :math:`\\frac{km}{s^2}`.
        """
        self._require_rust_core(position)

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)
        r_bf = self._rf_transform(position, epoch, R_b2rf, current_state)
        partials_bf, bnm_real, bnm_imag = self._rust_core.compute_partial_by_c_bf(
            r_bf, self._mu
        )
        self._bnm_ext_real = bnm_real
        self._bnm_ext_imag = bnm_imag
        return self._jacobian_C_transform_rf(partials_bf, epoch, R_b2rf)

    def _compute_partial_by_S(
        self,
        position: np.ndarray,
        current_state: dict,
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
            Dictionary containing the current state information. Needed to
            locate the harmonics body when it is integrated in the state
            vector rather than read from the ephemeris, and for the rotation
            when the pole is estimated.

        epoch : float
            Epoch at which the transformation is performed.

        Returns
        -------
        partials_a_by_S : np.ndarray
            Partial derivatives of acceleration with respect to the S coefficients. Size of (order+1)x(order+1).
            Expressed in units :math:`\\frac{km}{s^2}`.
        """
        self._require_rust_core(position)

        # Rotation: use estimated pole if in state, otherwise SPICE
        R_b2rf = self._get_R_b2rf_effective(current_state, epoch)
        r_bf = self._rf_transform(position, epoch, R_b2rf, current_state)
        partials_bf, bnm_real, bnm_imag = self._rust_core.compute_partial_by_s_bf(
            r_bf, self._mu
        )
        self._bnm_ext_real = bnm_real
        self._bnm_ext_imag = bnm_imag
        return self._jacobian_C_transform_rf(partials_bf, epoch, R_b2rf)

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
    #   da_rf/d(param) = -A(param) @ a_rf  +  J_rf @ (A(param) @ r_rf)
    #
    # where:
    #   a_rf   = direct SH acceleration in reference frame
    #   J_rf   = da_rf/dr_rf  (3x3 position Jacobian in reference frame)
    #   r_rf   = position relative to the harmonics body in reference frame
    #   A(RA)  = M @ G_z @ M^T          (J2000 z-axis generator; M = J2000 -> RF)
    #   A(DEC) = -M @ G̃_x(Ω) @ M^T     (rotated x-generator; Ω = RA + 90°)
    #   A(W)   = G̃_z = R @ G_z @ R^T   (G_z expressed in reference frame)
    #   A(PM_rate) = A(W) * t_days * (π/180)
    #
    # Derivation: the body-fixed-to-J2000 DCM is R_j = R_z(Ω) * R_x(i) * R_z(W)
    # and R = M * R_j. dR_j/dRA = G_z * R_j,  dR_j/dDEC = -G̃_x(Ω) * R_j,
    # dR/dW = R * G_z.
    #
    # The pole turns the field about the harmonics body, so only the direct
    # term moves, evaluated at the position relative to that body; the
    # indirect term compute_acceleration subtracts for a non-origin body
    # does not depend on the pole.
    # ------------------------------------------------------------------

    _G_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def _pole_partial_terms(self, position, current_state, epoch, body):
        """
        Compute the terms shared by the pole partials, all in the reference frame.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.

        Returns
        -------
        a_rf : np.ndarray
            ``(3,)`` direct harmonics acceleration, in km/s^2, without the
            indirect term.
        J_rf : np.ndarray
            ``(3, 3)`` position Jacobian, in 1/s^2.
        r_rf : np.ndarray
            ``(3,)`` position relative to the harmonics body, in km.
        R : np.ndarray
            ``(3, 3)`` body-fixed to reference-frame rotation, from
            :meth:`_get_R_b2rf_effective`.
        M : np.ndarray
            ``(3, 3)`` J2000 to reference-frame rotation.
        """
        a_rf = np.asarray(self.compute_acceleration(position, current_state, epoch, body))
        J_rf = self._compute_partial_by_position(position, current_state, epoch, body)
        r_rf = position
        if not self._sph_body_is_origin:
            r_origin_sph = self._sph_body_position(current_state, epoch)
            r_rf = position - r_origin_sph
            r_norm = float(np.linalg.norm(r_origin_sph))
            if r_norm > 0.0:
                a_rf = a_rf + self._mu_from_state(current_state) * r_origin_sph / r_norm**3

        R = self._get_R_b2rf_effective(current_state, epoch)
        ref_frame = self._ref_frame.name
        M = (
            np.eye(3)
            if ref_frame.upper() == "J2000"
            else self._spice_xfrm_cached("J2000", ref_frame, epoch)
        )
        return a_rf, J_rf, r_rf, R, M

    def _compute_partial_by_pole_RA(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Compute the partial of the harmonics acceleration with respect to the
        pole right ascension :math:`\\alpha`.

        A change in :math:`\\alpha` rotates the field about the J2000 z-axis, so
        with :math:`\\mathbf{G} = \\mathbf{M}\\mathbf{G}_z\\mathbf{M}^{\\mathsf{T}}`
        (the z-axis generator expressed in the reference frame)

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial\\alpha}
            = \\mathbf{G}\\,\\mathbf{a} - \\mathbf{J}\\,\\mathbf{G}\\,\\mathbf{r},

        with :math:`\\mathbf{a}`, :math:`\\mathbf{J}`, :math:`\\mathbf{r}` and
        :math:`\\mathbf{M}` from :meth:`_pole_partial_terms`.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` partial, in km/s^2/rad.
        """
        a_rf, J_rf, r_rf, _, M = self._pole_partial_terms(
            position, current_state, epoch, body
        )

        # RA rotates about the J2000 z-axis, expressed in the reference frame.
        # Sign verified against dual-number differentiation in Rust.
        G_z = M @ self._G_Z @ M.T
        partial = G_z @ a_rf - J_rf @ (G_z @ r_rf)
        return partial.reshape(3, 1)

    def _compute_partial_by_pole_DEC(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Compute the partial of the harmonics acceleration with respect to the
        pole declination :math:`\\delta`.

        A change in :math:`\\delta` rotates the field about the node axis, whose
        generator :math:`\\tilde{\\mathbf{G}}_x` is built from the pole direction
        (the third column of the body-fixed to J2000 rotation) and expressed in
        the reference frame. Then

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial\\delta}
            = -\\tilde{\\mathbf{G}}_x\\,\\mathbf{a}
            + \\mathbf{J}\\,\\tilde{\\mathbf{G}}_x\\,\\mathbf{r},

        with :math:`\\mathbf{a}`, :math:`\\mathbf{J}` and :math:`\\mathbf{r}` from
        :meth:`_pole_partial_terms`. The generator is zero when the pole lies
        within :math:`10^{-10}` of the J2000 z-axis.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` partial, in km/s^2/rad.
        """
        a_rf, J_rf, r_rf, R, M = self._pole_partial_terms(
            position, current_state, epoch, body
        )

        # Build G̃_x(Ω) from the north-pole direction in J2000 (the third
        # column of the body-fixed -> J2000 DCM), then express it in the
        # reference frame. R uses the estimated pole if it is in the state.
        r3 = M.T @ R[:, 2]
        cd = float(np.sqrt(r3[0] ** 2 + r3[1] ** 2))  # = cos(DEC)
        if cd > 1e-10:
            sO = r3[0] / cd  # sin(Ω) = cos(RA)
            neg_cO = r3[1] / cd  # -cos(Ω) = sin(RA)
        else:
            sO, neg_cO = 0.0, 0.0

        G_x_tilde = np.array([[0.0, 0.0, sO], [0.0, 0.0, neg_cO], [-sO, -neg_cO, 0.0]])
        G_x_tilde = M @ G_x_tilde @ M.T

        partial = -G_x_tilde @ a_rf + J_rf @ (G_x_tilde @ r_rf)
        return partial.reshape(3, 1)

    def _compute_partial_by_pole_PM(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Compute the partial of the harmonics acceleration with respect to the
        prime-meridian rate PM.

        With :math:`\\tilde{\\mathbf{G}}_z = \\mathbf{R}\\mathbf{G}_z\\mathbf{R}^{\\mathsf{T}}`
        (the body spin axis generator expressed in the reference frame),

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial W}
            = \\tilde{\\mathbf{G}}_z\\,\\mathbf{a}
            - \\mathbf{J}\\,\\tilde{\\mathbf{G}}_z\\,\\mathbf{r},
            \\qquad
            \\frac{\\partial\\mathbf{a}}{\\partial\\mathrm{PM}}
            = \\frac{\\partial\\mathbf{a}}{\\partial W}\\,d\\,\\frac{\\pi}{180},

        since :math:`W = W_0 + \\mathrm{PM}\\,d` with :math:`d` the days past
        J2000.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000 (SPICE ephemeris time).
        body : Body
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` partial, in km/s^2 per deg/day.
        """
        da_dW = self._compute_partial_by_pole_W0(position, current_state, epoch, body)

        # Convert from rad to deg/day: dW/d(PM_deg_per_day) = t_days * (π/180)
        import spiceypy as _spice

        t_days = epoch / _spice.spd()  # epoch is SPICE ET (seconds past J2000 TDB)
        partial = da_dW * t_days * (np.pi / 180.0)
        return partial.reshape(3, 1)

    def _compute_partial_by_pole_W0(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Compute the partial of the harmonics acceleration with respect to the
        prime-meridian angle at J2000, :math:`W_0`.

        With :math:`\\tilde{\\mathbf{G}}_z = \\mathbf{R}\\mathbf{G}_z\\mathbf{R}^{\\mathsf{T}}`
        (the body spin axis generator expressed in the reference frame),

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial W_0}
            = \\frac{\\partial\\mathbf{a}}{\\partial W}
            = \\tilde{\\mathbf{G}}_z\\,\\mathbf{a}
            - \\mathbf{J}\\,\\tilde{\\mathbf{G}}_z\\,\\mathbf{r},

        since :math:`W_0` enters :math:`W` with unit weight.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000 (SPICE ephemeris time).
        body : Body
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` partial, in km/s^2/rad.
        """
        a_rf, J_rf, r_rf, R, _ = self._pole_partial_terms(
            position, current_state, epoch, body
        )

        # G̃_z = R @ G_z @ R^T: G_z expressed in the reference frame.
        # R uses the estimated pole if it is in the state.
        G_z_tilde = R @ self._G_Z @ R.T

        # da/dW [km/s^2/rad]:  -A(W) @ a_rf + J_rf @ (A(W) @ r_rf)
        # Sign verified against dual-number differentiation in Rust.
        da_dW = G_z_tilde @ a_rf - J_rf @ (G_z_tilde @ r_rf)
        return da_dW.reshape(3, 1)

    def _mu_from_state(self, current_state: dict) -> float:
        """Return the GM that :meth:`compute_acceleration` applies.

        Parameters
        ----------
        current_state : dict or None
            The propagated state, keyed by ``(name, spice_id)``.

        Returns
        -------
        float
            The estimated ``("gm", spice_id)`` of the harmonics body when present,
            otherwise the nominal GM, in km^3/s^2.
        """
        if self._sph_body_spice_id is not None and current_state is not None:
            _mu_key = ("gm", self._sph_body_spice_id)
            if _mu_key in current_state:
                _mu_val = current_state[_mu_key]
                return float(_mu_val.values if hasattr(_mu_val, "values") else _mu_val)
        return self._mu

    def _compute_partial_by_mu(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Compute the partial of the harmonics acceleration with respect to the
        body's gravitational parameter GM.

        Every term of the expansion is linear in GM, so the partial is
        :math:`\\mathbf{a}/\\mu`, with :math:`\\mu` the GM from
        :meth:`_mu_from_state`.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000.
        body : Body
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` partial, in 1/km^2 (km/s^2 per km^3/s^2).
        """
        # The active GM (estimated or nominal) — must match what
        # compute_acceleration used, since a_SH is linear in GM and da/dGM = a/GM.
        mu_active = self._mu_from_state(current_state)
        a_rf = self.compute_acceleration(position, current_state, epoch, body)
        partial = a_rf / mu_active
        return partial.reshape(3, 1)
