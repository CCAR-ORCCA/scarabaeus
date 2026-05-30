# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Frame,
    Units,
    ArrayWUnits,
    EpochArray,
    Body,
    SpiceManager,
    StateArray,
    Spacecraft,
    nPlateModel,
)
from scarabaeus import constants
import warnings
import scarabaeus.utils.NumpyWrapper as np
import spiceypy

# ------------------#
#  Generate Units  #
# ------------------#
kg, km, sec, N, unitless = Units.get_units(["kg", "km", "sec", "N", "unitless"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class nPlateSRP(DynamicModel):
    """ Models the acceleration due to Solar Radiation Pressure (SRP) using an N-plate model.

        The spacecraft is decomposed into :math:`N` flat plates, each with its own
        area, surface normal, specular reflectivity :math:`\\rho_s`, and diffuse
        reflectivity :math:`\\rho_d`.  For each illuminated plate :math:`i`
        (i.e., :math:`\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{s}} > 0`), the
        SRP acceleration contribution is:

        .. math::

            \\mathbf{a}_i =
                -\\frac{\\nu\\, P_\\odot}{m}
                \\left(\\frac{r_\\odot}{r}\\right)^2
                A_i
                \\left(\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot\\right)
                \\left[
                    (1 - \\rho_s)\\,\\hat{\\mathbf{r}}_\\odot
                    + 2\\rho_s
                    \\left(\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot\\right)
                    \\hat{\\mathbf{n}}_i
                    + \\tfrac{2}{3}\\rho_d\\,\\hat{\\mathbf{n}}_i
                \\right]

        where :math:`\\nu` is the shadow factor (0 in eclipse, 1 in full Sun),
        :math:`P_\\odot` is the solar radiation pressure at 1 AU,
        :math:`r_\\odot / r` is the inverse-square scaling with heliocentric
        distance, and :math:`\\hat{\\mathbf{r}}_\\odot` is the unit vector from
        the spacecraft to the Sun (direction of incoming radiation); the dot
        product :math:`\\hat{\\mathbf{n}}_i \\cdot \\hat{\\mathbf{r}}_\\odot > 0`
        selects illuminated plates, and the leading minus sign yields acceleration
        directed away from the Sun.  The total acceleration is:

        .. math::

            \\mathbf{a}_\\text{SRP} = \\eta_\\text{SRP} \\sum_{i=1}^{N} \\mathbf{a}_i

        where :math:`\\eta_\\text{SRP}` is an estimable scale factor (default 1)
        retrieved from the state vector via a ``("eta_srp", spice_id)`` entry
        and can be solved for in the OD process.

        Parameters
        ----------
        state_vector : StateArray
            The state vector of the spacecraft containing position, velocity, and other estimated parameters.

        spacecraft : Spacecraft
            The spacecraft containing physical and plate model properties.

        See Also
        --------
        scarabaeus.StateArray : Represents state vectors with origins and reference frames.
        scarabaeus.Spacecraft : Defines spacecraft properties including mass and area.
        scarabaeus.Propagator : Uses this model when ``nPlate_SRP = True``.
        scarabaeus.CannonballSRP : Lower fidelity model for calculating SRP.

            Raises
            ------
            ValueError
                If any plate area in the n-plate model is not expressed in km².

            Notes
            -----
            The Sun’s position is retrieved dynamically from SPICE data.

        References
        ----------
        P. W. Kenneally, “High geometric fidelity solar radiation pressure modeling via graphics processing unit,”
        Master’s thesis, Colorado Boulder, 2016.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self, state_vector: StateArray, spacecraft: Spacecraft, base_frame: Frame
    ):
        # Origin and reference frame
        self._origin = state_vector.origin
        self._origin_name = self._origin.name
        self._frame = base_frame
        self._spacecraft = spacecraft
        # Precompute the SPICE name for the origin body so compute_acceleration does not
        # call eval() on every ODE function evaluation.
        self._origin_spice_name = getattr(constants, self._origin_name.upper()).spice_name

        # Body-specific parameters
        self._area = ArrayWUnits.get_value_in_target_units(self._spacecraft.area, km**2)
        self._ref_coeff = ArrayWUnits.get_value_in_target_units(
            self._spacecraft.ref_coeff, unitless
        )

        # Plate-specific parameters
        self._plate_model = self.spacecraft.n_plate_model

        # Constants
        self._P_SRP = ArrayWUnits.get_value_in_target_units(
            constants.srp_constant2, kg * km * sec**-2
        )

        # Cache eta_srp per body (spice_id -> float)
        self._eta_srp_map: dict[int, float] = {}
        self._initialize_eta_srp_map(state_vector)

        # Initialization check on the correct units for nPlate areas
        for ii in range(len(self.plate_model.names)):
            if self.plate_model.areas[ii].units != km**2:
                raise ValueError(f"units of nPlate areas not compliant.")

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def plate_model(self) -> nPlateModel:
        """ The associated nPlateModel used to calculate SRP. """
        return self._plate_model

    @property
    def spacecraft(self) -> Spacecraft:
        """
        The spacecraft associated with the SRP model.
        """
        return self._spacecraft

    @property
    def origin(self) -> Body:
        """
        The celestial body that serves as the origin for the state vector.
        """
        return self._origin

    @property
    def origin_name(self) -> str:
        """
        The name of the origin body.
        """
        return self._origin_name

    @property
    def frame(self) -> str:
        """
        The reference frame in which SRP calculations are performed.
        """
        return self._frame

    @property
    def area(self) -> float:
        """
        The total surface area of the spacecraft. Expressed in units of square meters.
        """
        return self._area

    @property
    def ref_coeff(self) -> float:
        """
        The reflectivity coefficient of the spacecraft. Unitless.
        """
        return self._ref_coeff

    @property
    def P_SRP(self) -> float:
        """
        The precomputed SRP constant for scaling factors. Expressed in units of :math:`\\frac{N}{m^2}`.
        """
        return self._P_SRP

    @property
    def eta_srp(self) -> float:
        """
        The SRP scale factor. Unitless.
        """
        return self._eta_srp

    @eta_srp.setter
    def eta_srp(self, input_val):
        if input_val is None:
            input_val = 0

        self._eta_srp = input_val

    # endregion => Properties  #
    # -------------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _initialize_eta_srp_map(self, state_array: StateArray) -> None:
        """
        Scan the StateArray and store eta_srp per body (by spice_id).

        Expects entries of the form:
            ('eta_srp', dim, est, dyn, owner, value)

        Only the first value is used if an array is provided.
        """
        for entry in state_array.state:
            if len(entry) < 6:
                continue

            name, _, _, _, owner, raw_val = entry
            if name != "eta_srp":
                continue

            spice_id = getattr(owner, "spice_id", None)
            if spice_id is None:
                raise ValueError(
                    "CannonballSRP: 'eta_srp' entry requires an owner with 'spice_id'. "
                    f"Got owner={owner!r}."
                )

            # Extract numeric value robustly
            if hasattr(raw_val, "quantity"):
                vals = raw_val.quantity.values
            elif hasattr(raw_val, "values"):
                vals = raw_val.values
            else:
                vals = raw_val

            eta = float(np.atleast_1d(vals)[0])
            self._eta_srp_map[spice_id] = eta

    def _get_eta_srp(self, current_state: dict, body: Body) -> float:
        """
        Resolve eta_srp for the given body.

        Priority:
        1. Cached from StateArray (self._eta_srp_map[spice_id])
        2. Dynamic from current_state[("eta_srp", spice_id)] if present
        3. Fallback: 1.0
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            if spice_id in self._eta_srp_map:
                return self._eta_srp_map[spice_id]

            key = ("eta_srp", spice_id)
            if key in current_state:
                val = current_state[key]
                return float(val.values if hasattr(val, "values") else val)

        return 1.0

    def inertial_to_body_DCM(self, epoch: EpochArray, tol: int = 50) -> float:
        """
        Computes the Direction Cosine Matrix (DCM) to transform vectors from the inertial
        frame to the spacecraft-fixed frame.

        Parameters
        ----------
        epoch : EpochArray
            The current epoch given in ephemeris time (seconds past J2000).

        tol : int, optional
            The time tolerance for retrieving attitude and angular velocity from the
            specified spacecraft clock time. Defaults to ``50``.

        Returns
        -------
        dcm : np.ndarray
            The DCM for transforming inertial vectors to spacecraft-fixed coordinates.

        Notes
        -----
        The ``epoch`` parameter is converted from ephemeris time to continuous encoded spacecraft clock "ticks". Due to
        this, the time tolerance should be given as sclock ticks and not seconds.
        """
        # Default DCM is the identity matrix
        default_DCM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Attempt to get the attitude from the SPICE Manager
        try:
            # Attempt to get the attitude from the SPICE Manager
            dcm, _, _ = SpiceManager.get_attitude(
                self.spacecraft.spice_id, epoch, self.frame, tol
            )
            return dcm.values

        except Exception:
            return default_DCM

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the acceleration due to solar radiation pressure (SRP) based on the
        n-plate model.

        Calculates the contribution of each spacecraft plate to the total
        acceleration, considering reflection, absorption, and the relative orientation
        of the spacecraft with respect to the Sun.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        total_srp_acc : np.ndarrays
            3x1 vector describing the total SRP acceleration. Assumed in units
            of :math:`\\frac{km}{s^2}`.
        """
        # Initialize acceleration
        nplate_srp_acceleration = np.array([0, 0, 0])

        # Compute the Sun-to-Origin vector from a Spice call
        sun2origin = SpiceManager.get_pos(
            self._origin_spice_name,
            epoch,
            self._frame,
            "SUN",
        ).values  # Assumed in km

        # Compute the Sun-to-Body vector
        sun2body = sun2origin + position

        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Get and validate the attitude mode
        attitude_mode = getattr(body, "attitudeMode", None)
        allowed_attitude_modes = (
            "CK",
            "nadir_pointing_to_sun",
            "inertialFixedAttitude",
        )
        if attitude_mode not in allowed_attitude_modes:
            raise ValueError(
                f"Invalid attitudeMode '{attitude_mode}'. Must be one of: {allowed_attitude_modes}."
            )

        # AttitudeMode cases
        if attitude_mode == "CK":
            quat_inertial2body = body.get_attitude(J2000, epoch)
            DCM_body2inertial = spiceypy.q2m(quat_inertial2body).T
        elif attitude_mode == "nadir_pointing_to_sun":
            warnings.warn(
                "nadir_pointing_to_sun attitude mode not yet fully implemented for nPlateSRP. "
                "It currently produces incorrect partial derivatives.",
                RuntimeWarning,
                stacklevel=2,
            )
            DCM_body2inertial = body.inertial_to_body_DCM_nadir_pointing_to_sun(
                self.origin_name,
                self._frame,
                ArrayWUnits(sun2body, km),
                current_state,
                epoch,
                body,
            )
        elif attitude_mode == "inertialFixedAttitude":
            if not (
                isinstance(body.fixedInertialAttitude, np.ndarray)
                and body.fixedInertialAttitude.shape == (3, 3)
            ):
                raise ValueError(
                    "fixedInertialAttitude must be a 3x3 numpy array. represent the rotation from Body fixed to inertial frame"
                )
            DCM_body2inertial = body.fixedInertialAttitude

        # Compute Body-to-Sun vectors in SC-fixed reference frame
        body2Sun_SCfixed = DCM_body2inertial.T @ -sun2body
        body2Sun_SCfixed_norm = np.linalg.norm(body2Sun_SCfixed)
        body2Sun_SCfixed_versor = body2Sun_SCfixed / body2Sun_SCfixed_norm

        # Get all normals — pass raw ET float to avoid creating an EpochArray object on
        # every ODE function evaluation (get_all_normals / _get_ck_normals accept floats).
        normals = np.array(
            self.plate_model.get_all_normals(epoch, self.spacecraft.spice_id)
        )

        # Compute nPlate SRP acceleration (Split into part 1, 2, and 3)
        # Compute part 1 of the acceleration formula, made of constants and squared distance
        m_t = self._spacecraft.mass(epoch).values  # Assumed in kg
        nplate_srp_acceleration_bf_p1 = (
            -(eta_srp / m_t) * self.P_SRP / (body2Sun_SCfixed_norm**2)
        )

        # Compute part 2 and 3 of the acceleration formula
        nplate_srp_acceleration_bf_p23 = np.array([0, 0, 0])
        T1 = np.zeros(
            3,
        )
        T2 = np.zeros(
            3,
        )
        T3 = np.zeros(
            3,
        )

        # Loop through plates of the nPlate model
        for ii in range(len(self.plate_model.names)):
            normal_plate_ii = normals[ii]
            normal_plate_ii_versor = normal_plate_ii / np.linalg.norm(normal_plate_ii)
            C_dref_plate_ii = self.plate_model.diff_ref_coeffs[ii]
            C_sref_plate_ii = self.plate_model.spec_ref_coeffs[ii]
            A_plate_ii = self.plate_model.areas[ii].values  # Assumed in km^2

            # Compute the angle theta between face's normal and body2Sun vector
            theta = self._angle_between(normal_plate_ii, body2Sun_SCfixed)
            # Compute the logic parameter H(theta)
            cos_theta = np.cos(theta)
            if cos_theta <= 0:
                H_theta = 0
            elif cos_theta > 0:
                H_theta = 1

            if H_theta == 1:
                part2 = A_plate_ii * cos_theta
                part3a = (1 - C_sref_plate_ii) * body2Sun_SCfixed_versor
                part3b = (
                    2
                    * ((1 / 3) * C_dref_plate_ii + C_sref_plate_ii * cos_theta)
                    * normal_plate_ii_versor
                )
                nplate_srp_acceleration_bf_p23 = (
                    nplate_srp_acceleration_bf_p23 + part2 * (part3a + part3b)
                )

                C_i = self.P_SRP * (eta_srp / m_t) * A_plate_ii
                A_i = -C_i * (1 - C_sref_plate_ii)
                B_i = (-2 / 3) * (C_dref_plate_ii) * C_i
                D_i = -2 * C_i * C_sref_plate_ii

                a = normal_plate_ii_versor[0]
                b = normal_plate_ii_versor[1]
                c = normal_plate_ii_versor[2]
                x = body2Sun_SCfixed[0]
                y = body2Sun_SCfixed[1]
                z = body2Sun_SCfixed[2]

                T1 += (
                    A_i
                    * (a * x + b * y + c * z)
                    / (np.sqrt(x**2 + y**2 + z**2) ** 4)
                    * np.array([x, y, z])
                )
                T2 += (
                    B_i
                    * (a * x + b * y + c * z)
                    / (np.sqrt(x**2 + y**2 + z**2) ** 3)
                    * np.array([a, b, c])
                )
                T3 += (
                    D_i
                    * (a * x + b * y + c * z) ** 2
                    / (np.sqrt(x**2 + y**2 + z**2) ** 4)
                    * np.array([a, b, c])
                )

        T_tot_bf = T1 + T2 + T3

        # Compute the total acceleration on the spacecraft (Multiply the constant terms outside the SUM)
        nplate_srp_acceleration_bf = (
            nplate_srp_acceleration_bf_p1 * nplate_srp_acceleration_bf_p23
        )

        # Transform the total acceleration from SC-fixed to original_inertial r.f.
        nplate_srp_acceleration = DCM_body2inertial @ nplate_srp_acceleration_bf
        T_tot = DCM_body2inertial @ T_tot_bf

        return T_tot

    def _compute_partial_by_position(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the SRP acceleration with respect to position.

        Determines how the SRP acceleration varies as a function of
        the spacecraft's position relative to the Sun.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        partial_a_by_pos : np.ndarrays
            The 3x3 Jacobian matrix of SRP acceleration with respect to position. Assumed in units
            of :math:`\\frac{1}{s^2}`.
        """
        # Initialize the total partial derivative matrix
        partial_a_by_pos_bf = np.zeros(
            (3, 3)
        )  # Jacobian matrix in 1/s^2, body-fixed frame
        partial_a_by_pos = np.zeros((3, 3))  # Jacobian matrix in 1/s^2, input frame

        # Compute the Sun-to-Origin vector
        sun2origin = SpiceManager.get_pos(
            self._origin_spice_name,
            epoch,
            self._frame,
            "SUN",
        ).values  # Assumed in km

        # Compute the Sun-to-Body vector
        sun2body = sun2origin + position

        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Get and validate the attitude mode
        attitude_mode = getattr(body, "attitudeMode", None)
        # allowed_attitude_modes = ("CK", "nadir_pointing_to_sun", "inertialFixedAttitude")
        # if attitude_mode not in allowed_attitude_modes:
        #     raise ValueError(
        #         f"Invalid attitudeMode '{attitude_mode}'. Must be one of: {allowed_attitude_modes}."
        #     )

        # AttitudeMode cases
        if attitude_mode == "CK":
            quat_inertial2body = body.get_attitude(J2000, epoch)
            DCM_body2inertial = spiceypy.q2m(quat_inertial2body).T
        elif attitude_mode == "nadir_pointing_to_sun":
            DCM_body2inertial = body.inertial_to_body_DCM_nadir_pointing_to_sun(
                self.origin_name,
                self._frame,
                ArrayWUnits(sun2body, km),
                current_state,
                epoch,
                body,
            )
        elif attitude_mode == "inertialFixedAttitude":
            if not (
                isinstance(body.fixedInertialAttitude, np.ndarray)
                and body.fixedInertialAttitude.shape == (3, 3)
            ):
                raise ValueError(
                    "fixedInertialAttitude must be a 3x3 numpy array. represent the rotation from Body fixed to inertial frame"
                )
            DCM_body2inertial = body.fixedInertialAttitude

        # Compute Body-to-Sun vectors in SC-fixed reference frame
        # body2Sun_SCfixed = (self.inertial_to_body_DCM(epoch) @ -sun2body.transpose()).transpose() #DCM_body2inertial.T @ -sun2body
        body2Sun_SCfixed = (DCM_body2inertial.T @ -sun2body.transpose()).transpose()
        body2Sun_SCfixed_norm = np.linalg.norm(body2Sun_SCfixed)

        # Compute part1 of the acceleration formula made of constants and squared distance
        dT_dxyz_tot = np.zeros((3, 3))
        normals = self.plate_model.get_all_normals(epoch, self.spacecraft.spice_id)

        for ii in range(len(self.plate_model.names)):
            normal_plate_ii_versor = normals[ii] / np.linalg.norm(normals[ii])
            C_dref_plate_ii = self.plate_model.diff_ref_coeffs[ii]
            C_sref_plate_ii = self.plate_model.spec_ref_coeffs[ii]
            A_plate_ii = self.plate_model.areas[ii].values  # Assumed in km^2

            # Compute C + A, B, D constant terms of the partials
            m_t = self._spacecraft.mass(epoch).values  # Assumed in kg
            C_i = self.P_SRP * (eta_srp / m_t) * A_plate_ii
            A_i = -C_i * (1 - C_sref_plate_ii)
            B_i = (-2 / 3) * (C_dref_plate_ii) * C_i
            D_i = -2 * (C_sref_plate_ii) * C_i

            # For simplicity, extract pos + face normal vectors
            a = normal_plate_ii_versor[0]
            b = normal_plate_ii_versor[1]
            c = normal_plate_ii_versor[2]
            x = body2Sun_SCfixed[0]
            y = body2Sun_SCfixed[1]
            z = body2Sun_SCfixed[2]
            r = body2Sun_SCfixed_norm

            abc_9 = np.array(
                [normal_plate_ii_versor, normal_plate_ii_versor, normal_plate_ii_versor]
            )

            # Compute partials of T1 terms
            dT1x_dx = (b * y + c * z) * (-4 * x**2 + r**2) + 2 * a * x * (
                -2 * x**2 + r**2
            )
            dT1x_dy = x * (-4 * y * (a * x + c * z) + b * (r**2 - 4 * y**2))
            dT1x_dz = x * (-4 * z * (a * x + b * y) + c * (r**2 - 4 * z**2))
            dT1y_dx = y * (-4 * x * (b * y + c * z) + a * (r**2 - 4 * x**2))
            dT1y_dy = (a * x + c * z) * (r**2 - 4 * y**2) + 2 * b * y * (
                r**2 - 2 * y**2
            )
            dT1y_dz = y * (-4 * z * (a * x + b * y) + c * (r**2 - 4 * z**2))
            dT1z_dx = z * (-4 * x * (b * y + c * z) + a * (r**2 - 4 * x**2))
            dT1z_dy = z * (-4 * y * (a * x + c * z) + b * (r**2 - 4 * y**2))
            dT1z_dz = (a * x + b * y) * (r**2 - 4 * z**2) + 2 * c * z * (
                r**2 - 2 * z**2
            )
            row_1 = np.array([dT1x_dx, dT1x_dy, dT1x_dz])
            row_2 = np.array([dT1y_dx, dT1y_dy, dT1y_dz])
            row_3 = np.array([dT1z_dx, dT1z_dy, dT1z_dz])
            T1 = np.array([row_1, row_2, row_3])

            dT1_dxyz = (A_i / (r**6)) * T1

            # Compute partials of T2 terms
            dT2_dx = -3 * x * (b * y + c * z) + a * (r**2 - 3 * x**2)
            dT2_dy = -3 * y * (a * x + c * z) + b * (r**2 - 3 * y**2)
            dT2_dz = -3 * z * (a * x + b * y) + c * (r**2 - 3 * z**2)
            vals = np.array([dT2_dx, dT2_dy, dT2_dz])
            T2 = np.array([vals, vals, vals])

            dT2_dxyz = (B_i / (r**5)) * (T2 * abc_9.transpose())

            # Compute partials of T3 terms
            dT3_dx = (
                -2
                * (a * x + b * y + c * z)
                * (2 * x * (b * y + c * z) + a * (2 * x**2 - r**2))
            )
            dT3_dy = (
                -2
                * (a * x + b * y + c * z)
                * (2 * y * (a * x + c * z) + b * (2 * y**2 - r**2))
            )
            dT3_dz = (
                -2
                * (a * x + b * y + c * z)
                * (2 * z * (a * x + b * y) + c * (2 * z**2 - r**2))
            )
            T3 = np.zeros((3, 3))
            T3[0] = np.array([dT3_dx, dT3_dy, dT3_dz])
            T3[1] = T3[0]
            T3[2] = T3[0]
            dT3_dxyz = (D_i / (r**6)) * (T3 * abc_9.transpose())

            # Put together the partials of T1, T2, and T3 terms
            dT_dxyz = dT1_dxyz + dT2_dxyz + dT3_dxyz

            # Compute the angle theta between face's normal and body2Sun vector
            theta = self._angle_between(normal_plate_ii_versor, body2Sun_SCfixed)

            # Compute the logic parameter H(theta)
            cos_theta = np.cos(theta)
            if cos_theta <= 0:
                H_theta = 0
            elif cos_theta > 0:
                H_theta = 1

            # Compute the plate component on the acceleration (The terms within the SUM)
            if H_theta == 1:
                dT_dxyz_tot = dT_dxyz_tot + dT_dxyz

        partial_a_by_pos_bf = dT_dxyz_tot

        # Transform the partial acceleration matrix from SC-fixed to original_inertial r.f.
        # partial_a_by_pos = self.inertial_to_body_DCM(epoch) @ partial_a_by_pos_bf

        DCM = DCM_body2inertial.T
        T = dT_dxyz_tot.T
        daix_dxi = (
            DCM[0, 0]
            * (T[0, 0] * -DCM[0, 0] + T[1, 0] * -DCM[1, 0] + T[2, 0] * -DCM[2, 0])
            + DCM[1, 0]
            * (T[0, 1] * -DCM[0, 0] + T[1, 1] * -DCM[1, 0] + T[2, 1] * -DCM[2, 0])
            + DCM[2, 0]
            * (T[0, 2] * -DCM[0, 0] + T[1, 2] * -DCM[1, 0] + T[2, 2] * -DCM[2, 0])
        )

        daix_dyi = (
            DCM[0, 0]
            * (T[0, 0] * -DCM[0, 1] + T[1, 0] * -DCM[1, 1] + T[2, 0] * -DCM[2, 1])
            + DCM[1, 0]
            * (T[0, 1] * -DCM[0, 1] + T[1, 1] * -DCM[1, 1] + T[2, 1] * -DCM[2, 1])
            + DCM[2, 0]
            * (T[0, 2] * -DCM[0, 1] + T[1, 2] * -DCM[1, 1] + T[2, 2] * -DCM[2, 1])
        )

        daix_dzi = (
            DCM[0, 0]
            * (T[0, 0] * -DCM[0, 2] + T[1, 0] * -DCM[1, 2] + T[2, 0] * -DCM[2, 2])
            + DCM[1, 0]
            * (T[0, 1] * -DCM[0, 2] + T[1, 1] * -DCM[1, 2] + T[2, 1] * -DCM[2, 2])
            + DCM[2, 0]
            * (T[0, 2] * -DCM[0, 2] + T[1, 2] * -DCM[1, 2] + T[2, 2] * -DCM[2, 2])
        )

        daiy_dxi = (
            DCM[0, 1]
            * (T[0, 0] * -DCM[0, 0] + T[1, 0] * -DCM[1, 0] + T[2, 0] * -DCM[2, 0])
            + DCM[1, 1]
            * (T[0, 1] * -DCM[0, 0] + T[1, 1] * -DCM[1, 0] + T[2, 1] * -DCM[2, 0])
            + DCM[2, 1]
            * (T[0, 2] * -DCM[0, 0] + T[1, 2] * -DCM[1, 0] + T[2, 2] * -DCM[2, 0])
        )

        daiy_dyi = (
            DCM[0, 1]
            * (T[0, 0] * -DCM[0, 1] + T[1, 0] * -DCM[1, 1] + T[2, 0] * -DCM[2, 1])
            + DCM[1, 1]
            * (T[0, 1] * -DCM[0, 1] + T[1, 1] * -DCM[1, 1] + T[2, 1] * -DCM[2, 1])
            + DCM[2, 1]
            * (T[0, 2] * -DCM[0, 1] + T[1, 2] * -DCM[1, 1] + T[2, 2] * -DCM[2, 1])
        )

        daiy_dzi = (
            DCM[0, 1]
            * (T[0, 0] * -DCM[0, 2] + T[1, 0] * -DCM[1, 2] + T[2, 0] * -DCM[2, 2])
            + DCM[1, 1]
            * (T[0, 1] * -DCM[0, 2] + T[1, 1] * -DCM[1, 2] + T[2, 1] * -DCM[2, 2])
            + DCM[2, 1]
            * (T[0, 2] * -DCM[0, 2] + T[1, 2] * -DCM[1, 2] + T[2, 2] * -DCM[2, 2])
        )

        daiz_dxi = (
            DCM[0, 2]
            * (T[0, 0] * -DCM[0, 0] + T[1, 0] * -DCM[1, 0] + T[2, 0] * -DCM[2, 0])
            + DCM[1, 2]
            * (T[0, 1] * -DCM[0, 0] + T[1, 1] * -DCM[1, 0] + T[2, 1] * -DCM[2, 0])
            + DCM[2, 2]
            * (T[0, 2] * -DCM[0, 0] + T[1, 2] * -DCM[1, 0] + T[2, 2] * -DCM[2, 0])
        )

        daiz_dyi = (
            DCM[0, 2]
            * (T[0, 0] * -DCM[0, 1] + T[1, 0] * -DCM[1, 1] + T[2, 0] * -DCM[2, 1])
            + DCM[1, 2]
            * (T[0, 1] * -DCM[0, 1] + T[1, 1] * -DCM[1, 1] + T[2, 1] * -DCM[2, 1])
            + DCM[2, 2]
            * (T[0, 2] * -DCM[0, 1] + T[1, 2] * -DCM[1, 1] + T[2, 2] * -DCM[2, 1])
        )

        daiz_dzi = (
            DCM[0, 2]
            * (T[0, 0] * -DCM[0, 2] + T[1, 0] * -DCM[1, 2] + T[2, 0] * -DCM[2, 2])
            + DCM[1, 2]
            * (T[0, 1] * -DCM[0, 2] + T[1, 1] * -DCM[1, 2] + T[2, 1] * -DCM[2, 2])
            + DCM[2, 2]
            * (T[0, 2] * -DCM[0, 2] + T[1, 2] * -DCM[1, 2] + T[2, 2] * -DCM[2, 2])
        )

        row_1 = [daix_dxi, daix_dyi, daix_dzi]
        row_2 = [daiy_dxi, daiy_dyi, daiy_dzi]
        row_3 = [daiz_dxi, daiz_dyi, daiz_dzi]
        partial_a_by_pos_math = np.array([row_1, row_2, row_3])

        return partial_a_by_pos_math

    def _compute_partial_by_eta(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Computes the partial derivative of SRP acceleration with respect to the SRP scale factor (`eta_srp`).

        This method determines how changes in the SRP scale factor affect the acceleration.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : Spacecraft
            The spacecraft for which SRP acceleration is calculated.

        Returns
        -------
        partial_by_eta : np.ndarray
            3x1 vector describing the partial derivative of SRP acceleration with respect to ``eta_srp``. Assumed in units
            of :math:`\\frac{km}{s^2}`.
        """
        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Compute the partial derivative
        partial_a_by_eta = (
            self.compute_acceleration(position, current_state, epoch, body) / eta_srp
        )

        # Return the transposed result
        return partial_a_by_eta.transpose()

    def _angle_between(
        self,
        vec_1: np.ndarray,
        vec_2: np.ndarray,
    ) -> float:
        """Compute the angle between two vectors

        Parameters
        ----------
        vec_1 : np.ndarray
            First vector.

        vec_2 : np.ndarray
            Second vector.

        Returns
        -------
        angle : float
            Angle between the two vectors in radians
        """
        # Compute the angle between two vectors
        cross = np.linalg.norm(np.cross(vec_1, vec_2))
        dot = np.dot(vec_1, vec_2)

        return np.arctan2(cross, dot)
