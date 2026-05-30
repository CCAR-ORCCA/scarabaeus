# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Units,
    ArrayWUnits,
    CelestialBody,
    Frame,
    ArrayWFrame,
)

import scarabaeus.utils.NumpyWrapper as np

# First, generate the default base units and frames
km, sec, rad = Units.get_units(["km", "sec", "rad"])
(J2000, ITRF93, ECLIPJ2000, IAUEARTH) = Frame.generate_common_frames()


#######################################################################
class ImpulsiveBurn(DynamicModel):
    """ Models an instantaneous velocity change (delta-v) maneuver.

    An impulsive burn approximates a finite-duration thrust as an
    instantaneous velocity discontinuity applied at epoch :math:`t_\\text{burn}`.
    The post-burn velocity is:

    .. math::

        \\mathbf{v}^+ = \\mathbf{v}^- + \\Delta\\mathbf{v}

    The delta-v vector can be supplied directly or decomposed into a scalar
    magnitude :math:`|\\Delta v|` and a unit direction vector :math:`\\hat{u}`:

    .. math::

        \\Delta\\mathbf{v} = |\\Delta v|\\,\\hat{u}

    Parameters
    ----------
    delta_v : ArrayWUnits, optional
        The delta-v vector applied during the burn (units: km/s).
    dv_magnitude : ArrayWUnits, optional
        Magnitude of the delta-v (units: km/s). Required when
        ``dv_unit_vector`` is provided instead of ``delta_v``.
    dv_unit_vector : ArrayWUnits, optional
        Unit direction vector of the delta-v (unitless). Required when
        ``dv_magnitude`` is provided instead of ``delta_v``.
    t_burn : float, optional
        Epoch at which the burn is applied (TDB seconds). Defaults to ``0``.
    base_frame : Frame, optional
        Reference frame for the delta-v vector. Defaults to J2000.

    Raises
    ------
    ValueError
        If neither ``delta_v`` nor both ``dv_magnitude`` and
        ``dv_unit_vector`` are provided.

    See Also
    --------
    scarabaeus.FiniteBurn : Continuous finite-duration thrust model.

    References
    ----------
    Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*
    (4th ed.). Microcosm Press.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        delta_v=None,
        dv_magnitude=None,
        dv_unit_vector=None,
        t_burn: float = 0,
        base_frame: Frame = None,
    ):
        # Initialize the delta-v vector
        if delta_v is not None:
            self._initialize_from_delta_v(delta_v)
        elif dv_magnitude is not None and dv_unit_vector is not None:
            self._initialize_from_magnitude_and_unit_vector(
                dv_magnitude, dv_unit_vector
            )
        else:
            raise ValueError(
                "Either delta_v or both dv_magnitude and dv_unit_vector must be provided."
            )

        # Set parameters
        self._t_burn = t_burn
        self.base_frame = base_frame

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def delta_v(self) -> ArrayWUnits:
        """
        The delta-v vector applied in the impulsive burn. Expressed in units of :math:`\\frac{km}{s}`.
        """
        return self._delta_v

    @property
    def dv_magnitude(self) -> ArrayWUnits:
        """
        The magnitude of the applied delta-v. Expressed in units of :math:`\\frac{km}{s}`.
        """
        return self._dv_magnitude

    @property
    def dv_unit_vector(self) -> ArrayWUnits:
        """
        The unit vector direction of the applied delta-v. Unitless.
        """
        return self._dv_unit_vector

    @property
    def t_burn(self) -> float:
        """
        The time at which the impulsive burn occurs.
        """
        return self._t_burn

    @property
    def base_frame(self) -> Frame:
        """
        The base frame at which the impulsive burn occurs
        """
        return self._base_frame

    @base_frame.setter
    def base_frame(self, input_val):
        """
        Set the reference frame for the impulsive burn.

        Parameters
        ----------
        input_val : Frame or None
            Reference frame. If ``None``, defaults to ``J2000``.
        """
        if input_val is None:
            input_val = J2000

        self._base_frame = input_val

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _initialize_from_delta_v(self, delta_v: ArrayWFrame):
        """
        Initializes the model from a provided delta-v vector.

        Parameters
        ----------
        delta_v : ArrayWFrame
            The 3x1 delta-v vector.
        """
        if not isinstance(delta_v, ArrayWFrame):
            raise TypeError(
                "The field 'delta_v' should contain an instance of ArrayWFrame."
            )
        self._delta_v = self._validate_input(delta_v.quantity, "delta_v")
        delta_v_val = ArrayWUnits.get_value_in_target_units(self.delta_v, km * sec**-1)
        self._dv_magnitude = np.linalg.norm(delta_v_val)
        self._dv_unit_vector = delta_v_val / self.dv_magnitude
        self.base_frame = delta_v.frame

    def _initialize_from_magnitude_and_unit_vector(
        self, dv_magnitude: float, dv_unit_vector: np.ndarray
    ) -> ArrayWUnits:
        """
        Initializes the model from a provided delta-v magnitude and unit vector.

        Parameters
        ----------
        dv_magnitude : ArrayWUnits
            The magnitude of the delta-v (size: 1x1, units: km/s).

        dv_unit_vector : ArrayWUnits
            Unit vector defining the burn direction (size: 3x1, unitless).
        """
        self._dv_magnitude = self._validate_input(dv_magnitude, "dv_magnitude")
        self._dv_unit_vector = self._validate_input(dv_unit_vector, "dv_unit_vector")
        self._delta_v = self.dv_magnitude * self.dv_unit_vector

    @staticmethod
    def _validate_input(value: ArrayWUnits, name: str) -> ArrayWUnits:
        """
        Validate that *value* is an ``ArrayWUnits`` instance.

        Parameters
        ----------
        value : ArrayWUnits
            The value to validate.
        name : str
            Name of the parameter (used in the error message).

        Returns
        -------
        ArrayWUnits
            The validated value, unchanged.

        Raises
        ------
        TypeError
            If *value* is not an instance of ``ArrayWUnits``.
        """
        if not isinstance(value, ArrayWUnits):
            raise TypeError(
                f"The field '{name}' should contain an instance of ArrayWUnits."
            )
        return value

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the acceleration due to the impulsive burn.

        Applies an instantaneous velocity change at the specified burn time..

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector in the reference frame (size: 3x1, units: km).

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch (units: sec).

        body : CelestialBody
            The celestial body associated with the simulation.

        Returns
        -------
        acc : np.ndarray
            The acceleration due to the impulsive burn (size: 3x1, units: km/s^2).
        """

        return np.zeros((3, 1))

    def _compute_partial_by_dv(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the partial derivative of the acceleration with respect to the delta-v.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : CelestialBody
            The celestial body associated with the simulation.

        Returns
        -------
        jacobian : np.ndarray
            The 3x3 Jacobian matrix of acceleration with respect to delta-v. Unitless.
        """

        return np.zeros((3, 3))

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
    ) -> np.ndarray:
        """
        Computes the partial derivative of the impulsive burn acceleration with respect to position.

        Since an impulsive burn directly modifies velocity rather than position,
        this derivative is always zero.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position. Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : CelestialBody
            The celestial body associated with the simulation.

        Returns
        -------
        partial_a_by_pos : np.ndarray
            A 3x3 matrix of zeros. Unitless.
        """
        return np.zeros((3, 3))

    def compute_TCM_covariance_gates(
        self,
        sig1=ArrayWUnits(0.00001 / 3, km / sec),
        sig2=ArrayWUnits(0.01 / 3, None),
        sig3=ArrayWUnits(0.000035 / 3, km / sec),
        sig4=ArrayWUnits(0.01 / 3, None),
    ) -> ArrayWUnits:
        """
        Computes the covariance matrix for a Trajectory Correction Maneuver (TCM).

        Computes the rotated covariance matrix P based on the input sigma values
        and the direction of the delta-v.

        Parameters
        ----------
        sig1 : ArrayWUnits, optional
            Sigma value for velocity uncertainty. Defaults to 0.00001 km/s.

        sig2 : ArrayWUnits, optional
            Sigma value for direction uncertainty Defaults to 0.01 unitless.

        sig3 : ArrayWUnits, optional
            Additional sigma for velocity. Defaults to 0.000035 km/s.

        sig4 : ArrayWUnits, optional
            Additional sigma for direction uncertainty. Defaults to 0.01 unitless.

        Returns
        -------
        covar_rotated : ArrayWUnits
            The rotated 3x3 covariance matrix. Expressed in units of :math:`\\frac{km^2}{s^2}`.
        """
        # Compute the magnitude of the vector y and normalize it
        magnitude_DV = self.dv_magnitude
        DV_hat = self.dv_unit_vector  # Normalize y

        # Define the original covariance matrix P in the local reference frame
        P11 = sig1**2 + magnitude_DV**2 * sig2**2
        P22 = sig3**2 + magnitude_DV**2 * sig4**2
        P33 = sig3**2 + magnitude_DV**2 * sig4**2
        units_P = P11.units
        P_local = np.diag(
            [
                P11.values,
                P22.values,
                P33.values,
            ]
        )

        # Helper function to create a rotation matrix that aligns the x-axis with the vector DV
        def rotation_matrix_from_vector(v: ArrayWUnits) -> np.ndarray:
            """
            Computes a rotation matrix that aligns the x-axis with a given vector.

            Constructs a 3x3 rotation matrix that aligns the x-axis of a reference
            frame with the input vector ``v``. The transformation is performed using Rodrigues'
            rotation formula.

            Parameters
            ----------
            v : ArrayWUnits
                Input vector to align with the x-axis (size: 3x1, unitless).

            Returns
            -------
            aligned : np.ndarray
                The 3x3 rotation matrix aligning the x-axis with ``v``.

            Notes
            -----
            * If ``v`` is already aligned with the x-axis, the identity matrix is returned.
            * The function normalizes the input vector before computing the transformation.
            """
            v = v.values  # Remove units
            x_axis = np.array([1, 0, 0])

            # Calculate the rotation axis (cross product between x-axis and vector v)
            cross_prod = np.cross(x_axis, v)
            sin_angle = np.linalg.norm(cross_prod)
            cos_angle = np.dot(x_axis, v)

            if (
                sin_angle == 0
            ):  # This handles the case where v is already aligned with the x-axis
                return np.eye(3)

            cross_prod = cross_prod / sin_angle  # Normalize rotation axis

            # Compute the skew-symmetric matrix of the cross product
            K = np.array(
                [
                    [0, -cross_prod[2], cross_prod[1]],
                    [cross_prod[2], 0, -cross_prod[0]],
                    [-cross_prod[1], cross_prod[0], 0],
                ]
            )

            # Compute the rotation matrix using Rodrigues' rotation formula
            R = np.eye(3) + sin_angle * K + (1 - cos_angle) * K @ K
            return R

        # Get the rotation matrix that aligns the x-axis with the direction of y_hat
        R = rotation_matrix_from_vector(DV_hat)

        # Rotate the local covariance matrix into the used reference frame
        # NOTE: R is from J2000 to the local frame, so we need to
        # transpose it to get the rotation from local to J2000
        P = ArrayWUnits(R.T @ P_local @ R, units_P)

        return P

    def compute_TCM_simplified_covariance(
        self,
        sig1=ArrayWUnits(0.01 / 3, None),
        sig2=ArrayWUnits(10e-5 / 3, km / sec),
        sig34=ArrayWUnits(2 * np.pi / 180 / 3, rad),
    ) -> ArrayWUnits:
        """
        Computes a simplified 3x3 covariance matrix for Trajectory Correction Maneuver (TCM) errors.

        Parameters
        ----------
        sig1 : ArrayWUnits, optional
            Proportional error component. Defaults to (0.01 / 3) unitless.

        sig2 : ArrayWUnits, optional
            Fixed velocity error. Defaults to (10e-5 / 3) km/s.

        sig34 : ArrayWUnits, optional
            Pointing error. Defaults to 2/3 degrees in radians.

        Returns
        -------
        iso_covar : ArrayWUnits
            The 3x3 isotropic covariance matrix. Expressed in units of :math:`\\frac{km^2}{s^2}`.
        """
        # Compute the magnitude of the delta-V vector (TCM magnitude)
        TCM_mag = ArrayWUnits(self.dv_magnitude, km / sec)

        # Compute the TCM error using the given model
        TCM_error = TCM_mag * (sig1 + np.sin(sig34)) + sig2

        # Generate an isotropic 3x3 covariance matrix
        P = ArrayWUnits(np.eye(3), None) * (
            TCM_error**2
        )  # Covariance is variance (sigma^2)

        return P
