# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import  (
    DynamicModel, 
    Frame, 
    StateArray, 
    CelestialBody,
    ArrayWUnits,
    Units
)
import scarabaeus.utils.NumpyWrapper as np
kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])


#######################################################################
class PointMassGravity(DynamicModel):
    """ Models the gravitational acceleration due to a point mass source using
    Keplerian dynamics.

    The gravitational acceleration is:

    .. math::

        \\mathbf{a} = -\\frac{\\mu}{\\|\\mathbf{r}\\|^3}\\,\\mathbf{r}

    where :math:`\\mu` is the gravitational parameter of the central body
    and :math:`\\mathbf{r}` is the position vector of the spacecraft relative
    to that body.  The gravitational parameter :math:`\\mu` can be estimated
    as a solve-for parameter; when the state vector contains a
    ``("gm", spice_id)`` entry, its value overrides the default at each
    function evaluation.

    Parameters
    ----------
    state_vector : StateArray
        State vector whose ``origin`` provides the central-body name,
        gravitational parameter, and SPICE ID.
    base_frame : Frame
        Reference frame in which accelerations are returned.

    See Also
    --------
    scarabaeus.StateArray : Represents state vectors with origin and reference frames.

    References
    ----------
    Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th Ed.
    Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*.
    """
    # -------------#
    # Constructor  #
    # -------------#
    def __init__(self,
                 state_vector : StateArray,
                 base_frame: Frame):
        self._origin_name = state_vector.origin.name
        self._ref_frame = base_frame
        self._origin_mu = ArrayWUnits.get_value_in_target_units(state_vector.origin.grav_param,km**3/sec**2)
        self._origin_spice_id = getattr(state_vector.origin, 'spice_id', None)

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def origin_name(self) -> str:
        """
        The name of the origin body.
        """
        return self._origin_name

    @property
    def origin_mu(self) -> float:
        """
        The gravitational parameter of the origin body. Assumed in units of
        of :math:`\\frac{km^3}{s^2}`.
        """
        return self._origin_mu

    @origin_mu.setter
    def origin_mu(self, input_val):
        """
        Set the gravitational parameter of the origin body.

        Parameters
        ----------
        input_val : float, np.ndarray, or None
            Gravitational parameter in km³·s⁻². If ``None``, defaults to
            ``ArrayWUnits(0, 'km**3/s**2')``.
        """
        if input_val is None:
            input_val = ArrayWUnits(0,'km**3/s**2')

        self._origin_mu = input_val

    @property
    def ref_frame(self) -> str:
        """
        The reference frame associated with the input state vector.
        """
        return self._ref_frame

    # endregion => Properties  #
    # -------------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the Keplerian acceleration acting on the spacecraft.

        The acceleration follows the inverse-square law, where the gravitational force
        exerted by the central body is proportional to the inverse cube of the position
        vector magnitude.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position relative to the central body.
            Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : CelestialBody
            The celestial body exerting the gravitational force.

        Returns
        -------
        acc_vec : np.ndarray
            3x1 vector describing the Keplerian acceleration. Assumed in :math:`\\frac{km}{s^2}`.
        """

        # Read updated GM from state if being estimated
        gm = self.origin_mu
        if self._origin_spice_id is not None:
            key = ("gm", self._origin_spice_id)
            if key in current_state:
                val = current_state[key]
                gm = float(val.values if hasattr(val, 'values') else val)

        # Compute the norm of the position vector
        position_norm = np.linalg.norm(position)

        # Calculate the Keplerian acceleration
        keplerian_acceleration = position_norm**-3 * (-gm * position)

        return keplerian_acceleration

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Computes the Jacobian matrix of the Keplerian acceleration with respect to position.

        This represents the sensitivity of the acceleration vector to changes in the
        spacecraft's position, which is useful for state transition modeling in orbit propagation.

        Parameters
        ----------
        position : np.ndarray
            3x1 vector describing the spacecraft's position relative to the central body.
            Assumed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : CelestialBody
            The celestial body exerting the gravitational force.

        Returns
        -------
        partial_a_by_pos : np.ndarray
            The 3x3 partial derivative (Jacobian) of acceleration with respect to
            position. Assumed in units of :math:`\\frac{1}{s^2}`.
        """

        # Initialize the total partial derivative matrix
        partial_a_by_pos = np.zeros((3, 3))

        # Identity matrix
        identity_matrix = np.eye(3)

        # Compute the norm of the position vector
        position_norm = np.linalg.norm(position)

        # Dyadic product of the origin-to-body vector
        rrt = np.outer(position,position)

        # Use estimated GM if present in state; fall back to nominal
        gm = self.origin_mu
        if self._origin_spice_id is not None:
            key = ("gm", self._origin_spice_id)
            if key in current_state:
                val = current_state[key]
                gm = float(val.values if hasattr(val, 'values') else val)
        C = -gm

        # Compute the partial derivative
        partial_a_by_pos = (C / position_norm**3) * identity_matrix - (
            3 * C / position_norm**5
        ) * rrt

        return partial_a_by_pos

    def _compute_partial_by_gm(
        self,
        position: ArrayWUnits,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> ArrayWUnits:
        """
        Computes the Jacobian matrix of the Keplerian acceleration with respect to GM.

        This represents the sensitivity of the acceleration vector to changes in the
        origin's GM, which is useful for state transition modeling in orbit propagation.

        Parameters
        ----------
        position : ArrayWUnits
            3x1 vector describing the spacecraft's position relative to the central body.
            Expressed in units of kilometers.

        current_state : dict
            Dictionary containing the current state parameters.

        epoch : float
            Current simulation epoch in seconds.

        body : CelestialBody
            The celestial body exerting the gravitational force.

        Returns
        -------
        partial_a_by_pos : np.ndarray
            The 3x1 partial derivative (Jacobian) of acceleration with respect to
            GM. Expressed in units of :math:`\\frac{km^3}{s^2}`.
        """

        # Compute the partial derivative
        position_norm = position.norm()
        partial_a_by_gm = -position * position_norm**-3

        return partial_a_by_gm.transpose()