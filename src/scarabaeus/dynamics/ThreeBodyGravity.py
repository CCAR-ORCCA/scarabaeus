# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    DynamicModel,
    Frame,
    StateArray,
    CelestialBody,
    SpiceManager,
    ArrayWUnits,
    Units,
)
import scarabaeus.utils.NumpyWrapper as np

kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])


#######################################################################
class ThreeBodyGravity(DynamicModel):
    """ Models third-body gravitational perturbations from one or more point masses.

    For each listed point mass, the acceleration follows the standard
    three-body formulation:

    .. math::

        \\mathbf{a}_{3b} = \\mu_i \\left(
            \\frac{\\mathbf{r}_{\\mathrm{body} \\to i}}{\\|\\mathbf{r}_{\\mathrm{body} \\to i}\\|^3}
            - \\frac{\\mathbf{r}_{\\mathrm{origin} \\to i}}{\\|\\mathbf{r}_{\\mathrm{origin} \\to i}\\|^3}
        \\right)

    If a third body is propagated in the :class:`StateArray` (i.e., its
    position is a dynamic state), its position is taken from
    ``current_state`` rather than from SPICE, enabling full coupled
    propagation.  Each :math:`\\mu_i` can also be estimated as a solve-for
    parameter; when the state vector contains a ``("gm", spice_id)`` entry
    for body :math:`i`, its value overrides the ephemeris default at each
    function evaluation.

    Parameters
    ----------
    state_vector : StateArray
        State vector of the primary spacecraft, used to determine the
        propagation origin and to identify any propagated third bodies.
    pnt_masses : list of str
        Names of the point-mass bodies (matching keys in
        ``scarabaeus.Constants``), e.g. ``["SUN", "MOON"]``.
    base_frame : Frame
        Reference frame in which accelerations are returned.

    Raises
    ------
    ValueError
        If the gravitational parameter units for any listed point mass are
        not in km³·s⁻².

    See Also
    --------
    scarabaeus.PointMassGravity : Single-body Newtonian gravity.
    scarabaeus.ForceModelTranslation : Aggregator that instantiates this model.

    References
    ----------
    Vallado, D. A. *Fundamentals of Astrodynamics and Applications*, 4th Ed.,
    Microcosm Press, 2013.
    """

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        state_vector: StateArray,
        pnt_masses: list,
        base_frame: Frame,
    ):
        self._state_vector = state_vector
        self._origin_name = state_vector.origin.name
        self._origin_mu = ArrayWUnits.get_value_in_target_units(
            state_vector.origin.grav_param, km**3 / sec**2
        )
        self._ref_frame = base_frame
        self._pnt_masses = pnt_masses

        # Cache which third bodies are propagated in the state vector
        # key: UPPERCASE body name (matching Constants key), value: owner object
        self._propagated_third_bodies = {}
        for name, _, _, _, owner, _ in state_vector.state:
            # If a position state is defined for a CelestialBody, mark as propagated
            if name == "position" and isinstance(owner, CelestialBody):
                self._propagated_third_bodies[owner.name.upper()] = owner

        # Cache integer SPICE ids for third-body GM state lookups
        self._pnt_mass_spice_ids: dict[str, int | None] = {}
        for pm in self._pnt_masses:
            # match constants query to barycenter or body center
            if pm.upper().endswith('BARYCENTER'):
                pm_query = pm.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
            else:
                pm_query = pm

            # make sure correct units
            pm_info = getattr(constants, pm_query.upper())
            if pm_info.GM.units != km**3 * sec**-2:
                raise ValueError(f"units of GM of {pm} not compliant.")

            # cache
            spice_name = pm_info.spice_name
            try:
                self._pnt_mass_spice_ids[pm] = SpiceManager.name_to_ID(spice_name)
            except Exception:
                self._pnt_mass_spice_ids[pm] = None

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def origin_name(self) -> str:
        """Name of the propagation-origin celestial body."""
        return self._origin_name

    @property
    def origin_mu(self) -> float:
        """Gravitational parameter of the origin body in km³·s⁻²."""
        return self._origin_mu

    @property
    def ref_frame(self) -> Frame:
        """Reference frame in which third-body accelerations are returned."""
        return self._ref_frame

    @property
    def pnt_masses(self) -> list[str]:
        """Ordered list of point-mass body name strings (Constants keys)."""
        return self._pnt_masses

    # ----------------#
    # region Helpers  #
    # ----------------#
    def _get_origin_to_point_mass(
        self,
        point_mass_name: str,
        current_state: dict,
        epoch: float,
    ) -> np.ndarray:
        """
        Return r_origin->point_mass (km).

        Logic
        -----
        1. If `point_mass_name` is marked as propagated in `_propagated_third_bodies`,
        its position MUST be found in `current_state` under key
        ("position", spice_id). If not, raise.
        2. Otherwise, fall back to SPICE ephemerides (legacy behavior).

        Parameters
        ----------
        point_mass_name : str
            Name key used in :class:`~scarabaeus.utils.constants` (e.g. ``'EARTH'``).
        current_state : dict
            Current full-state representation from the integrator.
            Expected keys: (name, spice_id).
        epoch : float
            Current epoch [TDB seconds].

        Returns
        -------
        np.ndarray
            3-element position vector [km] from origin to point mass.
        """
        name_uc = point_mass_name.upper()
        owner = self._propagated_third_bodies.get(name_uc)

        # 1) Propagated third body: must exist in current_state as ("position", spice_id)
        if owner is not None:
            pm_spice_id = getattr(owner, "spice_id", None)
            if pm_spice_id is None:
                raise ValueError(
                    f"ThreeBodyGravity: propagated body '{point_mass_name}' "
                    f"has no spice_id defined."
                )

            key = ("position", pm_spice_id)
            if key not in current_state:
                raise ValueError(
                    f"ThreeBodyGravity: body '{point_mass_name}' is marked as propagated "
                    f"but its position entry {key!r} is missing in current_state. "
                    "Ensure it is included in the StateArray and direct representation."
                )

            val = current_state[key]
            arr = val.values if hasattr(val, "values") else np.asarray(val)
            return np.asarray(arr, dtype=float)

        # 2) Not propagated: use SPICE (legacy behavior)
        if point_mass_name.upper().endswith('BARYCENTER'):
            # requested barycenter -> query by name and get barycenter ID
            pm_query = point_mass_name.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
            trgt_id = getattr(constants, pm_query.upper()).barycenter_id
            trgt_body = SpiceManager.id2name(trgt_id)
        else:
            # requested body -> query by name and get body center ID
            trgt_id = getattr(constants, point_mass_name.upper()).body_center_id
            trgt_body = SpiceManager.id2name(trgt_id)

        # get origin's SPICE ID
        origin_body_spice_name = getattr(constants, self._origin_name.upper()).spice_name

        r_origin_pm = SpiceManager.get_pos(trgt_body,
                                           epoch,
                                           self._ref_frame,
                                           origin_body_spice_name).values  # km
        return np.asarray(r_origin_pm, dtype=float)

    # ----------------#
    # region Methods  #
    # ----------------#
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Compute third-body gravitational acceleration.

        If a third body is propagated in the state vector, its position is taken
        from ``current_state``; otherwise SPICE is used.
        """
        if not isinstance(epoch, float):
            raise TypeError(f"Argument [epoch] must be float. Received: {type(epoch)}.")

        third_body_acceleration = np.zeros(3, dtype=float)
        for point_mass in self._pnt_masses:
            pm_spice_id = self._pnt_mass_spice_ids.get(point_mass)
            if pm_spice_id is not None:
                # Read updated GM from state if being estimated
                key = ("gm", pm_spice_id)
                if key in current_state:
                    val = current_state[key]
                    point_mass_mu = float(val.values if hasattr(val, 'values') else val)
                # otherwise use constants values
                else:
                    if point_mass.upper().endswith('BARYCENTER'):
                        # requested barycenter -> query by name and get barycenter GM
                        pm_query = point_mass.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
                        point_mass_mu = getattr(constants, pm_query).GM_barycenter.values
                    else:
                        # requested body -> query by name and get body GM
                        pm_query = point_mass.upper()
                        point_mass_mu = getattr(constants, pm_query).GM.values

            # r_origin->pm
            origin2pm = self._get_origin_to_point_mass(
                point_mass_name=point_mass,
                current_state=current_state,
                epoch=epoch,
            )
            origin2pm_norm = np.linalg.norm(origin2pm)

            body2pm = -position + origin2pm
            body2pm_norm = np.linalg.norm(body2pm)

            pm_acc = point_mass_mu * (-(origin2pm / origin2pm_norm**3) + (body2pm / body2pm_norm**3))

            third_body_acceleration += pm_acc

        return third_body_acceleration

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Compute da_3b/dr for third-body gravity.

        Uses propagated third-body positions when available; otherwise SPICE.
        """
        partial_a_by_pos = np.zeros((3, 3), dtype=float)
        I = np.eye(3)

        for point_mass in self._pnt_masses:
            pm_spice_id = self._pnt_mass_spice_ids.get(point_mass)
            if pm_spice_id is not None:
                # Read updated GM from state if being estimated
                key = ("gm", pm_spice_id)
                if key in current_state:
                    val = current_state[key]
                    point_mass_mu = float(val.values if hasattr(val, 'values') else val)
                # otherwise use constants values
                else:
                    if point_mass.upper().endswith('BARYCENTER'):
                        # requested barycenter -> query by name and get barycenter GM
                        pm_query = point_mass.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
                        point_mass_mu = getattr(constants, pm_query).GM_barycenter.values
                    else:
                        # requested body -> query by name and get body GM
                        pm_query = point_mass.upper()
                        point_mass_mu = getattr(constants, pm_query).GM.values

            # r_origin->pm from propagated state or SPICE
            origin2pm = self._get_origin_to_point_mass(
                point_mass_name=point_mass,
                current_state=current_state,
                epoch=epoch,
            )

            body2pm = -position + origin2pm
            r = body2pm
            r_norm = np.linalg.norm(r)

            if r_norm == 0.0:
                continue

            rrt = np.outer(r, r)
            dadr = -point_mass_mu * I / r_norm**3 + 3 * point_mass_mu * rrt / r_norm**5

            partial_a_by_pos += dadr

        return partial_a_by_pos

    def _compute_partial_by_third_body_position(
        self,
        position: np.ndarray,  # r_origin->body (the accelerated body)
        current_state: dict,
        epoch: float,
        body: CelestialBody,
        point_mass_name: str,  # e.g. "EARTH"
    ) -> np.ndarray | None:
        """
        Compute d a_3b / d r_origin->pm for a specific third body (point mass).

        Returns 3x3 if that point mass is propagated in the state.
        Returns None if not propagated (because then origin2pm comes from SPICE, not a solve-for).
        """
        # Check if point mass is propagated
        name_uc = point_mass_name.upper()
        owner = self._propagated_third_bodies.get(name_uc)
        if owner is None:
            return None  # not a state variable -> no partial in STM

        # Initialize
        if point_mass_name.upper().endswith('BARYCENTER'):
            pm_query = point_mass_name.upper().removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
            mu = getattr(constants, pm_query).GM.values
        else:
            pm_query = point_mass_name.upper()
            mu = getattr(constants, pm_query).GM.values

        I = np.eye(3)

        origin2pm = self._get_origin_to_point_mass(
            point_mass_name, current_state, epoch
        )
        r0 = origin2pm
        r = origin2pm - position  # body2pm

        r0n = np.linalg.norm(r0)
        rn = np.linalg.norm(r)
        if r0n == 0.0 or rn == 0.0:
            return np.zeros((3, 3), dtype=float)

        # J(u) = d/du [ u / ||u||^3 ] = I/||u||^3 - 3 u u^T / ||u||^5
        def J(u, un):
            uut = np.outer(u, u)
            return I / un**3 - 3.0 * uut / un**5

        # a = mu * ( -f(r0) + f(r) ), with r = r0 - position
        # da/dr0 = mu * ( -J(r0) + J(r) * d(r)/d(r0) )
        # d(r)/d(r0) = I
        dadr0 = mu * (-J(r0, r0n) + J(r, rn))
        return dadr0

    def _compute_partial_by_gm(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body,
        point_mass_name: str,
    ) -> np.ndarray:
        """
        Compute da_3b/dmu for a specific third-body gravitational parameter.

        Parameters
        ----------
        position : np.ndarray
            3-element position vector [km] of the accelerated body wrt origin.
        current_state : dict
            Current full-state representation from the integrator.
        epoch : float
            Current epoch [TDB seconds].
        body : object
            The accelerated body (spacecraft).
        point_mass_name : str
            Constants key of the third body (e.g. 'EARTH').

        Returns
        -------
        np.ndarray
            (3, 1) partial derivative of acceleration wrt mu_i [km^-2].
        """
        origin2pm = self._get_origin_to_point_mass(
            point_mass_name=point_mass_name,
            current_state=current_state,
            epoch=epoch,
        )
        origin2pm_norm = np.linalg.norm(origin2pm)
        body2pm = -position + origin2pm
        body2pm_norm = np.linalg.norm(body2pm)

        # da/dmu = -(r_origin->pm / |r_origin->pm|^3) + (r_body->pm / |r_body->pm|^3)
        partial = -(origin2pm / origin2pm_norm**3) + (body2pm / body2pm_norm**3)
        return partial.reshape(3, 1)
