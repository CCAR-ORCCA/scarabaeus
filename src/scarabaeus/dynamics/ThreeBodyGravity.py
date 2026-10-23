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
    Units
)
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.EphemerisCache import NoLoadedSPKError, build_ephemeris_cache

import warnings
from typing import List

kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])

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
    for body :math:`i`, its value overrides the body default at each
    function evaluation, for propagated and SPICE bodies alike.

    Parameters
    ----------
    origin : CelestialBody
        Propagation origin: supplies the name and gravitational parameter, and
        anchors the ephemeris cache.
    pnt_masses : list
        List of point mass bodies to be used in computing third body accelerations.
    propagated_bodies : list of CelestialBody, optional
        Third bodies that are integrated in the state vector rather than read
        from the ephemeris. Their positions are taken from the propagated state,
        so they are excluded from the SPICE ephemeris cache. 
        The other bodies' positions are read from the ephemeris cache using their SPICE ID, which can 
        be defined directly by providing direct :obj:`int` entries, or implied by 
        :class:`~scarabaeus.body.CelestialBody` entries. Direct and implied definitions 
        can be mixed within the list.
    base_frame : Frame
        Reference frame in which accelerations are returned.
    t0 : float
        Starting epoch of the force model. Assumed in seconds TDB.
    tf : float
        Final epoch of the force model. Assumed in seconds TDB.
    epoch_buffer : float, optional
        Margin added to the start and end epochs to allow an integrator space to 
        step before and after the defined time interval. Defaults to ``86400.0``, 
        equivalent to one day.

    See Also
    --------
    scarabaeus.PointMassGravity : Single-body Newtonian gravity.
    scarabaeus.ForceModelTranslation : Aggregator that instantiates this model.

    References
    ----------
    .. [1] Vallado, D. A. *Fundamentals of Astrodynamics and Applications*, 4th Ed.,
       Microcosm Press, 2013.
    """

    # errors and warnings (NoLoadedSPKError lives with the shared cache
    # builder; kept as a class attribute for backward compatibility)
    NoLoadedSPKError = NoLoadedSPKError
    class NoChebyCoeffsWarning(RuntimeWarning): pass

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        origin      : CelestialBody,
        point_masses: List[CelestialBody],
        base_frame  : Frame,
        t0          : float,
        tf          : float,
        propagated_bodies: List[CelestialBody] | None = None,
        epoch_buffer: float = 86400.0
    ):
        # save
        self._origin       = origin
        self._origin_name  = origin.name
        self._origin_mu    = ArrayWUnits.get_value_in_target_units(origin.grav_param, km**3 / sec**2)
        self._ref_frame    = base_frame

        # add some margin to time interval for the integrator
        self._t0 = t0 - epoch_buffer
        self._tf = tf + epoch_buffer

        ## save all bodies
        # bodies integrated alongside the spacecraft: their positions come from
        # the propagated state rather than from the ephemeris cache
        self._prop_bodies = [
            (body.name.upper(), body.spice_id, body.grav_param.values)
            for body in (propagated_bodies or [])
        ]

        # spice defined bodies in the SPK
        self._spice_bodies, prop_names = [], {name for name, _, _ in self._prop_bodies}
        for pm in point_masses:
            if pm.name.upper() not in prop_names:
                # not propagated in state -> save for spice cache
                self._spice_bodies.append(pm)

        # the spice bodies that attract, as (name, spice id, GM); their
        # positions come from the ephemeris cache
        self._cached_bodies = [
            (pm.name.upper(), pm.spice_id, float(pm.grav_param.values))
            for pm in self._spice_bodies
            if pm.spice_id != origin.spice_id
        ]
        if self._cached_bodies:
            self._ephemeris = self._cache_ephemerides()
        else:
            self._ephemeris = None

        self._pnt_masses = [pm.name for pm in point_masses]

        self._pnt_mass_spice_ids = {}
        for pm in point_masses:
            self._pnt_mass_spice_ids[pm.name] = pm.spice_id

        self._propagated_third_bodies = {
            body.name.upper(): body for body in (propagated_bodies or [])
        }

    def __getstate__(self):
        """Copy the attributes, leaving out the Rust ephemeris cache.

        ``copy.deepcopy`` and ``pickle`` both call this. ``SolutionOD``
        deep-copies the propagator, and so this model, in
        ``_integrate_reference`` and ``propagate_covariance``. The compiled
        cache cannot be copied, so it is dropped here and rebuilt in
        :meth:`__setstate__`.
        """
        state = self.__dict__.copy()
        state["_ephemeris"] = None
        return state

    def __setstate__(self, state):
        """Restore the copied attributes and rebuild the ephemeris cache
        from the loaded SPKs."""
        self.__dict__.update(state)
        if self._cached_bodies:
            self._ephemeris = self._cache_ephemerides()

    # region Properties
    @property
    def origin_name(self) -> str:
        """Name of the propagation-origin celestial body."""
        return self._origin_name

    @property
    def origin_mu(self) -> float:
        """Gravitational parameter of the origin body, in km^3/s^2."""
        return self._origin_mu

    @property
    def ref_frame(self) -> Frame:
        """Reference frame in which third-body accelerations are returned."""
        return self._ref_frame

    @property
    def pnt_masses(self) -> list[str]:
        """Ordered list of point-mass body name strings (Constants keys)."""
        return self._pnt_masses
    
    # endregion Properties

    # region Helpers
    def _get_origin_to_point_mass(
        self,
        point_mass_name: str,
        current_state: dict,
        epoch: float,
    ) -> np.ndarray:
        """
        Return the position of a point mass relative to the propagation origin.

        A third body that is propagated alongside the spacecraft (listed in
        ``_propagated_third_bodies``) is read from the integrator state under the
        key ``("position", spice_id)``. Any other body comes from the SPICE
        ephemerides, cached per epoch so the acceleration and its partials share
        one lookup.

        Parameters
        ----------
        point_mass_name : str
            Name key in :mod:`~scarabaeus.utils.Constants` (e.g. ``"EARTH"`` or
            ``"JUPITER_BARYCENTER"``).
        current_state : dict
            Current full-state representation from the integrator, keyed by
            ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            Position of the point mass relative to the origin, shape (3,), in km,
            expressed in this model's reference frame.

        Raises
        ------
        ValueError
            If the body is propagated but has no ``spice_id``, or its
            ``("position", spice_id)`` entry is missing from ``current_state``.
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

        # 2) Not propagated: use SPICE ephemerides.
        
        # Per-epoch position memo: compute_acceleration and _compute_partial_by_position
        # both query the SAME epoch within one RHS step, so the ephemeris lookup is
        # cached and reused (halves the SPICE calls); the cache resets when epoch changes.
        if self.__dict__.get("_pm_pos_cache") is None or self._pm_pos_cache_epoch != epoch:
            self._pm_pos_cache = {}
            self._pm_pos_cache_epoch = epoch
        cached_pos = self._pm_pos_cache.get(name_uc)
        if cached_pos is not None:
            return cached_pos

        # Body/origin name resolution is constant per point mass — resolve once.
        resolution = self.__dict__.get("_pm_resolution")
        if resolution is None:
            resolution = self._pm_resolution = {}
        resolved = resolution.get(name_uc)
        if resolved is None:
            if name_uc.endswith('BARYCENTER'):
                # requested barycenter -> query by name and get barycenter ID
                pm_query = name_uc.removesuffix('_BARYCENTER').removesuffix(' BARYCENTER').removesuffix('BARYCENTER')
                trgt_id = getattr(constants, pm_query.upper()).barycenter_id
            else:
                # requested body -> query by name and get body center ID
                trgt_id = getattr(constants, name_uc).body_center_id
            trgt_body = SpiceManager.id2name(trgt_id)
            origin_body_spice_name = getattr(constants, self._origin_name.upper()).spice_name
            resolved = resolution[name_uc] = (trgt_body, origin_body_spice_name)
        trgt_body, origin_body_spice_name = resolved

        # Raw-array SPICE position (km) — skips the ArrayWUnits wrap/unwrap.
        r_origin_pm = SpiceManager.get_pos_values(
            trgt_body, epoch, self._ref_frame, origin_body_spice_name
        )  # km
        arr = np.asarray(r_origin_pm, dtype=float)
        self._pm_pos_cache[name_uc] = arr
        return arr
    
    def _cache_ephemerides(self):
        """ Cache the ephemerides of the origin and the attracting spice bodies.

            The builder appends any centre bodies needed to chain SPK segments;
            they only locate the others.
        """
        body_ids = [self._origin.spice_id] + [sid for _, sid, _ in self._cached_bodies]
        return build_ephemeris_cache(body_ids, self._t0, self._tf)

    def _cached_bodies_positions(self, epoch: float) -> np.ndarray:
        """ (n, 3) positions of the cached bodies from the origin [km], one row
            per entry of ``_cached_bodies``, in the SPK native frame (J2000).
        """
        return self._ephemeris.get_relative_positions(
            [sid for _, sid, _ in self._cached_bodies], self._origin.spice_id, epoch
        )

    def _cached_bodies_gms(self, current_state: dict) -> np.ndarray:
        """ (n,) GMs of the cached bodies, a ``("gm", spice_id)`` state entry
            overriding the body's default.
        """
        return np.array([
            self._scalar_from_state(current_state.get(("gm", sid), gm))
            for _, sid, gm in self._cached_bodies
        ])

    #----------------#
    # region Methods #
    #----------------#
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """ Compute third-body gravitational acceleration.

            If a third body is propagated in the state vector, its position is taken
            from ``current_state``; otherwise from the ephemeris cache.

            Autograd can differentiate it with respect to ``position`` and the
            ``("gm", spice_id)`` state entries.
        """
        # contributions from propagated bodies
        a = self._propagated_bodies_acceleration(position, current_state)

        # contributions from spice bodies
        if self._ephemeris is not None:
            a = a + self._cached_bodies_acceleration(position, current_state, epoch)

        # return
        return a

    def _propagated_bodies_acceleration(
        self,
        position     : np.ndarray,
        current_state: dict,
    ) -> np.ndarray:
        """ Third-body acceleration from the bodies propagated in the state.

            Pure Python, so autograd can trace it with respect to ``position``
            and the GMs.
        """
        a = np.zeros(3, dtype = float)

        for name, pm_id, gm_def in self._prop_bodies:
            gm_key, pos_key = ('gm', pm_id), ('position', pm_id)

            gm = self._scalar_from_state(current_state.get(gm_key, gm_def))

            origin2pm      = np.array(current_state[pos_key])
            origin2pm_norm = np.linalg.norm(origin2pm)

            body2pm      = -position + origin2pm
            body2pm_norm = np.linalg.norm(body2pm)

            # A propagated body does not exert third-body gravity on itself:
            # for the self term body2pm == 0 -> 0/0 -> NaN. Skip it, matching
            # the guard already present in _compute_partial_by_position.
            if body2pm_norm == 0.0:
                continue

            a = a + gm * (-(origin2pm / origin2pm_norm**3) + (body2pm / body2pm_norm**3))

        return a

    def _cached_bodies_acceleration(
        self,
        position     : np.ndarray,
        current_state: dict,
        epoch        : float,
    ) -> np.ndarray:
        """ Third-body acceleration from the bodies in the ephemeris cache.

            Vectorised over the bodies, and traceable by autograd with respect
            to ``position`` and the GMs.
        """
        origin2pm = self._cached_bodies_positions(epoch)
        body2pm   = origin2pm - position
        gms       = self._cached_bodies_gms(current_state)

        origin2pm_norm3 = np.sum(origin2pm**2, axis=1, keepdims=True)**1.5
        body2pm_norm3   = np.sum(body2pm**2,   axis=1, keepdims=True)**1.5

        return gms @ (body2pm / body2pm_norm3 - origin2pm / origin2pm_norm3)

    def _compute_partial_by_position(
        self,
        position     : np.ndarray,
        current_state: dict,
        epoch        : float,
        body         : CelestialBody,
    ) -> np.ndarray:
        """ Compute da_3b/dr for third-body gravity.

            Uses propagated third-body positions when available; otherwise the
            ephemeris cache.
        """
        from scarabaeus import Utils
        partial_a_by_pos = np.zeros((3, 3), dtype=float)
        I = np.eye(3)

        ## propagated bodies
        for _, pm_id, gm_def in self._prop_bodies:
            gm_key = ('gm', pm_id)
            gm     = current_state.get(gm_key, gm_def)
            gm     = float(gm.values if hasattr(gm, 'values') else gm)

            origin2pm = np.array(current_state[('position', pm_id)])
            body2pm   = -position + origin2pm
            r_norm    = Utils.norm3D(body2pm)

            if r_norm == 0.0:
                continue

            rrt  = np.outer(body2pm, body2pm)
            dadr = -gm * I / r_norm**3 + 3 * gm * rrt / r_norm**5
            partial_a_by_pos += dadr

        ## SPICE bodies
        if self._ephemeris is not None:
            body2pm = self._cached_bodies_positions(epoch) - position
            gms     = self._cached_bodies_gms(current_state)
            r2      = np.sum(body2pm**2, axis=1)

            partial_a_by_pos += (
                -np.sum(gms / r2**1.5) * I
                + 3.0 * (body2pm.T * (gms / r2**2.5)) @ body2pm
            )

        return partial_a_by_pos

    def _compute_partial_by_third_body_position(
        self,
        position        : np.ndarray,
        current_state   : dict,
        epoch           : float,
        body            : CelestialBody,
        point_mass_name : str,
    ) -> np.ndarray | None:
        """ Compute d a_3b / d r_origin->pm for a specific third body propagated (point mass).

            Returns 3x3 if that point mass is propagated in the state.
            Returns None if not propagated (because then origin2pm comes from SPICE, not a solve-for).
        """
        from scarabaeus import Utils
        name_uc = point_mass_name.upper()

        # find in prop_bodies
        prop_entry = next(((n, pid, gm) for n, pid, gm in self._prop_bodies if n == name_uc), None)
        if not prop_entry:
            return None  # not a state variable

        _, pm_id, gm_def = prop_entry
        gm_key = ('gm', pm_id)
        gm     = current_state.get(gm_key, gm_def)
        gm     = float(gm.values if hasattr(gm, 'values') else gm)

        origin2pm = np.array(current_state[('position', pm_id)])
        r0        = origin2pm
        r         = origin2pm - position  # body2pm

        r0n = Utils.norm3D(r0)
        rn  = Utils.norm3D(r)
        if r0n == 0.0 or rn == 0.0:
            return np.zeros((3, 3), dtype=float)
        
        J = lambda u, un: np.eye(3) / un**3 - 3.0 * np.outer(u, u) / un**5

        return gm * (-J(r0, r0n) + J(r, rn))

    def _compute_partial_by_gm(
        self,
        position        : np.ndarray,
        current_state   : dict,
        epoch           : float,
        body            : CelestialBody,
        point_mass_name : str,
    ) -> np.ndarray:
        """ Compute da_3b/dmu for a specific third-body gravitational parameter.

            The origin-to-body position comes from the state for a propagated
            body, otherwise from the ephemeris cache. The acceleration is linear
            in each GM, so the partial does not depend on a GM value. Zero for a
            body that is not a point mass of this model.

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
        from scarabaeus import Utils
        name_uc = point_mass_name.upper()

        prop_id    = next((pid for n, pid, _ in self._prop_bodies if n == name_uc), None)
        cached_idx = next((i for i, (n, _, _) in enumerate(self._cached_bodies) if n == name_uc), None)
        if prop_id is not None:
            origin2pm = np.array(current_state[('position', prop_id)])
        elif cached_idx is not None:
            origin2pm = self._cached_bodies_positions(epoch)[cached_idx]
        else:
            return np.zeros((3, 1), dtype=float)

        origin2pm_norm = Utils.norm3D(origin2pm)
        body2pm        = -position + origin2pm
        body2pm_norm   = Utils.norm3D(body2pm)

        partial = -(origin2pm / origin2pm_norm**3) + (body2pm / body2pm_norm**3)
        return partial.reshape(3, 1)
