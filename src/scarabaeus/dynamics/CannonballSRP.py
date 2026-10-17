# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    CelestialBody,
    Body,
    EpochArray,
    StateArray,
    SpiceManager,
    Spacecraft,
    Frame,
    DynamicModel,
    ArrayWUnits,
    Units,
)
from scarabaeus import constants
from scarabaeus.utils.EphemerisCache import build_ephemeris_cache
import scarabaeus.utils.NumpyWrapper as np
kg, km, sec, N, unitless = Units.get_units(["kg", "km", "sec", "N", "unitless"])

_SUN_SPICE_ID = 10


# --------------------#
#  Class Definition  #
# --------------------#
class CannonballSRP(DynamicModel):
    """ Models the acceleration due to Solar Radiation Pressure (SRP) using the cannonball model.

    The spacecraft is treated as a perfect sphere that reflects solar radiation
    isotropically.  The SRP acceleration is:

    .. math::

        \\mathbf{a}_\\text{SRP} =
            -\\frac{\\nu\\,\\eta_\\text{SRP}\\, C_R\\, A}{m}
            \\left(\\frac{r_0}{r}\\right)^2
            P_0\\, \\hat{\\mathbf{r}}_\\odot

    where :math:`\\nu` is the shadow factor (0 in eclipse, 1 in full Sun),
    :math:`\\eta_\\text{SRP}` is an estimable scale factor (default 1),
    :math:`C_R = 1 + \\rho_s` is the reflectivity coefficient,
    :math:`A` is the spacecraft cross-sectional area,
    :math:`m` is the spacecraft mass,
    :math:`r_0` is the reference distance (1 AU),
    :math:`r` is the Sun–spacecraft distance,
    :math:`P_0` is the solar radiation pressure at 1 AU,
    and :math:`\\hat{\\mathbf{r}}_\\odot` is the unit vector from the spacecraft
    to the Sun (direction of incoming radiation; the leading minus sign means the
    acceleration is directed away from the Sun).
    :math:`\\eta_\\text{SRP}` is retrieved from the state vector via a
    ``("eta_srp", spice_id)`` entry and can be solved for in the OD process.

    Parameters
    ----------
    origin : CelestialBody
        Propagation origin. Used for the Sun-relative geometry; the estimable
        ``eta_srp`` is read from the propagated state at evaluation time.

    spacecraft : Spacecraft
        The spacecraft object with defined mass, cross-sectional area, and reflectivity coefficient.

    base_frame : Frame
        Frame of the propagator.

    See Also
    --------
    scarabaeus.StateArray : Represents state vectors with origins and reference frames.
    scarabaeus.Spacecraft : Defines spacecraft properties including mass and area.
    scarabaeus.Propagator : Uses this model when `cannonball_SRP=True`.

    Notes
    -----
    * The model assumes uniform reflection across the entire spacecraft surface.
    * The Sun's position is retrieved dynamically from SPICE data.
    * The acceleration follows an inverse-square law based on Sun-spacecraft distance.

    References
    ----------
    .. [1] Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods, and Applications.

    Examples
    --------
    .. code-block:: python

        # intial setup
        import scarabaeus as scb

        # Define spacecraft properties
        mass         = scb.ArrayWUnits(500, "kg")  # Spacecraft mass
        area         = scb.ArrayWUnits(20, "m^2")  # Cross-sectional area
        reflectivity = 1.3                         # Reflectivity coefficient

        # Create spacecraft object
        sc = scb.Spacecraft("Test_Sat", -1000, mass, area, reflectivity)

        # Define state vector with position, velocity, and SRP coefficient
        state = [
            (
                "position",
                3,
                "estimated",
                "dynamic",
                sc,
                scb.ArrayWUnits([7000, 0, 0], "km"),
            ),
            (
                "velocity",
                3,
                "estimated",
                "dynamic",
                sc,
                scb.ArrayWUnits([0, 7.5, 0], "km/s"),
            ),
            ("eta_srp", 1, "estimated", "static", sc, scb.ArrayWUnits(1, None)),
        ]

        # Create state vector
        state_vector = scb.StateArray(epoch = epochs[0], frame = 'J2000', origin = scb.CelestialBody('Earth'), state = state)

        # Initialize propagator with Cannonball SRP enabled
        prop = scb.Propagator(sc, state_vector, epochs, cannonball_SRP = True)
        prop.propagate()
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        origin: CelestialBody,
        spacecraft: Spacecraft,
        base_frame: Frame,
        t0: float = None,
        tf: float = None,
    ):
        # Origin and reference frame
        self._origin = origin
        self._origin_name = self._origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame

        # Body-specific parameters
        self._area = ArrayWUnits.get_value_in_target_units(self._spacecraft.area, km**2)
        self._ref_coeff = ArrayWUnits.get_value_in_target_units(
            self._spacecraft.ref_coeff, unitless
        )

        # SRP pressure constant
        self._p_srp = ArrayWUnits.get_value_in_target_units(
            constants.srp_constant2, kg * km * sec**-2
        )

        # Precompute constant scale factor (without eta_srp, without mass)
        self._const_scale_factor = self._p_srp * self._ref_coeff * self._area

        # eta_srp is read from the propagated state on every evaluation (see
        # _get_eta_srp). The map is kept only so callers that poke a value in
        # directly — the partial-derivative diagnostics do — still work.
        self._eta_srp_map: dict[int, float] = {}

        # Sun-position source. When the origin IS the Sun the vector is
        # identically zero. Otherwise, if the propagation span is known and
        # the frame is the SPK-native J2000, cache the Chebyshev records so
        # _sun_to_origin avoids the per-step spkpos; SPICE stays the fallback.
        self._origin_spice_id = getattr(self._origin, "spice_id", None)
        self._origin_is_sun = self._origin_spice_id == _SUN_SPICE_ID
        self._sun_eph = None
        self._sun_eph_span = None
        frame_name = getattr(base_frame, "name", str(base_frame))
        if (
            t0 is not None
            and tf is not None
            and self._origin_spice_id is not None
            and not self._origin_is_sun
            and frame_name.upper() == "J2000"
        ):
            self._sun_eph_span = (t0 - 86400.0, tf + 86400.0)
            self._sun_eph = self._build_sun_ephemeris()

    def __getstate__(self):
        """
        Copy the attributes, leaving out the Rust Sun ephemeris cache.

        ``copy.deepcopy`` and ``pickle`` both call this. ``SolutionOD``
        deep-copies the propagator, and so this model, in
        ``_integrate_reference`` and ``propagate_covariance``. The compiled
        cache cannot be copied, so it is dropped here and rebuilt in
        :meth:`__setstate__`.
        """
        state = self.__dict__.copy()
        state["_sun_eph"] = None
        return state

    def __setstate__(self, state):
        """Restore the copied attributes and rebuild the Sun ephemeris cache
        from the loaded SPKs."""
        self.__dict__.update(state)
        if self.__dict__.get("_sun_eph_span") is not None:
            self._sun_eph = self._build_sun_ephemeris()

    def _build_sun_ephemeris(self):
        """
        Cache the origin and Sun Chebyshev records over ``_sun_eph_span``.

        Returns
        -------
        EphemerisCache or None
            The cache, or ``None`` when the loaded SPKs do not cover the span;
            the Sun position then falls back to SPICE.
        """
        try:
            return build_ephemeris_cache(
                [self._origin_spice_id, _SUN_SPICE_ID], *self._sun_eph_span
            )
        except Exception:
            return None  # no/partial SPK coverage -> SPICE fallback

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def origin(self) -> Body:
        """
        The celestial body that serves as the origin for the state vector.

        :base: CannonballSRP
        :type: Body
        """
        return self._origin

    @property
    def origin_name(self) -> str:
        """
        The name of the origin body.

        :base: CannonballSRP
        :type: str
        """
        return self._origin_name

    @property
    def ref_frame(self) -> str:
        """
        The reference frame in which SRP calculations are performed.

        :base: CannonballSRP
        :type: str
        """
        return self._ref_frame

    @property
    def area(self) -> float:
        """
        The cross-sectional area of the spacecraft.

        :base: CannonballSRP
        :type: float
        """
        return self._area

    @property
    def mass(self) -> float:
        """
        The mass of the spacecraft, typically expressed in units of
        kilograms.

        :base: CannonballSRP
        :type: float
        """
        return self._mass

    @property
    def ref_coeff(self) -> float:
        """
        The reflectivity coefficient of the spacecraft.

        :base: CannonballSRP
        :type: float
        """
        return self._ref_coeff

    @property
    def p_srp(self) -> float:
        """
        Solar radiation pressure constant at 1 AU, in kg km/s^2.

        Converted at construction time from the ``"srpP_constant"`` entry in
        ``scarabaeus.Constants.CONSTANTS``.

        :base: CannonballSRP
        :type: float
        """
        return self._p_srp

    @property
    def eta_srp(self) -> float:
        """
        The SRP scale factor. Unitless.
        """
        return self._eta_srp

    @eta_srp.setter
    def eta_srp(self, input_val):
        """
        Set the SRP scale factor.

        Parameters
        ----------
        input_val : float or None
            Scale factor applied to the SRP acceleration. Unitless.
            If ``None``, defaults to ``0`` (no SRP contribution).
        """
        if input_val is None:
            input_val = 0

        self._eta_srp = input_val

    # endregion => Properties  #
    # -------------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _get_eta_srp(self, current_state: dict, body: Body) -> float:
        """
        Resolve eta_srp for the given body.

        Priority:
        1. The propagated state, ``current_state[("eta_srp", spice_id)]``
        2. Any value pushed into ``self._eta_srp_map`` by a caller
        3. Fallback: 1.0

        The state comes first so that an estimated eta_srp takes effect, which
        also matches how every other model resolves its parameters.
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            key = ("eta_srp", spice_id)
            if key in current_state:
                val = current_state[key]
                return self._scalar_from_state(val)

            if spice_id in self._eta_srp_map:
                return self._eta_srp_map[spice_id]

        return 1.0

    def _sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Sun->origin position (km, raw numpy), memoized per epoch.

        Both :meth:`compute_acceleration` and :meth:`_compute_partial_by_position`
        need this at the same epoch within one RHS step, so it is cached to avoid
        the duplicate SPICE call. The origin's SPICE name is constant and resolved
        once.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin position in km.
        """
        if self.__dict__.get("_s2o_epoch") != epoch:
            self._s2o_cache = self._lookup_sun_to_origin(epoch)
            self._s2o_epoch = epoch
        return self._s2o_cache

    def _lookup_sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Uncached Sun->origin lookup: identically zero for a heliocentric
        origin; Chebyshev ephemeris cache when available (falling back to
        SPICE for epochs outside the cached span); SPICE otherwise.
        """
        if self._origin_is_sun:
            return np.zeros(3)
        if self._sun_eph is not None:
            try:
                return np.asarray(
                    self._sun_eph.get_relative_position(
                        self._origin_spice_id, _SUN_SPICE_ID, epoch
                    ),
                    dtype=float,
                )
            except Exception:
                pass  # epoch outside the cached span -> SPICE fallback
        spice_name = self.__dict__.get("_origin_spice_name")
        if spice_name is None:
            spice_name = self._origin_spice_name = getattr(
                constants, self.origin_name.upper()
            ).spice_name
        return SpiceManager.get_pos_values(
            spice_name, epoch, self.ref_frame, "SUN"
        )

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Body
    ) -> np.ndarray:
        """
        Computes the acceleration due to solar radiation pressure (SRP).

        This method calculates the SRP acceleration experienced by the spacecraft based on its
        distance from the Sun and its physical properties.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector in the reference frame (size: 3x1, units: km).

        current_state : dict
            Dictionary containing the current state parameters, including `eta_srp`.

        epoch : float
            The current simulation epoch.

        body : Body
            The spacecraft for which SRP acceleration is computed.

        Returns
        -------
        srp_vec : np.ndarray
            The SRP acceleration vector (size: 3x1, units: km/s^2).
        """

        # Initialize acceleration
        cannonball_srp_acceleration = np.array([0, 0, 0])

        # Compute the Sun-to-Origin vector (raw km, memoized per epoch)
        sun2origin = self._sun_to_origin(epoch)
        # Compute the Sun-to-Body vector and its norm
        sun2body = sun2origin + position
        sun2body_norm = np.linalg.norm(sun2body)

        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Compute the scaling factor component for SRP acceleration vector
        m_t = self._spacecraft.mass(epoch).values  # Assumed in kg
        scaling_factor = (eta_srp * self._const_scale_factor) / m_t
        
        # Compute the total SRP acceleration
        cannonball_srp_acceleration = scaling_factor * (sun2body_norm**-3 * sun2body)

        return cannonball_srp_acceleration

    def _compute_partial_by_position(
        self, position: np.ndarray, current_state: dict, epoch: EpochArray, body: Body
    ) -> np.ndarray:
        """
        Computes the partial derivatives of the SRP acceleration with respect to position.

        This method determines how the SRP acceleration varies as a function of
        the spacecraft's position relative to the Sun.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector in the reference frame (size: 3x1, units: km).

        current_state : dict
            Dictionary containing the current state parameters, including `eta_srp`.

        epoch : EpochArray
            Current simulation epoch.

        body : Body
            The spacecraft affected by SRP.

        Returns
        -------
        partial_a_by_pos : np.ndarray
            The 3x3 Jacobian matrix of cannonball SRP acceleration with respect to position. Assumed
            in units of :math:`\\frac{1}{s^2}`.
        """

        # Initialize the total partial derivative matrix
        partial_a_by_pos = np.zeros((3, 3))

        # Identity matrix
        identity_matrix = np.eye(3)

        # Compute the Sun-to-Origin vector (raw km, memoized per epoch)
        sun2origin = self._sun_to_origin(epoch)
        # Compute the Body-to-Sun vector
        body2sun = sun2origin + position
        body2sun_norm = np.linalg.norm(body2sun)

        # Get the SRP scale factor
        eta_srp = self._get_eta_srp(current_state, body)

        # Compute the scaling factor component for SRP acceleration vector
        m_t = self._spacecraft.mass(epoch).values  # Assumed in kg
        scaling_factor = (eta_srp * self.p_srp * self.ref_coeff * self.area) / m_t

        # Dyadic product of the Sun-to-body vector
        dyadic_product = np.outer(body2sun.transpose(), body2sun)

        # Return the total partial derivative
        partial_a_by_pos = (
            body2sun_norm**-3 * (scaling_factor * identity_matrix)
        ) + body2sun_norm**-5 * (-3 * scaling_factor * dyadic_product)

        return partial_a_by_pos

    def _compute_partial_by_eta(
        self, position: np.ndarray, current_state: dict, epoch: EpochArray, body: Body
    ) -> np.ndarray:
        """
            Computes the partial derivative of SRP acceleration with respect to the SRP scale factor (`eta_srp`).

            This method determines how changes in the SRP scale factor affect the acceleration.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector in the reference frame (size: 3x1, units: km).

        current_state : dict
            Dictionary containing the current state parameters, including `eta_srp`.

        epoch : EpochArray
            Current simulation epoch.
        
        body : Body
            The spacecraft affected by SRP.

        Returns
        -------
        partial : np.ndarray
            The (3, 1) partial derivative vector of SRP acceleration with respect to `eta_srp`.
            Assumed in units of :math:`\\frac{km}{s^2}`.

        Notes
        -----
        The acceleration scales linearly with ``eta_srp``, so the partial is
        the acceleration at unit scale and is well defined at
        ``eta_srp = 0`` — which is a natural initial guess when the factor
        is being estimated as a correction.
        """
        # The acceleration is linear in the scale factor, a = eta * a_unit, so
        # the partial is a_unit. Evaluating it directly rather than dividing the
        # acceleration by eta keeps the partial defined at eta = 0 — a natural
        # starting guess, and one that otherwise raises ZeroDivisionError.
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            # eta cannot be addressed per body here, so fall back to scaling the
            # acceleration; _get_eta_srp returns 1.0 in this case, making the
            # division a no-op.
            return self._get_eta_srp(current_state, body) ** -1 * (
                self.compute_acceleration(position, current_state, epoch, body)
            )

        unit_state = dict(current_state or {})
        unit_state[("eta_srp", spice_id)] = 1.0

        return self.compute_acceleration(position, unit_state, epoch, body)
