# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Frame,
    StateArray,
    CelestialBody,
    ArrayWUnits,
    Units,
)
import numpy as np

kg, km, sec, N, unitless = Units.get_units(["kg", "km", "sec", "N", "unitless"])


class YarkovskyEffect(DynamicModel):
    """ Semi-empirical along-track Yarkovsky acceleration model.

    The Yarkovsky effect is a thermal radiation force that acts along the
    instantaneous velocity direction of a small body.  The acceleration is
    modelled as an inverse-square law scaled by a fitted parameter :math:`A_2`:

    .. math::

        \\mathbf{a}_\\text{Yark} =
            A_2 \\left(\\frac{r_0}{r}\\right)^2 \\hat{t}

    where :math:`A_2` is the Yarkovsky parameter (km/s² at :math:`r_0`),
    :math:`r_0` is a reference distance (default: 1 AU), :math:`r` is the
    heliocentric distance, and :math:`\\hat{t} = \\mathbf{v}/|\\mathbf{v}|`
    is the unit velocity vector.

    :math:`A_2` is estimated as a static parameter in the state vector (one
    per Yarkovsky-active body).  The model is fully state-driven — no SPICE
    calls are made during acceleration computation.

    Parameters
    ----------
    state_vector : StateArray
        State vector used to identify bodies carrying a Yarkovsky parameter.
    base_frame : Frame
        Reference frame in which accelerations are returned.
    param_name : str, optional
        Name of the Yarkovsky A2 parameter as stored in the state.
        Defaults to ``"A2_yark"``.
    r0 : ArrayWUnits, optional
        Reference distance for the inverse-square scaling.
        Defaults to 1 AU (``1.495978707e8`` km).

    Notes
    -----
    The state vector must contain ``("position", spice_id)``,
    ``("velocity", spice_id)``, and ``("A2_yark", spice_id)`` entries for
    each Yarkovsky-active body.  State keys follow the
    :meth:`~scarabaeus.StateArray._make_direct_representation` convention.

    See Also
    --------
    scarabaeus.CannonballSRP : Solar radiation pressure model using a similar scale-factor pattern.
    scarabaeus.nPlateSRP : Multi-plate SRP model.
    scarabaeus.ForceModelTranslation : Aggregator that instantiates this model.

    Raises
    ------
    ValueError
        If any ``param_name`` entry in *state_vector* belongs to an owner
        without a ``spice_id`` attribute.

    References
    ----------
    Farnocchia, D., Chesley, S. R., Vokrouhlický, D., Milani, A., Spoto, F.,
    & Bottke, W. F. (2013). Near Earth Asteroids with measurable Yarkovsky
    effect. *Icarus*, 224(1), 1–13. https://doi.org/10.1016/j.icarus.2013.02.004
    """

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        state_vector: StateArray,
        base_frame: Frame,
        param_name: str = "A2_yark",
        r0: ArrayWUnits | None = None,
    ):
        self._state_vector = state_vector
        self._ref_frame = base_frame
        self._param_name = param_name

        # Reference distance r0 [km]
        if r0 is None:
            self._r0_km = 1.495978707e8  # 1 au in km
        else:
            self._r0_km = float(r0.to(km).values)

        # Identify which bodies have A2_yark defined
        self._yarkovsky_bodies: set[int] = set()
        for name, _, _, _, owner, _ in state_vector.state:
            if name == self._param_name:
                spice_id = getattr(owner, "spice_id", None)
                if spice_id is None:
                    raise ValueError(
                        f"YarkovskyEffect: parameter '{self._param_name}' "
                        f"requires an owner with 'spice_id'. Got {owner!r}."
                    )
                self._yarkovsky_bodies.add(spice_id)

    # -------------------------#
    #       Properties         #
    # -------------------------#
    @property
    def ref_frame(self) -> Frame:
        """Reference frame in which Yarkovsky accelerations are returned."""
        return self._ref_frame

    @property
    def param_name(self) -> str:
        """Name of the A2 Yarkovsky parameter as stored in the state (e.g. ``"A2_yark"``)."""
        return self._param_name

    @property
    def yarkovsky_bodies(self) -> set[int]:
        """Set of SPICE integer IDs for bodies with a Yarkovsky parameter in the state."""
        return self._yarkovsky_bodies

    # ----------------#
    #     Helpers     #
    # ----------------#
    def _get_A2(self, current_state: dict, body: CelestialBody) -> float:
        """
        Get A2 associated with `body` from current_state.

        Expected key: (param_name, body.spice_id).

        Behavior
        --------
        - If body is NOT in `self._yarkovsky_bodies`:
            returns 0.0 (no Yarkovsky effect).
        - If body IS in `self._yarkovsky_bodies` but the parameter
          is missing in `current_state`: raises ValueError.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return 0.0

        if spice_id not in self._yarkovsky_bodies:
            return 0.0  # not configured for Yarkovsky

        key = (self._param_name, spice_id)
        if key not in current_state:
            raise ValueError(
                f"YarkovskyEffect: body with spice_id={spice_id} is marked as "
                f"Yarkovsky-active but '{self._param_name}' is missing in current_state."
            )

        val = current_state[key]
        return float(val.values if hasattr(val, "values") else val)

    def _get_velocity(self, current_state: dict, body: CelestialBody) -> np.ndarray:
        """
        Retrieve velocity vector for `body` from current_state.

        Expected key: ("velocity", body.spice_id).
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            raise ValueError(
                "YarkovskyEffect: cannot resolve velocity; body has no 'spice_id'."
            )

        key = ("velocity", spice_id)
        if key not in current_state:
            raise ValueError(
                f"YarkovskyEffect: missing velocity entry {key!r} in current_state."
            )

        v = current_state[key]
        v = v.values if hasattr(v, "values") else np.asarray(v)
        return np.asarray(v, dtype=float)

    # ----------------#
    #      Methods    #
    # ----------------#
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Compute Yarkovsky acceleration for `body`.

        Parameters
        ----------
        position : np.ndarray
            Position vector [km] of the body (w.r.t. central body) from the integrator.
        current_state : dict
            Direct representation of the full state, keyed by (name, spice_id).
        epoch : float
            Current epoch [TDB seconds]. (Not used in this compact model.)
        body : CelestialBody
            Target body.

        Returns
        -------
        np.ndarray
            3-element Yarkovsky acceleration vector [km/s^2].
            Zero if the body is not configured for Yarkovsky.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id not in self._yarkovsky_bodies:
            return np.zeros(3, dtype=float)

        r_vec = np.asarray(position, dtype=float)
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            return np.zeros(3, dtype=float)

        # Along-track direction from velocity
        v_vec = self._get_velocity(current_state, body)
        v_norm = np.linalg.norm(v_vec)
        if v_norm == 0.0:
            return np.zeros(3, dtype=float)

        t_hat = v_vec / v_norm
        A2 = self._get_A2(current_state, body)

        factor = A2 * (self._r0_km / r) ** 2
        return factor * t_hat

    def _compute_partial_by_A2_yark(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        da_Yark / dA2 for the given body.

        Returns
        -------
        np.ndarray
            3-element vector: (r0 / r)^2 * t_hat.
            Zero if the body is not Yarkovsky-active or data is insufficient.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id not in self._yarkovsky_bodies:
            return np.zeros(3, dtype=float)

        r_vec = np.asarray(position, dtype=float)
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            return np.zeros(3, dtype=float)

        try:
            v_vec = self._get_velocity(current_state, body)
        except ValueError:
            return np.zeros(3, dtype=float)

        v_norm = np.linalg.norm(v_vec)
        if v_norm == 0.0:
            return np.zeros(3, dtype=float)

        t_hat = v_vec / v_norm
        factor = (self._r0_km / r) ** 2
        return factor * t_hat

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        da_Yark / dr (3x3) for:
            a = A2 * (r0/r)^2 * t_hat
        where r = ||position|| and t_hat = v/||v||.

        Here we treat v (thus t_hat) as independent of position for this partial
        (i.e., the direct dependence through the model expression only).

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id not in self._yarkovsky_bodies:
            return np.zeros((3, 3), dtype=float)

        r_vec = np.asarray(position, dtype=float).reshape(3)
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            return np.zeros((3, 3), dtype=float)

        # Need t_hat from velocity
        try:
            v_vec = self._get_velocity(current_state, body)
        except ValueError:
            return np.zeros((3, 3), dtype=float)

        v_norm = np.linalg.norm(v_vec)
        if v_norm == 0.0:
            return np.zeros((3, 3), dtype=float)

        t_hat = v_vec / v_norm
        A2 = self._get_A2(current_state, body)

        # da/dr = A2 * t_hat * d[(r0/r)^2]/dr
        # d( r0^2 / r^2 )/dr_vec = -2 * r0^2 * r_vec / r^4
        # => da/dr = (-2*A2*r0^2/r^4) * (t_hat ⊗ r_vec)
        coef = -2.0 * A2 * (self._r0_km**2) / (r**4)
        return coef * np.outer(t_hat, r_vec)

    def _compute_partial_by_velocity(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        da_Yark / dv (3x3) for:
            a = A2 * (r0/r)^2 * (v/||v||)

        Using:
            d(v/||v||)/dv = (1/||v||) * (I - t_hat t_hat^T)

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id not in self._yarkovsky_bodies:
            return np.zeros((3, 3), dtype=float)

        r_vec = np.asarray(position, dtype=float).reshape(3)
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            return np.zeros((3, 3), dtype=float)

        try:
            v_vec = self._get_velocity(current_state, body)
        except ValueError:
            return np.zeros((3, 3), dtype=float)

        v_vec = np.asarray(v_vec, dtype=float).reshape(3)
        v_norm = np.linalg.norm(v_vec)
        if v_norm == 0.0:
            return np.zeros((3, 3), dtype=float)

        t_hat = v_vec / v_norm
        A2 = self._get_A2(current_state, body)

        s = (self._r0_km / r) ** 2  # scalar scale
        I = np.eye(3, dtype=float)

        return (A2 * s / v_norm) * (I - np.outer(t_hat, t_hat))
