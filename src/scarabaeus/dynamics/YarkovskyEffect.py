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

    where :math:`A_2` is the Yarkovsky parameter (km/s^2 at :math:`r_0`),
    :math:`r_0` is a reference distance (default: 1 AU), :math:`r` is the
    heliocentric distance, and :math:`\\hat{t} = \\mathbf{v}/|\\mathbf{v}|`
    is the unit velocity vector.

    :math:`A_2` is estimated as a static parameter in the state vector (one
    per Yarkovsky-active body).  The model is fully state-driven — no SPICE
    calls are made during acceleration computation.

    Parameters
    ----------
    base_frame : Frame
        Reference frame in which accelerations are returned.
    param_name : str, optional
        Name of the Yarkovsky A2 parameter as stored in the state.
        Defaults to ``"A2_yark"``.
    r0 : ArrayWUnits, optional
        Reference distance for the inverse-square scaling.
        Defaults to 1 AU (``1.495978707e8`` km).

    Raises
    ------
    ValueError
        If any ``param_name`` entry in the propagated state belongs to an owner
        without a ``spice_id`` attribute.

    See Also
    --------
    scarabaeus.CannonballSRP : Solar radiation pressure model using a similar scale-factor pattern.
    scarabaeus.nPlateSRP : Multi-plate SRP model.
    scarabaeus.ForceModelTranslation : Aggregator that instantiates this model.

    Notes
    -----
    The state vector must contain ``("position", spice_id)``,
    ``("velocity", spice_id)``, and ``("A2_yark", spice_id)`` entries for
    each Yarkovsky-active body.  State keys follow the
    :meth:`~scarabaeus.StateArray._make_direct_representation` convention.

    References
    ----------
    .. [1] Farnocchia, D., Chesley, S. R., Vokrouhlický, D., Milani, A., Spoto, F.,
       & Bottke, W. F. (2013). Near Earth Asteroids with measurable Yarkovsky
       effect. *Icarus*, 224(1), 1–13. https://doi.org/10.1016/j.icarus.2013.02.004
    """

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        base_frame: Frame,
        param_name: str = "A2_yark",
        r0: ArrayWUnits | None = None,
    ):
        self._ref_frame = base_frame
        self._param_name = param_name

        # Reference distance r0 [km]
        if r0 is None:
            self._r0_km = 1.495978707e8  # 1 au in km
        else:
            self._r0_km = float(r0.to(km).values)

        # A body is Yarkovsky-active when the propagated state carries its
        # parameter; that is checked per evaluation by _is_active, so no state
        # vector is needed here. The set stays as an override hook for callers
        # that want to force a body active.
        self._yarkovsky_bodies: set[int] = set()

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
    def _is_active(self, spice_id, current_state: dict) -> bool:
        """
        Report whether the Yarkovsky parameter applies to *spice_id*.

        A body is active when the propagated state carries its parameter, or
        when a caller has forced it into :attr:`yarkovsky_bodies`.

        Parameters
        ----------
        spice_id : int or None
            SPICE ID of the body being evaluated.
        current_state : dict
            Dictionary containing the current state parameters.

        Returns
        -------
        bool
            ``True`` when the model contributes for this body.
        """
        if spice_id is None:
            return False

        return (
            (self._param_name, spice_id) in current_state
            or spice_id in self._yarkovsky_bodies
        )

    def _get_A2(self, current_state: dict, body: CelestialBody) -> float:
        """
        Return the Yarkovsky :math:`A_2` parameter of ``body``.

        Parameters
        ----------
        current_state : dict
            Current full-state representation from the integrator, keyed by
            ``(name, spice_id)``.
        body : CelestialBody
            Body whose parameter is read.

        Returns
        -------
        float
            The value stored under ``(param_name, body.spice_id)``, or ``0.0``
            when the body has no ``spice_id`` or the model is not active for it
            (see :meth:`_is_active`).

        Raises
        ------
        KeyError
            If the body was forced into :attr:`yarkovsky_bodies` but the state
            carries no parameter for it.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return 0.0

        if not self._is_active(spice_id, current_state):
            return 0.0  # not configured for Yarkovsky

        val = current_state[(self._param_name, spice_id)]
        return self._scalar_from_state(val)

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
        if not self._is_active(spice_id, current_state):
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
        if not self._is_active(spice_id, current_state):
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
        Compute the partial of the Yarkovsky acceleration with respect to position.

        For :math:`\\mathbf{a} = A_2 (r_0/r)^2\\,\\hat{\\mathbf{t}}` with
        :math:`r = \\lVert\\mathbf{r}\\rVert` and
        :math:`\\hat{\\mathbf{t}} = \\mathbf{v}/\\lVert\\mathbf{v}\\rVert`,

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial\\mathbf{r}}
            = -\\frac{2 A_2 r_0^2}{r^4}\\,\\hat{\\mathbf{t}}\\,\\mathbf{r}^{\\mathsf{T}},

        treating :math:`\\hat{\\mathbf{t}}` as independent of position.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km.
        current_state : dict
            The propagated state, keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, in TDB seconds past J2000. Unused.
        body : CelestialBody
            The propagated body.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` partial, in 1/s^2; zeros when the body is not
            Yarkovsky-active or the position or velocity is zero.
        """
        spice_id = getattr(body, "spice_id", None)
        if not self._is_active(spice_id, current_state):
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
        if not self._is_active(spice_id, current_state):
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
