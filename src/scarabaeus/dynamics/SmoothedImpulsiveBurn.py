# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    DynamicModel,
    Units,
    ArrayWUnits,
    Frame,
    ImpulsiveBurn,
)
from scarabaeus.body.Body import Body

import scarabaeus.utils.NumpyWrapper as np

# First, generate the default base units and frames
km, sec = Units.get_units(["km", "sec"])
(J2000, ITRF93, ECLIPJ2000, IAUEARTH) = Frame.generate_common_frames()


#######################################################################
class SmoothedImpulsiveBurn(DynamicModel):
    """ Models impulsive maneuvers as continuous tanh-shaped acceleration pulses.

    A true impulsive maneuver applies an instantaneous velocity increment at
    :math:`t_{\\Delta v}`, which makes the state discontinuous and therefore
    forces the trajectory to be split into legs around every burn (see
    :class:`~scarabaeus.MissionSequence`). This model removes that constraint
    by replacing the Dirac impulse with a narrow, continuously differentiable
    pulse, so a single uninterrupted propagation can carry the maneuver and the
    variational equations can be integrated straight through it.

    The pulse is built from the product of two opposite-signed hyperbolic
    tangents, which yields a localized peak that decays to zero away from the
    maneuver epoch:

    .. math::

        s_{\\Delta v}(t) = -\\tanh^2\\!\\big(a (t - t_{\\Delta v})\\big) + 1
                         = \\operatorname{sech}^2\\!\\big(a (t - t_{\\Delta v})\\big)

    where :math:`a` is the *smoothing factor*. Integrating from the start of the
    arc gives the normalization

    .. math::

        \\int_{t_0}^{\\infty} s_{\\Delta v}\\,\\mathrm{d}t =
            \\frac{1}{a}\\Big(1 + \\tanh\\!\\big(a (t_{\\Delta v} - t_0)\\big)\\Big)

    so that dividing by it makes the pulse deliver exactly :math:`\\Delta v`:

    .. math::

        \\mathbf{u}^{\\text{smooth}}_{\\Delta v}(t) =
            \\frac{-\\tanh^2\\!\\big(a (t - t_{\\Delta v})\\big) + 1}
                  {\\frac{1}{a}\\big(1 + \\tanh(a (t_{\\Delta v} - t_0))\\big)}\\,
            \\Delta \\mathbf{v}

    As :math:`a \\to \\infty` this approaches
    :math:`\\Delta\\mathbf{v}\\,\\delta(t - t_{\\Delta v})` in the distributional
    sense. In practice :math:`a` must stay finite: too large a value produces
    gradients steep enough to collapse the integrator step size. See
    :attr:`smoothing_factor` for guidance on choosing it.

    The extended dynamics augment the orbital state with the maneuver
    parameters :math:`\\boldsymbol{\\theta}_{\\Delta v} = [\\Delta\\mathbf{v},\\,
    \\delta t_{\\Delta v}]`, whose own dynamics are zero:

    .. math::

        \\dot{\\mathbf{x}}_{\\Delta v} =
        \\begin{bmatrix} \\dot{\\mathbf{X}} \\\\ \\dot{\\boldsymbol{\\theta}}_{\\Delta v}
        \\end{bmatrix} =
        \\begin{bmatrix} \\mathbf{f} \\\\ \\mathbf{0}_{4\\times 1} \\end{bmatrix}
        + \\mathbf{B}_{\\Delta v}\\, \\mathbf{u}^{\\text{smooth}}_{\\Delta v}

    with :math:`\\mathbf{B}_{\\Delta v} = [\\mathbf{0}_{3\\times3};\\,
    \\mathbf{I}_3;\\, \\mathbf{0}_{4\\times3}]`. Because this model returns an
    acceleration, the propagator supplies :math:`\\mathbf{B}_{\\Delta v}`
    implicitly and the maneuver parameters are static state entries.

    Parameters
    ----------
    spacecraft : Body
        The maneuvering body. Its ``spice_id`` keys the parameter lookup.
    maneuvers : ImpulsiveBurn or list of ImpulsiveBurn
        The maneuvers to smooth. Each supplies the nominal delta-v vector and
        nominal epoch ``t_burn`` (TDB seconds past J2000). These are the same
        objects a leg-based :class:`~scarabaeus.MissionSequence` would consume.
    base_frame : Frame, optional
        Reference frame the delta-v vectors are expressed in. Defaults to J2000.
    t0 : float, optional
        Arc start epoch (TDB seconds past J2000) used as the lower limit of the
        normalization integral. When ``None``, the pulse is normalized over the
        whole real line (equivalent to assuming the maneuver sits many pulse
        widths after the arc start), which is the usual case.
    smoothing_factor : float, optional
        The smoothing factor :math:`a`, in :math:`s^{-1}`. Defaults to ``1.0``.

    Raises
    ------
    SmoothedImpulsiveBurn.NoManeuversError
        If ``maneuvers`` is empty.
    SmoothedImpulsiveBurn.InvalidSmoothingFactorError
        If ``smoothing_factor`` is not strictly positive.

    See Also
    --------
    scarabaeus.ImpulsiveBurn : Instantaneous delta-v applied across legs.
    scarabaeus.FiniteBurn : Continuous finite-duration thrust model.

    Notes
    -----
    For maneuver ``i`` (1-based), this model reads two solve-for parameters
    from the propagated state, falling back to the values supplied at
    construction when they are absent:

    ``dv_smooth{i}``
        The delta-v vector (size 3, units km/s).
    ``dt_dv{i}``
        A correction to the *nominal* maneuver epoch (size 1, units sec),
        so the effective epoch is ``t_burn + dt_dv``. Estimating an offset
        rather than the absolute epoch keeps the filter well conditioned:
        an absolute TDB epoch is order :math:`10^{8}` s, and its correction is
        typically order seconds. The partial is unchanged by this shift, since
        :math:`\\partial t_{\\Delta v} / \\partial \\delta t_{\\Delta v} = 1`.

    When exactly one maneuver is modeled, the unsuffixed names ``dv_smooth``
    and ``dt_dv`` are also accepted.

    References
    ----------
    .. [1] Tanh-based smoothing parameterization of impulsive maneuvers for
       state-transition-tensor propagation, Eqs. (17)-(20).
    """

    class NoManeuversError(Exception):
        """Raised when the model is built without any maneuver."""

    class InvalidSmoothingFactorError(Exception):
        """Raised when the smoothing factor is not strictly positive."""

    class NoSmoothingFactorRootError(Exception):
        """Raised when the optimal-smoothing-factor criterion has no root."""

    #: Largest attainable ``eta``. The criterion function
    #: :math:`h(u) = u\\,\\operatorname{sech}^2 u\\,(1 + 2u\\tanh u)` peaks at
    #: ``h = 1.098686`` for ``u = 1.182514``, so no smoothing factor can satisfy
    #: the midpoint relation beyond this. The published choice, 0.9, sits
    #: comfortably inside.
    ETA_MAX = 1.098686

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        spacecraft: Body,
        maneuvers,
        base_frame: Frame = None,
        t0: float = None,
        smoothing_factor: float = 1.0,
    ):
        if isinstance(maneuvers, ImpulsiveBurn):
            maneuvers = [maneuvers]
        maneuvers = list(maneuvers) if maneuvers is not None else []
        if not maneuvers:
            raise self.NoManeuversError(
                "SmoothedImpulsiveBurn requires at least one ImpulsiveBurn; "
                "received an empty maneuver list."
            )

        if not (float(smoothing_factor) > 0.0):
            raise self.InvalidSmoothingFactorError(
                "The smoothing factor 'a' must be strictly positive. "
                f"Received {smoothing_factor}."
            )

        self._spacecraft = spacecraft
        self._maneuvers = maneuvers
        self._frame = base_frame if base_frame is not None else J2000
        self._t0 = t0
        self._a = float(smoothing_factor)

        # Nominal delta-v vectors (km/s) and nominal epochs (TDB sec), pinned to
        # plain floats so the per-step evaluation does no unit arithmetic.
        self._dv_nominal = [
            np.asarray(
                ArrayWUnits.get_value_in_target_units(mvr.delta_v, km * sec**-1),
                dtype=float,
            ).reshape(3)
            for mvr in maneuvers
        ]
        self._t_burn_nominal = [float(mvr.t_burn) for mvr in maneuvers]

        # Caller-supplied overrides, mirroring the other maneuver models.
        self._dv_map: dict[int, np.ndarray] = {}
        self._dt_map: dict[int, float] = {}

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def spacecraft(self) -> Body:
        """The maneuvering body."""
        return self._spacecraft

    @property
    def maneuvers(self) -> list:
        """The list of :class:`~scarabaeus.ImpulsiveBurn` objects being smoothed."""
        return self._maneuvers

    @property
    def base_frame(self) -> Frame:
        """Reference frame the delta-v vectors are expressed in."""
        return self._frame

    @property
    def smoothing_factor(self) -> float:
        """
        The smoothing factor :math:`a`, in :math:`s^{-1}`.

        The pulse has an effective width of order :math:`1/a` and a peak
        acceleration of order :math:`a\\,\\|\\Delta v\\|/2`.

        Accuracy against the true impulse improves as :math:`O(1/a^2)` (the
        pulse is symmetric, so its leading error term is quadratic), but only
        while the integrator still resolves it. Past that point the adaptive
        step controller steps clean over the pulse, its error estimate never
        spikes, so it never refines, and the maneuver is silently under- or
        un-applied. That failure is not gradual: in the LEO case in
        ``tests/python_tests/unit_testing/dynamics_tests/test_SmoothedImpulsiveBurn.py``, the final
        position error falls from 1.1e-2 km to 7.2e-4 km to 2.9e-5 km for
        :math:`a = 0.05, 0.2, 1.0\\,s^{-1}`, then jumps to 1.7e+1 km at
        :math:`a = 5\\,s^{-1}`.

        Scarabaeus exposes no maximum-step control, so :math:`a` is the only
        lever: tightening ``atol``/``rtol`` does not rescue an unresolved
        pulse. Choose the smallest :math:`a` that meets the accuracy target,
        and validate it once against the equivalent two-leg impulsive
        propagation before trusting a new configuration.

        .. important::

            Trajectory fidelity and deviation-mapping fidelity pull :math:`a`
            in **opposite** directions, and
            :meth:`optimal_smoothing_factor` optimizes the second, not the
            first. It returns :math:`a \\propto 1/\\delta t_{\\Delta v}`, which
            for realistic epoch uncertainties is a far wider pulse than pure
            trajectory reproduction wants. Measured on the LEO case:

            ===================  ============  =================
            :math:`\\delta t`     :math:`a^*`   trajectory error
            ===================  ============  =================
            50 s                 0.0311        2.97e-02 km
            100 s                0.0155        1.18e-01 km
            300 s                0.0052        1.04e+00 km
            ===================  ============  =================

            against 2.9e-05 km at :math:`a = 1`. Use the criterion when the
            deliverable is a sensitivity mapping over an uncertain maneuver
            epoch; use a large :math:`a` when the deliverable is the
            trajectory itself. If you need both, they are different runs.
        """
        return self._a

    @property
    def t0(self) -> float:
        """Arc start epoch used as the lower limit of the normalization integral."""
        return self._t0

    @property
    def estimable_parameter_names(self) -> tuple[str, ...]:
        """
        The solve-for parameter names this model responds to.

        Returns
        -------
        tuple of str
            ``('dv_smooth1', 'dt_dv1', 'dv_smooth2', ...)``, plus the
            unsuffixed aliases when a single maneuver is modeled.
        """
        names: list[str] = []
        for idx in range(1, len(self._maneuvers) + 1):
            names.extend([f"dv_smooth{idx}", f"dt_dv{idx}"])
        if len(self._maneuvers) == 1:
            names.extend(["dv_smooth", "dt_dv"])
        return tuple(names)

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _get_dv(self, current_state: dict, body: Body, index: int) -> np.ndarray:
        """
        Resolve the delta-v vector for maneuver ``index`` (1-based).

        Priority: the propagated state, then a caller-supplied override, then
        the nominal value from the ``ImpulsiveBurn``. The state comes first so
        an estimated delta-v takes effect, matching how every other model
        resolves its parameters.

        Parameters
        ----------
        current_state : dict
            Dictionary containing the current state parameters.
        body : Body
            Owner whose ``spice_id`` keys the lookup.
        index : int
            1-based maneuver index.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` delta-v in km/s.
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            for name in self._param_aliases("dv_smooth", index):
                key = (name, spice_id)
                if key in current_state:
                    val = current_state[key]
                    val = val.values if hasattr(val, "values") else val
                    return np.asarray(val, dtype=float).reshape(3)

            if index in self._dv_map:
                return self._dv_map[index]

        return self._dv_nominal[index - 1]

    def _get_dt(self, current_state: dict, body: Body, index: int) -> float:
        """
        Resolve the epoch correction for maneuver ``index`` (1-based).

        Parameters
        ----------
        current_state : dict
            Dictionary containing the current state parameters.
        body : Body
            Owner whose ``spice_id`` keys the lookup.
        index : int
            1-based maneuver index.

        Returns
        -------
        float
            Offset in seconds added to the nominal epoch. Defaults to ``0.0``.
        """
        spice_id = getattr(body, "spice_id", None)

        if spice_id is not None:
            for name in self._param_aliases("dt_dv", index):
                key = (name, spice_id)
                if key in current_state:
                    val = current_state[key]
                    val = val.values if hasattr(val, "values") else val
                    return float(np.atleast_1d(val)[0])

            if index in self._dt_map:
                return self._dt_map[index]

        return 0.0

    def _param_aliases(self, base: str, index: int) -> tuple[str, ...]:
        """
        Return the accepted state-key names for a parameter of one maneuver.

        The unsuffixed alias is offered only for a single-maneuver model, where
        it is unambiguous.

        Parameters
        ----------
        base : str
            Unsuffixed parameter name, ``"dv_smooth"`` or ``"dt_dv"``.
        index : int
            1-based maneuver index appended to ``base``.

        Returns
        -------
        tuple of str
            ``(f"{base}{index}", base)`` for a single-maneuver model, otherwise
            ``(f"{base}{index}",)``.
        """
        if len(self._maneuvers) == 1:
            return (f"{base}{index}", base)
        return (f"{base}{index}",)

    def _pulse(self, epoch: float, t_dv: float) -> float:
        """
        The normalized pulse :math:`s_{\\Delta v}/N` at ``epoch``.

        Parameters
        ----------
        epoch : float
            Current epoch (TDB seconds past J2000).
        t_dv : float
            Effective maneuver epoch (TDB seconds past J2000).

        Returns
        -------
        float
            Pulse value in :math:`s^{-1}`, such that multiplying by a delta-v
            in km/s yields an acceleration in km/s^2.

        Notes
        -----
        ``sech^2`` is evaluated as ``1 - tanh^2`` rather than ``1/cosh^2``: for
        arguments far from the pulse, ``cosh`` overflows to infinity, while
        ``tanh`` saturates cleanly at +/-1 and yields an exact zero.
        """
        th = np.tanh(self._a * (epoch - t_dv))
        sech2 = 1.0 - th * th
        return sech2 / self._normalization(t_dv)

    def _normalization(self, t_dv: float) -> float:
        """
        The integral of the raw pulse over the arc, Eq. (18).

        Parameters
        ----------
        t_dv : float
            Effective maneuver epoch (TDB seconds past J2000).

        Returns
        -------
        float
            :math:`\\frac{1}{a}\\big(1 + \\tanh(a (t_{\\Delta v} - t_0))\\big)`,
            reducing to :math:`2/a` when no arc start is known.
        """
        if self._t0 is None:
            return 2.0 / self._a
        return (1.0 + np.tanh(self._a * (t_dv - self._t0))) / self._a

    def _d_normalization_d_t(self, t_dv: float) -> float:
        """
        Derivative of :meth:`_normalization` with respect to the maneuver epoch.

        Parameters
        ----------
        t_dv : float
            Effective maneuver epoch, in TDB seconds past J2000.

        Returns
        -------
        float
            :math:`\\operatorname{sech}^2\\big(a (t_{\\Delta v} - t_0)\\big)`,
            and zero when no arc start is known.
        """
        if self._t0 is None:
            return 0.0
        th = np.tanh(self._a * (t_dv - self._t0))
        return 1.0 - th * th

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Computes the smoothed acceleration from all modeled maneuvers.

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position vector. Unused: the pulse depends only on
            time and the maneuver parameters.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch (TDB seconds past J2000).
        body : Body
            The body whose maneuver parameters are read.

        Returns
        -------
        acc : np.ndarray
            ``(3,)`` acceleration in km/s^2.
        """
        acc = np.zeros(3)
        for index in range(1, len(self._maneuvers) + 1):
            dv = self._get_dv(current_state, body, index)
            t_dv = self._t_burn_nominal[index - 1] + self._get_dt(
                current_state, body, index
            )
            acc = acc + self._pulse(epoch, t_dv) * dv

        return acc

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Partial of the smoothed acceleration with respect to position.

        The pulse is a function of time and the maneuver parameters only, so
        this partial is identically zero.

        Parameters
        ----------
        position : np.ndarray
            Current position vector, in km. Unused.
        current_state : dict
            The propagated state. Unused.
        epoch : float
            Current epoch, in TDB seconds past J2000. Unused.
        body : Body
            The propagated body. Unused.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` zero matrix.
        """
        return np.zeros((3, 3))

    def _compute_partial_by_dv_smooth(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
        index: int = 1,
    ) -> np.ndarray:
        """
        Partial of the smoothed acceleration with respect to a delta-v vector.

        .. math::

            \\frac{\\partial \\mathbf{u}}{\\partial \\Delta\\mathbf{v}} =
                \\frac{s_{\\Delta v}(t)}{N}\\, \\mathbf{I}_3

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position vector. Unused.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch (TDB seconds past J2000).
        body : Body
            The body whose maneuver parameters are read.
        index : int, optional
            1-based maneuver index. Defaults to ``1``.

        Returns
        -------
        np.ndarray
            ``(3, 3)`` Jacobian in :math:`s^{-1}`.
        """
        t_dv = self._t_burn_nominal[index - 1] + self._get_dt(
            current_state, body, index
        )
        return self._pulse(epoch, t_dv) * np.eye(3)

    def _compute_partial_by_dt_dv(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
        index: int = 1,
    ) -> np.ndarray:
        """
        Partial of the smoothed acceleration with respect to the epoch offset.

        Differentiating Eq. (19) with respect to :math:`t_{\\Delta v}`, and using
        :math:`\\partial \\tau / \\partial t_{\\Delta v} = -1` for
        :math:`\\tau = t - t_{\\Delta v}`:

        .. math::

            \\frac{\\partial \\mathbf{u}}{\\partial t_{\\Delta v}} =
            \\left[
                \\frac{2 a \\operatorname{sech}^2(a\\tau)\\tanh(a\\tau)}{N}
                - \\frac{\\operatorname{sech}^2(a\\tau)\\,
                         \\operatorname{sech}^2(a (t_{\\Delta v} - t_0))}{N^2}
            \\right] \\Delta\\mathbf{v}

        The first term is the pulse sliding along the time axis; the second is
        the change in the normalization as the truncated integral shifts, and
        vanishes whenever the maneuver sits well inside the arc.

        Parameters
        ----------
        position : np.ndarray
            3x1 spacecraft position vector. Unused.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch (TDB seconds past J2000).
        body : Body
            The body whose maneuver parameters are read.
        index : int, optional
            1-based maneuver index. Defaults to ``1``.

        Returns
        -------
        np.ndarray
            ``(3, 1)`` Jacobian in :math:`km/s^3`.
        """
        dv = self._get_dv(current_state, body, index)
        t_dv = self._t_burn_nominal[index - 1] + self._get_dt(
            current_state, body, index
        )

        th = np.tanh(self._a * (epoch - t_dv))
        sech2 = 1.0 - th * th

        norm = self._normalization(t_dv)
        d_pulse = (2.0 * self._a * sech2 * th) / norm - (
            sech2 * self._d_normalization_d_t(t_dv)
        ) / (norm * norm)

        return (d_pulse * dv).reshape(3, 1)

    # ------------------------------------------------------------------
    # region Optimal smoothing factor
    # ------------------------------------------------------------------
    @staticmethod
    def mapped_dv_ratio(
        a: float, dt_dv: float, t_dv: float = None, t0: float = None
    ) -> float:
        """
        Mapped velocity deviation at the interval midpoint, per delta-v, Eq. (28).

        When only a maneuver-epoch deviation :math:`\\delta t_{\\Delta v}` is
        present, the mapped velocity deviation follows from the extended
        first- and second-order sensitivities

        .. math::

            \\delta \\dot v_x \\approx A^{\\Delta v}_{4,10}\\,\\delta t_{\\Delta v}
            + \\tfrac{1}{2}\\Lambda^{\\Delta v}_{4,10,10}\\,\\delta t_{\\Delta v}^2

        with :math:`A^{\\Delta v}_{4,10} = \\Delta v_x\\,\\partial_{t_{\\Delta v}} P`
        and :math:`\\Lambda^{\\Delta v}_{4,10,10} = \\Delta v_x\\,
        \\partial^2_{t_{\\Delta v}} P`, where :math:`P` is this model's
        normalized pulse integral. Carrying those derivatives through and
        integrating in time gives Eq. (28),

        .. math::

            \\frac{\\delta v_x}{\\Delta v_x} =
            \\frac{a}{S_2 + 1}
            \\Big[S_1^2 + (S_2 - 1) S_1 - S_2\\Big]
            \\Big(\\delta t_{\\Delta v} + a S_1 \\delta t_{\\Delta v}^2\\Big)

        with :math:`S_1 = \\tanh(a(t - t_{\\Delta v}))` and
        :math:`S_2 = \\tanh(a(t_{\\Delta v} - t_0))`, evaluated here at the
        midpoint :math:`t' = t_{\\Delta v} + \\tfrac{1}{2}\\delta t_{\\Delta v}`
        so that :math:`S_1 = \\tanh(a\\,\\delta t_{\\Delta v}/2)`.

        :math:`\\Delta v_x` divides out, which is why the criterion is
        independent of the maneuver magnitude.

        Parameters
        ----------
        a : float
            Smoothing factor, :math:`s^{-1}`.
        dt_dv : float
            Maneuver-epoch deviation :math:`\\delta t_{\\Delta v}`, in seconds.
        t_dv, t0 : float, optional
            Maneuver epoch and arc start. They enter only through
            :math:`S_2`; when either is omitted, :math:`S_2 = 1` is used, the
            limit reached whenever the maneuver sits more than a few pulse
            widths after the arc start. This is why :math:`t_{\\Delta v}` has
            almost no influence on the resulting smoothing factor.

        Returns
        -------
        float
            :math:`\\delta v_x(t') / \\Delta v_x`, dimensionless.
        """
        S1 = np.tanh(a * dt_dv / 2.0)
        S2 = 1.0 if (t_dv is None or t0 is None) else np.tanh(a * (t_dv - t0))

        bracket = S1 * S1 + (S2 - 1.0) * S1 - S2
        return (a / (S2 + 1.0)) * bracket * (dt_dv + a * S1 * dt_dv * dt_dv)

    @classmethod
    def criterion_residual(
        cls,
        a: float,
        dt_dv: float,
        eta: float = 0.9,
        t_dv: float = None,
        t0: float = None,
    ) -> float:
        """
        The criterion function :math:`g_{\\Delta v}`, Eq. (29).

        Requires the mapped velocity deviation at the midpoint to be a fixed
        fraction of the maneuver, with the sign that matches the real
        deviation: :math:`\\delta v_x(t') = -\\eta \\Delta v_x` for
        :math:`\\delta t_{\\Delta v} > 0` and :math:`+\\eta \\Delta v_x` for
        :math:`\\delta t_{\\Delta v} < 0`. The residual is
        :meth:`mapped_dv_ratio` plus :math:`\\eta\\,\\operatorname{sign}(\\delta t_{\\Delta v})`.

        Parameters
        ----------
        a : float
            Smoothing factor, in :math:`s^{-1}`.
        dt_dv : float
            Maneuver-epoch deviation :math:`\\delta t_{\\Delta v}`, in seconds.
        eta : float, optional
            Target fraction :math:`\\eta` of the maneuver. Defaults to ``0.9``.
        t_dv, t0 : float, optional
            Maneuver epoch and arc start, passed to :meth:`mapped_dv_ratio`.
            Defaults to ``None``.

        Returns
        -------
        float
            Zero when ``a`` satisfies the midpoint relation.
        """
        return cls.mapped_dv_ratio(a, dt_dv, t_dv, t0) + eta * np.sign(dt_dv)

    @classmethod
    def optimal_smoothing_factor(
        cls,
        dt_dv: float,
        eta: float = 0.9,
        t_dv: float = None,
        t0: float = None,
        a_bounds: tuple = (1e-3, 4.0),
        n_grid: int = 4000,
        branch: str = "low",
    ) -> float:
        """
        Solve :math:`g_{\\Delta v} = 0` for the smoothing factor.

        Implements the grid search for root approximation (GSRA): ``a`` is
        discretized over ``a_bounds``, :meth:`criterion_residual` is evaluated
        on the grid, and each sign change is refined by bisection. No analytic
        solution exists.

        In the :math:`S_2 \\to 1` limit the criterion collapses to a function of
        the single group :math:`u = a\\,\\delta t_{\\Delta v}/2`,

        .. math::

            u \\operatorname{sech}^2(u)\\,(1 + 2u\\tanh u) = \\eta

        so the smoothing factor scales as :math:`a \\propto 1/\\delta
        t_{\\Delta v}` and is independent of the maneuver magnitude, the
        maneuver epoch, and the orbit. Because :math:`h(u)` rises then falls,
        the equation has **two** roots, selected by ``branch``.

        Parameters
        ----------
        dt_dv : float
            The maneuver-epoch uncertainty to tune for, in seconds. Sign is
            irrelevant: the criterion is symmetric, so ``|dt_dv|`` is used.
        eta : float, optional
            Adjustment coefficient. Defaults to ``0.9``, reported as the value
            minimizing the position-deviation residual. Must be below
            :attr:`ETA_MAX`.
        t_dv, t0 : float, optional
            Passed through to :meth:`mapped_dv_ratio`; see the note there on
            why they barely matter.
        a_bounds : tuple of float, optional
            ``(a_min, a_max)`` search interval, defaults to ``(1e-3, 4.0)``.
        n_grid : int, optional
            Number of grid points for the search. Defaults to ``4000``.
        branch : {'low', 'high'}, optional
            Which of the two roots to return. ``'low'`` (default) gives the
            wider pulse, which is the numerically safer choice: it stays
            further from the step-resolution limit described in
            :attr:`smoothing_factor`.

        Returns
        -------
        float
            The smoothing factor, in :math:`s^{-1}`.

        Raises
        ------
        SmoothedImpulsiveBurn.NoSmoothingFactorRootError
            If ``eta`` is unreachable or no root lies inside ``a_bounds``.

        Examples
        --------
        >>> SmoothedImpulsiveBurn.optimal_smoothing_factor(100.0)
        0.015526...
        """
        if not (0.0 < eta < cls.ETA_MAX):
            raise cls.NoSmoothingFactorRootError(
                f"eta must lie in (0, {cls.ETA_MAX}); the criterion function "
                f"peaks at {cls.ETA_MAX} so no smoothing factor can satisfy "
                f"eta = {eta}."
            )
        if branch not in ("low", "high"):
            raise cls.NoSmoothingFactorRootError(
                f"branch must be 'low' or 'high'. Received {branch!r}."
            )

        dt = abs(float(dt_dv))
        if dt == 0.0:
            raise cls.NoSmoothingFactorRootError(
                "dt_dv must be non-zero: with no maneuver-epoch uncertainty "
                "the criterion is degenerate and any smoothing factor solves it."
            )

        a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
        grid = np.linspace(a_lo, a_hi, int(n_grid))
        g = np.array(
            [cls.criterion_residual(a, dt, eta, t_dv, t0) for a in grid]
        )

        sign_changes = np.where(np.diff(np.sign(g)) != 0)[0]
        if sign_changes.size == 0:
            raise cls.NoSmoothingFactorRootError(
                f"No root of the smoothing-factor criterion for "
                f"dt_dv = {dt_dv} s and eta = {eta} inside "
                f"a in [{a_lo}, {a_hi}]. Expected roots near "
                f"a = {2 * 0.77631 / dt:.4g} and {2 * 1.68209 / dt:.4g} "
                f"for eta = 0.9; widen a_bounds."
            )

        idx = sign_changes[0] if branch == "low" else sign_changes[-1]
        lo, hi = grid[idx], grid[idx + 1]

        # Bisection refinement of the bracketed root.
        f_lo = cls.criterion_residual(lo, dt, eta, t_dv, t0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = cls.criterion_residual(mid, dt, eta, t_dv, t0)
            if f_mid == 0.0 or (hi - lo) < 1e-15 * max(1.0, mid):
                return mid
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # endregion Optimal smoothing factor
    # ------------------------------------------------------------------

    def total_delivered_dv(
        self, epochs: np.ndarray, current_state: dict = None, body: Body = None
    ) -> np.ndarray:
        """
        Numerically integrate the pulse over ``epochs`` to check delivered delta-v.

        Diagnostic helper for the *analytic* normalization: on a grid fine
        enough to resolve the pulse (spacing well below :math:`1/a`) this
        returns the commanded delta-v to roundoff, confirming Eq. (18).

        .. warning::

            This does **not** predict whether a propagation will resolve the
            pulse. An adaptive integrator chooses its own internal steps, which
            are unrelated to the epochs a run reports on: a pulse this helper
            cannot see on a coarse output grid may still be integrated
            perfectly. The only reliable check that a given
            :attr:`smoothing_factor` is safe is to compare a propagation
            against the equivalent two-leg impulsive answer once, as
            ``tests/python_tests/unit_testing/dynamics_tests/test_SmoothedImpulsiveBurn.py`` does.

        Parameters
        ----------
        epochs : np.ndarray
            Monotonic grid of epochs (TDB seconds past J2000).
        current_state : dict, optional
            Parameter overrides; defaults to the nominal values.
        body : Body, optional
            Owner for parameter lookup. Defaults to the model's spacecraft.

        Returns
        -------
        np.ndarray
            ``(3,)`` integrated delta-v in km/s.
        """
        if current_state is None:
            current_state = {}
        if body is None:
            body = self._spacecraft

        epochs = np.asarray(epochs, dtype=float)
        acc = np.array(
            [
                self.compute_acceleration(None, current_state, float(t), body)
                for t in epochs
            ]
        )
        return np.trapezoid(acc, epochs, axis=0)

    # endregion Methods #
    # -------------------#
