# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Body,
    EpochArray,
    Spacecraft,
    Frame,
    DynamicModel,
    DynamicalUtils,
)
import scarabaeus.utils.NumpyWrapper as np


class PiecewisePolynomialAcceleration(DynamicModel):
    """Models an empirical acceleration as a piecewise polynomial in time over
    a fixed batch grid.

    The arc is divided into :math:`N_b` batches of length :math:`L` starting at
    :math:`t_0`, exactly as in
    :class:`~scarabaeus.PiecewiseFirstOrderGaussMarkov`. Within batch :math:`k`,
    with the local time :math:`\\tau = (t - t_0 - kL)/L \\in [0, 1]`, the
    acceleration is a polynomial of the chosen ``order``:

    ``order=0`` (zero-order hold)
        One constant 3-vector per batch,

        .. math::

            \\mathbf a(t) = \\mathbf c_k ,

        which jumps at the batch boundaries.

    ``order=1`` (first-order hold)
        One 3-vector per batch *boundary* (node), linearly interpolated in
        between,

        .. math::

            \\mathbf a(t) = (1 - \\tau)\\,\\mathbf n_k + \\tau\\,\\mathbf n_{k+1} ,

        so the acceleration is continuous across batches by construction: the
        end of batch :math:`k` and the start of batch :math:`k+1` share the
        node :math:`\\mathbf n_{k+1}`.

    The coefficients are components in the frame chosen with ``frame``, and
    the applied acceleration is :math:`\\mathbf T\\,\\mathbf a(t)`, where
    :math:`\\mathbf T` rotates that frame into the propagation frame.

    Unlike the Gauss-Markov models, nothing here is integrated: the
    coefficients are constants of the dynamics, estimated as a ``'static'``
    parameter. The acceleration is linear in them, so the partial the filter
    needs is the interpolation matrix itself,

    .. math::

        \\frac{\\partial \\mathbf a}{\\partial \\mathbf c} =
        \\begin{cases}
        [\\,\\mathbf 0 \\;\\cdots\\; \\mathbf I_3 \\;\\cdots\\; \\mathbf 0\\,]
            & \\text{order } 0 \\\\
        [\\,\\mathbf 0 \\;\\cdots\\; (1-\\tau)\\mathbf I_3 \\;\\;
            \\tau\\mathbf I_3 \\;\\cdots\\; \\mathbf 0\\,]
            & \\text{order } 1
        \\end{cases}

    which is exact rather than a difference approximation.

    Parameters
    ----------
    origin : Body
        Propagation origin.
    spacecraft : Spacecraft
        Spacecraft the empirical acceleration acts on.
    base_frame : Frame
        Reference frame of the propagator.
    t0 : EpochArray or float
        Reference epoch at which batch 0 starts, as an ``EpochArray`` or as raw
        ephemeris seconds (TDB). It is the origin of the batch grid, not the
        start of the current propagation: in a multi-leg
        :class:`~scarabaeus.MissionSequence` every leg must be given the same
        ``t0`` so the batches tile the whole arc and a coefficient means the
        same thing in every leg.
    batch_length : float
        Duration of each batch, in seconds.
    n_batches : int
        Number of batches.
    order : int
        Polynomial order within a batch: ``0`` for a zero-order hold (constant
        per batch), ``1`` for a first-order hold (linear per batch, continuous
        across batches). Required, since the two orders size the coefficient
        state differently.
    coefficients : np.ndarray, optional
        Default coefficient vector, used when the propagated state carries no
        ``a_ppoly`` entry. This is what makes the model usable as a plain force
        model — a known piecewise acceleration applied without estimating
        anything.
    frame : str, optional
        Frame the coefficients are expressed in: ``"inertial"`` (the
        propagation frame, the default), ``"RTN"`` (radial / along-track /
        cross-track, the RTN triad) or ``"body_fixed"``.
    body_fixed_frame : str or Frame, optional
        Frame the coefficients are expressed in, for ``frame="body_fixed"``
        (for example ``'IAU_BENNU'``).
    relative_body : str or Body, optional
        Body the RTN triad is built about, when that is not the propagation
        origin. Defaults to the propagation origin.

    Raises
    ------
    InvalidOrderError
        If ``order`` is not one of :attr:`ORDERS`.
    InvalidBatchGridError
        If ``batch_length <= 0`` or ``n_batches < 1``.
    InvalidEpochError
        If ``t0`` is neither an ``EpochArray`` nor a real number.
    CoefficientSizeError
        If ``coefficients`` does not match the order and batch count.
    DynamicalUtils.InvalidFrameError
        If ``frame`` is not one of the supported frames.
    DynamicalUtils.MissingBodyFixedFrameError
        If ``frame="body_fixed"`` is requested without ``body_fixed_frame``.

    See Also
    --------
    scarabaeus.PiecewiseFirstOrderGaussMarkov : Batch-segmented stochastic variant.
    scarabaeus.FourierAcceleration : Periodic empirical acceleration.
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.DynamicModel : Abstract base class for all force model components.

    Notes
    -----
    The estimated state is one stacked ``("a_ppoly", spice_id)`` entry, ordered
    batch-major (``order=0``) or node-major (``order=1``):

    .. code-block:: text

        order 0:  [ c0_x c0_y c0_z | c1_x c1_y c1_z | ... | c(N-1)_x ... ]   3*N
        order 1:  [ n0_x n0_y n0_z | n1_x n1_y n1_z | ... | nN_x     ... ]   3*(N+1)

    with :math:`N` the number of batches. :meth:`coefficient_labels` returns the
    matching names. The state vector must contain an entry of the form

    - ``('a_ppoly', n_coefficients, 'estimated', 'static', owner, value)``

    for the coefficients to be estimated.

    Batch membership follows the PFOGM convention: interior boundaries belong to
    the batch that starts there, and the closing endpoint belongs to the last
    batch. The propagator splits its arcs at the batch boundaries and pins each
    arc to its batch with :meth:`pin_batch_window`, so the zero-order hold's
    jump never falls inside an integration step.

    In the ``"RTN"`` frame the coefficients are radial, along-track and
    cross-track components, and the coefficient partial above is
    left-multiplied by :math:`\\mathbf T`. The RTN triad is built from the
    state, so the acceleration also gains the position and velocity partials
    :meth:`~scarabaeus.DynamicalUtils.frame_partials` returns; the inertial and
    body-fixed rotations depend on the epoch alone.

    Examples
    --------
    A first-order hold over four one-day batches, estimated:

    .. code-block:: python

        import scarabaeus as scb

        state = [
            ("position", 3, "estimated", "dynamic", sc, [7000, 0, 0]),
            ("velocity", 3, "estimated", "dynamic", sc, [0, 7.5, 0]),
            ("a_ppoly", 15, "estimated", "static", sc, [0.0] * 15),
        ]

        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            t0=arc_start,
            piecewise_polynomial=True,
            ppoly_order=1,
            ppoly_batch_length=86400.0,
            ppoly_n_batches=4,
        )
    """

    #: Polynomial orders the model supports: zero-order and first-order hold.
    ORDERS = (0, 1)

    class InvalidOrderError(ValueError):
        """Raised when ``order`` is not a supported polynomial order."""

    class InvalidBatchGridError(ValueError):
        """Raised when ``batch_length`` or ``n_batches`` cannot form a grid."""

    class InvalidEpochError(TypeError):
        """Raised when ``t0`` is neither an ``EpochArray`` nor a real number."""

    class CoefficientSizeError(ValueError):
        """Raised when the ``a_ppoly`` state does not match the order and grid."""

    class EpochOutsideGridError(ValueError):
        """Raised when an epoch falls outside the batch grid."""

    def __init__(
        self,
        origin: Body,
        spacecraft: Spacecraft,
        base_frame: Frame,
        t0: EpochArray | float,
        batch_length: float,
        n_batches: int,
        order: int,
        coefficients: np.ndarray | None = None,
        frame: str = "inertial",
        body_fixed_frame=None,
        relative_body=None,
    ):
        self._order = self._validate_order(order)
        self._validate_grid(batch_length, n_batches)

        self._origin = origin
        self._origin_name = self._origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame
        self._frame, self._body_fixed_frame, self._relative_body = (
            DynamicalUtils.resolve_frame(frame, body_fixed_frame, relative_body)
        )
        # Per-epoch SPICE lookups of the frame, kept by DynamicalUtils.
        self._frame_memo = {}

        self._t0 = self._epoch_to_et(t0)
        self._batch_length = float(batch_length)
        self._n_batches = int(n_batches)

        self._default_coefficients = (
            self._validate_coefficients(np.asarray(coefficients, dtype=float).ravel())
            if coefficients is not None
            else None
        )

        # Batch pinned for the integration arc in progress; see
        # :meth:`pin_batch_window`.
        self._pinned_batch: int | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_order(order) -> int:
        """
        Check the polynomial order.

        Parameters
        ----------
        order : int
            Requested order.

        Returns
        -------
        int
            The order, as an ``int``.

        Raises
        ------
        InvalidOrderError
            If ``order`` is not one of :attr:`ORDERS`.
        """
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or int(order) not in PiecewisePolynomialAcceleration.ORDERS
        ):
            raise PiecewisePolynomialAcceleration.InvalidOrderError(
                f"order must be 0 (zero-order hold, constant per batch) or 1 "
                f"(first-order hold, linear per batch and continuous across "
                f"batches), got {order!r}."
            )
        return int(order)

    @staticmethod
    def _validate_grid(batch_length, n_batches) -> None:
        """
        Check that the batch length and count form a grid.

        Parameters
        ----------
        batch_length : float
            Duration of each batch, in seconds.
        n_batches : int
            Number of batches.

        Raises
        ------
        InvalidBatchGridError
            If ``batch_length <= 0`` or ``n_batches < 1``.
        """
        if batch_length is None or float(batch_length) <= 0.0:
            raise PiecewisePolynomialAcceleration.InvalidBatchGridError(
                f"batch_length must be > 0 seconds, got {batch_length!r}."
            )
        if n_batches is None or int(n_batches) != n_batches or n_batches < 1:
            raise PiecewisePolynomialAcceleration.InvalidBatchGridError(
                f"n_batches must be an integer >= 1, got {n_batches!r}."
            )

    @staticmethod
    def _epoch_to_et(t0: EpochArray | float) -> float:
        """
        Normalise ``t0`` to raw ephemeris seconds (TDB).

        Both forms are accepted because the callers disagree: a script
        configures the model with an ``EpochArray``, while
        :class:`~scarabaeus.Propagator` binds its force models with the raw ET
        float it hands the integrator. Batch windows are compared against that
        same raw ET, so an ``EpochArray`` is read through ``.et`` rather than
        ``.times``, which follows the array's display system.

        Parameters
        ----------
        t0 : EpochArray or float
            Reference epoch of batch 0.

        Returns
        -------
        float
            Epoch in ephemeris seconds past J2000 (TDB).

        Raises
        ------
        InvalidEpochError
            If ``t0`` is neither an ``EpochArray`` nor a real number.
        """
        if isinstance(t0, EpochArray):
            return float(np.asarray(t0.et).ravel()[0])
        if isinstance(t0, (int, float, np.floating, np.integer)):
            return float(t0)
        raise PiecewisePolynomialAcceleration.InvalidEpochError(
            "t0 must be an EpochArray or ephemeris seconds (TDB) as a float; "
            f"received {type(t0).__name__}."
        )

    def _validate_coefficients(self, values: np.ndarray) -> np.ndarray:
        """
        Check a coefficient vector against the order and batch count.

        Parameters
        ----------
        values : np.ndarray
            Flat coefficient vector.

        Returns
        -------
        np.ndarray
            The same vector, validated.

        Raises
        ------
        CoefficientSizeError
            If the length is not :attr:`n_coefficients`.
        """
        expected = self.n_coefficients
        if values.size != expected:
            per = "batch" if self._order == 0 else "node (n_batches + 1 nodes)"
            raise self.CoefficientSizeError(
                f"an order-{self._order} piecewise polynomial over "
                f"{self._n_batches} batches has {expected} coefficients "
                f"(3 per {per}), got {values.size}. Size the 'a_ppoly' state "
                f"to {expected}, or check ppoly_order and ppoly_n_batches."
            )
        return values

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def origin(self) -> Body:
        """Celestial body serving as origin of the state vector."""
        return self._origin

    @property
    def origin_name(self) -> str:
        """Name of the origin body."""
        return self._origin_name

    @property
    def ref_frame(self):
        """Reference frame used by the model."""
        return self._ref_frame

    @property
    def frame(self) -> str:
        """Frame the coefficients are expressed in."""
        return self._frame

    @property
    def body_fixed_frame(self) -> str | None:
        """Body-fixed frame the coefficients are expressed in, if any."""
        return self._body_fixed_frame

    @property
    def relative_body(self) -> str | None:
        """Body the RTN triad is referred to, if any."""
        return self._relative_body

    @property
    def order(self) -> int:
        """Polynomial order within a batch: ``0`` or ``1``."""
        return self._order

    @property
    def t0(self) -> float:
        """Reference epoch of the first batch, in ephemeris seconds."""
        return self._t0

    @property
    def batch_length(self) -> float:
        """Duration of one batch, in seconds."""
        return self._batch_length

    @property
    def n_batches(self) -> int:
        """Number of batches."""
        return self._n_batches

    @property
    def n_coefficients(self) -> int:
        """Length of the stacked ``a_ppoly`` state: ``3*n_batches`` or ``3*(n_batches+1)``."""
        return 3 * (self._n_batches + self._order)

    @property
    def default_coefficients(self):
        """Coefficients applied when the state carries no ``a_ppoly`` entry."""
        return self._default_coefficients

    @default_coefficients.setter
    def default_coefficients(self, values):
        """Set the coefficients used when the state carries no ``a_ppoly``."""
        self._default_coefficients = (
            self._validate_coefficients(np.asarray(values, dtype=float).ravel())
            if values is not None
            else None
        )

    @property
    def depends_on_velocity(self) -> bool:
        """
        Whether the applied acceleration is a function of velocity.

        True for the RTN frame, whose triad is built from the velocity. The
        force model reads this to decide whether a
        :math:`\\partial \\mathbf a / \\partial \\mathbf v` block is needed.
        """
        return self._frame == "RTN"

    def coefficient_labels(self) -> list[str]:
        """
        Names of the stacked coefficients, in state order.

        Returns
        -------
        list of str
            ``['batch0_x', 'batch0_y', ...]`` for ``order=0`` or
            ``['node0_x', 'node0_y', ...]`` for ``order=1``, matching the
            layout of the ``a_ppoly`` state. The axes follow ``frame``:
            ``R``, ``T``, ``N`` in the RTN frame.
        """
        prefix = "batch" if self._order == 0 else "node"
        return [
            f"{prefix}{j}_{axis}"
            for j in range(self._n_batches + self._order)
            for axis in DynamicalUtils.ACCELERATION_FRAME_AXES[self._frame]
        ]

    # ------------------------------------------------------------------
    # Batch grid
    # ------------------------------------------------------------------
    def pin_batch_window(self, t_start: float, t_end: float) -> int | None:
        """
        Pin the batch that owns an integration arc, so the dynamics stay
        continuous across it.

        Batch membership is right-continuous, so the epoch closing an arc that
        ends on a batch boundary belongs to the *next* batch. For a zero-order
        hold the acceleration jumps there; for a first-order hold its slope
        does. Either way an integrator evaluating its final stage on the
        boundary sees a discontinuity at the very point it has to land on.
        Pinning the batch for the arc removes it from the arc's interior and
        endpoint alike.

        Only arcs that lie within a single batch are pinned; the propagator
        already splits arcs at interior batch boundaries, so that is the normal
        case. Anything else clears the pin and leaves the per-epoch lookup in
        charge.

        Parameters
        ----------
        t_start, t_end : float
            Arc bounds in ephemeris seconds (TDB); order does not matter.

        Returns
        -------
        int or None
            The pinned batch index, or ``None`` if the arc was not pinned.
        """
        self._pinned_batch = None

        t_lo, t_hi = min(t_start, t_end), max(t_start, t_end)
        try:
            k = self._get_active_batch_index(0.5 * (t_lo + t_hi))
        except ValueError:
            return None

        # Only pin when the whole arc really is inside that batch.
        batch_start = self._t0 + k * self._batch_length
        batch_end = batch_start + self._batch_length
        tol = max(1e-12, 10 * np.finfo(float).eps * max(1.0, abs(batch_end)))
        if t_lo < batch_start - tol or t_hi > batch_end + tol:
            return None

        self._pinned_batch = k
        return k

    def clear_batch_window(self) -> None:
        """Drop any pinned batch, restoring the per-epoch lookup."""
        self._pinned_batch = None

    def _get_active_batch_index(self, epoch: float) -> int:
        """
        Return the index of the batch active at ``epoch``.

        Batches tile ``[t0, t0 + n_batches * batch_length]``. Interior
        boundaries belong to the batch that starts there; the closing endpoint
        belongs to the last batch, so an arc sized as
        ``n_batches = arc_length / batch_length`` integrates to its final epoch
        instead of falling off the grid on the last step.

        While an arc is pinned by :meth:`pin_batch_window`, that batch is
        returned regardless of ``epoch``.

        Parameters
        ----------
        epoch : float
            Evaluation epoch in ephemeris seconds (TDB).

        Returns
        -------
        int
            Active batch index in ``[0, n_batches - 1]``.

        Raises
        ------
        EpochOutsideGridError
            If ``epoch`` falls outside the batch grid.
        """
        if self._pinned_batch is not None:
            return self._pinned_batch

        t = float(epoch)
        k = int(np.floor((t - self._t0) / self._batch_length))

        # Closing endpoint: the grid is otherwise half-open, so the last epoch
        # of a grid-length arc would index one batch past the end. The tolerance
        # covers rounding only — a few ULP at the magnitude of an ephemeris
        # second, so an epoch genuinely past the grid still raises.
        t_end = self._t0 + self._n_batches * self._batch_length
        endpoint_tol = max(1e-12, 10 * np.finfo(float).eps * max(1.0, abs(t_end)))
        if k == self._n_batches and abs(t - t_end) <= endpoint_tol:
            k = self._n_batches - 1

        if k < 0 or k >= self._n_batches:
            raise self.EpochOutsideGridError(
                f"Epoch {t} is outside the piecewise-polynomial batch grid "
                f"[{self._t0}, {t_end}] (batch index {k}, valid range "
                f"[0, {self._n_batches - 1}]). Widen the grid with "
                "ppoly_n_batches, or move its start with the force model's t0."
            )

        return k

    def get_active_batch_bounds(self, epoch: float) -> tuple[int, float, float]:
        """
        Return the active batch index and its time bounds.

        Parameters
        ----------
        epoch : float
            Current epoch in ephemeris seconds (TDB).

        Returns
        -------
        tuple[int, float, float]
            Tuple ``(batch_idx, batch_start, batch_end)``.
        """
        k = self._get_active_batch_index(epoch)
        t_start = self._t0 + k * self._batch_length
        t_end = t_start + self._batch_length
        return k, t_start, t_end

    def _interpolation_weights(self, epoch: float) -> tuple[int, float]:
        """
        Active batch and local time within it.

        Parameters
        ----------
        epoch : float
            Evaluation epoch in ephemeris seconds (TDB).

        Returns
        -------
        tuple[int, float]
            ``(k, tau)`` with ``tau = (epoch - batch_start) / batch_length``.
            ``tau`` is in ``[0, 1]`` up to rounding at a pinned arc's ends.
        """
        k = self._get_active_batch_index(epoch)
        tau = (float(epoch) - (self._t0 + k * self._batch_length)) / self._batch_length
        return k, tau

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    def _get_coefficients(self, current_state: dict, body: Body):
        """
        Resolve the polynomial coefficients for the given body.

        Priority:
        1. ``current_state[('a_ppoly', spice_id)]``
        2. the model's ``default_coefficients``
        3. ``None``, meaning the model contributes nothing

        Parameters
        ----------
        current_state : dict
            Dictionary containing the propagated state.
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray or None
            Coefficient vector of length :attr:`n_coefficients`.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return self._default_coefficients

        key = ("a_ppoly", spice_id)
        if key in current_state:
            val = current_state[key]
            values = np.atleast_1d(
                val.values if hasattr(val, "values") else val
            ).astype(float)
            return self._validate_coefficients(values.ravel())

        return self._default_coefficients

    def _frame_acceleration(self, current_state: dict, epoch: float, body: Body):
        """
        Interpolated acceleration in the frame the coefficients live in.

        Parameters
        ----------
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray or None
            Frame-local acceleration 3-vector in km/s^2, or ``None`` when
            neither the state nor the model configuration supplies
            coefficients.
        """
        p = self._get_coefficients(current_state, body)
        if p is None:
            return None

        nodes = p.reshape(-1, 3)
        if self._order == 0:
            return nodes[self._get_active_batch_index(epoch)].copy()

        k, tau = self._interpolation_weights(epoch)
        return (1.0 - tau) * nodes[k] + tau * nodes[k + 1]

    # ------------------------------------------------------------------
    # Acceleration and partials
    # ------------------------------------------------------------------
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Piecewise polynomial acceleration applied to the body.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector, used only to build the RTN triad.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            Acceleration 3-vector in km/s^2, in the propagation frame. Zero
            when neither the state nor the model configuration supplies
            coefficients.
        """
        a_frame = self._frame_acceleration(current_state, epoch, body)
        if a_frame is None:
            return np.zeros(3)
        return DynamicalUtils.from_frame(
            self, a_frame, position, current_state, epoch, body
        )

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to position.

        The frame-local acceleration is a function of time only, so only the
        frame it is expressed in can depend on position:

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf r}
            = \\sum_j a^{(f)}_j\\,
              \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf r},

        which is zero outside the RTN frame.

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        a_frame = self._frame_acceleration(current_state, epoch, body)
        if a_frame is None:
            return np.zeros((3, 3))
        return DynamicalUtils.frame_partials(
            self, a_frame, position, current_state, epoch, body
        )[0]

    def _compute_partial_by_velocity(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to velocity.

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf v}
            = \\sum_j a^{(f)}_j\\,
              \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf v},

        which is zero outside the RTN frame.

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        a_frame = self._frame_acceleration(current_state, epoch, body)
        if a_frame is None:
            return np.zeros((3, 3))
        return DynamicalUtils.frame_partials(
            self, a_frame, position, current_state, epoch, body
        )[1]

    def _compute_partial_by_coefficients(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the acceleration with respect to the stacked
        coefficients.

        The acceleration is linear in the coefficients, so the partial is the
        interpolation matrix: an identity block at the active batch for
        ``order=0``, and the two hat-function weights
        :math:`(1-\\tau)\\,\\mathbf I_3` and :math:`\\tau\\,\\mathbf I_3` at
        the batch's bounding nodes for ``order=1``, left-multiplied by the
        frame rotation :math:`\\mathbf T` (the identity in the inertial frame).
        It is exact at every epoch.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector, used only to build the RTN triad.
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current epoch in ephemeris seconds (TDB).
        body : Body
            Body the acceleration is evaluated for.

        Returns
        -------
        np.ndarray
            ``3 x n_coefficients`` dimensionless matrix, in the order documented
            by :meth:`coefficient_labels`.
        """
        T = DynamicalUtils.from_frame(
            self, np.eye(3), position, current_state, epoch, body
        )
        H = np.zeros((3, self.n_coefficients))
        if self._order == 0:
            k = self._get_active_batch_index(epoch)
            H[:, 3 * k : 3 * (k + 1)] = T
            return H

        k, tau = self._interpolation_weights(epoch)
        H[:, 3 * k : 3 * (k + 1)] = (1.0 - tau) * T
        H[:, 3 * (k + 1) : 3 * (k + 2)] = tau * T
        return H
