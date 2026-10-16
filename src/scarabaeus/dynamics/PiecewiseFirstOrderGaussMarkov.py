# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Body,
    EpochArray,
    StateArray,
    Spacecraft,
    Frame,
    DynamicModel,
    DynamicalUtils,
)
import scarabaeus.utils.NumpyWrapper as np


class PiecewiseFirstOrderGaussMarkov(DynamicModel):
    """ Piecewise first-order exponentially correlated Gauss-Markov acceleration model
    using a single stacked acceleration state.

    The propagation arc is divided into :math:`N_b` fixed-length time batches.
    Each batch :math:`k` carries an independent empirical acceleration
    :math:`\\mathbf{a}^{(k)}` that decays exponentially within the batch:

    .. math::

        \\dot{\\mathbf{a}}^{(k)} = -\\mathbf{B}^{(k)}\\,\\mathbf{a}^{(k)}

    Only the active batch :math:`k^*` at the current epoch contributes to the
    translational dynamics:
    :math:`\\ddot{\\mathbf{r}} = \\mathbf{a}_{\\mathrm{other}} +
    \\mathbf T\\,\\mathbf{a}^{(k^*)}`, where :math:`\\mathbf T` rotates the
    frame the components are expressed in (``frame``) into the propagation
    frame.
    The inverse correlation times :math:`\\beta_i^{(k)}` can be estimated as static
    parameters via ``("beta_pfogm", spice_id)`` state entries.

    The model partitions the propagation arc into fixed-length time batches and
    assigns one 3D empirical acceleration state to each batch. The full augmented
    acceleration state is stored as

    .. math::

        \\mathbf a_{pfogm} =
        \\begin{bmatrix}
        \\mathbf a^{(0)} \\\\
        \\mathbf a^{(1)} \\\\
        \\vdots \\\\
        \\mathbf a^{(N_b-1)}
        \\end{bmatrix}
        \\in \\mathbb{R}^{3N_b}

    where each batch state is

    .. math::

        \\mathbf a^{(k)} =
        \\begin{bmatrix}
        a_x^{(k)} \\\\
        a_y^{(k)} \\\\
        a_z^{(k)}
        \\end{bmatrix}

    The translational dynamics are

    .. math::

        \\dot{\\mathbf r} = \\mathbf v

    .. math::

        \\dot{\\mathbf v} = \\mathbf T\\,\\mathbf a^{(k^*)}

    where :math:`k^*` is the active batch index at the current epoch.

    Each batch follows first-order Gauss-Markov drift

    .. math::

        \\dot{\\mathbf a}^{(k)} = -\\mathbf B^{(k)} \\mathbf a^{(k)}

    with

    .. math::

        \\mathbf B^{(k)} =
        \\begin{bmatrix}
        \\beta_x^{(k)} & 0 & 0 \\\\
        0 & \\beta_y^{(k)} & 0 \\\\
        0 & 0 & \\beta_z^{(k)}
        \\end{bmatrix}

    The full stacked dynamics are therefore block diagonal.

    Parameters
    ----------
    origin : Body
        Propagation origin. ``a_pfogm`` and ``beta_pfogm`` are read from the
        propagated state at evaluation time.
    spacecraft : Spacecraft
        Spacecraft object associated with the propagated body.
    base_frame : Frame
        Reference frame of the propagator.
    t0 : EpochArray or float
        Reference epoch at which batch 0 starts, as an ``EpochArray`` or as raw
        ephemeris seconds (TDB). It is the origin of the batch grid, not the
        start of the current propagation: in a multi-leg
        :class:`~scarabaeus.MissionSequence` every leg must be given the same
        ``t0`` so the batches tile the whole arc and a batch index means the
        same thing in every leg.
    batch_length : float
        Duration of each time batch, in seconds.
    n_batches : int
        Number of piecewise time batches.
    default_beta : np.ndarray, optional
        Default 3-vector of inverse correlation coefficients
        ``[beta_x, beta_y, beta_z]`` used for every batch when ``beta_pfogm``
        is not provided in the state array.
    frame : str, optional
        Frame the components of every batch (and of their
        :math:`\\boldsymbol\\beta`) are expressed in: ``"inertial"`` (the
        propagation frame, the default), ``"RTN"`` (radial / along-track /
        cross-track, the RTN triad) or ``"body_fixed"``.
    body_fixed_frame : str or Frame, optional
        Frame the components are expressed in, for ``frame="body_fixed"``
        (for example ``'IAU_BENNU'``).
    relative_body : str or Body, optional
        Body the RTN triad is built about, when that is not the propagation
        origin. Defaults to the propagation origin.

    Raises
    ------
    ValueError
        If ``batch_length <= 0`` or ``n_batches < 1``.
    InvalidEpochError
        If ``t0`` is neither an ``EpochArray`` nor a real number.
    DynamicalUtils.InvalidFrameError
        If ``frame`` is not one of the supported frames.
    DynamicalUtils.MissingBodyFixedFrameError
        If ``frame="body_fixed"`` is requested without ``body_fixed_frame``.

    See Also
    --------
    scarabaeus.FirstOrderGaussMarkov : Single-arc variant of this model.
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.Propagator : Propagator that integrates the augmented state.
    scarabaeus.DynamicModel : Abstract base class for all force model components.

    Notes
    -----
    The state vector must contain:

    - ``("a_pfogm", 3 * n_batches, "estimated", "dynamic", owner, value)``

    Optionally, if the inverse correlation times are estimated:

    - ``("beta_pfogm", 3 * n_batches, "estimated", "static", owner, value)``

    If ``beta_pfogm`` is not provided, ``default_beta`` is repeated for all batches.

    Only the active batch contributes to :meth:`compute_acceleration`; all stacked
    batch states evolve according to their own drift, giving block-diagonal Jacobians
    and a simple selection matrix.

    Every batch lives in the chosen frame: in ``"RTN"`` its components are
    radial, along-track and cross-track, and the drift acts on those
    components. Only the applied acceleration is rotated, so the selection
    matrix becomes :math:`\\mathbf T` at the active batch. The RTN triad is
    built from the state, which adds the position and velocity partials
    :meth:`~scarabaeus.DynamicalUtils.frame_partials` returns; the inertial and
    body-fixed rotations depend on the epoch alone.
    """

    class InvalidEpochError(TypeError):
        """Raised when ``t0`` is neither an ``EpochArray`` nor a real number."""

    def __init__(
        self,
        origin: Body,
        spacecraft: Spacecraft,
        base_frame: Frame,
        t0: EpochArray | float,
        batch_length: float,
        n_batches: int,
        default_beta: np.ndarray | None = None,
        frame: str = "inertial",
        body_fixed_frame=None,
        relative_body=None,
    ):
        if batch_length <= 0.0:
            raise ValueError("batch_length must be > 0.")
        if n_batches < 1:
            raise ValueError("n_batches must be >= 1.")

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

        self._default_beta = (
            np.array(default_beta, dtype=float).reshape(3)
            if default_beta is not None
            else np.ones(3, dtype=float)
        )

        # Cached initial values keyed by body spice_id
        self._a0_map: dict[int, np.ndarray] = {}
        self._beta_map: dict[int, np.ndarray] = {}

        # Batch pinned for the integration arc in progress; see
        # :meth:`pin_batch_window`.
        self._pinned_batch: int | None = None


    @staticmethod
    def _epoch_to_et(t0: EpochArray | float) -> float:
        """
        Normalise ``t0`` to raw ephemeris seconds (TDB).

        Both forms are accepted because the two callers disagree: a script
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
        raise PiecewiseFirstOrderGaussMarkov.InvalidEpochError(
            "t0 must be an EpochArray or ephemeris seconds (TDB) as a float; "
            f"received {type(t0).__name__}."
        )

    @property
    def origin(self) -> Body:
        """Celestial body serving as origin of the state vector."""
        return self._origin

    @property
    def origin_name(self) -> str:
        """Name of the origin body."""
        return self._origin_name

    @property
    def ref_frame(self) -> str:
        """Reference frame used by the model."""
        return self._ref_frame

    @property
    def frame(self) -> str:
        """Frame the batch components are expressed in."""
        return self._frame

    @property
    def body_fixed_frame(self) -> str | None:
        """Body-fixed frame the components are expressed in, if any."""
        return self._body_fixed_frame

    @property
    def relative_body(self) -> str | None:
        """Body the RTN triad is referred to, if any."""
        return self._relative_body

    @property
    def depends_on_velocity(self) -> bool:
        """
        Whether the applied acceleration is a function of velocity.

        True for the RTN frame, whose triad is built from the velocity. The
        force model reads this to decide whether a
        :math:`\\partial \\mathbf a / \\partial \\mathbf v` block is needed.
        """
        return self._frame == "RTN"

    @property
    def t0(self) -> float:
        """Reference epoch of the first batch."""
        return self._t0

    @property
    def batch_length(self) -> float:
        """Duration of one batch."""
        return self._batch_length

    @property
    def n_batches(self) -> int:
        """Number of batches."""
        return self._n_batches

    @property
    def default_beta(self) -> np.ndarray:
        """Default inverse correlation coefficients used when 'beta_pfogm' is not provided."""
        return self._default_beta

    @default_beta.setter
    def default_beta(self, input_val):
        """Set the default inverse correlation coefficients."""
        self._default_beta = input_val

    def pin_batch_window(self, t_start: float, t_end: float) -> int | None:
        """
        Pin the batch that owns an integration arc, so the dynamics stay
        continuous across it.

        Batch membership is right-continuous, so the epoch closing an arc that
        ends on a batch boundary belongs to the *next* batch. An integrator
        evaluating its final stage there sees the acceleration and Jacobian jump
        at the very point it has to land on, shrinks its step to resolve a
        discontinuity that never moves, and gives up once the step falls below
        the spacing of the epoch — which at ephemeris-second magnitudes is large
        enough to hit in practice. Pinning the batch for the arc removes the
        jump from the arc's interior and endpoint alike.

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
        ValueError
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
            raise ValueError(
                f"Epoch {t} is outside the batch window "
                f"[{self._t0}, {t_end}]."
                f" Got batch index {k}, valid range [0, {self._n_batches - 1}]."
            )

        return k

    def _reshape_stacked(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape a stacked length-(3*n_batches) vector into shape (n_batches, 3).
        """
        x = np.atleast_1d(x).astype(float)
        expected_len = 3 * self._n_batches
        if x.size != expected_len:
            raise ValueError(
                f"Expected stacked vector of length {expected_len}, got {x.size}."
            )
        return x.reshape(self._n_batches, 3)

    def _flatten_stacked(self, x: np.ndarray) -> np.ndarray:
        """
        Flatten a (n_batches, 3) matrix into a length-(3*n_batches) vector.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self._n_batches, 3):
            raise ValueError(f"Expected shape ({self._n_batches}, 3), got {x.shape}.")
        return x.reshape(3 * self._n_batches)

    def _default_beta_stacked(self) -> np.ndarray:
        """
        Return the default stacked beta vector of length 3*n_batches.
        """
        return np.tile(self._default_beta, self._n_batches)

    def _get_a_pfogm(self, current_state: dict, body: Body) -> np.ndarray:
        """
        Resolve the current stacked GM acceleration state for the given body.

        Priority:
        1. current_state[('a_pfogm', spice_id)]
        2. cached StateArray value
        3. fallback zeros
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return np.zeros(3 * self._n_batches)

        key = ("a_pfogm", spice_id)
        if key in current_state:
            val = current_state[key]
            arr = np.atleast_1d(val.values if hasattr(val, "values") else val).astype(
                float
            )
            return self._reshape_stacked(arr).reshape(-1)

        if spice_id in self._a0_map:
            return self._a0_map[spice_id].copy()

        return np.zeros(3 * self._n_batches)

    def _get_beta_pfogm(self, current_state: dict, body: Body) -> np.ndarray:
        """
        Resolve the current stacked inverse correlation coefficients.

        Priority:
        1. current_state[('beta_pfogm', spice_id)]
        2. cached StateArray value
        3. fallback repeated default_beta
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return self._default_beta_stacked()

        key = ("beta_pfogm", spice_id)
        if key in current_state:
            val = current_state[key]
            arr = np.atleast_1d(val.values if hasattr(val, "values") else val).astype(
                float
            )
            return self._reshape_stacked(arr).reshape(-1)

        if spice_id in self._beta_map:
            return self._beta_map[spice_id].copy()

        return self._default_beta_stacked()

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Return the active batch Gauss-Markov acceleration applied to the body.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector, used by the RTN triad.

        current_state : dict
            Dictionary containing the propagated state.

        epoch : float
            Current propagation epoch.

        body : Body
            Body for which the acceleration is evaluated.

        Returns
        -------
        np.ndarray
            Active batch acceleration :math:`\\mathbf T\\,\\mathbf a^{(k^*)}`
            in the propagation frame, shape (3,).
        """
        return DynamicalUtils.from_frame(
            self,
            self._active_batch(current_state, epoch, body),
            position,
            current_state,
            epoch,
            body,
        )

    def _active_batch(self, current_state: dict, epoch: float, body: Body):
        """
        Frame-local acceleration of the batch active at ``epoch``.

        Parameters
        ----------
        current_state : dict
            Dictionary containing the propagated state.
        epoch : float
            Current propagation epoch.
        body : Body
            Body for which the acceleration is evaluated.

        Returns
        -------
        np.ndarray
            :math:`\\mathbf a^{(k^*)}`, shape (3,).
        """
        k = self._get_active_batch_index(epoch)
        a_mat = self._reshape_stacked(self._get_a_pfogm(current_state, body))
        return a_mat[k]

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of applied acceleration with respect to position.

        The empirical GM acceleration is an independent augmented state, so
        only the frame it is expressed in can depend on position:

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf r}
            = \\sum_j a^{(k^*)}_j\\,
              \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf r},

        which is zero outside the RTN frame.

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        return DynamicalUtils.frame_partials(
            self,
            self._active_batch(current_state, epoch, body),
            position,
            current_state,
            epoch,
            body,
        )[0]

    def _compute_partial_by_velocity(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of applied acceleration with respect to velocity.

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf v}
            = \\sum_j a^{(k^*)}_j\\,
              \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf v},

        which is zero outside the RTN frame.

        Returns
        -------
        np.ndarray
            3x3 matrix.
        """
        return DynamicalUtils.frame_partials(
            self,
            self._active_batch(current_state, epoch, body),
            position,
            current_state,
            epoch,
            body,
        )[1]

    def _compute_partial_by_acceleration_state(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the translational acceleration with respect to the
        full stacked piecewise GM state.

        If batch k is active, then

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf a_{pfogm}}
            =
            \\begin{bmatrix}
            \\mathbf 0 & \\cdots & \\mathbf T & \\cdots & \\mathbf 0
            \\end{bmatrix}

        where the frame rotation :math:`\\mathbf T` (the identity in the
        inertial frame) appears at the active batch.

        Returns
        -------
        np.ndarray
            3 x (3*n_batches) selection matrix.
        """
        k = self._get_active_batch_index(epoch)
        H = np.zeros((3, 3 * self._n_batches))
        H[:, 3 * k : 3 * (k + 1)] = DynamicalUtils.from_frame(
            self, np.eye(3), position, current_state, epoch, body
        )
        return H

    def _compute_acceleration_state_derivative(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Compute the deterministic drift of the full stacked GM acceleration state.

        For each batch k,

        .. math::

            \\dot{\\mathbf a}^{(k)} = -\\mathbf B^{(k)} \\mathbf a^{(k)}

        and the full stacked dynamics are block diagonal.

        Returns
        -------
        np.ndarray
            Stacked derivative vector of length 3*n_batches.
        """
        a_mat = self._reshape_stacked(self._get_a_pfogm(current_state, body))
        beta_mat = self._reshape_stacked(self._get_beta_pfogm(current_state, body))

        k = self._get_active_batch_index(epoch)

        for batch in range(self.n_batches):
            if batch == 0:
                if batch == k:
                    da_mat = np.array([-beta_mat[k] * a_mat[k]])
                else:
                    da_mat = np.array([0.0, 0.0, 0.0])
            else:
                if batch == k:
                    da_mat = np.vstack([da_mat, -beta_mat[k] * a_mat[k]])
                else:
                    da_mat = np.vstack([da_mat, np.array([0.0, 0.0, 0.0])])

        return da_mat

    def _compute_partial_of_acceleration_state_by_acceleration_state(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Jacobian of the stacked GM drift with respect to the stacked GM state.

        Since each batch evolves independently,

        .. math::

            \\frac{\\partial \\dot{\\mathbf a}_{pfogm}}
            {\\partial \\mathbf a_{pfogm}}
            =
            \\mathrm{blockdiag}(-\\mathbf B^{(0)}, -\\mathbf B^{(1)}, \\dots)

        Returns
        -------
        np.ndarray
            (3*n_batches) x (3*n_batches) block-diagonal Jacobian matrix.
        """
        beta_flat = self._get_beta_pfogm(current_state, body)
        J = np.zeros((3 * self._n_batches, 3 * self._n_batches))

        k = self._get_active_batch_index(epoch)
        beta_k = beta_flat[3 * k : 3 * (k + 1)]
        J[3 * k : 3 * (k + 1), 3 * k : 3 * (k + 1)] = -np.diag(beta_k)

        return J

    def _compute_partial_of_acceleration_state_by_beta(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Jacobian of the stacked GM drift with respect to the stacked inverse
        correlation coefficients.

        For each batch k,

        .. math::

            \\frac{\\partial \\dot{\\mathbf a}^{(k)}}
            {\\partial \\boldsymbol\\beta^{(k)}}
            =
            -\\mathrm{diag}(\\mathbf a^{(k)})

        Therefore the full Jacobian is block diagonal.

        Returns
        -------
        np.ndarray
            (3*n_batches) x (3*n_batches) block-diagonal Jacobian matrix.
        """
        a_flat = self._get_a_pfogm(current_state, body)
        J = np.zeros((3 * self._n_batches, 3 * self._n_batches))

        k = self._get_active_batch_index(epoch)
        a_k = a_flat[3 * k : 3 * (k + 1)]
        J[3 * k : 3 * (k + 1), 3 * k : 3 * (k + 1)] = -np.diag(a_k)

        return J

    def get_active_batch_bounds(self, epoch: float) -> tuple[int, float, float]:
        """
        Return the active batch index and its time bounds.

        Parameters
        ----------
        epoch : float
            Current epoch.

        Returns
        -------
        tuple[int, float, float]
            Tuple ``(batch_idx, batch_start, batch_end)``.
        """
        k = self._get_active_batch_index(epoch)
        t_start = self._t0 + k * self._batch_length
        t_end = t_start + self._batch_length
        return k, t_start, t_end

    def get_all_batch_accelerations(
        self,
        current_state: dict,
        body: Body,
    ) -> np.ndarray:
        """
        Return all batch acceleration states reshaped as (n_batches, 3).

        Returns
        -------
        np.ndarray
            Matrix whose k-th row is the acceleration vector of batch k.
        """
        return self._reshape_stacked(self._get_a_pfogm(current_state, body))
