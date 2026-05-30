# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Body,
    EpochArray,
    StateArray,
    Spacecraft,
    Frame,
    DynamicModel,
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
    :math:`\\ddot{\\mathbf{r}} = \\mathbf{a}_{\\mathrm{other}} + \\mathbf{a}^{(k^*)}`.
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

        \\dot{\\mathbf v} = \\mathbf a^{(k^*)}

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
    state_array : StateArray
        State definition containing ``a_pfogm`` and optionally ``beta_pfogm``.
    spacecraft : Spacecraft
        Spacecraft object associated with the propagated body.
    base_frame : Frame
        Reference frame of the propagator.
    t0 : float
        Reference epoch at which batch 0 starts.
    batch_length : float
        Duration of each time batch (same units as ``t0``).
    n_batches : int
        Number of piecewise time batches.
    default_beta : np.ndarray, optional
        Default 3-vector of inverse correlation coefficients
        ``[beta_x, beta_y, beta_z]`` used for every batch when ``beta_pfogm``
        is not provided in the state array.

    Raises
    ------
    ValueError
        If ``batch_length <= 0`` or ``n_batches < 1``.

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

    See Also
    --------
    scarabaeus.FirstOrderGaussMarkov : Single-arc variant of this model.
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.Propagator : Propagator that integrates the augmented state.
    scarabaeus.DynamicModel : Abstract base class for all force model components.
    """

    def __init__(
        self,
        state_array: StateArray,
        spacecraft: Spacecraft,
        base_frame: Frame,
        t0: float,
        batch_length: float,
        n_batches: int,
        default_beta: np.ndarray | None = None,
    ):
        if batch_length <= 0.0:
            raise ValueError("batch_length must be > 0.")
        if n_batches < 1:
            raise ValueError("n_batches must be >= 1.")

        self._origin = state_array.origin
        self._origin_name = self._origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame

        self._t0 = float(t0.times.values)
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

        self._initialize_maps(state_array)

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

    def _initialize_maps(self, state_array: StateArray) -> None:
        """
        Scan the StateArray and cache the initial stacked acceleration and beta
        vectors for each body.

        Recognized names:
        - a_pfogm
        - beta_pfogm
        """
        expected_len = 3 * self._n_batches

        for entry in state_array.state:
            if len(entry) < 6:
                continue

            name, _, _, _, owner, raw_val = entry
            spice_id = getattr(owner, "spice_id", None)
            if spice_id is None:
                continue

            if hasattr(raw_val, "quantity"):
                vals = raw_val.quantity.values
            elif hasattr(raw_val, "values"):
                vals = raw_val.values
            else:
                vals = raw_val

            vals = np.atleast_1d(vals).astype(float)

            if name == "a_pfogm":
                if vals.size != expected_len:
                    raise ValueError(
                        f"'a_pfogm' must have length {expected_len}, got {vals.size}."
                    )
                self._a0_map[spice_id] = vals.copy()

            elif name == "beta_pfogm":
                if vals.size != expected_len:
                    raise ValueError(
                        f"'beta_pfogm' must have length {expected_len}, got {vals.size}."
                    )
                self._beta_map[spice_id] = vals.copy()

    def _get_active_batch_index(self, epoch: float) -> int | None:
        """
        Return the active batch index, or None if epoch is outside all batch windows.
        """
        t = float(epoch)
        k = int(np.floor((t - self._t0) / self._batch_length))

        if k < 0 or k >= self._n_batches:
            raise ValueError(
                f"Epoch {t} is outside the batch window "
                f"[{self._t0}, {self._t0 + self._n_batches * self._batch_length})."
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
            Spacecraft position vector, unused here but retained for API consistency.

        current_state : dict
            Dictionary containing the propagated state.

        epoch : float
            Current propagation epoch.

        body : Body
            Body for which the acceleration is evaluated.

        Returns
        -------
        np.ndarray
            Active batch acceleration vector of shape (3,).
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

        Since the empirical GM acceleration is an independent augmented state,

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf r} = \\mathbf 0

        Returns
        -------
        np.ndarray
            3x3 zero matrix.
        """
        return np.zeros((3, 3))

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

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf v} = \\mathbf 0

        Returns
        -------
        np.ndarray
            3x3 zero matrix.
        """
        return np.zeros((3, 3))

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
            \\mathbf 0 & \\cdots & \\mathbf I_3 & \\cdots & \\mathbf 0
            \\end{bmatrix}

        where the identity block appears at the active batch.

        Returns
        -------
        np.ndarray
            3 x (3*n_batches) selection matrix.
        """
        k = self._get_active_batch_index(epoch)
        H = np.zeros((3, 3 * self._n_batches))
        H[:, 3 * k : 3 * (k + 1)] = np.eye(3)
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
