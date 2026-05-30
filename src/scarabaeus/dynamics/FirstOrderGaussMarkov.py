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


class FirstOrderGaussMarkov(DynamicModel):
    """ Models a first-order exponentially correlated Gauss-Markov acceleration process.

    An empirical acceleration state :math:`\\mathbf{a}_{gm}` is added to the
    equations of motion and evolves according to exponential decay:

    .. math::

        \\dot{\\mathbf{a}}_{gm} = -\\mathbf{B}\\,\\mathbf{a}_{gm}

    where :math:`\\mathbf{B} = \\mathrm{diag}(\\beta_x, \\beta_y, \\beta_z)`
    with :math:`\\beta_i = 1/\\tau_i` being the inverse correlation times.
    The spacecraft translational dynamics become
    :math:`\\ddot{\\mathbf{r}} = \\mathbf{a}_{\\mathrm{other}} + \\mathbf{a}_{gm}`.
    The :math:`\\beta_i` can be estimated as static parameters via
    ``("beta_fogm", spice_id)`` state entries.

    In deterministic propagation only the drift term is used:

    .. math::

        \\dot{\\mathbf a}_{gm} = -\\mathbf B \\, \\mathbf a_{gm}

    The continuous-time process noise intensity consistent with the stationary
    first-order Gauss-Markov model is:

    .. math::

        q_i = 2 \\, \\beta_i \\, \\sigma_i^2

    where :math:`\\sigma_i^2` is the stationary variance of the corresponding
    acceleration component.

    Parameters
    ----------
    state_array : StateArray
        State definition containing the ``a_fogm`` Gauss-Markov acceleration
        state and, optionally, the ``beta_fogm`` inverse correlation
        coefficients.
    spacecraft : Spacecraft
        Spacecraft object associated with the propagated body.
    base_frame : Frame
        Reference frame of the propagator.
    default_beta : np.ndarray, optional
        Default inverse correlation coefficients
        :math:`[\\beta_x, \\beta_y, \\beta_z]` in 1/TU.  Used when
        ``beta_fogm`` is not found in the state array.  Defaults to
        isotropic unit decay ``[1, 1, 1]``.

    Notes
    -----
    The state vector must contain entries of the form:

    - ``('a_fogm', 3, 'estimated', 'dynamic', owner, value)``

    Optionally, if the inverse correlation times are themselves estimated:

    - ``('beta_fogm', 3, 'estimated', 'static', owner, value)``

    The model is linear in :math:`\\mathbf a_{gm}` for fixed :math:`\\mathbf B`.
    If process noise is needed in a filter, this class provides the deterministic
    drift and Jacobians; the user should inject the corresponding process noise
    covariance separately.

    See Also
    --------
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.Propagator : Propagator that integrates the augmented state.
    scarabaeus.DynamicModel : Abstract base class for all force model components.
    scarabaeus.PiecewiseFirstOrderGaussMarkov : Batch-segmented variant of this model.

    References
    ----------
    Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control.
    Brown, R. G., & Hwang, P. Y. C. (2012). Introduction to Random Signals and Applied Kalman Filtering.

    Examples
    --------
    .. code-block:: python

        import scarabaeus as scb

        state = [
            ("position", 3, "estimated", "dynamic", sc, [7000, 0, 0]),
            ("velocity", 3, "estimated", "dynamic", sc, [0, 7.5, 0]),
            ("a_fogm", 3, "estimated", "dynamic", sc, [1e-9, 0.0, 0.0]),
            ("beta_fogm", 3, "estimated", "static", sc, [1/3600, 1/3600, 1/3600]),
        ]

        state_vector = scb.StateArray(
            epoch=epochs[0],
            frame="J2000",
            origin=scb.CelestialBody("Earth"),
            state=state,
        )

        gm_model = FirstOrderGaussMarkov(
            state_array=state_vector,
            spacecraft=sc,
            base_frame="J2000",
        )
    """

    def __init__(
        self,
        state_array: StateArray,
        spacecraft: Spacecraft,
        base_frame: Frame,
        default_beta: np.ndarray | None = None,
    ):
        self._origin = state_array.origin
        self._origin_name = self._origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame

        self._beta_map: dict[int, np.ndarray] = {}
        self._a0_map: dict[int, np.ndarray] = {}

        self._default_beta = (
            np.array(default_beta, dtype=float)
            if default_beta is not None
            else np.ones(3, dtype=float)
        )

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
    def default_beta(self) -> np.ndarray:
        """Default inverse correlation coefficients used when not specified in state."""
        return self._default_beta

    @default_beta.setter
    def default_beta(self, input_val):
        """Set the default inverse correlation coefficients."""
        self._default_beta = input_val

    def _initialize_maps(self, state_array: StateArray) -> None:
        """
        Scan the StateArray and cache initial GM acceleration and beta coefficients
        per body, identified by spice_id.

        Recognized names:
        - a_fogm
        - beta_fogm
        """
        tmp_a = {}
        tmp_b = {}

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

            if name == "a_fogm":
                tmp_a[spice_id] = vals[:3]
            elif name == "beta_fogm":
                tmp_b[spice_id] = vals[:3]

        self._a0_map = tmp_a
        self._beta_map = tmp_b

    def _get_a_fogm(self, current_state: dict, body: Body) -> np.ndarray:
        """
        Resolve the current Gauss-Markov acceleration state for the given body.

        Priority:
        1. current_state[('a_fogm', spice_id)]
        2. cached StateArray value
        3. fallback zeros
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return np.zeros(3)

        key = ("a_fogm", spice_id)
        if key in current_state:
            val = current_state[key]
            return np.atleast_1d(val.values if hasattr(val, "values") else val).astype(
                float
            )[:3]

        if spice_id in self._a0_map:
            return self._a0_map[spice_id]

        return np.zeros(3)

    def _get_beta_fogm(self, current_state: dict, body: Body) -> np.ndarray:
        """
        Resolve the inverse correlation coefficients for the given body.

        Priority:
        1. current_state[('beta_fogm', spice_id)]
        2. cached StateArray value
        3. fallback default_beta
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is None:
            return self._default_beta

        key = ("beta_fogm", spice_id)
        if key in current_state:
            val = current_state[key]
            return np.atleast_1d(val.values if hasattr(val, "values") else val).astype(
                float
            )[:3]

        return self._beta_map.get(spice_id, self._default_beta)

    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Returns the current Gauss-Markov acceleration applied to the body.

        Parameters
        ----------
        position : np.ndarray
            Spacecraft position vector, unused in this model but kept for API consistency.

        current_state : dict
            Dictionary containing the propagated state.

        epoch : float
            Current epoch, unused in this model but kept for API consistency.

        body : Body
            Body for which the acceleration is evaluated.

        Returns
        -------
        np.ndarray
            Current acceleration vector :math:`\\mathbf a_{gm}` in km/s^2.
        """
        return self._get_a_fogm(current_state, body)

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the applied acceleration with respect to position.

        Since the empirical GM acceleration is an independent augmented state,
        it does not directly depend on position.

        Returns
        -------
        np.ndarray
            Zero 3x3 matrix:

            .. math::

                \\frac{\\partial \\mathbf a_{gm}}{\\partial \\mathbf r} = \\mathbf 0
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
        Partial derivative of the applied acceleration with respect to velocity.

        Returns
        -------
        np.ndarray
            Zero 3x3 matrix:

            .. math::

                \\frac{\\partial \\mathbf a_{gm}}{\\partial \\mathbf v} = \\mathbf 0
        """
        return np.zeros((3, 3))

    def _compute_partial_by_acceleration_state(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the applied acceleration with respect to the
        Gauss-Markov acceleration state itself.

        Since the translational acceleration is exactly the GM state,

        .. math::

            \\mathbf a = \\mathbf a_{gm}

        therefore

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf a_{gm}} = \\mathbf I

        Returns
        -------
        np.ndarray
            3x3 identity matrix.
        """
        return np.eye(3)

    def _compute_acceleration_state_derivative(
        self,
        position,
        current_state: dict,
        epoch: float,
        body: Body,
    ) -> np.ndarray:
        """
        Computes the deterministic drift of the Gauss-Markov acceleration state.

        The first-order exponentially correlated model is

        .. math::

            \\dot{\\mathbf a}_{gm} = -\\mathbf B \\, \\mathbf a_{gm}

        Returns
        -------
        np.ndarray
            Time derivative of the GM acceleration state.
        """
        a_fogm = self._get_a_fogm(current_state, body)
        beta = self._get_beta_fogm(current_state, body)
        return -beta * a_fogm

    def _compute_partial_of_acceleration_state_by_acceleration_state(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Jacobian of the GM-state drift with respect to the GM acceleration state.

        For

        .. math::

            \\dot{\\mathbf a}_{gm} = -\\mathbf B \\, \\mathbf a_{gm}

        we have

        .. math::

            \\frac{\\partial \\dot{\\mathbf a}_{gm}}{\\partial \\mathbf a_{gm}}
            = -\\mathbf B

        Returns
        -------
        np.ndarray
            3x3 diagonal matrix ``diag(-beta_x, -beta_y, -beta_z)``.
        """
        beta = self._get_beta_fogm(current_state, body)
        return -np.diag(beta)

    def _compute_partial_of_acceleration_state_by_beta(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Jacobian of the GM-state drift with respect to the inverse correlation coefficients.

        For component-wise dynamics

        .. math::

            \\dot a_x = -\\beta_x a_x, \\,\\,
            \\dot a_y = -\\beta_y a_y, \\,\\,
            \\dot a_z = -\\beta_z a_z

        the Jacobian with respect to
        :math:`\\boldsymbol\\beta = [\\beta_x, \\beta_y, \\beta_z]^T` is

        .. math::

            \\frac{\\partial \\dot{\\mathbf a}_{gm}}{\\partial \\boldsymbol\\beta}
            =
            \\begin{bmatrix}
            -a_x & 0 & 0 \\\\
            0 & -a_y & 0 \\\\
            0 & 0 & -a_z
            \\end{bmatrix}

        Returns
        -------
        np.ndarray
            3x3 diagonal matrix with entries ``-a_fogm``.
        """
        a_fogm = self._get_a_fogm(current_state, body)
        return -np.diag(a_fogm)
