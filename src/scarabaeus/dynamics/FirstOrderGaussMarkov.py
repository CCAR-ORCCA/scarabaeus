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


class FirstOrderGaussMarkov(DynamicModel):
    """ Models a first-order exponentially correlated Gauss-Markov acceleration process.

    An empirical acceleration state :math:`\\mathbf{a}_{gm}` is added to the
    equations of motion and evolves according to exponential decay:

    .. math::

        \\dot{\\mathbf{a}}_{gm} = -\\mathbf{B}\\,\\mathbf{a}_{gm}

    where :math:`\\mathbf{B} = \\mathrm{diag}(\\beta_x, \\beta_y, \\beta_z)`
    with :math:`\\beta_i = 1/\\tau_i` being the inverse correlation times.
    The spacecraft translational dynamics become
    :math:`\\ddot{\\mathbf{r}} = \\mathbf{a}_{\\mathrm{other}} +
    \\mathbf T\\,\\mathbf{a}_{gm}`, where :math:`\\mathbf T` rotates the
    frame the components are expressed in (``frame``) into the propagation
    frame.
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
    origin : Body
        Propagation origin. ``a_fogm`` and ``beta_fogm`` are read from the
        propagated state at evaluation time.
    spacecraft : Spacecraft
        Spacecraft object associated with the propagated body.
    base_frame : Frame
        Reference frame of the propagator.
    default_beta : np.ndarray, optional
        Default inverse correlation coefficients
        :math:`[\\beta_x, \\beta_y, \\beta_z]` in 1/TU.  Used when
        ``beta_fogm`` is not found in the state array.  Defaults to
        isotropic unit decay ``[1, 1, 1]``.
    frame : str, optional
        Frame the components of :math:`\\mathbf a_{gm}` (and of
        :math:`\\boldsymbol\\beta`) are expressed in: ``"inertial"`` (the
        propagation frame, the default), ``"RTN"`` (radial / along-track /
        cross-track, the RTN triad) or ``"body_fixed"``.
    body_fixed_frame : str or Frame, optional
        Frame the components are expressed in, for ``frame="body_fixed"``
        (for example ``'IAU_BENNU'``).
    relative_body : str or Body, optional
        Body the RTN triad is built about, when that is not the propagation
        origin. Defaults to the propagation origin.

    See Also
    --------
    scarabaeus.StateArray : State vector used to initialize the model.
    scarabaeus.Propagator : Propagator that integrates the augmented state.
    scarabaeus.DynamicModel : Abstract base class for all force model components.
    scarabaeus.PiecewiseFirstOrderGaussMarkov : Batch-segmented variant of this model.

    Notes
    -----
    The state vector must contain entries of the form:

    - ``('a_fogm', 3, 'estimated', 'dynamic', owner, value)``

    Optionally, if the inverse correlation times are themselves estimated:

    - ``('beta_fogm', 3, 'estimated', 'static', owner, value)``

    The Gauss-Markov state lives in the chosen frame: in ``"RTN"`` its three
    components are radial, along-track and cross-track, each with its own
    correlation time, and the drift :math:`-\\mathbf B\\,\\mathbf a_{gm}`
    acts on those components. Only the applied acceleration
    :math:`\\mathbf T\\,\\mathbf a_{gm}` is rotated, so
    :math:`\\partial \\mathbf a / \\partial \\mathbf a_{gm} = \\mathbf T`.
    The RTN triad is built from the state, which adds the position and
    velocity partials :meth:`~scarabaeus.DynamicalUtils.frame_partials` returns;
    the inertial and body-fixed rotations depend on the epoch alone.

    The model is linear in :math:`\\mathbf a_{gm}` for fixed :math:`\\mathbf B`.
    If process noise is needed in a filter, this class provides the deterministic
    drift and Jacobians; the user should inject the corresponding process noise
    covariance separately.

    References
    ----------
    .. [1] Maybeck, P. S. (1979). Stochastic Models, Estimation, and Control.
    .. [2] Brown, R. G., & Hwang, P. Y. C. (2012). Introduction to Random Signals and Applied Kalman Filtering.

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
            origin=state_vector.origin,
            spacecraft=sc,
            base_frame="J2000",
        )
    """

    def __init__(
        self,
        origin: Body,
        spacecraft: Spacecraft,
        base_frame: Frame,
        default_beta: np.ndarray | None = None,
        frame: str = "inertial",
        body_fixed_frame=None,
        relative_body=None,
    ):
        self._origin = origin
        self._origin_name = self._origin.name
        self._spacecraft = spacecraft
        self._ref_frame = base_frame
        self._frame, self._body_fixed_frame, self._relative_body = (
            DynamicalUtils.resolve_frame(frame, body_fixed_frame, relative_body)
        )
        # Per-epoch SPICE lookups of the frame, kept by DynamicalUtils.
        self._frame_memo = {}

        self._beta_map: dict[int, np.ndarray] = {}
        self._a0_map: dict[int, np.ndarray] = {}

        self._default_beta = (
            np.array(default_beta, dtype=float)
            if default_beta is not None
            else np.ones(3, dtype=float)
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
        """Frame the Gauss-Markov components are expressed in."""
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
    def default_beta(self) -> np.ndarray:
        """Default inverse correlation coefficients used when not specified in state."""
        return self._default_beta

    @default_beta.setter
    def default_beta(self, input_val):
        """Set the default inverse correlation coefficients."""
        self._default_beta = input_val

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
            Spacecraft position vector, used by the RTN triad.

        current_state : dict
            Dictionary containing the propagated state.

        epoch : float
            Current epoch, used by the body-fixed rotation.

        body : Body
            Body for which the acceleration is evaluated.

        Returns
        -------
        np.ndarray
            Current acceleration :math:`\\mathbf T\\,\\mathbf a_{gm}` in the
            propagation frame, km/s^2.
        """
        a_gm = self._get_a_fogm(current_state, body)
        return DynamicalUtils.from_frame(
            self, a_gm, position, current_state, epoch, body
        )

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: EpochArray,
        body: Body,
    ) -> np.ndarray:
        """
        Partial derivative of the applied acceleration with respect to position.

        The empirical GM acceleration is an independent augmented state, so
        only the frame it is expressed in can depend on position: the RTN
        triad does, the inertial and body-fixed frames do not.

        Returns
        -------
        np.ndarray
            3x3 matrix,

            .. math::

                \\frac{\\partial (\\mathbf T\\,\\mathbf a_{gm})}
                     {\\partial \\mathbf r}
                = \\sum_j a_{gm,j}\\,
                  \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf r},

            which is zero outside the RTN frame.
        """
        a_gm = self._get_a_fogm(current_state, body)
        return DynamicalUtils.frame_partials(
            self, a_gm, position, current_state, epoch, body
        )[0]

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
            3x3 matrix,

            .. math::

                \\frac{\\partial (\\mathbf T\\,\\mathbf a_{gm})}
                     {\\partial \\mathbf v}
                = \\sum_j a_{gm,j}\\,
                  \\frac{\\partial \\hat{\\mathbf e}_j}{\\partial \\mathbf v},

            which is zero outside the RTN frame.
        """
        a_gm = self._get_a_fogm(current_state, body)
        return DynamicalUtils.frame_partials(
            self, a_gm, position, current_state, epoch, body
        )[1]

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

        Since the translational acceleration is the GM state rotated out of its
        frame,

        .. math::

            \\mathbf a = \\mathbf T\\,\\mathbf a_{gm}

        therefore

        .. math::

            \\frac{\\partial \\mathbf a}{\\partial \\mathbf a_{gm}} = \\mathbf T

        Returns
        -------
        np.ndarray
            3x3 rotation, the identity in the inertial frame.
        """
        return DynamicalUtils.from_frame(
            self, np.eye(3), position, current_state, epoch, body
        )

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
