# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import math
import numpy as np
from typing import Optional
from scarabaeus import CovarianceMatrix, ProcessNoise


# --------------------#
#  Class Definition  #
# --------------------#
class DynamicalModelCompensation(ProcessNoise):
    """ Dynamic Model Compensation (DMC) process noise model.

    An alternative to SNC that models process noise through direct augmentation
    of the state vector with stochastic parameters representing unmodeled
    accelerations or forces, allowing their estimation alongside the spacecraft state.

    **Augmented state and dynamics**

    DMC augments the state vector with a stochastic acceleration vector
    :math:`\\mathbf{w} \\in \\mathbb{R}^3`:

    .. math::

        \\mathbf{x}^T = [\\,\\mathbf{r}^T \\quad \\mathbf{v}^T \\quad \\mathbf{w}^T\\,]

    The stochastic acceleration obeys a first-order Gauss-Markov (exponentially
    correlated) process:

    .. math::

        \\dot{\\mathbf{w}} = -\\mathbf{B}\\,\\mathbf{w} + \\mathbf{u}(t)

    where :math:`\\mathbf{B} = \\mathrm{diag}(\\beta_x, \\beta_y, \\beta_z)` contains
    the inverse correlation times (decay rates), and :math:`\\mathbf{u}(t)` is a
    zero-mean input held constant over each step, with covariance
    :math:`\\mathbf{Q}` (3, 3), in km²/s⁶.

    **Discretization (gamma matrix)**

    The response of the state to a constant input :math:`\\mathbf{u}` over a
    step :math:`\\Delta t` is
    :math:`\\mathbf{\\Gamma}_k = \\int_{t_k}^{t_{k+1}}
    \\mathbf{\\Phi}(t_{k+1}, \\tau)\\,\\mathbf{B}_u\\,d\\tau` (9, 3),
    with :math:`\\mathbf{B}_u = [\\mathbf{0}\\;\\mathbf{0}\\;\\mathbf{I}_3]^T`:

    .. math::

        \\mathbf{\\Gamma}_r = \\frac{\\Delta t^2}{2}\\,\\mathbf{B}^{-1}
                          - \\Delta t\\,\\mathbf{B}^{-2}
                          + \\mathbf{B}^{-3}\\Bigl(\\mathbf{I} - e^{-\\mathbf{B}\\Delta t}\\Bigr)

    .. math::

        \\mathbf{\\Gamma}_v = \\Delta t\\,\\mathbf{B}^{-1}
                          - \\mathbf{B}^{-2}\\Bigl(\\mathbf{I} - e^{-\\mathbf{B}\\Delta t}\\Bigr)

    .. math::

        \\mathbf{\\Gamma}_w = \\mathbf{B}^{-1}\\Bigl(\\mathbf{I} - e^{-\\mathbf{B}\\Delta t}\\Bigr)

    All matrices are diagonal because :math:`\\mathbf{B}` is diagonal, so every
    block is computed element-wise. The full discrete-time process noise covariance is:

    .. math::

        \\mathbf{Q}_k = \\mathbf{\\Gamma}_k\\,\\mathbf{Q}\\,\\mathbf{\\Gamma}_k^T

    This treats :math:`\\mathbf{u}` as piecewise constant, as SNC does with the
    acceleration. Integrating white noise over the step instead gives a full
    rank :math:`\\mathbf{Q}_k` that grows more slowly with :math:`\\Delta t`;
    the form above has rank three, which the SRIF needs to factor it through
    :math:`\\mathbf{\\Gamma}_k`.

    **Decay rates**

    The filters discretize the noise with the decay rates the dynamics
    propagate the state with: the ``beta_fogm`` state of ``body`` when there
    is one, else the ``fogm_beta`` of the
    :class:`~scarabaeus.ForceModelTranslation` of that body. They are
    evaluated on each interval and again on every iteration, so an
    estimated decay rate is followed. ``beta`` is used only when no
    reference decay rates are given.

    **Covariance propagation**

    The propagated covariance is augmented identically to SNC:

    .. math::

        \\bar{\\mathbf{P}}_{k+1} =
            \\mathbf{\\Phi}_{k+1,k}\\,\\mathbf{P}_k\\,\\mathbf{\\Phi}_{k+1,k}^T
            + \\mathbf{Q}_k

    where :math:`\\mathbf{\\Phi}_{k+1,k}` is now the STM of the augmented
    :math:`[\\mathbf{r}, \\mathbf{v}, \\mathbf{w}]` system, including the exponential
    decay :math:`e^{-\\mathbf{B}\\Delta t}` in the :math:`\\mathbf{w}` partition.

    **Comparison with SNC**

    Unlike SNC, which treats the unmodeled acceleration as a purely random impulse
    reset at each epoch, DMC retains memory of the stochastic acceleration across
    epochs via the exponential decay :math:`e^{-\\mathbf{B}\\Delta t}`. As
    :math:`\\beta \\to \\infty` (very short correlation time) DMC approaches SNC;
    as :math:`\\beta \\to 0` (long correlation time) the stochastic acceleration
    becomes nearly constant and is effectively solved for as a deterministic
    parameter.

    Parameters
    ----------
    continuous_time_covariance : CovarianceMatrix
        Covariance :math:`\\mathbf{Q}` of the input driving the stochastic
        acceleration, held constant over each interval, in km²/s⁶.
    state_definition : optional
        State vector definition. Defaults to None.
    sequence_definition : optional
        Sequence definition for multi-leg trajectories. Defaults to None.
    beta : np.ndarray, optional
        Decay rates (inverse correlation times, 1/s) of the Gauss-Markov
        process, a scalar (isotropic) or a 3-element vector for x, y, z
        components. Used only when
        :meth:`_compute_discrete_time_covariances` is given no reference
        decay rates, which the filters always give. Defaults to ``None``.
    max_interval : ArrayWUnits or float, optional
        Longest interval the noise is accumulated over, in seconds when a
        float; a longer one, such as a data gap, receives the noise of one
        this long (see :meth:`~scarabaeus.ProcessNoise._noise_intervals`).
        Defaults to ``None``, no limit.
    body : Body or int, optional
        Body whose position, velocity and ``a_fogm`` the noise drives, or its
        SPICE ID. Defaults to ``None``, the owner of the first position in
        the state.

    Raises
    ------
    DynamicalModelCompensation.BetaSizeError
        If ``beta`` is neither a scalar nor a 3-element vector.
    DynamicalModelCompensation.NonPositiveBetaError
        If a component of ``beta`` is not strictly positive.

    See Also
    --------
    scarabaeus.StateNoiseCompensation : Simpler piecewise-constant SNC variant.
    scarabaeus.ProcessNoise : Abstract base class.
    scarabaeus.ProcessNoiseSettings : Configuration object consumed by the filter.

    Notes
    -----
    As :math:`\\beta \\to \\infty` (very short correlation time) DMC approaches SNC;
    as :math:`\\beta \\to 0` (long correlation time) the stochastic acceleration
    becomes nearly constant and is effectively estimated as a deterministic parameter.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class MissingBetaError(ValueError):
        """Raised when the noise is discretized with neither reference decay
        rates nor a ``beta`` given to the constructor."""

    class BetaSizeError(ValueError):
        """Raised when ``beta`` is neither a scalar nor a 3-element vector."""

    class NonPositiveBetaError(ValueError):
        """Raised when a component of ``beta`` is not strictly positive."""

    # endregion Errors #
    # ------------------#

    # The Gauss-Markov acceleration is driven along with the position and
    # velocity
    _noise_states = ("position", "velocity", "a_fogm")

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        continuous_time_covariance: CovarianceMatrix,
        state_definition=None,
        sequence_definition=None,
        beta: Optional[np.ndarray] = None,
        max_interval=None,
        body=None,
    ):
        super().__init__(
            continuous_time_covariance,
            state_definition,
            sequence_definition,
            max_interval,
            body,
        )

        # Validate beta if provided; the filters take it from the dynamics
        self.beta = None
        if beta is not None:
            beta = np.asarray(beta, dtype=float).reshape(-1)
            if beta.size == 1:
                # isotropic case
                beta = np.repeat(beta, 3)
            elif beta.size != 3:
                raise self.BetaSizeError(
                    "Beta must be either a scalar or a 3-element vector."
                )
            if not np.all(beta > 0):
                raise self.NonPositiveBetaError(
                    "All beta components must be strictly positive."
                )
            self.beta = beta

    # ----------------#
    # region Methods #
    # ----------------#
    def _compute_discrete_time_covariances(
        self,
        epochs: list,
        reference_state=None,
        reference_beta=None,
        reference_frame=None,
    ) -> None:
        """
        Compute discrete-time process noise covariances using DMC model.

        For DMC, the process noise applies to augmented stochastic parameters
        following a first-order Gauss-Markov process with time-constant matrix B.
        Each interval is cut to ``max_interval``, as in
        :meth:`~scarabaeus.ProcessNoise._noise_intervals`; the
        :math:`\\mathbf{w}` block levels off by itself, at
        :math:`Q_{ii}/\\beta_i^2`, but the position and velocity blocks do not.
        Covariances and gamma matrices are stored for the
        ``[position, velocity, a_fogm]`` block of :attr:`body` only;
        :meth:`gamma_at` and :meth:`covariance_at` place them in the state
        they are added to.

        When the Gauss-Markov acceleration is expressed in a frame other than
        the propagation one (RTN or body-fixed), its components are
        :math:`\\mathbf w = \\mathbf T^T \\mathbf w_{\\mathrm{in}}` and it
        drives the translational state through :math:`\\mathbf T \\mathbf w`.
        The input :math:`\\mathbf u` keeps its (inertial) covariance and
        the gamma matrix is rotated instead,

        .. math::

            \\mathbf\\Gamma' =
            \\begin{bmatrix}
            \\mathbf T\\,\\mathbf\\Gamma_r\\,\\mathbf T^T \\\\
            \\mathbf T\\,\\mathbf\\Gamma_v\\,\\mathbf T^T \\\\
            \\mathbf\\Gamma_w\\,\\mathbf T^T
            \\end{bmatrix},

        with :math:`\\mathbf T` held at its value at the interval's
        :meth:`~scarabaeus.ProcessNoise._noise_epochs` entry.

        Parameters
        ----------
        epochs : list
            Filtering timeline (measurement and sequence event epochs) the
            filters index the process noise by.
        reference_state : callable, optional
            Reference ``(positions, velocities)`` at an array of epochs, which
            an RTN covariance is rotated along (see
            :meth:`~scarabaeus.ProcessNoise._continuous_time_covariances`).
            Defaults to ``None``.
        reference_beta : callable, optional
            Decay rates the dynamics use at an array of epochs, of shape
            ``(len(epochs), 3)``, evaluated at
            :meth:`~scarabaeus.ProcessNoise._noise_epochs`; not finite on a
            leg that does not carry :attr:`body`, where the noise is zero.
            Defaults to ``None``, which uses ``beta``.
        reference_frame : callable, optional
            Rotation :math:`\\mathbf T` from the frame the Gauss-Markov
            acceleration is expressed in to the propagation frame, at an array
            of epochs, of shape ``(len(epochs), 3, 3)``, evaluated at
            :meth:`~scarabaeus.ProcessNoise._noise_epochs`. Defaults to
            ``None``, the identity: an inertial Gauss-Markov acceleration.

        Returns
        -------
        None

        Raises
        ------
        DynamicalModelCompensation.MissingBetaError
            If neither ``reference_beta`` nor ``beta`` is given.
        """
        epochs = np.asarray(epochs, dtype=float)
        if reference_beta is not None:
            betas = reference_beta(self._noise_epochs(epochs))
        elif self.beta is not None:
            betas = self.beta
        else:
            raise self.MissingBetaError(
                "DMC needs the decay rates of the Gauss-Markov process: give "
                "beta, or let the filter take them from the dynamics."
            )
        betas = np.broadcast_to(np.asarray(betas, dtype=float), (len(epochs), 3))
        rotations = (
            np.asarray(reference_frame(self._noise_epochs(epochs)), dtype=float)
            if reference_frame is not None
            else None
        )

        # Continuous-time covariance over each interval, inertial
        Q_tilde = self._continuous_time_covariances(epochs, reference_state)
        # Interval lengths, cut to max_interval
        intervals = self._noise_intervals(epochs)
        discrete_time_process_covariances = []
        gamma_matrices = []

        for k in range(len(epochs)):
            # Local DMC gamma for [r, v, a_fogm]; zero at the first epoch,
            # which has no interval to accumulate noise over, and over an
            # interval on a leg without the body, which has no decay rates
            if np.all(np.isfinite(betas[k])):
                gamma_local = self._construct_gamma_matrix_dmc(intervals[k], betas[k])
                if rotations is not None:
                    gamma_local = self._rotate_gamma(gamma_local, rotations[k])
            else:
                gamma_local = np.zeros((9, 3))
            gamma_matrices.append(gamma_local)

            discrete_time_process_covariances.append(
                gamma_local @ Q_tilde[k] @ gamma_local.T
            )

        self.Q_tilde = Q_tilde
        self.discrete_time_process_covariances = discrete_time_process_covariances
        self.gamma_matrices = gamma_matrices
        self.epochs = epochs

    def _construct_gamma_matrix_dmc(self, dt: float, beta: np.ndarray) -> np.ndarray:
        """
        Construct the 9x3 DMC gamma matrix for a given time step.

        The gamma matrix relates the input ``u``, held constant over the
        step, to the position, velocity, and stochastic-acceleration states
        via the first-order Gauss-Markov dynamics :math:`\\dot{w} = -B w + u`.
        Each diagonal block is computed analytically:

        .. math::

            \\Gamma_r = \\frac{\\Delta t^2}{2}B^{-1} - \\Delta t B^{-2}
                      + B^{-3}(I - e^{-B\\Delta t})

            \\Gamma_v = \\Delta t B^{-1} - B^{-2}(I - e^{-B\\Delta t})

            \\Gamma_w = B^{-1}(I - e^{-B\\Delta t})

        where :math:`B = \\mathrm{diag}(\\beta)` is the diagonal time-constant
        matrix.  The discrete process noise covariance is then
        :math:`Q_k = \\Gamma Q \\Gamma^T`.

        The terms cancel to :math:`\\Delta t^3/6`, :math:`\\Delta t^2/2` and
        :math:`\\Delta t` as :math:`\\beta\\Delta t \\to 0`, which loses
        every digit for long correlation times or short steps. The blocks are
        therefore evaluated as :math:`\\Gamma_r = \\Delta t^3\\varphi_3`,
        :math:`\\Gamma_v = \\Delta t^2\\varphi_2` and
        :math:`\\Gamma_w = \\Delta t\\,\\varphi_1` at
        :math:`-\\beta\\Delta t` (see :meth:`_phi_functions`).

        Parameters
        ----------
        dt : float
            Interval the noise acts over, in seconds.
        beta : np.ndarray
            Decay rate of each axis over the interval, in 1/s.

        Returns
        -------
        np.ndarray
            Gamma matrix of shape ``(9, 3)``, with row blocks
            ``[Gamma_r; Gamma_v; Gamma_w]``.
        """
        # Initialize gamma matrix (9 x 3)
        Gamma_r = np.zeros((3, 3))
        Gamma_v = np.zeros((3, 3))
        Gamma_w = np.zeros((3, 3))

        # For each diagonal component (x, y, z)
        for i in range(3):
            phi_1, phi_2, phi_3 = self._phi_functions(-beta[i] * dt)

            # Position block:
            Gamma_r[i, i] = dt**3 * phi_3

            # Velocity block:
            Gamma_v[i, i] = dt**2 * phi_2

            # Stochastic parameter block:
            Gamma_w[i, i] = dt * phi_1

        # Stack the three blocks vertically
        gamma = np.vstack((Gamma_r, Gamma_v, Gamma_w))

        return gamma

    @staticmethod
    def _rotate_gamma(gamma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Gamma matrix of a Gauss-Markov acceleration expressed in a rotated
        frame, driven by an input in the propagation frame.

        Parameters
        ----------
        gamma : np.ndarray
            Gamma matrix of shape ``(9, 3)`` from
            :meth:`_construct_gamma_matrix_dmc`, row blocks
            ``[Gamma_r; Gamma_v; Gamma_w]``.
        T : np.ndarray
            Rotation from the Gauss-Markov frame to the propagation frame,
            shape ``(3, 3)``.

        Returns
        -------
        np.ndarray
            ``[T Gamma_r T^T; T Gamma_v T^T; Gamma_w T^T]``, shape ``(9, 3)``.
        """
        return np.vstack(
            (T @ gamma[:3] @ T.T, T @ gamma[3:6] @ T.T, gamma[6:] @ T.T)
        )

    @staticmethod
    def _phi_functions(z: float) -> tuple:
        """
        Exponential integrator functions :math:`\\varphi_1, \\varphi_2, \\varphi_3`.

        .. math::

            \\varphi_k(z) = \\sum_{j \\ge 0} \\frac{z^j}{(j + k)!},
            \\qquad
            \\varphi_{k+1}(z) = \\frac{\\varphi_k(z) - 1/k!}{z}

        The recurrence divides a cancellation by :math:`z`, so for
        :math:`|z| < 1` the series of :math:`\\varphi_3` is summed instead and
        the recurrence run upwards, which adds terms of the same size.

        Parameters
        ----------
        z : float
            Argument, :math:`-\\beta\\Delta t` for the DMC gamma matrix.

        Returns
        -------
        tuple of float
            :math:`(\\varphi_1(z), \\varphi_2(z), \\varphi_3(z))`.
        """
        if abs(z) < 1.0:
            # Truncated after (20)!, below machine precision of 1/6
            phi_3 = 0.0
            for j in range(17, -1, -1):
                phi_3 = phi_3 * z + 1.0 / math.factorial(j + 3)
            phi_2 = 0.5 + z * phi_3
            phi_1 = 1.0 + z * phi_2
        else:
            phi_1 = np.expm1(z) / z
            phi_2 = (phi_1 - 1.0) / z
            phi_3 = (phi_2 - 0.5) / z
        return phi_1, phi_2, phi_3

    '''    
    def _discretize_dmc(self, Q: np.ndarray, dt: float) -> np.ndarray:
            """
            Discretize continuous-time covariance using DMC model.

            Implements the exact discretization for first-order Gauss-Markov process
            with exponential correlation. The resulting 9x9 matrix includes correlations
            between position (r), velocity (v), and stochastic parameters (w).

            Parameters
            ----------
            Q : np.ndarray
                Continuous-time covariance (3x3).
            dt : float
                Time step.

            Returns
            -------
            np.ndarray
                Discrete-time covariance (9x9).
            """
            # Extract diagonal elements of beta matrix
            B = self.beta_matrix
            beta = np.diag(B)

            # Validate input covariance
            if Q.shape != (3, 3):
                raise ValueError("DMC expects a 3x3 continuous-time covariance matrix.")
            if not np.allclose(Q, np.diag(np.diag(Q))):
                raise ValueError("Current DMC implementation supports diagonal Q only.")

            # Initialize 3x3 block matrices
            Qrr = np.zeros((3, 3))
            Qrv = np.zeros((3, 3))
            Qrw = np.zeros((3, 3))
            Qvv = np.zeros((3, 3))
            Qvw = np.zeros((3, 3))
            Qww = np.zeros((3, 3))

            # Precompute powers and exponentials
            dt2 = dt**2
            dt3 = dt**3

            # Calculate each diagonal element
            for i in range(3):
                b = beta[i]
                q = Q[i, i]

                exp_b_dt = np.exp(-b * dt)
                exp_2b_dt = np.exp(-2 * b * dt)

                # Q_rr block (position-position correlation)
                Qrr[i, i] = q * (
                    (1 / (3 * b**2)) * dt3
                    - (1 / b**3) * dt2
                    + (1 / b**4) * dt
                    - (2 / b**4) * dt * exp_b_dt
                    + (1 / (2 * b**5)) * (1 - exp_2b_dt)
                )

                # Q_rv block (position-velocity correlation)
                Qrv[i, i] = q * (
                    (1 / (2 * b**2)) * dt2
                    - (1 / b**3) * dt
                    + (1 / b**3) * exp_b_dt * dt
                    + (1 / b**4) * (1 - exp_b_dt)
                    - (1 / (2 * b**4)) * (1 - exp_2b_dt)
                )

                # Q_rw block (position-stochastic parameter correlation)
                Qrw[i, i] = q * (
                    (1 / (2 * b**3)) * (1 - exp_2b_dt) - (1 / b**2) * exp_b_dt * dt
                )

                # Q_vv block (velocity-velocity correlation)
                Qvv[i, i] = q * (
                    (1 / b**2) * dt
                    - (2 / b**3) * (1 - exp_b_dt)
                    + (1 / (2 * b**3)) * (1 - exp_2b_dt)
                )

                # Q_vw block (velocity-stochastic parameter correlation)
                Qvw[i, i] = q * ((1 / (2 * b**2)) * (1 + exp_2b_dt) - (1 / b**2) * exp_b_dt)

                # Q_ww block (stochastic parameter-stochastic parameter correlation)
                Qww[i, i] = q * ((1 / (2 * b)) * (1 - exp_2b_dt))

            # Assemble the 9x9 block matrix
            Q_dmc = np.block([[Qrr, Qrv, Qrw], [Qrv, Qvv, Qvw], [Qrw, Qvw, Qww]])

            return Q_dmc
            
    def _pad_covariance(
        self,
        Qk_dmc: np.ndarray,
        n: int,
        current_time: float,
        sequence_epochs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Pad covariance matrix for full state vector.

        For DMC, the process noise is a 9x9 matrix (position, velocity, and
        3 stochastic parameters). This needs to be padded to match the full
        state vector size if there are additional estimated parameters.

        Parameters
        ----------
        Qk_dmc : np.ndarray
            DMC process noise covariance (9x9).
        n : int
            Full state size.
        current_time : float
            Current epoch.
        sequence_epochs : list, optional
            Sequence epochs if using sequence definition.

        Returns
        -------
        np.ndarray
            Padded covariance matrix (n x n).
        """
        current_size = Qk_dmc.shape[0]

        if current_size >= n:
            # Already correct size or larger
            return Qk_dmc[:n, :n]

        if self.state_definition is not None:
            # Pad to full state size
            padding = ((0, n - current_size), (0, n - current_size))
            return np.pad(Qk_dmc, padding, mode="constant", constant_values=0)

        elif self.sequence_definition is not None and sequence_epochs is not None:
            # Find current leg index and get appropriate state size
            idx_leg = self._find_leg_index(current_time, sequence_epochs)

            n_leg = (
                self.sequence_definition.states_n[idx_leg]
                or self.sequence_definition.states_n[idx_leg + 1]
            )

            padding = ((0, n_leg - current_size), (0, n_leg - current_size))
            return np.pad(Qk_dmc, padding, mode="constant", constant_values=0)

        else:
            # Default padding
            padding = ((0, n - current_size), (0, n - current_size))
            return np.pad(Qk_dmc, padding, mode="constant", constant_values=0)
    '''

    # endregion Methods #
    # -------------------#
