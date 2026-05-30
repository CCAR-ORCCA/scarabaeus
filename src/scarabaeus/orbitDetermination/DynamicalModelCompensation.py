# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
from typing import Optional
from scarabaeus import CovarianceMatrix, ProcessNoise


# --------------------#
#  Class Definition  #
# --------------------#
class DynamicModelCompensation(ProcessNoise):
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

        \\dot{\\mathbf{w}} = -\\mathbf{B}\\,\\mathbf{w} + \\mathbf{v}(t)

    where :math:`\\mathbf{B} = \\mathrm{diag}(\\beta_x, \\beta_y, \\beta_z)` contains
    the inverse correlation times (decay rates), and :math:`\\mathbf{v}(t)` is zero-mean
    white noise with continuous-time PSD :math:`\\mathbf{Q}` (3×3).

    **Exact discretization (gamma matrix)**

    The mapping from the white-noise input :math:`\\mathbf{v}(t)` to the state
    perturbation over a step :math:`\\Delta t` is given by the exact analytic
    gamma matrix :math:`\\mathbf{\\Gamma}_k` (9×3):

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
        The continuous-time process noise covariance matrix.
    state_definition : optional
        State vector definition. Defaults to None.
    sequence_definition : optional
        Sequence definition for multi-leg trajectories. Defaults to None.
    beta : np.ndarray
        Time-constant vector (inverse correlation times) for the Gauss-Markov process.
        Can be a scalar (isotropic) or a 3-element vector for x, y, z components.

    Notes
    -----
    As :math:`\\beta \\to \\infty` (very short correlation time) DMC approaches SNC;
    as :math:`\\beta \\to 0` (long correlation time) the stochastic acceleration
    becomes nearly constant and is effectively estimated as a deterministic parameter.

    See Also
    --------
    scarabaeus.StateNoiseCompensation : Simpler piecewise-constant SNC variant.
    scarabaeus.ProcessNoise : Abstract base class.
    scarabaeus.ProcessNoiseSettings : Configuration object consumed by the filter.

    References
    ----------
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        continuous_time_covariance: CovarianceMatrix,
        state_definition=None,
        sequence_definition=None,
        beta: Optional[np.ndarray] = None,
    ):
        super().__init__(
            continuous_time_covariance, state_definition, sequence_definition
        )

        # Validate beta matrix if provided
        if beta is None:
            raise ValueError(
                "DMC requires a beta vector (inverse correlation times). "
                "Provide a 3-element vector [beta_x, beta_y, beta_z]."
            )
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size == 1:
            # isotropic case
            beta = np.repeat(beta, 3)
        elif beta.size != 3:
            raise ValueError("Beta must be either a scalar or a 3-element vector.")
        self.beta = beta
        self.beta_matrix = np.diag(beta)
        if np.any(self.beta <= 0):
            raise ValueError("All beta components must be strictly positive.")

    # ----------------#
    # region Methods #
    # ----------------#
    def _check_state_definition(self, state_definition) -> None:
        """
        Validate state definition for DMC compatibility.

        Required structure:
        1. The first three components of the state definition must be, in order:
        - position
        - velocity
        - a_fogm
        all belonging to the same body.
        2. No other body may contain an 'a_fogm' component.

        Parameters
        ----------
        state_definition : list
            State definition to validate. Each entry is expected to be a tuple like:
            (name, size, estimation, dynamics, owner, value)

        Raises
        ------
        ValueError
            If the required DMC structure is not satisfied.
        """
        if state_definition is None or len(state_definition) < 3:
            raise ValueError(
                "DMC requires at least the first three state components to be "
                "('position', 'velocity', 'a_fogm') for the same body."
            )

        # First three components must be position, velocity, a_fogm
        first = state_definition[0]
        second = state_definition[1]
        third = state_definition[2]

        try:
            name1, _, _, _, owner1, _ = first
            name2, _, _, _, owner2, _ = second
            name3, _, _, _, owner3, _ = third
        except Exception as e:
            raise ValueError(
                f"Invalid state definition entry format in first three components: {e}"
            )

        # Check ordering
        if name1 != "position" or name2 != "velocity" or name3 != "a_fogm":
            raise ValueError(
                "DMC requires the first three state components to be exactly "
                "('position', 'velocity', 'a_fogm') in that order."
            )

        # Check same owner/body
        if not (owner1 is owner2 and owner2 is owner3):
            raise ValueError(
                "DMC requires 'position', 'velocity', and 'a_fogm' to belong to the same body."
            )

        dmc_owner = owner1

        # Check that no other body has a_fogm
        for tup in state_definition[3:]:
            try:
                name, _, _, _, owner, _ = tup
            except Exception as e:
                raise ValueError(f"Invalid state definition entry format: {e}")

            if name == "a_fogm" and owner is not dmc_owner:
                raise ValueError(
                    "DMC currently supports only one body with 'a_fogm'. "
                    f"Found additional 'a_fogm' for body {owner}."
                )

    def _compute_discrete_time_covariances(self, measurement_times: list) -> None:
        """
        Compute discrete-time process noise covariances using DMC model.

        For DMC, the process noise applies to augmented stochastic parameters
        following a first-order Gauss-Markov process with time-constant matrix B.

        Parameters
        ----------
        measurement_times : list
            List of time epochs at which the process noise is evaluated.

        Returns
        -------
        None
        """
        Q_tilde = []
        discrete_time_process_covariances = []
        gamma_matrices = []
        n = self._get_state_size()

        sequence_epochs = None
        if self.sequence_definition is not None:
            sequence_epochs = [
                epochs.times for epochs in self.sequence_definition.epochs_vec
            ]

        for k in range(len(measurement_times)):
            Q_continuous = self.continuous_time_covariance.matrix_without_units()
            Q_tilde.append(Q_continuous)

            # Transform from RTN if needed
            Q_tilde[k] = self.transform_from_rtn(Q_tilde[k])

            if k == 0:
                discrete_time_process_covariances.append(np.zeros((n, n)))

                if self.state_definition is not None:
                    gamma_size = n
                elif (
                    self.sequence_definition is not None and sequence_epochs is not None
                ):
                    idx_leg = self._find_leg_index(
                        measurement_times[k], sequence_epochs
                    )
                    gamma_size = (
                        self.sequence_definition.states_n[idx_leg]
                        or self.sequence_definition.states_n[idx_leg + 1]
                    )
                else:
                    gamma_size = 9

                gamma_matrices.append(np.zeros((gamma_size, 3)))
            else:
                dt = measurement_times[k] - measurement_times[k - 1]

                # Local DMC gamma for [r, v, a_fogm]
                gamma_local = self._construct_gamma_matrix_dmc(dt)  # 9x3

                # Pad rows like SNC
                gamma_padded = self._pad_gamma_matrix_dmc(
                    gamma_local, n, measurement_times[k], sequence_epochs
                )
                gamma_matrices.append(gamma_padded)

                # Full-state discrete covariance
                Qk_discrete = gamma_padded @ Q_tilde[k] @ gamma_padded.T
                discrete_time_process_covariances.append(Qk_discrete)

        self.Q_tilde = Q_tilde
        self.discrete_time_process_covariances = discrete_time_process_covariances
        self.gamma_matrices = gamma_matrices

    def _construct_gamma_matrix_dmc(self, dt: float, n: int = 9) -> np.ndarray:
        """
        Construct the exact 9x3 DMC gamma matrix for a given time step.

        The gamma matrix relates continuous white-noise input ``v(t)`` to the
        position, velocity, and stochastic-acceleration states via the
        first-order Gauss-Markov dynamics ``ẇ = -B w + v``.  Each diagonal
        block is computed analytically:

        .. math::

            \\Gamma_r = \\frac{\\Delta t^2}{2}B^{-1} - \\Delta t B^{-2}
                      + B^{-3}(I - e^{-B\\Delta t})

            \\Gamma_v = \\Delta t B^{-1} - B^{-2}(I - e^{-B\\Delta t})

            \\Gamma_w = B^{-1}(I - e^{-B\\Delta t})

        where :math:`B = \\mathrm{diag}(\\beta)` is the diagonal time-constant
        matrix.  The discrete process noise covariance is then
        :math:`Q_k = \\Gamma Q \\Gamma^T`.

        Parameters
        ----------
        dt : float
            Time step in seconds between consecutive measurement epochs.
        n : int, optional
            Unused; retained for API consistency. Defaults to 9.

        Returns
        -------
        np.ndarray
            Gamma matrix of shape ``(9, 3)``, with row blocks
            ``[Gamma_r; Gamma_v; Gamma_w]``.
        """
        B = self.beta_matrix
        beta = np.diag(B)

        # Initialize gamma matrix (9 x 3)
        Gamma_r = np.zeros((3, 3))
        Gamma_v = np.zeros((3, 3))
        Gamma_w = np.zeros((3, 3))

        # For each diagonal component (x, y, z)
        for i in range(3):
            b = beta[i]
            exp_b_dt = np.exp(-b * dt)
            I_minus_exp = 1 - exp_b_dt

            # Position block:
            Gamma_r[i, i] = (
                (dt**2 / 2) * (1 / b) - dt * (1 / b**2) + (1 / b**3) * I_minus_exp
            )

            # Velocity block:
            Gamma_v[i, i] = dt * (1 / b) - (1 / b**2) * I_minus_exp

            # Stochastic parameter block:
            Gamma_w[i, i] = (1 / b) * I_minus_exp

        # Stack the three blocks vertically
        gamma = np.vstack((Gamma_r, Gamma_v, Gamma_w))

        return gamma

    def _pad_gamma_matrix_dmc(
        self,
        gamma_local: np.ndarray,
        n: int,
        current_time: float,
        sequence_epochs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Extend the 9x3 DMC gamma matrix to the active state size at ``current_time``.

        The local 9-row block corresponds to the leading
        ``[position, velocity, a_fogm]`` sub-state.  When additional static
        parameters are estimated, zero rows are appended so the matrix matches
        the full state dimension used by the filter.

        The boundary rule for sequence trajectories is identical to
        :meth:`StateNoiseCompensation._pad_gamma_matrix`: the first epoch after
        a leg-transition node uses the previous leg's state size.

        Parameters
        ----------
        gamma_local : np.ndarray
            DMC gamma matrix of shape ``(9, 3)`` from
            :meth:`_construct_gamma_matrix_dmc`.
        n : int
            Full state size when ``state_definition`` is set (non-sequence mode).
        current_time : float
            Current measurement epoch (TDB seconds).
        sequence_epochs : list, optional
            List of epoch arrays per leg from the sequence definition.

        Returns
        -------
        np.ndarray
            Zero-padded gamma matrix of shape ``(n_active, 3)``.
        """

        # -------------------------
        # Fixed-state case
        # -------------------------
        if self.state_definition is not None:
            padding = ((0, n - 9), (0, 0))
            return np.pad(gamma_local, padding, mode="constant", constant_values=0)

        if self.sequence_definition is None or sequence_epochs is None:
            return gamma_local

        # -------------------------
        # Helpers: robust epoch extraction
        # -------------------------
        def _to_float_array(ep):
            v = getattr(ep, "values", None)
            if v is not None:
                return np.asarray(v, dtype=float).ravel()
            t = getattr(ep, "times", None)
            if t is not None:
                vv = getattr(t, "values", None)
                if vv is not None:
                    return np.asarray(vv, dtype=float).ravel()
            return np.asarray(ep, dtype=float).ravel()

        ct = float(current_time)

        # Leg start / second timestamps from the propagation grid
        leg_start = []
        leg_second = []
        for ep in sequence_epochs:
            arr = _to_float_array(ep)
            if arr.size == 0:
                leg_start.append(None)
                leg_second.append(None)
            else:
                leg_start.append(float(arr[0]))
                leg_second.append(float(arr[1]) if arr.size > 1 else None)

        # Nominal leg index
        idx_leg = self._find_leg_index(ct, sequence_epochs)

        # -------------------------
        # Boundary rule identical to SNC
        # -------------------------
        atol = 1e-7
        for j in range(1, len(leg_start)):
            t0 = leg_start[j]
            t1 = leg_second[j]
            if t0 is None or t1 is None:
                continue

            if (ct > t0 + atol) and (ct <= t1 + atol):
                idx_leg = j - 1
                break

        # Active state size
        n_leg = self.sequence_definition.states_n[idx_leg]
        if not n_leg:
            n_leg = self.sequence_definition.states_n[idx_leg + 1]

        padding = ((0, int(n_leg) - 9), (0, 0))
        return np.pad(gamma_local, padding, mode="constant", constant_values=0)

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
