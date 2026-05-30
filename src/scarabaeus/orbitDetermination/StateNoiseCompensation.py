# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
from typing import Optional
from scarabaeus import CovarianceMatrix, ProcessNoise

# --------------------#
#  Class Definition  #
# --------------------#


class StateNoiseCompensation(ProcessNoise):
    """ State Noise Compensation (SNC) process noise model.

    Implements the SNC approach for modeling unmodeled accelerations in spacecraft
    dynamics. The continuous-time process noise is discretized using a piecewise
    constant acceleration model, particularly effective for atmospheric drag, solar
    radiation pressure variations, and other unmodeled forces.

    **Process noise model**

    The unmodeled acceleration is assumed piecewise constant between consecutive
    measurement epochs :math:`t_k` and :math:`t_{k+1}`:

    .. math::

        \\mathbf{a}(t) = \\mathbf{w}_k, \\quad t \\in [t_k,\\, t_{k+1})

    where :math:`\\mathbf{w}_k \\sim \\mathcal{N}(\\mathbf{0},\\, \\mathbf{Q})` and
    :math:`\\mathbf{Q}` (3×3) is the continuous-time power spectral density (PSD)
    of the acceleration uncertainty.

    **Discretization**

    Integrating the continuous dynamics over :math:`\\Delta t = t_{k+1} - t_k`
    (Tapley et al., 2004) gives the exact discrete-time process noise covariance:

    .. math::

        \\mathbf{Q}_k = \\begin{bmatrix}
            \\frac{\\Delta t^3}{3}\\,\\mathbf{Q}_c &
            \\frac{\\Delta t^2}{2}\\,\\mathbf{Q}_c & \\mathbf{0} \\\\[2pt]
            \\frac{\\Delta t^2}{2}\\,\\mathbf{Q}_c &
            \\Delta t\\,\\mathbf{Q}_c & \\mathbf{0} \\\\[2pt]
            \\mathbf{0} & \\mathbf{0} & \\mathbf{0}
        \\end{bmatrix}

    where the row/column partition is
    :math:`(\\mathbf{r},\\, \\mathbf{v},\\, \\mathbf{p})` and
    :math:`\\mathbf{Q}_c` (3×3) is the continuous-time PSD of the unmodeled
    acceleration. Static parameters :math:`\\mathbf{p}` receive no process noise.

    **Covariance propagation**

    At each time step the propagated covariance is augmented by the discrete
    process noise:

    .. math::

        \\bar{\\mathbf{P}}_{k+1} =
            \\mathbf{\\Phi}_{k+1,k}\\,\\mathbf{P}_k\\,\\mathbf{\\Phi}_{k+1,k}^T
            + \\mathbf{Q}_k

    where :math:`\\mathbf{\\Phi}_{k+1,k}` is the state transition matrix and
    :math:`\\mathbf{Q}_k` captures the uncertainty growth from the unmodeled
    acceleration over :math:`[t_k, t_{k+1}]`.

    Parameters
    ----------
    continuous_time_covariance : CovarianceMatrix
        The continuous-time process noise covariance matrix.
    state_definition : optional
        State vector definition. Defaults to None.
    sequence_definition : optional
        Sequence definition for multi-leg trajectories. Defaults to None.

    Notes
    -----
    Dynamic components must be position or velocity, and must precede static
    components in the state vector.

    See Also
    --------
    scarabaeus.DynamicModelCompensation : Exponentially-correlated DMC variant.
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
    ):
        super().__init__(
            continuous_time_covariance, state_definition, sequence_definition
        )

    # ----------------#
    # region Methods #
    # ----------------#
    def _check_state_definition(self, state_definition) -> None:
        """
        Validate state definition for SNC compatibility.

        Ensures dynamic components precede static components and are position/velocity.

        Parameters
        ----------
        state_definition : list
            State definition to validate.

        Raises
        ------
        ValueError
            If state definition violates SNC rules.

        Returns
        -------
        None
        """
        flag_static = False
        for i, tup in enumerate(state_definition):
            state_type, _, _, component_type, _, _ = tup
            if component_type == "static":
                flag_static = True
            elif component_type == "dynamic":
                if flag_static:
                    raise ValueError(
                        "With SNC enabled, dynamic components cannot follow static "
                        "components in the state definition."
                    )
                if state_type not in ("position", "velocity"):
                    raise ValueError(
                        "With SNC enabled, dynamic components must be either "
                        "position or velocity."
                    )

    def _compute_discrete_time_covariances(self, measurement_times: list) -> None:
        """
        Compute discrete-time process noise covariances using SNC model.

        Transforms continuous-time covariance into discrete-time using the
        piecewise constant acceleration assumption. Computes both discrete
        covariances and gamma matrices for Square Root Information Filter (SRIF).

        The discrete covariance is computed as: Q_k = Γ_k * Q * Γ_k^T
        where Γ_k = [[dt^3/2 * I_3], [dt^2 * I_3], [0]]

        Parameters
        ----------
        measurement_times : list
            List of time epochs at which the process noise is evaluated.

        Returns
        -------
        None
        """
        # Initialize
        Q_tilde = []
        discrete_time_process_covariances = []
        gamma_matrices = []
        n = self._get_state_size()

        sequence_epochs = None
        if self.sequence_definition is not None:
            sequence_epochs = [
                epochs.times for epochs in self.sequence_definition.epochs_vec
            ]

        # Loop over measurement epochs
        for k in range(len(measurement_times)):
            # Get continuous-time covariance
            Q_continuous = self.continuous_time_covariance.matrix_without_units()
            Q_tilde.append(Q_continuous)

            # Transform from RTN if necessary
            Q_tilde[k] = self.transform_from_rtn(Q_tilde[k])

            # Compute discrete-time covariance and gamma matrix
            if k == 0:
                # First epoch: initialize with zeros
                discrete_time_process_covariances.append(np.zeros((n, n)))

                # Determine appropriate gamma size
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
                    gamma_size = 6

                gamma_matrices.append(np.zeros((gamma_size, 3)))
            else:
                dt = measurement_times[k] - measurement_times[k - 1]

                # Construct gamma matrix (6x3 for pos/vel only)
                gamma_posvel = self._construct_gamma_matrix(dt)

                # Pad gamma matrix if needed
                gamma_padded = self._pad_gamma_matrix(
                    gamma_posvel, n, measurement_times[k], sequence_epochs
                )
                gamma_matrices.append(gamma_padded)

                # Compute discrete covariance: Q_k = Γ * Q_tilde * Γ^T
                Qk_discrete = gamma_padded @ Q_tilde[k] @ gamma_padded.T

                discrete_time_process_covariances.append(Qk_discrete)

        # Store computed values
        self.Q_tilde = Q_tilde
        self.discrete_time_process_covariances = discrete_time_process_covariances
        self.gamma_matrices = gamma_matrices

    def _construct_gamma_matrix(self, dt: float) -> np.ndarray:
        """
        Construct gamma matrix for position and velocity components.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Gamma matrix (6 x 3) for position and velocity.
        """
        return np.vstack(
            (
                (dt**3 / 2) * np.eye(3),  # Position rows
                (dt**2) * np.eye(3),  # Velocity rows
            )
        )

    def _pad_gamma_matrix(
        self,
        gamma_posvel: np.ndarray,
        n: int,
        current_time: float,
        sequence_epochs: Optional[list] = None,
    ) -> np.ndarray:
        """
        Extend the 6x3 SNC gamma matrix to match the active state size at ``current_time``.

        The 6-row ``gamma_posvel`` block covers only position and velocity.  For
        states larger than 6, zero rows are appended so that the matrix conforms
        to the full state dimension used by the filter at the given epoch.

        For sequence trajectories, a boundary rule is applied: the epoch
        immediately following a leg-transition node (i.e. any time in the
        half-open interval ``(t_start[j], t_second[j]]``) uses the *previous*
        leg's state size rather than the new leg's size.  This prevents a
        premature size switch before the filter has processed the node event.

        Parameters
        ----------
        gamma_posvel : np.ndarray
            SNC gamma matrix of shape ``(6, 3)``.
        n : int
            Full state size when ``state_definition`` is set (non-sequence mode).
        current_time : float
            Current measurement epoch (TDB seconds).
        sequence_epochs : list, optional
            List of epoch arrays, one per leg, from the sequence definition.

        Returns
        -------
        np.ndarray
            Zero-padded gamma matrix of shape ``(n_active, 3)``, where
            ``n_active`` is the state size appropriate for ``current_time``.
        """

        # -------------------------
        # Fixed-state case
        # -------------------------
        if self.state_definition is not None:
            padding = ((0, n - 6), (0, 0))
            return np.pad(gamma_posvel, padding, mode="constant", constant_values=0)

        if self.sequence_definition is None or sequence_epochs is None:
            return gamma_posvel

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

        # Leg start / second timestamps from the *propagation grid* (sequence epochs)
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
        # Boundary rule (NO exact match needed):
        # if ct is the first encountered epoch after node j,
        # i.e. ct in (leg_start[j], leg_second[j]] -> use j-1 size.
        # -------------------------
        atol = 1e-7  # just float-noise guard, not an "epsilon logic" for behavior
        for j in range(1, len(leg_start)):
            t0 = leg_start[j]
            t1 = leg_second[j]
            if t0 is None or t1 is None:
                continue

            # first encountered epoch after node is anything after t0 up to and including t1
            if (ct > t0 + atol) and (ct <= t1 + atol):
                idx_leg = j - 1
                break

        # State size for the chosen leg
        n_leg = self.sequence_definition.states_n[idx_leg]
        if not n_leg:
            n_leg = self.sequence_definition.states_n[idx_leg + 1]

        padding = ((0, int(n_leg) - 6), (0, 0))
        return np.pad(gamma_posvel, padding, mode="constant", constant_values=0)
