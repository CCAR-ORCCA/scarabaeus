# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
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

    where :math:`\\mathbf{w}_k \\sim \\mathcal{N}(\\mathbf{0},\\, \\mathbf{Q}_c)` and
    :math:`\\mathbf{Q}_c` (3, 3) is the covariance of the acceleration
    uncertainty, in km²/s⁴.

    **Discretization**

    Holding the acceleration constant over :math:`\\Delta t = t_{k+1} - t_k`
    and approximating the motion as rectilinear over the step, it maps into
    the position and velocity through

    .. math::

        \\mathbf{\\Gamma}_k = \\Delta t \\begin{bmatrix}
            \\frac{\\Delta t}{2}\\,\\mathbf{I}_3 \\\\[2pt]
            \\mathbf{I}_3
        \\end{bmatrix}

    so the discrete-time process noise covariance is

    .. math::

        \\mathbf{Q}_k = \\mathbf{\\Gamma}_k\\,\\mathbf{Q}_c\\,\\mathbf{\\Gamma}_k^T
        = \\Delta t^2 \\begin{bmatrix}
            \\frac{\\Delta t^2}{4}\\,\\mathbf{Q}_c &
            \\frac{\\Delta t}{2}\\,\\mathbf{Q}_c & \\mathbf{0} \\\\[2pt]
            \\frac{\\Delta t}{2}\\,\\mathbf{Q}_c &
            \\mathbf{Q}_c & \\mathbf{0} \\\\[2pt]
            \\mathbf{0} & \\mathbf{0} & \\mathbf{0}
        \\end{bmatrix}

    where the row/column partition is
    :math:`(\\mathbf{r},\\, \\mathbf{v},\\, \\mathbf{p})`. Static parameters
    :math:`\\mathbf{p}` receive no process noise. Integrating white noise over the step instead gives
    :math:`\\Delta t\\,[\\tfrac{\\Delta t^2}{3}, \\tfrac{\\Delta t}{2};
    \\tfrac{\\Delta t}{2}, 1] \\otimes \\mathbf{Q}_c`; the form above is
    larger for steps longer than a second, which suits an acceleration that is
    not estimated, and has rank three, which the SRIF needs to factor it
    through :math:`\\mathbf{\\Gamma}_k`. It grows quickly with
    :math:`\\Delta t`, so across long data gaps :math:`\\Delta t` should be
    capped with ``max_interval``.

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
    max_interval : ArrayWUnits or float, optional
        Longest interval the noise is accumulated over, in seconds when a
        float; a longer one, such as a data gap, receives the noise of one
        this long (see :meth:`~scarabaeus.ProcessNoise._noise_intervals`).
        Defaults to ``None``, no limit.
    body : Body or int, optional
        Body whose position and velocity the noise drives, or its SPICE ID.
        Defaults to ``None``, the owner of the first position in the state.

    See Also
    --------
    scarabaeus.DynamicalModelCompensation : Exponentially-correlated DMC variant.
    scarabaeus.ProcessNoise : Abstract base class.
    scarabaeus.ProcessNoiseSettings : Configuration object consumed by the filter.

    Notes
    -----
    The noise drives the position and velocity of ``body`` wherever they sit
    in the state; every other state, of that body or another, receives none.

    Dynamic components must be position or velocity, and must precede static
    components in the state vector.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        continuous_time_covariance: CovarianceMatrix,
        state_definition=None,
        sequence_definition=None,
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

    # ----------------#
    # region Methods #
    # ----------------#
    def _compute_discrete_time_covariances(
        self, epochs: list, reference_state=None
    ) -> None:
        """
        Compute discrete-time process noise covariances using SNC model.

        Transforms continuous-time covariance into discrete-time using the
        piecewise constant acceleration assumption. Computes both discrete
        covariances and gamma matrices for Square Root Information Filter (SRIF).

        The discrete covariance is computed as
        :math:`\\mathbf{Q}_k = \\Gamma_k \\mathbf{Q} \\Gamma_k^{\\mathsf{T}}`, with
        :math:`\\Gamma_k = [\\Delta t^2/2\\,\\mathbf{I}_3;\\ \\Delta t\\,\\mathbf{I}_3]`
        and :math:`\\Delta t` the interval length cut to ``max_interval``.
        Both are stored for the position/velocity block only; :meth:`gamma_at`
        and :meth:`covariance_at` size them for the state they are added to.

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

        Returns
        -------
        None
        """
        # Continuous-time covariance over each interval, inertial
        Q_tilde = self._continuous_time_covariances(epochs, reference_state)
        # Interval lengths, cut to max_interval
        intervals = self._noise_intervals(epochs)
        discrete_time_process_covariances = []
        gamma_matrices = []

        # Loop over timeline epochs
        for k in range(len(epochs)):
            if k == 0:
                # First epoch: no interval to accumulate noise over
                gamma_posvel = np.zeros((6, 3))
            else:
                gamma_posvel = self._construct_gamma_matrix(intervals[k])
            gamma_matrices.append(gamma_posvel)

            # Compute discrete covariance: Q_k = Γ * Q_tilde * Γ^T
            discrete_time_process_covariances.append(
                gamma_posvel @ Q_tilde[k] @ gamma_posvel.T
            )

        # Store computed values
        self.Q_tilde = Q_tilde
        self.discrete_time_process_covariances = discrete_time_process_covariances
        self.gamma_matrices = gamma_matrices
        self.epochs = np.asarray(epochs, dtype=float)

    def _construct_gamma_matrix(self, dt: float) -> np.ndarray:
        """
        Construct gamma matrix for position and velocity components.

        The position and velocity change over ``dt`` per unit of constant
        acceleration.

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
                (dt**2 / 2) * np.eye(3),  # Position rows
                dt * np.eye(3),  # Velocity rows
            )
        )
