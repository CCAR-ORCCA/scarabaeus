# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    FilterOD,
    CovarianceMatrix,
    SolutionOD,
    Propagator,
    FilterSettings,
    MissionSequence,
    MeasurementSpec,
)
from typing import List, Dict, Optional, Tuple
import scarabaeus.utils.NumpyWrapper as np
from scipy import linalg as spln
from scipy.linalg import solve_triangular


#######################################################################
class SRIF(FilterOD):
    """ Square Root Information Filter (SRIF) for spacecraft orbit determination.

    The SRIF is the **sequential** square-root information estimator.  It shares
    the fixed-reference-trajectory philosophy of the :class:`~scarabaeus.LKF` — the reference
    is never updated mid-pass — but represents the state uncertainty in the
    **square-root information domain** rather than the covariance domain.  This
    eliminates explicit matrix inversion from the inner loop and roughly doubles
    the effective numerical precision compared to the covariance-form LKF.

    **State representation**

    Instead of the covariance :math:`P` and deviation :math:`\\hat{x}`, the SRIF
    stores the pair :math:`(R_{xx},\\, z_x)` defined by:

    .. math::

        P = R_{xx}^{-1} R_{xx}^{-T}, \\qquad \\hat{x} = R_{xx}^{-1} z_x

    where :math:`R_{xx}` is the upper-triangular Cholesky factor of the
    information matrix :math:`\\Lambda = P^{-1} = R_{xx}^T R_{xx}`.

    **Time update**

    The square-root information matrix is propagated backward through the
    one-step STM :math:`\\Phi_{k,k-1}`:

    .. math::

        R_{xx,k|k-1} = R_{xx,k-1}\\,\\Phi_{k,k-1}^{-1}, \\qquad
        z_{x,k|k-1} = z_{x,k-1}

    (the information vector is unchanged when no process noise is active).
    When **SNC or DMC** process noise is present, the Thornton–Bierman
    augmented QR formulation updates both :math:`R_{xx}` and the process-noise
    sub-block in a single orthogonal transformation without ever forming
    :math:`Q_k` explicitly.

    **Measurement update**

    At each epoch :math:`t_k`, the whitened observation equation is appended
    below the current information block and the combined system is triangularized
    via a **Householder QR decomposition** (no matrix inversion):

    .. math::

        \\begin{bmatrix}
            R_{xx,k|k-1} & z_{x,k|k-1} \\\\
            \\tilde{R}_k^{-1} H_k & \\tilde{R}_k^{-1} y_k
        \\end{bmatrix}
        \\xrightarrow{\\text{QR}}
        \\begin{bmatrix}
            \\hat{R}_{xx,k} & \\hat{z}_{x,k} \\\\
            0 & e_k
        \\end{bmatrix}

    where :math:`\\tilde{R}_k` is the upper-triangular Cholesky factor of the
    measurement noise covariance :math:`R_k`, and :math:`e_k` is the normalized
    postfit residual.  The posterior estimate and covariance are recovered as:

    .. math::

        \\hat{x}_k = \\hat{R}_{xx,k}^{-1}\\,\\hat{z}_{x,k}, \\qquad
        \\hat{P}_k = \\hat{R}_{xx,k}^{-1}\\,\\hat{R}_{xx,k}^{-T}.

    This sequential update is numerically equivalent to the :class:`~scarabaeus.SRIFB`
    batch accumulation applied one epoch at a time, with intermediate time
    updates between epochs enabling SNC and DMC process noise.

    **Schmidt consider parameters**

    When consider parameters :math:`c` are present, the whitened measurement
    equation is augmented with the consider Jacobian :math:`H_{c,k}` before
    the QR step:

    .. math::

        \\begin{bmatrix}
            \\hat{R}_{xx,k|k-1} & 0 & \\hat{z}_{x,k|k-1} \\\\
            \\tilde{R}_k^{-1} H_{x,k} & \\tilde{R}_k^{-1} H_{c,k} &
            \\tilde{R}_k^{-1} y_k
        \\end{bmatrix}
        \\xrightarrow{\\text{QR}}
        \\begin{bmatrix}
            \\hat{R}_{xx,k} & \\hat{R}_{xc,k} & \\hat{z}_{x,k} \\\\
            0 & e_{c,k} & e_k
        \\end{bmatrix}

    The off-diagonal block :math:`\\hat{R}_{xc,k}` encodes the cross-
    correlation between estimated and consider uncertainties.  The consider
    covariance :math:`P_{cc}` is fixed throughout and only inflates the
    posterior covariance of the estimated state.

    **Backward smoother (Bierman–Thornton)**

    The forward pass stores at every epoch :math:`k` the posterior information
    pair :math:`(\\hat{R}_{xx,k},\\, \\hat{z}_{x,k})` and the process-noise
    sub-blocks needed to account for SNC/DMC.  The backward smoother then
    processes from epoch :math:`K` down to :math:`1`.

    At each backward step, the smoothed information square root
    :math:`\\tilde{R}_{xx,k+1}` from epoch :math:`k+1` is first
    back-propagated through the one-step STM:

    .. math::

        \\tilde{R}_{xx,k+1}^- = \\tilde{R}_{xx,k+1}\\,\\Phi_{k+1,k}^{-1}

    and then combined with the forward-filter information via an orthogonal
    transformation:

    .. math::

        \\begin{bmatrix}
            \\hat{R}_{xx,k} & \\hat{z}_{x,k} \\\\
            \\tilde{R}_{xx,k+1}^- & \\tilde{z}_{x,k+1}
        \\end{bmatrix}
        \\xrightarrow{\\text{QR}}
        \\begin{bmatrix}
            \\tilde{R}_{xx,k} & \\tilde{z}_{x,k} \\\\
            0 & e_k^s
        \\end{bmatrix}

    The smoothed estimate and covariance are recovered as:

    .. math::

        \\hat{x}_k^s = \\tilde{R}_{xx,k}^{-1}\\,\\tilde{z}_{x,k}, \\qquad
        \\hat{P}_k^s = \\tilde{R}_{xx,k}^{-1}\\,\\tilde{R}_{xx,k}^{-T}

    When SNC or DMC process noise is active, the stored Thornton–Bierman
    sub-blocks :math:`(\\bar{R}_{uu},\\, \\bar{R}_{ux},\\, \\tilde{b}_u)` enter
    the backward QR step to account for the noise contribution in the
    inter-epoch transition.

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Defines the reference trajectory and STMs.  A :class:`~scarabaeus.MissionSequence`
        can be used for multi-leg trajectories with maneuver events.
    measurements : MeasurementSpec
        Measurement specification object.
    settings : FilterSettings
        Filter configuration including initial covariance and optional SNC/DMC
        process noise model.
    traj_name : str, optional
        Name for the trajectory BSP file.  Auto-generated if ``None``.
    traj_dir : str, optional
        Directory for the trajectory BSP file.
    overwrite_traj : bool, optional
        If ``True``, overwrite any existing trajectory file.

    Notes
    -----
    The SRIF is preferred over :class:`~scarabaeus.LKF` when the problem
    is ill-conditioned (e.g., sparse tracking, very tight a-priori, or poorly
    observable parameters).  When no process noise is required, the batch
    formulation :class:`~scarabaeus.SRIFB` is simpler and equivalent.

    See Also
    --------
    scarabaeus.LKF : Covariance-form sequential filter; same algorithm, lower
        numerical precision.
    scarabaeus.SRIFB : Batch (non-sequential) SRIF; no process noise support.
    scarabaeus.FilterOD : Base class providing trajectory and measurement handling.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004).
    Statistical Orbit Determination.
    Elsevier Academic Press.
    ISBN 978-0-12-683630-1.

    .. [2] Bierman, G. J. (1977).
    Factorization Methods for Discrete Sequential Estimation.
    Academic Press.
    ISBN 978-0-12-097650-8.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q(self) -> int:
        """Dimension of the process-noise vector (3 for SNC and DMC)."""
        return self._q

    @q.setter
    def q(self, value: int):
        self._q = value

    @property
    def rhat_k(self) -> list:
        """:math:`\\hat{R}_{xx,k}^{-1}` stored at each epoch during the forward
        pass for use by the backward smoother."""
        return self._rhat_k

    @rhat_k.setter
    def rhat_k(self, value):
        self._rhat_k = value

    @property
    def bhat_k(self) -> list:
        """Posterior information vectors :math:`\\hat{z}_{x,k}` stored during the
        forward pass for the backward smoother."""
        return self._bhat_k

    @bhat_k.setter
    def bhat_k(self, value):
        self._bhat_k = value

    @property
    def btilde_u_sub_k(self) -> list:
        """Process-noise sub-block :math:`\\tilde{b}_u` from the Thornton–Bierman
        augmented QR factorisation (SNC/DMC smoother)."""
        return self._btilde_u_sub_k

    @btilde_u_sub_k.setter
    def btilde_u_sub_k(self, value):
        self._btilde_u_sub_k = value

    @property
    def rbar_u_sub_k(self) -> list:
        """Upper-triangular noise sub-block :math:`\\bar{R}_{uu}` from the
        augmented QR factorisation (SNC/DMC smoother)."""
        return self._rbar_u_sub_k

    @rbar_u_sub_k.setter
    def rbar_u_sub_k(self, value):
        self._rbar_u_sub_k = value

    @property
    def rbar_ux_sub_k(self) -> list:
        """Cross sub-block :math:`\\bar{R}_{ux}` from the augmented QR factorisation
        (SNC/DMC smoother)."""
        return self._rbar_ux_sub_k

    @rbar_ux_sub_k.setter
    def rbar_ux_sub_k(self, value):
        self._rbar_ux_sub_k = value

    @property
    def covariance_history(self) -> list:
        """Posterior covariance :math:`\\hat{P}_k` at each measurement epoch.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._covariance_history

    @covariance_history.setter
    def covariance_history(self, value):
        self._covariance_history = value

    @property
    def covariance_consider_history(self) -> list:
        """Consider-parameter covariance block at each epoch.
        Empty list when no consider parameters are active."""
        return self._covariance_consider_history

    @covariance_consider_history.setter
    def covariance_consider_history(self, value):
        self._covariance_consider_history = value

    @property
    def state_deviation_history(self) -> list:
        """Posterior state deviation :math:`\\hat{x}_k` at each measurement epoch.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._state_deviation_history

    @state_deviation_history.setter
    def state_deviation_history(self, value):
        self._state_deviation_history = value

    @property
    def prefit_residuals(self) -> dict:
        """Pre-update residuals :math:`y_k`, keyed by dataset name.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._prefit_residuals

    @prefit_residuals.setter
    def prefit_residuals(self, value):
        self._prefit_residuals = value

    @property
    def postfit_residuals(self) -> dict:
        """Post-update residuals, keyed by dataset name.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._postfit_residuals

    @postfit_residuals.setter
    def postfit_residuals(self, value):
        self._postfit_residuals = value

    @property
    def smoothed_state_deviation(self) -> list:
        """Smoothed state deviation at each epoch.
        Populated when smoothing is enabled in :class:`~scarabaeus.OutputSettings`."""
        return self._smoothed_state_deviation

    @smoothed_state_deviation.setter
    def smoothed_state_deviation(self, value):
        self._smoothed_state_deviation = value

    @property
    def smoothed_covariance(self) -> list:
        """Smoothed covariance at each epoch.
        Populated when smoothing is enabled in :class:`~scarabaeus.OutputSettings`."""
        return self._smoothed_covariance

    @smoothed_covariance.setter
    def smoothed_covariance(self, value):
        self._smoothed_covariance = value

    def __init__(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        settings: FilterSettings,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
    ):
        # parent class initialization
        super().__init__(
            propagator, measurements, settings, traj_name, traj_dir, overwrite_traj
        )

        # Define q and n for extracting information from the transformed matrices
        self.q = 3  # Noise vector

        # Track information needed for smoothing
        # Note the gamma matrices are stored in self.process_noise.gamma_matrices
        self.num_processed_steps = 0
        self.STM_k_from_k_minus_1 = []
        self.posterior_covariance = []
        self.rhat_k = []
        self.bhat_k = []
        self.btilde_u_sub_k = []
        self.rbar_u_sub_k = []
        self.rbar_ux_sub_k = []

        # Initialize output containers
        self._covariance_history = []
        self._covariance_consider_history = []
        self._state_deviation_history = []
        self._prefit_residuals = {}
        self._postfit_residuals = {}
        self._smoothed_state_deviation = []
        self._smoothed_covariance = []

        # Initialize process noise from settings if provided
        if self.settings.process_noise is not None:
            self._initialize_process_noise_from_settings()

        ## Print
        print("")
        print("=" * 80)
        print("Initializing Sequence Square Root Information Filter (SRIF)")
        print("=" * 80)
        if hasattr(self, "process_noise"):
            print(f"Process Noise Model: {self.settings.process_noise.type}")
        print("=" * 80)

    def _compute(
        self,
        meas_corr: list = None,
        printOutput: bool = True,
        underweighting_factor: float = 1.0,
    ) -> SolutionOD:
        """
        Execute one forward pass of the Square Root Information Filter.

        The SRIF propagates and updates the **square-root information matrix**
        :math:`R_{xx}` (upper triangular, such that :math:`P = (R_{xx}^T R_{xx})^{-1}`)
        and the **information vector** :math:`z_x = R_{xx}\\,\\hat{x}`, rather
        than the covariance or state deviation directly.  This improves
        numerical stability by avoiding explicit covariance inversion.

        Time update (square-root information propagation)
        --------------------------------------------------
        The square-root information matrix is propagated via:

        .. math::

            R_{xx,k|k-1} = R_{xx,k-1}\\,\\Phi_{k,k-1}^{-1}

        The information vector follows as
        :math:`z_{x,k|k-1} = R_{xx,k|k-1}\\,\\Phi_{k,k-1}\\,\\hat{x}_{k-1}`.
        Note that only :math:`R_{xx}` and :math:`z_x` are propagated forward;
        the state deviation :math:`\\hat{x}` is never stored directly during
        the time update — it is recovered at each epoch from
        :math:`\\hat{x}_k = R_{xx,k}^{-1} z_{x,k}`.

        When process noise is active (SNC or DMC), the predicted square-root
        information matrix and information vector are updated jointly via a
        QR factorisation of the augmented block matrix:

        .. math::

            \\begin{bmatrix}
                R_u & 0 & 0 \\\\
                -R_{xx} \\Gamma & R_{xx} & z_x
            \\end{bmatrix}

        where :math:`R_u = \\mathrm{chol}(Q^{-1})` and :math:`\\Gamma` is the
        discrete noise input matrix.  The factorisation is performed by
        ``_apply_process_noise_correction`` (defined locally inside
        ``_compute``).

        Measurement update (QR accumulation)
        -------------------------------------
        Whitened partials and residuals are appended below the current
        information block and a QR factorisation extracts the updated
        square-root information matrix and information vector:

        .. math::

            \\begin{bmatrix}
                R_{xx,k|k-1} & z_{x,k|k-1} \\\\
                R_y^{-1} H_x & R_y^{-1} y_k
            \\end{bmatrix}
            = Q_k \\begin{bmatrix} R_{xx,k} & z_{x,k} \\\\ * & * \\end{bmatrix}

        The posterior state deviation and covariance are recovered as:

        .. math::

            \\hat{x}_k = R_{xx,k}^{-1} z_{x,k}, \\quad
            \\hat{P}_k = R_{xx,k}^{-1} R_{xx,k}^{-T}

        Parameters
        ----------
        meas_corr : list, optional
            Correlation factors forwarded to :class:`~scarabaeus.CovarianceMatrix`.
        printOutput : bool, optional
            Print per-epoch progress. Defaults to True.
        underweighting_factor : float, optional
            Multiplicative scale on the measurement covariance.  Defaults to 1.0.

        Returns
        -------
        SolutionOD
            Solution object with state deviation history, covariance history,
            pre/postfit residuals, and smoother auxiliary arrays populated.
        """

        ## Utils
        def _propagate_information_deviation(Rxx_k1, Rxc_k1, dx_k1, STM_k, STM_k1):
            """
            SRIF time update: propagate square-root information matrix and information vector.

            Updates R_xx via  R_xx_bar = R_xx_hat @ Phi^{-1}  (Bierman 1977, Eq. 4.2.9)
            and carries dx forward only to form z_x_bar = R_xx_bar @ dx_bar on return.
            The state deviation dx is NOT the primary propagated quantity — R_xx and
            z_x = R_xx @ dx are. dx_bar is returned solely for the caller to reconstruct
            z_x_bar; it is not stored as a filter estimate.

            Shapes:
            Rxx_k1: (nx, nx) upper-tri  -- square-root information matrix at k-1
            Rxc_k1: (nx, nc) or None    -- consider cross-term at k-1
            dx_k1 : (nx, 1)             -- state deviation at k-1 (used to form z_x)
            STM_k : (n, n)              -- cumulative STM at k
            STM_k1: (n, n)              -- cumulative STM at k-1
            Returns:
            Rxx_k : (nx, nx) upper-tri  -- predicted square-root information matrix
            Rxc_bar: (nx, nc) or None
            dx_k  : (nx, 1)             -- predicted dx (used only to form z_x_bar)
            """
            # Get STM from k1 to k
            if getattr(self, "flag_node", False) and self.flag_node:
                prev_n = self.legs_n[counter_events - 1]
                curr_STM_prev_leg = self.trajectory._STMs[counter_events - 1][
                    -1
                ].reshape(prev_n, prev_n)
                STM_kk1 = curr_STM_prev_leg @ self._safe_inv(STM_k1)
            else:
                STM_kk1 = STM_k @ self._safe_inv(STM_k1)
            Phi_k, Psi_k, Phi_c_k = self._split_stm_phi_psi(STM_kk1)

            # Propagate square-root information matrix: R_xx_bar = R_xx_hat @ Phi^{-1}
            # Carry dx forward to form z_x_bar = R_xx_bar @ dx_bar (not stored directly)
            Rxx_k = Rxx_k1 @ self._safe_inv(Phi_k)
            dx_k = Phi_k @ dx_k1

            # Consider parameters cross-terms
            if self.has_consider:
                Rxc_bar = (Rxc_k1 - Rxx_k @ Psi_k) @ self._safe_inv(Phi_c_k)
            else:
                Rxc_bar = None

            # Store STM for smoothing
            self.STM_k_from_k_minus_1.append(Phi_k)
            return Rxx_k, Rxc_bar, dx_k, Phi_k, Psi_k, Phi_c_k

        def _apply_process_noise_correction(Rxx_k, zx_k, Rxc_k, k):
            """
            SRIF time update with additive process noise on x only.
            If consider is enabled, also propagates Rxc consistently through the same QR.
            Shapes:
            Rxx_k: (nx, nx) upper-tri
            zx_k : (nx, 1)
            Rxc_k: (nx, nc)  (only if has_consider)
            Ru   : (q, q)
            gamma_k: (nx, q)
            """
            Ru = self._safe_inv(spln.cholesky(self.process_noise.Q_tilde[k]))
            nx = self.n_est if self.has_consider else self.n
            q = self.q
            gamma_k = self.process_noise.gamma_matrices[k]  # (nx, q)

            # Ensure gamma matches the CURRENT information matrix size (state may still be 6 here)
            n_k = Rxx_k.shape[0]
            if gamma_k.shape[0] != n_k:
                gamma_k = gamma_k[
                    :n_k, :
                ]  # keep top rows (pos/vel first), drop padded zeros

            if self.has_consider:
                nc = self.n_con
                # Build augmented QR system:
                J = np.vstack(
                    (
                        np.hstack(
                            (
                                Ru,
                                np.zeros((q, nx)),
                                np.zeros((q, nc)),
                                np.zeros((q, 1)),
                            )
                        ),
                        np.hstack(
                            (
                                -Rxx_k @ gamma_k,
                                Rxx_k,
                                Rxc_k,
                                zx_k,
                            )
                        ),
                    )
                )

                _, Jhat = np.linalg.qr(J)

                # Extract updated info blocks (still upper-tri in the x columns)
                Rxx_k = Jhat[q : q + nx, q : q + nx]
                Rxc_k = Jhat[q : q + nx, q + nx : q + nx + nc]
                zx_k = Jhat[q : q + nx, q + nx + nc : q + nx + nc + 1].reshape(nx, 1)

                # Store smoothing blocks (keep your existing ones; add the new one if you want)
                self.rbar_u_sub_k.append(Jhat[0:q, 0:q])
                self.rbar_ux_sub_k.append(Jhat[0:q, q : q + nx])
                # optional (often useful for consider smoothing / debugging)
                if hasattr(self, "rbar_uc_sub_k"):
                    self.rbar_uc_sub_k.append(Jhat[0:q, q + nx : q + nx + nc])
                self.btilde_u_sub_k.append(Jhat[0:q, q + nx + nc : q + nx + nc + 1])
            else:
                J = np.vstack(
                    (
                        np.hstack(
                            (
                                Ru,
                                np.zeros((q, nx)),
                                np.zeros((q, 1)),
                            )
                        ),
                        np.hstack(
                            (
                                -Rxx_k @ gamma_k,
                                Rxx_k,
                                zx_k,
                            )
                        ),
                    )
                )
                _, Jhat = np.linalg.qr(J)
                Rxx_k = Jhat[q : q + nx, q : q + nx]
                zx_k = Jhat[q : q + nx, q + nx : q + nx + 1].reshape(nx, 1)
                Rxc_k = None

                self.rbar_u_sub_k.append(Jhat[0:q, 0:q])
                self.rbar_ux_sub_k.append(Jhat[0:q, q : q + nx])
                self.btilde_u_sub_k.append(Jhat[0:q, q + nx : q + nx + 1])

            return Rxx_k, Rxc_k, zx_k

        ## Initialize state length if sequence
        if self.flag_sequence is True:
            self.n = self.legs_n[0]

        # Create initial square root information matrix from initial covariance
        P0 = self.initial_covariance.matrix_without_units()
        if hasattr(self, "flag_MTM") and self.flag_MTM is True:
            MTM = self.trajectory._STMs[0].reshape(
                int(np.sqrt(len(self.trajectory._STMs[0]))), -1
            )
            P0 = MTM @ P0 @ MTM.T
            self.flag_MTM = False

        if self.has_consider:
            Pxx_hat_0 = P0[np.ix_(self.idx_est, self.idx_est)]
            Pcc_hat_0 = P0[np.ix_(self.idx_con, self.idx_con)]
            Pxc_hat_0 = P0[np.ix_(self.idx_est, self.idx_con)]
            R0 = self._sqrt_info_matrix((P0))
            Rxx_hat_k1 = R0[np.ix_(self.idx_est, self.idx_est)]
            Rxc_hat_k1 = R0[np.ix_(self.idx_est, self.idx_con)]
        else:
            Pxx_hat_0 = P0
            Rxx_hat_k1 = self._sqrt_info_matrix((Pxx_hat_0))
            Rxc_hat_k1 = None

        # Extract spacecraft times
        measurement_times = np.asarray(
            self.measurement_data.get_spacecraft_times(), dtype=float
        )
        if self.flag_sequence:
            event_times = np.asarray(self.events_epochs, dtype=float)
            timeline_times = np.unique(np.concatenate((measurement_times, event_times)))
        else:
            event_times = np.array([])
            timeline_times = measurement_times

        # Initialize storage as instance attributes
        STM_k1 = np.eye(self.n)
        dx_hat_k1 = np.zeros((self.n_est if self.has_consider else self.n, 1))
        Pcc_hat_k1 = Pcc_hat_0 if self.has_consider else None
        self.covariance_history = []
        self.covariance_consider_history = []
        self.state_deviation_history = []
        self.prefit_residuals = []
        self.postfit_residuals = []
        # Reset smoother-related arrays each call so multiple compute() calls do not
        # accumulate stale entries that corrupt the backward smoother indexing.
        self.num_processed_steps = 0
        self.STM_k_from_k_minus_1 = []
        self.posterior_covariance = []
        self.rhat_k = []
        self.bhat_k = []
        self.btilde_u_sub_k = []
        self.rbar_u_sub_k = []
        self.rbar_ux_sub_k = []
        counter_events = 0
        n_use = self.n_est if self.has_consider else self.n
        self.update_flag = True

        # Iterate through measurements
        for k in range(len(timeline_times)):
            current_time = timeline_times[k]
            idx_match = np.where(measurement_times == current_time)[0]
            has_measurement = len(idx_match) > 0
            self.update_flag = (
                has_measurement  # Only update if we have a measurement at this epoch
            )

            #### Events Handling ####
            if self.flag_sequence is True and counter_events < len(self.events_epochs):
                if current_time == self.events_epochs[counter_events]:
                    counter_events += 1
                    self.flag_node = True
                    print(
                        f"Event detected at the epoch {current_time} [TDB]: re-initializing the filter at this epoch."
                    )
                    self.n = self.legs_n[counter_events]
                    n_use = self.n_est if self.has_consider else self.n

            #### Extract Measurement Data for Epoch t2 ####
            block = None
            if self.update_flag:
                block = self.measurement_data.get_combined_for_t2(current_time)
                block = block[current_time]
                sigmas = block["sigma"]
                partials = block["partials"][:, : self.n]
                residuals = block["residuals"]
                instruments = block["instruments"]
                measurement_types = block["measurement_types"]
                dataset_names = block["dataset_names"]
                t2_all = block["t2"]

            #### Information Propagation ####
            # Extract STM
            if self.flag_sequence is True:
                k_offset = (
                    sum(len(stm) for stm in self.trajectory._STMs[:counter_events])
                    - counter_events
                )
                idx_leg = counter_events
                idx_in_leg = k - k_offset  # within-leg sample index

                STM_k = self.trajectory.get_STM_sequence(
                    epoch=float(current_time),
                    idx_leg=idx_leg,  # crucial for sequence
                )
            else:
                STM_k = self.trajectory.get_STM(epoch=current_time, idx=k)

            ## DEVIATION AND INFORMATION PREDICTION
            Rxx_bar_k, Rxc_bar_k, dx_bar_k, Phi_k, Psi_k, Phi_c_k = (
                _propagate_information_deviation(
                    Rxx_hat_k1, Rxc_hat_k1, dx_hat_k1, STM_k, STM_k1
                )
            )
            zx_bar_k = Rxx_bar_k @ dx_bar_k
            if self.has_consider:
                Pcc_bar_k = Phi_c_k @ Pcc_hat_k1 @ Phi_c_k.T

            # node handling applies AFTER time update, on BAR quantities
            if getattr(self, "flag_node", False) and self.flag_node:
                if self.has_consider:
                    n_est_prev = self.n_est  # OLD leg estimated size (before re-init)
                    nc = self.n_con  # OLD leg consider size

                    # Convert SRIF-Schmidt BAR objects to full covariance/deviation
                    Rxx_bar_k_inv = solve_triangular(
                        Rxx_bar_k, np.eye(n_est_prev), lower=False
                    )
                    Pxx_bar_k = Rxx_bar_k_inv @ Rxx_bar_k_inv.T
                    Sxc_bar_k = -self._safe_inv(Rxx_bar_k) @ Rxc_bar_k

                    # Full inflated covariance at k|k-1
                    # NOTE: here not inflated, just mapped to new definition
                    Pxc_bar_k = Sxc_bar_k @ Pcc_bar_k
                    P_full = np.block(
                        [[Pxx_bar_k, Pxc_bar_k], [Pxc_bar_k.T, Pcc_bar_k]]
                    )

                    # Full deviation vector (x part only is estimated; c mean=0 in deviations)
                    dx_full = np.zeros((n_est_prev + nc, 1))
                    dx_full[:n_est_prev, :] = dx_bar_k

                    # Apply node mapping (definition change)
                    dx_full = self._map_deviation_definition_node(
                        counter_events, dx_full
                    )
                    P_full = self._map_covariance_definition_node(
                        counter_events, P_full
                    )

                    # After mapping you can re-init consider partition safely
                    self._init_consider_partition(idx_leg=counter_events)
                    n_use = self.n_est if self.has_consider else self.n
                    nc = self.n_con

                    # Apply impulsive burn at node on the covariance
                    P_full = STM_k @ P_full @ STM_k.T

                    # Extract mapped blocks using NEW leg indices (after re-init)
                    Pxx_bar_k = P_full[np.ix_(self.idx_est, self.idx_est)]
                    Pxc_bar_k = P_full[np.ix_(self.idx_est, self.idx_con)]

                    # Refactor back to SRIF-Schmidt BAR objects
                    R0_node = self._sqrt_info_matrix((P_full))
                    Rxx_bar_k = R0_node[np.ix_(self.idx_est, self.idx_est)]
                    Rxc_bar_k = R0_node[np.ix_(self.idx_est, self.idx_con)]
                    Sxc_bar_k = -Rxx_bar_k @ Rxc_bar_k

                    # Map dx back to x and rebuild z using NEW leg indices
                    dx_bar_k = dx_full[self.idx_est]
                    zx_bar_k = Rxx_bar_k @ dx_bar_k

                else:
                    # Non-consider: just map x covariance/deviation at BAR level
                    prev_n = self.legs_n[counter_events - 1]
                    Rxx_bar_k_inv = solve_triangular(
                        Rxx_bar_k, np.eye(prev_n), lower=False
                    )
                    Pxx_bar_k = Rxx_bar_k_inv @ Rxx_bar_k_inv.T

                    dx_full = self._map_deviation_definition_node(
                        counter_events, dx_bar_k
                    )
                    P_full = self._map_covariance_definition_node(
                        counter_events, Pxx_bar_k
                    )

                    # After mapping you can re-init consider partition safely
                    self._init_consider_partition(idx_leg=counter_events)
                    n_use = self.n_est if self.has_consider else self.n
                    nc = self.n_con

                    # Apply impulsive burn at node on the covariance
                    P_full = STM_k @ P_full @ STM_k.T

                    # Re-extract mapped blocks
                    if self.has_consider:
                        Pxx_bar_k = P_full[np.ix_(self.idx_est, self.idx_est)]
                        Pcc_bar_k = P_full[np.ix_(self.idx_con, self.idx_con)]
                        Pxc_bar_k = P_full[np.ix_(self.idx_est, self.idx_con)]
                        R0 = self._sqrt_info_matrix((P_full))
                        Rxx_bar_k = R0[np.ix_(self.idx_est, self.idx_est)]
                        Rxc_bar_k = R0[np.ix_(self.idx_est, self.idx_con)]
                    else:
                        Pxx_bar_k = P_full[: self.n_est, : self.n_est]
                        Rxx_bar_k = self._sqrt_info_matrix((Pxx_bar_k))
                        Rxc_bar_k = None

                    dx_bar_k = dx_full[:n_use]
                    zx_bar_k = Rxx_bar_k @ dx_bar_k
                zx_bar_k = np.atleast_2d(zx_bar_k).reshape(-1, 1)

            # Add SNC if needed, else re-triangularize the predicted information terms via QR
            if self.SNC_flag or self.DMC_flag:
                Rxx_bar_k, Rxc_bar_k, zx_bar_k = _apply_process_noise_correction(
                    Rxx_bar_k, zx_bar_k, Rxc_bar_k, k
                )
            else:
                if self.has_consider:
                    nx, nc = self.n_est, self.n_con
                    _, J_num_k = np.linalg.qr(
                        np.hstack((Rxx_bar_k, Rxc_bar_k, zx_bar_k))
                    )
                    Rxx_bar_k = J_num_k[:nx, :nx]
                    Rxc_bar_k = J_num_k[:nx, nx : nx + nc]
                    zx_bar_k = J_num_k[:nx, nx + nc : nx + nc + 1]
                else:
                    _, J_num_k = np.linalg.qr(np.hstack((Rxx_bar_k, zx_bar_k)))
                    Rxx_bar_k = J_num_k[:n_use, :n_use]
                    zx_bar_k = J_num_k[:n_use, n_use : n_use + 1]

            #### Measurement information computations ####

            # Measurement information whitening and split H in estimated and considered
            if self.update_flag:
                Hx, Hc = self._split_partials_hx_hc(partials)
                measurement_covariance = underweighting_factor * np.array(
                    CovarianceMatrix(
                        sigmas,
                        current_time,
                        from_list=True,
                        corr_factors=meas_corr,
                    ).matrix
                )
                measurement_SRI = self._sqrt_info_matrix((measurement_covariance))
                partials_w = measurement_SRI @ Hx
                if self.has_consider:
                    partials_c_w = measurement_SRI @ Hc

            # Compute the posterior covariance and deviation
            if self.covariance_analysis is False:
                #### Measurement information state computations ####
                if self.update_flag:
                    residuals_w = measurement_SRI @ residuals

                    # Form cost function and minimize
                    if self.has_consider:
                        J_bar_k = np.vstack(
                            (
                                np.hstack((Rxx_bar_k, Rxc_bar_k, zx_bar_k)),
                                np.hstack((partials_w, partials_c_w, residuals_w)),
                            )
                        )
                    else:
                        J_bar_k = np.vstack(
                            (
                                np.hstack((Rxx_bar_k, zx_bar_k)),
                                np.hstack((partials_w, residuals_w)),
                            )
                        )
                    _, J_hat_k = np.linalg.qr(J_bar_k)

                    # Recover information terms
                    if self.has_consider:
                        nx = self.n_est
                        nc = self.n_con
                        Rxx_hat_k = J_hat_k[:nx, :nx]
                        Rxc_hat_k = J_hat_k[:nx, nx : nx + nc]
                        zx_hat_k = J_hat_k[:nx, nx + nc : nx + nc + 1].reshape(nx, 1)
                    else:
                        n = self.n
                        Rxx_hat_k = J_hat_k[:n, :n]
                        zx_hat_k = J_hat_k[:n, n : n + 1].reshape(n, 1)
                else:
                    Rxx_hat_k = Rxx_bar_k
                    Rxc_hat_k = Rxc_bar_k
                    zx_hat_k = zx_bar_k

                #### Save deviation, covariance ####
                Rxx_hat_k_inv = solve_triangular(Rxx_hat_k, np.eye(n_use), lower=False)
                Pxx_hat_k = Rxx_hat_k_inv @ Rxx_hat_k_inv.T
                dx_hat_k = Rxx_hat_k_inv @ zx_hat_k

                # Compute postfit and save deviation
                if self.update_flag:
                    postfit = residuals - Hx @ dx_hat_k

                if self.has_consider:
                    full_dev = np.zeros((self.n, 1))
                    full_dev[self.idx_est, :] = dx_hat_k
                    self.state_deviation_history.append(full_dev)
                else:
                    self.state_deviation_history.append(dx_hat_k)

                # Format residual output
                self.prefit_residuals.append(
                    [
                        residuals,
                        sigmas,
                        instruments,
                        measurement_types,
                        dataset_names,
                        t2_all,
                    ]
                )
                if self.update_flag:
                    self.postfit_residuals.append(
                        [
                            postfit,
                            sigmas,
                            instruments,
                            measurement_types,
                            dataset_names,
                            t2_all,
                        ]
                    )

            else:
                if self.update_flag:
                    # Form cost function and minimize
                    if self.has_consider:
                        J_bar_k = np.vstack(
                            (
                                np.hstack((Rxx_bar_k, Rxc_bar_k)),
                                np.hstack((partials_w, partials_c_w)),
                            )
                        )
                    else:
                        J_bar_k = np.vstack(
                            (
                                np.hstack((Rxx_bar_k)),
                                np.hstack((partials_w)),
                            )
                        )
                    _, J_hat_k = np.linalg.qr(J_bar_k)

                    # Recover information terms
                    if self.has_consider:
                        nx = self.n_est
                        nc = self.n_con
                        Rxx_hat_k = J_hat_k[:nx, :nx]
                        Rxc_hat_k = J_hat_k[:nx, nx : nx + nc]
                    else:
                        n = self.n
                        Rxx_hat_k = J_hat_k[:n, :n]
                else:
                    Rxx_hat_k = Rxx_bar_k
                    Rxc_hat_k = Rxc_bar_k

                #### Save covariance ####
                Rxx_hat_k_inv = solve_triangular(Rxx_hat_k, np.eye(n_use), lower=False)
                Pxx_hat_k = Rxx_hat_k_inv @ Rxx_hat_k_inv.T

                # Bookkeeping only: no estimated correction in covariance analysis
                zx_hat_k = zx_bar_k
                dx_hat_k = dx_bar_k

            # Make full posterior covariance in case of consider parameters
            if self.has_consider:
                Sxc_hat_k = -self._safe_inv(Rxx_hat_k) @ Rxc_hat_k
                Pc_hat_k = Pxx_hat_k + Sxc_hat_k @ Pcc_bar_k @ Sxc_hat_k.T
                Pxc_hat_k = Sxc_hat_k @ Pcc_bar_k
                P_full_cons = np.block(
                    [[Pc_hat_k, Pxc_hat_k], [Pxc_hat_k.T, Pcc_bar_k]]
                )
                P_full = np.block([[Pxx_hat_k, Pxc_hat_k], [Pxc_hat_k.T, Pcc_bar_k]])
            else:
                P_full = Pxx_hat_k

            # Save data for smoothing
            self.bhat_k.append(zx_hat_k)
            self.rhat_k.append(Rxx_hat_k_inv)
            self.posterior_covariance.append(P_full)

            # Store covariance as matrix (not flattened)
            self.covariance_history.append(P_full)
            self.covariance_consider_history.append(
                P_full_cons if self.has_consider else None
            )

            # Iterate
            STM_k1 = STM_k
            Rxx_hat_k1 = Rxx_hat_k
            dx_hat_k1 = dx_hat_k
            if self.has_consider:
                Rxc_hat_k1 = Rxc_hat_k
                Pcc_hat_k1 = Pcc_bar_k

            # Record how many steps have been processed
            self.num_processed_steps = k

            # Print progress
            if printOutput:
                epoch = current_time
                progress = (
                    (epoch - measurement_times[0])
                    / (measurement_times[-1] - measurement_times[0])
                    * 100
                )
                trace_P = np.trace(P_full)
                mean_sigma = float(np.mean(sigmas)) if self.update_flag else np.nan

                if self.covariance_analysis is True:
                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] "
                        f"tr(P)={trace_P:.4e}   "
                        f"⟨σ⟩={mean_sigma:.2e}"
                    )
                else:
                    norm_prefit = np.linalg.norm(residuals)
                    norm_postfit = np.linalg.norm(postfit)

                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] "
                        f"‖prefit‖={norm_prefit:.4e}   "
                        f"‖postfit‖={norm_postfit:.4e}   "
                        f"tr(P)={trace_P:.4e}   "
                        f"⟨σ⟩={mean_sigma:.2e}"
                    )

            # Reset flags after node
            if hasattr(self, "flag_node") and self.flag_node is True:
                self.flag_node = False
                self.update_flag = True

        # Reformat postfit residuals
        if self.covariance_analysis is False:
            self.prefit_residuals = self._unwrap_residuals(
                self.prefit_residuals, self.measurement_data.dataset_names
            )
            self.postfit_residuals = self._unwrap_residuals(
                self.postfit_residuals, self.measurement_data.dataset_names
            )

        # Return SolutionOD object
        return SolutionOD(self)

    def _smoother(self, printOutput: bool = True) -> SolutionOD:
        """
        SRIF backward smoother.

        Must be called after :meth:`compute`.  Performs a backward pass using
        the Thornton-Bierman square-root information smoother when process noise
        is active, or a direct STM back-propagation otherwise.

        **With process noise (SNC/DMC)** — square-root information backward pass:

        At each step ``k`` (from ``N-1`` down to ``1``), the augmented matrix:

        .. math::

            m_k = \\begin{bmatrix}
                R_{u,k} + R_{ux,k}\\Gamma_k & R_{ux,k}\\Phi_{k,k-1} & \\tilde{b}_{u,k} \\\\
                \\tilde{R}_{xx,k}\\Gamma_k   & \\tilde{R}_{xx,k}\\Phi_{k,k-1} & \\tilde{b}_{x,k}
            \\end{bmatrix}

        is reduced by QR factorisation to yield the smoothed information
        matrix and vector at step ``k-1``.

        **Without process noise** — direct back-propagation:

        .. math::

            \\tilde{x}_0 = \\Phi_N^{-1} \\hat{x}_N, \\quad
            \\tilde{P}_0 = \\Phi_N^{-1} \\hat{P}_N \\Phi_N^{-T}

        and forward-propagated to each interior epoch via
        :math:`\\tilde{x}_k = \\Phi_k \\tilde{x}_0`.

        Parameters
        ----------
        printOutput : bool, optional
            Print per-epoch smoothed postfit residual norm and covariance trace.

        Returns
        -------
        SolutionOD
            Solution object with ``deviation_smooth``, ``covariance_smooth``,
            and ``postfits_smoother`` populated.

        Raises
        ------
        ValueError
            If consider parameters or sequence mode are active (not supported).

        Notes
        -----
        Requires ``compute()`` to have been called first so that ``rhat_k``,
        ``bhat_k``, and the smoother auxiliary arrays are populated.
        """
        if self.has_consider:
            raise ValueError(
                "Smoother SRIF does not support consider parameters in the current implementation."
            )
        if self.flag_sequence:
            raise ValueError(
                "Smoother SRIF does not support sequence mode in the current implementation."
            )

        if printOutput:
            print("")
            print("=" * 80)
            print("Performing SRIF Backward Smoothing")
            print("=" * 80)

        # Initialize smoothed histories.
        # self.rhat_k stores Rxx_hat_k_inv (= P^{1/2}); the SRIF backward QR needs
        # Rxx_hat_k (the upper-triangular info matrix), so invert each entry.
        # Build new lists so the backward pass does not mutate the stored filter data.
        smoothed_rhat_k = [self._safe_inv(r) for r in self.rhat_k]
        smoothed_bhat_k = list(self.bhat_k)

        smoothed_state_devs = self.state_deviation_history.copy()
        smoothed_covars = self.posterior_covariance.copy()

        # Spacecraft epochs
        spacecraft_times = self.measurement_data.get_spacecraft_times()

        # Backward recursion
        if self.SNC_flag or self.DMC_flag:
            for k in range(self.num_processed_steps, 0, -1):
                m1 = np.vstack(
                    (
                        np.hstack(
                            (
                                self.rbar_u_sub_k[k]
                                + self.rbar_ux_sub_k[k]
                                @ self.process_noise.gamma_matrices[k],
                                self.rbar_ux_sub_k[k] @ self.STM_k_from_k_minus_1[k],
                                self.btilde_u_sub_k[k],
                            )
                        ),
                        np.hstack(
                            (
                                smoothed_rhat_k[k]
                                @ self.process_noise.gamma_matrices[k],
                                smoothed_rhat_k[k] @ self.STM_k_from_k_minus_1[k],
                                smoothed_bhat_k[k],
                            )
                        ),
                    )
                )
                _, tstar_k_minus_1 = np.linalg.qr(m1)

                bstar = tstar_k_minus_1[
                    self.q : self.q + self.n, self.q + self.n : self.q + self.n + 1
                ]
                rstar = tstar_k_minus_1[
                    self.q : self.q + self.n, self.q : self.q + self.n
                ]

                smoothed_bhat_k[k - 1] = bstar
                smoothed_rhat_k[k - 1] = rstar

                smoothed_state_devs[k - 1] = self._safe_inv(rstar) @ bstar
                smoothed_covars[k - 1] = self._safe_inv(rstar) @ self._safe_inv(rstar.T)
        else:
            # Retrieve the cumulative STM at the final measurement epoch by epoch
            # (direct _STMs indexing would use the propagation-step counter, which does
            # not align with the measurement counter when the propagation grid is finer).
            Phi_N = self.trajectory.get_STM(
                epoch=spacecraft_times[self.num_processed_steps],
                idx=self.num_processed_steps,
            )
            Phi_N_inv = self._safe_inv(Phi_N)

            # Smoothed state and covariance at t_0:
            #   x_s(0) = Phi_N^{-1} @ x_hat(N)
            #   P_s(0) = Phi_N^{-1} @ P_hat(N) @ Phi_N^{-T}
            smoothed_state_devs[0] = (
                Phi_N_inv @ smoothed_state_devs[self.num_processed_steps]
            )
            smoothed_covars[0] = (
                Phi_N_inv @ smoothed_covars[self.num_processed_steps] @ Phi_N_inv.T
            )

            # Propagate forward to each interior epoch:
            #   x_s(k) = Phi(0→k) @ x_s(0)
            #   P_s(k) = Phi(0→k) @ P_s(0) @ Phi(0→k)^T
            for k in range(1, self.num_processed_steps):
                STM = self.trajectory.get_STM(epoch=spacecraft_times[k], idx=k)
                smoothed_state_devs[k] = STM @ smoothed_state_devs[0]
                smoothed_covars[k] = STM @ smoothed_covars[0] @ STM.T

        # Build postfit residuals
        postfit_residuals_sm = []

        spacecraft_times = self.measurement_data.get_spacecraft_times()
        for k, epoch in enumerate(spacecraft_times):
            block = self.measurement_data.get_combined_for_t2(epoch)[epoch]
            residuals = block["residuals"]
            sigmas = block["sigma"]
            partials = block["partials"][:, : self.n]
            instruments = block["instruments"]
            measurement_types = block["measurement_types"]
            dataset_names = block["dataset_names"]
            t2_all = block["t2"]

            postfit = residuals - partials @ smoothed_state_devs[k]

            postfit_residuals_sm.append(
                [postfit, sigmas, instruments, measurement_types, dataset_names, t2_all]
            )

            if printOutput:
                prog = (
                    (epoch - spacecraft_times[0])
                    / (spacecraft_times[-1] - spacecraft_times[0])
                    * 100
                )
                print(
                    f"[{epoch:.2f} TDB | {prog:5.1f}%] "
                    f"‖postfit_s‖={np.linalg.norm(postfit):.4e}   "
                    f"tr(P_s)={np.trace(smoothed_covars[k]):.4e}"
                )

        # Store smoothed results in filter object for SolutionOD to extract
        self.smoothed_state_deviation = smoothed_state_devs
        self.smoothed_covariance = smoothed_covars  # Store as list of matrices
        self.postfit_residuals_smooth = self._unwrap_residuals(
            postfit_residuals_sm, self.measurement_data.dataset_names
        )

        # Return SolutionOD object (it will extract smoothed data)
        return SolutionOD(self)
