# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    FilterOD,
    SolutionOD,
    Propagator,
    FilterSettings,
    MissionSequence,
    MeasurementSpec,
    ProcessNoise,
)
from typing import List, Dict, Optional, Tuple
import scarabaeus.utils.NumpyWrapper as np


#######################################################################
class LKF(FilterOD):
    """Linearized Kalman Filter (LKF) for spacecraft orbit determination.

    The LKF is a sequential estimator that operates on a **fixed reference
    trajectory** computed once before filtering begins.  At each measurement
    epoch :math:`t_k` the filter estimates the *deviation* from that reference,
    :math:`\\hat{x}_k`, rather than the full state.  This is the key distinction
    from the Extended Kalman Filter (EKF): the EKF re-linearizes around the
    updated state after every measurement, effectively re-propagating a new
    reference at each step.

    **Why the LKF is preferred over the EKF in astrodynamics**

    For deep-space and interplanetary missions, tracking passes are typically
    short and separated by long coast arcs of hours to days.  In this sparse-
    measurement regime the EKF is prone to divergence: the re-linearization
    point drifts with each update, and the STM computed from the updated state
    can become inconsistent with the actual nonlinear dynamics over the long
    coast arcs, destabilizing the covariance propagation.  The LKF avoids
    this by keeping a single, dynamically consistent reference trajectory
    throughout the entire pass.  Iterated re-runs of the filter (outer
    iterations) progressively reduce the linearization error without
    sacrificing numerical stability.

    **Algorithm**

    Let :math:`\\hat{x}_k` be the estimated state deviation at epoch :math:`t_k`
    and :math:`\\hat{P}_k` be the associated covariance.

    *Time update* — propagate from :math:`t_{k-1}` to :math:`t_k` using the
    one-step STM :math:`\\Phi_{k,k-1} = \\Phi_k \\Phi_{k-1}^{-1}`:

    .. math::

        \\bar{x}_k = \\Phi_{k,k-1}\\,\\hat{x}_{k-1}

        \\bar{P}_k = \\Phi_{k,k-1}\\,\\hat{P}_{k-1}\\,\\Phi_{k,k-1}^T + Q_k

    where :math:`Q_k` is the discrete process noise covariance (zero when no
    SNC/DMC model is active).

    *Measurement update* — given the prefit residual
    :math:`y_k = z_k - h(\\bar{x}^*_k)`, the measurement Jacobian
    :math:`H_k`, and the measurement noise covariance :math:`R_k`:

    .. math::

        K_k = \\bar{P}_k H_k^T \\bigl(H_k \\bar{P}_k H_k^T + R_k\\bigr)^{-1}

        \\hat{x}_k = \\bar{x}_k + K_k\\,(y_k - H_k\\,\\bar{x}_k)

        \\hat{P}_k = (I - K_k H_k)\\,\\bar{P}_k\\,(I - K_k H_k)^T + K_k R_k K_k^T

    The last line uses the Joseph-stabilized form, which keeps the covariance
    positive definite under rounding better than :math:`(I - K_k H_k)\\bar{P}_k`.
    The result is symmetrized after each update: rounding leaves the product
    slightly asymmetric, and the next gain amplifies the asymmetry when the
    data are much more precise than :math:`\\bar{P}_k` (Doppler).

    **Consider parameters**

    When consider parameters :math:`c` are present, the filter runs a consider
    analysis, the same estimator as :class:`~scarabaeus.SRIF`,
    :class:`~scarabaeus.SRIFB` and :class:`~scarabaeus.LSB` ([1]_, Ch. 6). The
    estimate is conditioned on :math:`c = \\bar{c}`, so the gain uses
    :math:`\\bar{P}_{xx}`, the covariance with the considered states known:

    .. math::

        S_k = H_{x,k}\\,\\bar{P}_{xx,k}\\,H_{x,k}^T + R_k, \\quad
        K_k = \\bar{P}_{xx,k}\\,H_{x,k}^T S_k^{-1}

        \\hat{P}_{xx,k} = (I - K_k H_{x,k})\\,\\bar{P}_{xx,k}\\,
                          (I - K_k H_{x,k})^T + K_k R_k K_k^T

    The sensitivity :math:`S_x` carries the error in :math:`c` into the
    estimate:

    .. math::

        \\bar{S}_{x,k} = (\\Phi_{x}\\,\\hat{S}_{x,k-1} + \\Psi)\\,\\Phi_{c}^{-1},
        \\quad
        \\hat{S}_{x,k} = (I - K_k H_{x,k})\\,\\bar{S}_{x,k} - K_k H_{c,k}

    and the consider covariance is

    .. math::

        \\hat{P}_{c,k} = \\hat{P}_{xx,k} + \\hat{S}_{x,k}\\,P_{cc}\\,\\hat{S}_{x,k}^T,
        \\quad
        \\hat{P}_{xc,k} = \\hat{S}_{x,k}\\,P_{cc}

    At the first epoch and at sequence nodes,
    :math:`S_x = P_{xc} P_{cc}^{-1}` and :math:`P_{xx} = P_c - S_x P_{xc}^T`,
    so an a-priori correlation between :math:`x` and :math:`c` is kept.
    Process noise enters :math:`P_{xx}` only. The state deviation
    :math:`\\hat{x}_k` is **not** updated for consider parameters.

    **Smoother (Rauch–Tung–Striebel)**

    After the forward pass the LKF supports an optional
    Rauch–Tung–Striebel (RTS) backward smoother.  Starting from the last
    epoch :math:`K` and working backwards to :math:`k = 1`:

    .. math::

        G_k = \\hat{P}_k\\,\\Phi_{k+1,k}^T\\,\\bar{P}_{k+1}^{-1}

        \\hat{x}_k^s = \\hat{x}_k + G_k\\,(\\hat{x}_{k+1}^s - \\bar{x}_{k+1})

        \\hat{P}_k^s = \\hat{P}_k
                    + G_k\\,(\\hat{P}_{k+1}^s - \\bar{P}_{k+1})\\,G_k^T

    where :math:`G_k` is the smoother gain, and the superscript :math:`s`
    denotes smoothed quantities.  The smoother uses all measurements — past
    **and** future — to refine the estimate at each epoch, reducing uncertainty
    particularly over coast arcs between tracking passes.

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Propagator defining the reference trajectory and STMs.  A
        :class:`~scarabaeus.MissionSequence` can be used for multi-leg trajectories with
        maneuver events.
    measurements : MeasurementSpec
        Measurement specification object.
    settings : FilterSettings
        Filter configuration including initial covariance and optional process
        noise model.
    traj_name : str, optional
        Name for the trajectory BSP file.  Auto-generated if ``None``.
    traj_dir : str, optional
        Directory for the trajectory BSP file.
    overwrite_traj : bool, optional
        If ``True``, overwrite any existing trajectory file.
    verbose : bool, optional
        If ``False``, silence the printout of the propagation run at
        construction. Defaults to ``True``.

    See Also
    --------
    scarabaeus.SRIF : Square-root information form of the same algorithm,
        with superior numerical conditioning for ill-posed problems.
    scarabaeus.FilterOD : Base class providing trajectory and measurement handling.
    scarabaeus.SolutionOD : Container for the estimated state history and covariance.

    Notes
    -----
    For dense tracking or when superior numerical conditioning is required,
    prefer :class:`~scarabaeus.SRIF`, which delivers the same solution with
    roughly double the effective floating-point precision.  When the data are
    much more precise than the a-priori (:math:`H\\bar{P}H^T/R \\gtrsim 10^8`,
    typical for Doppler), the covariance form can use up most of the float64
    digits, and the LKF covariances may then differ from the SRIF ones by a
    few percent.  When process noise
    is not needed, the batch estimators :class:`~scarabaeus.LSB` and
    :class:`~scarabaeus.SRIFB` are simpler alternatives.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    # The shared history/residual properties (covariance_history,
    # covariance_consider_history, state_deviation_history, prefit_residuals,
    # postfit_residuals) are defined on FilterOD. Only LKF-specific ones below.

    @property
    def kalman_gains(self) -> list:
        """Kalman gain :math:`K_k` at each measurement update.
        Populated when ``settings.output.save_kalman_gains`` is ``True``."""
        return self._kalman_gains

    @kalman_gains.setter
    def kalman_gains(self, value):
        self._kalman_gains = value

    @property
    def innovation_covariances(self) -> list:
        """Innovation covariance :math:`S_k = H_k \\bar{P}_k H_k^T + R_k` at each
        measurement update. With consider parameters it also holds their
        contribution :math:`H_y P_{cc} H_y^T`, :math:`H_y = H_x S_x + H_c`.
        Populated when ``settings.output.save_innovation_covariances``
        or ``settings.output.save_nis_statistics`` is ``True``."""
        return self._innovation_covariances

    @innovation_covariances.setter
    def innovation_covariances(self, value):
        self._innovation_covariances = value

    @property
    def innovations(self) -> list:
        """Pre-update innovation :math:`\\nu_k = y_k - H_k \\bar{x}_k` at each
        measurement update, paired with :attr:`innovation_covariances`. Populated
        when ``settings.output.save_nis_statistics`` is ``True`` (not in covariance
        analysis)."""
        return self._innovations

    @innovations.setter
    def innovations(self, value):
        self._innovations = value

    def __init__(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        settings: FilterSettings,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
        verbose: bool = True,
    ):
        # Initialize base FilterOD class
        super().__init__(
            propagator,
            measurements,
            settings,
            traj_name,
            traj_dir,
            overwrite_traj,
            verbose=verbose,
        )

        # LKF-specific output containers (shared ones live on FilterOD).
        self._kalman_gains = []
        self._innovation_covariances = []
        self._innovations = []

        ## Print
        print("")
        print("=" * 80)
        print("Initializing Linearized Kalman Filter (LKF)")
        print("=" * 80)
        if self.process_noise is not None:
            print(f"Process Noise Model: {self._process_noise_description()}")
        print("=" * 80)

    # -----------------------------------
    #             Methods
    # -----------------------------------

    def _compute(
        self,
        meas_corr: list = None,
        printOutput: bool = True,
        underweighting_factor: float = 1.0,
    ) -> SolutionOD:
        """
        Execute one forward pass of the Linearized Kalman Filter.

        Iterates over all measurement epochs (and any intervening event epochs
        for sequence trajectories), alternating between a **time update** and a
        **measurement update** at each step.

        Parameters
        ----------
        meas_corr : list, optional
            Correlation factor list forwarded to :class:`~scarabaeus.CovarianceMatrix` for
            off-diagonal measurement noise terms.  Defaults to None.
        printOutput : bool, optional
            Print per-epoch progress (norms of prefit/postfit residuals,
            covariance trace, innovation RMS).  Defaults to True.
        underweighting_factor : float, optional
            Multiplicative scale applied to the measurement covariance ``R_k``.
            Values greater than 1 de-weight measurements (useful for filter
            initialisation or robustness tuning).  Defaults to 1.0.

        Returns
        -------
        SolutionOD
            Solution object containing state deviation history, covariance
            history, prefit and postfit residuals, and optional debug quantities
            as configured by :class:`~scarabaeus.OutputSettings`.

        Notes
        -----
        The time update predicts the covariance as

        .. math::

            \\bar{P}_k = \\Phi_{k,k-1}\\, \\hat{P}_{k-1}\\, \\Phi_{k,k-1}^T + Q_k

        where :math:`\\Phi_{k,k-1} = \\Phi_k \\Phi_{k-1}^{-1}` is the one-step
        STM and :math:`Q_k` is the discrete process noise (SNC or DMC, if active).

        The measurement update forms the innovation covariance and the Kalman
        gain,

        .. math::

            S_k = H_x \\bar{P}_k H_x^T + R_k, \\quad
            K_k = \\bar{P}_k H_x^T S_k^{-1}

        State correction:

        .. math::

            \\hat{x}_k = \\bar{x}_k + K_k (y_k - H_x \\bar{x}_k)

        Covariance correction (Joseph stabilised form, then symmetrized):

        .. math::

            \\hat{P}_k = (I - K_k H_x)\\, \\bar{P}_k\\, (I - K_k H_x)^T
                       + K_k R_k K_k^T

        When **consider parameters** are active, the same
        gain uses ``Pxx``, the covariance with the considered states known,
        and the sensitivity ``Sx`` carries their error:
        ``Pc = Pxx + Sx Pcc Sx^T`` (consider analysis, as in the SRIF).
        """

        # Initialize state length if sequence
        if self.flag_sequence is True:
            self.n = self.legs_n[0]

        # Initialize covariance and deviation
        P0 = self.initial_covariance.matrix_without_units()
        dx_bar_0 = self._apriori_deviation(0)
        if self.flag_MTM:
            MTM = self.trajectory._STMs[0].reshape(
                int(np.sqrt(len(self.trajectory._STMs[0]))), -1
            )
            P0 = MTM @ P0 @ MTM.T
            dx_bar_0 = MTM @ dx_bar_0
            self.flag_MTM = False

        # Extract blocks if consider parameters are used
        if self.has_consider:
            Pxx_hat_k1 = P0[np.ix_(self.idx_est, self.idx_est)]
            Pcc_bar_k1 = P0[np.ix_(self.idx_con, self.idx_con)]
        else:
            Pxx_hat_k1 = P0

        dx_hat_k1 = dx_bar_0[self.idx_est] if self.has_consider else dx_bar_0
        STM_k1 = np.eye(self.n)

        # Extract spacecraft times
        measurement_times, event_times, timeline_times = self._build_timeline()

        # Initialize Schmidt matrix Sx from the a-priori covariance, as at the
        # sequence nodes: Pc is the estimated block and Pxx = Pc - Sx Pxc^T the
        # part of it the considered states do not explain. The first
        # measurement is processed by the loop below, like every other one.
        if self.has_consider:
            Pxc_0 = P0[np.ix_(self.idx_est, self.idx_con)]
            Sx_hat_k1 = Pxc_0 @ self._safe_inv(Pcc_bar_k1)
            Pxx_hat_k1 = Pxx_hat_k1 - Sx_hat_k1 @ Pxc_0.T

        # Initialize storage arrays (these will be extracted by SolutionOD)
        self.covariance_history = []
        self.covariance_consider_history = []
        self.state_deviation_history = []
        self.prefit_residuals = []
        self.postfit_residuals = []
        # Forward solution at each sequence node, in the previous leg's definition
        self._node_deviations = []
        self._node_covariances = []

        # Debug storage, one entry per measurement update. Reset on every pass
        # so the histories of earlier iterations do not pile up.
        output = getattr(self.settings, "output", None)
        save_gains = bool(output is not None and output.save_kalman_gains)
        save_nis = bool(output is not None and output.save_nis_statistics)
        save_Sy = save_nis or bool(
            output is not None and output.save_innovation_covariances
        )
        self.kalman_gains = []
        self.innovation_covariances = []
        self.innovations = []
        counter_events = 0
        self.update_flag = True

        # Filter loop along measurement epochs
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
                        f"Event detected at epoch {current_time} [TDB]: re-initializing the filter at this epoch."
                    )
                    self.n = self.legs_n[counter_events]

            #### Extract Measurement Data for Epoch t2 ####
            block = None
            if self.update_flag:
                block = self._epoch_measurement_block(current_time)
                sigmas = block["sigma"]
                partials = block["partials"][:, : self.n]
                residuals = block["residuals"]
                instruments = block["instruments"]
                measurement_types = block["measurement_types"]
                dataset_names = block["dataset_names"]
                t2_all = block["t2"]

            #### Covariance Computations ####
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

            # Compute STM
            if self.flag_node:
                prev_n = self.legs_n[counter_events - 1]
                STM_k_prev_leg = self.trajectory._STMs[counter_events - 1][-1].reshape(
                    prev_n, prev_n
                )
                STM_kk1 = STM_k_prev_leg @ self._safe_inv(STM_k1)
            else:
                STM_kk1 = STM_k @ self._safe_inv(STM_k1)

            # Split STM / H in estimated and considered
            Phi_k, Psi_k, Phi_c_k = self._split_stm_phi_psi(STM_kk1)
            if self.update_flag and not self.flag_node:
                Hx, Hc = self._split_partials_hx_hc(partials)

            # COVARIANCE PREDICTION
            # Time propagation using STM_kk1 only (STM_kk1 already handles node bridging in your logic)
            Pxx_bar_k = Phi_k @ Pxx_hat_k1 @ Phi_k.T
            if self.has_consider:
                # Schmidt matrix time update for dynamic consider
                # Sx_bar maps current consider deviation (at k) to x deviation
                Sx_bar_k = (Phi_k @ Sx_hat_k1 + Psi_k) @ self._safe_inv(Phi_c_k)
                Pcc_bar_k = Phi_c_k @ Pcc_bar_k1 @ Phi_c_k.T

            if self.flag_node:
                # Forward solution at the node in the previous leg's definition,
                # taken before the partition is re-initialized for the new leg.
                # Mapping the final estimate back through the nodes recovers from
                # it the states dropped here. Covariance analysis carries no
                # deviation (dx_hat_k1 stays in the first leg's definition).
                if self.covariance_analysis is False:
                    dx_node_prev = Phi_k @ dx_hat_k1
                    self._node_deviations.append(
                        self._full_leg_deviation(counter_events - 1, dx_node_prev)
                    )
                # Covariance of the errors at the node in the previous leg's
                # definition (considered states always last). With consider
                # states it is the joint one, whose estimated block is the
                # consider-inflated Pc: Pxx leaves out the considered states'
                # contribution, and the matrix it forms with Pxc and Pcc need
                # not be positive definite.
                if self.has_consider:
                    Pxc_bar_k = Sx_bar_k @ Pcc_bar_k
                    Pc_bar_k = Pxx_bar_k + Sx_bar_k @ Pxc_bar_k.T
                    P_node = np.block(
                        [[Pc_bar_k, Pxc_bar_k], [Pxc_bar_k.T, Pcc_bar_k]]
                    )
                else:
                    P_node = Pxx_bar_k
                self._node_covariances.append(P_node)
                P_full = self._map_covariance_definition_node(counter_events, P_node)

                # After mapping you can re-init consider partition safely
                self._init_consider_partition(idx_leg=counter_events)
                if self.update_flag:
                    Hx, Hc = self._split_partials_hx_hc(partials)

                # Apply impulse in covariance from burn
                P_full = STM_k @ P_full @ STM_k.T

                # Back into Schmidt objects in the new definition: Pc is the
                # estimated block and Pxx = Pc - Sx Pxc^T the part of it the
                # considered states do not explain. Considered states augmented
                # here enter uncorrelated, so their Sx is zero.
                Pc_bar_k = P_full[np.ix_(self.idx_est, self.idx_est)]
                Pxx_bar_k = Pc_bar_k
                if self.has_consider:
                    Pcc_bar_k = P_full[np.ix_(self.idx_con, self.idx_con)]
                    Pxc_bar_k = P_full[np.ix_(self.idx_est, self.idx_con)]
                    Sx_bar_k = Pxc_bar_k @ self._safe_inv(Pcc_bar_k)
                    Pxx_bar_k = Pc_bar_k - Sx_bar_k @ Pxc_bar_k.T

            # Add process noise only to x
            if self.SNC_flag or self.DMC_flag:
                # Sized for the estimated states of the current leg (after any
                # node mapping above). The noise is not explained by the
                # considered states: Sx is unchanged.
                Pxx_bar_k += ProcessNoise.combined_covariance_at(
                    self._process_noise_models(), k, Pxx_bar_k.shape[0]
                )

            if self.update_flag:
                # Measurement covariance and partials
                measurement_covariance = self._measurement_covariance(
                    sigmas, current_time, meas_corr, underweighting_factor
                )

                # Kalman gain. With consider parameters it is the gain of the
                # estimate conditioned on c = c_bar (Pxx, not Pc), as in the
                # SRIF, SRIFB and LSB.
                Sy = Hx @ Pxx_bar_k @ Hx.T + measurement_covariance
                K = Pxx_bar_k @ Hx.T @ self._safe_inv(Sy)
                M = np.eye(Hx.shape[1]) - K @ Hx

                if save_Sy:
                    if self.has_consider:
                        # Covariance of the innovation, which the error in c
                        # also enters: Hy = Hx Sx + Hc
                        Hy = Hx @ Sx_bar_k + Hc
                        self.innovation_covariances.append(Sy + Hy @ Pcc_bar_k @ Hy.T)
                    else:
                        self.innovation_covariances.append(Sy)

            # COVARIANCE CORRECTION
            if self.update_flag:
                # Pxx Joseph
                Pxx_hat_k = M @ Pxx_bar_k @ M.T + K @ measurement_covariance @ K.T
                # Rounding leaves the product slightly asymmetric, and the next
                # gain, Pxx Hx^T, amplifies the asymmetry when the data are
                # much more precise than Pxx (Doppler)
                Pxx_hat_k = 0.5 * (Pxx_hat_k + Pxx_hat_k.T)
                if self.has_consider:
                    # Schmidt update for Sx
                    Sx_hat_k = M @ Sx_bar_k - K @ Hc
            else:
                Pxx_hat_k = Pxx_bar_k
                if self.has_consider:
                    Sx_hat_k = Sx_bar_k

            #### State Computations ####
            if self.covariance_analysis is False:
                # STATE PREDICTION
                if self.flag_node:
                    # Cross the node in the FULL leg definition, then come back
                    # to the estimated partition. Slicing `dx_full[:n_est]`
                    # instead silently drops the augmented components whenever
                    # a leg carries consider parameters. The prediction to the
                    # node uses the previous leg's partition (`Phi_k` has been
                    # re-split for the new leg by now).
                    dx_bar_k = self._map_deviation_at_node(
                        counter_events, dx_node_prev
                    )
                else:
                    dx_bar_k = Phi_k @ dx_hat_k1

                # Save prefit residuals if requested
                prefit = residuals.copy()
                if (
                    hasattr(self.settings, "output")
                    and self.settings.output.save_prefit_residuals
                ):
                    self.prefit_residuals.append(
                        [
                            prefit,
                            sigmas,
                            instruments,
                            measurement_types,
                            dataset_names,
                            t2_all,
                        ]
                    )

                # Compute innovation (used for state update)
                if self.update_flag:
                    innovation = residuals - Hx @ dx_bar_k
                    if save_nis:
                        self.innovations.append(innovation)

                # STATE CORRECTION
                if self.update_flag:
                    dx_hat_k = dx_bar_k + K @ innovation

                    if save_gains:
                        self.kalman_gains.append(K)
                else:
                    dx_hat_k = dx_bar_k

                # Save state deviation history
                if self.has_consider:
                    full_dev = np.zeros((self.n, 1))
                    full_dev[self.idx_est, :] = dx_hat_k
                    self.state_deviation_history.append(full_dev)
                else:
                    self.state_deviation_history.append(dx_hat_k)

                # True post-fit residual: residuals - H @ dx_hat_k (after update)
                # At k=0, dx_bar=0 so innovation==prefit; storing the true postfit
                # (I - H@K) @ innovation gives the physically correct value.
                if self.update_flag:
                    postfit = residuals - Hx @ dx_hat_k

                # Save prefit & postfit residuals if requested
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

                # Iterate
                dx_hat_k1 = dx_hat_k

            # Save covariance history (store as matrix, not flattened)
            if self.has_consider:
                Pxc_hat_k = Sx_hat_k @ Pcc_bar_k
                Pc_hat_k = Pxx_hat_k + Sx_hat_k @ Pcc_bar_k @ Sx_hat_k.T
                P_full = np.block([[Pc_hat_k, Pxc_hat_k], [Pxc_hat_k.T, Pcc_bar_k]])
                P_full_no_cons = np.block(
                    [[Pxx_hat_k, Pxc_hat_k], [Pxc_hat_k.T, Pcc_bar_k]]
                )
                self.covariance_history.append(P_full_no_cons)
            else:
                self.covariance_history.append(Pxx_hat_k)

            self.covariance_consider_history.append(
                P_full if self.has_consider else None
            )

            # Iterate
            STM_k1 = STM_k
            Pxx_hat_k1 = Pxx_hat_k
            if self.has_consider:
                Sx_hat_k1 = Sx_hat_k
                Pcc_bar_k1 = Pcc_bar_k

            # Reset flags
            if self.flag_node:
                self.flag_node = False
                self.update_flag = True

            # Print progress
            if printOutput is True:
                epoch = current_time
                progress = (
                    (epoch - measurement_times[0])
                    / (measurement_times[-1] - measurement_times[0])
                    * 100
                )

                trace_P = (
                    np.trace(Pxx_hat_k) if not self.has_consider else np.trace(P_full)
                )
                if self.update_flag:
                    trace_diag_S = np.sqrt(np.trace(np.diag(np.diag(Sy))))
                    mean_sigma = float(np.mean(sigmas))
                else:
                    trace_diag_S = np.nan
                    mean_sigma = np.nan

                if self.covariance_analysis:
                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] "
                        f"tr(P) = {trace_P:.4e}   "
                        f"√tr(S) = {trace_diag_S:.4e}   "
                        f"⟨std⟩ = {mean_sigma:.2e}"
                    )

                else:
                    norm_prefit = np.linalg.norm(residuals)
                    norm_postfit = np.linalg.norm(postfit)
                    update_magnitude = np.linalg.norm(K @ innovation)

                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] "
                        f"‖prefit‖ = {norm_prefit:.4e}   "
                        f"‖postfit‖ = {norm_postfit:.4e}   "
                        f"‖update‖ = {update_magnitude:.4e}   "
                        f"tr(P) = {trace_P:.4e}   "
                        f"√tr(S) = {trace_diag_S:.4e}   "
                        f"⟨std⟩ = {mean_sigma:.2e}"
                    )

        # Post-process residuals if needed
        if self.covariance_analysis is False:
            self.prefit_residuals = self._unwrap_residuals(
                self.prefit_residuals,
                self.measurement_data.dataset_names,
                epoch_attr="prefit_epochs",
            )
            self.postfit_residuals = self._unwrap_residuals(
                self.postfit_residuals,
                self.measurement_data.dataset_names,
                epoch_attr="postfit_epochs",
            )

        # Return SolutionOD object (no longer pass covariance directly)
        return SolutionOD(self)

    def _smoother(self, printOutput: bool = True) -> SolutionOD:
        """
        Rauch-Tung-Striebel (RTS) backward smoother.

        Must be called after :meth:`compute`.  Performs a single backward pass
        through the filtered state and covariance histories to compute smoothed
        estimates that use all available measurements (past and future).

        Parameters
        ----------
        printOutput : bool, optional
            Print per-epoch progress showing pre-fit, post-filter, and
            post-smoother residual norms and covariance trace. Defaults to
            ``True``.

        Returns
        -------
        SolutionOD
            Solution object with smoothed state deviations
            (``deviation_smooth``), smoothed covariance history
            (``covariance_smooth``), and smoothed postfit residuals
            (``postfits_smoother``).

        Raises
        ------
        ValueError
            If consider parameters or sequence mode are active (not supported).

        Notes
        -----
        The backward recursion runs from :math:`k = T-2` down to :math:`k = 0`.
        The one-step-ahead predicted covariance is

        .. math::

            \\bar{P}_{k+1|k} = \\Phi_{k+1,k}\\, \\hat{P}_k\\, \\Phi_{k+1,k}^T + Q_{k+1}

        RTS smoother gain:

        .. math::

            G_k = \\hat{P}_k\\, \\Phi_{k+1,k}^T\\, \\bar{P}_{k+1|k}^{-1}

        Smoothed state and covariance:

        .. math::

            \\tilde{x}_k = \\hat{x}_k + G_k (\\tilde{x}_{k+1} - \\Phi_{k+1,k}\\hat{x}_k)

            \\tilde{P}_k = \\hat{P}_k + G_k (\\tilde{P}_{k+1} - \\bar{P}_{k+1|k}) G_k^T

        Smoothing requires ``compute()`` to have been called first so that
        ``state_deviation_history`` and ``covariance_history`` are populated.
        """
        if self.has_consider:
            raise ValueError(
                "Smoother LKF does not support consider parameters in the current implementation."
            )
        if self.flag_sequence:
            raise ValueError(
                "Smoother LKF does not support sequence mode in the current implementation."
            )

        if printOutput:
            print("")
            print("=" * 80)
            print("Performing LKF Backward Smoothing")
            print("=" * 80)

        # Spacecraft epochs
        spacecraft_times = self.measurement_data.get_spacecraft_times()
        T = len(self.state_deviation_history)
        n = self.n

        # Rebuild filtered histories
        x_filt = [x.copy() for x in self.state_deviation_history]
        P_hist = self.covariance_history  # Already stored as matrices

        # Allocate smoothed arrays
        x_smooth = [None] * T
        P_smooth = [None] * T

        # Initialize last step
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_hist[-1]

        if printOutput:
            print("\n" + "=" * 80)
            print("Starting RTS smoother")
            print("=" * 80)

        # Backward pass
        for k in range(T - 2, -1, -1):
            # STM from k -> k+1
            STM0 = self.trajectory.get_STM(epoch=spacecraft_times[k], idx=k)
            STM1 = self.trajectory.get_STM(epoch=spacecraft_times[k + 1], idx=k + 1)
            Phi = STM1 @ self._safe_inv(STM0)

            # One-step-ahead covariance
            P_bar = Phi @ P_hist[k] @ Phi.T
            if self.SNC_flag or self.DMC_flag:
                P_bar += ProcessNoise.combined_covariance_at(
                    self._process_noise_models(), k + 1, n
                )

            # RTS gain
            G = P_hist[k] @ Phi.T @ self._safe_inv(P_bar)

            # Smooth state and covariance
            x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - Phi @ x_filt[k])
            P_smooth[k] = P_hist[k] + G @ (P_smooth[k + 1] - P_bar) @ G.T

            # Print progress
            if printOutput:
                # Extract information for printout
                block = self.measurement_data.get_combined_for_t2(spacecraft_times[k])
                block = block[spacecraft_times[k]]
                sigmas = block["sigma"]
                partials = block["partials"][:, : self.n]
                residuals = block["residuals"]

                prefit = residuals
                H = partials
                postfilt = prefit - H @ x_filt[k]
                postfitsm = prefit - H @ x_smooth[k]
                epoch = spacecraft_times[k]
                prog = (
                    (epoch - spacecraft_times[0])
                    / (spacecraft_times[-1] - spacecraft_times[0])
                    * 100
                )

                print(
                    f"[{epoch:.2f} TDB | {prog:5.1f}%] "
                    f"‖prefit‖={np.linalg.norm(prefit):.4e}   "
                    f"‖postfit‖={np.linalg.norm(postfilt):.4e}   "
                    f"‖postfit_s‖={np.linalg.norm(postfitsm):.4e}   "
                    f"tr(P_s)={np.trace(P_smooth[k]):.4e}"
                )

        # Rebuild smoothed postfit residuals
        postfit_residuals_sm = []
        for k in range(T):
            block = self.measurement_data.get_combined_for_t2(spacecraft_times[k])
            block = block[spacecraft_times[k]]
            sigmas = block["sigma"]
            partials = block["partials"][:, : self.n]
            residuals = block["residuals"]
            instruments = block["instruments"]
            measurement_types = block["measurement_types"]
            dataset_names = block["dataset_names"]
            t2_all = block["t2"]

            # Compute smoothed postfit residuals
            H = partials
            postfit = residuals - H @ x_smooth[k]
            postfit_residuals_sm.append(
                [
                    postfit,
                    sigmas,
                    instruments,
                    measurement_types,
                    dataset_names,
                    t2_all,
                ]
            )

        # Store smoothed results in filter object for SolutionOD to extract
        self.smoothed_state_deviation = x_smooth
        self.smoothed_covariance = P_smooth  # Store as list of matrices
        self.postfit_residuals_smooth = self._unwrap_residuals(
            postfit_residuals_sm,
            self.measurement_data.dataset_names,
            epoch_attr="postfit_epochs_smooth",
        )

        # Return SolutionOD object (it will extract smoothed data)
        return SolutionOD(self)
