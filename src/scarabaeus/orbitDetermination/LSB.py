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


#######################################################################
class LSB(FilterOD):
    """ Weighted Batch Least-Squares (LSB) estimator for spacecraft orbit determination.

    The LSB accumulates all measurements simultaneously ("batch") rather than
    processing them one by one as in a sequential filter.  All epochs contribute
    to a single linear system that is solved once to obtain the state deviation
    at the reference epoch :math:`t_0`.

    **Normal equations**

    Define :math:`A_k = H_k\\,\\Phi_{k,0}`, the measurement Jacobian at epoch
    :math:`t_k` mapped back to :math:`t_0` via the STM :math:`\\Phi_{k,0}`, and
    :math:`y_k` the prefit residual.  The weighted normal matrix and right-hand
    side are accumulated over all :math:`K` epochs:

    .. math::

        M = P_0^{-1} + \\sum_{k=1}^{K} A_k^T R_k^{-1} A_k, \\qquad
        b = \\sum_{k=1}^{K} A_k^T R_k^{-1} y_k

    where :math:`P_0^{-1}` is the a-priori information matrix.  The solution and
    its covariance are:

    .. math::

        \\hat{x}_0 = M^{-1} b, \\qquad \\hat{P}_0 = M^{-1}.

    :math:`M` is the **information matrix** :math:`\\Lambda` of the full batch,
    so the LSB solution is the maximum-likelihood estimate under Gaussian noise.

    **Consider parameters (Schmidt extension)**

    When consider parameters are present, the cross-term :math:`M_{xc}` is
    accumulated alongside :math:`M_{xx}` and the posterior covariance is
    inflated by the consider contribution:

    .. math::

        \\hat{P}_0 = M_{xx}^{-1} + S_{xc}\\,P_{cc}\\,S_{xc}^T

    where :math:`S_{xc} = -M_{xx}^{-1} M_{xc}` is the consider sensitivity
    matrix (Tapley et al., Chapter 7).

    **Numerical considerations**

    The LSB solves the normal equations directly via matrix inversion, which
    squares the condition number of the problem.  For ill-conditioned measurement
    geometries, :class:`~scarabaeus.SRIFB` provides the same solution through an orthogonal
    QR decomposition that avoids explicit normal-matrix inversion and is
    numerically preferable.

    **Process noise and stochastic parameters**

    Sequential process noise models such as SNC and DMC require intermediate
    time updates between measurement epochs and are therefore **not**
    supported in the purely batch formulation. Use :class:`~scarabaeus.SRIF` or
    :class:`~scarabaeus.LKF` when unmodelled accelerations must be compensated
    sequentially epoch by epoch.

    A batch analogue can still be constructed through so-called
    *"stochastics"* or *pseudo-epoch states*, where empirical accelerations
    are represented as estimated solve-for parameters within the batch state
    vector. In this approach, first-order Gauss–Markov or piecewise-constant
    stochastic accelerations are parameterized over predefined intervals and
    estimated simultaneously with the orbital state.

    Each stochastic parameter contributes additional columns to the design
    matrix

    .. math::

        A_k = H_k \\Phi_{k,0},

    allowing the batch filter to recover time-varying empirical
    accelerations without performing intermediate sequential covariance
    updates. This formulation is commonly referred to as a
    *pseudo-epoch-state filter* and is preferred when the stochastic signal
    can be adequately represented by deterministic parameters over the data
    arc.

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Defines the reference trajectory and STMs.
    measurements : MeasurementSpec
        Measurement specification object.
    settings : FilterSettings
        Filter configuration.  ``settings.process_noise`` must be ``None``.
    traj_name : str, optional
        Name for the trajectory BSP file.
    traj_dir : str, optional
        Directory for the trajectory BSP file.
    overwrite_traj : bool, optional
        If ``True``, overwrite any existing trajectory file.

    Raises
    ------
    NotImplementedError
        If ``settings.process_noise`` is not ``None`` (process noise is not
        supported in the batch formulation; use :class:`~scarabaeus.LKF` or :class:`~scarabaeus.SRIF`).

    Notes
    -----
    The LSB inverts the normal matrix directly, squaring the condition number.
    For ill-conditioned measurement geometries prefer :class:`~scarabaeus.SRIFB`,
    which produces the same solution via an orthogonal QR decomposition.
    Process noise (SNC/DMC) is not supported; use :class:`~scarabaeus.LKF` or
    :class:`~scarabaeus.SRIF` when unmodelled accelerations must be compensated.

    See Also
    --------
    scarabaeus.SRIFB : Numerically superior batch estimator in the square-root
        information domain; produces the same solution with better conditioning.
    scarabaeus.LKF : Sequential covariance-form filter; supports process noise.
    scarabaeus.FilterOD : Base class providing trajectory and measurement handling.

    References
    ----------
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def covariance_history(self) -> list:
        """Posterior covariance :math:`\\hat{P}_0` propagated to each measurement
        epoch via the trajectory STMs. Populated after calling :meth:`~FilterOD.fit`."""
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
        """State deviation :math:`\\hat{x}_k` propagated to each measurement epoch.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._state_deviation_history

    @state_deviation_history.setter
    def state_deviation_history(self, value):
        self._state_deviation_history = value

    @property
    def prefit_residuals(self) -> dict:
        """Pre-fit residuals :math:`y_k`, keyed by dataset name.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._prefit_residuals

    @prefit_residuals.setter
    def prefit_residuals(self, value):
        self._prefit_residuals = value

    @property
    def postfit_residuals(self) -> dict:
        """Post-fit residuals from the batch solution, keyed by dataset name.
        Populated after calling :meth:`~FilterOD.fit`."""
        return self._postfit_residuals

    @postfit_residuals.setter
    def postfit_residuals(self, value):
        self._postfit_residuals = value

    def __init__(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        settings: FilterSettings,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
    ):
        super().__init__(
            propagator, measurements, settings, traj_name, traj_dir, overwrite_traj
        )

        # Initialize output containers
        self._covariance_history = []
        self._covariance_consider_history = []
        self._state_deviation_history = []
        self._prefit_residuals = {}
        self._postfit_residuals = {}

        # Initialize process noise from settings if provided
        if self.settings.process_noise is not None:
            raise NotImplementedError("Process noise not a feature for LSB.")

        print("")
        print("=" * 80)
        print("Initializing Least Squares Batch (LSB) filter...")
        print("=" * 80)

    def _compute(
        self,
        meas_corr: list = None,
        map_back_sequence: bool = False,
        printOutput: bool = True,
    ) -> SolutionOD:
        """
        Compute the batch weighted least-squares solution.

        Iterates over all measurement epochs to accumulate the normal equations,
        then solves for the state deviation at the reference epoch and propagates
        the result forward to all epochs.

        For sequence trajectories, covariance and deviation are transferred
        across leg boundaries using
        :meth:`FilterOD._map_covariance_definition_node` and
        :meth:`FilterOD._map_deviation_definition_node`.  Optionally, the final
        leg solution is mapped backward through the entire sequence so that
        each leg's reference-epoch solution reflects the global batch estimate.

        Parameters
        ----------
        meas_corr : list, optional
            Correlation factors forwarded to :class:`~scarabaeus.CovarianceMatrix` for
            off-diagonal measurement noise terms.  Defaults to None.
        map_back_sequence : bool, optional
            If True, back-propagate the final leg solution to all previous legs
            through the sequence STMs.  Defaults to False.
        printOutput : bool, optional
            Print per-epoch progress (norm of the normal vector).
            Defaults to True.

        Returns
        -------
        SolutionOD
            Solution object with state deviation history, covariance history,
            and prefit/postfit residuals.
        """

        ## Ensure assumptions
        if self.flag_sequence:
            self._validate_sequence_batch()

        # Initialize arrays for storing information
        if self.flag_sequence:
            self.n = self.legs_n[0]

        # Initialize storage as instance attributes
        self.covariance_history = []
        self.covariance_consider_history = []
        self.state_deviation_history = []
        self.prefit_residuals = []
        self.postfit_residuals = []
        counter_events = 0

        # Initialize normal matrix and normal vector for least squares solution
        P0 = self.initial_covariance.matrix_without_units()
        I0 = self._safe_inv(P0)  # FULL info
        if self.has_consider:
            # estimated-only normal matrix starts from the estimated block prior
            Mxx = I0[np.ix_(self.idx_est, self.idx_est)]
            # coupling comes from the prior; keep as a fixed mapping term
            Mxc = I0[np.ix_(self.idx_est, self.idx_con)]
            # store the consider prior covariance (fixed, never updated)
            Pcc_bar_0 = P0[np.ix_(self.idx_con, self.idx_con)]
        else:
            Mxx = I0
        bx = np.zeros((self.n_est if self.has_consider else self.n, 1))

        # For sequence
        deviation_0_legs = []
        covariance_0_legs = []
        covariance_consider_0_legs = []
        last_STM_legs = []

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

        # Iterate through measurements
        print("")
        print("LS-Batch: measurements iteration initialization...")
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
                    self.update_flag = False
                    print(
                        f"Event detected at the epoch {current_time} [TDB]: re-initializing the filter at this epoch."
                    )
                    # Update size of the state vector (n)
                    self.n = self.legs_n[counter_events]

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

            # If update flag
            if self.update_flag:
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

                # Split STM / H in estimated and considered
                Phi_k, Psi_k, Phi_c_k = self._split_stm_phi_psi(STM_k)
                Hx, Hc = self._split_partials_hx_hc(partials)

                # Measurement covariance matrix
                measurement_covariance = np.array(
                    CovarianceMatrix(
                        sigmas,
                        current_time,
                        from_list=True,
                        corr_factors=meas_corr,
                    ).matrix
                )

                # Inverse of measurement covariance (W matrix)
                W = self._safe_inv(measurement_covariance)

                # Map partials back to the epoch
                Ax = Hx @ Phi_k
                if self.has_consider:
                    Ac = Hx @ Psi_k + Hc @ Phi_c_k

                # Update normal matrix and normal vector
                Mxx += Ax.T @ W @ Ax
                if not self.covariance_analysis:
                    bx += Ax.T @ W @ residuals

                if self.has_consider:
                    # Keep coupling accumulation ONLY (needed for consider covariance inflation)
                    Mxc += Ax.T @ W @ Ac
                    # DO NOT build Mcc or bc from measurements in Tapley's consider

                # Print progress
                if printOutput:
                    epoch = current_time
                    progress = (
                        (epoch - measurement_times[0])
                        / (measurement_times[-1] - measurement_times[0])
                        * 100
                    )
                    norm_nv = np.linalg.norm(bx)
                    print(
                        f"[{epoch: .2f} TDB | {progress:5.1f}%] "
                        f"‖normal_vector‖ = {norm_nv:.4e}"
                    )

            # If node augment deviation and covariance information
            if getattr(self, "flag_node", False):
                # solve current leg at t0 (estimated-only, and keep Schur blocks if consider)
                prev_n = self.legs_n[counter_events - 1]

                if self.has_consider:
                    if self.covariance_analysis:
                        deviation_0_leg = np.zeros((self.n_est, 1))
                    else:
                        deviation_0_leg = np.linalg.solve(Mxx, bx)

                    # Pxx_leg = inv(Mxx) (prefer solve instead of inv in production)
                    Pxx_leg = self._safe_inv(Mxx)

                    # 2) inflate covariance with fixed consider prior
                    Sxc = -Pxx_leg @ Mxc
                    Pxc_leg = Sxc @ Pcc_bar_0
                    Pcc_leg = Pcc_bar_0
                    Pc_leg = Pxx_leg + Sxc @ Pcc_bar_0 @ Sxc.T

                    # Store covariances:
                    # - "no-consider" full block (x,c) but with x part = Pxx_leg (not inflated)
                    P_leg_full = np.block([[Pxx_leg, Pxc_leg], [Pxc_leg.T, Pcc_leg]])

                    # - "consider-inflated" covariance block (x inflated)
                    P_cons_leg_full = np.block(
                        [[Pc_leg, Pxc_leg], [Pxc_leg.T, Pcc_leg]]
                    )

                    # IMPORTANT: do NOT build/append full_dev here if you want it gone for consider
                    # (you can append deviation_0_leg directly if you want estimated-only deviations stored)
                    deviation_0_legs.append(deviation_0_leg)  # estimated-only
                    covariance_0_legs.append(P_leg_full)

                else:
                    if self.covariance_analysis:
                        deviation_0_leg = np.zeros((self.n, 1))
                    else:
                        deviation_0_leg = np.linalg.solve(Mxx, bx)

                    Pxx_leg = self._safe_inv(Mxx)

                    deviation_0_legs.append(deviation_0_leg)
                    covariance_0_legs.append(Pxx_leg)
                covariance_consider_0_legs.append(
                    P_cons_leg_full if self.has_consider else None
                )

                # propagate to node with the last STM of prev leg
                STM_k_prev_leg = self.trajectory._STMs[counter_events - 1][-1].reshape(
                    prev_n, prev_n
                )
                last_STM_legs.append(STM_k_prev_leg)

                if self.has_consider:
                    dx_full_pre = np.vstack(
                        [
                            np.atleast_2d(deviation_0_leg).reshape(-1, 1),
                            np.zeros((self.n_con, 1)),
                        ]
                    )
                    dx_node_pre = STM_k_prev_leg @ dx_full_pre
                    P_node_pre = STM_k_prev_leg @ P_leg_full @ STM_k_prev_leg.T
                else:
                    dx_node_pre = STM_k_prev_leg @ deviation_0_leg
                    P_node_pre = STM_k_prev_leg @ Pxx_leg @ STM_k_prev_leg.T

                # map into next-leg definition
                dx_node = self._map_deviation_definition_node(
                    counter_events, dx_node_pre
                )
                P_node = self._map_covariance_definition_node(
                    counter_events, P_node_pre
                )

                # After mapping you can re-init consider partition safely
                self._init_consider_partition(idx_leg=counter_events)
                n_use = self.n_est if self.has_consider else self.n
                nc = self.n_con

                # re-initialize information-form quantities for next leg
                if self.has_consider:
                    # Reinitialize estimated-only normal matrix from the estimated marginal covariance
                    # BUT: since P_node already includes consider inflation, you should reinitialize from Pxx_hat if you want
                    # purely estimated normal equations. Easiest consistent approach:
                    Pxx0_next = P_node[np.ix_(self.idx_est, self.idx_est)]
                    Mxx = self._safe_inv(Pxx0_next)

                    # Keep the consider prior covariance for next leg from the node mapping
                    Pcc_bar_0 = P_node[np.ix_(self.idx_con, self.idx_con)]

                    # Reinitialize coupling from the node cross-cov
                    Pxc0_next = P_node[np.ix_(self.idx_est, self.idx_con)]
                    Mxc = -Mxx @ Pxc0_next @ self._safe_inv(Pcc_bar_0)

                    bx = Mxx @ dx_node[self.idx_est]
                else:
                    # Reinitialize normal matrix from the node covariance mapping
                    I_node = self._safe_inv(P_node)
                    Mxx = I_node[np.ix_(self.idx_est, self.idx_est)]
                    Mxc = I_node[np.ix_(self.idx_est, self.idx_con)]
                    Pcc_bar_0 = P_node[np.ix_(self.idx_con, self.idx_con)]
                    bx = Mxx @ dx_node[self.idx_est]

                bx = np.atleast_2d(bx).reshape(-1, 1)

                # Reset flags after node
                self.flag_node = False
                self.update_flag = True

        # Solve for final state deviation and covariance
        if self.covariance_analysis:
            dx_hat = np.zeros((self.n_est if self.has_consider else self.n, 1))
        else:
            dx_hat = np.linalg.solve(Mxx, bx)

        if self.has_consider:
            Pxx_hat = self._safe_inv(Mxx)

            # consider inflation with fixed prior
            Sxc = -Pxx_hat @ Mxc
            Pc = Pxx_hat + Sxc @ Pcc_bar_0 @ Sxc.T
            Pxc = Sxc @ Pcc_bar_0
            Pcc = Pcc_bar_0

            full_dev = np.zeros((self.n, 1))
            full_dev[self.idx_est, :] = dx_hat

            P_full = np.block([[Pxx_hat, Pxc], [Pxc.T, Pcc]])
            P_cons_full = np.block([[Pc, Pxc], [Pxc.T, Pcc]])

            deviation_0_legs.append(full_dev)
            covariance_0_legs.append(P_full)
        else:
            Pxx_hat = self._safe_inv(Mxx)

            deviation_0_legs.append(dx_hat)
            covariance_0_legs.append(Pxx_hat)
        covariance_consider_0_legs.append(P_cons_full if self.has_consider else None)

        # Optionally map back final leg information to previous legs
        if self.flag_sequence and map_back_sequence:
            deviation_0_legs_mapped = deviation_0_legs.copy()
            covariance_0_legs_mapped = covariance_0_legs.copy()
            last_leg_deviation = deviation_0_legs[-1]
            last_leg_covariance = covariance_0_legs[-1]
            if self.has_consider:
                covariance_consider_0_legs_mapped = covariance_consider_0_legs.copy()
                last_leg_covariance_consider = covariance_consider_0_legs[-1]
            for idx in range(counter_events, 0, -1):
                # Inverse STM (from t_node to t_0) of leg idx-1
                prev_STM_inv = self._safe_inv(last_STM_legs[idx - 1])

                # Map deviation
                prev_leg_deviation_tnode = self._map_deviation_definition_node(
                    idx, last_leg_deviation, flag_forward=False
                )
                prev_leg_deviation_t0 = prev_STM_inv @ prev_leg_deviation_tnode
                deviation_0_legs_mapped[idx - 1] = prev_leg_deviation_t0

                # Map covariance
                prev_leg_covariance_tnode = self._map_covariance_definition_node(
                    idx, last_leg_covariance, flag_forward=False
                )
                prev_leg_covariance_t0 = (
                    prev_STM_inv @ prev_leg_covariance_tnode @ prev_STM_inv.T
                )
                covariance_0_legs_mapped[idx - 1] = prev_leg_covariance_t0

                # Update last leg deviation and covariance
                last_leg_deviation = prev_leg_deviation_t0
                last_leg_covariance = prev_leg_covariance_t0

                # Consider covariance mapping
                if self.has_consider:
                    prev_leg_covariance_consider_tnode = (
                        self._map_covariance_definition_node(
                            idx, last_leg_covariance_consider, flag_forward=False
                        )
                    )
                    prev_leg_covariance_consider_t0 = (
                        prev_STM_inv
                        @ prev_leg_covariance_consider_tnode
                        @ prev_STM_inv.T
                    )
                    covariance_consider_0_legs_mapped[idx - 1] = (
                        prev_leg_covariance_consider_t0
                    )
                    last_leg_covariance_consider = prev_leg_covariance_consider_t0

        # Map the final leg state deviation and covariance to all times
        counter_events = 0
        if self.flag_sequence:
            self.n = self.legs_n[0]
        print("")
        print("LS-Batch: state and covariance mapping initialization...")
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

            # Split STM / H in estimated and considered
            # Phi_k, Psi_k = self._split_stm_phi_psi(STM_k)
            # Hx, Hc = self._split_partials_hx_hc(partials)

            # Map state deviation and covariance
            if self.flag_sequence and map_back_sequence:
                deviation_source = deviation_0_legs_mapped
                covariance_source = covariance_0_legs_mapped
                if self.has_consider:
                    covariance_consider_source = covariance_consider_0_legs_mapped
            else:
                deviation_source = deviation_0_legs
                covariance_source = covariance_0_legs
                if self.has_consider:
                    covariance_consider_source = covariance_consider_0_legs
            dx = STM_k @ deviation_source[counter_events]
            P = STM_k @ covariance_source[counter_events] @ STM_k.T
            if self.has_consider:
                P_cons = STM_k @ covariance_consider_source[counter_events] @ STM_k.T

            # Store results
            self.state_deviation_history.append(dx)
            self.covariance_history.append(P)
            self.covariance_consider_history.append(
                P_cons if self.has_consider else None
            )

            # Compute/store residuals only when meaningful
            if self.update_flag and not self.covariance_analysis:
                postfit = residuals - partials @ dx
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

            # Print progress
            if printOutput:
                epoch = current_time
                progress = (
                    (epoch - measurement_times[0])
                    / (measurement_times[-1] - measurement_times[0])
                    * 100
                )
                trace_P = np.trace(P if not self.has_consider else P_cons)

                if self.update_flag and not self.covariance_analysis:
                    norm_prefit = np.linalg.norm(residuals)
                    norm_postfit = np.linalg.norm(postfit)
                    mean_sigma = float(np.mean(sigmas))

                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] "
                        f"‖prefit‖={norm_prefit:.4e}   "
                        f"‖postfit‖={norm_postfit:.4e}   "
                        f"tr(P)={trace_P:.4e}   "
                        f"⟨σ⟩={mean_sigma:.2e}"
                    )
                else:
                    print(
                        f"[{epoch:.2f} TDB | {progress:5.1f}%] " f"tr(P)={trace_P:.4e}"
                    )

        # Post-process residuals
        if not self.covariance_analysis:
            self.prefit_residuals = self._unwrap_residuals(
                self.prefit_residuals, self.measurement_data.dataset_names
            )
            self.postfit_residuals = self._unwrap_residuals(
                self.postfit_residuals, self.measurement_data.dataset_names
            )

        # Return SolutionOD object
        return SolutionOD(self)
