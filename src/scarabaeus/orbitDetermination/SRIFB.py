# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    FilterOD,
    SolutionOD,
    Propagator,
    FilterSettings,
    MissionSequence,
    MeasurementSpec,
)
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import linalg as spln
from scipy.linalg import solve_triangular


#######################################################################
class SRIFB(FilterOD):
    """ Square Root Information Filter — Batch (SRIFB) for spacecraft orbit determination.

    The SRIFB is the **batch** counterpart of the sequential :class:`~scarabaeus.SRIF`.  All
    measurements across all epochs are accumulated at once in the square-root
    information domain without any intermediate time updates, so process noise
    (**SNC/DMC**) is not supported.  It produces the same solution as the
    :class:`~scarabaeus.LSB` normal-equation batch estimator but with superior numerical
    conditioning.

    **Algorithm**

    The SRIFB represents the accumulated information as the pair
    :math:`(R_{xx},\\, z_x)`, where :math:`R_{xx}` is the upper-triangular
    Cholesky factor of the information matrix and :math:`z_x = R_{xx}\\,\\hat{x}_0`.

    At each epoch :math:`t_k`, let :math:`\\tilde{R}_k` be the upper-triangular
    Cholesky factor of the measurement noise covariance :math:`R_k` (so
    :math:`R_k = \\tilde{R}_k^T \\tilde{R}_k`).  The whitened observation equation

    .. math::

        \\tilde{R}_k^{-1} H_k\\,\\Phi_{k,0}\\cdot\\Delta x_0
        \\;=\\;
        \\tilde{R}_k^{-1} y_k

    is appended below the running information block and the combined rectangular
    system is triangularized via a **Householder QR decomposition**:

    .. math::

        \\begin{bmatrix}
            R_{xx} & z_x \\\\
            \\tilde{R}_k^{-1} H_k\\,\\Phi_{k,0} & \\tilde{R}_k^{-1} y_k
        \\end{bmatrix}
        \\xrightarrow{\\;\\text{QR}\\;}
        \\begin{bmatrix}
            \\hat{R}_{xx} & \\hat{z}_x \\\\
            0 & e_k
        \\end{bmatrix}

    where :math:`e_k` collects the normalized postfit residuals for epoch
    :math:`k`.  After all :math:`K` epochs are processed, the state deviation
    at :math:`t_0` and its covariance are recovered as:

    .. math::

        \\hat{x}_0 = \\hat{R}_{xx}^{-1}\\,\\hat{z}_x, \\qquad
        \\hat{P}_0 = \\hat{R}_{xx}^{-1}\\,\\hat{R}_{xx}^{-T}.

    The estimated deviation is then propagated forward to every measurement
    epoch via the stored STMs.

    **Numerical advantage over LSB**

    The LSB forms the normal matrix :math:`M = P_0^{-1} + \\sum_k A_k^T R_k^{-1} A_k`
    explicitly and inverts it, squaring the condition number of the problem.  The
    SRIFB avoids this by working directly with the square root: each QR step is
    an orthogonal transformation that preserves the numerical rank of the
    information block throughout accumulation.

    **Schmidt consider parameters**

    The SRIFB supports the Schmidt consider-parameter extension.  The consider
    Jacobian :math:`H_{c,k}` is appended to each whitened measurement block
    before the QR step, and the resulting off-diagonal information block
    :math:`\\hat{R}_{xc}` encodes the cross-correlation between estimated and
    consider uncertainties.  The posterior covariance is:

    .. math::

        \\hat{P}_0 = \\hat{R}_{xx}^{-1}\\,\\hat{R}_{xx}^{-T}
                  + S_{xc}\\,P_{cc}\\,S_{xc}^T

    where :math:`S_{xc} = -\\hat{R}_{xx}^{-1}\\hat{R}_{xc}` is the consider
    sensitivity matrix.

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

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Defines the reference trajectory and STMs.  A :class:`~scarabaeus.MissionSequence`
        can be used for multi-leg trajectories with maneuver events.
    measurements : MeasurementSpec
        Measurement specification object.
    settings : FilterSettings
        Filter configuration.  ``settings.process_noise`` must be ``None``.
    traj_name : str, optional
        Name for the trajectory BSP file.  Auto-generated if ``None``.
    traj_dir : str, optional
        Directory for the trajectory BSP file.
    overwrite_traj : bool, optional
        If ``True``, overwrite any existing trajectory file.
    verbose : bool, optional
        If ``False``, silence the printout of the propagation run at
        construction. Defaults to ``True``.

    Raises
    ------
    ValueError
        If the initial covariance matrix is not positive definite (required for
        Cholesky factorisation).
    NotImplementedError
        If ``settings.process_noise`` is not ``None`` (process noise is not
        supported in the batch formulation; use :class:`~scarabaeus.SRIF` or :class:`~scarabaeus.LKF`).

    See Also
    --------
    scarabaeus.LSB : Normal-equation batch estimator; same solution, less stable.
    scarabaeus.SRIF : Sequential square-root information filter; supports process noise.
    scarabaeus.FilterOD : Base class providing trajectory and measurement handling.

    Notes
    -----
    Process noise (SNC/DMC) requires intermediate time updates and is therefore
    **not** supported in the batch formulation.  Use :class:`~scarabaeus.SRIF`
    or :class:`~scarabaeus.LKF` when unmodelled accelerations must be compensated
    sequentially.  For process-noise-free problems, SRIFB is the numerically
    preferred alternative to :class:`~scarabaeus.LSB`.

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

    # Batch estimator: sequential process noise is not applicable.
    _supports_process_noise = False

    # The solution is mapped back through the sequence nodes in ``_compute``.
    _maps_back_sequence = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    # The history/residual properties (covariance_history,
    # covariance_consider_history, state_deviation_history, prefit_residuals,
    # postfit_residuals) are inherited from FilterOD.

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
        # parent class initialization
        super().__init__(
            propagator,
            measurements,
            settings,
            traj_name,
            traj_dir,
            overwrite_traj,
            verbose=verbose,
        )

        # Raise error if needed
        ## Initial covariance matrix
        try:
            _ = self._sqrt_info_matrix(self.initial_covariance.matrix_without_units())
        except np.linalg.LinAlgError:
            raise ValueError(
                "The SRIF relies on Cholesky decomposition for computing the square root of the information matrix. Consequently, it is necessary for the initial covariance matrix to be positive definite."
            )

        ## Print
        print("")
        print("=" * 80)
        print("Initializing Sequence Square Root Information Filter Batch (SRIFB)")
        print("=" * 80)

    def _compute(
        self,
        meas_corr: list = None,
        map_back_sequence: bool = True,
        printOutput: bool = True,
    ) -> SolutionOD:
        """
        Computes the batch least-squares solution using the Square Root Information Filter (SRIF).

        Performs batch estimation of the spacecraft state by processing all available
        measurements in a least-squares sense. The formulation relies on the square root of the
        information matrix, leveraging Cholesky decomposition to ensure numerical stability.

        In sequence mode, maps the final leg deviation and covariance back through
        previous trajectory legs, ensuring continuity across estimation segments.
        States a leg drops at a node are recovered from that leg's forward
        solution (see :meth:`FilterOD._map_back_across_node`), so every leg
        reflects the solution of the whole data arc.

        Parameters
        ----------
        meas_corr : list, optional
            List of correction factors applied to the measurement covariance. Defaults to ``None``.

        map_back_sequence : bool, optional
            If ``True``, maps the final leg deviation and covariance back to previous legs.
            If ``False``, each leg keeps the solution of the data up to its end,
            which is not a consistent update of the whole sequence. Defaults to ``True``.

        printOutput : bool, optional
            If ``True``, prints progress information during execution. Defaults to ``True``.

        Returns
        -------
        SolutionOD
            An object containing the computed state deviation history,
            covariance evolution, and postfit residuals.
        """
        ## Ensure assumptions
        if self.flag_sequence:
            self._validate_sequence_batch()

        # Initialize arrays for storing information. Start from the first leg:
        # the last pass left its last leg's size, partition and node flag.
        if self.flag_sequence:
            self.n = self.legs_n[0]
            self.flag_node = False
            self._init_consider_partition()

        # Initialize storage as instance attributes
        self.covariance_history = []
        self.covariance_consider_history = []
        self.state_deviation_history = []
        self.prefit_residuals = []
        self.postfit_residuals = []
        counter_events = 0

        # For sequence
        deviation_0_legs = []
        covariance_0_legs = []
        covariance_consider_0_legs = []
        last_STM_legs = []
        deviation_node_legs = []
        covariance_node_legs = []
        information_node_legs = []

        # Create initial square root information matrix and information vector
        P0 = self.initial_covariance.matrix_without_units()
        if self.has_consider:
            # Keep Pcc prior
            Pcc_bar_0 = P0[np.ix_(self.idx_con, self.idx_con)]
            # Build full prior square-root information and extract blocks
            R0 = self._sqrt_info_matrix((P0))
            # Estimated/considered blocks in the same ordering as your idx_est/idx_con
            Rxx_hat_k = R0[np.ix_(self.idx_est, self.idx_est)]  # (nx,nx)
            Rxc_hat_k = R0[np.ix_(self.idx_est, self.idx_con)]  # (nx,nc)
        else:
            Rxx_hat_k = self._sqrt_info_matrix((P0))
            Rxc_hat_k = None
        # z_x = Rxx_hat_k * dx, starting from the a-priori deviation
        dx_bar_0 = self._apriori_deviation(0)
        zx_hat_k = Rxx_hat_k @ (dx_bar_0[self.idx_est] if self.has_consider else dx_bar_0)

        # Extract spacecraft times
        measurement_times, event_times, timeline_times = self._build_timeline()

        # Iterate through measurements
        print("")
        print("SRIF-Batch: measurements iteration initizialized...")
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
                    self.update_flag = (
                        False  # No update at the node epoch, only mapping
                    )
                    print(
                        f"Event detected at the epoch {current_time} [TDB]: re-initializing the filter at this epoch."
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

                # Measurement Information Whitening
                measurement_covariance = self._measurement_covariance(
                    sigmas, current_time, meas_corr
                )
                measurement_SRI = self._sqrt_info_matrix((measurement_covariance))

                # Whiten
                Hx_w = measurement_SRI @ Hx
                Ax_w_k = Hx_w @ Phi_k

                if not self.covariance_analysis:
                    r_w = measurement_SRI @ residuals

                if self.has_consider:
                    Hc_w = measurement_SRI @ Hc
                    # Consider sensitivity mapped to the same reference
                    Ac_w_k = Hx_w @ Psi_k + Hc_w @ Phi_c_k

                    if self.covariance_analysis:
                        J_k = np.vstack(
                            (
                                np.hstack((Rxx_hat_k, Rxc_hat_k)),
                                np.hstack((Ax_w_k, Ac_w_k)),
                            )
                        )
                    else:
                        J_k = np.vstack(
                            (
                                np.hstack((Rxx_hat_k, Rxc_hat_k, zx_hat_k)),
                                np.hstack((Ax_w_k, Ac_w_k, r_w)),
                            )
                        )

                    _, R_k = np.linalg.qr(J_k)

                    nx, nc = self.n_est, self.n_con
                    Rxx_hat_k = R_k[:nx, :nx]
                    Rxc_hat_k = R_k[:nx, nx : nx + nc]

                    if not self.covariance_analysis:
                        zx_hat_k = R_k[:nx, nx + nc : nx + nc + 1]

                else:
                    if self.covariance_analysis:
                        J_k = np.vstack((Rxx_hat_k, Ax_w_k))
                    else:
                        J_k = np.vstack(
                            (
                                np.hstack((Rxx_hat_k, zx_hat_k)),
                                np.hstack((Ax_w_k, r_w)),
                            )
                        )

                    _, R_k = np.linalg.qr(J_k)

                    n = self.n
                    Rxx_hat_k = R_k[:n, :n]

                    if not self.covariance_analysis:
                        zx_hat_k = R_k[:n, n : n + 1]

            # Print progress
            if printOutput:
                epoch = current_time
                progress = (
                    (epoch - measurement_times[0])
                    / (measurement_times[-1] - measurement_times[0])
                    * 100
                )
                norm_z = np.linalg.norm(zx_hat_k)
                print(f"[{epoch: .2f} TDB | {progress:5.1f}%] " f"‖z‖ = {norm_z:.4e}")

            # If node augment deviaton, covariance and information
            if self.flag_node:
                # Compute deviation and covariance at t0
                prev_n = self.legs_n[counter_events - 1]

                nx_leg = self.n_est if self.has_consider else prev_n
                Sxx_0_leg = solve_triangular(Rxx_hat_k, np.eye(nx_leg), lower=False)

                if self.covariance_analysis:
                    dx_hat_leg = np.zeros((nx_leg, 1))
                else:
                    dx_hat_leg = Sxx_0_leg @ zx_hat_k

                Pxx_hat_leg = Sxx_0_leg @ Sxx_0_leg.T

                if self.has_consider:
                    # build full covariance at the leg reference epoch using (Rxx_hat_k,Rxc_hat_k,Pcc_bar_0)
                    Sxc = -self._safe_inv(Rxx_hat_k) @ Rxc_hat_k
                    Pc = Pxx_hat_leg + Sxc @ Pcc_bar_0 @ Sxc.T
                    Pxc = Sxc @ Pcc_bar_0
                    P_full_0 = np.block([[Pxx_hat_leg, Pxc], [Pxc.T, Pcc_bar_0]])

                    dx_full_0 = np.zeros((prev_n, 1))
                    dx_full_0[self.idx_est, :] = dx_hat_leg

                    P_cons_full_0 = np.block([[Pc, Pxc], [Pxc.T, Pcc_bar_0]])
                else:
                    P_full_0 = Pxx_hat_leg
                    dx_full_0 = dx_hat_leg

                # store leg solution (optional)
                deviation_0_legs.append(dx_full_0)
                covariance_0_legs.append(P_full_0)
                covariance_consider_0_legs.append(
                    P_cons_full_0 if self.has_consider else None
                )

                # propagate to node epoch with last STM of previous leg
                curr_STM_prev_leg = self.trajectory._STMs[counter_events - 1][
                    -1
                ].reshape(prev_n, prev_n)
                last_STM_legs.append(curr_STM_prev_leg)

                dx_node = curr_STM_prev_leg @ dx_full_0
                # Covariance of the errors at the node: with consider states the
                # joint one, whose estimated block is the consider-inflated Pc.
                # P_full_0 pairs the conditional Pxx with Pxc and Pcc and need
                # not be positive definite.
                P_node = (
                    curr_STM_prev_leg
                    @ (P_cons_full_0 if self.has_consider else P_full_0)
                    @ curr_STM_prev_leg.T
                )

                # keep the node solution in the previous definition: mapping back
                # recovers from it the states dropped at this node
                deviation_node_legs.append(dx_node)
                covariance_node_legs.append(P_node)

                # map definition change at node (augment/reset)
                prev_has_consider = self.has_consider
                self._init_consider_partition(idx_leg=counter_events)

                # restart SRIF factors for next leg from FULL information
                if not (prev_has_consider or self.has_consider):
                    # carry the information itself across the node,
                    # R_node = Rxx Phi^-1: re-factoring P_node would invert a
                    # covariance too ill-conditioned after a dense arc
                    R_node = np.linalg.solve(curr_STM_prev_leg.T, Rxx_hat_k.T).T
                    # mapping back recovers the dropped states from it
                    information_node_legs.append(R_node)
                    Rxx_hat_k, zx_hat_k = self._map_information_definition_node(
                        counter_events, R_node, zx_hat_k
                    )
                    Rxc_hat_k = None
                else:
                    information_node_legs.append(None)
                    dx_node = self._map_deviation_definition_node(
                        counter_events, dx_node
                    )
                    P_node = self._map_covariance_definition_node(
                        counter_events, P_node
                    )
                    # update Pcc prior across legs (mapped)
                    Pcc_bar_0 = P_node[np.ix_(self.idx_con, self.idx_con)]

                    # build full information and extract Rxx_hat_k/Rxc_hat_k in new definition
                    R_node = self._sqrt_info_matrix((P_node))
                    Rxx_hat_k = R_node[np.ix_(self.idx_est, self.idx_est)]
                    Rxc_hat_k = R_node[np.ix_(self.idx_est, self.idx_con)]

                    # restart zx_hat_k from mapped deviation
                    zx_hat_k = Rxx_hat_k @ dx_node[self.idx_est]

                zx_hat_k = np.atleast_2d(zx_hat_k).reshape(-1, 1)

                self.flag_node = False
                self.update_flag = True

        # Forward solution at each sequence node, in the previous leg's
        # definition: SolutionOD maps final estimates back through it
        self._node_deviations = deviation_node_legs
        self._node_covariances = covariance_node_legs

        # Deviation and covariance last leg, after processing k measurements
        nx = self.n_est if self.has_consider else self.n
        Sxx = solve_triangular(Rxx_hat_k, np.eye(nx), lower=False)  # inv(Rxx_hat_k)
        Pxx_hat = Sxx @ Sxx.T

        if self.covariance_analysis:
            dx_hat = np.zeros((nx, 1))
        else:
            dx_hat = Sxx @ zx_hat_k

        if self.has_consider:
            Sxc = -self._safe_inv(Rxx_hat_k) @ Rxc_hat_k
            Pc = Pxx_hat + Sxc @ Pcc_bar_0 @ Sxc.T
            Pxc = Sxc @ Pcc_bar_0
            Pcc = Pcc_bar_0

            full_dev = np.zeros((self.n, 1))
            full_dev[self.idx_est, :] = dx_hat

            P_full = np.block([[Pxx_hat, Pxc], [Pxc.T, Pcc]])
            P_cons_full = np.block([[Pc, Pxc], [Pxc.T, Pcc]])

            deviation_0_legs.append(full_dev)
            covariance_0_legs.append(P_full)
            covariance_consider_0_legs.append(P_cons_full)
        else:
            deviation_0_legs.append(dx_hat)
            covariance_0_legs.append(Pxx_hat)
            covariance_consider_0_legs.append(None)

        # Map all the way back last leg information to previous legs
        if self.flag_sequence and map_back_sequence:
            deviation_0_legs_mapped = deviation_0_legs.copy()
            covariance_0_legs_mapped = covariance_0_legs.copy()
            covariance_consider_0_legs_mapped = covariance_consider_0_legs.copy()
            last_leg_deviation = deviation_0_legs[-1]
            # Map the covariance of the errors, joint when the last leg considers
            # states, like the forward node covariances it is conditioned on
            last_leg_covariance = (
                covariance_0_legs[-1]
                if covariance_consider_0_legs[-1] is None
                else covariance_consider_0_legs[-1]
            )
            for idx in range(counter_events, 0, -1):
                # Map deviation and covariance across the node
                (
                    prev_leg_deviation_tnode,
                    prev_leg_covariance_tnode,
                ) = self._map_back_across_node(
                    idx,
                    last_leg_deviation,
                    last_leg_covariance,
                    deviation_node_legs[idx - 1],
                    covariance_node_legs[idx - 1],
                    information_node_legs[idx - 1],
                )

                # ... and from t_node to t_0 of leg idx-1
                (
                    prev_leg_deviation_t0,
                    prev_leg_covariance_t0,
                ) = self._map_node_to_leg_epoch(
                    last_STM_legs[idx - 1],
                    prev_leg_deviation_tnode,
                    prev_leg_covariance_tnode,
                    label=f"Leg {idx - 1} covariance",
                )
                deviation_0_legs_mapped[idx - 1] = prev_leg_deviation_t0
                # stored as the leg's own pair, whichever legs consider states
                (
                    covariance_0_legs_mapped[idx - 1],
                    covariance_consider_0_legs_mapped[idx - 1],
                ) = self._split_consider_covariance(idx - 1, prev_leg_covariance_t0)

                # Update last leg deviation and covariance
                last_leg_deviation = prev_leg_deviation_t0
                last_leg_covariance = prev_leg_covariance_t0

        # Map the state deviation and covariance at epoch to all times k
        counter_events = 0
        if self.flag_sequence:
            self.n = self.legs_n[0]
        print("")
        print("SRIF-Batch: state and covariance mapping initialized...")
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
                block = self._epoch_measurement_block(current_time)
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

            # Map state deviation and covariance
            if self.flag_sequence and map_back_sequence:
                deviation_source = deviation_0_legs_mapped
                covariance_source = covariance_0_legs_mapped
                covariance_consider_source = covariance_consider_0_legs_mapped
            else:
                deviation_source = deviation_0_legs
                covariance_source = covariance_0_legs
                covariance_consider_source = covariance_consider_0_legs
            dx = STM_k @ deviation_source[counter_events]
            # per leg: self.has_consider is the last leg's
            P_cons = covariance_consider_source[counter_events]
            if P_cons is None:
                P = STM_k @ covariance_source[counter_events] @ STM_k.T
            else:
                # Map the joint covariance and condition it at this epoch: the
                # STM couples estimated and considered states, so the mapped
                # conditional block would no longer be conditional
                P, P_cons = self._split_consider_covariance(
                    counter_events, STM_k @ P_cons @ STM_k.T
                )

            # Store results
            self.state_deviation_history.append(dx)
            self.covariance_history.append(P)
            self.covariance_consider_history.append(P_cons)

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
                trace_P = np.trace(P if P_cons is None else P_cons)

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
                self.prefit_residuals,
                self.measurement_data.dataset_names,
                epoch_attr="prefit_epochs",
            )
            self.postfit_residuals = self._unwrap_residuals(
                self.postfit_residuals,
                self.measurement_data.dataset_names,
                epoch_attr="postfit_epochs",
            )

        # Return SolutionOD object
        return SolutionOD(self)
