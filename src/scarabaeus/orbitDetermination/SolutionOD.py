# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    FilterOD,
    EpochArray,
    OutputSettings,
    RangeIdeal,
    RangeRateIdeal,
    ArrayWUnits,
    Spacecraft,
    Units,
    Propagator,
    ArrayWFrame,
    StateArray,
    StateDefinition,
)
from typing import Optional, Dict, List, Tuple, Union
import scarabaeus.utils.NumpyWrapper as np
import json
import pickle
from pathlib import Path

# ------------------#
#  Generate Units  #
# ------------------#
sec = Units.get_units("sec")


# --------------------#
#  Class Definition  #
# --------------------#
class SolutionOD:
    """Stores and manages the results of an orbit determination process.

    Encapsulates all outputs from a filter including state deviations, covariances,
    residuals, and optional debugging quantities. Storage is controlled by OutputSettings.

    Parameters
    ----------
    filter_obj : FilterOD
        The filter object that produced this solution.
    output_settings : OutputSettings, optional
        Configuration controlling what quantities are stored.

    Attributes
    ----------
    filter : FilterOD
        Reference to the filter object.
    output_settings : OutputSettings
        Output configuration.
    timestamps : np.ndarray
        Epoch timestamps for the solution.
    deviation_est : np.ndarray, optional
        Estimated state deviation wrt the **last iteration's** reference trajectory
        [n_epochs x n_states].
    deviation_cumulative : np.ndarray, optional
        Cumulative state deviation wrt the **first** (initial) reference trajectory,
        accumulated across all iterations [n_epochs x n_states].
        Equal to ``deviation_est`` for single-iteration runs.
    state_est : np.ndarray, optional
        Absolute state estimate (reference + deviation_est) at each measurement
        epoch [n_epochs x n_states].  Layout: [pos_x, pos_y, pos_z, vel_x, vel_y,
        vel_z, param_0, ...].  Computed lazily on first access (SPICE queries).
    deviation_smooth : np.ndarray, optional
        Smoothed state deviation history [n_epochs x n_states].
    covariance_est : list of np.ndarray, optional
        Estimated covariance matrices at each epoch.
    covariance_smooth : list of np.ndarray, optional
        Smoothed covariance matrices (from RTS smoother).
    covariance_consider : list of np.ndarray, optional
        Consider parameter covariance matrices.
    prefits : list, optional
        Pre-fit measurement residuals.
    postfits : list, optional
        Post-fit measurement residuals.
    postfits_smoother : list, optional
        Post-fit residuals for smoothed solution.
    uq_metrics : dict, optional
        Uncertainty quantification metrics (mean, std, skewness, kurtosis, etc.).
    debug : dict, optional
        Debug quantities (Kalman gains, innovation covariances, NIS).

    Notes
    -----
    Storage of each quantity is controlled by the flags in the associated
    :class:`~scarabaeus.OutputSettings` instance.  Quantities not selected are
    left as ``None`` and not computed during the filter run.

    See Also
    --------
    scarabaeus.FilterOD : Base filter class that produces this solution.
    scarabaeus.OutputSettings : Controls which quantities are stored.

    References
    ----------
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        filter_obj: FilterOD,
        output_settings: Optional[OutputSettings] = None,
    ):
        self._filter = filter_obj

        # Use filter's output settings if not provided
        if output_settings is None:
            if hasattr(filter_obj.settings, "output"):
                self.output_settings = filter_obj.settings.output
            else:
                from scarabaeus import OutputSettings

                self.output_settings = OutputSettings()
        else:
            self.output_settings = output_settings

        # Extract timestamps
        self.timestamps = self._extract_timestamps()

        # Initialize all attributes to None
        self.deviation_est = None
        self.deviation_cumulative = None
        self._state_est_cache = None  # backing store for lazy state_est property
        self.deviation_smooth = None
        self.covariance_est = None
        self.covariance_smooth = None
        self.covariance_consider = None
        self.prefits = None
        self.postfits = None
        self.postfits_smoother = None
        self.uq_metrics = None
        self.debug = None
        # Multi-arc global parameter combination result (set by MultiFilterOD).
        # Keys: x_G, P_G, global_param_names, arc_global_abs, n_outer, converged.
        self.multi_arc_global: Optional[dict] = None

        # Store quantities based on output settings
        self._store_solution_data()

    # endregion Constructor #
    # -----------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#

    @property
    def filter(self) -> FilterOD:
        """The filter object used for orbit determination."""
        return self._filter

    @property
    def n_epochs(self) -> int:
        """Number of epochs in the solution."""
        return len(self.timestamps)

    @property
    def n_states(self) -> int:
        """Number of states in the state vector."""
        return self.filter.n

    @property
    def is_covariance_analysis(self) -> bool:
        """Whether this is a covariance analysis."""
        return self.filter.covariance_analysis

    @property
    def state_est(self) -> Optional[np.ndarray]:
        """
        Absolute state estimate at each measurement epoch [n_epochs x n_states].

        Computed lazily on first access (reference nominal + deviation_est at each
        epoch via trajectory SPICE queries) and cached. Returns None if
        ``deviation_est`` is not available or the filter has no trajectory.
        """
        if self._state_est_cache is None and self.deviation_est is not None:
            self._state_est_cache = self._compute_state_est()
        return self._state_est_cache

    @state_est.setter
    def state_est(self, value) -> None:
        """Allow direct assignment (e.g. from deserialize)."""
        self._state_est_cache = value

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def global_std_devs(self) -> np.ndarray:
        """
        Standard deviations of the combined global parameter estimate.

        Only meaningful when this solution was produced by ``MultiFilterOD.fit()``,
        which populates ``multi_arc_global`` with the combined covariance ``P_G``.

        Raises
        ------
        ValueError
            If ``multi_arc_global`` has not been set (single-arc solution).
        """
        if self.multi_arc_global is None:
            raise ValueError(
                "global_std_devs() is only available on solutions returned by "
                "MultiFilterOD.fit() (multi_arc_global is None)."
            )
        return np.sqrt(np.diag(self.multi_arc_global["P_G"]))

    def _extract_timestamps(self) -> np.ndarray:
        """Extract timestamps from measurement data."""
        return np.array(self.filter.measurement_data.get_spacecraft_times())

    def _store_solution_data(self) -> None:
        """
        Store solution data based on output settings.

        Populates attributes according to OutputSettings flags.
        """
        # Covariance analysis mode: only covariances
        if self.is_covariance_analysis:
            if self.output_settings.save_covariance_history:
                self.covariance_est = self.filter.covariance_history
            return

        # Full filter mode: store based on settings

        # Estimated state deviations
        if self.output_settings.save_state_deviation_history:
            if hasattr(self.filter, "state_deviation_history"):
                hist = self.filter.state_deviation_history

                # Fast path: already a numeric ndarray
                if isinstance(hist, np.ndarray) and hist.dtype != object:
                    self.deviation_est = np.atleast_2d(hist)
                else:
                    rows = [np.asarray(x).ravel() for x in hist]
                    widths = np.array([r.size for r in rows], dtype=int)

                    w_max = int(widths.max()) if widths.size else 0

                    # If all same, stack normally
                    if widths.size == 0:
                        self.deviation_est = np.zeros((0, 0), dtype=float)
                    elif np.all(widths == w_max):
                        self.deviation_est = np.vstack(rows).astype(float, copy=False)
                    else:
                        # Right-pad (left-aligned) with zeros: [x0..xk, 0..0]
                        dev = np.zeros((len(rows), w_max), dtype=float)
                        for i, r in enumerate(rows):
                            if r.size:
                                dev[i, : r.size] = r
                        self.deviation_est = dev

        # Cumulative deviation wrt first reference (across all iterations)
        if self.deviation_est is not None:
            cum = getattr(self.filter, "_cumulative_deviation", None)
            if cum is not None:
                self.deviation_cumulative = np.atleast_2d(np.asarray(cum, dtype=float))
            else:
                # Single-iteration run: cumulative == current deviation
                self.deviation_cumulative = self.deviation_est.copy()
            # state_est is computed lazily on first access via the property

        # Estimated covariances
        if self.output_settings.save_covariance_history:
            if hasattr(self.filter, "covariance_history"):
                self.covariance_est = self.filter.covariance_history

        # Pre-fit residuals
        if self.output_settings.save_prefit_residuals:
            if hasattr(self.filter, "prefit_residuals"):
                self.prefits = self.filter.prefit_residuals

        # Post-fit residuals
        if self.output_settings.save_postfit_residuals:
            if hasattr(self.filter, "postfit_residuals"):
                self.postfits = self.filter.postfit_residuals

        # Consider covariances
        if (
            self.output_settings.save_consider_covariance_history
            and self.filter.has_consider
        ):
            if hasattr(self.filter, "covariance_consider_history"):
                self.covariance_consider = self.filter.covariance_consider_history

        # Smoothed solution
        if self.output_settings.save_smoothed_solution:
            if hasattr(self.filter, "smoothed_state_deviation"):
                _ssd = self.filter.smoothed_state_deviation
                if _ssd:
                    self.deviation_smooth = np.array(_ssd)
            if hasattr(self.filter, "smoothed_covariance"):
                self.covariance_smooth = self.filter.smoothed_covariance
            if hasattr(self.filter, "postfit_residuals_smooth"):
                self.postfits_smoother = self.filter.postfit_residuals_smooth

        # UQ metrics (always for particle filters)
        if hasattr(self.filter, "particles"):
            self.uq_metrics = self.UQ_compute()

        # Debug data
        if any(
            [
                self.output_settings.save_kalman_gains,
                self.output_settings.save_innovation_covariances,
                self.output_settings.save_nis_statistics,
            ]
        ):
            self.debug = self._collect_debug_data()

    def _compute_state_est(self) -> Optional[np.ndarray]:
        """
        Compute the absolute state estimate at each measurement epoch.

        Returns an array of shape [n_epochs x n_states] where each row is
        ``nominal_state[k] + deviation_est[k]``, i.e. the true estimate in
        the last iteration's reference frame.

        Returns None if the filter does not expose trajectory state queries.
        """
        traj = getattr(self.filter, "trajectory", None)
        if traj is None or self.deviation_est is None:
            return None

        n = len(self.timestamps)
        n_states = (
            self.deviation_est.shape[1]
            if self.deviation_est.ndim > 1
            else self.deviation_est.shape[0]
        )
        out = np.zeros((n, n_states), dtype=float)

        flag_seq = getattr(self.filter, "flag_sequence", False)

        for k in range(n):
            _ts_k = self.timestamps[k]
            t_k = float(_ts_k.values if hasattr(_ts_k, 'values') else _ts_k)
            ep_k = EpochArray(np.array([t_k]), sys="TDB")

            if flag_seq:
                idx_leg = next(
                    (
                        (
                            i + 1
                            if ep_k[0].times.values == leg[-1]
                            and i != len(self.filter.legs_epochs) - 1
                            else i
                        )
                        for i, leg in enumerate(self.filter.legs_epochs)
                        if (
                            (ep_k[0].times.values in leg)
                            if isinstance(leg, np.ndarray)
                            else (ep_k[0].times.values == leg)
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            try:
                nom = traj.get_state(epoch_input=ep_k[0], idx_leg_input=idx_leg)
            except Exception:
                continue

            dev = self.deviation_est[k].ravel()
            pos_nom = np.asarray(nom["position"].values).ravel()
            vel_nom = np.asarray(nom["velocity"].values).ravel()

            row = np.concatenate([pos_nom + dev[:3], vel_nom + dev[3:6]])

            if nom["parameters"] is not None and len(dev) > 6:
                par_nom = np.asarray(nom["parameters"].values).ravel()
                row = np.concatenate([row, par_nom + dev[6:]])

            out[k, : len(row)] = row

        return out

    def _collect_debug_data(self) -> Dict:
        """
        Collect debug quantities into a dictionary.

        Returns
        -------
        dict
            Dictionary with debug data (Kalman gains, innovation cov, NIS).
        """
        debug_data = {}

        # Kalman gains
        if self.output_settings.save_kalman_gains:
            if hasattr(self.filter, "kalman_gains"):
                debug_data["kalman_gains"] = self.filter.kalman_gains

        # Innovation covariances
        if self.output_settings.save_innovation_covariances:
            if hasattr(self.filter, "innovation_covariances"):
                debug_data["innovation_covariances"] = (
                    self.filter.innovation_covariances
                )

        # NIS statistics
        if self.output_settings.save_nis_statistics:
            nis = self._compute_nis_statistics()
            if nis is not None:
                debug_data["nis_statistics"] = nis

        return debug_data if debug_data else None

    def _compute_nis_statistics(self) -> Optional[np.ndarray]:
        """
        Compute Normalized Innovation Squared (NIS) statistics.

        NIS = r^T * S^(-1) * r

        Returns
        -------
        np.ndarray or None
            NIS values at each epoch, or None if data unavailable.
        """
        if self.postfits is None or self.debug is None:
            return None

        if "innovation_covariances" not in self.debug:
            return None

        innovation_covs = self.debug["innovation_covariances"]

        nis_values = []
        for k, residual_data in enumerate(self.postfits):
            residuals, _, _, _, _, _ = residual_data

            if k < len(innovation_covs):
                try:
                    S_inv = self.filter._safe_inv(innovation_covs[k])
                    nis = float(residuals.T @ S_inv @ residuals)
                    nis_values.append(nis)
                except np.linalg.LinAlgError:
                    nis_values.append(np.nan)
            else:
                nis_values.append(np.nan)

        return np.array(nis_values) if nis_values else None

    def propagate_covariance(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
    ) -> List[np.ndarray]:
        """
        Propagate the estimated covariance to arbitrary epochs using trajectory STMs.

        For epochs inside the measurement arc, uses stored trajectory STMs.
        For epochs outside the arc, re-propagates from the nearest arc boundary.

        Parameters
        ----------
        epochs : EpochArray
            Epochs where you want the propagated covariance.
        use_smoothed : bool, optional
            If True and a smoothed solution is available, uses the smoothed covariance
            history (``covariance_smooth``) instead of the forward-filter history
            (``covariance_est``).  Defaults to True.
        """
        from scarabaeus import Trajectory

        if self.covariance_est is None or len(self.covariance_est) == 0:
            raise ValueError("No covariance history available.")
        if epochs is None or getattr(epochs, "size", 0) == 0:
            return []

        traj = getattr(self.filter, "trajectory", None)
        if traj is None:
            raise ValueError("Filter has no trajectory attached.")

        _dev_source = (
            self.deviation_smooth
            if (use_smoothed and self.deviation_smooth is not None)
            else self.deviation_est
        )

        # -------------------------
        # Helpers
        # -------------------------
        def _epoch_to_float(ep) -> float:
            try:
                return float(ep.times.values)
            except Exception:
                return float(ep)

        def _as_matrix(Phi) -> np.ndarray:
            Phi = np.asarray(Phi, dtype=float)
            if Phi.ndim == 1:
                n = int(np.sqrt(Phi.size))
                Phi = Phi.reshape(n, n)
            return Phi

        def _phi_ratio(Phi_t, Phi_ref) -> np.ndarray:
            Phi_t = _as_matrix(Phi_t)
            Phi_ref = _as_matrix(Phi_ref)
            return Phi_t @ np.linalg.solve(Phi_ref, np.eye(Phi_ref.shape[0]))

        def _get_leg_partition(i_leg):
            """Return (idx_est, idx_con) for leg i_leg from the filter's legs_model."""
            flag_seq = getattr(self.filter, "flag_sequence", False)
            legs_model = getattr(self.filter, "legs_model", None)
            if not flag_seq or legs_model is None or i_leg >= len(legs_model):
                return None, None
            state_def = legs_model[i_leg].state
            idx_est_l, idx_con_l = [], []
            cursor = 0
            for entry in state_def:
                name, dim, est_flag = entry[0], entry[1], entry[2]
                if est_flag == "considered":
                    idx_con_l.extend(range(cursor, cursor + dim))
                else:
                    idx_est_l.extend(range(cursor, cursor + dim))
                cursor += dim
            return np.asarray(idx_est_l, int), np.asarray(idx_con_l, int)

        def _prop(P_ref, Phi_t_ref) -> np.ndarray:
            Phi = _as_matrix(Phi_t_ref)
            P   = np.asarray(P_ref, float)
            n_phi, n_P = Phi.shape[0], P.shape[0]
            if n_phi != n_P:
                raise ValueError(
                    f"STM size ({n_phi}) does not match covariance size ({n_P}). "
                    "The propagator must integrate the full state (estimated + consider) "
                    "so that the STM encodes Phi_c and Psi for all parameter dynamics."
                )
            result = Phi @ P @ Phi.T
            return 0.5 * (result + result.T)

        def _build_epoch_array(t_arr):
            return EpochArray(np.asarray(t_arr, dtype=float), sys="TDB")

        def _get_stm(traj_obj, t, idx):
            return _as_matrix(traj_obj.get_STM(epoch=float(t), idx=idx))

        def _get_stm_seq(traj_obj, t, idx_leg):
            return _as_matrix(
                traj_obj.get_STM_sequence(epoch=float(t), idx_leg=idx_leg)
            )

        def _nearest_hist_idx(t) -> int:
            idx = int(np.argmin(np.abs(t_hist - float(t))))
            return max(0, min(idx, len(t_hist) - 1))

        def _state_dev_at(t_ref) -> np.ndarray:
            idx = int(np.argmin(np.abs(t_hist - float(t_ref))))
            return np.asarray(_dev_source[idx], dtype=float).ravel()

        def _reinitialize_at_estimated_state(t_ref):
            sv = self.filter.propagator.full_state_vector
            dev = _state_dev_at(t_ref)
            sv_est = self.filter._apply_state_deviation(sv, dev)
            self.filter.propagator.reinitialize(state_vector=sv_est)

        def _propagate_from_reference(t_ref, P_ref, t_targets) -> dict:
            t_targets = np.asarray(t_targets, dtype=float)
            if len(t_targets) == 0:
                return {}

            is_backward = bool(np.all(t_targets < float(t_ref)))
            all_t = np.unique(np.concatenate([[float(t_ref)], t_targets]))
            if is_backward:
                all_t = all_t[::-1]  # descending: t_ref first
            ep_arr = _build_epoch_array(all_t)

            # Build estimated state vector at t_ref
            sv = self.filter.propagator.full_state_vector
            dev = _state_dev_at(float(t_ref))
            sv_est = self.filter._apply_state_deviation(sv, dev)

            # Reinitialize with estimated state AND new tspan in one call.
            self.filter.propagator.reinitialize(
                state_vector=sv_est,
                tspan=ep_arr,
                backward=is_backward,
            )
            self.filter.propagator.propagate(display_progress=False)

            # For backward propagation, propagate() flips the output to ascending
            # time order so Trajectory.get_STM works. Mirror that flip here so
            # all_t stays consistent with the trajectory's epoch ordering.
            if is_backward:
                all_t = all_t[::-1]  # ascending: [..., t_ref] (t_ref is last)
                phi_ref_idx = len(all_t) - 1
            else:
                phi_ref_idx = 0

            tmp_traj = Trajectory(
                "tmp_covariance_propagation",
                state_array=self.filter.propagator.propagated_state_array,
            )
            tmp_traj.add_STMs(self.filter.propagator.STM)

            Phi_ref = _get_stm(tmp_traj, float(t_ref), idx=phi_ref_idx)

            result = {}
            for t in t_targets:
                idx_t = int(np.argmin(np.abs(all_t - float(t))))
                Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                result[float(t)] = _prop(
                    np.asarray(P_ref, float), _phi_ratio(Phi_t, Phi_ref)
                )

            return result

        def _try_stm_nonseq(t, P_ref, Phi_ref):
            idx_t = int(np.searchsorted(t_hist, float(t), side="left"))
            idx_t = max(0, min(idx_t, len(t_hist) - 1))
            try:
                Phi_t = _get_stm(traj, float(t), idx=idx_t)
                return _prop(P_ref, _phi_ratio(Phi_t, Phi_ref))
            except ValueError as e:
                if "Estimated interpolation error" in str(e):
                    return None
                raise

        # -------------------------
        # Setup
        # -------------------------
        t_req = np.asarray(
            [_epoch_to_float(epochs[k]) for k in range(epochs.size)], dtype=float
        )
        t_hist = np.asarray(self.timestamps, dtype=float)
        P_hist = (
            self.covariance_smooth
            if (
                use_smoothed
                and self.covariance_smooth is not None
                and len(self.covariance_smooth) > 0
            )
            else self.covariance_est
        )

        # =========================================================
        # NON-SEQUENCE CASE
        # =========================================================
        if not getattr(self.filter, "flag_sequence", False):

            t_arc_start = float(t_hist[0])
            t_arc_end = float(t_hist[-1])

            t_before = t_req[t_req < t_arc_start]
            t_inside = t_req[(t_req >= t_arc_start) & (t_req <= t_arc_end)]
            t_after = t_req[t_req > t_arc_end]

            out_map = {}

            # Inside arc — try stored STMs first, fallback to reprop
            P_ref_inside = np.asarray(P_hist[-1], float)
            Phi_ref_inside = _get_stm(traj, float(t_hist[-1]), idx=len(t_hist) - 1)

            fallback_groups = {}
            for t in t_inside:
                P_t = _try_stm_nonseq(t, P_ref_inside, Phi_ref_inside)
                if P_t is not None:
                    out_map[float(t)] = P_t
                else:
                    idx_near = _nearest_hist_idx(float(t))
                    fallback_groups.setdefault(idx_near, []).append(float(t))

            for idx_ref, t_targets in fallback_groups.items():
                out_map.update(
                    _propagate_from_reference(
                        float(t_hist[idx_ref]),
                        np.asarray(P_hist[idx_ref], float),
                        np.asarray(t_targets, float),
                    )
                )

            # Before arc
            if len(t_before) > 0:
                out_map.update(
                    _propagate_from_reference(
                        t_arc_start, np.asarray(P_hist[0], float), t_before
                    )
                )

            # After arc
            if len(t_after) > 0:
                out_map.update(
                    _propagate_from_reference(
                        t_arc_end, np.asarray(P_hist[-1], float), t_after
                    )
                )

            return [out_map[float(t)] for t in t_req]

        # =========================================================
        # SEQUENCE CASE
        # =========================================================
        legs_epochs = self.filter.legs_epochs
        nlegs = len(legs_epochs)

        def _leg_bounds(i):
            leg = legs_epochs[i]
            if isinstance(leg, np.ndarray):
                return float(leg[0]), float(leg[-1])
            return float(leg), float(leg)

        def _find_leg(t) -> int:
            for i in range(nlegs):
                t0, t1 = _leg_bounds(i)
                if t0 - 1e-12 <= float(t) <= t1 + 1e-12:
                    return i
            if float(t) < float(_leg_bounds(0)[0]):
                return -1
            return nlegs

        # Build the timeline that matches P_hist exactly.
        # covariance_est has one entry per filter timeline step (meas + event epochs);
        # t_hist contains only measurement epochs, so indexing P_hist by t_hist
        # position is off by one for every post-event entry.
        _meas_t = np.asarray(t_hist, dtype=float)
        _evt_t = (
            np.asarray(self.filter.events_epochs, dtype=float)
            if hasattr(self.filter, "events_epochs") and len(self.filter.events_epochs) > 0
            else np.array([])
        )
        _timeline = np.unique(np.concatenate([_meas_t, _evt_t]))

        def _timeline_idx_at_or_before(t: float) -> int:
            """Index into _timeline (and P_hist) of the last entry <= t."""
            j = int(np.searchsorted(_timeline, float(t), side="right") - 1)
            return max(0, min(j, len(P_hist) - 1))

        t_arc_global_start = float(_leg_bounds(0)[0])
        t_arc_global_end = float(_leg_bounds(nlegs - 1)[1])

        t_before_all = t_req[t_req < t_arc_global_start]
        t_after_all = t_req[t_req > t_arc_global_end]
        t_inside_all = t_req[
            (t_req >= t_arc_global_start) & (t_req <= t_arc_global_end)
        ]

        out_map = {}
        fallback_groups = {}

        for t in t_inside_all:
            i_leg = _find_leg(float(t))
            i_leg = max(0, min(i_leg, nlegs - 1))

            # Use the last timeline epoch at or before t as the forward reference.
            # This propagates covariance FORWARD in time so newly-introduced
            # parameters (e.g. dv_man after a burn) correctly start large and
            # shrink as measurements inform them.
            idx_ref = _timeline_idx_at_or_before(float(t))
            t_ref = float(_timeline[idx_ref])
            P_ref = np.asarray(P_hist[idx_ref], float)

            # Exact match: t lands on a known filter epoch — return directly.
            if abs(t_ref - float(t)) < 1e-3:
                out_map[float(t)] = 0.5 * (P_ref + P_ref.T)
                continue

            try:
                Phi_ref = _get_stm_seq(traj, t_ref, i_leg)
                Phi_t = _get_stm_seq(traj, float(t), i_leg)
                # _phi_ratio gives the full n_full × n_full one-step STM, which already
                # encodes Phi_c and Psi for dynamic consider parameters — no idx_est
                # partitioning needed or desired here.
                out_map[float(t)] = _prop(P_ref, _phi_ratio(Phi_t, Phi_ref))
            except ValueError as e:
                if "Estimated interpolation error" in str(e):
                    fallback_groups.setdefault(idx_ref, []).append(float(t))
                else:
                    raise

        for idx_ref, t_targets in fallback_groups.items():
            out_map.update(
                _propagate_from_reference(
                    float(_timeline[idx_ref]),
                    np.asarray(P_hist[idx_ref], float),
                    np.asarray(t_targets, float),
                )
            )

        if len(t_before_all) > 0:
            out_map.update(
                _propagate_from_reference(
                    t_arc_global_start, np.asarray(P_hist[0], float), t_before_all
                )
            )

        if len(t_after_all) > 0:
            out_map.update(
                _propagate_from_reference(
                    t_arc_global_end, np.asarray(P_hist[-1], float), t_after_all
                )
            )

        return [out_map[float(t)] for t in t_req]

    def propagate_state(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
    ) -> List[StateArray]:
        """
        Propagate the estimated state to arbitrary epochs using trajectory STMs.

        For epochs inside the measurement arc, uses stored trajectory STMs.
        For epochs outside the arc, re-propagates from the nearest arc boundary.

        Parameters
        ----------
        epochs : EpochArray
            Epochs where you want the estimated state.

        Returns
        -------
        list of StateArray
            One StateArray per requested epoch.
        """
        from scarabaeus import Trajectory

        if self.deviation_est is None:
            raise ValueError("No state deviation history available.")

        traj = getattr(self.filter, "trajectory", None)
        if traj is None:
            raise ValueError("Filter has no trajectory attached.")

        _dev_source = (
            self.deviation_smooth
            if (use_smoothed and self.deviation_smooth is not None)
            else self.deviation_est
        )

        # -------------------------
        # Helpers
        # -------------------------
        def _epoch_to_float(ep) -> float:
            try:
                return float(ep.times.values)
            except Exception:
                return float(ep)

        def _as_matrix(Phi) -> np.ndarray:
            Phi = np.asarray(Phi, dtype=float)
            if Phi.ndim == 1:
                n = int(np.sqrt(Phi.size))
                Phi = Phi.reshape(n, n)
            return Phi

        def _build_epoch_array(t_arr):
            return EpochArray(np.asarray(t_arr, dtype=float), sys="TDB")

        def _get_stm(traj_obj, t, idx):
            return _as_matrix(traj_obj.get_STM(epoch=float(t), idx=idx))

        def _get_stm_seq(traj_obj, t, idx_leg):
            return _as_matrix(
                traj_obj.get_STM_sequence(epoch=float(t), idx_leg=idx_leg)
            )

        def _nearest_hist_idx(t) -> int:
            idx = int(np.argmin(np.abs(t_hist - float(t))))
            return max(0, min(idx, len(t_hist) - 1))

        def _state_dev_at(t_ref) -> np.ndarray:
            idx = int(np.argmin(np.abs(t_hist - float(t_ref))))
            return np.asarray(_dev_source[idx], dtype=float).ravel()

        def _reprop(t_ref, t_targets):
            """Re-propagate from t_ref to t_targets. Returns (tmp_traj, all_t, is_backward)."""
            t_targets_arr = np.asarray(t_targets, float)
            is_backward = bool(np.all(t_targets_arr < float(t_ref)))
            all_t = np.unique(np.concatenate([[float(t_ref)], t_targets_arr]))
            if is_backward:
                all_t = all_t[::-1]  # descending: t_ref first for backward integration
            ep_arr = _build_epoch_array(all_t)
            sv = self.filter.propagator.full_state_vector
            dev = _state_dev_at(float(t_ref))
            sv_est = self.filter._apply_state_deviation(sv, dev)
            self.filter.propagator.reinitialize(
                state_vector=sv_est,
                tspan=ep_arr,
                backward=is_backward,
            )
            self.filter.propagator.propagate()

            # propagate() flips backward output to ascending order; mirror here
            if is_backward:
                all_t = all_t[::-1]  # ascending: [..., t_ref] (t_ref is last)
            tmp_traj = Trajectory(
                "tmp_state_propagation",
                state_array=self.filter.propagator.propagated_state_array,
            )
            tmp_traj.add_STMs(self.filter.propagator.STM)
            return tmp_traj, all_t, is_backward

        def _nominal_from_propagated(all_t, t) -> dict:
            """Extract nominal state directly from propagated_state_array — no SPICE."""
            idx = int(np.argmin(np.abs(all_t - float(t))))

            sa = self.filter.propagator.propagated_state_array
            pos_nom = None
            vel_nom = None
            par_blocks = []

            for entry in sa.state:
                name, size, _, _, _, awf = entry
                vals = np.asarray(awf.quantity.values)
                v = vals[idx] if vals.ndim == 2 else vals.ravel()[:size]
                if name == "position":
                    pos_nom = np.asarray(v).ravel()
                elif name == "velocity":
                    vel_nom = np.asarray(v).ravel()
                else:
                    par_blocks.append(np.asarray(v).ravel())

            return {
                "position": pos_nom,
                "velocity": vel_nom,
                "parameters": np.concatenate(par_blocks) if par_blocks else None,
            }

        def _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref) -> np.ndarray:
            Phi_t = _as_matrix(Phi_t)
            Phi_ref = _as_matrix(Phi_ref)
            return (
                Phi_t @ np.linalg.solve(Phi_ref, np.eye(Phi_ref.shape[0]))
            ) @ np.asarray(dx_ref, float).ravel()

        def _state_from_dev(t, dx, nominal=None) -> tuple:
            """
            Add deviation dx to nominal state at t.
            If nominal is None, queries SPICE (inside-arc only).
            If nominal is a dict, uses it directly (outside-arc safe).
            """
            if nominal is None:
                ep = _build_epoch_array([float(t)])
                ref = self.filter.trajectory.get_state(ep[0])
                pos_nom = np.asarray(ref["position"].values).ravel()
                vel_nom = np.asarray(ref["velocity"].values).ravel()
                par_nom = (
                    np.asarray(ref["parameters"].values).ravel()
                    if ref.get("parameters") is not None
                    else None
                )
            else:
                pos_nom = nominal["position"]
                vel_nom = nominal["velocity"]
                par_nom = nominal.get("parameters")

            pos = pos_nom + dx[:3]
            vel = vel_nom + dx[3:6]
            par = (par_nom + dx[6:]) if (par_nom is not None and len(dx) > 6) else None
            return pos, vel, par

        def _pack_as_state_array(t, pos, vel, par) -> StateArray:
            """Pack pos/vel/par into a StateArray."""
            ep = _build_epoch_array([float(t)])
            new_components = []
            par_offset = 0
            for entry in self.filter.propagator.full_state_vector.state:
                name, size, est_type, dyn_type, body, value_template = entry
                if name == "position":
                    vals = ArrayWFrame(
                        ArrayWUnits(pos, value_template.quantity.units),
                        value_template.frame,
                    )
                elif name == "velocity":
                    vals = ArrayWFrame(
                        ArrayWUnits(vel, value_template.quantity.units),
                        value_template.frame,
                    )
                else:
                    if par is not None:
                        param_vals = np.asarray(par).ravel()[
                            par_offset : par_offset + size
                        ]
                        par_offset += size
                    else:
                        param_vals = value_template.quantity.values
                    vals = ArrayWFrame(
                        ArrayWUnits(param_vals, value_template.quantity.units),
                        value_template.frame,
                    )
                new_components.append((name, size, est_type, dyn_type, body, vals))

            return StateArray(
                epoch=ep,
                origin=self.filter.propagator.full_state_vector.origin,
                state=StateDefinition.from_components(new_components),
            )

        # -------------------------
        # Setup
        # -------------------------
        t_req = np.asarray(
            [_epoch_to_float(epochs[k]) for k in range(epochs.size)], dtype=float
        )
        t_hist = np.asarray(self.timestamps, dtype=float)

        dx_ref_inside = _dev_source[-1].ravel()

        # =========================================================
        # NON-SEQUENCE CASE
        # =========================================================
        if not getattr(self.filter, "flag_sequence", False):

            t_arc_start = float(t_hist[0])
            t_arc_end = float(t_hist[-1])

            t_before = t_req[t_req < t_arc_start]
            t_inside = t_req[(t_req >= t_arc_start) & (t_req <= t_arc_end)]
            t_after = t_req[t_req > t_arc_end]

            pos_map = {}
            vel_map = {}
            par_map = {}

            Phi_ref_inside = _get_stm(traj, float(t_hist[-1]), idx=len(t_hist) - 1)

            # --- Inside arc: SPICE available ---
            fallback_groups = {}
            for t in t_inside:
                idx_t = int(np.searchsorted(t_hist, float(t), side="left"))
                idx_t = max(0, min(idx_t, len(t_hist) - 1))
                try:
                    Phi_t = _get_stm(traj, float(t), idx=idx_t)
                    dx = _apply_stm_to_dev(Phi_t, Phi_ref_inside, dx_ref_inside)
                    pos, vel, par = _state_from_dev(t, dx)  # uses SPICE
                    pos_map[float(t)] = pos
                    vel_map[float(t)] = vel
                    par_map[float(t)] = par
                except ValueError as e:
                    if "Estimated interpolation error" in str(e):
                        fallback_groups.setdefault(
                            _nearest_hist_idx(float(t)), []
                        ).append(float(t))
                    else:
                        raise

            for idx_ref, t_targets in fallback_groups.items():
                t_ref_f = float(t_hist[idx_ref])
                dx_ref = _dev_source[idx_ref].ravel()
                tmp_traj, all_t, is_backward = _reprop(
                    t_ref_f, np.asarray(t_targets, float)
                )
                phi_ref_idx = len(all_t) - 1 if is_backward else 0
                Phi_ref = _get_stm(tmp_traj, t_ref_f, idx=phi_ref_idx)
                for t in t_targets:
                    idx_t = int(np.argmin(np.abs(all_t - float(t))))
                    Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                    dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                    nom = _nominal_from_propagated(all_t, float(t))
                    pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                    pos_map[float(t)] = pos
                    vel_map[float(t)] = vel
                    par_map[float(t)] = par

            # --- Before arc: no SPICE, use tmp_traj nominal ---
            if len(t_before) > 0:
                dx_ref = _dev_source[0].ravel()
                tmp_traj, all_t, is_backward = _reprop(t_arc_start, t_before)
                phi_ref_idx = len(all_t) - 1 if is_backward else 0
                Phi_ref = _get_stm(tmp_traj, t_arc_start, idx=phi_ref_idx)
                for t in t_before:
                    idx_t = int(np.argmin(np.abs(all_t - float(t))))
                    Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                    dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                    nom = _nominal_from_propagated(all_t, float(t))
                    pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                    pos_map[float(t)] = pos
                    vel_map[float(t)] = vel
                    par_map[float(t)] = par

            # --- After arc: no SPICE, use tmp_traj nominal ---
            if len(t_after) > 0:
                dx_ref = _dev_source[-1].ravel()
                tmp_traj, all_t, is_backward = _reprop(t_arc_end, t_after)
                phi_ref_idx = len(all_t) - 1 if is_backward else 0
                Phi_ref = _get_stm(tmp_traj, t_arc_end, idx=phi_ref_idx)
                for t in t_after:
                    idx_t = int(np.argmin(np.abs(all_t - float(t))))
                    Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                    dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                    nom = _nominal_from_propagated(all_t, float(t))
                    pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                    pos_map[float(t)] = pos
                    vel_map[float(t)] = vel
                    par_map[float(t)] = par

            return [
                _pack_as_state_array(
                    t, pos_map[float(t)], vel_map[float(t)], par_map.get(float(t))
                )
                for t in t_req
            ]

        # =========================================================
        # SEQUENCE CASE
        # =========================================================
        legs_epochs = self.filter.legs_epochs
        nlegs = len(legs_epochs)

        def _leg_bounds(i):
            leg = legs_epochs[i]
            if isinstance(leg, np.ndarray):
                return float(leg[0]), float(leg[-1])
            return float(leg), float(leg)

        def _find_leg(t) -> int:
            for i in range(nlegs):
                t0, t1 = _leg_bounds(i)
                if t0 - 1e-12 <= float(t) <= t1 + 1e-12:
                    return i
            if float(t) < float(_leg_bounds(0)[0]):
                return -1
            return nlegs

        # Build the timeline that matches _dev_source exactly
        # (measurement epochs + event epochs, same as filter timeline).
        _meas_t = np.asarray(t_hist, dtype=float)
        _evt_t = (
            np.asarray(self.filter.events_epochs, dtype=float)
            if hasattr(self.filter, "events_epochs") and len(self.filter.events_epochs) > 0
            else np.array([])
        )
        _timeline = np.unique(np.concatenate([_meas_t, _evt_t]))

        def _timeline_idx_at_or_before(t: float) -> int:
            """Index into _timeline (and _dev_source) of the last entry <= t."""
            j = int(np.searchsorted(_timeline, float(t), side="right") - 1)
            return max(0, min(j, len(_dev_source) - 1))

        t_arc_global_start = float(_leg_bounds(0)[0])
        t_arc_global_end = float(_leg_bounds(nlegs - 1)[1])

        t_before_all = t_req[t_req < t_arc_global_start]
        t_after_all = t_req[t_req > t_arc_global_end]
        t_inside_all = t_req[
            (t_req >= t_arc_global_start) & (t_req <= t_arc_global_end)
        ]

        pos_map = {}
        vel_map = {}
        par_map = {}
        fallback_groups = {}

        # --- Inside arc: SPICE available ---
        for t in t_inside_all:
            i_leg = _find_leg(float(t))
            i_leg = max(0, min(i_leg, nlegs - 1))

            # Use the last timeline epoch at or before t as the forward reference.
            idx_ref = _timeline_idx_at_or_before(float(t))
            t_ref = float(_timeline[idx_ref])
            dx_ref = _dev_source[idx_ref].ravel()

            try:
                Phi_ref = _get_stm_seq(traj, t_ref, i_leg)
                Phi_t = _get_stm_seq(traj, float(t), i_leg)
                dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                pos, vel, par = _state_from_dev(t, dx)  # uses SPICE
                pos_map[float(t)] = pos
                vel_map[float(t)] = vel
                par_map[float(t)] = par
            except ValueError as e:
                if "Estimated interpolation error" in str(e):
                    fallback_groups.setdefault(idx_ref, []).append(float(t))
                else:
                    raise

        for idx_ref, t_targets in fallback_groups.items():
            t_ref_f = float(_timeline[idx_ref])
            dx_ref = _dev_source[idx_ref].ravel()
            tmp_traj, all_t, is_backward = _reprop(
                t_ref_f, np.asarray(t_targets, float)
            )
            phi_ref_idx = len(all_t) - 1 if is_backward else 0
            Phi_ref = _get_stm(tmp_traj, t_ref_f, idx=phi_ref_idx)
            for t in t_targets:
                idx_t = int(np.argmin(np.abs(all_t - float(t))))
                Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                nom = _nominal_from_propagated(all_t, float(t))
                pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                pos_map[float(t)] = pos
                vel_map[float(t)] = vel
                par_map[float(t)] = par

        # --- Before all legs ---
        if len(t_before_all) > 0:
            dx_ref = _dev_source[0].ravel()
            tmp_traj, all_t, is_backward = _reprop(t_arc_global_start, t_before_all)
            phi_ref_idx = len(all_t) - 1 if is_backward else 0
            Phi_ref = _get_stm(tmp_traj, t_arc_global_start, idx=phi_ref_idx)
            for t in t_before_all:
                idx_t = int(np.argmin(np.abs(all_t - float(t))))
                Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                nom = _nominal_from_propagated(all_t, float(t))
                pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                pos_map[float(t)] = pos
                vel_map[float(t)] = vel
                par_map[float(t)] = par

        # --- After all legs ---
        if len(t_after_all) > 0:
            dx_ref = _dev_source[-1].ravel()
            tmp_traj, all_t, is_backward = _reprop(t_arc_global_end, t_after_all)
            phi_ref_idx = len(all_t) - 1 if is_backward else 0
            Phi_ref = _get_stm(tmp_traj, t_arc_global_end, idx=phi_ref_idx)
            for t in t_after_all:
                idx_t = int(np.argmin(np.abs(all_t - float(t))))
                Phi_t = _get_stm(tmp_traj, float(t), idx=idx_t)
                dx = _apply_stm_to_dev(Phi_t, Phi_ref, dx_ref)
                nom = _nominal_from_propagated(all_t, float(t))
                pos, vel, par = _state_from_dev(t, dx, nominal=nom)
                pos_map[float(t)] = pos
                vel_map[float(t)] = vel
                par_map[float(t)] = par

        return [
            _pack_as_state_array(
                t, pos_map[float(t)], vel_map[float(t)], par_map.get(float(t))
            )
            for t in t_req
        ]

    def propagate_state_covariance(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
    ) -> List[tuple]:
        """
        Propagate both the estimated state and covariance to arbitrary epochs.

        Parameters
        ----------
        epochs : EpochArray
            Epochs where you want the estimated state and covariance.
        use_smoothed : bool, optional
            If True and a smoothed solution is available, uses smoothed deviations.
            Defaults to True.

        Returns
        -------
        list of (StateArray, np.ndarray)
            One tuple per requested epoch: (StateArray, P(t)).
        """
        state_list = self.propagate_state(epochs, use_smoothed=use_smoothed)
        cov_list = self.propagate_covariance(epochs, use_smoothed=use_smoothed)
        return list(zip(state_list, cov_list))

    def create_JSON(
        self,
        filepath: Union[str, Path],
    ) -> None:
        """
        Export solution results to JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to output JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        def _residuals_to_list(res_dict):
            if res_dict is None:
                return None
            out = {}
            for name, entries in res_dict.items():
                out[name] = [[float(r), float(s)] for r, s in entries]
            return out

        output_dict = {
            "metadata": {
                "n_epochs": self.n_epochs,
                "n_states": self.n_states,
                "is_covariance_analysis": self.is_covariance_analysis,
                **self.output_settings.metadata,
            },
            "timestamps": self.timestamps.tolist(),
        }

        if self.deviation_est is not None:
            output_dict["deviation_estimated"] = self.deviation_est.tolist()
        if self.deviation_cumulative is not None:
            output_dict["deviation_cumulative"] = self.deviation_cumulative.tolist()
        if self.state_est is not None:
            output_dict["state_estimated"] = self.state_est.tolist()
        if self.deviation_smooth is not None:
            output_dict["deviation_smoothed"] = self.deviation_smooth.tolist()
        if self.covariance_est is not None:
            output_dict["covariance_diagonals"] = [
                np.diag(P).tolist() for P in self.covariance_est
            ]
        if self.covariance_smooth is not None:
            output_dict["covariance_smooth_diagonals"] = [
                np.diag(P).tolist() for P in self.covariance_smooth
            ]
        if self.covariance_consider is not None:
            output_dict["covariance_consider_diagonals"] = [
                np.diag(P).tolist() if P is not None else None
                for P in self.covariance_consider
            ]
        if self.prefits is not None:
            output_dict["prefits"] = _residuals_to_list(self.prefits)
        if self.postfits is not None:
            output_dict["postfits"] = _residuals_to_list(self.postfits)
        if self.postfits_smoother is not None:
            output_dict["postfits_smoother"] = _residuals_to_list(
                self.postfits_smoother
            )
        if self.debug is not None and "nis_statistics" in self.debug:
            output_dict["nis_statistics"] = self.debug["nis_statistics"].tolist()

        with open(filepath, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"[SolutionOD] Saved → {filepath}")

    def UQ_compute(
        self,
        particle_states: Optional[np.ndarray] = None,
        confidence_levels: List[float] = [0.68, 0.95, 0.997],
    ) -> Dict:
        """
        Compute UQ metrics for particle-based filters.

        Computes mean, std, covariances, high-order statistical moments (skewness, kurtosis),
        weighted percentiles, and effective sample size from particle distributions.

        Parameters
        ----------
        particle_states : np.ndarray, optional
            Particle states [n_particles x n_states x n_epochs].
            If None, attempts to extract from filter.
        confidence_levels : list of float, optional
            Confidence levels for uncertainty bounds.

        Returns
        -------
        dict
            UQ metrics including mean, std, covariances, skewness, kurtosis,
            percentiles, and effective sample size.

        Raises
        ------
        ValueError
            If particle data is unavailable.
        """
        if particle_states is None:
            if hasattr(self.filter, "particles"):
                particle_states = self.filter.particles
            else:
                raise ValueError("No particle data available for UQ computation")

        n_particles, n_states, n_epochs = particle_states.shape

        uq_results = {
            "n_particles": n_particles,
            "n_epochs": n_epochs,
            "confidence_levels": confidence_levels,
        }

        # Initialize arrays
        means = np.zeros((n_epochs, n_states))
        stds = np.zeros((n_epochs, n_states))
        skewness = np.zeros((n_epochs, n_states))
        kurtosis = np.zeros((n_epochs, n_states))
        covariances = []
        percentiles = {
            level: np.zeros((n_epochs, n_states, 2)) for level in confidence_levels
        }

        # Extract weights if available
        weights = None
        if hasattr(self.filter, "weights"):
            weights = self.filter.weights

        for k in range(n_epochs):
            particles_k = particle_states[:, :, k]

            # Weights for this epoch
            w_k = (
                weights[:, k]
                if weights is not None
                else np.ones(n_particles) / n_particles
            )
            w_k = w_k / np.sum(w_k)

            # Mean, std, covariance
            mean_k = np.average(particles_k, weights=w_k, axis=0)
            var_k = np.average((particles_k - mean_k) ** 2, weights=w_k, axis=0)
            std_k = np.sqrt(var_k)
            means[k] = mean_k
            stds[k] = std_k
            covariances.append(np.cov(particles_k.T))

            # High-order moments
            for i in range(n_states):
                if std_k[i] > 1e-12:
                    z = (particles_k[:, i] - mean_k[i]) / std_k[i]
                    skewness[k, i] = np.average(z**3, weights=w_k)
                    kurtosis[k, i] = np.average(z**4, weights=w_k) - 3.0

            # Weighted percentiles
            for level in confidence_levels:
                lower_pct = (1 - level) / 2 * 100
                upper_pct = (1 + level) / 2 * 100
                for i in range(n_states):
                    sorted_idx = np.argsort(particles_k[:, i])
                    sorted_particles = particles_k[sorted_idx, i]
                    cumsum_weights = np.cumsum(w_k[sorted_idx])
                    lower_idx = min(
                        np.searchsorted(cumsum_weights, lower_pct / 100.0),
                        n_particles - 1,
                    )
                    upper_idx = min(
                        np.searchsorted(cumsum_weights, upper_pct / 100.0),
                        n_particles - 1,
                    )
                    percentiles[level][k, i, 0] = sorted_particles[lower_idx]
                    percentiles[level][k, i, 1] = sorted_particles[upper_idx]

        uq_results["mean"] = means
        uq_results["std"] = stds
        uq_results["covariances"] = covariances
        uq_results["skewness"] = skewness
        uq_results["kurtosis"] = kurtosis
        uq_results["percentiles"] = percentiles

        # Effective sample size
        if weights is not None:
            ess = np.zeros(n_epochs)
            for k in range(n_epochs):
                ess[k] = 1.0 / np.sum(weights[:, k] ** 2)
            uq_results["effective_sample_size"] = ess

        return uq_results

    def print_stats(
        self,
        verbose: bool = True,
    ) -> None:
        """
        Print statistical summary of the OD solution.

        Parameters
        ----------
        verbose : bool, optional
            Print detailed statistics. Defaults to True.
        """
        print("=" * 70)
        print("ORBIT DETERMINATION SOLUTION STATISTICS")
        print("=" * 70)

        print(f"\nSolution Overview:")
        print(f"  Number of epochs:        {self.n_epochs}")
        print(f"  Number of states:        {self.n_states}")
        print(f"  Covariance analysis:     {self.is_covariance_analysis}")

        # UQ Metrics
        if self.uq_metrics is not None:
            print(f"\nResidual Statistics:")
            print(f"  RMS:                     {self.uq_metrics['rms']:.6e}")
            print(f"  Mean:                    {self.uq_metrics['mean']:.6e}")
            print(f"  Std Dev:                 {self.uq_metrics['std']:.6e}")
            print(f"  Min:                     {self.uq_metrics['min']:.6e}")
            print(f"  Max:                     {self.uq_metrics['max']:.6e}")
            print(f"  Skewness:                {self.uq_metrics['skewness']:.4f}")
            print(f"  Kurtosis:                {self.uq_metrics['kurtosis']:.4f}")
            print(f"  Number of residuals:     {self.uq_metrics['n_residuals']}")

        # Chi-squared test
        if self.postfits is not None and verbose:
            n_measurements = sum(len(r[0]) for r in self.postfits)
            degrees_of_freedom = n_measurements - self.n_states

            chi_squared = sum(np.sum(r[0] ** 2) for r in self.postfits)
            normalized_chi_squared = chi_squared / degrees_of_freedom

            print(f"\nChi-Squared Test:")
            print(f"  Chi-squared value:       {chi_squared:.4f}")
            print(f"  Degrees of freedom:      {degrees_of_freedom}")
            print(f"  Normalized chi-squared:  {normalized_chi_squared:.4f}")

            # Rule of thumb: normalized chi-squared should be close to 1
            if normalized_chi_squared < 0.5:
                print(
                    f"  Status:                  Filter may be over-confident (χ²/dof < 0.5)"
                )
            elif normalized_chi_squared > 2.0:
                print(
                    f"  Status:                  Filter may be under-confident (χ²/dof > 2.0)"
                )
            else:
                print(f"  Status:                  Filter appears well-tuned")

        # NIS statistics
        if self.debug is not None and "nis_statistics" in self.debug:
            nis = self.debug["nis_statistics"]
            nis_clean = nis[~np.isnan(nis)]
            if len(nis_clean) > 0:
                print(f"\nNormalized Innovation Squared (NIS):")
                print(f"  Mean NIS:                {np.mean(nis_clean):.4f}")
                print(f"  Std NIS:                 {np.std(nis_clean):.4f}")
                print(f"  Min NIS:                 {np.min(nis_clean):.4f}")
                print(f"  Max NIS:                 {np.max(nis_clean):.4f}")

        # Covariance information
        if self.covariance_est is not None and verbose:
            print(f"\nCovariance Information:")

            # Position uncertainty (1-sigma)
            initial_pos_std = np.sqrt(np.diag(self.covariance_est[0])[:3])
            final_pos_std = np.sqrt(np.diag(self.covariance_est[-1])[:3])

            print(f"  Initial position uncertainty (1σ):")
            print(f"    X: {initial_pos_std[0]:.6e} km")
            print(f"    Y: {initial_pos_std[1]:.6e} km")
            print(f"    Z: {initial_pos_std[2]:.6e} km")
            print(f"    RSS: {np.linalg.norm(initial_pos_std):.6e} km")

            print(f"  Final position uncertainty (1σ):")
            print(f"    X: {final_pos_std[0]:.6e} km")
            print(f"    Y: {final_pos_std[1]:.6e} km")
            print(f"    Z: {final_pos_std[2]:.6e} km")
            print(f"    RSS: {np.linalg.norm(final_pos_std):.6e} km")

            # Velocity uncertainty (1-sigma)
            initial_vel_std = np.sqrt(np.diag(self.covariance_est[0])[3:6])
            final_vel_std = np.sqrt(np.diag(self.covariance_est[-1])[3:6])

            print(f"  Initial velocity uncertainty (1σ):")
            print(f"    VX: {initial_vel_std[0]:.6e} km/s")
            print(f"    VY: {initial_vel_std[1]:.6e} km/s")
            print(f"    VZ: {initial_vel_std[2]:.6e} km/s")
            print(f"    RSS: {np.linalg.norm(initial_vel_std):.6e} km/s")

            print(f"  Final velocity uncertainty (1σ):")
            print(f"    VX: {final_vel_std[0]:.6e} km/s")
            print(f"    VY: {final_vel_std[1]:.6e} km/s")
            print(f"    VZ: {final_vel_std[2]:.6e} km/s")
            print(f"    RSS: {np.linalg.norm(final_vel_std):.6e} km/s")

        # Smoothed solution info
        if self.deviation_smooth is not None:
            print(f"\nSmoothed Solution:")
            print(f"  Smoothed solution available: Yes")
            if self.covariance_smooth is not None:
                smoothed_pos_std = np.sqrt(np.diag(self.covariance_smooth[-1])[:3])
                print(
                    f"  Final smoothed position uncertainty (1σ): {np.linalg.norm(smoothed_pos_std):.6e} km"
                )

        print("=" * 70)

    def serialize(
        self,
        filepath: Union[str, Path],
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """
        Serialize SolutionOD object to binary format using pickle.

        Parameters
        ----------
        filepath : str or Path
            Path to output pickle file.
        protocol : int, optional
            Pickle protocol version. Defaults to highest available.

        Notes
        -----
        The filter object reference is NOT serialized to avoid circular dependencies
        and large file sizes. Only solution data is preserved.
        """
        filepath = Path(filepath)

        # Create a copy of the object's state without the filter reference
        state_dict = {
            "output_settings": self.output_settings,
            "timestamps": self.timestamps,
            "deviation_est": self.deviation_est,
            "deviation_cumulative": self.deviation_cumulative,
            "state_est": self._state_est_cache,
            "deviation_smooth": self.deviation_smooth,
            "covariance_est": self.covariance_est,
            "covariance_smooth": self.covariance_smooth,
            "covariance_consider": self.covariance_consider,
            "prefits": self.prefits,
            "postfits": self.postfits,
            "postfits_smoother": self.postfits_smoother,
            "uq_metrics": self.uq_metrics,
            "debug": self.debug,
            "n_states": self.n_states,
            "is_covariance_analysis": self.is_covariance_analysis,
        }

        # Serialize to file
        with open(filepath, "wb") as f:
            pickle.dump(state_dict, f, protocol=protocol)

        print(f"Solution serialized to {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")

    @classmethod
    def deserialize(
        cls,
        filepath: Union[str, Path],
        filter_obj: Optional[FilterOD] = None,
    ) -> "SolutionOD":
        """
        Deserialize SolutionOD object from binary file.

        Parameters
        ----------
        filepath : str or Path
            Path to pickle file.
        filter_obj : FilterOD, optional
            Filter object to associate with solution. If None, creates placeholder.

        Returns
        -------
        SolutionOD
            Deserialized solution object.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load state dictionary
        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)

        # Create a minimal filter object if none provided
        if filter_obj is None:
            # Create placeholder - user will need to provide actual filter for full functionality
            class PlaceholderFilter:
                def __init__(self, n_states, is_cov_analysis):
                    self.n = n_states
                    self.covariance_analysis = is_cov_analysis
                    self.settings = type(
                        "obj", (object,), {"output": state_dict["output_settings"]}
                    )()

            filter_obj = PlaceholderFilter(
                state_dict["n_states"], state_dict["is_covariance_analysis"]
            )

        # Create solution object
        solution = cls.__new__(cls)
        solution._filter = filter_obj
        solution.output_settings = state_dict["output_settings"]
        solution.timestamps = state_dict["timestamps"]
        solution.deviation_est = state_dict["deviation_est"]
        solution.deviation_cumulative = state_dict.get("deviation_cumulative")
        solution.state_est = state_dict.get("state_est")
        solution.deviation_smooth = state_dict["deviation_smooth"]
        solution.covariance_est = state_dict["covariance_est"]
        solution.covariance_smooth = state_dict["covariance_smooth"]
        solution.covariance_consider = state_dict["covariance_consider"]
        solution.prefits = state_dict["prefits"]
        solution.postfits = state_dict["postfits"]
        solution.postfits_smoother = state_dict["postfits_smoother"]
        solution.uq_metrics = state_dict["uq_metrics"]
        solution.debug = state_dict["debug"]

        print(f"Solution deserialized from {filepath}")
        return solution

    def estimated_trajectory(self, epochs: EpochArray, use_smoothed: bool = True):
        """
        Computes the estimated trajectory (position, velocity, parameters) at filter
        measurement epochs.

        This method is only valid at the epochs where the filter processed measurements
        (i.e. the epochs stored in ``solution.timestamps``).  Passing any other epoch
        raises a ``ValueError`` — use :meth:`propagate_state` for arbitrary epochs,
        including epochs outside the measurement arc.

        Parameters
        ----------
        epochs : EpochArray
            The measurement epochs at which to evaluate the estimated trajectory.
            Every epoch must match an entry in ``solution.timestamps`` to within 1 s.
        use_smoothed : bool, optional
            If True and a smoothed solution is available, use the smoothed deviations;
            otherwise use the filtered deviations.  Defaults to True.

        Returns
        -------
        estimated_pos : numpy.ndarray, shape (n, 3)
            Estimated position at each epoch [km].
        estimated_vel : numpy.ndarray, shape (n, 3)
            Estimated velocity at each epoch [km/s].
        estimated_params : numpy.ndarray or list or None
            Estimated parameters at each epoch, or ``None`` if no parameters exist.

        Raises
        ------
        ValueError
            If any requested epoch does not correspond to a filter measurement epoch.
            Use :meth:`propagate_state` for arbitrary epochs.
        """
        # ------------------------------------------------------------------
        # 1. Validate: every requested epoch must match a measurement epoch.
        # ------------------------------------------------------------------
        t_hist = np.asarray(self.timestamps, dtype=float)
        n = epochs.size
        _TOL_SEC = 1.0  # 1-second tolerance — tighter than any realistic cadence

        idx_devs = []
        bad_indices = []
        for k in range(n):
            t_k = float(epochs[k].times.values)
            nearest = int(np.argmin(np.abs(t_hist - t_k)))
            if abs(t_hist[nearest] - t_k) > _TOL_SEC:
                bad_indices.append(k)
            else:
                idx_devs.append(nearest)

        if bad_indices:
            raise ValueError(
                f"estimated_trajectory() received {len(bad_indices)} epoch(s) that do "
                f"not match any filter measurement epoch (input indices: {bad_indices}). "
                "This method only works at the epochs where the filter processed "
                "measurements. For arbitrary epochs — including propagation to tf — "
                "use solution.propagate_state(epochs) instead."
            )

        # ------------------------------------------------------------------
        # 2. Select deviation source.
        # ------------------------------------------------------------------
        if use_smoothed and self.deviation_smooth is not None:
            deviation_source = self.deviation_smooth
        else:
            deviation_source = self.deviation_est

        # ------------------------------------------------------------------
        # 3. Allocate output arrays.
        # ------------------------------------------------------------------
        estimated_position = np.zeros(shape=(n, 3))
        estimated_velocity = np.zeros(shape=(n, 3))
        if self.filter.trajectory.parameters is not None:
            if not self.filter.flag_sequence:
                shape = (
                    (n, 1)
                    if len(self.filter.trajectory.parameters.values.shape) == 1
                    else (n, self.filter.trajectory.parameters.values.shape[1])
                )
                estimated_parameters = np.zeros(shape=shape)
            else:
                estimated_parameters = [
                    [
                        np.zeros(shape=(self.filter.legs_n[i] - 6))
                        for _ in range(len(self.filter.legs_epochs[i]))
                    ]
                    for i in range(len(self.filter.trajectory.parameters))
                ]
        else:
            estimated_parameters = None

        # ------------------------------------------------------------------
        # 4. Compute estimated state at each epoch.
        # ------------------------------------------------------------------
        for k in range(n):
            idx_dev = idx_devs[k]  # index into deviation_source

            if self.filter.flag_sequence is True:
                idx_leg = next(
                    (
                        (
                            i + 1
                            if epochs[k].times.values == leg[-1]
                            and i != len(self.filter.legs_epochs) - 1
                            else i
                        )
                        for i, leg in enumerate(self.filter.legs_epochs)
                        if (
                            (epochs[k].times.values in leg)
                            if isinstance(leg, np.ndarray)
                            else (epochs[k].times.values == leg)
                        )
                    ),
                    None,
                )
            else:
                idx_leg = None

            # Nominal state from SPICE
            state_nominal = self.filter.trajectory.get_state(
                epoch_input=epochs[k], idx_leg_input=idx_leg
            )

            estimated_position[k] = (
                state_nominal["position"].values
                + deviation_source[idx_dev][0:3].flatten()
            )
            estimated_velocity[k] = (
                state_nominal["velocity"].values
                + deviation_source[idx_dev][3:6].flatten()
            )

            if state_nominal["parameters"] is not None:
                parameters_nominal = state_nominal["parameters"].values
                parameters_deviation = deviation_source[idx_dev][6:].flatten()

                if self.filter.flag_sequence is True:
                    if idx_leg is not None:
                        # k_offset = total measurement epochs in all previous legs;
                        # idx_dev - k_offset gives the within-leg measurement index.
                        k_offset = (
                            sum(
                                len(stm)
                                for stm in self.filter.trajectory._STMs[:idx_leg]
                            )
                            - idx_leg
                        )
                        estimated_parameters[idx_leg][idx_dev - k_offset] = (
                            parameters_nominal + parameters_deviation
                        )
                else:
                    estimated_parameters[k] = parameters_nominal + parameters_deviation

        return estimated_position, estimated_velocity, estimated_parameters

    def map_state_deviation_to_epoch(
        self, map_back_sequence: bool = False, use_smoothed: bool = True
    ):
        """
        Maps the state deviation to the initial epoch using the state transition matrix.

        If smoothed solution is available, simply returns the first smoothed deviation
        (which is already at the initial epoch). Otherwise, maps the filtered deviation
        backward through the STMs.

        Parameters
        ----------
        map_back_sequence : bool, optional
            If True, maps deviations back through the entire sequence;
            otherwise, maps only the last leg. Only used when smoothed solution
            is not available. Defaults to True.
        use_smoothed : bool, optional
            If True and smoothed solution is available, uses the first smoothed deviation.
            Otherwise maps filtered deviations backward. Defaults to True.

        Returns
        -------
        mapped : numpy.ndarray or list of numpy.ndarray
            The mapped state deviation(s) at the initial epoch.
            For sequences, returns a list with one deviation per leg if not mapping back.
            Otherwise returns a single deviation vector.
        """
        # If smoothed solution is available and requested, use the first smoothed
        # deviation. It lives at the first measurement epoch, which may be later
        # than t0, so map it back via Phi(t_first_meas, t0)^-1 (identity when they
        # coincide).
        if use_smoothed and self.deviation_smooth is not None:
            if not self.filter.flag_sequence:
                STM_t1_t0 = self.filter.trajectory.get_STM(
                    epoch=float(self.timestamps[0]), idx=None
                )
                return self.filter._safe_inv(STM_t1_t0) @ self.deviation_smooth[0]
            return self.deviation_smooth[0]

        deviation_source = self.deviation_est

        # Guard: if all measurements were rejected (e.g. chi2 editing), deviation is empty
        if deviation_source is None or deviation_source.size == 0:
            n_states = getattr(self.filter, "n_states", 6)
            return np.zeros(n_states)

        # -------------------------
        # Non-sequence: unchanged
        # -------------------------
        if not self.filter.flag_sequence:
            # deviation_source[-1] lives at the last measurement epoch, which may
            # precede the trajectory end when the reference is propagated with
            # padding, so take the STM at that epoch (not STMs_timestamp[-1]).
            STM_tend_t0 = self.filter.trajectory.get_STM(
                epoch=float(self.timestamps[-1]), idx=None
            )
            return self.filter._safe_inv(STM_tend_t0) @ deviation_source[-1]

        # -------------------------
        # Sequence mode
        # -------------------------
        nlegs = len(self.filter.legs_epochs)

        # 1) Leg-end STM for each leg, Phi_leg(leg_end, leg_start), using idx_leg (no
        #    direct _STMs access). Only valid for deviations that live at the leg-end
        #    (node) epoch; deviations at a leg's last measurement epoch use stm_at_dev.
        last_STMs_legs = [
            self.filter.trajectory.get_STM_sequence(
                epoch=float(self.filter.legs_epochs[i][-1]),
                idx_leg=i,  # crucial
            )
            for i in range(nlegs)
        ]

        # 2) Build the same "time axis" the filter iterated over.
        #    The SRIFB second pass iterates over timeline_times = unique(meas_times ∪ event_times).
        #    deviation_source[k] therefore corresponds to timeline_times[k], NOT to a
        #    measurement-only grid.  We must reconstruct the same timeline here so that
        #    idx_leg_end[i] correctly addresses deviation_source.
        t2_to_blocks = self.filter.measurement_data.indices_by_t2
        t2_sorted = np.asarray(sorted(t2_to_blocks.keys()), dtype=float)

        if (
            self.filter.flag_sequence
            and hasattr(self.filter, "events_epochs")
            and len(self.filter.events_epochs) > 0
        ):
            event_times_arr = np.asarray(self.filter.events_epochs, dtype=float)
            timeline_sorted = np.unique(np.concatenate((t2_sorted, event_times_arr)))
        else:
            timeline_sorted = t2_sorted

        # Helper: index into deviation_source of the last timeline epoch belonging to
        # leg i. An entry recorded exactly at an event-node epoch belongs to the next
        # leg (the filter increments its leg counter at the node), so skip it.
        def _last_dev_idx_for_leg(i: int) -> int:
            le = float(self.filter.legs_epochs[i][-1])
            j = int(np.searchsorted(timeline_sorted, le, side="right") - 1)
            if j >= 0 and i < nlegs - 1 and np.isclose(timeline_sorted[j], le):
                j -= 1  # skip the node entry (it belongs to leg i+1)
            if j < 0:
                raise ValueError(
                    f"No timeline time within leg {i} (leg end {le}, first time "
                    f"{timeline_sorted[0]})."
                )
            return j

        def _trim(dev, n):
            dev = np.asarray(dev, float).ravel()
            return dev[:n]

        def _map_back(Phi, dev, n):
            # Phi: (n,n), dev possibly padded
            dev_n = _trim(dev, n)
            return np.linalg.solve(Phi, dev_n)

        # 3) For each leg, the deviation-history index that belongs to it (its last
        #    measurement epoch) and the STM evaluated at that same epoch (a leg may be
        #    propagated past its last measurement).
        idx_leg_end = [_last_dev_idx_for_leg(i) for i in range(nlegs)]
        stm_at_dev = [
            self.filter.trajectory.get_STM_sequence(
                epoch=float(timeline_sorted[idx_leg_end[i]]), idx_leg=i
            )
            for i in range(nlegs)
        ]

        # -------------------------
        # Case A: per-leg mapping only (return list)
        # -------------------------
        if not map_back_sequence:
            # deviation at leg end, mapped to leg start, one per leg
            dev_end_list = [deviation_source[idx_leg_end[i]] for i in range(nlegs)]

            # Map each leg independently: last-measurement epoch -> leg start.
            # NOTE: this returns deviations at each leg start in that leg's own definition.
            new_prior_deviation = [
                _map_back(stm_at_dev[i], dev_end_list[i], self.filter.legs_n[i])
                for i in range(nlegs)
            ]
            return new_prior_deviation

        # -------------------------
        # Case B: map back through entire sequence (return list, one per leg start)
        # -------------------------
        # Start from the last leg's deviation at its last-measurement epoch, mapped
        # to the leg start with the STM at that same epoch.
        dev_end = deviation_source[idx_leg_end[-1]]
        n_last = self.filter.legs_n[-1]
        dev_start = _map_back(stm_at_dev[-1], dev_end, n_last)

        new_prior_deviation = [dev_start]

        for k in range(nlegs - 1, 0, -1):
            # 1) ensure dev_start is in leg-k true space before mapping definition
            dev_start = _trim(dev_start, self.filter.legs_n[k])

            # 2) map definition across node: leg k -> leg k-1 (reduction)
            dev_in_prev_def_at_node = self.filter._map_deviation_definition_node(
                k, dev_start, flag_forward=False
            )

            # 3) ensure it matches leg k-1 true size before STM mapping
            dev_in_prev_def_at_node = _trim(
                dev_in_prev_def_at_node, self.filter.legs_n[k - 1]
            )

            # 4) map within previous leg: node(end) -> start (after the definition
            #    reduction the deviation lives at the node epoch, so the leg-end STM
            #    applies here).
            dev_start = _map_back(
                last_STMs_legs[k - 1],
                dev_in_prev_def_at_node,
                self.filter.legs_n[k - 1],
            )

            new_prior_deviation = [dev_start] + new_prior_deviation

        return new_prior_deviation

    def save(self, filepath: Union[str, Path] = None) -> None:
        """
        Save all filter iterations to a single JSON file.

        If filepath is None, uses output_settings.output_dir /
        output_settings.solution_json_filename + '.json'.

        Each iteration is a key 'iter_1', 'iter_2', etc.
        The current (final) solution is also stored under 'final'.
        """
        # Resolve path
        if filepath is None:
            out_dir = (
                getattr(self.output_settings, "solution_output_path ", None) or "."
            )
            basename = (
                getattr(self.output_settings, "solution_output_name", None)
                or "solution"
            )
            filepath = Path(out_dir) / f"{basename}.json"
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        def _solution_to_dict(sol):
            def _res(r):
                if r is None:
                    return None
                return {k: [[float(a), float(b)] for a, b in v] for k, v in r.items()}

            d = {
                "timestamps": sol.timestamps.tolist(),
            }
            if sol.deviation_est is not None:
                d["deviation_estimated"] = sol.deviation_est.tolist()
            if sol.deviation_cumulative is not None:
                d["deviation_cumulative"] = sol.deviation_cumulative.tolist()
            if sol.state_est is not None:
                d["state_estimated"] = sol.state_est.tolist()
            if sol.deviation_smooth is not None:
                d["deviation_smoothed"] = sol.deviation_smooth.tolist()
            if sol.covariance_est is not None:
                d["covariance_diagonals"] = [
                    np.diag(P).tolist() for P in sol.covariance_est
                ]
            if sol.covariance_smooth is not None:
                d["covariance_smooth_diagonals"] = [
                    np.diag(P).tolist() for P in sol.covariance_smooth
                ]
            if sol.covariance_consider is not None:
                d["covariance_consider_diagonals"] = [
                    np.diag(P).tolist() if P is not None else None
                    for P in sol.covariance_consider
                ]
            d["prefits"] = _res(sol.prefits)
            d["postfits"] = _res(sol.postfits)
            if sol.postfits_smoother is not None:
                d["postfits_smoother"] = _res(sol.postfits_smoother)
            return d

        # Collect all iterations from filter history
        history = getattr(self._filter, "_solution_history", [])

        output = {
            "metadata": {
                "n_states": self.n_states,
                "is_covariance_analysis": self.is_covariance_analysis,
                "n_iterations": len(history),
                **self.output_settings.metadata,
            }
        }

        for i, sol in enumerate(history, start=1):
            output[f"iter_{i}"] = _solution_to_dict(sol)

        # Final solution (this object itself, same as last iter but explicit)
        output["final"] = _solution_to_dict(self)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[SolutionOD] All iterations saved → {filepath}")
