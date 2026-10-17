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
    SpiceManager,
)
from typing import Optional, Dict, List, Tuple, Union
import scarabaeus.utils.NumpyWrapper as np
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import scarabaeus
from scarabaeus.utils.Serialization import (
    awu_to_dict,
    dump_json,
    dump,
    format_suffix,
    to_jsonable,
    with_format_suffix,
)

# ------------------#
#  Generate Units  #
# ------------------#
sec = Units.get_units("sec")


# --------------------#
#  Class Definition  #
# --------------------#
class SolutionOD:
    r"""Stores and manages the results of an orbit determination process.

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
        accumulated across all iterations up to and including this one
        [n_epochs x n_states].
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

    See Also
    --------
    scarabaeus.FilterOD : Base filter class that produces this solution.
    scarabaeus.OutputSettings : Controls which quantities are stored.

    Notes
    -----
    Storage of each quantity is controlled by the flags in the associated
    :class:`~scarabaeus.OutputSettings` instance.  Quantities not selected are
    left as ``None`` and not computed during the filter run.

    :meth:`save` and :meth:`create_JSON` write self-describing files: the
    metadata labels every state index (owner, SPICE ID, frame, units) and
    records the a priori state and covariance, the convergence test and the
    provenance; the solution blocks add the estimate at the initial epoch.
    Covariances are written either in full or as diagonals, never both.

    **Choosing a mapped solution.** State and covariance mapping use the same
    data-cutoff convention. For an existing ``solution`` and target
    :class:`~scarabaeus.EpochArray` called ``epochs``:

    .. code-block:: python

        # Filtered: each target uses measurements through that target time.
        filtered = solution.propagate_state_covariance(
            epochs, use_smoothed=False, lookback_s=0.0)

        # Rolling prediction: each target uses measurements through 14 days earlier.
        prediction_14d = solution.propagate_state_covariance(
            epochs, use_smoothed=False, lookback_s=14 * 86400.0)

        # Reconstruction: every target uses the final filter update.
        reconstructed = solution.propagate_state_covariance(
            epochs, use_smoothed=False, use_all_data=True)

    Each result is a list of ``(StateArray, covariance)`` pairs in requested
    epoch order. For covariance-only analyses, use
    :meth:`propagate_covariance` with the same cutoff keywords; such analyses
    do not store state deviations. Covariance rows follow the nominal state
    component order, and entry ``(i, j)`` has the product of component ``i``
    and ``j`` units.

    For a selected update at epoch :math:`r`, mapping to :math:`t` uses

    .. math::

        A(t,r) &= \Phi(t,t_0)\Phi(r,t_0)^{-1}, \\
        \hat x(t) &= x_{\mathrm{nom}}(t) + A(t,r)\delta x(r), \\
        P(t) &= A(t,r)P(r)A(t,r)^T.

    These are linearized mappings around the fitted nominal reference. They
    add no measurements or process noise. Reconstruction maps the final
    covariance; it is not a stochastic backward smoother. When coverage is
    insufficient or STM interpolation is rejected, mapping reintegrates the
    nominal reference on copies of the propagator. The correction is added
    once.

    The default ``lookback_s=None`` retains the older API convention: a
    continuous arc maps its final update inside/after the measurement arc and
    its first update before it. Specify ``lookback_s=0.0, use_smoothed=False``
    for a causal filtered product. An explicit cutoff before the first update
    uses the prior; the prior covariance must be available. Simultaneous
    updates retain their order, and the last one at a cutoff is selected.

    Consider-augmented covariance is included by default when considered
    parameters are present. A smoothed augmented history is unavailable, so
    those queries need ``use_smoothed=False``. A paired query requires both
    smoothed histories or neither, preventing mixed state and covariance
    products.

    Sequences use local leg STMs, with shared nodes assigned to the later leg.
    Positive lookback and all-data reconstruction across sequence events
    currently raise ``NotImplementedError`` because they require event
    transition maps.

    **Reading and saving results.** :meth:`estimated_trajectory` reads each
    measurement epoch's own update and requires an epoch match within one
    microsecond. Use :meth:`propagate_state` for arbitrary epochs. Sequence
    parameter results are grouped by leg and contain only the requested rows
    for each leg, in request order. :meth:`map_state_deviation_to_epoch` maps
    corrections to nominal initial epochs and returns one vector per sequence
    leg in either sequence mapping mode.

    :meth:`UQ_compute` uses normalized particle weights consistently for the
    mean, population covariance, intervals and effective sample size. Its
    covariance diagonal equals the squared reported standard deviations; this
    is not an unbiased sample covariance with an N-1 correction.

    :meth:`serialize` preserves the update timeline, component frame/units,
    residual epochs, sequence layout and multi-arc global results alongside
    stored arrays. :meth:`deserialize` can inspect and export those results
    without the original filter. Trajectory queries and propagation require
    reattaching the matching fitted filter, since the pickle does not embed
    its propagator or SPICE kernels.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    class ElementConversionError(ValueError):
        """Raised when the estimate cannot be expressed in orbital elements —
        the body is not in the state vector, or its gravitational parameter is
        not available."""

    class DebugHistoryError(ValueError):
        """Raised when filter debug histories that pair up per measurement
        update differ in length."""

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        filter_obj: FilterOD,
        output_settings: Optional[OutputSettings] = None,
    ):
        """Initialize stored results and metadata from a filter.

        Parameters
        ----------
        filter_obj : FilterOD
            Filter whose available results and measurement epochs are read.
            The solution retains this object for subsequent trajectory queries.
        output_settings : OutputSettings or None, optional
            Quantities to retain. None uses ``filter_obj.settings.output`` when
            available, otherwise a new OutputSettings instance.

        Notes
        -----
        Initializes optional result attributes, captures initial frame/units,
        and calls :meth:`_store_solution_data`. Particle UQ and requested debug
        quantities may be computed during storage. The absolute state remains
        lazy, and construction does not run the orbit-determination filter.
        Stored filter arrays are not necessarily independent copies.
        """
        self._filter = filter_obj

        # Use filter's output settings if not provided
        if output_settings is None:
            if hasattr(filter_obj.settings, "output"):
                self.output_settings = filter_obj.settings.output
            else:
                self.output_settings = OutputSettings()
        else:
            self.output_settings = output_settings

        # Extract timestamps
        self.timestamps = self._extract_timestamps()

        # Frame + units of the estimated state, captured once here from the
        # filter's initial state vector. The filter runs on raw numbers and
        # never carries this metadata through the computation; it is recorded
        # at the start so the saved outputs can be labelled correctly.
        (self._state_frame, self._state_units,
         self._state_components) = self._capture_initial_frame_units()
        # Creation time, written as provenance by save()/create_JSON().
        self._created_utc = datetime.now(timezone.utc).isoformat()

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
        # Per-entry reception epochs (TDB seconds) paralleling prefits/postfits,
        # keyed by dataset name: {dataset: [t2, ...]}. Captured from the filter so
        # serialized residuals can carry a time tag.
        self.prefit_epochs = None
        self.postfit_epochs = None
        self.postfit_epochs_smooth = None
        # Epoch of every state-deviation / covariance row (= filter timeline grid,
        # measurements ∪ events). Longer than ``timestamps`` for sequences.
        self.timeline = None
        # Forward solution at each sequence node in the previous leg's definition
        # (sequential filters), used to map the final estimate back through it.
        self._node_deviations = None
        self._node_covariances = None
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
        """Number of states in the state vector (the full union for a sequence)."""
        # filter.n holds the width of the leg the filter last processed
        return getattr(self.filter, "n_global", self.filter.n)

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
        """Read spacecraft measurement epochs from the filter's measurement data.

        Returns
        -------
        ndarray
            TDB seconds returned by ``measurement_data.get_spacecraft_times()``,
            converted to a NumPy array without additional sorting or deduplication.

        Notes
        -----
        This is the measurement grid, not the deviation/covariance update grid.
        Sequence event epochs are provided separately by :meth:`_timeline_epochs`.
        """
        return np.array(self.filter.measurement_data.get_spacecraft_times())

    # ---------------------------------------------------------------------
    # Sequence layout helpers
    #
    # The state-deviation and covariance histories are built over the filter
    # *timeline* (unique union of measurement and event epochs), which for a
    # sequence is longer than the measurement-only ``timestamps``. These helpers
    # attach an epoch and a leg index to every row, and describe each leg's state
    # layout, so the saved solution is self-describing for sequence runs.
    # ---------------------------------------------------------------------
    def _timeline_epochs(self) -> Optional[np.ndarray]:
        """Select the epoch grid used to index deviation and covariance rows.

        Returns
        -------
        ndarray or None
            Flattened floating-point TDB seconds, preserving stored row order
            and duplicate epochs. None means no nonempty grid is available.

        Notes
        -----
        Uses ``self.timeline`` when it is not None, otherwise the filter's
        ``timeline_times``. If that selected grid is empty or missing, falls
        back to measurement ``timestamps``. An explicitly empty solution timeline
        therefore also falls back to measurements. This helper neither sorts
        nor validates row counts; consumers must check alignment with histories.
        """
        tl = self.timeline if self.timeline is not None else getattr(
            self.filter, "timeline_times", None
        )
        if tl is not None and len(tl) > 0:
            return np.asarray(tl, dtype=float).ravel()
        if self.timestamps is not None and len(self.timestamps) > 0:
            return np.asarray(self.timestamps, dtype=float).ravel()
        return None

    def _leg_index_for_epoch(self, t: float) -> Optional[int]:
        """Return the leg index a timeline epoch ``t`` belongs to.

        Parameters
        ----------
        t : float
            Epoch in TDB seconds.

        Returns
        -------
        int or None
            Zero-based index into ``filter.legs_epochs``. The later leg wins
            at shared boundaries or overlaps. Epochs at/after the final leg end
            map to the final leg. None is returned if legs are absent, t is
            before the first leg, or t falls in an uncovered interior gap.

        Notes
        -----
        Membership uses each leg's first and last stored epochs as an inclusive
        span, not exact membership in its sampling grid. Comparisons are exact;
        no relative tolerance is applied to large TDB timestamps.
        """
        legs = getattr(self.filter, "legs_epochs", None)
        if not legs:
            return None
        nlegs = len(legs)
        for i in reversed(range(nlegs)):
            leg = legs[i]
            leg_arr = np.asarray(leg, dtype=float).ravel()
            if leg_arr.size == 0:
                continue
            t0, t1 = leg_arr[0], leg_arr[-1]
            if t0 <= t <= t1:
                return i
        # Epoch at/after the final leg end, or otherwise unmatched.
        last = np.asarray(legs[-1], dtype=float).ravel()
        if last.size and t >= last[-1]:
            return nlegs - 1
        return None

    def _deviation_history(self, use_smoothed=True):
        """Return deviation rows aligned with a validated, sorted timeline.

        Parameters
        ----------
        use_smoothed : bool, optional
            Select a nonempty smoothed history when True; otherwise, or when
            smoothing is unavailable, select the filtered history. Defaults
            to ``True``.

        Returns
        -------
        timeline : ndarray, shape (n_updates,)
            Ascending TDB seconds, retaining event rows and simultaneous updates.
        deviations : list of ndarray
            Flattened corrections in timeline order and nominal component units.
            Sequence rows can have padding for shorter state definitions.
        smoothing : bool
            Whether the selected history contains smoothed corrections.

        Raises
        ------
        ValueError
            If the history is absent/empty, row counts differ, or epochs are
            nonfinite.

        Notes
        -----
        Sorting is stable: simultaneous rows keep their filter order, so callers
        can select the last update. Stored histories are not reordered in place.
        """
        smoothing = (use_smoothed and self.deviation_smooth is not None
                     and len(self.deviation_smooth) > 0)
        source = self.deviation_smooth if smoothing else self.deviation_est
        if source is None or len(source) == 0:
            raise ValueError("No state deviation history available.")
        timeline = self._timeline_epochs()
        if timeline is None or len(timeline) != len(source):
            raise ValueError("State deviation history and filter timeline lengths differ.")
        if not np.isfinite(timeline).all():
            raise ValueError("State deviation history epochs must be finite.")
        order = np.argsort(timeline, kind="stable")
        return timeline[order], [np.asarray(source[i], float).ravel() for i in order], smoothing

    @staticmethod
    def _matching_epoch_index(timeline, t):
        """Match a stored epoch to microsecond precision.

        Parameters
        ----------
        timeline : ndarray, shape (n_updates,)
            Finite TDB seconds; sorting is not required.
        t : float
            Requested epoch in TDB seconds.

        Returns
        -------
        int
            Closest stored epoch's index. If several rows share that exact
            stored time, return the last row in their original order.

        Raises
        ------
        ValueError
            If the timeline is empty or no epoch is within 1e-6 seconds.

        Notes
        -----
        This selects a row without propagating its correction. Callers evaluate
        the nominal state at the matched stored time, not at a nearby input time.
        """
        distances = np.abs(timeline - t)
        if (len(distances) == 0 or not np.isfinite(distances).all()
                or distances.min() > 1e-6):
            raise ValueError(f"Epoch {t} does not match a stored filter epoch.")
        nearest = int(np.argmin(distances))
        return int(np.flatnonzero(timeline == timeline[nearest])[-1])

    def _reference_model(self, leg=None):
        """Select the nominal initial state for an arc or sequence leg.

        Parameters
        ----------
        leg : int or None, optional
            Zero-based continuous-leg index, excluding event nodes. None selects
            the continuous arc's ``propagator.full_state_vector``. Defaults to
            ``None``.

        Returns
        -------
        StateArray
            Initial state with its actual epoch, component order, owners, units
            and frames. This references the filter's model; copy before editing.
        """
        if leg is not None:
            return self.filter.legs_model[leg]
        return self.filter.propagator.full_state_vector

    def _integrate_reference(self, times, leg=None):
        """Sample the nominal state and STMs without changing the fitted propagator.

        Parameters
        ----------
        times : iterable of float
            Sample epochs in TDB seconds; order and duplicates do not matter.
        leg : int or None, optional
            Zero-based sequence leg, or None for a continuous arc. A leg is
            extended with its own dynamics, without executing sequence events.
            Defaults to ``None``.

        Returns
        -------
        stms : dict of float to ndarray
            STMs at requested times and the nominal initial epoch, referenced
            to that initial epoch with identity STM0.
        states : dict of float to StateArray
            Uncorrected nominal states at the same epochs, retaining the selected
            model's component layout, units and frames.

        Raises
        ------
        ValueError
            If sequence propagators cannot be matched to their leg models.
            Integration and sampling failures also propagate to the caller.

        Notes
        -----
        Every matrix is integrated from the actual initial state epoch. Both
        sides of that epoch use separate copies, so mixed forward/backward
        requests share the same reference and never apply a correction twice.
        """
        from copy import deepcopy
        from scarabaeus import Trajectory

        model = self._reference_model(leg)
        propagator = self.filter.propagator
        if leg is not None:
            sequence = self.filter.trajectory.sequence_definition
            # Nodes and impulsive burns hold a propagator too; legs_model keeps
            # only the StateArray models, so pair each leg with its own.
            propagators = [p for m, p in zip(sequence.models, sequence.props)
                           if isinstance(m, StateArray) and isinstance(p, Propagator)]
            if len(propagators) != len(self.filter.legs_model):
                raise ValueError("Sequence leg propagators do not match its state models.")
            propagator = propagators[leg]
            # A leg starts from the state the sequence carried across its node,
            # not from its model's values (a template after the first leg).
            leg_start = float(np.asarray(self.filter.legs_epochs[leg], float).ravel()[0])
            model = self._nominal_state(leg_start, leg)
        epoch0 = float(np.asarray(model.epoch.et).ravel()[0])
        identity = np.eye(model.size)
        stms, states = {epoch0: identity}, {epoch0: deepcopy(model)}
        for backward in (True, False):
            targets = sorted(set(float(t) for t in times
                                 if (t < epoch0 if backward else t > epoch0)))
            if not targets:
                continue
            grid = np.unique(np.r_[epoch0, targets])
            prop = deepcopy(propagator)
            prop.reinitialize(
                state_vector=deepcopy(model),
                tspan=EpochArray(grid[::-1] if backward else grid, sys="TDB"),
                backward=backward, propagate_STM=True, STM0=identity.copy(),
            )
            prop.propagate(display_progress=False)
            sampled = prop.propagated_state_array
            trajectory = Trajectory("tmp_solution_reference", state_array=sampled)
            trajectory.add_STMs(prop.STM, timestamps=sampled.epoch)
            for t in targets:
                stms[t] = np.asarray(trajectory.get_STM(epoch=t), float)
                states[t] = sampled.get_state_at_epoch(EpochArray(t, sys="TDB"))
        return stms, states

    def _nominal_state(self, t, leg=None):
        """Read all nominal components in their original state-vector order.

        Parameters
        ----------
        t : float
            Requested epoch in TDB seconds.
        leg : int or None, optional
            Zero-based sequence leg, or None for a continuous arc. Defaults
            to ``None``.

        Returns
        -------
        StateArray
            Single-epoch uncorrected state with original component owners,
            units, frames and estimation/dynamics modes.

        Raises
        ------
        ValueError
            If a primary-body query cannot represent the full state, nominal
            parameters are missing, or parameter dimensions disagree. Messages
            containing ``Cannot extrapolate`` request integration fallback in
            :meth:`propagate_state`. Other trajectory query failures propagate.

        Notes
        -----
        Exact stored samples contain all bodies. Otherwise a single-body query
        provides Cartesian coordinates, converted to the model's units, and
        nominal parameters. Multiple-body states require integration when the
        requested epoch has no exact stored sample.
        """
        trajectory = self.filter.trajectory
        sampled = getattr(trajectory, "_state_array", None)
        if sampled is not None and leg is None:
            et = np.asarray(sampled.epoch.et, float).ravel()
            indices = np.flatnonzero(et == t)
            if len(indices):
                epoch = (sampled.epoch if np.ndim(sampled.epoch.et) == 0 else
                         sampled.epoch[int(indices[0])])
                return sampled.get_state_at_epoch(epoch)
        model = self._reference_model(leg)
        if sum(entry[0] == "position" for entry in model.state) != 1:
            # Trajectory.get_state exposes only its primary body's Cartesian
            # state. Integrate all bodies together rather than duplicating it.
            raise ValueError("Cannot extrapolate the full state from a primary-body query.")
        nominal = trajectory.get_state(
            epoch_input=EpochArray(t, sys="TDB"), idx_leg_input=leg)
        parameters = nominal.get("parameters")
        parameters = None if parameters is None else np.asarray(parameters.values).ravel()
        components, offset = [], 0
        for name, size, est, dyn, owner, template in model.state:
            if name in ("position", "velocity"):
                values = nominal[name].convert_to(template.quantity.units).values
            elif parameters is not None:
                values = parameters[offset:offset + size]
                offset += size
                if len(values) != size:
                    raise ValueError("Nominal parameter vector does not match the state definition.")
            else:
                raise ValueError("Cannot extrapolate missing nominal parameter values.")
            components.append((name, size, est, dyn, owner, ArrayWFrame(
                ArrayWUnits(np.asarray(values).ravel(), template.quantity.units), template.frame)))
        return StateArray(EpochArray(t, sys="TDB"), model.origin,
                          StateDefinition.from_components(components))

    @staticmethod
    def _correct_state(nominal, deviation):
        """Add one correction, retaining each component's owner, units and frame.

        Parameters
        ----------
        nominal : StateArray
            Uncorrected state at a single epoch.
        deviation : array_like, shape (nominal.size,)
            Additive correction in nominal component order, units and frames.
            Remove sequence storage padding before calling this helper.

        Returns
        -------
        StateArray
            Corrected state at the same epoch with all component metadata
            preserved. Neither input is modified.

        Raises
        ------
        ValueError
            If the correction dimension differs from the nominal state size.
        """
        deviation = np.asarray(deviation, float).ravel()
        if len(deviation) != nominal.size:
            raise ValueError("State deviation size does not match the nominal state.")
        components, offset = [], 0
        for name, size, est, dyn, owner, template in nominal.state:
            values = np.asarray(template.quantity.values).ravel() + deviation[offset:offset + size]
            components.append((name, size, est, dyn, owner, ArrayWFrame(
                ArrayWUnits(values, template.quantity.units), template.frame)))
            offset += size
        return StateArray(nominal.epoch, nominal.origin,
                          StateDefinition.from_components(components))

    def _legs_metadata(self) -> Optional[List[Dict]]:
        """Describe each sequence leg's epoch span and state-vector columns.

        Returns
        -------
        list of dict or None
            One entry per ``filter.legs_model``, in leg order, with ``index``,
            ``epoch_start``/``epoch_end`` in TDB seconds, ``n_states`` and a
            ``components`` list in the schema of :meth:`_components_metadata`.
            None means leg models or epoch grids are unavailable.

        Notes
        -----
        Column offsets restart at zero for each leg and exclude storage padding
        from wider legs. An unreadable model state yields an empty component
        list and zero size; a missing epoch grid yields None bounds. The result
        describes layouts without copying the propagated state histories.
        """
        legs_model = getattr(self.filter, "legs_model", None)
        legs_epochs = getattr(self.filter, "legs_epochs", None)
        if not legs_model or not legs_epochs:
            return None

        out = []
        for i, model in enumerate(legs_model):
            try:
                state_def = model.state
            except Exception:
                state_def = []

            components = self._components_metadata(state_def)
            cursor = components[-1]["index_end"] if components else 0

            leg_arr = (
                np.asarray(legs_epochs[i], dtype=float).ravel()
                if i < len(legs_epochs)
                else np.array([])
            )
            out.append(
                {
                    "index": i,
                    "epoch_start": float(leg_arr[0]) if leg_arr.size else None,
                    "epoch_end": float(leg_arr[-1]) if leg_arr.size else None,
                    "n_states": cursor,
                    "components": components,
                }
            )
        return out

    def _sequence_extras(self) -> Dict:
        """Assemble timeline and sequence-layout fields for serialization.

        Returns
        -------
        dict
            Available fields among ``timeline_epochs``, ``events_epochs``,
            ``leg_index`` and ``legs``. Epochs are lists of TDB seconds;
            ``leg_index`` parallels timeline rows and may contain None for
            uncovered epochs. ``legs`` uses :meth:`_legs_metadata`'s schema.

        Notes
        -----
        Starts with a shallow copy of metadata retained during deserialization.
        A current timeline replaces its saved counterpart. Event and leg fields
        are refreshed only when the filter exposes nonempty event epochs, so
        saved layout metadata remains usable without an attached live filter.
        This helper constructs a dictionary; it does not write a file.
        """
        extras: Dict = dict(getattr(self, "_saved_sequence_extras", {}))

        timeline = self._timeline_epochs()
        if timeline is not None:
            extras["timeline_epochs"] = [float(t) for t in timeline]

        events = getattr(self.filter, "events_epochs", None)
        if events is not None and len(events) > 0:
            extras["events_epochs"] = [
                float(e) for e in np.asarray(events, dtype=float).ravel()
            ]
            if timeline is not None:
                extras["leg_index"] = [
                    self._leg_index_for_epoch(float(t)) for t in timeline
                ]
            legs_meta = self._legs_metadata()
            if legs_meta is not None:
                extras["legs"] = legs_meta

        return extras

    @staticmethod
    def _components_metadata(state_def) -> List[Dict]:
        """Describe each component of a state definition, in state order.

        Parameters
        ----------
        state_def : iterable of tuple
            Entries ``(name, dim, estimation, dynamics, owner, value)`` of a
            ``StateDefinition`` (``StateArray.state``); trailing fields may be
            missing.

        Returns
        -------
        list of dict
            One dict per component with ``name``, ``dim``, ``estimation``,
            ``dynamics``, the owning body's ``owner`` name and ``spice_id``, the
            ``frame`` name, ``units`` (``Units.to_dict()``) and zero-based
            ``index_start``/``index_end`` (end exclusive) into the state vector.
            Fields the entry does not provide are None.

        Notes
        -----
        Labels only: values are not read, converted or copied.
        """
        components = []
        cursor = 0
        for entry in state_def:
            dim = int(entry[1])
            owner = entry[4] if len(entry) > 4 else None
            value = entry[5] if len(entry) > 5 else None
            units = getattr(getattr(value, "quantity", None), "units", None)
            components.append(
                {
                    "name": entry[0],
                    "dim": dim,
                    "estimation": entry[2] if len(entry) > 2 else None,
                    "dynamics": entry[3] if len(entry) > 3 else None,
                    "owner": getattr(owner, "name", None),
                    "spice_id": getattr(owner, "spice_id", None),
                    "frame": getattr(getattr(value, "frame", None), "name", None),
                    "units": units.to_dict() if hasattr(units, "to_dict") else None,
                    "index_start": cursor,
                    "index_end": cursor + dim,
                }
            )
            cursor += dim
        return components

    @staticmethod
    def _state_labels(components: List[Dict]) -> List[str]:
        """Name every state-vector index, e.g. ``"Sat.position[0] [km]"``.

        Parameters
        ----------
        components : list of dict
            Component descriptions as returned by :meth:`_components_metadata`.

        Returns
        -------
        list of str
            One label per state index: owner (when known), component name,
            element index (for components of dimension > 1) and units name
            (when known).
        """
        labels = []
        for c in components:
            owner = f"{c['owner']}." if c.get("owner") else ""
            units = (c.get("units") or {}).get("name")
            units = f" [{units}]" if units else ""
            for k in range(c["dim"]):
                element = f"[{k}]" if c["dim"] > 1 else ""
                labels.append(f"{owner}{c['name']}{element}{units}")
        return labels

    @staticmethod
    def _series_alignment(d: Dict) -> Dict:
        """Map each per-epoch array in a solution dict to its timestamp axis.

        Inspects the arrays actually present in the serialized solution ``d`` and,
        by matching row counts, records for each one the timestamp field that
        indexes it (``"timestamps"`` = measurement grid, ``"timeline_epochs"`` =
        deviation/covariance grid = measurements and event nodes) together with its
        length. Matching is based on lengths, not on the physical meaning of
        each row or a comparison of individual epoch values.

        Parameters
        ----------
        d : dict
            A serialized solution dict (as built by :meth:`create_JSON` or the
            per-iteration writer in :meth:`save`).

        Returns
        -------
        dict
            ``{array_name: {"n_rows": int, "timestamps": <field name or None>}}``
            for every recognised per-epoch array present in ``d``.

        Notes
        -----
        Only list-valued fields are inspected. ``timeline_epochs`` takes
        precedence over ``timestamps`` when both lengths match; None marks a
        series whose row count matches neither axis. The input is not modified.
        """
        # Candidate timestamp axes, longest first so an exact-length match prefers
        # the most specific axis when the two happen to be equal length.
        axes = {
            name: len(d[name])
            for name in ("timeline_epochs", "timestamps")
            if isinstance(d.get(name), list)
        }
        series = (
            "state_estimated",
            "deviation_estimated",
            "deviation_cumulative",
            "deviation_smoothed",
            "covariance_diagonals",
            "covariance_smooth_diagonals",
            "covariance_consider_diagonals",
            "covariance_full",
            "covariance_smooth_full",
            "covariance_consider_full",
            "leg_index",
            "nis_statistics",
        )
        alignment = {}
        for name in series:
            arr = d.get(name)
            if not isinstance(arr, list):
                continue
            n = len(arr)
            match = next((ax for ax, ln in axes.items() if ln == n), None)
            alignment[name] = {"n_rows": n, "timestamps": match}
        return alignment

    @staticmethod
    def _has_consider_rows(rows) -> bool:
        """
        Whether a consider covariance history holds any covariance.

        A sequence may consider states on some legs only, its rows being
        ``None`` elsewhere, while the filter's ``has_consider`` is its last
        leg's.

        Parameters
        ----------
        rows : list of (numpy.ndarray or None) or None
            Consider covariance history, one entry per timeline epoch.

        Returns
        -------
        bool
            Whether any entry of ``rows`` is a covariance.
        """
        return rows is not None and any(P is not None for P in rows)

    def _store_solution_data(self) -> None:
        """
        Populate solution attributes from available filter results.

        Returns
        -------
        None
            Updates this solution in place according to ``output_settings``.

        Notes
        -----
        Covariance-analysis mode stores selected covariance histories and the
        update timeline, then returns without deviations, residuals or UQ.
        Full-filter mode also retains selected deviations, cumulative deviations,
        residuals and their epochs, and smoothed quantities. Ragged deviation
        rows are right-padded with zeros to the widest state definition.

        Particle data triggers :meth:`UQ_compute`, and debug flags trigger
        :meth:`_collect_debug_data`. Absolute states are left for lazy evaluation
        through ``state_est``. Available arrays may be retained by reference;
        this is not a deep snapshot of the filter. Disabled or unavailable
        quantities are not cleared when this helper is called again.
        """
        # Covariance analysis mode: only covariances
        if self.is_covariance_analysis:
            if self.output_settings.save_covariance_history:
                self.covariance_est = self.filter.covariance_history
            if (self.output_settings.save_consider_covariance_history
                    and self._has_consider_rows(
                        getattr(self.filter, "covariance_consider_history", None))):
                self.covariance_consider = self.filter.covariance_consider_history
            self.timeline = getattr(self.filter, "timeline_times", None)
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

        # Cumulative deviation wrt first reference (across all iterations): the
        # deviation earlier iterations applied to the reference, plus this one.
        # Rows are paired by epoch, since the filter epochs move between passes.
        if self.deviation_est is not None:
            self.deviation_cumulative = self.deviation_est.copy()
            applied = getattr(self.filter, "_cumulative_deviation", None)
            if applied is not None:
                self.deviation_cumulative = FilterOD._accumulate_deviation(
                    np.atleast_2d(np.asarray(applied, dtype=float)),
                    getattr(self.filter, "_cumulative_epochs", None),
                    self.deviation_est,
                    np.asarray(getattr(self.filter, "timeline_times", []), dtype=float),
                )
            # state_est is computed lazily on first access via the property

        # Estimated covariances
        if self.output_settings.save_covariance_history:
            if hasattr(self.filter, "covariance_history"):
                self.covariance_est = self.filter.covariance_history

        # Filter timeline (epoch of every deviation/covariance row)
        self.timeline = getattr(self.filter, "timeline_times", None)

        # Forward node solutions of this pass (rebuilt by every filter run)
        self._node_deviations = getattr(self.filter, "_node_deviations", None)
        self._node_covariances = getattr(self.filter, "_node_covariances", None)

        # Pre-fit residuals
        if self.output_settings.save_prefit_residuals:
            if hasattr(self.filter, "prefit_residuals"):
                self.prefits = self.filter.prefit_residuals
                self.prefit_epochs = getattr(self.filter, "prefit_epochs", None)

        # Post-fit residuals
        if self.output_settings.save_postfit_residuals:
            if hasattr(self.filter, "postfit_residuals"):
                self.postfits = self.filter.postfit_residuals
                self.postfit_epochs = getattr(self.filter, "postfit_epochs", None)

        # Consider covariances
        if self.output_settings.save_consider_covariance_history and self._has_consider_rows(
            getattr(self.filter, "covariance_consider_history", None)
        ):
            self.covariance_consider = self.filter.covariance_consider_history

        # Smoothed solution
        if self.output_settings.save_smoothed_solution:
            if hasattr(self.filter, "smoothed_state_deviation"):
                _ssd = self.filter.smoothed_state_deviation
                if _ssd is not None and len(_ssd) > 0:
                    rows = [np.asarray(row, float).ravel() for row in _ssd]
                    width = max(len(row) for row in rows)
                    self.deviation_smooth = np.zeros((len(rows), width))
                    for i, row in enumerate(rows):
                        self.deviation_smooth[i, :len(row)] = row
            if hasattr(self.filter, "smoothed_covariance"):
                self.covariance_smooth = self.filter.smoothed_covariance
            if hasattr(self.filter, "postfit_residuals_smooth"):
                self.postfits_smoother = self.filter.postfit_residuals_smooth
                self.postfit_epochs_smooth = getattr(
                    self.filter, "postfit_epochs_smooth", None
                )

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

    def _parameter_layout(self, idx_leg=None, t=None) -> tuple:
        """
        State definition and sidecar body the reference parameters belong to.

        The reference trajectory returns, as ``parameters``, what its sidecar
        holds: for a single arc the primary body's sidecar, for a sequence the
        leg's, which carries every leg parameter. This names the state layout
        and the body that sidecar was written for, so the deviation rows can be
        picked with the same rule (:meth:`_parameter_deviation`).

        Parameters
        ----------
        idx_leg : int, optional
            Leg the deviation belongs to (sequences only).
        t : float, optional
            Epoch used to find the leg when ``idx_leg`` is not given.

        Returns
        -------
        tuple
            ``(state_definition, body_id)``; the definition is ``None`` when it
            cannot be resolved, and ``body_id`` is ``None`` for a sequence.
        """
        if not getattr(self.filter, "flag_sequence", False):
            traj = getattr(self.filter, "trajectory", None)
            return (
                getattr(traj, "state_definition", None),
                getattr(getattr(traj, "body", None), "spice_id", None),
            )
        if idx_leg is None and t is not None:
            idx_leg = self._leg_index_for_epoch(float(t))
        legs = getattr(self.filter, "legs_model", None)
        if idx_leg is None or not legs or not 0 <= idx_leg < len(legs):
            return None, None
        return getattr(legs[idx_leg], "state", None), None

    @staticmethod
    def _parameter_deviation(dev, state_definition, body_id=None) -> np.ndarray:
        """
        The rows of a state deviation that belong to a sidecar's parameters.

        They are not simply ``dev[6:]``: that assumes the state opens with the
        primary body's position and velocity and holds no others, and with a
        second propagated body it would also take that body's position and
        velocity, out of line with the nominal parameters it is added to. The
        rows are picked by walking the state definition and keeping the entries
        :meth:`SpiceManager._parameter_entries` assigns to the sidecar — the
        rule that wrote it, so the two cannot drift apart.

        Parameters
        ----------
        dev : array_like
            State deviation. It may span the full state, be right-padded past
            it (sequence legs of different widths are stacked), or hold the
            estimated partition only, in which case a considered parameter has
            no row and its deviation is zero.
        state_definition : list or None
            ``(name, dim, estimation, dynamics, owner, value)`` entries.
        body_id : int, optional
            SPICE id of the body whose sidecar holds the nominal parameters;
            ``None`` selects every parameter.

        Returns
        -------
        numpy.ndarray
            The parameter deviation, in state order. Falls back to ``dev[6:]``
            when the definition is missing or does not fit the deviation.
        """
        dev = np.asarray(dev, dtype=float).ravel()
        # Materialise once: selection is by identity, so both walks must see
        # the same tuples.
        entries = list(state_definition or [])
        if not entries:
            return dev[6:]
        selected = {
            id(entry) for entry in SpiceManager._parameter_entries(entries, body_id)
        }

        # (selected, row in the full state, row in the estimated partition, dim)
        blocks = []
        n_full = n_est = 0
        try:
            for entry in entries:
                dim = int(entry[1])
                considered = entry[2] == "considered"
                blocks.append(
                    (id(entry) in selected, n_full, None if considered else n_est, dim)
                )
                n_full += dim
                n_est += 0 if considered else dim
        except (TypeError, IndexError, ValueError):
            return dev[6:]

        if dev.size >= n_full:
            pieces = [dev[start : start + dim] for keep, start, _, dim in blocks if keep]
        elif dev.size == n_est:
            pieces = [
                np.zeros(dim) if start is None else dev[start : start + dim]
                for keep, _, start, dim in blocks
                if keep
            ]
        else:
            return dev[6:]
        return np.concatenate(pieces) if pieces else np.zeros(0)

    def _compute_state_est(self) -> Optional[np.ndarray]:
        """
        Compute the absolute state estimate at each measurement epoch.

        Returns
        -------
        ndarray or None
            Array with one row per measurement epoch and as many columns as the
            stored deviation history. Rows contain position, velocity and then
            nominal parameters plus their filtered corrections, in the last
            iteration's reference frame. Cartesian values are in km and km/s;
            parameters use the trajectory's native component units. Missing
            reference rows and inactive sequence padding remain NaN. None means
            the trajectory/deviation history is absent or every reference query
            failed.

        Raises
        ------
        ValueError
            If the deviation timeline is empty, nonfinite or has the wrong row
            count, a measurement epoch has no matching update, or nominal and
            deviation dimensions cannot be combined.

        Warns
        -----
        RuntimeWarning
            If one or more nominal trajectory queries fail. A partial failure
            returns the remaining rows; failure at every epoch returns None.

        Notes
        -----
        ``deviation_est`` is indexed on the filter *timeline* (measurements and
        event nodes), which is longer than the measurement-only ``timestamps``
        for sequences. Each measurement epoch is therefore mapped to its row in
        that timeline before the deviation is read; indexing ``deviation_est``
        directly by the measurement counter would misalign every row after the
        first event node that does not coincide with a measurement.

        The last simultaneous update wins. This reader assumes position and
        velocity occupy the first six deviation entries; the parameter rows are
        picked with :meth:`_parameter_deviation`. It uses filtered deviations
        without STM propagation or smoothing. Caching is performed by the ``state_est`` property, not here.
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
        out = np.full((n, n_states), np.nan, dtype=float)
        failures = []

        flag_seq = getattr(self.filter, "flag_sequence", False)

        timeline, deviations, _ = self._deviation_history(use_smoothed=False)

        for k in range(n):
            _ts_k = self.timestamps[k]
            t_k = float(_ts_k.values if hasattr(_ts_k, 'values') else _ts_k)
            ep_k = EpochArray(np.array([t_k]), sys="TDB")

            dev_idx = self._matching_epoch_index(timeline, t_k)
            idx_leg = self._leg_index_for_epoch(t_k) if flag_seq else None

            try:
                nom = traj.get_state(epoch_input=ep_k[0], idx_leg_input=idx_leg)
            except Exception as exc:
                failures.append(exc)
                continue

            dev = deviations[dev_idx]
            pos_nom = np.asarray(nom["position"].values).ravel()
            vel_nom = np.asarray(nom["velocity"].values).ravel()

            row = np.concatenate([pos_nom + dev[:3], vel_nom + dev[3:6]])

            par_dev = self._parameter_deviation(
                dev, *self._parameter_layout(idx_leg, t_k)
            )
            if nom["parameters"] is not None and par_dev.size:
                par_nom = np.asarray(nom["parameters"].values).ravel()
                row = np.concatenate([row, par_nom + par_dev[: len(par_nom)]])

            out[k, : len(row)] = row

        # An epoch outside the reference span is a legitimate miss, so single
        # failures are tolerated — but they must not leave a row of zeros, which reads
        # as a spacecraft at the centre of the origin body rather than as
        # missing data. Say so, and refuse to return an array that is entirely
        # made up of those rows.
        if failures:
            first = failures[0]
            if len(failures) == n:
                warnings.warn(
                    "state_est is unavailable: the reference state could not be "
                    f"read at any of the {n} measurement epochs "
                    f"(first error: {type(first).__name__}: {first}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            warnings.warn(
                f"state_est is missing {len(failures)} of {n} rows: the "
                "reference state could not be read at those epochs, which are "
                f"left as NaN (first error: {type(first).__name__}: {first}).",
                RuntimeWarning,
                stacklevel=2,
            )

        return out

    def _collect_debug_data(self) -> Dict:
        """
        Collect available debug quantities selected by output settings.

        Returns
        -------
        dict or None
            Selected keys among ``kalman_gains``, ``innovation_covariances`` and
            ``nis_statistics``. Filter histories are retained by reference;
            NIS, when available, is a newly computed array. None means no
            requested quantity was available.

        Notes
        -----
        This helper returns the dictionary without assigning ``self.debug``.
        See :meth:`_compute_nis_statistics` for the histories NIS is built from.
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
        r"""Normalized innovation squared of every filter measurement update.

        Returns
        -------
        ndarray or None
            One dimensionless value :math:`\nu_k^T S_k^{-1}\nu_k` per
            measurement update, in processing order. A singular :math:`S_k`
            gives NaN. None when the filter stores no innovations (covariance
            analysis, a filter other than LKF, or ``save_nis_statistics``
            off during the pass).

        Raises
        ------
        DebugHistoryError
            If the filter's innovation and innovation covariance histories
            differ in length.

        Notes
        -----
        :math:`\nu_k = y_k - H_k\bar{x}_k` is the pre-update innovation and
        :math:`S_k = H_k\bar{P}_k H_k^T + R_k` its covariance, read from the
        filter's ``innovations`` and ``innovation_covariances``, which hold one
        entry per update. Post-fit residuals do not give the statistic: they
        are correlated with the update and, once unwrapped, no longer grouped
        by update. For a consistent filter each value is chi-squared with as
        many degrees of freedom as measurements in its update.
        """
        innovations = getattr(self.filter, "innovations", None)
        covariances = getattr(self.filter, "innovation_covariances", None)
        if not innovations or covariances is None:
            return None
        if len(innovations) != len(covariances):
            raise SolutionOD.DebugHistoryError(
                f"{len(innovations)} innovations but {len(covariances)} innovation "
                "covariances; both must hold one entry per measurement update."
            )

        nis_values = []
        for nu, S in zip(innovations, covariances):
            nu = np.asarray(nu, dtype=float).ravel()
            try:
                nis_values.append(float(nu @ np.linalg.solve(np.asarray(S, dtype=float), nu)))
            except np.linalg.LinAlgError:
                nis_values.append(np.nan)

        return np.array(nis_values)

    def propagate_covariance(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
        *,
        lookback_s: Optional[float] = None,
        use_all_data: bool = False,
        include_consider: bool = True,
        apriori_covariance: Optional[np.ndarray] = None,
        apriori_epoch: Optional[float] = None,
    ) -> List[np.ndarray]:
        r"""Map covariance to arbitrary epochs using the trajectory STMs.

        With ``lookback_s=0`` the forward product uses the last update at or
        before each epoch; a positive lookback moves that data cutoff earlier.
        Reconstruction explicitly selects the final update with ``use_all_data``.
        In both cases :math:`P(t)=\Phi(t,t_k)P(t_k)\Phi(t,t_k)^T` with the full
        STM, including considered parameters and their cross-covariances.
        No process noise or measurement updates are added during this mapping.

        Stored STMs are used throughout their coverage, including before the
        first measurement and after the last one. For a continuous arc, missing
        or insufficiently accurate STMs are integrated on a copy of the
        propagator along the same nominal reference; this never changes the
        selected covariance or the filter's propagator.

        Parameters
        ----------
        epochs : EpochArray
            Epochs where you want the propagated covariance.
        use_smoothed : bool, optional
            If True and a smoothed solution is available, uses the smoothed covariance
            history (``covariance_smooth``) instead of the forward-filter history
            (``covariance_est``). Defaults to True. Set False for a causal
            filtered product or rolling forecast.
        lookback_s : float or None, optional
            Nonnegative interval between the plotted epoch and its data cutoff.
            Zero uses measurements through t; 14*86400 uses measurements only
            through t - 14 days. A cutoff cannot use a smoothed history, and
            a positive lookback cannot be combined with ``use_all_data=True``.
            Currently supported for continuous arcs. With None (the default),
            preserve the original mapping convention: continuous arcs use the
            final covariance inside/after the measurement arc and the first
            covariance before it; sequences use the last timeline update.
            This default is not a causal filtered product.
        use_all_data : bool, optional
            Map the final covariance to every requested epoch (deterministic
            reconstruction). Defaults to False. Currently requires a continuous
            arc; this is not a stochastic backward smoother.
        include_consider : bool, optional
            Include the consider-augmented covariance when the filter has
            considered parameters (on a sequence, on the legs that consider
            any; the others have no augmented one). Defaults to True. No smoothed augmented
            history exists, and the filtered one is only stored with
            ``save_consider_covariance_history``; in either case the
            estimated-only covariance is mapped and a RuntimeWarning is issued.
        apriori_covariance : ndarray, optional
            Prior for cutoffs before the first update. Defaults to the filter's
            initial covariance. A missing prior raises instead of using a
            posterior containing future measurements.
        apriori_epoch : float, optional
            TDB epoch of the prior. Defaults to the filter's initial state epoch.

        Returns
        -------
        list of ndarray
            Covariances in requested epoch order, including repeated epochs.
            Each matrix follows the full nominal state order and frames;
            element (i,j) has the product of component i and j units. Sequence
            legs may have different matrix sizes. Empty input returns [].

        Raises
        ------
        ValueError
            If histories or trajectory are unavailable; timeline lengths,
            epochs, cutoffs or dimensions are invalid; an explicit cutoff uses
            smoothing; or the required prior/consider covariance is missing.
        NotImplementedError
            For positive lookback or all-data reconstruction across sequence
            events, which require event transition maps.
        numpy.linalg.LinAlgError
            If a reference STM is singular.

        Notes
        -----
        For a selected covariance P(r), define A = Phi(t,t0) Phi(r,t0)^(-1).
        The mapped covariance is P(t) = A P(r) A.T, explicitly symmetrized to
        remove floating-point asymmetry. Mapping adds no process noise and
        performs no new measurement update. All-data reconstruction is a
        deterministic mapping of the final covariance, not a stochastic
        backward smoother. All STMs in a ratio use the same nominal reference.

        Examples
        --------
        For an existing solution and target EpochArray ``epochs``::

            filtered = solution.propagate_covariance(
                epochs, use_smoothed=False, lookback_s=0.0)
            prediction_14d = solution.propagate_covariance(
                epochs, use_smoothed=False, lookback_s=14 * 86400.0)
            reconstructed = solution.propagate_covariance(
                epochs, use_smoothed=False, use_all_data=True)

        See Also
        --------
        propagate_state, propagate_state_covariance
        """
        if epochs is None or getattr(epochs, "size", 0) == 0:
            return []
        legacy_reference = lookback_s is None
        lookback_s = 0.0 if lookback_s is None else float(lookback_s)
        if not np.isfinite(lookback_s) or lookback_s < 0:
            raise ValueError("lookback_s must be finite and nonnegative.")
        if use_all_data and lookback_s != 0:
            raise ValueError("use_all_data and lookback_s cannot be combined.")
        is_sequence = getattr(self.filter, "flag_sequence", False)
        if is_sequence and (lookback_s != 0 or use_all_data):
            raise NotImplementedError(
                "Rolling cutoffs and reconstruction across sequence events "
                "require event transition maps; use a continuous arc."
            )

        traj = getattr(self.filter, "trajectory", None)
        if traj is None:
            raise ValueError("Filter has no trajectory attached.")

        smoothing = (use_smoothed and self.covariance_smooth is not None
                     and len(self.covariance_smooth) > 0)
        if smoothing and not legacy_reference and not use_all_data:
            raise ValueError("A rolling data cutoff cannot use smoothed covariance.")
        # -------------------------
        # Helpers
        # -------------------------
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

        def _get_stm_seq(traj_obj, t, idx_leg):
            return _as_matrix(
                traj_obj.get_STM_sequence(epoch=float(t), idx_leg=idx_leg)
            )

        def _propagate_from_reference(t_ref, P_ref, t_targets) -> dict:
            if len(t_targets) == 0:
                return {}
            leg = self._leg_index_for_epoch(float(t_ref))
            stms, _ = self._integrate_reference([t_ref, *t_targets], leg)
            return {float(t): _prop(P_ref, _phi_ratio(stms[float(t)], stms[t_ref]))
                    for t in t_targets}

        # -------------------------
        # Setup
        # -------------------------
        t_req = np.asarray(epochs.et, dtype=float).ravel()
        if not np.isfinite(t_req).all():
            raise ValueError("Covariance mapping epochs must be finite.")
        t_hist = self._timeline_epochs()
        P_hist = self.covariance_smooth if smoothing else self.covariance_est
        P_consider = self.covariance_consider
        if P_consider is None or len(P_consider) == 0:
            P_consider = getattr(self.filter, "covariance_consider_history", None)
        if include_consider and (getattr(self.filter, "has_consider", False)
                                 or self._has_consider_rows(P_consider)):
            if smoothing:
                reason = "no smoothed consider-augmented covariance exists"
            elif not self._has_consider_rows(P_consider):
                reason = "no consider-augmented covariance history was saved"
            elif P_hist is None and any(P_c is None for P_c in P_consider):
                reason = "the legs without considered states have no saved covariance"
            else:
                reason = None
                # legs without considered states have no consider row: their
                # covariance is already the full one
                P_hist = list(P_consider) if P_hist is None else [
                    P if P_c is None else P_c for P, P_c in zip(P_hist, P_consider)]
            if reason is not None:
                warnings.warn(
                    f"Mapping the {'smoothed' if smoothing else 'filtered'} "
                    f"covariance without considered-parameter uncertainty: "
                    f"{reason}. Pass include_consider=False to silence this.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if P_hist is None or len(P_hist) == 0:
            raise ValueError("No covariance history available.")
        if t_hist is None or len(t_hist) != len(P_hist):
            raise ValueError("Covariance history and filter timeline lengths differ.")
        if not np.isfinite(t_hist).all():
            raise ValueError("Covariance history epochs must be finite.")
        # Simultaneous updates retain their filter order; the last one wins.
        order = np.argsort(t_hist, kind="stable")
        t_hist = t_hist[order]
        P_hist = [np.asarray(P_hist[i], dtype=float) for i in order]

        # =========================================================
        # NON-SEQUENCE CASE
        # =========================================================
        if not is_sequence:
            propagator = self.filter.propagator
            initial_state = propagator.full_state_vector
            epoch0 = float(np.asarray(initial_state.epoch.et).ravel()[0])
            prior_t = epoch0 if apriori_epoch is None else float(apriori_epoch)
            if apriori_covariance is None:
                prior = getattr(self.filter, "initial_covariance", None)
                if prior is None:
                    prior = getattr(getattr(self.filter, "settings", None),
                                    "initial_covariance", None)
                if prior is not None:
                    apriori_covariance = prior.matrix_without_units()

            references = []
            for t in t_req:
                index = (len(t_hist) - 1 if (use_all_data or
                         (legacy_reference and t >= t_hist[0])) else
                         int(np.searchsorted(t_hist, t - lookback_s, side="right")) - 1)
                if index < 0 and (smoothing or legacy_reference):
                    # Preserve boundary mapping for the legacy/smoothed product.
                    index = 0
                if index < 0:
                    if apriori_covariance is None:
                        raise ValueError("A prior covariance is required before the first update.")
                    references.append((prior_t, np.asarray(apriori_covariance, float)))
                else:
                    references.append((float(t_hist[index]), P_hist[index]))

            # Look up by epoch, never by measurement index on an unrelated STM
            # grid. Cache repeated reference STMs across mapping epochs.
            needed = {float(t) for t, (base_t, _) in zip(t_req, references) if t != base_t}
            needed.update(base_t for t, (base_t, _) in zip(t_req, references) if t != base_t)
            stms, missing = {}, []
            for t in sorted(needed):
                try:
                    stms[t] = _as_matrix(traj.get_STM(epoch=t))
                except ValueError as error:
                    message = str(error).lower()
                    if ("interpolation error" in message
                            or "cannot extrapolate" in message):
                        missing.append(t)
                    else:
                        raise

            if missing:
                from copy import deepcopy
                from scarabaeus import Trajectory

                # Reintegrate from the actual nominal state epoch, not from a
                # boundary time with the initial state's coordinates relabelled.
                # All STMs in a ratio must come from the same integration.
                stms = {epoch0: np.eye(P_hist[0].shape[0])}
                for backward in (True, False):
                    targets = sorted(t for t in needed
                                     if (t < epoch0 if backward else t > epoch0))
                    if not targets:
                        continue
                    times = np.unique(np.r_[epoch0, targets])
                    prop = deepcopy(propagator)
                    prop.reinitialize(
                        state_vector=deepcopy(initial_state),
                        tspan=_build_epoch_array(times[::-1] if backward else times),
                        backward=backward, propagate_STM=True,
                        STM0=np.eye(P_hist[0].shape[0]),
                    )
                    prop.propagate(display_progress=False)
                    tmp_traj = Trajectory(
                        "tmp_covariance_propagation",
                        state_array=prop.propagated_state_array,
                    )
                    tmp_traj.add_STMs(prop.STM, timestamps=prop.propagated_state_array.epoch)
                    for t in targets:
                        stms[t] = _as_matrix(tmp_traj.get_STM(epoch=t))

            out = []
            for t, (base_t, P_ref) in zip(t_req, references):
                if t == base_t:
                    out.append(0.5 * (P_ref + P_ref.T))
                else:
                    out.append(_prop(P_ref, _phi_ratio(stms[float(t)], stms[base_t])))
            return out

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
            # Shared event epochs belong to the newly initialized leg.
            for i in reversed(range(nlegs)):
                t0, t1 = _leg_bounds(i)
                if t0 - 1e-12 <= float(t) <= t1 + 1e-12:
                    return i
            if float(t) < float(_leg_bounds(0)[0]):
                return -1
            return nlegs

        # Use the actual covariance-row timeline, including event epochs and
        # simultaneous updates, rather than rebuilding a unique measurement grid.
        _timeline = t_hist

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
            if t_ref == float(t):
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
                if "interpolation error" in str(e).lower():
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
                    float(t_hist[0]), np.asarray(P_hist[0], float), t_before_all
                )
            )

        if len(t_after_all) > 0:
            out_map.update(
                _propagate_from_reference(
                    float(t_hist[-1]), np.asarray(P_hist[-1], float), t_after_all
                )
            )

        return [out_map[float(t)] for t in t_req]

    def propagate_state(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
        *,
        lookback_s: Optional[float] = None,
        use_all_data: bool = False,
        apriori_deviation=None,
        apriori_epoch: Optional[float] = None,
    ) -> List[StateArray]:
        r"""Map deviations with STMs and add them to the nominal state once.

        Parameters
        ----------
        epochs : EpochArray
            Target epochs, converted to TDB seconds internally. None or an empty
            array returns an empty list. Input order and duplicates are retained.
        use_smoothed : bool, optional
            Use a nonempty smoothed deviation history when available; otherwise
            use filtered deviations. Defaults to True.
        lookback_s : float or None, optional
            Nonnegative data-cutoff interval in seconds. Zero selects the last
            update at or before t; positive values select the last update at or
            before t-lookback_s. None preserves the legacy convention: continuous
            arcs use the final update inside/after the measurement arc and the
            first update before it; sequences use the last timeline update.
            The default is not a causal filtered estimate.
        use_all_data : bool, optional
            Reconstruct every target from the final update. Defaults to False.
            Cannot be combined with positive lookback. Continuous arcs only.
        apriori_deviation : array_like, optional
            Prior correction for an explicit cutoff before the first update,
            in full nominal component order and units. Defaults to the filter's
            initial deviation, then its settings, then a zero vector.
        apriori_epoch : float, optional
            TDB epoch of the prior correction. Defaults to the nominal initial
            state epoch. Ignored when all references are stored updates.

        Returns
        -------
        list of StateArray
            One corrected state per target. Component units, frames, owners and
            ordering follow the nominal model. Sequence leg sizes may differ.

        Raises
        ------
        ValueError
            If a history or trajectory is absent; histories, epochs, cutoffs or
            dimensions are invalid; an explicit cutoff selects smoothed data;
            or reference and target belong to different sequence legs.
        NotImplementedError
            For positive lookback or all-data reconstruction on sequences;
            these operations require event transition maps.
        numpy.linalg.LinAlgError
            If the reference STM is singular.

        Notes
        -----
        For a selected correction at r and STMs sharing nominal origin t0,

        .. math::

            \hat x(t) = x_{\mathrm{nom}}(t)
                + \Phi(t,t_0)\Phi(r,t_0)^{-1}\delta x_r.

        This is linearized mapping about the fitted reference. It performs no
        new filter update, adds no process noise, and is not a nonlinear
        integration of the corrected state. The correction is added once.
        Missing coverage or rejected STM interpolation triggers nominal
        reintegration on copies, preserving the selected reference update and
        the live filter. Shared sequence boundaries belong to the later leg.

        Examples
        --------
        For an existing solution and target EpochArray ``epochs``::

            filtered = solution.propagate_state(
                epochs, use_smoothed=False, lookback_s=0.0)
            prediction_14d = solution.propagate_state(
                epochs, use_smoothed=False, lookback_s=14 * 86400.0)
            reconstructed = solution.propagate_state(
                epochs, use_smoothed=False, use_all_data=True)

        See Also
        --------
        propagate_covariance : Covariance mapping with the same reference policy.
        propagate_state_covariance : Return matched state/covariance pairs.
        estimated_trajectory : Read each measurement epoch's own filter update.
        """
        if epochs is None or epochs.size == 0:
            return []
        t_hist, deviations, smoothing = self._deviation_history(use_smoothed)
        traj = getattr(self.filter, "trajectory", None)
        if traj is None:
            raise ValueError("Filter has no trajectory attached.")
        legacy_reference = lookback_s is None
        lookback_s = 0.0 if legacy_reference else float(lookback_s)
        if not np.isfinite(lookback_s) or lookback_s < 0:
            raise ValueError("lookback_s must be finite and nonnegative.")
        if use_all_data and lookback_s != 0:
            raise ValueError("use_all_data and lookback_s cannot be combined.")
        if smoothing and not legacy_reference and not use_all_data:
            raise ValueError("A rolling data cutoff cannot use smoothed deviations.")
        sequence = getattr(self.filter, "flag_sequence", False)
        if sequence and (lookback_s != 0 or use_all_data):
            raise NotImplementedError(
                "Rolling cutoffs and reconstruction across sequence events "
                "require event transition maps; use a continuous arc.")
        t_req = np.asarray(epochs.et, float).ravel()
        if not np.isfinite(t_req).all():
            raise ValueError("State mapping epochs must be finite.")

        # Group by leg so local STMs are never mixed across a definition change.
        groups = {}
        for t in t_req:
            leg = self._leg_index_for_epoch(t) if sequence else None
            if sequence and leg is None:
                first = float(np.asarray(self.filter.legs_epochs[0]).ravel()[0])
                if t > first:
                    raise ValueError(f"Epoch {t} lies in a gap between sequence legs.")
                leg = 0
            index = (len(t_hist) - 1 if (not sequence and (use_all_data or
                     (legacy_reference and t >= t_hist[0]))) else
                     int(np.searchsorted(t_hist, t - lookback_s, side="right")) - 1)
            if index < 0 and (smoothing or legacy_reference):
                index = 0
            model = self._reference_model(leg)
            if index < 0:
                if sequence:
                    raise ValueError("No state update in this sequence leg at the requested cutoff.")
                base_t = (float(np.asarray(model.epoch.et).ravel()[0])
                          if apriori_epoch is None else float(apriori_epoch))
                prior = apriori_deviation
                if prior is None:
                    prior = getattr(self.filter, "initial_deviation", None)
                if prior is None:
                    prior = getattr(getattr(self.filter, "settings", None), "initial_deviation", None)
                dx = np.zeros(model.size) if prior is None else np.asarray(prior, float).ravel()
            else:
                base_t, dx = float(t_hist[index]), deviations[index]
                if sequence:
                    if self._leg_index_for_epoch(base_t) != leg:
                        raise ValueError("State reference and target belong to different sequence legs.")
                    dx = dx[:model.size]  # sequence storage pads shorter definitions
            if not np.isfinite(base_t) or not np.isfinite(dx).all():
                raise ValueError("State reference epoch and deviation must be finite.")
            groups.setdefault(leg, []).append((float(t), base_t, dx))

        result = {}
        for leg, requests in groups.items():
            needed = {t for target, base, _ in requests for t in (target, base)}
            try:
                stms = {}
                for target, base, _ in requests:
                    if target == base:
                        continue
                    for t in (target, base):
                        if t not in stms:
                            value = (traj.get_STM(epoch=t) if leg is None else
                                     traj.get_STM_sequence(epoch=t, idx_leg=leg))
                            stms[t] = np.asarray(value, float)
                nominal = {t: self._nominal_state(t, leg) for t, _, _ in requests}
            except ValueError as error:
                message = str(error).lower()
                # Sequence STM lookups say "(no extrapolation)" off the leg grid.
                if not any(key in message for key in
                           ("interpolation error", "cannot extrapolate", "no extrapolation")):
                    raise
                # Rebuild *both* numerator and denominator on one reference.
                stms, nominal = self._integrate_reference(needed, leg)
            for t, base_t, dx in requests:
                mapped = dx if t == base_t else stms[t] @ np.linalg.solve(stms[base_t], dx)
                result[t] = self._correct_state(nominal[t], mapped)
        return [result[float(t)] for t in t_req]

    def propagate_state_covariance(
        self,
        epochs: EpochArray,
        use_smoothed: bool = True,
        *,
        lookback_s: Optional[float] = None,
        use_all_data: bool = False,
        include_consider: bool = True,
        apriori_deviation=None,
        apriori_covariance=None,
        apriori_epoch: Optional[float] = None,
    ) -> List[tuple]:
        """Map state and covariance using the same data cutoff and smoothing.

        Parameters
        ----------
        epochs : EpochArray
            Target epochs. None or an empty array returns an empty list.
        use_smoothed : bool, optional
            Use both smoothed histories if available. Defaults to True. If
            neither is stored, use filtered histories. If only one is stored,
            raise instead of mixing smoothed and filtered solution products.
        lookback_s : float or None, optional
            Data-cutoff interval in seconds, as in :meth:`propagate_state`.
            None preserves legacy reference selection; zero uses data through
            t; positive values use data through t-lookback_s.
        use_all_data : bool, optional
            Reconstruct every target from the final update. Defaults to False;
            cannot be combined with positive lookback. Continuous arcs only.
        include_consider : bool, optional
            Include consider-augmented covariance when the filter has considered
            parameters. Defaults to True. Falls back to the estimated-only
            covariance, with a RuntimeWarning, when no augmented history fits
            (always the case when smoothing).
        apriori_deviation : array_like, optional
            Prior correction in nominal component units for cutoffs before data.
            Defaults to the filter's initial deviation or zero.
        apriori_covariance : array_like, optional
            Full prior covariance for cutoffs before data. Defaults to the
            filter's initial covariance. Element (i,j) has the product of
            component i and j units, in the same order as the nominal state.
        apriori_epoch : float, optional
            Shared TDB epoch of the prior deviation and covariance. Defaults to
            the nominal initial state epoch.

        Returns
        -------
        list of tuple of (StateArray, ndarray)
            One pair per target in request order, retaining duplicates. Each
            covariance has shape (state.size, state.size) and uses its state's
            component ordering, frames and units.

        Raises
        ------
        ValueError
            If only one smoothed history is stored, output counts/dimensions
            differ, or either mapper rejects the input/history.
        NotImplementedError
            For cutoffs/reconstruction requiring sequence event transition maps.
        numpy.linalg.LinAlgError
            If an STM needed by either mapper is singular.

        Notes
        -----
        Both quantities are mapped about the fitted reference. Mapping adds no
        process noise; reconstruction is not a stochastic backward smoother.
        See the individual mappers for prior selection and integration fallback.

        Examples
        --------
        Return causal state/covariance pairs without later measurements::

            pairs = solution.propagate_state_covariance(
                epochs, use_smoothed=False, lookback_s=0.0)
            for state, covariance in pairs:
                sigma = np.sqrt(np.diag(covariance))

        See Also
        --------
        propagate_state, propagate_covariance, state_covariance_to_elements
        """
        if epochs is None or epochs.size == 0:
            return []
        smooth_state = self.deviation_smooth is not None and len(self.deviation_smooth) > 0
        smooth_cov = self.covariance_smooth is not None and len(self.covariance_smooth) > 0
        if use_smoothed and smooth_state != smooth_cov:
            raise ValueError("Both smoothed deviations and covariance are required for a paired mapping.")
        common = dict(use_smoothed=use_smoothed, lookback_s=lookback_s,
                      use_all_data=use_all_data, apriori_epoch=apriori_epoch)
        states = self.propagate_state(epochs, apriori_deviation=apriori_deviation, **common)
        covariances = self.propagate_covariance(
            epochs, include_consider=include_consider,
            apriori_covariance=apriori_covariance, **common)
        if len(states) != len(covariances) or len(states) != epochs.size:
            raise ValueError("State and covariance mappings returned different epoch counts.")
        for state, covariance in zip(states, covariances):
            if covariance.shape != (state.size, state.size):
                raise ValueError("Mapped state and covariance dimensions differ.")
        return list(zip(states, covariances))

    def state_covariance_to_elements(
        self,
        epochs: Optional[EpochArray] = None,
        body=None,
        grav_param=None,
        use_smoothed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Express the estimate and its covariance in Keplerian elements.

        Pairs the estimated state and covariance with
        :meth:`~scarabaeus.OrbitalElements.covariance_to_elements`, so the
        result is the same solution read in ``(a, e, i, RAAN, argp, M)``
        instead of Cartesian components. The state and covariance are taken
        from :meth:`propagate_state_covariance`, which keeps the two consistent
        with each other and with ``use_smoothed``.

        Parameters
        ----------
        epochs : EpochArray, optional
            Epochs to report. Defaults to the solution's measurement epochs,
            which for a long arc means one state and covariance propagation per
            measurement — pass a coarser set if that is more than you need.
        body : Body or int, optional
            Which object in the state vector to convert, as the body itself or
            its SPICE ID. Only needed when more than one body carries a
            position, as in a spacecraft-plus-asteroid estimate.
        grav_param : float, ArrayWUnits or CelestialBody, optional
            Gravitational parameter of the attracting body. Defaults to that of
            the state's own origin.
        use_smoothed : bool, optional
            Passed through to :meth:`propagate_state_covariance`. Defaults to
            True.

        Returns
        -------
        times : numpy.ndarray
            Epochs, seconds past J2000 TDB, one per row of ``elements``.
        elements : numpy.ndarray
            ``(n_epochs, 6)`` mean elements, ordered by
            :attr:`~scarabaeus.OrbitalElements.MEAN_ELEMENT_NAMES`, in km and
            radians.
        covariances : list of numpy.ndarray
            One ``(6, 6)`` element covariance per epoch.

        Raises
        ------
        ElementConversionError
            If ``body`` is ambiguous or absent from the state vector, or the
            gravitational parameter cannot be resolved.

        See Also
        --------
        scarabaeus.OrbitalElements.covariance_to_elements : The transformation applied here.
        SolutionOD.propagate_state_covariance : Supplies the state and covariance.

        Notes
        -----
        The mapping is linearised about the estimate, so it holds while the
        Cartesian uncertainty is small against the curvature of the element
        map. A near-circular or near-equatorial orbit leaves ``argp`` and
        ``RAAN`` weakly defined, and their variances inflate accordingly —
        that is the geometry showing through, not a defect in the transform.

        The elements are only Keplerian in an inertial frame centred on the
        attracting body; reading them off a state expressed otherwise gives
        numbers without the usual meaning.

        References
        ----------
        .. [1] Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods and
           Applications. Springer. ISBN 978-3-540-67280-7.
        """
        from scarabaeus import OrbitalElements

        if epochs is None:
            epochs = EpochArray(np.asarray(self.timestamps, dtype=float), sys="TDB")

        pairs = self.propagate_state_covariance(epochs, use_smoothed=use_smoothed)
        if not pairs:
            return np.zeros(0), np.zeros((0, 6)), []

        mu = self._resolve_grav_param(pairs[0][0], grav_param)
        km = Units.get_units("km")
        km_per_sec = km / sec

        times = np.zeros(len(pairs))
        elements = np.zeros((len(pairs), 6))
        covariances = []
        for k, (state, cov) in enumerate(pairs):
            (position, velocity), indices = self._body_state_entries(state, body)
            r = position.convert_to(km).values.ravel()
            v = velocity.convert_to(km_per_sec).values.ravel()
            # The covariance is ordered like the flat state vector, so the same
            # indices cut this body's block out of a multi-body matrix.
            block = np.asarray(cov, dtype=float)[np.ix_(indices, indices)]
            el, cov_el, _ = OrbitalElements.covariance_to_elements(mu, r, v, block)
            times[k] = float(np.asarray(state.epoch.times.values).ravel()[0])
            elements[k] = el
            covariances.append(cov_el)

        return times, elements, covariances

    @staticmethod
    def _body_state_entries(state: StateArray, body=None):
        """
        Pick one body's position and velocity out of the state vector.

        Walks the state components once, so the values and the rows they occupy
        in the flat state vector — and therefore in the covariance — come back
        together.

        Parameters
        ----------
        state : StateArray
            State to search.
        body : Body or int, optional
            The body, or its SPICE ID. May be omitted when the state carries a
            single position.

        Returns
        -------
        quantities : tuple of ArrayWUnits
            Position and velocity, in whatever units the state carries them.
        indices : numpy.ndarray
            The six rows those components occupy, position first.

        Raises
        ------
        ElementConversionError
            If the body is absent, or omitted while the state holds several.

        Notes
        -----
        The values cannot be read off ``StateArray.values0`` by index: that
        array carries one unit per component rather than one per element, so
        indexing it element-wise walks off the end of the units.
        """
        spice_id = getattr(body, "spice_id", body)
        found = {"position": [], "velocity": []}
        start = 0
        for entry in state.state:
            name, size, owner = entry[0], entry[1], entry[4]
            if name in found:
                owner_id = getattr(owner, "spice_id", None)
                if spice_id is None or owner_id == spice_id:
                    found[name].append(
                        (entry[5].quantity, np.arange(start, start + size), owner_id)
                    )
            start += size

        for name, matches in found.items():
            if len(matches) == 1:
                continue
            if not matches:
                raise SolutionOD.ElementConversionError(
                    f"no '{name}' for body {spice_id!r} in the state vector."
                )
            raise SolutionOD.ElementConversionError(
                f"the state vector carries {len(matches)} '{name}' components, "
                f"owned by {[owner_id for _, _, owner_id in matches]}; name the "
                "one to convert with the 'body' argument."
            )

        position, position_rows, _ = found["position"][0]
        velocity, velocity_rows, _ = found["velocity"][0]
        return (position, velocity), np.concatenate([position_rows, velocity_rows])

    @staticmethod
    def _resolve_grav_param(state: StateArray, grav_param=None) -> float:
        """
        Gravitational parameter to build elements with, in km^3/s^2.

        Parameters
        ----------
        state : StateArray
            State whose ``origin`` supplies the default.
        grav_param : float, ArrayWUnits or CelestialBody, optional
            An explicit value, or a body to read it from. A bare number is
            interpreted as km^3/s^2; a unit-bearing value is converted. Defaults
            to the state's origin.

        Returns
        -------
        float
            The gravitational parameter in km^3/s^2.

        Raises
        ------
        ElementConversionError
            If the selected body/value has no gravitational parameter.
        TypeError or ValueError
            If the resolved value cannot be converted to a floating-point scalar.

        Notes
        -----
        Unit-conversion errors propagate to the caller. For an ArrayWUnits
        value, only the first flattened entry is read; this helper does not
        check positivity or finiteness of the resulting gravitational parameter.
        """
        if grav_param is None:
            grav_param = state.origin
        # A body was handed in rather than a number.
        grav_param = getattr(grav_param, "grav_param", grav_param)
        if grav_param is None:
            raise SolutionOD.ElementConversionError(
                f"no gravitational parameter for origin {state.origin!r}; pass one "
                "with the 'grav_param' argument."
            )
        if isinstance(grav_param, ArrayWUnits):
            grav_param = grav_param.convert_to(Units.get_units("km") ** 3 / sec**2)
            return float(np.asarray(grav_param.values).ravel()[0])
        return float(grav_param)

    def create_JSON(
        self,
        filepath: Union[str, Path],
        fmt: str = None,
    ) -> None:
        """
        Export solution results to a JSON or pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.
        fmt : {"json", "pickle"}, optional
            On-disk format. When ``None`` (default) it is inferred from the file
            extension (``.pkl`` gives pickle, otherwise JSON); when given explicitly,
            the file extension is normalised to match.

        Notes
        -----
        Holds the same fields as the ``final`` block of :meth:`save`, with
        the metadata of :meth:`save` plus ``n_epochs`` at the top level, but
        no iteration history.
        """
        filepath = with_format_suffix(filepath, fmt)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        output_dict = {
            "metadata": {
                "n_epochs": self.n_epochs,
                "n_states": self.n_states,
                "is_covariance_analysis": self.is_covariance_analysis,
                **self._output_metadata(
                    "full" if self._full_covariance_output(True) else "diagonals"
                ),
            },
            **self._solution_to_dict(final=True),
            "epoch_solution": self._guarded_epoch_solution(),
        }

        dump(output_dict, filepath, fmt=fmt)
        print(f"[SolutionOD] Saved → {filepath}")

    def UQ_compute(
        self,
        particle_states: Optional[np.ndarray] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict:
        """
        Compute and store weighted uncertainty metrics for particle filters.

        Computes mean, std, covariances, high-order statistical moments (skewness, kurtosis),
        weighted percentiles, and effective sample size from particle distributions.

        Parameters
        ----------
        particle_states : np.ndarray, optional
            Finite states with nonempty shape (n_particles, n_states, n_epochs).
            If None, read ``filter.particles``. Values use each state component's
            native units.
        confidence_levels : list of float, optional
            Central interval probabilities strictly between 0 and 1. Defaults
            to [0.68, 0.95, 0.997]. Bounds use weighted empirical quantiles.

        Returns
        -------
        dict
            Mean, standard deviation, skewness and excess kurtosis arrays of
            shape (n_epochs, n_states); a list of (n_states, n_states)
            population covariances; ``percentiles[level]`` arrays of shape
            (n_epochs, n_states, 2), lower bound first; and an
            ``effective_sample_size`` array of shape (n_epochs,). Mean, standard
            deviation and bounds use component units; covariance uses their
            pairwise products. The same dictionary is saved as ``uq_metrics``.

        Raises
        ------
        ValueError
            If particle data is missing/nonfinite or its shape is invalid;
            weights are invalid or have zero mass; or levels are outside (0,1).

        Notes
        -----
        ``filter.weights`` may have shape (n_particles,) for static weights or
        (n_particles, n_epochs) for time-dependent weights. Missing weights
        imply a uniform distribution. Weights must be finite and nonnegative;
        they are normalized separately at each epoch without changing the
        filter. Covariance uses the weighted population moment (no N-1 sample
        correction), so its diagonal agrees with the squared returned standard
        deviations. Effective sample size is 1 / sum(w**2) using normalized w.
        Zero-variance components have zero skewness and excess kurtosis.
        """
        if particle_states is None:
            if hasattr(self.filter, "particles"):
                particle_states = self.filter.particles
            else:
                raise ValueError("No particle data available for UQ computation")

        particle_states = np.asarray(particle_states, dtype=float)
        if (particle_states.ndim != 3 or min(particle_states.shape) == 0
                or not np.isfinite(particle_states).all()):
            raise ValueError("Particles must be finite with nonempty shape (particles, states, epochs).")
        confidence_levels = list([0.68, 0.95, 0.997] if confidence_levels is None else confidence_levels)
        if any(not np.isfinite(level) or not 0 < level < 1 for level in confidence_levels):
            raise ValueError("Confidence levels must be finite and strictly between zero and one.")
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
        if weights is None:
            weights = np.ones((n_particles, n_epochs))
        else:
            weights = np.asarray(weights, float)
            if weights.shape == (n_particles,):
                weights = np.broadcast_to(weights[:, None], (n_particles, n_epochs))
        if (weights.shape != (n_particles, n_epochs) or not np.isfinite(weights).all()
                or (weights < 0).any() or (weights.max(axis=0) <= 0).any()):
            raise ValueError("Weights must be finite, nonnegative and have positive mass at each epoch.")
        # Scale first to avoid overflow when unnormalized weights are large.
        weights = weights / weights.max(axis=0)
        weights = weights / weights.sum(axis=0)

        for k in range(n_epochs):
            particles_k = particle_states[:, :, k]

            w_k = weights[:, k]

            # Mean, std, covariance
            mean_k = np.average(particles_k, weights=w_k, axis=0)
            var_k = np.average((particles_k - mean_k) ** 2, weights=w_k, axis=0)
            std_k = np.sqrt(var_k)
            means[k] = mean_k
            stds[k] = std_k
            centered = particles_k - mean_k
            # Population covariance of the weighted empirical distribution,
            # consistent with the returned standard deviations (also for N=1).
            covariances.append((centered.T * w_k) @ centered)

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
        uq_results["effective_sample_size"] = 1.0 / np.sum(weights**2, axis=0)
        self.uq_metrics = uq_results
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

        # Particle statistics, as written by UQ_compute
        if self.uq_metrics is not None:
            ess = np.asarray(self.uq_metrics["effective_sample_size"], dtype=float)
            std_final = np.asarray(self.uq_metrics["std"], dtype=float)[-1]
            print(f"\nParticle Statistics:")
            print(f"  Number of particles:     {self.uq_metrics['n_particles']}")
            print(f"  Number of epochs:        {self.uq_metrics['n_epochs']}")
            print(f"  Mean effective size:     {float(np.mean(ess)):.1f}")
            print(f"  Final-epoch max std:     {float(np.max(std_final)):.6e}")
            print(
                f"  Final-epoch max |skew|:  "
                f"{float(np.max(np.abs(self.uq_metrics['skewness'][-1]))):.4f}"
            )
            print(
                f"  Final-epoch max |kurt|:  "
                f"{float(np.max(np.abs(self.uq_metrics['kurtosis'][-1]))):.4f}"
            )

        # Chi-squared test
        # ``postfits`` is the ``{dataset: [[residual, sigma], ...]}`` mapping;
        # the cost is the sum of the squared normalized residuals.
        chi_squared, n_measurements = (
            self.filter._cost_function(self.postfits)
            if self.postfits is not None
            else (float("nan"), 0)
        )
        degrees_of_freedom = n_measurements - self.n_states
        if n_measurements > 0 and degrees_of_freedom > 0 and verbose:
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
        and large file sizes. Stored solution arrays, any already-computed
        absolute-state cache, residual epochs, full update timeline, frame/units,
        state components, sequence layout, run metadata (a priori, convergence,
        provenance), the epoch solution and multi-arc global results are
        preserved. Serializing does not force the lazy absolute-state cache to
        be evaluated.
        Reattach the matching fitted filter on deserialization to use trajectory
        or STM methods; propagators and SPICE kernels are not embedded.
        """
        filepath = Path(filepath)

        # Run metadata and the epoch solution need the live filter; failing to
        # build them must not block the pickle.
        try:
            run_metadata = self._run_metadata()
        except Exception as exc:  # noqa: BLE001
            print(f"[SolutionOD] Run metadata not serialized: {exc}")
            run_metadata = {}

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
            "timeline": self._timeline_epochs(),
            "state_frame": self._state_frame,
            "state_units": self._state_units,
            "state_components": getattr(self, "_state_components", []),
            "run_metadata": run_metadata,
            "epoch_solution": self._guarded_epoch_solution(),
            "prefit_epochs": self.prefit_epochs,
            "postfit_epochs": self.postfit_epochs,
            "postfit_epochs_smooth": self.postfit_epochs_smooth,
            "multi_arc_global": self.multi_arc_global,
            "sequence_extras": self._sequence_extras(),
            "filter_metadata": {
                name: getattr(self.filter, name, default)
                for name, default in (("has_consider", False), ("flag_sequence", False),
                                      ("legs_epochs", None), ("legs_n", None),
                                      ("events_epochs", None))
            },
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
            Matching fitted filter, needed for trajectory queries and STM
            mappings. If None, create a metadata-only placeholder for inspecting
            stored results and exporting them again.

        Returns
        -------
        SolutionOD
            Deserialized solution object.

        Raises
        ------
        FileNotFoundError
            If file does not exist.

        Notes
        -----
        Files written before timeline, residual-epoch and frame/unit metadata
        were saved remain readable: absent metadata defaults to None and the
        measurement timestamps serve as the fallback timeline. Missing sequence
        event metadata cannot be reconstructed without the original filter.
        Loading does not initialize or rerun the supplied filter. Only load
        pickle files from trusted sources.
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
            for name, value in state_dict.get("filter_metadata", {}).items():
                setattr(filter_obj, name, value)

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
        solution.timeline = state_dict.get("timeline")
        solution._state_frame = state_dict.get("state_frame")
        solution._state_units = state_dict.get("state_units")
        solution._state_components = state_dict.get("state_components", [])
        solution._saved_run_metadata = state_dict.get("run_metadata", {})
        solution._created_utc = solution._saved_run_metadata.get(
            "provenance", {}).get("created_utc")
        solution._saved_epoch_solution = state_dict.get("epoch_solution")
        solution.prefit_epochs = state_dict.get("prefit_epochs")
        solution.postfit_epochs = state_dict.get("postfit_epochs")
        solution.postfit_epochs_smooth = state_dict.get("postfit_epochs_smooth")
        solution.multi_arc_global = state_dict.get("multi_arc_global")
        solution._saved_sequence_extras = state_dict.get("sequence_extras", {})

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
            Every epoch must match an entry in ``solution.timestamps`` to within
            one microsecond. Queries snap to the matched stored epoch.
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
            Estimated parameters at each requested epoch, or ``None`` if absent.
            For sequences, a list per leg containing only the requested rows
            for that leg, in request order (empty for unrequested legs).

        Raises
        ------
        ValueError
            If any requested epoch does not correspond to a filter measurement epoch.
            Use :meth:`propagate_state` for arbitrary epochs.
        """
        t_req = np.asarray(epochs.et, float).ravel()
        if len(t_req) == 0:
            return np.empty((0, 3)), np.empty((0, 3)), None
        t_meas = np.asarray(self.timestamps, float).ravel()
        if not np.isfinite(t_req).all():
            raise ValueError("Estimated trajectory epochs must be finite.")
        try:
            matched = [float(t_meas[self._matching_epoch_index(t_meas, t)]) for t in t_req]
        except ValueError as error:
            raise ValueError(
                "estimated_trajectory() requires filter measurement epochs; "
                "use solution.propagate_state(epochs) for arbitrary epochs."
            ) from error
        timeline, deviations, _ = self._deviation_history(use_smoothed)
        trajectory = getattr(self.filter, "trajectory", None)
        if trajectory is None:
            raise ValueError("Filter has no trajectory attached.")
        sequence = getattr(self.filter, "flag_sequence", False)
        positions, velocities, parameters = [], [], []
        leg_parameters = [[] for _ in self.filter.legs_epochs] if sequence else None
        for t in matched:
            index = self._matching_epoch_index(timeline, t)
            leg = self._leg_index_for_epoch(t) if sequence else None
            nominal = trajectory.get_state(
                epoch_input=EpochArray(t, sys="TDB"), idx_leg_input=leg)
            dx = deviations[index]
            positions.append(np.asarray(nominal["position"].values).ravel() + dx[:3])
            velocities.append(np.asarray(nominal["velocity"].values).ravel() + dx[3:6])
            p = nominal.get("parameters")
            if p is None:
                values = np.empty(0)
            else:
                values = np.asarray(p.values).ravel()
                par_dev = self._parameter_deviation(dx, *self._parameter_layout(leg, t))
                if par_dev.size < values.size:
                    raise ValueError("Nominal parameters and deviation dimensions differ.")
                values = values + par_dev[: values.size]
            parameters.append(values)
            if sequence:
                if leg is None:
                    raise ValueError(f"Measurement epoch {t} does not belong to a sequence leg.")
                leg_parameters[leg].append(values)
        if not any(len(p) for p in parameters):
            output_parameters = None
        elif sequence:
            output_parameters = leg_parameters
        else:
            output_parameters = np.stack(parameters)
        return np.asarray(positions), np.asarray(velocities), output_parameters

    def map_state_deviation_to_epoch(
        self, map_back_sequence: bool = False, use_smoothed: bool = True
    ):
        r"""Map corrections to their nominal initial epochs using STM ratios.

        Parameters
        ----------
        map_back_sequence : bool, optional
            For sequences, False (default) maps each leg's own correction to
            its start. True starts with the final leg's correction and maps
            backward through earlier legs using the filter's definition-node
            maps. Both modes return a list in forward leg order. Ignored for
            continuous arcs.
        use_smoothed : bool, optional
            Use the first smoothed epoch in each selected arc/leg if a nonempty
            smoothed history is stored. Otherwise use its last filtered epoch.
            Defaults to True. The last update wins at simultaneous epochs.

        Returns
        -------
        ndarray or list of ndarray
            Correction at the nominal initial epoch for a continuous arc, or
            one correction per sequence leg start. Each vector uses its model's
            component order, units and frames, with storage padding removed.
            If no deviation history exists, return a zero vector sized from
            ``filter.n`` (falling back to ``filter.n_states`` or 6 for minimal
            filter objects).

        Raises
        ------
        ValueError
            If the timeline is invalid, the trajectory is unavailable, or a
            sequence leg contains no deviation rows.
        numpy.linalg.LinAlgError
            If a reference STM is singular.

        Notes
        -----
        A correction at stored epoch r is mapped to initial epoch s as

        .. math::

            \delta x(s) = \Phi(s,t_0)\Phi(r,t_0)^{-1}\delta x(r).

        The first measurement may follow the initial epoch, and the stored
        initial STM need not be identity. Both are handled explicitly. Shared
        sequence nodes belong to the later leg. Mapping through nodes uses
        the filter's existing definition transformations; it does not run a
        smoother or add new observations. States a later leg drops (e.g. the
        parameters of a burn followed by a coast) are recovered from the
        forward solution at the node when the filter stored one (LKF, SRIF,
        SRIFB, LSB), see :meth:`FilterOD._map_back_across_node`.

        See Also
        --------
        propagate_state : Map and combine corrections with nominal states.
        """
        smooth = (use_smoothed and self.deviation_smooth is not None
                  and len(self.deviation_smooth) > 0)
        source = self.deviation_smooth if smooth else self.deviation_est
        if source is None or len(source) == 0:
            n = getattr(self.filter, "n", getattr(self.filter, "n_states", 6))
            return np.zeros(n)
        timeline, deviations, _ = self._deviation_history(use_smoothed)
        trajectory = getattr(self.filter, "trajectory", None)
        if trajectory is None:
            raise ValueError("Filter has no trajectory attached.")

        def map_to_start(t, dx, leg=None):
            r"""Map a deviation from a stored epoch to the initial epoch.

            Parameters
            ----------
            t : float
                Stored epoch of ``dx``, TDB seconds past J2000.
            dx : numpy.ndarray
                Deviation at ``t``, in state order.
            leg : int, optional
                Sequence leg ``t`` belongs to; the deviation is then trimmed
                to that leg's states and mapped to the leg's first epoch.
                Defaults to ``None``, a single arc mapped to the epoch of
                ``propagator.full_state_vector``.

            Returns
            -------
            numpy.ndarray
                :math:`\Phi(s,t_0)\Phi(t,t_0)^{-1}\delta x(t)`.

            Raises
            ------
            numpy.linalg.LinAlgError
                If the STM at ``t`` is singular.
            """
            if leg is None:
                initial = getattr(getattr(self.filter, "propagator", None), "full_state_vector", None)
                start = None if initial is None else float(np.asarray(initial.epoch.et).ravel()[0])
                phi = np.asarray(trajectory.get_STM(epoch=t), float)
                phi0 = np.eye(phi.shape[0]) if start is None else np.asarray(trajectory.get_STM(epoch=start), float)
            else:
                start = float(np.asarray(self.filter.legs_epochs[leg]).ravel()[0])
                phi = np.asarray(trajectory.get_STM_sequence(epoch=t, idx_leg=leg), float)
                phi0 = np.asarray(trajectory.get_STM_sequence(epoch=start, idx_leg=leg), float)
                dx = dx[:self.filter.legs_n[leg]]
            return phi0 @ np.linalg.solve(phi, dx)

        if not getattr(self.filter, "flag_sequence", False):
            index = self._matching_epoch_index(timeline, timeline[0]) if smooth else len(timeline) - 1
            return map_to_start(float(timeline[index]), deviations[index])

        indices = []
        for leg, times in enumerate(self.filter.legs_epochs):
            rows = [i for i, t in enumerate(timeline) if self._leg_index_for_epoch(t) == leg]
            # Do not treat rows after the final leg span as belonging to it.
            end = float(np.asarray(times).ravel()[-1])
            rows = [i for i in rows if timeline[i] <= end]
            if not rows:
                raise ValueError(f"No state deviation history within sequence leg {leg}.")
            index = (self._matching_epoch_index(timeline, timeline[rows[0]])
                     if smooth else rows[-1])
            indices.append(index)
        if not map_back_sequence:
            return [map_to_start(float(timeline[index]), deviations[index], leg)
                    for leg, index in enumerate(indices)]

        last = len(indices) - 1
        index = indices[last]
        result = [map_to_start(float(timeline[index]), deviations[index], last)]
        node_deviations = getattr(self, "_node_deviations", None) or []
        node_covariances = getattr(self, "_node_covariances", None) or []
        for leg in range(last, 0, -1):
            if leg - 1 < min(len(node_deviations), len(node_covariances)):
                # States the previous leg drops at the node are recovered from
                # its forward solution there, conditioned on the mapped estimate
                at_node, _ = self.filter._map_back_across_node(
                    leg, result[0], None,
                    node_deviations[leg - 1], node_covariances[leg - 1])
            else:
                at_node = self.filter._map_deviation_definition_node(
                    leg, result[0], flag_forward=False)
            end = float(np.asarray(self.filter.legs_epochs[leg - 1]).ravel()[-1])
            result.insert(0, map_to_start(end, np.asarray(at_node).ravel(), leg - 1))
        return result

    def _reference_state_values(self) -> Optional[np.ndarray]:
        """Current reference initial state as one flat vector, in state order.

        Returns
        -------
        ndarray or None
            ``propagator._x_ref0`` (a sequence's global layout) when present,
            otherwise the values of ``propagator.full_state_vector`` at its
            epoch. None if the filter has neither.

        Notes
        -----
        :meth:`FilterOD.fit` moves this reference by every correction it
        applies, so it is the latest pass's reference, not the a priori state.
        """
        prop = getattr(self.filter, "propagator", None)
        x_ref = getattr(prop, "_x_ref0", None)
        if x_ref is not None:
            return np.asarray(x_ref, dtype=float).ravel()
        state = getattr(prop, "full_state_vector", None)
        if state is None:
            return None
        return np.asarray(state.values0.values, dtype=float).ravel()

    def _initial_epoch(self) -> Optional[float]:
        """Epoch of the filter's initial state.

        Returns
        -------
        float or None
            Epoch of ``propagator.full_state_vector``, TDB seconds past J2000,
            or None when the filter has no propagator or initial state.
        """
        state = getattr(getattr(self.filter, "propagator", None), "full_state_vector", None)
        if state is None:
            return None
        return float(np.asarray(state.epoch.et).ravel()[0])

    def _apriori_metadata(self) -> Optional[Dict]:
        """Describe the a priori state and covariance the filter started from.

        Returns
        -------
        dict or None
            ``epoch`` (TDB seconds); ``state``, the first reference (current
            reference minus the corrections :meth:`FilterOD.fit` applied);
            ``reference_state_current``; ``initial_deviation`` from the filter
            settings (None if not given); ``covariance`` (P0) and its
            ``sigmas``; ``covariance_layout``; and ``extra_apriori_sigmas``
            with their units. None if the filter has neither a propagator nor
            an initial covariance.

        Notes
        -----
        States and P0 use the units and frames of ``state_components``. For a
        sequence, states use the global layout while P0 is the first leg's
        (``covariance_layout`` is ``"legs[0]"``, see ``legs[0].components``).
        """
        initial_covariance = getattr(self.filter, "initial_covariance", None)
        reference = self._reference_state_values()
        if reference is None and initial_covariance is None:
            return None

        apriori: Dict = {"epoch": self._initial_epoch()}
        if reference is not None:
            applied = getattr(self.filter, "_delta0_total", None)
            applied = (np.zeros_like(reference) if applied is None
                       else np.asarray(applied, dtype=float).ravel())
            apriori["state"] = ((reference - applied).tolist()
                                if applied.size == reference.size else None)
            apriori["reference_state_current"] = reference.tolist()
        initial_deviation = getattr(self.filter, "initial_deviation", None)
        apriori["initial_deviation"] = (
            None if initial_deviation is None
            else np.asarray(initial_deviation, dtype=float).ravel().tolist()
        )
        if initial_covariance is not None:
            P0 = np.asarray(initial_covariance.matrix_without_units(), dtype=float)
            apriori["covariance"] = P0.tolist()
            apriori["sigmas"] = np.sqrt(np.clip(np.diag(P0), 0.0, None)).tolist()
            apriori["covariance_layout"] = (
                "legs[0]" if getattr(self.filter, "flag_sequence", False)
                else "state_components"
            )
            extra = getattr(initial_covariance, "extra_apriori_sigmas", None) or {}
            apriori["extra_apriori_sigmas"] = {}
            for name, sigma in extra.items():
                try:
                    apriori["extra_apriori_sigmas"][name] = awu_to_dict(sigma)
                except Exception:  # noqa: BLE001 - units without a dict form
                    apriori["extra_apriori_sigmas"][name] = {
                        "values": np.asarray(getattr(sigma, "values", sigma)).tolist(),
                        "units": None,
                    }
        return apriori

    def _run_metadata(self) -> Dict:
        """Describe the run that produced this solution, for saved metadata.

        Returns
        -------
        dict
            Available fields among ``state_components`` and ``state_labels``
            (:meth:`_components_metadata`, :meth:`_state_labels`), ``apriori``
            (:meth:`_apriori_metadata`), ``convergence`` (the filter's
            ``convergence_info``) and ``provenance`` (filter class, package
            version, UTC time the solution was created), as JSON-native types.

        Notes
        -----
        Starts from the metadata :meth:`serialize` stored, so a solution loaded
        without its filter keeps it; an attached filter refreshes each field.
        """
        meta: Dict = dict(getattr(self, "_saved_run_metadata", {}))

        components = getattr(self, "_state_components", None)
        if components:
            meta["state_components"] = components
            meta["state_labels"] = self._state_labels(components)
        apriori = self._apriori_metadata()
        if apriori is not None:
            meta["apriori"] = apriori
        convergence = getattr(self.filter, "convergence_info", None)
        if convergence is not None:
            meta["convergence"] = to_jsonable(convergence)
            # a single pass has no previous cost: null, not Infinity
            change = meta["convergence"].get("cost_relative_change")
            if change is not None and not np.isfinite(change):
                meta["convergence"]["cost_relative_change"] = None
        if getattr(self.filter, "propagator", None) is not None or "provenance" not in meta:
            meta["provenance"] = {
                "filter": type(self.filter).__name__,
                "scarabaeus_version": getattr(scarabaeus, "__version__", None),
                "created_utc": getattr(self, "_created_utc", None),
            }
        return to_jsonable(meta)

    def _epoch_solution(self) -> Optional[Dict]:
        r"""Estimate at the nominal initial epoch, as :meth:`FilterOD.fit` applies it.

        Returns
        -------
        dict or None
            ``epoch`` (TDB seconds); ``deviation``, the correction to the
            current reference; ``deviation_cumulative``, the correction to the
            first (a priori) reference; ``state``, the corrected initial state;
            and ``covariance`` and ``covariance_consider`` at that epoch. Fields
            that cannot be formed are None. Without an attached propagator,
            the epoch solution :meth:`serialize` stored, or None.

        Raises
        ------
        ValueError
            If the deviation history, timeline or trajectory is unusable, see
            :meth:`map_state_deviation_to_epoch`.
        numpy.linalg.LinAlgError
            If a reference STM is singular.

        Notes
        -----
        The deviation is the one :meth:`FilterOD.fit` tests for convergence
        (:meth:`map_state_deviation_to_epoch`, mapped to the global layout for
        a sequence). The reference and the corrections already applied are
        read from the filter, so this describes the filter's latest solution.

        The covariance comes from the same row r as the deviation (first
        smoothed, else last filtered) and is mapped as :math:`A P A^T` with
        :math:`A = \Phi(t_0)\Phi(r)^{-1}`, then symmetrized. The consider
        covariance is mapped from a filtered row only. A covariance whose size
        differs from the STM's, and any sequence covariance (the legs' layouts
        differ), is left None. A covariance analysis has no deviation fields.
        """
        prop = getattr(self.filter, "propagator", None)
        t0 = self._initial_epoch()
        if t0 is None:
            return getattr(self, "_saved_epoch_solution", None)
        out: Dict = {
            "epoch": t0,
            "deviation": None,
            "deviation_cumulative": None,
            "state": None,
            "covariance": None,
            "covariance_consider": None,
        }
        sequence = getattr(self.filter, "flag_sequence", False)

        if not self.is_covariance_analysis and self.deviation_est is not None:
            if sequence:
                dx = self.map_state_deviation_to_epoch(
                    map_back_sequence=not self.filter._maps_back_sequence)
                dx = prop._deviation_history_to_global_dx(dx)
            else:
                dx = self.map_state_deviation_to_epoch()
            dx = np.asarray(dx, dtype=float).ravel()
            applied = getattr(self.filter, "_delta0_total", None)
            applied = (np.zeros_like(dx) if applied is None
                       else np.asarray(applied, dtype=float).ravel())
            reference = self._reference_state_values()
            out["deviation"] = dx.tolist()
            if applied.size == dx.size:
                out["deviation_cumulative"] = (applied + dx).tolist()
            if reference is not None and reference.size == dx.size:
                out["state"] = (reference + dx).tolist()

        smooth = (not self.is_covariance_analysis
                  and self.deviation_smooth is not None and len(self.deviation_smooth) > 0
                  and self.covariance_smooth is not None and len(self.covariance_smooth) > 0)
        rows = self.covariance_smooth if smooth else self.covariance_est
        timeline = self._timeline_epochs()
        if (sequence or rows is None or len(rows) == 0 or timeline is None
                or len(timeline) != len(rows)):
            return out

        order = np.argsort(timeline, kind="stable")
        ordered = timeline[order]
        k = self._matching_epoch_index(ordered, ordered[0]) if smooth else len(ordered) - 1
        r = int(order[k])
        trajectory = self.filter.trajectory
        phi = np.asarray(trajectory.get_STM(epoch=float(timeline[r])), dtype=float)
        phi0 = np.asarray(trajectory.get_STM(epoch=t0), dtype=float)
        A = phi0 @ np.linalg.solve(phi, np.eye(phi.shape[0]))

        def _map(P):
            r"""Map a covariance from the stored epoch to the initial epoch.

            Parameters
            ----------
            P : array_like
                Covariance at the stored epoch, (n, n).

            Returns
            -------
            list of list of float or None
                :math:`A P A^T` with :math:`A = \Phi(t_0)\Phi(t_r)^{-1}`,
                symmetrized, or None when ``P`` does not match the STM size.
            """
            P = np.asarray(P, dtype=float)
            if P.shape != A.shape:
                return None
            P = A @ P @ A.T
            return (0.5 * (P + P.T)).tolist()

        out["covariance"] = _map(rows[r])
        consider = self.covariance_consider
        if (not smooth and consider is not None and len(consider) == len(rows)
                and consider[r] is not None):
            out["covariance_consider"] = _map(consider[r])
        return out

    @staticmethod
    def _residuals_to_list(res_dict, epoch_dict):
        """Serialize residuals with their epochs as ``[residual, sigma, epoch]`` rows.

        Parameters
        ----------
        res_dict : dict or None
            ``{dataset: [[residual, sigma], ...]}``.
        epoch_dict : dict or None
            Parallel ``{dataset: [t2, ...]}`` reception epochs in TDB seconds.

        Returns
        -------
        dict or None
            ``{dataset: [[residual, sigma, epoch], ...]}``, a missing epoch
            written as None. None if ``res_dict`` is None.
        """
        if res_dict is None:
            return None
        epoch_dict = epoch_dict or {}
        out = {}
        for name, entries in res_dict.items():
            eps = epoch_dict.get(name, [])
            out[name] = [
                [
                    float(r),
                    float(s),
                    float(eps[j]) if j < len(eps) else None,
                ]
                for j, (r, s) in enumerate(entries)
            ]
        return out

    def _full_covariance_output(self, final: bool) -> bool:
        """Whether a solution block stores full covariances, not diagonals.

        Parameters
        ----------
        final : bool
            Whether the block is the final solution.

        Returns
        -------
        bool
            True only for the final block with
            ``output_settings.save_full_covariance`` set.
        """
        return final and bool(getattr(self.output_settings, "save_full_covariance", False))

    def _solution_to_dict(self, final: bool = False) -> Dict:
        """Collect this solution's histories for :meth:`save` and :meth:`create_JSON`.

        Parameters
        ----------
        final : bool, optional
            Whether this is the final solution, whose covariances are written
            in full when ``output_settings.save_full_covariance`` is set.
            Defaults to False.

        Returns
        -------
        dict
            ``timestamps``, the fields of :meth:`_sequence_extras`, the available
            deviation, state and covariance histories, ``prefits``/``postfits``
            (plus ``postfits_smoother``), ``nis_statistics`` when computed, and
            ``timestamps_alignment``.

        Notes
        -----
        Covariances are written either as full matrices (``covariance_full``,
        ``covariance_smooth_full``, ``covariance_consider_full``) or as their
        diagonals (``*_diagonals``), never both: the diagonals of a full matrix
        are redundant. Consider rows of epochs without consider states are None.
        """
        d = {
            "timestamps": self.timestamps.tolist(),
            **self._sequence_extras(),
        }
        if self.deviation_est is not None:
            d["deviation_estimated"] = self.deviation_est.tolist()
        if self.deviation_cumulative is not None:
            d["deviation_cumulative"] = self.deviation_cumulative.tolist()
        if self.state_est is not None:
            d["state_estimated"] = self.state_est.tolist()
        if self.deviation_smooth is not None:
            d["deviation_smoothed"] = self.deviation_smooth.tolist()

        full = self._full_covariance_output(final)
        for name, rows in (
            ("covariance", self.covariance_est),
            ("covariance_smooth", self.covariance_smooth),
            ("covariance_consider", self.covariance_consider),
        ):
            if rows is None:
                continue
            d[f"{name}_full" if full else f"{name}_diagonals"] = [
                None if P is None
                else (np.asarray(P).tolist() if full else np.diag(P).tolist())
                for P in rows
            ]

        d["prefits"] = self._residuals_to_list(self.prefits, self.prefit_epochs)
        d["postfits"] = self._residuals_to_list(self.postfits, self.postfit_epochs)
        if self.postfits_smoother is not None:
            d["postfits_smoother"] = self._residuals_to_list(
                self.postfits_smoother, self.postfit_epochs_smooth
            )
        nis = (self.debug or {}).get("nis_statistics")
        if nis is not None:
            d["nis_statistics"] = np.asarray(nis).tolist()
        # Self-verifying map: which timestamp axis indexes each per-epoch array.
        d["timestamps_alignment"] = self._series_alignment(d)
        return d

    def _guarded_epoch_solution(self) -> Optional[Dict]:
        """:meth:`_epoch_solution`, or None with a printed reason if it fails.

        Returns
        -------
        dict or None
            The epoch solution, or None when building it raised; the error is
            printed rather than propagated so the rest of the file is saved.
        """
        try:
            return self._epoch_solution()
        except Exception as exc:  # noqa: BLE001
            print(f"[SolutionOD] Epoch solution not saved: {exc}")
            return None

    def _output_metadata(self, covariance_storage) -> Dict:
        """Metadata fields shared by :meth:`save` and :meth:`create_JSON`.

        Parameters
        ----------
        covariance_storage : str or dict
            How the covariances are written: ``"full"`` or ``"diagonals"``, or
            a dict giving that per solution block.

        Returns
        -------
        dict
            ``frame`` and ``units`` of the state, the residual and epoch
            conventions, ``covariance_storage``, the fields of
            :meth:`_run_metadata`, then ``output_settings.metadata``, which
            overrides any of them.
        """
        return {
            # Frame/units of the estimated state and covariances, captured at
            # construction so the solution is self-describing on disk.
            "frame": self._state_frame,
            "units": self._state_units,
            # Schema notes: each residual entry is [residual, sigma, epoch];
            # deviation/covariance rows are tagged by ``timeline_epochs`` and,
            # for sequences, ``leg_index`` (+ per-leg layout in ``legs``).
            "residual_columns": ["residual", "sigma", "epoch_tdb_sec"],
            "epoch_system": "TDB",
            "epoch_units": "s",
            "covariance_storage": covariance_storage,
            **self._run_metadata(),
            **self.output_settings.metadata,
        }

    def save(self, filepath: Union[str, Path] = None, fmt: str = None) -> None:
        """
        Save all filter iterations to a single JSON or pickle file.

        If filepath is None, uses output_settings.output_dir /
        output_settings.solution_json_filename + the format extension.

        Each iteration is a key 'iter_1', 'iter_2', etc.
        The current (final) solution is also stored under 'final'.

        Parameters
        ----------
        filepath : str or Path, optional
            Destination file. When None, derived from the output settings.
        fmt : {"json", "pickle"}, optional
            On-disk format. When ``None`` (default) it is inferred from the file
            extension (``.pkl`` gives pickle, otherwise JSON); when given explicitly,
            the file extension is normalised to match.

        Notes
        -----
        ``metadata`` describes the state (``frame``, ``units``,
        ``state_components``, ``state_labels``), the a priori state and
        covariance (``apriori``), the last convergence test
        (``convergence``), ``provenance`` and ``covariance_storage``. Each
        block holds the fields of :meth:`_solution_to_dict`.

        Iteration blocks store covariance diagonals. ``final`` stores full
        covariances when ``output_settings.save_full_covariance`` is set and
        diagonals otherwise, never both. Each ``iter_<k>`` block adds that
        pass's ``convergence`` test and its ``epoch_solution`` (the deviation
        from the a priori state at the initial epoch). ``final`` adds the full
        ``epoch_solution`` of :meth:`_epoch_solution`, or None if it cannot be
        formed.
        """
        # Resolve path
        if filepath is None:
            out_dir = (
                getattr(self.output_settings, "solution_output_path", None) or "."
            )
            basename = (
                getattr(self.output_settings, "solution_output_name", None)
                or "solution"
            )
            filepath = Path(out_dir) / f"{basename}{format_suffix(fmt)}"
        filepath = with_format_suffix(filepath, fmt)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Collect all iterations from filter history
        history = getattr(self._filter, "_solution_history", [])
        convergence = getattr(self._filter, "_convergence_hist", [])
        totals = getattr(self._filter, "_delta0_hist", [])
        t0 = self._initial_epoch()

        output = {
            "metadata": {
                "n_states": self.n_states,
                "is_covariance_analysis": self.is_covariance_analysis,
                "n_iterations": len(history),
                **self._output_metadata({
                    "iterations": "diagonals",
                    "final": "full" if self._full_covariance_output(True) else "diagonals",
                }),
            }
        }

        for i, sol in enumerate(history):
            block = sol._solution_to_dict()
            # Per-pass convergence test and correction w.r.t. the a priori state;
            # skipped when the histories do not run parallel to the solutions
            # (covariance analysis, or a filter without fit()).
            if len(convergence) == len(history):
                block["convergence"] = to_jsonable(convergence[i])
            if len(totals) == len(history):
                block["epoch_solution"] = {
                    "epoch": t0,
                    "deviation_cumulative": np.asarray(totals[i]).ravel().tolist(),
                }
            output[f"iter_{i + 1}"] = block

        # Final solution (this object itself, same as last iter but explicit)
        output["final"] = self._solution_to_dict(final=True)
        output["final"]["epoch_solution"] = self._guarded_epoch_solution()

        dump(output, filepath, fmt=fmt)
        print(f"[SolutionOD] All iterations saved → {filepath}")

        # ------------------------------------------------------------------
        # Companion artifacts (reference trajectory + estimated trajectory).
        # Written next to the solution JSON; each is independently guarded so a
        # failure never blocks the primary solution output.
        # ------------------------------------------------------------------
        base = str(filepath.with_suffix(""))
        if getattr(self.output_settings, "save_reference_trajectory", True):
            try:
                ref_bsp = self._save_reference_trajectory(base)
                print(f"[SolutionOD] Reference trajectory saved → {ref_bsp}")
            except Exception as exc:  # noqa: BLE001
                print(f"[SolutionOD] Reference trajectory not saved: {exc}")
        if getattr(self.output_settings, "save_estimated_trajectory", True):
            try:
                est_json = self._save_estimated_trajectory(base)
                if est_json is not None:
                    print(f"[SolutionOD] Estimated trajectory saved → {est_json}")
                else:
                    print(
                        "[SolutionOD] Estimated trajectory not saved: "
                        "no estimated parameters in the state."
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[SolutionOD] Estimated trajectory not saved: {exc}")

    def _capture_initial_frame_units(self):
        """
        Read frame and unit labels from the filter's initial state vector.

        Returns
        -------
        frame_name : str or None
            Name of the first non-None component frame, or None if absent.
        units : dict
            Component name to serialized units from ``Units.to_dict()``. An
            unavailable serialization is recorded as None.
        components : list of dict
            Per-component layout from :meth:`_components_metadata`. The triple
            is ``(None, {}, [])`` if ``propagator.full_state_vector`` is absent.

        Notes
        -----
        Called at construction to label saved results. It does not convert
        values or modify the state. ``full_state_vector`` is the state the
        filter estimates: the working frames of a single arc, or the global
        layout of a sequence. Units are keyed only by component name, so later
        components overwrite earlier entries with the same name; ``components``
        is the complete description of mixed-frame or multiple-body layouts.
        """
        prop = getattr(self._filter, "propagator", None)
        state = getattr(prop, "full_state_vector", None)
        try:
            components = self._components_metadata(state.state)
        except Exception:
            return None, {}, []

        frame_name = next(
            (c["frame"] for c in components if c["frame"] is not None), None
        )
        units = {c["name"]: c["units"] for c in components}
        return frame_name, units, components

    def _save_reference_trajectory(self, base: str) -> str:
        """
        Write the filter's propagated nominal reference as an SPK and sidecars.

        Parameters
        ----------
        base : str
            Output path prefix without the ``_reference.bsp`` suffix.

        Returns
        -------
        str
            Path ``<base>_reference.bsp`` returned after a successful write.

        Raises
        ------
        ValueError
            If the filter has no propagator or propagated state array.
        OSError
            If the existing SPK cannot be removed or an output cannot be written.
            Trajectory/SPICE serialization errors also propagate to the caller.

        Notes
        -----
        Uses the nominal states and STMs currently held by the propagator;
        estimated corrections are not applied. An existing SPK at this path is
        removed before writing. Sidecar formats and STM inclusion follow
        :meth:`_trajectory_write_formats`. If propagator settings cannot be
        serialized with ``to_dict()``, the trajectory is written without them.
        """
        from scarabaeus import Trajectory

        prop = getattr(self._filter, "propagator", None)
        if prop is None or getattr(prop, "propagated_state_array", None) is None:
            raise ValueError("Filter has no propagated reference trajectory.")

        states = prop.propagated_state_array
        try:
            settings = prop.to_dict()
        except Exception:
            settings = None

        ref = Trajectory(
            state_array=states,
            STMs=getattr(prop, "STM", None),
            STMs_timestamp=list(np.atleast_1d(states.epoch.et)),
            propagator_settings_dict=settings,
        )
        bsp = f"{base}_reference.bsp"
        if Path(bsp).exists():
            Path(bsp).unlink()
        ref.write_to_spk(bsp, **self._trajectory_write_formats())
        return bsp

    def _trajectory_write_formats(self) -> dict:
        """
        Build sidecar-writing options from the solution's output settings.

        Returns
        -------
        dict
            Keyword arguments ``stm_format``, ``sidecar_format`` and
            ``write_stms`` for :meth:`~scarabaeus.Trajectory.write_to_spk`.
            Both formats use ``trajectory_output_format``, falling back to
            ``"json"`` when missing or false-valued. ``write_stms`` is the
            boolean value of ``save_trajectory_stms``, defaulting to True.

        Notes
        -----
        This helper selects options without validating formats or writing files.
        Choosing a binary sidecar format can reduce storage for large STM
        histories. The underlying trajectory writer handles format validation.
        """
        fmt = getattr(self.output_settings, "trajectory_output_format", "json")
        return {
            "stm_format": fmt or "json",
            "sidecar_format": fmt or "json",
            "write_stms": bool(
                getattr(self.output_settings, "save_trajectory_stms", True)
            ),
        }

    def _save_estimated_trajectory(self, base: str) -> Optional[str]:
        """
        Write estimated Cartesian and parameter samples at measurement epochs.

        Parameters
        ----------
        base : str
            Output path prefix without the ``_estimated.json`` suffix.

        Returns
        -------
        str or None
            Path ``<base>_estimated.json`` after writing, or None when
            ``n_states <= 6``; that guard skips the file without querying states.

        Raises
        ------
        ValueError
            If :meth:`estimated_trajectory` rejects the stored epochs/history,
            or unequal sequence parameter layouts cannot form a NumPy array.
        OSError
            If the output file cannot be written. Trajectory query and JSON
            serialization errors also propagate to the caller.

        Notes
        -----
        Calls :meth:`estimated_trajectory` with its default smoothing selection.
        The JSON contains frame/unit metadata, measurement timestamps in TDB
        seconds, position in km, velocity in km/s, and native-unit parameters
        (or None). Existing output is replaced by the JSON writer.

        The size guard assumes a single body's six Cartesian components; it
        does not inspect component names to establish whether parameters exist.
        Sequence parameter lists are converted to an ndarray, so ragged leg
        layouts are not supported by this writer. These errors propagate here
        and are handled by the enclosing :meth:`save` method.
        """
        # No estimated parameters (position + velocity = 6) -> nothing to save.
        if self.n_states <= 6:
            return None

        epochs = EpochArray(np.asarray(self.timestamps, dtype=float), sys="TDB")
        est_pos, est_vel, est_params = self.estimated_trajectory(epochs)

        payload = {
            "frame": self._state_frame,
            "units": self._state_units,
            "timestamps": np.asarray(self.timestamps),
            "position": np.asarray(est_pos),
            "velocity": np.asarray(est_vel),
            "parameters": (
                np.asarray(est_params) if est_params is not None else None
            ),
        }
        path = f"{base}_estimated.json"
        dump_json(payload, path)
        return path
