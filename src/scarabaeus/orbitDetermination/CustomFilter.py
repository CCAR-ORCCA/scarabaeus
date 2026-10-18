# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional

from scarabaeus import (
    FilterOD,
    FilterSettings,
    MeasurementSpec,
    MissionSequence,
    ProcessNoise,
    Propagator,
    SolutionOD,
)
from scarabaeus.utils.CustomCallable import CustomCallable
from scarabaeus.utils.Utils import Utils

import numpy as np

# Names a user estimator may declare. Rejected at construction otherwise.
_CONTEXT_NAMES = ("filter", "timeline", "n_states", "covariance", "deviation")


class CustomFilter(FilterOD):
    """ Estimator whose whole algorithm is supplied from a script.

    ``CustomFilter`` completes the trio started by
    :class:`~scarabaeus.CustomDynamicModel` and
    :class:`~scarabaeus.CustomMeasurement`.  Where those replace one model,
    this replaces the **estimator itself**: the user writes the estimation
    algorithm and Scarabaeus provides the scaffolding around it — reference
    trajectory, measurement timeline, residuals and partials, state-transition
    matrices, process noise, and the :class:`~scarabaeus.SolutionOD` packaging
    that the rest of the toolchain consumes.

    Unlike :class:`~scarabaeus.LKF` or :class:`~scarabaeus.SRIF`, nothing here
    assumes a particular estimator structure. The algorithm is free to be

    * **sequential** — any Kalman/information/square-root recursion,
    * **batch** — accumulate normal equations over all epochs and solve once,
    * **sigma-point** (UKF/CKF) — propagate sigma points through the real
      nonlinear dynamics and measurements with :meth:`predict` and
      :meth:`predict_measurements`,
    * **Gaussian mixture** — run any of the above per component and recombine,
    * **polynomial chaos** — evaluate the model at the quadrature/collocation
      nodes and fit the expansion,
    * **sampling** (MCMC, particle) — score candidate states by their
      measurement likelihood.

    The estimator is a callable invoked once per :meth:`fit` iteration. It
    receives the filter itself — the toolkit below — and returns the histories
    the solution is built from.

    **Toolkit.** Read the problem:

    ==================================  =====================================
    :attr:`timeline`                    epochs the solution is reported on
    :attr:`measurement_epochs`          epochs that carry an observation
    :attr:`n_states`                    length of the estimated state
    :attr:`initial_covariance_matrix`   :math:`P_0`
    :attr:`initial_deviation`           :math:`\\bar x_0`
    :meth:`measurement_block`           residuals, partials, sigmas at an epoch
    :meth:`measurement_covariance`      :math:`R_k` from those sigmas
    :meth:`stm`                         :math:`\\Phi(t, t_0)`
    :meth:`stm_between`                 :math:`\\Phi(t_1, t_0)`
    :meth:`process_noise_at`            discrete :math:`Q_k`
    :meth:`safe_inv`                    pseudo-inverse used across the filters
    ==================================  =====================================

    Step outside the linearisation — these run the real nonlinear models and
    are what make sigma-point, mixture, chaos and sampling methods possible:

    ==============================  =========================================
    :meth:`predict`                 propagate an arbitrary initial state
    :meth:`predict_measurements`    computed measurements for such a state
    ==============================  =========================================

    Report the answer with :meth:`record`, or return a dict with the same keys.

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Reference trajectory generator, as for any filter. A sequence must
        carry the same state on every leg; one that adds or drops states at a
        node (e.g. finite-burn parameters) raises
        :class:`CustomFilter.SequenceError`.
    measurements : MeasurementSpec
        Measurement models and observations.
    settings : FilterSettings
        Filter configuration (initial covariance, process noise, output).
    estimator : callable
        The estimation algorithm. Invoked as ``estimator(filter=self)`` — declare
        ``filter`` (and optionally ``timeline``, ``n_states``, ``covariance``,
        ``deviation``) in its signature. Must either call :meth:`record` or
        return a mapping with ``"deviation"`` and ``"covariance"``.
    smoother : callable, optional
        Backward pass, invoked the same way when ``fit(if_sequential_smooth=True)``.
    traj_name, traj_dir, overwrite_traj, verbose
        Forwarded to :class:`~scarabaeus.FilterOD`.

    See Also
    --------
    scarabaeus.LKF : Reference sequential implementation to model an algorithm on.
    scarabaeus.CustomDynamicModel : User-defined force model.
    scarabaeus.CustomMeasurement : User-defined measurement model.

    Notes
    -----
    Residuals and partials returned by :meth:`measurement_block` are computed
    against the current **reference trajectory**, and :meth:`fit` re-linearises
    about the updated reference between iterations. An estimator that works in
    absolute states rather than deviations should use :meth:`predict` and
    :meth:`predict_measurements` instead.

    Those two run a full propagation (and, for the second, a full measurement
    pass) per call. A sigma-point filter therefore costs ``2n+1`` of them per
    iteration — acceptable offline, but it is the dominant cost, so cache
    aggressively inside the estimator.

    Examples
    --------
    A sequential Kalman recursion written from scratch:

    .. code-block:: python

        import numpy as np
        import scarabaeus as scb

        def sequential(filter):
            P = filter.initial_covariance_matrix
            x = filter.initial_deviation
            deviations, covariances = [], []

            previous = None
            for epoch in filter.timeline:
                if previous is not None:                      # time update
                    Phi = filter.stm_between(previous, epoch)
                    x, P = Phi @ x, Phi @ P @ Phi.T
                previous = epoch

                block = filter.measurement_block(epoch)
                if block is not None:                         # measurement update
                    H = block["partials"][:, : filter.n_states]
                    R = filter.measurement_covariance(block["sigma"], epoch)
                    S = H @ P @ H.T + R
                    K = P @ H.T @ filter.safe_inv(S)
                    x = x + K @ (block["residuals"] - H @ x)
                    P = (np.eye(filter.n_states) - K @ H) @ P

                deviations.append(x)
                covariances.append(P)

            filter.record(deviation=deviations, covariance=covariances)

        od = scb.CustomFilter(prop, measurements, settings, estimator=sequential)
        solution, n_iterations, converged = od.fit(max_iterations=3)

    A batch least-squares estimator over the same scaffolding:

    .. code-block:: python

        def batch(filter):
            n = filter.n_states
            information = filter.safe_inv(filter.initial_covariance_matrix)
            normal = np.zeros((n, 1))

            for epoch in filter.measurement_epochs:
                block = filter.measurement_block(epoch)
                H = block["partials"][:, :n] @ filter.stm(epoch)
                W = filter.safe_inv(filter.measurement_covariance(block["sigma"], epoch))
                information += H.T @ W @ H
                normal += H.T @ W @ block["residuals"]

            P0 = filter.safe_inv(information)
            x0 = P0 @ normal

            deviations, covariances = [], []
            for epoch in filter.timeline:
                Phi = filter.stm(epoch)
                deviations.append(Phi @ x0)
                covariances.append(Phi @ P0 @ Phi.T)

            filter.record(deviation=deviations, covariance=covariances)
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ResultError(ValueError):
        """Raised when the estimator does not report a usable result."""

    class ShapeError(ValueError):
        """Raised when a reported history has an unexpected shape."""

    class SequenceError(ValueError):
        """Raised when a mission sequence changes the state from one leg to
        the next."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        settings: FilterSettings,
        estimator=None,
        smoother=None,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            propagator,
            measurements,
            settings,
            traj_name,
            traj_dir,
            overwrite_traj,
            verbose=verbose,
        )

        if estimator is None:
            raise self.ResultError(
                "CustomFilter needs an 'estimator' callable implementing the "
                "estimation algorithm. See the class docstring for the toolkit "
                "it receives."
            )

        if self.flag_sequence:
            self._check_sequence_state()

        # Bound once, so a misspelt argument fails here and not mid-fit.
        self._estimator = CustomCallable(
            estimator, _CONTEXT_NAMES, "estimator", self.__class__.__name__
        )
        self._smoother_hook = (
            None
            if smoother is None
            else CustomCallable(
                smoother, _CONTEXT_NAMES, "smoother", self.__class__.__name__
            )
        )

        self._recorded: dict = {}

        print("")
        print("=" * 80)
        print("Initializing Custom Filter")
        print(f"Estimator: {getattr(estimator, '__name__', type(estimator).__name__)}")
        if self.process_noise is not None:
            print(f"Process Noise Model: {self._process_noise_description()}")
        print("=" * 80)

    # ----------------------#
    # region    Toolkit    #
    # ----------------------#
    @property
    def timeline(self) -> np.ndarray:
        """Epochs the solution is reported on (measurements together with events)."""
        if getattr(self, "timeline_times", None) is None:
            self._build_timeline()

        return self.timeline_times

    @property
    def measurement_epochs(self) -> np.ndarray:
        """Epochs that carry at least one observation."""
        measurement_times, _, _ = self._build_timeline()

        return measurement_times

    @property
    def n_states(self) -> int:
        """Length of the estimated state."""
        return self.n

    @property
    def initial_covariance_matrix(self) -> np.ndarray:
        """A priori covariance :math:`P_0`, without units."""
        return self.initial_covariance.matrix_without_units()

    @property
    def initial_deviation(self) -> np.ndarray:
        """A priori state deviation :math:`\\bar x_0`, shape ``(n, 1)``."""
        deviation = getattr(self, "dx0", None)
        if deviation is None:
            return np.zeros((self.n, 1))

        return np.atleast_2d(np.asarray(deviation, dtype=float)).reshape(-1, 1)

    def measurement_block(self, epoch) -> dict | None:
        """
        Return the measurement block at *epoch*.

        Parameters
        ----------
        epoch : float
            Epoch, TDB seconds past J2000.

        Returns
        -------
        dict or None
            Keys include ``residuals``, ``partials``, ``sigma``, ``instruments``,
            ``measurement_types``, ``dataset_names`` and ``t2``. ``None`` when the
            epoch carries no observation (an event node, for instance).
        """
        try:
            return self._epoch_measurement_block(epoch)
        except (KeyError, IndexError):
            return None

    def measurement_covariance(
        self, sigmas, epoch, meas_corr=None, underweighting_factor: float = 1.0
    ) -> np.ndarray:
        """
        Build the measurement-noise covariance :math:`R_k`.

        Parameters
        ----------
        sigmas : array_like
            Per-measurement standard deviations from :meth:`measurement_block`.
        epoch : float
            Epoch the measurements belong to.
        meas_corr : list, optional
            Off-diagonal correlation terms.
        underweighting_factor : float, optional
            Multiplicative inflation of ``R``. Defaults to 1.0.

        Returns
        -------
        numpy.ndarray
            Measurement covariance ``(m, m)``.
        """
        return self._measurement_covariance(
            sigmas, epoch, meas_corr, underweighting_factor
        )

    def stm(self, epoch, index=None, idx_leg=None) -> np.ndarray:
        """
        State-transition matrix from the initial epoch to *epoch*.

        Parameters
        ----------
        epoch : float
            Target epoch, TDB seconds.
        index : int, optional
            Position of *epoch* in the timeline. Supplying it lets the
            trajectory pick the stored matrix directly instead of searching.
        idx_leg : int, optional
            Leg index, for sequence trajectories.

        Returns
        -------
        numpy.ndarray
            ``Phi(epoch, t0)``, shape ``(n, n)``.
        """
        if idx_leg is not None:
            matrix = self.trajectory.get_STM_sequence(epoch=epoch, idx_leg=idx_leg)
        else:
            matrix = self.trajectory.get_STM(epoch=epoch, idx=index)

        matrix = np.asarray(getattr(matrix, "values", matrix), dtype=float)
        size = int(np.sqrt(matrix.size))

        return matrix.reshape(size, size)

    def stm_between(self, epoch_from, epoch_to, idx_leg=None) -> np.ndarray:
        """
        One-step state-transition matrix between two epochs.

        .. math::

            \\Phi(t_1, t_0) = \\Phi(t_1, t_{\\text{ref}})\\,
                              \\Phi(t_0, t_{\\text{ref}})^{-1}

        Parameters
        ----------
        epoch_from, epoch_to : float
            Bounding epochs, TDB seconds.
        idx_leg : int, optional
            Leg index, for sequence trajectories.

        Returns
        -------
        numpy.ndarray
            ``Phi(epoch_to, epoch_from)``, shape ``(n, n)``.
        """
        return self.stm(epoch_to, idx_leg=idx_leg) @ self.safe_inv(
            self.stm(epoch_from, idx_leg=idx_leg)
        )

    def process_noise_at(self, index: int) -> np.ndarray | None:
        """
        Discrete process-noise covariance :math:`Q_k` at a timeline index.

        Parameters
        ----------
        index : int
            Position in :attr:`timeline`.

        Returns
        -------
        numpy.ndarray or None
            ``Q_k``, or ``None`` when no process noise is configured.
        """
        if not (self.SNC_flag or self.DMC_flag) or self.process_noise is None:
            return None

        return ProcessNoise.combined_covariance_at(
            self._process_noise_models(), index, self.n
        )

    def safe_inv(self, matrix, min_svd: float = 1e-14) -> np.ndarray:
        """
        Numerically guarded inverse, shared with the built-in filters.

        Parameters
        ----------
        matrix : numpy.ndarray
            Matrix to invert.
        min_svd : float, optional
            Singular-value floor. Defaults to ``1e-14``.

        Returns
        -------
        numpy.ndarray
            The (pseudo-)inverse.
        """
        return self._safe_inv(matrix, min_svd)

    # endregion Toolkit #
    # ----------------------#

    # --------------------------------#
    # region Nonlinear evaluation     #
    # --------------------------------#
    def predict(self, initial_state, epochs=None):
        """
        Propagate an arbitrary initial state through the real dynamics.

        This is the escape hatch from the linearisation: sigma-point, mixture,
        chaos and sampling estimators need the nonlinear flow, not ``Phi``.

        Parameters
        ----------
        initial_state : array_like
            Initial state values, shape ``(n,)``, in the reference state's own
            order and units.
        epochs : EpochArray, optional
            Epochs to report. Defaults to the propagator's own span.

        Returns
        -------
        numpy.ndarray
            Propagated states, shape ``(len(epochs), n)``.

        Notes
        -----
        Each call runs a full propagation; cache the results inside the
        estimator rather than calling it per epoch.
        """
        propagator = self._propagator_for(initial_state, epochs)
        propagator.propagate(display_progress=False)

        return np.asarray(propagator.ys, dtype=float)[:, : self.n]

    def predict_measurements(self, initial_state, epochs=None) -> dict:
        """
        Computed measurements for an arbitrary initial state.

        Propagates *initial_state* and evaluates every configured measurement
        model along the resulting trajectory, giving :math:`h(x)` without any
        linearisation.

        Parameters
        ----------
        initial_state : array_like
            Initial state values, shape ``(n,)``.
        epochs : EpochArray, optional
            Epochs to evaluate at. Defaults to the propagator's own span.

        Returns
        -------
        dict
            Mapping of dataset name to the computed measurements.

        Notes
        -----
        This is the most expensive call in the toolkit: one propagation plus a
        full measurement pass. A sigma-point filter needs ``2n+1`` per iteration.
        """
        propagator = self._propagator_for(initial_state, epochs)
        propagator.propagate(display_progress=False)

        from scarabaeus import Trajectory

        trajectory = Trajectory(state_array=propagator.propagated_state_array)
        target = self.propagator.primary_body
        span = epochs if epochs is not None else propagator.tspan

        computed = {}
        for descriptor in self.measurements:
            model = descriptor["model"]
            name = descriptor.get("dataset_name") or model.name
            model.update_reference_state(propagator.full_state_vector)
            _, _, _, values = model.computed_measurements(
                target=target, epoch_array=span
            )
            computed[name] = values

        self._last_predicted_trajectory = trajectory

        return computed

    def _propagator_for(self, initial_state, epochs=None) -> Propagator:
        """
        Build a propagator seeded with *initial_state*.

        A fresh :class:`~scarabaeus.Propagator` is constructed rather than a copy
        of the reference one: the force models hold a Rust ephemeris cache that
        cannot be deep-copied, and they are re-initialised against the new state
        anyway.

        Parameters
        ----------
        initial_state : array_like
            Initial state values, shape ``(n,)``, in the reference state's own
            component order and units.
        epochs : EpochArray, optional
            Propagation span. Defaults to the reference propagator's.

        Returns
        -------
        Propagator
            Propagator over the requested span, seeded with the given state.
        """
        from scarabaeus import ArrayWFrame, ArrayWUnits, StateArray, StateDefinition

        values = np.asarray(initial_state, dtype=float).ravel()
        reference = self.propagator.full_state_vector

        # Rebuild the state definition component by component, substituting the
        # supplied values while keeping every owner, unit and frame intact.
        components, offset = [], 0
        for name, dim, estimation, dynamics, owner, template in reference.state:
            block = values[offset : offset + dim]
            offset += dim

            quantity = ArrayWUnits(
                block.copy() if dim > 1 else float(block[0]),
                template.quantity.units,
            )
            components.append(
                (name, dim, estimation, dynamics, owner,
                 ArrayWFrame(quantity, template.frame))
            )

        state_vector = StateArray(
            epoch=reference.epoch,
            origin=reference.origin,
            state=StateDefinition.from_components(components),
        )

        return Propagator(
            primary_body=self.propagator.primary_body,
            state_vector=state_vector,
            tspan=epochs if epochs is not None else self.propagator.tspan,
            force_models=self.propagator.force_models,
            propagate_STM=False,
            # The force models are shared, not copied: an initialized model holds
            # a Rust ephemeris cache that cannot be deep-copied. They are re-bound
            # to this state, so the reference propagator must be re-initialised
            # afterwards — fit() does that at every iteration boundary.
            copy_force_models=False,
        )

    # endregion Nonlinear evaluation #
    # --------------------------------#

    # ----------------#
    # region Results #
    # ----------------#
    def record(self, deviation=None, covariance=None, prefit=None, postfit=None):
        """
        Report the estimator's results.

        Parameters
        ----------
        deviation : sequence, optional
            State deviation per timeline epoch, each ``(n, 1)``.
        covariance : sequence, optional
            Covariance per timeline epoch, each ``(n, n)``.
        prefit, postfit : list, optional
            Residual records in the layout the built-in filters use:
            ``[residuals, sigmas, instruments, types, dataset_names, epochs]``.
        """
        if deviation is not None:
            self._recorded["deviation"] = deviation
        if covariance is not None:
            self._recorded["covariance"] = covariance
        if prefit is not None:
            self._recorded["prefit"] = prefit
        if postfit is not None:
            self._recorded["postfit"] = postfit

    def _store_result(self, result) -> None:
        """
        Move the estimator's output onto the attributes SolutionOD reads.

        Parameters
        ----------
        result : dict or None
            Whatever the estimator returned; merged over anything already
            reported through :meth:`record`.

        Raises
        ------
        CustomFilter.ResultError
            If no deviation or covariance history was reported.
        CustomFilter.ShapeError
            If a history does not match the timeline length or the state size.
        """
        recorded = dict(self._recorded)
        if isinstance(result, dict):
            recorded.update(result)

        if "covariance" not in recorded:
            raise self.ResultError(
                "The estimator reported no covariance history. Call "
                "filter.record(deviation=..., covariance=...) or return a dict "
                "with those keys."
            )

        expected = len(self.timeline)
        covariance = [
            np.asarray(entry, dtype=float).reshape(self.n, self.n)
            for entry in recorded["covariance"]
        ]
        if len(covariance) != expected:
            raise self.ShapeError(
                f"The estimator reported {len(covariance)} covariance entries "
                f"for {expected} timeline epochs."
            )
        self.covariance_history = covariance

        if self.covariance_analysis:
            self.state_deviation_history = [np.zeros((self.n, 1))] * expected
        else:
            if "deviation" not in recorded:
                raise self.ResultError(
                    "The estimator reported no deviation history. Call "
                    "filter.record(deviation=...) or return it."
                )
            deviation = [
                np.asarray(entry, dtype=float).reshape(self.n, 1)
                for entry in recorded["deviation"]
            ]
            if len(deviation) != expected:
                raise self.ShapeError(
                    f"The estimator reported {len(deviation)} deviation entries "
                    f"for {expected} timeline epochs."
                )
            self.state_deviation_history = deviation

        if self.covariance_analysis:
            return

        dataset_names = self.measurement_data.dataset_names
        self.prefit_residuals = self._unwrap_residuals(
            recorded.get("prefit", []), dataset_names, epoch_attr="prefit_epochs"
        )
        self.postfit_residuals = self._unwrap_residuals(
            recorded.get("postfit", []), dataset_names, epoch_attr="postfit_epochs"
        )

    # endregion Results #
    # ----------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _check_sequence_state(self) -> None:
        """
        Refuse a mission sequence whose legs do not all carry the same state.

        The toolkit reports one state size for the whole timeline, and no
        forward solution is stored at the nodes to recover dropped states when
        mapping back.

        Raises
        ------
        CustomFilter.SequenceError
            If a leg adds, drops or reorders states relative to the one before.
        """
        for counter_events in range(1, len(self.legs_model)):
            prev_list, next_list = self._node_state_lists(counter_events, True)
            Q, _new_n, _new_blocks = Utils.build_reorder_operator(prev_list, next_list)
            if Q.shape[0] != Q.shape[1] or not np.array_equal(Q, np.eye(Q.shape[0])):
                raise self.SequenceError(
                    f"CustomFilter needs every leg of the mission sequence to "
                    f"carry the same state, but leg {counter_events} changes it "
                    f"(e.g. a finite burn adding or dropping its parameters). "
                    f"Use LKF, SRIF, SRIFB or LSB for such sequences."
                )

    def _context(self) -> dict:
        """
        Assemble the keyword arguments handed to the estimator.

        Returns
        -------
        dict
            Mapping of every context name to its value.
        """
        return {
            "filter": self,
            "timeline": self.timeline,
            "n_states": self.n,
            "covariance": self.initial_covariance_matrix,
            "deviation": self.initial_deviation,
        }

    def _compute(self, meas_corr: list = None, printOutput: bool = True) -> SolutionOD:
        """
        Run one pass of the user estimator and package the solution.

        Parameters
        ----------
        meas_corr : list, optional
            Measurement-correlation terms; forwarded to the estimator only via
            :meth:`measurement_covariance`, which the estimator calls itself.
        printOutput : bool, optional
            Print a one-line summary. Defaults to True.

        Returns
        -------
        SolutionOD
            The packaged solution.
        """
        if self.flag_sequence is True:
            self.n = self.legs_n[0]

        self._recorded = {}
        self._build_timeline()

        result = self._estimator(self._context())
        self._store_result(result)

        if printOutput:
            trace = float(np.trace(self.covariance_history[-1]))
            print(
                f"Custom estimator finished over {len(self.timeline)} epochs "
                f"| final tr(P) = {trace:.6e}"
            )

        return SolutionOD(self)

    def _smoother(self, printOutput: bool = True) -> SolutionOD:
        """
        Run the user smoother, when one was supplied.

        Parameters
        ----------
        printOutput : bool, optional
            Print a one-line summary. Defaults to True.

        Returns
        -------
        SolutionOD
            The packaged smoothed solution.

        Raises
        ------
        NotImplementedError
            If no smoother callable was given.
        """
        if self._smoother_hook is None:
            raise NotImplementedError(
                "This CustomFilter was built without a 'smoother' callable, so "
                "fit(if_sequential_smooth=True) has nothing to run."
            )

        self._recorded = {}
        result = self._smoother_hook(self._context())
        self._store_result(result)

        if printOutput:
            print(f"Custom smoother finished over {len(self.timeline)} epochs")

        return SolutionOD(self)

    def __repr__(self) -> str:
        return f"CustomFilter(estimator={self._estimator!r})"

    # endregion Methods #
    # -------------------#
