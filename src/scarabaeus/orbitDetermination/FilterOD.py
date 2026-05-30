# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    StateArray,
    CovarianceMatrix,
    Trajectory,
    FilterDataManager,
    Propagator,
    FilterSettings,
    ProcessNoiseSettings,
    StateNoiseCompensation,
    DynamicModelCompensation,
    StateDefinition,
    ArrayWFrame,
    ArrayWUnits,
    ScenarioSetup,
    MissionSequence,
    MeasurementSpec,
)
import scarabaeus.utils.NumpyWrapper as np
import os
import time
import scipy.linalg as spln
from typing import List, Dict, Tuple, Optional
import warnings


# -----------------------#
#    Class Definition   #
# -----------------------#
class FilterOD:
    """Parent class for orbit determination filters used for spacecraft state and
    parameter estimation from tracking measurements.

    The framework supports single- and multi-object estimation problems through
    flexible dynamical and measurement model definitions. The class propagates a
    reference solution, constructs the linearized measurement sensitivity matrix,
    and estimates corrections to the reference trajectory, dynamical states, and
    model parameters.

    Supported estimation problems include spacecraft translational dynamics,
    stochastic and deterministic parameters, multi-arc or mission-sequence
    configurations, and coupled dynamical systems.

    **Maximum-a-posteriori estimate**

    Every filter in this framework estimates the *deviation* :math:`\\Delta x_0`
    from the reference trajectory :math:`x^*(t)` that minimises the weighted
    least-squares cost:

    .. math::

        J(\\Delta x_0) = \\Delta x_0^T P_0^{-1} \\Delta x_0
        + \\sum_k \\Bigl[y_k - H_k \\Phi_{k,0}\\,\\Delta x_0\\Bigr]^T
          R_k^{-1}
          \\Bigl[y_k - H_k \\Phi_{k,0}\\,\\Delta x_0\\Bigr]

    where :math:`y_k = z_k - h(x^*(t_k))` is the prefit residual and
    :math:`P_0^{-1}` is the a-priori information matrix.  Under Gaussian
    assumptions the minimiser is the **maximum-a-posteriori (MAP)** estimate
    :math:`\\hat{x}_0 = x^*(t_0) + \\widehat{\\Delta x}_0`.

    **Iteration**

    Because the measurement model is linearised around :math:`x^*`, the
    single-pass estimate is only first-order accurate.  The
    :meth:`fit` method re-runs the filter iteratively: after each pass the
    reference is re-propagated from :math:`x^*(t_0) + \\widehat{\\Delta x}_0`
    and the filter is re-applied on the updated reference.  At convergence the
    accumulated deviation satisfies :math:`\\widehat{\\Delta x}_0 \\approx 0`,
    meaning the current reference **is** the MAP estimate and the covariance is
    the true posterior covariance linearised around the solution.

    **Schmidt consider parameters**

    Parameters that affect measurements but are poorly constrained and should
    not be estimated can be declared as *consider parameters*
    (:class:`~scarabaeus.FilterSettings`).  Their uncertainty is propagated through the
    filter and inflates the covariance of the estimated state (Schmidt, 1966),
    but their values are never updated by measurements.  This prevents
    over-confidence when unmodelled force parameters are present.

    Parameters
    ----------
    propagator : Propagator or MissionSequence
        Propagator or mission sequence object that defines the reference trajectory
        and state transition matrices (STMs) for orbit determination.
    measurements : MeasurementSpec
        Measurement specification object.
    settings : FilterSettings
        Filter configuration settings.
    traj_name : str, optional
        Name for the trajectory BSP file.
    traj_dir : str, optional
        Directory to write trajectory.
    overwrite_traj : bool
        If True, overwrite existing trajectory file.

    Notes
    -----
    This class is not instantiated directly; use a concrete subclass
    (:class:`~scarabaeus.LKF`, :class:`~scarabaeus.SRIF`,
    :class:`~scarabaeus.LSB`, or :class:`~scarabaeus.SRIFB`).
    Call :meth:`fit` to run the filter and retrieve results via
    :class:`~scarabaeus.SolutionOD`.

    See Also
    --------
    scarabaeus.LKF : Linearized Kalman Filter (sequential, covariance form).
    scarabaeus.SRIF : Square Root Information Filter (sequential, square-root form).
    scarabaeus.LSB : Batch Least-Squares estimator.
    scarabaeus.SRIFB : Batch Square Root Information Filter.
    scarabaeus.SolutionOD : Container for the estimated state history and covariance.

    References
    ----------
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def propagator(self):
        """Propagator or ScenarioSetup used to build the reference trajectory.
        Replaced by a :class:`~scarabaeus.ScenarioSetup` when a :class:`~scarabaeus.MissionSequence` is provided.
        """
        return self._propagator

    @propagator.setter
    def propagator(self, value):
        self._propagator = value

    @property
    def measurements(self):
        """Measurement specification object passed at construction time."""
        return self._measurements

    @measurements.setter
    def measurements(self, value):
        self._measurements = value

    @property
    def settings(self):
        """Filter configuration (covariance, process noise, output options)."""
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    @property
    def trajectory(self):
        """Propagated reference trajectory stored in a SPICE BSP kernel with a
        companion JSON file for STMs and estimated parameters."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value):
        self._trajectory = value

    @property
    def measurement_data(self):
        """Container for all measurement datasets, residuals, and editing flags."""
        return self._measurement_data

    @measurement_data.setter
    def measurement_data(self, value):
        self._measurement_data = value

    @property
    def n(self) -> int:
        """Dimension of the estimated state vector."""
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value

    @property
    def has_consider(self) -> bool:
        """``True`` if consider parameters are present in the state vector."""
        return self._has_consider

    @has_consider.setter
    def has_consider(self, value: bool):
        self._has_consider = value

    @property
    def flag_sequence(self) -> bool:
        """``True`` for multi-leg (mission-sequence) trajectories."""
        return self._flag_sequence

    @flag_sequence.setter
    def flag_sequence(self, value: bool):
        self._flag_sequence = value

    @property
    def SNC_flag(self) -> bool:
        """``True`` when State Noise Compensation (SNC) process noise is active."""
        return self._SNC_flag

    @SNC_flag.setter
    def SNC_flag(self, value: bool):
        self._SNC_flag = value

    @property
    def DMC_flag(self) -> bool:
        """``True`` when Dynamic Model Compensation (DMC) process noise is active."""
        return self._DMC_flag

    @DMC_flag.setter
    def DMC_flag(self, value: bool):
        self._DMC_flag = value

    @property
    def covariance_analysis(self) -> bool:
        """``True`` when running in covariance-analysis mode (no measurement updates)."""
        return self._covariance_analysis

    @covariance_analysis.setter
    def covariance_analysis(self, value: bool):
        self._covariance_analysis = value

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        settings: FilterSettings,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
    ):
        # Store inputs
        self.propagator = propagator
        self.measurements = measurements
        self.settings = settings
        self.initial_covariance = settings.initial_covariance

        # Build trajectory and measurements
        self.trajectory, self.measurement_data = self._initialize(
            propagator, measurements, traj_name, traj_dir, overwrite_traj
        )

        self.first_iteration_measurement_data = self.measurement_data

        # Initialize flags
        self.SNC_flag = False
        self.DMC_flag = False
        self.covariance_analysis = False
        self.update_flag = True
        self.flag_desat = False
        self.flag_sequence = self.trajectory.sequence_definition is not None

        # Setup based on trajectory type
        if self.flag_sequence:
            self._initialize_sequence_attributes()
        else:
            self._initialize_nonsequence_attributes()

        # Generate consider parameter partition (if applicable)
        self._init_consider_partition()

        # Validate and apply settings
        self._apply_settings()

        # Track cumulative correction w.r.t. iteration-1 reference
        # frozen "initial" state definition
        self._ref0_state_def = self.trajectory.state_definition
        # cumulative total in ref0 coordinates
        self._delta0_total = np.zeros((self.n, 1), dtype=float)
        # optional history
        self._delta0_hist = []
        # per-epoch cumulative deviation [n_epochs x n_states] across all iterations
        self._cumulative_deviation = None
        # solution history
        self._solution_history = []
        self._solution_history_smooth = []

    # endregion Constructor #
    # -----------------------#

    # -----------------------------------
    #             Methods
    # -----------------------------------

    def _initialize_nonsequence_attributes(self) -> None:
        """
        Initialize filter attributes for single-arc (non-sequence) trajectories.

        Infers the state vector dimension ``n`` from the shape of the first stored
        STM (assumed square), stores the state definition, and detects whether the
        first STM is actually a Maneuver Transition Matrix (MTM).

        An MTM is identified by checking whether the lower-right 3x3 block of
        ``STM[3:6, -3:]`` equals the identity matrix, which is the signature of
        an analytically included impulsive delta-V in the state vector.

        Attributes Set
        --------------
        n : int
            Dimension of the estimated state vector.
        state_vector : list
            State vector definition copied from ``self.trajectory``.
        flag_MTM : bool
            True if the first trajectory STM is a Maneuver Transition Matrix.

        Returns
        -------
        None
        """
        self.n = int(np.sqrt(len(self.trajectory._STMs[0])))
        self.state_vector = self.trajectory.state_definition

        # Check if dV is introduced analytically via MTM
        # If the first STM is an MTM, dV is already in the state vector
        self.flag_MTM = False
        if not self.flag_sequence:
            MTM_candidate = self.trajectory._STMs[0].reshape(
                int(np.sqrt(len(self.trajectory._STMs[0]))), -1
            )
            if len(MTM_candidate) > 6:
                self.flag_MTM = np.array_equal(MTM_candidate[3:6, -3:], np.eye(3))

    def _initialize_sequence_attributes(self) -> None:
        """
        Initialize filter attributes for multi-leg (sequence) trajectories.

        Parses the sequence definition to extract per-leg epoch arrays, state
        sizes, and models.  Separates single-epoch "event" entries (maneuver
        nodes) from multi-epoch "leg" entries (propagation segments) so the
        filter can handle covariance mapping across trajectory discontinuities.

        Attributes Set
        --------------
        sequence_epochs : list of np.ndarray
            Epoch arrays for every leg, including single-point event epochs.
        n : int
            Full state vector dimension (from the propagator).
        events_epochs : list of float
            Scalar epoch values for trajectory events (maneuver nodes).
        legs_n : list of int
            State dimension for each propagation leg.
        legs_epochs : list of np.ndarray
            Epoch arrays for propagation legs only (multi-epoch entries).
        legs_model : list of StateArray
            State model objects for each propagation leg.
        new_sigmas : list
            Extra apriori sigma values from ``initial_covariance.extra_apriori_sigmas``,
            used to initialise covariance blocks for newly augmented states.
        flag_node : bool
            Set to False; toggled to True by the filter when an event epoch is reached.

        Returns
        -------
        None
        """

        self.sequence_epochs = [
            (
                np.array([epoch.times.values])
                if isinstance(epoch.times.values, (float, int))
                else epoch.times.values
            )
            for epoch in self.trajectory.sequence_definition.epochs_vec
        ]
        self.n = self.propagator.full_state_vector.size
        self.events_epochs = [
            sublist[0] for sublist in self.sequence_epochs if len(sublist) == 1
        ]

        self.legs_n = [
            num
            for num in self.trajectory._sequence_definition.states_n
            if isinstance(num, (int, np.integer))
        ]
        self.legs_epochs = [lst for lst in self.sequence_epochs if len(lst) > 1]
        self.legs_model = [
            model
            for model in self.trajectory._sequence_definition.models
            if isinstance(model, StateArray)
        ]
        self.new_sigmas = [
            elem.values
            for elem in self.initial_covariance.extra_apriori_sigmas.values()
        ]
        self.flag_node = False

    def _apply_settings(self) -> None:
        """
        Apply :class:`~scarabaeus.FilterSettings` to the filter after trajectory initialisation.

        Processes optional settings fields in order:

        1. ``initial_deviation`` — if provided, calls :meth:`_set_initial_deviation`.
        2. ``process_noise``     — if provided, calls :meth:`_configure_process_noise`
           which sets ``SNC_flag`` or ``DMC_flag`` and stores noise parameters.
        3. ``measurement_sigmas`` — if provided, overrides per-dataset sigma values
           via :meth:`_apply_measurement_overrides`.
        4. ``covariance_analysis`` — copies the boolean flag onto ``self``.

        Returns
        -------
        None
        """
        # Initial deviation
        if self.settings.initial_deviation is not None:
            self._set_initial_deviation(self.settings.initial_deviation)

        # Process noise
        if self.settings.process_noise is not None:
            self._configure_process_noise(self.settings.process_noise)

        # Measurement overrides
        if self.settings.measurement_sigmas is not None:
            self._apply_measurement_overrides(self.settings.measurement_sigmas)

        # Flags
        self.covariance_analysis = self.settings.covariance_analysis

    def _set_initial_deviation(self, x0: np.ndarray) -> None:
        """Set initial state deviation."""
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 1:
            x0 = x0.reshape(-1, 1)

        expected_size = self.n if hasattr(self, "n") else None
        if expected_size and x0.shape[0] != expected_size:
            raise ValueError(
                f"Initial deviation size ({x0.shape[0]}) != state dim ({expected_size})."
            )
        self.initial_deviation = x0

    def _configure_process_noise(self, pn: ProcessNoiseSettings) -> None:
        """Configure process noise model."""
        self.process_noise_type = pn.type
        self.process_noise_Q_cont = pn.Q_cont

        if pn.type == "DMC" and pn.beta is not None:
            self.process_noise_beta = pn.beta
            self.DMC_flag = True

        if pn.type == "SNC":
            self.SNC_flag = True

    def _apply_measurement_overrides(self, overrides: Dict[str, float]) -> None:
        """Apply measurement sigma overrides by dataset name."""
        available = {ds.set_name for ds in self.measurement_data.datasets}
        unknown = set(overrides.keys()) - available
        if unknown:
            raise ValueError(
                f"measurement_sigmas contains unknown dataset name(s): {unknown}. "
                f"Available datasets: {available}."
            )

        for ds in self.measurement_data.datasets:
            if ds.set_name in overrides:
                sigma = overrides[ds.set_name]
                new_sigma = (
                    float(sigma.values) if hasattr(sigma, "values") else float(sigma)
                )
                ds.data["sigma"] = np.full_like(np.array(ds.data["sigma"]), new_sigma)
        self.measurement_data._build_indices()

    def _initialize_process_noise_from_settings(self) -> None:
        """
        Initialize process noise model from settings.

        Creates the appropriate process noise object (SNC, DMC, or STOC)
        based on settings and computes discrete-time covariances.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If process noise type is not recognized.
        """
        pn_settings = self.settings.process_noise

        # Get state and sequence definitions
        state_def = self.trajectory.state_definition
        seq_def = self.trajectory.sequence_definition

        # Create appropriate process noise object
        if pn_settings.type == "SNC":
            self.process_noise = StateNoiseCompensation(
                continuous_time_covariance=pn_settings.Q_cont,
                state_definition=state_def,
                sequence_definition=seq_def,
            )

        elif pn_settings.type == "DMC":
            self.process_noise = DynamicModelCompensation(
                continuous_time_covariance=pn_settings.Q_cont,
                state_definition=state_def,
                sequence_definition=seq_def,
                beta=pn_settings.beta,
            )

        else:
            raise ValueError(
                f"Unknown process noise type: {pn_settings.type}. "
                "Must be 'SNC' or 'DMC'."
            )

        # Compute discrete-time covariances
        spacecraft_times = self.measurement_data.get_spacecraft_times()
        self.process_noise._compute_discrete_time_covariances(spacecraft_times)

        # Set flag
        self.SNC_flag = pn_settings.type == "SNC"
        self.DMC_flag = pn_settings.type == "DMC"

    def _initialize(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
    ) -> Tuple[Trajectory, FilterDataManager]:
        """
        Initialize the reference trajectory and measurement data used by the filter.

        This method propagates the supplied dynamical model, writes the propagated
        reference trajectory to an SPICE BSP file, attaches the associated STMs, and
        constructs the measurement datasets required by the orbit determination
        filter.  If a :class:`~scarabaeus.MissionSequence` is supplied, it is first
        propagated into a :class:`~scarabaeus.ScenarioSetup`, which is then used as
        the downstream propagator representation.

        Measurement descriptors are converted into concrete measurement datasets by
        updating each measurement model with the current reference state and calling
        its dataset-generation routine.  The resulting datasets are collected into a
        :class:`~scarabaeus.FilterDataManager`, which also applies the configured
        measurement editing rules.  Outlier flags are preserved across repeated
        filter iterations by accumulating edited measurements with a logical OR.

        Parameters
        ----------
        propagator : Propagator or MissionSequence
            Dynamical model used to generate the reference trajectory.  A standard
            propagator is propagated directly.  A mission sequence is first converted
            into a propagated :class:`~scarabaeus.ScenarioSetup`.
        measurements : MeasurementSpec
            Iterable measurement specification.  Each entry must contain a
            ``"model"`` field and may optionally contain ``"observed_meas"``,
            ``"dataset_name"``, and ``"epochs"`` fields.
        traj_name : str, optional
            Name of the trajectory BSP file.  If not provided, a unique name is
            generated from the target body name and the current time.
        traj_dir : str, optional
            Directory where the trajectory BSP file is written.  Defaults to
            ``./data/kernels/scenario/`` relative to the current working directory.
        overwrite_traj : bool, optional
            If ``True``, remove an existing BSP file with the same name before
            writing the new trajectory.

        Returns
        -------
        trajectory : Trajectory
            Propagated reference trajectory with STMs attached.
        fdm : FilterDataManager
            Measurement data manager containing the generated datasets, editing
            metadata, and accumulated outlier flags.

        Raises
        ------
        ValueError
            If the propagated model does not contain an STM.
        ValueError
            If the target body cannot be inferred from
            ``propagator.primary_body``.
        ValueError
            If a measurement descriptor does not contain the required ``"model"``
            field.

        Notes
        -----
        This method is called during filter construction and again during iterative
        re-linearization.  For this reason, measurement editing flags are not reset
        after the first call: once a measurement has been marked as an outlier, it
        remains excluded in subsequent iterations.
        """

        # Directories
        traj_dir = traj_dir or (os.getcwd() + "/data/kernels/scenario/")
        os.makedirs(traj_dir, exist_ok=True)

        # Propagate trajectory
        if isinstance(propagator, MissionSequence):
            scenario = propagator.propagate()
            self.propagator = scenario  # swap to ScenarioSetup for downstream use
            propagator = scenario
        else:
            propagator.propagate()

        # Trajectory filename
        body_obj = getattr(propagator, "primary_body", None)
        body_str = (
            body_obj.name if (body_obj and hasattr(body_obj, "name")) else "target"
        )
        if traj_name is None:
            traj_name = f"{body_str}_traj_{int(time.time())}.bsp"
            print(f"Auto-generated trajectory name: {traj_name}")
        traj_path = os.path.join(traj_dir, traj_name)
        if overwrite_traj and os.path.isfile(traj_path):
            os.remove(traj_path)

        # Build trajectory
        if isinstance(propagator, ScenarioSetup):
            trajectory = Trajectory(traj_name, leg_array=propagator)
            trajectory.write_to_spk(traj_path)

        else:
            trajectory = Trajectory(
                traj_name, state_array=propagator.propagated_state_array
            )
            trajectory.write_to_spk(traj_path)
        if hasattr(propagator, "STM") and propagator.STM is not None:
            trajectory.add_STMs(propagator.STM)
        else:
            raise ValueError("Propagator must have STM for OD.")

        # Build measurement datasets
        if body_obj is None:
            raise ValueError(
                "Cannot infer target spacecraft from propagator.primary_body."
            )

        datasets = []
        for i, meas in enumerate(measurements):
            if "model" not in meas:
                raise ValueError(
                    f"Measurement descriptor {i} missing required keys 'model'."
                )

            model = meas["model"]
            observed_meas = meas.get("observed_meas", None)
            dataset_name = meas.get("dataset_name", None)
            epochs = meas.get("epochs", None)
            model.update_reference_state(self.propagator.full_state_vector)
            ds = model.generate_measurement_dataset(
                dataset_name,
                target=body_obj,
                observed_meas=observed_meas,
                epochs=epochs,
            )
            datasets.append(ds)

        # Build FilterDataManager
        fdm = FilterDataManager(
            datasets,
            editing_method=self.settings.editing_method,
            editing_datasets=self.settings.editing_datasets,
            editing_kwargs=self.settings.editing_kwargs,
            save_path=self.settings.output.fdm_output_path,
            save_name=self.settings.output.fdm_output_name,
        )

        # First call: initialize from whatever editing produced
        if not hasattr(self, "_edited_outlier_flags"):
            self._edited_outlier_flags = {
                ds.set_name: np.array(ds.data["outlier_flag"], dtype=bool).copy()
                for ds in fdm.datasets
            }
        # Subsequent calls: merge new flags with accumulated ones (OR)
        else:
            for ds in fdm.datasets:
                if ds.set_name in self._edited_outlier_flags:
                    new_flags = np.array(ds.data["outlier_flag"], dtype=bool)
                    self._edited_outlier_flags[ds.set_name] |= new_flags
                    ds.data["outlier_flag"] = self._edited_outlier_flags[
                        ds.set_name
                    ].copy()
            fdm._build_indices()

        return trajectory, fdm

    def fit(
        self,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-6,
        verbose: bool = True,
        if_sequential_smooth: bool = False,
    ):
        """
        Perform iterative orbit determination with automatic convergence checking.

        Runs the filter iteratively, applying state corrections from each solution
        to the next iteration until convergence or maximum iterations is reached.

        Parameters
        ----------
        traj_name : str, optional
            Base name for trajectory files. Iteration number will be appended.
            If None, auto-generates names. Defaults to None.
        traj_dir : str, optional
            Directory for trajectory files. Defaults to None (current directory).
        overwrite_traj : bool, optional
            Whether to overwrite existing trajectories. Defaults to True.
        max_iterations : int, optional
            Maximum number of iterations. Defaults to 10.
        convergence_threshold : float, optional
            RMS deviation threshold for convergence. Defaults to 1e-6.
        verbose : bool, optional
            If True, prints iteration progress. Defaults to True.
        if_sequential_smooth : bool, optional
            If True, applies sequential smoothing after filtering. Defaults to False.

        Returns
        -------
        solution : SolutionOD
            Final filter solution from the last iteration.
        iteration_count : int
            Number of iterations performed.
        converged : bool
            Whether convergence was achieved.

        Examples
        --------
        >>> # Automatic iteration with convergence checking
        >>> lkf = LKF(propagator, covariance, measurements)
        >>> solution, n_iters, converged = lkf.iterate(
        ...     max_iterations=5,
        ...     convergence_threshold=1e-6
        ... )
        >>> print(f"Converged in {n_iters} iterations: {converged}")
        """

        if if_sequential_smooth and type(self).__name__ in ("LSB", "SRIFB"):
            raise NotImplementedError(
                f"Sequential smoothing (if_sequential_smooth=True) is not supported "
                f"for batch filters ({type(self).__name__}). Smoothing requires "
                f"epoch-by-epoch time updates, which batch filters do not perform. "
                f"Use SRIF or LKF instead."
            )

        if self.covariance_analysis:
            max_iterations = 1  # Force single pass semantics
            if_sequential_smooth = False  # No post-fit residual smoothing in cov-only

        if verbose:
            print("\n" + "=" * 80)
            print("STARTING ITERATIVE ORBIT DETERMINATION")
            print("=" * 80)
            print(f"Max iterations: {max_iterations}")
            if not self.covariance_analysis:
                print(f"Convergence threshold: {convergence_threshold:.2e}")
            else:
                print("Mode: covariance analysis (prediction-only)")
            print("=" * 80 + "\n")

        converged = False
        solution = None
        rms_deviation = 0.0

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration}")
                print(f"{'='*60}")

            # Compute filter solution
            solution = self._compute(printOutput=verbose)
            self._solution_history.append(solution)

            # Apply sequential smoothing if requested
            if if_sequential_smooth and (not self.covariance_analysis):
                solution = self._smoother(printOutput=verbose)
                if self.settings.output.save_smoothed_solution:
                    self._solution_history_smooth.append(solution)

            if self.covariance_analysis:
                # No deviation / no convergence test / no reinit
                converged = True
                if verbose:
                    print(f"\n{'='*60}")
                    print("✓ DONE (covariance analysis: single pass)")
                    print(f"{'='*60}\n")
                break

            # Get state deviation from solution
            deviation = solution.map_state_deviation_to_epoch()
            if self.flag_sequence:
                # NOTE: In any case, you assume that the nodes are continuous, even if
                # they are not explicitly mapped back with the proper flag, so it is acceptable to discard
                # redundant position estimates, for example.
                deviation = self.propagator._deviation_history_to_global_dx(deviation)

            # Accumulate total deviation w.r.t. first reference
            self._delta0_total = self._delta0_total + deviation
            self._delta0_hist.append(self._delta0_total.copy())

            # Accumulate per-epoch deviation wrt first reference
            if solution.deviation_est is not None:
                if self._cumulative_deviation is None:
                    self._cumulative_deviation = solution.deviation_est.copy()
                else:
                    self._cumulative_deviation = (
                        self._cumulative_deviation + solution.deviation_est
                    )

            # Check convergence
            rms_deviation = np.sqrt(np.mean(deviation**2))

            if verbose:
                print(f"RMS State Deviation: {rms_deviation:.6e}")

            if rms_deviation < convergence_threshold:
                converged = True
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"✓ CONVERGED after {iteration} iteration(s)!")
                    print(f"{'='*60}\n")
                break

            # Reinitialize for next iteration (if not last iteration)
            if iteration < max_iterations:
                # Generate trajectory name for next iteration
                if traj_name is not None:
                    # Strip .bsp extension if present
                    base_name = traj_name.replace(".bsp", "")
                    next_traj_name = f"{base_name}_it{iteration+1}.bsp"
                else:
                    next_traj_name = None

                if verbose:
                    print(f"-> Reinitializing for iteration {iteration+1}...")

                self._reinitialize(
                    state_deviation=deviation,
                    traj_name=next_traj_name,
                    traj_dir=traj_dir,
                    overwrite_traj=overwrite_traj,
                )

        if (not self.covariance_analysis) and (not converged) and verbose:
            print(f"\n{'='*60}")
            print(f"!! Did NOT converge after {max_iterations} iterations")
            print(f"  Final RMS deviation: {rms_deviation:.6e}")
            print(f"{'='*60}\n")

        # Automatically save final solution if requested through OutputSettings
        if solution is not None and getattr(self.settings, "output", None) is not None:
            out = self.settings.output
            if getattr(out, "solution_output_path", None) is not None:
                solution.save()

        return solution, iteration, converged

    def _reinitialize(
        self,
        state_deviation: Optional[np.ndarray] = None,
        state_vector: Optional[StateArray] = None,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
    ) -> None:
        """
        Reinitialize the filter for iterative estimation.

        Updates the state vector by applying deviations from a previous filter iteration,
        re-propagates the trajectory, and rebuilds measurement datasets. This method
        enables iterative orbit determination where each iteration refines the reference
        trajectory using corrections from the previous filter solution.

        Parameters
        ----------
        state_deviation : np.ndarray, optional
            State deviation vector from previous filter iteration (e.g., from
            solution.map_state_deviation_to_epoch()). If provided, this deviation
            is added to the current state vector. Shape should be (n,) or (n, 1).
            Defaults to None.
        state_vector : StateArray, optional
            New state vector to use. If None, uses the current propagator's state vector.
            If state_deviation is provided, it will be applied to this state vector.
            Defaults to None.
        traj_name : str, optional
            Name for the new trajectory BSP file. If None, auto-generates a name.
            Defaults to None.
        traj_dir : str, optional
            Directory to write the new trajectory. If None, uses current directory.
            Defaults to None.
        overwrite_traj : bool, optional
            If True, overwrites existing trajectory file with the same name.
            Defaults to True.

        Returns
        -------
        None
            Updates self.trajectory and self.measurement_data in place.

        Raises
        ------
        ValueError
            If state_deviation is provided but state_vector cannot be determined.

        Notes
        -----
        This method modifies the FilterOD object in place by:
        1. Applying state deviation to the state vector (if provided)
        2. Re-propagating the trajectory with the updated state
        3. Rebuilding measurement datasets
        4. Re-initializing filter attributes (sequence, ground stations, etc.)

        The body objects are NOT modified - the same spacecraft body is used across
        iterations for simplicity. Each iteration operates on the same SPICE ID.

        Examples
        --------
        >>> # Manual iteration control
        >>> lkf = LKF(propagator, covariance, measurements)
        >>>
        >>> # Iteration 1
        >>> solution_it1 = lkf._compute()
        >>> deviation_it1 = solution_it1.map_state_deviation_to_epoch()
        >>>
        >>> # Iteration 2
        >>> lkf._reinitialize(state_deviation=deviation_it1)
        >>> solution_it2 = lkf._compute()
        >>>
        >>> # Iteration 3
        >>> deviation_it2 = solution_it2.map_state_deviation_to_epoch()
        >>> lkf._reinitialize(state_deviation=deviation_it2)
        >>> solution_it3 = lkf._compute()

        See Also
        --------
        iterate : Automatic iterative estimation with convergence checking.
        _apply_state_deviation : Applies deviation to state vector components.
        _initialize : Builds trajectory and measurement datasets.
        """
        if self.flag_sequence:
            # Reinitialize the propagator with updated state
            self.propagator._reinitialize_from_deviation_global(state_deviation)
        else:
            # Determine which state vector to use
            if state_vector is None:
                # Use current propagator's state vector
                if (
                    not hasattr(self.propagator, "full_state_vector")
                    or self.propagator.full_state_vector is None
                ):
                    raise ValueError(
                        "Cannot determine state vector. Either provide state_vector parameter "
                        "or ensure propagator has a valid state_vector attribute."
                    )
                state_vector = self.propagator.full_state_vector

            # Apply state deviation if provided
            if state_deviation is not None:
                state_vector = self._apply_state_deviation(
                    state_vector, state_deviation
                )

            # Reinitialize the propagator with updated state
            self.propagator.reinitialize(state_vector=state_vector)

        # Rebuild trajectory and measurement data
        self.trajectory, self.measurement_data = self._initialize(
            propagator=self.propagator,
            measurements=self.measurements,
            traj_name=traj_name,
            traj_dir=traj_dir,
            overwrite_traj=overwrite_traj,
        )

        # Re-check sequence definition (may have changed)
        self.flag_sequence = self.trajectory.sequence_definition is not None

        # Re-initialize sequence or non-sequence attributes
        if self.flag_sequence:
            self._initialize_sequence_attributes()
        else:
            self._initialize_nonsequence_attributes()

        # Re-generate consider parameter partition (if applicable)
        self._init_consider_partition()
        self._apply_settings()

    def _apply_state_deviation(
        self,
        state_vector: StateArray,
        state_deviation: np.ndarray,
    ) -> StateArray:
        """
        Apply state deviation to state vector components.

        Adds deviation from filter solution to each component of the state vector,
        properly handling position, velocity, and other parameter types while
        preserving units and frames.

        Parameters
        ----------
        state_vector : StateArray
            Original state vector to update.
        state_deviation : np.ndarray
            Deviation vector to apply, shape (n,) or (n, 1).

        Returns
        -------
        StateArray
            New state vector with deviation applied.

        Raises
        ------
        ValueError
            If deviation size doesn't match state vector size.
        """
        # Flatten deviation
        deviation = np.atleast_1d(state_deviation.flatten())

        # Check size consistency
        expected_size = sum(entry[1] for entry in state_vector.state)
        if len(deviation) != expected_size:
            raise ValueError(
                f"State deviation size ({len(deviation)}) does not match "
                f"state vector size ({expected_size})."
            )

        # Build new state with deviation applied
        new_state = []
        dev_idx = 0

        for state_entry in state_vector.state:
            state_type, size, estimate_flag, component_type, body, value = state_entry

            # Extract deviation for this component
            component_deviation = deviation[dev_idx : dev_idx + size]
            dev_idx += size

            # Apply deviation based on state type
            new_value = ArrayWFrame(
                value.quantity + ArrayWUnits(component_deviation, value.quantity.units),
                value.frame,
            )

            # Keep same body (no body updates across iterations)
            new_state.append(
                (state_type, size, estimate_flag, component_type, body, new_value)
            )

        # Create new StateArray with updated state
        return StateArray(
            epoch=state_vector.epoch,
            origin=state_vector.origin,
            state=StateDefinition.from_components(new_state),
        )

    def _unwrap_residuals(
        self, postfit_residuals: list, dataset_names: list[str]
    ) -> dict:
        """
        Organizes postfit residuals into a dictionary categorized by dataset.

        Parameters
        ----------
        postfit_residuals : list
            List of residuals and associated metadata.
        dataset_names : list
            List of dataset names corresponding to different measurements.

        Returns
        -------
        dict
            Dictionary mapping dataset names to corresponding residuals and sigmas.
        """
        # Initialize dictionary with empty lists for each dataset
        unwrapped = {name: [] for name in dataset_names}

        # Loop through epochs
        for epoch_residuals in postfit_residuals:
            residuals, sigmas, _, _, sets, _ = epoch_residuals

            # Assign residuals to their corresponding dataset
            for res, sigma, dataset in zip(residuals.flatten(), sigmas, sets):
                if dataset in unwrapped:
                    unwrapped[dataset].append([res, sigma])

        return unwrapped

    def _map_covariance_definition_node(
        self, counter_events: int, old_covariance: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjusts the covariance matrix to match a new state vector definition.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event for state transition.

        old_covariance : numpy.ndarray
            Previous covariance matrix to be updated.

        flag_forward : bool, optional
            If ``True``, it maps from the previous state definition to the new one (augmentation);
            Defaults to ``True``.

        Returns
        -------
        updated_covar : numpy.ndarray
            Updated covariance matrix reflecting the new state definition.
        """
        # Get state definitions
        if flag_forward:
            new_list = self.legs_model[counter_events].state_definition
            old_list = self.legs_model[counter_events - 1].state_definition
        else:
            new_list = self.legs_model[counter_events - 1].state_definition
            old_list = self.legs_model[counter_events].state_definition

        # Define new state size and covariance matrix
        new_state_size = sum(element[1] for element in new_list)
        new_covariance = np.zeros((new_state_size, new_state_size))

        # Create state mappings and compute offsets
        old_state_mapping = self._create_state_mapping(old_list)
        new_state_mapping = self._create_state_mapping(new_list)
        old_offsets = self._compute_offsets(old_state_mapping)
        new_offsets = self._compute_offsets(new_state_mapping)

        # Fill in the new covariance matrix
        for new_id, (_, new_size) in new_state_mapping.items():
            if new_id in old_state_mapping:
                old_offset = old_offsets[new_id]
                new_offset = new_offsets[new_id]

                new_covariance[
                    new_offset : new_offset + new_size,
                    new_offset : new_offset + new_size,
                ] = old_covariance[
                    old_offset : old_offset + new_size,
                    old_offset : old_offset + new_size,
                ]

                for other_new_id, (_, other_new_size) in new_state_mapping.items():
                    if other_new_id in old_state_mapping and new_id != other_new_id:
                        other_old_offset = old_offsets[other_new_id]
                        other_new_offset = new_offsets[other_new_id]

                        new_covariance[
                            new_offset : new_offset + new_size,
                            other_new_offset : other_new_offset + other_new_size,
                        ] = old_covariance[
                            old_offset : old_offset + new_size,
                            other_old_offset : other_old_offset + other_new_size,
                        ]
                        new_covariance[
                            other_new_offset : other_new_offset + other_new_size,
                            new_offset : new_offset + new_size,
                        ] = old_covariance[
                            other_old_offset : other_old_offset + other_new_size,
                            old_offset : old_offset + new_size,
                        ]
            else:
                new_offset = new_offsets[new_id]
                for i in range(new_size):
                    if new_id not in self.initial_covariance.extra_apriori_sigmas:
                        raise ValueError(
                            f"No sigma provided for new state element '{new_id}'."
                        )
                    new_covariance[new_offset + i, new_offset + i] = (
                        float(
                            self.initial_covariance.extra_apriori_sigmas[
                                new_id
                            ].values.ravel()[i]
                        )
                        ** 2
                    )

        return new_covariance

    def _map_deviation_definition_node(
        self, counter_events: int, old_deviation: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjusts the deviation vector to align with a new state vector definition.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event for state transition.

        old_deviation : numpy.ndarray
            Previous state deviation vector.

        flag_forward : bool, optional
            If ``True``, it maps from the previous state definition to the new one (augmentation);
            Defaults to ``True``.

        Returns
        -------
        updated_dev : numpy.ndarray
            Updated deviation vector reflecting the new state definition.
        """
        if flag_forward:
            new_list = self.legs_model[counter_events].state_definition
            old_list = self.legs_model[counter_events - 1].state_definition
        else:
            new_list = self.legs_model[counter_events - 1].state_definition
            old_list = self.legs_model[counter_events].state_definition

        new_state_size = sum(element[1] for element in new_list)
        new_deviation = np.zeros((new_state_size, 1))

        old_state_mapping = self._create_state_mapping(old_list)
        new_state_mapping = self._create_state_mapping(new_list)
        old_offsets = self._compute_offsets(old_state_mapping)
        new_offsets = self._compute_offsets(new_state_mapping)

        for new_id, (_, new_size) in new_state_mapping.items():
            # NOTE: if not in old_state_mapping, the deviation is already
            # initialized to zero. It's an expected behavior when augmenting
            # since the deviation is initialized to zero at the beginning of the filter.
            # When not augmenting, an error is raised.
            if new_id in old_state_mapping:
                old_offset = old_offsets[new_id]
                new_offset = new_offsets[new_id]

                old_deviation = np.asarray(old_deviation).ravel()  # (N,)
                new_deviation = np.asarray(new_deviation).ravel()  # (M,)

                new_deviation[new_offset : new_offset + new_size] = old_deviation[
                    old_offset : old_offset + new_size
                ]
            else:
                if flag_forward == False:
                    raise ValueError(
                        "Deviation mapping has exhibited unexpected behavior, with deviation elements missing during reduction."
                    )

        return new_deviation

    def _map_stm_definition_node(
        self, counter_events: int, old_STM: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjusts the State Transition Matrix (STM) to match a new state definition.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event for state transition.

        old_STM : numpy.ndarray
            Previous STM to be updated.

        flag_forward : bool, optional
            If ``True``, it maps from the previous state definition to the new one (augmentation);
            Defaults to ``True``.

        Returns
        -------
        updated_stm : numpy.ndarray
            Updated State Transition Matrix reflecting the new state definition.
        """
        # Get state definitions
        if flag_forward:
            new_list = self.legs_model[counter_events].state_definition
            old_list = self.legs_model[counter_events - 1].state_definition
        else:
            new_list = self.legs_model[counter_events - 1].state_definition
            old_list = self.legs_model[counter_events].state_definition

        # Define new state size and initialize the STM
        new_state_size = sum(element[1] for element in new_list)
        new_STM = np.eye(new_state_size)  # Initialize STM as identity matrix

        # Create state mappings and compute offsets
        old_state_mapping = self._create_state_mapping(old_list)
        new_state_mapping = self._create_state_mapping(new_list)
        old_offsets = self._compute_offsets(old_state_mapping)
        new_offsets = self._compute_offsets(new_state_mapping)

        # Fill in the new STM
        for new_id, (_, new_size) in new_state_mapping.items():
            if new_id in old_state_mapping:
                old_offset = old_offsets[new_id]
                new_offset = new_offsets[new_id]

                new_STM[
                    new_offset : new_offset + new_size,
                    new_offset : new_offset + new_size,
                ] = old_STM[
                    old_offset : old_offset + new_size,
                    old_offset : old_offset + new_size,
                ]

                for other_new_id, (_, other_new_size) in new_state_mapping.items():
                    if other_new_id in old_state_mapping and new_id != other_new_id:
                        other_old_offset = old_offsets[other_new_id]
                        other_new_offset = new_offsets[other_new_id]

                        new_STM[
                            new_offset : new_offset + new_size,
                            other_new_offset : other_new_offset + other_new_size,
                        ] = old_STM[
                            old_offset : old_offset + new_size,
                            other_old_offset : other_old_offset + other_new_size,
                        ]
                        new_STM[
                            other_new_offset : other_new_offset + other_new_size,
                            new_offset : new_offset + new_size,
                        ] = old_STM[
                            other_old_offset : other_old_offset + other_new_size,
                            old_offset : old_offset + new_size,
                        ]
            else:
                raise ValueError(
                    "STM mapping is not supported for new state elements. Missing STM elements."
                )

        return new_STM

    def _create_state_mapping(self, state_list):
        """
        Generates a mapping from state variable names to their indices and sizes.

        Parameters
        ----------
        state_list : list
            List of state variable names and their associated sizes.

        Returns
        -------
        dict
            Dictionary mapping state variable names to (index, size) tuples.
        """
        return {element[0]: (idx, element[1]) for idx, element in enumerate(state_list)}

    def _compute_offsets(self, state_mapping):
        """
        Computes the starting index (offset) for each state variable based on its size.

        Parameters
        ----------
        state_mapping : dict
            Mapping of state variable names to their indices and sizes.

        Returns
        -------
        dict
            Dictionary mapping state variable names to their offsets in the state vector.
        """
        return {
            key: sum(value[1] for value in list(state_mapping.values())[:i])
            for i, (key, value) in enumerate(state_mapping.items())
        }

    def _validate_sequence_batch(self):
        """
        Validates consistency in state vector definitions across sequential estimation legs.

        Ensures that each state vector in a sequence contains all necessary elements from previous legs.

        Returns
        -------
        None
        """
        # Initialize list of state vectors
        list_state_vectors = [
            obj.state for obj in self.legs_model if isinstance(obj, StateArray)
        ]

        # Take consecutive pairs of state_vectors
        for i in range(1, len(list_state_vectors)):
            prev_state_vec = list_state_vectors[i - 1]
            curr_state_vec = list_state_vectors[i]

            # Initialize lists of IDs for previous and current state vectors
            prev_ids = [prev_tuple[0] for prev_tuple in prev_state_vec]
            curr_ids = [curr_tuple[0] for curr_tuple in curr_state_vec]

            # Ensure that every element in the previous state vector is present in the current state vector (i.e., only increment components)
            for prev_id in prev_ids:
                if prev_id not in curr_ids:
                    print(
                        f"Warning: State vector leg {i} does not contain element '{prev_id}'"
                    )

                    # raise Exception(
                    #    f"Inconsistency detected: State vector leg {i} does not contain element '{prev_id}' from state vector leg {i-1}.\n"
                    #    "The current implementation needs that for the batch filtering."
                    # )

    def observability_test(self, print_obs=False, threshold=1e-1, meas_corr=None):
        """
        Compute the observability and information matrices for the current filter setup.

        The **observability matrix** :math:`\\mathcal{O}` is formed by stacking the
        measurement sensitivity matrices mapped back to the reference epoch :math:`t_0`
        via the State Transition Matrix (STM):

        .. math::

            \\mathcal{O} = \\begin{bmatrix}
                H_1\\,\\Phi_{1,0} \\\\
                H_2\\,\\Phi_{2,0} \\\\
                \\vdots \\\\
                H_k\\,\\Phi_{k,0}
            \\end{bmatrix}

        where :math:`H_k` is the measurement Jacobian at epoch :math:`t_k` and
        :math:`\\Phi_{k,0}` is the STM from :math:`t_0` to :math:`t_k`.  The system
        is **observable** if and only if :math:`\\mathcal{O}` has full column rank
        :math:`n` (the state dimension).

        The **information matrix** (Fisher information / weighted normal matrix)
        :math:`\\Lambda` accumulates the measurement information across all epochs:

        .. math::

            \\Lambda = \\sum_{k} H_{k0}^T\\, R_k^{-1}\\, H_{k0}

        where :math:`H_{k0} = H_k\\,\\Phi_{k,0}` and :math:`R_k` is the measurement
        noise covariance at epoch :math:`k`.  The system is observable if and only if
        :math:`\\Lambda` is positive definite, i.e. all its eigenvalues exceed
        ``threshold``.

        Parameters
        ----------
        print_obs : bool, optional
            If True, prints the state dimension, rank of :math:`\\Lambda`, and
            whether it is strictly positive definite.
        threshold : float, optional
            Minimum eigenvalue threshold used to assess positive definiteness of
            :math:`\\Lambda`. Defaults to ``1e-1``.
        meas_corr : list, optional
            Off-diagonal correlation coefficients for the per-epoch measurement
            noise covariance :math:`R_k`.  Must be a flat list of
            :math:`n_m (n_m - 1) / 2` values covering the upper triangular part
            of the correlation matrix in row-major order, where :math:`n_m` is
            the number of simultaneous measurements at each epoch.  When ``None``,
            :math:`R_k` is diagonal (measurements assumed uncorrelated).

        Returns
        -------
        info_matrix, obs_matrix : tuple[numpy.ndarray, numpy.ndarray]
            A tuple with the following values corresponding to
            their respective indices:

            * ``[0]`` = info_matrix : numpy.ndarray of shape ``(n, n)``
                The information matrix :math:`\\Lambda`.
            * ``[1]`` = obs_matrix : numpy.ndarray of shape ``(m, n)``
                The observability matrix :math:`\\mathcal{O}`, where ``m`` is the
                total number of measurements across all epochs.
        """
        # Check is not sequence
        if self.flag_sequence is True:
            raise ValueError(
                "The observability test is not available for the sequence mode."
            )

        # Initialize
        spacecraft_times = self.measurement_data.get_spacecraft_times()

        # Construct H observability matrix
        for k in range(len(spacecraft_times)):
            #### Extract Measurement Data for Epoch t2 ####
            block = self.measurement_data.get_combined_for_t2(
                spacecraft_times[k]
            )  # returns dict keyed by t2_val
            block = block[spacecraft_times[k]]
            sigmas = block["sigma"]
            partials = block["partials"][:, : self.n]

            # STM from t0 to tk
            STM_k0 = self.trajectory.get_STM(epoch=spacecraft_times[k], idx=k)

            # Measurement covariance
            measurement_covariance = np.array(
                CovarianceMatrix(
                    sigmas,
                    spacecraft_times[k],
                    from_list=True,
                    corr_factors=meas_corr,
                ).matrix
            )

            # Compute H and Lambda matrices
            # NOTE: each row i-th is Htilde_i * STM(t_0, t_i)
            H_k0 = partials @ STM_k0
            if k == 0:
                observability_matrix = H_k0
                information_matrix = (
                    H_k0.T @ self._safe_inv(measurement_covariance) @ H_k0
                )
            else:
                observability_matrix = np.vstack((observability_matrix, H_k0))
                information_matrix += (
                    H_k0.T @ self._safe_inv(measurement_covariance) @ H_k0
                )

        # Print observability properties
        if print_obs is True:
            # Analyze matrix H
            print("Length of the state vector:", self.n)
            print(
                "Observability matrix H rank: ",
                np.linalg.matrix_rank(information_matrix),
            )
            if np.linalg.matrix_rank(information_matrix) < self.n:
                print("Observability matrix H un-sufficient rank.")
            else:
                print("Observability matrix H sufficient rank.")
            # Analyze matrix Lambda
            eigvals_Lambda = np.linalg.eigvals(information_matrix)
            if np.all(eigvals_Lambda > threshold):
                print("The information matrix Lambda is strictly positive definite.")
            else:
                print(
                    "The information matrix Lambda is not strictly positive definite."
                )

        return information_matrix, observability_matrix

    def _init_consider_partition(self, idx_leg=0) -> None:
        """
        Initialize the partition of state variables into estimated and considered parameters.

        This method processes the trajectory state definition to categorize state variables
        based on their estimation flags. Variables marked as "considered" are separated from
        those to be estimated, and their indices are stored for later use in filtering operations.

        The method also computes the total state dimension and performs sanity checks to ensure
        consistency between the state definition size and the STM (State Transition Matrix)
        inferred size.

        Parameters
        ----------
        idx_leg : int, optional
            Index of the leg to use for state definition when in sequence mode. Defaults to 0

        Raises
        ------
        warnings.UserWarning
            If the state definition size differs from the STM inferred size, indicating
            potential inconsistencies in state ordering or STM mapping.


        Attributes Set
        ---------------
        idx_est : np.ndarray
            Integer array of indices for estimated state variables.
        idx_con : np.ndarray
            Integer array of indices for considered (not estimated) state variables.
        n_full : int
            Total dimension of the full state vector.
        n_est : int
            Number of estimated state variables.
        n_con : int
            Number of considered (not estimated) state variables.
        has_consider : bool
            Flag indicating whether there are any considered parameters (True if n_con > 0).
        """
        if self.flag_sequence:
            state_def = self.legs_model[idx_leg].state
        else:
            state_def = (
                self.trajectory.state_definition
            )  # list of tuples (name, dim, est, dyn, owner, value)

        idx_est = []
        idx_con = []

        cursor = 0
        for entry in state_def:
            name, dim, est_flag, dyn_flag, owner, value = entry
            if est_flag == "considered":
                idx_con.extend(range(cursor, cursor + dim))
            else:
                idx_est.extend(range(cursor, cursor + dim))
            cursor += dim

        self.idx_est = np.asarray(idx_est, dtype=int)
        self.idx_con = np.asarray(idx_con, dtype=int)

        self.n_full = cursor
        self.n_est = int(self.idx_est.size)
        self.n_con = int(self.idx_con.size)

        self.has_consider = self.n_con > 0

    @property
    def apriori_covariance_est(self) -> "np.ndarray":
        """
        Apriori covariance restricted to the estimated parameter subspace.

        Returns the ``(n_est x n_est)`` block of the full initial covariance
        matrix corresponding to estimated (non-considered) state variables.
        When no consider parameters are present the full matrix is returned.

        Returns
        -------
        np.ndarray
            Square apriori covariance of shape ``(n_est, n_est)``.
        """
        P0 = self.initial_covariance.matrix_without_units()
        if self.has_consider:
            return P0[np.ix_(self.idx_est, self.idx_est)]
        return P0

    def _split_stm_phi_psi(self, STM: np.ndarray) -> tuple:
        """
        Partition the State Transition Matrix into estimated and consider sub-matrices.

        Extracts three sub-blocks from the full-state STM according to the
        estimated (``idx_est``) and considered (``idx_con``) parameter indices.

        Parameters
        ----------
        STM : np.ndarray
            Full State Transition Matrix of shape ``(n_full, n_full)``.

        Returns
        -------
        Phi : np.ndarray
            STM block for estimated-to-estimated coupling,
            shape ``(n_est, n_est)``.
        Psi : np.ndarray
            STM block for consider-to-estimated coupling (sensitivity of
            estimated states to consider perturbations),
            shape ``(n_est, n_con)``.  Zero-column matrix when no consider
            parameters exist.
        Phi_c : np.ndarray
            STM block for consider-to-consider coupling,
            shape ``(n_con, n_con)``.  Zero matrix when no consider
            parameters exist.
        """
        if not self.has_consider:
            Phi = STM[np.ix_(self.idx_est, self.idx_est)]
            Psi = np.zeros((Phi.shape[0], 0))
            Phi_c = np.zeros((0, 0))
            return Phi, Psi, Phi_c

        Phi = STM[np.ix_(self.idx_est, self.idx_est)]
        Psi = STM[np.ix_(self.idx_est, self.idx_con)]
        Phi_c = STM[np.ix_(self.idx_con, self.idx_con)]
        return Phi, Psi, Phi_c

    def _split_partials_hx_hc(self, H: np.ndarray) -> tuple:
        """
        Partition the measurement partial-derivative matrix into estimated and consider columns.

        Parameters
        ----------
        H : np.ndarray
            Full measurement partial-derivative matrix of shape ``(m, n_full)``,
            where ``m`` is the number of measurements at a given epoch and
            ``n_full`` is the total state dimension.

        Returns
        -------
        Hx : np.ndarray
            Columns corresponding to estimated parameters,
            shape ``(m, n_est)``.
        Hc : np.ndarray
            Columns corresponding to consider parameters,
            shape ``(m, n_con)``.  Zero-column matrix ``(m, 0)`` when no
            consider parameters are configured.

        Raises
        ------
        IndexError
            If ``idx_est`` or ``idx_con`` contain out-of-range column indices.
        """
        if not self.has_consider:
            Hx = H
            Hc = np.zeros((H.shape[0], 0))
            return Hx, Hc

        Hx = H[:, self.idx_est]
        Hc = H[:, self.idx_con]
        return Hx, Hc

    def _safe_inv(self, A, min_svd=1e-14):
        """
        Compute a numerically robust matrix inverse.

        This routine first attempts a standard matrix inversion using
        :func:`numpy.linalg.inv`. If the matrix is singular or numerically
        ill-conditioned, the function falls back to a Moore-Penrose
        pseudo-inverse computed from the singular value decomposition (SVD).

        A runtime warning is issued whenever the pseudo-inverse is used.

        Parameters
        ----------
        A : ndarray of shape (n, n)
            Matrix to invert.

        min_svd : float, optional
            Minimum allowed singular value used during pseudo-inverse
            construction. Singular values smaller than this threshold are
            clipped to ``min_svd`` to avoid numerical overflow and improve
            stability. Default is ``1e-14``.

        Returns
        -------
        ndarray of shape (n, n)
            Inverse or pseudo-inverse of ``A``.

        Warns
        -----
        RuntimeWarning
            Raised when the matrix inversion fails and the function falls
            back to the pseudo-inverse.

        Notes
        -----
        The pseudo-inverse is computed as

        .. math::

            A^+ = V \\Sigma^{-1} U^T

        where

        .. math::

            A = U \\Sigma V^T

        is the singular value decomposition of ``A``.

        This function is useful in orbit determination and estimation
        problems where rank-deficient or poorly conditioned matrices may
        appear during filtering or covariance propagation.
        """

        try:
            return np.linalg.inv(A)

        except np.linalg.LinAlgError:
            warnings.warn(
                "Matrix is singular or ill-conditioned. "
                "Falling back to pseudo-inverse (pinv).",
                RuntimeWarning,
            )

        # SVD-based pseudo-inverse
        U, S, Vt = np.linalg.svd(A)

        # Clip small singular values
        S = np.maximum(S, min_svd)

        return Vt.T @ np.diag(1.0 / S) @ U.T

    def _sqrt_info_matrix(self, P, min_eig=1e-14):
        """
        Compute the square-root information matrix.

        Computes an upper-triangular matrix :math:`R` such that

        .. math::

            R^T R \\approx P^{-1}

        where :math:`P` is a covariance matrix.

        The routine first attempts a Cholesky decomposition of the inverse
        covariance matrix. If Cholesky factorization fails (typically due to
        loss of positive definiteness or numerical conditioning issues),
        the function falls back to an eigenvalue-based construction.

        Parameters
        ----------
        P : ndarray of shape (n, n)
            Covariance matrix.

        min_eig : float, optional
            Minimum eigenvalue allowed during the eigenvalue fallback.
            Eigenvalues smaller than this threshold are clipped to
            ``min_eig`` to preserve numerical stability. Default is
            ``1e-14``.

        Returns
        -------
        ndarray of shape (n, n)
            Square-root information matrix.

        Warns
        -----
        RuntimeWarning
            Raised when Cholesky decomposition fails and the routine falls
            back to the eigenvalue-based formulation.

        Notes
        -----
        Primary method
        ^^^^^^^^^^^^^^

        The preferred computation is

        .. math::

            R = \\mathrm{chol}(P^{-1})

        where ``chol`` denotes the upper-triangular Cholesky factor.

        Fallback method
        ^^^^^^^^^^^^^^^

        If Cholesky decomposition fails, the matrix is reconstructed using
        the eigendecomposition

        .. math::

            P = V \\Lambda V^T

        and the square-root information matrix is approximated as

        .. math::

            R = V \\Lambda^{-1/2} V^T

        after clipping small eigenvalues.

        This approach improves robustness in estimation problems involving
        near-singular covariance matrices or rank-deficient systems.
        """

        try:
            R = spln.cholesky(self._safe_inv(P), lower=False)
            return R

        except np.linalg.LinAlgError:
            warnings.warn(
                "Cholesky factorization failed. Falling back to "
                "eigenvalue-based square-root information matrix.",
                RuntimeWarning,
            )

        # Eigenvalue fallback
        eigvals, eigvecs = np.linalg.eigh(P)

        # Clip small eigenvalues
        eigvals = np.maximum(eigvals, min_eig)

        # R ≈ sqrt(inv(P))
        R = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return R
