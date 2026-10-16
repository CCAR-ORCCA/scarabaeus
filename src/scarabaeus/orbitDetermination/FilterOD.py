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
    DynamicalModelCompensation,
    StateDefinition,
    ArrayWFrame,
    ArrayWUnits,
    ScenarioSetup,
    MissionSequence,
    MeasurementSpec,
    SpiceManager,
    Units,
    DynamicalUtils,
)
import scarabaeus.utils.NumpyWrapper as np
import functools
import os
import time
import scipy.linalg as spln
from typing import Callable, List, Dict, Tuple, Optional, Union
import warnings
from scarabaeus.utils.Utils import Utils


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
        Filter configuration settings. Its ``measurement_sigmas``, keyed by
        dataset name, are set once, at construction and before the first
        editing pass, on the model behind each dataset they name (see
        :meth:`set_measurement_sigmas`). They therefore win over the sigma the
        model was built with, ``model.sigma`` reports them, and the datasets of
        every later iteration are built with them. The model is the only
        source: a sigma set on it afterwards (through
        :meth:`set_measurement_sigmas`, ``MeasurementSpec.edit`` or
        ``model.sigma = ...``, e.g. from a :meth:`fit` callback) takes over
        from the next iteration on. An ``ArrayWUnits`` value is converted to
        the dataset's units; a bare float is taken to be in them already.
    traj_name : str, optional
        Name for the trajectory BSP file. :meth:`fit` names the trajectories of
        later iterations after it.
    traj_dir : str, optional
        Directory to write trajectory, for this and every later iteration.
        Defaults to ``./data/kernels/scenario/`` under the current directory.
    overwrite_traj : bool
        If True, overwrite existing trajectory file.
    verbose : bool, optional
        If False, silence the printout of the propagation run at construction.
        Defaults to True.

    See Also
    --------
    scarabaeus.LKF : Linearized Kalman Filter (sequential, covariance form).
    scarabaeus.SRIF : Square Root Information Filter (sequential, square-root form).
    scarabaeus.LSB : Batch Least-Squares estimator.
    scarabaeus.SRIFB : Batch Square Root Information Filter.
    scarabaeus.SolutionOD : Container for the estimated state history and covariance.

    Notes
    -----
    This class is not instantiated directly; use a concrete subclass
    (:class:`~scarabaeus.LKF`, :class:`~scarabaeus.SRIF`,
    :class:`~scarabaeus.LSB`, or :class:`~scarabaeus.SRIFB`).
    Call :meth:`fit` to run the filter and retrieve results via
    :class:`~scarabaeus.SolutionOD`.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ConvergenceCriterionError(ValueError):
        """Raised when an unknown convergence criterion is requested."""

    class MissingAprioriSigmaError(ValueError):
        """Raised when a newly augmented state has no ``extra_apriori_sigmas`` entry."""

    class AprioriSigmaUnitsError(ValueError):
        """Raised when an ``extra_apriori_sigmas`` entry cannot be expressed in the
        units of the state element it initialises."""

    class InitialCovarianceSizeError(ValueError):
        """Raised when the initial covariance does not match the size of the
        initial state (the first leg's on a sequence)."""

    class MissingGaussMarkovModelError(ValueError):
        """Raised when DMC process noise is used on a leg whose dynamics do not
        integrate the first-order Gauss-Markov acceleration it drives."""

    class ProcessNoiseBetaMismatchError(ValueError):
        """Raised when the DMC ``beta`` given in the settings differs from the
        decay rates of the Gauss-Markov acceleration in the dynamics."""

    class DuplicateProcessNoiseBodyError(ValueError):
        """Raised when two process noise settings drive the same body."""

    class NoiseReferenceUnavailableError(ValueError):
        """Raised when an RTN process noise drives a body whose reference
        trajectory is not in the trajectory SPK: on a sequence, any body other
        than the primary body of each leg."""

    class UnknownSigmaDatasetError(ValueError):
        """Raised when a measurement sigma names a dataset the filter does not
        have."""

    class MeasurementSigmaUnitsError(ValueError):
        """Raised when a measurement sigma cannot be expressed in the units of
        the dataset it is given for."""

    class SharedSigmaModelError(ValueError):
        """Raised when measurement sigmas would give one model two sigmas: the
        datasets one model builds (its components, or the entries that reuse
        it) share its sigma, so a sigma given for one of them must hold for
        all of them."""

    class SequenceStateDroppedWarning(UserWarning):
        """Issued when a sequence leg drops a state the previous leg carried.

        Dropping is supported (the state is marginalised at the node and
        recovered when mapping back), so this is informational; silence it with
        ``warnings.filterwarnings("ignore", category=FilterOD.SequenceStateDroppedWarning)``.
        """

    class UnnoisedBodyWarning(UserWarning):
        """Issued when process noise is configured but an estimated, propagated
        body has none.

        Leaving a body noise-free is legitimate, so this is informational;
        silence it with
        ``warnings.filterwarnings("ignore", category=FilterOD.UnnoisedBodyWarning)``.
        """

    class CovarianceNotPSDWarning(RuntimeWarning):
        """Issued when a covariance mapped back across a sequence is not
        positive semi-definite beyond round-off."""

    # endregion Errors #
    # ------------------#

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

    # ------------------------------------------------------------------
    # Solution-history attributes shared by every filter (LKF / LSB /
    # SRIF / SRIFB). They are populated during :meth:`_compute` and read
    # back by :class:`~scarabaeus.SolutionOD`. Defined once here so the
    # subclasses don't each repeat identical getter/setter plumbing.
    # ------------------------------------------------------------------
    @property
    def covariance_history(self) -> list:
        """Posterior covariance at each measurement epoch. Populated after
        calling :meth:`fit`."""
        return self._covariance_history

    @covariance_history.setter
    def covariance_history(self, value):
        self._covariance_history = value

    @property
    def covariance_consider_history(self) -> list:
        """Consider-augmented covariance at each measurement epoch (when consider
        parameters are active). Populated after calling :meth:`fit`."""
        return self._covariance_consider_history

    @covariance_consider_history.setter
    def covariance_consider_history(self, value):
        self._covariance_consider_history = value

    @property
    def state_deviation_history(self) -> list:
        """State deviation at each measurement epoch. Populated after calling
        :meth:`fit`."""
        return self._state_deviation_history

    @state_deviation_history.setter
    def state_deviation_history(self, value):
        self._state_deviation_history = value

    @property
    def prefit_residuals(self) -> dict:
        """Pre-update measurement residuals, keyed by epoch. Populated after
        calling :meth:`fit`."""
        return self._prefit_residuals

    @prefit_residuals.setter
    def prefit_residuals(self, value):
        self._prefit_residuals = value

    @property
    def postfit_residuals(self) -> dict:
        """Post-update measurement residuals, keyed by epoch. Populated after
        calling :meth:`fit`."""
        return self._postfit_residuals

    @postfit_residuals.setter
    def postfit_residuals(self, value):
        self._postfit_residuals = value

    @property
    def cost_history(self) -> list:
        """Weighted least-squares cost :math:`J = \\sum_i (r_i/\\sigma_i)^2` of the
        post-fit residuals of every iteration performed by :meth:`fit`, as
        ``[(J, n_residuals), ...]``."""
        return self._cost_hist

    # Whether this filter supports sequential process noise (SNC/DMC).
    # Batch estimators (LSB, SRIFB) override this to False.
    _supports_process_noise = True

    # Whether ``_compute`` already maps the last leg's solution back through the
    # sequence nodes, so that every leg of the deviation history holds the
    # solution of the whole data arc. Batch estimators (LSB, SRIFB) override
    # this to True; sequential filters hold per-leg solutions instead.
    _maps_back_sequence = False

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
        verbose: bool = True,
    ):
        # Store inputs
        self.propagator = propagator
        self.measurements = measurements
        self.settings = settings
        self.initial_covariance = settings.initial_covariance

        # Build trajectory and measurements
        self._traj_dir = traj_dir
        self.trajectory, self.measurement_data = self._initialize(
            propagator,
            measurements,
            traj_name,
            traj_dir,
            overwrite_traj,
            verbose=verbose,
        )
        # fit() names later iterations after this one, so re-linearising does
        # not leave a new timestamped SPK behind on every pass
        self._traj_name = os.path.basename(self.trajectory.spk_path)

        self.first_iteration_measurement_data = self.measurement_data

        # Initialize flags
        self.SNC_flag = False
        self.DMC_flag = False
        self.covariance_analysis = False
        self.update_flag = True
        self.flag_desat = False
        # Always defined so the filter loops can test them directly (rather than
        # via hasattr); the sequence/non-sequence initializers refine them below.
        self.flag_node = False
        self.flag_MTM = False
        self.flag_sequence = self.trajectory.sequence_definition is not None

        # Setup based on trajectory type
        if self.flag_sequence:
            self._initialize_sequence_attributes()
        else:
            self._initialize_nonsequence_attributes()

        # Generate consider parameter partition (if applicable)
        self._init_consider_partition()

        # The filters index P0 with the initial state's partition
        n_initial = self.legs_n[0] if self.flag_sequence else self.n
        P0_shape = np.shape(self.initial_covariance.matrix_without_units())
        if P0_shape != (n_initial, n_initial):
            raise self.InitialCovarianceSizeError(
                f"The initial covariance is {P0_shape[0]}x{P0_shape[-1]}, but "
                f"the initial state has {n_initial} elements."
                + (
                    " On a sequence it covers the first leg only: states "
                    "added by later legs take their a-priori sigma from "
                    "extra_apriori_sigmas."
                    if self.flag_sequence
                    else ""
                )
            )

        # Validate and apply settings
        self._apply_settings()

        # Track cumulative correction w.r.t. iteration-1 reference
        # frozen "initial" state definition
        self._ref0_state_def = self.trajectory.state_definition
        # cumulative correction applied to the reference, in ref0 coordinates
        self._delta0_total = np.zeros((self.n, 1), dtype=float)
        # optional history of the total after each pass
        self._delta0_hist = []
        # per-epoch deviation [n_epochs x n_states] applied to the reference so far,
        # and the filter epochs its rows belong to
        self._cumulative_deviation = None
        self._cumulative_epochs = None
        # solution history
        self._solution_history = []
        self._solution_history_smooth = []
        # weighted least-squares cost per iteration, as (J, n_residuals)
        self._cost_hist = []
        # convergence test of every iteration, parallel to _delta0_hist
        self._convergence_hist = []
        # summary of the last convergence test performed by fit()
        self.convergence_info = None

        # Shared output containers (populated during _compute, read by
        # SolutionOD). Subclasses add only their own extra containers.
        self._covariance_history = []
        self._covariance_consider_history = []
        self._state_deviation_history = []
        self._prefit_residuals = {}
        self._postfit_residuals = {}

        # Process-noise model: configured here when provided and supported,
        # otherwise rejected for batch estimators that cannot use it.
        self.process_noise = None
        if self.settings.process_noise is not None:
            if self._supports_process_noise:
                self._initialize_process_noise_from_settings()
            else:
                raise NotImplementedError(
                    f"Process noise is not supported by {type(self).__name__}."
                )

    # endregion Constructor #
    # -----------------------#

    # -----------------------------------
    #             Methods
    # -----------------------------------

    def _initialize_nonsequence_attributes(self) -> None:
        """
        Initialize filter attributes for a single-arc (non-sequence) trajectory.

        Notes
        -----
        Sets the following attributes:

        ``n`` : int
            Dimension of the state vector, from the size of the first stored STM.
        ``state_vector`` : list
            State definition of :attr:`trajectory`.
        ``flag_MTM`` : bool
            ``True`` when the first stored STM is a Maneuver Transition Matrix
            (MTM), i.e. a state longer than six whose velocity rows of the last
            three columns, ``STM[3:6, -3:]``, form the identity: the signature of
            an impulsive delta-v carried in the state vector.
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
        Initialize filter attributes for a multi-leg (sequence) trajectory.

        Single-epoch entries of the sequence definition are events (maneuver
        nodes) and multi-epoch entries are propagation legs; keeping them apart
        lets the filter map the covariance across the trajectory's
        discontinuities.

        Notes
        -----
        Sets the following attributes:

        ``sequence_epochs`` : list of np.ndarray
            Epoch array of every sequence entry, events included.
        ``n`` : int
            Full state dimension, from the propagator. The filters reassign it
            to the current leg's width while they run.
        ``n_global`` : int
            Full state dimension (the union of all legs), kept fixed while
            ``n`` changes.
        ``events_epochs`` : list of float
            Epoch of each event.
        ``legs_n`` : list of int
            State dimension of each propagation leg.
        ``legs_epochs`` : list of np.ndarray
            Epoch arrays of the propagation legs only.
        ``legs_model`` : list of StateArray
            State model of each propagation leg.
        ``new_sigmas`` : list of ArrayWUnits
            The ``initial_covariance.extra_apriori_sigmas`` values, units intact,
            kept for inspection. The a-priori variances actually placed on the
            covariance diagonal are built in :meth:`_map_covariance_definition_node`,
            which converts each sigma to the units of the state element it
            initialises.
        ``flag_node`` : bool
            ``False``; the filter sets it when it reaches an event epoch.
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
        # _compute moves self.n to each leg's width in turn. Not n_full:
        # _init_consider_partition sets that to the current leg's width
        self.n_global = self.n
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
        # Kept as ArrayWUnits: a bare number here invites the same units trap
        # the covariance seeding had, since these sigmas are only meaningful
        # against the units of the state element they initialise.
        self.new_sigmas = list(
            self.initial_covariance.extra_apriori_sigmas.values()
        )
        self.flag_node = False

    def _apply_settings(self) -> None:
        """
        Apply :class:`~scarabaeus.FilterSettings` to the filter after trajectory initialisation.

        Processes optional settings fields in order:

        1. ``initial_deviation`` — if provided, calls :meth:`_set_initial_deviation`.
        2. ``process_noise``     — if provided, calls :meth:`_configure_process_noise`
           which sets ``SNC_flag`` and/or ``DMC_flag`` and stores noise parameters.
        3. ``covariance_analysis`` — copies the boolean flag onto ``self``.

        ``measurement_sigmas`` is not among them: :meth:`_initialize` sets it on
        the models once, at construction, and every later iteration builds its
        datasets from the models (see :meth:`set_measurement_sigmas`).

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

    def _configure_process_noise(
        self, pn: Union[ProcessNoiseSettings, List[ProcessNoiseSettings]]
    ) -> None:
        """
        Configure process noise model, or one per noised body for a list.

        Parameters
        ----------
        pn : ProcessNoiseSettings or list of ProcessNoiseSettings
            Process noise settings, one per noised body for a list.

        Returns
        -------
        None
            Sets ``process_noise_type``, ``process_noise_Q_cont``,
            ``process_noise_beta`` (single values for a single settings
            object, lists otherwise) and ``SNC_flag``/``DMC_flag``.
        """
        settings = pn if isinstance(pn, list) else [pn]
        # Single values for a single model, as before; lists otherwise
        single = len(settings) == 1
        types = [s.type for s in settings]
        self.process_noise_type = types[0] if single else types
        self.process_noise_Q_cont = (
            settings[0].Q_cont if single else [s.Q_cont for s in settings]
        )

        if "DMC" in types:
            self.process_noise_beta = (
                settings[0].beta if single else [s.beta for s in settings]
            )
            self.DMC_flag = True

        if "SNC" in types:
            self.SNC_flag = True

    def set_measurement_sigmas(
        self, sigmas: Dict[str, Union[ArrayWUnits, float]]
    ) -> None:
        """
        Set the measurement sigma of datasets by name, through their models.

        Each sigma is set on the model that built the dataset, so
        ``model.sigma`` reports it and the datasets of every later iteration
        are built with it, and is written to the dataset's current ``sigma``
        column as well. The construction applies ``settings.measurement_sigmas``
        this way; call it between iterations, from a :meth:`fit` callback, to
        reweight the data.

        Parameters
        ----------
        sigmas : dict
            ``{dataset_name: sigma}``. An ``ArrayWUnits`` sigma is converted to
            the dataset's units; a bare float is taken to be in them already.
            One model has one sigma, so the datasets it builds -- the
            components of a two-component measurement, or the entries of a
            spec that reuse it -- share it. A component left out keeps its
            current sigma; any other dataset of that model must be given the
            same sigma.

        Raises
        ------
        UnknownSigmaDatasetError
            If a name is not one of the filter's datasets.
        MeasurementSigmaUnitsError
            If a sigma cannot be converted to its dataset's units.
        SharedSigmaModelError
            If the datasets of one model would be given different sigmas.

        Examples
        --------
        >>> def reweight(filt, iteration, solution):
        ...     if iteration == 1:
        ...         filt.set_measurement_sigmas({"GS1 Range": ArrayWUnits(5.0, m)})
        >>> solution, n_iters, converged = lkf.fit(callback=reweight)
        """
        self._set_model_sigmas(sigmas, self.measurement_data.datasets)
        self.measurement_data._build_indices()

    def _set_model_sigmas(
        self, sigmas: Dict[str, Union[ArrayWUnits, float]], datasets: list
    ) -> None:
        """
        Set measurement sigmas on the models that built ``datasets``, and on the
        datasets' ``sigma`` columns.

        Everything is checked before anything is set, so a refused call leaves
        the models and the datasets as they were.

        Parameters
        ----------
        sigmas : dict
            ``{dataset_name: sigma}``, as for :meth:`set_measurement_sigmas`.
        datasets : list of MeasurementDataSet
            The datasets the models built, matched to them through
            ``_sigma_sources``, which :meth:`_initialize` records.

        Raises
        ------
        UnknownSigmaDatasetError, MeasurementSigmaUnitsError, SharedSigmaModelError
            As for :meth:`set_measurement_sigmas`.
        """
        by_name = {}
        for ds in datasets:
            by_name.setdefault(ds.set_name, []).append(ds)
        unknown = set(sigmas) - set(by_name)
        if unknown:
            raise self.UnknownSigmaDatasetError(
                f"measurement_sigmas names unknown dataset(s) {sorted(unknown)}. "
                f"Available datasets: {sorted(by_name)}."
            )

        values = {}
        for name, sigma in sigmas.items():
            units = by_name[name][0].units
            # The sigma column is in the dataset's units, so a sigma in other
            # units (km for a range dataset in m, say) is converted into them,
            # and one in incompatible units is refused.
            if getattr(sigma, "units", None) is not None and units is not None:
                try:
                    sigma = sigma.convert_to(units)
                except ValueError as err:
                    raise self.MeasurementSigmaUnitsError(
                        f"measurement_sigmas['{name}'] is in {sigma.units}, which "
                        f"cannot be converted to the dataset's units ({units}): {err}"
                    ) from err
            values[name] = float(getattr(sigma, "values", sigma))

        # One sigma per model, one value per dataset it builds in a call. Every
        # entry of a model that any of these sigmas reaches is checked, since
        # setting the model changes the datasets of all of them.
        reached = [m for m, built in self._sigma_sources if set(built) & set(values)]
        new_sigma = {}
        for model, built in self._sigma_sources:
            if not any(model is m for m in reached):
                continue
            # A dataset not named keeps the sigma it was built with
            vector = [
                values[n]
                if n in values
                else float(np.asarray(by_name[n][0].data["sigma"], dtype=float).ravel()[0])
                for n in built
            ]
            units = by_name[built[0]][0].units
            if any(by_name[n][0].units != units for n in built):
                units = None
            if id(model) in new_sigma and new_sigma[id(model)][1] != vector:
                raise self.SharedSigmaModelError(
                    f"'{model.name}' builds the datasets "
                    f"{new_sigma[id(model)][2]} and {built}, which would be given "
                    f"the sigmas {new_sigma[id(model)][1]} and {vector}. One model "
                    "has one sigma: give its datasets the same one, or give "
                    "each its own model."
                )
            new_sigma[id(model)] = (model, vector, built, units)

        for model, vector, _, units in new_sigma.values():
            value = vector[0] if len(vector) == 1 else np.array(vector)
            model.sigma = ArrayWUnits(value, units) if units is not None else value
        for name, value in values.items():
            for ds in by_name[name]:
                ds.data["sigma"] = np.full_like(
                    np.asarray(ds.data["sigma"], dtype=float), value
                )

    def _process_noise_settings(self) -> List[ProcessNoiseSettings]:
        """
        The process noise settings, one per noised body.

        Returns
        -------
        list of ProcessNoiseSettings
            ``settings.process_noise`` as a list, empty without process noise.
        """
        pn = self.settings.process_noise
        if pn is None:
            return []
        return list(pn) if isinstance(pn, list) else [pn]

    def _process_noise_models(self) -> list:
        """
        The process noise models, one per noised body.

        Returns
        -------
        list of ProcessNoise
            :attr:`process_noise` as a list, empty without process noise.
        """
        if self.process_noise is None:
            return []
        if isinstance(self.process_noise, list):
            return self.process_noise
        return [self.process_noise]

    def _process_noise_description(self) -> str:
        """
        Process noise types for the filter printout.

        Returns
        -------
        str
            The model type (e.g. ``'SNC'``) for a single model, otherwise each
            type with the body it drives, e.g. ``'DMC (sc), SNC (ast)'``.
        """
        settings = self._process_noise_settings()
        if len(settings) == 1:
            return settings[0].type
        return ", ".join(
            f"{s.type} ({getattr(model.body, 'name', model.body)})"
            for s, model in zip(settings, self._process_noise_models())
        )

    def _initialize_process_noise_from_settings(self) -> None:
        """
        Initialize the process noise models from settings.

        Creates one process noise object (SNC or DMC) per
        :class:`~scarabaeus.ProcessNoiseSettings`, driving the states of its
        ``body`` (the primary body of the trajectory when not given), and
        computes their discrete-time covariances. A single model is kept as
        is, several as a list, which the filters stack (see
        :meth:`~scarabaeus.ProcessNoise.combined_gamma_at`).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If process noise type is not recognized.
        DuplicateProcessNoiseBodyError
            If two settings drive the same body.
        NoiseReferenceUnavailableError
            If an RTN noise drives a body whose reference trajectory is not
            available (see :meth:`_reference_position_velocity`).
        ProcessNoiseBetaMismatchError
            If a DMC ``beta`` is given and differs from the decay rates of the
            Gauss-Markov acceleration in the dynamics.

        Warns
        -----
        UnnoisedBodyWarning
            If an estimated, propagated body has no process noise.
        """
        all_settings = self._process_noise_settings()

        # Get state and sequence definitions
        state_def = self.trajectory.state_definition
        seq_def = self.trajectory.sequence_definition
        primary = (
            self.trajectory._legs[0]["body"]
            if self.flag_sequence
            else self.trajectory.body
        )

        models = []
        for pn_settings in all_settings:
            body = pn_settings.body if pn_settings.body is not None else primary

            # Create appropriate process noise object
            if pn_settings.type == "SNC":
                model = StateNoiseCompensation(
                    continuous_time_covariance=pn_settings.Q_cont,
                    state_definition=state_def,
                    sequence_definition=seq_def,
                    max_interval=pn_settings.max_interval,
                    body=body,
                )

            elif pn_settings.type == "DMC":
                model = DynamicalModelCompensation(
                    continuous_time_covariance=pn_settings.Q_cont,
                    state_definition=state_def,
                    sequence_definition=seq_def,
                    beta=pn_settings.beta,
                    max_interval=pn_settings.max_interval,
                    body=body,
                )

            else:
                raise ValueError(
                    f"Unknown process noise type: {pn_settings.type}. "
                    "Must be 'SNC' or 'DMC'."
                )

            # The models add up, so a second one on a body would double its noise
            body_id = getattr(model.body, "spice_id", model.body)
            if any(getattr(m.body, "spice_id", m.body) == body_id for m in models):
                raise self.DuplicateProcessNoiseBodyError(
                    "Two ProcessNoiseSettings drive "
                    f"{getattr(model.body, 'name', model.body)}: give each body "
                    "a single one."
                )
            models.append(model)

        self.process_noise = models[0] if len(models) == 1 else models
        self._warn_unnoised_bodies()

        # Compute discrete-time covariances over the filtering timeline
        self._build_timeline()

        # DMC decays at the rates the dynamics integrate the Gauss-Markov
        # acceleration with, so a beta given in the settings is only checked
        for pn_settings, model in zip(all_settings, models):
            if pn_settings.type != "DMC" or pn_settings.beta is None:
                continue
            reference = self._reference_fogm_beta(
                model._noise_epochs(self.timeline_times), model.body
            )
            # A leg without the body has no rate (and no noise)
            reference = reference[np.all(np.isfinite(reference), axis=1)]
            if not np.allclose(reference, model.beta, rtol=1e-6, atol=0.0):
                raise self.ProcessNoiseBetaMismatchError(
                    f"The DMC beta in ProcessNoiseSettings ({model.beta}) "
                    "differs from the decay rates of the Gauss-Markov acceleration "
                    f"in the dynamics ({np.unique(reference, axis=0)}). DMC takes "
                    "them from the dynamics: omit beta from ProcessNoiseSettings."
                )

        # Set flag
        types = {pn_settings.type for pn_settings in all_settings}
        self.SNC_flag = "SNC" in types
        self.DMC_flag = "DMC" in types

    def _warn_unnoised_bodies(self) -> None:
        """
        Warn about every estimated, propagated body the process noise does
        not drive.

        Returns
        -------
        None

        Warns
        -----
        UnnoisedBodyWarning
            Once per body with an estimated, dynamic ``position`` and no
            process noise.
        """
        if self.flag_sequence:
            definitions = [model.state for model in self.legs_model]
        else:
            definitions = [self.trajectory.state_definition]
        noised = {
            getattr(model.body, "spice_id", model.body)
            for model in self._process_noise_models()
        }

        unnoised = {}
        for definition in definitions:
            for name, _, est_flag, dyn_flag, owner, _ in definition:
                owner_id = getattr(owner, "spice_id", owner)
                if (
                    name == "position"
                    and est_flag == "estimated"
                    and dyn_flag == "dynamic"
                    and owner_id not in noised
                ):
                    unnoised.setdefault(owner_id, owner)

        for owner in unnoised.values():
            warnings.warn(
                f"{getattr(owner, 'name', owner)} is estimated and propagated "
                "but has no process noise. Add a ProcessNoiseSettings with its "
                "body, unless it is meant to be noise-free.",
                self.UnnoisedBodyWarning,
                stacklevel=3,
            )

    def _initialize(
        self,
        propagator: Propagator | MissionSequence,
        measurements: MeasurementSpec,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
        verbose: bool = True,
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
            Directory where the trajectory BSP file is written.  Defaults to the
            directory the filter was built with, else ``./data/kernels/scenario/``
            relative to the current working directory.
        overwrite_traj : bool, optional
            If ``True``, remove an existing BSP file with the same name before
            writing the new trajectory.
        verbose : bool, optional
            Passed to the propagation of a mission sequence or scenario, whose
            segment-by-segment printout it controls.  A plain propagator has no
            such printout and ignores it.  Defaults to ``True``.

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
        traj_dir = traj_dir or self._traj_dir or (os.getcwd() + "/data/kernels/scenario/")
        os.makedirs(traj_dir, exist_ok=True)

        # Propagate trajectory
        if isinstance(propagator, MissionSequence):
            scenario = propagator.propagate(verbose=verbose)
            self.propagator = scenario  # swap to ScenarioSetup for downstream use
            propagator = scenario
        elif isinstance(propagator, ScenarioSetup):
            propagator.propagate(verbose=verbose)
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
            trajectory.write_to_spk(traj_path, **self._trajectory_write_formats())

        else:
            trajectory = Trajectory(
                traj_name, state_array=propagator.propagated_state_array
            )
            trajectory.write_to_spk(traj_path, **self._trajectory_write_formats())
        if hasattr(propagator, "STM") and propagator.STM is not None:
            # A propagated state array carries one STM per output epoch
            trajectory.add_STMs(
                propagator.STM,
                None if isinstance(propagator, ScenarioSetup) else trajectory.epoch,
            )
        else:
            raise ValueError("Propagator must have STM for OD.")

        # Build measurement datasets
        if body_obj is None:
            raise ValueError(
                "Cannot infer target spacecraft from propagator.primary_body."
            )

        datasets = []
        sigma_sources = []
        for i, meas in enumerate(measurements):
            if "model" not in meas:
                raise ValueError(
                    f"Measurement descriptor {i} missing required keys 'model'."
                )

            model = meas["model"]
            observed_meas = meas.get("observed_meas", None)
            dataset_name = meas.get("dataset_name", None)
            epochs = meas.get("epochs", None)

            # Datasets are always regenerated here, even when the spec already
            # carries one from MeasurementSpec.compute(). Those are computed
            # against whatever reference the caller chose for inspecting
            # residuals, typically the true trajectory, whereas the filter needs
            # them against its own reference, which is perturbed and changes on
            # every iterate() pass. Reusing them would quietly feed the filter
            # residuals taken about the wrong trajectory.
            #
            # Outlier flags do carry over, because they live on the observations
            # (ObsData.meas_outliers) and are reseeded into each regenerated
            # dataset.
            model.update_reference_state(self.propagator.full_state_vector)
            # The analytical light-time partials map a deviation between the
            # endpoints of an observable through the reference STMs. Models
            # that do not read it ignore it.
            model.set_trajectory(trajectory)
            ds = model.generate_measurement_dataset(
                dataset_name,
                target=body_obj,
                observed_meas=observed_meas,
                epochs=epochs,
            )
            datasets.append(ds)
            # A two-component model (ra/dec, sample/line) builds one dataset per
            # component, all of them carrying its one sigma
            built = ds if isinstance(ds, (list, tuple)) else [ds]
            sigma_sources.append((model, [d.set_name for d in built]))
        self._sigma_sources = sigma_sources

        # First call: settings.measurement_sigmas go on the models, once, and
        # before the editing below, which therefore weights the residuals by
        # them, as every later iteration does
        first_call = not hasattr(self, "_edited_outlier_flags")
        if first_call and self.settings.measurement_sigmas is not None:
            flat = [
                d for ds in datasets
                for d in (ds if isinstance(ds, (list, tuple)) else [ds])
            ]
            self._set_model_sigmas(self.settings.measurement_sigmas, flat)

        # Build FilterDataManager
        fdm = FilterDataManager(
            datasets,
            editing_method=self.settings.editing_method,
            editing_datasets=self.settings.editing_datasets,
            editing_kwargs=self.settings.editing_kwargs,
            save_path=self.settings.output.fdm_output_path,
            save_name=self.settings.output.fdm_output_name,
            save_format=getattr(self.settings.output, "fdm_output_format", "json"),
        )

        # First call: initialize from whatever editing produced
        if first_call:
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
        convergence_threshold_cost: float = 1e-3,
        convergence_criterion: str = "any",
        verbose: bool = True,
        if_sequential_smooth: bool = False,
        callback: Optional[Callable] = None,
    ):
        """
        Perform iterative orbit determination with automatic convergence checking.

        Runs the filter iteratively, applying state corrections from each solution
        to the next iteration until convergence or maximum iterations is reached.

        Two convergence tests are evaluated at every iteration:

        * **State deviation** — RMS of the estimated deviation mapped back to the
          epoch, compared against ``convergence_threshold``.  This is the strict
          test: it declares convergence only when the reference trajectory stops
          moving.
        * **Cost function** — relative change of the normalised weighted
          least-squares cost :math:`J/n = \\sum_i (r_i/\\sigma_i)^2 / n` between
          consecutive iterations, compared against ``convergence_threshold_cost``.
          When the problem is weakly observable the deviation keeps wandering
          along the unobservable directions while the measurement fit no longer
          improves; the run has converged as far as the data can tell, and only
          this test detects it.

        ``convergence_criterion`` selects how the two are combined.  The cost is
        normalised by the number of residuals so that measurement editing (which
        changes that number between iterations) is not mistaken for a cost change.

        Parameters
        ----------
        traj_name : str, optional
            Base name for trajectory files. Iteration number will be appended.
            If None, the name the filter was built with is the base.
            Defaults to None.
        traj_dir : str, optional
            Directory for trajectory files. If None, the directory the filter
            was built with. Defaults to None.
        overwrite_traj : bool, optional
            Whether to overwrite existing trajectories. Defaults to True.
        max_iterations : int, optional
            Maximum number of iterations. Defaults to 10.
        convergence_threshold : float, optional
            RMS deviation threshold for convergence. Defaults to 1e-6.
        convergence_threshold_cost : float, optional
            Threshold on the *relative* change of the normalised weighted cost
            :math:`|\\Delta (J/n)| / (J/n)` between consecutive iterations.
            Defaults to 1e-3.
        convergence_criterion : str, optional
            How the two tests are combined:

            * ``'any'`` — converged as soon as either test passes (default).
            * ``'deviation'`` — RMS deviation only (pre-cost-test behaviour).
            * ``'cost'`` — cost-function stagnation only; the criterion of
              choice for weakly observable problems.
            * ``'both'`` — both tests must pass.
        verbose : bool, optional
            If True, prints iteration progress. Defaults to True.
        if_sequential_smooth : bool, optional
            If True, applies sequential smoothing after filtering. Defaults to False.
        callback : callable, optional
            Called as ``callback(filter, iteration, solution)`` after every
            iteration, once its convergence test is done (its outcome is the
            last entry of ``_convergence_hist``) and before the filter is
            reinitialized for the next one. Its return value is ignored. It is
            the place to reweight or edit the data between iterations: a sigma
            set on a model (:meth:`set_measurement_sigmas`, or
            ``model.sigma = ...``) or an outlier flagged through
            ``MeasurementSpec.flag`` takes effect when the datasets are rebuilt
            for the next iteration, so one made after the last iteration does
            nothing within this call. Other settings are not guaranteed to be
            picked up mid-run. Defaults to None.

        Returns
        -------
        solution : SolutionOD
            Final filter solution from the last iteration.
        iteration_count : int
            Number of iterations performed.
        converged : bool
            Whether convergence was achieved.

        Raises
        ------
        ConvergenceCriterionError
            If ``convergence_criterion`` is not one of the accepted values.
        NotImplementedError
            If ``if_sequential_smooth=True`` on a batch filter (LSB, SRIFB).

        Notes
        -----
        The cost is always evaluated on the *filter* post-fit residuals, also
        when ``if_sequential_smooth=True``, so the iterations are compared on a
        like-for-like basis.  Its per-iteration history is available as
        :attr:`cost_history`, and the outcome of the last convergence test as
        ``convergence_info``. Every iteration's test is kept in
        ``_convergence_hist``, which :meth:`SolutionOD.save` writes to each
        ``iter_<k>`` block.

        Reweighting the data from ``callback`` changes the scale of the cost, so
        the cost test of the next iteration compares two differently weighted
        costs; prefer ``convergence_criterion='deviation'`` across a change of
        sigmas.

        Examples
        --------
        >>> # Automatic iteration with convergence checking
        >>> lkf = LKF(propagator, covariance, measurements)
        >>> solution, n_iters, converged = lkf.fit(
        ...     max_iterations=5,
        ...     convergence_threshold=1e-6
        ... )
        >>> print(f"Converged in {n_iters} iterations: {converged}")

        >>> # Weakly observable problem: stop when the fit stops improving
        >>> solution, n_iters, converged = lkf.fit(
        ...     max_iterations=10,
        ...     convergence_threshold_cost=1e-3,
        ...     convergence_criterion="cost",
        ... )

        >>> # Loosen the range sigma after the first iteration
        >>> def reweight(filt, iteration, solution):
        ...     if iteration == 1:
        ...         filt.set_measurement_sigmas({"GS1 Range": ArrayWUnits(5.0, m)})
        >>> solution, n_iters, converged = lkf.fit(callback=reweight)
        """

        criterion = str(convergence_criterion).lower()
        if criterion not in ("any", "deviation", "cost", "both"):
            raise self.ConvergenceCriterionError(
                f"Unknown convergence_criterion '{convergence_criterion}'. "
                "Choose 'any', 'deviation', 'cost', or 'both'."
            )

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
                print(f"Convergence criterion: {criterion}")
                print(f"  RMS deviation threshold:  {convergence_threshold:.2e}")
                print(f"  Relative cost threshold:  {convergence_threshold_cost:.2e}")
            else:
                print("Mode: covariance analysis (prediction-only)")
            print("=" * 80 + "\n")

        if traj_name is None:
            traj_name = self._traj_name

        converged = False
        solution = None
        rms_deviation = 0.0
        cost = float("nan")
        cost_normalized = float("nan")
        cost_normalized_prev = float("nan")
        cost_change = float("inf")
        n_residuals = 0
        converged_deviation = False
        converged_cost = False

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration}")
                print(f"{'='*60}")

            # Compute filter solution
            solution = self._compute(printOutput=verbose)
            self._solution_history.append(solution)

            # Cost of this pass, from the filter post-fit residuals (i.e. before
            # any smoothing, so that all iterations are comparable)
            if not self.covariance_analysis:
                cost, n_residuals = self._cost_function()
                self._cost_hist.append((cost, n_residuals))

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
                if callback is not None:
                    callback(self, iteration, solution)
                break

            # Get state deviation from solution
            if self.flag_sequence:
                # NOTE: A state shared by several legs is taken from the first leg
                # carrying it. That is consistent only if every leg holds the same
                # solution. Batch filters map the last leg's solution back through
                # the nodes themselves; for sequential filters, whose legs hold the
                # solution of the data up to that leg only, the final estimate is
                # mapped back here. Per-leg solutions would mix e.g. the initial
                # state of the first arc with burn parameters fitted together with
                # a different, discarded state.
                deviation = solution.map_state_deviation_to_epoch(
                    map_back_sequence=not self._maps_back_sequence
                )
                deviation = self.propagator._deviation_history_to_global_dx(deviation)
            else:
                deviation = solution.map_state_deviation_to_epoch()

            # Total deviation w.r.t. first reference after this pass. Filters return
            # either (n,) or (n, 1); a flat one would broadcast the total to (n, n).
            deviation = np.asarray(deviation, dtype=float).reshape(-1, 1)
            self._delta0_hist.append(self._delta0_total + deviation)

            # Check convergence
            rms_deviation = np.sqrt(np.mean(deviation**2))

            # Relative change of the normalized cost. Normalizing by the number
            # of residuals keeps the test meaningful when measurement editing
            # changes how many measurements are processed between iterations.
            cost_normalized = cost / n_residuals if n_residuals > 0 else float("nan")
            cost_change = float("inf")
            if np.isfinite(cost_normalized) and np.isfinite(cost_normalized_prev):
                cost_change = abs(cost_normalized - cost_normalized_prev) / max(
                    abs(cost_normalized_prev), 1e-300
                )

            converged_deviation = bool(rms_deviation < convergence_threshold)
            converged_cost = bool(cost_change < convergence_threshold_cost)

            if criterion == "deviation":
                converged = converged_deviation
            elif criterion == "cost":
                converged = converged_cost
            elif criterion == "both":
                converged = converged_deviation and converged_cost
            else:  # "any"
                converged = converged_deviation or converged_cost

            self._convergence_hist.append(
                {
                    "iteration": iteration,
                    "rms_deviation": float(rms_deviation),
                    "cost": float(cost),
                    "n_residuals": int(n_residuals),
                    "cost_normalized": float(cost_normalized),
                    # no previous cost on the first pass: null, not Infinity
                    "cost_relative_change": (
                        float(cost_change) if np.isfinite(cost_change) else None
                    ),
                    "converged_deviation": converged_deviation,
                    "converged_cost": converged_cost,
                    "converged": bool(converged),
                }
            )

            if verbose:
                print(f"RMS State Deviation: {rms_deviation:.6e}")
                if n_residuals > 0:
                    print(
                        f"Cost function J = Σ(r/σ)²: {cost:.6e} "
                        f"(n = {n_residuals}, J/n = {cost_normalized:.6e})"
                    )
                    if np.isfinite(cost_change):
                        print(f"Relative cost change: {cost_change:.6e}")
                else:
                    print(
                        "Cost function: unavailable (no weighted residuals) "
                        "-> cost criterion inactive"
                    )

            if callback is not None:
                callback(self, iteration, solution)

            if converged:
                if verbose:
                    triggers = []
                    if converged_deviation:
                        triggers.append("RMS deviation")
                    if converged_cost:
                        triggers.append("cost function")
                    print(f"\n{'='*60}")
                    print(f"✓ CONVERGED after {iteration} iteration(s)!")
                    print(f"  Triggered by: {' and '.join(triggers)}")
                    print(f"{'='*60}\n")
                break

            cost_normalized_prev = cost_normalized

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
                    verbose=verbose,
                )

                # Only a correction applied to the reference is accumulated: the
                # a priori of later passes, and the cumulative deviation their
                # solutions report on top of their own, are relative to it. The
                # last pass's correction is never applied, so a later fit() call
                # resumes from the same reference without counting it twice.
                self._delta0_total = self._delta0_total + deviation
                if solution.deviation_est is not None:
                    self._cumulative_deviation = solution.deviation_cumulative.copy()
                    self._cumulative_epochs = np.asarray(
                        getattr(self, "timeline_times", []), dtype=float
                    )

        if not self.covariance_analysis:
            self.convergence_info = {
                "criterion": criterion,
                "iterations": iteration,
                "converged": converged,
                "rms_deviation": rms_deviation,
                "rms_deviation_threshold": convergence_threshold,
                "converged_deviation": converged_deviation,
                "cost": cost,
                "cost_normalized": cost_normalized,
                "cost_relative_change": cost_change,
                "cost_relative_change_threshold": convergence_threshold_cost,
                "converged_cost": converged_cost,
                "n_residuals": n_residuals,
            }

        if (not self.covariance_analysis) and (not converged) and verbose:
            print(f"\n{'='*60}")
            print(f"!! Did NOT converge after {max_iterations} iterations")
            print(f"  Final RMS deviation:        {rms_deviation:.6e}")
            print(f"  Final relative cost change: {cost_change:.6e}")
            print(f"{'='*60}\n")

        # Automatically save final solution if requested through OutputSettings
        if solution is not None and getattr(self.settings, "output", None) is not None:
            out = self.settings.output
            if getattr(out, "solution_output_path", None) is not None:
                solution.save()

        return solution, iteration, converged

    @staticmethod
    def _accumulate_deviation(
        cumulative: Optional[np.ndarray],
        cumulative_epochs: Optional[np.ndarray],
        deviation: np.ndarray,
        epochs: np.ndarray,
        max_epoch_shift: float = 1.0,
    ) -> np.ndarray:
        """
        Add one iteration's per-epoch deviation to the running total, row by epoch.

        The filter epochs are not fixed across iterations: they are spacecraft
        times ``t2``, which move with the reference trajectory, and the set of
        epochs can change (data edited in one iteration but not the next,
        measurements that share an epoch in one iteration and not the next). Rows
        are therefore paired by epoch rather than by position. A row of
        ``deviation`` is paired with the row of ``cumulative`` whose epoch is its
        nearest neighbour, when that pairing is mutual and the epochs are within
        ``max_epoch_shift`` seconds. A row with no partner has no cumulative
        value: it is set to NaN and a warning names how many rows that affects.

        Parameters
        ----------
        cumulative : np.ndarray or None
            Running total from the previous iterations [n_prev x n_states], or
            None on the first iteration.
        cumulative_epochs : np.ndarray or None
            Epochs of the rows of ``cumulative``.
        deviation : np.ndarray
            This iteration's deviation [n x n_states].
        epochs : np.ndarray
            Epochs of the rows of ``deviation``.
        max_epoch_shift : float, optional
            Largest epoch difference [s] accepted when pairing rows.

        Returns
        -------
        np.ndarray
            Running total over this iteration's epochs [n x n_states].
        """
        deviation = np.asarray(deviation, dtype=float)
        if cumulative is None:
            return deviation.copy()
        cumulative = np.asarray(cumulative, dtype=float)

        n_new = deviation.shape[0]
        have_epochs = (
            cumulative_epochs is not None
            and len(cumulative_epochs) == cumulative.shape[0]
            and len(epochs) == n_new
            and n_new > 0
            and cumulative.shape[0] > 0
        )
        if not have_epochs or cumulative.shape[1:] != deviation.shape[1:]:
            if cumulative.shape == deviation.shape:
                return cumulative + deviation
            warnings.warn(
                f"Cumulative deviation dropped: the previous iteration's "
                f"{cumulative.shape} cannot be matched to this iteration's "
                f"{deviation.shape}. deviation_cumulative is NaN.",
                RuntimeWarning,
                stacklevel=3,
            )
            return np.full_like(deviation, np.nan)

        prev = np.asarray(cumulative_epochs, dtype=float)
        new = np.asarray(epochs, dtype=float)

        def nearest(src, ref):
            """Index into sorted ``ref`` of the nearest neighbour of each ``src``."""
            if len(ref) == 1:
                return np.zeros(len(src), dtype=int)
            j = np.clip(np.searchsorted(ref, src), 1, len(ref) - 1)
            left = j - 1
            return np.where(np.abs(src - ref[left]) <= np.abs(src - ref[j]), left, j)

        j = nearest(new, prev)  # new row -> previous row
        back = nearest(prev, new)  # previous row -> new row
        paired = (back[j] == np.arange(n_new)) & (np.abs(new - prev[j]) <= max_epoch_shift)

        out = np.full_like(deviation, np.nan)
        out[paired] = cumulative[j[paired]] + deviation[paired]
        n_unpaired = int(n_new - paired.sum())
        if n_unpaired:
            warnings.warn(
                f"{n_unpaired} of {n_new} filter epochs have no counterpart in the "
                f"previous iteration; their deviation_cumulative rows are NaN.",
                RuntimeWarning,
                stacklevel=3,
            )
        return out

    def _reinitialize(
        self,
        state_deviation: Optional[np.ndarray] = None,
        state_vector: Optional[StateArray] = None,
        traj_name: Optional[str] = None,
        traj_dir: Optional[str] = None,
        overwrite_traj: bool = True,
        verbose: bool = True,
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
            Directory to write the new trajectory. If None, the directory the
            filter was built with. Defaults to None.
        overwrite_traj : bool, optional
            If True, overwrites existing trajectory file with the same name.
            Defaults to True.
        verbose : bool, optional
            If True, the re-propagation of a mission sequence prints its
            segment-by-segment progress. Defaults to True.

        Returns
        -------
        None
            Updates self.trajectory and self.measurement_data in place.

        Raises
        ------
        ValueError
            If state_deviation is provided but state_vector cannot be determined.

        See Also
        --------
        fit : Automatic iterative estimation with convergence checking.
        _apply_state_deviation : Applies deviation to state vector components.
        _initialize : Builds trajectory and measurement datasets.

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
        Work between iterations goes in a :meth:`fit` callback, which runs
        before this method rebuilds the data, rather than in calls to this
        method and ``_compute``:

        >>> lkf = LKF(propagator, covariance, measurements)
        >>> def between_iterations(filt, iteration, solution):
        ...     if iteration == 1:
        ...         filt.set_measurement_sigmas({"GS1 Range": ArrayWUnits(5.0, m)})
        >>> solution, n_iters, converged = lkf.fit(
        ...     max_iterations=3, callback=between_iterations
        ... )
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
            verbose=verbose,
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

    def _trajectory_write_formats(self) -> dict:
        """
        On-disk formats for the sidecars written beside a trajectory SPK.

        The STM history is the largest artifact an OD run produces — one matrix
        per output epoch, growing with the square of the state size — so a
        finely sampled arc or an extra solve-for parameter can push it into
        hundreds of megabytes of JSON. ``OutputSettings.trajectory_output_format``
        selects the compact binary form instead; note that
        ``fdm_output_format`` is a different setting covering the measurement
        records, not these.

        Returns
        -------
        dict
            Keyword arguments for :meth:`~scarabaeus.Trajectory.write_to_spk`.
        """
        output = getattr(self, "settings", None) and self.settings.output
        fmt = getattr(output, "trajectory_output_format", "json")
        return {
            "stm_format": fmt,
            "sidecar_format": fmt,
            # The filter serves its own STM lookups from the in-memory history,
            # so the sidecar it writes here is for whoever reads the trajectory
            # back later. A run that never does can skip the largest artifact
            # an OD produces.
            "write_stms": bool(getattr(output, "save_trajectory_stms", True)),
        }

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
        self, postfit_residuals: list, dataset_names: list[str], epoch_attr: str = None
    ) -> dict:
        """
        Organizes postfit residuals into a dictionary categorized by dataset.

        Parameters
        ----------
        postfit_residuals : list
            List of residuals and associated metadata.
        dataset_names : list
            List of dataset names corresponding to different measurements.
        epoch_attr : str, optional
            When given, the matching reception epoch (``t2``, TDB seconds) of every
            residual entry is collected into a parallel ``{dataset: [t2, ...]}``
            dict and stored on ``self`` under this attribute name (e.g.
            ``"prefit_epochs"``). The residual entries themselves are left as
            ``[residual, sigma]`` so existing consumers are unaffected.

        Returns
        -------
        dict
            Dictionary mapping dataset names to corresponding residuals and sigmas.
        """
        # Initialize dictionary with empty lists for each dataset
        unwrapped = {name: [] for name in dataset_names}
        epochs = {name: [] for name in dataset_names}

        # Loop through epochs
        for epoch_residuals in postfit_residuals:
            residuals, sigmas, _, _, sets, t2_all = epoch_residuals

            # ``t2_all`` is per-measurement, aligned with ``sets``/``sigmas``.
            t2_arr = np.atleast_1d(np.asarray(t2_all, dtype=float)).ravel()

            # Assign residuals to their corresponding dataset
            for i, (res, sigma, dataset) in enumerate(
                zip(residuals.flatten(), sigmas, sets)
            ):
                if dataset in unwrapped:
                    unwrapped[dataset].append([res, sigma])
                    epochs[dataset].append(
                        float(t2_arr[i]) if i < t2_arr.size else float(t2_arr[-1])
                    )

        if epoch_attr is not None:
            setattr(self, epoch_attr, epochs)

        return unwrapped

    def _cost_function(self, residuals: Optional[dict] = None) -> Tuple[float, int]:
        """
        Weighted least-squares cost of the measurement residuals.

        Evaluates the measurement part of the cost minimised by every filter of
        this framework (see the class docstring),

        .. math::

            J = \\sum_i \\left(\\frac{r_i}{\\sigma_i}\\right)^2 ,

        summed over every residual of the current pass.  The a-priori term
        :math:`\\Delta x_0^T P_0^{-1} \\Delta x_0` is deliberately excluded: it is
        the measurement fit that keeps changing (or stops changing) between
        iterations, and the prior term is not comparable across the legs of a
        mission sequence, where the state definition changes.

        Residuals with a non-positive or non-finite sigma are skipped, so that
        edited/unusable measurements do not contaminate the cost.

        Parameters
        ----------
        residuals : dict, optional
            ``{dataset: [[residual, sigma], ...]}`` mapping, as produced by
            :meth:`_unwrap_residuals`. Defaults to :attr:`postfit_residuals`.

        Returns
        -------
        cost : float
            Weighted sum of squares. ``nan`` when no usable residual is found
            (e.g. covariance analysis, or a custom filter that records none).
        n_residuals : int
            Number of residuals contributing to ``cost``.
        """
        if residuals is None:
            residuals = self.postfit_residuals

        if not isinstance(residuals, dict) or len(residuals) == 0:
            return float("nan"), 0

        cost = 0.0
        n_residuals = 0
        for entries in residuals.values():
            if len(entries) == 0:
                continue
            arr = np.atleast_2d(np.asarray(entries, dtype=float))
            res, sigma = arr[:, 0], arr[:, 1]
            valid = np.isfinite(res) & np.isfinite(sigma) & (sigma > 0.0)
            cost += float(np.sum((res[valid] / sigma[valid]) ** 2))
            n_residuals += int(np.count_nonzero(valid))

        if n_residuals == 0:
            return float("nan"), 0

        return cost, n_residuals

    def _map_covariance_definition_node(
        self, counter_events: int, old_covariance: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjust the covariance to a new leg's state definition via the reorder
        operator ``Q`` (``P' = Q P Q^T``). Newly augmented states receive their
        a-priori variance on the diagonal (uncorrelated, ``P_yz = 0``).

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event.
        old_covariance : numpy.ndarray
            Covariance in the previous state definition.
        flag_forward : bool, optional
            ``True`` maps previous -> new (augmentation); ``False`` reduces.
            Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            Covariance in the new state definition.
        """
        old_model, new_model = self._node_state_models(counter_events, flag_forward)
        Q, _new_n, new_blocks = Utils.build_reorder_operator(
            old_model.state_definition, new_model.state_definition
        )

        block_units = self._state_block_units(new_model)
        new_block_variances = []
        for name, start, size in new_blocks:
            sig = self._apriori_sigma_values(name, size, block_units.get(start))
            new_block_variances.append((start, sig ** 2))

        return Utils.reorder_covariance(Q, old_covariance, new_block_variances)

    def _apriori_sigma_values(self, name: str, size: int, target_units):
        """
        Return the a-priori sigma of a newly augmented state block, expressed in
        the units of the state element it initialises.

        The values supplied through ``extra_apriori_sigmas`` carry their own
        units, which need not match the ones the state element was declared
        with (e.g. a pointing bias declared in radians given a sigma in
        degrees). The covariance is built in the state's own units, so the
        supplied sigma is converted before it is squared onto the diagonal.

        Parameters
        ----------
        name : str
            Name of the newly augmented state element.
        size : int
            Number of scalars in the block.
        target_units : Units or None
            Units of the state element. ``None`` when the element's units could
            not be resolved, in which case the supplied values are used as-is.

        Returns
        -------
        numpy.ndarray
            Sigma values in ``target_units``, truncated to ``size`` entries.

        Raises
        ------
        MissingAprioriSigmaError
            If no ``extra_apriori_sigmas`` entry exists for ``name``.
        AprioriSigmaUnitsError
            If the supplied sigma cannot be converted to ``target_units``.
        """
        sigmas = self.initial_covariance.extra_apriori_sigmas
        if name not in sigmas:
            raise self.MissingAprioriSigmaError(
                f"No sigma provided for new state element '{name}'."
            )

        sigma = sigmas[name]
        if isinstance(target_units, Units):
            try:
                sigma = sigma.convert_to(target_units)
            except (ValueError, TypeError) as exc:
                raise self.AprioriSigmaUnitsError(
                    f"A-priori sigma for '{name}' is given in "
                    f"'{sigma.units}', which cannot be converted to the units "
                    f"the state element is defined in ('{target_units}')."
                ) from exc

        return np.asarray(sigma.values).ravel()[:size]

    @staticmethod
    def _state_block_units(state_model) -> dict:
        """
        Map each state block's scalar offset to the units of its values.

        The offsets match the ones produced by
        :meth:`~scarabaeus.Utils.build_reorder_operator`, so a block reported as
        newly augmented can be paired with the units it is defined in.
        ``state_definition`` drops the values (and with them the units), which
        is why the full ``state`` entries are walked here.

        Parameters
        ----------
        state_model : StateArray
            Leg state model to inspect.

        Returns
        -------
        dict
            ``{start_offset: Units or None}``; ``None`` for blocks whose units
            are not a single homogeneous ``Units`` object.
        """
        units_by_start = {}
        cursor = 0
        for entry in state_model.state:
            size = int(entry[1])
            quantity = getattr(entry[5], "quantity", None) if len(entry) > 5 else None
            units = getattr(quantity, "units", None)
            units_by_start[cursor] = units if isinstance(units, Units) else None
            cursor += size
        return units_by_start

    def _map_deviation_definition_node(
        self, counter_events: int, old_deviation: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjust the state deviation to a new leg's definition via ``X' = Q X``.
        Newly augmented states start at their a-priori mean, see
        :meth:`_apriori_deviation`.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event.
        old_deviation : numpy.ndarray
            Deviation in the previous state definition.
        flag_forward : bool, optional
            ``True`` maps previous -> new (augmentation); ``False`` reduces.
            Reducing while new states appear is unexpected and raises.

        Returns
        -------
        numpy.ndarray
            Deviation in the new state definition (1-D).
        """
        old_list, new_list = self._node_state_lists(counter_events, flag_forward)
        Q, _new_n, new_blocks = Utils.build_reorder_operator(old_list, new_list)
        if new_blocks and not flag_forward:
            raise ValueError(
                "Deviation mapping has exhibited unexpected behavior, with "
                "deviation elements missing during reduction."
            )
        deviation = Utils.reorder_vector(Q, old_deviation).ravel()
        if new_blocks:
            x_bar = self._apriori_deviation(counter_events).ravel()
            for _name, start, size in new_blocks:
                deviation[start : start + size] = x_bar[start : start + size]
        return deviation

    def _map_information_definition_node(
        self, counter_events: int, R: np.ndarray, z: np.ndarray
    ):
        """
        Carry a square-root information array across a sequence node.

        ``[R | z]`` describes the solution at the node epoch in the definition of
        leg ``counter_events - 1`` (``R`` need not be triangular). It is moved to
        the definition of leg ``counter_events`` without passing through the
        covariance, which after a long, dense arc is too ill-conditioned to be
        inverted back into information reliably:

        - states the new leg drops are marginalised: with their columns first, a
          QR factorisation isolates their rows, and the remaining rows hold the
          information on the kept states alone;
        - the kept columns move to their positions in the new definition;
        - newly augmented states contribute a-priori rows
          ``diag(1/sigma) dx = diag(1/sigma) x_bar`` (uncorrelated, as in
          :meth:`_map_covariance_definition_node`, and centred on
          :meth:`_apriori_deviation`);
        - a last QR factorisation restores the upper-triangular form.

        Parameters
        ----------
        counter_events : int
            Index of the sequence event (node) being crossed forward.
        R : numpy.ndarray
            ``(n_prev, n_prev)`` square-root information matrix at the node.
        z : numpy.ndarray
            Information vector (``n_prev`` elements).

        Returns
        -------
        R_new : numpy.ndarray
            ``(n_new, n_new)`` upper-triangular square-root information matrix.
        z_new : numpy.ndarray
            ``(n_new, 1)`` information vector.
        """
        old_model, new_model = self._node_state_models(counter_events, True)
        Q, new_n, new_blocks = Utils.build_reorder_operator(
            old_model.state_definition, new_model.state_definition
        )
        R = np.asarray(R, dtype=float)
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        kept = np.flatnonzero(Q.any(axis=0))
        dropped = np.flatnonzero(~Q.any(axis=0))

        # Marginalise the dropped states
        if dropped.size:
            n_d = dropped.size
            _, T = np.linalg.qr(np.hstack((R[:, dropped], R[:, kept], z)))
            R_kept = T[n_d:, n_d:-1]
            z_kept = T[n_d:, -1:]
        else:
            R_kept, z_kept = R, z

        # Kept information in the new definition, then the a-priori rows
        R_rows = [np.zeros((R_kept.shape[0], new_n))]
        R_rows[0][:, Q[:, kept].argmax(axis=0)] = R_kept
        z_rows = [z_kept]
        if new_blocks:
            block_units = self._state_block_units(new_model)
            x_bar = self._apriori_deviation(counter_events).ravel()
            for name, start, size in new_blocks:
                sig = self._apriori_sigma_values(name, size, block_units.get(start))
                prior = np.zeros((size, new_n))
                prior[np.arange(size), start + np.arange(size)] = 1.0 / sig
                R_rows.append(prior)
                z_rows.append((x_bar[start : start + size] / sig).reshape(-1, 1))

        _, T = np.linalg.qr(np.hstack((np.vstack(R_rows), np.vstack(z_rows))))
        m = min(T.shape[0], new_n)
        R_new = np.zeros((new_n, new_n))
        z_new = np.zeros((new_n, 1))
        R_new[:m] = T[:m, :new_n]
        z_new[:m] = T[:m, new_n:]
        return R_new, z_new

    def _map_back_across_node(
        self,
        counter_events: int,
        smoothed_deviation,
        smoothed_covariance: np.ndarray,
        node_deviation,
        node_covariance: np.ndarray,
        node_information: Optional[np.ndarray] = None,
    ):
        """
        Carry a solution backward across a sequence node.

        Batch filters use it on the solution of the last leg; for sequential
        filters :meth:`SolutionOD.map_state_deviation_to_epoch` uses it on the
        final estimate, with the forward solution stored at each node.

        The solution of the leg after the node is reduced to the previous leg's
        definition at the node epoch. States augmented at the node are simply
        discarded. States the previous leg *drops* at the node (e.g. the
        parameters of a burn arc followed by a coast) have no counterpart after
        it, so they are recovered by conditioning the previous leg's forward
        solution at the node on the smoothed values of the kept states
        (a Rauch-Tung-Striebel step without process noise).

        The conditioning is done in square-root information form. With the
        dropped columns first, a QR factorisation of the forward square-root
        information ``R`` at the node isolates the rows
        ``[R_dd  R_dk]`` that carry the dropped states, and

        .. math::

            G = -R_{dd}^{-1} R_{dk}, \\qquad
            \\delta x_d^s = \\delta x_d + G (\\delta x_k^s - \\delta x_k),

            P_{dk}^s = G P_{kk}^s, \\qquad
            P_{dd}^s = R_{dd}^{-1} R_{dd}^{-T} + G P_{kk}^s G^T.

        This is the covariance-form step
        :math:`G = P_{dk} P_{kk}^{-1}`,
        :math:`P_{dd}^s = P_{dd} + G (P_{kk}^s - P_{kk}) G^T` written without
        inverting :math:`P_{kk}` or differencing two nearly equal covariances,
        which after a dense arc loses positive definiteness and lets the
        recovered sigmas exceed their a priori. Here :math:`P^s` is
        :math:`[I; G] P_{kk}^s [I; G]^T` plus the conditional covariance of the
        dropped states, so it is positive semi-definite by construction.

        It is exact: the data after the node depend on the dropped states only
        through the kept ones. Considered states carry no deviation, so they
        stay at zero. Never updated, a dropped considered state also keeps its
        forward covariance; its cross terms :math:`G P_{kk}^s` hold as they
        are, being :math:`P_{kk}^s P_{kk}^{-1} P_{kc}` either way.

        Parameters
        ----------
        counter_events : int
            Index of the sequence event (node) being crossed backward.
        smoothed_deviation : numpy.ndarray or None
            Deviation at the node in the definition of leg ``counter_events``.
            ``None`` maps the covariance only.
        smoothed_covariance : numpy.ndarray or None
            Covariance at the node in the definition of leg ``counter_events``.
            ``None`` maps the deviation only.
        node_deviation : numpy.ndarray or None
            Forward deviation of leg ``counter_events - 1`` propagated to the
            node, in that leg's definition. Unused when ``smoothed_deviation``
            is ``None``.
        node_covariance : numpy.ndarray
            Forward covariance of leg ``counter_events - 1`` propagated to the
            node, in that leg's definition. Read when ``node_information`` is
            ``None``, to build the square-root information from it, and for the
            covariance of the dropped considered states.
        node_information : numpy.ndarray, optional
            Forward square-root information of leg ``counter_events - 1`` at the
            node, in that leg's definition (``R`` with ``R^T R = P^{-1}``; it
            need not be triangular). Supplied by SRIFB, which carries it across
            the node anyway and so avoids re-factoring ``node_covariance``.

        Returns
        -------
        deviation : numpy.ndarray or None
            Column deviation at the node in the definition of leg
            ``counter_events - 1``; ``None`` when ``smoothed_deviation`` is.
        covariance : numpy.ndarray or None
            Covariance at the node in the same definition; ``None`` when
            ``smoothed_covariance`` is.
        """
        prev_list, next_list = self._node_state_lists(counter_events, True)
        Q, _new_n, _new_blocks = Utils.build_reorder_operator(prev_list, next_list)
        kept = np.flatnonzero(Q.any(axis=0))
        dropped = np.flatnonzero(~Q.any(axis=0))

        covariance = None
        if smoothed_covariance is not None:
            covariance = Q.T @ np.asarray(smoothed_covariance, dtype=float) @ Q
        deviation = None
        if smoothed_deviation is not None:
            deviation = Utils.reorder_vector(Q.T, smoothed_deviation)

        if dropped.size == 0:
            return deviation, covariance

        # Rows of the forward information that carry the dropped states
        if node_information is None:
            R_f = self._sqrt_info_matrix(node_covariance)
        else:
            R_f = np.asarray(node_information, dtype=float)
        n_d = dropped.size
        _, T = np.linalg.qr(np.hstack((R_f[:, dropped], R_f[:, kept])))
        R_dd = T[:n_d, :n_d]
        G = -spln.solve_triangular(R_dd, T[:n_d, n_d:], lower=False)
        _idx_est, idx_con, _n = self._partition_indices(
            self._leg_state_definition(counter_events - 1)
        )

        if covariance is not None:
            S_dd = spln.solve_triangular(R_dd, np.eye(n_d), lower=False)
            P_kk_s = covariance[np.ix_(kept, kept)]
            covariance[np.ix_(dropped, kept)] = G @ P_kk_s
            covariance[np.ix_(kept, dropped)] = (G @ P_kk_s).T
            covariance[np.ix_(dropped, dropped)] = S_dd @ S_dd.T + G @ P_kk_s @ G.T
            # considered states are never updated: hold their forward covariance
            con = np.intersect1d(dropped, np.asarray(idx_con, dtype=int))
            if con.size:
                covariance[np.ix_(con, con)] = np.asarray(
                    node_covariance, dtype=float
                )[np.ix_(con, con)]
            covariance = 0.5 * (covariance + covariance.T)

        if deviation is not None:
            dx_f = np.asarray(node_deviation, dtype=float).reshape(-1, 1)
            deviation[dropped] = dx_f[dropped] + G @ (deviation[kept] - dx_f[kept])
            deviation[idx_con] = 0.0

        return deviation, covariance

    def _map_node_to_leg_epoch(
        self, STM: np.ndarray, deviation, covariance, label: str = "covariance"
    ):
        """
        Carry a node solution back to its leg's reference epoch.

        Applies :math:`\\Phi^{-1}` to the deviation and
        :math:`\\Phi^{-1} P \\Phi^{-T}` to the covariance through linear solves
        rather than an explicit inverse, symmetrises the result and warns with
        :class:`CovarianceNotPSDWarning` if it is not positive semi-definite.

        Parameters
        ----------
        STM : numpy.ndarray
            ``(n, n)`` STM of the leg from its reference epoch to the node.
        deviation : numpy.ndarray or None
            Deviation at the node; ``None`` maps the covariance only.
        covariance : numpy.ndarray or None
            Covariance at the node; ``None`` maps the deviation only.
        label : str, optional
            Name of the covariance used in the warning. Defaults to
            ``"covariance"``.

        Returns
        -------
        deviation, covariance : numpy.ndarray or None
            The deviation and covariance at the leg's reference epoch.
        """
        STM = np.asarray(STM, dtype=float)
        try:
            if deviation is not None:
                deviation = np.linalg.solve(STM, deviation)
            if covariance is not None:
                covariance = np.linalg.solve(STM, np.linalg.solve(STM, covariance).T)
        except np.linalg.LinAlgError:
            STM_inv = self._safe_inv(STM)
            if deviation is not None:
                deviation = STM_inv @ deviation
            if covariance is not None:
                covariance = STM_inv @ covariance @ STM_inv.T

        if covariance is not None:
            covariance = 0.5 * (covariance + covariance.T)
            self._check_covariance_psd(covariance, label)
        return deviation, covariance

    def _split_consider_covariance(self, idx_leg: int, covariance: np.ndarray):
        """
        Split a leg's joint error covariance into the pair the filters store.

        On a leg with considered states, ``covariance_history`` holds the
        "no-consider" covariance, whose estimated block is conditional on the
        considered states, and ``covariance_consider_history`` the joint one.
        A solution mapped back from a later leg arrives joint; the conditional
        block is

        .. math::

            P_{xx} = P_c - P_{xc} P_{cc}^{-1} P_{xc}^T,

        which on the leg it was computed in is the filter's own :math:`P_{xx}`,
        since there :math:`P_{xc} = S_{xc} P_{cc}`.

        Parameters
        ----------
        idx_leg : int
            Leg whose definition ``covariance`` is in.
        covariance : numpy.ndarray
            Joint error covariance in that definition.

        Returns
        -------
        covariance, covariance_consider : numpy.ndarray, numpy.ndarray or None
            The conditional and the joint covariance, or ``covariance`` and
            ``None`` when the leg has no considered states.
        """
        idx_est, idx_con, _n = self._partition_indices(
            self._leg_state_definition(idx_leg)
        )
        if idx_con.size == 0:
            return covariance, None
        P_xc = covariance[np.ix_(idx_est, idx_con)]
        P_xx = covariance[np.ix_(idx_est, idx_est)] - P_xc @ np.linalg.solve(
            covariance[np.ix_(idx_con, idx_con)], P_xc.T
        )
        conditional = covariance.copy()
        conditional[np.ix_(idx_est, idx_est)] = 0.5 * (P_xx + P_xx.T)
        return conditional, covariance

    def _check_covariance_psd(self, P: np.ndarray, label: str, tol: float = 1e-9):
        """
        Warn if ``P`` is not positive semi-definite beyond round-off.

        The test runs on the correlation matrix, so the units spread across the
        state do not set the threshold.

        Parameters
        ----------
        P : numpy.ndarray
            Symmetric covariance matrix.
        label : str
            Name of the covariance used in the warning.
        tol : float, optional
            Most negative correlation-matrix eigenvalue accepted, as a
            magnitude. Default is ``1e-9``.

        Warns
        -----
        CovarianceNotPSDWarning
            If the smallest correlation-matrix eigenvalue is below ``-tol`` or
            a variance is negative.
        """
        d = np.sqrt(np.abs(np.diag(P)))
        d[d == 0.0] = 1.0
        min_eig = np.linalg.eigh(P / np.outer(d, d))[0].min()
        negative = np.flatnonzero(np.diag(P) < 0.0)
        if min_eig < -tol or negative.size:
            warnings.warn(
                f"{label} is not positive semi-definite (smallest correlation "
                f"eigenvalue {min_eig:.3e}, negative variances at {negative.tolist()}).",
                self.CovarianceNotPSDWarning,
            )

    def _map_stm_definition_node(
        self, counter_events: int, old_STM: np.ndarray, flag_forward: bool = True
    ) -> np.ndarray:
        """
        Adjust the STM to a new leg's definition via ``Phi' = Q Phi Q^T``.

        STM mapping does not support newly augmented states (they have no source
        partials); encountering one raises.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event.
        old_STM : numpy.ndarray
            STM in the previous state definition.
        flag_forward : bool, optional
            ``True`` maps previous -> new; ``False`` reduces. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            STM in the new state definition.
        """
        old_list, new_list = self._node_state_lists(counter_events, flag_forward)
        Q, _new_n, new_blocks = Utils.build_reorder_operator(old_list, new_list)
        if new_blocks:
            raise ValueError(
                "STM mapping is not supported for new state elements. "
                "Missing STM elements."
            )
        return Utils.reorder_stm(Q, old_STM, new_blocks)

    def _node_state_lists(self, counter_events: int, flag_forward: bool):
        """
        Return the ``(old_list, new_list)`` state definitions for a node mapping.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event (node).
        flag_forward : bool
            If ``True`` the new leg is ``counter_events`` and the old leg is the
            previous one (augmentation); the roles reverse when reducing.

        Returns
        -------
        old_list, new_list : list
            The previous and target leg state definitions.
        """
        old_model, new_model = self._node_state_models(counter_events, flag_forward)
        return old_model.state_definition, new_model.state_definition

    def _node_state_models(self, counter_events: int, flag_forward: bool):
        """
        Return the ``(old_model, new_model)`` leg state models for a node mapping.

        Same selection as :meth:`_node_state_lists`, but keeping the
        :class:`~scarabaeus.StateArray` objects so callers can also read the
        units of each state element, which ``state_definition`` drops.

        Parameters
        ----------
        counter_events : int
            Index of the current sequence event (node).
        flag_forward : bool
            If ``True`` the new model is ``counter_events`` and the old one is
            the previous leg (augmentation); the roles reverse when reducing.

        Returns
        -------
        old_model, new_model : StateArray
            The previous and target leg state models.
        """
        if flag_forward:
            return (
                self.legs_model[counter_events - 1],
                self.legs_model[counter_events],
            )
        return (
            self.legs_model[counter_events],
            self.legs_model[counter_events - 1],
        )

    def _validate_sequence_batch(self):
        """
        Reports state elements dropped between consecutive sequence legs.

        Dropping is supported by the batch filters (the element is marginalised
        at the node and recovered when mapping back), so each leg that drops
        elements issues one :class:`SequenceStateDroppedWarning` naming them.

        Returns
        -------
        None

        Warns
        -----
        SequenceStateDroppedWarning
            Once per leg that drops elements of the previous leg.
        """
        # Initialize list of state vectors
        list_state_vectors = [
            obj.state for obj in self.legs_model if isinstance(obj, StateArray)
        ]

        # Take consecutive pairs of state_vectors
        for i in range(1, len(list_state_vectors)):
            prev_ids = [prev_tuple[0] for prev_tuple in list_state_vectors[i - 1]]
            curr_ids = [curr_tuple[0] for curr_tuple in list_state_vectors[i]]

            dropped = [prev_id for prev_id in prev_ids if prev_id not in curr_ids]
            if dropped:
                warnings.warn(
                    f"State vector leg {i} does not contain element(s) {dropped} "
                    f"of leg {i - 1}; they are recovered when mapping back.",
                    self.SequenceStateDroppedWarning,
                    stacklevel=2,
                )

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
        Split the state into estimated and considered parameters.

        Entries flagged ``"considered"`` go to the consider partition and every
        other entry to the estimated one; see :meth:`_partition_indices`.

        Parameters
        ----------
        idx_leg : int, optional
            Leg whose state definition is used in sequence mode. Defaults to ``0``.

        Notes
        -----
        Sets the following attributes:

        ``idx_est`` : np.ndarray
            Indices of the estimated state elements.
        ``idx_con`` : np.ndarray
            Indices of the considered state elements.
        ``n_full`` : int
            Dimension of the full state vector.
        ``n_est`` : int
            Number of estimated state elements.
        ``n_con`` : int
            Number of considered state elements.
        ``has_consider`` : bool
            ``True`` when ``n_con > 0``.
        """
        if self.flag_sequence:
            state_def = self.legs_model[idx_leg].state
        else:
            state_def = (
                self.trajectory.state_definition
            )  # list of tuples (name, dim, est, dyn, owner, value)

        idx_est, idx_con, cursor = self._partition_indices(state_def)

        self.idx_est = idx_est
        self.idx_con = idx_con

        self.n_full = cursor
        self.n_est = int(self.idx_est.size)
        self.n_con = int(self.idx_con.size)

        self.has_consider = self.n_con > 0

    @staticmethod
    def _partition_indices(state_def):
        """
        Split a state definition into estimated and considered index arrays.

        Pure counterpart of :meth:`_init_consider_partition`, so a partition
        can be computed for any leg without disturbing the filter's current
        one.

        Parameters
        ----------
        state_def : list of tuple
            ``(name, dim, est_flag, dyn_flag, owner, value)`` entries.

        Returns
        -------
        tuple
            ``(idx_est, idx_con, n_full)``.
        """
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

        return (
            np.asarray(idx_est, dtype=int),
            np.asarray(idx_con, dtype=int),
            cursor,
        )

    def _leg_state_definition(self, idx_leg: int):
        """State definition of leg ``idx_leg``, or the trajectory's when flat."""
        if self.flag_sequence:
            return self.legs_model[idx_leg].state
        return self.trajectory.state_definition

    def _map_deviation_at_node(
        self, counter_events: int, dx_est_prev: np.ndarray
    ) -> np.ndarray:
        """
        Carry the running deviation across a sequence node.

        The operator built by :meth:`_map_deviation_definition_node` is sized
        from the legs' **full** state definitions, but the filter's running
        deviation is the **estimated partition only** — it multiplies ``Hx``,
        which has ``n_est`` columns. Handing that partition straight to the
        mapping mismatches whenever a leg carries consider parameters. It only
        appears to work while the previous leg's full size happens to equal its
        estimated size, which holds for a plain position/velocity leg and stops
        holding the moment a leg adds burn parameters.

        This embeds the partition back into the previous leg's full definition,
        maps it, and returns the new leg's estimated partition. Consider
        parameters carry no tracked deviation — only covariance — so they enter
        the full vector as zeros.

        Parameters
        ----------
        counter_events : int
            Index of the sequence event (node) being crossed.
        dx_est_prev : numpy.ndarray
            Deviation in the previous leg's estimated partition.

        Returns
        -------
        numpy.ndarray
            Column vector of the deviation in the new leg's estimated
            partition.
        """
        idx_est_new, _, _ = self._partition_indices(
            self._leg_state_definition(counter_events)
        )
        dx_prev_full = self._full_leg_deviation(counter_events - 1, dx_est_prev)

        dx_full = np.asarray(
            self._map_deviation_definition_node(counter_events, dx_prev_full)
        ).reshape(-1, 1)

        return dx_full[idx_est_new, 0].reshape(-1, 1)

    def _full_leg_deviation(self, idx_leg: int, dx_est: np.ndarray) -> np.ndarray:
        """
        Embed a deviation of a leg's estimated partition into its full definition.

        Consider parameters carry no tracked deviation, so they enter as zeros.
        A deviation that is already full-sized (no consider parameters on the
        leg) passes through untouched.

        Parameters
        ----------
        idx_leg : int
            Index of the sequence leg the deviation belongs to.
        dx_est : numpy.ndarray
            Deviation in the leg's estimated partition (or full definition).

        Returns
        -------
        numpy.ndarray
            Column vector of the deviation in the leg's full definition.
        """
        idx_est, _, n_full = self._partition_indices(
            self._leg_state_definition(idx_leg)
        )
        dx = np.asarray(dx_est, dtype=float).reshape(-1, 1)

        if dx.size == n_full:
            return dx
        if dx.size == idx_est.size:
            dx_full = np.zeros((n_full, 1))
            dx_full[idx_est, 0] = dx.ravel()
            return dx_full
        raise ValueError(
            f"Deviation of size {dx.size} matches neither the full "
            f"({n_full}) nor the estimated ({idx_est.size}) dimension "
            f"of leg {idx_leg}."
        )

    def _apriori_deviation(self, idx_leg: int = 0) -> np.ndarray:
        """
        A-priori state deviation of a leg with respect to the current reference.

        The a-priori state is the reference the first iteration started from.
        Once :meth:`fit` has moved the reference by ``_delta0_total``, the
        a-priori state sits at ``-_delta0_total`` from it, and that is the
        mean the prior information must carry. Starting every iteration from
        a zero a-priori deviation instead re-centres the prior on the latest
        estimate: the prior then constrains nothing, and directions the data
        observe weakly (e.g. burn parameters of an arc with no tracking)
        drift further at every iteration.

        Parameters
        ----------
        idx_leg : int, optional
            Index of the sequence leg. Ignored for single-arc trajectories.
            Defaults to 0.

        Returns
        -------
        numpy.ndarray
            Column vector in the leg's full state definition. Considered
            states are never corrected, so their entries are zero.
        """
        delta = getattr(self, "_delta0_total", None)
        if not self.flag_sequence:
            if delta is None:
                return np.zeros((self.n, 1))
            return -np.asarray(delta, dtype=float).reshape(-1, 1)

        x_bar = np.zeros((self.legs_n[idx_leg], 1))
        if delta is None or not np.any(delta):
            return x_bar

        sequence = self.trajectory._sequence_definition
        leg_names = [
            name
            for name, seg_type in zip(sequence.names, sequence.types)
            if seg_type in ("Leg", "Finite Burn")
        ]
        delta = np.asarray(delta, dtype=float).ravel()
        for ls, le, gs, ge in self.propagator._leg_to_global_maps[leg_names[idx_leg]]:
            x_bar[ls:le, 0] = -delta[gs:ge]
        return x_bar

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

    def _build_timeline(self):
        """
        Build the filtering timeline shared by every filter's ``_compute`` loop.

        Returns
        -------
        (measurement_times, event_times, timeline_times) : tuple of np.ndarray
            ``measurement_times`` are the spacecraft measurement epochs;
            ``event_times`` are the sequence event epochs (empty for single-arc
            trajectories); ``timeline_times`` is their sorted union — the array
            the per-epoch loop iterates over.
        """
        measurement_times = np.asarray(
            self.measurement_data.get_spacecraft_times(), dtype=float
        )
        if self.flag_sequence:
            event_times = np.asarray(self.events_epochs, dtype=float)
            timeline_times = np.unique(
                np.concatenate((measurement_times, event_times))
            )
        else:
            event_times = np.array([])
            timeline_times = measurement_times
        # Cache the timeline so downstream consumers (e.g. SolutionOD) can attach
        # an epoch to every state-deviation / covariance row. These histories are
        # built over ``timeline_times`` (measurements ∪ events), which is longer
        # than the measurement-only ``timestamps`` whenever a sequence has event
        # epochs that do not coincide with a measurement epoch.
        self.timeline_times = timeline_times
        # The filters index the process noise by position in this timeline, so
        # it is discretized over the same epochs: an event epoch splits the
        # interval it falls in. Redone whenever the timeline changes (e.g. the
        # measurements are rebuilt between iterations).
        # An RTN noise is redone every time: it is rotated along the reference
        # trajectory, which is rebuilt between iterations. So is a DMC noise,
        # which decays at the reference Gauss-Markov rates (estimated or not)
        # and, for an RTN or body-fixed Gauss-Markov acceleration, is rotated
        # into its frame along the reference trajectory.
        # Each model reads the reference of the body it drives.
        for model in self._process_noise_models():
            reference_state = functools.partial(
                self._reference_position_velocity, body=model.body
            )
            if isinstance(model, DynamicalModelCompensation):
                model._compute_discrete_time_covariances(
                    timeline_times,
                    reference_state,
                    functools.partial(self._reference_fogm_beta, owner=model.body),
                    functools.partial(self._reference_fogm_frame, owner=model.body),
                )
            elif model.frame == "RTN" or not np.array_equal(
                model.epochs, timeline_times
            ):
                model._compute_discrete_time_covariances(
                    timeline_times, reference_state
                )
        return measurement_times, event_times, timeline_times

    def _reference_leg_of(self, times) -> np.ndarray:
        """
        Index into ``trajectory._legs`` of the leg each time falls in: the
        first leg ending at or after it, the last one past the end. Zero on a
        single arc.

        Parameters
        ----------
        times : array_like
            TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(len(times),)``.
        """
        times = np.asarray(times, dtype=float)
        if not self.flag_sequence:
            return np.zeros(times.shape, dtype=int)
        leg_ends = np.array([np.max(leg["epoch_leg"].et) for leg in self.trajectory._legs])
        return np.minimum(np.searchsorted(leg_ends, times), len(leg_ends) - 1)

    def _reference_position_velocity(self, times, body=None):
        """
        Position and velocity of the reference trajectory at ``times``.

        Read from the trajectory SPK, which is loaded once written. On a
        sequence each time is taken relative to the origin, and in the frame,
        of the leg it falls in. The legs meet at their nodes, where SPICE
        returns only one of the two states, so the times should lie inside a
        leg.

        Parameters
        ----------
        times : array_like
            TDB seconds past J2000.
        body : Body, optional
            Body whose reference to read. A single arc writes every
            propagated body to the SPK, a sequence only the primary body of
            each leg. Defaults to ``None``, the primary body.

        Returns
        -------
        (positions, velocities) : tuple of np.ndarray
            Arrays of shape ``(len(times), 3)``, in km and km/s.

        Raises
        ------
        NoiseReferenceUnavailableError
            If, on a sequence, ``body`` is not the primary body of every leg.
        """
        times = np.asarray(times, dtype=float)
        if self.flag_sequence:
            legs = [
                (leg["body"], leg["origin"], leg["frame"])
                for leg in self.trajectory._legs
            ]
            if body is not None:
                body_id = getattr(body, "spice_id", body)
                other = next(
                    (leg_body for leg_body, _, _ in legs if leg_body.spice_id != body_id),
                    None,
                )
                if other is not None:
                    raise self.NoiseReferenceUnavailableError(
                        f"The reference trajectory of {getattr(body, 'name', body)} "
                        "is not available: a sequence writes only the primary "
                        f"body of each leg ({other.name}) to the trajectory SPK. "
                        "Give its process noise in an inertial frame."
                    )
        else:
            legs = [
                (
                    self.trajectory.body if body is None else body,
                    self.trajectory.origin,
                    self.trajectory.ref_frame,
                )
            ]

        states = np.array([
            SpiceManager.get_state(body, t, frame, origin).values
            for t, (body, origin, frame) in zip(
                times, (legs[i] for i in self._reference_leg_of(times))
            )
        ]).reshape(-1, 6)
        return states[:, :3], states[:, 3:]

    def _reference_fogm_models(self, owner=None) -> list:
        """
        The first-order Gauss-Markov model each leg integrates, with the leg's
        reference initial state and the body that owns its ``a_fogm``.

        Parameters
        ----------
        owner : Body, optional
            Body whose Gauss-Markov acceleration to read. Defaults to
            ``None``, the owner of the first ``a_fogm`` of each leg.

        Returns
        -------
        list of tuple
            One ``(state, leg_owner, gauss_markov)`` per leg, in leg order;
            ``leg_owner`` and ``gauss_markov`` are ``None`` on a leg whose
            state carries no ``a_fogm`` of ``owner``.

        Raises
        ------
        MissingGaussMarkovModelError
            If a leg's dynamics do not integrate the Gauss-Markov acceleration
            of the body that owns ``a_fogm``.
        """
        if self.flag_sequence:
            sequence = self.trajectory._sequence_definition
            props = [sequence.props[leg["seq_idx"]] for leg in self.trajectory._legs]
        else:
            props = [self.propagator]

        owner_id = getattr(owner, "spice_id", owner)
        legs = []
        for prop in props:
            state = prop.full_state_vector
            leg_owner = next(
                (
                    entry[4]
                    for entry in state.state
                    if entry[0] == "a_fogm"
                    and (
                        owner is None
                        or getattr(entry[4], "spice_id", None) == owner_id
                    )
                ),
                None,
            )
            if leg_owner is None:
                # Nothing to decay on this leg, nor noise to drive
                legs.append((state, None, None))
                continue
            gauss_markov = next(
                (
                    force_model.force_models["FirstOrderGaussMarkov"]
                    for force_model in prop.force_models
                    if "FirstOrderGaussMarkov"
                    in getattr(force_model, "dynamics_mode_list", ())
                    and getattr(force_model.primary_body, "spice_id", None)
                    == leg_owner.spice_id
                ),
                None,
            )
            if gauss_markov is None:
                raise self.MissingGaussMarkovModelError(
                    f"DMC drives the a_fogm of {leg_owner.name}, but no force model on "
                    "its leg integrates it: enable first_order_gauss_markov in the "
                    "ForceModelTranslation of that body."
                )
            legs.append((state, leg_owner, gauss_markov))
        return legs

    def _reference_fogm_beta(self, times, owner=None):
        """
        Decay rates of the first-order Gauss-Markov acceleration at ``times``,
        as the dynamics integrate it on the leg each time falls in.

        They are the ``beta_fogm`` state value when the state carries one
        (which the filter may estimate), otherwise the force model's own
        (``fogm_beta``), from the leg's reference initial state.

        Parameters
        ----------
        times : array_like
            TDB seconds past J2000.
        owner : Body, optional
            Body whose Gauss-Markov acceleration to read. Defaults to
            ``None``, the owner of the first ``a_fogm`` of each leg.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(times), 3)``, in 1/s. NaN on a leg whose
            state carries no ``a_fogm`` of ``owner``.

        Raises
        ------
        MissingGaussMarkovModelError
            If a leg's dynamics do not integrate the Gauss-Markov acceleration
            of the body that owns ``a_fogm``.
        """
        betas = []
        for state, leg_owner, gauss_markov in self._reference_fogm_models(owner):
            if gauss_markov is None:
                betas.append(np.full(3, np.nan))
                continue
            beta = gauss_markov._get_beta_fogm(
                state._make_direct_representation(), leg_owner
            )
            betas.append(
                np.broadcast_to(np.asarray(beta, dtype=float).reshape(-1), (3,))
            )
        return np.array(betas)[self._reference_leg_of(times)]

    def _reference_fogm_frame(self, times, owner=None):
        """
        Rotation from the frame the first-order Gauss-Markov acceleration is
        expressed in to the propagation frame, at ``times``, along the
        reference trajectory.

        The identity for an inertial Gauss-Markov acceleration and on a leg
        that carries none. A body-fixed rotation depends on the epoch alone,
        so the reference trajectory is read only when some leg's acceleration
        is RTN.

        Parameters
        ----------
        times : array_like
            TDB seconds past J2000.
        owner : Body, optional
            Body whose Gauss-Markov acceleration to read. Defaults to
            ``None``, the owner of the first ``a_fogm`` of each leg.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(times), 3, 3)``.

        Raises
        ------
        MissingGaussMarkovModelError
            If a leg's dynamics do not integrate the Gauss-Markov acceleration
            of the body that owns ``a_fogm``.
        """
        times = np.asarray(times, dtype=float)
        rotations = np.tile(np.eye(3), (len(times), 1, 1))
        legs = self._reference_fogm_models(owner)
        frames = {gm.frame for _, _, gm in legs if gm is not None}
        if frames <= {"inertial"}:
            return rotations

        if "RTN" in frames:
            positions, velocities = self._reference_position_velocity(times, owner)
        else:
            positions = velocities = np.full((len(times), 3), np.nan)
        for i, leg in enumerate(self._reference_leg_of(times)):
            _, leg_owner, gauss_markov = legs[leg]
            if gauss_markov is None or gauss_markov.frame == "inertial":
                continue
            rotations[i] = DynamicalUtils.frame_matrix(
                gauss_markov,
                positions[i],
                {("velocity", leg_owner.spice_id): velocities[i]},
                times[i],
                leg_owner,
            )
        return rotations

    def _epoch_measurement_block(self, epoch):
        """
        Return the combined measurement block for a single ``t2`` epoch.

        Thin wrapper over :meth:`FilterDataManager.get_combined_for_t2` that
        drops the redundant ``{epoch: block}`` indirection repeated at every
        call site.

        Parameters
        ----------
        epoch : float
            Reception (``t2``) epoch, TDB seconds past J2000.

        Returns
        -------
        dict
            The measurement block for ``epoch`` (keys such as ``sigma``,
            ``partials``, ``residuals``, ``instruments``, ``t2``).
        """
        return self.measurement_data.get_combined_for_t2(epoch)[epoch]

    def _measurement_covariance(
        self, sigmas, epoch, meas_corr=None, underweighting_factor=1.0
    ):
        """
        Build the measurement-noise covariance ``R`` for one epoch.

        Equivalent to ``underweighting_factor * CovarianceMatrix(sigmas, epoch,
        from_list=True, corr_factors=meas_corr).matrix`` as a numpy array.

        Parameters
        ----------
        sigmas : array_like
            Per-measurement standard deviations for this epoch.
        epoch : float
            Measurement epoch, TDB seconds past J2000.
        meas_corr : list, optional
            Off-diagonal correlation factors forwarded to
            :class:`~scarabaeus.CovarianceMatrix`. Defaults to ``None``.
        underweighting_factor : float, optional
            Multiplicative scale applied to ``R`` (``>1`` de-weights
            measurements). Batch filters use the default ``1.0``.

        Returns
        -------
        numpy.ndarray
            The measurement-noise covariance matrix ``R``.
        """
        return underweighting_factor * np.array(
            CovarianceMatrix(
                sigmas, epoch, from_list=True, corr_factors=meas_corr
            ).matrix
        )

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

        The factorisation is carried out on the correlation matrix, so the
        spread of units across the state (e.g. km next to km/s next to a
        unitless scale factor) does not reach the numerics, and without forming
        the inverse explicitly. If Cholesky factorization still fails (loss of
        positive definiteness), the function falls back to an eigenvalue-based
        construction.

        Parameters
        ----------
        P : ndarray of shape (n, n)
            Covariance matrix.

        min_eig : float, optional
            Smallest eigenvalue of the correlation matrix kept by the eigenvalue
            fallback, relative to its largest one. Default is ``1e-14``.

        Returns
        -------
        ndarray of shape (n, n)
            Upper-triangular square-root information matrix.

        Warns
        -----
        RuntimeWarning
            Raised when Cholesky decomposition fails and the routine falls
            back to the eigenvalue-based formulation.

        Notes
        -----
        Primary method
        ^^^^^^^^^^^^^^

        With :math:`P = D C D`, :math:`D = \\mathrm{diag}(\\sqrt{P_{ii}})`, and
        :math:`J` the reversal permutation, the lower Cholesky factor of the
        reversed correlation matrix :math:`J C J = L L^T` gives

        .. math::

            R = J L^{-1} J D^{-1},

        which is upper triangular with positive diagonal, i.e. exactly
        :math:`\\mathrm{chol}(P^{-1})`.

        Fallback method
        ^^^^^^^^^^^^^^^

        If Cholesky decomposition fails, :math:`C = V \\Lambda V^T` is used with
        the eigenvalues clipped from below, and the symmetric square root
        :math:`V \\Lambda^{-1/2} V^T` is brought back to upper-triangular form
        by a QR factorisation (which leaves :math:`R^T R` unchanged), so callers
        that back-substitute through :math:`R` still see a triangular matrix.
        """
        P = np.asarray(P, dtype=float)
        P = 0.5 * (P + P.T)
        n = P.shape[0]

        # Unit scaling: factor the correlation matrix, undo the scaling after
        scale = np.sqrt(np.abs(np.diag(P)))
        scale[scale == 0.0] = 1.0
        C = P / np.outer(scale, scale)

        try:
            L = spln.cholesky(C[::-1, ::-1], lower=True)
            R_C = spln.solve_triangular(L, np.eye(n), lower=True)[::-1, ::-1]
            return R_C / scale

        except np.linalg.LinAlgError:
            warnings.warn(
                "Cholesky factorization failed. Falling back to "
                "eigenvalue-based square-root information matrix.",
                RuntimeWarning,
            )

        # Eigenvalue fallback, clipping relative to the largest eigenvalue
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, min_eig * np.max(eigvals))

        # Symmetric square root of inv(C), re-triangularised
        _, R_C = np.linalg.qr(eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)
        R_C = R_C * np.where(np.diag(R_C) < 0.0, -1.0, 1.0)[:, None]

        return R_C / scale
