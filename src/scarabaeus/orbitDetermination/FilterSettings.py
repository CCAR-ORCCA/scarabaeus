# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional, Dict
from scarabaeus import CovarianceMatrix, ProcessNoiseSettings, OutputSettings
import numpy as np

# -----------------------#
#    Class Definition   #
# -----------------------#


class FilterSettings:
    """ Comprehensive filter configuration for orbit determination.

    Bundles the initial covariance, optional process noise model, measurement
    noise overrides, output configuration, and measurement editing settings
    into a single object passed to any :class:`~scarabaeus.FilterOD` subclass.

    Parameters
    ----------
    initial_covariance : CovarianceMatrix
        Initial covariance matrix :math:`P_0`.
    initial_deviation : np.ndarray, optional
        Initial state deviation :math:`x_0`. ``None`` means zero deviation.
    process_noise : ProcessNoiseSettings, optional
        Process noise model configuration. ``None`` disables process noise.
    measurement_sigmas : dict, optional
        Sigma overrides for measurement noise, keyed by dataset name.
    covariance_analysis : bool
        If ``True``, run in covariance-analysis mode (no measurement updates).
        Defaults to ``False``.
    output : OutputSettings, optional
        Output configuration. Defaults to all-default :class:`~scarabaeus.OutputSettings`.
    editing_method : str, optional
        Measurement editing mode: ``'lasso'``, ``'chi2'``, ``'date_range'``,
        or ``'interactive_date_range'``. ``None`` disables editing.
    editing_datasets : str or list, optional
        Which datasets to edit: ``None`` = all, a string = one by name,
        a list of strings = a specific subset.
    editing_kwargs : dict, optional
        Extra keyword arguments forwarded to the editing method.

    Notes
    -----
    Pass a single :class:`~scarabaeus.FilterSettings` instance to any
    :class:`~scarabaeus.FilterOD` subclass constructor.  When ``output`` is
    omitted, an all-default :class:`~scarabaeus.OutputSettings` is created
    automatically.

    See Also
    --------
    scarabaeus.FilterOD : Base filter class that consumes this settings object.
    scarabaeus.ProcessNoiseSettings : Process noise model configuration.
    scarabaeus.OutputSettings : Controls what the solution stores and writes.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initial_covariance(self) -> "CovarianceMatrix":
        """Initial covariance matrix :math:`P_0`."""
        return self._initial_covariance

    @initial_covariance.setter
    def initial_covariance(self, value):
        self._initial_covariance = value

    @property
    def initial_deviation(self) -> Optional[np.ndarray]:
        """Initial state deviation :math:`x_0`. ``None`` means zero deviation."""
        return self._initial_deviation

    @initial_deviation.setter
    def initial_deviation(self, value):
        self._initial_deviation = value

    @property
    def process_noise(self) -> Optional["ProcessNoiseSettings"]:
        """Process noise model configuration (:class:`~scarabaeus.ProcessNoiseSettings`)."""
        return self._process_noise

    @process_noise.setter
    def process_noise(self, value):
        self._process_noise = value

    @property
    def measurement_sigmas(self) -> Optional[Dict[str, float]]:
        """Measurement noise sigma overrides by dataset name."""
        return self._measurement_sigmas

    @measurement_sigmas.setter
    def measurement_sigmas(self, value):
        self._measurement_sigmas = value

    @property
    def covariance_analysis(self) -> bool:
        """If ``True``, run in covariance-analysis mode (no measurement updates)."""
        return self._covariance_analysis

    @covariance_analysis.setter
    def covariance_analysis(self, value: bool):
        self._covariance_analysis = value

    @property
    def output(self) -> "OutputSettings":
        """Output configuration (:class:`~scarabaeus.OutputSettings`). Always non-``None``
        after construction — defaults to an all-default :class:`~scarabaeus.OutputSettings`."""
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def editing_method(self) -> Optional[str]:
        """Measurement editing mode. One of ``'lasso'``, ``'chi2'``,
        ``'date_range'``, ``'interactive_date_range'``, or ``None``."""
        return self._editing_method

    @editing_method.setter
    def editing_method(self, value):
        self._editing_method = value

    @property
    def editing_datasets(self) -> Optional[str | list]:
        """Which datasets to edit: ``None`` = all, a single name string, or a
        list of dataset name strings."""
        return self._editing_datasets

    @editing_datasets.setter
    def editing_datasets(self, value):
        self._editing_datasets = value

    @property
    def editing_kwargs(self) -> Optional[Dict]:
        """Extra keyword arguments forwarded to the editing method.
        E.g. ``{"sigma_scale": 4.0}`` for ``'chi2'``, or
        ``{"t_start": ..., "t_end": ..., "action": "flag"}`` for ``'date_range'``."""
        return self._editing_kwargs

    @editing_kwargs.setter
    def editing_kwargs(self, value):
        self._editing_kwargs = value

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        initial_covariance: "CovarianceMatrix",
        initial_deviation: Optional[np.ndarray] = None,
        process_noise: Optional["ProcessNoiseSettings"] = None,
        measurement_sigmas: Optional[Dict[str, float]] = None,
        covariance_analysis: bool = False,
        output: Optional["OutputSettings"] = None,
        editing_method: Optional[str] = None,
        editing_datasets: Optional[str | list] = None,
        editing_kwargs: Optional[Dict] = None,
    ):
        self.initial_covariance = initial_covariance
        self.initial_deviation = initial_deviation
        self.process_noise = process_noise
        self.measurement_sigmas = measurement_sigmas
        self.covariance_analysis = covariance_analysis
        self.output = output if output is not None else OutputSettings()
        self.editing_method = editing_method
        self.editing_datasets = editing_datasets
        self.editing_kwargs = editing_kwargs
