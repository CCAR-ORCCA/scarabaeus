# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional, Dict

# -----------------------#
#    Class Definition   #
# -----------------------#


class OutputSettings:
    """ Configuration for :class:`~scarabaeus.SolutionOD` output and storage.

    Controls which quantities the filter stores after a run and where JSON
    and ``.npz`` output files are written.  All flags default to sensible
    values; override only what you need.

    Parameters
    ----------
    save_prefit_residuals : bool
        Store prefit residuals in the solution. Defaults to ``True``.
    save_postfit_residuals : bool
        Store postfit residuals in the solution. Defaults to ``True``.
    save_covariance_history : bool
        Store covariance evolution across measurement epochs. Defaults to ``True``.
    save_consider_covariance_history : bool
        Store consider-parameter covariance evolution. Defaults to ``True``.
    save_state_deviation_history : bool
        Store state deviation history across measurement epochs. Defaults to ``True``.
    save_kalman_gains : bool
        Store Kalman gain matrices at each epoch (debugging). Defaults to ``False``.
    save_innovation_covariances : bool
        Store innovation covariance matrices at each epoch (debugging). Defaults to ``False``.
    save_nis_statistics : bool
        Store Normalized Innovation Squared (NIS) statistics. Defaults to ``False``.
    save_smoothed_solution : bool
        Run the RTS/Bierman smoother after the forward pass and store the smoothed
        solution. Defaults to ``True``.
    solution_output_path : str, optional
        Directory for the JSON solution file. ``None`` disables JSON output.
    solution_output_name : str, optional
        Base filename (without extension) for the JSON solution file.
        Defaults to ``'solution'``.
    fdm_output_path : str, optional
        Directory to write per-dataset ``.npz`` filter-data files.
        ``None`` disables ``.npz`` output.
    fdm_output_name : str, optional
        Base name prefix for filter-data files.
    metadata : dict, optional
        Arbitrary user metadata (version, producer, description, etc.).

    Notes
    -----
    All boolean flags default to sensible values for a standard OD run.
    Override only what you need; unset flags revert to their defaults when
    an instance is created with no arguments.

    See Also
    --------
    scarabaeus.SolutionOD : OD result container that respects these settings.
    scarabaeus.FilterSettings : Bundles this object with covariance and process noise.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def save_prefit_residuals(self) -> bool:
        """Store prefit residuals in the solution. Defaults to ``True``."""
        return self._save_prefit_residuals

    @save_prefit_residuals.setter
    def save_prefit_residuals(self, value: bool):
        self._save_prefit_residuals = value

    @property
    def save_postfit_residuals(self) -> bool:
        """Store postfit residuals in the solution. Defaults to ``True``."""
        return self._save_postfit_residuals

    @save_postfit_residuals.setter
    def save_postfit_residuals(self, value: bool):
        self._save_postfit_residuals = value

    @property
    def save_covariance_history(self) -> bool:
        """Store covariance evolution across measurement epochs. Defaults to ``True``."""
        return self._save_covariance_history

    @save_covariance_history.setter
    def save_covariance_history(self, value: bool):
        self._save_covariance_history = value

    @property
    def save_consider_covariance_history(self) -> bool:
        """Store consider-parameter covariance evolution. Defaults to ``True``."""
        return self._save_consider_covariance_history

    @save_consider_covariance_history.setter
    def save_consider_covariance_history(self, value: bool):
        self._save_consider_covariance_history = value

    @property
    def save_state_deviation_history(self) -> bool:
        """Store state deviation history across measurement epochs. Defaults to ``True``."""
        return self._save_state_deviation_history

    @save_state_deviation_history.setter
    def save_state_deviation_history(self, value: bool):
        self._save_state_deviation_history = value

    @property
    def save_kalman_gains(self) -> bool:
        """Store Kalman gain matrices at each epoch (debugging). Defaults to ``False``."""
        return self._save_kalman_gains

    @save_kalman_gains.setter
    def save_kalman_gains(self, value: bool):
        self._save_kalman_gains = value

    @property
    def save_innovation_covariances(self) -> bool:
        """Store innovation covariance matrices at each epoch (debugging). Defaults to ``False``."""
        return self._save_innovation_covariances

    @save_innovation_covariances.setter
    def save_innovation_covariances(self, value: bool):
        self._save_innovation_covariances = value

    @property
    def save_nis_statistics(self) -> bool:
        """Store Normalized Innovation Squared (NIS) statistics. Defaults to ``False``."""
        return self._save_nis_statistics

    @save_nis_statistics.setter
    def save_nis_statistics(self, value: bool):
        self._save_nis_statistics = value

    @property
    def save_smoothed_solution(self) -> bool:
        """If ``True``, run the RTS/Bierman smoother after the forward pass and store
        the smoothed solution. Defaults to ``True``."""
        return self._save_smoothed_solution

    @save_smoothed_solution.setter
    def save_smoothed_solution(self, value: bool):
        self._save_smoothed_solution = value

    @property
    def solution_output_path(self) -> Optional[str]:
        """Directory to save the JSON solution file. ``None`` disables JSON output."""
        return self._solution_output_path

    @solution_output_path.setter
    def solution_output_path(self, value):
        self._solution_output_path = value

    @property
    def solution_output_name(self) -> Optional[str]:
        """Base filename (without extension) for the JSON solution file."""
        return self._solution_output_name

    @solution_output_name.setter
    def solution_output_name(self, value):
        self._solution_output_name = value

    @property
    def fdm_output_path(self) -> Optional[str]:
        """Directory to write per-dataset ``.npz`` filter-data files.
        ``None`` disables ``.npz`` output."""
        return self._fdm_output_path

    @fdm_output_path.setter
    def fdm_output_path(self, value):
        self._fdm_output_path = value

    @property
    def fdm_output_name(self) -> Optional[str]:
        """Base name prefix for filter-data files.
        When set, files are named ``{fdm_output_name}_{set_name}.npz``."""
        return self._fdm_output_name

    @fdm_output_name.setter
    def fdm_output_name(self, value):
        self._fdm_output_name = value

    @property
    def metadata(self) -> Dict:
        """Arbitrary user metadata (version, producer, description, etc.)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict):
        self._metadata = value

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        save_prefit_residuals: bool = True,
        save_postfit_residuals: bool = True,
        save_covariance_history: bool = True,
        save_consider_covariance_history: bool = True,
        save_state_deviation_history: bool = True,
        save_kalman_gains: bool = False,
        save_innovation_covariances: bool = False,
        save_nis_statistics: bool = False,
        save_smoothed_solution: bool = True,
        solution_output_path: Optional[str] = None,
        solution_output_name: Optional[str] = "solution",
        fdm_output_path: Optional[str] = None,
        fdm_output_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.save_prefit_residuals = save_prefit_residuals
        self.save_postfit_residuals = save_postfit_residuals
        self.save_covariance_history = save_covariance_history
        self.save_consider_covariance_history = save_consider_covariance_history
        self.save_state_deviation_history = save_state_deviation_history
        self.save_kalman_gains = save_kalman_gains
        self.save_innovation_covariances = save_innovation_covariances
        self.save_nis_statistics = save_nis_statistics
        self.save_smoothed_solution = save_smoothed_solution
        self.solution_output_path = solution_output_path
        self.solution_output_name = solution_output_name
        self.fdm_output_path = fdm_output_path
        self.fdm_output_name = fdm_output_name
        self.metadata = metadata if metadata is not None else {}
