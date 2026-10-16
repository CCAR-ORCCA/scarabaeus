# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import os
from typing import Optional, Dict, Literal

# -----------------------#
#    Class Definition   #
# -----------------------#


class OutputSettings:
    """ Configuration for :class:`~scarabaeus.SolutionOD` output and storage.

    Controls which quantities the filter stores after a run and where the JSON
    output files are written.  All flags default to sensible values; override
    only what you need.

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
        Directory to write per-dataset JSON filter-data files.
        ``None`` disables filter-data output.
    fdm_output_name : str, optional
        Base name prefix for filter-data files.
    fdm_output_format : {"json", "pickle"}, optional
        On-disk format for the filter-data files — the per-dataset measurement
        records the filter rebuilds each iteration. Defaults to ``"json"``.
        This does **not** control the trajectory sidecars; see
        ``trajectory_output_format``.
    save_trajectory_stms : bool
        Whether to write the STM sidecar beside each trajectory SPK. It is the
        largest artifact an OD run produces — one matrix per output epoch, per
        iteration — and the filter serves its own STM lookups from memory, so a
        run that does not read the sidecar back can turn it off. Defaults to
        ``True``.
    trajectory_output_format : {"json", "pickle"}, optional
        On-disk format for the sidecars written beside each trajectory SPK: the
        STM history (``*_stms.*``) and the parameter / settings / mass-profile
        records. Defaults to ``"json"``.

        The STM sidecar is by far the largest artifact an OD run produces — one
        matrix per output epoch, growing with the square of the state size — so
        a finely sampled arc or an extra solve-for parameter can push it into
        hundreds of megabytes of text. ``"pickle"`` writes the same content as
        a compact binary, which both writes and reads back considerably faster.
        Either format is read back transparently.
    metadata : dict, optional
        Arbitrary user metadata (version, producer, description, etc.). Its
        keys are merged last into the saved ``metadata`` block, so they
        override the generated fields of the same name.
    save_full_covariance : bool
        Write the final solution's covariances as full matrices
        (``covariance_full``, ``covariance_smooth_full``,
        ``covariance_consider_full``) instead of their diagonals
        (``*_diagonals``). Iteration blocks always store diagonals, and a block
        never holds both. Defaults to ``True``.

    See Also
    --------
    scarabaeus.SolutionOD : OD result container that respects these settings.
    scarabaeus.FilterSettings : Bundles this object with covariance and process noise.

    Notes
    -----
    All boolean flags default to sensible values for a standard OD run.
    Override only what you need; unset flags revert to their defaults when
    an instance is created with no arguments.
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
        """Directory to write per-dataset JSON filter-data files.
        ``None`` disables filter-data output."""
        return self._fdm_output_path

    @fdm_output_path.setter
    def fdm_output_path(self, value):
        self._fdm_output_path = value

    @property
    def fdm_output_name(self) -> Optional[str]:
        """Base name prefix for filter-data files.
        When set, files are named ``{fdm_output_name}_{set_name}.json``."""
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
        fdm_output_format: Optional[Literal["json", "pickle"]] = "json",
        trajectory_output_format: Optional[Literal["json", "pickle"]] = "json",
        metadata: Optional[Dict] = None,
        output_dir: Optional[str] = None,
        run_name: Optional[str] = "scenario",
        save_reference_trajectory: bool = True,
        save_estimated_trajectory: bool = True,
        save_trajectory_stms: bool = True,
        save_full_covariance: bool = True,
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

        # Unified output convention: a single directory + run name from which
        # per-artifact paths are derived (see ``path_for``). Explicit
        # solution_/fdm_ paths still win for backward compatibility.
        self._output_dir = output_dir
        self._run_name = run_name or "scenario"

        # When the unified convention is used, derive the legacy solution/fdm
        # paths from it unless they were given explicitly.
        if output_dir is not None and solution_output_path is None:
            solution_output_path = output_dir
        if output_dir is not None and fdm_output_path is None:
            fdm_output_path = output_dir

        self.solution_output_path = solution_output_path
        self.solution_output_name = solution_output_name
        self.fdm_output_path = fdm_output_path
        self.fdm_output_name = fdm_output_name
        self.fdm_output_format = fdm_output_format or "json"
        self.trajectory_output_format = trajectory_output_format or "json"

        # OD artifact toggles (extend SolutionOD.save output).
        self.save_reference_trajectory = save_reference_trajectory
        self.save_estimated_trajectory = save_estimated_trajectory
        self.save_trajectory_stms = save_trajectory_stms
        self.save_full_covariance = save_full_covariance

        self.metadata = metadata if metadata is not None else {}

    # ------------------------------------------------------------------
    # Unified output-path convention
    # ------------------------------------------------------------------
    @property
    def output_dir(self) -> Optional[str]:
        """Single directory all artifacts for this run are written under."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def run_name(self) -> str:
        """Base name (prefix) shared by every artifact of this run."""
        return self._run_name

    @run_name.setter
    def run_name(self, value):
        self._run_name = value or "scenario"

    def path_for(self, artifact: str, ext: str = "json") -> str:
        """
        Build a standardized ``<output_dir>/<run_name>_<artifact>.<ext>`` path.

        Parameters
        ----------
        artifact : str
            Artifact suffix, e.g. ``"solution"``, ``"reference"``, ``"estimated"``,
            ``"stms"``, ``"settings"``, ``"parameters"``.
        ext : str
            File extension without the dot. Defaults to ``"json"``.

        Returns
        -------
        str
            The resolved path. Falls back to the current directory when
            ``output_dir`` is unset.
        """
        directory = self._output_dir or self.solution_output_path or "."
        return os.path.join(directory, f"{self._run_name}_{artifact}.{ext}")
