# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from scarabaeus import CovarianceMatrix, StateArray
import warnings


# --------------------#
#  Class Definition  #
# --------------------#
class ProcessNoise(ABC):
    """ Abstract base class for process noise modeling in orbit determination filters.

    Provides a common interface for different process noise models used to account
    for unmodeled dynamics in spacecraft orbit determination. Handles conversion
    between continuous-time and discrete-time representations.

    Parameters
    ----------
    continuous_time_covariance : CovarianceMatrix
        The continuous-time process noise covariance matrix (power spectral density).
    state_definition : optional
        State vector definition. Defaults to None.
    sequence_definition : optional
        Sequence definition for multi-leg trajectories. Defaults to None.

    Notes
    -----
    Process noise accounts for unmodeled spacecraft dynamics and is primarily used
    in sequential filters to address model uncertainties and filter saturation.
    Concrete sub-classes must implement :meth:`_compute_discrete_time_covariances`.

    See Also
    --------
    scarabaeus.StateNoiseCompensation : Piecewise-constant SNC implementation.
    scarabaeus.DynamicModelCompensation : Exponentially-correlated DMC implementation.
    scarabaeus.ProcessNoiseSettings : Configuration object consumed by the filter.

    References
    ----------
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        continuous_time_covariance: CovarianceMatrix,
        state_definition=None,
        sequence_definition=None,
    ):
        self.continuous_time_covariance = continuous_time_covariance
        self.state_definition = state_definition
        self.sequence_definition = sequence_definition

        self._init_consider_partition()
        self._validate_definitions()

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def frame(self) -> str:
        """The reference frame of the process noise covariance."""
        return self.continuous_time_covariance.frame

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    @abstractmethod
    def _compute_discrete_time_covariances(self, measurement_times: list) -> None:
        """
        Compute discrete-time process noise covariances.

        Parameters
        ----------
        measurement_times : list
            List of time epochs at which the process noise is evaluated.

        Returns
        -------
        None
        """
        pass

    def _validate_definitions(self) -> None:
        """
        Validate state definitions for process noise compatibility.

        Returns
        -------
        None
        """
        if self.state_definition is not None:
            self._check_state_definition(self.state_definition)
        elif self.sequence_definition is not None:
            for model in self.sequence_definition.models:
                if isinstance(model, StateArray):
                    self._check_state_definition(model.state)

    @abstractmethod
    def _check_state_definition(self, state_definition) -> None:
        """
        Validate a single state definition.

        Parameters
        ----------
        state_definition : list
            State definition to validate.

        Returns
        -------
        None
        """
        pass

    def _get_state_size(self) -> int:
        """
        Get the size of the state vector.

        Returns
        -------
        int
            Size of the state vector.
        """
        if self.state_definition is not None:
            # return sum([component[1] for component in self.state_definition])
            return self.n_est
        elif self.sequence_definition is not None:
            return self.sequence_definition.states_n[0]
        else:
            return 6

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
        if self.state_definition is not None:
            state_def = self.state_definition
        else:
            state_def = self.sequence_definition.models[idx_leg].state

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

    def transform_from_rtn(self, Q: np.ndarray) -> np.ndarray:
        """
        Rotate the process noise covariance from RTN frame to the inertial frame.

        The RTN (Radial-Transverse-Normal) basis vectors are constructed from
        the current position and velocity in the state definition.  The rotation
        is applied as ``Q_inertial = T @ Q_rtn @ T^T`` where ``T`` is the
        3x3 RTN basis matrix.

        This method is a no-op when ``self.continuous_time_covariance.frame`` is
        not ``'RTN'`` — the input matrix is returned unchanged.  It also returns
        ``Q`` unchanged if ``position`` or ``velocity`` cannot be found in the
        state definition.

        Parameters
        ----------
        Q : np.ndarray
            Process noise covariance matrix expressed in the RTN frame.

        Returns
        -------
        np.ndarray
            Covariance matrix expressed in the inertial frame, or the original
            ``Q`` when no rotation is needed.
        """
        if self.continuous_time_covariance.frame != "RTN":
            return Q

        # Get position and velocity from state definition
        position = None
        velocity = None
        for item in self.state_definition:
            if item[0] == "position":
                position = item[-1].values
            elif item[0] == "velocity":
                velocity = item[-1].values

        if position is None or velocity is None:
            return Q

        # Compute RTN basis vectors
        r = position / np.linalg.norm(position)
        h = np.cross(position, velocity)
        n = h / np.linalg.norm(h)
        t = np.cross(n, r)
        rtn_basis = np.vstack((r, t, n))

        # Transform to inertial frame
        return rtn_basis @ Q @ rtn_basis.T

    def _find_leg_index(self, current_time: float, sequence_epochs: list) -> int:
        """
        Find the leg index for a given time in a sequence.

        Parameters
        ----------
        current_time : float
            Current epoch.
        sequence_epochs : list
            List of epoch arrays for each leg.

        Returns
        -------
        int
            Leg index.
        """
        for i, sublist in enumerate(sequence_epochs):
            times = sublist.values if hasattr(sublist, "values") else sublist

            # Check if current time is in this leg's epochs
            if isinstance(times, np.ndarray):
                if current_time in times:
                    # Handle transition at end of leg
                    if current_time == times[-1] and i != len(sequence_epochs) - 1:
                        return i + 1
                    return i
            else:
                if current_time == times:
                    return i + 1 if i != len(sequence_epochs) - 1 else i

        return 0  # Default to first leg

    # endregion Methods #
    # -------------------#
