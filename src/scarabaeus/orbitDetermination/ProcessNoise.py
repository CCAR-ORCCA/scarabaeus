# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as spln
from scarabaeus import ArrayWUnits, CovarianceMatrix, StateArray, Units
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
        Covariance of the process noise input, held constant over each
        timeline interval.
    state_definition : optional
        State vector definition. Defaults to None.
    sequence_definition : optional
        Sequence definition for multi-leg trajectories. Defaults to None.
    max_interval : ArrayWUnits or float, optional
        Longest interval the noise is accumulated over, in seconds when a
        float. See :meth:`_noise_intervals`. Defaults to ``None``, no limit.
    body : Body or int, optional
        Body whose dynamics the noise drives, or its SPICE ID. Defaults to
        ``None``, the owner of the first position in the state.

    Raises
    ------
    ProcessNoise.NoiseBodyNotInStateError
        If the state of no leg carries the noise states of ``body``.
    ProcessNoise.ConsideredNoiseStateError
        If a noise state of ``body`` is considered.
    ProcessNoise.MissingNoiseStateError
        If a leg carries some, but not all, of the noise states of ``body``.

    See Also
    --------
    scarabaeus.StateNoiseCompensation : Piecewise-constant SNC implementation.
    scarabaeus.DynamicalModelCompensation : Exponentially-correlated DMC implementation.
    scarabaeus.ProcessNoiseSettings : Configuration object consumed by the filter.

    Notes
    -----
    Process noise accounts for unmodeled spacecraft dynamics and is primarily used
    in sequential filters to address model uncertainties and filter saturation.
    Concrete sub-classes must implement :meth:`_compute_discrete_time_covariances`.

    The noise drives the states of ``body`` listed in :attr:`_noise_states`,
    wherever they sit in the state of each leg, so several propagated bodies
    are noised with one model each, which the filters stack (see
    :meth:`combined_gamma_at`). On a leg whose state does not carry ``body``
    the noise is zero.

    References
    ----------
    .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class MissingReferenceStateError(ValueError):
        """Raised when an RTN process noise is discretized without the reference
        trajectory its rotation to the inertial frame is built from."""

    class NonPositiveIntervalError(ValueError):
        """Raised when the longest interval the noise is accumulated over is
        not a positive time."""

    class NoiseBodyNotInStateError(ValueError):
        """Raised when the state of no leg carries the body the noise drives."""

    class ConsideredNoiseStateError(ValueError):
        """Raised when a state the noise drives is considered rather than
        estimated, so the filters have no rows to add the noise to."""

    class MissingNoiseStateError(ValueError):
        """Raised when a leg carries some, but not all, of the states the noise
        drives for its body."""

    # endregion Errors #
    # ------------------#

    # States of the body the noise drives, in the row order of the gamma
    # matrices
    _noise_states = ("position", "velocity")

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        continuous_time_covariance: CovarianceMatrix,
        state_definition=None,
        sequence_definition=None,
        max_interval=None,
        body=None,
    ):
        self.continuous_time_covariance = continuous_time_covariance
        self.state_definition = state_definition
        self.sequence_definition = sequence_definition
        self.max_interval = self.interval_in_seconds(max_interval)
        # Resolved to the owner object in the state by _validate_definitions
        self.body = body
        # Epochs the discrete-time matrices were last computed on
        self.epochs = None

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
    def _compute_discrete_time_covariances(
        self, epochs: list, reference_state=None
    ) -> None:
        """
        Compute discrete-time process noise covariances.

        Parameters
        ----------
        epochs : list
            Filtering timeline (measurement and sequence event epochs) the
            filters index the process noise by.
        reference_state : callable, optional
            Maps an array of epochs to the reference ``(positions,
            velocities)``, each of shape ``(len(epochs), 3)``. Required when
            the continuous-time covariance is given in the RTN frame (see
            :meth:`_continuous_time_covariances`). Defaults to ``None``.

        Returns
        -------
        None
        """
        pass

    def _continuous_time_covariances(self, epochs, reference_state=None) -> list:
        """
        Inertial continuous-time covariance acting over each timeline interval.

        Entry ``k`` drives the interval ending at ``epochs[k]``. A covariance
        given in the RTN frame is rotated with the RTN frame of the reference
        trajectory at the middle of the part of that interval the noise acts
        over (see :meth:`_noise_intervals`; at ``epochs[0]`` for the first
        entry, which has no interval), so the rotation follows the spacecraft
        along the arc and onto every leg of a sequence.

        Parameters
        ----------
        epochs : array_like
            Filtering timeline.
        reference_state : callable, optional
            See :meth:`_compute_discrete_time_covariances`. Defaults to
            ``None``.

        Returns
        -------
        list of np.ndarray
            One ``(3, 3)`` covariance per epoch.

        Raises
        ------
        ProcessNoise.MissingReferenceStateError
            If the covariance is in the RTN frame and ``reference_state`` is
            not given.
        """
        Q = self.continuous_time_covariance.matrix_without_units()
        if self.frame != "RTN":
            return [Q] * len(epochs)
        if reference_state is None:
            raise self.MissingReferenceStateError(
                "An RTN process noise is rotated to the inertial frame along "
                "the reference trajectory, which was not provided."
            )

        positions, velocities = reference_state(self._noise_epochs(epochs))
        return [
            self.transform_from_rtn(Q, position, velocity)
            for position, velocity in zip(positions, velocities)
        ]

    def _noise_intervals(self, epochs) -> np.ndarray:
        """
        Time the process noise acts over in each timeline interval.

        Entry ``k`` belongs to the interval ending at ``epochs[k]`` and is
        zero for the first entry, which has no interval. An interval longer
        than :attr:`max_interval`, such as a data gap, is cut to it: the
        input acts over the last :attr:`max_interval` of the interval only.
        The noise held constant over a whole gap grows as :math:`\\Delta t^4`
        in position and soon swamps the uncertainty mapped across the gap by
        the dynamics; cut, it stops growing at the value of an interval
        :attr:`max_interval` long, and stays continuous and non-decreasing in
        the interval length.

        Parameters
        ----------
        epochs : array_like
            Filtering timeline.

        Returns
        -------
        np.ndarray
            One duration per epoch, in seconds.
        """
        epochs = np.asarray(epochs, dtype=float)
        intervals = np.diff(epochs, prepend=epochs[:1])
        if self.max_interval is not None:
            intervals = np.minimum(intervals, self.max_interval)
        return intervals

    def _noise_epochs(self, epochs) -> np.ndarray:
        """
        Epoch each timeline interval's noise is evaluated at.

        Entry ``k`` is the middle of the part of the interval ending at
        ``epochs[k]`` that the noise acts over (see :meth:`_noise_intervals`),
        and ``epochs[0]`` for the first entry, which has no interval. The
        interval boundaries are event epochs on a sequence, so every entry
        lies inside a single leg.

        Parameters
        ----------
        epochs : array_like
            Filtering timeline.

        Returns
        -------
        np.ndarray
            One epoch per timeline epoch, in seconds past J2000.
        """
        epochs = np.asarray(epochs, dtype=float)
        intervals = self._noise_intervals(epochs)
        return np.concatenate((epochs[:1], epochs[1:] - 0.5 * intervals[1:]))

    @staticmethod
    def interval_in_seconds(interval) -> Optional[float]:
        """
        A time span in seconds, checked to be positive.

        Parameters
        ----------
        interval : ArrayWUnits, float or None
            Time span, in seconds when a float. ``None`` is returned as is.

        Returns
        -------
        float or None
            The span in seconds.

        Raises
        ------
        ProcessNoise.NonPositiveIntervalError
            If the span is not positive.
        """
        if interval is None:
            return None
        if isinstance(interval, ArrayWUnits):
            interval = interval.convert_to(Units.get_units("sec")).values
        seconds = np.asarray(interval, dtype=float).item()
        if not seconds > 0.0:
            raise ProcessNoise.NonPositiveIntervalError(
                f"The longest interval the process noise is accumulated over "
                f"must be a positive time, got {seconds} s."
            )
        return seconds

    def noise_rows_at(self, k: int) -> np.ndarray:
        """
        Rows of the estimated states the noise drives at timeline index ``k``.

        They are the rows of the states of :attr:`body` listed in
        :attr:`_noise_states`, on the leg the filter is on at ``epochs[k]``:
        the leg an event epoch starts, as the filters map the state through
        the node before adding the noise. Without a state definition, the
        leading rows.

        Parameters
        ----------
        k : int
            Index in the epochs the matrices were computed on.

        Returns
        -------
        np.ndarray
            Integer array, empty on a leg whose state does not carry
            :attr:`body`.
        """
        if self._leg_rows is None:
            return np.arange(self.gamma_matrices[k].shape[0])
        leg = np.searchsorted(self._event_epochs, self.epochs[k], side="right")
        return self._leg_rows[min(leg, len(self._leg_rows) - 1)]

    def gamma_at(self, k: int, n: int) -> np.ndarray:
        """
        Process noise gamma matrix at timeline index ``k`` for an ``n``-state.

        The noise drives only the states of its body (see
        :meth:`noise_rows_at`), so the rows of every other state are zero.
        The size is chosen where the matrix is used: a filter carries fewer
        states when some are considered, and a different number on each leg
        of a sequence.

        Parameters
        ----------
        k : int
            Index in the epochs the matrices were computed on.
        n : int
            Size of the state the noise is added to.

        Returns
        -------
        np.ndarray
            Gamma matrix of shape ``(n, m)``, with ``m`` the size of the
            noise vector, 3.
        """
        gamma = self.gamma_matrices[k]
        rows = self.noise_rows_at(k)
        full = np.zeros((n, gamma.shape[1]))
        if rows.size:
            full[rows] = gamma
        return full

    def covariance_at(self, k: int, n: int) -> np.ndarray:
        """
        Discrete-time process noise covariance at timeline index ``k`` for an
        ``n``-state, zero outside the states of its body (see
        :meth:`gamma_at`).

        Parameters
        ----------
        k : int
            Index in the epochs the matrices were computed on.
        n : int
            Size of the state the noise is added to.

        Returns
        -------
        np.ndarray
            Covariance of shape ``(n, n)``.
        """
        Q_block = self.discrete_time_process_covariances[k]
        rows = self.noise_rows_at(k)
        Q = np.zeros((n, n))
        if rows.size:
            Q[np.ix_(rows, rows)] = Q_block
        return Q

    def input_covariance_at(self, k: int) -> np.ndarray:
        """
        Inertial covariance of the noise input over the interval ending at
        timeline index ``k``, which the SRIF factors.

        Parameters
        ----------
        k : int
            Index in the epochs the matrices were computed on.

        Returns
        -------
        np.ndarray
            Covariance of shape ``(3, 3)``.
        """
        return self.Q_tilde[k]

    @staticmethod
    def combined_gamma_at(models: list, k: int, n: int) -> np.ndarray:
        """
        Gamma matrices of several models at timeline index ``k``, side by side.

        Each model drives the states of its own body (see
        :meth:`noise_rows_at`), so the noise inputs are independent and stack:

        .. math::

            \\mathbf{\\Gamma}_k = [\\,\\mathbf{\\Gamma}_k^{(1)} \\;\\cdots\\;
                \\mathbf{\\Gamma}_k^{(m)}\\,]

        with three noise components per body.

        Parameters
        ----------
        models : list of ProcessNoise
            One model per noised body, discretized on the same epochs.
        k : int
            Index in the epochs the matrices were computed on.
        n : int
            Size of the state the noise is added to.

        Returns
        -------
        np.ndarray
            Gamma matrix of shape ``(n, 3 m)``, for ``m`` models.
        """
        return np.hstack([model.gamma_at(k, n) for model in models])

    @staticmethod
    def combined_covariance_at(models: list, k: int, n: int) -> np.ndarray:
        """
        Discrete-time process noise covariance of several models at timeline
        index ``k``, the sum of theirs as their inputs are independent:

        .. math::

            \\mathbf{Q}_k = \\sum_i \\mathbf{Q}_k^{(i)}

        Parameters
        ----------
        models : list of ProcessNoise
            One model per noised body, discretized on the same epochs.
        k : int
            Index in the epochs the matrices were computed on.
        n : int
            Size of the state the noise is added to.

        Returns
        -------
        np.ndarray
            Covariance of shape ``(n, n)``.
        """
        return sum(model.covariance_at(k, n) for model in models)

    @staticmethod
    def combined_input_covariance_at(models: list, k: int) -> np.ndarray:
        """
        Covariance of the stacked noise inputs of several models (see
        :meth:`combined_gamma_at`) at timeline index ``k``, block diagonal as
        the inputs are independent:

        .. math::

            \\tilde{\\mathbf{Q}}_k = \\mathrm{blockdiag}(\\tilde{\\mathbf{Q}}_k^{(1)},
                \\ldots, \\tilde{\\mathbf{Q}}_k^{(m)})

        Parameters
        ----------
        models : list of ProcessNoise
            One model per noised body, discretized on the same epochs.
        k : int
            Index in the epochs the matrices were computed on.

        Returns
        -------
        np.ndarray
            Covariance of shape ``(3 m, 3 m)``, for ``m`` models.
        """
        return spln.block_diag(*(model.input_covariance_at(k) for model in models))

    def _validate_definitions(self) -> None:
        """
        Locate the states the noise drives on every leg.

        Sets the event epochs splitting the legs of a sequence and, per leg,
        the rows in the estimated state of the :attr:`_noise_states` of
        :attr:`body`, which is resolved to its owner object in the state.
        Without a state or sequence definition the noise drives the leading
        rows (see :meth:`noise_rows_at`).

        Returns
        -------
        None

        Raises
        ------
        ProcessNoise.NoiseBodyNotInStateError
            If the state of no leg carries the noise states of :attr:`body`,
            or, with no ``body`` given, carries a position.
        """
        self._event_epochs = np.array([])
        if self.state_definition is not None:
            definitions = [self.state_definition]
        elif self.sequence_definition is not None:
            definitions = [
                model.state
                for model in self.sequence_definition.models
                if isinstance(model, StateArray)
            ]
            # Single-epoch entries are the events between the legs
            epochs = [
                np.atleast_1d(np.asarray(epoch.times.values, dtype=float))
                for epoch in self.sequence_definition.epochs_vec
            ]
            self._event_epochs = np.array([e[0] for e in epochs if e.size == 1])
        else:
            self._leg_rows = None
            return

        owners = [entry[4] for definition in definitions for entry in definition]
        if self.body is None:
            body_id = next(
                (
                    getattr(entry[4], "spice_id", entry[4])
                    for definition in definitions
                    for entry in definition
                    if entry[0] == "position"
                ),
                None,
            )
            if body_id is None:
                raise self.NoiseBodyNotInStateError(
                    "The process noise drives the owner of the first position "
                    "in the state, which carries none: name the body."
                )
        else:
            body_id = getattr(self.body, "spice_id", self.body)
        self.body = next(
            (owner for owner in owners if getattr(owner, "spice_id", owner) == body_id),
            self.body,
        )

        self._leg_rows = [self._noise_rows(definition) for definition in definitions]
        if not any(rows.size for rows in self._leg_rows):
            raise self.NoiseBodyNotInStateError(
                f"The process noise drives the {', '.join(self._noise_states)} of "
                f"{getattr(self.body, 'name', self.body)}, which the state of no "
                "leg carries."
            )

    def _noise_rows(self, state_definition) -> np.ndarray:
        """
        Rows in the estimated state of the :attr:`_noise_states` of
        :attr:`body`, in that order.

        Parameters
        ----------
        state_definition : iterable
            Entries ``(name, size, estimation, dynamics, owner, value)`` of one
            leg.

        Returns
        -------
        np.ndarray
            Integer array, empty when the leg does not carry :attr:`body`.

        Raises
        ------
        ProcessNoise.ConsideredNoiseStateError
            If a noise state of :attr:`body` is considered.
        ProcessNoise.MissingNoiseStateError
            If the leg carries some, but not all, of the noise states.
        """
        body_id = getattr(self.body, "spice_id", self.body)
        body_name = getattr(self.body, "name", self.body)
        rows = {}
        # Cursor over the estimated states only, which the filters size the
        # noise to
        cursor = 0
        for name, dim, est_flag, _, owner, _ in state_definition:
            if (
                name in self._noise_states
                and getattr(owner, "spice_id", owner) == body_id
            ):
                if est_flag == "considered":
                    raise self.ConsideredNoiseStateError(
                        f"The process noise drives the {name} of {body_name}, "
                        "which is considered: estimate it, or noise another body."
                    )
                rows[name] = np.arange(cursor, cursor + dim)
            if est_flag != "considered":
                cursor += dim

        if not rows:
            return np.array([], dtype=int)
        missing = [name for name in self._noise_states if name not in rows]
        if missing:
            raise self.MissingNoiseStateError(
                f"The process noise drives the {', '.join(self._noise_states)} "
                f"of {body_name}, but a leg carries no {', '.join(missing)} for it."
            )
        return np.concatenate([rows[name] for name in self._noise_states])

    def _init_consider_partition(self, idx_leg=0) -> None:
        """
        Split the state into estimated and considered parameters.

        Entries flagged ``"considered"`` go to the consider partition and every
        other entry to the estimated one.

        Parameters
        ----------
        idx_leg : int, optional
            Leg of :attr:`sequence_definition` whose state is used when no
            single-arc ``state_definition`` is set. Defaults to ``0``.

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

    def transform_from_rtn(
        self, Q: np.ndarray, position: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a covariance from the RTN frame of a state to the inertial frame.

        The rows of ``T`` are the radial, transverse and normal unit vectors
        built from ``position`` and ``velocity``, so an RTN vector maps to the
        inertial frame as ``T^T @ a_rtn`` and the covariance as
        ``Q_inertial = T^T @ Q_rtn @ T``.

        This method is a no-op when ``self.continuous_time_covariance.frame`` is
        not ``'RTN'`` — the input matrix is returned unchanged.

        Parameters
        ----------
        Q : np.ndarray
            ``(3, 3)`` covariance expressed in the RTN frame.
        position : np.ndarray
            Inertial position the RTN frame is built from.
        velocity : np.ndarray
            Inertial velocity the RTN frame is built from.

        Returns
        -------
        np.ndarray
            Covariance matrix expressed in the inertial frame, or the original
            ``Q`` when no rotation is needed.
        """
        if self.continuous_time_covariance.frame != "RTN":
            return Q

        # Compute RTN basis vectors
        r = position / np.linalg.norm(position)
        h = np.cross(position, velocity)
        n = h / np.linalg.norm(h)
        t = np.cross(n, r)
        rtn_basis = np.vstack((r, t, n))

        # Transform to inertial frame
        return rtn_basis.T @ Q @ rtn_basis

    # endregion Methods #
    # -------------------#
