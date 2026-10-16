# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
import numpy as np
from scarabaeus import (
    CovarianceMatrix,
    ProcessNoise,
)

# -----------------------#
#    Class Definition   #
# -----------------------#


class ProcessNoiseSettings:
    """ Process noise configuration for the filter.

    A lightweight container that selects the process noise model type and
    stores its continuous-time covariance and model-specific parameters.
    Passed to :class:`~scarabaeus.FilterSettings` and consumed by :class:`~scarabaeus.FilterOD`
    subclasses to build the concrete :class:`~scarabaeus.ProcessNoise` object.

    Parameters
    ----------
    type : str
        Process noise model type: ``'SNC'`` (State Noise Compensation),
        ``'DMC'`` (Dynamic Model Compensation), or ``'STOC'`` (stochastic).
    Q_cont : CovarianceMatrix
        Covariance :math:`Q_c` of the input held constant over each timeline
        interval: for SNC the unmodeled acceleration (km²/s⁴), for DMC the
        input driving the stochastic acceleration (km²/s⁶).
    beta : np.ndarray, optional
        DMC only — decay rates :math:`\\beta` (1/s) of the first-order
        Gauss–Markov stochastic acceleration. The filter takes them from the
        dynamics: the ``beta_fogm`` state when there is one, else the
        ``fogm_beta`` of the :class:`~scarabaeus.ForceModelTranslation`. When
        given, they must match those, or the filter raises
        :class:`~scarabaeus.FilterOD.ProcessNoiseBetaMismatchError`.
        Defaults to ``None``.
    max_interval : ArrayWUnits or float, optional
        Longest interval the noise is accumulated over, in seconds when a
        float. A longer interval, such as a data gap, receives the noise of
        one this long, so the covariance does not blow up across it. See
        :meth:`~scarabaeus.ProcessNoise._noise_intervals`. Defaults to
        ``None``, no limit.
    body : Body or int, optional
        Body whose dynamics the noise drives, or its SPICE ID. The noise
        acts on that body's position and velocity (and ``a_fogm`` for DMC),
        wherever they sit in the state. Give the filter one settings object
        per body to noise several propagated bodies. Defaults to ``None``,
        the primary body of the trajectory.

    Raises
    ------
    ValueError
        If ``type`` is not one of ``'SNC'``, ``'DMC'``, or ``'STOC'``.
    ProcessNoise.NonPositiveIntervalError
        If ``max_interval`` is not a positive time.

    See Also
    --------
    scarabaeus.FilterSettings : Bundles this object with the initial covariance.
    scarabaeus.StateNoiseCompensation : SNC process noise model.
    scarabaeus.DynamicalModelCompensation : DMC process noise model.

    Notes
    -----
    The concrete :class:`~scarabaeus.ProcessNoise` subclass
    (:class:`~scarabaeus.StateNoiseCompensation` or
    :class:`~scarabaeus.DynamicalModelCompensation`) is instantiated lazily
    by the filter, not here.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def type(self) -> str:
        """Process noise model type: ``'SNC'``, ``'DMC'``, or ``'STOC'``."""
        return self._type

    @type.setter
    def type(self, value: str):
        self._type = value

    @property
    def Q_cont(self) -> "CovarianceMatrix":
        """Covariance of the input held constant over each timeline interval:
        the unmodeled acceleration (SNC), or the input driving the stochastic
        acceleration (DMC)."""
        return self._Q_cont

    @Q_cont.setter
    def Q_cont(self, value):
        self._Q_cont = value

    @property
    def beta(self) -> Optional[np.ndarray]:
        """DMC only — decay rates :math:`\\beta` (1/s) of the Gauss–Markov
        stochastic acceleration, checked against the dynamics, or ``None``
        to take them from the dynamics."""
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def max_interval(self) -> Optional[float]:
        """Longest interval the noise is accumulated over, in seconds, or
        ``None`` for no limit."""
        return self._max_interval

    @max_interval.setter
    def max_interval(self, value):
        self._max_interval = ProcessNoise.interval_in_seconds(value)

    @property
    def body(self):
        """Body whose dynamics the noise drives (a Body or its SPICE ID), or
        ``None`` for the primary body of the trajectory."""
        return self._body

    @body.setter
    def body(self, value):
        self._body = value

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        type: str,
        Q_cont: "CovarianceMatrix",
        beta: Optional[np.ndarray] = None,
        max_interval=None,
        body=None,
    ):
        self.type = type
        self.Q_cont = Q_cont
        self.beta = beta
        self.max_interval = max_interval
        self.body = body

        if self.type not in ("SNC", "DMC", "STOC"):
            raise ValueError(
                f"Invalid process noise type: {self.type}. "
                "Must be 'SNC', 'DMC', or 'STOC'."
            )
