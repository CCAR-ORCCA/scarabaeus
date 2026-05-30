# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
import numpy as np
from scarabaeus import (
    CovarianceMatrix,
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
        Continuous-time process noise power spectral density matrix :math:`Q_c`.
    beta : np.ndarray, optional
        DMC only — diagonal decay-rate vector :math:`\\beta` (units: 1/time) for
        the first-order Gauss–Markov stochastic acceleration model.
        Required when ``type`` is ``'DMC'``.

    Raises
    ------
    ValueError
        If ``type`` is not one of ``'SNC'``, ``'DMC'``, or ``'STOC'``.
    ValueError
        If ``type`` is ``'DMC'`` and ``beta`` is ``None``.

    Notes
    -----
    The concrete :class:`~scarabaeus.ProcessNoise` subclass
    (:class:`~scarabaeus.StateNoiseCompensation` or
    :class:`~scarabaeus.DynamicModelCompensation`) is instantiated lazily
    by the filter, not here.

    See Also
    --------
    scarabaeus.FilterSettings : Bundles this object with the initial covariance.
    scarabaeus.StateNoiseCompensation : SNC process noise model.
    scarabaeus.DynamicModelCompensation : DMC process noise model.
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
        """Continuous-time process noise power spectral density matrix :math:`Q_c`."""
        return self._Q_cont

    @Q_cont.setter
    def Q_cont(self, value):
        self._Q_cont = value

    @property
    def beta(self) -> Optional[np.ndarray]:
        """DMC only — diagonal decay-rate vector :math:`\\beta` (units: 1/time) for
        the first-order Gauss–Markov stochastic acceleration model."""
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        type: str,
        Q_cont: "CovarianceMatrix",
        beta: Optional[np.ndarray] = None,
    ):
        self.type = type
        self.Q_cont = Q_cont
        self.beta = beta

        if self.type not in ("SNC", "DMC", "STOC"):
            raise ValueError(
                f"Invalid process noise type: {self.type}. "
                "Must be 'SNC', 'DMC', or 'STOC'."
            )
        if self.type == "DMC" and self.beta is None:
            raise ValueError("DMC requires beta to be specified.")
