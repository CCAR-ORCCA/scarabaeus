# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional, Sequence
import numpy as np
from scarabaeus import ArrayWUnits, UncertaintyQuantification, CovarianceMatrix


# --------------------#
#  Class Definition  #
# --------------------#
class MonteCarloSamples(UncertaintyQuantification):
    """
    Uncertainty represented by a cloud of (optionally weighted) samples.

    The samples :math:`\\mathbf{x}^{(i)}`, :math:`i = 1, \\ldots, N`, stand for
    the distribution directly, with no Gaussian assumption. With weights
    :math:`w_i` (normalised to sum to one; equal weights when none are given)
    the moments are

    .. math::

        \\bar{\\mathbf{x}} = \\sum_i w_i\\, \\mathbf{x}^{(i)}, \\qquad
        P = \\frac{1}{1 - \\sum_i w_i^2} \\sum_i w_i
            (\\mathbf{x}^{(i)} - \\bar{\\mathbf{x}})
            (\\mathbf{x}^{(i)} - \\bar{\\mathbf{x}})^T,

    the unbiased weighted covariance, which reduces to the usual
    :math:`1/(N-1)` sample covariance for equal weights.

    Parameters
    ----------
    samples : np.ndarray
        Samples, shape ``(n_samples, n_dimensions)``.
    epoch : int
        J2000 ephemeris time (ET seconds) the samples refer to.
    frame : str, optional
        Reference frame. Defaults to ``'J2000'``.
    definition : optional
        State or sequence definition. Defaults to ``None``.
    labels : sequence of str, optional
        Name of each dimension. Defaults to ``x0, x1, ...``.
    weights : np.ndarray, optional
        Non-negative weight of each sample, shape ``(n_samples,)``. Normalised
        to sum to one. Defaults to equal weights.

    Raises
    ------
    SamplesShapeError
        If ``samples`` is not a 2-D array of finite values.
    LabelsSizeError
        If ``labels`` does not name every dimension.
    WeightsError
        If ``weights`` is not one non-negative finite value per sample with a
        positive sum.
    """

    # ---------------#
    # region Errors #
    # ---------------#
    class SamplesShapeError(ValueError):
        """The samples are not a finite 2-D array."""

    class LabelsSizeError(ValueError):
        """The labels do not name every dimension of the samples."""

    class WeightsError(ValueError):
        """The weights are not one non-negative finite value per sample."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        samples: np.ndarray,
        epoch: int,
        frame: str = "J2000",
        definition=None,
        labels: Optional[Sequence[str]] = None,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Store the samples, their labels and normalised weights.

        Parameters
        ----------
        samples : np.ndarray
            Samples, shape ``(n_samples, n_dimensions)``.
        epoch : int
            J2000 ephemeris time (ET seconds).
        frame : str, optional
            Reference frame name. Defaults to ``'J2000'``.
        definition : optional
            State or sequence definition object.
        labels : sequence of str, optional
            Name of each dimension.
        weights : np.ndarray, optional
            Non-negative weight of each sample.

        Raises
        ------
        SamplesShapeError
            If ``samples`` is not a finite 2-D array.
        LabelsSizeError
            If ``labels`` does not have one entry per dimension.
        WeightsError
            If ``weights`` is malformed.
        """
        super().__init__(epoch, frame, definition)

        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or not np.all(np.isfinite(samples)):
            raise MonteCarloSamples.SamplesShapeError(
                f"Samples must be a finite 2-D array (n_samples, n_dimensions); "
                f"got shape {samples.shape}."
            )
        self._samples = samples
        self._n_samples, self._n_dim = samples.shape

        if labels is None:
            labels = [f"x{i}" for i in range(self._n_dim)]
        if len(labels) != self._n_dim:
            raise MonteCarloSamples.LabelsSizeError(
                f"{len(labels)} labels given for {self._n_dim} dimensions."
            )
        self._labels = list(labels)

        if weights is None:
            weights = np.full(self._n_samples, 1.0 / self._n_samples)
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (self._n_samples,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or weights.sum() <= 0.0
        ):
            raise MonteCarloSamples.WeightsError(
                f"Weights must be {self._n_samples} non-negative finite values "
                f"with a positive sum; got shape {weights.shape}."
            )
        self._weights = weights / weights.sum()

    # endregion Constructor #
    # -----------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def samples(self) -> np.ndarray:
        """The samples, shape ``(n_samples, n_dimensions)``."""
        return self._samples

    @property
    def weights(self) -> np.ndarray:
        """Normalised weight of each sample, shape ``(n_samples,)``."""
        return self._weights

    @property
    def labels(self) -> list:
        """Name of each dimension."""
        return self._labels

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._n_samples

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""
        return self._n_dim

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def to_array(self) -> np.ndarray:
        """
        Return the samples as an array.

        Returns
        -------
        np.ndarray
            The samples, shape ``(n_samples, n_dimensions)``.
        """
        return self._samples

    def get_mean(self) -> np.ndarray:
        """
        Weighted sample mean.

        Returns
        -------
        np.ndarray
            Mean, shape ``(n_dimensions,)``.
        """
        return self._weights @ self._samples

    def get_covariance(self) -> np.ndarray:
        """
        Unbiased weighted sample covariance.

        Returns
        -------
        np.ndarray
            Covariance, shape ``(n_dimensions, n_dimensions)``.
        """
        centred = self._samples - self.get_mean()
        cov = (centred * self._weights[:, None]).T @ centred
        return cov / (1.0 - np.sum(self._weights**2))

    def get_standard_deviations(self) -> np.ndarray:
        """
        Square root of the diagonal of :meth:`get_covariance`.

        Returns
        -------
        np.ndarray
            1-sigma of each dimension, shape ``(n_dimensions,)``.
        """
        return np.sqrt(np.diag(self.get_covariance()))

    def get_percentile(self, percentile) -> np.ndarray:
        """
        Weighted percentile of each dimension.

        The sorted samples are placed at the weight accumulated before each
        one, rescaled to ``[0, 1]``, and the percentile is interpolated
        linearly between them. For equal weights these positions are
        :math:`i / (N - 1)`, so the result matches
        ``np.percentile(samples, percentile, axis=0)``.

        Parameters
        ----------
        percentile : float or array_like
            Percentile(s) in ``[0, 100]``.

        Returns
        -------
        np.ndarray
            Shape ``(n_dimensions,)`` for a scalar percentile, otherwise
            ``(len(percentile), n_dimensions)``.
        """
        q = np.atleast_1d(np.asarray(percentile, dtype=float)) / 100.0
        out = np.empty((q.size, self._n_dim))
        for j in range(self._n_dim):
            order = np.argsort(self._samples[:, j])
            w = self._weights[order]
            positions = np.cumsum(w) - w
            if positions[-1] > 0.0:
                positions /= positions[-1]
            out[:, j] = np.interp(q, positions, self._samples[order, j])
        return out[0] if np.ndim(percentile) == 0 else out

    def effective_sample_size(self) -> float:
        """
        Kish effective sample size :math:`1 / \\sum_i w_i^2`.

        Returns
        -------
        float
            ``n_samples`` for equal weights, smaller otherwise.
        """
        return float(1.0 / np.sum(self._weights**2))

    def to_covariance_matrix(self) -> CovarianceMatrix:
        """
        Gaussian summary of the samples as a covariance matrix.

        Returns
        -------
        CovarianceMatrix
            :meth:`get_covariance` with unitless entries, at this epoch and
            frame.
        """
        cov = self.get_covariance()
        matrix = [
            [ArrayWUnits(cov[i, j], None) for j in range(self._n_dim)]
            for i in range(self._n_dim)
        ]
        return CovarianceMatrix(
            matrix, self._epoch, self._frame, definition=self._definition
        )

    # endregion Methods #
    # -------------------#
