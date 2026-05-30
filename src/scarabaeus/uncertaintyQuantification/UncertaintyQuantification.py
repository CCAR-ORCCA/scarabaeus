# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from abc import ABC, abstractmethod
import numpy as np

#--------------------#
#  Class Definition  #
#--------------------#
class UncertaintyQuantification(ABC):
    """
    Abstract base class for uncertainty quantification.

    Provides a common interface for different uncertainty representations
    including covariance matrices, sigma points, and other uncertainty
    propagation methods.

    Parameters
    ----------
    epoch : int
        The epoch of the uncertainty representation.
    frame : str, optional
        The reference frame. Defaults to 'J2000'.
    definition : optional
        Sequence or state definition. Defaults to None.
    """

    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(
        self,
        epoch: int,
        frame: str = 'J2000',
        definition=None
    ):
        """
        Store epoch, frame, and optional state definition.

        Parameters
        ----------
        epoch : int
            J2000 ephemeris time (ET seconds) for this uncertainty
            representation.
        frame : str, optional
            Reference frame name (e.g., ``'J2000'``).  Defaults to
            ``'J2000'``.
        definition : optional
            State or sequence definition object.  Passed through unchanged;
            interpretation is left to concrete subclasses.
        """
        self._epoch = epoch
        self._frame = frame
        self._definition = definition

    #----------------------#
    # region    Properties #
    #----------------------#
    @property
    def epoch(self) -> int:
        """The epoch of the uncertainty representation."""
        return self._epoch

    @property
    def frame(self) -> str:
        """The reference frame of the uncertainty representation."""
        return self._frame

    @property
    def definition(self):
        """The sequence or state definition."""
        return self._definition

    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#
    @abstractmethod
    def to_array(self) -> np.ndarray:
        """
        Convert uncertainty representation to numpy array.

        Returns
        -------
        np.ndarray
            Numpy array representation without units.
        """
        pass

    @abstractmethod
    def get_standard_deviations(self) -> np.ndarray:
        """
        Get standard deviations for each parameter.

        Returns
        -------
        np.ndarray
            Array of standard deviations.
        """
        pass

    # endregion Methods #
    #-------------------#

# NOTE: FOR THE FUTURE
'''
#==============================#
#   CLASS: SIGMA POINTS        #
#==============================#
class SigmaPoints(UncertaintyQuantification):
    """
    Represents uncertainty using sigma points (unscented transform).

    Used for nonlinear uncertainty propagation through the unscented transform
    method, which represents the distribution using a set of deterministically
    chosen sample points.

    Parameters
    ----------
    points : np.ndarray
        Array of sigma points, shape (n_points, n_dimensions).
    weights_mean : np.ndarray
        Weights for computing the mean, shape (n_points,).
    weights_cov : np.ndarray
        Weights for computing the covariance, shape (n_points,).
    epoch : int
        The epoch of the sigma points.
    frame : str, optional
        The reference frame. Defaults to 'J2000'.
    definition : optional
        Sequence or state definition. Defaults to None.

    Raises
    ------
    ValueError
        If dimensions are inconsistent or weights don't sum to 1.
    """

    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(
        self,
        points: np.ndarray,
        weights_mean: np.ndarray,
        weights_cov: np.ndarray,
        epoch: int,
        frame: str = 'J2000',
        definition=None
    ):
        """Initialize a SigmaPoints object."""
        super().__init__(epoch, frame, definition)
        
        self._validate_inputs(points, weights_mean, weights_cov)
        
        self._points = points
        self._weights_mean = weights_mean
        self._weights_cov = weights_cov
        self._n_points, self._n_dim = points.shape

    #----------------------#
    # region    Properties #
    #----------------------#
    @property
    def points(self) -> np.ndarray:
        """The sigma points array."""
        return self._points

    @property
    def weights_mean(self) -> np.ndarray:
        """Weights for mean computation."""
        return self._weights_mean

    @property
    def weights_cov(self) -> np.ndarray:
        """Weights for covariance computation."""
        return self._weights_cov

    @property
    def n_points(self) -> int:
        """Number of sigma points."""
        return self._n_points

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""
        return self._n_dim

    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#
    def _validate_inputs(
        self,
        points: np.ndarray,
        weights_mean: np.ndarray,
        weights_cov: np.ndarray
    ) -> None:
        """
        Validate sigma points and weights.

        Parameters
        ----------
        points : np.ndarray
            Sigma points array.
        weights_mean : np.ndarray
            Mean weights.
        weights_cov : np.ndarray
            Covariance weights.

        Raises
        ------
        ValueError
            If dimensions are inconsistent or weights invalid.
        """
        if points.ndim != 2:
            raise ValueError("Points must be a 2D array.")

        n_points = points.shape[0]
        
        if len(weights_mean) != n_points or len(weights_cov) != n_points:
            raise ValueError(
                f"Weight arrays must have length {n_points} to match number of points."
            )

        if not np.isclose(np.sum(weights_mean), 1.0):
            raise ValueError("Mean weights must sum to 1.")

    def to_array(self) -> np.ndarray:
        """
        Convert sigma points to numpy array.

        Returns
        -------
        np.ndarray
            The sigma points array.
        """
        return self._points

    def get_mean(self) -> np.ndarray:
        """
        Compute weighted mean of sigma points.

        Returns
        -------
        np.ndarray
            Mean state vector.
        """
        return np.sum(self._weights_mean[:, np.newaxis] * self._points, axis=0)

    def get_covariance(self) -> np.ndarray:
        """
        Compute weighted covariance from sigma points.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        mean = self.get_mean()
        deviations = self._points - mean
        return np.sum(
            self._weights_cov[:, np.newaxis, np.newaxis] * 
            (deviations[:, :, np.newaxis] * deviations[:, np.newaxis, :]),
            axis=0
        )

    def get_standard_deviations(self) -> np.ndarray:
        """
        Get standard deviations from sigma points.

        Returns
        -------
        np.ndarray
            Array of standard deviations.
        """
        cov = self.get_covariance()
        return np.sqrt(np.diag(cov))

    def to_covariance_matrix(self) -> 'CovarianceMatrix':
        """
        Convert sigma points to a CovarianceMatrix object.

        Returns
        -------
        CovarianceMatrix
            Equivalent covariance matrix representation.
        """
        cov = self.get_covariance()
        
        # Convert to list of lists with ArrayWUnits
        matrix_with_units = [
            [ArrayWUnits(cov[i, j], None) for j in range(self._n_dim)]
            for i in range(self._n_dim)
        ]
        
        return CovarianceMatrix(
            matrix_with_units,
            self._epoch,
            self._frame,
            definition=self._definition
        )

    # endregion Methods #
    #-------------------#


#==============================#
#   CLASS: MONTE CARLO         #
#==============================#
class MonteCarloSamples(UncertaintyQuantification):
    """
    Represents uncertainty using Monte Carlo samples.

    Used for uncertainty propagation through random sampling, providing
    a non-parametric representation of the probability distribution.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples, shape (n_samples, n_dimensions).
    epoch : int
        The epoch of the samples.
    frame : str, optional
        The reference frame. Defaults to 'J2000'.
    definition : optional
        Sequence or state definition. Defaults to None.

    Raises
    ------
    ValueError
        If samples array is not 2D.
    """

    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(
        self,
        samples: np.ndarray,
        epoch: int,
        frame: str = 'J2000',
        definition=None
    ):
        """Initialize a MonteCarloSamples object."""
        super().__init__(epoch, frame, definition)
        
        if samples.ndim != 2:
            raise ValueError("Samples must be a 2D array.")
        
        self._samples = samples
        self._n_samples, self._n_dim = samples.shape

    #----------------------#
    # region    Properties #
    #----------------------#
    @property
    def samples(self) -> np.ndarray:
        """The Monte Carlo samples array."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._n_samples

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""
        return self._n_dim

    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#
    def to_array(self) -> np.ndarray:
        """
        Convert samples to numpy array.

        Returns
        -------
        np.ndarray
            The samples array.
        """
        return self._samples

    def get_mean(self) -> np.ndarray:
        """
        Compute mean of samples.

        Returns
        -------
        np.ndarray
            Mean state vector.
        """
        return np.mean(self._samples, axis=0)

    def get_covariance(self) -> np.ndarray:
        """
        Compute sample covariance.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        return np.cov(self._samples, rowvar=False)

    def get_standard_deviations(self) -> np.ndarray:
        """
        Get standard deviations from samples.

        Returns
        -------
        np.ndarray
            Array of standard deviations.
        """
        return np.std(self._samples, axis=0)

    def to_covariance_matrix(self) -> CovarianceMatrix:
        """
        Convert samples to a CovarianceMatrix object.

        Returns
        -------
        CovarianceMatrix
            Equivalent covariance matrix representation.
        """
        cov = self.get_covariance()
        
        # Convert to list of lists with ArrayWUnits
        matrix_with_units = [
            [ArrayWUnits(cov[i, j], None) for j in range(self._n_dim)]
            for i in range(self._n_dim)
        ]
        
        return CovarianceMatrix(
            matrix_with_units,
            self._epoch,
            self._frame,
            definition=self._definition
        )

    def get_percentile(self, percentile: float, axis: int = None) -> np.ndarray:
        """
        Compute percentile of samples.

        Parameters
        ----------
        percentile : float
            Percentile to compute (0-100).
        axis : int, optional
            Axis along which to compute. If None, compute for each dimension.

        Returns
        -------
        np.ndarray
            Percentile values.
        """
        if axis is None:
            return np.percentile(self._samples, percentile, axis=0)
        return np.percentile(self._samples, percentile, axis=axis)

    # endregion Methods #
    #-------------------#
'''