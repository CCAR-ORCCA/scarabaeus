# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits

import scarabaeus.utils.NumpyWrapper as np

class Noise:
    """Generates AWGN noise for measurements and sensor data.

    Provides methods to draw independent samples from a univariate normal
    distribution with mean :math:`\\mu` and standard deviation :math:`\\sigma`,

    .. math::

        p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)

    returning unitless arrays, unit-carrying
    :class:`~scarabaeus.ArrayWUnits` objects, or noisy copies of existing
    datasets.

    See Also
    --------
    scarabaeus.Units : Unit system used to annotate noise samples.
    scarabaeus.ArrayWUnits : Array type that carries physical units.

    Notes
    -----
    All samples are drawn via ``numpy.random.normal``.  Seed the global
    NumPy random state before calling any method for reproducible results.
    The noisy output of :meth:`apply_AWGN` preserves the units of the input
    data.

    References
    ----------
    Grimmett, G.; Stirzaker, D. (2001).
    *Probability and Random Processes* (3rd ed.).
    Oxford University Press.
    ISBN 978-0198572220.
    """

    def __init__(self) -> None:
        """Initialize a Noise object."""
        pass

    def generate_AWGN(self, mu: float, sigma: float, count: int = 1) -> np.ndarray:
        """Generate an array of unitless Additive White Gaussian Noise samples.

        Each sample :math:`x_i \\sim \\mathcal{N}(\\mu,\\,\\sigma^2)` is drawn
        independently.

        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sigma : float
            Standard deviation of the normal distribution.
        count : int, optional
            Number of samples to generate.  Defaults to ``1``.

        Returns
        -------
        noise : numpy.ndarray
            Array of shape ``(count,)`` containing the noise samples.
        """
        awgn = np.random.normal(mu, sigma, count)

        return awgn

    def generate_AWGN_with_units(
        self, mu: float, sigma: float, units: Units, count: int = 1
    ) -> ArrayWUnits:
        """Generate an AWGN array annotated with physical units.

        Delegates sampling to :meth:`generate_AWGN` and wraps the result in
        an :class:`~scarabaeus.ArrayWUnits` object.

        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sigma : float
            Standard deviation of the normal distribution.
        units : Units
            Physical units to attach to the noise samples.
        count : int, optional
            Number of samples to generate.  Defaults to ``1``.

        Returns
        -------
        noise : ArrayWUnits
            Array of shape ``(count,)`` with the specified *units*.
        """
        awgn = self.generate_AWGN(mu=mu, sigma=sigma, count=count)

        awgn_with_units = ArrayWUnits(awgn, units)

        return awgn_with_units

    def apply_AWGN(
        self, data: ArrayWUnits, mu: float, sigma: float
    ) -> ArrayWUnits:
        """Apply AWGN to an existing dataset, preserving its physical units.

        Generates :math:`n = \\texttt{data.size}` noise samples
        :math:`\\epsilon_i \\sim \\mathcal{N}(\\mu,\\,\\sigma^2)` in the same
        units as *data*, then returns :math:`\\tilde{y}_i = y_i + \\epsilon_i`.

        Parameters
        ----------
        data : ArrayWUnits
            Original dataset to corrupt.
        mu : float
            Mean of the noise distribution (typically ``0``).
        sigma : float
            Standard deviation of the noise distribution.

        Returns
        -------
        noisy_data : ArrayWUnits
            Corrupted copy of *data* with units unchanged.
        """
        # Get length of data
        n = data.size
        # Get units of data
        u = data.units

        # Generate AWGN with units u
        awgn = self.generate_AWGN_with_units(mu=mu, sigma=sigma, units=u, count=n)

        # Add together two array with units objects
        noisy_data = data + awgn

        return noisy_data
