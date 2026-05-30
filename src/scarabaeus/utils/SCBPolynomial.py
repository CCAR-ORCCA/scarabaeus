# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus.utils.NumpyWrapper as np


# ----------------------#
#   Class Definition    #
# ----------------------#
class SCBPolynomial:
    """Lightweight polynomial wrapper for time-varying scalar profiles.

    Stores coefficients in ascending order (lowest to highest degree) and
    optionally enforces a validity domain :math:`[t_0, t_f]`.  Designed as a
    drop-in replacement for ``numpy.polynomial.Polynomial`` without its
    autograd incompatibilities.

    Given coefficients :math:`[a_0, a_1, \\ldots, a_n]`, the polynomial is

    .. math::

        p(t) = a_0 + a_1\\,t + a_2\\,t^2 + \\cdots + a_n\\,t^n
             = \\sum_{k=0}^{n} a_k\\,t^k

    Parameters
    ----------
    coefficients : array-like
        Polynomial coefficients ``[a0, a1, …, an]`` in ascending degree order.
    domain : tuple of float, optional
        ``(t0, tf)`` bounding the valid evaluation range.  Evaluation outside
        this range raises ``ValueError``.

    See Also
    --------
    numpy.polynomial.Polynomial : Standard NumPy polynomial (autograd-incompatible).

    References
    ----------
    Press, W. H.; Teukolsky, S. A.; Vetterling, W. T.; Flannery, B. P. (2007).
    *Numerical Recipes: The Art of Scientific Computing* (3rd ed.).
    Cambridge University Press.
    ISBN 978-0521880688.
    """

    def __init__(self, coefficients: np.ndarray, domain: tuple = None):
        """Initialize a polynomial wrapper.

        Parameters
        ----------
        coefficients : array-like
            Coefficients :math:`[a_0, a_1, \\ldots, a_n]` in ascending degree
            order (i.e. ``coefficients[k]`` multiplies :math:`t^k`).
        domain : tuple of float, optional
            ``(t0, tf)`` validity interval.  Evaluation outside this range
            raises ``ValueError``.
        """
        self.coefficients = np.array(coefficients)
        self._degree = len(coefficients) - 1
        self.domain = domain

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate :math:`p(t) = \\sum_{k=0}^{n} a_k\\,t^k`.

        Parameters
        ----------
        t : float or numpy.ndarray
            Scalar or array of evaluation points.  If *domain* was set,
            all values must satisfy :math:`t_0 \\le t \\le t_f`.

        Returns
        -------
        result : float or numpy.ndarray
            Polynomial value(s) at *t*.  Returns a scalar when *t* is scalar.

        Raises
        ------
        ValueError
            If any element of *t* falls outside the polynomial domain.
        """
        t = np.atleast_1d(t)
        if self.domain is not None:
            t_lo = min(self.domain)
            t_hi = max(self.domain)
            if np.any((t < t_lo) | (t > t_hi)):
                raise ValueError(f"Evaluation outside polynomial domain {self.domain}")

        result = np.polyval(self.coefficients[::-1], t)
        return result.squeeze() if result.size > 1 else result.item()

    # ----------------#
    # region Methods #
    # ----------------#
    @classmethod
    def fit(
        cls, t_array: np.ndarray, y_array: np.ndarray, deg: int, domain: tuple = None
    ):
        """
        Fit a polynomial of degree `deg` to the given data.

        Parameters
        ----------
        t_array : np.ndarray
            Independent variable (time).
        y_array : np.ndarray
            Dependent variable (e.g., mass values).
        deg : int
            Degree of the polynomial to fit.
        domain : tuple, optional
            Domain [t0, tf] for the polynomial.

        Returns
        -------
        SCBPolynomial
            Fitted polynomial instance.
        """
        coeffs = np.polyfit(t_array, y_array, deg)
        coeffs = coeffs[::-1]  # reverse to match increasing order
        return cls(coeffs, domain=domain)

    @classmethod
    def from_fit(
        cls, t_array: np.ndarray, y_array: np.ndarray, deg: int, domain: tuple = None
    ):
        """
        Alias for :meth:`fit`.

        Parameters
        ----------
        t_array : np.ndarray
            Independent variable (time).
        y_array : np.ndarray
            Dependent variable values to fit.
        deg : int
            Degree of the polynomial to fit.
        domain : tuple, optional
            Domain ``[t0, tf]`` for the polynomial.

        Returns
        -------
        SCBPolynomial
            Fitted polynomial instance.
        """
        return cls.fit(t_array, y_array, deg, domain)

    def convert(self, domain: tuple):
        """Return a copy of this polynomial with a new validity domain.

        Parameters
        ----------
        domain : tuple of float
            New ``(t0, tf)`` interval to enforce.

        Returns
        -------
        poly : SCBPolynomial
            A shallow copy sharing the same coefficients but with *domain*
            replacing the original.
        """
        return SCBPolynomial(self.coefficients.copy(), domain=domain)

    def derivative(self):
        """Return the first derivative :math:`p'(t)` as a new polynomial.

        Applies the power rule coefficient-wise:

        .. math::

            p'(t) = a_1 + 2\\,a_2\\,t + \\cdots + n\\,a_n\\,t^{n-1}
                  = \\sum_{k=1}^{n} k\\,a_k\\,t^{k-1}

        Returns
        -------
        dpoly : SCBPolynomial
            Degree-:math:`(n-1)` polynomial representing :math:`p'(t)`,
            inheriting the same *domain*.
        """
        coeffs = [i * c for i, c in enumerate(self.coefficients)][1:]
        return SCBPolynomial(coeffs, domain=self.domain)

    def integrate(self, a: float, b: float) -> float:
        """Compute the definite integral of :math:`p(t)` over :math:`[a, b]`.

        Constructs the antiderivative

        .. math::

            P(t) = \\sum_{k=0}^{n} \\frac{a_k}{k+1}\\,t^{k+1}

        and returns :math:`P(b) - P(a)`.

        Parameters
        ----------
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.

        Returns
        -------
        integral : float
            Value of :math:`\\int_a^b p(t)\\,\\mathrm{d}t`.
        """
        antideriv_coeffs = [c / (i + 1) for i, c in enumerate(self.coefficients)]
        antideriv = SCBPolynomial(antideriv_coeffs)
        return antideriv(b) - antideriv(a)

    @property
    def deg(self) -> int:
        """Degree :math:`n` of the polynomial (number of coefficients minus one)."""
        return self._degree

    def __repr__(self) -> str:
        """Return a concise string showing coefficients and domain."""
        return (
            f"SCBPolynomial(coeffs={self.coefficients.tolist()}, domain={self.domain})"
        )
