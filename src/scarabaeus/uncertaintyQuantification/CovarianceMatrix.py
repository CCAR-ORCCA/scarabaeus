# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from typing import Optional
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import numpy as np
from scarabaeus import ArrayWUnits, UncertaintyQuantification


# --------------------#
#  Class Definition  #
# --------------------#
class CovarianceMatrix(UncertaintyQuantification):
    """
    Covariance matrix for uncertainty quantification in orbit determination.

    The covariance matrix :math:`P` is the central object describing the uncertainty
    of the estimated state vector :math:`\\mathbf{x} \\in \\mathbb{R}^n`.  It is
    defined as the expected outer product of the estimation error
    :math:`\\boldsymbol{\\epsilon} = \\mathbf{x} - \\hat{\\mathbf{x}}`:

    .. math::

        P = E\\!\\left[\\boldsymbol{\\epsilon}\\,\\boldsymbol{\\epsilon}^T\\right]
          = \\begin{bmatrix}
              \\sigma_1^2      & P_{12} & \\cdots & P_{1n} \\\\
              P_{21}          & \\sigma_2^2 & \\cdots & P_{2n} \\\\
              \\vdots          & \\vdots     & \\ddots & \\vdots \\\\
              P_{n1}          & P_{n2} & \\cdots & \\sigma_n^2
            \\end{bmatrix}

    where :math:`\\sigma_i = \\sqrt{P_{ii}}` is the standard deviation of the
    :math:`i`-th state component and :math:`P_{ij}` is the covariance between
    components :math:`i` and :math:`j`.

    **Fundamental properties**

    * **Symmetry**: :math:`P = P^T`, i.e. :math:`P_{ij} = P_{ji}`.
    * **Positive semi-definiteness**: :math:`\\mathbf{v}^T P \\mathbf{v} \\geq 0`
      for all :math:`\\mathbf{v} \\in \\mathbb{R}^n`.
    * **Correlation coefficient**: the normalised off-diagonal element is

      .. math::

          \\rho_{ij} = \\frac{P_{ij}}{\\sigma_i \\sigma_j}, \\qquad \\rho_{ij} \\in [-1, 1]

    **Role in the OD filter**

    In the batch least-squares and sequential filters implemented in Scarabaeus,
    :math:`P` is propagated and updated according to:

    * *Propagation* (time update):

      .. math::

          P^- = \\Phi\\, P^+\\, \\Phi^T + Q

      where :math:`\\Phi` is the state-transition matrix (STM) and :math:`Q` is
      the process-noise covariance.

    * *Measurement update* (information filter form):

      .. math::

          P^+ = \\left(P^{-\\,-1} + H^T R^{-1} H\\right)^{-1}

      where :math:`H` is the measurement sensitivity (partial-derivative) matrix and
      :math:`R` is the measurement noise covariance.

    * *Batch least-squares normal equations*:

      .. math::

          P = \\left(\\Lambda_0^{-1} + \\sum_k H_k^T R_k^{-1} H_k\\right)^{-1}

      where :math:`\\Lambda_0` is the a-priori information matrix.

    **Construction paths**

    +---------------------------+--------------------------------------------------+
    | ``from_list=False``       | Provide a full :math:`n \\times n` matrix directly. |
    +---------------------------+--------------------------------------------------+
    | ``from_list=True``        | Provide a flat list of :math:`n` standard         |
    | (diagonal build)          | deviations :math:`\\sigma_i`; optionally supply   |
    |                           | ``corr_factors`` for off-diagonal terms.          |
    +---------------------------+--------------------------------------------------+

    Parameters
    ----------
    row_list : list[list]
        Full matrix data (list of rows) when ``from_list=False``, or flat list
        of standard deviations :math:`[\\sigma_1, \\ldots, \\sigma_n]` when
        ``from_list=True``.
    epoch : int
        J2000 ephemeris time (ET seconds) tagging this covariance.
    frame : str, optional
        Reference frame in which the position/velocity components of :math:`P`
        are expressed.  Defaults to ``'J2000'``.
    from_list : bool, optional
        If ``True``, build a diagonal (or correlated) covariance from
        ``row_list`` treated as a sigma list.  Defaults to ``False``.
    corr_factors : list, optional
        Upper-triangular correlation coefficients :math:`\\rho_{ij}` packed
        row-by-row (length :math:`n(n-1)/2`).  Only used when
        ``from_list=True``.  Defaults to ``None`` (identity correlation,
        i.e. diagonal :math:`P`).
    extra_apriori_sigmas : dict[str, ArrayWUnits], optional
        Additional a-priori standard deviations for parameters that enter
        in multi-arc / sequence scenarios but are not part of the current
        arc state.  Keys are parameter names.  Defaults to ``None``.
    definition : optional
        State or sequence definition object.  Defaults to ``None``.

    Raises
    ------
    ValueError
        If any value in ``extra_apriori_sigmas`` is not an ``ArrayWUnits``.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        row_list: list[list],
        epoch: int,
        frame: str = "J2000",
        from_list: bool = False,
        corr_factors: Optional[list] = None,
        extra_apriori_sigmas: Optional[dict[str, ArrayWUnits]] = None,
        definition=None,
    ):
        """
        Build a covariance matrix from a row-list or a diagonal-sigma list.

        Delegates to :meth:`_initialize_matrix` for the actual construction.
        If *extra_apriori_sigmas* is provided, the expanded size ``max_n`` is
        recorded for sequence-scenario bookkeeping.

        Parameters
        ----------
        row_list : list[list]
            Full matrix data (list of rows), or flat diagonal standard
            deviations when *from_list* is ``True``.
        epoch : int
            J2000 ephemeris time (ET seconds).
        frame : str, optional
            Reference frame name.  Defaults to ``'J2000'``.
        from_list : bool, optional
            If ``True``, treat *row_list* as a flat list of standard
            deviations and build a diagonal covariance.  Defaults to
            ``False``.
        corr_factors : list, optional
            Off-diagonal correlation coefficients applied when *from_list*
            is ``True``.
        extra_apriori_sigmas : dict[str, ArrayWUnits], optional
            Additional standard deviations for sequence-estimation scenarios;
            keys are parameter names.
        definition : optional
            State or sequence definition object.

        Raises
        ------
        ValueError
            If any value in *extra_apriori_sigmas* is not an ``ArrayWUnits``.
        """
        super().__init__(epoch, frame, definition)

        self.extra_apriori_sigmas = extra_apriori_sigmas or {}

        self._validate_sequence_sigmas(extra_apriori_sigmas)
        self._initialize_matrix(row_list, from_list, corr_factors)

        if extra_apriori_sigmas is not None:
            self.max_n = len(extra_apriori_sigmas) + len(row_list)

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def matrix(self) -> list[list[ArrayWUnits]]:
        """The covariance matrix as a list of lists."""
        return self._matrix

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _validate_sequence_sigmas(
        self, extra_apriori_sigmas: Optional[dict[str, ArrayWUnits]]
    ) -> None:
        """
        Validate that extra_apriori_sigmas contains only ArrayWUnits objects.

        Parameters
        ----------
        extra_apriori_sigmas : dict[str, ArrayWUnits] or None
            Dictionary mapping parameter names to standard deviations for sequence scenarios.

        Raises
        ------
        ValueError
            If any element is not an ArrayWUnits object.
        """
        if extra_apriori_sigmas is not None:
            if not isinstance(extra_apriori_sigmas, dict):
                raise ValueError(
                    "extra_apriori_sigmas must be a dict mapping parameter names to ArrayWUnits."
                )
            if not all(
                isinstance(v, ArrayWUnits) for v in extra_apriori_sigmas.values()
            ):
                raise ValueError(
                    "All values in extra_apriori_sigmas must be ArrayWUnits objects."
                )

    def _initialize_matrix(
        self, row_list: list[list], from_list: bool, corr_factors: Optional[list]
    ) -> None:
        """
        Initialize the covariance matrix from input data.

        Parameters
        ----------
        row_list : list[list]
            Matrix data or diagonal elements.
        from_list : bool
            Whether to construct from diagonal elements.
        corr_factors : list or None
            Correlation factors for construction.
        """
        if from_list:
            corr_matrix = (
                self._build_correlation_matrix(row_list, corr_factors)
                if corr_factors is not None
                else np.eye(len(row_list))
            )
            self._build_from_diagonal(row_list, corr_matrix)
        else:
            self._matrix = [list(row) for row in row_list]

    def to_array(self) -> np.ndarray:
        """
        Convert covariance matrix to numpy array without units.

        Returns
        -------
        np.ndarray
            The covariance matrix as a numpy array without units.
        """
        return np.array([[elem.values for elem in row] for row in self._matrix])

    def matrix_without_units(self) -> np.ndarray:
        """
        Extract values from the covariance matrix without units.

        Returns
        -------
        np.ndarray
            The covariance matrix as a numpy array without units.
        """
        return self.to_array()

    def get_standard_deviations(self) -> np.ndarray:
        """
        Return the 1-sigma uncertainties of each state component.

        The standard deviation of the :math:`i`-th component is the square
        root of the corresponding diagonal element:

        .. math::

            \\sigma_i = \\sqrt{P_{ii}}, \\qquad i = 1, \\ldots, n

        These values are the formal uncertainties (1-sigma) reported by the
        filter and are commonly used for residual editing and solution assessment.

        Returns
        -------
        np.ndarray
            1-D array of length :math:`n` containing :math:`\\sigma_i`.
        """
        cov_array = self.to_array()
        return np.sqrt(np.diag(cov_array))

    def _build_from_diagonal(
        self, sigmas: list[ArrayWUnits], corr_matrix: np.ndarray
    ) -> None:
        """
        Construct a covariance matrix from standard deviations and a correlation matrix.

        Each element of :math:`P` is assembled as:

        .. math::

            P_{ij} =
            \\begin{cases}
                \\sigma_i^2           & i = j \\\\
                \\rho_{ij}\\,\\sigma_i\\,\\sigma_j & i \\neq j
            \\end{cases}

        where :math:`\\sigma_i` is the standard deviation of component :math:`i` and
        :math:`\\rho_{ij} \\in [-1, 1]` is the correlation coefficient between
        components :math:`i` and :math:`j` supplied via ``corr_matrix``.

        The resulting matrix is symmetric and positive semi-definite provided that
        ``corr_matrix`` is itself a valid correlation matrix.

        Parameters
        ----------
        sigmas : list[ArrayWUnits]
            Standard deviations :math:`\\sigma_1, \\ldots, \\sigma_n`.
        corr_matrix : np.ndarray
            :math:`n \\times n` correlation matrix :math:`R` with
            :math:`R_{ii} = 1` and :math:`R_{ij} = \\rho_{ij}`.
        """
        n = len(sigmas)
        self._matrix = [
            [
                sigmas[i] ** 2 if i == j else corr_matrix[i, j] * sigmas[i] * sigmas[j]
                for j in range(n)
            ]
            for i in range(n)
        ]

    def _build_correlation_matrix(
        self, diag_cov: list, upper_tri_vec: list
    ) -> np.ndarray:
        """
        Assemble a symmetric correlation matrix from its upper-triangular entries.

        The correlation matrix :math:`R \\in \\mathbb{R}^{n \\times n}` is built as:

        .. math::

            R_{ij} =
            \\begin{cases}
                1           & i = j \\\\
                \\rho_{ij}  & i < j \\quad (\\text{from } \\mathtt{upper\\_tri\\_vec}) \\\\
                \\rho_{ji}  & i > j \\quad (\\text{symmetry})
            \\end{cases}

        The upper-triangular entries are packed row-by-row, so for an
        :math:`n \\times n` matrix the expected length of ``upper_tri_vec`` is

        .. math::

            \\frac{n(n-1)}{2}

        For example, for :math:`n = 3` the vector is
        :math:`[\\rho_{12},\\, \\rho_{13},\\, \\rho_{23}]`.

        Parameters
        ----------
        diag_cov : list
            Diagonal elements (used only to infer :math:`n`; values are ignored).
        upper_tri_vec : list
            Correlation coefficients :math:`\\rho_{ij}` for :math:`i < j`,
            packed row-by-row.  Each element must be an ``ArrayWUnits``
            with a scalar ``.values`` attribute.

        Returns
        -------
        np.ndarray
            :math:`n \\times n` symmetric correlation matrix with unit diagonal.

        Raises
        ------
        ValueError
            If ``len(upper_tri_vec) != n(n-1)/2``.
        """
        n = len(diag_cov)
        expected_length = (n * (n - 1)) // 2

        if len(upper_tri_vec) != expected_length:
            raise ValueError(
                f"Correlation factors vector has length {len(upper_tri_vec)}, "
                f"but {expected_length} elements are required for a {n}x{n} matrix."
            )

        corr_matrix = np.eye(n)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                value = float(upper_tri_vec[idx].values)
                corr_matrix[i, j] = value
                corr_matrix[j, i] = value
                idx += 1

        return corr_matrix

    def reinitialize_with_matrix(
        self,
        new_matrix: np.ndarray,
        new_epoch: Optional[float] = None,
        new_frame: Optional[str] = None,
    ) -> Self:
        """
        Return a new ``CovarianceMatrix`` with updated numerical values.

        This is the standard way to create the *updated* covariance after a
        filter step.  Given the current (prior) covariance :math:`P^-` and the
        numerically computed updated array :math:`\\tilde{P}^+`, the typical
        call pattern is:

        .. code-block:: python

            P_plus = P_minus.reinitialize_with_matrix(P_plus_array, new_epoch=t_k)

        The unit metadata stored in the original matrix is re-attached to the
        new values wherever the sizes match; elements that fall outside the old
        size receive ``None`` units.

        Parameters
        ----------
        new_matrix : np.ndarray
            Square :math:`n \\times n` array of updated covariance values.
            Must satisfy :math:`n > 0` and ``new_matrix.shape[0] == new_matrix.shape[1]``.
        new_epoch : float, optional
            Ephemeris time (ET seconds) of the updated covariance.
            If ``None``, the current epoch is preserved.
        new_frame : str, optional
            Reference frame of the updated covariance.
            If ``None``, the current frame is preserved.

        Returns
        -------
        CovarianceMatrix
            A new instance carrying ``new_matrix`` with inherited units.

        Raises
        ------
        ValueError
            If ``new_matrix`` is not a ``numpy.ndarray`` or is not square.
        """
        if not isinstance(new_matrix, np.ndarray):
            raise ValueError("new_matrix must be a numpy.ndarray.")

        if new_matrix.shape[0] != new_matrix.shape[1]:
            raise ValueError("new_matrix must be square.")

        new_size = new_matrix.shape[0]
        matrix_with_units = self._apply_units_to_matrix(new_matrix, new_size)

        updated_epoch = new_epoch if new_epoch is not None else self._epoch
        updated_frame = new_frame if new_frame is not None else self._frame

        return CovarianceMatrix(matrix_with_units, updated_epoch, updated_frame)

    def _apply_units_to_matrix(
        self, new_matrix: np.ndarray, new_size: int
    ) -> list[list[ArrayWUnits]]:
        """
        Apply units from original matrix to new matrix values.

        Parameters
        ----------
        new_matrix : np.ndarray
            New matrix values.
        new_size : int
            Size of the new matrix.

        Returns
        -------
        list[list[ArrayWUnits]]
            Matrix with units applied.
        """
        if not hasattr(self, "_matrix"):
            return [
                [ArrayWUnits(new_matrix[i, j], None) for j in range(new_size)]
                for i in range(new_size)
            ]

        old_size = len(self._matrix)

        return [
            [
                ArrayWUnits(
                    new_matrix[i, j],
                    (
                        self._matrix[i][j].units
                        if i < old_size and j < len(self._matrix[i])
                        else None
                    ),
                )
                for j in range(new_size)
            ]
            for i in range(new_size)
        ]

    # Deprecated alias for backward compatibility
    def covariance_matrix_from_list(
        self, sigmas: list[ArrayWUnits], corr_matrix: np.ndarray
    ) -> None:
        """
        DEPRECATED: Use _build_from_diagonal instead.

        Constructs covariance matrix from diagonal elements and correlation matrix.
        """
        self._build_from_diagonal(sigmas, corr_matrix)

    # Deprecated alias for backward compatibility
    def vector_to_correlation_matrix(
        self, diag_cov: list, upper_tri_vec: list
    ) -> np.ndarray:
        """
        DEPRECATED: Use _build_correlation_matrix instead.

        Converts vectors into a correlation matrix.
        """
        return self._build_correlation_matrix(diag_cov, upper_tri_vec)

    # endregion Methods #
    # -------------------#
