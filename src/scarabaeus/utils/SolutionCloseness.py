# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus.utils.NumpyWrapper as np


# --------------------#
#  Class Definition  #
# --------------------#
class SolutionCloseness:
    """Statistical distance between orbit determination solutions.

    Two OD solutions of the same object rarely agree exactly, and the useful
    question is not how far apart the estimates are but whether they are
    *consistent*: whether the difference is small compared with the uncertainty
    each solution claims. Comparing state differences alone answers neither —
    a 10 km difference is nothing for a solution with a 100 km covariance and
    damning for one with a 1 km covariance.

    The measure of Bhaskaran & Rush [1]_ answers it directly. Around each
    solution :math:`S_i = (c_i, P_i)` grow the :math:`t`-sigma ellipsoid

    .. math::

        (x - c_i)^T P_i^{-1} (x - c_i) = t^2

    and let both grow together. The closeness :math:`d(S_1, S_2)` is the
    smallest :math:`t` at which they first touch. It reads directly as a number
    of sigma: solutions that touch below :math:`1\\sigma` overlap comfortably,
    while a value of several sigma means the two cannot both be right.

    The measure is symmetric, scale free, and reduces to the intuitive answers
    in the simple cases — for equal covariances it is exactly half the
    Mahalanobis distance between the centres, and for isotropic covariances
    :math:`a^2 I` and :math:`b^2 I` it is :math:`\\|c_2 - c_1\\| / (a + b)`.

    Solutions come either as :class:`~scarabaeus.SolutionOD` objects, which are
    read for their state estimate and covariance at a chosen epoch, or as plain
    ``(mean, covariance)`` pairs.

    See Also
    --------
    scarabaeus.SolutionOD : Solution object read by :meth:`between`.

    Notes
    -----
    The tangency point satisfies
    :math:`\\Lambda_1 (x - c_1) = -\\alpha \\Lambda_2 (x - c_2)`, so it lies at

    .. math::

        x(\\alpha) = (\\Lambda_1 + \\alpha \\Lambda_2)^{-1}
                     (\\Lambda_1 c_1 + \\alpha \\Lambda_2 c_2)

    for a single unknown :math:`\\alpha > 0`, fixed by requiring the two
    Mahalanobis distances at that point to be equal. That difference runs
    monotonically from negative to positive as :math:`\\alpha` sweeps
    :math:`0 \\to \\infty` (the point slides from :math:`c_1` to :math:`c_2`),
    so the root is unique and bracketing finds it without a search grid.

    Both solutions must be expressed in the same frame and units; nothing here
    can detect otherwise, and comparing a J2000 solution against a body-fixed
    one produces a plausible number that means nothing.

    References
    ----------
    .. [1] Bhaskaran, S., & Rush, J. (2004). Navigation performance measures
       for orbit determination solution comparison. *AIAA/AAS Astrodynamics
       Specialist Conference*, AIAA 2004-4982.

    Examples
    --------
    .. code-block:: python

        # two filter solutions of the same arc
        result = SolutionCloseness.between(solution_a, solution_b)
        print(f"{result['d']:.2f} sigma apart")

        # position only, at the first epoch instead of the last
        result = SolutionCloseness.between(
            solution_a, solution_b, index=0, states=slice(0, 3)
        )

        # a whole family at once
        D = SolutionCloseness.matrix([sol_a, sol_b, sol_c], labels=["A", "B", "C"])
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class SolutionClosenessError(ValueError):
        """Raised when two solutions cannot be compared as posed."""

    # endregion Errors #
    # ------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    @classmethod
    def between(
        cls,
        first,
        second,
        index: int = -1,
        covariance: str = "est",
        states=None,
        max_alpha: float = 1e12,
    ) -> dict:
        """
        Closeness between two OD solutions, in sigma.

        Parameters
        ----------
        first, second : SolutionOD or tuple
            The two solutions. Either :class:`~scarabaeus.SolutionOD` objects,
            or ``(mean, covariance)`` pairs with the mean shaped ``(n,)`` and
            the covariance ``(n, n)``.
        index : int, optional
            Epoch row to compare at, for ``SolutionOD`` inputs. Defaults to
            ``-1``, the final estimate; ``0`` is the first epoch.
        covariance : str, optional
            Which stored covariance to read: ``'est'`` (default), ``'smooth'``
            or ``'consider'``.
        states : slice or sequence of int, optional
            Which state components to compare. Defaults to all of them; pass
            ``slice(0, 3)`` to compare position alone, which is often what a
            navigation team actually cares about.
        max_alpha : float, optional
            Bracketing limit for the tangency parameter. Only reached for
            solutions many orders of magnitude apart in scale.

        Returns
        -------
        dict
            ``d`` the closeness in sigma, ``t_sq`` its square, ``alpha`` the
            tangency parameter, ``x_tang`` the touching point, and
            ``mahalanobis`` the pair of distances from each centre to that
            point — equal to ``d`` on both sides, which is the check that the
            solve converged.

        Raises
        ------
        SolutionClosenessError
            If the shapes disagree, a covariance is singular, or the requested
            history is not stored on the solution.
        """
        c1, P1 = cls._as_mean_covariance(first, index, covariance, states, "first")
        c2, P2 = cls._as_mean_covariance(second, index, covariance, states, "second")

        if c1.shape != c2.shape:
            raise cls.SolutionClosenessError(
                f"the two solutions have different state sizes: {c1.shape[0]} "
                f"and {c2.shape[0]}. Use 'states' to compare a common subset."
            )

        delta = c2 - c1
        if not np.any(delta):
            return {
                "d": 0.0,
                "t_sq": 0.0,
                "alpha": None,
                "x_tang": c1.copy(),
                "mahalanobis": (0.0, 0.0),
            }

        lambda_1 = cls._inverse(P1, "first")
        lambda_2 = cls._inverse(P2, "second")

        alpha = cls._tangency_alpha(c1, lambda_1, c2, lambda_2, max_alpha)
        x_tang = cls._tangency_point(alpha, c1, lambda_1, c2, lambda_2)

        m1 = float((x_tang - c1) @ lambda_1 @ (x_tang - c1))
        m2 = float((x_tang - c2) @ lambda_2 @ (x_tang - c2))
        t_sq = 0.5 * (m1 + m2)

        return {
            "d": float(np.sqrt(max(t_sq, 0.0))),
            "t_sq": float(t_sq),
            "alpha": float(alpha),
            "x_tang": x_tang,
            "mahalanobis": (float(np.sqrt(max(m1, 0.0))), float(np.sqrt(max(m2, 0.0)))),
        }

    @classmethod
    def matrix(cls, solutions, **kwargs) -> np.ndarray:
        """
        Pairwise closeness between every solution in a set.

        The off-diagonal entries are what tell a family of solutions apart: a
        tight block means the solutions agree within their own uncertainty,
        while an isolated row is a solution that disagrees with all the others.

        Parameters
        ----------
        solutions : sequence
            Solutions in any form :meth:`between` accepts.
        **kwargs
            Passed through to :meth:`between`.

        Returns
        -------
        np.ndarray
            Symmetric ``(N, N)`` matrix of closeness values in sigma, zero on
            the diagonal.
        """
        count = len(solutions)
        distances = np.zeros((count, count))

        for i in range(count):
            for j in range(i + 1, count):
                value = cls.between(solutions[i], solutions[j], **kwargs)["d"]
                distances[i, j] = distances[j, i] = value

        return distances

    @classmethod
    def plot_matrix(
        cls,
        solutions,
        labels=None,
        scheme: str = "sigma",
        title: str = "OD solution consistency",
        show_values: bool = True,
        distances=None,
        **kwargs,
    ):
        """
        Plot the pairwise closeness of a family of solutions.

        Parameters
        ----------
        solutions : sequence
            Solutions in any form :meth:`between` accepts.
        labels : sequence of str, optional
            Axis labels; defaults to ``S1 ... SN``.
        scheme : str, optional
            Colour banding: ``'sigma'`` (1/2/3-sigma bands, the default),
            ``'binary'`` (consistent or not, split at 0.8), or ``'continuous'``
            for a smooth colour bar.
        title : str, optional
            Figure title.
        show_values : bool, optional
            Print each value in its cell. Defaults to ``True``.
        distances : np.ndarray, optional
            A previously computed matrix, to avoid recomputing it.
        **kwargs
            Passed through to :meth:`matrix`.

        Returns
        -------
        distances, fig : np.ndarray and matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import BoundaryNorm, ListedColormap

        if distances is None:
            distances = cls.matrix(solutions, **kwargs)
        count = len(distances)
        if labels is None:
            labels = [f"S{i + 1}" for i in range(count)]

        bands = {
            "sigma": (
                [0.0, 1.0, 2.0, 3.0, np.inf],
                ["#1a9e74", "#6cc9a9", "#d95f30", "#8b1a1a"],
                ["< 1 sigma", "1 - 2", "2 - 3", "> 3"],
            ),
            "binary": (
                [0.0, 0.8, np.inf],
                ["#3a7abf", "#d95f30"],
                ["consistent (<= 0.8)", "inconsistent (> 0.8)"],
            ),
        }

        figure_size = count * 0.62 + 2.6
        fig, ax = plt.subplots(figsize=(figure_size + 1.2, figure_size))

        # only the upper triangle carries information; the matrix is symmetric
        upper_only = np.ma.masked_where(
            np.tril(np.ones_like(distances, dtype=bool)), distances
        )

        if scheme in bands:
            bounds, colours, band_labels = bands[scheme]
            colour_map = ListedColormap(colours)
            colour_map.set_bad(color="#f4f4f4")
            ax.imshow(
                upper_only,
                cmap=colour_map,
                norm=BoundaryNorm(bounds, colour_map.N),
                aspect="equal",
            )
            ax.legend(
                handles=[
                    mpatches.Patch(facecolor=c, edgecolor="#999", label=lab)
                    for c, lab in zip(colours, band_labels)
                ],
                title="closeness",
                title_fontsize=8,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
            )
        else:
            colour_map = plt.cm.RdYlGn_r.copy()
            colour_map.set_bad(color="#f4f4f4")
            image = ax.imshow(
                upper_only,
                cmap=colour_map,
                vmin=0.0,
                vmax=max(float(np.max(distances)), 1.0),
                aspect="equal",
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="closeness [sigma]")

        for i in range(count):
            ax.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#d8d8d8", zorder=2)
            )

        if show_values:
            font_size = max(6, min(9, 90 // max(count, 1)))
            for i in range(count):
                for j in range(i + 1, count):
                    ax.text(
                        j,
                        i,
                        f"{distances[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        color="white",
                        fontweight="bold",
                        zorder=4,
                    )

        ax.set_xticks(range(count))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(count))
        ax.set_yticklabels(labels, fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for k in range(count + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.8, zorder=1)
            ax.axvline(k - 0.5, color="white", linewidth=0.8, zorder=1)

        upper = distances[np.triu_indices(count, k=1)]
        ax.set_title(
            f"{title}\nmin {np.min(upper):.2f} | mean {np.mean(upper):.2f} | "
            f"max {np.max(upper):.2f} sigma",
            fontsize=10,
            pad=10,
        )
        fig.tight_layout()

        return distances, fig

    # endregion Methods #
    # ----------------#

    # ------------------#
    # region Internals #
    # ------------------#
    @classmethod
    def _tangency_alpha(cls, c1, lambda_1, c2, lambda_2, max_alpha) -> float:
        """
        Solve for the tangency parameter.

        The two Mahalanobis distances at :math:`x(\\alpha)` cross exactly once
        as ``alpha`` sweeps zero to infinity, because the point slides
        monotonically from one centre to the other. Bracketing that crossing
        and bisecting is enough — no search grid, and no root to miss.

        Parameters
        ----------
        c1, c2 : np.ndarray
            Solution means.
        lambda_1, lambda_2 : np.ndarray
            Inverse covariances.
        max_alpha : float
            Upper limit on the bracket expansion.

        Returns
        -------
        float
            The tangency parameter.

        Raises
        ------
        SolutionClosenessError
            If no crossing can be bracketed below ``max_alpha``.
        """

        def imbalance(alpha: float) -> float:
            x = cls._tangency_point(alpha, c1, lambda_1, c2, lambda_2)
            return float(
                (x - c1) @ lambda_1 @ (x - c1) - (x - c2) @ lambda_2 @ (x - c2)
            )

        # expand outwards from unity until the crossing is bracketed
        low, high = 1.0, 1.0
        low_value, high_value = imbalance(low), imbalance(high)

        while low_value > 0.0 and low > 1.0 / max_alpha:
            low /= 4.0
            low_value = imbalance(low)
        while high_value < 0.0 and high < max_alpha:
            high *= 4.0
            high_value = imbalance(high)

        if low_value > 0.0 or high_value < 0.0:
            raise cls.SolutionClosenessError(
                "could not bracket the tangency point between alpha = "
                f"{1.0 / max_alpha:g} and {max_alpha:g}. The two covariances "
                "differ by an extreme factor; check that both solutions are in "
                "the same units."
            )

        for _ in range(200):
            middle = np.sqrt(low * high)  # geometric: alpha spans many decades
            value = imbalance(middle)
            if value == 0.0 or (high / low) < 1.0 + 1e-14:
                break
            if value < 0.0:
                low = middle
            else:
                high = middle

        return float(np.sqrt(low * high))

    @staticmethod
    def _tangency_point(alpha, c1, lambda_1, c2, lambda_2) -> np.ndarray:
        """
        Point where the two scaled ellipsoids touch, for a given alpha.

        Parameters
        ----------
        alpha : float
            Tangency parameter.
        c1, c2 : np.ndarray
            Solution means.
        lambda_1, lambda_2 : np.ndarray
            Inverse covariances.

        Returns
        -------
        np.ndarray
            The touching point.
        """
        return np.linalg.solve(
            lambda_1 + alpha * lambda_2, lambda_1 @ c1 + alpha * (lambda_2 @ c2)
        )

    @classmethod
    def _inverse(cls, covariance, which: str) -> np.ndarray:
        """
        Invert a covariance, reporting which solution failed.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix.
        which : str
            Label used in the error message.

        Returns
        -------
        np.ndarray
            The inverse covariance.

        Raises
        ------
        SolutionClosenessError
            If the matrix is singular.
        """
        try:
            return np.linalg.inv(covariance)
        except np.linalg.LinAlgError as exc:
            raise cls.SolutionClosenessError(
                f"the {which} solution's covariance is singular, so its "
                "uncertainty ellipsoid is degenerate. Compare a subset of the "
                "states with 'states=' if some are unobservable."
            ) from exc

    @classmethod
    def _as_mean_covariance(cls, solution, index, covariance, states, which: str):
        """
        Reduce a solution to the ``(mean, covariance)`` pair to compare.

        Parameters
        ----------
        solution : SolutionOD or tuple
            The solution.
        index : int
            Epoch row, for ``SolutionOD`` inputs.
        covariance : str
            Which stored covariance to read.
        states : slice or sequence of int or None
            State components to keep.
        which : str
            Label used in error messages.

        Returns
        -------
        mean, covariance : np.ndarray

        Raises
        ------
        SolutionClosenessError
            If the solution does not carry what was asked for.
        """
        if isinstance(solution, (tuple, list)) and len(solution) == 2:
            mean = np.asarray(solution[0], dtype=float).ravel()
            matrix = np.asarray(solution[1], dtype=float)
        else:
            mean, matrix = cls._from_solution_od(solution, index, covariance, which)

        mean = np.asarray(mean, dtype=float).ravel()
        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape != (mean.size, mean.size):
            raise cls.SolutionClosenessError(
                f"the {which} solution's covariance is {matrix.shape}, which "
                f"does not match its {mean.size}-element mean."
            )

        if states is not None:
            index_array = (
                np.arange(mean.size)[states]
                if isinstance(states, slice)
                else np.asarray(states, dtype=int)
            )
            mean = mean[index_array]
            matrix = matrix[np.ix_(index_array, index_array)]

        return mean, matrix

    @classmethod
    def _from_solution_od(cls, solution, index, covariance, which: str):
        """
        Read a state estimate and covariance out of a ``SolutionOD``.

        The covariance history runs on the filter timeline while the state
        estimate runs on the measurement epochs; for a sequence run those grids
        differ in length, so the state row is matched by epoch rather than by
        position.

        Parameters
        ----------
        solution : SolutionOD
            Solution to read.
        index : int
            Row of the covariance history.
        covariance : str
            ``'est'``, ``'smooth'`` or ``'consider'``.
        which : str
            Label used in error messages.

        Returns
        -------
        mean, covariance : np.ndarray

        Raises
        ------
        SolutionClosenessError
            If the requested history is not stored.
        """
        attribute = {
            "est": "covariance_est",
            "smooth": "covariance_smooth",
            "consider": "covariance_consider",
        }.get(covariance)
        if attribute is None:
            raise cls.SolutionClosenessError(
                f"unknown covariance '{covariance}'; choose 'est', 'smooth' or "
                "'consider'."
            )

        history = getattr(solution, attribute, None)
        if history is None or len(history) == 0:
            raise cls.SolutionClosenessError(
                f"the {which} solution has no '{covariance}' covariance stored. "
                "Enable it through OutputSettings before running the filter."
            )
        history = np.asarray(history, dtype=float)
        matrix = history[index]

        states = getattr(solution, "state_est", None)
        if states is None or len(states) == 0:
            raise cls.SolutionClosenessError(
                f"the {which} solution has no state estimate stored, so there "
                "is nothing to compare. A covariance-analysis run carries "
                "uncertainty but no estimate; pass (mean, covariance) pairs "
                "instead."
            )
        states = np.asarray(states, dtype=float)

        if len(states) == len(history):
            return states[index], matrix

        # Grids differ (sequence run): match the state row by epoch.
        timeline = getattr(solution, "timeline", None)
        timestamps = getattr(solution, "timestamps", None)
        if timeline is None or timestamps is None:
            return states[index], matrix

        epoch = np.asarray(timeline, dtype=float).ravel()[index]
        nearest = int(
            np.argmin(np.abs(np.asarray(timestamps, dtype=float).ravel() - epoch))
        )
        return states[nearest], matrix

    # endregion Internals #
    # ------------------#
