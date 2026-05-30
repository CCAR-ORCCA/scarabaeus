# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import warnings

import scarabaeus.utils.NumpyWrapper as np


# --------------------#
#  Class Definition  #
# --------------------#
class UtilsOD:
    """ Utility methods for Orbit Determination solution analysis and manipulation.

    All methods are static and operate on raw NumPy arrays or
    :class:`~scarabaeus.SolutionOD` objects.  No instantiation is required::

        d = scb.UtilsOD.closeness(c1, P1, c2, P2)['d']
        c_m, P_m = scb.UtilsOD.merge(c1, P1, c2, P2)

    See Also
    --------
    scarabaeus.SolutionOD : OD solution container.
    scarabaeus.Utils : General-purpose Scarabaeus utilities.

    References
    ----------
    .. [1] McElrath, T. P., Watkins, M. M., Portock, B. M., Graat, E. J.,
       Baird, D. T., Wawrzyniak, G. G., Guinn, J. R., Antreasian, P. G.,
       Attiyah, A. A., Baalke, R. C., & Taber, W. L. (2004).
       *Mars Exploration Rovers Orbit Determination Filter Strategy.*
       In *AIAA/AAS Astrodynamics Specialist Conference and Exhibit*,
       Providence, Rhode Island, 16–19 August 2004.
       AIAA Paper 2004-5082.
       https://doi.org/10.2514/6.2004-5082
    """

    # ----------------#
    # region Methods #
    # ----------------#

    # ------------------------------------------------------------------
    # region  Closeness measure
    # ------------------------------------------------------------------

    @staticmethod
    def closeness(
        c1: np.ndarray,
        P1: np.ndarray,
        c2: np.ndarray,
        P2: np.ndarray,
        alpha_range: tuple = (1e-6, 1e6),
        n_grid: int = 10_000,
        tol: float = 1e-10,
    ) -> dict:
        """
        Compute the sigma-ellipsoid closeness measure between two OD solutions.

        Given two Gaussian solutions S1 = (c1, P1) and S2 = (c2, P2), returns
        the smallest *t* > 0 such that the *t*-sigma uncertainty ellipsoids are
        tangent (AIAA 2004-4982 [1]_).  A value *d* < 1 means the solutions are
        mutually consistent within their 1-sigma bounds.

        Parameters
        ----------
        c1 : ndarray, shape (n,)
            Mean state vector of solution 1.
        P1 : ndarray, shape (n, n)
            Covariance matrix of solution 1 (symmetric positive definite).
        c2 : ndarray, shape (n,)
            Mean state vector of solution 2.
        P2 : ndarray, shape (n, n)
            Covariance matrix of solution 2 (symmetric positive definite).
        alpha_range : (float, float), optional
            Search range ``(alpha_min, alpha_max)`` for the Lagrange multiplier.
            Defaults to ``(1e-6, 1e6)``.
        n_grid : int, optional
            Number of logarithmically spaced grid points for the initial root
            bracket search.  Defaults to ``10_000``.
        tol : float, optional
            Bisection convergence tolerance for alpha root refinement.
            Defaults to ``1e-10``.

        Returns
        -------
        result : dict
            Dictionary with the following keys:

            * ``'d'``         – float, closeness in units of sigma.
            * ``'t_sq'``      – float, ``d ** 2``.
            * ``'alpha'``     – float, Lagrange multiplier at the tangency point.
            * ``'x_tang'``    – ndarray shape (n,), tangency point in state space.
            * ``'all_roots'`` – list of all positive alpha roots found.
            * ``'all_t_sq'``  – list of t² values for each root.

        Notes
        -----
        The algorithm evaluates the stationarity condition

        .. math::

            f(\\alpha) = \\Delta c^\\top (P_2 + \\alpha P_1)^{-1}
                         (\\alpha^2 P_1 - P_2)
                         (P_2 + \\alpha P_1)^{-1} \\Delta c = 0

        on a logarithmic grid, brackets sign changes, and refines them with
        bisection.  If no root is found a :class:`UserWarning` is issued and
        ``'d'`` is returned as ``nan``.

        See Also
        --------
        UtilsOD.closeness_matrix : Pairwise matrix for N solutions.

        References
        ----------
        .. [1] Woffinden & Geller (2004). AIAA 2004-4982.

        Examples
        --------
        ::

            import numpy as np
            import scarabaeus as scb

            c1 = np.zeros(6)
            P1 = np.eye(6) * 4.0
            c2 = np.array([1.0, 0.5, 0.2, 0.0, 0.0, 0.0])
            P2 = np.eye(6) * 2.0

            res = scb.UtilsOD.closeness(c1, P1, c2, P2)
            print(f"d = {res['d']:.4f} sigma")
        """
        c1 = np.asarray(c1, dtype=float).ravel()
        c2 = np.asarray(c2, dtype=float).ravel()
        P1 = np.asarray(P1, dtype=float)
        P2 = np.asarray(P2, dtype=float)
        dc = c2 - c1

        if np.allclose(dc, 0):
            return {
                "d": 0.0,
                "t_sq": 0.0,
                "alpha": None,
                "x_tang": c1.copy(),
                "all_roots": [],
                "all_t_sq": [],
            }

        def _f(a):
            M = P2 + a * P1
            try:
                Mi = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                return np.nan
            return float(dc @ Mi @ (a**2 * P1 - P2) @ Mi @ dc)

        def _tsq(a):
            Mi = np.linalg.inv(P2 + a * P1)
            return float(dc @ Mi @ P2 @ Mi @ dc)

        alphas = np.geomspace(alpha_range[0], alpha_range[1], n_grid)
        fvals = np.array([_f(a) for a in alphas])

        roots = []
        for i in range(len(fvals) - 1):
            if np.isnan(fvals[i]) or np.isnan(fvals[i + 1]):
                continue
            if fvals[i] * fvals[i + 1] < 0:
                lo, hi = alphas[i], alphas[i + 1]
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    fm = _f(mid)
                    if abs(fm) < tol or (hi - lo) < tol * mid:
                        break
                    if _f(lo) * fm < 0:
                        hi = mid
                    else:
                        lo = mid
                roots.append(0.5 * (lo + hi))

        if not roots:
            warnings.warn(
                "closeness(): no positive alpha roots found — ellipsoids may be nested "
                "or alpha_range too narrow.",
                stacklevel=2,
            )
            return {
                "d": np.nan,
                "t_sq": np.nan,
                "alpha": None,
                "x_tang": None,
                "all_roots": [],
                "all_t_sq": [],
            }

        t_sq_vals = [_tsq(a) for a in roots]
        best = int(np.argmin(t_sq_vals))
        best_a = roots[best]
        best_tsq = t_sq_vals[best]

        L1 = np.linalg.inv(P1)
        L2 = np.linalg.inv(P2)
        x_tang = np.linalg.solve(L1 + best_a * L2, L1 @ c1 + best_a * L2 @ c2)

        return {
            "d": float(np.sqrt(max(best_tsq, 0.0))),
            "t_sq": float(best_tsq),
            "alpha": float(best_a),
            "x_tang": x_tang,
            "all_roots": roots,
            "all_t_sq": t_sq_vals,
        }

    @staticmethod
    def closeness_matrix(
        solutions: list,
        labels: list = None,
        verbose: bool = True,
        **closeness_kwargs,
    ) -> tuple:
        """
        Compute the N×N pairwise closeness matrix for a list of OD solutions.

        Parameters
        ----------
        solutions : list of (c, P) or SolutionOD
            Each entry is either a ``(mean_vector, covariance_matrix)`` tuple or
            a :class:`~scarabaeus.SolutionOD` object (final epoch is used).
        labels : list of str, optional
            Labels for each solution.  Defaults to ``["S1", "S2", …]``.
        verbose : bool, optional
            Print progress for each pair.  Defaults to ``True``.
        **closeness_kwargs
            Forwarded to :meth:`closeness` (e.g. ``alpha_range``, ``n_grid``).

        Returns
        -------
        D : ndarray, shape (N, N)
            Symmetric matrix where ``D[i, j] = d(Si, Sj)`` in sigma units.
            Diagonal entries are zero.
        labels : list of str
            Labels used (auto-generated when not supplied).

        See Also
        --------
        UtilsOD.plot_closeness_matrix : Visualise the returned matrix.
        """
        n = len(solutions)
        if labels is None:
            labels = [f"S{i + 1}" for i in range(n)]

        pairs = [UtilsOD._as_c_P(s) for s in solutions]
        D = np.zeros((n, n))
        total = n * (n - 1) // 2
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                count += 1
                c1, P1 = pairs[i]
                c2, P2 = pairs[j]
                res = UtilsOD.closeness(c1, P1, c2, P2, **closeness_kwargs)
                D[i, j] = D[j, i] = res["d"]
                if verbose:
                    print(
                        f"  [{count}/{total}] d({labels[i]}, {labels[j]}) "
                        f"= {res['d']:.4f} σ"
                    )

        return D, labels

    @staticmethod
    def plot_closeness_matrix(
        D: np.ndarray,
        labels: list = None,
        scheme: str = "merb",
        title: str = "OD Solution Consistency Matrix",
        show_values: bool = True,
        figsize: tuple = None,
        save_path: str = None,
    ) -> None:
        """
        Visualise an N×N pairwise closeness matrix.

        Parameters
        ----------
        D : ndarray, shape (N, N)
            Closeness matrix returned by :meth:`closeness_matrix`.
        labels : list of str, optional
            Axis tick labels.  Defaults to ``["S1", "S2", …]``.
        scheme : {'merb', 'mera', 'sigma', 'continuous'}, optional
            Colormap scheme:

            * ``'merb'`` *(default)* — five-band categorical map.
            * ``'mera'`` — two-band binary map at the 0.8 σ boundary.
            * ``'sigma'`` — four-band map at 1, 2, 3 σ boundaries.
            * ``'continuous'`` — ``RdYlGn_r`` continuous colormap.

        title : str, optional
            Figure title.  Defaults to ``'OD Solution Consistency Matrix'``.
        show_values : bool, optional
            Annotate upper-triangle cells with numeric values.  Defaults to
            ``True``.
        figsize : (float, float), optional
            Figure size in inches.  Auto-scaled from N when ``None``.
        save_path : str, optional
            Save the figure to this path at 300 dpi instead of displaying it.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import BoundaryNorm, ListedColormap

        n = D.shape[0]
        if labels is None:
            labels = [f"S{i + 1}" for i in range(n)]

        _schemes = {
            "merb": {
                "bounds": [0.0, 0.2, 0.5, 0.8, 1.2, 9999],
                "colors": ["#1a9e74", "#6cc9a9", "#3a7abf", "#d95f30", "#8b1a1a"],
                "labels": ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–1.2", "≥ 1.2"],
            },
            "mera": {
                "bounds": [0.0, 0.8, 9999],
                "colors": ["#3a7abf", "#d95f30"],
                "labels": ["≤ 0.8 (consistent)", "> 0.8 (inconsistent)"],
            },
            "sigma": {
                "bounds": [0.0, 1.0, 2.0, 3.0, 9999],
                "colors": ["#1a9e74", "#6cc9a9", "#d95f30", "#8b1a1a"],
                "labels": ["≤ 1σ", "1σ–2σ", "2σ–3σ", "> 3σ"],
            },
        }
        cfg = _schemes.get(scheme)

        if figsize is None:
            sz = n * 0.62 + 1.8
            figsize = (sz + 1.0, sz)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f8f8f8")
        ax.set_facecolor("#f8f8f8")

        mask = np.tril(np.ones_like(D, dtype=bool), k=-1)
        D_plot = np.ma.masked_where(mask, D)

        if cfg is not None:
            cmap = ListedColormap(cfg["colors"])
            cmap.set_bad(color="#f8f8f8")
            norm = BoundaryNorm(cfg["bounds"], cmap.N)
            ax.imshow(D_plot, cmap=cmap, norm=norm, aspect="equal")
        else:
            vmax = max(float(np.nanmax(D)), 1.5)
            cmap = plt.cm.RdYlGn_r.copy()
            cmap.set_bad(color="#f8f8f8")
            im = ax.imshow(D_plot, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

        for i in range(n):
            ax.add_patch(
                plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#cccccc", zorder=2)
            )
            ax.text(
                i,
                i,
                "—",
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                zorder=3,
            )

        if show_values and cfg is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    val = D[i, j]
                    idx = min(
                        int(np.searchsorted(cfg["bounds"][1:], val)),
                        len(cfg["colors"]) - 1,
                    )
                    r, g, b = (
                        int(cfg["colors"][idx].lstrip("#")[k : k + 2], 16) / 255
                        for k in (0, 2, 4)
                    )
                    tc = (
                        "white"
                        if 0.299 * r + 0.587 * g + 0.114 * b < 0.55
                        else "#1a1a1a"
                    )
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=max(6, min(9, 90 // n)),
                        color=tc,
                        fontweight="bold",
                        zorder=4,
                    )

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for k in range(n + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.8, zorder=1)
            ax.axvline(k - 0.5, color="white", linewidth=0.8, zorder=1)

        if cfg is not None:
            patches = [
                mpatches.Patch(facecolor=c, edgecolor="#999", linewidth=0.5, label=l)
                for c, l in zip(cfg["colors"], cfg["labels"])
            ]
            ax.legend(
                handles=patches,
                title="Closeness d (σ)",
                title_fontsize=8,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                framealpha=0.9,
                edgecolor="#cccccc",
            )
        else:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Closeness d (σ)", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

        ax.set_title(title, fontsize=11, fontweight="500", pad=12)
        ax.set_xlabel("Solution", fontsize=9, labelpad=6)
        ax.set_ylabel("Solution", fontsize=9, labelpad=6)

        upper = D[np.triu_indices(n, k=1)]
        fig.text(
            0.5,
            0.01,
            (
                f"N = {n}   |   min d = {np.nanmin(upper):.3f} σ   "
                f"mean d = {np.nanmean(upper):.3f} σ   "
                f"max d = {np.nanmax(upper):.3f} σ"
            ),
            ha="center",
            fontsize=8,
            color="#555555",
            style="italic",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    # endregion Closeness measure

    # ------------------------------------------------------------------
    # region  Estimate combination
    # ------------------------------------------------------------------

    @staticmethod
    def merge(
        c1: np.ndarray,
        P1: np.ndarray,
        c2: np.ndarray,
        P2: np.ndarray,
    ) -> tuple:
        """
        Combine two independent Gaussian estimates via the information filter.

        Given two unbiased estimates of the same quantity with covariances P1
        and P2, the optimal linear combination is:

        .. math::

            P_m = (P_1^{-1} + P_2^{-1})^{-1}, \\quad
            c_m = P_m (P_1^{-1} c_1 + P_2^{-1} c_2)

        This is equivalent to a Kalman update where one estimate treats the
        other as a measurement [2]_.

        Parameters
        ----------
        c1 : ndarray, shape (n,)
            Mean of estimate 1.
        P1 : ndarray, shape (n, n)
            Covariance of estimate 1.
        c2 : ndarray, shape (n,)
            Mean of estimate 2.
        P2 : ndarray, shape (n, n)
            Covariance of estimate 2.

        Returns
        -------
        c_merged : ndarray, shape (n,)
            Combined mean.
        P_merged : ndarray, shape (n, n)
            Combined covariance (always ≤ min(P1, P2) element-wise on the
            diagonal).

        Raises
        ------
        np.linalg.LinAlgError
            If either covariance matrix is singular.

        Examples
        --------
        ::

            c_m, P_m = scb.UtilsOD.merge(c1, P1, c2, P2)
        """
        c1 = np.asarray(c1, dtype=float).ravel()
        c2 = np.asarray(c2, dtype=float).ravel()
        P1 = np.asarray(P1, dtype=float)
        P2 = np.asarray(P2, dtype=float)

        L1 = np.linalg.inv(P1)
        L2 = np.linalg.inv(P2)
        P_m = np.linalg.inv(L1 + L2)
        P_m = 0.5 * (P_m + P_m.T)  # enforce symmetry
        c_m = P_m @ (L1 @ c1 + L2 @ c2)
        return c_m, P_m

    @staticmethod
    def merge_many(solutions: list) -> tuple:
        """
        Combine N independent Gaussian estimates sequentially.

        Applies :meth:`merge` iteratively to the list of solutions, accumulating
        the combined mean and covariance.

        Parameters
        ----------
        solutions : list of (c, P) or SolutionOD
            Ordered list of solutions.  Each entry is a ``(mean, covariance)``
            tuple or a :class:`~scarabaeus.SolutionOD` object (final epoch).

        Returns
        -------
        c_merged : ndarray, shape (n,)
            Combined mean.
        P_merged : ndarray, shape (n, n)
            Combined covariance.

        Raises
        ------
        ValueError
            If fewer than two solutions are provided.

        Examples
        --------
        ::

            c_m, P_m = scb.UtilsOD.merge_many([sol1, sol2, sol3])
        """
        if len(solutions) < 2:
            raise ValueError("merge_many() requires at least two solutions.")

        pairs = [UtilsOD._as_c_P(s) for s in solutions]
        c_m, P_m = pairs[0]
        for c2, P2 in pairs[1:]:
            c_m, P_m = UtilsOD.merge(c_m, P_m, c2, P2)
        return c_m, P_m

    # endregion Estimate combination

    # ------------------------------------------------------------------
    # region  Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def mahalanobis(x: np.ndarray, c: np.ndarray, P: np.ndarray) -> float:
        """
        Compute the Mahalanobis distance between a point and a Gaussian.

        .. math::

            d_M = \\sqrt{(x - c)^\\top P^{-1} (x - c)}

        Parameters
        ----------
        x : ndarray, shape (n,)
            Query point.
        c : ndarray, shape (n,)
            Distribution mean.
        P : ndarray, shape (n, n)
            Covariance matrix (symmetric positive definite).

        Returns
        -------
        d : float
            Mahalanobis distance in units of sigma.
        """
        dx = np.asarray(x, dtype=float).ravel() - np.asarray(c, dtype=float).ravel()
        return float(np.sqrt(dx @ np.linalg.inv(np.asarray(P, dtype=float)) @ dx))

    @staticmethod
    def nees(
        x_true: np.ndarray,
        c_est: np.ndarray,
        P_est: np.ndarray,
    ) -> float:
        """
        Compute the Normalized Estimation Error Squared (NEES).

        A standard filter consistency metric [2]_:

        .. math::

            \\epsilon = (x_{true} - c_{est})^\\top P_{est}^{-1} (x_{true} - c_{est})

        For a consistent filter, the expected value of NEES equals the state
        dimension *n*.

        Parameters
        ----------
        x_true : ndarray, shape (n,)
            True state vector.
        c_est : ndarray, shape (n,)
            Filter estimated mean.
        P_est : ndarray, shape (n, n)
            Filter estimated covariance.

        Returns
        -------
        epsilon : float
            NEES value.  Expected value for a consistent filter: ``n``.
        """
        dx = (
            np.asarray(x_true, dtype=float).ravel()
            - np.asarray(c_est, dtype=float).ravel()
        )
        return float(dx @ np.linalg.inv(np.asarray(P_est, dtype=float)) @ dx)

    @staticmethod
    def sigma_overlap(
        c1: np.ndarray,
        P1: np.ndarray,
        c2: np.ndarray,
        P2: np.ndarray,
        n_sigma: float = 1.0,
    ) -> bool:
        """
        Check whether two sigma ellipsoids overlap.

        Two ellipsoids overlap if the sum of their Mahalanobis radii (each
        evaluated at the midpoint between the means) exceeds the distance
        between the means [2]_.  This is a fast, conservative heuristic; use
        :meth:`closeness` for the exact tangency value.

        Parameters
        ----------
        c1 : ndarray, shape (n,)
            Mean of distribution 1.
        P1 : ndarray, shape (n, n)
            Covariance of distribution 1.
        c2 : ndarray, shape (n,)
            Mean of distribution 2.
        P2 : ndarray, shape (n, n)
            Covariance of distribution 2.
        n_sigma : float, optional
            Number of sigma defining the ellipsoids.  Defaults to ``1.0``.

        Returns
        -------
        overlap : bool
            ``True`` if the *n_sigma*-ellipsoids overlap.
        """
        c1 = np.asarray(c1, dtype=float).ravel()
        c2 = np.asarray(c2, dtype=float).ravel()
        dc = c2 - c1
        r1 = n_sigma / np.sqrt(dc @ np.linalg.inv(np.asarray(P1, dtype=float)) @ dc)
        r2 = n_sigma / np.sqrt(dc @ np.linalg.inv(np.asarray(P2, dtype=float)) @ dc)
        return bool((r1 + r2) >= 1.0)

    # endregion Diagnostics

    # ------------------------------------------------------------------
    # region  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_c_P(sol) -> tuple:
        """
        Convert a solution to a ``(c, P)`` tuple.

        Accepts a raw ``(c, P)`` tuple or a
        :class:`~scarabaeus.SolutionOD` object (uses the final epoch).

        Parameters
        ----------
        sol : tuple or SolutionOD
            The solution to convert.

        Returns
        -------
        c : ndarray, shape (n,)
        P : ndarray, shape (n, n)

        Raises
        ------
        TypeError
            If ``sol`` is not a recognised type.
        ValueError
            If the required data is not present in a ``SolutionOD`` object.
        """
        if isinstance(sol, tuple) and len(sol) == 2:
            return (
                np.asarray(sol[0], dtype=float).ravel(),
                np.asarray(sol[1], dtype=float),
            )

        try:
            from scarabaeus.orbitDetermination.SolutionOD import SolutionOD as _SOD

            if isinstance(sol, _SOD):
                if sol.covariance_est is None:
                    raise ValueError(
                        "SolutionOD has no covariance_est. "
                        "Enable OutputSettings.save_covariance_history."
                    )
                P = np.asarray(sol.covariance_est[-1], dtype=float)
                if sol.deviation_est is not None:
                    c = np.asarray(sol.deviation_est[-1], dtype=float).ravel()
                elif sol.state_est is not None:
                    c = np.asarray(sol.state_est[-1], dtype=float).ravel()
                else:
                    raise ValueError(
                        "SolutionOD has neither deviation_est nor state_est."
                    )
                return c, P
        except ImportError:
            pass

        raise TypeError(
            f"Each solution must be a (c, P) tuple or a SolutionOD object. "
            f"Received {type(sol)}."
        )

    # endregion Internal helpers

    # endregion Methods #
    # ------------------#
