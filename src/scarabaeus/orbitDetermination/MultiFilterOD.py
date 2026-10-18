# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# Multi-arc orbit determination combining any FilterOD instances.
#
# MATHEMATICAL OVERVIEW  (French et al. OSIRIS-REx / NAPEOS 2009 formulation)
# ----------------------------------------------------------------
# N arcs, each with local parameters a_i and shared global
# parameters g.  After each arc converges, extract (dx_0, P_0)
# at the arc's initial epoch t_0 from the row the arc correction
# uses (first smoothed epoch, else last filtered epoch t_r):
#
#   M = Φ(t_0) Φ(t_r)^{-1}
#   dx_0 = M · dx_r,  P_0 = M · P_r · M^T
#
# Batch filters (LSB, SRIFB) also store their rows at t_r, so
# every filter type is mapped back the same way.
#
# Form normal equations and partition (Eq. 1):
#
#   N_i = P_0^{-1},  B_i = N_i · dx_0
#
#   N_i = [ N_a_i   N_ag_i ]    B_i = [ B_a_i ]
#         [ N_ag_i^T  N_g_i ]         [ B_g_i ]
#
# Schur-complement to eliminate local parameters (Eq. 4):
#
#   Ñ_g_i = N_g_i - N_ag_i^T · N_a_i^{-1} · N_ag_i
#   B̃_g_i = B_g_i - N_ag_i^T · N_a_i^{-1} · B_a_i
#
# Combine arcs around common reference ḡ (Eq. 7–8):
#
#   Λ    = P_prior^{-1} + Σ Ñ_g_i
#   N̂    = Σ (B̃_g_i + Ñ_g_i · (x_g_i - ḡ))
#   P̂_g  = Λ^{-1}
#   x̂_g  = ḡ + P̂_g · N̂
#
# where x_g_i = x_G_prop_i + dx_0[global_idx] is arc i's
# absolute global estimate and ḡ is an arbitrary common
# reference (paper: "solution is insensitive to choice").
#
# Back-substitute for local parameters (Eq. 11–12):
#
#   P̂_a_i = (N_a_i - N_ag_i · P̂_g · N_ag_i^T)^{-1}
#   Δx_a_i = P̂_a_i · (B_a_i - N_ag_i · P̂_g · N̂)
# =============================================================

import numpy as _np
from typing import Callable, List, Optional


class MultiFilterOD:
    """ Multi-arc orbit determination combining N independent FilterOD instances.

    Each arc has its own propagator, measurements, settings, and filter type
    (SRIF, LKF, LSB, SRIFB, or any :class:`~scarabaeus.FilterOD` subclass).
    The class runs all arcs to convergence and then fuses their estimates of
    any parameters designated as *global* (shared across arcs) using the
    NAPEOS partitioned normal-equation formulation with Schur-complement
    reduction.

    **Algorithm overview**

    The full state vector of arc :math:`i` is partitioned into arc-local
    parameters :math:`\\mathbf{a}_i` and global (shared) parameters
    :math:`\\mathbf{g}`:

    .. math::

        \\mathbf{x}_i = \\begin{bmatrix} \\mathbf{a}_i \\\\ \\mathbf{g} \\end{bmatrix}

    **Step 1 — Per-arc convergence.**
    Each arc filter is run to inner convergence independently, yielding a
    state deviation :math:`\\delta\\mathbf{x}_{0,i}` and covariance
    :math:`P_{0,i}` referred to the arc's initial epoch :math:`t_0`.

    The row used is the one the arc correction is built from: the first
    smoothed epoch when the arc was smoothed, otherwise the last filtered
    epoch :math:`t_r`. Every filter type, batch (LSB, SRIFB) included,
    stores its rows at their own epoch, so the row is mapped back with the
    cumulative state-transition matrix :math:`\\Phi`:

    .. math::

        M = \\Phi(t_0)\\,\\Phi(t_r)^{-1},
        \\qquad
        \\delta\\mathbf{x}_{0,i} = M\\,\\delta\\mathbf{x}_{r,i},
        \\qquad
        P_{0,i} = M\\,P_{r,i}\\,M^T

    **Step 2 — Normal equations.**
    The information matrix and right-hand side for arc :math:`i` are

    .. math::

        N_i = P_{0,i}^{-1},
        \\qquad
        \\mathbf{b}_i = N_i\\,\\delta\\mathbf{x}_{0,i}

    When ``subtract_arc_priors=True`` the arc's own apriori information is
    removed so it is counted only once via ``global_apriori_cov``:

    .. math::

        N_i \\leftarrow N_i - P_{\\mathrm{apr},i}^{-1}

    **Step 3 — Partitioning and Schur complement.**
    The normal matrix is split into local and global blocks:

    .. math::

        N_i =
        \\begin{bmatrix}
            N_{a,i}   & N_{ag,i} \\\\
            N_{ag,i}^T & N_{g,i}
        \\end{bmatrix},
        \\qquad
        \\mathbf{b}_i =
        \\begin{bmatrix} \\mathbf{b}_{a,i} \\\\ \\mathbf{b}_{g,i} \\end{bmatrix}

    Local parameters are eliminated via the Schur complement:

    .. math::

        \\tilde{N}_{g,i} = N_{g,i} - N_{ag,i}^T N_{a,i}^{-1} N_{ag,i}

        \\tilde{\\mathbf{b}}_{g,i} = \\mathbf{b}_{g,i}
            - N_{ag,i}^T N_{a,i}^{-1} \\mathbf{b}_{a,i}

    **Step 4 — Global combination.**
    An arbitrary reference :math:`\\bar{\\mathbf{g}}` (arc-0 absolute estimate)
    is used to form the combined information and normal vector:

    .. math::

        \\Lambda = P_{\\mathrm{prior}}^{-1}
                  + \\sum_{i=1}^{N} \\tilde{N}_{g,i}

        \\hat{\\mathbf{n}} = \\sum_{i=1}^{N}
            \\Bigl( \\tilde{\\mathbf{b}}_{g,i}
                  + \\tilde{N}_{g,i}\\,
                    (\\mathbf{x}_{G,i} - \\bar{\\mathbf{g}}) \\Bigr)

    giving the combined global estimate and covariance:

    .. math::

        \\hat{P}_G = \\Lambda^{-1},
        \\qquad
        \\hat{\\mathbf{x}}_G = \\bar{\\mathbf{g}} + \\hat{P}_G\\,\\hat{\\mathbf{n}}

    The solution is insensitive to the choice of :math:`\\bar{\\mathbf{g}}`.

    **Step 5 — Local back-substitution.**
    With :math:`\\hat{P}_G` known, the arc-local correction is recovered by a
    second Schur inversion:

    .. math::

        \\hat{P}_{a,i} =
            \\bigl( N_{a,i} - N_{ag,i}\\,\\hat{P}_G\\,N_{ag,i}^T \\bigr)^{-1}

        \\Delta\\mathbf{x}_{a,i} =
            \\hat{P}_{a,i}\\,
            \\bigl( \\mathbf{b}_{a,i} - N_{ag,i}\\,\\hat{P}_G\\,\\hat{\\mathbf{n}} \\bigr)

    **Step 6 — Reinitialization.**
    Each arc is restarted from the corrected reference trajectory
    :math:`\\mathbf{x}_{\\mathrm{ref}} + [\\Delta\\mathbf{x}_{a,i};\\,\\Delta\\mathbf{x}_G]`
    and the outer loop repeats until the relative RMS change in
    :math:`\\hat{\\mathbf{x}}_G` falls below ``convergence_threshold``, or the
    relative change of the total weighted cost :math:`J/n` falls below
    ``convergence_threshold_cost`` (see ``convergence_criterion``).

    Parameters
    ----------
    filters : list of FilterOD
        One filter per arc, fully configured (propagator, measurements,
        settings).  All filters must estimate the same set of global
        parameters (matching base names) plus their own arc-local parameters.
    global_param_names : list of str
        Base names of parameters shared across arcs (e.g. ``["gm", "cnm_2_0"]``).
        Every arc must estimate at least one such parameter.
    global_apriori_cov : np.ndarray of shape (n_G, n_G), optional
        Apriori covariance for the global parameter vector.  Enters as
        :math:`P_{\\mathrm{prior}}^{-1}` in the global combination step.
        Set to ``None`` (default) to omit (improper prior).
    subtract_arc_priors : bool, optional
        If ``True`` (default), each arc's own apriori covariance
        :math:`P_{\\mathrm{apr},i}` is subtracted from :math:`N_i` before
        combination, preventing the prior from being counted once per arc.
        Set to ``False`` to retain the classic behaviour in which every arc
        contributes its full posterior information matrix including the prior.

    Raises
    ------
    ValueError
        If any arc filter has ``flag_sequence=True`` (multi-leg arcs are not
        supported).
    ValueError
        If any arc filter estimates no parameter whose name appears in
        ``global_param_names``.

    See Also
    --------
    scarabaeus.FilterOD : Base filter class used per-arc.
    scarabaeus.LKF : Sequential covariance-form filter.
    scarabaeus.SRIF : Sequential square-root information filter.
    scarabaeus.LSB : Batch least-squares estimator.
    scarabaeus.SRIFB : Batch square-root information estimator.

    Notes
    -----
    * Multi-leg (sequence) arcs are not supported; each arc must be a
      single-leg filter.
    * All arcs must estimate the *same* set of global parameters (matched by
      base name); the dimension :math:`n_G` must be consistent across arcs.
    * The outer loop reinitializes arcs after each global combination; the
      number of outer iterations required is typically 2–4 for well-posed
      problems.
    * If no ``global_apriori_cov`` is supplied, the combined problem has an
      improper prior and the solution is determined solely by the arc data.

    References
    ----------
    .. [1] French, A. S., Leonard, J. M., Geeraert, J. L., Page, B. R., Antreasian, P. G.,
       Moreau, M. C., McMahon, J. W., Scheeres, D. J., & the OSIRIS-REx Team.
       "Multi-Arc Filtering During the Navigation Campaign of the OSIRIS-REx Mission."
       KinetX / NASA GSFC / University of Colorado. (Primary reference for the
       partitioned normal-equation and Schur-complement multi-arc formulation
       implemented here.)
    .. [2] European Space Operations Centre (2009).
       "NAPEOS — Navigation Package for Earth Orbiting Satellites," May 2009,
       pp. 1–150.
    .. [3] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical Orbit
       Determination*. Elsevier Academic Press. ISBN 978-0-12-683630-1.
    .. [4] Bierman, G. J. (1977).
       *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ConvergenceCriterionError(ValueError):
        """Raised when an unknown convergence criterion is requested."""

    class ArcSolutionError(ValueError):
        """Raised when an arc solution lacks the histories needed at t_0."""

    # endregion Errors #
    # ------------------#

    def __init__(
        self,
        filters: list,
        global_param_names: List[str],
        global_apriori_cov: Optional[_np.ndarray] = None,
        subtract_arc_priors: bool = True,
    ):
        self.filters = filters
        self.global_param_names = set(global_param_names)
        self.global_apriori_cov = global_apriori_cov
        self.subtract_arc_priors = subtract_arc_priors

        for i, filt in enumerate(filters):
            if getattr(filt, "flag_sequence", False):
                raise ValueError(
                    f"Arc filter {i} uses a MissionSequence (flag_sequence=True). "
                    "MultiFilterOD does not support sequence/multi-leg filters as arcs."
                )
            _, global_est_idx, _, _ = self._get_partition(filt)
            if len(global_est_idx) == 0:
                raise ValueError(
                    f"Arc filter {i} has no estimated parameters matching "
                    f"{sorted(self.global_param_names)}.  Every arc must estimate "
                    "at least one global parameter."
                )

    # ─────────────────────────────────────────────────────────
    # State partitioning
    # ─────────────────────────────────────────────────────────

    def _get_partition(self, filt) -> tuple:
        """
        Partition state indices into local and global sub-sets for one arc filter.

        Iterates over the filter's trajectory state definition and classifies
        each **estimated** parameter as either *local* (arc-specific) or
        *global* (shared across arcs, whose name appears in
        ``self.global_param_names``).

        Parameters
        ----------
        filt : FilterOD
            Arc filter whose state definition is to be partitioned.

        Returns
        -------
        local_est_idx : list of int
            Column indices in the *estimated* subspace (``n_est``-dimensional)
            for arc-local parameters.
        global_est_idx : list of int
            Column indices in the estimated subspace for global parameters.
        local_full_idx : list of int
            Column indices in the *full* state vector (``n``-dimensional,
            including consider parameters) for arc-local parameters.
        global_full_idx : list of int
            Column indices in the full state vector for global parameters.
        """
        local_est_idx: list = []
        global_est_idx: list = []
        local_full_idx: list = []
        global_full_idx: list = []
        est_cursor = 0
        full_cursor = 0

        for (
            name,
            dim,
            est_flag,
            dyn_flag,
            owner,
            value,
        ) in filt.trajectory.state_definition:
            if est_flag == "estimated":
                est_slice = list(range(est_cursor, est_cursor + dim))
                full_slice = list(range(full_cursor, full_cursor + dim))
                if name in self.global_param_names:
                    global_est_idx.extend(est_slice)
                    global_full_idx.extend(full_slice)
                else:
                    local_est_idx.extend(est_slice)
                    local_full_idx.extend(full_slice)
                est_cursor += dim
            full_cursor += dim

        return local_est_idx, global_est_idx, local_full_idx, global_full_idx

    # ─────────────────────────────────────────────────────────
    # Absolute global values from propagator reference state
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_global_abs(filt, global_est_idx: list) -> _np.ndarray:
        """
        Extract absolute (physical) values of global parameters from the arc's reference state.

        Reads the current propagator reference state and concatenates the
        numeric values of all *estimated* parameters in order, then indexes
        into that array at the global positions to return only the global subset.

        Parameters
        ----------
        filt : FilterOD
            Arc filter whose propagator reference state is queried.
        global_est_idx : list of int
            Indices into the estimated-parameter sub-vector that correspond to
            global parameters (as returned by :meth:`_get_partition`).

        Returns
        -------
        np.ndarray
            1-D array of length ``n_G`` containing the absolute (non-deviation)
            reference values of the global parameters for this arc.
        """
        est_vals: list = []
        for (
            name,
            dim,
            est_flag,
            dyn_flag,
            body,
            value,
        ) in filt.propagator.full_state_vector.state:
            if est_flag != "estimated":
                continue
            est_vals.extend(value.quantity.values.flatten()[:dim].tolist())
        return _np.array(est_vals)[global_est_idx]

    # ─────────────────────────────────────────────────────────
    # Numerics helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _safe_inv(A, min_svd=1e-14):
        """Invert a matrix, falling back to a floored-SVD pseudo-inverse.

        Parameters
        ----------
        A : np.ndarray
            Square matrix to invert.
        min_svd : float, optional
            Floor applied to the singular values when the direct inverse fails.
            Defaults to ``1e-14``.

        Returns
        -------
        np.ndarray
            :math:`A^{-1}`, or :math:`V \\, \\mathrm{diag}(1/\\max(s_i, s_{min}))
            \\, U^T` when :math:`A` is singular.
        """
        try:
            return _np.linalg.inv(A)
        except _np.linalg.LinAlgError:
            pass
        U, S, Vt = _np.linalg.svd(A)
        S = _np.maximum(S, min_svd)
        return Vt.T @ _np.diag(1.0 / S) @ U.T

    # ─────────────────────────────────────────────────────────
    # Extract (dx_0, P_0) at t_0  —  unified for all filter types
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_dx_P_at_t0(filt, solution) -> tuple:
        """
        Return the estimated deviation and covariance at :math:`t_0`.

        Parameters
        ----------
        filt : FilterOD
            Filter that has been run.
        solution : SolutionOD
            Solution returned by ``filt``.

        Returns
        -------
        tuple of np.ndarray
            ``(dx_0, P_0)`` in the estimated subspace.

        Raises
        ------
        ArcSolutionError
            If the arc's solution holds no deviation or covariance history
            aligned with its timeline (e.g. covariance analysis, or history
            storage turned off in its output settings).

        Notes
        -----
        The row used is the one :meth:`FilterOD.fit` maps back for the arc
        correction (:meth:`SolutionOD.map_state_deviation_to_epoch`): the first
        smoothed epoch when the arc was smoothed, otherwise the last filtered
        one. Every filter, sequential (LKF, SRIF) or batch (LSB, SRIFB), stores
        its deviation and covariance at the epoch :math:`t_r` of their row, so
        the row is mapped to :math:`t_0` with

        .. math::

            M = \\Phi(t_0) \\, \\Phi(t_r)^{-1}, \\qquad
            \\delta x_0 = M \\, \\delta x_r, \\qquad
            P_0 = M \\, P_r \\, M^T ,

        :math:`\\Phi` being the arc's cumulative STM. :math:`\\Phi(t_0)` is the
        identity unless the stored STM starts elsewhere.

        With considered states (always last, and constant) only the estimated
        block :math:`M_{ee}` of :math:`M` is applied: the stored estimated
        covariance is conditional on the considered states, which is the
        information the arc's normal equations carry, and considered deviations
        are zero.
        """
        smoothed = all(
            h is not None and len(h) > 0
            for h in (solution.deviation_smooth, solution.covariance_smooth)
        )
        deviations = solution.deviation_smooth if smoothed else solution.deviation_est
        covariances = solution.covariance_smooth if smoothed else solution.covariance_est
        timeline = solution._timeline_epochs()
        if (
            timeline is None
            or deviations is None
            or covariances is None
            or len(deviations) == 0
            or not len(timeline) == len(deviations) == len(covariances)
        ):
            raise MultiFilterOD.ArcSolutionError(
                "The arc solution needs state deviation and covariance histories "
                "aligned with its timeline; enable save_state_deviation_history "
                "and save_covariance_history and run an estimation pass."
            )

        # The last update at the selected epoch wins
        t_r = timeline.min() if smoothed else timeline.max()
        k = int(_np.flatnonzero(timeline == t_r)[-1])
        dx_r = _np.asarray(deviations[k], dtype=float).ravel()
        P_r = _np.asarray(covariances[k], dtype=float)

        epoch = filt.propagator.full_state_vector.epoch
        t_0 = float(_np.asarray(epoch.et, dtype=float).ravel()[0])
        Phi_r = _np.asarray(filt.trajectory.get_STM(epoch=float(t_r)), dtype=float)
        Phi_0 = _np.asarray(filt.trajectory.get_STM(epoch=t_0), dtype=float)
        M = Phi_0 @ _np.linalg.inv(Phi_r)

        idx_est = getattr(filt, "idx_est", None)
        est = (
            _np.arange(dx_r.size)
            if idx_est is None
            else _np.asarray(idx_est, dtype=int)
        )
        M_ee = M[_np.ix_(est, est)]
        dx_0 = M_ee @ dx_r[est]
        P_0 = M_ee @ P_r[_np.ix_(est, est)] @ M_ee.T
        return dx_0, 0.5 * (P_0 + P_0.T)

    # ─────────────────────────────────────────────────────────
    # Normal-equation blocks + Schur complement
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normal_equation_blocks(
        dx_0: _np.ndarray,
        P_0: _np.ndarray,
        local_idx: list,
        global_idx: list,
        P_apriori_est: Optional[_np.ndarray] = None,
    ) -> dict:
        """
        Form N = P_0^{-1}, B = N dx_0, partition, and Schur-complement.

        If P_apriori_est is provided, it is subtracted from N so that the
        normal equations reflect only the data likelihood of this arc,
        preventing the arc's prior from being counted multiple times when
        arcs are combined in _combine_global.

        Returns dict with keys:
          N_a, N_g, N_ag, B_a, B_g  — raw partitioned blocks
          N_tilde_g, B_tilde_g      — Schur-complemented (local-eliminated)
        """
        N = MultiFilterOD._safe_inv(P_0)
        if P_apriori_est is not None:
            N = N - MultiFilterOD._safe_inv(P_apriori_est)
        B = N @ dx_0

        a, g = local_idx, global_idx
        N_a = N[_np.ix_(a, a)] if a else _np.zeros((0, 0))
        N_g = N[_np.ix_(g, g)]
        N_ag = N[_np.ix_(a, g)] if a else _np.zeros((0, len(g)))
        B_a = B[a] if a else _np.zeros(0)
        B_g = B[g]

        if len(a) > 0:
            N_a_inv_N_ag = _np.linalg.solve(N_a, N_ag)
            N_tilde_g = N_g - N_ag.T @ N_a_inv_N_ag
            B_tilde_g = B_g - N_ag.T @ _np.linalg.solve(N_a, B_a)
        else:
            N_tilde_g = N_g.copy()
            B_tilde_g = B_g.copy()

        return dict(
            N_a=N_a,
            N_g=N_g,
            N_ag=N_ag,
            B_a=B_a,
            B_g=B_g,
            N_tilde_g=N_tilde_g,
            B_tilde_g=B_tilde_g,
        )

    # ─────────────────────────────────────────────────────────
    # Global combination  (Eq. 7–8)
    # ─────────────────────────────────────────────────────────

    def _combine_global(self, arc_data: list, n_G: int) -> tuple:
        """
        Combine Schur-complemented arc contributions into a single global estimate.

        Implements Equations 7–8 of the NAPEOS multi-arc formulation.  An
        arbitrary reference :math:`\\bar{g}` (taken as arc 0's absolute
        estimate) is used to shift the problem; the solution is insensitive
        to this choice.

        .. math::

            \\Lambda = P_{\\text{prior}}^{-1} + \\sum_i \\tilde{N}_{g,i}

            \\hat{N} = \\sum_i \\left( \\tilde{B}_{g,i}
                      + \\tilde{N}_{g,i} (x_{G,i} - \\bar{g}) \\right)

            \\hat{P}_G = \\Lambda^{-1}, \\quad
            \\hat{x}_G = \\bar{g} + \\hat{P}_G \\hat{N}

        Parameters
        ----------
        arc_data : list of dict
            One dict per arc, each containing the keys ``N_tilde_g``,
            ``B_tilde_g``, and ``x_G_abs`` as produced by
            :meth:`_normal_equation_blocks`.
        n_G : int
            Dimension of the global parameter vector.

        Returns
        -------
        x_G_combined : np.ndarray
            Combined absolute global parameter estimate, shape ``(n_G,)``.
        P_G : np.ndarray
            Combined global parameter covariance, shape ``(n_G, n_G)``.
        """
        g_bar = arc_data[0]["x_G_abs"].copy()
        Lambda = _np.zeros((n_G, n_G))
        N_hat = _np.zeros(n_G)

        if self.global_apriori_cov is not None:
            Lambda += MultiFilterOD._safe_inv(_np.asarray(self.global_apriori_cov, dtype=float))

        for d in arc_data:
            Lambda += d["N_tilde_g"]
            N_hat += d["B_tilde_g"] + d["N_tilde_g"] @ (d["x_G_abs"] - g_bar)

        P_G = MultiFilterOD._safe_inv(Lambda)
        x_G = g_bar + P_G @ N_hat
        return x_G, P_G

    # ─────────────────────────────────────────────────────────
    # Local back-substitution  (Eq. 11–12)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _solve_local(arc_d: dict, P_G: _np.ndarray, B_hat_g: _np.ndarray) -> tuple:
        """
        Back-substitute to obtain the arc-local parameter correction.

        Implements Equations 11–12 of the NAPEOS formulation.  Given the
        combined global covariance :math:`\\hat{P}_G` and combined normal
        vector :math:`\\hat{N}`, the local Schur complement is inverted to
        yield the update for arc-local parameters:

        .. math::

            \\hat{P}_a = \\left( N_a - N_{ag}\\, \\hat{P}_G\\, N_{ag}^T \\right)^{-1}

            \\Delta x_a = \\hat{P}_a \\left( B_a - N_{ag}\\, \\hat{P}_G\\, \\hat{N} \\right)

        Parameters
        ----------
        arc_d : dict
            Arc normal-equation blocks from :meth:`_normal_equation_blocks`,
            containing at least ``N_a``, ``N_ag``, and ``B_a``.
        P_G : np.ndarray
            Combined global parameter covariance of shape ``(n_G, n_G)``.
        B_hat_g : np.ndarray
            Combined global normal vector :math:`\\hat{N}` of shape ``(n_G,)``,
            computed once in :meth:`fit` and reused across arcs.

        Returns
        -------
        delta_a : np.ndarray
            Local parameter correction vector, shape ``(n_a,)``.
            Zero-length array when the arc has no local parameters.
        P_a : np.ndarray
            Local parameter covariance, shape ``(n_a, n_a)``.
            Zero-area matrix when the arc has no local parameters.
        """
        N_a = arc_d["N_a"]
        N_ag = arc_d["N_ag"]
        B_a = arc_d["B_a"]

        if N_a.size == 0:
            return _np.zeros(0), _np.zeros((0, 0))

        P_a = MultiFilterOD._safe_inv(N_a - N_ag @ P_G @ N_ag.T)
        delta_a = P_a @ (B_a - N_ag @ P_G @ B_hat_g)
        return delta_a, P_a

    # ─────────────────────────────────────────────────────────
    # Reinitialize arc
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _reinitialize_arc(
        filt,
        arc_d: dict,
        delta_a: _np.ndarray,
        delta_G: _np.ndarray,
        traj_name: Optional[str] = None,
        rebuild: bool = False,
    ) -> None:
        """
        Apply combined local and global corrections to an arc filter and re-propagate.

        Assembles the full-state correction vector by placing ``delta_a`` at
        the local parameter indices and ``delta_G`` at the global parameter
        indices, then calls :meth:`FilterOD._reinitialize` to re-propagate the
        reference trajectory and rebuild measurement datasets.

        If the total correction norm is below ``1e-20`` the call is skipped to
        avoid unnecessary re-propagation, unless ``rebuild`` is set.

        Parameters
        ----------
        filt : FilterOD
            Arc filter to reinitialise.
        arc_d : dict
            Arc data dict from :meth:`_normal_equation_blocks`, containing
            ``local_full_idx``, ``global_full_idx``, and ``n_full``.
        delta_a : np.ndarray
            Local parameter correction, shape ``(n_a,)``.
        delta_G : np.ndarray
            Global parameter correction ``x_G_combined - x_G_abs_prop``,
            shape ``(n_G,)``.
        traj_name : str, optional
            Name for the new trajectory BSP file after reinitialisation.
        rebuild : bool
            Reinitialise even when the correction is zero, so that the
            datasets are rebuilt after a :meth:`fit` callback changed the
            data they are built from.
        """
        correction = _np.zeros(arc_d["n_full"])
        if len(arc_d["local_full_idx"]) > 0 and delta_a.size > 0:
            correction[arc_d["local_full_idx"]] = delta_a
        correction[arc_d["global_full_idx"]] = delta_G

        if _np.linalg.norm(correction) < 1e-20 and not rebuild:
            return

        filt._reinitialize(
            state_deviation=correction.reshape(-1, 1),
            traj_name=traj_name,
        )

    # ─────────────────────────────────────────────────────────
    # Main fit
    # ─────────────────────────────────────────────────────────

    def fit(
        self,
        max_outer_iters: int = 5,
        max_inner_iters: int = 10,
        convergence_threshold: float = 1e-6,
        convergence_threshold_cost: float = 1e-3,
        convergence_criterion: str = "any",
        if_sequential_smooth: bool = False,
        traj_name: Optional[str] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> List["SolutionOD"]:
        """
        Run multi-arc orbit determination.

        Parameters
        ----------
        max_outer_iters : int
            Maximum outer (global-combination) iterations.
        max_inner_iters : int
            Maximum inner (per-arc) iterations per outer step.
        convergence_threshold : float
            Relative RMS convergence threshold for both inner and outer loops.
        convergence_threshold_cost : float
            Threshold on the relative change of the normalised weighted cost
            :math:`J/n = \\sum_i (r_i/\\sigma_i)^2 / n` between iterations. In the
            outer loop the cost is summed over all arcs. Weakly observable
            global parameters keep drifting between outer iterations without
            improving the fit; this test declares convergence in that case.
        convergence_criterion : str
            How the RMS and cost tests are combined, for both loops:
            ``'any'`` (default), ``'deviation'``, ``'cost'``, or ``'both'``.
            See :meth:`~scarabaeus.FilterOD.fit`.
        if_sequential_smooth : bool
            Apply smoothing to each arc after inner convergence.
        traj_name : str, optional
            Base name for trajectory files; arc index and outer iteration
            number are appended automatically.
        verbose : bool
            Print progress banners.
        callback : callable, optional
            Called as ``callback(multi_filter, outer, arc_solutions)`` after
            every outer iteration, with this object, the outer iteration number
            and the solution of each arc in the order of ``filters``. It runs
            once the outer convergence test is done (the first outer iteration
            has none) and before the arcs are reinitialised for the next one.
            Its return value is ignored. It is the place to reweight or edit
            the data between outer iterations, as the ``callback`` of
            :meth:`~scarabaeus.FilterOD.fit` is between the iterations of one
            filter: a sigma set on an arc's model
            (:meth:`~scarabaeus.FilterOD.set_measurement_sigmas`, or
            ``model.sigma = ...``) or an outlier flagged through
            ``MeasurementSpec.flag`` takes effect when that arc's datasets are
            rebuilt for the next outer iteration, so one made after the last
            outer iteration does nothing within this call. The inner fit of
            each arc runs without a callback. Defaults to None.

        Returns
        -------
        list[SolutionOD]
            One solution per arc from the final outer iteration.
            Each solution's ``multi_arc_global`` attribute contains:

            * ``x_G``                — (n_G,) combined absolute estimate
            * ``P_G``                — (n_G, n_G) combined covariance
            * ``global_param_names`` — sorted list of global parameter names
            * ``arc_global_abs``     — per-arc absolute estimates
            * ``n_outer``            — outer iterations performed
            * ``converged``          — True if outer loop converged
            * ``criterion``          — convergence criterion used
            * ``cost``               — total weighted cost of the last outer step
            * ``cost_normalized``    — that cost divided by ``n_residuals``
            * ``cost_relative_change`` — its relative change vs. the previous step
            * ``n_residuals``        — residuals contributing to ``cost``

        Raises
        ------
        ConvergenceCriterionError
            If ``convergence_criterion`` is not one of the accepted values.

        Notes
        -----
        Reweighting the data from ``callback`` changes the scale of the cost, so
        the cost test of the next outer iteration compares two differently
        weighted costs; prefer ``convergence_criterion='deviation'`` across a
        change of sigmas.

        Examples
        --------
        >>> # Loosen the range sigma of every arc after the first outer iteration
        >>> def reweight(multi, outer, arc_solutions):
        ...     if outer == 1:
        ...         for filt in multi.filters:
        ...             filt.set_measurement_sigmas({"GS1 Range": ArrayWUnits(5.0, m)})
        >>> arc_solutions = multi.fit(callback=reweight)
        """
        criterion = str(convergence_criterion).lower()
        if criterion not in ("any", "deviation", "cost", "both"):
            raise self.ConvergenceCriterionError(
                f"Unknown convergence_criterion '{convergence_criterion}'. "
                "Choose 'any', 'deviation', 'cost', or 'both'."
            )

        x_G_prev: Optional[_np.ndarray] = None
        cost_normalized_prev = float("nan")
        cost_total = float("nan")
        cost_normalized = float("nan")
        cost_change = float("inf")
        n_residuals_total = 0
        arc_solutions = [None] * len(self.filters)
        arc_global_abs_final = [None] * len(self.filters)
        x_G_combined: Optional[_np.ndarray] = None
        P_G: Optional[_np.ndarray] = None
        n_G: Optional[int] = None
        outer_converged = False
        outer = 0

        if verbose:
            print("\n" + "=" * 70)
            print("  MULTI-ARC ORBIT DETERMINATION")
            print(f"  Arcs         : {len(self.filters)}")
            print(f"  Global params: {sorted(self.global_param_names)}")
            print(f"  Max outer / inner iters: {max_outer_iters} / {max_inner_iters}")
            print("=" * 70)

        for outer in range(1, max_outer_iters + 1):
            if verbose:
                print(f"\n{'='*70}")
                print(f"  OUTER ITERATION {outer} / {max_outer_iters}")
                print(f"{'='*70}")

            arc_data: list = []
            cost_total = 0.0
            n_residuals_total = 0

            # ── Inner loop: converge each arc independently ────────────────
            for i, filt in enumerate(self.filters):
                if verbose:
                    print(f"\n  --- Arc {i+1}/{len(self.filters)} [outer={outer}] ---")

                arc_base = (
                    f"{traj_name.replace('.bsp','')}_arc{i+1}" if traj_name else None
                )
                arc_traj = f"{arc_base}_outer{outer}.bsp" if arc_base else None

                solution, n_inner, inner_conv = filt.fit(
                    max_iterations=max_inner_iters,
                    convergence_threshold=convergence_threshold,
                    convergence_threshold_cost=convergence_threshold_cost,
                    convergence_criterion=criterion,
                    if_sequential_smooth=if_sequential_smooth,
                    verbose=verbose,
                    traj_name=arc_traj,
                )
                arc_solutions[i] = solution

                # Accumulate the arc's weighted cost for the outer-loop test
                cost_arc, n_res_arc = filt._cost_function()
                if n_res_arc > 0:
                    cost_total += cost_arc
                    n_residuals_total += n_res_arc

                local_idx, global_idx, local_full_idx, global_full_idx = (
                    self._get_partition(filt)
                )
                n_G_arc = len(global_idx)
                if n_G is None:
                    n_G = n_G_arc
                elif n_G != n_G_arc:
                    raise ValueError(
                        f"Arc {i} has {n_G_arc} global components; expected {n_G}."
                    )

                # (dx_0, P_0) at t_0 — unified across all filter types
                dx_0, P_0 = self._extract_dx_P_at_t0(filt, solution)

                # Normal-equation blocks + Schur complement
                ne = self._normal_equation_blocks(
                    dx_0,
                    P_0,
                    local_idx,
                    global_idx,
                    P_apriori_est=(
                        filt.apriori_covariance_est
                        if self.subtract_arc_priors
                        else None
                    ),
                )

                # Absolute global estimate for this arc
                x_G_abs_prop = self._extract_global_abs(filt, global_idx)
                x_G_abs_i = x_G_abs_prop + dx_0[_np.array(global_idx)]
                arc_global_abs_final[i] = x_G_abs_i.copy()

                arc_data.append(
                    {
                        **ne,
                        "x_G_abs": x_G_abs_i,
                        "x_G_abs_prop": x_G_abs_prop,
                        "local_idx": local_idx,
                        "global_idx": global_idx,
                        "local_full_idx": local_full_idx,
                        "global_full_idx": global_full_idx,
                        "n_full": filt.n,
                    }
                )

            # ── Combine global parameters (Eq. 7–8) ───────────────────────
            x_G_combined, P_G = self._combine_global(arc_data, n_G)

            if verbose:
                print(f"\n  Combined global estimate (absolute):")
                print(f"    x̂_G = {x_G_combined}")
                print(f"    σ_G = {_np.sqrt(_np.diag(P_G))}")
                for i, d in enumerate(arc_data):
                    print(f"    Arc {i+1} estimate: {d['x_G_abs']}")

            # ── Outer convergence check ────────────────────────────────────
            # Total weighted cost over all arcs, normalized by the number of
            # residuals so that measurement editing does not look like a change.
            cost_normalized = (
                cost_total / n_residuals_total
                if n_residuals_total > 0
                else float("nan")
            )
            cost_change = float("inf")
            if _np.isfinite(cost_normalized) and _np.isfinite(cost_normalized_prev):
                cost_change = abs(cost_normalized - cost_normalized_prev) / max(
                    abs(cost_normalized_prev), 1e-300
                )

            if x_G_prev is not None:
                delta = x_G_combined - x_G_prev
                scale = _np.maximum(_np.abs(x_G_combined), 1.0)
                rms_outer = float(_np.sqrt(_np.mean((delta / scale) ** 2)))

                converged_deviation = bool(rms_outer < convergence_threshold)
                converged_cost = bool(cost_change < convergence_threshold_cost)

                if criterion == "deviation":
                    outer_converged = converged_deviation
                elif criterion == "cost":
                    outer_converged = converged_cost
                elif criterion == "both":
                    outer_converged = converged_deviation and converged_cost
                else:  # "any"
                    outer_converged = converged_deviation or converged_cost

                if verbose:
                    print(
                        f"\n  Outer relative RMS Δx_G = {rms_outer:.3e}"
                        f"  (threshold = {convergence_threshold:.3e})"
                    )
                    if n_residuals_total > 0:
                        print(
                            f"  Outer cost J = Σ(r/σ)² = {cost_total:.3e}"
                            f"  (n = {n_residuals_total}, J/n = {cost_normalized:.3e})"
                        )
                        print(
                            f"  Outer relative cost change = {cost_change:.3e}"
                            f"  (threshold = {convergence_threshold_cost:.3e})"
                        )
                    else:
                        print(
                            "  Outer cost: unavailable (no weighted residuals) "
                            "-> cost criterion inactive"
                        )

            if callback is not None:
                callback(self, outer, list(arc_solutions))

            if outer_converged:
                if verbose:
                    triggers = []
                    if converged_deviation:
                        triggers.append("RMS Δx_G")
                    if converged_cost:
                        triggers.append("cost function")
                    print(f"\n  {'='*60}")
                    print(
                        f"  ✓ Multi-arc CONVERGED after {outer} outer iteration(s)"
                    )
                    print(f"    Triggered by: {' and '.join(triggers)}")
                    print(f"  {'='*60}")
                break

            x_G_prev = x_G_combined.copy()
            cost_normalized_prev = cost_normalized

            # ── Back-substitute and reinitialize (Eq. 11–12) ──────────────
            if outer < max_outer_iters:
                # N̂ = Σ(B̃_g_i + Ñ_g_i·(x_g_i - ḡ))  — same sum used in _combine_global
                g_bar = arc_data[0]["x_G_abs"]
                B_hat_g = sum(
                    d["B_tilde_g"] + d["N_tilde_g"] @ (d["x_G_abs"] - g_bar)
                    for d in arc_data
                )

                for i, (filt, d) in enumerate(zip(self.filters, arc_data)):
                    if verbose:
                        print(f"\n  Reinitializing arc {i+1} with combined x_G …")

                    delta_a, _ = self._solve_local(d, P_G, B_hat_g)
                    delta_G = x_G_combined - d["x_G_abs_prop"]

                    arc_base = (
                        f"{traj_name.replace('.bsp','')}_arc{i+1}"
                        if traj_name
                        else None
                    )
                    upd_traj = (
                        f"{arc_base}_outer{outer}_update.bsp" if arc_base else None
                    )
                    self._reinitialize_arc(
                        filt,
                        d,
                        delta_a,
                        delta_G,
                        traj_name=upd_traj,
                        rebuild=callback is not None,
                    )

        if not outer_converged and verbose:
            print(
                f"\n  !! Multi-arc did NOT converge after {max_outer_iters} outer iterations"
            )

        multi_arc_global = {
            "x_G": x_G_combined,
            "P_G": P_G,
            "global_param_names": sorted(self.global_param_names),
            "arc_global_abs": arc_global_abs_final,
            "n_outer": outer,
            "converged": outer_converged,
            "criterion": criterion,
            "cost": cost_total,
            "cost_normalized": cost_normalized,
            "cost_relative_change": cost_change,
            "n_residuals": n_residuals_total,
        }
        for sol in arc_solutions:
            if sol is not None:
                sol.multi_arc_global = multi_arc_global

        return arc_solutions
