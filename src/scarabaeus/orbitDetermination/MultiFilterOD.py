# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# Multi-arc orbit determination combining any FilterOD instances.
#
# MATHEMATICAL OVERVIEW  (French et al. OSIRIS-REx / NAPEOS 2009 formulation)
# ----------------------------------------------------------------
# N arcs, each with local parameters a_i and shared global
# parameters g.  After each arc converges, extract (dx_0, P_0)
# at a common epoch t_0:
#
#   - batch / smoother  : take directly from solution
#   - sequential only   : map final-epoch result back via Φ_N
#                         dx_0 = Φ_N · dx_N,  P_0 = Φ_N · P_N · Φ_N^T
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
from typing import List, Optional


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
    :math:`P_{0,i}` referred to a common reference epoch :math:`t_0`.

    For *batch* or *smoother* outputs the solution is already at :math:`t_0`.
    For *sequential* filters (LKF, SRIF) the final-epoch result is mapped
    back via the cumulative state-transition matrix :math:`\\Phi_N`:

    .. math::

        \\delta\\mathbf{x}_{0,i} = \\Phi_N\\,\\delta\\mathbf{x}_{N,i},
        \\qquad
        P_{0,i} = \\Phi_N\\,P_{N,i}\\,\\Phi_N^T

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
    :math:`\\hat{\\mathbf{x}}_G` falls below ``convergence_threshold``.

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

    See Also
    --------
    scarabaeus.FilterOD : Base filter class used per-arc.
    scarabaeus.LKF : Sequential covariance-form filter.
    scarabaeus.SRIF : Sequential square-root information filter.
    scarabaeus.LSB : Batch least-squares estimator.
    scarabaeus.SRIFB : Batch square-root information estimator.

    References
    ----------
    French, A. S., Leonard, J. M., Geeraert, J. L., Page, B. R., Antreasian, P. G.,
        Moreau, M. C., McMahon, J. W., Scheeres, D. J., & the OSIRIS-REx Team.
        "Multi-Arc Filtering During the Navigation Campaign of the OSIRIS-REx Mission."
        KinetX / NASA GSFC / University of Colorado. (Primary reference for the
        partitioned normal-equation and Schur-complement multi-arc formulation
        implemented here.)
    European Space Operations Centre (2009).
        "NAPEOS — Navigation Package for Earth Orbiting Satellites," May 2009,
        pp. 1–150.
    Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination.
        Elsevier Academic Press. ISBN 978-0-12-683630-1.
    Bierman, G. J. (1977).
        *Factorization Methods for Discrete Sequential Estimation*. Academic Press.
    """

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
        """Numerically robust matrix inverse (falls back to pseudo-inverse)."""
        try:
            return _np.linalg.inv(A)
        except _np.linalg.LinAlgError:
            pass
        U, S, Vt = _np.linalg.svd(A)
        S = _np.maximum(S, min_svd)
        return Vt.T @ _np.diag(1.0 / S) @ U.T

    # ─────────────────────────────────────────────────────────
    # Filter-type helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_srif_type(filt) -> bool:
        return (
            hasattr(filt, "rhat_k") and hasattr(filt, "bhat_k") and len(filt.rhat_k) > 0
        )

    @staticmethod
    def _get_stm_final(filt) -> _np.ndarray:
        """Cumulative STM Φ_N from t_0 to the last processed measurement epoch."""
        spacecraft_times = filt.measurement_data.get_spacecraft_times()
        N = (
            filt.num_processed_steps
            if hasattr(filt, "num_processed_steps") and filt.num_processed_steps > 0
            else len(filt.covariance_history)
        )
        return filt.trajectory.get_STM(epoch=spacecraft_times[N], idx=N)

    # ─────────────────────────────────────────────────────────
    # Extract (dx_0, P_0) at t_0  —  unified for all filter types
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_dx_P_at_t0(filt, solution) -> tuple:
        """
        Return (dx_0, P_0) in the estimated subspace at t_0.

        SRIF-type
        ---------
        R_N = inv(rhat_k[-1]),  P_N = R_N^T R_N  at t_N.
        Map to t_0 via Φ_N:  dx_0 = Φ_N dx_N,  P_0 = Φ_N P_N Φ_N^T.

        Batch / smoother
        ----------------
        Solution is already expressed at t_0 — take directly.

        Sequential KF (no smoother)
        ---------------------------
        P and dx live at the final epoch t_N.
        Map to t_0 via Φ_N (same formula as SRIF).
        """
        srif_type = MultiFilterOD._is_srif_type(filt)

        if srif_type:
            R_N = MultiFilterOD._safe_inv(filt.rhat_k[-1])
            P_N = R_N.T @ R_N
            dx_N = _np.asarray(solution.deviation_est[-1]).flatten()
            Phi = MultiFilterOD._get_stm_final(filt)
            return Phi @ dx_N, Phi @ P_N @ Phi.T

        P_last = _np.asarray(filt.covariance_history[-1])
        dx_last = _np.asarray(solution.deviation_est[-1]).flatten()

        if getattr(filt, "if_sequential_smooth", False) or getattr(
            filt, "is_batch", False
        ):
            return dx_last, P_last

        Phi = MultiFilterOD._get_stm_final(filt)
        return Phi @ dx_last, Phi @ P_last @ Phi.T

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
    ) -> None:
        """
        Apply combined local and global corrections to an arc filter and re-propagate.

        Assembles the full-state correction vector by placing ``delta_a`` at
        the local parameter indices and ``delta_G`` at the global parameter
        indices, then calls :meth:`FilterOD._reinitialize` to re-propagate the
        reference trajectory and rebuild measurement datasets.

        If the total correction norm is below ``1e-20`` the call is skipped to
        avoid unnecessary re-propagation.

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
        """
        correction = _np.zeros(arc_d["n_full"])
        if len(arc_d["local_full_idx"]) > 0 and delta_a.size > 0:
            correction[arc_d["local_full_idx"]] = delta_a
        correction[arc_d["global_full_idx"]] = delta_G

        if _np.linalg.norm(correction) < 1e-20:
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
        if_sequential_smooth: bool = False,
        traj_name: Optional[str] = None,
        verbose: bool = True,
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
        if_sequential_smooth : bool
            Apply smoothing to each arc after inner convergence.
        traj_name : str, optional
            Base name for trajectory files; arc index and outer iteration
            number are appended automatically.
        verbose : bool
            Print progress banners.

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
        """
        x_G_prev: Optional[_np.ndarray] = None
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
                    if_sequential_smooth=if_sequential_smooth,
                    verbose=verbose,
                    traj_name=arc_traj,
                )
                arc_solutions[i] = solution

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
            if x_G_prev is not None:
                delta = x_G_combined - x_G_prev
                scale = _np.maximum(_np.abs(x_G_combined), 1.0)
                rms_outer = float(_np.sqrt(_np.mean((delta / scale) ** 2)))
                if verbose:
                    print(
                        f"\n  Outer relative RMS Δx_G = {rms_outer:.3e}"
                        f"  (threshold = {convergence_threshold:.3e})"
                    )
                if rms_outer < convergence_threshold:
                    outer_converged = True
                    if verbose:
                        print(f"\n  {'='*60}")
                        print(
                            f"  ✓ Multi-arc CONVERGED after {outer} outer iteration(s)"
                        )
                        print(f"  {'='*60}")
                    break

            x_G_prev = x_G_combined.copy()

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
                        filt, d, delta_a, delta_G, traj_name=upd_traj
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
        }
        for sol in arc_solutions:
            if sol is not None:
                sol.multi_arc_global = multi_arc_global

        return arc_solutions
