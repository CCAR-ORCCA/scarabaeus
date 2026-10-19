# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import types

import numpy as np
import pytest
import src.scarabaeus as scb

# --------------------#
# region    Fixtures  #
# --------------------#


def stub(global_names=("gm",), flag_sequence=False):
    """Minimal FilterOD-like stub for MultiFilterOD construction tests."""
    state_def = [
        ("position", 3, "estimated", "dynamic", None, None),
        ("velocity", 3, "estimated", "dynamic", None, None),
    ]
    for name in global_names:
        state_def.append((name, 1, "estimated", "static", None, None))

    class StubTraj:
        pass

    class StubFilter:
        pass

    traj = StubTraj()
    traj.state_definition = state_def
    f = StubFilter()
    f.flag_sequence = flag_sequence
    f.trajectory = traj
    return f

# endregion Fixtures #


# ---------------#
# region  Tests  #
# ---------------#


class TestInitialization:

    def test_attributes_stored(self):
        """Valid input → filters, global_param_names, apriori_cov, subtract_arc_priors set."""
        s = stub(["gm"])
        multi = scb.MultiFilterOD(
            filters=[s, s],
            global_param_names=["gm"],
            global_apriori_cov=np.eye(1),
            subtract_arc_priors=False,
        )
        assert len(multi.filters) == 2
        assert multi.global_param_names == {"gm"}
        assert multi.global_apriori_cov is not None
        assert multi.subtract_arc_priors is False

    @pytest.mark.parametrize("apriori, subtract", [
        (None, True),
        (np.eye(1), True),
        (None, False),
    ])
    def test_defaults_and_overrides(self, apriori, subtract):
        """global_apriori_cov and subtract_arc_priors are stored as given."""
        s = stub(["gm"])
        multi = scb.MultiFilterOD(
            filters=[s], global_param_names=["gm"],
            global_apriori_cov=apriori, subtract_arc_priors=subtract,
        )
        assert multi.global_apriori_cov is apriori
        assert multi.subtract_arc_priors is subtract

    @pytest.mark.parametrize("bad_filters, match", [
        ([stub(["gm"], flag_sequence=True)], "flag_sequence=True"),
        ([stub(global_names=[])], "no estimated parameters matching"),
    ])
    def test_invalid_filter_raises(self, bad_filters, match):
        """Sequence filter or missing global param each raise ValueError."""
        with pytest.raises(ValueError, match=match):
            scb.MultiFilterOD(bad_filters, ["gm"])

    def test_second_filter_in_list_raises(self):
        """Error message identifies the offending arc index."""
        with pytest.raises(ValueError, match="Arc filter 1"):
            scb.MultiFilterOD([stub(["gm"]), stub([], False)], ["gm"])


class TestSafeInv:

    @pytest.mark.parametrize("A, expected", [
        (np.diag([2.0, 4.0, 8.0]), np.diag([0.5, 0.25, 0.125])),
        (np.eye(3),                 np.eye(3)),
        (np.array([[4.0]]),         np.array([[0.25]])),
    ])
    def test_invertible(self, A, expected):
        np.testing.assert_allclose(scb.MultiFilterOD._safe_inv(A), expected, atol=1e-12)

    def test_satisfies_A_inv_A_identity(self):
        A = np.array([[3.0, 1.0], [1.0, 5.0]])
        np.testing.assert_allclose(scb.MultiFilterOD._safe_inv(A) @ A, np.eye(2), atol=1e-12)

    def test_singular_returns_finite(self):
        A = np.outer([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        result = scb.MultiFilterOD._safe_inv(A)
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))


class TestExtractDxPAtT0:
    """(dx_0, P_0) is the arc-correction row mapped back to the initial epoch."""

    T0 = 1000.0

    @staticmethod
    def shift(dt):
        """Constant-velocity STM over ``dt`` for a (position, velocity) state."""
        return np.array([[1.0, dt], [0.0, 1.0]])

    @classmethod
    def arc(cls, timeline, deviations, covariances, smoothed=False, stm=None,
            idx_est=None):
        """Filter and solution stubs; the STM defaults to ``shift(t - T0)``."""
        stm = stm or (lambda t: cls.shift(t - cls.T0))

        class StubTraj:
            def get_STM(self, epoch, idx=None):
                return stm(epoch)

        class StubEpoch:
            et = np.array([cls.T0])

        class StubState:
            epoch = StubEpoch()

        class StubPropagator:
            full_state_vector = StubState()

        class StubFilter:
            pass

        class StubSolution:
            deviation_est = deviation_smooth = None
            covariance_est = covariance_smooth = None

            def _timeline_epochs(self):
                return np.asarray(timeline, dtype=float)

        f = StubFilter()
        f.trajectory = StubTraj()
        f.propagator = StubPropagator()
        if idx_est is not None:
            f.idx_est = np.asarray(idx_est)
        sol = StubSolution()
        if smoothed:
            sol.deviation_smooth, sol.covariance_smooth = deviations, covariances
        else:
            sol.deviation_est, sol.covariance_est = deviations, covariances
        return f, sol

    def test_filtered_row_is_mapped_back_not_forward(self):
        """dx_N = Φ(t_N) dx_0 must return dx_0 (the old code applied Φ again)."""
        dx_true = np.array([1.0, 0.5])
        P_true = np.diag([4.0, 1.0])
        Phi_N = self.shift(30.0)
        f, sol = self.arc(
            [1000.0, 1010.0, 1030.0],
            [np.zeros(2), np.ones(2), Phi_N @ dx_true],
            [np.eye(2), np.eye(2), Phi_N @ P_true @ Phi_N.T],
        )

        dx_0, P_0 = scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)

        np.testing.assert_allclose(dx_0, dx_true, atol=1e-12)
        np.testing.assert_allclose(P_0, P_true, atol=1e-12)

    def test_last_update_wins_at_the_final_epoch(self):
        """Simultaneous rows at t_N: the last stored one is used."""
        f, sol = self.arc(
            [1000.0, 1030.0, 1030.0, 1010.0],
            [np.zeros(2), np.array([9.0, 9.0]), np.array([3.0, 0.0]), np.ones(2)],
            [np.eye(2)] * 4,
        )

        dx_0, _ = scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)

        np.testing.assert_allclose(dx_0, [3.0, 0.0], atol=1e-12)

    def test_smoothed_history_uses_its_first_epoch(self):
        """A smoothed arc is read at its first epoch and mapped from there."""
        dx_true = np.array([2.0, -1.0])
        Phi_1 = self.shift(5.0)
        f, sol = self.arc(
            [1005.0, 1020.0],
            [Phi_1 @ dx_true, np.array([7.0, 7.0])],
            [Phi_1 @ np.eye(2) @ Phi_1.T, np.eye(2)],
            smoothed=True,
        )

        dx_0, P_0 = scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)

        np.testing.assert_allclose(dx_0, dx_true, atol=1e-12)
        np.testing.assert_allclose(P_0, np.eye(2), atol=1e-12)

    def test_stored_stm_need_not_start_at_t0(self):
        """M = Φ(t_0) Φ(t_r)^{-1}, not Φ(t_r)^{-1} alone."""
        f, sol = self.arc(
            [1000.0, 1030.0], [np.zeros(2), np.array([0.0, 1.0])],
            [np.eye(2)] * 2, stm=lambda t: self.shift(t - 900.0),
        )

        dx_0, _ = scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)

        np.testing.assert_allclose(dx_0, [-30.0, 1.0], atol=1e-12)

    def test_considered_states_are_held_at_nominal(self):
        """Estimated block of M on the conditional covariance; consider dropped."""
        Phi = np.array([[1.0, 30.0, 5.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        P_r = np.array([[4.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 9.0]])
        dx_r = np.array([1.0, 2.0, 0.0])
        f, sol = self.arc(
            [1000.0, 1030.0], [np.zeros(3), dx_r], [np.eye(3), P_r],
            stm=lambda t: Phi if t > 1000.0 else np.eye(3), idx_est=[0, 1],
        )

        dx_0, P_0 = scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)

        M_ee = np.linalg.inv(Phi)[:2, :2]
        np.testing.assert_allclose(dx_0, M_ee @ dx_r[:2], atol=1e-12)
        np.testing.assert_allclose(P_0, M_ee @ P_r[:2, :2] @ M_ee.T, atol=1e-12)

    @pytest.mark.parametrize("timeline, deviations", [
        ([1000.0], None),
        ([1000.0, 1010.0], [np.zeros(2)]),
    ])
    def test_missing_or_misaligned_history_raises(self, timeline, deviations):
        f, sol = self.arc(timeline, deviations, None if deviations is None
                          else [np.eye(2)] * len(deviations))
        with pytest.raises(scb.MultiFilterOD.ArcSolutionError, match="aligned"):
            scb.MultiFilterOD._extract_dx_P_at_t0(f, sol)


class TestNormalEquationBlocks:

    @pytest.mark.parametrize("local_idx, global_idx, P_0, dx_0, exp_N_tilde, exp_B_tilde", [
        # No local params: N_tilde_g == N_g, B_tilde_g == B_g
        ([], [0, 1, 2], 2.0 * np.eye(3), np.array([1.0, 0.0, 0.0]),
         0.5 * np.eye(3), np.array([0.5, 0.0, 0.0])),
        # Diagonal P_0 → zero cross-term → N_tilde_g == N_g block
        ([0, 1, 2], [3], np.eye(4), np.array([1.0, 2.0, 3.0, 4.0]),
         np.array([[1.0]]), np.array([4.0])),
    ])
    def test_schur_complement(self, local_idx, global_idx, P_0, dx_0, exp_N_tilde, exp_B_tilde):
        blocks = scb.MultiFilterOD._normal_equation_blocks(dx_0, P_0, local_idx, global_idx)
        np.testing.assert_allclose(blocks["N_tilde_g"], exp_N_tilde, atol=1e-12)
        np.testing.assert_allclose(blocks["B_tilde_g"], exp_B_tilde, atol=1e-12)

    def test_all_keys_present(self):
        blocks = scb.MultiFilterOD._normal_equation_blocks(
            np.zeros(3), np.eye(3), local_idx=[], global_idx=[0, 1, 2]
        )
        for key in ("N_a", "N_g", "N_ag", "B_a", "B_g", "N_tilde_g", "B_tilde_g"):
            assert key in blocks

    def test_empty_local_shapes(self):
        blocks = scb.MultiFilterOD._normal_equation_blocks(
            np.zeros(2), np.eye(2), local_idx=[], global_idx=[0, 1]
        )
        assert blocks["N_a"].shape == (0, 0)
        assert blocks["N_ag"].shape == (0, 2)

    def test_prior_subtraction_reduces_N_tilde_g(self):
        P_0, P_apr = 2.0 * np.eye(3), 4.0 * np.eye(3)
        no_sub = scb.MultiFilterOD._normal_equation_blocks(
            np.zeros(3), P_0, [], [0, 1, 2]
        )
        sub = scb.MultiFilterOD._normal_equation_blocks(
            np.zeros(3), P_0, [], [0, 1, 2], P_apriori_est=P_apr
        )
        expected = no_sub["N_tilde_g"] - scb.MultiFilterOD._safe_inv(P_apr)
        np.testing.assert_allclose(sub["N_tilde_g"], expected, atol=1e-12)


class TestCombineGlobal:

    def make_multi(self, apriori=None):
        return scb.MultiFilterOD(
            filters=[stub(["gm"])], global_param_names=["gm"],
            global_apriori_cov=apriori,
        )

    @pytest.mark.parametrize("arc_data, n_G, exp_P_G, exp_x_G", [
        # Single arc, no prior: P_G = 1/N_tilde_g, x_G = g_bar + P_G @ B
        (
            [{"N_tilde_g": np.array([[2.0]]), "B_tilde_g": np.array([3.0]), "x_G_abs": np.array([10.0])}],
            1, np.array([[0.5]]), np.array([11.5]),
        ),
        # Two arcs: Lambda = 2 + 3 = 5, N_hat = 1 + (2 + 3*(6-5)) = 6
        (
            [
                {"N_tilde_g": np.array([[2.0]]), "B_tilde_g": np.array([1.0]), "x_G_abs": np.array([5.0])},
                {"N_tilde_g": np.array([[3.0]]), "B_tilde_g": np.array([2.0]), "x_G_abs": np.array([6.0])},
            ],
            1, np.array([[0.2]]), np.array([6.2]),
        ),
    ])
    def test_combine_analytical(self, arc_data, n_G, exp_P_G, exp_x_G):
        x_G, p_g = self.make_multi()._combine_global(arc_data, n_G)
        np.testing.assert_allclose(p_g, exp_P_G, rtol=1e-10)
        np.testing.assert_allclose(x_G, exp_x_G, rtol=1e-10)

    def test_apriori_cov_tightens_P_G(self):
        arc_data = [{"N_tilde_g": np.array([[2.0]]), "B_tilde_g": np.array([1.0]), "x_G_abs": np.array([5.0])}]
        _, p_no = self.make_multi()._combine_global(arc_data, 1)
        _, p_with = self.make_multi(apriori=np.array([[1.0]]))._combine_global(arc_data, 1)
        assert p_with[0, 0] < p_no[0, 0]

    @pytest.mark.parametrize("n_G", [1, 2, 3])
    def test_output_shapes(self, n_G):
        arc_data = [{"N_tilde_g": np.eye(n_G), "B_tilde_g": np.ones(n_G), "x_G_abs": np.zeros(n_G)}]
        x_G, p_g = self.make_multi()._combine_global(arc_data, n_G)
        assert x_G.shape == (n_G,)
        assert p_g.shape == (n_G, n_G)


class TestSolveLocal:

    def test_no_local_params_returns_empty(self):
        arc_d = {"N_a": np.zeros((0, 0)), "N_ag": np.zeros((0, 1)), "B_a": np.zeros(0)}
        delta_a, p_a = scb.MultiFilterOD._solve_local(arc_d, np.array([[0.5]]), np.array([2.0]))
        assert delta_a.shape == (0,)
        assert p_a.shape == (0, 0)

    def test_analytical_1d(self):
        """N_a=4, N_ag=1, B_a=3, P_G=0.5, B_hat_g=2 → analytical delta_a and P_a."""
        arc_d = {"N_a": np.array([[4.0]]), "N_ag": np.array([[1.0]]), "B_a": np.array([3.0])}
        delta_a, p_a = scb.MultiFilterOD._solve_local(arc_d, np.array([[0.5]]), np.array([2.0]))
        np.testing.assert_allclose(p_a, [[1.0 / 3.5]], rtol=1e-10)
        np.testing.assert_allclose(delta_a, [2.0 / 3.5], rtol=1e-10)

    def test_zero_cross_term_decouples_local(self):
        """N_ag=0 → delta_a = N_a^{-1} B_a, independent of global."""
        arc_d = {"N_a": np.diag([2.0, 4.0]), "N_ag": np.zeros((2, 1)), "B_a": np.array([6.0, 8.0])}
        delta_a, _ = scb.MultiFilterOD._solve_local(arc_d, np.eye(1), np.zeros(1))
        np.testing.assert_allclose(delta_a, [3.0, 2.0], atol=1e-12)

    @pytest.mark.parametrize("n_local", [1, 2, 3])
    def test_output_shapes(self, n_local):
        arc_d = {"N_a": np.eye(n_local), "N_ag": np.zeros((n_local, 1)), "B_a": np.zeros(n_local)}
        delta_a, p_a = scb.MultiFilterOD._solve_local(arc_d, np.eye(1), np.zeros(1))
        assert delta_a.shape == (n_local,)
        assert p_a.shape == (n_local, n_local)


class TestFitCallback:
    """
    fit(callback=...) runs once per outer iteration, after its convergence test
    and before the arcs are reinitialised, so that what it changes is picked up
    when the arcs rebuild their datasets.
    """

    @staticmethod
    def _multi(log):
        """Two arcs whose numerics are stubbed; the global estimate never moves."""
        def arc(i):
            f = stub(["gm"])
            f.n = 7
            f.apriori_covariance_est = np.eye(7)
            f.solution = types.SimpleNamespace(arc=i)

            def fit(**kwargs):
                log.append(("fit", i))
                return f.solution, 1, True

            f.fit = fit
            f._cost_function = lambda: (1.0, 10)
            return f

        multi = scb.MultiFilterOD([arc(0), arc(1)], ["gm"])
        multi._extract_dx_P_at_t0 = lambda filt, sol: (np.zeros(7), np.eye(7))
        multi._normal_equation_blocks = lambda *a, **k: {
            "B_tilde_g": np.zeros(1), "N_tilde_g": np.eye(1)}
        multi._extract_global_abs = lambda filt, idx: np.array([1.0])
        multi._combine_global = lambda arc_data, n_G: (np.array([1.0]), np.eye(1))
        multi._solve_local = lambda d, P_G, B: (np.zeros(0), np.zeros((0, 0)))
        multi._reinitialize_arc = (
            lambda filt, d, da, dG, traj_name=None, rebuild=False:
            log.append(("reinit", filt.solution.arc, rebuild))
        )
        return multi

    def test_runs_between_the_test_and_the_reinitialisation(self):
        log = []
        multi = self._multi(log)

        def callback(m, outer, arc_solutions):
            assert m is multi
            assert [sol.arc for sol in arc_solutions] == [0, 1]
            log.append(("callback", outer))

        multi.fit(max_outer_iters=3, convergence_criterion="deviation",
                  verbose=False, callback=callback)

        assert log == [
            ("fit", 0), ("fit", 1), ("callback", 1),
            ("reinit", 0, True), ("reinit", 1, True),
            ("fit", 0), ("fit", 1), ("callback", 2),
        ]

    def test_without_a_callback_a_zero_correction_is_not_rebuilt(self):
        log = []
        multi = self._multi(log)

        solutions = multi.fit(max_outer_iters=3, convergence_criterion="deviation",
                              verbose=False)

        assert ("reinit", 0, False) in log
        assert solutions[0].multi_arc_global["converged"]
        assert solutions[0].multi_arc_global["n_outer"] == 2

    @pytest.mark.parametrize("rebuild, rebuilt", [(False, False), (True, True)])
    def test_rebuild_reinitialises_a_zero_correction(self, rebuild, rebuilt):
        calls = []
        filt = types.SimpleNamespace(
            _reinitialize=lambda **kwargs: calls.append(kwargs))
        arc_d = {"n_full": 7, "local_full_idx": list(range(6)),
                 "global_full_idx": [6]}
        multi = scb.MultiFilterOD([stub(["gm"])], ["gm"])

        multi._reinitialize_arc(filt, arc_d, np.zeros(6), np.zeros(1),
                                traj_name="arc.bsp", rebuild=rebuild)

        assert bool(calls) is rebuilt

# endregion Tests #
