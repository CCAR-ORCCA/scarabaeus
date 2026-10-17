# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import pytest
import src.scarabaeus as scb

# --------------------#
# region    Fixtures  #
# --------------------#

N = 6


def eye(n=N):
    return np.eye(n)


def c(vals, n=N):
    v = np.zeros(n)
    v[: len(vals)] = vals
    return v

# endregion Fixtures #

# ---------------#
# region  Tests  #
# ---------------#


class TestMahalanobis:

    @pytest.mark.parametrize("x, mean, cov, expected", [
        (c([0.0]),       c([0.0]),  eye(),       0.0),   # x == mean
        (c([1.0]),       c([0.0]),  eye(),       1.0),   # unit separation, unit cov
        (c([2.0]),       c([0.0]),  eye(),       2.0),   # 2× separation
        (c([5.0]),       c([0.0]),  eye(),       5.0),   # 5× separation
        (c([2.0]),       c([0.0]),  4.0 * eye(), 1.0),   # scaled cov: d = 2/2 = 1
        (c([1.5, -0.5]), c([0.3, 1.0]), 2.0 * eye(), None),  # general — just check float
    ])
    def test_mahalanobis(self, x, mean, cov, expected):
        d = scb.UtilsOD.mahalanobis(x, mean, cov)
        assert isinstance(d, float)
        if expected is not None:
            assert d == pytest.approx(expected, rel=1e-10)

    def test_equals_nees_square_root(self):
        x, mean, cov = c([1.5]), c([0.3]), 2.0 * eye()
        d = scb.UtilsOD.mahalanobis(x, mean, cov)
        eps = scb.UtilsOD.nees(x, mean, cov)
        assert d == pytest.approx(np.sqrt(eps), rel=1e-10)


class TestNEES:

    @pytest.mark.parametrize("x_true, c_est, cov, expected", [
        (c([0.0]), c([0.0]), eye(),       0.0),  # no error
        (c([1.0]), c([0.0]), eye(),       1.0),  # unit displacement, unit cov
        (c([3.0]), c([0.0]), eye(),       9.0),  # 3× displacement → NEES = 9
        (c([2.0]), c([0.0]), 4.0 * eye(), 1.0),  # scaled cov: (2²)/4 = 1
    ])
    def test_nees(self, x_true, c_est, cov, expected):
        eps = scb.UtilsOD.nees(x_true, c_est, cov)
        assert isinstance(eps, float)
        assert eps == pytest.approx(expected, rel=1e-10)


class TestMerge:

    @pytest.mark.parametrize("c1, cov1, c2, cov2, expected_c, expected_cov_diag", [
        # Equal means preserved, equal cov → merged cov = cov/2
        (c([2.0]), eye(), c([2.0]), eye(), c([2.0]), 0.5 * np.ones(N)),
        # Equal cov → merged mean is arithmetic mean
        (c([1.0]), eye(), c([3.0]), eye(), c([2.0]), 0.5 * np.ones(N)),
        # Unequal cov (scalar 1D): c_m=0.5, P_m=0.75
        (np.zeros(1), np.eye(1), np.array([2.0]), 3.0 * np.eye(1), np.array([0.5]), np.array([0.75])),
    ])
    def test_merged_mean_and_cov(self, c1, cov1, c2, cov2, expected_c, expected_cov_diag):
        c_m, cov_m = scb.UtilsOD.merge(c1, cov1, c2, cov2)
        np.testing.assert_allclose(c_m, expected_c, atol=1e-10)
        np.testing.assert_allclose(np.diag(cov_m), expected_cov_diag, atol=1e-10)

    @pytest.mark.parametrize("cov1_scale, cov2_scale", [
        (1.0, 1.0),
        (2.0, 5.0),
        (4.0, 9.0),
    ])
    def test_merged_variance_tighter_than_both(self, cov1_scale, cov2_scale):
        cov1, cov2 = cov1_scale * eye(), cov2_scale * eye()
        _, cov_m = scb.UtilsOD.merge(np.zeros(N), cov1, np.zeros(N), cov2)
        assert np.all(np.diag(cov_m) < np.diag(cov1))
        assert np.all(np.diag(cov_m) < np.diag(cov2))

    def test_result_shapes(self):
        c_m, cov_m = scb.UtilsOD.merge(np.zeros(N), eye(), np.zeros(N), eye())
        assert c_m.shape == (N,)
        assert cov_m.shape == (N, N)

    def test_merged_covariance_symmetric(self):
        _, cov_m = scb.UtilsOD.merge(c([1.0]), 2.0 * eye(), c([3.0]), 5.0 * eye())
        np.testing.assert_allclose(cov_m, cov_m.T, atol=1e-12)


class TestMergeMany:

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            scb.UtilsOD.merge_many([(np.zeros(N), eye())])

    def test_two_solutions_matches_merge(self):
        c1, c2 = c([1.0]), c([3.0])
        cov1, cov2 = 2.0 * eye(), 4.0 * eye()
        c_direct, cov_direct = scb.UtilsOD.merge(c1, cov1, c2, cov2)
        c_many, cov_many = scb.UtilsOD.merge_many([(c1, cov1), (c2, cov2)])
        np.testing.assert_allclose(c_many, c_direct, atol=1e-12)
        np.testing.assert_allclose(cov_many, cov_direct, atol=1e-12)

    @pytest.mark.parametrize("n_sols", [3, 4, 5])
    def test_result_shapes_n_solutions(self, n_sols):
        sols = [(c([float(i)]), eye()) for i in range(n_sols)]
        c_m, cov_m = scb.UtilsOD.merge_many(sols)
        assert c_m.shape == (N,)
        assert cov_m.shape == (N, N)

    def test_n_equal_solutions_covariance(self):
        # 3 identical covariances P → merged = P/3
        cov = 6.0 * eye()
        sols = [(np.zeros(N), cov.copy()) for _ in range(3)]
        _, cov_m = scb.UtilsOD.merge_many(sols)
        np.testing.assert_allclose(cov_m, 2.0 * eye(), atol=1e-10)


class TestCloseness:

    def test_identical_solutions_zero(self):
        v = c([1.0, 2.0])
        res = scb.UtilsOD.closeness(v, eye(), v, eye())
        assert res["d"] == pytest.approx(0.0, abs=1e-12)
        assert res["t_sq"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("separation, expected_d", [
        (1.0, 0.5),   # |dc|=1 → d=0.5
        (2.0, 1.0),   # |dc|=2 → d=1.0 (boundary)
        (4.0, 2.0),   # |dc|=4 → d=2.0
    ])
    def test_equal_cov_d_equals_half_separation(self, separation, expected_d):
        # For cov1=cov2=I: d = |dc|/2
        c1 = np.zeros(N)
        c2 = c([separation])
        res = scb.UtilsOD.closeness(c1, eye(), c2, eye())
        assert res["d"] == pytest.approx(expected_d, rel=1e-3)

    def test_return_keys_present(self):
        res = scb.UtilsOD.closeness(np.zeros(N), eye(), c([2.0]), eye())
        for key in ("d", "t_sq", "alpha", "x_tang", "all_roots", "all_t_sq"):
            assert key in res

    def test_d_equals_sqrt_t_sq(self):
        res = scb.UtilsOD.closeness(np.zeros(N), eye(), c([2.0]), eye())
        assert res["d"] == pytest.approx(np.sqrt(max(res["t_sq"], 0.0)), rel=1e-10)

    def test_tangency_at_midpoint_equal_cov(self):
        c1, c2 = np.zeros(N), c([2.0])
        res = scb.UtilsOD.closeness(c1, eye(), c2, eye())
        np.testing.assert_allclose(res["x_tang"][0], 1.0, atol=1e-3)

    def test_symmetric(self):
        c1, c2 = c([0.0]), c([3.0])
        cov1, cov2 = 2.0 * eye(), 5.0 * eye()
        d12 = scb.UtilsOD.closeness(c1, cov1, c2, cov2)["d"]
        d21 = scb.UtilsOD.closeness(c2, cov2, c1, cov1)["d"]
        assert d12 == pytest.approx(d21, rel=1e-4)

    def test_no_root_warns_and_returns_nan(self):
        # Root for cov1=cov2=I is at alpha=1; alpha_range (100, 1000) misses it
        c1, c2 = np.zeros(N), c([1.0])
        with pytest.warns(UserWarning):
            res = scb.UtilsOD.closeness(c1, eye(), c2, eye(), alpha_range=(100.0, 1000.0))
        assert np.isnan(res["d"])


class TestClosenessMatrix:

    @pytest.mark.parametrize("n_sols", [2, 3, 4])
    def test_shape(self, n_sols):
        sols = [(np.zeros(N), eye())] * n_sols
        d_mat, _ = scb.UtilsOD.closeness_matrix(sols, verbose=False)
        assert d_mat.shape == (n_sols, n_sols)

    def test_diagonal_is_zero(self):
        sols = [(c([float(i)]), eye()) for i in range(3)]
        d_mat, _ = scb.UtilsOD.closeness_matrix(sols, verbose=False)
        np.testing.assert_allclose(np.diag(d_mat), 0.0, atol=1e-12)

    def test_symmetric(self):
        sols = [(c([float(i)]), eye()) for i in range(3)]
        d_mat, _ = scb.UtilsOD.closeness_matrix(sols, verbose=False)
        np.testing.assert_allclose(d_mat, d_mat.T, atol=1e-10)

    @pytest.mark.parametrize("n, expected_labels", [
        (2, ["S1", "S2"]),
        (3, ["S1", "S2", "S3"]),
    ])
    def test_default_labels(self, n, expected_labels):
        sols = [(np.zeros(N), eye())] * n
        _, labels = scb.UtilsOD.closeness_matrix(sols, verbose=False)
        assert labels == expected_labels

    def test_custom_labels_preserved(self):
        sols = [(np.zeros(N), eye())] * 2
        _, labels = scb.UtilsOD.closeness_matrix(sols, labels=["A", "B"], verbose=False)
        assert labels == ["A", "B"]

    def test_offdiagonal_matches_pairwise_closeness(self):
        c1, c2, cov = c([0.0]), c([2.0]), eye()
        d_mat, _ = scb.UtilsOD.closeness_matrix([(c1, cov), (c2, cov)], verbose=False)
        expected = scb.UtilsOD.closeness(c1, cov, c2, cov)["d"]
        assert d_mat[0, 1] == pytest.approx(expected, rel=1e-10)
        assert d_mat[1, 0] == pytest.approx(expected, rel=1e-10)


class TestSigmaOverlap:

    @pytest.mark.parametrize("c1, c2, cov1, cov2, n_sigma, expected", [
        (c([0.0]),    c([0.0]),    eye(), eye(), 1.0, True),   # same mean
        (c([0.0]),    c([1000.0]), eye(), eye(), 1.0, False),  # far apart
        (c([0.0]),    c([1.5]),    eye(), eye(), 1.0, True),   # r1+r2 > 1
        (c([0.0]),    c([3.0]),    eye(), eye(), 1.0, False),  # r1+r2 < 1
        (c([0.0]),    c([3.0]),    eye(), eye(), 3.0, True),   # larger n_sigma → overlap
    ])
    def test_sigma_overlap(self, c1, c2, cov1, cov2, n_sigma, expected):
        result = scb.UtilsOD.sigma_overlap(c1, cov1, c2, cov2, n_sigma=n_sigma)
        assert isinstance(result, bool)
        assert result is expected

    def test_symmetric(self):
        c1, c2, cov = c([0.0]), c([1.5]), eye()
        assert scb.UtilsOD.sigma_overlap(c1, cov, c2, cov) == scb.UtilsOD.sigma_overlap(c2, cov, c1, cov)


class TestAsCP:

    @pytest.mark.parametrize("sol, expected_c, expected_cov", [
        ((np.array([1.0, 2.0]), np.eye(2)), np.array([1.0, 2.0]), np.eye(2)),
        ((np.array([[1.0, 2.0]]), np.eye(2)), np.array([1.0, 2.0]), np.eye(2)),  # 2D c → ravelled
    ])
    def test_tuple_passthrough(self, sol, expected_c, expected_cov):
        c_out, cov_out = scb.UtilsOD._as_c_P(sol)
        np.testing.assert_allclose(c_out, expected_c)
        np.testing.assert_allclose(cov_out, expected_cov)
        assert c_out.ndim == 1

    @pytest.mark.parametrize("bad_input", [
        "not_a_solution",
        42,
        (np.zeros(3),),                          # tuple too short
        (np.zeros(3), np.eye(3), np.eye(3)),     # tuple too long
    ])
    def test_wrong_type_raises(self, bad_input):
        with pytest.raises((TypeError, Exception)):
            scb.UtilsOD._as_c_P(bad_input)

# endregion Tests #
