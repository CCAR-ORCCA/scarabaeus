# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Tests for the Utils state-space change (Q reorder/selection) operators."""
import numpy as np
import pytest

from scarabaeus.utils.Utils import Utils


# --------------------#
# region    Fixtures #
# --------------------#
class _Owner:
    """Minimal stand-in for a state-component owner with a SPICE id."""

    def __init__(self, spice_id):
        self.spice_id = spice_id


def _blk(name, size, owner):
    """A state-definition entry: (name, size, est, dyn, owner, awf)."""
    return (name, size, "est", "dyn", owner, None)


# endregion Fixtures #
# --------------------#

# --------------#
# region Tests #
# --------------#
class TestBuildReorderOperator:
    """Cases for build_reorder_operator and the Q it produces."""

    def test_delete_and_swap(self):
        """
        Reproduce the Chapter-8 example: from a 5-scalar state, delete the 2nd
        variable and swap the 3rd and 4th. The resulting Q must equal the textbook
        operator, and ``X' = Q X`` / ``P' = Q P Q^T`` must follow.
        """
        o = _Owner(-1)
        old = [_blk(n, 1, o) for n in ("a", "b", "c", "d", "e")]
        new = [_blk(n, 1, o) for n in ("a", "d", "c", "e")]  # drop b, swap c/d

        Q, new_n, new_blocks = Utils.build_reorder_operator(old, new)

        expected_Q = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        assert new_n == 4
        assert new_blocks == []  # nothing augmented
        np.testing.assert_array_equal(Q, expected_Q)

        X = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        np.testing.assert_array_equal(
            Utils.reorder_vector(Q, X), np.array([[10.0], [40.0], [30.0], [50.0]])
        )

        P = np.arange(25, dtype=float).reshape(5, 5)
        P = P + P.T  # symmetric
        np.testing.assert_array_equal(Utils.reorder_covariance(Q, P), Q @ P @ Q.T)

    def test_is_similarity_transform(self):
        """Pure reorder: P' = Q P Q^T and STM' = Q STM Q^T exactly."""
        o = _Owner(-1000)
        old = [_blk("position", 3, o), _blk("velocity", 3, o), _blk("Cd", 1, o)]
        new = [_blk("position", 3, o), _blk("Cd", 1, o), _blk("velocity", 3, o)]
        Q, _n, new_blocks = Utils.build_reorder_operator(old, new)
        assert new_blocks == []

        rng = np.random.default_rng(0)
        A = rng.standard_normal((7, 7))
        P = A @ A.T
        STM = rng.standard_normal((7, 7))
        np.testing.assert_allclose(Utils.reorder_covariance(Q, P), Q @ P @ Q.T, atol=0, rtol=0)
        np.testing.assert_allclose(Utils.reorder_stm(Q, STM), Q @ STM @ Q.T, atol=0, rtol=0)

    def test_contraction_drops_states(self):
        """Dropping a state removes its row/column from the covariance."""
        o = _Owner(-1000)
        old = [_blk("position", 3, o), _blk("velocity", 3, o), _blk("Cd", 1, o)]
        new = [_blk("position", 3, o), _blk("velocity", 3, o)]  # drop Cd
        Q, new_n, new_blocks = Utils.build_reorder_operator(old, new)
        assert new_n == 6 and new_blocks == []

        P = np.diag([1.0, 2, 3, 4, 5, 6, 99])  # 99 == Cd variance, must vanish
        P2 = Utils.reorder_covariance(Q, P)
        assert P2.shape == (6, 6)
        np.testing.assert_array_equal(np.diag(P2), [1, 2, 3, 4, 5, 6])

    def test_expansion_adds_apriori(self):
        """A newly augmented state: zero mean, a-priori variance, identity STM."""
        o = _Owner(-1000)
        old = [_blk("position", 3, o), _blk("velocity", 3, o)]
        new = [_blk("position", 3, o), _blk("velocity", 3, o), _blk("Cd", 1, o)]
        Q, new_n, new_blocks = Utils.build_reorder_operator(old, new)
        assert new_n == 7
        assert new_blocks == [("Cd", 6, 1)]  # appended at offset 6

        # deviation: new state starts at zero
        X = np.arange(1, 7, dtype=float).reshape(6, 1)
        np.testing.assert_array_equal(
            Utils.reorder_vector(Q, X).ravel(), [1, 2, 3, 4, 5, 6, 0]
        )

        # covariance: a-priori variance on the new diagonal, uncorrelated
        P = np.eye(6)
        P2 = Utils.reorder_covariance(Q, P, new_block_variances=[(6, np.array([0.49]))])
        assert P2.shape == (7, 7)
        assert P2[6, 6] == pytest.approx(0.49)
        assert np.all(P2[6, :6] == 0) and np.all(P2[:6, 6] == 0)

        # STM: identity block for the new state
        STM = np.full((6, 6), 2.0)
        S2 = Utils.reorder_stm(Q, STM, new_blocks)
        assert S2[6, 6] == 1.0
        assert np.all(S2[6, :6] == 0) and np.all(S2[:6, 6] == 0)

    def test_multibody_occurrence_keying(self):
        """Repeated names across bodies are matched by (name, spice_id, occurrence)."""
        a, b = _Owner(-1000), _Owner(-1001)
        old = [_blk("position", 3, a), _blk("position", 3, b)]
        new = [_blk("position", 3, b), _blk("position", 3, a)]  # swap the two bodies
        Q, _n, new_blocks = Utils.build_reorder_operator(old, new)
        assert new_blocks == []

        P = np.diag([1.0, 1, 1, 9, 9, 9])  # body a -> 1, body b -> 9
        P2 = Utils.reorder_covariance(Q, P)
        # after swap, body b (9) comes first, body a (1) second
        np.testing.assert_array_equal(np.diag(P2), [9, 9, 9, 1, 1, 1])


# endregion Tests #
# --------------#
