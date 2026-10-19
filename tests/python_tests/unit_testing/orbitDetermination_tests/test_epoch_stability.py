# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Filter epochs must not depend on the last bit of t2.

Each measurement model solves light time for its own t2, so a range point and a
Doppler point received at the same t3 agree on t2 only to rounding. Whether they
round to the same float can change from one iteration of FilterOD.fit to the
next, which used to change the number of filter epochs and crash the fit when it
added the per-epoch deviations of two iterations.
"""
import warnings

import numpy as np
import pytest

import scarabaeus as scb
from scarabaeus import MeasurementDataSet
from scarabaeus.orbitDetermination.FilterDataManager import FilterDataManager
from scarabaeus.orbitDetermination.FilterOD import FilterOD


NAMES = ["t1", "t2", "t3", "computed", "observed",
         "residuals", "partials", "sigma", "outlier_flag"]

# The epoch that broke a real DSS-34 fit, and one ulp at that magnitude
T = 661296844.8702171
ULP = np.spacing(T)


def _dataset(name, t2, flags=None):
    n = len(t2)
    flags = np.zeros(n) if flags is None else flags
    values = [
        np.asarray(t2, float) - 1.0,
        np.asarray(t2, float),
        np.asarray(t2, float) + 1.0,
        np.zeros(n), np.zeros(n), np.zeros(n),
        np.zeros((n, 6)), np.ones(n), np.asarray(flags, float),
    ]
    return MeasurementDataSet(
        set_name=name, measurements=values, names=NAMES,
        instrument=scb.GroundStation("DSS-24"), target=None,
        epochs_t3=None, measurement_type="TestRange",
    )


class TestT2Grouping:
    @pytest.mark.parametrize("offset", [0.0, ULP, -ULP, 5e-7])
    def test_near_equal_t2_share_one_epoch(self, offset):
        """The epoch count is the same whether or not the two t2 round equal."""
        rng = _dataset("range", [T - 60.0, T + offset])
        dop = _dataset("doppler", [T, T + 60.0])
        idx = FilterDataManager._combine_indices_by_t2([rng, dop])

        assert len(idx) == 3
        key = min(T, T + offset)
        assert [e["dataset"].set_name for e in idx[key]] == ["range", "doppler"]

    def test_block_keeps_each_measurement_t2(self):
        rng = _dataset("range", [T + ULP])
        dop = _dataset("doppler", [T])
        fdm = FilterDataManager(datasets=[rng, dop])
        (key,) = fdm.get_spacecraft_times()
        block = fdm.get_combined_for_t2(key)[key]

        assert key == T
        assert np.array_equal(np.sort(block["t2"]), [T, T + ULP])

    def test_separate_epochs_not_merged(self):
        ds = _dataset("d", [T, T + 2 * FilterDataManager.T2_MERGE_TOL])
        assert len(FilterDataManager._combine_indices_by_t2([ds])) == 2

    def test_groups_do_not_chain(self):
        """Points each within the tolerance of the next stay in separate groups
        once they are more than the tolerance from the group's first point."""
        tol = FilterDataManager.T2_MERGE_TOL
        ds = _dataset("d", [T, T + 0.6 * tol, T + 1.2 * tol])
        assert len(FilterDataManager._combine_indices_by_t2([ds])) == 2

    def test_all_flagged_gives_no_epochs(self):
        ds = _dataset("d", [T, T + 60.0], flags=[1, 1])
        assert FilterDataManager._combine_indices_by_t2([ds]) == {}


class TestAccumulateDeviation:
    acc = staticmethod(FilterOD._accumulate_deviation)

    def test_first_iteration_is_a_copy(self):
        dev = np.ones((3, 6))
        out = self.acc(None, None, dev, np.array([0.0, 60.0, 120.0]))
        assert np.array_equal(out, dev) and out is not dev

    def test_rows_paired_by_epoch_when_epochs_shift(self):
        """t2 moves a little with the reference trajectory; rows still pair up."""
        prev = np.arange(3.0)[:, None] * np.ones((1, 6))
        out = self.acc(prev, np.array([0.0, 60.0, 120.0]),
                       np.ones((3, 6)), np.array([0.003, 60.002, 120.001]))
        assert np.array_equal(out, prev + 1)

    def test_epoch_count_change_does_not_raise(self):
        """The failure seen on real tracking data: 309 epochs, then 310."""
        prev = np.ones((3, 6))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = self.acc(prev, np.array([0.0, 60.0, 120.0]),
                           np.full((4, 6), 2.0), np.array([0.0, 60.0, 60.0 + 1e-7, 120.0]))
        assert out.shape == (4, 6)
        assert np.array_equal(out[[0, 3]], np.full((2, 6), 3.0))
        # One of the two near-equal epochs pairs with 60.0, the other has no partner
        assert np.isnan(out[1:3]).all(axis=1).sum() == 1
        assert any("no counterpart" in str(x.message) for x in w)
