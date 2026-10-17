# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==============================================#
#  Unit Tests for the FilterDataManager class  #
# ==============================================#
import src.scarabaeus as scb

import numpy as np
import pytest

# --------------------#
# region    Fixtures #
# --------------------#
NAMES = ["t1", "t2", "t3", "computed", "observed",
         "residuals", "partials", "sigma", "outlier_flag"]


def _dataset(name, t2, flags, residuals=None):
    """Minimal MeasurementDataSet with controllable t2 and outlier flags."""
    n = len(t2)
    residuals = np.zeros(n) if residuals is None else np.asarray(residuals, float)
    values = [
        np.asarray(t2, float) - 1.0,     # t1
        np.asarray(t2, float),           # t2
        np.asarray(t2, float) + 1.0,     # t3
        np.zeros(n),                     # computed
        residuals.copy(),                # observed
        residuals,                       # residuals
        np.zeros((n, 6)),                # partials
        np.ones(n),                      # sigma
        np.asarray(flags, float),        # outlier_flag
    ]
    return scb.MeasurementDataSet(
        set_name=name, measurements=values, names=NAMES,
        instrument=scb.GroundStation("DSS-24"), target=None,
        epochs_t3=None, measurement_type="TestRange",
    )

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that object is constructed correctly.
    """
    pytest.skip()


# Outlier gating
class TestFilterGating:
    def test_flagged_measurements_excluded(self):
        ds = _dataset("d", t2=[10.0, 20.0, 30.0], flags=[0, 1, 0])
        idx = scb.FilterDataManager._combine_indices_by_t2([ds])

        assert sorted(idx) == [10.0, 30.0], "flagged t2 must not appear"
        for entries in idx.values():
            for e in entries:
                assert 1 not in e["indices"], "flagged index must not be selected"

    def test_fully_flagged_epoch_dropped_not_empty(self):
        """A t2 with every measurement flagged must vanish, not survive empty.

        get_spacecraft_times() is built from these keys and the filters then
        index straight back in, so an empty entry would raise rather than
        simply skip the epoch.
        """
        ds = _dataset("d", t2=[10.0, 20.0], flags=[1, 0])
        idx = scb.FilterDataManager._combine_indices_by_t2([ds])

        assert 10.0 not in idx
        mgr = scb.FilterDataManager([ds])
        assert 10.0 not in mgr.get_spacecraft_times()

    def test_shared_epoch_keeps_unflagged_dataset(self):
        """Two datasets at one t2; flagging one must not drop the other."""
        a = _dataset("a", t2=[10.0], flags=[1])
        b = _dataset("b", t2=[10.0], flags=[0])
        idx = scb.FilterDataManager._combine_indices_by_t2([a, b])

        assert 10.0 in idx
        names = [e["dataset"].set_name for e in idx[10.0]]
        assert names == ["b"]

    def test_unflagged_data_passes_through_unchanged(self):
        ds = _dataset("d", t2=[10.0, 20.0, 30.0], flags=[0, 0, 0])
        idx = scb.FilterDataManager._combine_indices_by_t2([ds])
        assert sorted(idx) == [10.0, 20.0, 30.0]

    def test_missing_outlier_flag_column_tolerated(self):
        """Datasets built without an outlier_flag column still combine."""
        ds = _dataset("d", t2=[10.0, 20.0], flags=[0, 0])
        del ds.data["outlier_flag"]
        idx = scb.FilterDataManager._combine_indices_by_t2([ds])
        assert sorted(idx) == [10.0, 20.0]

# endregion Tests #
# --------------#
