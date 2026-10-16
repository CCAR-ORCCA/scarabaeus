# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Unit Tests for the MeasurementRegistry class  #
# ================================================#
import pytest

import src.scarabaeus as scb


# --------------------#
# region    Fixtures #
# --------------------#
class Station:
    """Stand-in for a ground station; only its name is read."""

    def __init__(self, name):
        self.name = name


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestRegistry:
    def test_known_types_resolve(self):
        for name in ("SeqRange", "Doppler", "IdealRange", "IdealRangeRate", "Centroid"):
            assert scb.MeasurementRegistry.get(name).model_class is not None

    def test_type_carries_block_and_units(self):
        """Keeping units with the type is what stops a sigma and its
        observations being built in different units."""
        sr = scb.MeasurementRegistry.get("SeqRange")
        dop = scb.MeasurementRegistry.get("Doppler")
        assert sr.meas_name == "2W_sranging"
        assert dop.meas_name == "2W_doppler"
        assert sr.units is not dop.units

    def test_labels_keep_plotting_substrings(self):
        """Plotting classifies on 'Range Rate' before 'Range'."""
        assert scb.MeasurementRegistry.get("Doppler").label == "Range Rate"
        assert scb.MeasurementRegistry.get("SeqRange").label == "Range"

    def test_unknown_type_lists_options(self):
        with pytest.raises(KeyError, match="SeqRange"):
            scb.MeasurementRegistry.get("NotAType")

    def test_extensible(self):
        scb.MeasurementRegistry.register(
            "TestType", scb.MeasurementType(object, "blk", None, "Range"))
        assert scb.MeasurementRegistry.get("TestType").meas_name == "blk"
        del scb.MeasurementRegistry.all()["TestType"]


class TestDatasetNaming:
    def test_default_name_from_station_and_label(self):
        s = scb.MeasurementStream("f", "SeqRange", Station("DSS-34"))
        assert scb.MeasurementRegistry.assign_dataset_names([s]) == ["DSS-34 Range"]

    def test_counter_appended_on_repeat(self):
        """Datasets are keyed by name in the solution, so repeats must not
        collide. The multi-pass scripts hit this with eight streams."""
        gs = Station("DSS-25")
        streams = [scb.MeasurementStream(f"p{i}", "SeqRange", gs) for i in range(3)]
        assert scb.MeasurementRegistry.assign_dataset_names(streams) == [
            "DSS-25 Range", "DSS-25 Range 2", "DSS-25 Range 3"]

    def test_counter_is_per_station_and_type(self):
        a, b = Station("DSS-24"), Station("DSS-25")
        streams = [
            scb.MeasurementStream("p1", "SeqRange", a),
            scb.MeasurementStream("p2", "SeqRange", b),
            scb.MeasurementStream("p3", "SeqRange", a),
            scb.MeasurementStream("p4", "Doppler", a),
        ]
        assert scb.MeasurementRegistry.assign_dataset_names(streams) == [
            "DSS-24 Range", "DSS-25 Range", "DSS-24 Range 2", "DSS-24 Range Rate"]

    def test_counter_keeps_plotting_substring_intact(self):
        gs = Station("DSS-25")
        streams = [scb.MeasurementStream(f"p{i}", "Doppler", gs) for i in range(2)]
        for name in scb.MeasurementRegistry.assign_dataset_names(streams):
            assert "Range Rate" in name

    def test_explicit_name_wins(self):
        s = scb.MeasurementStream("f", "SeqRange", Station("DSS-34"),
                                  name="GS1 Real Range")
        assert scb.MeasurementRegistry.assign_dataset_names([s]) == ["GS1 Real Range"]

    def test_explicit_names_deduplicated(self):
        gs = Station("DSS-34")
        streams = [scb.MeasurementStream("a", "SeqRange", gs, name="Same"),
                   scb.MeasurementStream("b", "SeqRange", gs, name="Same")]
        assert scb.MeasurementRegistry.assign_dataset_names(streams) == ["Same", "Same 2"]


# endregion Tests #
# --------------#
