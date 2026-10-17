# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ======================================#
#  Unit Tests for the Maneuver class   #
# ======================================#
import numpy as np
import pytest

import src.scarabaeus as scb

kg, sec, N = scb.Units.get_units(["kg", "sec", "N"])

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def maneuver():
    """
    Full maneuver with already-7-element mass_flow, ux, uy, uz inputs,
    exercising each setter's non-padding branch.
    """
    return scb.Maneuver(
        thrust=scb.ArrayWUnits(0.5, N),
        mass_flow=scb.ArrayWUnits(np.array([3e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), kg / sec),
        start_time=scb.EpochArray(1000.0, "TDB"),
        end_time=scb.EpochArray(2000.0, "TDB"),
        ux=scb.ArrayWUnits(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), None),
        uy=scb.ArrayWUnits(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), None),
        uz=scb.ArrayWUnits(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), None),
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(maneuver):
    """
    Verifies correct attribute assignment on construction, including
    all property getters and the mass_flow/ux/uy/uz setters'
    non-padding branch (already-7-element input, stored descending
    with the constant term at index 6).
    """
    assert maneuver.thrust is not None

    assert np.allclose(maneuver.mass_flow.values, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3e-4])

    assert maneuver.start_time.times.values == pytest.approx(1000.0)
    assert maneuver.end_time.times.values == pytest.approx(2000.0)

    assert np.allclose(maneuver.ux.values, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(maneuver.uy.values, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(maneuver.uz.values, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestErrorHandling:
    @pytest.mark.parametrize(
        "case",
        ["not_epoch_array", "negative", "start_ge_end"],
        ids=["Not EpochArray", "Negative Epoch", "start_time >= end_time"],
    )
    def test_start_time(self, case):
        """
        Verifies TypeError for a non-EpochArray value, ValueError for a
        negative epoch, and ValueError when start_time >= end_time.
        """
        match case:
            case "not_epoch_array":
                m = scb.Maneuver()
                with pytest.raises(TypeError, match="Start time must be EpochArray"):
                    m.start_time = "not an epoch"
            case "negative":
                m = scb.Maneuver()
                with pytest.raises(ValueError, match="Start time cannot be negative"):
                    m.start_time = scb.EpochArray(-5.0, "TDB")
            case "start_ge_end":
                m = scb.Maneuver(end_time=scb.EpochArray(100.0, "TDB"))
                with pytest.raises(
                    ValueError, match="Start time cannot be greater than or equal to end time"
                ):
                    m.start_time = scb.EpochArray(200.0, "TDB")

    @pytest.mark.parametrize(
        "case",
        ["not_epoch_array", "negative", "end_le_start"],
        ids=["Not EpochArray", "Negative Epoch", "end_time <= start_time"],
    )
    def test_end_time(self, case):
        """
        Verifies TypeError for a non-EpochArray value, ValueError for a
        negative epoch, and ValueError when end_time <= start_time.
        """
        match case:
            case "not_epoch_array":
                m = scb.Maneuver()
                with pytest.raises(TypeError, match="End time must be EpochArray"):
                    m.end_time = "not an epoch"
            case "negative":
                m = scb.Maneuver()
                with pytest.raises(ValueError, match="End time cannot be negative"):
                    m.end_time = scb.EpochArray(-5.0, "TDB")
            case "end_le_start":
                m = scb.Maneuver(start_time=scb.EpochArray(200.0, "TDB"))
                with pytest.raises(
                    ValueError, match="End time cannot be less than or equal to start time"
                ):
                    m.end_time = scb.EpochArray(100.0, "TDB")


# endregion Tests #
# ----------------#