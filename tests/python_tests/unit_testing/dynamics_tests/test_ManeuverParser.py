# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the ManeuverParser class   #
# ============================================#
import io

import pytest

import src.scarabaeus as scb

kg, sec, N = scb.Units.get_units(["kg", "sec", "N"])


def _maneuver_row(num_maneuvers, start_str, ux, uy, uz, thrust, mass_flow, duration):
    """
    Builds a comma-delimited row containing ``num_maneuvers`` identical
    maneuver blocks, per the 13-column-repeating layout documented in
    ManeuverParser._parse_maneuver_spec.
    """
    cols = [""] * (13 * num_maneuvers + 1)
    cols[1] = str(num_maneuvers)
    for n in range(num_maneuvers):
        idx_add = 13 * n
        cols[3 + idx_add] = start_str
        cols[4 + idx_add] = str(ux)
        cols[5 + idx_add] = str(uy)
        cols[6 + idx_add] = str(uz)
        cols[7 + idx_add] = str(thrust)
        cols[9 + idx_add] = str(mass_flow)
        cols[13 + idx_add] = str(duration)
    return ",".join(cols)


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """Verifies a freshly constructed parser has an empty burn list."""
    parser = scb.ManeuverParser()
    assert parser.get_burns() == []


def test_set_burns_get_burns():
    """Verifies set_burns/get_burns round-trip the internal burn list."""
    parser = scb.ManeuverParser()
    burns = [scb.Maneuver(), scb.Maneuver()]

    parser.set_burns(burns)

    assert parser.get_burns() == burns
    assert parser.get_burns() is not burns  # stored as a new list


@pytest.mark.parametrize(
    "num_maneuvers",
    [1, 2],
    ids=["Single Maneuver Row", "Two Maneuvers in One Row"],
)
def test_parse_maneuver_spec(num_maneuvers):
    """
    Verifies a comma-delimited maneuver spec is parsed into Maneuver
    objects with the correct start/end times and polynomial
    parameters, for one and two maneuvers packed in a single row.
    """
    start_str = "2024 JAN 01 00:00:00"
    duration = 60.0
    row = _maneuver_row(num_maneuvers, start_str, 1.0, 0.0, 0.0, 0.5, 3e-4, duration)
    content = "header\n" + row + "\n"

    # _parse_maneuver_spec calls SpiceManager.str2et(), which needs a
    # leapseconds kernel in the pool; load it directly rather than relying
    # on ManeuverParser's own module-level SpiceManager.clear_kernels() call
    # not having wiped it out from under the shared test-suite fixture.
    scb.SpiceManager.load_kernel("data/kernels/locked/lsk/naif0012.tls")

    parser = scb.ManeuverParser()
    burns = parser._parse_maneuver_spec(io.StringIO(content))

    et0 = scb.SpiceManager.cal2et(start_str)

    assert len(burns) == num_maneuvers
    for burn in burns:
        assert burn.start_time.times.values == pytest.approx(et0)
        assert burn.end_time.times.values == pytest.approx(et0 + duration)
        assert burn.thrust.values[-1] == pytest.approx(0.0005)  # 0.5 N -> kN
        assert burn.mass_flow.values[-1] == pytest.approx(3e-4)
        assert burn.ux.values[-1] == pytest.approx(1.0)
        assert burn.uy.values[-1] == pytest.approx(0.0)
        assert burn.uz.values[-1] == pytest.approx(0.0)


class TestUpdateBurnByStartTime:
    def test_invalid_maneuver_raises(self):
        """Verifies ValueError when the replacement isn't a Maneuver."""
        parser = scb.ManeuverParser()
        with pytest.raises(ValueError, match="has to be a Maneuver object"):
            parser._update_burn_by_start_time("not a maneuver", [])

    def test_success(self):
        """Verifies the burn matching on start_time is replaced in place."""
        parser = scb.ManeuverParser()
        burn0 = scb.Maneuver(start_time=scb.EpochArray(1000.0, "TDB"))
        burn1 = scb.Maneuver(start_time=scb.EpochArray(2000.0, "TDB"))
        fb_list = [burn0, burn1]

        replacement = scb.Maneuver(start_time=scb.EpochArray(2000.0, "TDB"), thrust=scb.ArrayWUnits(1.0, N))
        updated_list, idx = parser._update_burn_by_start_time(replacement, fb_list)

        assert idx == 1
        assert updated_list[1] is replacement
        assert updated_list[0] is burn0

    def test_no_match_raises(self):
        """Verifies ValueError when no burn matches the given start_time."""
        parser = scb.ManeuverParser()
        fb_list = [scb.Maneuver(start_time=scb.EpochArray(1000.0, "TDB"))]
        replacement = scb.Maneuver(start_time=scb.EpochArray(9999.0, "TDB"))

        with pytest.raises(ValueError, match="No burn found with start_time"):
            parser._update_burn_by_start_time(replacement, fb_list)


# endregion Tests #
# ----------------#