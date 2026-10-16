# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the Spacecraft class  #
# =======================================#

import src.scarabaeus as scb
import numpy as np
import pytest

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


# endregion Fixtures #
# --------------------#
@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return scb.ArrayWUnits(1.5, None)


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft"""
    return scb.Spacecraft(
        "orbiter_ref_it1",
        -1001,
        scb.ArrayWUnits(2000.0, kg),
        sc_area,
        sc_ref_coeff,)


# --------------#
# region Tests #
# --------------#
def test_initialization(sc_area, sc_ref_coeff):
    """
    Verifies that object is constructed correctly, including the
    thruster and body_fixed_frame_id properties.
    """
    name = "orbiter_ref_it1"
    spice_id = -1001
    tot_mass = scb.ArrayWUnits(2000.0, kg)
    thruster = "dummy_thruster_obj"
    body_fixed_frame_id = 12345
    spacecraft = scb.Spacecraft(
        name, spice_id, tot_mass, sc_area, sc_ref_coeff,
        thruster=thruster, body_fixed_frame_id=body_fixed_frame_id,
    )

    assert spacecraft._name == name
    assert spacecraft._spice_id == spice_id
    assert spacecraft._mass_profile == tot_mass
    assert spacecraft.area == sc_area
    assert spacecraft._ref_coeff == sc_ref_coeff
    assert spacecraft.thruster == thruster
    assert spacecraft.body_fixed_frame_id == body_fixed_frame_id


class TestInstrumentManagement:
    def test_add(self, sc):
        """
        Verifies that instruments can be added to the spacecraft.
        """
        # create instrument
        name = "sc_unit_antenna"
        test_antenna = scb.Antenna(name, turn_ratio=880.0 / 749.0, spice_id=-9901)

        # add instrument to spacecraft
        sc.add_instrument([test_antenna])

        # ensure instrument is stored in spacecraft
        assert test_antenna in sc.instrument_list

    def test_dependent_spice_ids(self, sc):
        """
        Verifies that dependent spice ids are correctly returned.
        """
        # create instrument
        name = "sc_unit_antenna"
        test_antenna = scb.Antenna(name, turn_ratio=880.0 / 749.0, spice_id=-9901)

        # add instrument to spacecraft
        sc.add_instrument([test_antenna])

        # ensure dependent spice ids are correct
        dependent_ids = sc.get_dependent_spice_ids()
        assert test_antenna.spice_id in dependent_ids["instruments"]


def test_body_fixed_frame_id_invalid_type(sc):
    """Verifies TypeError when body_fixed_frame_id is set to a non-int."""
    with pytest.raises(TypeError, match="body_fixed_frame_id"):
        sc.body_fixed_frame_id = "not an int"


class TestGetAttitude:
    @pytest.mark.parametrize(
        "from_frame, tol, expected",
        [("not a frame", 0, TypeError), (J2000, "not an int", TypeError)],
        ids=["Invalid from_frame Type", "Invalid tol Type"],
    )
    def test_invalid_args(self, sc, from_frame, tol, expected):
        """Verifies TypeError for a non-Frame from_frame and a non-int tol."""
        epoch = scb.EpochArray(0.0, "TDB")
        with pytest.raises(expected):
            sc.get_attitude(from_frame, epoch, tol)

    def test_returns_quaternion(self, tmp_path):
        """
        Verifies that get_attitude returns a 4-element quaternion derived
        from a synthetic CK segment (via write_ck_type3, reusing Cassini's
        SCLK already in the locked generic kernel set).
        """
        sc_attitude = scb.Spacecraft("CASSINI_TEST_SC", -82)

        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        et_list = [et0 + 60.0 * i for i in range(5)]
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        avvs = np.tile([0.001, 0.002, 0.003], (5, 1))

        ck_path = str(tmp_path / "test_sc_attitude.bc")
        try:
            scb.SpiceManager.write_ck_type3(
                ck_path, inst_id=-82000, ref_frame="J2000", sclk_id=-82,
                et_list=et_list, quats=quats, avvs=avvs, segid="TEST SC ATTITUDE SEG",
            )
            epoch = scb.EpochArray(et0 + 120.0, "TDB")
            q = sc_attitude.get_attitude(J2000, epoch)

            assert isinstance(q, np.ndarray)
            assert q.shape == (4,)
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)


@pytest.mark.parametrize(
    "case",
    ["keyed_by_spice_id", "plain_velocity_key_raw_list", "frame_as_string", "missing_velocity"],
    ids=["Keyed by (velocity, spice_id)", "Plain 'velocity' Key (Raw List)",
         "Frame Given as String", "Missing Velocity Raises KeyError"],
)
def test_inertial_to_body_DCM_nadirPointingSun(sc, case):
    """
    Verifies the deprecated nadir-to-sun DCM: a velocity keyed by
    ("velocity", spice_id) as an ArrayWUnits, a plain "velocity" key
    given as a raw list, a frame given as a plain string instead of a
    Frame object, and a KeyError when no velocity entry is found. Every
    call warns, and the result is the transpose of
    DynamicalUtils.nadirPointingSun.
    """
    et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    sun2body_pos = scb.ArrayWUnits(np.array([1.0e8, 0.0, 0.0]), km)
    frame = "J2000" if case == "frame_as_string" else J2000

    match case:
        case "keyed_by_spice_id":
            vel = scb.ArrayWUnits(np.array([0.0, 30.0, 0.0]), km / sec)
            current_state = {("velocity", sc.spice_id): vel}
        case "plain_velocity_key_raw_list" | "frame_as_string":
            current_state = {"velocity": [0.0, 30.0, 0.0]}
        case "missing_velocity":
            current_state = {}

    if case == "missing_velocity":
        with pytest.warns(DeprecationWarning), pytest.raises(
            KeyError, match="needs the spacecraft velocity"
        ):
            sc.inertial_to_body_DCM_nadirPointingSun(
                "EARTH", frame, sun2body_pos, current_state, et0, sc
            )
        return

    with pytest.warns(DeprecationWarning, match="DynamicalUtils.nadirPointingSun"):
        dcm = sc.inertial_to_body_DCM_nadirPointingSun(
            "EARTH", frame, sun2body_pos, current_state, et0, sc
        )
    assert dcm.shape == (3, 3)
    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-8)

    sun2earth_vel = scb.SpiceManager.get_vel(
        trgt_bdy="EARTH", epoch_time=et0, reference_frame="J2000", obsvr_bdy="SUN"
    ).values
    body2inertial, _, _ = scb.DynamicalUtils.nadirPointingSun(
        sun2body_pos.values, sun2earth_vel + np.array([0.0, 30.0, 0.0])
    )
    np.testing.assert_allclose(dcm, body2inertial.T, atol=1e-15)