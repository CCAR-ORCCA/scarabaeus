# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================#
#  Unit Tests for the CelestialBody class  #
# ==========================================#

import src.scarabaeus as scb
import pytest

kg, km, sec, mu = scb.Units.get_units(["kg", "km", "sec", "mu"])
ITRF93 = scb.Frame("ITRF93")


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that object is constructed correctly for all valid initialization configurations.
    """
    # parameters
    test_name = "TEST_CB"
    test_mass = scb.ArrayWUnits(1, kg)
    test_rad = scb.ArrayWUnits(1, km)
    test_GM = scb.ArrayWUnits(1, km**3 / sec**2)
    test_frame = ITRF93
    test_id = 10000000

    # construct body
    cb = scb.CelestialBody(
        name=test_name,
        mass=test_mass,
        mean_radius=test_rad,
        grav_param=test_GM,
        base_frame=test_frame,
        spice_id=test_id,)

    # check
    assert cb.name == test_name
    assert cb.mass() == test_mass
    assert cb.mean_radius == test_rad
    assert cb.grav_param == test_GM
    assert cb.base_frame == test_frame
    assert cb.spice_id == test_id


class TestErrorHandling:
    @pytest.mark.parametrize(
        "mean_radius, expected",
        [("not an AWU", TypeError), (scb.ArrayWUnits(1, kg), ValueError)],
        ids=["Invalid Type", "Wrong Units"],
    )
    def test_mean_radius(self, mean_radius, expected):
        """
        Verifies that the mean_radius setter raises TypeError for a
        non-ArrayWUnits value and ValueError for an ArrayWUnits not
        expressed in km.
        """
        with pytest.raises(expected):
            scb.CelestialBody(
                name="TEST_CB_MEAN_RADIUS",
                mass=scb.ArrayWUnits(1, kg),
                mean_radius=mean_radius,
                grav_param=scb.ArrayWUnits(1, km**3 / sec**2),
                base_frame=ITRF93,
                spice_id=10000003,
            )

    @pytest.mark.parametrize(
        "grav_param, expected",
        [("not an AWU", TypeError), (scb.ArrayWUnits(1, kg), ValueError)],
        ids=["Invalid Type", "Wrong Units"],
    )
    def test_grav_param(self, grav_param, expected):
        """
        Verifies that the grav_param setter raises TypeError for a
        non-ArrayWUnits value and ValueError for an ArrayWUnits not
        expressed in km^3/s^2.
        """
        with pytest.raises(expected):
            scb.CelestialBody(
                name="TEST_CB_GRAV_PARAM",
                mass=scb.ArrayWUnits(1, kg),
                mean_radius=scb.ArrayWUnits(1, km),
                grav_param=grav_param,
                base_frame=ITRF93,
                spice_id=10000004,
            )

    def test_gs_list_invalid_element(self):
        """Verifies TypeError when gs_list contains a non-GroundStation element."""
        with pytest.raises(TypeError, match="GroundStation"):
            scb.CelestialBody(
                name="TEST_CB_GS_LIST",
                mass=scb.ArrayWUnits(1, kg),
                mean_radius=scb.ArrayWUnits(1, km),
                grav_param=scb.ArrayWUnits(1, km**3 / sec**2),
                base_frame=ITRF93,
                spice_id=10000005,
                gs_list=["not a groundstation"],
            )


def test_repr():
    """Verifies that repr(cb) returns the body's name."""
    cb = scb.CelestialBody(
        name="TEST_CB_REPR",
        mass=scb.ArrayWUnits(1, kg),
        mean_radius=scb.ArrayWUnits(1, km),
        grav_param=scb.ArrayWUnits(1, km**3 / sec**2),
        base_frame=ITRF93,
        spice_id=10000006,
    )
    assert repr(cb) == "TEST_CB_REPR"


def test_disp_properties(capsys):
    """
    Verifies that disp_properties prints the body's name, mass value
    (evaluated via mass(), not the bound method itself), mean_radius,
    grav_param, base_frame, spice_id, and gs_list.
    """
    cb = scb.CelestialBody(
        name="TEST_CB_DISP",
        mass=scb.ArrayWUnits(500, kg),
        mean_radius=scb.ArrayWUnits(1, km),
        grav_param=scb.ArrayWUnits(1, km**3 / sec**2),
        base_frame=ITRF93,
        spice_id=10000007,
    )
    cb.disp_properties()
    out = capsys.readouterr().out

    assert "TEST_CB_DISP" in out
    assert "500" in out
    assert "bound method" not in out
    assert "10000007" in out


@pytest.mark.parametrize(
    "cb_name, expected",
    [
        ("Jupiter", None),
        ("Not Jupiter", ValueError),
        (123, TypeError),
        ("G", ValueError),
    ],
    ids=["Valid Query", "Not Defined", "Invalid Type", "Incomplete Constant"],
)
def test_from_constants(cb_name, expected):
    """
    Verifies that alternate construction method builds a CelestialBody
    correctly for a valid name, raises TypeError for a non-string
    cb_name, and raises ValueError both for a name not defined in
    constants and for a defined-but-incomplete constant (e.g. "G",
    which exists in the constants module but isn't a full PCInfo).
    """
    if expected is not None:
        with pytest.raises(expected):
            scb.CelestialBody.from_constants(cb_name)
        return

    jov_consts = scb.constants.JUPITER
    cb = scb.CelestialBody.from_constants(cb_name)
    assert cb.name == cb_name
    assert cb.mean_radius == jov_consts.mean_radius
    assert cb.grav_param == jov_consts.GM
    assert cb._base_frame == jov_consts.ref_name
    assert cb.spice_id == jov_consts.body_center_id
    assert cb.barycenter_id == jov_consts.barycenter_id


class TestOptionalInputs:
    """
    mass, grav_param, mean_radius and base_frame are optional; given only
    one of mass and grav_param, the other follows from mu = G*M.
    """

    def test_mass_from_grav_param(self):
        """Verifies that a body given only GM gets mass = GM/G in kg."""
        gm = scb.ArrayWUnits(126686534.92180079, mu)
        cb = scb.CelestialBody(name="TEST_CB_GM_ONLY", grav_param=gm, spice_id=10000010)

        assert cb.grav_param is gm
        assert cb.mass().units == kg
        assert cb.mass().values == pytest.approx(126686534.92180079 / 6.67430e-20, rel=1e-14)

    @pytest.mark.parametrize(
        "mass",
        [scb.ArrayWUnits(2.0e15, kg), scb.ArrayWUnits(2.0e18, scb.Units.get_units("g"))],
        ids=["kg", "g"],
    )
    def test_grav_param_from_mass(self, mass):
        """Verifies that a body given only its mass, in any mass units, gets GM = G*M in km^3/s^2."""
        cb = scb.CelestialBody(name="TEST_CB_MASS_ONLY", mass=mass, spice_id=10000011)

        assert cb.mass() is mass
        assert cb.grav_param.units == mu
        assert cb.grav_param.values == pytest.approx(6.67430e-20 * 2.0e15, rel=1e-14)

    def test_both_kept_as_given(self):
        """Verifies that mass and GM given together are both kept, with no rederivation."""
        mass = scb.ArrayWUnits(1, kg)
        gm = scb.ArrayWUnits(1, mu)
        cb = scb.CelestialBody(name="TEST_CB_BOTH", mass=mass, grav_param=gm, spice_id=10000012)

        assert cb.mass() is mass
        assert cb.grav_param is gm

    def test_name_and_id_only(self, capsys):
        """Verifies that a body needs only a name and SPICE ID, and still displays."""
        cb = scb.CelestialBody(name="TEST_CB_BARE", spice_id=10000013)

        assert cb.mass_profile is None
        assert cb.grav_param is None
        assert cb.mean_radius is None
        assert cb.base_frame is None
        assert cb.base_frame_name is None

        cb.disp_properties()
        assert "TEST_CB_BARE" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "mass",
        [5.0, scb.ArrayWUnits(5.0, km)],
        ids=["Not an AWU", "Not a mass"],
    )
    def test_underivable_mass(self, mass):
        """Verifies InvalidMassError when GM must be derived from a mass that is not an AWU mass."""
        with pytest.raises(scb.CelestialBody.InvalidMassError):
            scb.CelestialBody(name="TEST_CB_BAD_MASS", mass=mass, spice_id=10000014)
