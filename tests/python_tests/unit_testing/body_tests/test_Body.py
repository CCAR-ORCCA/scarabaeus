# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =================================#
#  Unit Tests for the Body class  #
# =================================#
import src.scarabaeus as scb
from scarabaeus import SpiceManager
from scarabaeus.utils.SCBPolynomial import SCBPolynomial
import pytest
import spiceypy as spice
import numpy as np

N, kg, sec, km, dam, m, mm, mu, rad, mrad, deg, g, hr = scb.Units.get_units(
    ["N", "kg", "sec", "km", "dam", "m", "mm", "mu", "rad", "mrad", "deg", "g", "hr"])

# --------------------#
# region    Fixtures #
# --------------------#

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestInitialization:
    @pytest.mark.parametrize(
        "init_args",
        [("Sun", 10, None, None),
         ("ORCCA", -1234, None, None),
         ("ORCCA", -1234, scb.ArrayWUnits(300, kg), None),
         ("TEST_BODY", -1000, scb.ArrayWUnits(300, kg), 3),],
        ids=[ "Name in Kernels",
              "Name & New ID",
              "Name, ID, Const Mass",
              "Name, ID, Mass, Mass Poly",],)
    def test_proper_init(self, init_args):
        """
        Verifies that object is constructed correctly for all valid initialization configurations.
        """
        # construct body
        body = scb.Body(*init_args)

        # verify that name is correct
        assert body._name == init_args[0]
        assert body.spice_id == init_args[1]
        assert body._mass_profile == init_args[2]
        assert body._mass_poly_deg == init_args[3]

    def test_improper_init(self):
        """
        Verifies that a ValueError is raised when no spice_id is given and
        the body name isn't found in the kernel pool.
        """
        with pytest.raises(ValueError, match="not found in the kernel pool"):
            scb.Body("TOTALLY_UNKNOWN_BODY_XYZ", None)


@pytest.mark.parametrize(
    "body, date_str",
    [(scb.Body("Sun"), "2000 JAN 01 12:00:00.000"),
     (scb.Body("Sun"), "2000 JAN 02 12:00:00.000"),],
    ids=["State at J2000", "State at J2000 + 1 day"],)
def test_get_state(body, date_str):
    epoch = scb.EpochArray(date_str, "TDB")

    spice_state = np.array(
        spice.spkezr("SUN", float(epoch.times.values), "J2000", "None", "Earth")[0]
    )
    body_state = body.get_state(epoch)
    assert np.array_equal(body_state.quantity.values, spice_state)


@pytest.mark.parametrize(
    "epoch_0, reference_frame, origin, expected",
    [
        ("not an epoch", "J2000", "EARTH", TypeError),
        (None, 123, "EARTH", TypeError),
        (None, "J2000", 123, TypeError),
        (None, "NOT_A_REAL_FRAME_XYZ", "EARTH", RuntimeError),
    ],
    ids=[
        "Invalid epoch_0 Type",
        "Invalid reference_frame Type",
        "Invalid origin Type",
        "SPICE Error Wrapped as RuntimeError",
    ],
)
def test_get_state_invalid_args(epoch_0, reference_frame, origin, expected):
    """
    Verifies that get_state raises TypeError for a non-EpochArray
    epoch_0 or a non-string reference_frame/origin, and wraps any
    underlying SPICE error (e.g. an unrecognized frame name) in a
    RuntimeError.
    """
    body = scb.Body("Sun")
    if epoch_0 is None:
        epoch_0 = scb.EpochArray("2000 JAN 01 12:00:00.000", "TDB")

    with pytest.raises(expected):
        body.get_state(epoch_0, reference_frame=reference_frame, origin=origin)


@pytest.mark.parametrize(
    "t, expected",
    [(50.0, 495.0), (9999.0, ValueError)],
    ids=["Within Range", "Outside Range"],
)
def test_get_mass(t, expected):
    """
    Verifies that get_mass evaluates the fitted polynomial mass profile
    for a time within its accumulated tspan, and raises ValueError for
    a time outside every stored tspan.
    """
    body = scb.Body("TESTSAT_GETMASS", -999050)
    body.fit_mass_profile_from_data(
        np.array([0.0, 100.0, 200.0]), np.array([500.0, 490.0, 480.0]),
        domain=(0.0, 200.0), units=kg,
    )

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected, match="outside the valid range"):
            body.get_mass(t)
    else:
        result = body.get_mass(t)
        assert isinstance(result, scb.ArrayWUnits)
        assert float(result.values) == pytest.approx(expected)


def test_reset_mass_history():
    """
    Verifies that reset_mass_history restores the mass the body was built with
    and forgets every fitted profile, so propagating a mission again starts
    from that mass instead of the one the previous run ended with.
    """
    body = scb.Body("TESTSAT_RESETMASS", -999051, mass=scb.ArrayWUnits(500.0, kg))
    body.fit_mass_profile_from_data(
        np.array([0.0, 100.0, 200.0]), np.array([500.0, 490.0, 480.0]),
        domain=(0.0, 200.0), units=kg,
    )
    profiles = body._mass_profiles

    body.reset_mass_history()

    assert float(body.mass(150.0).values) == pytest.approx(500.0)
    # cleared in place: force-model copies of the body share these lists
    assert body._mass_profiles is profiles
    assert body._mass_profiles == [] and body._mass_profiles_tspan == []


@pytest.mark.parametrize(
    "case, spice_id, mass_kind, t_kind, expected",
    [
        ("static", -999040, "static", None, "static"),
        ("poly_no_t", -999041, "poly", None, ValueError),
        ("poly_non_tdb_epoch", -999042, "poly", "non_tdb_epoch", ValueError),
        ("poly_tdb_epoch", -999043, "poly", "tdb_epoch", 490.0),
        ("poly_float_t", -999044, "poly", "float", 490.0),
        ("poly_invalid_t_type", -999045, "poly", "invalid", TypeError),
        ("unsupported_internal_type", -999046, "unsupported", None, TypeError),
    ],
    ids=[
        "Static ArrayWUnits Mass",
        "Polynomial: No t Given",
        "Polynomial: Non-TDB EpochArray",
        "Polynomial: TDB EpochArray",
        "Polynomial: Float t",
        "Polynomial: Invalid t Type",
        "Unsupported Internal Mass Type",
    ],
)
def test_mass(case, spice_id, mass_kind, t_kind, expected):
    """
    Verifies Body.mass() for a static ArrayWUnits mass (returned as-is
    regardless of t), a time-varying SCBPolynomial mass evaluated via an
    EpochArray (TDB required) or a bare float, the corresponding error
    paths (missing t, non-TDB epoch, invalid t type), and the fallback
    TypeError for an unsupported internal _mass_profile type.

    Mass/epoch objects are built inside the test body (not the
    parametrize decorator) because a non-TDB EpochArray needs SPICE
    kernels loaded, which happens via a module-scoped fixture that runs
    after collection-time code.
    """
    mass_arg = {
        "static": scb.ArrayWUnits(300, kg),
        "poly": SCBPolynomial([500.0, -0.1], domain=(0.0, 1000.0)),
        "unsupported": None,
    }[mass_kind]

    name = f"TESTSAT_MASS_{case.upper()}"
    body = scb.Body(name, spice_id, mass=mass_arg)
    if case == "unsupported_internal_type":
        body._mass_profile = "not a valid mass type"

    t_arg = {
        None: None,
        "non_tdb_epoch": lambda: scb.EpochArray("2024 JAN 01 00:00:00", "UTC"),
        "tdb_epoch": lambda: scb.EpochArray(100.0, "TDB"),
        "float": lambda: 100.0,
        "invalid": lambda: "not a valid t",
    }[t_kind]
    if callable(t_arg):
        t_arg = t_arg()

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            body.mass(t_arg)
    elif expected == "static":
        assert body.mass() is body._mass_profile
        assert body.mass(scb.EpochArray(0.0, "TDB")) is body._mass_profile
    else:
        result = body.mass(t_arg)
        assert isinstance(result, scb.ArrayWUnits)
        assert float(result.values) == pytest.approx(expected)
