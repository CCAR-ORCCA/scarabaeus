# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the FiniteBurn class  #
# =======================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec, N = scb.Units.get_units(["km", "kg", "sec", "N"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
# setup
@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "TDB")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return 1.5


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft."""
    return scb.Spacecraft("ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff)


@pytest.fixture
def state(epoch, origin, sc):
    """Define StateArray object."""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def maneuver(epoch, origin):
    """
    Define maneuver.

    thrust is given as a full 7-element (already-degree-6) coefficient
    array rather than a scalar, so that the Maneuver.thrust setter's
    non-padding branch is exercised here.
    """
    thrust = scb.ArrayWUnits(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), N)
    mass_flow = scb.ArrayWUnits(3e-4, kg / sec)
    start_time = epoch
    end_time = epoch + scb.ArrayWUnits((60 * 24 * 60 * 60), sec)
    ux = scb.ArrayWUnits(1, None)
    uy = scb.ArrayWUnits(0, None)
    uz = scb.ArrayWUnits(0, None)

    return scb.Maneuver(thrust, mass_flow, start_time, end_time, ux, uy, uz)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(sc, maneuver, state):
    """
    Verifies that object is constructed correctly, including the
    Maneuver.thrust setter's non-padding branch (already-7-element
    input): a constant-thrust polynomial [0.5, 0, 0, 0, 0, 0, 0] is
    stored in descending order with the constant term at index 6.
    """
    # intialize
    finite_burn = scb.FiniteBurn(sc, J2000, maneuver)

    # ensure correct properties
    assert finite_burn.maneuver == maneuver
    assert finite_burn._spacecraft == sc
    assert finite_burn._ref_frame == J2000

    # thrust setter's non-padding branch: descending storage, constant at index 6
    assert np.allclose(maneuver.thrust.values, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005])


def test_evaluate_polynomial(sc, maneuver, state):
    """
    Verifies _evaluate_polynomial returns:
      - the input ArrayWUnits unchanged for a scalar (size=1),
      - the correct polynomial value for a 7-element coefficient array
        (descending order: [c6, c5, ..., c0] → y = c6*t^6 + ... + c1*t + c0),
      - a ValueError for any other input size.
    """
    fb = scb.FiniteBurn(sc, J2000, maneuver)

    # Scalar: returned as-is
    scalar = awu(5.0, N)
    result = fb._evaluate_polynomial(scalar, t=10.0)
    assert result is scalar

    # Descending [1, 2, 3, 0, 0, 0, 0] → c6=1, c5=2, c4=3, rest 0
    # y = t^6 + 2*t^5 + 3*t^4; at t=2 → 64 + 64 + 48 = 176
    poly = awu(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]), N)
    result = fb._evaluate_polynomial(poly, t=2.0)
    assert np.isclose(float(result.values), 176.0)

    # At t=0 only the constant term (index 6, here 0.0) contributes
    result_t0 = fb._evaluate_polynomial(poly, t=0.0)
    assert np.isclose(float(result_t0.values), 0.0)

    # Units are carried through
    assert result.units == N

    # Invalid size raises ValueError
    bad = awu(np.array([1.0, 2.0, 3.0]), N)
    with pytest.raises(ValueError):
        fb._evaluate_polynomial(bad, t=1.0)


# NOTE: the rest of this class' methods are tested in integration tests
