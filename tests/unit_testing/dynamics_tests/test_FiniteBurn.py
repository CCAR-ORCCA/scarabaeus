# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu


import pytest
import numpy as np
km, kg, sec, N = scb.Units.get_units(['km', 'kg', 'sec', 'N'])
J2000   = scb.Frame('J2000')

#--------------------#
# region    Fixtures #
#--------------------#
# setup
@pytest.fixture
def epoch():
    """ Epoch to define state. """
    return scb.EpochArray('2000 JAN 01 12:00:00.000', 'TDB')

@pytest.fixture
def origin():
    """ Origin body to define state. """
    return scb.CelestialBody.from_constants('Earth')

@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)

@pytest.fixture
def sc_ref_coeff():
    return 1.5

# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """ Define spacecraft. """
    return scb.Spacecraft('ORCCA', -1234, area = sc_area, ref_coeff = sc_ref_coeff)

@pytest.fixture
def state(epoch, origin):
    """ Define StateArray object. """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0 , 0.0]), km    , J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0   , 70.0, 0.0]), km/sec, J2000)

    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]

    return scb.StateArray(epoch, origin,  state=scb.StateDefinition.from_components(state_dict))

@pytest.fixture
def maneuver(epoch, origin):
    """ Define maneuver """
    thrust     = scb.ArrayWUnits(0.5, N)
    mass_flow  = scb.ArrayWUnits( 3e-4 , kg/sec)
    start_time = epoch
    end_time   = epoch + scb.ArrayWUnits((60 * 24 * 60 * 60), sec)
    ux         = scb.ArrayWUnits(1, None)
    uy         = scb.ArrayWUnits(0, None)
    uz         = scb.ArrayWUnits(0, None)

    return scb.Maneuver(thrust, mass_flow, start_time, end_time, ux, uy, uz)


# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(sc, maneuver, state):
    """
     Verifies that object is constructed correctly.
    """
    # intialize
    finite_burn = scb.FiniteBurn(sc,J2000,maneuver, state)
    
    # ensure correct properties
    assert finite_burn.maneuver              == maneuver
    assert finite_burn._state_vector         == state
    assert finite_burn._spacecraft         == sc
    assert finite_burn._ref_frame           == J2000




def test_evaluate_polynomial():
    """
    
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests