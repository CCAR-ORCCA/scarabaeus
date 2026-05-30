# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(['km', 'kg', 'sec'])
J2000   = scb.Frame('J2000')

#--------------------#
# region    Fixtures #
#--------------------#
# setup
@pytest.fixture
def epoch():
    """ Epoch to define state. """
    return scb.EpochArray('2000 JAN 01 12:00:00.000', 'UTC')

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
    """ Spacecraft to define the cannonball model for. """
    return scb.Spacecraft('ORCCA', -1234, area = sc_area, ref_coeff = sc_ref_coeff)

@pytest.fixture
def state(epoch, origin):
    """ StateArray object to construct CannonballSRP. """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0 , 0.0]), km    , J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0   , 70.0, 0.0]), km/sec, J2000)

    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]

    return scb.StateArray(epoch, origin,  state=scb.StateDefinition.from_components(state_dict))

# validation
@pytest.fixture
def exp_const_scale_factor():
    """ Expected constant scale factor computed by CannonballSRP. """
    return awu(152969156.355, km**3*kg*sec**-2)

@pytest.fixture
def frame():
    """ Define frame """

    return J2000

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(state, sc, origin, sc_area, sc_ref_coeff, exp_const_scale_factor):
    """
        Verifies that object is constructed correctly.
    """
    # initialize
    cb_srp = scb.CannonballSRP(state, sc, J2000)
    
    # ensure correct properties
    assert cb_srp.origin              == origin
    assert cb_srp.origin_name         == origin.name
    assert cb_srp._spacecraft         == sc
    assert cb_srp.ref_frame           == J2000

    assert cb_srp.area                == scb.ArrayWUnits.get_value_in_target_units(sc_area, km**2)  # stored in km^2
    assert cb_srp.ref_coeff           == sc_ref_coeff

    assert cb_srp._const_scale_factor == scb.ArrayWUnits.get_value_in_target_units(exp_const_scale_factor, (kg*km**3)/sec**2)

# NOTE: the rest of this class' methods are tested in integration tests