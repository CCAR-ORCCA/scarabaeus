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

@pytest.fixture
def order():
    """ Order of spherical harmonics expansion. """
    return 3
@pytest.fixture
def cs_file(earth_config_file):
    """ File containing spherical harmonics coefficients. """
    return str(earth_config_file)

@pytest.fixture
def body():
    """ Central body for spherical harmonics expansion. """
    return scb.CelestialBody.from_constants('Earth')

@pytest.fixture
def epoch():
    """ Epoch to define state. """
    return scb.EpochArray('2000 JAN 01 12:00:00.000', 'UTC')

@pytest.fixture
def origin():
    """ Origin body to define state. """
    earth_dict = {'name'      : scb.constants.EARTH.name,
                  'spice_name': scb.constants.EARTH.spice_name,
                  'ref_name'  : scb.constants.EARTH.ref_name, 
                  'SPICE_ID'  : scb.constants.EARTH.body_center_id,
                  'mass'      : scb.constants.EARTH.mass,
                  'mu'        : scb.constants.EARTH.GM,
                  'radius'    : scb.constants.EARTH.equatorial_radius}
    return earth_dict

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
def state(epoch, body):
    """ Define StateArray object. """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0 , 0.0]), km    , J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0   , 70.0, 0.0]), km/sec, J2000)

    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]

    return scb.StateArray(epoch, body,  state=scb.StateDefinition.from_components(state_dict))

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(origin, order, cs_file, body, state):
    """
    Verifies that object is constructed correctly.

    """
    # intialize
    spher_harm_gravity = scb.SphericalHarmonicsGravity(order, cs_file, body, state, base_frame=J2000)
    
    # ensure correct properties
    assert spher_harm_gravity._order             == order
    assert spher_harm_gravity._cs_file           == cs_file
    assert spher_harm_gravity._body              == body
    assert spher_harm_gravity._origin            == body
    assert spher_harm_gravity._origin_name       == origin['name']
    assert spher_harm_gravity._ref_frame         == J2000

def test_acceleration_transform_rf():
    """
    
    """
    pytest.skip()

def test_acceleration_transform_rf_raw():
    """
    
    """
    pytest.skip()

def test_jacobian_pos_transform_rf():
    """
    
    """
    pytest.skip()

def test_jacobian_C_transform_rf():
    """
    
    """
    pytest.skip()

def test_jacobian_pos_transform_rf_raw():
    """
    
    """
    pytest.skip()

def test_rf_transform():
    """
    
    """
    pytest.skip()

def test_rf_transform_raw():
    """
    
    """
    pytest.skip()

def test_compute_bnm():
    """
    
    """
    pytest.skip()

def test_compute_bnm_raw():
    """
    
    """
    pytest.skip()

def test_normalize_coeffs():
    """
    
    """
    pytest.skip()

def test_kronecker_delta():
    """
    
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests