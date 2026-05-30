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

@pytest.fixture
def n_model(nplate_config_file):
    """ Define n-plate model. """
    # construct N-plate model using a configuration file
    plate_config = str(nplate_config_file)
    return scb.nPlateModel(plate_config)

# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff,n_model):
    """ Define spacecraft. """
    return scb.Spacecraft('ORCCA', -1234, area = sc_area, ref_coeff = sc_ref_coeff,n_plate_model=n_model)

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


# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(sc_area,sc_ref_coeff,origin, state, sc):
    """
        Verifies that object is constructed correctly.
    """
    # initialize
    nplate_srp = scb.nPlateSRP(state, sc, J2000)
    
    # ensure correct properties
    assert nplate_srp.origin              == origin
    assert nplate_srp.origin_name         == origin.name
    assert nplate_srp._spacecraft         == sc
    assert nplate_srp._frame              == J2000

    assert nplate_srp.area                == scb.ArrayWUnits.get_value_in_target_units(sc_area, km**2)  # stored in km^2
    assert nplate_srp.ref_coeff           == sc_ref_coeff

    assert nplate_srp._plate_model        == sc.n_plate_model


def test_inertial_to_body_DCM():
    """
    
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests