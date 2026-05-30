# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np

kg, km, sec = scb.Units.get_units(['kg', 'km', 'sec'])

CASE_1 = 0
CASE_2 = 1
CASE_3 = 2

#--------------------#
# region    Fixtures #
#--------------------#
# general
@pytest.fixture
def start_epoch():
    return scb.SpiceManager.cal2et('2028 FEB 08 05:31:08.0')

@pytest.fixture
def end_epoch():
    return scb.SpiceManager.cal2et('2028 FEB 08 07:04:44.0')

@pytest.fixture
def epochs():
    pass

@pytest.fixture
def frame():
    return scb.Frame('J2000')

@pytest.fixture
def origin():
    scb.CelestialBody.from_constants('EARTH')

# spacecraft 
@pytest.fixture
def sc_mass():
    return awu(1000, kg)

@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)

@pytest.fixture
def sc_ref_coeff():
    return 1.5

@pytest.fixture
def spacecraft_cannonball(sc_mass, sc_area, sc_ref_coeff):
    scb.Spacecraft('ORRCA', -1234, sc_mass, sc_area, sc_ref_coeff)

@pytest.fixture
def state_0(spacecraft_cannonball, start_epoch, frame, origin):
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0     , 0.0]), km    , frame)
    vel_0 = scb.ArrayWFrame(np.array([0.0   , 7.546049, 0.0]), km/sec, frame)

    state_dict = [
        ("position", 3, "estimated", "dynamic", spacecraft_cannonball, pos_0),
        ("velocity", 3, "estimated", "dynamic", spacecraft_cannonball, vel_0),
    ]

    return scb.StateArray(start_epoch, origin, state_dict)

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
class TestCannonballSRP:
    """ All DynamicModel inherited methods. """
    @pytest.fixture
    def cannonball(state_0, spacecraft_cannonball, frame):
        scb.CannonballSRP(state_0, spacecraft_cannonball, frame)

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case, cannonball):
        """
            Verifies that the acceleration due to solar radiation pressure is calculated as expected.
        """
        print(cannonball)

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
            Verifies that the partial derivatives of the SRP acceleration with respect to position are 
            calculated as expected.
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_eta(self, case):
        """
            Verifies that the partial derivative of SRP acceleration with respect to the SRP scale factor is 
            calculated as expected.
        """
        pytest.skip()

class TestFiniteBurn:
    """ All DynamicModel inherited methods. """
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case):
        """
            Verifies that the acceleration from the finite burn force is calculated as expected.
        """
        pytest.skip()
    
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
            Verifies that partial derivative of the finite burn force with respect to the position vector is
            calculated as expected.
        """
        pytest.skip()

class TestImpulsiveBurn:
    """ All DynamicModel inherited methods. """
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case):
        """
        
        """
        pytest.skip()
    
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_dv(self, case):
        """
        
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
        
        """
        pytest.skip()

class TestnPlateSRP:
    """ All DynamicModel inherited methods. """
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case):
        """
            Verifies that the acceleration due to solar radiation pressure is calculated as expected.
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
            Verifies that the partial derivatives of the SRP acceleration with respect to position are 
            calculated as expected.
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_eta(self, case):
        """
            Verifies that the partial derivative of SRP acceleration with respect to the SRP scale factor is 
            calculated as expected.
        """
        pytest.skip()

class TestPointMassGravity:
    """ All DynamicModel inherited methods. """
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case):
        """
        
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
        
        """
        pytest.skip()

class TestSphericalHarmonicsGravity:
    """ All DynamicModel inherited methods. """
    pass

class TestThreeBodyGravity:
    """ All DynamicModel inherited methods. """
    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_acceleration(self, case):
        """
        
        """
        pytest.skip()

    @pytest.mark.parametrize(
            'case',
            [(CASE_1),
             (CASE_2),
             (CASE_3)],
             ids = ['Case 1', 'Case 2', 'Case 3']
    )
    def test_compute_partial_by_position(self, case):
        """
        
        """
        pytest.skip()