"""
    Scarabaeus system testing configuration. Called in addition to all methods 
    and setup defined in the main testing suite configuration (see conftest.py 
    in testing suite root). Includes additional setup logic unique to system 
    testing.
"""
import src.scarabaeus as scb
from scarabaeus import ArrayWUnits as awu

import pytest
import numpy as np

import json, os
from pathlib import Path

# units and frames used across test
kg, km, sec, rad = scb.Units.get_units(["kg", "km", "sec", "rad"])
J2000 = scb.Frame("J2000")

@pytest.fixture(scope="session")
def epochs():
    """
    Generates an array of epochs for testing. The epochs are spaced 60 seconds apart, starting from a specific time and ending after 58000 seconds.
    """
    time_0 = scb.SpiceManager.cal2et("2029 APR 01 00:00:00")
    time_f = scb.SpiceManager.cal2et("2029 APR 01 01:30:00")  # time_0 + 90 min
    dt = 60
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array

@pytest.fixture(scope="session")
def antenna():
    """
    Generates a test antenna.
    """

    name = "test_antenna"
    test_antenna = scb.Antenna(name, turn_ratio=880.0 / 749.0, spice_id=-1200)
    return test_antenna

@pytest.fixture(scope="session")
def camera():
    """
    Generates a test camera for use in centroiding measurements.
    """

    name = "Test Camera"
    spice_id = -1000
    measurement_type = "image"
    fov_angular = scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad)
    fov_pixels = (2000, 2000)
    focal_length = scb.ArrayWUnits(150 * 10**-6, km)
    ifov_angle = scb.ArrayWUnits(np.array([25 * 10**-6, 20 * 25**-6]), rad)
    camera = scb.Camera(
        name=name,
        associated_body_spice_id=spice_id,
        camera_frame=J2000,
        fov_angular=fov_angular,
        fov_pixels=fov_pixels,
        focal_length=focal_length,
        ifov_angle=ifov_angle,
        measurement_type=measurement_type,
    )

    return camera

@pytest.fixture(scope="session")
def GS1():
    """
    Generates a test ground station for use in radiometric measurements.
    """
    return scb.GroundStation("DSS-14")

@pytest.fixture(scope="session")
def GS2():
    """
    Generates a second test ground station for use in differential radiometric measurements.
    """
    return scb.GroundStation("DSS-63")

def get_common_params():
    """
    Returns common parameters for spacecraft initialization.
    """
    spice_id = -1000
    tot_mass = scb.ArrayWUnits(2000.0, kg)
    sc_area = scb.ArrayWUnits(1e-06, km**2)
    sc_ref_coeff = 1.5

    return spice_id, tot_mass, sc_area, sc_ref_coeff

@pytest.fixture(scope="session")
def sc(antenna, camera):
    """
    Generates a test spacecraft with the provided antenna and camera.
    """

    name = "Orbiter"
    spice_id, tot_mass, sc_area, sc_ref_coff = get_common_params()
    sc1 = scb.Spacecraft(
        name=name,
        spice_id=spice_id,
        tot_mass=tot_mass,
        area=sc_area,
        ref_coeff=sc_ref_coff,
    )
    sc1.add_instrument([antenna])
    sc1.add_instrument([camera])

    return sc1

@pytest.fixture(scope="session")
def init_state(epochs, origin, sc):
    # compute initial state
    a = 6778  # km, ~400km altitude LEO
    mu = origin.grav_param.values  # 398600.435436096 km^3/s^2
    v = np.sqrt((mu / a))
    pos0 = scb.ArrayWFrame(
        awu(
            np.array([a, 0, 0]),
            km,
        ),
        J2000,
    )
    vel0 = scb.ArrayWFrame(
        awu(
            np.array([0, v, 0]),
            km / sec,
        ),
        J2000,
    )
    x0 = scb.StateArray(
        epochs[0],
        origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )
    return x0

@pytest.fixture(scope="session")
def origin():

    origin = scb.CelestialBody.from_constants("Earth")

    return origin