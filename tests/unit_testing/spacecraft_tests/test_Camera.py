# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest

km, deg, rad = scb.Units.get_units(['km', 'deg','rad'])
J2000   = scb.Frame('J2000')
#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization():
    """
        Verifies that object is constructed correctly.
    """
    name = "Test Camera"
    spice_id = -12345
    measurement_type = "image"
    fov_angular = scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad)
    fov_pixels = (2000, 2000)
    focal_length =  scb.ArrayWUnits(150 * 10**-6, km)
    ifov_angle = scb.ArrayWUnits(np.array([25 * 10**-6, 20 * 25**-6]), rad)

    camera = scb.Camera(name, spice_id, J2000, fov_angular, fov_pixels, focal_length, ifov_angle, measurement_type= measurement_type)

    assert camera._name == name
    assert camera._spice_id == spice_id
    assert camera._measurement_type == measurement_type
    assert camera._fov_angular == fov_angular
    assert camera._fov_pixels == fov_pixels
    assert camera._focal_length == focal_length
    assert camera._ifov_angle == ifov_angle