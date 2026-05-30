# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
import numpy as np
#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
@pytest.mark.parametrize("grav_param, r, v, expected", [
    (398600.4418, np.array([7000, 0, 0]), np.array([0, 7.54604911, 0]), (6999.99224475, 1.10789209e-06, 0.0, 0.0, np.pi, -np.pi)),
    (398600.4418, np.array([42164.1696, 0, 0]), np.array([0, 3.074661212, 0]), (42164.20008787, 7.23075324e-07, 0.0, 0.0, 0.0, 0.0)),
    (398600.4418, np.array([42164.1696 / np.sqrt(2), 42164.1696 / np.sqrt(2), 0]), np.array([-3.074661212 / np.sqrt(2), 3.074661212 / np.sqrt(2), 0]), (42164.20008787, 7.23075324e-7, 0.0, 0.0, 0.78539816,0.0)),
])
def test_rv2coe(grav_param, r, v, expected):
    """     Tests the rv2coe method of the OrbitalElements class.
    """
    # Arrange
    orbital_elements = scb.OrbitalElements()
    
    # Act
    result = orbital_elements.rv2coe(grav_param, r, v)
    
    # Assert
    assert np.allclose(result, expected, atol=1e-5), f"Expected {expected}, but got {result}"
