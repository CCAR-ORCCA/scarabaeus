# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

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
    name = "test_antenna"
    test_antenna = scb.Antenna(name,turn_ratio=  880.0/749.0,  spice_id= -1000)

    assert test_antenna.name == name
    assert test_antenna._turn_ratio == 880.0/749.0
    assert test_antenna.spice_id == -1000