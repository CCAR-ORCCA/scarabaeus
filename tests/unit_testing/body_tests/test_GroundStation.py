# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

@pytest.fixture
def instrument():
    """
        Fixture that returns a sample instrument for testing.
    """
    return scb.GroundStation('DSS-14', 399014)

@pytest.fixture
def target():
    """
        Fixture that returns a sample target for testing.
    """
    return scb.Spacecraft('Test Spacecraft', 123456)

#--------------#
# region Tests #
#--------------#
@pytest.mark.parametrize(
    'init_args',
    [('DSS-14', 399014),
     ('DSS-24', None)
     ],
        ids = ['with ID', 'Without ID']
)
def test_initialization(init_args):
    """
        Verifies that object is constructed correctly for all valid initialization configurations.
    """
    # construct body
    gs = scb.GroundStation(*init_args)

    # verify that name is correct
    assert gs.name == init_args[0]
    if init_args[1] is not None:
        assert gs.spice_id == init_args[1]


def test_station_visibility(target, instrument):
    """
        Verifies that station visibility is properly detected.
    """
    pytest.skip('UPDATE')
    # gs = scb.GroundStation('DSS-24', 399024)

    # gs.station_visibility(instrument,target)
    
