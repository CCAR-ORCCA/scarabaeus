# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.timeAndFrame.EpochArray import EpochArray
import pytest
import numpy as np

km, kg, sec, hr,day = scb.Units.get_units(['km', 'kg', 'sec', 'hr','day'])
J2000   = scb.Frame('J2000')


#--------------------#
# region    Fixtures #
#--------------------#
# setup
@pytest.fixture
def epoch():
    """ Epoch to define state. """
    return scb.EpochArray('2000 JAN 01 00:00:00.000', 'TDB')

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(epoch):
    """
        Verifies that object is constructed correctly.

    """
    pytest.skip('See system_testing/integration_testing/test_times.py for other EpochArray tests.')

def test_split_fractional():
    """
        Verifies splitFractional operation.

    """
    fractional = 123.456
    split = EpochArray._split_fractional_part(fractional)
    assert split[0] == pytest.approx(123.0, 1e-6)
    assert split[1] == pytest.approx(0.456, 1e-6)
    
def test_enforce_precision():
    """
        Verifies enforcePrecision operation.

    """
    pytest.skip('NOT IMPLEMENTED')
