# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
import os

km, kg, sec, hr,day, m, rad, unitless = scb.Units.get_units(['km', 'kg', 'sec', 'hr','day', 'm', 'rad', 'unitless'])

#--------------------#
# region    Fixtures #
#--------------------#

@pytest.fixture
def epoch():
    """ Epoch to define state. """
    epoch_0 = scb.EpochArray('2000 JAN 01 00:00:00.000', 'TDB')
    return epoch_0

@pytest.fixture
def bplane_file(earth_bplane_fk):
    file = str(earth_bplane_fk)

    return file

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(epoch,bplane_file):
    """
        Verifies that object is constructed correctly.
    """
    bplane_spice_id = 12345678
    sc_name="MOON"
    bplane_name="EARTH_BPLANE"
    sc_spice_id=301
    target_name="EARTH_BARYCENTER"
    target_spice_id=3
    Bplane = scb.Bplane(epoch, bplane_name, bplane_spice_id, sc_name, sc_spice_id, target_name, target_spice_id, bplane_file, v_hat=False, new_bplane=False)
    
    assert Bplane.epoch == epoch
    assert Bplane._bplane_name == bplane_name
    assert Bplane._bplane_spice_id == bplane_spice_id
    assert Bplane._sc_name == sc_name
    assert Bplane._sc_spice_id == 'MOON'
    assert Bplane._target_name == 'EARTH_BARYCENTER'
    assert Bplane._target_spice_id == 'EARTH BARYCENTER'
    assert str(Bplane._fk_file) == bplane_file

def test_compute_jacobian():
    """
        DESC
    """
    pytest.skip()
    
def test_target_covariance():
    """
        DESC
    """
    pytest.skip()
        
def test_target_spacecraft_covariance():
    """
        DESC
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests