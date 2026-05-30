# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
km, kg, sec = scb.Units.get_units(['km', 'kg', 'sec'])

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#
@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)

@pytest.fixture
def sc_ref_coeff():
    return scb.ArrayWUnits(1.5, None)

# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """  Define spacecraft """
    return scb.Spacecraft(
    "orbiter_ref_it1",
    -1001,
    scb.ArrayWUnits(2000.0, kg),
    sc_area,
    sc_ref_coeff,
)
#--------------#
# region Tests #
#--------------#
def test_initialization(sc_area, sc_ref_coeff):
    """
        Verifies that object is constructed correctly.
    """
    name = "orbiter_ref_it1"
    spice_id = -1001
    tot_mass = scb.ArrayWUnits(2000.0, kg) 
    spacecraft =  scb.Spacecraft( name, spice_id, tot_mass, sc_area, sc_ref_coeff)

    assert spacecraft._name == name
    assert spacecraft._spice_id == spice_id
    assert spacecraft._mass_profile == tot_mass
    assert spacecraft.area == sc_area
    assert spacecraft._ref_coeff == sc_ref_coeff

def test_add_instrument(sc):
    """
        Verifies that instruments can be added to the spacecraft.
    """
    # create instrument
    name = "test_antenna"
    test_antenna = scb.Antenna(name,turn_ratio=  880.0/749.0,  spice_id= -1000)

    # add instrument to spacecraft
    sc.add_instrument([test_antenna])

    # ensure instrument is stored in spacecraft
    assert test_antenna in sc.instrument_list

def test_get_dependent_spice_ids(sc):
    """
        Verifies that dependent spice ids are correctly returned.
    """
    # create instrument
    name = "test_antenna"
    test_antenna = scb.Antenna(name,turn_ratio=  880.0/749.0,  spice_id= -1000)

    # add instrument to spacecraft
    sc.add_instrument([test_antenna])

    # ensure dependent spice ids are correct
    dependent_ids = sc.get_dependent_spice_ids()
    assert test_antenna.spice_id in dependent_ids['instruments']