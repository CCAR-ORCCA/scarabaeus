# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest
import os
km, kg, sec = scb.Units.get_units(['km', 'kg', 'sec'])
J2000   = scb.Frame('J2000')

#--------------------#
# region    Fixtures #
#--------------------#

@pytest.fixture
def epochs():
    time_0 = scb.SpiceManager.jd2et(2461809.72995654 + 1 / 3)
    time_f = scb.SpiceManager.jd2et(2461809.72995654 + 1)
    dt = 60 * 60
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array

@pytest.fixture
def origin():

    origin = scb.CelestialBody.from_constants("SUN")

    return origin 

@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft("Orbiter",-1000,  scb.ArrayWUnits(2000.0, kg), Orbiter_area,Orbiter_cr_srp)

@pytest.fixture
def state_dict(sc):
    leg1_state_definition = [
    (
        "position",
        3,
        "estimated",
        "dynamic",
        sc,
        scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000),
    ),
    (
        "velocity",
        3,
        "estimated",
        "dynamic",
        sc,
        scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km / sec, J2000),
    ),
]
    return leg1_state_definition

@pytest.fixture
def state(epochs, origin, state_dict):
    """ Define stateArray object """
    return scb.StateArray(epochs[0], origin,  state=scb.StateDefinition.from_components(state_dict))

@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(primary_body=sc,)

@pytest.fixture
def propagator(sc, state,epochs,force_model):
        
    prop = scb.Propagator(
    primary_body=sc,
    state_vector=state,
    tspan=epochs,
    force_models=force_model)
    return prop

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(state_dict, state, epochs, origin,propagator):
    """
        Verifies that object is constructed correctly.
    """
    leg1_model = scb.StateArray(epoch=epochs[0],origin=origin,state=scb.StateDefinition.from_components(state_dict))
    MS = scb.MissionSequence("ORCCA_SEQUENCE")
    MS.addLeg("First Leg", scb.ArrayWUnits(58000, sec), leg1_model, propagator, scb.ArrayWUnits(360, sec))
    Scenario = scb.ScenarioSetup(MS)

    assert Scenario._mission_sequence == MS
    assert Scenario._len_sequence == len(MS.names)
    assert Scenario._propagator == propagator
    assert Scenario._origin == origin

def test_merge_state_list_with_old(state_dict, state, epochs, origin,propagator):
    """
        Verifies that the state list is merged with the old state list correctly.
    """
    leg1_model = scb.StateArray(epoch=epochs[0],origin=origin,state=scb.StateDefinition.from_components(state_dict))
    MS = scb.MissionSequence("ORCCA_SEQUENCE")
    MS.addLeg("First Leg", scb.ArrayWUnits(58000, sec), leg1_model, propagator, scb.ArrayWUnits(360, sec))
    Scenario = scb.ScenarioSetup(MS)

    update_list = Scenario._merge_state_list_with_old(state_dict)
    excepected_list = state_dict + state_dict
    assert update_list == excepected_list

def test_update_stm():
    """
        DESC
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests