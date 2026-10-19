# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
#=============================================#
#  Integration Tests for the ScenarioSetup   #
#=============================================#
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

    # origin property getter/setter
    assert Scenario.origin == origin
    new_origin = scb.CelestialBody.from_constants("EARTH")
    Scenario.origin = new_origin
    assert Scenario.origin == new_origin
    Scenario.origin = origin

    # tolerance property
    assert Scenario.atol == 2.220446049250313e-11

    # propagated_state_array before propagate() has been called: warns and
    # has no last-leg result to fall back on yet
    with pytest.warns(UserWarning), pytest.raises(AttributeError):
        _ = Scenario.propagated_state_array

    # accumulators are empty until propagate() runs
    assert Scenario.total_parameters == []
    assert Scenario.STM == []
    assert Scenario.total_components == {}

    # total_position/total_velocity require propagate() to have populated totals
    with pytest.raises(RuntimeError):
        _ = Scenario.total_position
    with pytest.raises(RuntimeError):
        _ = Scenario.total_velocity

    # full_state_vector reflects the scenario-global state definition
    full_state = Scenario.full_state_vector
    assert isinstance(full_state, scb.StateArray)
    assert full_state.origin == origin

def test_merge_state_list_with_old(state_dict, state, epochs, origin, propagator):
    """
    Verifies that _merge_state_list_with_old returns a list of the same
    length as the input whose values are inherited from the previous state vector.
    """
    leg1_model = scb.StateArray(epoch=epochs[0], origin=origin, state=scb.StateDefinition.from_components(state_dict))
    MS = scb.MissionSequence("ORCCA_SEQUENCE")
    MS.addLeg("First Leg", scb.ArrayWUnits(58000, sec), leg1_model, propagator, scb.ArrayWUnits(360, sec))
    Scenario = scb.ScenarioSetup(MS)

    update_list = Scenario._merge_state_list_with_old(state_dict)

    assert len(update_list) == len(state_dict)
    for (orig_name, orig_size, orig_est, orig_dyn, orig_body, orig_awf), \
            (new_name, new_size, new_est, new_dyn, new_body, new_awf) in zip(state_dict, update_list):
        assert new_name == orig_name
        assert new_size == orig_size
        assert new_est == orig_est
        assert new_dyn == orig_dyn
        assert new_body is orig_body
        assert np.allclose(new_awf.quantity.values, orig_awf.quantity.values)

def test_update_stm():
    """
        DESC
    """
    pytest.skip()

# NOTE: the rest of this class' methods are tested in integration tests