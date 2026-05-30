# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import os
import pytest
import numpy as np

km, sec = scb.Units.get_units(['km', 'sec'])
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
    # initialize
    epoch   = scb.EpochArray(scb.SpiceManager.cal2et('2028 FEB 08 05:31:08.0'), 'TDB')
    earth   = scb.CelestialBody.from_constants('Earth')
    test_sc = scb.Spacecraft('ORCCA', -1234)
    pos_0   = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0   = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km/sec, J2000)  
    state_dict = [
        ("position", 3, "estimated", "dynamic", test_sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", test_sc, vel_0),
    ]

    state_arr = scb.StateArray(epoch, earth, state=scb.StateDefinition.from_components(state_dict))

    assert state_arr._epoch == epoch
    assert state_arr._origin == earth
    assert state_arr._state == scb.StateDefinition.from_components(state_dict).components


def test_add_to_state_vector(setup):
    """
        Verifies that components can be added to state vector.
    """
    pytest.skip('NEED TO NOT USE .FROM_COMPONENTS ON THIS')
    # # initialize
    # epoch   = scb.EpochArray(scb.SpiceManager.cal2et('2028 FEB 08 05:31:08.0'), 'TDB')
    # earth   = scb.CelestialBody.from_constants('Earth')
    # test_sc = scb.Spacecraft('ORCCA', -1234)
    # pos_0   = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    # pos_1   = scb.ArrayWFrame(np.array([8000.0, 0.0, 0.0]), km, J2000)
    # vel_0   = scb.ArrayWFrame(np.array([0.0, 7.546049, 0.0]), km/sec, J2000)  
    # state_dict = [
    #     ("position", 3, "estimated", "dynamic", test_sc, pos_0),
    #     ("velocity", 3, "estimated", "dynamic", test_sc, vel_0),
    # ]

    # state_arr = scb.StateArray(epoch, earth, state=scb.StateDefinition.from_components(state_dict))

    # # add mass component
    # state_arr.add_to_state_vector(scb.StateDefinition.from_components([("eta_srp", 1, "estimated", "static", test_sc, scb.ArrayWFrame(scb.ArrayWUnits(1.5, None), J2000))]))

    # assert state_arr._state == scb.StateDefinition.from_components(state_dict + [("eta_srp", 1, "estimated", "static", test_sc, scb.ArrayWFrame(scb.ArrayWUnits(1.5, None), J2000))]).components


# NOTE: the rest of this class' methods are tested in integration tests