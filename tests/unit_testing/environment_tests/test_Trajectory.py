# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import os
import pytest
import numpy as np

km, sec, kg, hr = scb.Units.get_units(["km", "sec", "kg", 'hr'])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


# setup
### use conftest fixture
@pytest.fixture
def epochs():
    """Epoch to define state."""
    return scb.EpochArray.interval('2021 January 1, 00:00:00', '2021 January 7, 00:00:00',
                                   scb.ArrayWUnits(1, hr), sys = 'TDB')


### use conftest fixture
@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


### remove
@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)


### remove
@pytest.fixture
def sc_ref_coeff():
    return 1.5


### use conftest fixture
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft"""
    return scb.Spacecraft(
        "ORCCA",
        -1234,
        tot_mass=scb.ArrayWUnits(1500.0, kg),
        area=sc_area,
        ref_coeff=sc_ref_coeff,
    )


### use conftest fixture
@pytest.fixture
def state(epochs, origin, sc):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(
        np.array(
            [
                [7000.0, 0.0, 0.0],
                [8000.0, 0.0, 0.0],
                [7000.0, 0.0, 0.0],
                [8000.0, 0.0, 0.0],
            ]
        ),
        km,
        J2000,
    )
    vel_0 = scb.ArrayWFrame(
        np.array(
            [[0.0, 70.0, 0.0], [0.0, 80.0, 10.0], [0.0, 70.0, 0.0], [0.0, 80.0, 10.0]]
        ),
        km / sec,
        J2000,
    )

    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]
    state = scb.StateArray(
        epochs[0:4], origin, state=scb.StateDefinition.from_components(state_dict)
    )
    return state


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(state):
    """
    Verifies that object is constructed correctly.
    """
    orbiter_traj = scb.Trajectory(state_array=state)


# NOTE: the rest of this class' methods are tested in integration tests
