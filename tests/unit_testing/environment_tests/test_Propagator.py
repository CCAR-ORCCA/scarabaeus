# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import os
import pytest
import numpy as np

km, kg, sec, hr = scb.Units.get_units(["km", "kg", "sec", 'hr'])
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
    return awu(1e-06, km**2)


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
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    state_dict = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]
    return scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition.from_components(state_dict)
    )


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,
    )


# endregion Fixtures #
# --------------------#

# --------------#
# region Tests #
# --------------#


def test_initialization(sc, state, epochs, force_model):
    """
    Verifies that object is constructed correctly.
    """
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=epochs,
        force_models=force_model,
    )
    assert prop._primary_body == sc
    assert prop._prop_origin == state.origin
    assert prop._t0 == state.epoch
    assert prop._tspan == epochs


def test_convert_to_base_frame():
    """
    DESC
    """
    pytest.skip()


def test_accumulate_and_save_units_and_frames():
    """
    DESC
    """
    pytest.skip()


def test_accumulate_and_save_stm_units():
    """
    DESC
    """
    pytest.skip()


def test_accumulate_and_save_frame_ratios():
    """
    DESC
    """
    pytest.skip()


def test_get_event_function():
    """
    DESC
    """
    pytest.skip()


def test_translate_origin():
    """
    DESC
    """
    pytest.skip()


def test_convert_to_original_frames():
    """
    DESC
    """
    pytest.skip()


def test_precompute_indices():
    """
    DESC
    """
    pytest.skip()


def test_to_dict():
    """
    DESC
    """
    pytest.skip()


def test_initialize_dynamic_mass():
    """
    DESC
    """
    pytest.skip()


# NOTE: the rest of this class' methods are tested in integration tests
