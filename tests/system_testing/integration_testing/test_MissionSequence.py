# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import os
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])


J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


### use conftest fixture
@pytest.fixture
def epochs():
    time_0 = scb.SpiceManager.jd2et(2461809.72995654 + 1 / 3)
    time_f = scb.SpiceManager.jd2et(2461809.72995654 + 1)
    dt = 60 * 60
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array


### use conftest fixture
@pytest.fixture
def origin():

    origin = scb.CelestialBody.from_constants("SUN")

    return origin


### use conftest fixture
@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp
    )


### use conftest fixture
@pytest.fixture
def state_dict(sc):
    state_definition = [
        (
            "position",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(
                scb.ArrayWUnits(
                    np.array(
                        [-2.6561672815488e08, 1.8837080839972e08, 1.0199307996915e08]
                    ),
                    km,
                ),
                J2000,
            ),
        ),
        (
            "velocity",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(
                scb.ArrayWUnits(
                    np.array([-9.0779264528807, -12.4823165310294, -4.4599484708606]),
                    km / sec,
                ),
                J2000,
            ),
        ),
    ]
    return state_definition


@pytest.fixture
def state(epochs, origin, state_dict):
    """Define stateArray object"""
    return scb.StateArray(
        epochs[0], origin, state=scb.StateDefinition.from_components(state_dict)
    )


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,
    )


@pytest.fixture
def propagator(sc, state, epochs, force_model):

    prop = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model
    )
    return prop


@pytest.fixture
def dv1_model():
    return scb.ImpulsiveBurn(
        scb.ArrayWFrame(scb.ArrayWUnits(np.array([5]), km / sec), J2000),
        scb.ArrayWFrame(
            scb.ArrayWUnits(np.array([0.16384638, 0.9830783, 0.08192319]), km / km),
            J2000,
        ),
    )


@pytest.fixture
def state_fb_def():
    leg_state_definition_fb_M1 = [
        (
            "position",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(np.full(3, np.nan), km, J2000),
        ),
        (
            "velocity",
            3,
            "estimated",
            "dynamic",
            sc,
            scb.ArrayWFrame(np.full(3, np.nan), km / sec, J2000),
        ),
    ]
    return leg_state_definition_fb_M1


@pytest.fixture
def leg_model_fb(origin, state_fb_def):
    return scb.StateArray(
        epoch=scb.EpochArray(np.full(1, np.nan), sys="TDB"),
        origin=origin,
        state=scb.StateDefinition.from_components(state_fb_def),
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
@pytest.mark.parametrize("init_args", [("TEST SEQUENCE",)], ids=["name"])
def test_initialization(init_args):
    """
    Verifies that object is constructed correctly.
    """
    miss_seq = scb.MissionSequence(*init_args)
    assert miss_seq.name == "TEST SEQUENCE"


# fix this
def test_add_leg(epochs, origin, state_dict, propagator):
    """
    Verifies that legs are added to the sequence correctly.
    """
    leg1_model = scb.StateArray(
        epoch=epochs[0],
        origin=origin,
        state=scb.StateDefinition.from_components(state_dict),
    )

    miss_seq = scb.MissionSequence("TEST SEQUENCE")
    miss_seq.addLeg(
        "LEG 1",
        scb.ArrayWUnits(58000, sec),
        leg1_model,
        propagator,
        scb.ArrayWUnits(360, sec),
    )
    assert miss_seq._names == ["LEG 1"]


# fix this
def test_add_burn(epochs, origin, state_dict, propagator, dv1_model):
    """
    Verifies that legs are added to the sequence correctly.
    """
    leg1_model = scb.StateArray(
        epoch=epochs[0],
        origin=origin,
        state=scb.StateDefinition.from_components(state_dict),
    )

    miss_seq = scb.MissionSequence("TEST SEQUENCE")
    miss_seq.addLeg(
        "LEG 1",
        scb.ArrayWUnits(58000, sec),
        leg1_model,
        propagator,
        scb.ArrayWUnits(360, sec),
    )
    miss_seq.addBurn("Burn 1", dv1_model, propagator)
    assert miss_seq._names == ["LEG 1", "Burn 1"]


# fix this
# def test_add_finite_burn(epochs, origin, state_dict, propagator, leg_model_fb):
#     """
#     Verifies that legs are added to the sequence correctly.
#     """
#     leg1_model = scb.StateArray(
#         epoch=epochs[0],
#         origin=origin,
#         state=scb.StateDefinition.from_components(state_dict),
#     )

#     miss_seq = scb.MissionSequence("TEST SEQUENCE")
#     miss_seq.addLeg(
#         "LEG 1",
#         scb.ArrayWUnits(58000, sec),
#         leg1_model,
#         propagator,
#         scb.ArrayWUnits(360, sec),
#     )
#     miss_seq.add_finite_burn(
#         "Burn 2",
#         scb.ArrayWUnits(30, sec),
#         leg_model_fb,
#         propagator,
#         scb.ArrayWUnits(1, sec),
#     )
#     assert miss_seq._names == ["LEG 1", "Burn 2"]
