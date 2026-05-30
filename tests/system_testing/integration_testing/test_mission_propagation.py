# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np
import os

kg, km, sec, rad, N = scb.Units.get_units(["kg", "km", "sec", "rad", "N"])
J2000 = scb.Frame("J2000")


@pytest.fixture(scope="module")
def epochs_fb():
    time_0 = scb.SpiceManager.cal2et("2029 APR 01 01:30:00")
    time_f = scb.SpiceManager.cal2et("2029 APR 01 01:31:00")
    dt = 1
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array


@pytest.fixture(scope="module")
def epochs2():
    time_0 = scb.SpiceManager.cal2et("2029 APR 01 01:31:00")
    time_f = scb.SpiceManager.cal2et("2029 APR 01 03:01:00")
    dt = 60
    epoch_array = scb.EpochArray(np.arange(time_0, time_f, dt), sys="TDB")

    return epoch_array


@pytest.fixture(scope="module")
def maneuver():
    M1 = scb.Maneuver()
    M1.thrust = awu(0.200, N)
    M1.mass_flow = awu(0.012763666015287, kg / sec)
    M1.start_time = scb.EpochArray(923000647.3506516, 'TDB')
    M1.end_time = scb.EpochArray(923004047.3506516, 'TDB')
    M1.ux = awu(1, None)
    M1.uy = awu(0, None)
    M1.uz = awu(1, None)

    return M1


@pytest.fixture(scope="module")
def leg2_model(origin, sc):
    pos0 = scb.ArrayWFrame(
        np.full(3, np.nan),
        km,
        J2000,
    )
    vel0 = scb.ArrayWFrame(
        np.full(3, np.nan),
        km / sec,
        J2000,
    )
    leg2_model = scb.StateArray(
        epoch=scb.EpochArray(np.full(1, np.nan), sys="TDB"),
        origin=origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )
    return leg2_model


@pytest.fixture(scope="module")
def leg_model_fb(origin, sc):
    pos0 = scb.ArrayWFrame(
        np.full(3, np.nan),
        km,
        J2000,
    )
    vel0 = scb.ArrayWFrame(
        np.full(3, np.nan),
        km / sec,
        J2000,
    )
    return scb.StateArray(
        epoch=scb.EpochArray(np.full(1, np.nan), sys="TDB"),
        origin=origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )


@pytest.fixture(scope="module")
def force_model(sc):
    third_bodies = ["MERCURY", "VENUS", "SUN"]
    return scb.ForceModelTranslation(
        primary_body=sc, cannonball_SRP=True, third_bodies=third_bodies
    )


@pytest.fixture(scope="module")
def propagator1(sc, init_state, epochs, force_model):
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=init_state,
        tspan=epochs,
        force_models=force_model,
    )
    return prop


@pytest.fixture(scope="module")
def propagator_fb(sc, leg_model_fb, epochs_fb, maneuver):
    force_model_fb = scb.ForceModelTranslation(
        primary_body=sc, cannonball_SRP=True, finite_burn=True, maneuver=maneuver
    )
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=leg_model_fb,
        tspan=epochs_fb,
        force_models=force_model_fb,
    )
    return prop


@pytest.fixture(scope="module")
def propagator2(sc, leg2_model, epochs2, force_model):

    prop = scb.Propagator(
        primary_body=sc,
        state_vector=leg2_model,
        tspan=epochs2,
        force_models=force_model,
    )
    return prop


@pytest.fixture(scope="module")
def dv1_model():
    time_0 = scb.SpiceManager.cal2et("2029 APR 01 01:30:00")
    return scb.ImpulsiveBurn(
        dv_magnitude=scb.ArrayWUnits(np.array([5]), km / sec),
        dv_unit_vector=scb.ArrayWUnits(np.array([[0.164, 0.983, 0.0819]]), None),
        t_burn=scb.ArrayWUnits(time_0, sec),
    )


# endregion Fixtures #
# --------------------#


class TestEnvironment:

    # Test a scenario with both finite burn and impulsive burn models
    # propagator with a force model that includes cannonball SRP
    # validate legs and impulsive burn against
    # check that sequnce is consistent

    def test_scenario_fb(
        self,
        init_state,
        leg2_model,
        leg_model_fb,
        propagator1,
        propagator2,
        propagator_fb,
    ):
        dt = scb.ArrayWUnits(60, sec)
        MS = scb.MissionSequence("TEST_SEQUENCE")
        MS.addLeg("leg1", scb.ArrayWUnits(5400, sec), init_state, propagator1, dt, None)
        MS.addFiniteBurn(
            "fb1",
            scb.ArrayWUnits(60, sec),
            leg_model_fb,
            propagator_fb,
            scb.ArrayWUnits(1, sec),
        )
        MS.addLeg("leg2", scb.ArrayWUnits(5400, sec), leg2_model, propagator2, dt, None)

        mission = scb.ScenarioSetup(MS)
        mission.propagate()

        epochs = mission.propagated_state_array.epoch
        actual_duration = epochs[-1] - epochs[0]
        expected_duration = awu(5400 + 60 + 5400, sec)

        np.testing.assert_allclose(
            actual_duration.values,
            expected_duration.values,
            atol=1e-6,
            err_msg="The propagated state array duration does not match the MissionSequence total.",
        )

        temp_filename = "temp_mission_test.bsp"
        expected_full_path = os.path.join("data/kernels/scenario", temp_filename)

        try:
            orbiter_traj = scb.Trajectory(
                temp_filename, state_array=mission.propagated_state_array
            )
            assert os.path.exists(
                expected_full_path
            ), f"BSP was not created at {expected_full_path}"

            expected_initial_state = mission.propagated_state_array.state[0]
            traj_initial_state = orbiter_traj.get_state(
                mission.propagated_state_array.epoch[0]
            )
            assert (
                traj_initial_state["position"] == expected_initial_state[5][0].quantity
            ), "Initial position mismatch."
            assert (
                traj_initial_state["velocity"] == expected_initial_state[5][1].quantity
            ), "Initial velocity mismatch."
            assert (
                traj_initial_state["epoch"] == mission.propagated_state_array.epoch[0]
            ), "Epoch mismatch."
        finally:
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)

    def test_scenario_dv(
        self, init_state, leg2_model, dv1_model, propagator1, propagator2
    ):
        dt = scb.ArrayWUnits(60, sec)
        MS = scb.MissionSequence("TEST_SEQUENCE")
        MS.addLeg("leg1", scb.ArrayWUnits(5400, sec), init_state, propagator1, dt, None)
        MS.addBurn("dv1", dv1_model, propagator1)
        MS.addLeg("leg2", scb.ArrayWUnits(5400, sec), leg2_model, propagator2, dt, None)

        mission = scb.ScenarioSetup(MS)
        mission.propagate()

        epochs = mission.propagated_state_array.epoch
        actual_duration = epochs[-1] - epochs[0]
        expected_duration = awu(5400 + 60 + 5400, sec)

        np.testing.assert_allclose(
            actual_duration.values,
            expected_duration.values,
            atol=1e-6,
            err_msg="The propagated state array duration does not match the MissionSequence total.",
        )

        temp_filename = "temp_mission_test.bsp"
        expected_full_path = os.path.join("data/kernels/scenario", temp_filename)

        try:
            orbiter_traj = scb.Trajectory(
                temp_filename, state_array=mission.propagated_state_array
            )
            assert os.path.exists(
                expected_full_path
            ), f"BSP was not created at {expected_full_path}"

            expected_initial_state = mission.propagated_state_array.state[0]
            traj_initial_state = orbiter_traj.get_state(
                mission.propagated_state_array.epoch[0]
            )
            assert (
                traj_initial_state["position"] == expected_initial_state[5][0].quantity
            ), "Initial position mismatch."
            assert (
                traj_initial_state["velocity"] == expected_initial_state[5][1].quantity
            ), "Initial velocity mismatch."
            assert (
                traj_initial_state["epoch"] == mission.propagated_state_array.epoch[0]
            ), "Epoch mismatch."
        finally:
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)

    def test_scenario_dv_fb(
        self,
        init_state,
        leg2_model,
        dv1_model,
        leg_model_fb,
        propagator1,
        propagator2,
        propagator_fb,
    ):
        dt = scb.ArrayWUnits(60, sec)
        MS = scb.MissionSequence("TEST_SEQUENCE")
        MS.addLeg("leg1", scb.ArrayWUnits(5400, sec), init_state, propagator1, dt, None)
        MS.addBurn("dv1", dv1_model, propagator1)
        MS.addLeg("leg2", scb.ArrayWUnits(5400, sec), leg2_model, propagator2, dt, None)
        MS.addFiniteBurn(
            "fb1",
            scb.ArrayWUnits(60, sec),
            leg_model_fb,
            propagator_fb,
            scb.ArrayWUnits(1, sec),
        )

        mission = scb.ScenarioSetup(MS)
        mission.propagate()

        epochs = mission.propagated_state_array.epoch
        actual_duration = epochs[-1] - epochs[0]
        expected_duration = awu(5400 + 60 + 5400, sec)

        np.testing.assert_allclose(
            actual_duration.values,
            expected_duration.values,
            atol=1e-6,
            err_msg="The propagated state array duration does not match the MissionSequence total.",
        )

        temp_filename = "temp_mission_test.bsp"
        expected_full_path = os.path.join("data/kernels/scenario", temp_filename)

        try:
            orbiter_traj = scb.Trajectory(
                temp_filename, state_array=mission.propagated_state_array
            )
            assert os.path.exists(
                expected_full_path
            ), f"BSP was not created at {expected_full_path}"

            expected_initial_state = mission.propagated_state_array.state[0]
            traj_initial_state = orbiter_traj.get_state(
                mission.propagated_state_array.epoch[0]
            )
            assert (
                traj_initial_state["position"] == expected_initial_state[5][0].quantity
            ), "Initial position mismatch."
            assert (
                traj_initial_state["velocity"] == expected_initial_state[5][1].quantity
            ), "Initial velocity mismatch."
            assert (
                traj_initial_state["epoch"] == mission.propagated_state_array.epoch[0]
            ), "Epoch mismatch."
        finally:
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)
