# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Integration Tests for Mission Propagation     #
# ================================================#

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
    third_bodies = scb.CelestialBody.from_constants(["MERCURY", "VENUS", "SUN"])
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
def init_state_srp(origin, sc, epochs):
    pos0 = scb.ArrayWFrame(np.array([6778.0, 0.0, 0.0]), km, J2000)
    vel0 = scb.ArrayWFrame(np.array([0.0, 7.6, 0.0]), km / sec, J2000)
    eta0 = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.0]), None), None)
    state_def = (
        scb.StateDefinition()
        .position(sc, pos0)
        .velocity(sc, vel0)
        .param("eta_srp", sc, eta0, dynamics="static")
    )
    return scb.StateArray(epoch=epochs[0], origin=origin, state=state_def)


@pytest.fixture(scope="module")
def propagator_srp(sc, init_state_srp, epochs, force_model):
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=init_state_srp,
        tspan=epochs,
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

    def test_finite_burn_leg(
        self,
        tmp_path,
        init_state,
        leg2_model,
        leg_model_fb,
        propagator1,
        propagator2,
        propagator_fb,
    ):
        """End-to-end propagation of a mission sequence that includes a
        finite burn leg alongside a standard propagation leg."""
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
        expected_full_path = str(tmp_path / temp_filename)

        try:
            orbiter_traj = scb.Trajectory(state_array=mission.propagated_state_array)
            orbiter_traj.write_to_spk(expected_full_path)
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
            # write_to_spk() furnsh's the SPK into the global SPICE kernel
            # pool so get_state() can query it; unload it before deleting
            # the file, or a later test's kernel scan (e.g. ThreeBodyGravity's
            # ephemeris cache) will try to reopen a now-missing path.
            scb.SpiceManager.unload_kernel_from_pool(expected_full_path)
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)

    def test_impulsive_burn_leg(
        self, tmp_path, init_state, leg2_model, dv1_model, propagator1, propagator2
    ):
        """End-to-end propagation of a mission sequence with an impulsive
        delta-V maneuver between two propagation legs, checking total
        duration and trajectory file consistency."""
        dt = scb.ArrayWUnits(60, sec)
        MS = scb.MissionSequence("TEST_SEQUENCE")
        MS.addLeg("leg1", scb.ArrayWUnits(5400, sec), init_state, propagator1, dt, None)
        MS.addBurn("dv1", dv1_model, propagator1)
        MS.addLeg("leg2", scb.ArrayWUnits(5400, sec), leg2_model, propagator2, dt, None)

        mission = scb.ScenarioSetup(MS)
        mission.propagate()

        epochs = mission.propagated_state_array.epoch
        actual_duration = epochs[-1] - epochs[0]
        expected_duration = awu(5400 + 5400, sec)  # impulsive burn is instantaneous

        np.testing.assert_allclose(
            actual_duration.values,
            expected_duration.values,
            atol=1e-6,
            err_msg="The propagated state array duration does not match the MissionSequence total.",
        )

        temp_filename = "temp_mission_test.bsp"
        expected_full_path = str(tmp_path / temp_filename)

        try:
            orbiter_traj = scb.Trajectory(state_array=mission.propagated_state_array)
            orbiter_traj.write_to_spk(expected_full_path)
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
            # write_to_spk() furnsh's the SPK into the global SPICE kernel
            # pool so get_state() can query it; unload it before deleting
            # the file, or a later test's kernel scan (e.g. ThreeBodyGravity's
            # ephemeris cache) will try to reopen a now-missing path.
            scb.SpiceManager.unload_kernel_from_pool(expected_full_path)
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)

        # Trajectory can also be built directly from the propagated
        # ScenarioSetup (leg_array=), rather than its propagated_state_array.
        # This mission has 3 sequence entries (leg1, dv1 [Impulsive Burn],
        # leg2), so it also exercises the Impulsive-Burn-skip branch and the
        # duplicate-boundary-sample drop between leg1 and leg2.
        leg_traj = scb.Trajectory(leg_array=mission)

        assert leg_traj.epoch == mission.total_epochsTDB
        assert np.allclose(leg_traj.position.values, mission.total_position.quantity.values)
        assert np.allclose(leg_traj.velocity.values, mission.total_velocity.quantity.values)
        assert len(leg_traj.body) == mission.total_epochsTDB.size
        assert len(leg_traj.origin) == mission.total_epochsTDB.size
        assert len(leg_traj.ref_frame) == mission.total_epochsTDB.size

    def test_impulsive_and_finite_burn(
        self,
        tmp_path,
        init_state,
        leg2_model,
        dv1_model,
        leg_model_fb,
        propagator1,
        propagator2,
        propagator_fb,
    ):
        """End-to-end propagation of a mission sequence combining both
        impulsive delta-V and finite burn maneuvers."""
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
        expected_full_path = str(tmp_path / temp_filename)

        try:
            orbiter_traj = scb.Trajectory(state_array=mission.propagated_state_array)
            orbiter_traj.write_to_spk(expected_full_path)
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
            # write_to_spk() furnsh's the SPK into the global SPICE kernel
            # pool so get_state() can query it; unload it before deleting
            # the file, or a later test's kernel scan (e.g. ThreeBodyGravity's
            # ephemeris cache) will try to reopen a now-missing path.
            scb.SpiceManager.unload_kernel_from_pool(expected_full_path)
            if os.path.exists(expected_full_path):
                os.remove(expected_full_path)

    def test_static_parameter(self, sc, init_state_srp, propagator_srp):
        """
        Verifies that a static scalar parameter (eta_srp) accumulated across
        a propagated leg is repeated over the segment's epochs in
        _total_components, and collapsed back to a single representative
        value in the scenario's full_state_vector.
        """
        MS = scb.MissionSequence("STATIC_PARAM_SEQUENCE")
        MS.addLeg(
            "leg1",
            scb.ArrayWUnits(300, sec),
            init_state_srp,
            propagator_srp,
            scb.ArrayWUnits(60, sec),
            None,
        )
        mission = scb.ScenarioSetup(MS)
        mission.propagate(verbose=False)

        eta_key = ("eta_srp", 1, sc.spice_id, None, 0)
        eta_total = mission.total_components[eta_key]
        eta_vals = np.asarray(eta_total["awf"].quantity.values)
        assert eta_vals.shape == (eta_total["epochs"].size, 1)
        np.testing.assert_allclose(eta_vals, 1.0)

        fsv_eta = next(
            awf for name, *_, awf in mission.full_state_vector.state if name == "eta_srp"
        )
        assert np.asarray(fsv_eta.quantity.values).shape == ()
