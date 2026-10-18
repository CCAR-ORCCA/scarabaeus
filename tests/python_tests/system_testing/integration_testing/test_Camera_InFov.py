# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Integration Tests for the Camera class   #
# ============================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import numpy as np
import pytest

km, sec, rad = scb.Units.get_units(["km", "sec", "rad"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(scope="module")
def orbiter_traj(sc, epochs, tmp_path_factory):
    """
    LEO trajectory BSP kernel for the test spacecraft (SPICE ID -1000).
    Writes the BSP to a temporary file and furnshes it into the SPICE
    kernel pool so target_in_fov can resolve the spacecraft state.
    """
    origin = scb.CelestialBody.from_constants("EARTH")

    a = 6778  # km, ~400 km altitude LEO
    mu = origin.grav_param.values
    v = np.sqrt(mu / a)

    pos0 = scb.ArrayWFrame(awu(np.array([a, 0.0, 0.0]), km), J2000)
    vel0 = scb.ArrayWFrame(awu(np.array([0.0, v, 0.0]), km / sec), J2000)

    x0 = scb.StateArray(
        epochs[0],
        origin,
        state=scb.StateDefinition().position(sc, pos0).velocity(sc, vel0),
    )

    cannon = scb.ForceModelTranslation(primary_body=sc, cannonball_SRP=True)
    prop = scb.Propagator(
        primary_body=sc, state_vector=x0, tspan=epochs, force_models=cannon
    )
    prop.propagate()

    tmp_bsp = str(tmp_path_factory.mktemp("camera_integ") / "orbiter.bsp")
    traj = scb.Trajectory(state_array=prop.propagated_state_array)
    traj.write_to_spk(tmp_bsp)
    return traj


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#


class TestTargetInFov:
    """Integration tests for Camera.target_in_fov using real SPICE ephemeris."""

    def test_single_epoch_returns_bool(self, camera, orbiter_traj, epochs):
        """
        Verifies that a single-epoch target_in_fov query returns a scalar
        bool (True or False) without raising an exception.
        """
        earth = scb.CelestialBody.from_constants("EARTH")
        result = camera.target_in_fov(earth, epochs[0])

        assert isinstance(result, (bool, np.bool_))

    def test_multiple_epochs_returns_array(self, camera, orbiter_traj, epochs):
        """
        Verifies that a multi-epoch query returns a boolean ndarray whose
        length matches the number of input epochs.
        """
        earth = scb.CelestialBody.from_constants("EARTH")
        result = camera.target_in_fov(earth, epochs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (epochs.size,)
        assert np.issubdtype(result.dtype, np.bool_)

    def test_consistent_single_vs_first_of_array(self, camera, orbiter_traj, epochs):
        """
        Verifies that the scalar result for a single epoch matches the first
        element returned by the multi-epoch query.
        """
        earth = scb.CelestialBody.from_constants("EARTH")
        single = camera.target_in_fov(earth, epochs[0])
        multi = camera.target_in_fov(earth, epochs)

        assert bool(single) == bool(multi[0])

    def test_falls_back_to_inertial_frame_on_spice_error(self, orbiter_traj, epochs):
        """
        Verifies target_in_fov's except-fallback branch: when the primary
        SpiceManager.get_state query (in the camera's own _camera_frame)
        fails -- e.g. because the frame isn't recognized by SPICE -- it
        falls back to querying in J2000 relative to associated_body_spice_id
        and rotating with get_rotation_to_camera.

        Builds its own Camera (rather than mutating the shared session
        fixture) with a Frame whose name has been mutated to something
        SPICE doesn't recognize, but the same associated_body_spice_id
        (-1000) as `orbiter_traj` so the fallback query has real ephemeris
        to resolve against.
        """
        bogus_frame = scb.Frame("J2000")
        bogus_frame._name = "BOGUS_CAMERA_FRAME_FOR_FALLBACK_TEST"

        fallback_camera = scb.Camera(
            name="Fallback Test Camera",
            associated_body_spice_id=-1000,
            camera_frame=bogus_frame,
            fov_angular=scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad),
            fov_pixels=(2000, 2000),
            focal_length=scb.ArrayWUnits(150e-6, km),
            ifov_angle=scb.ArrayWUnits(np.array([25e-6, 25e-6]), rad),
            measurement_type="image",
        )

        earth = scb.CelestialBody.from_constants("EARTH")
        result = fallback_camera.target_in_fov(earth, epochs[0])

        assert isinstance(result, (bool, np.bool_))


# endregion Tests #
# ----------------#
