# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Integration Tests for the GroundStation class #
# ================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import numpy as np
import pytest
import spiceypy as spice
km, sec = scb.Units.get_units(["km", "sec"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(scope="module", autouse=True)
def orbiter_traj(sc, epochs, tmp_path_factory, setup):
    """
    LEO trajectory BSP kernel for the test spacecraft (SPICE ID -1000).
    Writes and furnshes the BSP so SPICE can resolve the spacecraft state
    during ground station queries.
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

    tmp_bsp = str(tmp_path_factory.mktemp("gs_integ") / "orbiter.bsp")
    traj = scb.Trajectory(state_array=prop.propagated_state_array)
    traj.write_to_spk(tmp_bsp)
    spice.furnsh(tmp_bsp)
    return traj, tmp_bsp


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#


class TestStationVisibility:
    """Integration tests for GroundStation.station_visibility."""

    def test_single_epoch_returns_bool(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that a single-epoch visibility query returns a scalar bool
        without raising an exception.
        """
        result = GS1.station_visibility(sc, epochs[0])

        assert isinstance(result, (bool, np.bool_))

    def test_multiple_epochs_returns_array(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that a multi-epoch query returns a boolean ndarray whose
        length matches the number of input epochs.
        """
        result = GS1.station_visibility(sc, epochs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (epochs.size,)
        assert np.issubdtype(result.dtype, np.bool_)

    def test_returns_angles_tuple(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that return_angles=True yields a (visible, elevation, azimuth)
        3-tuple with scalar types for a single epoch.
        """
        result = GS1.station_visibility(sc, epochs[0], return_angles=True)

        assert len(result) == 3
        visible, elevation, azimuth = result
        assert isinstance(visible, (bool, np.bool_))
        assert isinstance(elevation, (float, np.floating))
        assert isinstance(azimuth, (float, np.floating))

    def test_multiple_epochs_returns_angles_arrays(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that return_angles=True with multiple epochs yields a
        (visible_array, elevation_array, azimuth_array) 3-tuple whose
        arrays each have length matching the epoch array.
        """
        result = GS1.station_visibility(sc, epochs, return_angles=True)

        assert len(result) == 3
        visible, elevation, azimuth = result
        assert isinstance(visible, np.ndarray)
        assert isinstance(elevation, np.ndarray)
        assert isinstance(azimuth, np.ndarray)
        assert visible.shape == (epochs.size,)
        assert elevation.shape == (epochs.size,)
        assert azimuth.shape == (epochs.size,)


class TestElevation:
    """Integration tests for GroundStation.elevation."""

    def test_single_epoch_returns_float(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that a single-epoch elevation query returns a scalar float
        without raising an exception.
        """
        result = GS1.elevation(sc, epochs[0])

        assert isinstance(result, (float, np.floating))

    def test_multiple_epochs_returns_array(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that a multi-epoch elevation query returns a float ndarray
        whose length matches the number of input epochs.
        """
        result = GS1.elevation(sc, epochs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (epochs.size,)

    def test_visible_implies_positive_elevation(self, GS1, sc, orbiter_traj, epochs):
        """
        Verifies that any epoch where the spacecraft is visible has an
        elevation angle strictly greater than zero.
        """
        visible = GS1.station_visibility(sc, epochs)
        elev = GS1.elevation(sc, epochs)

        if np.any(visible):
            assert np.all(elev[visible] > 0.0)


# endregion Tests #
# ----------------#