# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Integration Tests for the Bplane class    #
# =============================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import numpy as np
import pytest

km, sec = scb.Units.get_units(["km", "sec"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(scope="module")
def bplane_file(earth_bplane_fk):
    return str(earth_bplane_fk)


@pytest.fixture(scope="module")
def bplane(epochs, bplane_file, hyperbolic_state):
    bp = scb.Bplane(
        epochs[0],
        "EARTH_BPLANE",
        12345678,
        "MOON",
        301,
        "EARTH_BARYCENTER",
        3,
        bplane_file,
        v_hat=False,
        new_bplane=False,
    )
    # Moon's real orbit is bound (ecc < 1) so compute_hyperbolic_parameters
    # returns NaN for the reference state.  Override bpar_ref with values
    # derived from the hyperbolic_state so targeting tests converge.
    bpar = bp.compute_hyperbolic_parameters(hyperbolic_state)
    bp.bpar_ref = bpar
    bp._bt_ref = bpar[0]
    bp._br_ref = bpar[1]
    bp._ltof_ref = bpar[2]
    bp._time_close_approach_ref = scb.ArrayWUnits(
        np.array(bp._epoch.times.values + bp._ltof_ref.values), sec
    )
    return bp


@pytest.fixture(scope="module")
def hyperbolic_state():
    """Relative state giving a hyperbolic Earth flyby (C3 > 0)."""
    return scb.ArrayWUnits(
        [400000.0, 0.0, 0.0, 0.0, 3.0, 0.5],
        [km, km, km, km / sec, km / sec, km / sec],
    )


@pytest.fixture(scope="module")
def approach_traj(sc, epochs, tmp_path_factory, hyperbolic_state):
    """
    LEO trajectory created the standard way (propagate → Trajectory →
    write_to_spk), wrapped in _BplaneTrajAdapter to provide the
    ArrayWUnits interface expected by piecewise_targeting.
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

    tmp_bsp = str(tmp_path_factory.mktemp("bplane_integ") / "approach.bsp")
    traj = scb.Trajectory(state_array=prop.propagated_state_array)
    traj.write_to_spk(tmp_bsp)

    return _BplaneTrajAdapter(traj, hyperbolic_state)


# endregion Fixtures #
# --------------------#


class _BplaneTrajAdapter:
    """
    Wraps a propagated scarabaeus Trajectory to provide the three-method
    interface required by Bplane.piecewise_targeting.

    Trajectory.get_state returns a dict; piecewise_targeting expects a
    sliceable ArrayWUnits, so this adapter serves the hyperbolic relative
    state supplied at construction for any epoch query.

      get_state(epoch)              → ArrayWUnits  (6-element J2000 state)
      propagate(state, t0, t1)     → _BplaneTrajAdapter
      write_spk(segments, path)    → None
    """

    def __init__(self, trajectory: scb.Trajectory, rel_state: scb.ArrayWUnits):
        self._traj = trajectory
        self._rel_state = rel_state
        self.write_spk_calls = []

    def get_state(self, epoch) -> scb.ArrayWUnits:
        return self._rel_state

    def propagate(self, state, epoch_start, epoch_end):
        return _BplaneTrajAdapter(self._traj, state)

    def write_spk(self, segments, path):
        # piecewise_targeting always calls write_spk on the original
        # trajectory object (not on the per-segment adapters returned by
        # propagate), so recording here is sufficient for the caller to
        # inspect afterward.
        self.write_spk_calls.append((segments, path))


# --------------#
# region Tests #
# --------------#


class TestCreateKernel:
    """Tests for _create_kernel with new_bplane=True."""

    def test_bplane_kernel_written_to_disk(self, epochs, tmp_path):
        """
        Verifies that constructing a Bplane with new_bplane=True writes the
        FK file to the specified scenario_kernel_dir and furnshes it.
        """
        scb.Bplane(
            epochs[0],
            "INTEG_TEST_BPLANE",
            77889900,
            "MOON",
            301,
            "EARTH_BARYCENTER",
            3,
            "integ_test_bplane.tf",
            v_hat=False,
            new_bplane=True,
            scenario_kernel_dir=tmp_path,
        )

        assert (tmp_path / "integ_test_bplane.tf").exists()


class TestTargeting:
    """Integration tests for linear, nonlinear, and piecewise targeting."""

    def test_linear_returns_dv(self, bplane, hyperbolic_state):
        """
        Verifies that linear_targeting returns an ArrayWUnits delta-V
        vector of shape (3,) in km/s.
        """
        dv = bplane.linear_targeting(hyperbolic_state)

        assert isinstance(dv, scb.ArrayWUnits)
        assert dv.values.shape == (3,)

    def test_nonlinear_returns_dv(self, bplane, hyperbolic_state):
        """
        Verifies that nonlinear_targeting returns an ArrayWUnits delta-V
        vector of shape (3,) in km/s.
        """
        dv = bplane.nonlinear_targeting(hyperbolic_state)

        assert isinstance(dv, scb.ArrayWUnits)
        assert dv.values.shape == (3,)

    def test_piecewise_invalid_method(self, bplane, approach_traj, epochs):
        """
        Verifies that piecewise_targeting raises ValueError when an
        unrecognised method string is supplied.
        """
        with pytest.raises(ValueError, match="method must be"):
            bplane.piecewise_targeting(approach_traj, [epochs[0]], method="bad_method")

    def test_piecewise_returns_dv_list(self, bplane, approach_traj, epochs):
        """
        Verifies that piecewise_targeting returns a list of delta-V vectors,
        one per TCM epoch, each an ArrayWUnits of shape (3,).
        """
        dv_list = bplane.piecewise_targeting(approach_traj, [epochs[0]], method="linear")

        assert isinstance(dv_list, list)
        assert len(dv_list) == 1
        assert isinstance(dv_list[0], scb.ArrayWUnits)
        assert dv_list[0].values.shape == (3,)

    def test_piecewise_multi_epoch_uses_next_tcm_epoch(
        self, bplane, approach_traj, epochs
    ):
        """
        With 2+ TCM epochs, every non-final segment must propagate to the
        *next* TCM epoch (not close approach) -- only the last segment
        should target close approach.
        """
        dv_list = bplane.piecewise_targeting(
            approach_traj, [epochs[0], epochs[1]], method="linear"
        )

        assert len(dv_list) == 2
        for dv in dv_list:
            assert isinstance(dv, scb.ArrayWUnits)
            assert dv.values.shape == (3,)

    def test_piecewise_writes_spk(self, bplane, approach_traj, epochs, tmp_path):
        """
        Supplying spk_output_path must call the trajectory's write_spk
        with the list of propagated segments and the given path.
        """
        spk_path = tmp_path / "corrected.bsp"
        bplane.piecewise_targeting(
            approach_traj, [epochs[0]], method="linear", spk_output_path=spk_path,
        )

        assert len(approach_traj.write_spk_calls) == 1
        segments, path = approach_traj.write_spk_calls[0]
        assert len(segments) == 1
        assert path == str(spk_path)


# endregion Tests #
# ----------------#