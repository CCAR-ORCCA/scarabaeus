# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the Trajectory class  #
# =======================================#
"""
Trajectory stores the propagated state history of a spacecraft or
celestial body in a defined reference frame and origin body.

Tests
-----
test_initialization
    Verifies correct attribute assignment on construction from a
    StateArray.
"""
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
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


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


# ---------------------------------------------------#
# region  Mixed-unit estimated parameters            #
# ---------------------------------------------------#
@pytest.fixture
def state_mixed_units(epochs, origin, sc):
    """
    StateArray whose body solves for two parameters with DIFFERENT units.

    A delta-v (km/s) alongside a maneuver-epoch correction (sec) is the
    motivating case, but the same shape occurs for FiniteBurn's dimensionless
    k_thrust next to the angular dRA/dDEC.
    """
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(
        epochs[0],
        origin,
        state=(
            scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .param(
                "dv_smooth1",
                sc,
                scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                estimation="estimated",
                dynamics="static",
            )
            .param(
                "dt_dv1",
                sc,
                scb.ArrayWFrame(scb.ArrayWUnits(np.array([-12.5]), sec), None),
                estimation="estimated",
                dynamics="static",
            )
        ),
    )


def test_mixed_unit_parameters_are_accepted(state_mixed_units):
    """
    Parameters of one body must not be required to share units.

    This previously raised "Inconsistent units across parameters", which made
    it impossible to solve for a delta-v and a maneuver epoch together.
    """
    traj = scb.Trajectory(state_array=state_mixed_units)

    # Values stay stacked, so every existing consumer keeps its shape.
    assert traj.parameters is not None
    assert traj.parameters.values.shape[1] == 4

    # The honest per-parameter units are reported separately.
    names = [n for n, _, _ in traj.parameter_units]
    dims = [d for _, d, _ in traj.parameter_units]
    assert names == ["dv_smooth1", "dt_dv1"]
    assert dims == [3, 1]

    # No single unit describes the block, so the stacked array is left
    # dimensionless rather than mislabelled with whichever unit came first.
    assert not np.any(traj.parameters.units.dimensions.powers)


def test_uniform_unit_parameters_keep_shared_units(epochs, origin, sc):
    """The uniform case is unchanged: the stacked array still carries units."""
    traj = scb.Trajectory(
        state_array=scb.StateArray(
            epochs[0],
            origin,
            state=(
                scb.StateDefinition()
                .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
                .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
                .param(
                    "dv_smooth1",
                    sc,
                    scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
                    estimation="estimated",
                    dynamics="static",
                )
            ),
        )
    )

    assert traj.parameters.units == (km / sec)
    assert [n for n, _, _ in traj.parameter_units] == ["dv_smooth1"]


def test_mixed_unit_parameters_round_trip_through_sidecar(origin, sc, tmp_path):
    """
    The sidecar must record units per parameter, not one unit per file.

    Reading back a mixed-unit body yields the parameters individually, each
    with its own units, since no single ArrayWUnits could describe them.
    """
    epochs = scb.EpochArray(np.arange(0.0, 600.0, 60.0), sys="TDB")
    state = (
        scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5460491, 0.0]), km / sec, J2000))
        .param(
            "dv_smooth1",
            sc,
            scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000),
            estimation="estimated",
            dynamics="static",
        )
        .param(
            "dt_dv1",
            sc,
            scb.ArrayWFrame(scb.ArrayWUnits(np.array([-12.5]), sec), None),
            estimation="estimated",
            dynamics="static",
        )
    )
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=scb.StateArray(epochs[0], origin, state=state),
        tspan=epochs,
        force_models=scb.ForceModelTranslation(primary_body=sc),
        propagate_STM=False,
    )
    prop.propagate()

    spk = str(tmp_path / "mixed_units.bsp")
    scb.Trajectory(state_array=prop.propagated_state_array).write_to_spk(
        spk, append_if_exists=False
    )

    # Default return stays type-stable: one stacked ArrayWUnits, always.
    stacked = scb.SpiceManager.get_parameters(
        sc.name, 300.0, J2000, origin.name, folder_path_override=str(tmp_path)
    )
    assert isinstance(stacked, scb.ArrayWUnits)
    assert np.size(stacked.values) == 4
    # Mixed units have no single honest label, so the stack is dimensionless.
    assert not np.any(stacked.units.dimensions.powers)

    out = scb.SpiceManager.get_parameters(
        sc.name,
        300.0,
        J2000,
        origin.name,
        folder_path_override=str(tmp_path),
        per_parameter=True,
    )

    assert isinstance(out, dict), "per_parameter=True must split by parameter"
    assert set(out) == {"dv_smooth1", "dt_dv1"}
    assert out["dv_smooth1"].units == (km / sec)
    assert out["dt_dv1"].units == sec
    np.testing.assert_allclose(
        np.ravel(out["dv_smooth1"].values), [1e-3, -2e-3, 3e-3], atol=1e-9
    )
    np.testing.assert_allclose(
        float(np.ravel(out["dt_dv1"].values)[0]), -12.5, atol=1e-6
    )


# ---------------------------------------------------#
# region  STM interpolation gate                     #
# ---------------------------------------------------#
"""
Every ``get_STM`` / ``get_STM_sequence`` lookup that lands between two stored
epochs interpolates linearly and rejects the result when the estimated error
exceeds ``max_interp_error``. The estimate is curvature-based and reported
relative to the matrix being interpolated, because the error of linear
interpolation is set by the second derivative, not the first, and because an
STM's Frobenius norm grows with the arc.

Tests
-----
test_error_estimate_matches_repropagated_truth
    The reported estimate agrees with the error measured against a finely
    propagated reference.
test_error_estimate_falls_as_the_square_of_the_step
    Halving the grid step cuts the estimate by about four, which is what makes
    "use a finer grid" an actionable instruction.
test_fine_grid_on_a_long_arc_is_accepted
    A long arc no longer trips the gate at a fine step; the first-difference
    form it replaces failed here regardless of the grid.
test_coarse_grid_is_rejected
    A step genuinely too coarse to interpolate is still refused.
test_tolerance_can_be_relaxed_and_disabled
    The trajectory's tolerance is honoured by the lookups, and None skips.
test_negative_tolerance_is_reported
    A tolerance that is not a positive relative number is refused.
test_tolerance_can_be_set_at_construction
    The constructor argument reaches the property, and defaults to 1e-3.
test_two_point_grid_has_no_curvature_to_check
    With only two nodes there is no third point to take a curvature from, so
    the lookup interpolates instead of failing.
"""

_stm_interp_rel_error = scb.Trajectory._stm_interp_rel_error


def _analytic_stm_grid(step, n, size=4):
    """
    A smooth, exactly-known matrix history on a uniform grid.

    Built from ``exp(A t)`` for a fixed ``A``, so the true value at any epoch is
    available in closed form and the gate's estimate can be checked against a
    real interpolation error rather than against itself.
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(3)
    A = rng.normal(scale=0.35, size=(size, size))
    times = np.arange(n, dtype=float) * step
    flats = [expm(A * t).ravel() for t in times]
    return times, flats, A


def test_error_estimate_matches_repropagated_truth():
    """The reported estimate agrees with the error measured against truth."""
    from scipy.linalg import expm

    step = 0.25
    times, flats, A = _analytic_stm_grid(step, 12)

    i = 6
    alpha = 0.5
    reported = _stm_interp_rel_error(times, flats, i, alpha)

    t_mid = times[i - 1] + alpha * step
    interpolated = (1.0 - alpha) * flats[i - 1] + alpha * flats[i]
    truth = expm(A * t_mid).ravel()
    measured = np.linalg.norm(interpolated - truth) / np.linalg.norm(truth)

    assert reported == pytest.approx(measured, rel=0.05)


def test_error_estimate_falls_as_the_square_of_the_step():
    """Halving the grid step cuts the estimate by about four."""
    # The bracket must start at the same epoch every time: refining the grid
    # while holding the *index* fixed would sample a different part of the
    # curve and measure nothing about the step.
    t_bracket = 1.2
    estimates = []
    for step in (0.4, 0.2, 0.1):
        i = int(round(t_bracket / step)) + 1
        times, flats, _ = _analytic_stm_grid(step, i + 3)
        assert times[i - 1] == pytest.approx(t_bracket)
        estimates.append(_stm_interp_rel_error(times, flats, i, 0.5))

    assert estimates[0] / estimates[1] == pytest.approx(4.0, rel=0.15)
    assert estimates[1] / estimates[2] == pytest.approx(4.0, rel=0.15)


def _trajectory_with_stms(state, epochs, sc, step_seconds, hours):
    """Propagate a short arc and hang its STMs off a Trajectory."""
    grid = scb.EpochArray.interval(
        float(np.asarray(epochs.et).ravel()[0]),
        float(np.asarray(epochs.et).ravel()[0]) + hours * 3600.0,
        scb.ArrayWUnits(step_seconds, sec),
        sys="TDB",
    )
    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=grid,
        force_models=scb.ForceModelTranslation(primary_body=sc),
        display_progress=False,
    )
    propagator.propagate()
    trajectory = scb.Trajectory("gate", state_array=propagator.propagated_state_array)
    trajectory.add_STMs(propagator.STM, timestamps=grid)
    return trajectory, np.asarray(grid.et).ravel()


def test_fine_grid_on_a_long_arc_is_accepted(state, epochs, sc):
    """A long arc no longer trips the gate at a fine step."""
    trajectory, times = _trajectory_with_stms(state, epochs, sc, 10.0, hours=8)

    # Probes strictly between nodes, so every one of them interpolates.
    for t in times[1:-1:60]:
        trajectory.get_STM(epoch=float(t) + 4.0)


def test_coarse_grid_is_rejected(state, epochs, sc):
    """A step genuinely too coarse to interpolate is still refused."""
    trajectory, times = _trajectory_with_stms(state, epochs, sc, 900.0, hours=8)

    with pytest.raises(ValueError, match="relative interpolation error"):
        for t in times[1:-1]:
            trajectory.get_STM(epoch=float(t) + 400.0)


def test_tolerance_can_be_relaxed_and_disabled(state, epochs, sc):
    """The trajectory's tolerance is honoured by the lookups, and None skips."""
    trajectory, times = _trajectory_with_stms(state, epochs, sc, 900.0, hours=4)
    probe = float(times[2]) + 400.0

    with pytest.raises(ValueError, match="relative interpolation error"):
        trajectory.get_STM(epoch=probe)

    trajectory.max_interp_error = 1.0
    relaxed = trajectory.get_STM(epoch=probe)

    trajectory.max_interp_error = None
    unchecked = trajectory.get_STM(epoch=probe)

    assert np.allclose(relaxed, unchecked)


def test_negative_tolerance_is_reported(state):
    """A tolerance that is not a positive relative number is refused."""
    trajectory = scb.Trajectory("gate", state_array=state)

    with pytest.raises(ValueError, match="positive relative tolerance"):
        trajectory.max_interp_error = -1.0


@pytest.mark.parametrize("configured", [1e-2, None])
def test_tolerance_can_be_set_at_construction(state, configured):
    """The constructor argument reaches the property."""
    trajectory = scb.Trajectory(
        "gate", state_array=state, max_interp_error=configured
    )

    assert trajectory.max_interp_error == configured
    assert scb.Trajectory("gate", state_array=state).max_interp_error == 1e-3


def test_two_point_grid_has_no_curvature_to_check():
    """With only two nodes the lookup interpolates instead of failing."""
    times = np.array([0.0, 1.0])
    flats = [np.eye(3).ravel(), (2.0 * np.eye(3)).ravel()]

    assert _stm_interp_rel_error(times, flats, 1, 0.5) is None


# ---------------------------------------------------#
# region  Skipping the STM sidecar                   #
# ---------------------------------------------------#
"""
The STM sidecar is the largest artifact a trajectory writes — one matrix per
output epoch, growing with the square of the state size, and an OD run writes
one per iteration. The filter serves its own STM lookups from the in-memory
history, so a run that never reads the sidecar back can skip writing it.

Tests
-----
test_stm_sidecar_written_by_default
    The default behaviour is unchanged.
test_stm_sidecar_can_be_skipped
    write_stms=False leaves the SPK and the other sidecars in place.
test_skipping_the_sidecar_keeps_the_stms_in_memory
    Only the file is skipped; lookups still work.
test_output_settings_flag_reaches_the_writer
    OutputSettings.save_trajectory_stms is what the filter and the solution
    pass down to write_to_spk.
"""


def _written_trajectory(state, epochs, sc, tmp_path, **write_kwargs):
    """Propagate a short arc and write it, returning the trajectory and paths."""
    trajectory, _ = _trajectory_with_stms(state, epochs, sc, 60.0, hours=1)
    spk = tmp_path / "arc.bsp"
    trajectory.write_to_spk(str(spk), **write_kwargs)
    return trajectory, spk


def test_stm_sidecar_written_by_default(state, epochs, sc, tmp_path):
    """The default behaviour is unchanged."""
    _written_trajectory(state, epochs, sc, tmp_path)

    assert (tmp_path / "arc_stms.json").exists()


def test_stm_sidecar_can_be_skipped(state, epochs, sc, tmp_path):
    """write_stms=False leaves the SPK and the other sidecars in place."""
    _written_trajectory(state, epochs, sc, tmp_path, write_stms=False)

    assert not (tmp_path / "arc_stms.json").exists()
    assert (tmp_path / "arc.bsp").exists(), "the state history must still be written"


def test_skipping_the_sidecar_keeps_the_stms_in_memory(state, epochs, sc, tmp_path):
    """Only the file is skipped; lookups still work."""
    trajectory, _ = _written_trajectory(
        state, epochs, sc, tmp_path, write_stms=False
    )

    stm = trajectory.get_STM(epoch=float(np.asarray(trajectory.STMs_timestamp)[2]))

    assert stm.shape[0] == stm.shape[1]
    assert np.isfinite(stm).all()


@pytest.mark.parametrize("configured, expected", [(True, True), (False, False)])
def test_output_settings_flag_reaches_the_writer(configured, expected):
    """OutputSettings.save_trajectory_stms is what gets passed to write_to_spk."""

    class StubSolution:
        output_settings = scb.OutputSettings(save_trajectory_stms=configured)

    formats = scb.SolutionOD._trajectory_write_formats(StubSolution())

    assert formats["write_stms"] is expected


# ---------------------------------------------------#
# region  Multi-body SPK output                      #
# ---------------------------------------------------#
"""
A trajectory whose state carries more than one propagated body writes one SPK
segment per body — into a single file by default, or into one file per body
with ``per_body_spk``. Either way each body's parameters sidecar is named after
that body: they all write ``_parameters.json``, so a shared name left only the
last body's parameters on disk.

Tests
-----
test_parameter_entries_excludes_position_and_velocity
    A second propagated body's pos/vel are not parameters of the first.
test_parameter_entries_narrow_to_one_body
    A sidecar describes its own body's parameters, not every body's.
test_body_tagged_path_is_unique_per_body
    The per-body path/stem is derived from the body name.
"""

from scarabaeus.timeAndFrame.SpiceManager import SpiceManager as _SpiceManager


class _Owner:
    def __init__(self, name, spice_id):
        self._name = name
        self.name = name
        self.spice_id = spice_id


def _two_body_state_definition():
    """State of [sc pos, sc vel, body pos, body vel, eta_srp, gm]."""
    sc_owner = _Owner("ORBITER", -1092)
    body_owner = _Owner("Earth", 399)
    return [
        ("position", 3, "estimated", "dynamic", sc_owner, None),
        ("velocity", 3, "estimated", "dynamic", sc_owner, None),
        ("position", 3, "estimated", "dynamic", body_owner, None),
        ("velocity", 3, "estimated", "dynamic", body_owner, None),
        ("eta_srp", 1, "estimated", "static", sc_owner, None),
        ("gm", 1, "estimated", "static", body_owner, None),
    ]


def test_parameter_entries_excludes_position_and_velocity():
    """A second propagated body's pos/vel are not parameters of the first."""
    entries = _SpiceManager._parameter_entries(_two_body_state_definition())

    assert [e[0] for e in entries] == ["eta_srp", "gm"], (
        "the old rule skipped the first two entries, which labelled the second "
        "body's position and velocity as parameters"
    )


@pytest.mark.parametrize(
    "body_id, expected", [(-1092, ["eta_srp"]), (399, ["gm"])]
)
def test_parameter_entries_narrow_to_one_body(body_id, expected):
    """A sidecar describes its own body's parameters, not every body's."""
    entries = _SpiceManager._parameter_entries(
        _two_body_state_definition(), body_id
    )

    assert [e[0] for e in entries] == expected


def test_body_tagged_path_is_unique_per_body():
    """The per-body path/stem is derived from the body name."""
    tagged = [
        scb.Trajectory._body_tagged_path("/out/mission.bsp", _Owner(n, i))
        for n, i in (("ORBITER", -1092), ("Earth", 399))
    ]

    assert tagged == ["/out/mission_ORBITER.bsp", "/out/mission_Earth.bsp"]
    assert len(set(tagged)) == 2

