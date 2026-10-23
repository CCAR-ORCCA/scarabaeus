# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Physical and data-integrity regressions for SolutionOD post-processing."""
from types import SimpleNamespace
import pickle

import numpy as np
import pytest
import scarabaeus as scb


T0 = 9e8
km, sec, kg = scb.Units.get_units(["km", "sec", "kg"])


def solution_for(**attributes):
    filt = SimpleNamespace(
        n=6, covariance_analysis=False, has_consider=False,
        settings=SimpleNamespace(output=scb.OutputSettings()),
        measurement_data=SimpleNamespace(get_spacecraft_times=lambda: T0 + np.array([50., 100.])),
    )
    for name, value in attributes.items():
        setattr(filt, name, value)
    return scb.SolutionOD(filt)


def phi(dt):
    return np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])


def flat(state):
    return np.concatenate([np.asarray(entry[5].quantity.values).ravel() for entry in state.state])


@pytest.fixture
def ballistic():
    earth = scb.CelestialBody.from_constants("EARTH")
    earth.grav_param = scb.ArrayWUnits(0., km**3/sec**2)
    craft = scb.Spacecraft("solution_free_flight", -99124, scb.ArrayWUnits(1000., kg))
    frame = scb.Frame("J2000")
    initial = scb.StateArray(
        scb.EpochArray(T0, sys="TDB"), earth,
        scb.StateDefinition().position(craft, scb.ArrayWFrame([7000., 0., 0.], km, frame))
        .velocity(craft, scb.ArrayWFrame([0., 1., 0.], km/sec, frame)),
    )
    prop = scb.Propagator(
        primary_body=craft, state_vector=initial,
        tspan=scb.EpochArray(T0 + np.arange(0., 101., 10.), sys="TDB"),
        force_models=scb.ForceModelTranslation(primary_body=craft),
    )
    prop.propagate(display_progress=False)
    trajectory = scb.Trajectory("solution_free_flight", state_array=prop.propagated_state_array)
    trajectory.add_STMs(prop.STM, timestamps=prop.propagated_state_array.epoch)
    correction = np.array([1., 2., 3., .1, .2, .3])
    sol = solution_for(
        propagator=prop, trajectory=trajectory,
        state_deviation_history=[phi(50) @ correction*2, phi(100) @ correction],
        covariance_history=[phi(50) @ np.eye(6)*2 @ phi(50).T,
                            phi(100) @ np.eye(6) @ phi(100).T],
        initial_covariance=SimpleNamespace(matrix_without_units=lambda: np.eye(6)*4),
    )
    return sol, correction


@pytest.mark.parametrize("force_fallback", [False, True])
def test_state_propagation_adds_correction_once_and_preserves_filter(ballistic, monkeypatch, force_fallback):
    sol, dx = ballistic
    prop = sol.filter.propagator
    original = (prop.full_state_vector, prop.tspan, prop.STM, prop.propagated_state_array)
    times = np.array([150., -50., 25., 100., 150.])
    if force_fallback:
        def unavailable(epoch):
            raise ValueError("Estimated relative interpolation error exceeds threshold")
        monkeypatch.setattr(sol.filter.trajectory, "get_STM", unavailable)
    actual = sol.propagate_state(scb.EpochArray(T0 + times, sys="TDB"), use_smoothed=False)
    for state, dt, scale in zip(actual, times, [1., 2., 2., 1., 1.]):
        expected = np.array([7000., dt, 0., 0., 1., 0.]) + phi(dt) @ dx*scale
        np.testing.assert_allclose(flat(state), expected, atol=1e-9, rtol=1e-12)
        assert float(state.epoch.et) == T0 + dt
    assert all(a is b for a, b in zip(original, (
        prop.full_state_vector, prop.tspan, prop.STM, prop.propagated_state_array)))
    repeated = sol.propagate_state(scb.EpochArray(T0 + times, sys="TDB"), use_smoothed=False)
    np.testing.assert_allclose([flat(x) for x in repeated], [flat(x) for x in actual])


@pytest.mark.parametrize("lookback,all_data,factors", [(0., False, [0., 2., 1.]),
                                                      (50., False, [0., 0., 1.]),
                                                      (0., True, [1., 1., 1.])])
def test_paired_mapping_uses_same_cutoff(ballistic, lookback, all_data, factors):
    sol, dx = ballistic
    times = [25., 75., 150.]
    pairs = sol.propagate_state_covariance(
        scb.EpochArray(T0 + np.array(times), sys="TDB"), use_smoothed=False,
        lookback_s=lookback, use_all_data=all_data)
    for (state, cov), dt, factor in zip(pairs, times, factors):
        np.testing.assert_allclose(flat(state), [7000., dt, 0., 0., 1., 0.] + phi(dt) @ dx*factor, atol=1e-9)
        cov_factor = factor if factor else 4.
        np.testing.assert_allclose(cov, phi(dt) @ np.eye(6)*cov_factor @ phi(dt).T, atol=1e-8)


def test_paired_mapping_rejects_partial_smoothing(ballistic):
    sol, _ = ballistic
    sol.deviation_smooth = sol.deviation_est.copy()
    with pytest.raises(ValueError, match="Both smoothed"):
        sol.propagate_state_covariance(scb.EpochArray(T0 + 50., sys="TDB"))


def test_initial_mapping_honors_nonidentity_stm_origin(ballistic, monkeypatch):
    sol, dx = ballistic
    transform = np.diag(np.arange(1., 7.))
    monkeypatch.setattr(sol.filter.trajectory, "get_STM", lambda epoch: phi(epoch-T0) @ transform)
    np.testing.assert_allclose(sol.map_state_deviation_to_epoch(use_smoothed=False), dx)
    sol.deviation_smooth = sol.deviation_est.copy()
    np.testing.assert_allclose(sol.map_state_deviation_to_epoch(), 2*dx)


def test_propagated_state_preserves_distinct_bodies_and_parameter_blocks(ballistic):
    sol, _ = ballistic
    initial = sol.filter.propagator.full_state_vector
    craft2 = scb.Spacecraft("solution_second_body", -99125, scb.ArrayWUnits(100., kg))
    frame = scb.Frame("J2000")
    definition = scb.StateDefinition.from_components(list(initial.state))
    definition.position(craft2, scb.ArrayWFrame([-2000., 3000., 1000.], km, frame))
    definition.velocity(craft2, scb.ArrayWFrame([2., 3., 4.], km/sec, frame))
    definition.param("range_bias_test", craft2,
                     scb.ArrayWFrame(scb.ArrayWUnits([10.], km), None), dynamics="static")
    nominal = scb.StateArray(scb.EpochArray(T0+50., sys="TDB"), initial.origin, definition)
    sol.filter.propagator = SimpleNamespace(full_state_vector=nominal)
    sol.filter.trajectory = SimpleNamespace(_state_array=nominal, get_STM=lambda epoch: np.eye(13))
    sol.deviation_est = np.array([np.arange(13.), np.arange(13.)*2])
    corrected = sol.propagate_state(scb.EpochArray(T0+50., sys="TDB"),
                                     use_smoothed=False, lookback_s=0.)[0]
    np.testing.assert_allclose(flat(corrected)[:6], [7000., 1., 2., 3., 5., 5.])
    np.testing.assert_allclose(flat(corrected)[6:12], [-1994., 3007., 1008., 11., 13., 15.])
    assert flat(corrected)[12] == 22.
    assert corrected.state[2][4] is craft2
    assert corrected.state[0][4] is initial.state[0][4]


def measurement_solution():
    def nominal(epoch_input, idx_leg_input=None):
        return dict(position=scb.ArrayWUnits([100., 200., 300.], km),
                    velocity=scb.ArrayWUnits([1., 2., 3.], km/sec),
                    parameters=scb.ArrayWUnits([10., 20.], km))
    sol = solution_for(trajectory=SimpleNamespace(get_state=nominal), n=8)
    sol.timeline = T0 + np.array([100., 25., 50., 100.])
    sol.deviation_est = np.array([np.full(8, 1.), np.full(8, 2.), np.full(8, 3.), np.full(8, 4.)])
    return sol


def test_estimated_trajectory_uses_last_simultaneous_update_and_parameter_width():
    sol = measurement_solution()
    pos, vel, par = sol.estimated_trajectory(scb.EpochArray(T0 + np.array([100., 50., 100.]), sys="TDB"))
    np.testing.assert_allclose(pos, np.array([100., 200., 300.]) + np.array([4., 3., 4.])[:, None])
    np.testing.assert_allclose(par, np.array([10., 20.]) + np.array([4., 3., 4.])[:, None])
    assert vel.shape == (3, 3)
    np.testing.assert_allclose(sol.state_est[1], [104., 204., 304., 5., 6., 7., 14., 24.])


def test_nearby_epoch_requires_propagation():
    sol = measurement_solution()
    with pytest.raises(ValueError, match="propagate_state"):
        sol.estimated_trajectory(scb.EpochArray(T0 + 50.1, sys="TDB"))
    pos, vel, params = sol.estimated_trajectory(scb.EpochArray([], sys="TDB"))
    assert pos.shape == vel.shape == (0, 3)
    assert params is None


def test_sequence_measurement_queries_use_leg_spans_and_requested_parameter_rows():
    sol = measurement_solution()
    sol.filter.flag_sequence = True
    sol.filter.legs_epochs = [T0 + np.array([0., 100.]), T0 + np.array([100., 200.])]
    sol.filter.legs_n = [8, 8]
    called = []
    get_state = sol.filter.trajectory.get_state
    def nominal(epoch_input, idx_leg_input=None):
        called.append(idx_leg_input)
        return get_state(epoch_input, idx_leg_input)
    sol.filter.trajectory.get_state = nominal
    _, _, params = sol.estimated_trajectory(scb.EpochArray(T0 + np.array([100., 50., 100.]), sys="TDB"))
    assert called == [1, 0, 1]
    np.testing.assert_allclose(params[0], [[13., 23.]])
    np.testing.assert_allclose(params[1], [[14., 24.], [14., 24.]])
    # Sequence storage can include inactive padding from a wider leg.
    sol.deviation_est = np.pad(sol.deviation_est, ((0, 0), (0, 1)))
    assert np.isnan(sol.state_est[:, -1]).all()
    np.testing.assert_allclose(sol.state_est[:, 6:8], [[13., 23.], [14., 24.]])


def test_state_est_missing_reference_is_nan():
    sol = measurement_solution()
    get_state = sol.filter.trajectory.get_state
    def nominal(epoch_input, idx_leg_input=None):
        if float(epoch_input.et) == T0 + 50.:
            raise ValueError("Missing reference sample")
        return get_state(epoch_input, idx_leg_input)
    sol.filter.trajectory.get_state = nominal
    with pytest.warns(RuntimeWarning, match="left as NaN"):
        states = sol.state_est
    assert np.isnan(states[0]).all()
    assert np.isfinite(states[1]).all()


def test_timeline_mismatch_is_not_silently_indexed():
    sol = measurement_solution()
    sol.timeline = sol.timeline[:-1]
    with pytest.raises(ValueError, match="timeline lengths differ"):
        sol.estimated_trajectory(scb.EpochArray(T0 + 50., sys="TDB"))
    with pytest.raises(ValueError, match="timeline lengths differ"):
        sol.map_state_deviation_to_epoch()


def test_sequence_mapping_keeps_measurement_before_node():
    sol = solution_for(flag_sequence=True, legs_n=[6, 6],
                       legs_epochs=[T0 + np.array([0., 100.]), T0 + np.array([100., 200.])])
    sol.timeline = T0 + np.array([0., 50., 100., 150.])
    sol.deviation_est = np.array([np.ones(6)*i for i in [1., 2., 3., 4.]])
    sol.filter.trajectory = SimpleNamespace(get_STM_sequence=lambda epoch, idx_leg: np.eye(6))
    # np.isclose(9e8+50, 9e8+100) is True; it must not discard the 50 s row.
    np.testing.assert_allclose(sol.map_state_deviation_to_epoch(use_smoothed=False), [np.ones(6)*2, np.ones(6)*4])
    sol.deviation_smooth = sol.deviation_est.copy()
    np.testing.assert_allclose(sol.map_state_deviation_to_epoch(), [np.ones(6), np.ones(6)*3])


def two_leg_sequence(ballistic, lookup_error):
    """Leg, Node, Leg solution whose stored sequence STMs all fail with ``lookup_error``."""
    from copy import deepcopy
    sol, dx = ballistic
    first = sol.filter.propagator
    initial_second = first.propagated_state_array.get_state_at_epoch(scb.EpochArray(T0 + 100., sys="TDB"))
    # A different second-leg velocity exposes accidental reuse of another leg.
    initial_second = sol._correct_state(initial_second, [0., 0., 0., 0., 1., 0.])
    second = deepcopy(first)
    second.reinitialize(state_vector=initial_second,
                        tspan=scb.EpochArray(T0 + np.arange(100., 201., 10.), sys="TDB"))
    second.propagate(display_progress=False)
    # Like a later sequence leg's model, the second one is only a layout
    # template: the leg starts from the state carried across the node.
    template = sol._correct_state(initial_second, [0., 0., 0., 0., -2., 0.])
    sol.filter.flag_sequence = True
    sol.filter.legs_epochs = [T0 + np.array([0., 100.]), T0 + np.array([100., 200.])]
    sol.filter.legs_model = [first.full_state_vector, template]
    sol.filter.legs_n = [6, 6]
    sol.filter.propagator = second  # ScenarioSetup can expose the final propagator.
    sol.timeline = T0 + np.array([50., 100., 150., 200.])
    sol.deviation_est = np.array([phi(50) @ dx, 2*dx, phi(50) @ (2*dx), phi(100) @ (2*dx)])
    sol.covariance_est = [phi(50) @ phi(50).T, np.eye(6)*2,
                          phi(50)*2 @ phi(50).T, phi(100)*2 @ phi(100).T]
    def unavailable(epoch, idx_leg):
        raise ValueError(lookup_error)
    def stored(epoch_input, idx_leg_input):
        state = [first, second][idx_leg_input].propagated_state_array.get_state_at_epoch(epoch_input)
        return {entry[0]: entry[5].quantity for entry in state.state}
    # The node holds the next leg's propagator too.
    sol.filter.trajectory = SimpleNamespace(
        get_STM_sequence=unavailable, get_state=stored,
        sequence_definition=SimpleNamespace(
            models=[first.full_state_vector, object(), template],
            props=[first, second, second]),
    )
    return sol, dx, first, second


def test_sequence_fallback_uses_own_leg_model_and_actual_reference_epoch(ballistic):
    sol, dx, first, second = two_leg_sequence(
        ballistic, "Estimated relative interpolation error exceeds threshold")
    original_stms = first.STM, second.STM
    epochs = scb.EpochArray(T0 + np.array([-25., 125., 225.]), sys="TDB")
    pairs = sol.propagate_state_covariance(epochs, use_smoothed=False)
    for (state, covariance), dt, factor, nominal in zip(
        pairs, [-25., 25., 125.], [1., 2., 2.],
        [[7000., -25., 0., 0., 1., 0.], [7000., 150., 0., 0., 2., 0.],
         [7000., 350., 0., 0., 2., 0.]],
    ):
        np.testing.assert_allclose(flat(state), nominal + phi(dt) @ dx*factor, atol=1e-9)
        np.testing.assert_allclose(covariance, phi(dt)*factor @ phi(dt).T, atol=1e-8)
    assert first.STM is original_stms[0]
    assert second.STM is original_stms[1]


def test_sequence_state_fallback_outside_leg_stm_grid(ballistic):
    # Trajectory.get_STM_sequence refuses epochs off a leg's grid this way.
    sol, dx, _, _ = two_leg_sequence(
        ballistic, "Epoch 1.0 before first STM timestamp 2.0 (no extrapolation).")
    states = sol.propagate_state(scb.EpochArray(T0 + np.array([-25., 225.]), sys="TDB"),
                                 use_smoothed=False)
    for state, dt, factor, nominal in zip(
        states, [-25., 125.], [1., 2.],
        [[7000., -25., 0., 0., 1., 0.], [7000., 350., 0., 0., 2., 0.]],
    ):
        np.testing.assert_allclose(flat(state), nominal + phi(dt) @ dx*factor, atol=1e-9)


def test_empty_deviation_uses_actual_filter_dimension():
    sol = solution_for(n=11)
    assert sol.map_state_deviation_to_epoch().shape == (11,)


def test_numpy_smoothed_history_can_be_stored():
    sol = solution_for(smoothed_state_deviation=np.ones((2, 6)))
    sol.output_settings.save_smoothed_solution = True
    sol._store_solution_data()
    np.testing.assert_array_equal(sol.deviation_smooth, np.ones((2, 6)))


@pytest.mark.parametrize("scale", [1., 100., 1e308])
def test_weighted_uq_covariance_and_ess_use_same_distribution(scale):
    sol = solution_for(weights=np.array([.25, .75])*scale)
    particles = np.array([[[0., 1.], [0., 2.]], [[2., 3.], [4., 6.]]])
    metrics = sol.UQ_compute(particles, confidence_levels=[.5])
    np.testing.assert_allclose(metrics["mean"], [[1.5, 3.], [2.5, 5.]])
    np.testing.assert_allclose(metrics["covariances"], [[[.75, 1.5], [1.5, 3.]]]*2)
    np.testing.assert_allclose(metrics["std"]**2, [[.75, 3.]]*2)
    np.testing.assert_allclose(metrics["effective_sample_size"], [1.6, 1.6])
    assert sol.uq_metrics is metrics


def test_uq_single_particle_is_a_point_distribution():
    metrics = solution_for().UQ_compute(np.array([[[42.]]]))
    np.testing.assert_array_equal(metrics["covariances"], [[[0.]]])
    np.testing.assert_array_equal(metrics["std"], [[0.]])
    np.testing.assert_array_equal(metrics["effective_sample_size"], [1.])


@pytest.mark.parametrize("weights", [[0., 0.], [-1., 2.], [1., np.nan], [1., 2., 3.]])
def test_uq_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="Weights"):
        solution_for(weights=weights).UQ_compute(np.ones((2, 1, 1)))


def test_serialization_preserves_metadata_and_exports_without_filter(tmp_path):
    sol = measurement_solution()
    sol._state_frame, sol._state_units = "J2000", ["km"]*8
    sol.prefit_epochs = {"range": [T0 + 50.]}
    sol.postfit_epochs = sol.prefit_epochs.copy()
    sol.postfit_epochs_smooth = sol.prefit_epochs.copy()
    sol.multi_arc_global = {"P_G": np.diag([4., 9.])}
    sol.filter.has_consider = True
    sol.filter.flag_sequence = True
    sol.filter.legs_epochs = [T0 + np.array([0., 100.]), T0 + np.array([100., 200.])]
    sol.filter.events_epochs = [T0 + 100.]
    path = tmp_path / "solution.pkl"
    sol.serialize(path)
    restored = scb.SolutionOD.deserialize(path)
    np.testing.assert_array_equal(restored._timeline_epochs(), sol.timeline)
    np.testing.assert_array_equal(restored.global_std_devs(), [2., 3.])
    assert restored._state_frame == "J2000"
    assert restored._state_units == sol._state_units
    assert restored.prefit_epochs == sol.prefit_epochs
    assert restored.postfit_epochs_smooth == sol.postfit_epochs_smooth
    assert restored.filter.has_consider
    assert restored._sequence_extras() == sol._sequence_extras()
    restored.create_JSON(tmp_path / "restored.json")
    restored.serialize(tmp_path / "roundtrip.pkl")


def test_legacy_pickle_gets_missing_attribute_defaults(tmp_path):
    sol = solution_for()
    path = tmp_path / "legacy.pkl"
    sol.serialize(path)
    with path.open("rb") as stream:
        data = pickle.load(stream)
    for key in ("timeline", "state_frame", "state_units", "sequence_extras", "filter_metadata",
                "multi_arc_global", "prefit_epochs", "postfit_epochs", "postfit_epochs_smooth"):
        data.pop(key)
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    restored = scb.SolutionOD.deserialize(path)
    assert restored.multi_arc_global is None
    assert restored._state_frame is None
    np.testing.assert_array_equal(restored._timeline_epochs(), sol.timestamps)
    restored.create_JSON(tmp_path / "legacy.json")
