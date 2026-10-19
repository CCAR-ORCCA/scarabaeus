# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Physical regression checks for the public SolutionOD covariance mapper."""
from types import SimpleNamespace

import numpy as np
import pytest
import scarabaeus as scb


EPOCH0 = 9e8
DAY = 86400.0


def ballistic_stm(epoch):
    return np.array([[1.0, epoch - EPOCH0], [0.0, 1.0]])


def ballistic_covariance(epoch, diagonal):
    dt = epoch - EPOCH0
    position, velocity = diagonal
    return np.array([[position + dt**2 * velocity, dt * velocity],
                     [dt * velocity, velocity]])


@pytest.fixture
def solution():
    # Two final updates; the second has received more information.
    times = EPOCH0 + np.array([DAY, 2 * DAY, 2 * DAY])
    history = np.array([ballistic_covariance(t, diagonal)
                        for t, diagonal in zip(times, (
                            (16.0, 4e-10), (4.0, 1e-10), (1.0, 1e-12)))])
    filt = SimpleNamespace(
        n=2, covariance_analysis=True, has_consider=False,
        settings=SimpleNamespace(output=scb.OutputSettings()),
        measurement_data=SimpleNamespace(get_spacecraft_times=lambda: times),
        trajectory=SimpleNamespace(get_STM=ballistic_stm),
        propagator=SimpleNamespace(full_state_vector=SimpleNamespace(
            epoch=scb.EpochArray(EPOCH0, sys="TDB"))),
        initial_covariance=SimpleNamespace(
            matrix_without_units=lambda: np.diag([100.0, 1e-8])),
        covariance_history=history,
    )
    return scb.SolutionOD(filt)


def mapped(solution, epochs, **kwargs):
    kwargs.setdefault("lookback_s", 0.0)
    return solution.propagate_covariance(
        scb.EpochArray(np.atleast_1d(epochs), sys="TDB"),
        use_smoothed=False, **kwargs,
    )


@pytest.mark.parametrize("offset,lookback,all_data,diagonal", [
    (-DAY, 0, False, (100.0, 1e-8)),  # Prior mapped backwards.
    (DAY / 2, 0, False, (100.0, 1e-8)),  # Before any measurements.
    (DAY, 0, False, (16.0, 4e-10)),
    (DAY + 3600, 0, False, (16.0, 4e-10)),  # ET tolerance must not skip an hour.
    (2 * DAY, 0, False, (1.0, 1e-12)),  # Include both updates at the cutoff.
    (2 * DAY + 3600, 0, False, (1.0, 1e-12)),  # Past last measurement, stored STM.
    (10 * DAY, 14 * DAY, False, (100.0, 1e-8)),  # Rolling cutoff before data.
    (15 * DAY, 14 * DAY, False, (16.0, 4e-10)),
    (DAY / 2, 0, True, (1.0, 1e-12)),
    (2 * DAY + 3600, 0, True, (1.0, 1e-12)),
])
def test_data_cutoffs_and_stm_mapping(solution, offset, lookback, all_data, diagonal):
    epoch = EPOCH0 + offset
    actual = mapped(solution, epoch, lookback_s=lookback, use_all_data=all_data)[0]
    np.testing.assert_allclose(actual, ballistic_covariance(epoch, diagonal),
                               rtol=1e-12, atol=1e-12)


def test_unsorted_history_and_repeated_requests_preserve_last_update(solution):
    order = [1, 0, 2]
    solution.timestamps = solution.timestamps[order]
    solution.covariance_est = solution.covariance_est[order]
    epochs = EPOCH0 + np.array([2 * DAY, DAY + 3600, 2 * DAY])
    actual = mapped(solution, epochs)
    for cov, t, diagonal in zip(actual, epochs, ((1, 1e-12), (16, 4e-10), (1, 1e-12))):
        np.testing.assert_allclose(cov, ballistic_covariance(t, diagonal))


def test_covariance_analysis_retains_consider_history_and_timeline(solution):
    filt = solution.filter
    filt.has_consider = True
    filt.covariance_consider_history = 2 * filt.covariance_history
    filt.timeline_times = solution.timestamps.copy()
    sol = scb.SolutionOD(filt)
    assert sol.covariance_consider is filt.covariance_consider_history
    np.testing.assert_array_equal(sol.timeline, filt.timeline_times)
    t = EPOCH0 + DAY / 2
    np.testing.assert_allclose(mapped(sol, t, use_all_data=True)[0],
                               ballistic_covariance(t, (2.0, 2e-12)))
    np.testing.assert_allclose(mapped(sol, t, use_all_data=True, include_consider=False)[0],
                               ballistic_covariance(t, (1.0, 1e-12)))


def test_missing_consider_history_is_not_silently_ignored(solution):
    expected = mapped(solution, EPOCH0 + DAY)
    solution.filter.has_consider = True
    with pytest.warns(RuntimeWarning, match="no consider-augmented"):
        actual = mapped(solution, EPOCH0 + DAY)
    np.testing.assert_allclose(actual, expected)


def test_smoothed_consider_maps_smoothed_covariance_with_warning(solution):
    # Plotting maps the smoothed covariance with include_consider left True.
    solution.filter.has_consider = True
    solution.filter.covariance_consider_history = 2 * solution.covariance_est
    solution.covariance_smooth = [ballistic_covariance(t, (1.0, 1e-12))
                                  for t in solution.timestamps]
    t = EPOCH0 + DAY + 3600
    with pytest.warns(RuntimeWarning, match="smoothed covariance"):
        actual = solution.propagate_covariance(scb.EpochArray([t], sys="TDB"))[0]
    np.testing.assert_allclose(actual, ballistic_covariance(t, (1.0, 1e-12)))


def test_considered_acceleration_and_cross_covariance_are_mapped(solution):
    # A constant uncertain acceleration c: x(t)=x0+v0*dt+c*dt^2/2.
    # Retain both Var(c) and Cov(x0,c); mapping only the estimated block fails.
    def stm(epoch):
        dt = epoch - EPOCH0
        return np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
    prior = np.array([[4., 0., .1], [0., .01, 0.], [.1, 0., .25]])
    solution.filter.trajectory.get_STM = stm
    solution.filter.has_consider = True
    solution.timestamps = np.array([EPOCH0])
    solution.covariance_consider = [prior]
    solution.covariance_est = [np.diag([4., .01, 0.])]
    dt = 10.
    actual = mapped(solution, EPOCH0 + dt)[0]
    assert actual[0, 0] == pytest.approx(4 + dt**2*.01 + dt**4*.25/4 + dt**2*.1)
    assert actual[0, 2] == pytest.approx(.1 + dt**2*.25/2)
    assert actual[1, 2] == pytest.approx(dt*.25)
    assert actual[2, 2] == .25


def test_smoothed_selection_and_rolling_cutoff_conflict(solution):
    solution.covariance_smooth = [ballistic_covariance(t, (1.0, 1e-12))
                                  for t in solution.timestamps]
    t = EPOCH0 + DAY + 3600
    actual = solution.propagate_covariance(scb.EpochArray([t], sys="TDB"))[0]
    np.testing.assert_allclose(actual, ballistic_covariance(t, (1.0, 1e-12)))
    with pytest.raises(ValueError, match="smoothed"):
        solution.propagate_covariance(scb.EpochArray([t], sys="TDB"), lookback_s=DAY)


def test_existing_default_mapping_convention_is_preserved(solution):
    # Other callers, including propagate_state_covariance, map the final
    # estimate inside the arc. They opt into a causal product explicitly.
    epochs = EPOCH0 + np.array([DAY/2, DAY+3600, 3*DAY])
    actual = solution.propagate_covariance(
        scb.EpochArray(epochs, sys="TDB"), use_smoothed=False)
    for cov, t, diagonal in zip(actual, epochs, ((16., 4e-10), (1., 1e-12), (1., 1e-12))):
        np.testing.assert_allclose(cov, ballistic_covariance(t, diagonal))


@pytest.mark.parametrize("kwargs", [
    {"lookback_s": -1}, {"lookback_s": np.nan},
    {"lookback_s": DAY, "use_all_data": True},
])
def test_invalid_cutoffs_raise(solution, kwargs):
    with pytest.raises(ValueError):
        mapped(solution, EPOCH0 + DAY, **kwargs)


def test_missing_prior_raises_and_override_has_its_own_epoch(solution):
    solution.filter.initial_covariance = None
    with pytest.raises(ValueError, match="prior covariance"):
        mapped(solution, EPOCH0 + DAY/2)
    prior_t = EPOCH0 + 100
    t = EPOCH0 + 200
    actual = mapped(solution, t, apriori_epoch=prior_t,
                    apriori_covariance=np.diag([9., .04]))[0]
    np.testing.assert_allclose(actual, [[409., 4.], [4., .04]])


def test_timeline_shape_mismatch_raises(solution):
    solution.timeline = solution.timestamps[:1]
    with pytest.raises(ValueError, match="timeline lengths"):
        mapped(solution, EPOCH0 + DAY)


def test_sequence_uses_event_rows_and_keeps_simultaneous_updates(solution):
    solution.filter.flag_sequence = True
    solution.filter.legs_epochs = [np.array([EPOCH0, EPOCH0+10]),
                                   np.array([EPOCH0+10, EPOCH0+20])]
    solution.timeline = EPOCH0 + np.array([0., 10., 10., 15.])
    solution.covariance_est = [np.eye(2)*p for p in (100., 50., 25., 4.)]
    solution.filter.trajectory.get_STM_sequence = (
        lambda epoch, idx_leg: np.array([[1., epoch-EPOCH0-10*idx_leg], [0., 1.]])
    )
    actual = mapped(solution, EPOCH0 + np.array([10., 12., 15.]))
    np.testing.assert_array_equal(actual[0], np.eye(2)*25.)
    np.testing.assert_allclose(actual[1], [[125., 50.], [50., 25.]])
    np.testing.assert_array_equal(actual[2], np.eye(2)*4.)


@pytest.mark.parametrize("failure", ["coverage", "interpolation"])
def test_reintegration_preserves_reference_and_cutoff(solution, failure, monkeypatch):
    """Exercise the real propagator fallback against exact free-flight dynamics.

    Both forward and backward targets force integration outside the stored
    grid. The early target must keep the prior even while the final target
    uses the posterior. Repeated calls must leave the fitted propagator intact.
    """
    km, sec, kg = scb.Units.get_units(["km", "sec", "kg"])
    earth = scb.CelestialBody.from_constants("EARTH")
    earth.grav_param = scb.ArrayWUnits(0., km**3/sec**2)
    craft = scb.Spacecraft("covariance_free_flight", -99123, scb.ArrayWUnits(1000., kg))
    frame = scb.Frame("J2000")
    initial = scb.StateArray(
        scb.EpochArray(EPOCH0, sys="TDB"), earth,
        scb.StateDefinition().position(craft, scb.ArrayWFrame([7000., 0., 0.], km, frame))
        .velocity(craft, scb.ArrayWFrame([0., 1., 0.], km/sec, frame)),
    )
    prop = scb.Propagator(
        primary_body=craft, state_vector=initial,
        tspan=scb.EpochArray(EPOCH0 + np.arange(0., 101., 10.), sys="TDB"),
        force_models=scb.ForceModelTranslation(primary_body=craft),
    )
    prop.propagate(display_progress=False)
    trajectory = scb.Trajectory("covariance_free_flight", state_array=prop.propagated_state_array)
    trajectory.add_STMs(prop.STM, timestamps=prop.propagated_state_array.epoch)
    if failure == "interpolation":
        # Simulate the actual Trajectory error spelling, previously not caught.
        def unavailable(epoch):
            raise ValueError("Estimated relative interpolation error exceeds threshold")
        monkeypatch.setattr(trajectory, "get_STM", unavailable)
    filt = solution.filter
    filt.propagator, filt.trajectory, filt.n = prop, trajectory, 6
    prior = np.diag([4., 9., 16., .01, .04, .09])
    filt.initial_covariance = SimpleNamespace(matrix_without_units=lambda: prior)
    solution.timestamps = EPOCH0 + np.array([50., 100.])
    def expected(dt, factor):
        p, v = np.diag(prior)[:3]*factor, np.diag(prior)[3:]*factor
        return np.block([[np.diag(p+dt**2*v), np.diag(dt*v)],
                         [np.diag(dt*v), np.diag(v)]])
    solution.covariance_est = [expected(50, .5), expected(100, .25)]
    # These identify accidental reinitialization of the live filter propagator.
    original_state = prop.full_state_vector
    original_span = prop.tspan
    original_stm = prop.STM
    epochs = EPOCH0 + np.array([150., -50., 25., 150.])
    actual = mapped(solution, epochs)
    for cov, dt, factor in zip(actual, epochs-EPOCH0, (.25, 1., 1., .25)):
        np.testing.assert_allclose(cov, expected(dt, factor), rtol=1e-10, atol=1e-10)
    assert prop.full_state_vector is original_state
    assert prop.tspan is original_span
    assert prop.STM is original_stm
    np.testing.assert_allclose(mapped(solution, epochs), actual, rtol=1e-10, atol=1e-10)
