# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the CustomMeasurement class  #
# =============================================#
import src.scarabaeus as scb
import numpy as np
import pytest

kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])
J2000 = scb.Frame("J2000")

SIGMA = scb.ArrayWUnits(1e-3, km)


def plain_range(position):
    """Geometric range — the observable RangeIdeal computes."""
    return np.linalg.norm(position)


def biased_range(position, params):
    """Geometric range plus an estimated station bias."""
    return np.linalg.norm(position) + params["station_bias"][0]


def range_partial_by_position(position):
    """d(range)/d(r) = r_hat, as a (1, 3) row."""
    return (position / np.linalg.norm(position)).reshape(1, 3)


def bias_partial(position):
    """d(range)/d(bias) = 1."""
    return np.ones((1, 1))


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(autouse=True)
def isolate_custom_registry():
    """Keep the process-wide custom-parameter registry from leaking between tests."""
    saved = dict(scb.StateDefinition._custom_state_components)
    yield
    scb.StateDefinition._custom_state_components.clear()
    scb.StateDefinition._custom_state_components.update(saved)


@pytest.fixture
def gs():
    """Observing ground station."""
    return scb.GroundStation("DSS-14")


@pytest.fixture
def sc():
    """Observed spacecraft."""
    return scb.Spacecraft(
        "Orbiter",
        -1000,
        scb.ArrayWUnits(2000.0, kg),
        scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )


@pytest.fixture
def origin():
    """Propagation origin."""
    return scb.CelestialBody.from_constants("EARTH")


@pytest.fixture
def epochs():
    """Short epoch grid covering the propagated arc."""
    t0 = scb.SpiceManager.jd2et(2461809.72995654)

    return scb.EpochArray(np.arange(t0, t0 + 5400.0, 300.0), sys="TDB")


@pytest.fixture
def trajectory(sc, origin, epochs, tmp_path):
    """Propagate the spacecraft and publish an SPK the measurements can query."""
    state_def = (
        scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.array([7000.0, 100.0, -200.0]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.array([0.5, 7.5, 0.3]), km / sec, J2000))
    )
    state = scb.StateArray(epoch=epochs[0], origin=origin, state=state_def)

    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=epochs,
        force_models=scb.ForceModelTranslation(primary_body=sc),
    )
    propagator.propagate(display_progress=False)

    bsp = tmp_path / "custom_measurement.bsp"
    # Loaded into the shared kernel pool; not unloaded here, since clearing the
    # pool would also drop the session metakernel every later test relies on.
    scb.Trajectory(state_array=propagator.propagated_state_array).write_to_spk(str(bsp))

    return state_def, state


@pytest.fixture
def state_with_bias(sc, gs, origin, epochs):
    """State definition carrying an estimated station bias owned by the station."""
    scb.CustomParameter("station_bias", units=km, default=0.0)

    state_def = (
        scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.array([7000.0, 100.0, -200.0]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.array([0.5, 7.5, 0.3]), km / sec, J2000))
        .custom_param("station_bias", gs, scb.ArrayWFrame(0.25, km, None))
    )

    return state_def, scb.StateArray(epoch=epochs[0], origin=origin, state=state_def)


# endregion Fixtures #
# --------------------#


# ----------------------------#
# region    Construction     #
# ----------------------------#
class TestConstructionErrors:
    def test_signature_rejects_unknown_argument(self, gs):
        """Callables may only declare names present in the evaluation context."""
        with pytest.raises(scb.CustomCallable.SignatureError, match="altitude"):
            scb.CustomMeasurement("bad", gs, lambda altitude: 0.0, sigma=SIGMA)

    def test_observable_shape_is_validated(self, gs, sc, trajectory, epochs):
        """A wrong-sized observable is reported instead of silently accepted."""
        state_def, state = trajectory
        model = scb.CustomMeasurement(
            "bad", gs, lambda position: np.zeros(2), sigma=SIGMA, state_definition=state_def
        )
        model.update_reference_state(state)

        with pytest.raises(scb.CustomMeasurement.ShapeError):
            model.computed_measurements(target=sc, epoch_array=epochs, frame=J2000)

    def test_missing_units_is_reported(self, gs):
        """Construction without units or sigma cannot label the measurement."""
        with pytest.raises(scb.CustomMeasurement.MissingSigmaError):
            scb.CustomMeasurement("no units", gs, plain_range)

    def test_missing_units_is_the_base_missing_sigma_error(self, gs):
        """The error is the base class's, so one except clause covers every model."""
        assert scb.CustomMeasurement.MissingSigmaError is scb.Measurement.MissingSigmaError
        with pytest.raises(scb.Measurement.MissingSigmaError):
            scb.CustomMeasurement("no units", gs, plain_range)


# endregion Construction #
# ----------------------------#


# --------------------------------#
# region    Built-in agreement    #
# --------------------------------#
class TestBuiltinAgreement:
    def test_range_values(self, gs, sc, trajectory, epochs):
        """A custom geometric range reproduces RangeIdeal's computed measurements."""
        state_def, state = trajectory

        builtin = scb.RangeIdeal("builtin", gs, sigma=SIGMA, state_definition=state_def)
        builtin.update_reference_state(state)
        custom = scb.CustomMeasurement(
            "custom", gs, plain_range, sigma=SIGMA, state_definition=state_def
        )
        custom.update_reference_state(state)

        _, _, _, expected = builtin.computed_measurements(
            target=sc, epoch_array=epochs, frame=J2000
        )
        _, _, _, actual = custom.computed_measurements(
            target=sc, epoch_array=epochs, frame=J2000
        )

        assert np.allclose(actual.quantity.values, expected.quantity.values, rtol=0, atol=0)

    def test_range_partials(self, gs, sc, trajectory, epochs):
        """A custom geometric range reproduces RangeIdeal's H-tilde."""
        state_def, state = trajectory

        builtin = scb.RangeIdeal("builtin", gs, sigma=SIGMA, state_definition=state_def)
        builtin.update_reference_state(state)
        custom = scb.CustomMeasurement(
            "custom", gs, plain_range, sigma=SIGMA, state_definition=state_def
        )
        custom.update_reference_state(state)

        expected = np.asarray(builtin.compute_partials(target=sc, epoch_array=epochs))
        actual = np.asarray(custom.compute_partials(target=sc, epoch_array=epochs))

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected, rtol=1e-6, atol=1e-8)


# endregion Built-in agreement #
# --------------------------------#


# ----------------------------#
# region    Partials         #
# ----------------------------#
class TestPartials:
    def test_position_matches_analytic(self, gs, sc, trajectory, epochs):
        """The finite-difference position Jacobian matches the analytic one."""
        state_def, state = trajectory

        numerical = scb.CustomMeasurement(
            "fd", gs, plain_range, sigma=SIGMA, state_definition=state_def
        )
        analytic = scb.CustomMeasurement(
            "analytic",
            gs,
            plain_range,
            sigma=SIGMA,
            state_definition=state_def,
            partial_by_position=range_partial_by_position,
        )
        for model in (numerical, analytic):
            model.update_reference_state(state)

        numeric_h = np.asarray(numerical._partials(sc, epochs[0], J2000))
        analytic_h = np.asarray(analytic._partials(sc, epochs[0], J2000))

        assert np.allclose(numeric_h, analytic_h, rtol=1e-6, atol=1e-9)

    def test_velocity_absent_when_independent(self, gs):
        """An observable that ignores velocity reports no velocity dependence."""
        model = scb.CustomMeasurement("range", gs, plain_range, sigma=SIGMA)

        assert model.depends_on_position is True
        assert model.depends_on_velocity is False


# endregion Partials #
# ----------------------------#


# ----------------------------#
# region    Parameters       #
# ----------------------------#
class TestParameters:
    def test_shifts_the_measurement(self, gs, sc, trajectory, state_with_bias, epochs):
        """A declared parameter read from the state changes the observable."""
        plain_def, plain_state = trajectory
        bias_def, bias_state = state_with_bias

        unbiased = scb.CustomMeasurement(
            "unbiased", gs, plain_range, sigma=SIGMA, state_definition=plain_def
        )
        unbiased.update_reference_state(plain_state)

        biased = scb.CustomMeasurement(
            "biased",
            gs,
            biased_range,
            sigma=SIGMA,
            state_definition=bias_def,
            parameters=[scb.CustomParameter("station_bias", units=km, register=False)],
        )
        biased.update_reference_state(bias_state)

        _, _, _, plain = unbiased.computed_measurements(
            target=sc, epoch_array=epochs, frame=J2000
        )
        _, _, _, shifted = biased.computed_measurements(
            target=sc, epoch_array=epochs, frame=J2000
        )

        assert np.allclose(shifted.quantity.values - plain.quantity.values, 0.25)

    def test_partial_matches_analytic(self, gs, sc, trajectory, state_with_bias, epochs):
        """The finite-difference parameter Jacobian matches the analytic one."""
        bias_def, bias_state = state_with_bias

        numerical = scb.CustomMeasurement(
            "fd",
            gs,
            biased_range,
            sigma=SIGMA,
            state_definition=bias_def,
            parameters=[scb.CustomParameter("station_bias", units=km, register=False)],
        )
        analytic = scb.CustomMeasurement(
            "analytic",
            gs,
            biased_range,
            sigma=SIGMA,
            state_definition=bias_def,
            parameters=[
                scb.CustomParameter(
                    "station_bias", units=km, partial=bias_partial, register=False
                )
            ],
        )
        for model in (numerical, analytic):
            model.update_reference_state(bias_state)

        numeric_h = np.asarray(numerical._partials(sc, epochs[0], J2000))
        analytic_h = np.asarray(analytic._partials(sc, epochs[0], J2000))

        # d(range)/d(bias) is exactly 1 and sits in the last column
        assert np.isclose(analytic_h[-1], 1.0)
        assert np.allclose(numeric_h, analytic_h, rtol=1e-6, atol=1e-9)

    def test_one_column_per_state_element(self, gs, sc, trajectory, state_with_bias, epochs):
        """The Jacobian row is laid out to match the state definition."""
        bias_def, bias_state = state_with_bias

        model = scb.CustomMeasurement(
            "biased",
            gs,
            biased_range,
            sigma=SIGMA,
            state_definition=bias_def,
            parameters=[scb.CustomParameter("station_bias", units=km, register=False)],
        )
        model.update_reference_state(bias_state)

        row = model._partials(sc, epochs[0], J2000)
        expected = sum(component[1] for component in bias_state.state)

        assert len(row) == expected == 7


# endregion Parameters #
# ----------------------------#


# ----------------------------#
# region    Layout           #
# ----------------------------#
class TestLayout:
    def test_observer_position_column_is_negated(self, gs, sc, origin, trajectory, epochs):
        """An instrument-owned position column carries the opposite sign."""
        _, _ = trajectory

        state_def = (
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 100.0, -200.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.5, 7.5, 0.3]), km / sec, J2000))
            .position(gs, scb.ArrayWFrame(np.zeros(3), km, J2000))
            .velocity(gs, scb.ArrayWFrame(np.zeros(3), km / sec, J2000))
        )
        state = scb.StateArray(epoch=epochs[0], origin=origin, state=state_def)

        model = scb.CustomMeasurement(
            "custom", gs, plain_range, sigma=SIGMA, state_definition=state_def
        )
        model.update_reference_state(state)

        row = np.asarray(model._partials(sc, epochs[0], J2000))

        assert np.allclose(row[6:9], -row[0:3])

    def test_multicomponent_observable_returns_one_row_each(self, gs, sc, trajectory, epochs):
        """A vector observable produces one Jacobian row per component."""
        state_def, state = trajectory

        def range_and_squared(position):
            radius = np.linalg.norm(position)
            return np.array([radius, radius**2])

        model = scb.CustomMeasurement(
            "two component",
            gs,
            range_and_squared,
            sigma=SIGMA,
            dimension=2,
            state_definition=state_def,
        )
        model.update_reference_state(state)

        rows = model._partials(sc, epochs[0], J2000)

        assert len(rows) == 2
        assert all(len(row) == 6 for row in rows)
        # d(r^2)/dr = 2 r d(r)/dr, so the second row is the first scaled by 2r
        radius = np.linalg.norm(model._relative_state(sc, epochs[0], J2000)[0:3])
        assert np.allclose(np.asarray(rows[1])[:3], 2 * radius * np.asarray(rows[0])[:3],
                           rtol=1e-5)


# endregion Layout #
# ----------------------------#
