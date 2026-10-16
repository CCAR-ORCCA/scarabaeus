# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ========================================#
#  Unit Tests for the CustomFilter class  #
# ========================================#
import os

import src.scarabaeus as scb
import numpy as np
import pytest

kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])
J2000 = scb.Frame("J2000")

POS = np.array([-1.1123095885148e08, 8.9094345479316e07, 3.8656500948069e07])
VEL = np.array([-20.6936999825159, -16.7800270812616, -6.6437327193572])
DELTA_POS = np.array([1.0, -0.5, 0.75])
DELTA_VEL = np.array([1e-3, -5e-4, 7.5e-4])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """Truth trajectory, observations and a perturbed reference to estimate from."""
    directory = tmp_path_factory.mktemp("custom_filter")

    origin = scb.CelestialBody.from_constants("SUN")
    station = scb.GroundStation("DSS-14")
    third_bodies = scb.CelestialBody.from_constants(["MERCURY", "VENUS", "EARTH"])

    t0 = scb.SpiceManager.jd2et(2461809.72995654 + 1 / 3)
    epochs = scb.EpochArray(np.arange(t0, t0 + 86400.0, 1800.0), sys="TDB")

    mass = scb.ArrayWUnits(2000.0, kg)
    area = scb.ArrayWUnits(1e-06, km**2)
    reflectivity = scb.ArrayWUnits(1.5, None)

    def build(name, spice_id, position, velocity):
        spacecraft = scb.Spacecraft(name, spice_id, mass, area, reflectivity)
        definition = (
            scb.StateDefinition()
            .position(spacecraft, scb.ArrayWFrame(position, km, J2000))
            .velocity(spacecraft, scb.ArrayWFrame(velocity, km / sec, J2000))
        )
        state = scb.StateArray(epoch=epochs[0], origin=origin, state=definition)

        return spacecraft, definition, state

    def force_model(spacecraft):
        return scb.ForceModelTranslation(
            primary_body=spacecraft, third_bodies=third_bodies
        )

    # Truth trajectory, published so the measurement models can query it.
    truth_sc, _, truth_state = build("cf_truth", -1000, POS, VEL)
    truth = scb.Propagator(
        primary_body=truth_sc,
        state_vector=truth_state,
        tspan=epochs,
        force_models=force_model(truth_sc),
    )
    truth.propagate(display_progress=False)
    scb.Trajectory(state_array=truth.propagated_state_array).write_to_spk(
        str(directory / "cf_truth.bsp")
    )

    # Perturbed reference the filters start from.
    reference_sc, reference_def, reference_state = build(
        "cf_reference", -1011, POS + DELTA_POS, VEL + DELTA_VEL
    )

    range_sigma = scb.ArrayWUnits(1e-3, km)
    rate_sigma = scb.ArrayWUnits(1e-5, km / sec)
    range_model = scb.RangeIdeal(
        "R", station, sigma=range_sigma, state_definition=reference_def
    )
    rate_model = scb.RangeRateIdeal(
        "RR", station, sigma=rate_sigma, state_definition=reference_def
    )

    observed = {}
    for model, label, units in (
        (range_model, "cf_range", km),
        (rate_model, "cf_rate", km / sec),
    ):
        model.write_observed_measurements(
            target=truth_sc, epoch_array=epochs, noisy=False, file_name=label
        )
        observed[label] = model.observed_measurements(
            file_name=f"data/measurements/radiometric/{label}.json",
            meas_name="meas_ideal",
            units=units,
        )

    measurements = scb.MeasurementSpec.from_dict(
        [
            {
                "model": range_model,
                "observed_meas": observed["cf_range"],
                "dataset_name": "Range",
            },
            {
                "model": rate_model,
                "observed_meas": observed["cf_rate"],
                "dataset_name": "Range Rate",
            },
        ]
    )
    covariance = scb.CovarianceMatrix(
        [scb.ArrayWUnits(3.0, km)] * 3 + [scb.ArrayWUnits(3e-3, km / sec)] * 3,
        epochs[1],
        from_list=True,
    )

    yield {
        "directory": directory,
        "epochs": epochs,
        "spacecraft": reference_sc,
        "state": reference_state,
        "force_model": force_model,
        "measurements": measurements,
        "covariance": covariance,
    }

    for label in ("cf_range", "cf_rate"):
        path = f"data/measurements/radiometric/{label}.json"
        if os.path.isfile(path):
            os.remove(path)


@pytest.fixture
def make_filter(scenario):
    """Build a CustomFilter (or any FilterOD) over the shared scenario."""
    counter = {"index": 0}

    def factory(estimator=None, cls=scb.CustomFilter, **kwargs):
        counter["index"] += 1
        tag = f"cf_case{counter['index']}"
        propagator = scb.Propagator(
            primary_body=scenario["spacecraft"],
            state_vector=scenario["state"],
            tspan=scenario["epochs"],
            force_models=scenario["force_model"](scenario["spacecraft"]),
        )
        settings = scb.FilterSettings(
            initial_covariance=scenario["covariance"], output=scb.OutputSettings()
        )
        if estimator is not None:
            kwargs["estimator"] = estimator

        return cls(
            propagator=propagator,
            measurements=scenario["measurements"],
            settings=settings,
            traj_name=f"{tag}.bsp",
            traj_dir=str(scenario["directory"]),
            **kwargs,
        )

    return factory


def sequential(filter):
    """Textbook sequential Kalman recursion, written against the toolkit."""
    covariance = filter.initial_covariance_matrix
    deviation = filter.initial_deviation
    deviations, covariances = [], []

    previous = None
    for epoch in filter.timeline:
        if previous is not None:
            transition = filter.stm_between(previous, epoch)
            deviation = transition @ deviation
            covariance = transition @ covariance @ transition.T
        previous = epoch

        block = filter.measurement_block(epoch)
        if block is not None:
            partials = block["partials"][:, : filter.n_states]
            noise = filter.measurement_covariance(block["sigma"], epoch)
            innovation_covariance = partials @ covariance @ partials.T + noise
            gain = covariance @ partials.T @ filter.safe_inv(innovation_covariance)
            deviation = deviation + gain @ (
                block["residuals"] - partials @ deviation
            )
            residual_map = np.eye(filter.n_states) - gain @ partials
            covariance = (
                residual_map @ covariance @ residual_map.T + gain @ noise @ gain.T
            )
            # Symmetrized, as in the LKF
            covariance = 0.5 * (covariance + covariance.T)

        deviations.append(deviation)
        covariances.append(covariance)

    filter.record(deviation=deviations, covariance=covariances)


def batch(filter):
    """Batch least squares accumulated over every measurement epoch."""
    size = filter.n_states
    information = filter.safe_inv(filter.initial_covariance_matrix)
    normal = np.zeros((size, 1))

    for epoch in filter.measurement_epochs:
        block = filter.measurement_block(epoch)
        transition = filter.stm(epoch)
        partials = block["partials"][:, :size] @ transition
        weight = filter.safe_inv(
            filter.measurement_covariance(block["sigma"], epoch)
        )
        information = information + partials.T @ weight @ partials
        normal = normal + partials.T @ weight @ block["residuals"]

    covariance0 = filter.safe_inv(information)
    deviation0 = covariance0 @ normal

    deviations, covariances = [], []
    for epoch in filter.timeline:
        transition = filter.stm(epoch)
        deviations.append(transition @ deviation0)
        covariances.append(transition @ covariance0 @ transition.T)

    filter.record(deviation=deviations, covariance=covariances)


def initial_state(od):
    """Converged initial state held by a filter's reference propagator."""
    return np.asarray(od.propagator.full_state_vector.values0.values).ravel()


# endregion Fixtures #
# --------------------#


# ----------------------------#
# region    Construction     #
# ----------------------------#
class TestEstimatorValidation:
    """An estimator callable is required and checked before anything runs."""

    def test_required(self, make_filter):
        """A CustomFilter without an algorithm has nothing to run."""
        with pytest.raises(scb.CustomFilter.ResultError, match="estimator"):
            make_filter()

    def test_signature_validated(self, make_filter):
        """An estimator declaring an unknown argument is refused up front."""
        with pytest.raises(scb.CustomCallable.SignatureError, match="sigma_points"):
            make_filter(lambda sigma_points: None)


class TestSequenceState:
    """A mission sequence must carry the same state on every leg."""

    @pytest.fixture
    def legs(self):
        """A coast leg, a burn leg adding a thrust scale, and another coast."""
        origin = scb.CelestialBody.from_constants("SUN")
        epoch = scb.EpochArray(np.array([8.0e8]), sys="TDB")[0]
        spacecraft = scb.Spacecraft(
            "cf_sequence",
            -1012,
            scb.ArrayWUnits(2000.0, kg),
            scb.ArrayWUnits(1e-06, km**2),
            scb.ArrayWUnits(1.5, None),
        )

        def leg(with_burn):
            definition = (
                scb.StateDefinition()
                .position(spacecraft, scb.ArrayWFrame(POS, km, J2000))
                .velocity(spacecraft, scb.ArrayWFrame(VEL, km / sec, J2000))
            )
            if with_burn:
                definition = definition.param(
                    "k_thrust1",
                    spacecraft,
                    scb.ArrayWFrame(scb.ArrayWUnits(1.0, None), None),
                    estimation="estimated",
                    dynamics="static",
                )
            return scb.StateArray(epoch=epoch, origin=origin, state=definition)

        return leg(False), leg(True), leg(False)

    @staticmethod
    def _sequence_filter(legs_model):
        # only the attributes the check reads: building a sequence filter
        # would propagate every leg without exercising anything more of it
        od = object.__new__(scb.CustomFilter)
        od.flag_sequence = True
        od.legs_model = list(legs_model)
        return od

    def test_changing_state_refused(self, legs):
        """A finite burn adding its parameters at a node is refused up front."""
        od = self._sequence_filter(legs)

        with pytest.raises(scb.CustomFilter.SequenceError, match="leg 1"):
            od._check_sequence_state()

    def test_same_state_accepted(self, legs):
        """Legs carrying the same state, as with impulsive burns, pass."""
        coast, _, other_coast = legs

        self._sequence_filter([coast, other_coast])._check_sequence_state()


# endregion Construction #
# ----------------------------#


# ----------------------------#
# region    Toolkit          #
# ----------------------------#
class TestToolkit:
    """The helpers an estimator callable uses to build its own recursion."""

    def test_exposes_the_problem(self, make_filter, scenario):
        """timeline, state size and a priori covariance describe the problem."""
        od = make_filter(sequential)

        assert od.n_states == 6
        assert len(od.timeline) >= len(scenario["epochs"])
        assert od.initial_covariance_matrix.shape == (6, 6)
        assert od.initial_deviation.shape == (6, 1)

    def test_measurement_block_has_residuals_and_partials(self, make_filter):
        """Every measurement epoch carries the arrays an estimator needs."""
        od = make_filter(sequential)
        epoch = od.measurement_epochs[0]

        block = od.measurement_block(epoch)

        assert block is not None
        assert block["partials"].shape[1] >= od.n_states
        assert block["residuals"].shape[0] == block["partials"].shape[0]
        assert len(block["sigma"]) == block["partials"].shape[0]

    def test_measurement_covariance_is_square(self, make_filter):
        """R is built from the block's sigmas, one row per measurement."""
        od = make_filter(sequential)
        epoch = od.measurement_epochs[0]
        block = od.measurement_block(epoch)

        noise = od.measurement_covariance(block["sigma"], epoch)
        rows = block["partials"].shape[0]

        assert noise.shape == (rows, rows)
        assert np.allclose(noise, noise.T)

    def test_stm_between_composes(self, make_filter):
        """stm_between(a, b) is Phi(b) Phi(a)^-1."""
        od = make_filter(sequential)
        first, second = od.timeline[0], od.timeline[3]

        expected = od.stm(second) @ od.safe_inv(od.stm(first))

        assert np.allclose(od.stm_between(first, second), expected)


# endregion Toolkit #
# ----------------------------#


# ----------------------------#
# region    Estimators       #
# ----------------------------#
class TestEstimators:
    """End-to-end fits using estimator callables of different shapes."""

    def test_sequential_matches_lkf(self, make_filter):
        """A user-written Kalman recursion reproduces the built-in LKF."""
        reference = make_filter(cls=scb.LKF)
        reference.fit(max_iterations=2, verbose=False, traj_name="cf_lkf.bsp")

        custom = make_filter(sequential)
        custom.fit(max_iterations=2, verbose=False, traj_name="cf_seq.bsp")

        assert np.allclose(
            initial_state(custom), initial_state(reference), rtol=0, atol=1e-9
        )

    def test_batch_runs(self, make_filter):
        """A batch normal-equations estimator solves the same problem."""
        od = make_filter(batch)
        solution, _, _ = od.fit(max_iterations=2, verbose=False, traj_name="cf_bat.bsp")

        truth = np.concatenate([POS, VEL])
        error = initial_state(od) - truth

        assert solution is not None
        # The a priori is off by |DELTA_POS| ~ 1.35 km; the fit must do far better.
        assert np.linalg.norm(error[:3]) < 0.1 * np.linalg.norm(DELTA_POS)

    def test_result_may_be_returned_instead_of_recorded(self, make_filter):
        """Returning a dict is equivalent to calling record()."""
        def returning(filter):
            size, span = filter.n_states, len(filter.timeline)
            return {
                "deviation": [np.zeros((size, 1))] * span,
                "covariance": [filter.initial_covariance_matrix] * span,
            }

        od = make_filter(returning)
        solution, _, _ = od.fit(max_iterations=1, verbose=False, traj_name="cf_ret.bsp")

        assert solution is not None
        assert len(od.covariance_history) == len(od.timeline)


# endregion Estimators #
# ----------------------------#


# ----------------------------#
# region    Result contract  #
# ----------------------------#
class TestResultContract:
    """record()'s output is checked for completeness before a fit can finish."""

    def test_missing_covariance(self, make_filter):
        """An estimator that reports nothing is refused, with a pointer to record()."""
        od = make_filter(lambda filter: None)

        with pytest.raises(scb.CustomFilter.ResultError, match="covariance"):
            od.fit(max_iterations=1, verbose=False, traj_name="cf_none.bsp")

    def test_wrong_history_length(self, make_filter):
        """A history that does not span the timeline is refused."""
        def truncated(filter):
            filter.record(
                deviation=[np.zeros((filter.n_states, 1))],
                covariance=[filter.initial_covariance_matrix],
            )

        od = make_filter(truncated)

        with pytest.raises(scb.CustomFilter.ShapeError, match="timeline epochs"):
            od.fit(max_iterations=1, verbose=False, traj_name="cf_short.bsp")


# endregion Result contract #
# ----------------------------#


# --------------------------------#
# region Nonlinear evaluation     #
# --------------------------------#
def test_predict_propagates_an_arbitrary_state(make_filter):
    """The nonlinear hook responds to a shifted initial state."""
    captured = {}

    def probe(filter):
        base = np.concatenate([POS + DELTA_POS, VEL + DELTA_VEL])
        shifted = base.copy()
        shifted[0] += 10.0

        captured["base"] = filter.predict(base)
        captured["shifted"] = filter.predict(shifted)

        size, span = filter.n_states, len(filter.timeline)
        filter.record(
            deviation=[np.zeros((size, 1))] * span,
            covariance=[filter.initial_covariance_matrix] * span,
        )

    od = make_filter(probe)
    od.fit(max_iterations=1, verbose=False, traj_name="cf_probe.bsp")

    assert captured["base"].shape[1] == 6
    assert np.allclose(captured["base"][0][:3], POS + DELTA_POS, atol=1e-6)
    # A 10 km shift must still be visible at the end of the arc.
    separation = np.linalg.norm(captured["shifted"][-1][:3] - captured["base"][-1][:3])
    assert separation > 1.0


# endregion Nonlinear evaluation #
# --------------------------------#


# ----------------------------#
# region    Smoothing        #
# ----------------------------#
def test_smoother_absent_is_reported(make_filter):
    """Requesting smoothing without a smoother callable is refused."""
    od = make_filter(sequential)

    with pytest.raises(NotImplementedError, match="smoother"):
        od.fit(
            max_iterations=1,
            verbose=False,
            if_sequential_smooth=True,
            traj_name="cf_smooth.bsp",
        )


# endregion Smoothing #
# ----------------------------#
