# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =====================================#
#  Unit Tests for the FilterOD class  #
# =====================================#
import src.scarabaeus as scb
import numpy as np
import pytest
import warnings

m, km, kg, sec, hr, rad, deg = scb.Units.get_units(["m", "km", "kg", "sec", "hr", "rad", "deg"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture(autouse=True)
def scenario_dir(tmp_path, monkeypatch):
    """Filters built without a traj_dir write under the cwd: keep them out of the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def epochs():
    # meas_data_file (conftest) uses doy=39 (2028 Feb 8) at ~05:31 TDB;
    # start at midnight with 1hr steps so the BSP covers the measurement epoch
    return scb.EpochArray.interval(
        "2028 Feb 8 00:00:00", "2028 Feb 9 00:00:00", scb.ArrayWUnits(1, hr), sys="TDB")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc_area():
    return scb.ArrayWUnits(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return scb.ArrayWUnits(1.5, None)


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft"""
    return scb.Spacecraft(
        "orbiter_ref_it1",
        -1001,
        scb.ArrayWUnits(2000.0, kg),
        sc_area,
        sc_ref_coeff,
    )


@pytest.fixture
def state(epochs, origin, sc):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(primary_body=sc,)


@pytest.fixture
def propagator(sc, state, epochs, force_model):

    prop = scb.Propagator(
        primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model)
    return prop


@pytest.fixture
def covar_mat(epochs):
    # Set up initial covariance matrix P (valid also for it-2, and it-3)
    pos_sigma = scb.ArrayWUnits(5, km)
    vel_sigma = scb.ArrayWUnits(5, km)
    state_sigma_list = [
        pos_sigma,
        pos_sigma,
        pos_sigma,
        vel_sigma,
        vel_sigma,
        vel_sigma,]
    return scb.CovarianceMatrix(state_sigma_list, epochs[1], from_list=True)


@pytest.fixture
def measurements(meas_data_file):
    GS1 = scb.GroundStation("DSS-14")

    # (2) Initialize Measurement Models
    Range_GS1 = scb.RangeIdeal(
        "GS1 Ideal Range Model", GS1, sigma=scb.ArrayWUnits(1e-3, km))
    RangeRate_GS1 = scb.RangeRateIdeal(
        "GS1 Ideal Range Rate Model", GS1, sigma=scb.ArrayWUnits(1e-5, km / sec))

    obs_quantities_range = Range_GS1.observed_measurements(
                            file_name=str(meas_data_file),
                            meas_name="meas_ideal",
                            units=km,)

    measurements_list = scb.MeasurementSpec.many(
        scb.MeasurementSpec(
            model=Range_GS1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",),
        scb.MeasurementSpec(
            model=Range_GS1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",))
    return measurements_list


@pytest.fixture
def filter_od(propagator, covar_mat, measurements):
    """Bare FilterOD over the shared scenario."""
    settings = scb.FilterSettings(initial_covariance=covar_mat)
    return scb.FilterOD(propagator, measurements, settings)


class ScriptedFilter(scb.FilterOD):
    """
    FilterOD whose passes are scripted, so that the convergence loop of
    :meth:`~scarabaeus.FilterOD.fit` can be exercised without propagating.

    Each entry of ``script`` is a ``(deviation_rms, residual)`` pair: the pass
    reports a deviation of that RMS and a single unit-sigma residual, so the
    cost of the pass is ``residual ** 2``.
    """

    def script(self, passes: list) -> None:
        self._script = list(passes)
        self._pass = 0

    def _compute(self, printOutput: bool = True):
        deviation_rms, residual = self._script[min(self._pass, len(self._script) - 1)]
        self._pass += 1
        self.postfit_residuals = {"scripted": [[residual, 1.0]]}

        deviation = np.full((self.n, 1), deviation_rms)
        return type(
            "ScriptedSolution",
            (),
            {
                "map_state_deviation_to_epoch": lambda _self: deviation,
                "deviation_est": None,
            },
        )()

    def _reinitialize(self, *args, **kwargs):
        """No trajectory to rebuild: the next pass is scripted."""
        return None


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(propagator, covar_mat, measurements):
    """
    Verifies that object is constructed correctly.
    """
    settings = scb.FilterSettings(initial_covariance=covar_mat)
    test_filter_od = scb.FilterOD(propagator, measurements, settings)

    assert test_filter_od.propagator == propagator
    assert test_filter_od.measurements == measurements
    assert test_filter_od.settings == settings
    assert test_filter_od.initial_covariance == settings.initial_covariance


@pytest.mark.parametrize("n_sigmas", [5, 7])
def test_initial_covariance_size_reported(propagator, measurements, epochs, n_sigmas):
    """A P0 that does not match the initial state is reported at construction."""
    covariance = scb.CovarianceMatrix(
        [scb.ArrayWUnits(5, km)] * n_sigmas, epochs[1], from_list=True)
    settings = scb.FilterSettings(initial_covariance=covariance)

    with pytest.raises(scb.FilterOD.InitialCovarianceSizeError,
                       match=f"{n_sigmas}x{n_sigmas}.* 6 elements"):
        scb.FilterOD(propagator, measurements, settings)


class TestCostFunction:
    """_cost_function's weighting and its handling of unusable residuals."""

    def test_weights_residuals(self, filter_od):
        """The cost is the weighted sum of squares of the post-fit residuals."""
        filter_od.postfit_residuals = {
            "Range": [[2.0, 1.0], [3.0, 3.0]],
            "Range Rate": [[1.0, 2.0]],
        }

        cost, n_residuals = filter_od._cost_function()

        assert cost == pytest.approx(4.0 + 1.0 + 0.25)
        assert n_residuals == 3

    def test_skips_unusable_residuals(self, filter_od):
        """Residuals whose sigma is zero, negative or NaN do not enter the cost."""
        filter_od.postfit_residuals = {
            "Range": [[1.0, 1.0], [5.0, 0.0], [5.0, -1.0], [5.0, np.nan]],
            "Range Rate": [],
        }

        cost, n_residuals = filter_od._cost_function()

        assert cost == pytest.approx(1.0)
        assert n_residuals == 1

    def test_without_residuals(self, filter_od):
        """No usable residual reports NaN, so the cost test stays inactive."""
        cost, n_residuals = filter_od._cost_function({})

        assert np.isnan(cost)
        assert n_residuals == 0


class TestFitConvergence:
    """fit()'s convergence loop, driven by a ScriptedFilter's fake passes."""

    def test_cost_history_records_every_iteration(self, propagator, covar_mat, measurements):
        """fit() stores the cost of each pass, alongside its residual count."""
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 4.0), (1.0, 3.0), (1.0, 2.0)])

        od.fit(max_iterations=3, convergence_criterion="deviation", verbose=False)

        assert od.cost_history == [(16.0, 1), (9.0, 1), (4.0, 1)]

    def test_cost_criterion_converges_when_deviation_does_not(
        self, propagator, covar_mat, measurements
    ):
        """
        Weakly observable case: the deviation keeps moving while the fit no longer
        improves, so only the cost test can declare convergence.
        """
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)])

        _, iterations, converged = od.fit(
            max_iterations=5,
            convergence_threshold=1e-6,
            convergence_threshold_cost=1e-3,
            verbose=False,
        )

        assert converged
        assert iterations == 2  # first pass has no previous cost to compare against
        assert od.convergence_info["converged_cost"] is True
        assert od.convergence_info["converged_deviation"] is False

    def test_deviation_criterion_ignores_a_stalled_cost(
        self, propagator, covar_mat, measurements
    ):
        """The 'deviation' criterion keeps iterating on a stalled cost."""
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 2.0)])

        _, iterations, converged = od.fit(
            max_iterations=3, convergence_criterion="deviation", verbose=False
        )

        assert not converged
        assert iterations == 3

    def test_both_criterion_requires_the_two_tests(self, propagator, covar_mat, measurements):
        """The 'both' criterion is not satisfied by a stalled cost alone."""
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 2.0)])

        _, _, converged = od.fit(
            max_iterations=3, convergence_criterion="both", verbose=False
        )

        assert not converged

        # Same stalled cost, but now the deviation is below threshold as well
        od_small = ScriptedFilter(propagator, measurements, settings)
        od_small.script([(1e-9, 2.0)])

        _, iterations, converged = od_small.fit(
            max_iterations=3, convergence_criterion="both", verbose=False
        )

        assert converged
        assert iterations == 2

    def test_only_applied_corrections_accumulate(self, propagator, covar_mat, measurements):
        """
        Each pass's total is the corrections applied so far plus its own, and
        the last pass's correction, which no further pass applies to the
        reference, is left out of the running total.
        """
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 4.0), (2.0, 3.0), (4.0, 2.0)])

        od.fit(max_iterations=3, convergence_criterion="deviation", verbose=False)

        assert [float(np.unique(total)[0]) for total in od._delta0_hist] == [1.0, 3.0, 7.0]
        assert od._delta0_total.shape == (od.n, 1)
        assert np.all(od._delta0_total == 3.0)

    def test_unknown_criterion_reported(self, filter_od):
        """An unsupported criterion is refused before any pass is run."""
        with pytest.raises(
            scb.FilterOD.ConvergenceCriterionError, match="convergence_criterion"
        ):
            filter_od.fit(max_iterations=1, convergence_criterion="rms", verbose=False)

    def test_quiet_fit_repropagates_quietly(self, propagator, covar_mat, measurements):
        """fit(verbose=False) hands its verbosity to every re-propagation."""
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 4.0), (1.0, 3.0), (1.0, 2.0)])
        seen = []
        od._reinitialize = lambda *args, **kwargs: seen.append(kwargs.get("verbose"))

        od.fit(max_iterations=3, convergence_criterion="deviation", verbose=False)

        assert seen == [False, False]


class _PropagationReached(Exception):
    """Raised by the recording propagators below once propagate() is called."""


class _RecordingScenario(scb.ScenarioSetup):
    def propagate(self, verbose=True):
        self.verbose_seen = verbose
        raise _PropagationReached


class _RecordingSequence(scb.MissionSequence):
    def propagate(self, map_STM_to_t0=False, verbose=True):
        self.verbose_seen = verbose
        raise _PropagationReached


class TestInitializeVerbosity:
    """_initialize forwards its verbosity to a sequence's or scenario's own
    propagation, whose segment-by-segment printout it controls."""

    @pytest.mark.parametrize("cls", [_RecordingScenario, _RecordingSequence])
    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbosity_reaches_propagate(self, filter_od, tmp_path, cls, verbose):
        recorder = cls.__new__(cls)  # only propagate() is reached

        with pytest.raises(_PropagationReached):
            filter_od._initialize(
                recorder, filter_od.measurements, traj_dir=str(tmp_path), verbose=verbose
            )

        assert recorder.verbose_seen is verbose

    @pytest.mark.parametrize(
        "filter_cls", [scb.FilterOD, scb.LKF, scb.SRIF, scb.SRIFB, scb.LSB, scb.CustomFilter]
    )
    @pytest.mark.parametrize("verbose", [True, False])
    def test_constructor_verbosity_reaches_propagate(
        self, covar_mat, measurements, tmp_path, filter_cls, verbose
    ):
        recorder = _RecordingSequence.__new__(_RecordingSequence)
        settings = scb.FilterSettings(initial_covariance=covar_mat)

        with pytest.raises(_PropagationReached):
            filter_cls(
                recorder, measurements, settings, traj_dir=str(tmp_path), verbose=verbose
            )

        assert recorder.verbose_seen is verbose


class TestTrajectoryFiles:
    """Re-linearised trajectories go where, and under the name, the filter was built with."""

    def test_iterations_follow_constructor_name_and_dir(
        self, propagator, covar_mat, measurements, tmp_path
    ):
        """fit() without names writes <name>_it<k>.bsp beside the first SPK, not timestamped files elsewhere."""

        class RelinearisingFilter(ScriptedFilter):
            def _reinitialize(self, *args, **kwargs):
                return scb.FilterOD._reinitialize(self, *args, **kwargs)

        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = RelinearisingFilter(
            propagator, measurements, settings, traj_name="ref.bsp", traj_dir=str(tmp_path)
        )
        od.script([(1e-9, 1.0)] * 3)

        od.fit(
            max_iterations=3,
            convergence_threshold=1e-12,
            convergence_criterion="deviation",
            verbose=False,
        )

        bsps = sorted(p.name for p in tmp_path.glob("*.bsp"))
        assert bsps == ["ref.bsp", "ref_it2.bsp", "ref_it3.bsp"]
        assert od.trajectory.spk_path == str(tmp_path / "ref_it3.bsp")


# ------------------------------------------------------------------#
# region  Sequence node mapping and a-priori sigmas                  #
# ------------------------------------------------------------------#
# A sequence leg may introduce states the previous leg did not carry (finite
# burn solve-fors, for instance). The filter maps the covariance onto the new
# definition and seeds those states from `extra_apriori_sigmas`, which carry
# their own units and need not match the ones the state was declared with.


@pytest.fixture
def burn_leg_model(epochs, origin, sc):
    """Leg state that adds the finite-burn solve-fors, declared in radians."""
    return scb.StateArray(
        epochs[0],
        origin,
        state=(
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
            .param(
                "k_thrust1",
                sc,
                scb.ArrayWFrame(scb.ArrayWUnits(1.0, None), None),
                estimation="estimated",
                dynamics="static",
            )
            .param(
                "dRA1",
                sc,
                scb.ArrayWFrame(0.0, rad, J2000),
                estimation="estimated",
                dynamics="static",
            )
        ),
    )


@pytest.fixture
def sequence_filter(state, burn_leg_model, epochs):
    """
    Filter carrying just the two attributes node mapping reads.

    The mapping methods need only `legs_model` and `initial_covariance`;
    building a full filter would pull in a propagator and measurements without
    exercising anything more of the mapping itself.
    """
    od = object.__new__(scb.FilterOD)
    od.legs_model = [state, burn_leg_model]
    od.initial_covariance = scb.CovarianceMatrix(
        [scb.ArrayWUnits(1.0, km)] * 3 + [scb.ArrayWUnits(1e-3, km / sec)] * 3,
        epochs[0],
        from_list=True,
        extra_apriori_sigmas={
            "k_thrust1": scb.ArrayWUnits(0.05, None),
            "dRA1": scb.ArrayWUnits(1.0, deg),
        },
    )
    return od


@pytest.fixture
def coast_covariance():
    """Covariance in the leg that predates the burn solve-fors."""
    return np.diag([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])


@pytest.fixture
def burn_consider_leg_model(epochs, origin, sc):
    """Burn leg estimating k_thrust1 and considering dRA1."""
    return scb.StateArray(
        epochs[0],
        origin,
        state=(
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
            .param(
                "k_thrust1",
                sc,
                scb.ArrayWFrame(scb.ArrayWUnits(1.0, None), None),
                estimation="estimated",
                dynamics="static",
            )
            .param(
                "dRA1",
                sc,
                scb.ArrayWFrame(0.0, rad, J2000),
                estimation="considered",
                dynamics="static",
            )
        ),
    )


class TestSequenceNodeMapping:
    """Mapping a leg's covariance onto the next leg's (possibly augmented) state."""

    def test_state_lists_orders_old_then_new(self, sequence_filter):
        """Node mapping hands back (old, new) for augmentation and for reduction."""
        coast, burn = sequence_filter.legs_model

        old_list, new_list = sequence_filter._node_state_lists(1, True)
        assert old_list is coast.state_definition
        assert new_list is burn.state_definition

        old_list, new_list = sequence_filter._node_state_lists(1, False)
        assert old_list is burn.state_definition
        assert new_list is coast.state_definition

    def test_apriori_sigma_converted_to_state_units(
        self, sequence_filter, coast_covariance
    ):
        """A sigma in degrees seeds a radian state as radians, not as a bare number."""
        P = sequence_filter._map_covariance_definition_node(1, coast_covariance)

        # dRA1 is the last block; 1 deg must arrive as (pi/180)^2 rad^2
        assert np.diag(P)[-1] == pytest.approx(np.deg2rad(1.0) ** 2)

        # the unitless thrust scale is untouched by the conversion
        assert np.diag(P)[-2] == pytest.approx(0.05**2)

    def test_apriori_sigma_already_in_state_units_unchanged(
        self, sequence_filter, coast_covariance
    ):
        """A sigma that already matches the state's units passes through untouched."""
        sequence_filter.initial_covariance.extra_apriori_sigmas["dRA1"] = scb.ArrayWUnits(
            np.deg2rad(0.5), rad
        )

        P = sequence_filter._map_covariance_definition_node(1, coast_covariance)

        assert np.diag(P)[-1] == pytest.approx(np.deg2rad(0.5) ** 2)

    def test_augmented_states_start_uncorrelated(self, sequence_filter, coast_covariance):
        """New states get a diagonal variance and no cross-covariance with the old ones."""
        P = sequence_filter._map_covariance_definition_node(1, coast_covariance)

        assert P.shape == (8, 8)
        assert np.allclose(P[:6, :6], coast_covariance)
        assert np.allclose(P[:6, 6:], 0.0)
        assert np.allclose(P[6:, :6], 0.0)

    def test_incompatible_apriori_sigma_units_reported(
        self, sequence_filter, coast_covariance
    ):
        """A sigma whose dimensions cannot reach the state's units is refused."""
        sequence_filter.initial_covariance.extra_apriori_sigmas["dRA1"] = scb.ArrayWUnits(
            1.0, km
        )

        with pytest.raises(scb.FilterOD.AprioriSigmaUnitsError, match="dRA1"):
            sequence_filter._map_covariance_definition_node(1, coast_covariance)

    def test_missing_apriori_sigma_reported(self, sequence_filter, coast_covariance):
        """A newly augmented state with no supplied sigma is refused."""
        del sequence_filter.initial_covariance.extra_apriori_sigmas["dRA1"]

        with pytest.raises(scb.FilterOD.MissingAprioriSigmaError, match="dRA1"):
            sequence_filter._map_covariance_definition_node(1, coast_covariance)

    def test_new_sigmas_keep_their_units(self, sequence_filter, state, burn_leg_model, epochs):
        """The cached a-priori sigmas stay unit-carrying, not bare numbers."""
        from types import SimpleNamespace

        sequence_definition = SimpleNamespace(
            epochs_vec=[state.epoch, burn_leg_model.epoch],
            states_n=[state.size, burn_leg_model.size],
            models=[state, burn_leg_model],
        )
        sequence_filter.trajectory = SimpleNamespace(
            sequence_definition=sequence_definition,
            _sequence_definition=sequence_definition,
        )
        sequence_filter.propagator = SimpleNamespace(full_state_vector=burn_leg_model)

        sequence_filter._initialize_sequence_attributes()

        assert sequence_filter.new_sigmas
        for sigma in sequence_filter.new_sigmas:
            # a raw float here is the units trap that hit the covariance seeding
            assert isinstance(sigma, scb.ArrayWUnits)

        by_units = {str(s.units) for s in sequence_filter.new_sigmas}
        assert "deg" in by_units


def _random_covariance(n, seed):
    """Well-conditioned, fully correlated covariance."""
    A = np.random.default_rng(seed).normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


class TestSequenceNodeMapBack:
    """Carrying the whole-arc batch solution backward across a node."""

    @pytest.mark.parametrize("route", ["covariance", "information"])
    def test_dropped_states_match_joint_update(
        self, sequence_filter, state, burn_leg_model, route
    ):
        """
        A burn leg followed by a coast drops the burn solve-fors at the node.
        Data after the node only see the kept position/velocity; recovering the
        dropped states must give what a joint update of the whole state gives,
        whether the forward solution at the node arrives as a covariance (LSB)
        or as a square-root information matrix (SRIFB).
        """
        sequence_filter.legs_model = [burn_leg_model, state]
        sequence_filter.flag_sequence = True

        # forward solution of the burn leg at the node (pos, vel, k_thrust1, dRA1)
        P_f = _random_covariance(8, seed=1)
        dx_f = np.random.default_rng(2).normal(size=(8, 1))

        # data after the node: z = H x_kept + v
        rng = np.random.default_rng(3)
        H = rng.normal(size=(4, 6))
        R = np.diag([0.5, 0.2, 0.3, 0.1])
        z = rng.normal(size=(4, 1))

        def update(dx, P, H):
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            return dx + K @ (z - H @ dx), (np.eye(P.shape[0]) - K @ H) @ P

        # what the coast leg solves for, and the reference joint update
        dx_kept, P_kept = update(dx_f[:6], P_f[:6, :6], H)
        dx_joint, P_joint = update(dx_f, P_f, np.hstack((H, np.zeros((4, 2)))))

        if route == "covariance":
            node = {"node_covariance": P_f}
        else:
            # any R with R^T R = inv(P_f): SRIFB's is not triangular in general
            Q_o, _ = np.linalg.qr(np.random.default_rng(14).normal(size=(8, 8)))
            R_f = Q_o @ np.linalg.cholesky(np.linalg.inv(P_f)).T
            node = {"node_covariance": None, "node_information": R_f}

        dx, P = sequence_filter._map_back_across_node(1, dx_kept, P_kept, dx_f, **node)

        assert dx.shape == (8, 1)
        assert np.allclose(dx, dx_joint)
        assert np.allclose(P, P_joint)

    def test_recovery_stays_accurate_after_a_dense_arc(
        self, sequence_filter, state, burn_leg_model
    ):
        """
        After a dense arc the smoothed kept covariance is many orders of
        magnitude below the forward one at the node. The covariance-form step
        P_dd + G (P_kk^s - P_kk) G^T then differences two nearly equal large
        matrices, loses the small result (a few percent here, an indefinite
        matrix and sigmas above the a priori on a real arc), while the
        information form has nothing to cancel.

        The forward information makes the dropped states A x_k plus noise of
        sigma c, so the exact answer is known in closed form.
        """
        sequence_filter.legs_model = [burn_leg_model, state]
        sequence_filter.flag_sequence = True

        rng = np.random.default_rng(13)
        s_k, c = 1e4, 1e-6
        A = rng.normal(size=(2, 6))
        R_f = np.zeros((8, 8))
        R_f[:6, :6] = np.eye(6) / s_k
        R_f[6:, :6] = -A / c
        R_f[6:, 6:] = np.eye(2) / c
        P_kk_s = 1e-12 * _random_covariance(6, seed=15)

        dx_f = np.zeros((8, 1))
        dx_f[:6] = rng.normal(size=(6, 1))
        dx_f[6:] = A @ dx_f[:6]
        dx_kk_s = rng.normal(size=(6, 1))

        dx, P = sequence_filter._map_back_across_node(
            1, dx_kk_s, P_kk_s, dx_f, None, node_information=R_f
        )

        def rel(a, b):
            return np.linalg.norm(a - b) / np.linalg.norm(b)

        assert rel(P[6:, 6:], c**2 * np.eye(2) + A @ P_kk_s @ A.T) < 1e-8
        assert rel(P[6:, :6], A @ P_kk_s) < 1e-8
        assert rel(dx[6:], A @ dx_kk_s) < 1e-8
        assert np.linalg.eigvalsh(P).min() > 0.0

    def test_augmented_states_are_discarded(self, sequence_filter):
        """A coast followed by a burn only reduces: the solve-fors added there go."""
        sequence_filter.flag_sequence = True
        dx_s = np.arange(8.0)  # 1-D on purpose: the result is still a column
        P_s = _random_covariance(8, seed=4)

        dx, P = sequence_filter._map_back_across_node(1, dx_s, P_s, None, None)

        assert dx.shape == (6, 1)
        assert np.allclose(dx.ravel(), dx_s[:6])
        assert np.allclose(P, P_s[:6, :6])

    def test_dropped_considered_states_are_not_updated(
        self, sequence_filter, state, burn_consider_leg_model
    ):
        """
        Considered states are not solved for, recovered or not: their deviation
        stays at zero and their error at the forward one. The reference is the
        covariance of the errors themselves: the kept ones updated by the data
        after the node, the dropped estimated one corrected through them, the
        considered one untouched.
        """
        sequence_filter.legs_model = [burn_consider_leg_model, state]
        sequence_filter.flag_sequence = True

        # forward joint error covariance at the node (pos, vel, k_thrust1, dRA1)
        P_f = _random_covariance(8, seed=5)
        dx_f = np.zeros((8, 1))
        dx_f[:7, 0] = np.random.default_rng(6).normal(size=7)

        # data after the node: z = H x_kept + v
        rng = np.random.default_rng(7)
        H = rng.normal(size=(4, 6))
        R = np.diag([0.5, 0.2, 0.3, 0.1])
        P_kk = P_f[:6, :6]
        K = P_kk @ H.T @ np.linalg.inv(H @ P_kk @ H.T + R)
        dx_kept = dx_f[:6] + K @ (rng.normal(size=(4, 1)) - H @ dx_f[:6])
        P_kept = (np.eye(6) - K @ H) @ P_kk

        # errors after the node as a linear map of the forward errors and v
        G = P_f[6:7, :6] @ np.linalg.inv(P_kk)
        L = np.zeros((8, 12))
        L[:6, :6] = np.eye(6) - K @ H
        L[:6, 8:] = -K
        L[6:7] = np.hstack((np.zeros((1, 6)), [[1.0, 0.0]], np.zeros((1, 4))))
        L[6:7] += G @ (L[:6] - np.hstack((np.eye(6), np.zeros((6, 6)))))
        L[7, 7] = 1.0
        Sigma = np.zeros((12, 12))
        Sigma[:8, :8] = P_f
        Sigma[8:, 8:] = R

        dx, P = sequence_filter._map_back_across_node(
            1, dx_kept, P_kept, dx_f, P_f
        )

        assert dx[-1, 0] == 0.0
        assert dx[-2, 0] != pytest.approx(dx_f[-2, 0])
        assert P[-1, -1] == pytest.approx(P_f[-1, -1])
        assert np.allclose(P, L @ Sigma @ L.T)

    def test_sequential_estimate_mapped_back_through_node(
        self, sequence_filter, state, burn_leg_model
    ):
        """
        LKF and SRIF end on the coast leg's estimate, which no longer holds the
        burn solve-fors. Mapping it back to the burn leg must recover them from
        the forward solution the filter stored at the node, as a joint update
        of the whole state does. Without that solution the reduction cannot be
        inverted.
        """
        from types import SimpleNamespace

        sequence_filter.legs_model = [burn_leg_model, state]
        sequence_filter.flag_sequence = True
        sequence_filter.legs_epochs = [np.array([0.0, 10.0]), np.array([10.0, 20.0])]
        sequence_filter.legs_n = [8, 6]
        # identity STMs: only the node is exercised, not the in-leg mapping
        sequence_filter.trajectory = SimpleNamespace(
            get_STM_sequence=lambda epoch, idx_leg: np.eye(sequence_filter.legs_n[idx_leg])
        )

        # forward solution of the burn leg at the node, then data on the coast
        P_f = _random_covariance(8, seed=7)
        dx_f = np.random.default_rng(8).normal(size=(8, 1))
        rng = np.random.default_rng(9)
        H = rng.normal(size=(4, 6))
        R = np.diag([0.5, 0.2, 0.3, 0.1])
        z = rng.normal(size=(4, 1))

        def update(dx, P, H):
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            return dx + K @ (z - H @ dx)

        dx_kept = update(dx_f[:6], P_f[:6, :6], H)
        dx_joint = update(dx_f, P_f, np.hstack((H, np.zeros((4, 2)))))

        # one filtered row per leg, the coast row padded to the burn leg's size
        rows = [dx_f.ravel(), np.concatenate((dx_kept.ravel(), np.zeros(2)))]
        sol = object.__new__(scb.SolutionOD)
        sol._filter = sequence_filter
        sol.deviation_smooth = None
        sol.deviation_est = np.vstack(rows)
        sol._deviation_history = lambda use_smoothed=True: (
            np.array([5.0, 15.0]), rows, False
        )
        sol._node_deviations = [dx_f]
        sol._node_covariances = [P_f]

        burn, coast = sol.map_state_deviation_to_epoch(
            map_back_sequence=True, use_smoothed=False
        )

        assert np.allclose(coast, dx_kept.ravel())
        assert np.allclose(burn, dx_joint.ravel())

        sol._node_deviations = sol._node_covariances = None
        with pytest.raises(ValueError, match="missing during reduction"):
            sol.map_state_deviation_to_epoch(map_back_sequence=True, use_smoothed=False)


class TestMapNodeToLegEpoch:
    """Carrying a node solution back to its leg's reference epoch."""

    def test_round_trip(self):
        """Phi^-1 undoes the forward mapping of both deviation and covariance."""
        od = object.__new__(scb.FilterOD)
        rng = np.random.default_rng(16)
        STM = np.eye(6) + 0.3 * rng.normal(size=(6, 6))
        dx0 = rng.normal(size=(6, 1))
        P0 = _random_covariance(6, seed=17)

        with warnings.catch_warnings():
            warnings.simplefilter("error", scb.FilterOD.CovarianceNotPSDWarning)
            dx, P = od._map_node_to_leg_epoch(STM, STM @ dx0, STM @ P0 @ STM.T)

        assert np.allclose(dx, dx0)
        assert np.allclose(P, P0)
        assert np.array_equal(P, P.T)

    def test_indefinite_covariance_reported(self):
        od = object.__new__(scb.FilterOD)

        with pytest.warns(scb.FilterOD.CovarianceNotPSDWarning, match="smoothed"):
            od._map_node_to_leg_epoch(np.eye(2), None, np.diag([1.0, -1.0]), "smoothed")


class TestSplitConsiderCovariance:
    """A mapped-back joint covariance stored as the leg's own pair."""

    def test_leg_without_considered_states_has_no_consider_row(
        self, sequence_filter, burn_consider_leg_model, state
    ):
        sequence_filter.legs_model = [burn_consider_leg_model, state]
        sequence_filter.flag_sequence = True
        P = _random_covariance(6, seed=18)

        conditional, joint = sequence_filter._split_consider_covariance(1, P)

        assert conditional is P
        assert joint is None

    def test_conditional_block_is_the_filter_one(
        self, sequence_filter, burn_consider_leg_model, state
    ):
        """
        On a consider leg P_c = P_xx + S P_cc S^T and P_xc = S P_cc, so the
        split gives back the filter's own P_xx beside the same P_xc and P_cc.
        """
        sequence_filter.legs_model = [burn_consider_leg_model, state]
        sequence_filter.flag_sequence = True
        P_xx = _random_covariance(7, seed=19)
        P_cc = np.array([[0.3]])
        P_xc = np.random.default_rng(20).normal(size=(7, 1)) @ P_cc
        J = np.block([[P_xx + P_xc @ np.linalg.solve(P_cc, P_xc.T), P_xc], [P_xc.T, P_cc]])

        conditional, joint = sequence_filter._split_consider_covariance(0, J)

        assert joint is J
        assert np.allclose(conditional, np.block([[P_xx, P_xc], [P_xc.T, P_cc]]))
        assert np.array_equal(conditional, conditional.T)


class TestValidateSequenceBatch:
    """Dropping states between legs is reported as a warning, not printed."""

    def test_dropped_states_named_in_warning(self, sequence_filter, state, burn_leg_model):
        sequence_filter.legs_model = [burn_leg_model, state]

        with pytest.warns(scb.FilterOD.SequenceStateDroppedWarning, match="k_thrust1"):
            sequence_filter._validate_sequence_batch()

    def test_augmentation_alone_is_silent(self, sequence_filter):
        with warnings.catch_warnings():
            warnings.simplefilter("error", scb.FilterOD.SequenceStateDroppedWarning)
            sequence_filter._validate_sequence_batch()


def _information(P, dx):
    """Square-root information array [R | z] of a Gaussian (upper-triangular R)."""
    R = np.linalg.cholesky(np.linalg.inv(P)).T
    return R, R @ dx


def _moments(R, z):
    """Covariance and mean of a square-root information array."""
    R_inv = np.linalg.inv(R)
    return R_inv @ R_inv.T, R_inv @ z


class TestSequenceNodeInformation:
    """SRIFB carries the information array, not the covariance, across a node."""

    def test_augmentation_matches_covariance_route(self, sequence_filter):
        """A coast followed by a burn: new solve-fors enter at their a priori."""
        sequence_filter.flag_sequence = True
        P = _random_covariance(6, seed=7)
        dx = np.random.default_rng(8).normal(size=(6, 1))
        x_bar = np.zeros((8, 1))
        x_bar[6:, 0] = [0.01, -0.002]
        sequence_filter._apriori_deviation = lambda idx_leg: x_bar

        R_new, z_new = sequence_filter._map_information_definition_node(
            1, *_information(P, dx)
        )
        P_new, dx_new = _moments(R_new, z_new)

        assert np.allclose(np.tril(R_new, -1), 0.0)
        assert np.allclose(P_new, sequence_filter._map_covariance_definition_node(1, P))
        assert np.allclose(
            dx_new.ravel(), sequence_filter._map_deviation_definition_node(1, dx)
        )

    def test_reduction_marginalises_dropped_states(
        self, sequence_filter, state, burn_leg_model
    ):
        """A burn followed by a coast: the dropped solve-fors are marginalised."""
        sequence_filter.legs_model = [burn_leg_model, state]
        P = _random_covariance(8, seed=9)
        dx = np.random.default_rng(10).normal(size=(8, 1))

        R_new, z_new = sequence_filter._map_information_definition_node(
            1, *_information(P, dx)
        )
        P_new, dx_new = _moments(R_new, z_new)

        assert R_new.shape == (6, 6)
        assert np.allclose(np.tril(R_new, -1), 0.0)
        assert np.allclose(P_new, P[:6, :6])
        assert np.allclose(dx_new, dx[:6])


class TestAprioriDeviation:
    """Each iteration's a priori is re-centred on the original a-priori state."""

    def test_single_arc_undoes_accumulated_correction(self):
        """With no correction yet the a priori is the reference itself."""
        od = object.__new__(scb.FilterOD)
        od.flag_sequence = False
        od.n = 3
        assert np.array_equal(od._apriori_deviation(), np.zeros((3, 1)))

        od._delta0_total = np.array([[1.0], [-2.0], [0.5]])
        assert np.array_equal(od._apriori_deviation(), -od._delta0_total)

    def test_sequence_leg_reads_its_global_blocks(self):
        """A leg picks its own blocks out of the global correction."""
        from types import SimpleNamespace

        od = object.__new__(scb.FilterOD)
        od.flag_sequence = True
        od.legs_n = [6, 8]
        od.trajectory = SimpleNamespace(
            _sequence_definition=SimpleNamespace(
                names=["coast", "impulse", "burn"],
                types=["Leg", "Maneuver", "Finite Burn"],
            )
        )
        od.propagator = SimpleNamespace(
            _leg_to_global_maps={"burn": [(0, 6, 0, 6), (6, 8, 9, 11)]}
        )
        assert np.array_equal(od._apriori_deviation(1), np.zeros((8, 1)))

        od._delta0_total = np.arange(1.0, 12.0).reshape(-1, 1)
        x_bar = od._apriori_deviation(1)

        assert np.array_equal(x_bar[:6, 0], -np.arange(1.0, 7.0))
        assert np.array_equal(x_bar[6:, 0], [-10.0, -11.0])


class TestSqrtInfoMatrix:
    """Square-root information of a covariance whose entries span many units."""

    def test_matches_cholesky_of_the_inverse(self):
        """km, km/s and unitless blocks side by side give chol(inv(P)) exactly."""
        od = object.__new__(scb.FilterOD)
        D = np.diag([10.0, 10.0, 10.0, 1e-4, 1e-4, 1e-4, 0.05, 0.02])
        P = D @ _random_covariance(8, seed=11) @ D

        R = od._sqrt_info_matrix(P)

        assert np.allclose(np.tril(R, -1), 0.0)
        assert np.allclose(R, np.linalg.cholesky(np.linalg.inv(P)).T, rtol=1e-9)

    def test_fallback_stays_upper_triangular(self):
        """SRIF back-substitutes through R, so the fallback must be triangular too."""
        od = object.__new__(scb.FilterOD)
        P = _random_covariance(6, seed=12)
        P[0, 0] = -1.0

        with pytest.warns(RuntimeWarning, match="Cholesky"):
            R = od._sqrt_info_matrix(P)

        assert np.allclose(np.tril(R, -1), 0.0)
        assert np.all(np.diag(R) > 0.0)
        assert np.all(np.isfinite(R))


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Trajectory sidecar formats                                 #
# ------------------------------------------------------------------#
# The STM history is the biggest artifact an OD run writes — one matrix per
# output epoch, growing with the square of the state size — so it is the one
# worth writing as binary. It is a different setting from the measurement
# records, which is a distinction that has caught people out.


def output_settings_with(**kwargs):
    """OutputSettings carrying only the fields under test."""
    return scb.OutputSettings(**kwargs)


def filter_with_output(output):
    """A bare FilterOD carrying just the settings the sidecar writer reads."""
    from types import SimpleNamespace

    od = object.__new__(scb.FilterOD)
    od.settings = SimpleNamespace(output=output)
    return od


class TestTrajectorySidecarFormats:
    """_trajectory_write_formats' choice of format for the STM sidecars."""

    def test_defaults_to_json(self):
        """Trajectory sidecars are human-readable unless asked otherwise."""
        od = filter_with_output(output_settings_with())

        formats = od._trajectory_write_formats()
        assert formats["stm_format"] == "json"
        assert formats["sidecar_format"] == "json"

    def test_trajectory_output_format_reaches_the_sidecars(self):
        """OutputSettings can put the STM history in the compact binary form."""
        od = filter_with_output(output_settings_with(trajectory_output_format="pickle"))

        formats = od._trajectory_write_formats()
        assert formats["stm_format"] == "pickle"
        assert formats["sidecar_format"] == "pickle"

    def test_measurement_format_does_not_move_the_sidecars(self):
        """fdm_output_format covers the measurement records, not the STMs."""
        settings = output_settings_with(fdm_output_format="pickle")
        od = filter_with_output(settings)

        assert settings.fdm_output_format == "pickle"
        # the trajectory sidecars are a separate decision and stay as they were
        assert od._trajectory_write_formats()["stm_format"] == "json"


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Measurement sigma overrides                                #
# ------------------------------------------------------------------#
# FilterOD.set_measurement_sigmas (which the construction uses to apply
# FilterSettings.measurement_sigmas) sets each sigma on the model behind the
# dataset it names, and on the dataset's sigma column. The column is in the
# dataset's own units, so a sigma carrying units has to be converted into them
# rather than having its bare value copied.
def _sigma_dataset(name, units, n=3):
    """A dataset with a unit sigma column, built without running a model."""
    names = ["t1", "t2", "t3", "computed", "observed",
             "residuals", "partials", "sigma", "outlier_flag"]
    t2 = np.arange(n, dtype=float)
    values = [t2 - 1.0, t2, t2 + 1.0, np.zeros(n), np.zeros(n), np.zeros(n),
              np.zeros((n, 6)), np.ones(n), np.zeros(n)]
    return scb.MeasurementDataSet(
        set_name=name, measurements=values, names=names,
        instrument=scb.GroundStation("DSS-24"), target=None,
        epochs_t3=None, measurement_type="TestRange", units=units,
    )


def _sigma_model():
    """Stands in for a measurement model: the filter reads and sets its sigma only."""
    from types import SimpleNamespace

    return SimpleNamespace(name="stub", sigma=None)


def _sigma_filter(datasets, sources):
    """A FilterOD over ``datasets``, built by ``sources`` = [(model, names)], without running a filter."""
    filt = scb.FilterOD.__new__(scb.FilterOD)
    filt.measurement_data = scb.FilterDataManager(datasets)
    filt._sigma_sources = sources
    return filt


def _override_filter(units, n=3):
    """A FilterOD holding one dataset 'd', built by one model."""
    ds = _sigma_dataset("d", units, n)
    model = _sigma_model()
    return _sigma_filter([ds], [(model, ["d"])]), ds, model


def test_sigma_override_with_units_is_converted_to_dataset_units():
    filt, ds, model = _override_filter(units=m)
    filt.set_measurement_sigmas({"d": scb.ArrayWUnits(0.002, km)})
    assert np.allclose(ds.data["sigma"], 2.0)
    assert model.sigma.units == m and np.isclose(model.sigma.values, 2.0)


def test_sigma_override_bare_float_is_taken_in_dataset_units():
    filt, ds, model = _override_filter(units=m)
    filt.set_measurement_sigmas({"d": 2.0})
    assert np.allclose(ds.data["sigma"], 2.0)
    assert model.sigma.units == m and np.isclose(model.sigma.values, 2.0)


def test_sigma_override_incompatible_units_are_refused():
    filt, ds, model = _override_filter(units=m)
    with pytest.raises(scb.FilterOD.MeasurementSigmaUnitsError,
                       match="measurement_sigmas\\['d'\\]"):
        filt.set_measurement_sigmas({"d": scb.ArrayWUnits(1.0, km / sec)})
    assert np.allclose(ds.data["sigma"], 1.0), "sigma left untouched on error"
    assert model.sigma is None


def test_sigma_override_without_dataset_units_takes_the_value_as_given():
    filt, ds, model = _override_filter(units=None)
    filt.set_measurement_sigmas({"d": scb.ArrayWUnits(3.0, km)})
    assert np.allclose(ds.data["sigma"], 3.0)
    assert model.sigma == 3.0


def test_sigma_override_unknown_dataset_name_is_refused():
    filt, _, _ = _override_filter(units=m)
    with pytest.raises(scb.FilterOD.UnknownSigmaDatasetError, match="unknown dataset"):
        filt.set_measurement_sigmas({"nope": 1.0})


def test_sigma_override_of_one_component_keeps_the_other():
    """A two-component model gets a per-component sigma; the component not named keeps its own."""
    ra, dec = _sigma_dataset("cam_ra", deg), _sigma_dataset("cam_dec", deg)
    model = _sigma_model()
    filt = _sigma_filter([ra, dec], [(model, ["cam_ra", "cam_dec"])])

    filt.set_measurement_sigmas({"cam_ra": 2.0})

    assert np.allclose(ra.data["sigma"], 2.0)
    assert np.allclose(dec.data["sigma"], 1.0)
    assert model.sigma.units == deg
    assert np.allclose(model.sigma.values, [2.0, 1.0])


def test_sigma_override_conflicting_on_a_shared_model_is_refused():
    """Two datasets of one model cannot be given two sigmas, and nothing is set when refused."""
    p, q = _sigma_dataset("p", m), _sigma_dataset("q", m)
    model = _sigma_model()
    filt = _sigma_filter([p, q], [(model, ["p"]), (model, ["q"])])

    with pytest.raises(scb.FilterOD.SharedSigmaModelError, match="One model"):
        filt.set_measurement_sigmas({"p": 2.0})
    assert model.sigma is None
    assert np.allclose(p.data["sigma"], 1.0)

    filt.set_measurement_sigmas({"p": 2.0, "q": 2.0})
    assert np.isclose(model.sigma.values, 2.0)
    assert np.allclose(q.data["sigma"], 2.0)


def _range_column(od):
    """The sigma column of the 'GS1 Range' datasets, in metres."""
    columns = [
        scb.ArrayWUnits(np.asarray(ds.data["sigma"], dtype=float), ds.units).convert_to(m).values
        for ds in od.measurement_data.datasets if ds.set_name == "GS1 Range"
    ]
    assert columns
    return np.concatenate(columns)


class TestMeasurementSigmasOnTheModel:
    """settings.measurement_sigmas go on the model once; the model is then the only source."""

    def test_construction_sets_the_model_sigma(self, propagator, covar_mat, measurements):
        settings = scb.FilterSettings(
            initial_covariance=covar_mat,
            measurement_sigmas={"GS1 Range": scb.ArrayWUnits(2.0, m)},
        )
        od = scb.FilterOD(propagator, measurements, settings)

        model = next(iter(measurements))["model"]
        assert np.isclose(model.sigma.convert_to(m).values, 2.0)
        assert np.allclose(_range_column(od), 2.0)

    def test_a_later_model_sigma_wins_over_the_settings(self, propagator, covar_mat, measurements):
        """A refit (MeasurementSpec.edit) or a manual change of model.sigma is no longer overwritten."""
        settings = scb.FilterSettings(
            initial_covariance=covar_mat,
            measurement_sigmas={"GS1 Range": scb.ArrayWUnits(2.0, m)},
        )
        od = scb.FilterOD(propagator, measurements, settings, traj_name="ref.bsp")

        next(iter(measurements))["model"].sigma = scb.ArrayWUnits(4.0, m)
        # own name: an auto name taken in the same second unloads the first SPK (known flake)
        od._reinitialize(traj_name="ref_rebuilt.bsp", verbose=False)

        assert np.allclose(_range_column(od), 4.0)


class TestFitCallback:
    """fit(callback=...) runs between iterations, before the data are rebuilt."""

    def test_called_after_every_iteration(self, propagator, covar_mat, measurements):
        settings = scb.FilterSettings(initial_covariance=covar_mat)
        od = ScriptedFilter(propagator, measurements, settings)
        od.script([(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)])
        calls = []

        _, iterations, _ = od.fit(
            max_iterations=5,
            convergence_threshold_cost=1e-3,
            verbose=False,
            callback=lambda filt, it, sol: calls.append(
                (filt, it, filt._convergence_hist[-1]["iteration"], sol)
            ),
        )

        # converges on the cost at iteration 2, and that iteration is reported too
        assert iterations == 2
        assert [c[1] for c in calls] == [1, 2]
        assert all(c[0] is od and c[2] == c[1] for c in calls)
        assert calls[-1][3] is not None

    def test_sigma_set_in_the_callback_reaches_the_next_iteration(
        self, propagator, covar_mat, measurements, tmp_path
    ):
        class RelinearisingFilter(ScriptedFilter):
            def _reinitialize(self, *args, **kwargs):
                return scb.FilterOD._reinitialize(self, *args, **kwargs)

        settings = scb.FilterSettings(
            initial_covariance=covar_mat,
            measurement_sigmas={"GS1 Range": scb.ArrayWUnits(2.0, m)},
        )
        od = RelinearisingFilter(
            propagator, measurements, settings, traj_name="ref.bsp", traj_dir=str(tmp_path)
        )
        od.script([(1e-9, 1.0)] * 2)
        columns = []

        def reweight(filt, iteration, solution):
            columns.append(_range_column(filt).copy())
            if iteration == 1:
                filt.set_measurement_sigmas({"GS1 Range": scb.ArrayWUnits(5.0, m)})

        od.fit(
            max_iterations=2,
            convergence_threshold=1e-12,
            convergence_criterion="deviation",
            verbose=False,
            callback=reweight,
        )

        assert np.allclose(columns[0], 2.0)
        assert np.allclose(columns[1], 5.0)
        assert np.isclose(next(iter(measurements))["model"].sigma.convert_to(m).values, 5.0)


# endregion                                                          #
# ------------------------------------------------------------------#
