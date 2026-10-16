# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the SolutionOD class  #
# =======================================#
import types

import src.scarabaeus as scb
import numpy as np
import pytest
from types import SimpleNamespace
km, kg, sec, hr = scb.Units.get_units(["km", "kg", "sec", "hr"])
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
    # measurement data (conftest meas_data_file) uses doy=39 (2028 Feb 8)
    # epochs must start at Feb 8 and use 1hr steps so the BSP covers
    # the measurement time (~05:31) with enough interpolation points
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


@pytest.fixture
def sc(sc_area, sc_ref_coeff):
    """Define spacecraft"""
    return scb.Spacecraft(
        "orbiter_ref_it1",
        -1001,
        scb.ArrayWUnits(2000.0, kg),
        sc_area,
        sc_ref_coeff,)


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

    prop = scb.Propagator(primary_body=sc, state_vector=state, tspan=epochs, force_models=force_model)
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
    range_gs1 = scb.RangeIdeal(
        "GS1 Ideal Range Model", GS1, sigma=scb.ArrayWUnits(1e-3, km))

    obs_quantities_range = range_gs1.observed_measurements(
        file_name=str(meas_data_file),
        meas_name="meas_ideal",
        units=km,)

    measurements_list = scb.MeasurementSpec.many(
        scb.MeasurementSpec(
            model=range_gs1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",),
        scb.MeasurementSpec(
            model=range_gs1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",),)
    return measurements_list


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
    solution_od = scb.SolutionOD(test_filter_od)

    assert solution_od._filter == test_filter_od


def test_global_std_devs(propagator, covar_mat, measurements):
    """
    Verifies that global_std_devs returns correct standard deviations when
    multi_arc_global is populated (as MultiFilterOD.fit() would set it).
    """
    settings = scb.FilterSettings(initial_covariance=covar_mat)
    test_filter_od = scb.FilterOD(propagator, measurements, settings)
    solution = scb.SolutionOD(test_filter_od)

    n = solution.n_states
    P_G = np.diag([4.0] * n)
    solution.multi_arc_global = {"P_G": P_G}

    std_devs = solution.global_std_devs()
    assert std_devs.shape == (n,)
    assert np.allclose(std_devs, [2.0] * n)


class TestDeviationPadding:
    """
    Tests for the right-pad branch in SolutionOD._store_solution_data.

    Triggered when state_deviation_history rows have unequal lengths.
    A minimal stub provides the required FilterOD interface without a full run.
    """

    DEVIATION_ONLY = scb.OutputSettings(
        save_state_deviation_history=True,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def make_solution(self, history, **filter_attrs):
        """Return a SolutionOD built from a stub filter with the given history.

        ``filter_attrs`` are set on the stub filter as well.
        """
        timestamps = [float(i) for i in range(len(history))]

        class StubFilter:
            covariance_analysis = False
            has_consider = False
            state_deviation_history = history

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return timestamps

        stub = StubFilter()
        for name, value in filter_attrs.items():
            setattr(stub, name, value)
        return scb.SolutionOD(stub, self.DEVIATION_ONLY)

    def test_jagged_rows_padded_to_max_width(self):
        """Rows with varying lengths are right-padded with zeros to the max width."""
        history = [
            np.array([1.0, 2.0, 3.0]),       # width 3
            np.array([4.0, 5.0]),             # width 2 — shorter
            np.array([6.0, 7.0, 8.0, 9.0]),  # width 4 — longest
        ]
        sol = self.make_solution(history)
        assert sol.deviation_est.shape == (3, 4)
        np.testing.assert_array_equal(sol.deviation_est[0], [1.0, 2.0, 3.0, 0.0])
        np.testing.assert_array_equal(sol.deviation_est[1], [4.0, 5.0, 0.0, 0.0])
        np.testing.assert_array_equal(sol.deviation_est[2], [6.0, 7.0, 8.0, 9.0])

    def test_empty_history_yields_zero_shape(self):
        """Empty history results in a (0, 0) deviation array."""
        sol = self.make_solution([])
        assert sol.deviation_est.shape == (0, 0)

    def test_uniform_rows_use_vstack_path(self):
        """Equal-width rows take the fast vstack path (no padding needed)."""
        history = [np.array([1.0, 2.0, 3.0])] * 3
        sol = self.make_solution(history)
        assert sol.deviation_est.shape == (3, 3)
        np.testing.assert_array_equal(sol.deviation_est[0], [1.0, 2.0, 3.0])

    def test_cumulative_deviation_matches_deviation_est(self):
        """Without _cumulative_deviation on the filter, cumulative == deviation_est."""
        history = [np.array([1.0, 2.0]), np.array([3.0])]
        sol = self.make_solution(history)
        np.testing.assert_array_equal(sol.deviation_cumulative, sol.deviation_est)

    def test_cumulative_deviation_includes_the_current_pass(self):
        """The cumulative deviation is what earlier passes applied to the
        reference plus this pass's own deviation, not the earlier passes alone."""
        history = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        applied = np.array([[10.0, 20.0], [30.0, 40.0]])
        sol = self.make_solution(history, _cumulative_deviation=applied)

        np.testing.assert_array_equal(sol.deviation_est, np.vstack(history))
        np.testing.assert_array_equal(
            sol.deviation_cumulative, np.vstack(history) + applied
        )


class TestStateCount:
    """n_states reports the whole state vector, not the last leg processed."""

    @staticmethod
    def solution_over(filter_od):
        sol = object.__new__(scb.SolutionOD)
        sol._filter = filter_od
        return sol

    def test_sequence_reports_the_union_of_its_legs(self):
        """A sequence filter's n is the width of whichever leg it ran last."""
        from types import SimpleNamespace

        sol = self.solution_over(SimpleNamespace(n=6, n_global=12))
        assert sol.n_states == 12

    def test_single_arc_reports_the_filter_width(self):
        from types import SimpleNamespace

        sol = self.solution_over(SimpleNamespace(n=7))
        assert sol.n_states == 7


class TestTrajectorySidecarFormat:
    """The saved reference trajectory honours the configured sidecar format.

    Writing the STM history as JSON dominates the cost of saving a solution on
    a finely sampled arc, so the setting has to reach this writer and not only
    the filter's own.
    """

    @staticmethod
    def solution_with(fmt=None):
        """A SolutionOD carrying only the output settings under test."""
        settings = scb.OutputSettings(
            **({"trajectory_output_format": fmt} if fmt else {})
        )

        class StubFilter:
            covariance_analysis = False
            has_consider = False

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        return scb.SolutionOD(StubFilter(), settings)

    def test_defaults_to_json(self):
        """Sidecars stay human-readable unless asked otherwise."""
        formats = self.solution_with()._trajectory_write_formats()
        assert formats["stm_format"] == "json"
        assert formats["sidecar_format"] == "json"

    def test_pickle_reaches_the_writer(self):
        """The compact binary form is selected for both sidecar kinds."""
        formats = self.solution_with("pickle")._trajectory_write_formats()
        assert formats["stm_format"] == "pickle"
        assert formats["sidecar_format"] == "pickle"


class TestDebugData:
    """
    Indirect tests for SolutionOD._collect_debug_data via constructor.

    Each test enables one or more debug OutputSettings flags and checks
    that sol.debug reflects the correct keys/values from the stub filter.
    """

    NO_DEV = scb.OutputSettings(
        save_state_deviation_history=False,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def make_solution(self, output_settings, **filter_attrs):
        """Build SolutionOD from a one-epoch stub, optionally injecting filter attrs."""

        class StubFilter:
            covariance_analysis = False
            has_consider = False

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        stub = StubFilter()
        for key, val in filter_attrs.items():
            setattr(stub, key, val)
        return scb.SolutionOD(stub, output_settings)

    def debug_settings(self, **flags):
        """Return OutputSettings with all storage off except the given debug flags."""
        s = scb.OutputSettings(
            save_state_deviation_history=False,
            save_covariance_history=False,
            save_prefit_residuals=False,
            save_postfit_residuals=False,
            save_consider_covariance_history=False,
            save_smoothed_solution=False,
        )
        for k, v in flags.items():
            setattr(s, k, v)
        return s

    def test_kalman_gains_stored(self):
        """save_kalman_gains=True + filter.kalman_gains → debug['kalman_gains'] set."""
        gains = [np.eye(6), 2 * np.eye(6)]
        sol = self.make_solution(
            self.debug_settings(save_kalman_gains=True),
            kalman_gains=gains,
        )
        assert sol.debug is not None
        assert "kalman_gains" in sol.debug
        assert sol.debug["kalman_gains"] is gains

    def test_innovation_covariances_stored(self):
        """save_innovation_covariances=True + attr → debug['innovation_covariances'] set."""
        covs = [np.eye(2), np.eye(2)]
        sol = self.make_solution(
            self.debug_settings(save_innovation_covariances=True),
            innovation_covariances=covs,
        )
        assert sol.debug is not None
        assert "innovation_covariances" in sol.debug
        assert sol.debug["innovation_covariances"] is covs

    def test_both_debug_keys_stored(self):
        """Both flags set + both attrs present → debug has both keys."""
        gains = [np.eye(6)]
        covs = [np.eye(2)]
        sol = self.make_solution(
            self.debug_settings(save_kalman_gains=True, save_innovation_covariances=True),
            kalman_gains=gains,
            innovation_covariances=covs,
        )
        assert "kalman_gains" in sol.debug
        assert "innovation_covariances" in sol.debug

    def test_debug_none_when_flag_set_but_attr_absent(self):
        """Flag set but filter lacks the attribute → debug_data stays empty → None."""
        sol = self.make_solution(self.debug_settings(save_kalman_gains=True))
        assert sol.debug is None

    def test_debug_none_when_no_debug_flags(self):
        """No debug flags → _collect_debug_data never called → sol.debug is None."""
        sol = self.make_solution(self.NO_DEV)
        assert sol.debug is None

    def test_nis_statistics_absent_without_innovations(self):
        """save_nis_statistics=True but no filter innovations → no nis_statistics."""
        sol = self.make_solution(self.debug_settings(save_nis_statistics=True))
        assert sol.debug is None

    def test_nis_statistics_from_innovations(self):
        """One ν^T S^{-1} ν per update; update sizes may differ; singular S → NaN."""
        S_2 = np.array([[4.0, 1.0], [1.0, 2.0]])
        nu_2 = np.array([[1.0], [-2.0]])
        sol = self.make_solution(
            self.debug_settings(save_nis_statistics=True),
            innovations=[nu_2, np.array([3.0]), np.array([1.0])],
            innovation_covariances=[S_2, np.array([[9.0]]), np.zeros((1, 1))],
        )
        nis = sol.debug["nis_statistics"]
        expected = (nu_2.T @ np.linalg.inv(S_2) @ nu_2).item()
        np.testing.assert_allclose(nis[:2], [expected, 1.0], rtol=1e-12)
        assert np.isnan(nis[2])

    def test_nis_statistics_misaligned_histories_raise(self):
        """Innovations and their covariances must pair up per update."""
        with pytest.raises(scb.SolutionOD.DebugHistoryError, match="per measurement update"):
            self.make_solution(
                self.debug_settings(save_nis_statistics=True),
                innovations=[np.ones(2), np.ones(2)],
                innovation_covariances=[np.eye(2)],
            )


class TestPrintStatsChiSquared:
    """The chi-squared block reads the ``{dataset: [[residual, sigma], ...]}`` postfits."""

    def solution(self, postfits):
        sol = TestDebugData().make_solution(TestDebugData.NO_DEV, n=2)
        sol.filter._cost_function = types.MethodType(scb.FilterOD._cost_function, sol.filter)
        sol.postfits = postfits
        return sol

    def test_chi_squared_over_all_datasets(self, capsys):
        """Σ(r/σ)² over every dataset, dof = residuals − states."""
        sol = self.solution({
            "range": [[1.0, 1.0], [4.0, 2.0]],
            "doppler": [[3.0, 1.0], [0.0, 1.0], [5.0, -1.0]],
        })

        sol.print_stats(verbose=True)

        out = capsys.readouterr().out
        assert "Chi-squared value:       14.0000" in out
        assert "Degrees of freedom:      2" in out

    def test_no_postfits_skips_the_block(self, capsys):
        sol = self.solution(None)

        sol.print_stats(verbose=True)

        assert "Chi-Squared Test" not in capsys.readouterr().out


class TestConsiderHistoryStorage:
    """
    A sequence may consider states on some legs only. The filter's
    ``has_consider`` is its last leg's, so the consider history is kept
    whenever any of its rows holds a covariance.
    """

    @staticmethod
    def solution_over(rows, covariance_analysis=False):
        """SolutionOD over a stub whose last leg considers nothing."""

        class StubFilter:
            has_consider = False
            covariance_history = []
            covariance_consider_history = rows

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        StubFilter.covariance_analysis = covariance_analysis
        settings = scb.OutputSettings(save_consider_covariance_history=True)
        return scb.SolutionOD(StubFilter(), settings)

    @pytest.mark.parametrize("covariance_analysis", [False, True])
    def test_kept_when_only_an_earlier_leg_considers(self, covariance_analysis):
        rows = [None, np.eye(3), None]
        sol = self.solution_over(rows, covariance_analysis)
        assert sol.covariance_consider is rows

    def test_not_kept_when_no_leg_considers(self):
        sol = self.solution_over([None, None])
        assert sol.covariance_consider is None


class TestPropagateState:
    """
    Error-path unit tests for SolutionOD.propagate_state.

    Functional correctness is covered by TestSolutionOD in the system tests.
    These tests exercise the two early-exit guards at the top of the method.
    """

    def make_stub_sol(self, output_settings, **filter_attrs):
        """Build a SolutionOD from a one-epoch stub, optionally injecting filter attrs."""

        class StubFilter:
            covariance_analysis = False
            has_consider = False

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        stub = StubFilter()
        for key, val in filter_attrs.items():
            setattr(stub, key, val)
        return scb.SolutionOD(stub, output_settings)

    NO_STORAGE = scb.OutputSettings(
        save_state_deviation_history=False,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    DEV_ONLY = scb.OutputSettings(
        save_state_deviation_history=True,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def test_raises_without_deviation_est(self):
        """ValueError when deviation_est is None (state history not saved)."""
        sol = self.make_stub_sol(self.NO_STORAGE)
        assert sol.deviation_est is None
        dummy = scb.EpochArray(np.array([0.0]), sys="TDB")
        with pytest.raises(ValueError, match="No state deviation history"):
            sol.propagate_state(dummy)

    def test_raises_without_trajectory(self):
        """ValueError when filter.trajectory is absent even if deviation_est is set."""
        sol = self.make_stub_sol(
            self.DEV_ONLY,
            state_deviation_history=[np.zeros(6)],
        )
        assert sol.deviation_est is not None
        dummy = scb.EpochArray(np.array([0.0]), sys="TDB")
        with pytest.raises(ValueError, match="no trajectory attached"):
            sol.propagate_state(dummy)


class TestMapStateDeviationToEpoch:
    """
    Unit tests for SolutionOD.map_state_deviation_to_epoch.

    Covers the smoothed fast-path and the empty-deviation guard using stubs.
    The non-sequence STM path is covered by TestSolutionOD in the system tests.
    """

    NO_STORAGE = scb.OutputSettings(
        save_state_deviation_history=False,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    DEV_ONLY = scb.OutputSettings(
        save_state_deviation_history=True,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def make_stub_sol(self, output_settings, **filter_attrs):
        """Build a one-epoch SolutionOD from a minimal stub."""

        class StubFilter:
            covariance_analysis = False
            has_consider = False

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        stub = StubFilter()
        for key, val in filter_attrs.items():
            setattr(stub, key, val)
        return scb.SolutionOD(stub, output_settings)

    @staticmethod
    def identity_stm_filter_attrs(n_states=6):
        """
        Stub attributes for the smoothed path.

        The smoothed deviation sits at the first measurement epoch, so the
        method maps it back through Phi(t1, t0)^-1. An identity STM makes that
        mapping a no-op, which is what lets these tests assert on
        ``deviation_smooth[0]`` while still exercising the real code path.
        """

        class StubTrajectory:
            @staticmethod
            def get_STM(epoch=None, idx=None):
                return np.eye(n_states)

        return {
            "trajectory": StubTrajectory(),
            "flag_sequence": False,
            "_safe_inv": np.linalg.inv,
        }

    def test_smoothed_path_returns_deviation_smooth_first_row(self):
        """use_smoothed=True with deviation_smooth set → returns deviation_smooth[0]."""
        sol = self.make_stub_sol(
            self.NO_STORAGE, **self.identity_stm_filter_attrs()
        )
        sol.deviation_smooth = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        ])
        sol.timeline = np.array([0.0, 1.0])
        result = sol.map_state_deviation_to_epoch(use_smoothed=True)
        np.testing.assert_array_equal(result, sol.deviation_smooth[0])

    def test_smoothed_path_ignores_deviation_est(self):
        """Smooth path returns deviation_smooth[0] regardless of deviation_est content."""
        sol = self.make_stub_sol(
            self.NO_STORAGE, **self.identity_stm_filter_attrs()
        )
        sol.deviation_smooth = np.ones((2, 6))
        sol.deviation_est = np.zeros((2, 6))
        sol.timeline = np.array([0.0, 1.0])
        result = sol.map_state_deviation_to_epoch(use_smoothed=True)
        np.testing.assert_array_equal(result, np.ones(6))

    def test_use_smoothed_false_skips_smooth_path(self):
        """use_smoothed=False must not return deviation_smooth[0] even if it exists."""
        sol = self.make_stub_sol(self.NO_STORAGE)
        sol.deviation_smooth = np.ones((2, 6)) * 99.0
        # deviation_est is None → guard fires instead (not the smooth path)
        result = sol.map_state_deviation_to_epoch(use_smoothed=False)
        assert not np.allclose(result, 99.0)

    def test_none_deviation_returns_zeros(self):
        """deviation_est=None (no storage) → guard returns np.zeros(6) (default n)."""
        sol = self.make_stub_sol(self.NO_STORAGE)
        assert sol.deviation_est is None
        result = sol.map_state_deviation_to_epoch(use_smoothed=False)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_size_zero_deviation_returns_zeros(self):
        """Empty state_deviation_history → deviation_est.size==0 → guard returns zeros."""
        sol = self.make_stub_sol(self.DEV_ONLY, state_deviation_history=[])
        assert sol.deviation_est is not None
        assert sol.deviation_est.size == 0
        result = sol.map_state_deviation_to_epoch(use_smoothed=False)
        np.testing.assert_array_equal(result, np.zeros(6))

    @pytest.mark.parametrize("n_states_attr, expected_len", [
        (4, 4),
        (9, 9),
    ])
    def test_guard_respects_filter_n_states(self, n_states_attr, expected_len):
        """Guard uses filter.n_states when present to size the zeros vector."""
        sol = self.make_stub_sol(self.NO_STORAGE)
        sol.filter.n_states = n_states_attr  # type: ignore[attr-defined]
        result = sol.map_state_deviation_to_epoch(use_smoothed=False)
        assert len(result) == expected_len
        np.testing.assert_array_equal(result, np.zeros(expected_len))


class TestStateCovarianceToElements:
    """
    Tests for reading the estimate into Keplerian elements.

    The transformation itself is covered in ``test_OrbitalElements``; what is
    exercised here is the plumbing between the two — locating a body inside the
    state vector, converting its units, cutting its block out of the
    covariance, and resolving the gravitational parameter. The state and
    covariance are supplied directly rather than fitted, so the elements a
    given state must produce are known independently of any filter.
    """

    MU_EARTH = 398600.4418
    ELEMENTS = (7500.0, 0.02, 0.9, 0.4, 1.1, 2.0)
    EPOCH = 8.0e8

    NO_STORAGE = scb.OutputSettings(
        save_state_deviation_history=True,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def make_solution(self, states_and_covariances):
        """A SolutionOD whose propagation returns the given states and covariances."""

        class StubFilter:
            covariance_analysis = False
            has_consider = False
            state_deviation_history = [np.zeros(6)]

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [TestStateCovarianceToElements.EPOCH]

        solution = scb.SolutionOD(StubFilter(), self.NO_STORAGE)
        solution.propagate_state_covariance = (
            lambda epochs, use_smoothed=True: states_and_covariances
        )
        return solution

    def make_state(self, owners_r_v, origin=None):
        """
        A StateArray holding one position/velocity pair per owner.

        Parameters
        ----------
        owners_r_v : list of (Body, ndarray, ndarray)
            Owner and its position and velocity, in km and km/s.
        origin : CelestialBody, optional
            State origin. Defaults to Earth.
        """
        definition = scb.StateDefinition()
        for owner, r, v in owners_r_v:
            definition = definition.position(
                owner, scb.ArrayWFrame(np.asarray(r, dtype=float), km, J2000)
            ).velocity(
                owner, scb.ArrayWFrame(np.asarray(v, dtype=float), km / sec, J2000)
            )
        epoch = scb.EpochArray(np.array([self.EPOCH]), sys="TDB")
        return scb.StateArray(
            epoch,
            origin if origin is not None else scb.CelestialBody.from_constants("Earth"),
            state=definition,
        )

    def reference_state(self):
        """The position and velocity the class-level elements correspond to."""
        return scb.OrbitalElements.coe2rv(self.MU_EARTH, *self.ELEMENTS)

    def test_elements_match_a_direct_conversion(self, sc):
        """The elements returned are those of the state that was handed in."""
        # Arrange
        r, v = self.reference_state()
        covariance = np.diag([1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10])
        solution = self.make_solution([(self.make_state([(sc, r, v)]), covariance)])

        # Act
        times, elements, covariances = solution.state_covariance_to_elements(
            grav_param=self.MU_EARTH
        )

        # Assert
        assert times.shape == (1,) and elements.shape == (1, 6)
        assert len(covariances) == 1 and covariances[0].shape == (6, 6)
        assert times[0] == pytest.approx(self.EPOCH)
        assert elements[0] == pytest.approx(np.asarray(self.ELEMENTS), abs=1e-9)

        jacobian = scb.OrbitalElements.elements_by_state_jacobian(self.MU_EARTH, r, v)
        assert covariances[0] == pytest.approx(jacobian @ covariance @ jacobian.T)

    def test_grav_param_comes_from_the_origin(self, sc):
        """With no grav_param given, the state's own origin supplies it."""
        # Arrange
        r, v = self.reference_state()
        state = self.make_state([(sc, r, v)])

        # Act
        resolved = scb.SolutionOD._resolve_grav_param(state, None)

        # Assert
        expected = float(
            np.asarray(
                scb.CelestialBody.from_constants("Earth").grav_param.values
            ).ravel()[0]
        )
        assert resolved == pytest.approx(expected)

    def test_explicit_grav_param_overrides_the_origin(self, sc):
        """A gravitational parameter passed in is the one used."""
        # Arrange
        r, v = self.reference_state()
        solution = self.make_solution(
            [(self.make_state([(sc, r, v)]), np.eye(6) * 1e-6)]
        )

        # Act
        _, elements, _ = solution.state_covariance_to_elements(
            grav_param=self.MU_EARTH
        )

        # Assert
        assert elements[0] == pytest.approx(np.asarray(self.ELEMENTS), abs=1e-9)

    def test_units_are_honoured(self, sc):
        """A state carrying metres is converted, not read as kilometres."""
        # Arrange
        r, v = self.reference_state()
        metre, metre_per_sec = scb.Units.get_units("m"), scb.Units.get_units("m") / sec
        definition = (
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(r * 1e3, metre, J2000))
            .velocity(sc, scb.ArrayWFrame(v * 1e3, metre_per_sec, J2000))
        )
        state = scb.StateArray(
            scb.EpochArray(np.array([self.EPOCH]), sys="TDB"),
            scb.CelestialBody.from_constants("Earth"),
            state=definition,
        )
        solution = self.make_solution([(state, np.eye(6) * 1e-6)])

        # Act
        _, elements, _ = solution.state_covariance_to_elements(
            grav_param=self.MU_EARTH
        )

        # Assert
        assert elements[0] == pytest.approx(np.asarray(self.ELEMENTS), abs=1e-8)

    def test_trailing_parameters_are_excluded(self, sc):
        """Estimated parameters after the state do not enter the element covariance."""
        # Arrange
        r, v = self.reference_state()
        cartesian = np.diag([1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10])
        augmented = np.diag(list(np.diag(cartesian)) + [5.0, 6.0])
        state = self.make_state([(sc, r, v)])

        # Act
        _, _, plain = self.make_solution(
            [(state, cartesian)]
        ).state_covariance_to_elements(grav_param=self.MU_EARTH)
        _, _, with_parameters = self.make_solution(
            [(state, augmented)]
        ).state_covariance_to_elements(grav_param=self.MU_EARTH)

        # Assert
        assert plain[0] == pytest.approx(with_parameters[0])

    def test_second_body_needs_naming(self, sc):
        """With two bodies in the state, the one to convert has to be said."""
        # Arrange
        r, v = self.reference_state()
        asteroid = scb.CelestialBody.from_constants("Mars")
        solution = self.make_solution(
            [(self.make_state([(sc, r, v), (asteroid, r * 2, v / 2)]), np.eye(12) * 1e-6)]
        )

        # Act / Assert
        with pytest.raises(scb.SolutionOD.ElementConversionError, match="body"):
            solution.state_covariance_to_elements(grav_param=self.MU_EARTH)

    def test_named_body_selects_its_own_block(self, sc):
        """Naming a body picks its state and its rows of the covariance."""
        # Arrange
        r, v = self.reference_state()
        asteroid = scb.CelestialBody.from_constants("Mars")
        state = self.make_state([(sc, r, v), (asteroid, r * 1.5, v / 1.5)])
        # Distinct scales per block, so a mis-cut covariance cannot pass.
        covariance = np.diag([1e-4] * 3 + [1e-10] * 3 + [7.0] * 3 + [9.0] * 3)
        solution = self.make_solution([(state, covariance)])

        # Act
        _, elements, covariances = solution.state_covariance_to_elements(
            body=asteroid, grav_param=self.MU_EARTH
        )

        # Assert
        expected_elements = scb.OrbitalElements.rv2mean_elements(
            self.MU_EARTH, r * 1.5, v / 1.5
        )
        assert elements[0] == pytest.approx(expected_elements)

        jacobian = scb.OrbitalElements.elements_by_state_jacobian(
            self.MU_EARTH, r * 1.5, v / 1.5
        )
        assert covariances[0] == pytest.approx(
            jacobian @ np.diag([7.0] * 3 + [9.0] * 3) @ jacobian.T
        )

    def test_absent_body_is_reported(self, sc):
        """A body that is not in the state vector is named in the error."""
        # Arrange
        r, v = self.reference_state()
        solution = self.make_solution(
            [(self.make_state([(sc, r, v)]), np.eye(6) * 1e-6)]
        )

        # Act / Assert
        with pytest.raises(scb.SolutionOD.ElementConversionError, match="-99999"):
            solution.state_covariance_to_elements(
                body=-99999, grav_param=self.MU_EARTH
            )

    def test_no_epochs_gives_empty_results(self, sc):
        """A propagation that returns nothing gives empty arrays, not an error."""
        # Arrange
        solution = self.make_solution([])

        # Act
        times, elements, covariances = solution.state_covariance_to_elements()

        # Assert
        assert times.shape == (0,)
        assert elements.shape == (0, 6)
        assert covariances == []


class TestParameterDeviation:
    """
    Tests for picking the parameter rows out of a state deviation.

    The state carries a second propagated body, so the old ``dev[6:]`` would
    also take that body's position and velocity. The deviation is
    ``arange``, so each value names the row it came from.

    Rows: 0-5 spacecraft pos/vel, 6-11 Earth pos/vel, 12 the spacecraft's
    ``eta_srp``, 13 Earth's ``gm``, 14 a ground station's range bias.
    """

    SC = SimpleNamespace(spice_id=-1001, _name="SC")
    EARTH = SimpleNamespace(spice_id=399, _name="Earth")
    STATION = SimpleNamespace(spice_id=399014, _name="DSS14")

    NO_STORAGE = scb.OutputSettings(
        save_state_deviation_history=True,
        save_covariance_history=False,
        save_prefit_residuals=False,
        save_postfit_residuals=False,
        save_consider_covariance_history=False,
        save_smoothed_solution=False,
    )

    def state_definition(self, bias_estimation="estimated"):
        """The multi-body state described in the class docstring."""
        return [
            ("position", 3, "estimated", "dynamic", self.SC, None),
            ("velocity", 3, "estimated", "dynamic", self.SC, None),
            ("position", 3, "estimated", "dynamic", self.EARTH, None),
            ("velocity", 3, "estimated", "dynamic", self.EARTH, None),
            ("eta_srp", 1, "estimated", "static", self.SC, None),
            ("gm", 1, "estimated", "static", self.EARTH, None),
            ("range_bias", 1, bias_estimation, "static", self.STATION, None),
        ]

    def test_primary_sidecar_skips_second_body_states(self):
        """The primary's rows are its own parameters plus the unpropagated owner's."""
        dev = np.arange(15.0)
        rows = scb.SolutionOD._parameter_deviation(dev, self.state_definition(), -1001)
        np.testing.assert_array_equal(rows, [12.0, 14.0])

    def test_no_body_takes_every_parameter(self):
        """Without a body (a sequence leg) every parameter is kept, no pos/vel."""
        dev = np.arange(15.0)
        rows = scb.SolutionOD._parameter_deviation(dev, self.state_definition())
        np.testing.assert_array_equal(rows, [12.0, 13.0, 14.0])

    def test_padded_row_uses_the_full_layout(self):
        """Trailing padding from a wider leg is ignored."""
        dev = np.arange(17.0)
        rows = scb.SolutionOD._parameter_deviation(dev, self.state_definition(), -1001)
        np.testing.assert_array_equal(rows, [12.0, 14.0])

    def test_estimated_partition_zero_fills_considered(self):
        """A deviation without the considered rows gives them zero deviation."""
        dev = np.arange(14.0)
        rows = scb.SolutionOD._parameter_deviation(
            dev, self.state_definition(bias_estimation="considered"), -1001
        )
        np.testing.assert_array_equal(rows, [12.0, 0.0])

    def test_missing_definition_falls_back(self):
        """No state definition keeps the old single-body rule."""
        dev = np.arange(9.0)
        rows = scb.SolutionOD._parameter_deviation(dev, None)
        np.testing.assert_array_equal(rows, [6.0, 7.0, 8.0])

    def test_state_est_adds_matching_rows(self):
        """state_est adds each nominal parameter its own deviation, not Earth's states."""
        # Arrange
        definition = self.state_definition()
        nominal = {
            "position": SimpleNamespace(values=np.array([7000.0, 0.0, 0.0])),
            "velocity": SimpleNamespace(values=np.array([0.0, 7.5, 0.0])),
            "parameters": SimpleNamespace(values=np.array([1.0, 2.0e-3])),
        }

        class StubTrajectory:
            state_definition = definition
            body = TestParameterDeviation.SC

            @staticmethod
            def get_state(epoch_input=None, idx_leg_input=None):
                return nominal

        class StubFilter:
            covariance_analysis = False
            has_consider = False
            flag_sequence = False
            trajectory = StubTrajectory()
            state_deviation_history = [np.arange(15.0)]

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [8.0e8]

        # Act
        solution = scb.SolutionOD(StubFilter(), self.NO_STORAGE)
        row = solution.state_est[0]

        # Assert
        np.testing.assert_array_equal(row[:6], [7000.0, 1.0, 2.0, 3.0, 11.5, 5.0])
        np.testing.assert_array_equal(row[6:8], [1.0 + 12.0, 2.0e-3 + 14.0])

    def test_sequence_layout_comes_from_the_leg(self):
        """A sequence reads the leg's definition, found from the epoch if needed."""
        # Arrange
        legs = [SimpleNamespace(state="leg 0"), SimpleNamespace(state="leg 1")]

        class StubFilter:
            covariance_analysis = False
            has_consider = False
            flag_sequence = True
            legs_model = legs
            legs_epochs = [np.array([0.0, 10.0]), np.array([10.0, 20.0])]
            state_deviation_history = [np.zeros(6)]

            class measurement_data:
                @staticmethod
                def get_spacecraft_times():
                    return [0.0]

        solution = scb.SolutionOD(StubFilter(), self.NO_STORAGE)

        # Act / Assert
        assert solution._parameter_layout(0) == ("leg 0", None)
        assert solution._parameter_layout(t=15.0) == ("leg 1", None)
        assert solution._parameter_layout(5) == (None, None)


# endregion Tests #
