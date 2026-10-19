# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Functional Tests for Orbit Determination   #
# =============================================#

import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import json
import pytest
import numpy as np
from scipy.stats import chi2 as chi2_dist

from typing import TypedDict
import spiceypy as spice

# --------------#
# region Setup #
# --------------#
km, AU, kg, sec, rad = scb.Units.get_units(["km", "AU", "kg", "sec", "rad"])
J2000 = scb.Frame("J2000")

# -----------------#
# region Fixtures #
# -----------------#
# shared data type from trajectory fixtures -> makes typehints work
TrajData = TypedDict(
    "TrajData",
    {
        "x0": scb.StateArray,
        "pos_0": scb.ArrayWFrame,
        "vel_0": scb.ArrayWFrame,
        "trajectory": scb.StateArray,
        "traj_obj": scb.Trajectory,
        "T": awu,
        "ephemeris": str,
        "force_kwargs": dict,
    },
)


@pytest.fixture(scope="module",autouse=True)
def traj_keplerian(sc, epochs, tmp_path_factory) -> TrajData:
    """
    Propagated Keplerian trajectory information for testing.
    """
    earth = scb.CelestialBody.from_constants("EARTH")

    # initial state
    a = awu(1, AU).convert_to(km).values
    mu = earth.grav_param.values
    e, i, RAAN = 0.46, np.deg2rad(2), np.deg2rad(0.0)

    v = np.sqrt((mu / a) * ((1 - e) / (1 + e)))
    x0_vals = [
        -a * (1 + e) * np.cos(RAAN),
        -a * (1 + e) * np.sin(RAAN),
        0,
        v * np.sin(RAAN) * np.cos(i),
        -v * np.cos(RAAN) * np.cos(i),
        -v * np.sin(i),
    ]

    pos_0 = scb.ArrayWFrame(x0_vals[0:3], km, J2000)
    vel_0 = scb.ArrayWFrame(x0_vals[3:], km / sec, J2000)

    x0 = scb.StateArray(
        epoch=epochs[0],
        origin=earth,
        state=scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0),
    )

    # propagate
    force_kwargs = {}
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=x0,
        tspan=epochs,
        force_models=scb.ForceModelTranslation(sc, **force_kwargs),
    )
    prop.propagate()
    states = prop.propagated_state_array

    # save to trajectory
    bsp_path = str(tmp_path_factory.mktemp("od_keplerian") / "test_OD.bsp")
    orbiter_traj = scb.Trajectory(state_array=states)
    orbiter_traj.write_to_spk(bsp_path)
    # spice.furnsh(bsp_path)

    # get orbital period
    T = awu(2 * np.pi * np.sqrt(a**3 / mu), sec)

    # save off to dict and return
    traj_data: TrajData = {
        "x0": x0,
        "pos_0": pos_0,
        "vel_0": vel_0,
        "trajectory": states,
        "traj_obj": orbiter_traj,
        "T": T,
        "ephemeris": bsp_path,
        "force_kwargs": force_kwargs,
    }
    return traj_data


@pytest.fixture(scope="module", autouse= True)
def traj_full_dyn(sc, epochs, tmp_path_factory) -> TrajData:
    """
    Propagated full-dynamics trajectory with SRP and third-body perturbations.
    """
    earth = scb.CelestialBody.from_constants("EARTH")

    # initial state
    a = awu(1, AU).convert_to(km).values
    mu = earth.grav_param.values
    e, i, RAAN = 0.46, np.deg2rad(2), np.deg2rad(0.0)

    v = np.sqrt((mu / a) * ((1 - e) / (1 + e)))
    x0_vals = [
        -a * (1 + e) * np.cos(RAAN),
        -a * (1 + e) * np.sin(RAAN),
        0,
        v * np.sin(RAAN) * np.cos(i),
        -v * np.cos(RAAN) * np.cos(i),
        -v * np.sin(i),
    ]

    pos_0 = scb.ArrayWFrame(x0_vals[0:3], km, J2000)
    vel_0 = scb.ArrayWFrame(x0_vals[3:], km / sec, J2000)

    x0 = scb.StateArray(
        epoch=epochs[0],
        origin=earth,
        state=scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0),
    )

    # propagate with SRP and third-body perturbations
    third_bodies = scb.CelestialBody.from_constants(["MERCURY", "VENUS", "SUN"])
    force_kwargs = {"cannonball_SRP": True, "third_bodies": third_bodies}
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=x0,
        tspan=epochs,
        force_models=scb.ForceModelTranslation(sc, **force_kwargs),
    )
    prop.propagate()
    states = prop.propagated_state_array

    # save to trajectory
    bsp_path = str(tmp_path_factory.mktemp("od_full_dyn") / "test_OD_full_dyn.bsp")
    orbiter_traj = scb.Trajectory(state_array=states)
    orbiter_traj.write_to_spk(bsp_path)
    # spice.furnsh(bsp_path)

    # get orbital period
    T = awu(2 * np.pi * np.sqrt(a**3 / mu), sec)

    traj_data: TrajData = {
        "x0": x0,
        "pos_0": pos_0,
        "vel_0": vel_0,
        "trajectory": states,
        "traj_obj": orbiter_traj,
        "T": T,
        "ephemeris": bsp_path,
        "force_kwargs": force_kwargs,
    }
    return traj_data


@pytest.fixture(
    scope="module",
    ids=["Keplerian", "Full Dynamics"],
    params=["traj_keplerian", "traj_full_dyn"],
)
def active_traj(request) -> TrajData:
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def range_ideal(GS1):
    # Initialize measurement uncertainty
    range_sigma = scb.ArrayWUnits(1e-3, km)
    rangerate_sigma = scb.ArrayWUnits(1e-5, km / sec)

    # (2) Initialize Measurement Models
    Range_GS1 = scb.RangeIdeal("GS1 Ideal Range Model", GS1, sigma=range_sigma)
    RangeRate_GS1 = scb.RangeRateIdeal(
        "GS1 Ideal Range Rate Model", GS1, sigma=rangerate_sigma
    )
    return Range_GS1, RangeRate_GS1


@pytest.fixture(scope="module")
def gen_clean_meas_from_traj(active_traj, range_ideal, sc, epochs):
    """
    Generate observed measurements from a set of given trajectory
    data.
    """
    # need range and rangerate info from this
    # this should probably be a factory fixture to take inputs from multiple
    # kinds of trajectories
    # (3) Compute Measurements (on true trajectory)
    spice.furnsh(active_traj["ephemeris"])

    Range_GS1, RangeRate_GS1 = range_ideal
    Range_GS1.write_observed_measurements(
        target=sc,
        epoch_array=epochs,
        frame=J2000,
        noisy=False,
        file_name="ideal_range",
    )

    RangeRate_GS1.write_observed_measurements(
        target=sc,
        epoch_array=epochs,
        frame=J2000,
        noisy=False,
        file_name="ideal_range_rate",
    )
    obs_quantities_range = Range_GS1.observed_measurements(
        file_name="data/measurements/radiometric/ideal_range.json",
        meas_name="meas_ideal",
        units=km,
    )

    obs_quantities_rangerate = RangeRate_GS1.observed_measurements(
        file_name="data/measurements/radiometric/ideal_range_rate.json",
        meas_name="meas_ideal",
        units=km / sec,
    )

    measurements_list = scb.MeasurementSpec.many(
        scb.MeasurementSpec(
            model=Range_GS1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",
        ),
        scb.MeasurementSpec(
            model=RangeRate_GS1,
            observed_meas=obs_quantities_rangerate,
            dataset_name="GS1 Range Rate",
        ),
    )
    return measurements_list


@pytest.fixture(scope="module")
def gen_noisy_meas_from_traj(active_traj, range_ideal, sc, epochs):
    """
    Generate observed measurements from a set of given trajectory
    data.
    """
    # need range and rangerate info from this
    # this should probably be a factory fixture to take inputs from multiple
    # kinds of trajectorie
    # (3) Compute Measurements (on true trajectory)
    spice.furnsh(active_traj["ephemeris"])

    Range_GS1, RangeRate_GS1 = range_ideal
    Range_GS1.write_observed_measurements(
        target=sc,
        epoch_array=epochs,
        frame=J2000,
        noisy=True,
        file_name="ideal_range",
    )

    RangeRate_GS1.write_observed_measurements(
        target=sc,
        epoch_array=epochs,
        frame=J2000,
        noisy=True,
        file_name="ideal_range_rate",
    )
    obs_quantities_range = Range_GS1.observed_measurements(
        file_name="data/measurements/radiometric/ideal_range.json",
        meas_name="meas_ideal",
        units=km,
    )

    obs_quantities_rangerate = RangeRate_GS1.observed_measurements(
        file_name="data/measurements/radiometric/ideal_range_rate.json",
        meas_name="meas_ideal",
        units=km / sec,
    )

    measurements_list = scb.MeasurementSpec.many(
        scb.MeasurementSpec(
            model=Range_GS1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range",
        ),
        scb.MeasurementSpec(
            model=RangeRate_GS1,
            observed_meas=obs_quantities_rangerate,
            dataset_name="GS1 Range Rate",
        ),
    )
    return measurements_list


# endregion Fixtures #
# -----------------#


# --------------#
# region Tests #
# --------------#


class FilterTests:
    """Base class with shared fixtures and helpers for all filter test classes."""

    @pytest.fixture(scope="class")
    def filter_class(self):
        raise NotImplementedError("Subclass must implement filter_class fixture")

    @pytest.fixture(scope="class")
    def filter_settings(self, epochs):
        pos_sigma = scb.ArrayWUnits(5, km)
        vel_sigma = scb.ArrayWUnits(5e-4, km * sec**-1)
        state_sigma_list = [
            pos_sigma,
            pos_sigma,
            pos_sigma,
            vel_sigma,
            vel_sigma,
            vel_sigma,
        ]
        covar_mat = scb.CovarianceMatrix(state_sigma_list, epochs[1], from_list=True)
        return scb.FilterSettings(initial_covariance=covar_mat)

    @pytest.fixture(scope="class")
    def filter_solution(
        self,
        active_traj,
        filter_class,
        filter_settings,
        gen_clean_meas_from_traj,
        gen_noisy_meas_from_traj,
        epochs,
        origin,
        tmp_path_factory,
    ):
        """Run the filter with clean and noisy measurements, return both solutions."""
        spice.furnsh(active_traj["ephemeris"])
        traj = active_traj
        filter_name = filter_class.__name__
        # Every iteration writes a reference SPK; keep them out of the repo
        traj_dir = str(tmp_path_factory.mktemp(f"od_{filter_name}"))

        # Reference spacecraft needs a different SPICE ID to avoid kernel conflicts
        sc_ref = scb.Spacecraft(
            f"{filter_name}_ref",
            -1001,
            scb.ArrayWUnits(2000.0, kg),
            scb.ArrayWUnits(1e-06, km**2),
            scb.ArrayWUnits(1.5, None),
        )

        # Small perturbation from truth initial state
        delta_pos = scb.ArrayWUnits(np.array([1.0, 1.0, 1.0]), km)
        delta_vel = scb.ArrayWUnits(np.array([1e-3, 1e-3, 1e-3]), km / sec)
        pos_pert = scb.ArrayWFrame(traj["pos_0"].quantity + delta_pos, J2000)
        vel_pert = scb.ArrayWFrame(traj["vel_0"].quantity + delta_vel, J2000)

        state_pert = scb.StateArray(
            epoch=epochs[0],
            origin=origin,
            state=scb.StateDefinition()
            .position(sc_ref, pos_pert)
            .velocity(sc_ref, vel_pert),
        )

        def make_propagator():
            return scb.Propagator(
                primary_body=sc_ref,
                state_vector=state_pert,
                tspan=epochs,
                # Same dynamics as the truth: a two-body filter on the
                # Sun-perturbed arc misses by ~1000 km and fails NEES/NIS by design
                force_models=scb.ForceModelTranslation(sc_ref, **traj["force_kwargs"]),
            )

        clean_filter = filter_class(
            propagator=make_propagator(),
            measurements=gen_clean_meas_from_traj,
            settings=filter_settings,
            traj_name=f"{filter_name}_clean.bsp",
            traj_dir=traj_dir,
        )
        noisy_filter = filter_class(
            propagator=make_propagator(),
            measurements=gen_noisy_meas_from_traj,
            settings=filter_settings,
            traj_name=f"{filter_name}_noisy.bsp",
            traj_dir=traj_dir,
        )
        iterate_kwargs = {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "verbose": False,
        }
        return clean_filter.fit(**iterate_kwargs), noisy_filter.fit(
            **iterate_kwargs
        )

    @staticmethod
    def compute_NEES_stat(sol, truth_traj, epochs, n_state=6, n_samples=10):
        """Compute NEES statistic over sampled epochs, returning (sum_nees, dof)."""
        est_pos, est_vel, _ = sol.estimated_trajectory(epochs)

        n_check = min(len(sol.covariance_est), epochs.size)
        indices = np.linspace(0, n_check - 1, n_samples, dtype=int)

        nees_vals = []
        for k in indices:
            true_state = truth_traj.get_state(epoch_input=epochs[k])
            true_pos = true_state["position"].values
            true_vel = true_state["velocity"].values

            error = np.concatenate([true_pos - est_pos[k], true_vel - est_vel[k]])
            P = sol.covariance_est[k][:n_state, :n_state]
            nees_vals.append(float(error @ np.linalg.inv(P) @ error.T))

        dof = len(nees_vals) * n_state
        return float(np.sum(nees_vals)), dof

    @staticmethod
    def compute_NIS_stat(sol):
        """Compute NIS = sum_i (r_i / sigma_i)^2 at each measurement epoch, returning an array."""
        datasets = list(sol.postfits.keys())
        n_epochs = len(sol.postfits[datasets[0]])

        nis_vals = []
        for k in range(n_epochs):
            nis_k = sum(
                (sol.postfits[ds][k][0] / sol.postfits[ds][k][1]) ** 2
                for ds in datasets
            )
            nis_vals.append(nis_k)

        return np.array(nis_vals)


class TestLKF(FilterTests):
    """
    LKF statistical and residual tests.
    """

    @pytest.fixture(scope="class")
    def filter_class(self):
        return scb.LKF

    def test_state_errors(self, filter_solution, traj_keplerian, epochs):
        """Ensure LKF converges and estimated positions stay within 3-sigma of truth."""
        
        (clean_sol, _, clean_converged), (noisy_sol, _, _) = filter_solution
        assert clean_converged, "LKF did not converge on clean measurements"

        truth_traj = traj_keplerian["traj_obj"]

        for sol in (clean_sol, noisy_sol):
            est_pos, _, _ = sol.estimated_trajectory(epochs)
            for k, cov in [(0, sol.covariance_est[0]), (-1, sol.covariance_est[-1])]:
                true_pos = truth_traj.get_state(epoch_input=epochs[k])[
                    "position"
                ].values
                pos_err = np.linalg.norm(est_pos[k] - true_pos)
                pos_3sigma = 3 * np.sqrt(np.trace(cov[:3, :3]))
                assert (
                    pos_err < pos_3sigma
                ), f"Position error {pos_err:.3f} km exceeds 3-sigma {pos_3sigma:.3f} km at epoch {k}"

    def test_NEES(self, filter_solution, traj_keplerian, epochs):
        """NEES sum must fall within chi-squared 90% confidence bounds."""
        (clean_sol, _, clean_converged), _ = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NEES not meaningful")

        truth_traj = traj_keplerian["traj_obj"]
        sum_nees, dof = self.compute_NEES_stat(clean_sol, truth_traj, epochs)

        alpha = 0.015
        lower = chi2_dist.ppf(alpha / 2, dof)
        upper = chi2_dist.ppf(1 - (alpha / 2), dof)

        assert (
            sum_nees < upper
        ), f"NEES sum {sum_nees:.2f} > chi(0.95, {dof})={upper:.2f}: LKF overconfident"
        assert (
            sum_nees > lower
        ), f"NEES sum {sum_nees:.2f} < chi(0.05, {dof})={lower:.2f}: LKF underconfident"

    def test_NIS(self, filter_solution):
        """NIS mean must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.4
        r1 = chi2_dist.ppf(alpha / 2, N * ny) / N
        r2 = chi2_dist.ppf(1 - alpha / 2, N * ny) / N
        mean_nis = np.mean(nis_vals)

        assert (
            mean_nis > r1
        ), f"Mean NIS {mean_nis:.2f} < {r1:.2f}: filter measurement model underconfident"
        assert (
            mean_nis < r2
        ), f"Mean NIS {mean_nis:.2f} > {r2:.2f}: filter measurement model overconfident"

    def test_pre_and_postfits(self, filter_solution):
        """Postfit RMS must be smaller than prefit RMS for each dataset."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; residuals not meaningful")

        for ds in noisy_sol.prefits:
            pre = np.array([r for r, _ in noisy_sol.prefits[ds]])
            post = np.array([r for r, _ in noisy_sol.postfits[ds]])
            prefit_rms = np.sqrt(np.mean(pre**2))
            postfit_rms = np.sqrt(np.mean(post**2))
            assert (
                postfit_rms < prefit_rms
            ), f"{ds}: postfit RMS {postfit_rms:.4e} >= prefit RMS {prefit_rms:.4e}"


class TestLSB(FilterTests):
    """
    LSB statistical and residual tests.
    """

    @pytest.fixture(scope="class")
    def filter_class(self):
        return scb.LSB

    def test_state_errors(self, filter_solution, traj_keplerian, epochs):
        """Ensure LSB converges and estimated positions stay within 3-sigma of truth."""
        (clean_sol, _, clean_converged), (noisy_sol, _, _) = filter_solution

        assert clean_converged, "LSB did not converge on clean measurements"

        truth_traj = traj_keplerian["traj_obj"]

        for sol in (clean_sol, noisy_sol):
            est_pos, _, _ = sol.estimated_trajectory(epochs)
            for k, cov in [(0, sol.covariance_est[0]), (-1, sol.covariance_est[-1])]:
                true_pos = truth_traj.get_state(epoch_input=epochs[k])[
                    "position"
                ].values
                pos_err = np.linalg.norm(est_pos[k] - true_pos)
                pos_3sigma = 3 * np.sqrt(np.trace(cov[:3, :3]))
                assert (
                    pos_err < pos_3sigma
                ), f"Position error {pos_err:.3f} km exceeds 3-sigma {pos_3sigma:.3f} km at epoch {k}"

    def test_NEES(self, filter_solution, traj_keplerian, epochs):
        """NEES sum must fall within chi-squared confidence bounds."""
        (clean_sol, _, clean_converged), _ = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NEES not meaningful")

        truth_traj = traj_keplerian["traj_obj"]
        sum_nees, dof = self.compute_NEES_stat(clean_sol, truth_traj, epochs)

        alpha = 0.015
        lower = chi2_dist.ppf(alpha / 2, dof)
        upper = chi2_dist.ppf(1 - (alpha / 2), dof)

        assert (
            sum_nees < upper
        ), f"NEES sum {sum_nees:.2f} > chi(0.95, {dof})={upper:.2f}: LSB overconfident"
        assert (
            sum_nees > lower
        ), f"NEES sum {sum_nees:.2f} < chi(0.05, {dof})={lower:.2f}: LSB underconfident"

    def test_NIS(self, filter_solution):
        """NIS mean must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.4
        r1 = chi2_dist.ppf(alpha / 2, N * ny) / N
        r2 = chi2_dist.ppf(1 - alpha / 2, N * ny) / N
        mean_nis = np.mean(nis_vals)

        assert (
            mean_nis > r1
        ), f"Mean NIS {mean_nis:.2f} < {r1:.2f}: filter measurement model underconfident"
        assert (
            mean_nis < r2
        ), f"Mean NIS {mean_nis:.2f} > {r2:.2f}: filter measurement model overconfident"


class TestSRIF(FilterTests):
    """
    SRIF statistical and residual tests.
    """

    @pytest.fixture(scope="class")
    def filter_class(self):
        return scb.SRIF

    def test_state_errors(self, filter_solution, traj_keplerian, epochs):
        """Ensure SRIF converges and estimated positions stay within 3-sigma of truth."""
        (clean_sol, _, clean_converged), (noisy_sol, _, _) = filter_solution

        assert clean_converged, "SRIF did not converge on clean measurements"

        truth_traj = traj_keplerian["traj_obj"]

        for sol in (clean_sol, noisy_sol):
            est_pos, _, _ = sol.estimated_trajectory(epochs)
            for k, cov in [(0, sol.covariance_est[0]), (-1, sol.covariance_est[-1])]:
                true_pos = truth_traj.get_state(epoch_input=epochs[k])[
                    "position"
                ].values
                pos_err = np.linalg.norm(est_pos[k] - true_pos)
                pos_3sigma = 3 * np.sqrt(np.trace(cov[:3, :3]))
                assert (
                    pos_err < pos_3sigma
                ), f"Position error {pos_err:.3f} km exceeds 3-sigma {pos_3sigma:.3f} km at epoch {k}"

    def test_NEES(self, filter_solution, traj_keplerian, epochs):
        """NEES sum must fall within chi-squared confidence bounds."""
        (clean_sol, _, clean_converged), _ = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NEES not meaningful")

        truth_traj = traj_keplerian["traj_obj"]
        sum_nees, dof = self.compute_NEES_stat(clean_sol, truth_traj, epochs)

        alpha = 0.015
        lower = chi2_dist.ppf(alpha / 2, dof)
        upper = chi2_dist.ppf(1 - (alpha / 2), dof)

        assert (
            sum_nees < upper
        ), f"NEES sum {sum_nees:.2f} > chi(0.95, {dof})={upper:.2f}: SRIF overconfident"
        assert (
            sum_nees > lower
        ), f"NEES sum {sum_nees:.2f} < chi(0.05, {dof})={lower:.2f}: SRIF underconfident"

    def test_NIS(self, filter_solution):
        """NIS mean must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.4
        r1 = chi2_dist.ppf(alpha / 2, N * ny) / N
        r2 = chi2_dist.ppf(1 - alpha / 2, N * ny) / N
        mean_nis = np.mean(nis_vals)

        assert (
            mean_nis > r1
        ), f"Mean NIS {mean_nis:.2f} < {r1:.2f}: filter measurement model underconfident"
        assert (
            mean_nis < r2
        ), f"Mean NIS {mean_nis:.2f} > {r2:.2f}: filter measurement model overconfident"

    def test_pre_and_postfits(self, filter_solution):
        """Postfit RMS must be smaller than prefit RMS for each dataset."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; residuals not meaningful")

        for ds in noisy_sol.prefits:
            pre = np.array([r for r, _ in noisy_sol.prefits[ds]])
            post = np.array([r for r, _ in noisy_sol.postfits[ds]])
            prefit_rms = np.sqrt(np.mean(pre**2))
            postfit_rms = np.sqrt(np.mean(post**2))
            assert (
                postfit_rms < prefit_rms
            ), f"{ds}: postfit RMS {postfit_rms:.4e} >= prefit RMS {prefit_rms:.4e}"


class TestSRIFB(FilterTests):
    """
    SRIFB statistical and residual tests.
    """

    @pytest.fixture(scope="class")
    def filter_class(self):
        return scb.SRIFB

    def test_state_errors(self, filter_solution, traj_keplerian, epochs):
        """Ensure SRIFB converges and estimated positions stay within 3-sigma of truth."""
        (clean_sol, _, clean_converged), (noisy_sol, _, _) = filter_solution

        assert clean_converged, "SRIFB did not converge on clean measurements"

        truth_traj = traj_keplerian["traj_obj"]

        for sol in (clean_sol, noisy_sol):
            est_pos, _, _ = sol.estimated_trajectory(epochs)
            for k, cov in [(0, sol.covariance_est[0]), (-1, sol.covariance_est[-1])]:
                true_pos = truth_traj.get_state(epoch_input=epochs[k])[
                    "position"
                ].values
                pos_err = np.linalg.norm(est_pos[k] - true_pos)
                pos_3sigma = 3 * np.sqrt(np.trace(cov[:3, :3]))
                assert (
                    pos_err < pos_3sigma
                ), f"Position error {pos_err:.3f} km exceeds 3-sigma {pos_3sigma:.3f} km at epoch {k}"

    def test_NEES(self, filter_solution, traj_keplerian, epochs):
        """NEES sum must fall within chi-squared confidence bounds."""
        (clean_sol, _, clean_converged), _ = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NEES not meaningful")

        truth_traj = traj_keplerian["traj_obj"]
        sum_nees, dof = self.compute_NEES_stat(clean_sol, truth_traj, epochs)

        alpha = 0.015
        lower = chi2_dist.ppf(alpha / 2, dof)
        upper = chi2_dist.ppf(1 - (alpha / 2), dof)

        assert (
            sum_nees < upper
        ), f"NEES sum {sum_nees:.2f} > chi(0.95, {dof})={upper:.2f}: SRIFB overconfident"
        assert (
            sum_nees > lower
        ), f"NEES sum {sum_nees:.2f} < chi(0.05, {dof})={lower:.2f}: SRIFB underconfident"

    def test_NIS(self, filter_solution):
        """NIS sum must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.4
        r1 = chi2_dist.ppf(alpha / 2, N * ny) / N
        r2 = chi2_dist.ppf(1 - alpha / 2, N * ny) / N
        mean_nis = np.mean(nis_vals)

        assert (
            mean_nis > r1
        ), f"Mean NIS {mean_nis:.2f} < {r1:.2f}: filter measurement model underconfident"
        assert (
            mean_nis < r2
        ), f"Mean NIS {mean_nis:.2f} > {r2:.2f}: filter measurement model overconfident"

    def test_pre_and_postfits(self, filter_solution):
        """Postfit RMS must be smaller than prefit RMS for each dataset."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; residuals not meaningful")

        for ds in noisy_sol.prefits:
            pre = np.array([r for r, _ in noisy_sol.prefits[ds]])
            post = np.array([r for r, _ in noisy_sol.postfits[ds]])
            prefit_rms = np.sqrt(np.mean(pre**2))
            postfit_rms = np.sqrt(np.mean(post**2))
            assert (
                postfit_rms < prefit_rms
            ), f"{ds}: postfit RMS {postfit_rms:.4e} >= prefit RMS {prefit_rms:.4e}"


class TestSolutionOD(FilterTests):
    """
    SolutionOD properties and methods, backed by a converged LKF solution.
    """

    @pytest.fixture(scope="class")
    def filter_class(self):
        return scb.LKF

    @pytest.fixture(scope="class")
    def lkf_sol(self, filter_solution):
        (clean_sol, _, clean_converged), _ = filter_solution
        if not clean_converged:
            pytest.skip("LKF did not converge; SolutionOD tests skipped")
        return clean_sol

    def test_solution_properties(self, lkf_sol):
        """n_epochs, n_states, and is_covariance_analysis are set correctly."""
        assert lkf_sol.n_epochs > 0
        assert lkf_sol.n_states > 0
        assert lkf_sol.is_covariance_analysis is False

    def test_state_est(self, lkf_sol):
        """Lazily-computed state_est has shape (n_epochs, n_states)."""
        state = lkf_sol.state_est
        assert state is not None
        assert state.ndim == 2
        assert state.shape == (lkf_sol.n_epochs, lkf_sol.n_states)

    def test_global_std_devs_raises(self, lkf_sol):
        """global_std_devs raises ValueError for single-arc solutions."""
        with pytest.raises(ValueError, match="multi_arc_global"):
            lkf_sol.global_std_devs()

    def test_print_stats(self, lkf_sol):
        """print_stats completes without raising an exception."""
        lkf_sol.print_stats(verbose=False)

    def test_propagate_covariance(self, lkf_sol):
        """propagate_covariance returns a list of (n_states x n_states) matrices."""
        t_vals = np.asarray(lkf_sol.timestamps, dtype=float)[:2]
        test_epochs = scb.EpochArray(t_vals, sys="TDB")
        covs = lkf_sol.propagate_covariance(test_epochs)
        assert isinstance(covs, list)
        assert len(covs) > 0
        n = lkf_sol.n_states
        for cov in covs:
            assert np.asarray(cov).shape == (n, n)

    @pytest.mark.parametrize("use_smoothed", [True, False])
    def test_propagate_state_returns_state_arrays(self, lkf_sol, use_smoothed):
        """propagate_state returns one StateArray per epoch with finite pos/vel."""
        t_vals = np.asarray(lkf_sol.timestamps, dtype=float)[:3]
        test_epochs = scb.EpochArray(t_vals, sys="TDB")
        states = lkf_sol.propagate_state(test_epochs, use_smoothed=use_smoothed)
        assert len(states) == len(t_vals)
        for sa in states:
            assert isinstance(sa, scb.StateArray)
            assert sa.size > 0
            for name, _, _, _, _, awf in sa.state:
                if name in ("position", "velocity"):
                    vals = np.asarray(awf.quantity.values).ravel()
                    assert np.all(np.isfinite(vals)), f"{name} contains non-finite values"

    def test_propagate_state_outside_arc(self, lkf_sol):
        """Epoch beyond arc end is propagated and returns a valid StateArray."""
        t_end = float(np.asarray(lkf_sol.timestamps, dtype=float)[-1])
        dt = 60.0
        beyond = scb.EpochArray(np.array([t_end + dt]), sys="TDB")
        states = lkf_sol.propagate_state(beyond)
        assert len(states) == 1
        assert isinstance(states[0], scb.StateArray)
        assert states[0].size > 0

    def test_propagate_state_empty_epochs(self, lkf_sol):
        """Empty EpochArray returns an empty list without error."""
        empty = scb.EpochArray(np.array([]), sys="TDB")
        states = lkf_sol.propagate_state(empty)
        assert states == []

    def test_map_state_deviation_to_epoch(self, lkf_sol):
        """Non-sequence path: inverse-STM mapping returns a finite (n_states,) array."""
        mapped = lkf_sol.map_state_deviation_to_epoch(use_smoothed=False)
        assert isinstance(mapped, np.ndarray)
        assert mapped.shape == (lkf_sol.n_states,)
        assert np.all(np.isfinite(mapped))

    def test_create_JSON(self, lkf_sol, tmp_path):
        """create_JSON writes a file containing the expected metadata."""
        out_file = tmp_path / "solution.json"
        lkf_sol.create_JSON(out_file)
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "metadata" in data
        assert data["metadata"]["n_epochs"] == lkf_sol.n_epochs
        assert data["metadata"]["n_states"] == lkf_sol.n_states
        assert "is_covariance_analysis" in data["metadata"]

    def test_serialize_deserialize(self, lkf_sol, tmp_path):
        """serialize/deserialize round-trip preserves timestamps."""
        pkl_file = tmp_path / "solution.pkl"
        lkf_sol.serialize(pkl_file)
        assert pkl_file.exists()
        recovered = scb.SolutionOD.deserialize(pkl_file)
        np.testing.assert_array_equal(recovered.timestamps, lkf_sol.timestamps)


# class TestMultiFilterOD:
#     """
#     Multi-arc OD functional tests: two LKF arcs with velocity as the shared
#     global parameter, run on the Keplerian truth trajectory.
#     """

#     @pytest.fixture(scope="class")
#     def gs1(self, request):
#         return request.getfixturevalue("GS1")

#     @pytest.fixture(scope="class")
#     def arc_solutions(self, traj_keplerian, epochs, origin, sc, gs1):
#         """Build two LKF arc filters and run MultiFilterOD.fit()."""
#         spice.furnsh(traj_keplerian["ephemeris"])

#         range_sigma = scb.ArrayWUnits(1e-3, km)
#         range_model = scb.RangeIdeal("MultiArc GS1 Range", gs1, sigma=range_sigma)
#         range_model.write_observed_measurements(
#             target=sc, epoch_array=epochs, frame=J2000,
#             noisy=False, file_name="multi_arc_range",
#         )
#         obs = range_model.observed_measurements(
#             file_name="data/measurements/radiometric/multi_arc_range.json",
#             meas_name="meas_ideal", units=km,
#         )
#         measurements = scb.MeasurementSpec.many(
#             scb.MeasurementSpec(
#                 model=range_model, observed_meas=obs, dataset_name="GS1 Range"
#             ),
#         )

#         delta_pos = scb.ArrayWUnits(np.array([1.0, 1.0, 1.0]), km)
#         delta_vel = scb.ArrayWUnits(np.array([1e-3, 1e-3, 1e-3]), km * sec**-1)

#         def make_arc(name, spice_id):
#             sc_ref = scb.Spacecraft(
#                 name, spice_id,
#                 scb.ArrayWUnits(2000.0, kg),
#                 scb.ArrayWUnits(1e-6, km**2),
#                 scb.ArrayWUnits(1.5, None),
#             )
#             pos = scb.ArrayWFrame(traj_keplerian["pos_0"].quantity + delta_pos, J2000)
#             vel = scb.ArrayWFrame(traj_keplerian["vel_0"].quantity + delta_vel, J2000)
#             x0 = scb.StateArray(
#                 epochs[0], origin,
#                 state=scb.StateDefinition().position(sc_ref, pos).velocity(sc_ref, vel),
#             )
#             prop = scb.Propagator(
#                 primary_body=sc_ref, state_vector=x0, tspan=epochs,
#                 force_models=scb.ForceModelTranslation(sc_ref),
#             )
#             covar = scb.CovarianceMatrix(
#                 [scb.ArrayWUnits(5, km)] * 3 + [scb.ArrayWUnits(5e-4, km * sec**-1)] * 3,
#                 epochs[1], from_list=True,
#             )
#             settings = scb.FilterSettings(initial_covariance=covar)
#             return scb.LKF(
#                 propagator=prop, measurements=measurements,
#                 settings=settings, traj_name=f"{name}.bsp",
#             )

#         arc1 = make_arc("multi_arc1", -2001)
#         arc2 = make_arc("multi_arc2", -2002)

#         multi = scb.MultiFilterOD(
#             filters=[arc1, arc2],
#             global_param_names=["velocity"],
#         )
#         solutions = multi.fit(max_outer_iters=2, verbose=False)
#         return solutions, multi

#     def test_returns_one_solution_per_arc(self, arc_solutions):
#         """fit() returns exactly one SolutionOD per arc."""
#         solutions, _ = arc_solutions
#         assert len(solutions) == 2
#         for sol in solutions:
#             assert isinstance(sol, scb.SolutionOD)

#     def test_multi_arc_global_populated(self, arc_solutions):
#         """Every solution has multi_arc_global with the expected keys."""
#         solutions, _ = arc_solutions
#         required = {"x_G", "P_G", "global_param_names", "arc_global_abs", "n_outer", "converged"}
#         for sol in solutions:
#             assert sol.multi_arc_global is not None
#             assert required.issubset(sol.multi_arc_global.keys())

#     def test_global_std_devs_shape_and_finite(self, arc_solutions):
#         """global_std_devs returns a finite (3,) array (one per velocity component)."""
#         solutions, _ = arc_solutions
#         std_devs = solutions[0].global_std_devs()
#         assert std_devs.shape == (3,)
#         assert np.all(np.isfinite(std_devs))
#         assert np.all(std_devs > 0)

#     def test_outer_loop_converged(self, arc_solutions):
#         """Two identical arcs converge in the first outer iteration."""
#         solutions, _ = arc_solutions
#         assert solutions[0].multi_arc_global["converged"] is True

#     def test_global_param_names_recorded(self, arc_solutions):
#         """multi_arc_global records 'velocity' as the global parameter."""
#         solutions, _ = arc_solutions
#         assert "velocity" in solutions[0].multi_arc_global["global_param_names"]

#     def test_global_covariance_symmetric_positive(self, arc_solutions):
#         """Combined global covariance is symmetric and has positive diagonal."""
#         solutions, _ = arc_solutions
#         p_g = np.asarray(solutions[0].multi_arc_global["P_G"])
#         np.testing.assert_allclose(p_g, p_g.T, atol=1e-10)
#         assert np.all(np.diag(p_g) > 0)
