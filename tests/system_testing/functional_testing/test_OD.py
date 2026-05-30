# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Tests to verify that Scarabaeus orbit determination (OD) systems
function properly.

Verifies two different OD scenarios:
- Keplerian interplanetary orbit from Earth orbit to the asteroid belt
-
"""

import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np
from scipy.stats import chi2 as chi2_dist

from typing import TypedDict

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
    },
)


@pytest.fixture(scope="module")
def traj_keplerian(sc, epochs) -> TrajData:
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
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=x0,
        tspan=epochs,
        force_models=scb.ForceModelTranslation(sc),
    )
    prop.propagate()
    states = prop.propagated_state_array

    # save to trajectory
    bsp_path = "test_OD.bsp"  # <- need to get this working
    orbiter_traj = scb.Trajectory(bsp_path, state_array=states)

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
    }
    return traj_data


@pytest.fixture(scope="module")
def traj_full_dyn(sc, epochs) -> TrajData:
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
    third_bodies = ["MERCURY", "VENUS", "SUN"]
    prop = scb.Propagator(
        primary_body=sc,
        state_vector=x0,
        tspan=epochs,
        force_models=scb.ForceModelTranslation(
            sc,
            cannonball_SRP=True,
            third_bodies=third_bodies,
        ),
    )
    prop.propagate()
    states = prop.propagated_state_array

    # save to trajectory
    bsp_path = "test_OD_full_dyn.bsp"
    orbiter_traj = scb.Trajectory(bsp_path, state_array=states)

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


# --------------#
# region Tests #
# --------------#
class FilterTests:
    """
    Base class for all filters that provides common tests for each.
    """

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
    ):
        """Run the filter with clean and noisy measurements, return both solutions."""
        traj = active_traj
        filter_name = filter_class.__name__

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
                force_models=scb.ForceModelTranslation(sc_ref),
            )

        clean_filter = filter_class(
            propagator=make_propagator(),
            measurements=gen_clean_meas_from_traj,
            settings=filter_settings,
            traj_name=f"{filter_name}_clean.bsp",
        )
        noisy_filter = filter_class(
            propagator=make_propagator(),
            measurements=gen_noisy_meas_from_traj,
            settings=filter_settings,
            traj_name=f"{filter_name}_noisy.bsp",
        )
        iterate_kwargs = {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "verbose": False,
        }
        return clean_filter.iterate(**iterate_kwargs), noisy_filter.iterate(
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

        alpha = 0.02
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
        (clean_sol, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; residuals not meaningful")

        for ds in clean_sol.prefits:
            pre = np.array([r for r, _ in clean_sol.prefits[ds]])
            post = np.array([r for r, _ in clean_sol.postfits[ds]])
            prefit_rms = np.mean(pre)
            postfit_rms = np.mean(post)
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
                pos_3sigma = np.sqrt(np.trace(cov[:3, :3]))
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
        """NIS sum must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.7
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
        """NIS sum must fall within chi-squared confidence bounds."""
        (_, _, clean_converged), (noisy_sol, _, _) = filter_solution

        if not clean_converged:
            pytest.skip("Filter did not converge; NIS not meaningful")

        nis_vals = self.compute_NIS_stat(noisy_sol)
        ny = len(noisy_sol.postfits)
        N = len(nis_vals)

        alpha = 0.3
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
