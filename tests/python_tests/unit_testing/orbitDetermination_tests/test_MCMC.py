# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================#
#  Unit Tests for the MCMC class  #
# ================================#
import copy
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src.scarabaeus as scb
import numpy as np
import pytest

kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])
J2000 = scb.Frame("J2000")

POS = np.array([-1.1123095885148e08, 8.9094345479316e07, 3.8656500948069e07])
VEL = np.array([-20.6936999825159, -16.7800270812616, -6.6437327193572])
DELTA_POS = np.array([1.0, -0.5, 0.75])
DELTA_VEL = np.array([1e-3, -5e-4, 7.5e-4])
LABELS = ("mcmc_range", "mcmc_rate")

# Whitened chain moments must match the batch's (zero mean, identity
# covariance) to this tolerance: about five standard errors for the chains run
# here, whose smallest effective sample size is several hundred.
WHITENED_TOL = 0.25


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """Truth trajectory, observations and the pieces to fit a filter to them."""
    directory = tmp_path_factory.mktemp("mcmc")

    origin = scb.CelestialBody.from_constants("SUN")
    station = scb.GroundStation("DSS-14")
    third_bodies = scb.CelestialBody.from_constants(["MERCURY", "VENUS", "EARTH"])

    t0 = scb.SpiceManager.jd2et(2461809.72995654 + 1 / 3)
    epochs = scb.EpochArray(np.arange(t0, t0 + 86400.0, 1800.0), sys="TDB")

    mass = scb.ArrayWUnits(2000.0, kg)
    area = scb.ArrayWUnits(1e-06, km**2)
    reflectivity = scb.ArrayWUnits(1.5, None)

    def build(name, spice_id, position, velocity, velocity_estimation="estimated"):
        spacecraft = scb.Spacecraft(name, spice_id, mass, area, reflectivity)
        definition = (
            scb.StateDefinition()
            .position(spacecraft, scb.ArrayWFrame(position, km, J2000))
            .velocity(
                spacecraft,
                scb.ArrayWFrame(velocity, km / sec, J2000),
                estimation=velocity_estimation,
            )
        )
        state = scb.StateArray(epoch=epochs[0], origin=origin, state=definition)

        return spacecraft, definition, state

    def force_model(spacecraft):
        return scb.ForceModelTranslation(
            primary_body=spacecraft, third_bodies=third_bodies
        )

    # Truth trajectory, published so the measurement models can query it.
    truth_sc, truth_def, truth_state = build("mcmc_truth", -1400, POS, VEL)
    truth = scb.Propagator(
        primary_body=truth_sc,
        state_vector=truth_state,
        tspan=epochs,
        force_models=force_model(truth_sc),
    )
    truth.propagate(display_progress=False)
    scb.Trajectory(state_array=truth.propagated_state_array).write_to_spk(
        str(directory / "mcmc_truth.bsp")
    )

    range_sigma = scb.ArrayWUnits(1e-3, km)
    rate_sigma = scb.ArrayWUnits(1e-5, km / sec)

    def models(definition):
        return (
            scb.RangeIdeal("R", station, sigma=range_sigma, state_definition=definition),
            scb.RangeRateIdeal(
                "RR", station, sigma=rate_sigma, state_definition=definition
            ),
        )

    # Noise-free observations of the truth, shared by every filter.
    observed = {}
    for model, label, units in zip(models(truth_def), LABELS, (km, km / sec)):
        model.write_observed_measurements(
            target=truth_sc, epoch_array=epochs, noisy=False, file_name=label
        )
        observed[label] = model.observed_measurements(
            file_name=f"data/measurements/radiometric/{label}.json",
            meas_name="meas_sim",
            units=units,
        )

    def covariance(velocity_sigma):
        return scb.CovarianceMatrix(
            [scb.ArrayWUnits(3.0, km)] * 3
            + [scb.ArrayWUnits(velocity_sigma, km / sec)] * 3,
            epochs[1],
            from_list=True,
        )

    def fitted(
        name, spice_id, delta_vel, velocity_estimation, velocity_sigma, cls=scb.LSB
    ):
        """Fit a filter to convergence from a perturbed reference."""
        spacecraft, definition, state = build(
            name, spice_id, POS + DELTA_POS, VEL + delta_vel, velocity_estimation
        )
        range_model, rate_model = models(definition)
        measurements = scb.MeasurementSpec.from_dict(
            [
                {
                    "model": range_model,
                    "observed_meas": observed[LABELS[0]],
                    "dataset_name": "Range",
                },
                {
                    "model": rate_model,
                    "observed_meas": observed[LABELS[1]],
                    "dataset_name": "Range Rate",
                },
            ]
        )
        propagator = scb.Propagator(
            primary_body=spacecraft,
            state_vector=state,
            tspan=epochs,
            force_models=force_model(spacecraft),
        )
        filter_obj = cls(
            propagator=propagator,
            measurements=measurements,
            settings=scb.FilterSettings(
                initial_covariance=covariance(velocity_sigma),
                output=scb.OutputSettings(),
            ),
            traj_name=f"{name}.bsp",
            traj_dir=str(directory),
        )
        solution, _iterations, converged = filter_obj.fit(
            max_iterations=6, verbose=False, traj_name=f"{name}.bsp"
        )
        assert converged

        return filter_obj, solution

    lsb, lsb_solution = fitted("mcmc_lsb", -1411, DELTA_VEL, "estimated", 3e-3)
    # Velocity at its true value and considered: the position is estimated.
    # A 1 mm/s consider sigma: at 3 m/s the joint position covariance is
    # 11 orders above the conditional one, and the Schur complement that
    # recovers the stored conditional covariance loses its thin direction
    # to round-off, in the LSB as in the MCMC.
    consider, _ = fitted("mcmc_consider", -1412, np.zeros(3), "considered", 1e-6)

    yield {
        "lsb": lsb,
        "lsb_solution": lsb_solution,
        "consider": consider,
        "n_epochs": len(lsb.state_deviation_history),
    }

    for label in LABELS:
        path = f"data/measurements/radiometric/{label}.json"
        if os.path.isfile(path):
            os.remove(path)


@pytest.fixture(scope="module")
def sampled(scenario):
    """MCMC of the converged LSB, sampled once for the read-only tests."""
    mcmc = scb.MCMC(scenario["lsb"], verbose=False)
    solution = mcmc.sample(n_steps=4000, n_particles=500, seed=0, verbose=False)

    return mcmc, solution


def whitened_moments(mean, covariance, mean_ref, covariance_ref):
    """Mean and covariance in the coordinates where the reference is N(0, I)."""
    L = np.linalg.cholesky(covariance_ref)
    mean_w = np.linalg.solve(L, np.ravel(mean) - np.ravel(mean_ref))
    covariance_w = np.linalg.solve(L, np.linalg.solve(L, covariance).T)

    return mean_w, covariance_w


@pytest.fixture(scope="module")
def burn_sequence(tmp_path_factory):
    """
    LSB fitted to noise-free data over a coast, a finite burn adding dRA1,
    dDEC1 and k_thrust1, and (optionally) a final coast that drops them.
    """
    N, rad = scb.Units.get_units(["N", "rad"])
    t_burn = 923000647.3506516
    t_burn_end = t_burn + 600.0
    u = 1.0 / np.sqrt(2.0)
    origin = scb.CelestialBody.from_constants("SUN")

    def fit(name, spice_id, k_estimation, final_coast):
        directory = tmp_path_factory.mktemp(name)
        start_time = t_burn - 600.0
        end_time = t_burn_end + 600.0 if final_coast else t_burn_end

        maneuver = scb.Maneuver()
        maneuver.thrust = scb.ArrayWUnits(0.2, N)
        maneuver.mass_flow = scb.ArrayWUnits(0.2 / (2200.0 * 9.80665), kg / sec)
        maneuver.start_time = scb.EpochArray(t_burn, "TDB")
        maneuver.end_time = scb.EpochArray(t_burn_end, "TDB")
        maneuver.ux = scb.ArrayWUnits(u, None)
        maneuver.uy = scb.ArrayWUnits(0.0, None)
        maneuver.uz = scb.ArrayWUnits(u, None)

        spacecraft = scb.Spacecraft(
            name,
            spice_id,
            scb.ArrayWUnits(2300.0, kg),
            scb.ArrayWUnits(1e-6, km**2),
            scb.ArrayWUnits(1.5, None),
        )

        def propagator(definition, t0, t1, burn=None):
            state = scb.StateArray(
                epoch=scb.EpochArray(t0, sys="TDB"), origin=origin, state=definition
            )
            force_model = scb.ForceModelTranslation(
                primary_body=spacecraft,
                cannonball_SRP=True,
                finite_burn=burn is not None,
                maneuver=burn,
            )
            return scb.Propagator(
                primary_body=spacecraft,
                state_vector=state,
                tspan=scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB"),
                force_models=force_model,
            )

        coast = (
            scb.StateDefinition()
            .position(
                spacecraft,
                scb.ArrayWFrame(
                    np.array([-2.65643964e08, 1.88333350e08, 1.01979698e08]), km, J2000
                ),
            )
            .velocity(
                spacecraft,
                scb.ArrayWFrame(
                    np.array([-9.07510541, -12.48432651, -4.46105273]), km / sec, J2000
                ),
            )
        )
        burn = (
            scb.StateDefinition.empty_pv(spacecraft, frame=J2000)
            .param("dRA1", spacecraft, scb.ArrayWFrame(0.4, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("dDEC1", spacecraft, scb.ArrayWFrame(0.3, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("k_thrust1", spacecraft, scb.ArrayWFrame(0.9, None, None),
                   estimation=k_estimation, dynamics="static")
        )

        sequence = scb.MissionSequence(name)
        sequence.addLeg(name="coast", prop=propagator(coast, start_time, t_burn))
        sequence.addFiniteBurn(
            name="burn", prop=propagator(burn, t_burn, t_burn_end, maneuver)
        )
        if final_coast:
            sequence.addLeg(
                name="final coast",
                prop=propagator(
                    scb.StateDefinition.empty_pv(spacecraft, frame=J2000),
                    t_burn_end,
                    end_time,
                ),
            )

        station = scb.GroundStation("DSS-14")
        epochs = scb.EpochArray(
            np.arange(start_time + 30.0, end_time - 30.0, 60.0), sys="TDB"
        )
        measurements = scb.MeasurementSpec.from_dict(
            [
                {
                    "model": scb.RangeIdeal(
                        f"{name}_range", station,
                        sigma=scb.ArrayWUnits(1e-3, km), sequence_definition=sequence,
                    ),
                    "dataset_name": f"{name}_range",
                    "epochs": epochs,
                },
                {
                    "model": scb.RangeRateIdeal(
                        f"{name}_rate", station,
                        sigma=scb.ArrayWUnits(1e-6, km / sec),
                        sequence_definition=sequence,
                    ),
                    "dataset_name": f"{name}_rate",
                    "epochs": epochs,
                },
            ]
        )
        settings = scb.FilterSettings(
            initial_covariance=scb.CovarianceMatrix(
                [scb.ArrayWUnits(100.0, km)] * 3
                + [scb.ArrayWUnits(0.01, km / sec)] * 3,
                scb.EpochArray(start_time, sys="TDB"),
                from_list=True,
                extra_apriori_sigmas={
                    "dRA1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                    "dDEC1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                    "k_thrust1": scb.ArrayWUnits(0.05, None),
                },
            ),
            output=scb.OutputSettings(
                solution_output_path=str(directory),
                solution_output_name=name,
                save_reference_trajectory=False,
                save_estimated_trajectory=False,
            ),
        )
        traj_name = f"{name}.bsp"
        lsb = scb.LSB(
            propagator=sequence,
            settings=settings,
            measurements=measurements,
            traj_name=traj_name,
            traj_dir=str(directory),
        )
        lsb.fit(max_iterations=3, verbose=False, traj_name=traj_name)
        assert lsb.convergence_info["converged"]

        return lsb

    return fit


def assert_sigmas_match(mcmc, lsb, consider=False, tol=0.1):
    """Per-state sigma ratios of the chain against the batch, at every epoch."""
    mcmc_history = mcmc.covariance_consider_history if consider else mcmc.covariance_history
    lsb_history = lsb.covariance_consider_history if consider else lsb.covariance_history
    assert len(mcmc_history) == len(lsb_history)
    for P, P_ref in zip(mcmc_history, lsb_history):
        if P_ref is None:
            assert P is None
            continue
        sigma_ref = np.sqrt(np.diag(np.asarray(P_ref, dtype=float)))
        sigma = np.sqrt(np.diag(np.asarray(P, dtype=float)))
        np.testing.assert_allclose(sigma / sigma_ref, 1.0, atol=tol)


# endregion Fixtures #
# --------------------#


# ---------------------------#
# region    Initialization #
# ---------------------------#
def test_rejects_non_filters(sampled):
    mcmc, _ = sampled
    with pytest.raises(scb.MCMC.FilterTypeError):
        scb.MCMC("not a filter", verbose=False)
    with pytest.raises(scb.MCMC.FilterTypeError):
        scb.MCMC(mcmc, verbose=False)


@pytest.mark.parametrize(
    "attributes, error",
    [
        ({"covariance_analysis": True}, scb.MCMC.CovarianceAnalysisError),
        ({"convergence_info": None}, scb.MCMC.UnfittedFilterError),
        ({"convergence_info": {"converged": False}}, scb.MCMC.FilterNotConvergedError),
        ({"SNC_flag": True}, scb.MCMC.ProcessNoiseError),
        ({"DMC_flag": True}, scb.MCMC.ProcessNoiseError),
    ],
)
def test_rejects_filters_it_cannot_sample(scenario, attributes, error):
    filter_obj = copy.copy(scenario["lsb"])
    # setattr, not __dict__: several of these flags are FilterOD properties.
    for name, value in attributes.items():
        setattr(filter_obj, name, value)
    with pytest.raises(error):
        scb.MCMC(filter_obj, verbose=False)


def test_accepts_a_solution_and_leaves_the_filter_alone(scenario):
    lsb = scenario["lsb"]
    history = lsb.covariance_history
    mcmc = scb.MCMC(scenario["lsb_solution"], verbose=False)

    assert mcmc.source_filter == "LSB"
    assert mcmc.samples is None and mcmc.labels is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scb.MCMC.ShortChainWarning)
        mcmc.sample(n_steps=200, n_particles=10, seed=1, verbose=False)
    assert lsb.covariance_history is history
    assert mcmc.covariance_history is not history


def test_fit_is_not_supported(scenario):
    with pytest.raises(scb.MCMC.FitNotSupportedError):
        scb.MCMC(scenario["lsb"], verbose=False).fit()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_steps": 0}, scb.MCMC.SampleSizeError),
        ({"n_steps": 10.5}, scb.MCMC.SampleSizeError),
        ({"n_particles": -1}, scb.MCMC.SampleSizeError),
        ({"n_walkers": 11}, scb.MCMC.WalkerCountError),
        ({"moves": "gibbs"}, scb.MCMC.UnknownMoveError),
        ({"log_prior": lambda dx: -np.inf}, scb.MCMC.InitialWalkersError),
    ],
)
def test_sample_arguments(scenario, kwargs, error):
    mcmc = scb.MCMC(scenario["lsb"], verbose=False)
    with pytest.raises(error):
        mcmc.sample(**{"n_steps": 100, "seed": 0, "verbose": False, **kwargs})


# endregion Initialization #
# --------------------------#


# ---------------------#
# region    Posterior #
# ---------------------#
def test_posterior_matches_the_batch(scenario, sampled):
    """With the Gaussian a-priori the linearized posterior is the LSB one."""
    lsb = scenario["lsb"]
    mcmc, _ = sampled

    for k in (0, scenario["n_epochs"] - 1):
        P_ref = np.asarray(lsb.covariance_history[k], dtype=float)
        mean_w, covariance_w = whitened_moments(
            mcmc.state_deviation_history[k],
            mcmc.covariance_history[k],
            lsb.state_deviation_history[k],
            P_ref,
        )
        np.testing.assert_allclose(mean_w, 0.0, atol=WHITENED_TOL)
        np.testing.assert_allclose(covariance_w, np.eye(6), atol=WHITENED_TOL)

    # The global state is the first epoch's: its moments are the samples'.
    np.testing.assert_allclose(
        mcmc.posterior_mean, np.ravel(mcmc.state_deviation_history[0]), atol=1e-12
    )


def test_postfit_residuals_are_about_the_mean(sampled):
    mcmc, _ = sampled
    prefit = mcmc.prefit_residuals
    postfit = mcmc.postfit_residuals

    assert prefit.keys() == postfit.keys()
    # Noise-free data and a converged reference: the residuals are tiny.
    for name in postfit:
        residuals, sigma = np.asarray(postfit[name], dtype=float).T
        assert residuals.size == len(prefit[name]) > 0
        assert np.all(np.abs(residuals) < sigma)


def test_solution_carries_particles_and_uq_metrics(scenario, sampled):
    mcmc, solution = sampled

    assert isinstance(solution, scb.SolutionOD)
    assert solution.filter is mcmc
    assert mcmc.particles.shape == (500, 6, scenario["n_epochs"])
    assert solution.uq_metrics["n_particles"] == 500
    assert solution.uq_metrics["n_epochs"] == scenario["n_epochs"]

    # The particles at the first epoch are the samples.
    np.testing.assert_array_equal(mcmc.particles[:, :, 0], mcmc.samples.samples)


def test_samples_and_diagnostics(sampled):
    mcmc, _ = sampled
    samples = mcmc.samples
    diagnostics = mcmc.diagnostics

    assert isinstance(samples, scb.MonteCarloSamples)
    assert samples.labels == mcmc.labels and len(mcmc.labels) == 6
    assert samples.n_samples == 500
    assert mcmc.chain.shape == (4000, diagnostics["n_walkers"], 6)
    assert mcmc.log_prob.shape == (4000, diagnostics["n_walkers"])
    assert diagnostics["n_walkers"] == max(
        mcmc.walkers_per_dimension * 6, mcmc.min_walkers
    )
    assert diagnostics["source_filter"] == "LSB"
    assert diagnostics["mean_acceptance_fraction"] >= mcmc.min_acceptance
    assert diagnostics["burn_in"] <= 2000 and diagnostics["thin"] >= 1
    assert np.all(diagnostics["effective_sample_size"] > 0.0)


def test_seed_makes_it_reproducible(scenario):
    runs = []
    for _ in range(2):
        mcmc = scb.MCMC(scenario["lsb"], verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scb.MCMC.ShortChainWarning)
            mcmc.sample(
                n_steps=300, n_particles=50, moves="de", seed=7, verbose=False
            )
        runs.append(mcmc)

    np.testing.assert_array_equal(runs[0].chain, runs[1].chain)
    np.testing.assert_array_equal(runs[0].samples.samples, runs[1].samples.samples)


def test_short_chain_warns(scenario):
    mcmc = scb.MCMC(scenario["lsb"], verbose=False)
    with pytest.warns(scb.MCMC.ShortChainWarning):
        mcmc.sample(n_steps=50, n_particles=10, seed=0, verbose=False)
    assert mcmc.diagnostics["burn_in"] <= 25


def test_sampler_settings(scenario):
    mcmc = scb.MCMC(
        scenario["lsb"],
        verbose=False,
        burn_in_factor=0.0,
        thin_factor=0.0,
        tau_multiple=0,
        min_acceptance=1.0,
    )
    mcmc.min_walkers = 40
    with pytest.warns(scb.MCMC.AcceptanceFractionWarning):
        mcmc.sample(n_steps=100, n_particles=10, seed=0, verbose=False)

    assert mcmc.diagnostics["n_walkers"] == 40
    assert mcmc.diagnostics["burn_in"] == 0 and mcmc.diagnostics["thin"] == 1


def test_consider_spread(scenario):
    """Considered velocity: the joint covariance is the Schmidt batch one."""
    lsb = scenario["consider"]
    mcmc = scb.MCMC(lsb, verbose=False)
    mcmc.sample(n_steps=3000, n_particles=2000, seed=3, verbose=False)
    position = slice(0, 3)

    # The conditional position covariance is the chain's.
    mean_w, covariance_w = whitened_moments(
        np.ravel(mcmc.state_deviation_history[0])[position],
        np.asarray(mcmc.covariance_history[0])[position, position],
        np.ravel(lsb.state_deviation_history[0])[position],
        np.asarray(lsb.covariance_history[0], dtype=float)[position, position],
    )
    np.testing.assert_allclose(mean_w, 0.0, atol=WHITENED_TOL)
    np.testing.assert_allclose(covariance_w, np.eye(3), atol=WHITENED_TOL)

    # The joint one adds the linear consider spread.
    J_ref = np.asarray(lsb.covariance_consider_history[0], dtype=float)
    _, covariance_w = whitened_moments(
        np.zeros(6), mcmc.covariance_consider_history[0], np.zeros(6), J_ref
    )
    np.testing.assert_allclose(covariance_w, np.eye(6), atol=WHITENED_TOL)

    # The considered entries of the particles are the a-priori spread.
    sigma_vel = np.sqrt(np.diag(J_ref)[3:])
    particles_sigma = mcmc.samples.get_standard_deviations()[3:]
    np.testing.assert_allclose(particles_sigma / sigma_vel, 1.0, atol=0.1)
    assert mcmc.chain.shape[-1] == 3


# endregion Posterior #
# ---------------------#


# --------------------#
# region    Sequence #
# --------------------#
def test_sequence_matches_the_batch(burn_sequence):
    """Burn parameters added at a node and dropped at the next: LSB's posterior."""
    lsb = burn_sequence("mcmc_seq_est", -1421, "estimated", final_coast=True)
    mcmc = scb.MCMC(lsb, verbose=False)
    mcmc.sample(n_steps=6000, n_particles=300, seed=2, verbose=False)
    n_epochs = len(lsb.covariance_history)

    assert_sigmas_match(mcmc, lsb)
    assert mcmc.labels[6:] == [
        "mcmc_seq_est.dRA1 [rad]",
        "mcmc_seq_est.dDEC1 [rad]",
        "mcmc_seq_est.k_thrust1 [unitless]",
    ]
    # Zero-padded on the coasts, which carry only position and velocity.
    assert mcmc.particles.shape == (300, 9, n_epochs)
    assert np.all(mcmc.particles[:, 6:, 0] == 0.0)
    assert np.all(mcmc.particles[:, 6:, -1] == 0.0)
    assert np.any(mcmc.particles[:, 6:, 15] != 0.0)
    for k in (0, n_epochs // 2, n_epochs - 1):
        sigma = np.sqrt(np.diag(np.asarray(lsb.covariance_history[k], dtype=float)))
        mean = np.ravel(mcmc.state_deviation_history[k])[: sigma.size]
        np.testing.assert_allclose(mean / sigma, 0.0, atol=WHITENED_TOL)


def test_sequence_consider_matches_the_batch(burn_sequence):
    """k_thrust1 considered from the node it is added at to the end."""
    lsb = burn_sequence("mcmc_seq_con", -1422, "considered", final_coast=False)
    mcmc = scb.MCMC(lsb, verbose=False)
    mcmc.sample(n_steps=6000, n_particles=300, seed=4, verbose=False)

    assert_sigmas_match(mcmc, lsb)
    assert_sigmas_match(mcmc, lsb, consider=True)
    assert mcmc.chain.shape[-1] == 8


def test_dropped_consider_warns(burn_sequence):
    """The chain keeps a dropped considered state, unlike the filter."""
    lsb = burn_sequence("mcmc_seq_drop", -1423, "considered", final_coast=True)
    mcmc = scb.MCMC(lsb, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scb.MCMC.ShortChainWarning)
        with pytest.warns(scb.MCMC.ConsiderDroppedWarning, match="k_thrust1"):
            mcmc.sample(n_steps=200, n_particles=20, seed=0, verbose=False)


# endregion Sequence #
# --------------------#


# -----------------#
# region    Prior #
# -----------------#
def test_gaussian_log_prior_reproduces_the_default():
    """A Gaussian log_prior equal to the a-priori changes only the constant."""
    rng = np.random.default_rng(11)
    n = 4
    A = rng.standard_normal((20, n))
    D = A.T @ A
    d = A.T @ rng.standard_normal(20)
    I_bar = np.diag(rng.uniform(0.5, 2.0, n))
    x_bar = rng.standard_normal(n)

    M = I_bar + D
    x_hat = np.linalg.solve(M, I_bar @ x_bar + d)
    L = np.linalg.cholesky(np.linalg.inv(M))

    default = scb.MCMC._log_posterior(x_hat, L, D, d, x_bar, None)
    custom = scb.MCMC._log_posterior(
        x_hat, L, D, d, x_bar, lambda dx: -0.5 * dx @ I_bar @ dx
    )
    theta = rng.standard_normal((50, n))
    offset = custom(theta) - default(theta)
    np.testing.assert_allclose(offset, offset[0], atol=1e-10)


def test_bounded_prior_truncates_the_posterior(scenario):
    lsb = scenario["lsb"]
    mcmc = scb.MCMC(lsb, verbose=False)
    x_bar = np.ravel(mcmc._apriori_deviation(0))
    # Cut the first position component at the batch solution.
    bound = float(np.ravel(lsb.state_deviation_history[0])[0] - x_bar[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scb.MCMC.ShortChainWarning)
        mcmc.sample(
            n_steps=1500,
            n_particles=200,
            log_prior=lambda dx: 0.0 if dx[0] <= bound else -np.inf,
            seed=5,
            verbose=False,
        )

    assert np.all(mcmc.chain[..., 0] - x_bar[0] <= bound)
    assert np.all(mcmc.samples.samples[:, 0] - x_bar[0] <= bound)
    assert mcmc.posterior_mean[0] - x_bar[0] < bound


# endregion Prior #
# -----------------#


# -----------------------#
# region    Corner plot #
# -----------------------#
def test_corner_plot(sampled):
    mcmc, _ = sampled
    fig = scb.Plotting.plot_corner(
        mcmc.samples,
        covariance=mcmc.posterior_covariance,
        indices=[0, 1, 2],
        truths=np.zeros(6),
        title="MCMC",
    )
    assert len(fig.axes) == 9
    plt.close(fig)

    fig = scb.Plotting.plot_corner(mcmc.samples.samples[:, :2], labels=["a", "b"])
    assert len(fig.axes) == 4
    plt.close(fig)


@pytest.mark.parametrize(
    "samples, kwargs, error",
    [
        (np.zeros((1, 2)), {}, scb.Plotting.CornerSamplesError),
        (np.zeros((10, 2)), {"indices": [0, 2]}, scb.Plotting.CornerSamplesError),
        (np.zeros((10, 2)), {"labels": ["a"]}, scb.Plotting.CornerOverlayError),
        (np.zeros((10, 2)), {"truths": np.zeros(3)}, scb.Plotting.CornerOverlayError),
        (
            np.zeros((10, 2)),
            {"covariance": np.eye(3)},
            scb.Plotting.CornerOverlayError,
        ),
    ],
)
def test_corner_plot_errors(samples, kwargs, error):
    with pytest.raises(error):
        scb.Plotting.plot_corner(samples, **kwargs)


# endregion Corner plot #
# -----------------------#

