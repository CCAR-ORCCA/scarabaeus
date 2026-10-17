# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================#
#  Unit Tests for the LKF class  #
# ================================#
import src.scarabaeus as scb
import numpy as np
import pytest

km, kg, sec, hr = scb.Units.get_units(["km", "kg", "sec", "hr"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#


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
        sc_ref_coeff,)


@pytest.fixture
def state(epochs, origin, sc):
    """Define stateArray object"""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000)
    a_fogm_0 = scb.ArrayWFrame(np.array([0.0, 0.0, 0.0]), km * sec**-2, J2000)

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0)
                                                .param("a_fogm", sc, a_fogm_0),)


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,
        first_order_gauss_markov=True,
        fogm_beta=np.array([1e-4, 1e-4, 1e-4]),
    )


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
    accel_sigma = scb.ArrayWUnits(1e-6, km * sec**-2)
    state_sigma_list = [
        pos_sigma,
        pos_sigma,
        pos_sigma,
        vel_sigma,
        vel_sigma,
        vel_sigma,
        accel_sigma,
        accel_sigma,
        accel_sigma,]
    return scb.CovarianceMatrix(state_sigma_list, epochs[1], from_list=True)


@pytest.fixture
def measurements(meas_data_file):
    GS1 = scb.GroundStation("DSS-14")

    # (2) Initialize Measurement Models
    Range_GS1 = scb.RangeIdeal(
        "GS1 Ideal Range Model", GS1, sigma=scb.ArrayWUnits(1e-3, km))

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
            dataset_name="GS1 Range",),)
    return measurements_list


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(propagator, covar_mat, measurements, epochs):
    """
    Verifies that object is constructed correctly.
    """
    Q_diag = scb.ArrayWUnits(1e-8, km * sec**-2)
    Q_sigma_list = [Q_diag, Q_diag, Q_diag]
    Q_cont = scb.CovarianceMatrix(Q_sigma_list, epochs[0], from_list=True)

    settings = scb.FilterSettings(
        initial_covariance=covar_mat,
        process_noise=scb.ProcessNoiseSettings(
            type="DMC",
            Q_cont=Q_cont,
            beta=np.array([1e-4, 1e-4, 1e-4]),
        ),
    )
    lkf_it1 = scb.LKF(
        propagator=propagator,
        settings=settings,
        measurements=measurements,
        traj_name="orbiter_ref_it1.bsp",)

    assert lkf_it1.propagator == propagator
    assert lkf_it1.measurements == measurements
    assert lkf_it1.settings == settings
    assert lkf_it1.initial_covariance == settings.initial_covariance


def _burn_arc_lkf(estimation, out_dir, k_thrust_sigma=1e-9, corr_factors=None,
                  filter_class=scb.LKF):
    """Fitted covariance-analysis filter (LKF unless told otherwise) on a short
    finite-burn arc with SNC, with k_thrust either considered or estimated
    (a-priori correlations packed as in CovarianceMatrix's corr_factors)."""
    N, rad = scb.Units.get_units(["N", "rad"])
    t0 = 923000647.3506516
    t1 = t0 + 1200.0
    u = 1.0 / np.sqrt(2.0)

    maneuver = scb.Maneuver()
    maneuver.thrust = scb.ArrayWUnits(0.2, N)
    maneuver.mass_flow = scb.ArrayWUnits(0.2 / (2200.0 * 9.80665), kg / sec)
    maneuver.start_time = scb.EpochArray(t0, "TDB")
    maneuver.end_time = scb.EpochArray(t1, "TDB")
    maneuver.ux = scb.ArrayWUnits(u, None)
    maneuver.uy = scb.ArrayWUnits(0.0, None)
    maneuver.uz = scb.ArrayWUnits(u, None)

    sc = scb.Spacecraft("lkf_burn_sc", -60000, scb.ArrayWUnits(2300.0, kg),
                        scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))
    definition = (
        scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(
            np.array([-2.65649403e08, 1.88325868e08, 1.01977022e08]), km, J2000))
        .velocity(sc, scb.ArrayWFrame(
            np.array([-9.07473227, -12.48458138, -4.46117482]), km / sec, J2000))
        .param("dRA1", sc, scb.ArrayWFrame(0.4, rad, J2000),
               estimation="estimated", dynamics="static")
        .param("dDEC1", sc, scb.ArrayWFrame(0.3, rad, J2000),
               estimation="estimated", dynamics="static")
        .param("k_thrust1", sc, scb.ArrayWFrame(0.9, None, None),
               estimation=estimation, dynamics="static")
    )
    state = scb.StateArray(epoch=scb.EpochArray(t0, sys="TDB"),
                           origin=scb.CelestialBody.from_constants("SUN"),
                           state=definition)
    propagator = scb.Propagator(
        primary_body=sc, state_vector=state,
        tspan=scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB"),
        force_models=scb.ForceModelTranslation(
            primary_body=sc, cannonball_SRP=True, finite_burn=True, maneuver=maneuver))

    gs = scb.GroundStation("DSS-14")
    meas_epochs = scb.EpochArray(np.arange(t0, t1 - 30.0, 60.0), sys="TDB")
    measurements = scb.MeasurementSpec.from_dict([
        {"model": scb.RangeIdeal("R", gs, sigma=scb.ArrayWUnits(1e-3, km),
                                 state_definition=definition),
         "dataset_name": "R", "epochs": meas_epochs},
        {"model": scb.RangeRateIdeal("RR", gs, sigma=scb.ArrayWUnits(1e-6, km / sec),
                                     state_definition=definition),
         "dataset_name": "RR", "epochs": meas_epochs},
    ])

    settings = scb.FilterSettings(
        initial_covariance=scb.CovarianceMatrix(
            [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.01, km / sec)] * 3
            + [scb.ArrayWUnits(np.deg2rad(0.5), rad)] * 2
            + [scb.ArrayWUnits(k_thrust_sigma, None)],
            scb.EpochArray(t0, sys="TDB"), from_list=True, corr_factors=corr_factors),
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=True,
        # Noise large against the 1e-9 sigma of k_thrust, so a gain that
        # missed it would be off by orders of magnitude
        process_noise=scb.ProcessNoiseSettings(
            type="SNC",
            Q_cont=scb.CovarianceMatrix([scb.ArrayWUnits(6e-8, km * sec**-2)] * 3,
                                        scb.EpochArray(t0, sys="TDB"), from_list=True)),
    )
    traj_name = f"{filter_class.__name__.lower()}_burn_{estimation}.bsp"
    lkf = filter_class(propagator=propagator, settings=settings,
                       measurements=measurements, traj_name=traj_name,
                       traj_dir=str(out_dir))
    lkf.fit(max_iterations=1, verbose=False, traj_name=traj_name)
    return lkf


def _run_burn_arc_covariance(estimation, out_dir):
    """Estimated-state covariance history of the burn arc, with k_thrust
    either considered or estimated with a tiny a-priori sigma."""
    lkf = _burn_arc_lkf(estimation, out_dir)
    return np.array([np.asarray(P, float)[:8, :8] for P in lkf.covariance_history])


def test_consider_gain_includes_process_noise(tmp_path):
    """
    A considered parameter with a tiny a-priori sigma must give the same
    covariance for the estimated states as estimating it, also with SNC:
    the gain has to see the process noise added to Pxx.
    """
    P_considered = _run_burn_arc_covariance("considered", tmp_path / "considered")
    P_estimated = _run_burn_arc_covariance("estimated", tmp_path / "estimated")

    assert P_considered.shape == P_estimated.shape
    np.testing.assert_allclose(P_considered, P_estimated, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("estimation", ["considered", "estimated"])
def test_matches_srif(tmp_path, estimation):
    """
    The LKF and the SRIF are the same estimator: with k_thrust considered,
    the estimate is conditioned on its a-priori value and the consider
    covariance is Pxx + Sx Pcc Sx^T. An a-priori correlation between the
    estimated and considered states must survive the start of the pass.
    """
    n = 9
    # k_thrust (index 8) correlated with x (0) and dRA1 (6)
    corr = [scb.ArrayWUnits(0.3 if j == 8 and i in (0, 6) else 0.0, None)
            for i in range(n) for j in range(i + 1, n)]
    lkf = _burn_arc_lkf(estimation, tmp_path / "lkf", 0.05, corr)
    srif = _burn_arc_lkf(estimation, tmp_path / "srif", 0.05, corr,
                         filter_class=scb.SRIF)

    histories = ["covariance_history"]
    if estimation == "considered":
        histories.append("covariance_consider_history")
    for name in histories:
        P_lkf = np.array([np.asarray(P, float) for P in getattr(lkf, name)])
        P_srif = np.array([np.asarray(P, float) for P in getattr(srif, name)])
        assert P_lkf.shape == P_srif.shape
        for P_l, P_s in zip(P_lkf, P_srif):
            s_l, s_s = np.sqrt(np.diag(P_l)), np.sqrt(np.diag(P_s))
            np.testing.assert_allclose(s_l, s_s, rtol=1e-8, atol=0.0)
            np.testing.assert_allclose(P_l / np.outer(s_l, s_l),
                                       P_s / np.outer(s_s, s_s), rtol=0.0, atol=1e-8)
        # The a-priori correlation of x with k_thrust is still there
        s0 = np.sqrt(np.diag(P_lkf[0]))
        assert abs(P_lkf[0][0, 8] / (s0[0] * s0[8])) > 0.1
