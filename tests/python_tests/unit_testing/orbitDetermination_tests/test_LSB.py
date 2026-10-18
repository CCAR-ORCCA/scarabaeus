# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================#
#  Unit Tests for the LSB class  #
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

    return scb.StateArray(epochs[0], origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def force_model(sc):
    return scb.ForceModelTranslation(
        primary_body=sc,)


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
    obs_quantities_range = Range_GS1.observed_measurements(
        file_name=str(meas_data_file),
        meas_name="meas_ideal",
        units=km,)

    measurements_list = scb.MeasurementSpec.many(
        scb.MeasurementSpec(
            model=Range_GS1,
            observed_meas=obs_quantities_range,
            dataset_name="GS1 Range", ),
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
def test_initialization(propagator, covar_mat, measurements):
    """
    Verifies that object is constructed correctly.
    """
    settings = scb.FilterSettings(initial_covariance=covar_mat)
    lSB_it1 = scb.LSB(
        propagator=propagator,
        settings=settings,
        measurements=measurements,
        traj_name="orbiter_ref_it1.bsp",)

    assert lSB_it1.propagator == propagator
    assert lSB_it1.measurements == measurements
    assert lSB_it1.settings == settings
    assert lSB_it1.initial_covariance == settings.initial_covariance


def _run_burn_arc_covariance(filter_cls, estimation, k_thrust_sigma, out_dir):
    """Batch covariance analysis on a short finite-burn arc, with k_thrust
    either considered or estimated."""
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

    sc = scb.Spacecraft("batch_burn_sc", -60000, scb.ArrayWUnits(2300.0, kg),
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
            scb.EpochArray(t0, sys="TDB"), from_list=True),
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=True,
    )
    traj_name = f"batch_burn_{filter_cls.__name__}_{estimation}.bsp"
    batch = filter_cls(propagator=propagator, settings=settings, measurements=measurements,
                       traj_name=traj_name, traj_dir=str(out_dir))
    batch.fit(max_iterations=1, verbose=False, traj_name=traj_name)
    return batch


@pytest.mark.parametrize("filter_cls", [scb.LSB, scb.SRIFB])
def test_consider_covariance_is_conditional_at_every_epoch(filter_cls, tmp_path):
    """
    The batch maps its solution at epoch to every measurement time. With a
    considered state the stored covariance must stay the one given that
    state, i.e. what estimating it with a negligible a-priori sigma gives,
    and the joint one must keep the considered state's a-priori.
    """
    considered = _run_burn_arc_covariance(filter_cls, "considered", 0.05,
                                          tmp_path / "considered")
    known = _run_burn_arc_covariance(filter_cls, "estimated", 1e-7, tmp_path / "known")

    assert len(considered.covariance_history) == len(known.covariance_history) > 2
    for P, J, P_known in zip(considered.covariance_history,
                             considered.covariance_consider_history,
                             known.covariance_history):
        # compared as correlations: LSB's normal equations lose digits on the
        # small cross terms
        sigma = np.sqrt(np.diag(P_known)[:8])
        np.testing.assert_allclose((P[:8, :8] - P_known[:8, :8]) / np.outer(sigma, sigma),
                                   0.0, atol=1e-5)
        np.testing.assert_allclose(J[8, 8], 0.05**2, rtol=1e-12)
        # the consider covariance only adds to the conditional one (LSB's
        # inverted normal matrix is symmetric only to round-off)
        J_xx = 0.5 * (J[:8, :8] + J[:8, :8].T)
        assert np.linalg.eigvalsh(J_xx - P[:8, :8]).min() > -1e-9 * np.abs(J).max()


def _run_burn_sequence(filter_cls, out_dir):
    """Batch covariance analysis over coast, finite burn (adding dRA1, dDEC1
    and a considered k_thrust1), coast."""
    N, rad = scb.Units.get_units(["N", "rad"])
    t_m1 = 923000647.3506516
    t_m1_end = t_m1 + 600.0
    start_time, end_time = t_m1 - 600.0, t_m1_end + 600.0
    u = 1.0 / np.sqrt(2.0)

    maneuver = scb.Maneuver()
    maneuver.thrust = scb.ArrayWUnits(0.2, N)
    maneuver.mass_flow = scb.ArrayWUnits(0.2 / (2200.0 * 9.80665), kg / sec)
    maneuver.start_time = scb.EpochArray(t_m1, "TDB")
    maneuver.end_time = scb.EpochArray(t_m1_end, "TDB")
    maneuver.ux = scb.ArrayWUnits(u, None)
    maneuver.uy = scb.ArrayWUnits(0.0, None)
    maneuver.uz = scb.ArrayWUnits(u, None)

    sc = scb.Spacecraft("batch_seq_sc", -60000, scb.ArrayWUnits(2300.0, kg),
                        scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None))
    origin = scb.CelestialBody.from_constants("SUN")

    def prop(definition, t0, t1, burn=None):
        state = scb.StateArray(epoch=scb.EpochArray(t0, sys="TDB"), origin=origin,
                               state=definition)
        force_model = scb.ForceModelTranslation(
            primary_body=sc, cannonball_SRP=True, finite_burn=burn is not None,
            maneuver=burn)
        return scb.Propagator(
            primary_body=sc, state_vector=state,
            tspan=scb.EpochArray(np.arange(t0, t1 + 10.0, 10.0), sys="TDB"),
            force_models=force_model)

    coast = (scb.StateDefinition()
             .position(sc, scb.ArrayWFrame(
                 np.array([-2.65643964e08, 1.88333350e08, 1.01979698e08]), km, J2000))
             .velocity(sc, scb.ArrayWFrame(
                 np.array([-9.07510541, -12.48432651, -4.46105273]), km / sec, J2000)))
    burn = (scb.StateDefinition.empty_pv(sc, frame=J2000)
            .param("dRA1", sc, scb.ArrayWFrame(0.4, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("dDEC1", sc, scb.ArrayWFrame(0.3, rad, J2000),
                   estimation="estimated", dynamics="static")
            .param("k_thrust1", sc, scb.ArrayWFrame(0.9, None, None),
                   estimation="considered", dynamics="static"))

    sequence = scb.MissionSequence("batch_sequence")
    sequence.addLeg(name="coast", prop=prop(coast, start_time, t_m1))
    sequence.addFiniteBurn(name="burn", prop=prop(burn, t_m1, t_m1_end, maneuver))
    sequence.addLeg(name="final coast", prop=prop(
        scb.StateDefinition.empty_pv(sc, frame=J2000), t_m1_end, end_time))

    gs = scb.GroundStation("DSS-14")
    meas_epochs = scb.EpochArray(np.arange(start_time + 30.0, end_time - 30.0, 60.0),
                                 sys="TDB")
    measurements = scb.MeasurementSpec.from_dict([
        {"model": scb.RangeIdeal("R", gs, sigma=scb.ArrayWUnits(1e-3, km),
                                 sequence_definition=sequence),
         "dataset_name": "R", "epochs": meas_epochs},
        {"model": scb.RangeRateIdeal("RR", gs, sigma=scb.ArrayWUnits(1e-6, km / sec),
                                     sequence_definition=sequence),
         "dataset_name": "RR", "epochs": meas_epochs},
    ])

    settings = scb.FilterSettings(
        initial_covariance=scb.CovarianceMatrix(
            [scb.ArrayWUnits(100.0, km)] * 3 + [scb.ArrayWUnits(0.01, km / sec)] * 3,
            scb.EpochArray(start_time, sys="TDB"), from_list=True,
            extra_apriori_sigmas={
                "dRA1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                "dDEC1": scb.ArrayWUnits(np.deg2rad(0.5), rad),
                "k_thrust1": scb.ArrayWUnits(0.05, None),
            }),
        output=scb.OutputSettings(
            solution_output_path=str(out_dir), solution_output_name="solution",
            save_reference_trajectory=False, save_estimated_trajectory=False),
        covariance_analysis=True,
    )
    traj_name = f"batch_sequence_{filter_cls.__name__}.bsp"
    batch = filter_cls(propagator=sequence, settings=settings, measurements=measurements,
                       traj_name=traj_name, traj_dir=str(out_dir))
    batch.fit(max_iterations=1, verbose=False, traj_name=traj_name)
    return batch


@pytest.mark.parametrize("filter_cls", [scb.LSB, scb.SRIFB])
def test_repeat_compute_on_sequence(filter_cls, tmp_path):
    """
    A pass ends on the last leg with its state size, consider partition and
    node flag still set, and fit does not reset them after its last pass. A
    further _compute must start again from the first leg: with the leg map-back
    it reproduces the fit pass, and without it each leg keeps its own size.
    """
    batch = _run_burn_sequence(filter_cls, tmp_path)
    P_fit = [np.array(P, float) for P in batch.covariance_history]
    dx_fit = [np.array(dx, float) for dx in batch.state_deviation_history]

    batch._compute(printOutput=False)
    assert len(batch.covariance_history) == len(P_fit)
    for P, P_ref in zip(batch.covariance_history, P_fit):
        np.testing.assert_array_equal(P, P_ref)
    for dx, dx_ref in zip(batch.state_deviation_history, dx_fit):
        np.testing.assert_array_equal(dx, dx_ref)

    batch._compute(map_back_sequence=False, printOutput=False)
    times = batch.timeline_times
    in_burn = (times >= batch.events_epochs[0]) & (times < batch.events_epochs[1])
    assert in_burn.any() and not in_burn.all()
    # the burn leg adds dRA1, dDEC1 (estimated) and k_thrust1 (considered)
    for P, J, dx, burn in zip(batch.covariance_history, batch.covariance_consider_history,
                              batch.state_deviation_history, in_burn):
        assert np.shape(P) == ((9, 9) if burn else (6, 6))
        assert (J is not None) == burn
        assert np.isfinite(dx).all()
