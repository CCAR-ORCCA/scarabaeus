# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =================================#
#  Unit Tests for the SRIF class  #
# =================================#
import src.scarabaeus as scb
import numpy as np
import pytest
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

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
        process_noise=scb.ProcessNoiseSettings(type="SNC", Q_cont=Q_cont),)
    SRIF_it1 = scb.SRIF(
        propagator=propagator,
        settings=settings,
        measurements=measurements,
        traj_name="orbiter_ref_it1.bsp", )

    assert SRIF_it1.propagator == propagator
    assert SRIF_it1.measurements == measurements
    assert SRIF_it1.settings == settings
    assert SRIF_it1.initial_covariance == settings.initial_covariance
