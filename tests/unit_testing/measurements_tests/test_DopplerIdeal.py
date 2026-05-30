# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest

km, sec, kg, hr = scb.Units.get_units(["km", "sec", "kg", "hr"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def epochs():
    return scb.EpochArray.interval(
        "2028 Feb 1 00:00:00", "2028 Feb 2 00:00:00", scb.ArrayWUnits(1, hr), sys="TDB"
    )


@pytest.fixture
def origin():
    return scb.Body("Earth", spice_id=399)


@pytest.fixture
def GS1():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")


@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp
    )


@pytest.fixture
def antenna():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.Antenna(name="Antenna_for_doppler", spice_id=-1200)


@pytest.fixture
def doppler_transmit_frequency():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.ArrayWUnits(8.8 * 10**9, sec**-1)


@pytest.fixture
def doppler_sigma():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.ArrayWUnits(1e-3, sec)


@pytest.fixture
def eps_dict():
    """
    Fixture to create an eps_dict containing ArrayWUnits objects
    for testing the _build_perturbation_vector method.
    """

    return {
        "dx": scb.ArrayWUnits(np.array([1.5]), km),
        "dy": scb.ArrayWUnits(np.array([2.0]), km),
        "dz": scb.ArrayWUnits(np.array([0.5]), km),
        "dvx": scb.ArrayWUnits(np.array([0.001]), km * sec**-1),
        "dvy": scb.ArrayWUnits(np.array([0.002]), km * sec**-1),
        "dvz": scb.ArrayWUnits(np.array([0.003]), km * sec**-1),
    }


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(antenna, GS1, doppler_transmit_frequency, doppler_sigma):
    """
    Verifies that object is constructed correctly.
    """
    name = "doppler_test"
    Doppler_GS1 = scb.DopplerIdeal(
        name=name,
        instrument=GS1,
        sigma=doppler_sigma,
        antenna_name=antenna.name,
        doppler_transmit_frequency=doppler_transmit_frequency,
    )
    assert Doppler_GS1._name == name
    assert Doppler_GS1._instrument == GS1
    assert Doppler_GS1._sigma == doppler_sigma
    assert Doppler_GS1._antenna_name == antenna.name
    assert Doppler_GS1._doppler_transmit_frequency == doppler_transmit_frequency


def test_update_reference_state(antenna, GS1, origin, doppler_transmit_frequency, doppler_sigma, epochs):
    """
    Verifies that the reference state is updated correctly.
    """
    name = "doppler_test"
    meas = scb.DopplerIdeal(
        name=name,
        instrument=GS1,
        sigma=doppler_sigma,
        antenna_name=antenna.name,
        doppler_transmit_frequency=doppler_transmit_frequency,
    )
    x0_vals = [
        7000.0,
        0.0,
        0.0,
        0.0,
        7.5,
        0.0,
    ]  # Example state vector (position in km, velocity in km/s)
    pos_0 = scb.ArrayWFrame(x0_vals[0:3], km, J2000)
    vel_0 = scb.ArrayWFrame(x0_vals[3:], km / sec, J2000)

    new_state = scb.StateArray(
        epoch=epochs[0],
        origin=origin,
        state=scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0),
    )
    state0 = meas._reference_state_vector
    meas.update_reference_state(new_state)

    assert meas._reference_state_vector is not state0  # Ensure it's a new object
    assert meas._reference_state_vector == new_state


def test_build_perturbation(
    eps_dict, antenna, GS1, doppler_transmit_frequency, doppler_sigma
):
    """
    Verifies that the perturbation is built correctly.
    """
    name = "doppler_test"
    meas = scb.DopplerIdeal(
        name=name,
        instrument=GS1,
        sigma=doppler_sigma,
        antenna_name=antenna.name,
        doppler_transmit_frequency=doppler_transmit_frequency,
    )
    delta = meas._build_perturbation_vector(eps_dict)

    assert isinstance(delta, scb.ArrayWUnits)
    assert delta == scb.ArrayWUnits(
        np.array([1.5, 2.0, 0.5, 0.001, 0.002, 0.003]),
        [km, km, km, km * sec**-1, km * sec**-1, km * sec**-1],
    )


def test_validation_logic(sc, epochs, GS1, antenna):
    """Verify that the function prevents invalid input combinations."""
    sigma = scb.ArrayWUnits(1e-3, sec**-1)
    meas_model = scb.DopplerIdeal(
        name="dopplerIdeal",
        instrument=GS1,
        sigma=sigma,
        antenna_name=antenna.name,
        doppler_transmit_frequency=scb.ArrayWUnits(8.8 * 10**9, sec**-1),
    )  # Example instantiation

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock
        )


def test_compute_measurement():
    pytest.skip("NOT IMPLEMENTED")


def test_get_measurements_in_time_frame():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_h_tilde_pos():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_h_tilde_vel():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_h_tilde_eta_srp():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_h_tilde_gs_location():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_h_tilde_range_bias():
    pytest.skip("NOT IMPLEMENTED")


def test_compute_measurement_partials():
    pytest.skip("NOT IMPLEMENTED")


def test_get_partials_in_time_frame():
    pytest.skip("NOT IMPLEMENTED")


def test_generate_filter_dataset():
    pytest.skip("NOT IMPLEMENTED")
