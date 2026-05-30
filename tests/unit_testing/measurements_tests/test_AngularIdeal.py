# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest

kg, km, sec, rad, hr = scb.Units.get_units(["kg", "km", "sec", "rad", "hr"])


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def epochs():

    return scb.EpochArray.interval(
        "2028 Feb 1 00:00:00", "2028 Feb 2 00:00:00", scb.ArrayWUnits(1, hr), sys="TDB"
    )


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


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(sc):
    """
    Verifies that object is constructed correctly.
    """
    sigma = scb.ArrayWUnits([10e-6, 10e-6], rad)
    meas_model = scb.AngularIdeal(
        name="AngularIdeal", observer=sc, sigma=sigma
    )  # Example instantiation

    assert meas_model._name == "AngularIdeal"
    assert meas_model._instrument == sc
    assert meas_model._sigma == sigma


def test_validation_logic(sc, epochs):
    """Verify that the function prevents invalid input combinations."""
    sigma = scb.ArrayWUnits([10e-6, 10e-6], rad)
    meas_model = scb.AngularIdeal(
        name="AngularIdeal", observer=sc, sigma=sigma
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
