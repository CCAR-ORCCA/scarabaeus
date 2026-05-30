# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
import numpy as np

(J2000, ITRF93, ECLIPJ2000, IAUEARTH) = scb.Frame.generate_common_frames()
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


@pytest.fixture
def camera(sc):
    """
    Fixture to create a sample camera instrument for testing.
    """
    cam_fov_angular = scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad)
    cam_fov_pixels = (2000, 2000)
    cam_ifov_angle = scb.ArrayWUnits(np.array([25 * 10**-6, 20 * 25**-6]), rad)
    cam_focal_length = scb.ArrayWUnits(150 * 10**-6, km)

    return scb.Camera(
        name="ORCCA_camera",
        associated_body_spice_id=sc.spice_id,
        camera_frame=J2000,
        fov_angular=cam_fov_angular,
        fov_pixels=cam_fov_pixels,
        ifov_angle=cam_ifov_angle,
        focal_length=cam_focal_length,
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(camera):
    """
    Verifies that object is constructed correctly.
    """
    centroiding_sigma = scb.ArrayWUnits(np.array([1, 1]), None)

    Centroid = scb.CentroidingIdeal(
        name="Opnav model", camera=camera, sigma=centroiding_sigma
    )

    assert Centroid.name == "Opnav model"
    assert Centroid._camera == camera
    assert Centroid._sigma == centroiding_sigma


def test_measurement_dataset_val_logic(sc, epochs, camera):
    """Verify that the function prevents invalid input combinations."""
    sigma = scb.ArrayWUnits(np.array([1, 1]), None)
    meas_model = scb.CentroidingIdeal(
        name="CentroidIdeal", camera=camera, sigma=sigma
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
