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
def test_initialization(GS1):
    """
    Verifies that object is constructed correctly.
    """
    range_sigma = scb.ArrayWUnits(1e-3, km)
    Range_GS1 = scb.RangeIdeal(
        name="GS1 Ideal Range Model", instrument=GS1, sigma=range_sigma
    )

    assert Range_GS1._name == "GS1 Ideal Range Model"
    assert Range_GS1._instrument == GS1
    assert Range_GS1._sigma == range_sigma


def test_measurement_dataset_val_logic(sc, epochs, GS1):
    """Verify that the function prevents invalid input combinations."""
    meas_model = scb.RangeIdeal(
        name="test_range_ideal", instrument=GS1, sigma=scb.ArrayWUnits(0.001, km)
    )

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock
        )


# # Versioning
# __verison__ = "0.0.0"
# __author__ = "Kian Shakerin"

# # Imports
# import sys
# import os

# import pytest
# import numpy as np
# import spiceypy

# sys.path.append("./src/")
# import scarabaeus as scb

# #---------------------------#
# #           Setup           #
# #---------------------------#
# # Perform necessary setup here
# # Generate common units
# kg, km, sec, rad, meter, AU, min, hour, day, deg, newton = scb.UnitsArray.generate_common_units()

# # Clearning the kernel pool before the test to eliminate any load errors
# # spiceypy.kclear()

# # Define the bodies for the test
# target = scb.Body("TEST_target", spice_id = -9998)
# observer = scb.GroundStation("TEST_obs", spice_id = -9999)

# # Define the reference frame and origin body for the test
# ref_frame_name = 'J2000'
# origin_body = 'Earth'

# # Define position and velocity arrays defining the trajectory data points
# pos1 = scb.ArrayWUnits(
#     np.array([[1.0, -1.0, 0.0],
#             [2.0, -2.0, 0.0],
#             [3.0, -3.0, 0.0],
#             [4.0, -4.0, 0.0]
#     ]),
#     km
# )

# vel1 = scb.ArrayWUnits(
#     np.array([[1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0]
#     ]),
#     km/sec
# )

# # make a "perturbed" trajectory
# pos1_pert = scb.ArrayWUnits(
#     np.array([[1.1, -1.1, 0.0],
#             [2.1, -2.1, 0.0],
#             [3.1, -3.1, 0.0],
#             [4.1, -4.1, 0.0]
#     ]),
#     km
# )

# vel1_pert = scb.ArrayWUnits(
#     np.array([[1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0],
#             [1.0, -1.0, 0.0]
#     ]),
#     km/sec
# )

# pos2 = scb.ArrayWUnits(
#     np.array([[0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#     ]),
#     km)

# vel2 = scb.ArrayWUnits(
#     np.array([[0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0]
#     ]),
#     km/sec
# )


# # Define epochs corresponding to the trajectory data points
# times = scb.ArrayWUnits(np.array([0.0, 1.0, 2.0, 3.0]), sec)
# epochs = scb.EpochArray(times, timeFrame = 'TDB')

# # Initialize the trajectories
# # target_trajectory = scb.Trajectory('TEST_target.bsp', pos1, vel1, epochs, ref_frame_name, origin_body, target)
# # observer_trajectory = scb.Trajectory('TEST_obs.bsp', pos2, vel2, epochs, ref_frame_name, origin_body, observer)

# # Adjust kernel priority
# # scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")
# # scb.SpiceManager.increase_kernel_priority("./kernels/TEST_obs.bsp")

# # Initialize Range Ideal Models
# range_model_sigma = scb.RangeIdealModel("RANGE_TEST_NOISY", observer, 1)
# range_model_no_sigma = scb.RangeIdealModel("RANGE_TEST", observer)

# #---------------------------#
# #           Tests           #
# #---------------------------#
# # Fixtures (write fixtures for tests here)
# @pytest.fixture
# def relative_positions():
#     return pos1 - pos2

# @pytest.fixture
# def range_measurements():
#     return [
#         scb.ArrayWUnits(np.sqrt(1**2 + (-1)**2), km),
#         scb.ArrayWUnits(np.sqrt(2**2 + (-2)**2), km),
#         scb.ArrayWUnits(np.sqrt(3**2 + (-3)**2), km),
#         scb.ArrayWUnits(np.sqrt(4**2 + (-4)**2), km)
#     ]

# @pytest.fixture
# def range_partials(relative_positions, range_measurements):
#     """
#     Test that the range partials computed are correct.
#     For now our state size is 6 so our partials should be
#     a 1x6 ArrayWUnits object.
#     Range partials are of the form:
#     [rho_x/rho_mag, rho_y/rho_mag, rho_z/rho_mag, 0, 0, 0]
#     This function compares the partials at t=0 (rather epochs[0])
#     """
#     partials = []
#     for i in range(4):
#         partials.append([
#             relative_positions[i][0].values.tolist() / range_measurements[i].values.tolist(),
#             relative_positions[i][1].values.tolist() / range_measurements[i].values.tolist(),
#             relative_positions[i][2].values.tolist() / range_measurements[i].values.tolist(),
#             0.0,
#             0.0,
#             0.0
#         ])

#     return partials

# @pytest.fixture
# def computed_range_measurements():
#     return [
#         scb.ArrayWUnits(np.sqrt(1.1**2 + (-1.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(2.1**2 + (-2.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(3.1**2 + (-3.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(4.1**2 + (-4.1)**2), km)
#     ]

# @pytest.fixture
# def residiuals(range_measurements, computed_range_measurements):
#     residiuals = []
#     for i, j in zip(range_measurements, computed_range_measurements):
#         residiuals.append(j - i)
#     return residiuals


# def test_sigma_property():
#     """
#     Test that the sigma class property is set correctly.
#     """
#     np.testing.assert_equal(range_model_sigma.sigma, 1)

# def test_compute_range():
#     """
#     Test that the compute_range function return the correct range value.
#     """
#     np.testing.assert_equal(
#         range_model_no_sigma.compute_range(target, epochs[0]),
#         scb.ArrayWUnits(np.sqrt(2), km)
#     )

# def test_get_range_measurements_in_timeframe_epochlist(range_measurements):
#     """
#     Test that the compute_range_measurements_in_timeframe function returns
#     the correct range values when given an EpochArray object. No noise is
#     added to the measurement.
#     """
#     np.testing.assert_array_equal(
#         range_model_no_sigma.get_range_measurements_in_timeframe(target, epoch_list=epochs).data["range"],
#         range_measurements
#     )

# def test_get_range_measurements_in_timeframe_epoch_start_end(range_measurements):
#     """
#     Test that thecompute_range_measurements_in_timeframe function returns
#     the correct range values when given a start epoch and an end epoch.
#     In this test we use the default timestep of t=1. No noise is added to the
#     measurements.
#     """
#     epoch_start = epochs[0]
#     epoch_end = epochs[-1]
#     np.testing.assert_array_equal(
#         range_model_no_sigma.get_range_measurements_in_timeframe(target, None, epoch_start, epoch_end).data["range"],
#         range_measurements
#     )

# def test_compute_range_partials(range_partials):
#     """
#     Test that the compute_range_partials function returs the correct
#     values for the partials at a given epoch.
#     """
#     epoch = epochs[0]
#     np.testing.assert_array_equal(
#         range_model_no_sigma.compute_range_partials(target, epoch),
#         range_partials[0]
#     )

# def test_compute_range_partials_in_timeframe_epochlist(range_partials):
#     """
#     Test that the copute range partials in timeframe function returns
#     the correct ragne partials when given an EpochArray object.
#     Args:
#         range_partials (_type_): Fixture with the expected partials
#     """
#     np.testing.assert_array_equal(
#         range_model_no_sigma.get_range_partials_in_timeframe(target, epochs).data['range partials'],
#         range_partials
#     )

# def test_compute_range_partials_in_timeframe_epoch_start_end(range_partials):
#     """
#     Test that the copute range partials in timeframe function returns
#     the correct ragne partials when given an EpochArray object.
#     Args:
#         range_partials (_type_): Fixture with the expected partials
#     """
#     epoch_start = epochs[0]
#     epoch_end = epochs[-1]
#     np.testing.assert_array_equal(
#         range_model_no_sigma.get_range_partials_in_timeframe(target, None, epoch_start, epoch_end).data['range partials'],
#         range_partials
#     )


# # def test_compute_light_time():
# #     pass

# # TODO: Write the filter datatset test.
# # This test wwe will need to make a new "perturbed" trajector

# def test_generate_filter_data(range_measurements, range_partials, computed_range_measurements, residiuals):
#     # Instantiate a perturbed trajectory
#     #perturbed_trajectory

#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target_pert.bsp")

#     range_measurements_perturbed = range_model_no_sigma.get_range_measurements_in_timeframe(
#         target,
#         epochs
#     )

#     # Furnsh the unperturbed trajectory
#     #spiceypy.furnsh("./kernels/TEST_target.bsp")
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")

#     # Compute the filter dataset
#     dataset = range_model_no_sigma.generate_filter_dataset(
#         dataset_name = "TEST_DATASET",
#         target = target,
#         epoch_list = epochs,
#         observed_measurements = range_measurements_perturbed
#     )

#     # asserts
#     # Observed Measurements
#     np.testing.assert_array_equal(
#         dataset.data['observed'],
#         range_measurements
#     )
#     # Computed Measurements
#     np.testing.assert_array_equal(
#         dataset.data['computed'],
#         computed_range_measurements
#     )
#     # Residuals
#     np.testing.assert_array_equal(
#         dataset.data['residuals'],
#         residiuals
#     )
#     # Partials
#     np.testing.assert_array_equal(
#         dataset.data['partials'],
#         range_partials
#     )
