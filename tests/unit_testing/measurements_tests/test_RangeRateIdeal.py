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
    rangerate_sigma = scb.ArrayWUnits(1e-5, km / sec)
    name = "GS1 Ideal RangeRate Model"
    RangeRate_GS1 = scb.RangeRateIdeal(name=name, instrument=GS1, sigma=rangerate_sigma)

    assert RangeRate_GS1._name == name
    assert RangeRate_GS1._instrument == GS1
    assert RangeRate_GS1._sigma == rangerate_sigma


def test_measurement_dataset_val_logic(sc, epochs, GS1):
    """Verify that the function prevents invalid input combinations."""
    meas_model = scb.RangeRateIdeal(
        name="test_range_rate_ideal",
        instrument=GS1,
        sigma=scb.ArrayWUnits(0.001, km / sec),
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
# #spiceypy.kclear()

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
# # scb.SpiceManager.increase_kernel_priority("./kernels/TEST_obs.bsp")
# # scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")

# # Initialize Range Ideal Models
# range_rate_model_sigma = scb.RangeRateIdealModel("RANGE_RATE_TEST_NOISY", observer, 1)
# range_rate_model_no_sigma = scb.RangeRateIdealModel("RANGE_RATE_TEST", observer)

# #---------------------------#
# #           Tests           #
# #---------------------------#
# # Fixtures (write fixtures for tests here)
# @pytest.fixture
# def relative_positions():
#     return pos1 - pos2

# @pytest.fixture
# def relative_velocities():
#     return vel1 - vel2

# @pytest.fixture
# def perturbed_relative_positions():
#     return pos1_pert - pos2

# @pytest.fixture
# def perturbed_relative_velocities():
#     return vel1_pert - vel2

# @pytest.fixture
# def range_measurements():
#     return [
#         scb.ArrayWUnits(np.sqrt(1**2 + (-1)**2), km),
#         scb.ArrayWUnits(np.sqrt(2**2 + (-2)**2), km),
#         scb.ArrayWUnits(np.sqrt(3**2 + (-3)**2), km),
#         scb.ArrayWUnits(np.sqrt(4**2 + (-4)**2), km)
#     ]

# @pytest.fixture
# def computed_range_measurements():
#     return [
#         scb.ArrayWUnits(np.sqrt(1.1**2 + (-1.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(2.1**2 + (-2.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(3.1**2 + (-3.1)**2), km),
#         scb.ArrayWUnits(np.sqrt(4.1**2 + (-4.1)**2), km)
#     ]


# @pytest.fixture
# def range_rate_measurements(range_measurements, relative_positions, relative_velocities):
#     """
#     Fixture to create range rate measurements to compare against
#     Args:
#         range_measurements (_type_): _description_
#         relative_positions (_type_): _description_
#         relative_velocities (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     range_rates = []
#     for i, j, k in zip(range_measurements, relative_positions, relative_velocities):
#         rr = np.dot(j.values, k.values) / i.values
#         range_rates.append(scb.ArrayWUnits(rr, km/sec))

#     return range_rates

# @pytest.fixture
# def range_rate_partials(range_measurements, range_rate_measurements, relative_positions, relative_velocities):
#     range_rate_partials = []
#     for i, j, k, l in zip(range_measurements, relative_positions, relative_velocities, range_rate_measurements):
#         rr_partial = np.array([
#             k.values[0] / i.values - (j.values[0] * l.values) / (i.values ** 2),
#             k.values[1] / i.values - (j.values[1] * l.values) / (i.values ** 2),
#             k.values[2] / i.values - (j.values[2] * l.values) / (i.values ** 2),
#             j.values[0] / i.values,
#             j.values[1] / i.values,
#             j.values[2] / i.values
#         ])
#         range_rate_partials.append(rr_partial)

#     return range_rate_partials

# # @pytest.fixture
# # def perturbed_trajectory():
# #     # make a "perturbed" trajectory
# #     # Clearning the kernel pool before the test to eliminate any load errors
# #     target_trajectory_perturbed = scb.Trajectory('TEST_target_pert.bsp', pos1_pert, vel1_pert, epochs, ref_frame_name, origin_body, target)
# #     yield target_trajectory_perturbed
# #     os.remove('./kernels/TEST_target_pert.bsp')


# @pytest.fixture
# def computed_range_rate_measurements(computed_range_measurements, perturbed_relative_positions, perturbed_relative_velocities):
#     computed_range_rates = []
#     for i, j, k in zip(computed_range_measurements, perturbed_relative_positions, perturbed_relative_velocities):
#         rr = np.dot(j.values, k.values) / i.values
#         computed_range_rates.append(scb.ArrayWUnits(rr, km/sec))

#     return computed_range_rates

# @pytest.fixture
# def range_rate_residiuals(range_rate_measurements, computed_range_rate_measurements):
#     residiuals = []
#     for i, j in zip(range_rate_measurements, computed_range_rate_measurements):
#         residiuals.append(j - i)
#     return residiuals

# def test_sigma_property():
#     """
#     Test that the sigma class property is set correctly
#     """
#     np.testing.assert_equal(range_rate_model_sigma.sigma, 1)

# def test_compute_range_rate(range_rate_measurements):
#     """
#     Test that the compute_range_rate function returns the correct range value.
#     """
#     np.testing.assert_equal(
#         range_rate_model_no_sigma.compute_range_rate(target, epochs[0]),
#         range_rate_measurements[0]
#     )

# def test_get_range_rate_measurements_in_timeframe_epochlist(range_rate_measurements):
#     """
#     Test that the compute_range_measurements_in_timeframe function returns
#     the correct range values when given an EpochArray object. No noise is
#     added to the measurement.
#     """
#     np.testing.assert_array_equal(
#         range_rate_model_no_sigma.get_range_rate_measurements_in_timeframe(target, epoch_list=epochs).data["range rate"],
#         range_rate_measurements
#     )

# def test_get_range_measurements_in_timeframe_epoch_start_end(range_rate_measurements):
#     """
#     Test that thecompute_range_rate_measurements_in_timeframe function returns
#     the correct range values when given a start epoch and an end epoch.
#     In this test we use the default timestep of t=1. No noise is added to the
#     measurements.
#     """
#     epoch_start = epochs[0]
#     epoch_end = epochs[-1]
#     np.testing.assert_array_equal(
#         range_rate_model_no_sigma.get_range_rate_measurements_in_timeframe(target, None, epoch_start, epoch_end).data["range rate"],
#         range_rate_measurements
#     )

# def test_compute_range_rate_partials(range_rate_partials):
#     """
#     Test that the compute_range_rate_partials function returs the correct
#     values for the partials at a given epoch.
#     """
#     epoch = epochs[0]
#     np.testing.assert_array_equal(
#         range_rate_model_no_sigma.compute_range_rate_partials(target, epoch),
#         range_rate_partials[0]
#     )

# def test_compute_range_rate_partials_in_timeframe_epochlist(range_rate_partials):
#     """
#     Test that the copute range rate partials in timeframe function returns
#     the correct ragne partials when given an EpochArray object.
#     Args:
#         range_partials (_type_): Fixture with the expected partials
#     """
#     np.testing.assert_array_equal(
#         range_rate_model_no_sigma.get_range_rate_partials_in_timeframe(target, epochs).data['range rate partials'],
#         range_rate_partials
#     )

# def test_compute_range_rate_partials_in_timeframe_epoch_start_end(range_rate_partials):
#     """
#     Test that the copute range rate partials in timeframe function returns
#     the correct ragne partials when given an EpochArray object.
#     Args:
#         range_partials (_type_): Fixture with the expected partials
#     """
#     epoch_start = epochs[0]
#     epoch_end = epochs[-1]
#     np.testing.assert_array_equal(
#         range_rate_model_no_sigma.get_range_rate_partials_in_timeframe(target, None, epoch_start, epoch_end).data['range rate partials'],
#         range_rate_partials
#     )

# def test_generate_filter_data(range_rate_measurements, range_rate_partials, computed_range_rate_measurements, range_rate_residiuals):
#     # Adjust kernel priority so the perterbed trajectory is higher.
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target_pert.bsp")

#     # Compute range rate measurements for the perturbed trajectory
#     range_rate_measurements_perturbed = range_rate_model_no_sigma.get_range_rate_measurements_in_timeframe(
#         target,
#         epochs
#     )

#     # Adjust kernel priority so the unperterbed trajectory is higher.
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")

#     # Compute the filter dataset
#     dataset = range_rate_model_no_sigma.generate_filter_dataset(
#         dataset_name = "TEST_DATASET",
#         target = target,
#         epoch_list = epochs,
#         observed_measurements = range_rate_measurements_perturbed
#     )

#     # asserts
#     # Observed Measurements
#     np.testing.assert_array_equal(
#         dataset.data['observed'],
#         range_rate_measurements
#     )
#     # Computed Measurements
#     np.testing.assert_array_equal(
#         dataset.data['computed'],
#         computed_range_rate_measurements
#     )
#     # Residuals
#     np.testing.assert_array_equal(
#         dataset.data['residuals'],
#         range_rate_residiuals
#     )
#     # Partials
#     np.testing.assert_array_equal(
#         dataset.data['partials'],
#         range_rate_partials
#     )
