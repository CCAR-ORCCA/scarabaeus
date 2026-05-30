# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization():
    """
        Verifies that object is constructed correctly.
    """
    pytest.skip()



    
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
# spiceypy.kclear()

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
# #target_trajectory = scb.Trajectory('TEST_target.bsp', pos1, vel1, epochs, ref_frame_name, origin_body, target)
# #observer_trajectory = scb.Trajectory('TEST_obs.bsp', pos2, vel2, epochs, ref_frame_name, origin_body, observer)

# # Initialize Range Ideal Models
# #range_rate_model_sigma = scb.RangeRateIdealModel("RANGE_RATE_TEST_NOISY", observer, 1)
# #range_rate_model_no_sigma = scb.RangeRateIdealModel("RANGE_RATE_TEST", observer)


# #---------------------------#
# #           Tests           #
# #---------------------------#
# # Fixtures (write fixtures for tests here)
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
# def range_residiuals(range_measurements, computed_range_measurements):
#     residiuals = []
#     for i, j in zip(range_measurements, computed_range_measurements):
#         residiuals.append(j - i)
#     return residiuals

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
#     print(range_rates)
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

# @pytest.fixture
# def computed_range_rate_measurements(computed_range_measurements, perturbed_relative_positions, perturbed_relative_velocities):
#     computed_range_rates = []
#     for i, j, k in zip(computed_range_measurements, perturbed_relative_positions, perturbed_relative_velocities):
#         rr = np.dot(j.values, k.values) / i.values
#         computed_range_rates.append(scb.ArrayWUnits(rr, km/sec))
#     print(computed_range_rates)
#     return computed_range_rates

# @pytest.fixture
# def range_rate_residiuals(range_rate_measurements, computed_range_rate_measurements):
#     residiuals = []
#     for i, j in zip(range_rate_measurements, computed_range_rate_measurements):
#         residiuals.append(j - i)
#     return residiuals

# @pytest.fixture
# def generate_range_dataset() -> scb.MeasurementDataSet:
#     """
#     This fixtures returns a range dataset with residuals.

#     Returns:
#         scb.MeasurementDataSet: Dataset to be passed into the FilterDataManager
#     """
#     # Initialize a range model
#     range_model_no_sigma = scb.RangeIdealModel("RANGE_TEST", observer)

#     # Promote perturbed trajectory priority and compute measurements
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target_pert.bsp")     
#     range_measurements_perturbed = range_model_no_sigma.get_range_measurements_in_timeframe(
#         target,
#         epochs
#     )

#     # Promote unperturbed trajectory priority and compute dataset
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")
#     range_dataset = range_model_no_sigma.generate_filter_dataset(
#         dataset_name = "TEST_DATASET_RANGE",
#         target = target,
#         epoch_list = epochs,
#         observed_measurements = range_measurements_perturbed
#     )
#     #print(range_dataset.data)
#     return range_dataset

# @pytest.fixture
# def generate_range_rate_dataset() -> scb.MeasurementDataSet:
#     """
#     This fixtures returns a range rate dataset with residuals.

#     Returns:
#         scb.MeasurementDataSet: Dataset to be passed into the FilterDataManager
#     """
#     # Initialize the range rate model
#     range_rate_model_no_sigma = scb.RangeRateIdealModel("RANGE_RATE_TEST", observer)

#     # Promote perturbed trajectory priority and compute measurements
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target_pert.bsp")
#     range_rate_measurements_perturbed = range_rate_model_no_sigma.get_range_rate_measurements_in_timeframe(
#         target,
#         epochs
#     )

#     # Promote unperturbed trajectory priority and compute dataset
#     scb.SpiceManager.increase_kernel_priority("./kernels/TEST_target.bsp")
#     range_rate_dataset = range_rate_model_no_sigma.generate_filter_dataset(
#         dataset_name = "TEST_DATASET_RANGE_RATE",
#         target = target,
#         epoch_list = epochs,
#         observed_measurements = range_rate_measurements_perturbed
#     )
#     #print(range_rate_dataset.data)
#     return range_rate_dataset


# @pytest.fixture
# def range_datalist(range_measurements, computed_range_measurements, range_residiuals, range_partials):
#     datalist = []
#     for t, o, c, r, p in zip(epochs, range_measurements, computed_range_measurements, range_residiuals, range_partials):
#         datalist.append([
#             t.time.values.tolist(),
#             o.values.tolist(),
#             c.values.tolist(),
#             r.values.tolist(),
#             p,
#             None,
#             observer,
#             "range",
#             "TEST_DATASET_RANGE"
#         ])
    
#     return datalist

# @pytest.fixture
# def range_rate_datalist(range_rate_measurements, computed_range_rate_measurements, range_rate_residiuals, range_rate_partials):
#     datalist = []
#     for t, o, c, r, p in zip(epochs, range_rate_measurements, computed_range_rate_measurements, range_rate_residiuals, range_rate_partials):
#         datalist.append([
#             t.time.values.tolist(),
#             o.values.tolist(),
#             c.values.tolist(),
#             r.values.tolist(),
#             p,
#             None,
#             observer,
#             "range rate",
#             "TEST_DATASET_RANGE_RATE"
#         ])
    
#     return datalist

# @pytest.fixture
# def combined_datalist(range_datalist:pytest.fixture, range_rate_datalist:pytest.fixture):
#     combined = []
#     for i in range(len(range_datalist)):
#         combined.append([
#             range_datalist[i][0],
#             np.array([range_datalist[i][1], range_rate_datalist[i][1]]),
#             np.array([range_datalist[i][2], range_rate_datalist[i][2]]),
#             np.array([range_datalist[i][3], range_rate_datalist[i][3]]),
#             np.array([range_datalist[i][4], range_rate_datalist[i][4]]),
#             np.array([range_datalist[i][5], range_rate_datalist[i][5]]),
#             [range_datalist[i][6], range_rate_datalist[i][6]],
#             [range_datalist[i][7], range_rate_datalist[i][7]],
#             [range_datalist[i][8], range_rate_datalist[i][8]]
#         ])

#     return combined

# def test_convert_dataset_to_list(generate_range_dataset: pytest.fixture, range_datalist:pytest.fixture):
#     """
#     Test that the convert_dataset_to_list method converts a MeasurementDataset 
#     into a list of data.

#     Args:
#         generate_range_dataset (pytest.fixture): range data
#     """
#     data_list = scb.FilterDataManager.convert_dataset_to_list(generate_range_dataset)
    
#     # We need to perform the tests this way because the list generated is not of
#     # standard dimensions so the test suite cannot perform tests correctly.
#     for i in range(len(range_datalist)):
#         np.testing.assert_equal(data_list[i][0], range_datalist[i][0])
#         np.testing.assert_equal(data_list[i][1], range_datalist[i][1])
#         np.testing.assert_equal(data_list[i][2], range_datalist[i][2])
#         np.testing.assert_equal(data_list[i][3], range_datalist[i][3])
#         np.testing.assert_array_equal(data_list[i][4], range_datalist[i][4])
#         np.testing.assert_equal(data_list[i][5], range_datalist[i][5])
#         np.testing.assert_equal(data_list[i][6], range_datalist[i][6])
#         np.testing.assert_equal(data_list[i][7], range_datalist[i][7])
#         np.testing.assert_equal(data_list[i][8], range_datalist[i][8])


# def test_combine_multiple_datasets_combined_data(generate_range_dataset: pytest.fixture, generate_range_rate_dataset:pytest.fixture, combined_datalist:pytest.fixture):
#     """
#     Test that the combine_multiple_datasets method correctly combines datasets.

#     Args:
#         generate_range_dataset (pytest.fixture): Range data
#         generate_range_rate_dataset (pytest.fixture): Range rate data
#     """
#     datasets = [generate_range_dataset, generate_range_rate_dataset]
#     combined_datasets, _ = scb.FilterDataManager.combine_multiple_datasets(datasets)

#     for i in range(len(combined_datalist)):
#         np.testing.assert_equal(combined_datasets[i][0], combined_datalist[i][0])
#         np.testing.assert_array_equal(combined_datasets[i][1], combined_datalist[i][1])
#         np.testing.assert_array_equal(combined_datasets[i][2], combined_datalist[i][2])
#         np.testing.assert_array_equal(combined_datasets[i][3], combined_datalist[i][3])
#         np.testing.assert_array_equal(combined_datasets[i][4], combined_datalist[i][4])
#         np.testing.assert_array_equal(combined_datasets[i][5], combined_datalist[i][5])
#         np.testing.assert_array_equal(combined_datasets[i][6], combined_datalist[i][6])
#         np.testing.assert_array_equal(combined_datasets[i][7], combined_datalist[i][7])
#         np.testing.assert_array_equal(combined_datasets[i][8], combined_datalist[i][8])

# def test_combine_multiple_datasets_names(generate_range_dataset: pytest.fixture, generate_range_rate_dataset:pytest.fixture, combined_datalist:pytest.fixture):
#     """
#     Test that the combine_multiple_datasets method correctly outputs the names of the
#     combined datasets.

#     Args:
#         generate_range_dataset (pytest.fixture): Range data
#         generate_range_rate_dataset (pytest.fixture): Range rate data
#     """
#     datasets = [generate_range_dataset, generate_range_rate_dataset]
#     _, names = scb.FilterDataManager.combine_multiple_datasets(datasets)

#     np.testing.assert_equal(names[0], "TEST_DATASET_RANGE")
#     np.testing.assert_equal(names[1], "TEST_DATASET_RANGE_RATE")

# def test_combine_multiple_datasets_none_input():

#     np.testing.assert_equal(
#         scb.FilterDataManager.combine_multiple_datasets(None),
#         None
#     )

# def test_combine_multiple_datasets_empty_list_input():
#     np.testing.assert_equal(
#         scb.FilterDataManager.combine_multiple_datasets([]),
#         None
#     )

# def test_combine_multiple_datasets_non_list_input():
#     np.testing.assert_equal(
#         scb.FilterDataManager.combine_multiple_datasets("TEST"),
#         None
#     )  

# def test_combine_multiple_datasets_one_dataset_input(generate_range_dataset:pytest.fixture, range_datalist:pytest.fixture):
#     np.testing.assert_equal(
#         scb.FilterDataManager.combine_multiple_datasets([generate_range_dataset]),
#         range_datalist
#     )  

