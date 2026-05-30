# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest

km = scb.Units.get_units('km')

#--------------#
# region Tests #
#--------------#
def test_initialization():
    """
        Verifies that object is constructed correctly.
    """
    noise = scb.Noise()

def test_generate_AWGN():
    """
        Verifies that AWGN is generated correctly.
    """
    noise = scb.Noise()
    awgn = noise.generate_AWGN(mu=0, sigma=1, count=5)
    assert type(awgn) == np.ndarray
    assert len(awgn) == 5

def test_generate_AWGN_with_units():
    """
        Verifies that AWGN with units is generated correctly.
    """
    noise = scb.Noise()
    awgn_with_units = noise.generate_AWGN_with_units(mu=0, sigma=1, units= km, count=5)
    assert type(awgn_with_units) == scb.ArrayWUnits
    assert awgn_with_units.size == 5
    assert awgn_with_units.units == km

def test_apply_AWGN():
    """
        Verifies that AWGN is applied correctly.
    """
    noise = scb.Noise()
    data = scb.ArrayWUnits(np.array([1,2,3,4,5]), km)
    data_with_noise = noise.apply_AWGN(data=data, mu=0, sigma=1)
    assert type(data_with_noise) == scb.ArrayWUnits
    assert data_with_noise.size == 5
    assert data_with_noise.units == km






# # Versioning
# __verison__ = "0.0.0"
# __author__ = "Kian Shakerin"

# # Imports
# import sys

# import pytest
# import numpy as np

# sys.path.append("./src/")
# import scarabaeus as scb

# #---------------------------#
# #           Setup           #
# #---------------------------#
# # Generate common units
# kg, km, sec, rad, meter, AU, min, hour, day, deg, Newton = scb.UnitsArray.generate_common_units()

# #---------------------------#
# #           Tests           #
# #---------------------------#
# # NOTE: This may have been overkill...

# # Single AWGN no units
# @pytest.fixture
# def AWGN_single():
#     return scb.Noise().generate_AWGN(
#         mu = 0,
#         sigma = 1
#     )

# def test_generate_AWGN_type_single(AWGN_single):
#     assert type(AWGN_single) == np.ndarray

# def test_generate_AWGN_size_single(AWGN_single):
#     assert len(AWGN_single) == 1

# def test_generate_AWGN_value_type_single(AWGN_single):
#     assert all([type(n) == np.float64 for n in AWGN_single])

# # Multiple AWGN no units
# @pytest.fixture
# def AWGN_multiple():
#     return scb.Noise().generate_AWGN(
#         mu = 0,
#         sigma = 1,
#         count = 3
#     )

# def test_generate_AWGN_type_single(AWGN_multiple):
#     assert type(AWGN_multiple) == np.ndarray

# def test_generate_AWGN_type_multiple(AWGN_multiple):
#     assert all([type(n) == np.float64 for n in AWGN_multiple])

# def test_generate_AWGN_length(AWGN_multiple):
#     assert len(AWGN_multiple) == 3

# # AWGN with units
# @pytest.fixture
# def AWGN_with_units():
#     return scb.Noise().generate_AWGN_with_units(
#         mu = 0,
#         sigma = 1,
#         units = km,
#         count = 3
#     )

# def test_generate_AWGN_with_units_type(AWGN_with_units):
#     assert type(AWGN_with_units) == scb.ArrayWUnits

# def test_generate_AWGN_with_units_size(AWGN_with_units):
#     assert AWGN_with_units.size == 3

# # Apply AWGN with units
# @pytest.fixture
# def apply_AWGN_with_units():
#     data = scb.ArrayWUnits(np.array([1,2,3,4,5]), km)
#     return scb.Noise().apply_AWGN(
#         data = data,
#         mu = 0,
#         sigma = 1
#     )

# def test_apply_AWGN_type(apply_AWGN_with_units):
#     assert type(apply_AWGN_with_units) == scb.ArrayWUnits

# def test_apply_AWGN_size(apply_AWGN_with_units):
#     assert apply_AWGN_with_units.size == 5

# def test_apply_AWGN_units(apply_AWGN_with_units):
#     assert apply_AWGN_with_units.units == km

# # # data = scb.ArrayWUnits(np.array([1,2,3,4,5]), km)
# # # a = scb.Noise().apply_AWGN(
# # #     data = data,    
# # #     mu = 0,
# # #     sigma = 1
# # # )

# # unitx = scb.UnitsArray.from_array(np.array([km, km, km]))

# # b = scb.ArrayWUnits(np.array([[1,2],[3,4]]), km )#np.array([1,2,3]), km)
# # c = scb.ArrayWUnits(np.array([[1,2],[3,4]]), km )#np.array([1,2,3]), km)

# # d = b + c

# 0 == 0