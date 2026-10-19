# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================#
#  Unit Tests for the Noise class  #
# ==================================#
import src.scarabaeus as scb
import numpy as np
import pytest

km = scb.Units.get_units("km")


# --------------#
# region Tests #
# --------------#
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
    awgn_with_units = noise.generate_AWGN_with_units(mu=0, sigma=1, units=km, count=5)
    assert type(awgn_with_units) == scb.ArrayWUnits
    assert awgn_with_units.size == 5
    assert awgn_with_units.units == km


def test_apply_AWGN():
    """
    Verifies that AWGN is applied correctly.
    """
    noise = scb.Noise()
    data = scb.ArrayWUnits(np.array([1, 2, 3, 4, 5]), km)
    data_with_noise = noise.apply_AWGN(data=data, mu=0, sigma=1)
    assert type(data_with_noise) == scb.ArrayWUnits
    assert data_with_noise.size == 5
    assert data_with_noise.units == km