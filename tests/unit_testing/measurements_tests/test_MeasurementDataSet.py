# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus import ArrayWUnits as awu

import pytest
km, sec, min, kg = scb.Units.get_units(["km", "sec", "min", "kg"])

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#
@pytest.fixture()
def values():
    return [awu([1, 2, 3], km), awu([4, 5, 6], km), awu([7, 8, 9], km)]

@pytest.fixture()
def names():
    return ["observed", "computed", "residuals"]

@pytest.fixture()
def epoch_array():
    return scb.EpochArray([1096810090.12,1096810099.13,1096810060.14,1096810999.15],'TDB')

@pytest.fixture()
def target():
    return scb.Body("Earth", spice_id=399)

@pytest.fixture()
def instrument():
    return scb.Instrument("TEST_INSTR", -9999, "TEST_MEAS_TYPE")

#--------------#
# region Tests #
#--------------#
def test_initialization(values, names, epoch_array, target, instrument):
    """
        Verifies that object is constructed correctly.
    """
    dataset_name = "Test DataSet"
    data_set = scb.MeasurementDataSet(set_name=dataset_name,measurements=values, names=names, instrument=instrument,target=target,epochs_t3=epoch_array)
    