# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

#--------------------#
# region    Fixtures #
#--------------------#
@pytest.fixture
def instrument():
    """
        Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(instrument,media_corr_file):
    """
        Verifies that object is constructed correctly.
    """
    name="GS1 Media Corrections"
    tropo_seasonal_file_path = str(media_corr_file)
    media_correction = scb.MediaCorrections(name=name, instrument=instrument, tropo_seasonal_file_path=tropo_seasonal_file_path)

    assert media_correction._name == name
    assert media_correction._instrument == instrument
