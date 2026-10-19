# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ======================================#
#  Unit Tests for the Instrument class #
# ======================================#

import src.scarabaeus as scb
import pytest

# --------------------#
# region    Fixtures #
# --------------------#

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that object is constructed correctly.
    """
    name = "Test Instrument"
    spice_id = -12345
    measurement_type = "Range"

    instrument = scb.Instrument(name, spice_id, measurement_type)

    assert instrument._name == name
    assert instrument._spice_id == spice_id
    assert instrument._measurement_type == measurement_type
    assert instrument.measurement_type == measurement_type

def test_field_of_view():
    """
    Verifies that the field_of_view property is not implemented in the base Instrument class.
    """
    pytest.skip("Field of view property is not implemented")
