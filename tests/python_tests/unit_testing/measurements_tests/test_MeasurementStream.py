# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==============================================#
#  Unit Tests for the MeasurementStream class  #
# ==============================================#
import pytest

import src.scarabaeus as scb


# --------------------#
# region    Fixtures #
# --------------------#
class Station:
    """Stand-in for a ground station; only its name is read."""

    name = "DSS-34"


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestStreamCoercion:
    def test_tuple(self):
        s = scb.MeasurementStream.coerce(("f.json", "SeqRange", Station()))
        assert s.file == "f.json" and s.meas_type == "SeqRange"

    def test_passthrough(self):
        s = scb.MeasurementStream("f.json", "SeqRange", Station())
        assert scb.MeasurementStream.coerce(s) is s

    def test_short_tuple_raises(self):
        with pytest.raises(ValueError, match="file, meas_type, station"):
            scb.MeasurementStream.coerce(("f.json", "SeqRange"))

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            scb.MeasurementStream.coerce("just a string")

    def test_a_four_element_tuple_is_rejected(self):
        """
        The fields after ``station`` are not in a guessable order, so
        ``(file, type, station, 0.5)`` written for a sigma would name the
        dataset "0.5" and set no sigma.
        """
        with pytest.raises(ValueError, match="exactly"):
            scb.MeasurementStream.coerce(("f.json", "SeqRange", Station(), 0.5))

    def test_the_rejection_names_the_fields_to_use_instead(self):
        """The error says how to pass the extra value properly."""
        with pytest.raises(ValueError) as excinfo:
            scb.MeasurementStream.coerce(("f.json", "SeqRange", Station(), 0.5))
        message = str(excinfo.value)
        assert "MeasurementStream(" in message
        for field in ("name", "sigma", "bias", "media"):
            assert field in message


# endregion Tests #
# --------------#
