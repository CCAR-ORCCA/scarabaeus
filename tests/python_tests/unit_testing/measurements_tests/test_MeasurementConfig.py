# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==============================================#
#  Unit Tests for the MeasurementConfig class  #
# ==============================================#
import src.scarabaeus as scb


# --------------#
# region Tests #
# --------------#
class TestConfig:
    def test_sigma_by_type(self):
        cfg = scb.MeasurementConfig(sigma={"SeqRange": 1.0})
        assert cfg.get_sigma("SeqRange") == 1.0

    def test_sigma_by_dataset_name_wins(self):
        cfg = scb.MeasurementConfig(sigma={"SeqRange": 1.0, "GS1 Real Range": 2.0})
        assert cfg.get_sigma("SeqRange", "GS1 Real Range") == 2.0

    def test_missing_sigma_is_none(self):
        """There is no default; the model that needs one reports it missing."""
        assert scb.MeasurementConfig().get_sigma("SeqRange") is None

    def test_bias_defaults_to_zero(self):
        assert scb.MeasurementConfig().get_bias("SeqRange") == 0.0

    def test_extra_is_extensible(self):
        cfg = scb.MeasurementConfig(extra={"anything": 1})
        assert cfg.extra["anything"] == 1


# endregion Tests #
# --------------#
