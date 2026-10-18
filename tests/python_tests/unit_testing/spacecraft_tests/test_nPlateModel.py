# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ========================================#
#  Unit Tests for the nPlateModel class  #
# ========================================#
import json

import src.scarabaeus as scb
import pytest

# --------------------#
# region    Fixtures #
# --------------------#

# --------------------#
# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(nplate_config_file):
    """
    Verifies that object is constructed correctly, including all
    property getters (names, areas, ref_coeffs, abs_ref_coeffs,
    diff_ref_coeffs, spec_ref_coeffs, normals, plate_types, plate_ids,
    frames, double_sided). A configuration without the optional "frames"
    and "double sided" keys has single-sided plates and no named frames.
    """
    # construct N-plate model using a configuration file
    plate_config = str(nplate_config_file)
    n_model = scb.nPlateModel(plate_config)

    assert n_model.names == ["plate 1", "plate 2"]
    assert [a.values for a in n_model.areas] == [pytest.approx(1e-06), pytest.approx(1e-06)]
    assert n_model.ref_coeffs == [1.5, 1.1]
    assert n_model.abs_ref_coeffs == [0.33, 0.33]
    assert n_model.diff_ref_coeffs == [0.33, 0.33]
    assert n_model.spec_ref_coeffs == [0.34, 0.34]
    assert n_model.normals == [[1, 0, 0], [-1, 0, 0]]
    assert n_model.plate_types == ["Fixed", "CK"]
    assert n_model.plate_ids == [None, -64027]
    assert n_model.frames == [None, None]
    assert n_model.double_sided == [False, False]
    # the Fixed plate is in the body frame, the CK plate is not
    assert n_model.body_frame_plates == [True, False]


def _config(**overrides):
    """Three-plate configuration (Fixed, Fixed, CK), with keys overridden."""
    data = {
        "names": ["bus +x", "array", "gimballed"],
        "areas": [1e-06, 2e-06, 3e-06],
        "ref coeffs": [1.5, 1.1, 1.2],
        "abs coeffs": [0.33, 0.33, 0.33],
        "dref coeffs": [0.33, 0.33, 0.33],
        "sref coeffs": [0.34, 0.34, 0.34],
        "normal vectors": [[1, 0, 0], [0, 1, 0], [-1, 0, 0]],
        "types": ["Fixed", "Fixed", "CK"],
        "ids": [None, None, -64027],
    }
    data.update(overrides)
    return data


def test_frames_and_double_sided():
    """
    Verifies the optional "frames" (SPICE frame name or ID per Fixed plate)
    and "double sided" keys, and that only Fixed plates without a frame
    are in the spacecraft body frame.
    """
    n_model = scb.nPlateModel(
        plates_data=_config(
            frames=[None, "ECLIPJ2000", None], **{"double sided": [False, True, False]}
        )
    )

    assert n_model.frames == [None, "ECLIPJ2000", None]
    assert n_model.double_sided == [False, True, False]
    assert n_model.body_frame_plates == [True, False, False]

    # a frame may also be given by its SPICE ID
    n_model.frames = [None, 17, None]
    assert n_model.body_frame_plates == [True, False, False]


def test_to_dict_round_trip():
    """
    Verifies that to_dict writes "frames" and "double sided", and that
    from_dict rebuilds the same model from it.
    """
    n_model = scb.nPlateModel(
        plates_data=_config(
            frames=[None, 17, None], **{"double sided": [False, True, False]}
        )
    )

    data = n_model.to_dict()
    assert data["frames"] == [None, 17, None]
    assert data["double sided"] == [False, True, False]

    restored = scb.nPlateModel.from_dict(json.loads(json.dumps(data)))
    assert restored.to_dict() == data


def test_repr(nplate_config_file):
    """Verifies the printable representation includes the plate count."""
    n_model = scb.nPlateModel(str(nplate_config_file))
    assert repr(n_model) == "N-Plate Model containing 2 plates"


class TestErrorHandling:
    def test_names_unique(self, nplate_config_file):
        """Verifies ValueError when plate names aren't unique."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(ValueError, match="must be unique"):
            n_model.names = ["dup", "dup"]

    def test_areas_type(self, nplate_config_file):
        """Verifies TypeError when an areas element isn't int/float."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(TypeError, match="areas"):
            n_model.areas = [1.0, "not a number"]

    def test_normals_length(self, nplate_config_file):
        """Verifies ValueError when a normal vector isn't a 1x3 vector."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(ValueError, match="1x3 vector"):
            n_model.normals = [[1, 0, 0], [1, 0]]

    def test_plate_types_invalid(self, nplate_config_file):
        """Verifies ValueError when a plate type isn't "Fixed" or "CK"."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(ValueError, match="plates_type"):
            n_model.plate_types = ["Fixed", "Not a valid type"]

    def test_plate_ids_type(self, nplate_config_file):
        """Verifies TypeError when a plate id isn't int or None."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(TypeError, match="spice ids"):
            n_model.plate_ids = [None, "not an int"]

    def test_ref_coeffs_type(self, nplate_config_file):
        """
        Verifies TypeError (via the shared _validate_numerical helper) when
        a ref_coeffs element isn't int/float.
        """
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(TypeError, match="ref coeffs"):
            n_model.ref_coeffs = [1.5, "not a number"]

    @pytest.mark.parametrize(
        "frame", [1.5, True, [17]], ids=["float", "bool", "list"]
    )
    def test_frames_type(self, frame):
        """
        Verifies ConfigurationError when a "frames" entry is neither a frame
        name, a frame ID nor null (a bool is not taken for an ID).
        """
        with pytest.raises(scb.nPlateModel.ConfigurationError, match="frames"):
            scb.nPlateModel(plates_data=_config(frames=[None, frame, None]))

    def test_ck_plate_with_frame(self):
        """
        Verifies ConfigurationError when a CK plate is also given a frame:
        its C-kernel already orients it.
        """
        with pytest.raises(scb.nPlateModel.ConfigurationError, match="CK plate"):
            scb.nPlateModel(plates_data=_config(frames=[None, None, "J2000"]))

    def test_double_sided_type(self):
        """Verifies ConfigurationError when a "double sided" entry isn't a bool."""
        with pytest.raises(scb.nPlateModel.ConfigurationError, match="double sided"):
            scb.nPlateModel(plates_data=_config(**{"double sided": [False, 1, False]}))

    def test_mismatched_column_count(self, tmp_path):
        """
        Verifies ValueError when a config property has a different number
        of entries than "names".
        """
        data = {
            "names": ["plate 1", "plate 2"],
            "areas": [1e-06],  # mismatched: only 1 entry, "names" has 2
            "ref coeffs": [1.5, 1.1],
            "abs coeffs": [0.33, 0.33],
            "dref coeffs": [0.33, 0.33],
            "sref coeffs": [0.34, 0.34],
            "normal vectors": [[1, 0, 0], [-1, 0, 0]],
            "types": ["Fixed", "CK"],
            "ids": [None, -64027],
        }
        config_path = tmp_path / "bad_nplate_config.json"
        config_path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="Mismatched column count"):
            scb.nPlateModel(str(config_path))
