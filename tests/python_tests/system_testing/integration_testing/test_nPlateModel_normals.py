# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =====================================================#
#  Integration Tests for nPlateModel Normal Queries   #
# =====================================================#
import json

import numpy as np
import pytest
import spiceypy

import src.scarabaeus as scb

# --------------------#
# region    Fixtures #
# --------------------#


# 90 degree rotation about Z: the synthetic CK's constant quaternion
SYNTHETIC_QUAT = [np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]


def make_config(tmp_path, types, ids, frames=None):
    data = {
        "names": ["plate 1", "plate 2"],
        "areas": [1e-06, 1e-06],
        "ref coeffs": [1.5, 1.1],
        "abs coeffs": [0.33, 0.33],
        "dref coeffs": [0.33, 0.33],
        "sref coeffs": [0.34, 0.34],
        "normal vectors": [[1, 0, 0], [-1, 0, 0]],
        "types": types,
        "ids": ids,
    }
    if frames is not None:
        data["frames"] = frames
    config_path = tmp_path / "nplate_normals_config.json"
    config_path.write_text(json.dumps(data))
    return str(config_path)


@pytest.fixture
def synthetic_ck(tmp_path):
    """
    Writes a synthetic CK segment for a dedicated instrument ID, reusing
    Cassini's real SCLK kernel (spice_id -82) already present in the
    locked generic kernel set.
    """
    inst_id = -82001
    sc_id = -82

    et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    et_list = [et0 + 60.0 * i for i in range(5)]
    quats = np.tile(SYNTHETIC_QUAT, (5, 1))
    avvs = np.tile([0.0, 0.0, 0.0], (5, 1))

    ck_path = str(tmp_path / "test_nplate_normals.bc")
    scb.SpiceManager.write_ck_type3(
        ck_path, inst_id=inst_id, ref_frame="J2000", sclk_id=sc_id,
        et_list=et_list, quats=quats, avvs=avvs, segid="TEST NPLATE NORMALS SEG",
    )
    try:
        yield inst_id, sc_id, et0 + 120.0
    finally:
        scb.SpiceManager.unload_kernel_from_pool(ck_path)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestGetAllNormals:
    def test_fixed_only(self, tmp_path):
        """
        Verifies that a model with no CK-type plates returns its stored
        normals unchanged (the non-SPICE branch of get_all_normals).
        """
        config = make_config(tmp_path, types=["Fixed", "Fixed"], ids=[None, None])
        n_model = scb.nPlateModel(config)

        result = n_model.get_all_normals(0.0, -82)

        assert result is n_model.normals

    def test_with_ck_plate(self, synthetic_ck, tmp_path):
        """
        Verifies that a CK-type plate's normal is rotated using a real
        synthetic CK segment, while the Fixed-type plate passes through
        unchanged. The C-matrix rotates J2000 into the plate frame, so the
        J2000 normal is its transpose applied to the configured one.
        """
        inst_id, sc_id, epoch = synthetic_ck
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, inst_id])
        n_model = scb.nPlateModel(config)

        result = n_model.get_all_normals(epoch, sc_id)

        assert len(result) == 2
        # Fixed plate (index 0) is passed through unchanged
        assert result[0] == [1, 0, 0]
        # CK plate (index 1): [-1, 0, 0] in the plate frame is +y in J2000
        expected = spiceypy.q2m(SYNTHETIC_QUAT).T @ np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(expected, [0.0, 1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(result[1], expected, atol=1e-12)

    def test_ck_plate_in_other_inertial_frame(self, synthetic_ck, tmp_path):
        """
        Verifies that inertial_frame re-expresses CK normals in that frame,
        leaving the body-frame plate as configured.
        """
        inst_id, sc_id, epoch = synthetic_ck
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, inst_id])
        n_model = scb.nPlateModel(config)

        result = n_model.get_all_normals(epoch, sc_id, inertial_frame="ECLIPJ2000")

        expected = spiceypy.pxform("J2000", "ECLIPJ2000", epoch) @ (
            spiceypy.q2m(SYNTHETIC_QUAT).T @ np.array([-1.0, 0.0, 0.0])
        )
        assert result[0] == [1, 0, 0]
        np.testing.assert_allclose(result[1], expected, atol=1e-12)

    @pytest.mark.parametrize("frame", ["ECLIPJ2000", 17], ids=["name", "id"])
    def test_fixed_plate_in_named_frame(self, tmp_path, frame):
        """
        Verifies that a Fixed plate with a frame (by SPICE name or ID) is
        rotated from that frame into J2000, while the plate without one
        stays in the body frame. No CK is needed for either.
        """
        config = make_config(
            tmp_path, types=["Fixed", "Fixed"], ids=[None, None], frames=[None, frame]
        )
        n_model = scb.nPlateModel(config)
        epoch = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

        result = n_model.get_all_normals(epoch, -82)

        expected = spiceypy.pxform("ECLIPJ2000", "J2000", epoch) @ np.array(
            [-1.0, 0.0, 0.0]
        )
        assert result[0] == [1, 0, 0]
        np.testing.assert_allclose(result[1], expected, atol=1e-15)

    @pytest.mark.parametrize(
        "frame, match",
        [
            ("ORCCA_NO_SUCH_FRAME", "Unable to orient"),
            (-82999999, "no loaded kernel defines"),
        ],
        ids=["name", "id"],
    )
    def test_unknown_frame_raises(self, tmp_path, frame, match):
        """
        Verifies PlateOrientationError when a plate's frame is unknown to
        SPICE, and that SPICE is left usable afterwards.
        """
        config = make_config(
            tmp_path, types=["Fixed", "Fixed"], ids=[None, None], frames=[None, frame]
        )
        n_model = scb.nPlateModel(config)

        with pytest.raises(scb.nPlateModel.PlateOrientationError, match=match):
            n_model.get_all_normals(0.0, -82)
        assert not spiceypy.failed()

    def test_ck_query_fails_raises(self, tmp_path):
        """
        Verifies PlateOrientationError, naming the plate and its CK ID, when
        no CK kernel is loaded for a CK-type plate, and that SPICE is left
        usable afterwards.
        """
        # -64099 has no CK segment loaded anywhere in this test session
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, -64099])
        n_model = scb.nPlateModel(config)

        with pytest.raises(
            scb.nPlateModel.PlateOrientationError,
            match=r'CK plate "plate 2" \(CK ID -64099\)',
        ):
            n_model.get_all_normals(0.0, -82)
        assert not spiceypy.failed()

    def test_missing_sclk_raises(self, tmp_path):
        """
        Verifies PlateOrientationError when the spacecraft has no SCLK kernel,
        so the epoch cannot be converted to clock ticks for the CK query, and
        that SPICE is left usable afterwards.
        """
        # -64099 has no SCLK kernel loaded anywhere in this test session
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, -64099])
        n_model = scb.nPlateModel(config)

        with pytest.raises(
            scb.nPlateModel.PlateOrientationError, match="no spacecraft clock"
        ):
            n_model.get_all_normals(0.0, -64099)
        assert not spiceypy.failed()


class TestGetCkNormals:
    def test_include_ids(self, synthetic_ck, tmp_path):
        """
        Verifies that _get_ck_normals(..., include_IDs=True) returns
        [plate_id, normal] pairs for each CK-type plate.
        """
        inst_id, sc_id, epoch = synthetic_ck
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, inst_id])
        n_model = scb.nPlateModel(config)

        result = n_model._get_ck_normals(epoch, sc_id, include_IDs=True)

        assert len(result) == 1
        plate_id, normal = result[0]
        assert plate_id == inst_id
        assert np.linalg.norm(normal) == pytest.approx(1.0)

    def test_epoch_array_tdb(self, synthetic_ck, tmp_path):
        """
        Verifies that a TDB EpochArray is accepted and produces the same
        result as the equivalent raw ET value.
        """
        inst_id, sc_id, epoch = synthetic_ck
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, inst_id])
        n_model = scb.nPlateModel(config)

        epoch_array = scb.EpochArray(epoch, "TDB")
        result_array = n_model._get_ck_normals(epoch_array, sc_id)
        result_raw = n_model._get_ck_normals(epoch, sc_id)

        assert np.allclose(result_array, result_raw)

    def test_epoch_array_wrong_system_raises(self, synthetic_ck, tmp_path):
        """Verifies ValueError when the EpochArray isn't in TDB."""
        inst_id, sc_id, epoch = synthetic_ck
        config = make_config(tmp_path, types=["Fixed", "CK"], ids=[None, inst_id])
        n_model = scb.nPlateModel(config)

        epoch_array = scb.EpochArray(epoch, "TT")
        with pytest.raises(ValueError, match="ephemeris time"):
            n_model._get_ck_normals(epoch_array, sc_id)

    def test_invalid_epoch_type_raises(self, nplate_config_file):
        """Verifies TypeError when epoch isn't an EpochArray or numeric."""
        n_model = scb.nPlateModel(str(nplate_config_file))
        with pytest.raises(TypeError, match="EpochArray or numeric"):
            n_model._get_ck_normals("not an epoch", -82)


# endregion Tests #
# ----------------#