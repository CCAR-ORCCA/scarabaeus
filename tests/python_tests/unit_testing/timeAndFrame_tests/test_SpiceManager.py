# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest
import spiceypy as spice
import os
import json
import warnings
import pickle

(J2000, ITRF93, ECLIPJ2000, IAUEARTH) = scb.Frame.generate_common_frames()

furnshKernelFilename = os.getcwd() + "/data/kernels/locked/locked_generic.tm"
scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)

km, kg, sec, hr, day, m, rad, unitless = scb.Units.get_units(
    ["km", "kg", "sec", "hr", "day", "m", "rad", "unitless"]
)

# J2000 epoch: 2000 JAN 01 12:00:00 TDB = 0 ET (must say TDB explicitly;
# without it SPICE parses as UTC, which leads TDB by ~64.184 s).
J2000_CAL = "2000 JAN 01 12:00:00 TDB"
J2000_ET = 0.0

# A second known epoch for round-trip tests
REF_CAL = "2024 MAR 20 03:06:00"

# --------------------#
# region    Fixtures  #
# --------------------#

MOON_ID = spice.bodn2c("MOON")
EARTH_ID = spice.bodn2c("EARTH")

EYE6 = np.eye(6).tolist()
TWO6 = (2 * np.eye(6)).tolist()


def write_json(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def write_pickle(tmp_path, name, data):
    from scarabaeus.utils.Serialization import dump_pickle
    return dump_pickle(data, tmp_path / name)


def flat_params_json(epoch_lo=0.0, epoch_hi=100.0, values=None):
    """Flat (non-sequence) parameters JSON for MOON/J2000/EARTH."""
    if values is None:
        values = [3.0, 5.0]
    return {
        "body_ID": MOON_ID,
        "origin_ID": EARTH_ID,
        "reference_frame": "J2000",
        "parameters": values,
        "epochsTDB": [epoch_lo, epoch_hi],
        "unitsPower": [0, 0, 1, 0],
        "unitsScale": [1.0],
    }


def sequence_params_json(parameters, epochs=(0.0, 100.0), extra_legs=None):
    """Single-leg sequence parameters JSON for MOON/J2000/EARTH.

    ``extra_legs``, if given, is a dict merged in *before* ``leg0`` (both of
    get_parameters's leg loops stop at the first match, so a leg meant to
    exercise a skip/continue branch must be iterated before the valid one).
    """
    legs = {}
    if extra_legs:
        legs.update(extra_legs)
    legs["leg0"] = {
        "epochsTDB": list(epochs),
        "body_ID": MOON_ID,
        "origin_ID": EARTH_ID,
        "reference_frame": "J2000",
        "parameters": parameters,
    }
    return {
        "body_ID": MOON_ID,
        "origin_ID": EARTH_ID,
        "reference_frame": "J2000",
        "legs": legs,
    }


def stm_json(epochs, stms_dict):
    """Minimal STM JSON for MOON/J2000/EARTH."""
    return {
        "body_ID": MOON_ID,
        "origin_ID": EARTH_ID,
        "reference_frame": "J2000",
        "epochsTDB": epochs,
        "STMs": stms_dict,
    }


def sequence_stm_json(epochs, stms_dict, legs):
    """STM JSON as a sequence run writes it: the STMs plus, for context, the
    same ``legs`` section its parameters sidecar carries."""
    return {**stm_json(epochs, stms_dict), "legs": legs}


def per_leg_stm_json(leg_blocks):
    """STM JSON with one STM block per leg, as _prepare_stm_data writes a
    sequence: ``leg_blocks`` maps a leg key to ``(grid, matrices)``."""
    legs = {
        key: {
            "body_ID": MOON_ID,
            "origin_ID": EARTH_ID,
            "reference_frame": "J2000",
            "STM_epochsTDB": list(grid),
        }
        for key, (grid, _) in leg_blocks.items()
    }
    stms = {
        key: {str(float(t)): M for t, M in zip(grid, mats)}
        for key, (grid, mats) in leg_blocks.items()
    }
    epochs = sorted({float(t) for grid, _ in leg_blocks.values() for t in grid})
    return {**stm_json(epochs, stms), "legs": legs}

# endregion Fixtures #
# --------------------#

# ---------------#
# region  Tests  #
# ---------------#


class TestTimeConversions:

    def test_cal2et_j2000_epoch(self):
        """J2000 epoch must convert to 0 ET (by definition)."""
        et = scb.SpiceManager.cal2et(J2000_CAL)
        assert isinstance(et, float)
        assert et == pytest.approx(J2000_ET, abs=1e-3)

    def test_cal2et_matches_spice(self):
        et_scb = scb.SpiceManager.cal2et(REF_CAL)
        et_spc = spice.str2et(REF_CAL)
        assert et_scb == pytest.approx(et_spc, rel=1e-12)

    def test_str2et_matches_cal2et(self):
        """str2et is an alias — must return identical results."""
        assert scb.SpiceManager.str2et(REF_CAL) == scb.SpiceManager.cal2et(REF_CAL)

    def test_utc2et_matches_spice(self):
        utc_str = "2024-03-20T03:06:00"
        et_scb = scb.SpiceManager.utc2et(utc_str)
        et_spc = spice.utc2et(utc_str)
        assert et_scb == pytest.approx(et_spc, rel=1e-12)

    def test_et2cal_round_trip(self):
        """et → calendar string → et must recover the original epoch."""
        et_original = scb.SpiceManager.cal2et(REF_CAL)
        cal_str = scb.SpiceManager.et2cal(et_original)
        assert isinstance(cal_str, str)
        et_recovered = scb.SpiceManager.cal2et(cal_str)
        assert et_recovered == pytest.approx(et_original, abs=1e-3)

    def test_et2cal_0d_array_returns_scalar_string(self):
        """A 0-D ndarray must be treated like a scalar float."""
        et_original = scb.SpiceManager.cal2et(REF_CAL)
        cal_str = scb.SpiceManager.et2cal(np.array(et_original))
        assert scb.SpiceManager.cal2et(str(cal_str)) == pytest.approx(et_original, abs=1e-3)

    def test_et2cal_nd_array_preserves_shape(self):
        """An N-D ndarray must come back reshaped to the original shape."""
        et0 = scb.SpiceManager.cal2et(REF_CAL)
        et1 = et0 + 3600.0
        result = scb.SpiceManager.et2cal(np.array([[et0, et1], [et1, et0]]))
        assert result.shape == (2, 2)

    def test_et2cal_plain_list_returns_array(self):
        """A plain list (neither float nor ndarray) hits the final fallback branch."""
        et0 = scb.SpiceManager.cal2et(REF_CAL)
        et1 = et0 + 3600.0
        result = scb.SpiceManager.et2cal([et0, et1])
        assert len(result) == 2

    def test_et2utc_returns_string(self):
        et = scb.SpiceManager.cal2et(REF_CAL)
        utc_str = scb.SpiceManager.et2utc(et)
        assert isinstance(utc_str, str)
        assert len(utc_str) > 0

    def test_et2jd_j2000(self):
        """ET=0 (TDB) is ~64 s before J2000 UTC, so JD ≈ 2451544.999."""
        jd_str = scb.SpiceManager.et2jd(J2000_ET)
        assert isinstance(jd_str, str)
        jd_val = float(jd_str.split()[-1])
        assert jd_val == pytest.approx(2451545.0, abs=1e-3)

    def test_jd2et_round_trip(self):
        """JD → ET round-trip must be self-consistent via SPICE."""
        et_ref = scb.SpiceManager.cal2et(REF_CAL)
        jd_str = scb.SpiceManager.et2jd(et_ref)
        jd_val = float(jd_str.split()[-1])
        et_recovered = scb.SpiceManager.jd2et(jd_val)
        assert et_recovered == pytest.approx(et_ref, abs=1.0)  # 1-second tolerance for TDB/UTC drift

    def test_et2YDS_j2000(self):
        """J2000 epoch is year 2000, DOY 1 (Jan 1)."""
        years, doys, _ = scb.SpiceManager.et2YDS([J2000_ET])
        assert years[0] == 2000
        assert doys[0] == 1

    def test_et2YDS_returns_three_lists(self):
        et = scb.SpiceManager.cal2et(REF_CAL)
        result = scb.SpiceManager.et2YDS([et])
        assert len(result) == 3
        years, doys, secs = result
        assert len(years) == len(doys) == len(secs) == 1

    def test_YDS2et_round_trip(self):
        """YDS → ET → YDS must recover the same (year, doy, sec)."""
        et_ref = scb.SpiceManager.cal2et(REF_CAL)
        years, doys, secs = scb.SpiceManager.et2YDS([et_ref])
        et_back = scb.SpiceManager.YDS2et(years[0], doys[0], secs[0])
        assert et_back[0] == pytest.approx(et_ref, abs=1e-3)


class TestBodyFrameLookups:

    @pytest.mark.parametrize("body_name,expected_id", [
        ("EARTH", 399), ("SUN", 10), ("MOON", 301),
    ])
    def test_get_id_from_string(self, body_name, expected_id):
        assert scb.SpiceManager.get_id_from_string(body_name) == expected_id

    @pytest.mark.parametrize("spice_id,expected_substring", [
        (399, "EARTH"), (10, "SUN"),
    ])
    def test_get_name_from_id(self, spice_id, expected_substring):
        name = scb.SpiceManager.get_name_from_id(spice_id)
        assert expected_substring in name.upper()

    def test_name_to_ID_earth(self):
        assert scb.SpiceManager.name_to_ID("EARTH") == 399

    def test_name_to_ID_unknown_returns_none(self):
        result = scb.SpiceManager.name_to_ID("NONEXISTENT_BODY_XYZ")
        assert result is None

    def test_frame_name_to_frame_id_j2000(self):
        fid = scb.SpiceManager.frame_name_to_frame_id("J2000")
        assert isinstance(fid, int)
        assert fid == 1

    def test_frame_id_to_frame_name_j2000(self):
        name = scb.SpiceManager.frame_id_to_frame_name(1)
        assert "J2000" in name.upper()

    def test_get_frame_w_spice_id_earth(self):
        result = scb.SpiceManager.get_frame_w_spice_id(399)
        assert result is not None

    def test_get_frame_w_spice_id_unrecognized_raises(self):
        """An id_number with no associated frame must raise ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            scb.SpiceManager.get_frame_w_spice_id(999999999)


class TestStateAndPosition:

    # Use Earth–Moon system: always covered by generic kernels.
    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_get_pos(self):
        pos = scb.SpiceManager.get_pos("MOON", self.ET, "J2000", "EARTH")
        assert hasattr(pos, "values")
        assert pos.values.shape == (3,)
        assert np.linalg.norm(pos.values) > 1e3  # Moon is ~384 000 km away

    def test_get_vel(self):
        vel = scb.SpiceManager.get_vel("MOON", self.ET, "J2000", "EARTH")
        assert hasattr(vel, "values")
        assert vel.values.shape == (3,)

    def test_get_pos_callback_matches_get_pos(self):
        """The _get_pos_callback closure must reproduce get_pos at a given epoch."""
        callback = scb.SpiceManager._get_pos_callback("MOON", "J2000", "EARTH", "None")
        result = callback(self.ET)
        expected = scb.SpiceManager.get_pos("MOON", self.ET, "J2000", "EARTH")
        np.testing.assert_allclose(result, expected.values, rtol=1e-12)

    def test_get_vel_callback_matches_get_vel(self):
        """The _get_vel_callback closure must reproduce get_vel at a given epoch."""
        callback = scb.SpiceManager._get_vel_callback("MOON", "J2000", "EARTH", "None")
        result = callback(self.ET)
        expected = scb.SpiceManager.get_vel("MOON", self.ET, "J2000", "EARTH")
        np.testing.assert_allclose(result, expected.values, rtol=1e-12)

    def test_get_state_shape(self):
        state = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        assert hasattr(state, "values")
        assert state.values.shape == (6,)

    def test_get_state_pos_vel_consistent(self):
        """First 3 elements of get_state must match get_pos."""
        pos = scb.SpiceManager.get_pos("MOON", self.ET, "J2000", "EARTH")
        state = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        np.testing.assert_allclose(state.values[:3], pos.values, rtol=1e-12)

    def test_get_lighttime_positive(self):
        lt = scb.SpiceManager.get_lighttime("MOON", self.ET, "J2000", "EARTH")
        assert float(lt.values) > 0
        assert float(lt.values) < 2.0  # Moon light travel time < 2 seconds

    def test_get_xfrm_j2000_to_eclipj2000_shape(self):
        xfrm = scb.SpiceManager.get_xfrm("J2000", "ECLIPJ2000", self.ET)
        assert xfrm.shape == (3, 3)

    def test_get_xfrm_identity_same_frame(self):
        """Transforming from a frame to itself must give the identity matrix."""
        xfrm = scb.SpiceManager.get_xfrm("J2000", "J2000", self.ET)
        np.testing.assert_allclose(xfrm, np.eye(3), atol=1e-12)

    def test_get_xfrm_orthogonal(self):
        """A rotation matrix must be orthogonal: R @ R.T = I."""
        xfrm = scb.SpiceManager.get_xfrm("J2000", "ECLIPJ2000", self.ET)
        np.testing.assert_allclose(xfrm @ xfrm.T, np.eye(3), atol=1e-10)

    def test_get_xfrm_callback_matches_get_xfrm(self):
        """The _get_xfrm_callback closure must reproduce get_xfrm at a given epoch."""
        callback = scb.SpiceManager._get_xfrm_callback("J2000", "ECLIPJ2000")
        result = callback(self.ET)
        expected = scb.SpiceManager.get_xfrm("J2000", "ECLIPJ2000", self.ET)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestObjectInputCoercion:
    """Frame and Body objects must be accepted wherever strings are documented.

    The five state-query methods share the signature (trgt_bdy, et, frame, obsvr_bdy).
    Passing Frame/Body objects hits the isinstance-guard branches and the result
    must be numerically identical to the string-based call.
    """

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    EARTH = scb.CelestialBody.from_constants("EARTH")
    MOON = scb.CelestialBody.from_constants("MOON")

    @pytest.mark.parametrize("func", [
        scb.SpiceManager.get_pos,
        scb.SpiceManager.get_vel,
        scb.SpiceManager.get_state,
        scb.SpiceManager.get_lighttime,
    ], ids=["get_pos", "get_vel", "get_state", "get_lighttime"])
    def test_accepts_frame_and_body_objects(self, func):
        """Result with Frame/Body objects must match result with plain strings."""
        result_str = func("MOON", self.ET, "J2000", "EARTH")
        result_obj = func(self.MOON, self.ET, J2000, self.EARTH)
        np.testing.assert_allclose(result_obj.values, result_str.values, rtol=1e-12)

    def test_get_state_precise_accepts_frame_and_body_objects(self):
        """get_state_precise with Frame/Body objects must match the string-based call."""
        result_str = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        result_obj = scb.SpiceManager.get_state_precise(
            self.MOON, self.ET, J2000, self.EARTH, epoch_time_low=0.0
        )
        np.testing.assert_allclose(result_obj.values, result_str.values, rtol=1e-12)

    def test_visibility_accepts_body_objects(self):
        ET0 = self.ET
        ET1 = ET0 + 30 * 86400.0
        kwargs = {
            "step_size": 3600.0, "angle_limit": 0.0, "inframe": "J2000", "abcorr": "NONE",
            "crdsys": "LATITUDINAL", "coord": "LATITUDE", "relate": ">",
        }
        num_str, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", ET0, ET1, **kwargs
        )
        num_obj, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            self.EARTH, self.MOON, ET0, ET1, **kwargs
        )
        assert num_obj == num_str


class TestStatePrecise:

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_get_state_precise_shape(self):
        state = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        assert hasattr(state, "values")
        assert state.values.shape == (6,)

    def test_get_state_precise_matches_get_state(self):
        """epoch_time_low=0.0 reproduces get_state exactly."""
        st_std = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        st_prec = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        np.testing.assert_array_equal(st_prec.values, st_std.values)

    def test_get_state_precise_nonzero(self):
        state = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        assert np.linalg.norm(state.values[:3]) > 1e3

    @pytest.mark.parametrize("low", [0.25, 0.5, 0.75])
    def test_get_state_precise_representable_low(self, low):
        """ET + low is a float here, so the pair must return exactly the SPICE state there.

        A linear interpolation over 1 s would be off by up to a/8 (~0.3 mm for the Moon).
        """
        st_prec = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=low
        )
        st_true = spice.spkezr("MOON", self.ET + low, "J2000", "NONE", "EARTH")[0]
        np.testing.assert_allclose(st_prec.values, st_true, rtol=0.0, atol=1e-9)

    def test_get_state_precise_sub_ulp_low(self):
        """A low part below one float step of ET moves the state along the chord to the next float."""
        ulp = np.spacing(self.ET)
        st_0 = spice.spkezr("MOON", self.ET, "J2000", "NONE", "EARTH")[0]
        st_1 = spice.spkezr("MOON", self.ET + ulp, "J2000", "NONE", "EARTH")[0]
        st_prec = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.3 * ulp
        )
        np.testing.assert_allclose(
            st_prec.values[:3], st_0[:3] + 0.3 * (st_1[:3] - st_0[:3]), rtol=0.0, atol=1e-9
        )
        # The shift itself (~36 um) is well above the tolerance, so the low part is not ignored
        assert np.linalg.norm(st_prec.values[:3] - st_0[:3]) > 1e-8


class TestUtilities:

    def test_matrix_times_vector_identity(self):
        v = np.array([1.0, 2.0, 3.0])
        result = scb.SpiceManager.matrix_times_vector(np.eye(3), v)
        np.testing.assert_allclose(result, v)

    def test_matrix_times_vector_known_rotation(self):
        """90-degree rotation around z maps [1,0,0] to [0,1,0]."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        result = scb.SpiceManager.matrix_times_vector(R, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_def_new_body(self):
        """Defining a new body must make it retrievable by name."""
        name = "TEST_BODY_SCB_UNIT"
        spice_id = -99991
        scb.SpiceManager.def_new_body(name, spice_id)
        assert spice.bodn2c(name) == spice_id

    def test_check_kernel_status_in_pool_loaded(self):
        """A freshly (re-)loaded metakernel must be reported as in the pool.

        Loads explicitly rather than relying on module-level load state:
        conftest's module-scoped autouse fixture clears the kernel pool
        before each test module runs, so furnshKernelFilename is not
        guaranteed to still be loaded by the time this test executes.
        """
        scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)
        status = scb.SpiceManager.check_kernel_status_in_pool(furnshKernelFilename)
        assert status is True

    def test_check_kernel_status_not_loaded(self):
        status = scb.SpiceManager.check_kernel_status_in_pool("/nonexistent/path/fake.bsp")
        assert status is False


class TestAdditionalLookups:

    @pytest.mark.parametrize("spice_id,expected_substring", [
        (399, "EARTH"), (10, "SUN"),
    ])
    def test_id2name(self, spice_id, expected_substring):
        name = scb.SpiceManager.id2name(spice_id)
        assert expected_substring in name.upper()

    def test_utc2tdb_matches_utc2et(self):
        """utc2tdb is an alias for utc2et — must return identical values."""
        utc = "2024-03-20T03:06:00"
        assert scb.SpiceManager.utc2tdb(utc) == scb.SpiceManager.utc2et(utc)

    def test_list_all_frames_returns_list(self):
        frames = scb.SpiceManager.list_all_frames(max_id=5)
        assert isinstance(frames, list)

    def test_list_all_frames_includes_j2000(self):
        """J2000 has frame ID 1, well within max_id=5."""
        frames = scb.SpiceManager.list_all_frames(max_id=5)
        assert any("J2000" in f for f in frames)

    def test_list_all_frames_skips_ids_that_raise(self, monkeypatch):
        """A frame ID that raises must be skipped, not crash the scan.

        spice.frmnam never actually raises (it returns "" for unknown IDs),
        so the try/except is exercised here via a monkeypatched lookup.
        """
        def fake_lookup(frame_id):
            if frame_id == 0:
                raise RuntimeError("simulated SPICE failure")
            return scb.SpiceManager.frame_id_to_frame_name.__wrapped__(frame_id)

        monkeypatch.setattr(scb.SpiceManager, "frame_id_to_frame_name", fake_lookup)
        frames = scb.SpiceManager.list_all_frames(max_id=2)
        assert isinstance(frames, list)


class TestKernelManagement:

    def test_print_kernels_does_not_raise(self, capsys):
        scb.SpiceManager.print_kernels()
        out = capsys.readouterr().out
        assert isinstance(out, str)

    @pytest.mark.parametrize("arg", [
        furnshKernelFilename,
        [furnshKernelFilename],
    ], ids=["single_file", "list"])
    def test_load_kernel(self, arg):
        scb.SpiceManager.load_kernel(arg)
        scb.SpiceManager.unload_kernel_from_pool(furnshKernelFilename)

    def test_unload_nonexistent_does_not_raise(self):
        """Unloading a file that was never loaded must silently do nothing."""
        scb.SpiceManager.unload_kernel_from_pool("/nonexistent/path/fake.bsp")

    def test_increase_kernel_priority_does_not_raise(self):
        scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)
        scb.SpiceManager.increase_kernel_priority(furnshKernelFilename)
        scb.SpiceManager.unload_kernel_from_pool(furnshKernelFilename)


class TestWriteFrameFiles:
    """_write_pck_frame / _write_ck_frame / _write_tk_frame each generate a
    text FK kernel and raise if the requested frame_id is already
    registered in the SPICE pool (frame ID 1 == J2000, always registered)."""

    def test_write_pck_frame_creates_file(self, tmp_path):
        file_path = str(tmp_path / "eros_fixed")
        scb.SpiceManager._write_pck_frame(file_path, "EROS_FIXED_TEST", 2, 2000433)
        assert os.path.isfile(file_path + ".pck")
        content = open(file_path + ".pck").read()
        assert "EROS_FIXED_TEST" in content

    def test_write_pck_frame_registered_id_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="already registered"):
            scb.SpiceManager._write_pck_frame(str(tmp_path / "bad"), "X", 2, 1)

    def test_write_pck_frame_overwrites_existing_file(self, tmp_path):
        file_path = str(tmp_path / "eros_fixed")
        scb.SpiceManager._write_pck_frame(file_path, "EROS_FIXED_TEST", 2, 2000433)
        scb.SpiceManager._write_pck_frame(file_path, "EROS_FIXED_TEST", 2, 2000433)
        assert os.path.isfile(file_path + ".pck")

    def test_write_ck_frame_creates_file(self, tmp_path):
        file_path = str(tmp_path / "mgs_spacecraft")
        scb.SpiceManager._write_ck_frame(file_path, "MGS_SPACECRAFT_TEST", -94001, -94, -94, -94)
        assert os.path.isfile(file_path + ".tf")
        content = open(file_path + ".tf").read()
        assert "MGS_SPACECRAFT_TEST" in content

    def test_write_ck_frame_registered_id_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="already registered"):
            scb.SpiceManager._write_ck_frame(str(tmp_path / "bad"), "X", 1, -94, -94, -94)

    def test_write_tk_frame_creates_file_with_default_identity(self, tmp_path):
        file_path = str(tmp_path / "mars_fixed")
        scb.SpiceManager._write_tk_frame(file_path, "MARS_FIXED_TEST", 1400499, 499, "IAU_MARS")
        assert os.path.isfile(file_path + ".tf")
        content = open(file_path + ".tf").read()
        assert "1 0 0 0 1 0 0 0 1" in content

    def test_write_tk_frame_creates_file_with_custom_matrix(self, tmp_path):
        file_path = str(tmp_path / "mars_custom")
        scb.SpiceManager._write_tk_frame(
            file_path, "MARS_CUSTOM_TEST", 1400500, 499, "IAU_MARS",
            matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        )
        content = open(file_path + ".tf").read()
        assert "0 1 0 1 0 0 0 0 1" in content

    def test_write_tk_frame_registered_id_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="already registered"):
            scb.SpiceManager._write_tk_frame(str(tmp_path / "bad"), "X", 1, 499, "IAU_MARS")


def test_get_intervals_returns_start_end_jd_strings():
    spk_file = os.getcwd() + "/data/kernels/locked/spk/de432s.bsp"
    interval = scb.SpiceManager.get_intervals(spk_file, 301)  # Moon
    assert isinstance(interval, list)
    assert len(interval) == 2
    for s in interval:
        assert str(s).startswith("JD ")
        assert float(str(s).removeprefix("JD ")) > 0


class TestWriteSPKMetadata:
    """write_spk_metadata_json writes JSON sidecars only (no SPK segment),
    covering all four optional sidecars via the non-sequence code path.
    A minimal state_definition is used: index 2 is the only "parameter"
    entry, holding (name, *_, frame) where frame provides frame.name for
    the "parameters_body" field."""

    @staticmethod
    def _fixtures(tmp_path, name="testsat"):
        kg = scb.Units.get_units(["kg"])
        body = scb.Body(f"{name.upper()}", spice_id=-999001, mass=scb.ArrayWUnits(500.0, kg))
        mu_units = scb.Units.get_units(["km"]) ** 3 / scb.Units.get_units(["sec"]) ** 2
        state_definition = [None, None, ("mu", None, None, None, J2000)]
        parameters_json = scb.ArrayWUnits(4902.8, mu_units)
        stms = np.array([np.eye(6), np.eye(6)])
        mass_profile = scb.SCBPolynomial([500.0], domain=(0.0, 100.0))
        return body, state_definition, parameters_json, stms, mass_profile

    def test_writes_all_four_sidecars(self, tmp_path):
        body, state_definition, parameters_json, stms, mass_profile = self._fixtures(tmp_path)
        spk_name = str(tmp_path / "test_meta.bsp")

        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=body,
            origin_ID=399,
            ref_frame=J2000,
            epochs_TDB=[0.0, 100.0],
            degree_poly=1,
            parameters_JSON=parameters_json,
            STMs_JSON=stms,
            state_definition=state_definition,
            sequence_definition=None,
            leg_data=None,
            propagator_settings={"integrator": "RK4"},
            mass_profile=mass_profile,
        )

        base = str(tmp_path / "test_meta")
        assert os.path.isfile(base + "_parameters.json")
        assert os.path.isfile(base + "_stms.json")
        assert os.path.isfile(base + "_settings.json")
        assert os.path.isfile(base + "_mass_profile.json")

        params = json.load(open(base + "_parameters.json"))
        assert params["parameters_def"] == ["mu"]
        assert params["parameters"] == pytest.approx(4902.8)

        settings = json.load(open(base + "_settings.json"))
        assert settings == {"integrator": "RK4"}

        mass_data = json.load(open(base + "_mass_profile.json"))
        assert mass_data["mass_poly"]["coeffs"] == [500.0]

    def test_no_sidecars_writes_nothing(self, tmp_path):
        """All four sidecar args are optional; omitting them writes no files."""
        body, *_ = self._fixtures(tmp_path)
        spk_name = str(tmp_path / "test_meta_empty.bsp")
        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name, body=body, origin_ID=399, ref_frame=J2000,
            epochs_TDB=[0.0, 100.0], degree_poly=1,
        )
        assert list(tmp_path.iterdir()) == []


class TestWriteSPKSegmentType13:
    """write_spk_segment_type13 writes a real type-13 SPK segment plus the
    same optional JSON sidecars as write_spk_metadata_json, and exercises
    _open_spk_file's new-file vs. append-to-existing-file branches."""

    @staticmethod
    def _state_fixtures():
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        epochs_tdb = [et0, et0 + 100.0]
        state_spice = np.array([
            [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
            [7000.0, 100.0, 0.0, -0.1, 7.5, 0.0],
        ])
        return epochs_tdb, state_spice

    def test_creates_new_spk_file_with_sidecars(self, tmp_path):
        kg = scb.Units.get_units(["kg"])
        body = scb.Body("TESTSAT002", spice_id=-999002, mass=scb.ArrayWUnits(500.0, kg))
        epochs_tdb, state_spice = self._state_fixtures()
        mu_units = scb.Units.get_units(["km"]) ** 3 / scb.Units.get_units(["sec"]) ** 2
        state_definition = [None, None, ("mu", None, None, None, J2000)]

        spk_name = str(tmp_path / "test_seg.bsp")
        try:
            scb.SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name,
                SPK_SEG_ID="TESTSEG",
                body=body,
                origin_ID=399,
                ref_frame=J2000,
                epochs_TDB=epochs_tdb,
                degree_poly=1,
                state_SPICE=state_spice,
                parameters_JSON=scb.ArrayWUnits(4902.8, mu_units),
                STMs_JSON=np.array([np.eye(6), np.eye(6)]),
                state_definition=state_definition,
                propagator_settings={"integrator": "RK4"},
                mass_profile=scb.SCBPolynomial([500.0], domain=(epochs_tdb[0], epochs_tdb[1])),
            )
            assert os.path.isfile(spk_name)
            base = str(tmp_path / "test_seg")
            assert os.path.isfile(base + "_parameters.json")
            assert os.path.isfile(base + "_stms.json")
            assert os.path.isfile(base + "_settings.json")
            assert os.path.isfile(base + "_mass_profile.json")
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)

    def test_second_write_appends_to_existing_file(self, tmp_path):
        """A second write_spk_segment_type13 call on the same SPK_name must
        hit _open_spk_file's append branch (spkopa) instead of creating a
        fresh file (spkopn) -- verified by both segments being present
        afterward (file size is unreliable: filesystem block rounding can
        mask a small second segment)."""
        body = scb.Body("TESTSAT003", spice_id=-999003)
        epochs_tdb, state_spice = self._state_fixtures()
        spk_name = str(tmp_path / "test_seg_append.bsp")
        try:
            scb.SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name, SPK_SEG_ID="SEG1", body=body, origin_ID=399,
                ref_frame=J2000, epochs_TDB=epochs_tdb, degree_poly=1, state_SPICE=state_spice,
            )
            scb.SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name, SPK_SEG_ID="SEG2", body=body, origin_ID=399,
                ref_frame=J2000, epochs_TDB=epochs_tdb, degree_poly=1, state_SPICE=state_spice,
            )

            scb.SpiceManager.unload_kernel_from_pool(spk_name)
            handle = spice.dafopr(spk_name)
            spice.dafbfs(handle)
            names = []
            while spice.daffna():
                names.append(spice.dafgn())
            spice.dafcls(handle)
            assert names == ["SEG1", "SEG2"]
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)

    def test_warns_only_when_appended_epochs_are_already_covered(self, tmp_path, capsys):
        """Appending to an existing SPK is silent; re-covering epochs the file
        already holds for that body is not. Segments do not merge, the later one
        outranks the earlier, so the warning has to fire exactly when old states
        stop being the ones SPICE returns -- and not when consecutive legs merely
        chain end to end, which is how a sequence is written."""
        body = scb.Body("TESTSAT004", spice_id=-999004)
        other = scb.Body("TESTSAT005", spice_id=-999005)
        state_spice = self._state_fixtures()[1]
        spk_name = str(tmp_path / "test_seg_overlap.bsp")

        def write(b, t0, t1, seg_id):
            scb.SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name, SPK_SEG_ID=seg_id, body=b, origin_ID=399,
                ref_frame=J2000, epochs_TDB=[t0, t1], degree_poly=1,
                state_SPICE=state_spice,
            )
            return capsys.readouterr().out

        try:
            assert "Warning" not in write(body, 0.0, 100.0, "SEG1")       # new file
            assert "Warning" not in write(body, 200.0, 300.0, "SEG2")     # disjoint
            assert "Warning" not in write(body, 100.0, 200.0, "SEG3")     # touches both
            assert "Warning" not in write(other, 0.0, 100.0, "SEG4")      # other body

            out = write(body, 50.0, 250.0, "SEG5")                        # real overlap
            assert "Warning" in out and "-999004" in out
            assert "2000 JAN 01 12:00:50.000 to 2000 JAN 01 12:04:10.000 TDB" in out
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)

    def test_overlapping_coverage_is_empty_for_a_body_not_in_the_file(self, tmp_path):
        """A body with no segments has no coverage to shadow; spkcov reports an
        empty window for an unknown ID rather than raising, so no guard is needed."""
        body = scb.Body("TESTSAT006", spice_id=-999006)
        epochs_tdb, state_spice = self._state_fixtures()
        spk_name = str(tmp_path / "test_seg_absent.bsp")
        try:
            scb.SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name, SPK_SEG_ID="SEG1", body=body, origin_ID=399,
                ref_frame=J2000, epochs_TDB=epochs_tdb, degree_poly=1,
                state_SPICE=state_spice,
            )
            scb.SpiceManager.unload_kernel_from_pool(spk_name)

            assert scb.SpiceManager._overlapping_coverage(
                spk_name, -999999, epochs_tdb[0], epochs_tdb[-1]
            ) == []
            assert scb.SpiceManager._overlapping_coverage(
                spk_name, body.spice_id, epochs_tdb[0], epochs_tdb[-1]
            ) == [(epochs_tdb[0], epochs_tdb[-1])]
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)


class TestType13HermiteConsistency:
    """Type 13 is not just a different label on the same records: its position
    curve is built from the sampled velocities, so what SPICE hands back is one
    function together with its own derivative, and it does not step where the
    interpolation window moves.

    Type 9 fitted position and velocity with two independent Lagrange
    polynomials over a window re-centred on the request time.  The velocity it
    returned was therefore not d/dt of the position it returned, and halfway
    between two nodes -- where the window switches, and where neither window's
    polynomial is pinned to anything -- the position curve jumped.  On the
    integrator's own adaptive grid that jump is the spike this switch removes.
    """

    MU = 398600.435507  # km^3/s^2
    RADIUS = 7000.0  # km
    BODY = scb.Body("TESTSAT013", spice_id=-999013)
    TARGET = "-999013"

    @classmethod
    def _analytic_states(cls, ets, et0):
        """A circular orbit: smooth, not a polynomial (so no interpolant can be
        exact by accident), and with position and velocity that are exactly each
        other's derivative."""
        n = np.sqrt(cls.MU / cls.RADIUS ** 3)
        theta = n * (np.asarray(ets) - et0)
        zero = np.zeros_like(theta)
        return np.column_stack([
            cls.RADIUS * np.cos(theta),
            cls.RADIUS * np.sin(theta),
            zero,
            -cls.RADIUS * n * np.sin(theta),
            cls.RADIUS * n * np.cos(theta),
            zero,
        ])

    @classmethod
    def _write(cls, tmp_path):
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        period = 2 * np.pi * np.sqrt(cls.RADIUS ** 3 / cls.MU)

        # Deliberately uneven, the way an adaptive integrator's steps are, and
        # coarse enough that the interpolant is doing real work between nodes.
        fractions = np.array([0.0, 0.03, 0.11, 0.17, 0.31, 0.38, 0.52,
                              0.61, 0.70, 0.79, 0.88, 0.94, 1.0])
        ets = et0 + fractions * period

        spk_name = str(tmp_path / "hermite.bsp")
        scb.SpiceManager.write_spk_segment_type13(
            SPK_name=spk_name,
            SPK_SEG_ID="HERMITE SEGMENT",
            body=cls.BODY,
            origin_ID=EARTH_ID,
            ref_frame=J2000,
            epochs_TDB=list(ets),
            degree_poly=3,
            state_SPICE=cls._analytic_states(ets, et0),
        )
        return spk_name, ets

    @classmethod
    def _states_at(cls, ets):
        return np.array([
            spice.spkezr(cls.TARGET, float(et), "J2000", "NONE", "EARTH")[0]
            for et in ets
        ])

    def test_velocity_is_the_derivative_of_position(self, tmp_path):
        spk_name, ets = self._write(tmp_path)
        try:
            # Sampled between the nodes, where the two curves are free to
            # disagree; at a node both are pinned to the written state.
            sample = np.linspace(ets[0], ets[-1], 200)[1:-1]
            h = 0.2  # s, short against every interval, long against roundoff
            numerical = (
                self._states_at(sample + h)[:, :3]
                - self._states_at(sample - h)[:, :3]
            ) / (2.0 * h)
            returned = self._states_at(sample)[:, 3:]
            speed = np.linalg.norm(returned, axis=1).max()

            # Differencing one cubic is off by O(h^2 * jerk), parts per billion
            # of the orbital speed here.  Two independent polynomials disagree
            # by parts per thousand.
            assert np.abs(numerical - returned).max() < 1e-6 * speed
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)

    def test_position_does_not_jump_where_the_window_moves(self, tmp_path):
        spk_name, ets = self._write(tmp_path)
        try:
            # Dense enough to straddle every node and every midpoint between
            # nodes: the two places an interpolation window can change.
            sample = np.linspace(ets[0], ets[-1], 4001)
            pos = self._states_at(sample)[:, :3]
            step = np.linalg.norm(np.diff(pos, axis=0), axis=1)

            # On a circular orbit a continuous interpolant advances by nearly
            # the same distance every sample.  A window-switch discontinuity
            # shows up as one step out of scale with the rest, and stays that
            # size however fine the sampling gets -- so compare the steps to
            # each other rather than to a tolerance in km.
            assert step.max() < 3.0 * np.median(step)
        finally:
            scb.SpiceManager.unload_kernel_from_pool(spk_name)


class TestPrepareStmData:
    """_prepare_stm_data's STMs-normalization branches, called directly
    (no MissionSequence needed for the non-sequence code path)."""

    STATE_DEFINITION = [None, None, ("mu", None, None, None, J2000)]

    def test_none_stms_gives_empty_dict(self):
        data = scb.SpiceManager._prepare_stm_data(
            None, [0.0, 100.0], -999, 399, J2000, None, self.STATE_DEFINITION, None, None,
        )
        assert data["STMs"] == {}

    def test_single_2d_ndarray(self):
        data = scb.SpiceManager._prepare_stm_data(
            np.eye(6), [0.0], -999, 399, J2000, None, self.STATE_DEFINITION, None, None,
        )
        assert list(data["STMs"].keys()) == ["0.0"]

    def test_list_of_ndarrays(self):
        data = scb.SpiceManager._prepare_stm_data(
            [np.eye(6), 2 * np.eye(6)], [0.0, 100.0], -999, 399, J2000, None,
            self.STATE_DEFINITION, None, None,
        )
        assert list(data["STMs"].keys()) == ["0.0", "100.0"]

    def test_1d_array_uses_atleast_2d(self):
        data = scb.SpiceManager._prepare_stm_data(
            np.array([1.0, 2.0, 3.0]), [0.0], -999, 399, J2000, None,
            self.STATE_DEFINITION, None, None,
        )
        assert data["STMs"]["0.0"] == [[1.0, 2.0, 3.0]]

    def test_list_of_plain_lists(self):
        """Elements of the STMs iterable that are already plain lists (not
        ndarrays) are appended as-is."""
        data = scb.SpiceManager._prepare_stm_data(
            [[[1, 0], [0, 1]]], [0.0], -999, 399, J2000, None,
            self.STATE_DEFINITION, None, None,
        )
        assert data["STMs"]["0.0"] == [[1, 0], [0, 1]]

    def test_parameter_labels_skip_every_position_and_velocity(self):
        """The STM spans the whole state, so every parameter column is
        labelled, a ground station's bias included, and a second propagated
        body's position and velocity are not. The old "skip the first two
        entries" rule listed Earth's pos/vel as parameters."""
        sc = _StubOwner("ORBITER", -999)
        earth = _StubOwner("EARTH", 399)
        station = _StubOwner("DSS-14", 399014)
        state_definition = [
            ("position", 3, "estimated", "dynamic", sc, None),
            ("velocity", 3, "estimated", "dynamic", sc, None),
            ("eta_srp", 1, "estimated", "static", sc, None),
            ("position", 3, "estimated", "dynamic", earth, None),
            ("velocity", 3, "estimated", "dynamic", earth, None),
            ("gm", 1, "estimated", "static", earth, None),
            ("range_bias", 1, "considered", "static", station, None),
        ]

        data = scb.SpiceManager._prepare_stm_data(
            np.eye(19), [0.0], -999, 399, J2000, None, state_definition, None, None,
        )

        assert data["parameters_def"] == ["eta_srp", "gm", "range_bias"]
        assert data["parameters_body"] == ["ORBITER", "EARTH", "DSS-14"]


class TestWriteStmSidecar:

    def test_pickle_format_writes_pkl_file(self, tmp_path):
        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_stm_sidecar(spk_name, {"a": 1}, stm_format="pickle")
        assert os.path.isfile(str(tmp_path / "test_stms.pkl"))

    def test_unknown_format_raises(self, tmp_path):
        spk_name = str(tmp_path / "test.bsp")
        with pytest.raises(ValueError, match="Unknown stm_format"):
            scb.SpiceManager._write_stm_sidecar(spk_name, {"a": 1}, stm_format="xml")


class TestWriteParametersToJsonDefault:
    """_write_parameters_to_json's `_default` fallback serializer, exercised
    with values that aren't natively JSON-serializable."""

    def test_serializes_mixed_wrapper_types(self, tmp_path):
        km = scb.Units.get_units(["km"])
        spk_name = str(tmp_path / "test.bsp")
        data = {
            "awf": scb.ArrayWFrame(np.array([1.0, 2.0, 3.0]), km, J2000),
            "awu": scb.ArrayWUnits(np.array([4.0, 5.0]), km),
            # np.float64 subclasses Python float (natively JSON-serializable);
            # np.int64 does not subclass int, so it exercises the numpy-scalar
            # branch of `_default`.
            "np_scalar": np.int64(7),
        }
        scb.SpiceManager._write_parameters_to_json(spk_name, data, "_default_test.json")
        result = json.load(open(str(tmp_path / "test_default_test.json")))
        assert result["awf"]["frame"] == "J2000"
        assert result["awf"]["values"] == [1.0, 2.0, 3.0]
        assert result["awu"]["values"] == [4.0, 5.0]
        assert result["np_scalar"] == 7

    def test_arraywframe_duck_type_with_failing_properties(self, tmp_path):
        """A duck-typed "ArrayWFrame" (matched by class name, not isinstance)
        whose .quantity/.frame properties both raise must degrade gracefully
        to None fields rather than propagating the exception."""

        class FakeAWF:
            @property
            def quantity(self):
                raise RuntimeError("no quantity")

            @property
            def frame(self):
                raise RuntimeError("no frame")

        FakeAWF.__name__ = "ArrayWFrame"

        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"fake": FakeAWF()}, "_fake_awf.json")
        result = json.load(open(str(tmp_path / "test_fake_awf.json")))
        assert result["fake"] == {
            "frame": None, "values": None, "unitsPower": [], "unitsScale": [],
        }

    def test_arraywframe_duck_type_with_failing_units(self, tmp_path):
        """A duck-typed "ArrayWFrame" whose quantity has no .values and whose
        .units raises must fall back to str(quantity) and empty unit lists."""

        class FakeQuantity:
            def __repr__(self):
                return "fakequantity"

        class FakeAWF:
            quantity = FakeQuantity()
            frame = None

        FakeAWF.__name__ = "ArrayWFrame"

        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"fake": FakeAWF()}, "_fake_awf2.json")
        result = json.load(open(str(tmp_path / "test_fake_awf2.json")))
        assert result["fake"]["values"] == "fakequantity"
        assert result["fake"]["unitsPower"] == []

    def test_arraywunits_duck_type_with_failing_values(self, tmp_path):
        """A duck-typed "ArrayWUnits" whose .values raises must fall back to
        str(o) for values and empty unit lists."""

        class FakeAWU:
            @property
            def values(self):
                raise RuntimeError("no values")

        FakeAWU.__name__ = "ArrayWUnits"

        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"fake": FakeAWU()}, "_fake_awu.json")
        result = json.load(open(str(tmp_path / "test_fake_awu.json")))
        assert result["fake"]["unitsPower"] == []
        assert isinstance(result["fake"]["values"], str)

    def test_duck_typed_values_attribute(self, tmp_path):
        class DuckValues:
            values = np.array([1.0, 2.0, 3.0])

        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"duck": DuckValues()}, "_duck.json")
        result = json.load(open(str(tmp_path / "test_duck.json")))
        assert result["duck"] == [1.0, 2.0, 3.0]

    def test_last_resort_stringifies_unknown_object(self, tmp_path):
        class Weird:
            pass

        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"weird": Weird()}, "_weird.json")
        result = json.load(open(str(tmp_path / "test_weird.json")))
        assert result["weird"].startswith("<")

    def test_pkl_suffix_writes_pickle(self, tmp_path):
        spk_name = str(tmp_path / "test.bsp")
        scb.SpiceManager._write_parameters_to_json(spk_name, {"a": np.float64(1.5)}, "_pickle_test.pkl")
        import pickle
        with open(str(tmp_path / "test_pickle_test.pkl"), "rb") as f:
            assert pickle.load(f) == {"a": 1.5}


class TestOpenCKFile:

    def test_appends_default_extension(self, tmp_path):
        ck_path_noext = str(tmp_path / "test_ck_noext")
        path, handle = scb.SpiceManager._open_ck_file(ck_path_noext)
        try:
            assert path == ck_path_noext + ".bc"
            assert os.path.isfile(path)
        finally:
            spice.dafcls(handle)

    def test_creates_missing_parent_directory(self, tmp_path):
        ck_path = str(tmp_path / "subdir" / "test_ck.bc")
        path, handle = scb.SpiceManager._open_ck_file(ck_path)
        try:
            assert os.path.isdir(str(tmp_path / "subdir"))
        finally:
            spice.dafcls(handle)

    def test_reopening_removes_existing_file(self, tmp_path):
        ck_path = str(tmp_path / "test_ck.bc")
        path1, handle1 = scb.SpiceManager._open_ck_file(ck_path)
        spice.dafcls(handle1)

        path2, handle2 = scb.SpiceManager._open_ck_file(ck_path)
        try:
            assert path2 == path1
            assert os.path.isfile(path2)
        finally:
            spice.dafcls(handle2)
            scb.SpiceManager.unload_kernel_from_pool(path2)


class TestEnsureQuatAvArrays:

    def test_ensure_quat_array_transposes_4byN(self):
        q = scb.SpiceManager._ensure_quat_array(np.ones((4, 5)), 5)
        assert q.shape == (5, 4)

    def test_ensure_quat_array_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="quats must be shape"):
            scb.SpiceManager._ensure_quat_array(np.ones((3, 3)), 5)

    def test_ensure_av_array_transposes_3byN(self):
        a = scb.SpiceManager._ensure_av_array(np.ones((3, 5)), 5)
        assert a.shape == (5, 3)

    def test_ensure_av_array_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="avvs must be shape"):
            scb.SpiceManager._ensure_av_array(np.ones((2, 2)), 5)


class TestSkyGeometry:

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_get_elevation_angle(self):
        elev = scb.SpiceManager.get_elevation_angle("MOON", self.ET)
        assert hasattr(elev, "values")
        assert -np.pi / 2 <= float(elev.values) <= np.pi / 2

    def test_get_sep_angle(self):
        """SEP angle is between 0 and π radians by definition."""
        sep = scb.SpiceManager.get_sep_angle("MOON", self.ET)
        assert hasattr(sep, "values")
        assert 0.0 <= float(sep.values) <= np.pi


class TestVisibility:

    ET0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
    ET1 = ET0 + 30 * 86400.0  # 30-day window

    KWARGS = {
        "step_size": 3600.0, "angle_limit": 0.0, "inframe": "J2000", "abcorr": "NONE",
        "crdsys": "LATITUDINAL", "coord": "LATITUDE", "relate": ">",
    }

    def test_visibility_earth_moon_30_days(self):
        """Moon latitude > 0° in J2000 must occur multiple times in 30 days."""
        num_w, windows = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", self.ET0, self.ET1, **self.KWARGS,
        )
        assert isinstance(num_w, int)
        assert isinstance(windows, list)
        assert num_w == len(windows)
        assert num_w > 0

    def test_windows_have_ordered_bounds(self):
        """Every visibility window must have its start time before its end time."""
        _, windows = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", self.ET0, self.ET1, **self.KWARGS,
        )
        for start, end in windows:
            assert start <= end

    def test_impossible_angle_limit_returns_zero_windows(self):
        """Latitude > 91° is geometrically impossible; must return 0 windows."""
        num_w, windows = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", self.ET0, self.ET1,
            step_size=3600.0, angle_limit=91.0, inframe="J2000", abcorr="NONE",
            crdsys="LATITUDINAL", coord="LATITUDE", relate=">",
        )
        assert num_w == 0
        assert windows == []

    def test_single_epoch_array_accepted(self):
        """Single-value EpochArray inputs must give the same count as bare floats."""
        num_float, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", self.ET0, self.ET1, **self.KWARGS,
        )
        ea0 = scb.EpochArray(self.ET0, "TDB")
        ea1 = scb.EpochArray(self.ET1, "TDB")
        num_ea, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", ea0, ea1, **self.KWARGS,
        )
        assert num_ea == num_float

    def test_multi_value_epoch_array_raises(self):
        """EpochArray with more than one entry must raise ValueError."""
        bad_ea = scb.EpochArray(np.array([self.ET0, self.ET1]), "TDB")
        with pytest.raises(ValueError, match="epoch_start"):
            scb.SpiceManager.get_observer_target_visibility_windows(
                "EARTH", "MOON", bad_ea, self.ET1, **self.KWARGS,
            )

    def test_bad_epoch_type_raises(self):
        """Passing a non-numeric, non-EpochArray epoch must raise TypeError."""
        with pytest.raises(TypeError):
            scb.SpiceManager.get_observer_target_visibility_windows(
                "EARTH", "MOON", "2024 JAN 01", self.ET1, **self.KWARGS,
            )

    def test_default_inframe_accepts_str_and_body_observer(self):
        """Omitting inframe (default "obs") must resolve the observer's body-fixed
        frame for both a plain string and a Body object, giving the same result."""
        kwargs = {k: v for k, v in self.KWARGS.items() if k != "inframe"}
        num_str, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            "EARTH", "MOON", self.ET0, self.ET1, **kwargs,
        )
        earth = scb.CelestialBody.from_constants("EARTH")
        num_obj, _ = scb.SpiceManager.get_observer_target_visibility_windows(
            earth, "MOON", self.ET0, self.ET1, **kwargs,
        )
        assert num_str == num_obj


class TestJSONMethods:
    """get_parameters / get_STMs / get_propagator_settings raise when no
    matching JSON files are found in the folder."""

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    @pytest.mark.parametrize("func_call", [
        lambda sm, et, fp: sm.get_parameters("MOON", et, "J2000", "EARTH", folder_path_override=fp),
        lambda sm, et, fp: sm.get_STMs("MOON", et, "J2000", "EARTH", folder_path_override=fp),
        lambda sm, et, fp: sm.get_propagator_settings("MOON", "J2000", "EARTH", folder_path_override=fp),
    ], ids=["get_parameters", "get_STMs", "get_propagator_settings"])
    def test_raises_without_json(self, func_call, tmp_path):
        with pytest.raises(Exception):
            func_call(scb.SpiceManager, self.ET, str(tmp_path))


class TestSidecarIsKind:
    """SpiceManager._sidecar_is_kind tells sidecars apart by their payload."""

    LEGS = sequence_params_json([1.0])["legs"]
    SETTINGS = {"body_ID": MOON_ID, "integrator": "DOP853", "force_models": [], "STM0": None}

    @pytest.mark.parametrize(
        "data, kind",
        [
            (flat_params_json(), "parameters"),
            (sequence_params_json([1.0]), "parameters"),
            (stm_json([0.0], {"0.0": EYE6}), "stms"),
            (sequence_stm_json([0.0], {"0.0": EYE6}, LEGS), "stms"),
            (SETTINGS, "settings"),
        ],
        ids=["flat-parameters", "sequence-parameters", "flat-stms", "sequence-stms", "settings"],
    )
    def test_each_sidecar_is_exactly_one_kind(self, data, kind):
        kinds = [k for k in ("parameters", "stms", "settings") if scb.SpiceManager._sidecar_is_kind(data, k)]
        assert kinds == [kind]

    def test_non_dict_is_no_kind(self):
        assert not scb.SpiceManager._sidecar_is_kind([1.0, 2.0], "parameters")


class TestGetParameters:
    """Comprehensive tests for SpiceManager.get_parameters."""

    MOON_BODY = scb.CelestialBody.from_constants("MOON")
    EARTH_BODY = scb.CelestialBody.from_constants("EARTH")

    def test_flat_json_returns_array_w_units(self, tmp_path):
        write_json(tmp_path, "moon_params.json", flat_params_json())
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert hasattr(result, "values")

    def test_uses_kernel_folder_when_no_override_given(self, tmp_path, monkeypatch):
        """Omitting folder_path_override must fall back to cls.kernel_folder."""
        write_json(tmp_path, "moon_params.json", flat_params_json())
        monkeypatch.setattr(scb.SpiceManager, "kernel_folder", str(tmp_path))
        result = scb.SpiceManager.get_parameters("MOON", 50.0, "J2000", "EARTH")
        assert hasattr(result, "values")

    def test_flat_json_linear_interpolation(self, tmp_path):
        # polyfit([0,100], [3,5], 1) at t=50 → 4.0
        write_json(tmp_path, "moon_params.json", flat_params_json(values=[3.0, 5.0]))
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4.0, abs=1e-10)

    def test_flat_json_body_frame_enums_accepted(self, tmp_path):
        write_json(tmp_path, "moon_params.json", flat_params_json())
        result_str = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        result_obj = scb.SpiceManager.get_parameters(
            self.MOON_BODY, 50.0, J2000, self.EARTH_BODY,
            folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result_str.values, result_obj.values)

    def test_multiple_matching_files_warns(self, tmp_path):
        write_json(tmp_path, "moon_params_a.json", flat_params_json(values=[3.0, 3.0]))
        write_json(tmp_path, "moon_params_b.json", flat_params_json(values=[7.0, 7.0]))
        with pytest.warns(RuntimeWarning, match="Multiple matching JSON files"):
            scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_sequence_stm_sidecar_is_not_a_second_match(self, tmp_path):
        """A sequence STM sidecar embeds the same "legs" as the parameters
        sidecar beside it, but it is not a parameters sidecar: the lookup must
        resolve to the one file without a "Multiple matching" warning."""
        params = sequence_params_json(
            ["mu", {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]}]
        )
        write_json(tmp_path, "run_parameters.json", params)
        write_json(
            tmp_path, "run_stms.json",
            sequence_stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}, params["legs"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
            )
        assert float(result.values.ravel()[0]) == pytest.approx(4902.8, rel=1e-12)

    def test_sequence_json_constant_parameter(self, tmp_path):
        data = {
            "body_ID": MOON_ID,
            "origin_ID": EARTH_ID,
            "reference_frame": "J2000",
            "legs": {
                "leg0": {
                    "epochsTDB": [0.0, 100.0],
                    "body_ID": MOON_ID,
                    "origin_ID": EARTH_ID,
                    "reference_frame": "J2000",
                    "parameters": [
                        "mu",
                        {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]},
                    ],
                }
            },
        }
        write_json(tmp_path, "moon_seq.json", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4902.8, rel=1e-6)

    def test_sequence_json_timeseries_parameter_interpolates(self, tmp_path):
        data = {
            "body_ID": MOON_ID,
            "origin_ID": EARTH_ID,
            "reference_frame": "J2000",
            "legs": {
                "leg0": {
                    "epochsTDB": [0.0, 100.0],
                    "body_ID": MOON_ID,
                    "origin_ID": EARTH_ID,
                    "reference_frame": "J2000",
                    "parameters": [
                        "mu",
                        {
                            "epochsTDB": [0.0, 100.0],
                            "values": [[1.0], [3.0]],
                            "unitsPower": [0, 0, 3, -2],
                            "unitsScale": [1.0],
                        },
                    ],
                }
            },
        }
        write_json(tmp_path, "moon_seq.json", data)
        # linear interp between 1.0 and 3.0 at t=50 → 2.0
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(2.0, abs=1e-6)

    def test_sequence_json_epoch_outside_all_legs_raises(self, tmp_path):
        data = {
            "body_ID": MOON_ID,
            "origin_ID": EARTH_ID,
            "reference_frame": "J2000",
            "legs": {
                "leg0": {
                    "epochsTDB": [0.0, 100.0],
                    "body_ID": MOON_ID,
                    "origin_ID": EARTH_ID,
                    "reference_frame": "J2000",
                    "parameters": [
                        "mu",
                        {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]},
                    ],
                }
            },
        }
        write_json(tmp_path, "moon_seq.json", data)
        with pytest.raises(ValueError, match="epoch_time does not fall within any leg"):
            scb.SpiceManager.get_parameters(
                "MOON", 9999.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_ignores_files_with_unrecognized_extension(self, tmp_path):
        """A stray non-.json/.pkl file in the folder must be skipped, not loaded."""
        (tmp_path / "notes.txt").write_text("not a sidecar file")
        write_json(tmp_path, "moon_params.json", flat_params_json())
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert hasattr(result, "values")

    def test_sequence_json_non_dict_leg_is_skipped(self, tmp_path):
        """A non-dict entry under "legs" must be skipped, not crash the lookup."""
        data = sequence_params_json(
            ["mu", {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]}],
            extra_legs={"leg_bad": "not a dict"},
        )
        write_json(tmp_path, "moon_seq.json", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4902.8, rel=1e-6)

    def test_sequence_json_empty_epochs_leg_is_skipped(self, tmp_path):
        """A leg with an empty epochsTDB list must be skipped in favor of a valid leg."""
        data = sequence_params_json(
            ["mu", {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]}],
            extra_legs={
                "leg_empty": {
                    "epochsTDB": [],
                    "body_ID": MOON_ID,
                    "origin_ID": EARTH_ID,
                    "reference_frame": "J2000",
                    "parameters": ["mu", {"values": [1.0]}],
                }
            },
        )
        write_json(tmp_path, "moon_seq.json", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4902.8, rel=1e-6)

    def test_sequence_json_leg_without_parameters_is_skipped(self, tmp_path):
        """A leg whose epoch range matches but has no "parameters" key must be skipped."""
        data = sequence_params_json(
            ["mu", {"values": [4902.8], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]}],
            extra_legs={
                "leg_noparam": {
                    "epochsTDB": [0.0, 100.0],
                    "body_ID": MOON_ID,
                    "origin_ID": EARTH_ID,
                    "reference_frame": "J2000",
                }
            },
        )
        write_json(tmp_path, "moon_seq.json", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4902.8, rel=1e-6)

    @pytest.mark.parametrize(
        "parameters, match",
        [
            ("just a string", "Unexpected 'parameters' format"),
            (["mu"], "Unexpected 'parameters' format"),
            (["mu", {"values": [1.0]}, "extra"], "Malformed parameters list"),
            (["mu", {"unitsPower": [0, 0, 3, -2]}], "Malformed parameters entry"),
            (["mu", "not-a-dict"], "Malformed parameters entry"),
            (["mu", {"values": None}], "Missing 'values'"),
        ],
        ids=[
            "Not A List",
            "Too Short",
            "Odd Length",
            "Entry Missing Values Key",
            "Entry Not Dict",
            "Values Explicitly None",
        ],
    )
    def test_sequence_json_malformed_parameters_raises(self, tmp_path, parameters, match):
        data = sequence_params_json(parameters)
        write_json(tmp_path, "moon_seq.json", data)
        with pytest.raises(ValueError, match=match):
            scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_sequence_json_timeseries_falls_back_to_leg_epochs(self, tmp_path):
        """A time-series parameter dict without its own epochsTDB must use the leg's."""
        data = sequence_params_json(
            ["mu", {"values": [[1.0], [3.0]], "unitsPower": [0, 0, 3, -2], "unitsScale": [1.0]}],
            epochs=(0.0, 100.0),
        )
        write_json(tmp_path, "moon_seq.json", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(2.0, abs=1e-6)

    def test_sequence_json_timeseries_length_mismatch_raises(self, tmp_path):
        data = sequence_params_json(
            ["mu", {"epochsTDB": [0.0, 50.0, 100.0], "values": [[1.0], [2.0]]}]
        )
        write_json(tmp_path, "moon_seq.json", data)
        with pytest.raises(ValueError, match="length mismatch"):
            scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_sequence_json_vector_param_with_mismatched_units_size_raises(self, tmp_path):
        """Each parameter's unitsPower is one 1x4 block, built into that
        parameter's own Units. A block of any other length (one power here)
        is malformed, and Dimensions() rejects it."""
        data = sequence_params_json(
            [
                "mu",
                {
                    "epochsTDB": [0.0, 100.0],
                    "values": [[1.0, 2.0], [3.0, 4.0]],
                    "unitsPower": [1],
                    "unitsScale": [2.0],
                },
            ]
        )
        write_json(tmp_path, "moon_seq.json", data)
        with pytest.raises(ValueError, match=r"must be a 1x4 array"):
            scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_sequence_pickle_1d_timeseries_values_reshaped(self, tmp_path):
        """A pickled sidecar may store "values" as a genuine 1-D ndarray (unlike
        JSON, which always decodes nested lists); this must be reshaped to a
        column vector before interpolation, same as the list-of-lists case."""
        data = sequence_params_json(
            [
                "mu",
                {
                    "epochsTDB": [0.0, 100.0],
                    "values": np.array([1.0, 3.0]),
                    "unitsPower": [0, 0, 3, -2],
                    "unitsScale": [1.0],
                },
            ]
        )
        write_pickle(tmp_path, "moon_seq.pkl", data)
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert float(result.values.ravel()[0]) == pytest.approx(2.0, abs=1e-6)

    # A finite-burn leg: a unitless thrust scale next to two angles in rad,
    # laid out the way the sequence writer emits a multi-parameter leg
    BURN_LEG = [
        "k_thrust1", {"values": [0.9], "unitsPower": [], "unitsScale": []},
        "dRA1", {"values": [0.4], "unitsPower": [0, 0, 0, 1], "unitsScale": [0, 0, 0, 1]},
        "dDEC1", {"values": [0.3], "unitsPower": [0, 0, 0, 1], "unitsScale": [0, 0, 0, 1]},
    ]

    def test_sequence_json_mixed_units_stack_is_dimensionless(self, tmp_path):
        """Every parameter of a leg is read, and a stack whose parameters
        disagree on units is not labelled with any one of them."""
        write_json(tmp_path, "moon_seq.json", sequence_params_json(self.BURN_LEG))
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result.values.ravel(), [0.9, 0.4, 0.3])
        assert np.array_equal(result.units.dimensions.powers, [0, 0, 0, 0])

    def test_sequence_json_mixed_units_per_parameter(self, tmp_path):
        write_json(tmp_path, "moon_seq.json", sequence_params_json(self.BURN_LEG))
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path),
            per_parameter=True,
        )
        assert list(result) == ["k_thrust1", "dRA1", "dDEC1"]
        assert float(np.ravel(result["k_thrust1"].values)[0]) == pytest.approx(0.9)
        assert result["k_thrust1"].units.unitless
        assert result["dRA1"].units == rad
        assert float(np.ravel(result["dDEC1"].values)[0]) == pytest.approx(0.3)
        assert result["dDEC1"].units == rad

    def test_sequence_json_shared_units_are_kept(self, tmp_path):
        write_json(tmp_path, "moon_seq.json", sequence_params_json(self.BURN_LEG[2:]))
        result = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result.values.ravel(), [0.4, 0.3])
        assert result.units == rad
    def test_sequence_json_every_parameter_of_a_leg_round_trips(self, tmp_path):
        """A leg holding several parameters is written whole and read back
        whole: the stack is dimensionless because the units differ, and
        ``per_parameter`` gives each parameter its own values and units. The
        writer used to keep only the first pair, and the reader concatenated
        the 1x4 power blocks into one Dimensions, which rejects more than one."""
        definition = _StubSequenceDefinition(["Leg"])
        leg_data = {
            "by_seq_idx": {
                0: {"body_ID": MOON_ID, "origin_ID": EARTH_ID, "reference_frame": "J2000"}
            }
        }
        dv = scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000)
        bias = scb.ArrayWFrame(np.array([0.005]), km, J2000)
        legs = scb.SpiceManager._prepare_sequence_data(
            definition, leg_data, [[("dv", dv), ("range_bias", bias)]]
        )
        scb.SpiceManager._write_parameters_to_json(
            str(tmp_path / "moon_seq.bsp"),
            {"body_ID": MOON_ID, "origin_ID": EARTH_ID, "reference_frame": "J2000",
             "legs": legs},
            "_parameters.json",
        )

        stacked = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        split = scb.SpiceManager.get_parameters(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path),
            per_parameter=True,
        )

        np.testing.assert_allclose(stacked.values.ravel(), [1e-3, -2e-3, 3e-3, 0.005])
        assert stacked.units == unitless
        assert list(split) == ["dv", "range_bias"]
        np.testing.assert_allclose(split["dv"].values, [1e-3, -2e-3, 3e-3])
        assert split["dv"].units == km / sec
        assert float(split["range_bias"].values) == pytest.approx(0.005)
        assert split["range_bias"].units == km


class TestGetSTMs:
    """Comprehensive tests for SpiceManager.get_STMs."""

    MOON_BODY = scb.CelestialBody.from_constants("MOON")
    EARTH_BODY = scb.CelestialBody.from_constants("EARTH")

    def test_exact_epoch_returns_stored_stm(self, tmp_path):
        write_json(tmp_path, "moon_stms.json", stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}))
        result = scb.SpiceManager.get_STMs(
            "MOON", 0.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result, np.eye(6))

    def test_interpolated_epoch_linear(self, tmp_path):
        write_json(tmp_path, "moon_stms.json", stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}))
        # at t=50, alpha=0.5: 0.5*I + 0.5*(2I) = 1.5*I
        result = scb.SpiceManager.get_STMs(
            "MOON", 50.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result, 1.5 * np.eye(6))

    def test_epoch_out_of_range_raises(self, tmp_path):
        write_json(tmp_path, "moon_stms.json", stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}))
        with pytest.raises(ValueError):
            scb.SpiceManager.get_STMs(
                "MOON", 9999.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_body_frame_enums_accepted(self, tmp_path):
        write_json(tmp_path, "moon_stms.json", stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}))
        result_str = scb.SpiceManager.get_STMs(
            "MOON", 0.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        result_obj = scb.SpiceManager.get_STMs(
            self.MOON_BODY, 0.0, J2000, self.EARTH_BODY,
            folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result_str, result_obj)

    def test_no_matching_file_raises(self, tmp_path):
        with pytest.raises(
            scb.SpiceManager.SidecarNotFoundError, match="No STM sidecar"
        ):
            scb.SpiceManager.get_STMs(
                "MOON", 0.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_sequence_stm_sidecar_with_legs_matches(self, tmp_path):
        """The "legs" a sequence STM sidecar carries for context must not stop
        it being read as an STM sidecar, even beside its parameters sidecar."""
        params = sequence_params_json(["mu", {"values": [1.0], "unitsPower": [0, 0, 0, 0], "unitsScale": [1.0]}])
        write_json(tmp_path, "run_parameters.json", params)
        write_json(
            tmp_path, "run_stms.json",
            sequence_stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}, params["legs"]),
        )
        result = scb.SpiceManager.get_STMs(
            "MOON", 50.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
        )
        np.testing.assert_allclose(result, 1.5 * np.eye(6))

    def test_pkl_sidecar_matches(self, tmp_path):
        """A "_stms.pkl" sidecar must be read via pickle and match a .json one."""
        write_pickle(tmp_path, "moon_stms.pkl", stm_json([0.0, 100.0], {"0.0": EYE6, "100.0": TWO6}))
        result = scb.SpiceManager.get_STMs(
            "MOON", 0.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        np.testing.assert_allclose(result, np.eye(6))

    def test_ignores_files_with_unrelated_suffix(self, tmp_path):
        """A file not ending in "_stms.json"/"_stms.pkl" must be skipped (not
        parsed as a sidecar), still leaving no match found.

        Kept as the only file in the folder: the loop breaks on the first
        match, so pairing it with a valid sidecar makes coverage of the skip
        branch depend on unreliable os.listdir ordering.
        """
        (tmp_path / "notes.txt").write_text("not a sidecar file")
        with pytest.raises(
            scb.SpiceManager.SidecarNotFoundError, match="No STM sidecar"
        ):
            scb.SpiceManager.get_STMs(
                "MOON", 0.0, "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_interpolates_at_realistic_epochs(self, tmp_path):
        """Between two stored epochs near ET 7.6e8 s the STM is interpolated,
        not snapped to a stored one: a relative tolerance there spans
        thousands of seconds."""
        et0 = 7.6e8
        write_json(
            tmp_path, "moon_stms.json",
            stm_json([et0, et0 + 100.0], {str(et0): EYE6, str(et0 + 100.0): TWO6}),
        )
        result = scb.SpiceManager.get_STMs(
            "MOON", et0 + 10.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
        )
        np.testing.assert_allclose(result, 1.1 * np.eye(6))

    def test_sequence_stm_finds_correct_leg(self, tmp_path):
        """A per-leg sequence file reads each leg's own block, whose STM size
        may differ from the previous leg's; the boundary epoch both legs hold
        reads the end of the earlier leg."""
        write_json(tmp_path, "moon_stms.json", per_leg_stm_json({
            "leg_0": ([0.0, 100.0], [EYE6, TWO6]),
            "leg_1": ([100.0, 200.0], [np.eye(9).tolist(), (3 * np.eye(9)).tolist()]),
        }))

        def stm_at(t):
            return scb.SpiceManager.get_STMs(
                "MOON", t, "J2000", "EARTH", sidecar_path=str(tmp_path)
            )

        np.testing.assert_allclose(stm_at(50.0), 1.5 * np.eye(6))
        np.testing.assert_allclose(stm_at(100.0), 2 * np.eye(6))
        np.testing.assert_allclose(stm_at(150.0), 2 * np.eye(9))

    def test_sequence_stm_epoch_outside_all_legs_raises(self, tmp_path):
        write_json(tmp_path, "moon_stms.json", per_leg_stm_json({
            "leg_0": ([0.0, 100.0], [EYE6, TWO6]),
        }))
        with pytest.raises(ValueError, match="not in any leg"):
            scb.SpiceManager.get_STMs(
                "MOON", 9999.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
            )


class TestGetPropagatorSettings:
    """Comprehensive tests for SpiceManager.get_propagator_settings."""

    MOON_BODY = scb.CelestialBody.from_constants("MOON")
    EARTH_BODY = scb.CelestialBody.from_constants("EARTH")

    @staticmethod
    def _settings_json(extra=None):
        data = {
            "body_ID": MOON_ID,
            "origin_ID": EARTH_ID,
            "reference_frame": "J2000",
            "integrator": "RK4",
            "step_size": 60.0,
        }
        if extra:
            data.update(extra)
        return data

    def test_returns_matching_settings(self, tmp_path):
        write_json(tmp_path, "moon_settings.json", self._settings_json())
        result = scb.SpiceManager.get_propagator_settings(
            "MOON", "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert result["integrator"] == "RK4"
        assert result["step_size"] == pytest.approx(60.0)

    def test_body_frame_enums_accepted(self, tmp_path):
        write_json(tmp_path, "moon_settings.json", self._settings_json())
        result_str = scb.SpiceManager.get_propagator_settings(
            "MOON", "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        result_obj = scb.SpiceManager.get_propagator_settings(
            self.MOON_BODY, J2000, self.EARTH_BODY, folder_path_override=str(tmp_path)
        )
        assert result_str == result_obj

    def test_pkl_sidecar_matches(self, tmp_path):
        write_pickle(tmp_path, "moon_settings.pkl", self._settings_json())
        result = scb.SpiceManager.get_propagator_settings(
            "MOON", "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert result["integrator"] == "RK4"

    def test_ignores_files_with_unrelated_suffix(self, tmp_path):
        """A file not ending in "_settings.json"/"_settings.pkl" must be
        skipped, kept as the only file so the outcome doesn't depend on
        unreliable os.listdir ordering relative to a valid sidecar.
        """
        (tmp_path / "notes.txt").write_text("not a sidecar file")
        with pytest.raises(
            scb.SpiceManager.SidecarNotFoundError,
            match="No propagator-settings sidecar",
        ):
            scb.SpiceManager.get_propagator_settings(
                "MOON", "J2000", "EARTH", folder_path_override=str(tmp_path)
            )

    def test_no_matching_file_raises(self, tmp_path):
        with pytest.raises(
            scb.SpiceManager.SidecarNotFoundError,
            match="No propagator-settings sidecar",
        ):
            scb.SpiceManager.get_propagator_settings(
                "MOON", "J2000", "EARTH", folder_path_override=str(tmp_path)
            )


class TestGenerateMetakernel:

    def test_creates_file(self, tmp_path):
        out_file = str(tmp_path / "out.tm")
        scb.SpiceManager.generate_metakernel(str(tmp_path), out_file)
        assert os.path.isfile(out_file)

    def test_file_has_kpl_header(self, tmp_path):
        out_file = str(tmp_path / "out.tm")
        scb.SpiceManager.generate_metakernel(str(tmp_path), out_file)
        content = open(out_file).read()
        assert "KPL/MK" in content
        assert "KERNELS_TO_LOAD" in content

    def test_lists_files_in_directory(self, tmp_path):
        (tmp_path / "kernel_a.bsp").touch()
        (tmp_path / "kernel_b.tls").touch()
        out_file = str(tmp_path / "out.tm")
        scb.SpiceManager.generate_metakernel(str(tmp_path), out_file)
        content = open(out_file).read()
        assert "kernel_a.bsp" in content
        assert "kernel_b.tls" in content


class TestGetStateAntenna:
    """get_state_antenna with inertial frames — no CK kernel required."""

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_zero_offset_matches_get_state(self):
        """Zero offset with same inertial frame must reproduce get_state exactly."""
        expected = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        result = scb.SpiceManager.get_state_antenna(
            "MOON", self.ET, J2000, "EARTH", np.zeros(3), J2000
        )
        np.testing.assert_allclose(result, expected.values, rtol=1e-12)

    def test_offset_shifts_position_only(self):
        """Antenna offset must shift position but leave velocity unchanged."""
        offset = np.array([100.0, 0.0, 0.0])
        baseline = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        result = scb.SpiceManager.get_state_antenna(
            "MOON", self.ET, J2000, "EARTH", offset, J2000
        )
        np.testing.assert_allclose(result[:3], baseline.values[:3] + offset, rtol=1e-12)
        np.testing.assert_allclose(result[3:], baseline.values[3:], rtol=1e-12)

    def test_result_shape(self):
        result = scb.SpiceManager.get_state_antenna(
            "MOON", self.ET, J2000, "EARTH", np.zeros(3), J2000
        )
        assert result.shape == (6,)


class TestGetAttitude:
    """get_attitude raises when CK/SCLK kernels are absent, and succeeds
    against a synthetic CK segment written via write_ck_type3 (Cassini's
    SCLK, cas00084.tsc, is already in the locked generic kernel set)."""

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_raises_without_sclk_kernel(self):
        """Calling get_attitude without an SCLK kernel must raise a SPICE error."""
        with pytest.raises(Exception):
            scb.SpiceManager.get_attitude(-74, self.ET, "J2000", 1.0)

    @staticmethod
    def _write_synthetic_ck(tmp_path):
        """5 records (not 3 or 4) sidesteps write_ck_type3's shape-detection
        ambiguity for square (N,4)/(N,3) arrays when N==4 or N==3."""
        ck_path = str(tmp_path / "test_attitude.bc")
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        et_list = [et0 + 60.0 * i for i in range(5)]
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        avvs = np.tile([0.001, 0.002, 0.003], (5, 1))
        scb.SpiceManager.write_ck_type3(
            ck_path, inst_id=-82000, ref_frame="J2000", sclk_id=-82,
            et_list=et_list, quats=quats, avvs=avvs, segid="TEST ATTITUDE SEG",
        )
        return ck_path, et0

    def test_returns_state_angvel_angacc(self, tmp_path):
        ck_path, et0 = self._write_synthetic_ck(tmp_path)
        try:
            state, angvel, _ = scb.SpiceManager.get_attitude(-82, et0 + 120.0, "J2000", 1.0)
            np.testing.assert_allclose(state, np.eye(3))
            np.testing.assert_allclose(angvel, [0.001, 0.002, 0.003])
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)

    def test_body_and_frame_objects_accepted(self, tmp_path):
        ck_path, et0 = self._write_synthetic_ck(tmp_path)
        try:
            cassini = scb.Body("CASSINI", spice_id=-82)
            result_obj = scb.SpiceManager.get_attitude(cassini, et0 + 120.0, J2000, 1.0)
            result_raw = scb.SpiceManager.get_attitude(-82, et0 + 120.0, "J2000", 1.0)
            np.testing.assert_allclose(result_obj[0], result_raw[0])
            np.testing.assert_allclose(result_obj[1], result_raw[1])
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)

    def test_non_float_epochs_accepted(self, tmp_path):
        """A numpy scalar and a one-epoch EpochArray give the float result."""
        ck_path, et0 = self._write_synthetic_ck(tmp_path)
        try:
            et = et0 + 120.0
            expected = scb.SpiceManager.get_attitude(-82, et, "J2000", 1.0)
            for epoch in (np.float64(et), scb.EpochArray(et, "TDB")):
                result = scb.SpiceManager.get_attitude(-82, epoch, "J2000", 1.0)
                np.testing.assert_allclose(result[0], expected[0])
                np.testing.assert_allclose(result[1], expected[1])
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)


class TestCKSCLKMethods:
    """Methods that require CK or SCLK kernels not present in the locked
    generic kernel set. Tests are skipped until those kernels are available."""

    def test_ckbrief(self, tmp_path):
        """ckbrief must report the ID and SCLK/TDB coverage of a written CK segment."""
        ck_path = str(tmp_path / "test_ckbrief.bc")
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        et_list = [et0, et0 + 60.0, et0 + 120.0]
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        try:
            scb.SpiceManager.write_ck_type3(
                ck_path, inst_id=-82000, ref_frame="J2000", sclk_id=-82,
                et_list=et_list, quats=quats, segid="TEST CKBRIEF SEG",
            )
            brief = scb.SpiceManager.ckbrief(ck_path, disp=True)

            assert len(brief) == 1
            entry = brief[0]
            assert entry["ID"] == -82000
            assert len(entry["SCLK_INTERVAL"]) == 1
            assert len(entry["TDB_INTERVAL"]) == 1

            sclk_lo, sclk_hi = entry["SCLK_INTERVAL"][0]
            assert sclk_lo < sclk_hi

            tdb_lo, tdb_hi = entry["TDB_INTERVAL"][0]
            assert isinstance(tdb_lo, scb.EpochArray)
            assert isinstance(tdb_hi, scb.EpochArray)
            np.testing.assert_allclose(tdb_lo.times.values, et_list[0], atol=1e-6)
            np.testing.assert_allclose(tdb_hi.times.values, et_list[-1], atol=1e-6)
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)

    def test_list_all_ck_ids(self, tmp_path):
        """list_all_ck_ids must report the NAIF ID(s) present in a written CK."""
        ck_path = str(tmp_path / "test_ck_ids.bc")
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        et_list = [et0, et0 + 60.0, et0 + 120.0]
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        try:
            scb.SpiceManager.write_ck_type3(
                ck_path, inst_id=-82000, ref_frame="J2000", sclk_id=-82,
                et_list=et_list, quats=quats, segid="TEST CK IDS SEG",
            )
            ids = scb.SpiceManager.list_all_ck_ids(ck_path)
            assert list(ids) == [-82000]
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)

    def test_sclk2et(self):
        """sclk2et must (approximately) invert et2sclk for Cassini's SCLK."""
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        sclk_str = scb.SpiceManager.et2sclk(-82, et0)
        et_back = scb.SpiceManager.sclk2et(-82, sclk_str)
        # SCLK string truncates sub-tick precision, so allow a loose tolerance.
        assert et_back == pytest.approx(et0, abs=1e-2)

    def test_sce2c(self):
        """sce2c must return monotonically increasing ticks for increasing ET."""
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        ticks0 = scb.SpiceManager.sce2c(-82, et0)
        ticks1 = scb.SpiceManager.sce2c(-82, et0 + 60.0)
        assert isinstance(ticks0, float)
        assert ticks1 > ticks0

    def test_et2sclk(self):
        """et2sclk must return a non-empty SCLK string for Cassini's clock."""
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        sclk_str = scb.SpiceManager.et2sclk(-82, et0)
        assert isinstance(sclk_str, str)
        assert len(sclk_str) > 0

    def test_write_ck_type3(self, tmp_path):
        """write_ck_type3 must produce a CK readable back via ckbrief."""
        ck_path = str(tmp_path / "test_write_ck.bc")
        et0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")
        et_list = [et0, et0 + 60.0, et0 + 120.0]
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        try:
            result_path = scb.SpiceManager.write_ck_type3(
                ck_path, inst_id=-82000, ref_frame="J2000", sclk_id=-82,
                et_list=et_list, quats=quats, segid="TEST WRITE CK SEG",
            )
            assert result_path == ck_path
            assert os.path.isfile(ck_path)
            brief = scb.SpiceManager.ckbrief(ck_path)
            assert brief[0]["ID"] == -82000
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)


class TestWriteCKType3Errors:
    """write_ck_type3's input-validation branches and its dcm_cb calling
    convention (an alternative to passing quats directly)."""

    ET0 = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_empty_et_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one epoch"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "a.bc"), -82000, "J2000", -82, [],
            )

    def test_non_increasing_et_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="et_list must be strictly increasing"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "b.bc"), -82000, "J2000", -82,
                [self.ET0, self.ET0 - 1.0],
                quats=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            )

    def test_no_quats_and_no_dcm_cb_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Provide either quats or dcm_cb"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "c.bc"), -82000, "J2000", -82,
                [self.ET0, self.ET0 + 60.0],
            )

    def test_dcm_cb_builds_quats(self, tmp_path):
        """When quats is omitted, dcm_cb(et) -> DCM is used to build them."""
        ck_path = str(tmp_path / "d.bc")
        et_list = [self.ET0 + 60.0 * i for i in range(5)]
        try:
            result_path = scb.SpiceManager.write_ck_type3(
                ck_path, -82000, "J2000", -82, et_list,
                dcm_cb=lambda t: np.eye(3),
            )
            assert os.path.isfile(result_path)
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)

    def test_dcm_cb_wrong_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"dcm_cb must return \(3,3\)"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "e.bc"), -82000, "J2000", -82,
                [self.ET0 + 60.0 * i for i in range(5)],
                dcm_cb=lambda t: np.eye(2),
            )

    def test_non_increasing_sclk_raises(self, tmp_path, monkeypatch):
        """A degenerate SCLK kernel/ET combination that yields non-increasing
        encoded ticks despite a strictly increasing et_list.

        Real ET/SCLK pairs at this epoch don't naturally trigger this (the
        gap between "et diff swallowed by float64 precision" and "SCLK tick
        resolution" left no usable delta), so sce2c is monkeypatched to
        return a constant tick regardless of input.
        """
        monkeypatch.setattr(spice, "sce2c", lambda sclk_id, t: 100.0)
        with pytest.raises(ValueError, match="Encoded SCLK times must be strictly increasing"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "f.bc"), -82000, "J2000", -82,
                [self.ET0, self.ET0 + 60.0],
                quats=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            )

    def test_invalid_make_intervals_raises(self, tmp_path):
        with pytest.raises(ValueError, match="make_intervals must be"):
            scb.SpiceManager.write_ck_type3(
                str(tmp_path / "g.bc"), -82000, "J2000", -82,
                [self.ET0, self.ET0 + 60.0],
                quats=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                make_intervals="bogus",
            )

    def test_single_interval_with_multiple_records(self, tmp_path):
        """make_intervals="single" must produce one interval even when
        nrec > 1 (as opposed to the default "per_sample" behavior)."""
        ck_path = str(tmp_path / "h.bc")
        et_list = [self.ET0 + 60.0 * i for i in range(5)]
        try:
            result_path = scb.SpiceManager.write_ck_type3(
                ck_path, -82000, "J2000", -82, et_list,
                quats=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
                make_intervals="single",
            )
            assert os.path.isfile(result_path)
        finally:
            scb.SpiceManager.unload_kernel_from_pool(ck_path)


class TestValidateEpochInput:
    """_validate_epoch_input accepts a single-value EpochArray, a bare
    float/int, and rejects a multi-value EpochArray or any other type."""

    @pytest.mark.parametrize(
        "input_epoch, expected",
        [
            (scb.EpochArray([100.0], "TDB"), "single_epocharray"),
            (scb.EpochArray([100.0, 200.0], "TDB"), ValueError),
            (100.0, "unchanged"),
            (100, "unchanged"),
            ("not valid", TypeError),
        ],
        ids=[
            "Single-Value EpochArray",
            "Multi-Value EpochArray",
            "Float",
            "Int",
            "Invalid Type",
        ],
    )
    def test_validate_epoch_input(self, input_epoch, expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                scb.SpiceManager._validate_epoch_input(input_epoch)
        elif expected == "single_epocharray":
            result = scb.SpiceManager._validate_epoch_input(input_epoch)
            assert np.allclose(result, [100.0])
        else:
            assert scb.SpiceManager._validate_epoch_input(input_epoch) == input_epoch


class TestParameterSidecarSkipped:
    """A state vector with no estimated parameters writes no parameters sidecar.

    A params-free sequence hands the writer one ``None`` per leg. That list is
    truthy, so it used to slip past the ``is not None`` guard and leave a file
    whose ``parameters`` fields were all null — which then matched in
    get_parameters and shadowed runs that estimated something.
    """

    BODY = scb.CelestialBody.from_constants("EARTH")
    EPOCHS = [0.0, 60.0, 120.0]

    @pytest.mark.parametrize("payload", [
        None,
        [None, None],
        [],
        [[], []],
    ], ids=["none", "one-none-per-leg", "empty-list", "empty-per-leg"])
    def test_no_sidecar_written_without_parameters(self, payload, tmp_path):
        spk = str(tmp_path / "no_params.bsp")
        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk, body=self.BODY, origin_ID=EARTH_ID, ref_frame=J2000,
            epochs_TDB=self.EPOCHS, degree_poly=7, parameters_JSON=payload,
            STMs_JSON=None, state_definition=None, sequence_definition=None,
        )
        assert list(tmp_path.glob("*_parameters.*")) == []

    def test_sidecar_written_when_a_parameter_exists(self, tmp_path):
        spk = str(tmp_path / "with_params.bsp")
        scb.SpiceManager.write_spk_metadata_json(
            SPK_name=spk, body=self.BODY, origin_ID=EARTH_ID, ref_frame=J2000,
            epochs_TDB=self.EPOCHS, degree_poly=7,
            parameters_JSON=scb.ArrayWUnits(np.array([[1.0], [1.1], [1.2]]), km),
            STMs_JSON=None,
            state_definition=[
                ("position", 3, None, "dynamic", self.BODY),
                ("velocity", 3, None, "dynamic", self.BODY),
                ("eta_srp", 1, None, "static", self.BODY),
            ],
            sequence_definition=None,
        )
        written = tmp_path / "with_params_parameters.json"
        assert written.is_file()

        contents = json.loads(written.read_text())
        assert contents["parameters_def"] == ["eta_srp"]
        assert contents["parameters"] == [[1.0], [1.1], [1.2]]

    @pytest.mark.parametrize("payload, expected", [
        (None, False),
        ([None, None], False),
        ([], False),
        ([[], []], False),
        ([None, scb.ArrayWUnits(np.array([[1.0]]), km)], True),
        ([[scb.ArrayWUnits(np.array([[1.0]]), km)]], True),
        (scb.ArrayWUnits(np.array([[1.0], [2.0]]), km), True),
    ], ids=["none", "per-leg-none", "empty", "empty-per-leg",
            "one-leg-populated", "nested-leg", "flat-array"])
    def test_has_parameter_payload(self, payload, expected):
        assert scb.SpiceManager._has_parameter_payload(payload) is expected


class TestSegmentStateCount:
    """A type-13 segment needs (degree+1)//2 states, and says so before writing.

    CSPICE reports this as SPICE(TOOFEWSTATES) from deep inside spkw13, naming
    neither the segment nor the fix, and only after the SPK has been opened.
    Each node carries a derivative as well as a position, so a Hermite window is
    half the width the same degree cost under type 9: grids type 9 turned away
    are now written, and only the cases below still fall short.
    """

    BODY = scb.CelestialBody.from_constants("EARTH")

    @staticmethod
    def write(tmp_path, n_states, degree_poly):
        epochs = [60.0 * k for k in range(n_states)]
        states = [[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]] * n_states
        scb.SpiceManager.write_spk_segment_type13(
            SPK_name=str(tmp_path / "segment.bsp"),
            SPK_SEG_ID="TEST SEGMENT",
            body=TestSegmentStateCount.BODY,
            # Not the body itself: CSPICE rejects a segment centred on its
            # own body with SPICE(BODYANDCENTERSAME)
            origin_ID=spice.bodn2c("SUN"),
            ref_frame=J2000,
            epochs_TDB=epochs,
            degree_poly=degree_poly,
            state_SPICE=states,
        )

    @pytest.mark.parametrize("n_states, degree_poly", [
        (1, 3),
        (3, 7),
        (5, 11),
    ], ids=["one-state-cubic", "three-states-degree-7", "five-states-degree-11"])
    def test_too_few_states_is_reported(self, n_states, degree_poly, tmp_path):
        with pytest.raises(
            scb.SpiceManager.TooFewStatesError, match="TEST SEGMENT"
        ):
            self.write(tmp_path, n_states, degree_poly)

        # The check runs before the SPK is opened, so nothing is left behind
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("degree_poly", [2, 4, 8], ids=["two", "four", "eight"])
    def test_even_degree_is_reported(self, degree_poly, tmp_path):
        """Type 13 fits a position and a velocity per node, so the degree it
        produces is always odd; CSPICE rejects an even one outright."""
        with pytest.raises(
            scb.SpiceManager.EvenDegreeError, match="TEST SEGMENT"
        ):
            self.write(tmp_path, n_states=8, degree_poly=degree_poly)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("n_states, degree_poly", [
        (2, 3),
        (3, 3),
        (4, 7),
    ], ids=["two-states-cubic", "three-states-cubic", "four-states-degree-7"])
    def test_enough_states_writes_the_segment(self, n_states, degree_poly, tmp_path):
        # The two cubic cases were TooFewStatesError under type 9, which wanted
        # four nodes for a cubic where Hermite needs two.
        self.write(tmp_path, n_states, degree_poly)

        written = tmp_path / "segment.bsp"
        assert written.is_file()
        scb.SpiceManager.unload_kernel_from_pool(str(written))


class TestSidecarResolution:
    """Sidecars are found by where they are, not by a folder layout.

    The lookups used to scan one folder — kernel_folder, else
    <cwd>/data/kernels/scenario — so a run whose files sit anywhere else, or
    whose SPK is named to someone else's convention, could not be read at all.
    """

    ET = 50.0

    @staticmethod
    def scenario(tmp_path, basename="run7"):
        """An SPK with its three sidecars beside it, as write_to_spk leaves them."""
        (tmp_path / f"{basename}.bsp").write_bytes(b"not a real spk")
        write_json(tmp_path, f"{basename}_parameters.json", flat_params_json())
        write_json(tmp_path, f"{basename}_stms.json", stm_json([0.0, 100.0], {
            "0.0": np.eye(6).tolist(), "100.0": (2 * np.eye(6)).tolist(),
        }))
        write_json(tmp_path, f"{basename}_settings.json",
                   {"integrator": "DOP853", "rtol": 1e-12, "force_models": []})
        return str(tmp_path / f"{basename}.bsp")

    @pytest.mark.parametrize("kind, expected_first", [
        ("parameters", "run7_parameters.json"),
        ("stms", "run7_stms.json"),
        ("settings", "run7_settings.json"),
    ])
    def test_spk_resolves_its_own_sidecars_first(self, kind, expected_first, tmp_path):
        spk = self.scenario(tmp_path)
        files = scb.SpiceManager.sidecar_files(spk, kind=kind)
        assert os.path.basename(files[0]) == expected_first

    def test_folder_prefers_the_matching_suffix(self, tmp_path):
        self.scenario(tmp_path)
        files = scb.SpiceManager.sidecar_files(str(tmp_path), kind="stms")
        assert os.path.basename(files[0]) == "run7_stms.json"
        # the other sidecars stay reachable, just later
        assert {os.path.basename(f) for f in files} == {
            "run7_stms.json", "run7_parameters.json", "run7_settings.json"
        }

    def test_explicit_file_is_used_whatever_it_is_called(self, tmp_path):
        write_json(tmp_path, "whatever-we-called-it.json", flat_params_json())
        odd = str(tmp_path / "whatever-we-called-it.json")

        assert scb.SpiceManager.sidecar_files(odd) == [odd]
        result = scb.SpiceManager.get_parameters(
            "MOON", self.ET, "J2000", "EARTH", sidecar_path=odd
        )
        assert float(result.values.ravel()[0]) == pytest.approx(4.0, abs=1e-10)

    def test_several_locations_are_searched_in_order(self, tmp_path):
        here, there = tmp_path / "here", tmp_path / "there"
        here.mkdir(), there.mkdir()
        write_json(there, "moon_params.json", flat_params_json())

        result = scb.SpiceManager.get_parameters(
            "MOON", self.ET, "J2000", "EARTH", sidecar_path=[str(here), str(there)]
        )
        assert hasattr(result, "values")

    def test_missing_locations_are_skipped_not_raised(self, tmp_path):
        write_json(tmp_path, "moon_params.json", flat_params_json())

        assert scb.SpiceManager.sidecar_files(str(tmp_path / "nope")) == []
        result = scb.SpiceManager.get_parameters(
            "MOON", self.ET, "J2000", "EARTH",
            sidecar_path=[str(tmp_path / "nope"), str(tmp_path)],
        )
        assert hasattr(result, "values")

    def test_error_names_every_location_searched(self, tmp_path):
        with pytest.raises(scb.SpiceManager.SidecarNotFoundError) as excinfo:
            scb.SpiceManager.get_parameters(
                "MOON", self.ET, "J2000", "EARTH",
                sidecar_path=[str(tmp_path / "one"), str(tmp_path / "two")],
            )
        assert "one" in str(excinfo.value) and "two" in str(excinfo.value)

    def test_folder_path_override_still_works(self, tmp_path):
        self.scenario(tmp_path)
        result = scb.SpiceManager.get_parameters(
            "MOON", self.ET, "J2000", "EARTH", folder_path_override=str(tmp_path)
        )
        assert hasattr(result, "values")


class TestSidecarPayloadKind:
    """Each lookup returns its own payload, never a neighbour's.

    The sidecars written beside one SPK share body_ID / origin_ID /
    reference_frame, so identity alone does not tell them apart — and a
    settings payload carries no identity at all.
    """

    def test_settings_lookup_ignores_the_stm_sidecar(self, tmp_path):
        # Settings first alphabetically would still lose to a stale scan order
        write_json(tmp_path, "run_stms.json", stm_json([0.0], {"0.0": np.eye(6).tolist()}))
        write_json(tmp_path, "run_settings.json",
                   {"integrator": "DOP853", "force_models": []})

        settings = scb.SpiceManager.get_propagator_settings(
            "MOON", "J2000", "EARTH", sidecar_path=str(tmp_path)
        )
        assert settings["integrator"] == "DOP853"
        assert "STMs" not in settings

    def test_settings_without_identifiers_are_found_by_location(self, tmp_path):
        """Propagator.to_dict() records no body/origin/frame; the file's place
        beside its SPK is what identifies it."""
        write_json(tmp_path, "run_settings.json",
                   {"integrator": "IAS15", "force_models": []})

        settings = scb.SpiceManager.get_propagator_settings(
            "MOON", "J2000", "EARTH", sidecar_path=str(tmp_path)
        )
        assert settings["integrator"] == "IAS15"

    def test_parameters_lookup_ignores_the_stm_sidecar(self, tmp_path):
        write_json(tmp_path, "run_stms.json", stm_json([0.0, 100.0], {
            "0.0": np.eye(6).tolist(), "100.0": np.eye(6).tolist(),
        }))

        with pytest.raises(scb.SpiceManager.SidecarNotFoundError):
            scb.SpiceManager.get_parameters(
                "MOON", 50.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
            )

    def test_stm_lookup_ignores_the_parameters_sidecar(self, tmp_path):
        write_json(tmp_path, "run_parameters.json", flat_params_json())

        with pytest.raises(scb.SpiceManager.SidecarNotFoundError):
            scb.SpiceManager.get_STMs(
                "MOON", 50.0, "J2000", "EARTH", sidecar_path=str(tmp_path)
            )


class TestSTMLookup:
    """STM retrieval from a sidecar.

    The STM history is the largest artifact an OD run writes — one matrix per
    output epoch — and the filter reads one entry per measurement. A lookup
    that walks the whole epoch grid therefore turns retrieval into O(N*M),
    which is the difference between seconds and a run that appears hung.
    """

    @staticmethod
    def grid_sidecar(tmp_path, n_epochs=500, n_state=6):
        """An STM sidecar on a one-second grid, each matrix tagged by its epoch."""
        epochs = [float(i) for i in range(n_epochs)]
        # scale each matrix by its epoch so a mixed-up lookup is visible
        stms = {str(t): (np.eye(n_state) * (t + 1.0)).tolist() for t in epochs}
        write_json(tmp_path, "run_stms.json", stm_json(epochs, stms))
        scb.SpiceManager.clear_sidecar_cache()
        return str(tmp_path)

    @pytest.mark.parametrize("epoch", [0.0, 1.0, 250.0, 499.0],
                             ids=["first", "second", "middle", "last"])
    def test_exact_epochs_return_the_stored_matrix(self, tmp_path, epoch):
        """An epoch on the grid returns its own matrix, not a neighbour's."""
        location = self.grid_sidecar(tmp_path)

        stm = scb.SpiceManager.get_STMs(
            "MOON", epoch, "J2000", "EARTH", sidecar_path=location
        )

        np.testing.assert_allclose(stm, np.eye(6) * (epoch + 1.0))

    def test_intermediate_epochs_interpolate(self, tmp_path):
        """An epoch between grid points is interpolated linearly."""
        location = self.grid_sidecar(tmp_path)

        stm = scb.SpiceManager.get_STMs(
            "MOON", 10.25, "J2000", "EARTH", sidecar_path=location
        )

        # halfway convention: matrices scale as (t + 1), so does the blend
        np.testing.assert_allclose(stm, np.eye(6) * 11.25, rtol=1e-12)

    @pytest.mark.parametrize("epoch", [-1.0, 500.0], ids=["before", "after"])
    def test_epochs_outside_the_grid_are_reported(self, tmp_path, epoch):
        """An epoch the sidecar does not span is refused rather than clamped."""
        location = self.grid_sidecar(tmp_path)

        with pytest.raises(ValueError, match="out of stored range"):
            scb.SpiceManager.get_STMs(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )

    def test_epoch_grid_is_parsed_once(self, tmp_path):
        """The epoch grid is converted once per sidecar, not once per lookup.

        Rebuilding it makes every lookup O(N); this is the regression guard,
        checked by identity rather than by timing so it cannot go flaky.
        """
        location = self.grid_sidecar(tmp_path)
        for epoch in (5.0, 100.0, 400.0):
            scb.SpiceManager.get_STMs(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )

        sidecar = next(
            payload
            for _, payload in scb.SpiceManager._sidecar_cache.values()
            if isinstance(payload, dict) and "STMs" in payload
        )
        first = scb.SpiceManager._sidecar_epochs_cached(sidecar)
        second = scb.SpiceManager._sidecar_epochs_cached(sidecar)

        assert first is second

    def test_clearing_the_cache_drops_the_epoch_grid(self, tmp_path):
        """The derived grid follows the sidecar cache it was built from."""
        location = self.grid_sidecar(tmp_path)
        scb.SpiceManager.get_STMs("MOON", 5.0, "J2000", "EARTH", sidecar_path=location)
        assert scb.SpiceManager._sidecar_epoch_cache

        scb.SpiceManager.clear_sidecar_cache()

        assert not scb.SpiceManager._sidecar_epoch_cache


class TestParameterLookup:
    """Parameter retrieval from a sidecar.

    A solved-for parameter is stored as a time series and interpolated on
    demand. The interpolation depends only on the file, so refitting it per
    lookup makes every call O(N) — the slowdown appears only once a parameter
    is estimated, which is what makes it confusing in the field.
    """

    @staticmethod
    def series_sidecar(tmp_path, n_epochs=400, slope=3e-4):
        """A sidecar holding one scalar parameter varying linearly in time."""
        epochs = (np.arange(n_epochs, dtype=float) * 3.0)
        values = (1.0 + slope * epochs).reshape(-1, 1)
        write_json(tmp_path, "run_parameters.json", {"legs": {"0": {
            "body_ID": MOON_ID, "origin_ID": EARTH_ID, "reference_frame": "J2000",
            "epochsTDB": epochs.tolist(),
            "parameters": ["eta_srp", {
                "epochsTDB": epochs.tolist(), "values": values.tolist(),
                "unitsPower": [0, 0, 0, 0], "unitsScale": [0.0, 0.0, 0.0, 0.0],
            }],
        }}})
        scb.SpiceManager.clear_sidecar_cache()
        return str(tmp_path), epochs, values

    def test_interpolates_the_stored_series(self, tmp_path):
        """A lookup returns the fitted value, matching a direct fit."""
        location, epochs, values = self.series_sidecar(tmp_path)
        coefficients = np.polyfit(
            epochs, values, int(scb.SpiceManager.poly_interp_par_deg)
        )

        for epoch in (1.0, 300.0, float(epochs[-1]) - 1.0):
            result = scb.SpiceManager.get_parameters(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )
            np.testing.assert_allclose(
                np.asarray(result.values).ravel(),
                np.polyval(coefficients, epoch).ravel(),
                rtol=1e-12,
            )

    def test_the_fit_is_computed_once(self, tmp_path):
        """The interpolation is fitted once per sidecar, not once per lookup.

        Refitting per call is what turns a solved-for parameter into an O(N)
        lookup; identity of the cached coefficients is the durable guard.
        """
        location, _, _ = self.series_sidecar(tmp_path)
        for epoch in (10.0, 200.0, 900.0):
            scb.SpiceManager.get_parameters(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )

        payload = next(
            value
            for _, value in scb.SpiceManager._sidecar_cache.values()
            if isinstance(value, dict) and "legs" in value
        )
        derived = scb.SpiceManager._sidecar_epoch_cache[id(payload)][1]

        fits = [key for key in derived if key[0] == "param_fit"]
        assert len(fits) == 1

        # asking again must not rebuild: the sentinel would replace the fit
        sentinel = object()
        again = scb.SpiceManager._sidecar_derived(payload, fits[0], lambda: sentinel)
        assert again is not sentinel

    @staticmethod
    def flat_sidecar(tmp_path, n_epochs=400):
        """A sidecar in the non-sequence layout, which fits its own way."""
        epochs = np.arange(n_epochs, dtype=float)
        write_json(tmp_path, "run_parameters.json", {
            "body_ID": MOON_ID, "origin_ID": EARTH_ID, "reference_frame": "J2000",
            "epochsTDB": epochs.tolist(),
            "parameters": (1.0 + 3e-4 * epochs).tolist(),
            "unitsPower": [0, 0, 0, 0], "unitsScale": [0.0, 0.0, 0.0, 0.0],
        })
        scb.SpiceManager.clear_sidecar_cache()
        return str(tmp_path), epochs

    def test_flat_sidecar_interpolates(self, tmp_path):
        """The non-sequence layout returns the fitted value too."""
        location, epochs = self.flat_sidecar(tmp_path)
        coefficients = np.polyfit(
            epochs, 1.0 + 3e-4 * epochs, int(scb.SpiceManager.poly_interp_par_deg)
        )

        for epoch in (1.0, 200.0, 398.0):
            result = scb.SpiceManager.get_parameters(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )
            assert float(np.asarray(result.values).ravel()[0]) == pytest.approx(
                float(np.polyval(coefficients, epoch)), rel=1e-12
            )

    def test_flat_sidecar_fit_is_computed_once(self, tmp_path):
        """The non-sequence layout caches its fit as well.

        This branch was missed when the sequence one was first cached, and it
        is the layout a single-leg run actually writes — so it is the one that
        was still refitting per lookup in the field.
        """
        location, _ = self.flat_sidecar(tmp_path)
        for epoch in (10.0, 100.0, 300.0):
            scb.SpiceManager.get_parameters(
                "MOON", epoch, "J2000", "EARTH", sidecar_path=location
            )

        payload = next(
            value
            for _, value in scb.SpiceManager._sidecar_cache.values()
            if isinstance(value, dict) and "parameters" in value
        )
        derived = scb.SpiceManager._sidecar_epoch_cache[id(payload)][1]

        assert [key for key in derived if key[0] == "flat_fit"]

    def test_rewriting_the_sidecar_refits(self, tmp_path):
        """A rewritten file is picked up, cached fit and all."""
        location, epochs, _ = self.series_sidecar(tmp_path)
        scb.SpiceManager.get_parameters(
            "MOON", 300.0, "J2000", "EARTH", sidecar_path=location
        )

        # same shape, different values
        constant = np.full((len(epochs), 1), 5.0)
        write_json(tmp_path, "run_parameters.json", {"legs": {"0": {
            "body_ID": MOON_ID, "origin_ID": EARTH_ID, "reference_frame": "J2000",
            "epochsTDB": epochs.tolist(),
            "parameters": ["eta_srp", {
                "epochsTDB": epochs.tolist(), "values": constant.tolist(),
                "unitsPower": [0, 0, 0, 0], "unitsScale": [0.0, 0.0, 0.0, 0.0],
            }],
        }}})

        result = scb.SpiceManager.get_parameters(
            "MOON", 300.0, "J2000", "EARTH", sidecar_path=location
        )
        assert float(np.asarray(result.values).ravel()[0]) == pytest.approx(5.0)


# STM sidecar payload
def _stm_payload(n_epochs=6, size=4):
    """Prepare an STM payload the way write_to_spk does."""
    stms = np.arange(n_epochs * size * size, dtype=float).reshape(n_epochs, size, size)
    epochs = np.arange(n_epochs, dtype=float) * 10.0
    return (
        scb.SpiceManager._prepare_stm_data(
            STMs=stms,
            epochs_TDB=epochs,
            body_ID=-1234,
            origin_ID=399,
            ref_frame=J2000,
            parameters_JSON=None,
            state_definition=[None, None],
            sequence_definition=None,
            leg_data=None,
            STM_units=None,
        ),
        stms,
        epochs,
    )


class TestStmSidecarPayload:
    """
    The STM sidecar holds one matrix per output epoch, so a long arc on a fine
    grid runs to hundreds of thousands of matrices. The payload keeps them as
    arrays rather than nested Python lists: converting up front creates millions
    of float objects that must all be live while the file is written, which is
    what turned a linear cost into a superlinear one.
    """

    def test_stm_payload_keeps_matrices_as_arrays(self):
        """The prepared payload holds arrays, not nested lists."""
        payload, _, epochs = _stm_payload()

        stored = payload["STMs"][str(epochs[0])]
        assert isinstance(stored, np.ndarray), (
            "STMs must stay as arrays: materialising them as nested lists is what "
            "drives the memory blow-up on a long arc."
        )
        assert stored.shape == (4, 4)

    def test_stm_payload_matrices_survive_the_round_trip(self):
        """Values are unchanged by the array representation."""
        payload, stms, epochs = _stm_payload()

        for k, t in enumerate(epochs):
            np.testing.assert_allclose(
                np.asarray(payload["STMs"][str(t)]), stms[k]
            )

    @pytest.mark.parametrize("fmt, suffix", [("json", ".json"), ("pickle", ".pkl")])
    def test_stm_sidecar_writes_json_and_pickle_alike(self, tmp_path, fmt, suffix):
        """Both formats are written and read back to the same matrices."""
        payload, stms, epochs = _stm_payload()
        spk = str(tmp_path / "arc.bsp")

        scb.SpiceManager._write_stm_sidecar(spk, payload, stm_format=fmt)

        written = tmp_path / f"arc_stms{suffix}"
        assert written.exists()

        if suffix == ".json":
            with open(written) as handle:
                reloaded = json.load(handle)
        else:
            with open(written, "rb") as handle:
                reloaded = pickle.load(handle)

        for k, t in enumerate(epochs):
            np.testing.assert_allclose(np.asarray(reloaded["STMs"][str(t)]), stms[k])

    def test_stm_sidecar_json_is_written_compact(self, tmp_path):
        """The STM file is not pretty-printed, which is 1.6x the bytes for no
        readability at this size."""
        payload, _, _ = _stm_payload(n_epochs=40)
        spk = str(tmp_path / "arc.bsp")

        scb.SpiceManager._write_stm_sidecar(spk, payload, stm_format="json")
        text = (tmp_path / "arc_stms.json").read_text()

        # One float per line is the pretty-printed shape; compact output keeps the
        # matrices on their own line each.
        assert "\n" not in text.strip() or text.count("\n") < 10


# Scalar constant leg parameters
LEG_PARAM_SC_NAME = "LEGPARAM_SC"
LEG_PARAM_SC_ID = -1409


@pytest.fixture
def leg_param_sc():
    """Spacecraft registering the body name get_parameters resolves."""
    return scb.Spacecraft(
        LEG_PARAM_SC_NAME,
        LEG_PARAM_SC_ID,
        scb.ArrayWUnits(100.0, kg),
        scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )


def _sequence_parameter_sidecar(tmp_path, values, epochs_key=None):
    """Write a one-leg sequence parameters sidecar holding one parameter."""
    entry = {
        "frame": "J2000",
        "values": values,
        "unitsPower": [0, 0, 0, 0],
        "unitsScale": [1.0, 1.0, 1.0, 1.0],
    }
    if epochs_key is not None:
        entry["epochsTDB"] = epochs_key
    payload = {
        "legs": {
            "leg_0": {
                "body_ID": LEG_PARAM_SC_ID,
                "origin_ID": 399,
                "reference_frame": "J2000",
                "name": "Leg 1",
                "epochsTDB": [0.0, 100.0, 200.0, 300.0],
                "parameters": ["eta_srp", entry],
            }
        }
    }
    path = tmp_path / "arc_parameters.json"
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


def _read_parameter(tmp_path, epoch):
    """Read the single parameter back through the public getter."""
    return scb.SpiceManager.get_parameters(
        LEG_PARAM_SC_NAME,
        epoch,
        J2000,
        "Earth",
        sidecar_path=str(tmp_path / "arc_parameters.json"),
    )


class TestScalarConstantLegParameters:
    """
    A leg parameter in a sequence sidecar is either constant over the leg or a
    time series with one row per epoch. A size-1 parameter (a solved-for SRP
    scale factor, say) is serialised by ``ndarray.tolist()`` as a bare float,
    which is neither a list nor a nested list. It must still be read as a
    constant.
    """

    def test_scalar_constant_leg_parameter_is_read(self, tmp_path, leg_param_sc):
        """A parameter stored as a bare float comes back as its value."""
        _sequence_parameter_sidecar(tmp_path, 1.0)

        out = _read_parameter(tmp_path, 150.0)

        np.testing.assert_allclose(np.ravel(out.values), [1.0])

    def test_vector_constant_leg_parameter_is_read(self, tmp_path, leg_param_sc):
        """A parameter stored as a flat list still comes back unchanged."""
        _sequence_parameter_sidecar(tmp_path, [1.0, -2.0, 3.0])

        out = _read_parameter(tmp_path, 150.0)

        np.testing.assert_allclose(np.ravel(out.values), [1.0, -2.0, 3.0])

    def test_leg_parameter_time_series_is_interpolated(self, tmp_path, leg_param_sc):
        """A genuine per-epoch series is interpolated, not read as a constant."""
        # Linear ramp, so the value at the midpoint is unambiguous.
        _sequence_parameter_sidecar(
            tmp_path,
            [[0.0], [1.0], [2.0], [3.0]],
            epochs_key=[0.0, 100.0, 200.0, 300.0],
        )

        out = _read_parameter(tmp_path, 150.0)

        np.testing.assert_allclose(np.ravel(out.values), [1.5], atol=1e-9)


# Sequence legs registry
class _StubOwner:
    """Body stand-in: the registry reads a name off each component's owner."""

    def __init__(self, name="ORCCA", spice_id=-1234):
        self._name = name
        self.name = name
        self.spice_id = spice_id


class _StubModel:
    """Sequence model exposing only what the registry builder reads."""

    def __init__(self):
        # Position and velocity only: no estimated parameters to register.
        owner = _StubOwner()
        self.state_definition = [
            ("position", 3, "estimated", "dynamic", owner, None),
            ("velocity", 3, "estimated", "dynamic", owner, None),
        ]


class _StubSequenceDefinition:
    """Minimal stand-in for a propagated MissionSequence."""

    def __init__(self, types):
        self.types = list(types)
        self.names = [f"seg_{i}" for i in range(len(types))]
        self.models = [_StubModel() for _ in types]
        # Built with interval() so `.times` carries units: the registry reads
        # `epoch.times.values`, which a bare-array EpochArray does not provide.
        self.epochs_vec = [
            scb.EpochArray.interval(
                100.0 * i, 100.0 * (i + 1), scb.ArrayWUnits(25.0, sec), sys="TDB"
            )
            for i in range(len(types))
        ]


def _registered_legs(types):
    """Build the sidecar leg registry for a sequence of the given segment types."""
    definition = _StubSequenceDefinition(types)
    active = [i for i, t in enumerate(types) if t in ("Leg", "Finite Burn")]
    leg_data = {
        "by_seq_idx": {
            i: {"body_ID": -1234, "origin_ID": 399, "reference_frame": "J2000"}
            for i in active
        }
    }
    return scb.SpiceManager._prepare_sequence_data(definition, leg_data, None)


class TestSequenceLegsRegistry:
    """
    The STM sidecar carries a per-leg registry so a solution can resolve state
    history segment by segment. A finite burn is a propagated span, not an
    event, and the rest of the library treats it as an active segment, so it
    has to appear in that registry alongside the coast legs. It used to be
    dropped, and a solution then had no state history across the burns.
    """

    def test_finite_burn_legs_are_registered(self):
        """Every active segment of a coast/burn/burn/coast arc appears."""
        types = ["Leg", "Finite Burn", "Finite Burn", "Leg"]

        legs = _registered_legs(types)

        assert len(legs) == 4, (
            "finite burns are propagated spans and must be registered alongside "
            f"the coast legs, got {sorted(legs)}"
        )
        assert sorted(legs) == ["leg_0", "leg_1", "leg_2", "leg_3"]
        assert [legs[k]["name"] for k in sorted(legs)] == [
            "seg_0", "seg_1", "seg_2", "seg_3"
        ]

    @pytest.mark.parametrize("event", ["Impulsive Burn", "Node"])
    def test_events_are_not_registered(self, event):
        """Events have no propagated span, so they stay out of the registry."""
        legs = _registered_legs(["Leg", event, "Leg"])

        assert len(legs) == 2

    def test_leg_indices_account_for_finite_burns(self):
        """A burn occupies a leg slot, so the legs after it are not shifted."""
        # Node between each span, as MissionSequence inserts.
        types = ["Leg", "Node", "Finite Burn", "Node", "Leg"]

        legs = _registered_legs(types)

        # Three active spans, numbered consecutively: the two nodes are the only
        # entries `num_events` counts.
        assert sorted(legs) == ["leg_0", "leg_1", "leg_2"]
        assert [legs[k]["name"] for k in sorted(legs)] == ["seg_0", "seg_2", "seg_4"]

    def test_every_named_parameter_of_a_leg_is_kept(self):
        """A leg's (name, value) pairs are all flattened into the payload, not
        only the first: a maneuver epoch next to its delta-v, or a station's
        range bias, used to be missing from the sidecar."""
        definition = _StubSequenceDefinition(["Leg", "Node", "Leg"])
        leg_data = {
            "by_seq_idx": {
                i: {"body_ID": -1234, "origin_ID": 399, "reference_frame": "J2000"}
                for i in (0, 2)
            }
        }
        dv = scb.ArrayWFrame(np.array([1e-3, -2e-3, 3e-3]), km / sec, J2000)
        dt = scb.ArrayWFrame(np.array([-12.5]), sec, J2000)
        bias = scb.ArrayWFrame(np.array([0.005]), km, J2000)

        legs = scb.SpiceManager._prepare_sequence_data(
            definition, leg_data, [[("dv", dv), ("dt_dv", dt)], [("range_bias", bias)]]
        )

        assert legs["leg_0"]["parameters"][0::2] == ["dv", "dt_dv"]
        assert legs["leg_0"]["parameters"][1::2] == [dv, dt]
        assert legs["leg_1"]["parameters"][0::2] == ["range_bias"]
        assert legs["leg_1"]["parameters"][1::2] == [bias]

# endregion Tests #
