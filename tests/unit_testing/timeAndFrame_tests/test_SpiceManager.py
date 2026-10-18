# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest
import spiceypy as spice
import os
import json

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


def stm_json(epochs, stms_dict):
    """Minimal STM JSON for MOON/J2000/EARTH."""
    return {
        "body_ID": MOON_ID,
        "origin_ID": EARTH_ID,
        "reference_frame": "J2000",
        "epochsTDB": epochs,
        "STMs": stms_dict,
    }

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
        """epoch_time_low=0.0 selects the start of the 1-second interval."""
        st_std = scb.SpiceManager.get_state("MOON", self.ET, "J2000", "EARTH")
        st_prec = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        np.testing.assert_allclose(st_prec.values, st_std.values, rtol=1e-10)

    def test_get_state_precise_nonzero(self):
        state = scb.SpiceManager.get_state_precise(
            "MOON", self.ET, "J2000", "EARTH", epoch_time_low=0.0
        )
        assert np.linalg.norm(state.values[:3]) > 1e3


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
        """The locked_generic metakernel file itself should be in the pool."""
        status = scb.SpiceManager.check_kernel_status_in_pool(furnshKernelFilename)
        assert isinstance(status, bool)

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
            origin_ID=EARTH_ID,
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
    """get_attitude raises when CK/SCLK kernels are absent."""

    ET = scb.SpiceManager.cal2et("2024 JAN 01 00:00:00")

    def test_raises_without_sclk_kernel(self):
        """Calling get_attitude without an SCLK kernel must raise a SPICE error."""
        with pytest.raises(Exception):
            scb.SpiceManager.get_attitude(-74, self.ET, "J2000", 1.0)


class TestCKSCLKMethods:
    """Methods that require CK or SCLK kernels not present in the locked
    generic kernel set. Tests are skipped until those kernels are available."""

    def test_ckbrief(self):
        pytest.skip("Requires a CK file not in the locked generic kernel set.")

    def test_list_all_ck_ids(self):
        pytest.skip("Requires a CK file not in the locked generic kernel set.")

    def test_sclk2et(self):
        pytest.skip("Requires a SCLK kernel not in the locked generic kernel set.")

    def test_sce2c(self):
        pytest.skip("Requires a SCLK kernel not in the locked generic kernel set.")

    def test_et2sclk(self):
        pytest.skip("Requires a SCLK kernel not in the locked generic kernel set.")

    def test_write_ck_type3(self):
        pytest.skip("Requires a SCLK kernel not in the locked generic kernel set.")

# endregion Tests #


# ---------------------------------------------------#
# region  STM sidecar payload                        #
# ---------------------------------------------------#
"""
The STM sidecar holds one matrix per output epoch, so a long arc on a fine grid
runs to hundreds of thousands of matrices. The payload keeps them as arrays
rather than nested Python lists: converting up front creates millions of float
objects that must all be live while the file is written, which is what turned a
linear cost into a superlinear one.

Tests
-----
test_stm_payload_keeps_matrices_as_arrays
    The prepared payload holds arrays, not nested lists.
test_stm_payload_matrices_survive_the_round_trip
    Values are unchanged by the array representation.
test_stm_sidecar_writes_json_and_pickle_alike
    Both formats are written and read back to the same matrices.
test_stm_sidecar_json_is_written_compact
    The STM file is not pretty-printed, which is 1.6x the bytes for no
    readability at this size.
"""

import json as _json
import pickle as _pickle


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


def test_stm_payload_keeps_matrices_as_arrays():
    """The prepared payload holds arrays, not nested lists."""
    payload, _, epochs = _stm_payload()

    stored = payload["STMs"][str(epochs[0])]
    assert isinstance(stored, np.ndarray), (
        "STMs must stay as arrays: materialising them as nested lists is what "
        "drives the memory blow-up on a long arc."
    )
    assert stored.shape == (4, 4)


def test_stm_payload_matrices_survive_the_round_trip():
    """Values are unchanged by the array representation."""
    payload, stms, epochs = _stm_payload()

    for k, t in enumerate(epochs):
        np.testing.assert_allclose(
            np.asarray(payload["STMs"][str(t)]), stms[k]
        )


@pytest.mark.parametrize("fmt, suffix", [("json", ".json"), ("pickle", ".pkl")])
def test_stm_sidecar_writes_json_and_pickle_alike(tmp_path, fmt, suffix):
    """Both formats are written and read back to the same matrices."""
    payload, stms, epochs = _stm_payload()
    spk = str(tmp_path / "arc.bsp")

    scb.SpiceManager._write_stm_sidecar(spk, payload, stm_format=fmt)

    written = tmp_path / f"arc_stms{suffix}"
    assert written.exists()

    if suffix == ".json":
        with open(written) as handle:
            reloaded = _json.load(handle)
    else:
        with open(written, "rb") as handle:
            reloaded = _pickle.load(handle)

    for k, t in enumerate(epochs):
        np.testing.assert_allclose(np.asarray(reloaded["STMs"][str(t)]), stms[k])


def test_stm_sidecar_json_is_written_compact(tmp_path):
    """The STM file is not pretty-printed."""
    payload, _, _ = _stm_payload(n_epochs=40)
    spk = str(tmp_path / "arc.bsp")

    scb.SpiceManager._write_stm_sidecar(spk, payload, stm_format="json")
    text = (tmp_path / "arc_stms.json").read_text()

    # One float per line is the pretty-printed shape; compact output keeps the
    # matrices on their own line each.
    assert "\n" not in text.strip() or text.count("\n") < 10


# ---------------------------------------------------#
# region  Scalar constant leg parameters             #
# ---------------------------------------------------#
"""
A leg parameter in a sequence sidecar is either constant over the leg or a time
series with one row per epoch. A size-1 parameter — a solved-for SRP scale
factor, say — is serialised by ``ndarray.tolist()`` as a bare float, which is
neither a list nor a nested list. It must still be read as a constant.

Tests
-----
test_scalar_constant_leg_parameter_is_read
    A parameter stored as a bare float comes back as its value.
test_vector_constant_leg_parameter_is_read
    A parameter stored as a flat list still comes back unchanged.
test_leg_parameter_time_series_is_interpolated
    A genuine per-epoch series is still interpolated rather than treated as a
    constant.
"""


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
        _json.dump(payload, handle)
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


def test_scalar_constant_leg_parameter_is_read(tmp_path, leg_param_sc):
    """A parameter stored as a bare float comes back as its value."""
    _sequence_parameter_sidecar(tmp_path, 1.0)

    out = _read_parameter(tmp_path, 150.0)

    np.testing.assert_allclose(np.ravel(out.values), [1.0])


def test_vector_constant_leg_parameter_is_read(tmp_path, leg_param_sc):
    """A parameter stored as a flat list still comes back unchanged."""
    _sequence_parameter_sidecar(tmp_path, [1.0, -2.0, 3.0])

    out = _read_parameter(tmp_path, 150.0)

    np.testing.assert_allclose(np.ravel(out.values), [1.0, -2.0, 3.0])


def test_leg_parameter_time_series_is_interpolated(tmp_path, leg_param_sc):
    """A genuine per-epoch series is interpolated, not read as a constant."""
    # Linear ramp, so the value at the midpoint is unambiguous.
    _sequence_parameter_sidecar(
        tmp_path,
        [[0.0], [1.0], [2.0], [3.0]],
        epochs_key=[0.0, 100.0, 200.0, 300.0],
    )

    out = _read_parameter(tmp_path, 150.0)

    np.testing.assert_allclose(np.ravel(out.values), [1.5], atol=1e-9)


# ---------------------------------------------------#
# region  Sequence legs registry                     #
# ---------------------------------------------------#
"""
The STM sidecar carries a per-leg registry so a solution can resolve state
history segment by segment. A finite burn is a propagated span, not an event,
and the rest of the library treats it as an active segment — so it has to
appear in that registry alongside the coast legs. It used to be dropped, and a
solution then had no state history across the burns.

Tests
-----
test_finite_burn_legs_are_registered
    Every active segment of a [Leg, Finite Burn, Finite Burn, Leg] arc appears.
test_events_are_not_registered
    Impulsive burns and nodes are still excluded; they have no span.
test_leg_indices_account_for_finite_burns
    The registry keys follow the sequence slots, so a burn does not shift the
    coast legs that come after it.
"""


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


def test_finite_burn_legs_are_registered():
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
def test_events_are_not_registered(event):
    """Events have no propagated span, so they stay out of the registry."""
    legs = _registered_legs(["Leg", event, "Leg"])

    assert len(legs) == 2


def test_leg_indices_account_for_finite_burns():
    """A burn occupies a leg slot, so the legs after it are not shifted."""
    # Node between each span, as MissionSequence inserts.
    types = ["Leg", "Node", "Finite Burn", "Node", "Leg"]

    legs = _registered_legs(types)

    # Three active spans, numbered consecutively: the two nodes are the only
    # entries `num_events` counts.
    assert sorted(legs) == ["leg_0", "leg_1", "leg_2"]
    assert [legs[k]["name"] for k in sorted(legs)] == ["seg_0", "seg_2", "seg_4"]

