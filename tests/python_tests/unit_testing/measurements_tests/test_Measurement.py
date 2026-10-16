# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ========================================#
#  Unit Tests for the Measurement class  #
# ========================================#
import src.scarabaeus as scb
import spiceypy as spice
import pytest
import numpy as np
import json
import types

km, sec, kg, hr, rad, Hz = scb.Units.get_units(["km", "sec", "kg", "hr", "rad", "Hz"])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def epochs():
    return scb.EpochArray.interval(
        "2028 FEB 08 13:31:08.00",
        "2028 FEB 09 05:31:08.00",
        scb.ArrayWUnits(1, hr),
        sys="TDB",)


@pytest.fixture
def eps_dict():
    """
    Fixture to create an eps_dict containing ArrayWUnits objects
    for testing the _build_perturbation_vector method.
    """

    return {
        "dx": scb.ArrayWUnits(np.array([1.5]), km),
        "dy": scb.ArrayWUnits(np.array([2.0]), km),
        "dz": scb.ArrayWUnits(np.array([0.5]), km),
        "dvx": scb.ArrayWUnits(np.array([0.001]), km * sec**-1),
        "dvy": scb.ArrayWUnits(np.array([0.002]), km * sec**-1),
        "dvz": scb.ArrayWUnits(np.array([0.003]), km * sec**-1),
    }


@pytest.fixture
def GS1():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-55")


@pytest.fixture
def sc():
    Orbiter_area = scb.ArrayWUnits(1e-06, km**2)
    Orbiter_cr_srp = scb.ArrayWUnits(1.5, None)
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp)


@pytest.fixture
def aux_dict():
    """
    Provides a dictionary with two indices:
    Index 0: Normal valid data.
    Index 1: 'Invalid' flags that the function should normalize to 0.
    """
    return {
        "ul_freq": [2.1e9, 2.3e9],
        "rcv_time_tag_delay": [0.05, -1],  # Test -1 -> 0
        "transmit_time_tag_delay": [0.04, -1],  # Test -1 -> 0
        "ul_zheight_corr": [1.1, -99.0],  # Test -99.0 -> 0
        "dl_zheight_corr": [1.2, -99.0],  # Test -99.0 -> 0
        "array_delay": np.array([0.5, -99.0]),  # Test array logic + flag
        "ul_stn_cal": [150, 200],
        "dl_stn_cal": [140, 190],
        "scft_transpd_delay": 1200,  # Non-indexed
        "scft_transpd_delay_s": 0.004,  # Non-indexed
    }


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(GS1):
    """
    Verifies that object is constructed correctly.
    """
    meas_name = "Test Measurement"
    meas = scb.Measurement(name=meas_name, instrument=GS1)

    assert meas._name == meas_name
    assert meas._instrument == GS1


def test_build_perturbation(eps_dict, GS1):
    """
    Verifies that the perturbation is built correctly.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    delta = meas._build_perturbation_vector(eps_dict)

    assert isinstance(delta, scb.ArrayWUnits)
    assert delta == scb.ArrayWUnits(
        np.array([1.5, 2.0, 0.5, 0.001, 0.002, 0.003]),
        [km, km, km, km * sec**-1, km * sec**-1, km * sec**-1],)


def test_generate_electronic_delays_dictionary(GS1, aux_dict):
    """
    Verifies that the electronic delays dictionary is generated correctly.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    delays_dict0 = meas._generate_electronic_delays_dictionary(aux_dict, idx=0)

    expected_keys = [
        "ul_freq",
        "ul_stn_cal",
        "dl_stn_cal",
        "rcv_tt_dly",
        "trn_tt_dly",
        "ul_zheight_corr",
        "dl_zheight_corr",
        "array_delay",
        "scft_transpd_delay",
        "scft_transpd_delay_s",]

    assert all(key in delays_dict0 for key in expected_keys)

    assert delays_dict0["rcv_tt_dly"] == 0.05
    assert delays_dict0["ul_zheight_corr"] == 1.1
    assert delays_dict0["array_delay"] == 0.5
    assert delays_dict0["ul_freq"] == 2.1e9
    assert delays_dict0["scft_transpd_delay"] == 1200

    # --- Test Index 1: Flag Normalization (-1 and -99.0 to 0) ---
    delays_dict1 = meas._generate_electronic_delays_dictionary(aux_dict, idx=1)

    # These should all be 0 based on the function's if-statements
    assert delays_dict1["rcv_tt_dly"] == 0
    assert delays_dict1["trn_tt_dly"] == 0
    assert delays_dict1["ul_zheight_corr"] == 0
    assert delays_dict1["dl_zheight_corr"] == 0
    assert delays_dict1["array_delay"] == 0

    # Frequency should still map correctly
    assert delays_dict1["ul_freq"] == 2.3e9


def test_compute_CN_lt(GS1):
    """
    Verifies the computation of the one-way light time from a
    receiver to a transmitting ground station.
    """
    meas_name = "Test Measurement"
    meas = scb.Measurement(name=meas_name, instrument=GS1)

    # Get Earth's position relative to SSB epoch
    earth_pos = spice.spkpos("EARTH", 886901468.24, "J2000", "NONE", "SSB")[0]

    # Then offset by 7000 km
    receiver_pos_ssb = earth_pos + np.array([7000, 0, 0])
    light_time = meas._compute_CN_lt(
        receiver_pos=receiver_pos_ssb,
        transmitter_id="DSS-55",
        t_rcv=886901468.24,
        delta=np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]),)

    assert 0.002 < light_time < 0.05, "Light time outside physical bounds"


def test_compute_CN_lt_pair_epoch(GS1):
    """
    Verifies that a reception epoch given as (high, low) solves for the instant high + low.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    t_rcv = 886901468.24
    receiver_pos_ssb = spice.spkpos("EARTH", t_rcv, "J2000", "NONE", "SSB")[0] + np.array(
        [7000, 0, 0]
    )
    delta = np.zeros(6)

    # t_rcv + 0.25 is a float, so both calls describe the same instant
    lt_float = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv + 0.25, delta)
    lt_pair = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, delta, t_rcv_l=0.25)
    assert lt_pair == pytest.approx(lt_float, abs=1e-14)


def test_compute_CN_lt_delta_moves_down_leg_transmitter_once(GS1):
    """
    Verifies that on the down leg the perturbation moves the transmitter only, so a 1 km
    shift along the line of sight changes the light time by 1 km / c; on the up leg the
    receiver already carries it and the light time does not move.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    c = 299792.458
    t_rcv = 886901468.24
    receiver_pos_ssb = spice.spkpos("EARTH", t_rcv, "J2000", "NONE", "SSB")[0] + np.array(
        [7000, 0, 0]
    )
    los = spice.spkpos("DSS-55", t_rcv, "J2000", "NONE", "SSB")[0] - receiver_pos_ssb
    delta = np.concatenate([los / np.linalg.norm(los), np.zeros(3)])  # 1 km outward

    lt_dwn = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, np.zeros(6), "down")
    lt_dwn_pert = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, delta, "down")
    assert (lt_dwn_pert - lt_dwn) * c == pytest.approx(1.0, rel=1e-3)

    lt_up = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, np.zeros(6), "up")
    lt_up_pert = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, delta, "up")
    assert lt_up_pert == lt_up


def test_compute_CN_lt_up_leg_station_in_ssb_space_time(GS1):
    """
    Verifies that the up-leg station is taken in the SSB space-time frame (Moyer 2000,
    eq. 4-10): the converged light time closes on that position, which sits centimetres
    from the raw SPICE station used on the down leg.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    c = 299792.458
    t_rcv = 886901468.24
    receiver_pos_ssb = spice.spkpos("EARTH", t_rcv, "J2000", "NONE", "SSB")[0] + np.array(
        [7000, 0, 0]
    )
    lt_up = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, np.zeros(6), "up")
    lt_raw = meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, np.zeros(6), "down")

    t_trn = t_rcv - lt_up
    gs_BC = spice.spkpos("EARTH", t_trn, "J2000", "NONE", "SSB")[0] + (
        meas._compute_GS_in_SSB_space_time_RLT("DSS-55", t_trn)[:3]
    )
    assert np.linalg.norm(gs_BC - receiver_pos_ssb) / c == pytest.approx(lt_up, abs=1e-12)
    assert 1e-5 < abs(lt_up - lt_raw) * c < 1e-3  # 1 cm to 1 m, in km


def test_compute_CN_lt_raises_when_not_converged(GS1, monkeypatch):
    """
    Verifies that a light-time root search reporting no convergence raises
    LightTimeConvergenceError instead of returning its last iterate.
    """
    meas = scb.Measurement(name="Test Measurement", instrument=GS1)
    t_rcv = 886901468.24
    receiver_pos_ssb = spice.spkpos("EARTH", t_rcv, "J2000", "NONE", "SSB")[0] + np.array(
        [7000, 0, 0]
    )

    class _NotConverged:
        converged = False
        root = 0.0

    monkeypatch.setitem(
        scb.Measurement._compute_CN_lt.__globals__,
        "root_scalar",
        lambda *args, **kwargs: _NotConverged(),
    )
    with pytest.raises(scb.Measurement.LightTimeConvergenceError, match="up leg"):
        meas._compute_CN_lt(receiver_pos_ssb, "DSS-55", t_rcv, np.zeros(6))


class TestRelativisticLightTimeCorrection:
    def test_solar_gravity(self, GS1):
        """
        Verifies the relativistic light-time correction due to solar
        gravity (Shapiro delay).
        """
        c = 299792.458  # km/s
        mu_S = 132712440018.0  # km^3/s^2
        gamma = 1.0
        AU = 149597870.7  # km
        r1_r2 = AU  # Sun to body 1
        r2_r3 = AU  # Sun to body 2
        r12_r23 = 2 * AU  # near solar conjunction

        # Compute manually
        sun_term = (1 + gamma) * mu_S / c**2
        numerator = r1_r2 + r2_r3 + r12_r23 + sun_term
        denominator = r1_r2 + r2_r3 - r12_r23 + sun_term
        expected = ((1 + gamma) * mu_S / c**3) * np.log(numerator / denominator)

        meas = scb.Measurement(name="Test", instrument=GS1)
        result = meas._compute_RLT_SUN_lt(r1_r2, r2_r3, r12_r23)

        assert np.isclose(result, expected, rtol=1e-10)
        assert 1e-4 < result < 3e-4, f"Shapiro delay out of physical range: {result}"

    def test_planetary_gravity(self, GS1):
        """
        Verifies the relativistic light-time correction due to planetary
        body gravity along the signal path.
        """
        meas = scb.Measurement(name="Test", instrument=GS1)

        t0 = scb.SpiceManager.cal2et("2029 APR 01 09:04:07.350")
        light_time = 0.02
        t1 = t0
        t2 = t0 + light_time
        t3 = t0 + 2 * light_time

        # SSB-to-GS state
        ssb_to_gs_t1 = spice.spkezr("DSS-55", t1, "J2000", "NONE", "SSB")[0]
        ssb_to_gs_t3 = spice.spkezr("DSS-55", t3, "J2000", "NONE", "SSB")[0]

        delta = np.zeros(6)
        body_id = "10"  # SPICE ID

        # Test "up" leg (t1 -> t2)
        dt_up = meas._compute_RLT_BODIES_lt(
            t2_t1=t1,
            t3_t2=t2,
            body_id=body_id,
            ssb_to_gs_t_31_RLT=ssb_to_gs_t1,
            way="up",
            delta=delta,
        )

        # Test "dwn" leg (t2 -> t3)
        dt_dwn = meas._compute_RLT_BODIES_lt(
            t2_t1=t2,
            t3_t2=t3,
            body_id=body_id,
            ssb_to_gs_t_31_RLT=ssb_to_gs_t3,
            way="dwn",
            delta=delta,
        )

        # Must be positive
        assert dt_up > 0, "GR delay must be positive"
        assert dt_dwn > 0, "GR delay must be positive"

        # verify earth dominates — delay should be order ~nanoseconds to microseconds in LEO
        assert 1e-12 < dt_up < 1e-4, f"dt_up out of physical range: {dt_up}"
        assert 1e-12 < dt_dwn < 1e-4, f"dt_dwn out of physical range: {dt_dwn}"

        # up and dwn should be close to each other (symmetric geometry, small time gap)
        assert np.isclose(dt_up, dt_dwn, rtol=1e-2), "up/dwn delays should be similar"

    def test_body_list_is_configurable(self, GS1):
        """
        Verifies that the body term sums over the bodies it is given: an empty
        list turns it off, and the default list is the sum of its bodies taken
        one at a time.
        """
        t2 = scb.SpiceManager.cal2et("2029 APR 01 09:04:07.350")
        t3 = t2 + 0.02
        ssb_to_gs_t3 = spice.spkezr("DSS-55", t3, "J2000", "NONE", "SSB")[0]

        def dt(bodies):
            return scb.Measurement._compute_RLT_BODIES_lt(
                t2, t3, "10", ssb_to_gs_t3, way="dwn", bodies=bodies
            )

        assert dt(()) == 0
        singles = sum(dt((body,)) for body in scb.constants.RLT_BODIES)
        assert np.isclose(dt(None), singles, rtol=1e-12)
        assert np.isclose(dt(scb.constants.RLT_BODIES), singles, rtol=1e-12)

    def test_body_on_path_raises(self, GS1):
        """
        Verifies that a target listed among the bodies, where the delay
        diverges, raises RLTBodyOnPathError instead of returning inf.
        """
        t2 = scb.SpiceManager.cal2et("2029 APR 01 09:04:07.350")
        t3 = t2 + 2000.0
        ssb_to_gs_t3 = spice.spkezr("DSS-55", t3, "J2000", "NONE", "SSB")[0]
        jupiter = str(scb.constants.JUPITER.barycenter_id)

        with pytest.raises(scb.Measurement.RLTBodyOnPathError, match=f"Body {jupiter}"):
            scb.Measurement._compute_RLT_BODIES_lt(
                t2, t3, jupiter, ssb_to_gs_t3, way="dwn"
            )


class TestPreciseRTLT:
    """
    Checks shared by the full and light precise RTLT drivers, DSS-55 to Phobos.
    """

    t3 = 886901468.24
    c = 299792.458  # km/s
    no_delays = {
        "rcv_tt_dly": 0.0,
        "trn_tt_dly": 0.0,
        "array_delay": 0.0,
        "ul_zheight_corr": 0.0,
        "dl_zheight_corr": 0.0,
        "scft_transpd_delay_s": 0.0,
    }

    def _rtlt(self, driver, delays=None, gs_delta_awf=None, light_time_settings=None):
        return driver(
            sc_spice_id="401",
            gs_spice_id="DSS-55",
            tt_et=self.t3,
            electronic_delays_dict=self.no_delays if delays is None else delays,
            solar_corona_dict={"ul_freq": 0.0, "M2": 0.0},
            delta_partials=scb.ArrayWUnits(np.zeros((1, 6)), km),
            gs_delta_awf=gs_delta_awf,
            light_time_settings=light_time_settings,
        )

    @pytest.mark.parametrize("name", ["_compute_precise_RTLT", "_compute_geocentric_RTLT"])
    def test_light_time_settings_reach_the_solution(self, name, monkeypatch):
        """
        Verifies that the defaults given explicitly reproduce the solution bit for bit,
        that one pass per leg leaves t2 and t1 short of convergence, that a tolerance
        larger than any light-time correction still makes that one pass rather than
        none, and that et_tai_iterations reaches the ET - TAI series.
        """
        driver = getattr(scb.Measurement, name)
        rtlt, parts, t2, t1 = self._rtlt(driver)

        explicit = self._rtlt(driver, light_time_settings=dict(scb.Measurement.LIGHT_TIME_SETTINGS))
        assert (explicit[0], explicit[2], explicit[3]) == (rtlt, t2, t1)
        assert np.array_equal(explicit[1], parts)

        one_pass = self._rtlt(driver, light_time_settings={"max_iterations": 1})
        assert one_pass[2] != t2 and one_pass[3] != t1
        assert abs(one_pass[2] - t2) < 1e-3 and abs(one_pass[3] - t1) < 1e-3

        loose = self._rtlt(driver, light_time_settings={"tolerance": 1e6})
        assert (loose[0], loose[2], loose[3]) == (one_pass[0], one_pass[2], one_pass[3])

        seen = []
        et_from_tai = scb.Measurement._compute_et_from_tai

        def spy(*args, **kwargs):
            seen.append(args[4])
            return et_from_tai(*args, **kwargs)

        monkeypatch.setattr(scb.Measurement, "_compute_et_from_tai", staticmethod(spy))
        self._rtlt(driver, light_time_settings={"et_tai_iterations": 5})
        assert seen == [5]

    @pytest.mark.parametrize("name", ["_compute_precise_RTLT", "_compute_geocentric_RTLT"])
    def test_gs_delta_moves_station_on_both_legs(self, name):
        """
        Verifies that the station offset is applied at t1 and t3: a 1 km shift along the
        ITRF93 local vertical changes the RTLT by minus its projection on the two lines of
        sight over c, and a zero offset is identical to none.
        """
        driver = getattr(scb.Measurement, name)
        rtlt, _, t2, t1 = self._rtlt(driver)

        up = spice.spkpos("DSS-55", self.t3, "ITRF93", "NONE", "EARTH")[0]
        up = up / np.linalg.norm(up)
        delta_awf = scb.ArrayWFrame(up, km, scb.Frame("ITRF93"))
        zero_awf = scb.ArrayWFrame(np.zeros(3), km, scb.Frame("ITRF93"))

        sc_t2 = spice.spkpos("401", t2, "J2000", "NONE", "SSB")[0]
        expected = 0.0
        for t in (t1, self.t3):
            los = sc_t2 - spice.spkpos("DSS-55", t, "J2000", "NONE", "SSB")[0]
            delta_J2000 = spice.pxform("ITRF93", "J2000", t) @ up
            expected -= np.dot(los / np.linalg.norm(los), delta_J2000) / self.c

        rtlt_delta = self._rtlt(driver, gs_delta_awf=delta_awf)[0]
        assert rtlt_delta - rtlt == pytest.approx(expected, rel=1e-3)
        assert self._rtlt(driver, gs_delta_awf=zero_awf)[0] == rtlt

    @pytest.mark.parametrize("name", ["_compute_precise_RTLT", "_compute_geocentric_RTLT"])
    def test_delays_enter_rtlt_but_not_light_time_epochs(self, name):
        """
        Verifies that the transponder and z-height delays add to the RTLT and leave the
        returned t2 and t1 on the light-time solution (Moyer 2000, eq. 11-7).
        """
        driver = getattr(scb.Measurement, name)
        delays = dict(
            self.no_delays,
            scft_transpd_delay_s=1.2e-6,
            ul_zheight_corr=4.0e-8,
            dl_zheight_corr=3.0e-8,
        )
        rtlt, _, t2, t1 = self._rtlt(driver)
        rtlt_d, _, t2_d, t1_d = self._rtlt(driver, delays=delays)

        assert rtlt_d - rtlt == pytest.approx(1.27e-6, abs=1e-12)
        assert (t2_d, t1_d) == (t2, t1)

    @pytest.mark.parametrize("name", ["_compute_precise_RTLT", "_compute_geocentric_RTLT"])
    def test_solar_corona_at_transmit_frequency(self, name):
        """
        Verifies that, given the ramp, the corona delay on both legs is at the transmit
        frequency at t1 rather than at the time-tag ul_freq, so it scales by
        (ul_freq / f_T(t1))^2, and that a flat ramp, or one that starts after t1,
        leaves it at ul_freq. Phobos is near solar conjunction at t3.
        """
        driver = getattr(scb.Measurement, name)
        ul_freq, M2, rate = 7.18e9, 880 / 749, 2e5  # [Hz], [-], [Hz/s]
        years, doys, secs = scb.SpiceManager.et2YDS([self.t3])

        def corona(ramp_start, ramp_rate):
            corona_dict = {"ul_freq": ul_freq, "M2": M2}
            if ramp_start is not None:
                # One record before t3, so that f_T(t3) = ul_freq
                corona_dict["ramp"] = [
                    [secs[0] - ramp_start], [doys[0]],
                    [ul_freq - ramp_rate * ramp_start], [ramp_rate], [1], [years[0]],
                ]
            return driver(
                sc_spice_id="401",
                gs_spice_id="DSS-55",
                tt_et=self.t3,
                electronic_delays_dict=self.no_delays,
                solar_corona_dict=corona_dict,
                delta_partials=scb.ArrayWUnits(np.zeros((1, 6)), km),
                gs_delta_awf=None,
            )

        rtlt_0, parts_0, t2_0, t1_0 = corona(None, 0.0)
        rtlt_r, parts_r, t2_r, t1_r = corona(7200.0, rate)
        assert parts_0[12, 0] > 0 and parts_0[14, 0] > 0

        f_T1 = ul_freq - rate * (rtlt_r - parts_r[12, 0] - parts_r[14, 0])
        for k in (12, 14):
            assert parts_r[k, 0] / parts_0[k, 0] == pytest.approx((ul_freq / f_T1) ** 2, rel=1e-9)
        assert np.array_equal(np.delete(parts_r, [12, 14]), np.delete(parts_0, [12, 14]))
        assert (t2_r, t1_r) == (t2_0, t1_0)

        # A flat ramp, and a ramp that starts after t1, give the time-tag frequency
        assert np.array_equal(corona(7200.0, 0.0)[1], parts_0)
        assert np.array_equal(corona(1.0, rate)[1], parts_0)


class TestGeocentricRTLT:
    """
    The geocentric light-time solution against the barycentric one, for Earth
    orbiters written to a two-body SPK and seen from DSS-14.
    """

    c = 299792.458  # km/s
    et0 = 828273600.0  # 2026 APR 01 TDB: the GEO orbit below is then over DSS-14
    no_delays = TestPreciseRTLT.no_delays

    @pytest.fixture
    def orbiters(self, tmp_path):
        mu = 398600.435436
        path = str(tmp_path / "orbiters.bsp")
        handle = spice.spkopn(path, "orbiters", 0)
        ets = np.arange(self.et0 - 3600.0, self.et0 + 2 * 86400.0, 300.0)
        for spice_id, a in ((-900, 7000.0), (-901, 42164.0)):
            x0 = np.array([a, 0, 0, 0, np.sqrt(mu / a) * np.cos(0.5), np.sqrt(mu / a) * np.sin(0.5)])
            states = np.array([spice.prop2b(mu, x0, t - self.et0) for t in ets])
            spice.spkw05(handle, spice_id, 399, "J2000", ets[0], ets[-1], "x", mu, len(ets), states, ets)
        spice.spkcls(handle)
        spice.furnsh(path)
        yield
        spice.unload(path)

    def _visible_epoch(self, spice_id):
        station = spice.spkpos("DSS-14", self.et0, "ITRF93", "NONE", "EARTH")[0]
        for t3 in self.et0 + np.arange(0.0, 2 * 86400.0 - 3600.0, 300.0):
            azel = spice.azlcpo("ELLIPSOID", spice_id, t3, "NONE", False, 1, station, "EARTH", "ITRF93")[0]
            if azel[2] > np.radians(20.0):
                return t3
        raise AssertionError(f"{spice_id} never rises above 20 deg at DSS-14")

    def _rtlt(self, driver, spice_id, t3):
        return driver(
            sc_spice_id=spice_id,
            gs_spice_id="DSS-14",
            tt_et=t3,
            electronic_delays_dict=self.no_delays,
            solar_corona_dict={"ul_freq": 0.0, "M2": 0.0},
            delta_partials=scb.ArrayWUnits(np.zeros((1, 6)), km),
            gs_delta_awf=None,
        )

    @pytest.mark.parametrize("spice_id", ["-900", "-901"])
    def test_agrees_with_barycentric_solution(self, orbiters, spice_id):
        """
        Verifies that the two frames give the same two-way range to 1 mm in LEO and at
        GEO. The barycentric solution carries the Sun's delay, about 2e-8 of the range
        (0.8 m at GEO), which the geocentric one must absorb through the inverse
        Moyer eq. 4-10 transform of the spacecraft.
        """
        t3 = self._visible_epoch(spice_id)
        rtlt_b, parts_b, t2_b, t1_b = self._rtlt(scb.Measurement._compute_precise_RTLT, spice_id, t3)
        rtlt_g, parts_g, t2_g, t1_g = self._rtlt(scb.Measurement._compute_geocentric_RTLT, spice_id, t3)

        assert abs(rtlt_g - rtlt_b) * self.c / 2 < 1e-6
        assert parts_g[4, 0] + parts_g[5, 0] == 0.0
        assert t2_g == pytest.approx(t2_b, abs=1e-6)
        assert t1_g == pytest.approx(t1_b, abs=1e-6)


@pytest.mark.parametrize("frame", ["barycentric", "Geocentric "])
def test_resolve_light_time_frame_accepts_known_frames(frame):
    assert scb.Measurement._resolve_light_time_frame(frame) == frame.strip().lower()


def test_resolve_light_time_frame_rejects_unknown_frame():
    with pytest.raises(scb.Measurement.LightTimeFrameError, match="heliocentric"):
        scb.Measurement._resolve_light_time_frame("heliocentric")


def test_resolve_light_time_settings_fills_in_the_defaults():
    defaults = scb.Measurement.LIGHT_TIME_SETTINGS
    resolved = scb.Measurement._resolve_light_time_settings(None)
    assert resolved == defaults and resolved is not defaults

    resolved = scb.Measurement._resolve_light_time_settings({"tolerance": 1e-9})
    assert resolved == dict(defaults, tolerance=1e-9)
    assert defaults["tolerance"] == 1e-12


@pytest.mark.parametrize(
    "settings",
    [
        {"tol": 1e-9},
        {"tolerance": 0.0},
        {"tolerance": -1e-9},
        {"tolerance": np.inf},
        {"tolerance": np.nan},
        {"tolerance": True},
        {"tolerance": "1e-9"},
        {"max_iterations": 0},
        {"max_iterations": 1.5},
        {"max_iterations": True},
        {"et_tai_iterations": 0},
    ],
)
def test_resolve_light_time_settings_rejects_bad_settings(settings):
    with pytest.raises(scb.Measurement.LightTimeSettingsError, match=next(iter(settings))):
        scb.Measurement._resolve_light_time_settings(settings)





def test_measurement_dataset_val(sc, epochs, GS1):
    """Verify that the function prevents invalid input combinations."""
    meas_model = scb.Measurement(name="test_mesurement", instrument=GS1)

    # Test 1: Provide nothing
    with pytest.raises(ValueError, match="Provide exactly one of"):
        meas_model.generate_measurement_dataset("test", target=sc)

    # Test 2: Provide both
    obs_mock = (epochs, np.zeros(epochs.size), None, np.zeros(epochs.size))
    with pytest.raises(ValueError, match="Provide only one of"):
        meas_model.generate_measurement_dataset(
            "test", target=sc, epochs=epochs, observed_meas=obs_mock)


class TestRTLTGradients:
    """
    The exact round-trip light-time gradients (Moyer 12-12, 12-17, 12-23) against
    central differences of a flat-space round trip with straight-line motion.
    The velocities are a hundredth of c so the transmitter denominators are
    visible above the difference noise.
    """

    c = 299792.458  # km/s
    r_sc0 = np.array([1.2e8, -4.0e7, 2.0e7])  # km
    v_sc = np.array([-1500.0, 2400.0, 800.0])  # km/s
    r_gs0 = np.array([6000.0, 1000.0, -2000.0])  # km
    v_gs = np.array([900.0, -2100.0, 1300.0])  # km/s
    t3 = 1000.0  # s

    def _round_trip(self, d_dn=0.0, d_up=0.0, dr_sc=np.zeros(3), dr_gs=np.zeros(3)):
        """ρ = t3 − t1, with delays d_dn, d_up [s] inside each leg's light time."""
        r_sc = lambda t: self.r_sc0 + dr_sc + self.v_sc * t
        r_gs = lambda t: self.r_gs0 + dr_gs + self.v_gs * t
        t2 = self.t3
        for _ in range(60):
            t2 = self.t3 - np.linalg.norm(r_sc(t2) - r_gs(self.t3)) / self.c - d_dn
        t1 = t2
        for _ in range(60):
            t1 = t2 - np.linalg.norm(r_sc(t2) - r_gs(t1)) / self.c - d_up
        n_dn = (r_sc(t2) - r_gs(self.t3)) / np.linalg.norm(r_sc(t2) - r_gs(self.t3))
        n_up = (r_sc(t2) - r_gs(t1)) / np.linalg.norm(r_sc(t2) - r_gs(t1))
        return self.t3 - t1, n_dn, n_up

    def _central(self, h, **shift):
        (key, unit), = shift.items()
        return (self._round_trip(**{key: h * unit})[0]
                - self._round_trip(**{key: -h * unit})[0]) / (2.0 * h)

    def test_leg_weights_are_delay_partials(self):
        _, n_dn, n_up = self._round_trip()
        w_dn, w_up = scb.Measurement.rtlt_leg_weights(n_dn, n_up, self.v_sc, self.v_gs, self.c)

        assert abs(w_dn - 1.0) > 1e-3 and abs(w_up - 1.0) > 1e-3
        assert w_dn == pytest.approx(self._central(1e-3, d_dn=1.0), rel=1e-9)
        assert w_up == pytest.approx(self._central(1e-3, d_up=1.0), rel=1e-9)

    def test_leg_weights_are_one_without_motion(self):
        _, n_dn, n_up = self._round_trip()
        w = scb.Measurement.rtlt_leg_weights(n_dn, n_up, np.zeros(3), np.zeros(3), self.c)
        assert w == (1.0, 1.0)

    def test_station_and_spacecraft_gradients(self):
        _, n_dn, n_up = self._round_trip()
        g_sc = scb.Measurement.rtlt_grad_pos_exact(n_dn, n_up, self.v_sc, self.v_gs, self.c)
        g3, g1 = scb.Measurement.rtlt_grad_station_exact(n_dn, n_up, self.v_sc, self.v_gs, self.c)

        # Moving both ends together leaves the round trip unchanged.
        np.testing.assert_allclose(g3 + g1, -g_sc, rtol=1e-12)
        for k, e in enumerate(np.eye(3)):
            assert g_sc[k] == pytest.approx(self._central(100.0, dr_sc=e), rel=1e-7)
            assert g3[k] + g1[k] == pytest.approx(self._central(100.0, dr_gs=e), rel=1e-7)


# ---------------------------------------------------#
# region  Empty visibility mask                      #
# ---------------------------------------------------#
"""
``check_visibility=True`` drops the epochs an instrument cannot observe. A
station that sees nothing over the requested span is an ordinary outcome — a
script generating data for several instruments should skip that one and carry
on — so it warns and writes no file rather than raising and losing the run.
No file is written because an empty measurement set cannot be built: the units
are read off the measurements themselves.

Tests
-----
test_no_visible_epochs_warns_instead_of_raising
    The call returns, warns, and leaves no file behind.
test_no_visible_epochs_names_the_instrument_and_the_mask
    The warning says which model gave up and what mask was applied.
"""

import warnings


def _never_visible_model(GS1):
    """A range model whose station cannot see the target at the test epochs."""
    return scb.RangeIdeal(
        "unreachable range", GS1, sigma=scb.ArrayWUnits(1e-3, km)
    )


def _force_empty_mask(monkeypatch, model):
    """Make the visibility check reject every epoch, whatever the geometry."""
    monkeypatch.setattr(
        type(model),
        "_compute_visibility_mask",
        lambda self, target, epoch_array, elevation_mask=10.0: np.zeros(
            np.atleast_1d(epoch_array.et).size, dtype=bool
        ),
    )


def test_no_visible_epochs_warns_instead_of_raising(
    monkeypatch, tmp_path, GS1, sc, epochs
):
    """The call returns, warns, and leaves no file behind."""
    model = _never_visible_model(GS1)
    _force_empty_mask(monkeypatch, model)

    with pytest.warns(RuntimeWarning, match="never visible"):
        result = model.write_observed_measurements(
            target=sc,
            epoch_array=epochs,
            noisy=False,
            file_name="nothing_visible",
            check_visibility=True,
            folder_path_override=str(tmp_path),
        )

    assert result is None
    assert not (tmp_path / "nothing_visible.json").exists(), (
        "an empty measurement set cannot be assembled, so no file is written"
    )


def test_no_visible_epochs_names_the_instrument_and_the_mask(
    monkeypatch, tmp_path, GS1, sc, epochs
):
    """The warning says which model gave up and what mask was applied."""
    model = _never_visible_model(GS1)
    _force_empty_mask(monkeypatch, model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.write_observed_measurements(
            target=sc,
            epoch_array=epochs,
            noisy=False,
            file_name="nothing_visible",
            check_visibility=True,
            elevation_mask=25.0,
            folder_path_override=str(tmp_path),
        )

    message = str(
        next(w.message for w in caught if issubclass(w.category, RuntimeWarning))
    )
    assert "unreachable range" in message
    assert "nothing_visible" in message
    assert "25.0" in message


@pytest.mark.parametrize(
    "mask, expected",
    [
        (10.0, 10.0),
        (0.0, 0.0),
        (1.0, 1.0),  # whole degrees are not ambiguous
        (2.5, 2.5),  # nor is anything above pi/2
        (scb.ArrayWUnits(0.5, scb.Units.get_units("deg")), 0.5),
        (scb.ArrayWUnits(0.2, rad), np.degrees(0.2)),
    ],
)
def test_resolve_elevation_mask(mask, expected):
    assert scb.Measurement.resolve_elevation_mask(mask) == pytest.approx(expected)


@pytest.mark.parametrize(
    "mask, match",
    [
        (0.5, "could be 28.65 deg given in radians"),
        (0.17, "radians"),
        (np.pi / 2, "radians"),
        (100.0, "outside 0 to 90"),
        (-5.0, "outside 0 to 90"),
        (scb.ArrayWUnits(2.0, rad), "outside 0 to 90"),
        (scb.ArrayWUnits(10.0, km), "not a single angle"),
        ("ten", "neither a number"),
        (None, "neither a number"),
    ],
)
def test_resolve_elevation_mask_rejects(mask, match):
    with pytest.raises(scb.Measurement.ElevationMaskError, match=match):
        scb.Measurement.resolve_elevation_mask(mask)


def test_sub_degree_mask_reaches_the_station(monkeypatch, tmp_path, GS1, sc, epochs):
    """Resolved for the warning and again for the station, a bare 0.5 would be refused."""
    seen = []

    def station_visibility(self, target, epoch_array, elevation_mask=10.0, **kwargs):
        seen.append(elevation_mask)
        return np.zeros(np.atleast_1d(epoch_array.et).size, dtype=bool)

    monkeypatch.setattr(type(GS1), "station_visibility", station_visibility)
    model = _never_visible_model(GS1)
    with pytest.warns(RuntimeWarning, match="currently 0.5 deg"):
        model.write_observed_measurements(
            target=sc,
            epoch_array=epochs,
            noisy=False,
            file_name="nothing_visible",
            check_visibility=True,
            elevation_mask=scb.ArrayWUnits(0.5, scb.Units.get_units("deg")),
            folder_path_override=str(tmp_path),
        )
    assert seen == [0.5]


def _stub_forward_model(monkeypatch, model, n):
    """
    Return canned measurements for *n* epochs.

    The GS1/sc fixtures have no SPK coverage at the test epochs, so the real
    forward model cannot run here. These two tests are about what
    write_observed_measurements does with the result -- assembling the
    measurement block and checking its shape -- not about the geometry.
    """
    meas = types.SimpleNamespace(
        quantity=types.SimpleNamespace(values=np.arange(n, dtype=float))
    )
    monkeypatch.setattr(
        type(model),
        "computed_measurements",
        lambda self, *args, **kwargs: (None, None, None, meas),
    )


def test_ragged_measurement_block_is_refused(monkeypatch, tmp_path, GS1, sc, epochs):
    """
    A column whose length disagrees with the number of simulated epochs is a
    refusal, not a file.

    Every column in the measurement block is one value per observation. A
    model's _write_aux_data copies its columns out of computed_measurements_dict,
    filled either by _read_aux (from whatever pass was read) or by a
    default_*_aux helper on its own grid -- neither guaranteed to match the
    epochs being simulated. observed_measurements rebuilds its epochs from the
    year/doy/sec columns, so a mismatch used to be written out silently and came
    back with the measurements paired to the wrong times.
    """
    model = scb.RangeIdeal("ragged range", GS1, sigma=scb.ArrayWUnits(1e-3, km))
    _stub_forward_model(monkeypatch, model, np.atleast_1d(epochs.et).size)
    monkeypatch.setattr(
        type(model),
        "_write_aux_data",
        lambda self, dict_DB, dict_meas: dict_meas.update(bogus=[0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="ragged"):
        model.write_observed_measurements(
            target=sc,
            epoch_array=epochs,
            noisy=False,
            file_name="ragged",
            folder_path_override=str(tmp_path),
        )

    assert not (tmp_path / "ragged.json").exists(), (
        "a ragged block must not reach disk"
    )


def test_time_tags_round_trip_through_the_file(monkeypatch, tmp_path, GS1, sc, epochs):
    """
    The tags written are the epochs simulated, recoverable exactly.

    write_observed_measurements converts the epochs to station time (UTC) with
    SpiceManager.et2YDS (spice.et2utc); observed_measurements converts them back
    with SpiceManager.YDS2et (spice.str2et). Same LSK, same DELTET model, so the
    round trip is exact -- which is what lets a simulated file be reloaded as if
    it were real data and recompute against the same spacecraft state.
    """
    model = scb.RangeIdeal("round trip", GS1, sigma=scb.ArrayWUnits(1e-3, km))
    n = np.atleast_1d(epochs.et).size
    _stub_forward_model(monkeypatch, model, n)
    model.write_observed_measurements(
        target=sc,
        epoch_array=epochs,
        noisy=False,
        file_name="round_trip",
        folder_path_override=str(tmp_path),
    )
    with open(tmp_path / "round_trip.json") as handle:
        written = json.load(handle)["meas_sim"]
    back = np.asarray(
        scb.SpiceManager.YDS2et(written["year"], written["doy"], written["sec"])
    )
    assert back.size == n
    np.testing.assert_allclose(back, np.atleast_1d(epochs.et), rtol=0, atol=1e-9)


def test_unset_sigma_reports_what_is_missing(GS1):
    """
    A model with no sigma says so, instead of AttributeError on NoneType.

    sigma is optional on every model constructor and in MeasurementConfig, whose
    docstring used to say real data usually left it unset. Dataset generation
    read self.sigma.values directly, so build() followed by compute() raised
    "'NoneType' object has no attribute 'values'" from inside
    generate_measurement_dataset, naming neither the model nor the dataset.
    """
    model = scb.RangeIdeal("no sigma", GS1)
    assert model.sigma is None

    with pytest.raises(ValueError, match="has no measurement sigma") as excinfo:
        model._require_sigma("DSN Range")

    message = str(excinfo.value)
    assert "no sigma" in message, "the model is named"
    assert "DSN Range" in message, "the dataset is named"
    assert "MeasurementConfig" in message, "the fix is named"


def test_bare_numeric_sigma_is_accepted(GS1):
    """
    A plain number works, so a refit that could not resolve units is survivable.

    MeasurementSpec.edit assigns ArrayWUnits(value, units), but falls back to the
    bare value when it has no units to use -- which is what happened whenever the
    model had no prior sigma to take them from. That float then hit
    self.sigma.values and raised. edit() now resolves units from the dataset, and
    this is the backstop.
    """
    model = scb.DopplerReal("float sigma", GS1)
    model.sigma = 0.0123
    assert model._require_sigma("any") == pytest.approx(0.0123)
    np.testing.assert_allclose(
        np.array([model._require_sigma("any") for _ in range(3)]), 0.0123
    )


def test_sigma_is_expressed_in_the_measurement_units(GS1):
    """
    The sigma column is in the computed measurement's units, so a sigma set in
    other units of the same dimension is converted rather than copied: a 5 m
    sigma on a range computed in km used to weight it as 5 km.
    """
    m = scb.Units.get_units("m")
    model = scb.RangeIdeal("m sigma", GS1, sigma=scb.ArrayWUnits(5.0, m))
    assert model._require_sigma("r", units=km) == pytest.approx(0.005)
    assert model._require_sigma("r") == pytest.approx(5.0), "no units: as stands"

    model.sigma = 5.0
    assert model._require_sigma("r", units=km) == pytest.approx(5.0)

    model.sigma = scb.ArrayWUnits(5.0, km / sec)
    with pytest.raises(scb.Measurement.SigmaUnitsError, match="'r'"):
        model._require_sigma("r", units=km)


def test_multi_component_sigma_stays_per_component(GS1):
    """A two-component sigma still yields one row per epoch, two columns."""
    model = scb.AngularIdeal(
        "angles", observer=GS1, sigma=scb.ArrayWUnits([10e-6, 20e-6], rad)
    )
    sigmas = np.array([model._require_sigma("angles") for _ in range(4)])
    assert sigmas.shape == (4, 2)
    np.testing.assert_allclose(sigmas[:, 1], 20e-6)


MODELS_WITH_SIGMA = [
    "RangeIdeal", "RangeRateIdeal", "DopplerIdeal", "AngularIdeal",
    "CentroidingIdeal", "DopplerReal", "SequentialRangingReal",
    "CustomMeasurement", "DiffOneWayRangeIdeal",
]


@pytest.mark.parametrize("model_name", MODELS_WITH_SIGMA)
def test_every_model_exposes_a_sigma_setter(model_name):
    """
    sigma is settable on every model.

    MeasurementSpec.flag assigns to model.sigma to apply a refitted value; every
    model inherits the base-class property.
    """
    model_class = getattr(scb, model_name)
    prop = getattr(model_class, "sigma")
    assert prop.fset is not None, f"{model_name}.sigma has no setter"

    # __new__ bypasses the constructors, which differ widely; the property is
    # the whole subject here.
    model = model_class.__new__(model_class)
    value = scb.ArrayWUnits(0.0123, km)
    model.sigma = value
    assert model.sigma == value


IDEAL_MODELS = [
    "RangeIdeal", "RangeRateIdeal", "DopplerIdeal", "AngularIdeal",
    "CentroidingIdeal", "CustomMeasurement", "DiffOneWayRangeIdeal",
]


@pytest.mark.parametrize("model_name", IDEAL_MODELS)
def test_missing_epochs_raise_value_error(model_name):
    """
    With neither epoch_array nor a start/end pair, the ideal models say so.

    They share Measurement._resolve_epoch_array with the real models, which
    raises ValueError before the model touches any of its own state.
    """
    model_class = getattr(scb, model_name)
    # __new__ bypasses the constructors, which differ widely; the check runs
    # before computed_measurements reads anything set by them.
    model = model_class.__new__(model_class)
    with pytest.raises(ValueError, match="No epochs to evaluate"):
        model.computed_measurements(None)


# The state key and units each ideal model passes to Measurement._extract_bias
IDEAL_BIAS_EXTRACTORS = [
    ("RangeIdeal", "range_bias", 0.5, km),
    ("RangeRateIdeal", "rangerate_bias", 0.5, km / sec),
    ("DopplerIdeal", "rangerate_bias_Hz", 0.5, Hz),
    ("DiffOneWayRangeIdeal", "dor_bias", 0.5, km),
    ("AngularIdeal", "angular_bias", [1e-5, -2e-5], rad),
]


@pytest.mark.parametrize("model_name, state_key, bias, units", IDEAL_BIAS_EXTRACTORS)
def test_numeric_meas_bias_is_taken_in_observable_units(model_name, state_key, bias, units):
    """
    A numeric meas_bias, as documented and as MeasurementSpec.build() passes it,
    comes back as an ArrayWUnits in the observable's units.
    """
    model_class = getattr(scb, model_name)
    model = model_class.__new__(model_class)
    model._reference_state_vector = None
    model._meas_bias = bias

    out = model._extract_bias(state_key, units)

    assert isinstance(out, scb.ArrayWUnits)
    assert out.units.name == units.name
    np.testing.assert_array_equal(out.values, np.asarray(bias, dtype=float))


@pytest.mark.parametrize("model_name, state_key, bias, units", IDEAL_BIAS_EXTRACTORS)
def test_meas_bias_with_units_keeps_them(model_name, state_key, bias, units):
    """
    An ArrayWFrame meas_bias returns its quantity; an ArrayWUnits one is
    returned as given.
    """
    model_class = getattr(scb, model_name)
    model = model_class.__new__(model_class)
    model._reference_state_vector = None
    values = np.atleast_1d(np.asarray(bias, dtype=float))

    model._meas_bias = scb.ArrayWFrame(values, rad, scb.Frame("J2000"))
    out = model._extract_bias(state_key, units)
    assert isinstance(out, scb.ArrayWUnits)
    assert out.units.name == "rad"
    np.testing.assert_array_equal(out.values, values)

    given = scb.ArrayWUnits(values, kg)
    model._meas_bias = given
    assert model._extract_bias(state_key, units) is given

    model._meas_bias = None
    assert model._extract_bias(state_key, units) is None


# The state component that overrides meas_bias
IDEAL_BIAS_STATE_COMPONENTS = {
    "RangeIdeal": "range_bias_GS",
    "RangeRateIdeal": "rangerate_bias_GS",
    "DopplerIdeal": "rangerate_bias_Hz_GS",
    "DiffOneWayRangeIdeal": "dor_bias_GS",
    "AngularIdeal": "angular_bias_GS",
}


@pytest.mark.parametrize("model_name, state_key, bias, units", IDEAL_BIAS_EXTRACTORS)
def test_meas_bias_holds_until_the_state_carries_this_stations_bias(
    model_name, state_key, bias, units
):
    """
    With a reference state attached, meas_bias still applies unless the state
    holds this station's bias, which overrides it.
    """
    model_class = getattr(scb, model_name)
    model = model_class.__new__(model_class)
    # Bodies are matched by SPICE ID only. Plain stand-ins keep the names out
    # of the kernel pool, where earlier tests may have bound them elsewhere.
    station = types.SimpleNamespace(name="Station", spice_id=399101)
    other = types.SimpleNamespace(name="Other", spice_id=399102)
    model._instrument = station
    model._meas_bias = bias
    model._reference_state_vector = None
    fallback = model._extract_bias(state_key, units)
    values = np.atleast_1d(np.asarray(bias, dtype=float))

    component = IDEAL_BIAS_STATE_COMPONENTS[model_name]
    state = [(
        component, values.size, "estimated", "parameter", other,
        scb.ArrayWFrame(2 * values, fallback.units, scb.Frame("J2000")),
    )]
    model._reference_state_vector = types.SimpleNamespace(state=state)

    out = model._extract_bias(state_key, units)
    assert isinstance(out, scb.ArrayWUnits)
    assert out.units.name == units.name
    np.testing.assert_array_equal(out.values, values)

    state.append((
        component, values.size, "estimated", "parameter", station,
        scb.ArrayWFrame(3 * values, fallback.units, scb.Frame("J2000")),
    ))
    out = model._extract_bias(state_key, units)
    assert isinstance(out, scb.ArrayWUnits)
    np.testing.assert_array_equal(out.values, 3 * values)


# ---------------------------------------------------#
# region  Measurement block name resolution          #
# ---------------------------------------------------#
# Real files name their blocks by measurement type, simulated files always use
# "meas_sim". Reusing the real name on a simulated file is the easy mistake, so
# the failure has to say what the file actually holds.
SIMULATED_FILE = {
    "name": "sim_sr",
    "meas_sim": {"outlier_flag": [0, 0, 0]},
    "aux_SFDU_9": {"sec": [1.0]},
}

REAL_FILE = {
    "2W_sranging": {"outlier_flag": [0, 0]},
    "2W_doppler": {"outlier_flag": [0, 0]},
    "aux_SFDU_9": {"sec": [1.0]},
}


class TestPresentKey:
    def test_requested_name_is_returned_when_present(self):
        assert scb.Measurement._resolve_meas_key(REAL_FILE, "2W_sranging") == "2W_sranging"

    def test_simulated_default_resolves(self):
        assert scb.Measurement._resolve_meas_key(SIMULATED_FILE, "meas_sim") == "meas_sim"


class TestLegacyAlias:
    """"meas_ideal" was renamed to "meas_sim"; old files stay readable."""

    def test_modern_name_falls_back_to_legacy_key(self):
        legacy = {"meas_ideal": {"outlier_flag": [0]}}
        with pytest.warns(DeprecationWarning, match="meas_ideal"):
            assert scb.Measurement._resolve_meas_key(legacy, "meas_sim") == "meas_ideal"

    def test_legacy_name_falls_back_to_modern_key(self):
        with pytest.warns(DeprecationWarning):
            assert scb.Measurement._resolve_meas_key(SIMULATED_FILE, "meas_ideal") == "meas_sim"


class TestMissingKey:
    def test_real_name_on_a_simulated_file_names_the_simulated_block(self):
        """The mistake this error exists for."""
        with pytest.raises(KeyError) as excinfo:
            scb.Measurement._resolve_meas_key(SIMULATED_FILE, "2W_sranging", "sim_sr.json")
        message = excinfo.value.args[0]
        assert "2W_sranging" in message, "must name what was asked for"
        assert "meas_sim" in message, "must name the block that is there"
        assert "simulated" in message.lower()
        assert "sim_sr.json" in message, "must name the file looked in"

    def test_absent_name_on_a_real_file_lists_the_blocks(self):
        with pytest.raises(KeyError) as excinfo:
            scb.Measurement._resolve_meas_key(REAL_FILE, "3W_doppler", "pass.json")
        message = excinfo.value.args[0]
        assert "3W_doppler" in message
        assert "2W_sranging" in message and "2W_doppler" in message
        # Not a simulated file, so it should not claim to be one.
        assert "This is a simulated file" not in message

    def test_file_name_is_optional(self):
        with pytest.raises(KeyError):
            scb.Measurement._resolve_meas_key(REAL_FILE, "nope")

    def test_non_dict_entries_are_not_listed_as_blocks(self):
        """"name" is a string in a written file, not a measurement block."""
        with pytest.raises(KeyError) as excinfo:
            scb.Measurement._resolve_meas_key(SIMULATED_FILE, "missing")
        assert "'name'" not in excinfo.value.args[0]

    def test_a_missing_name_raises_instead_of_passing_through(self):
        """Raising here names the alternatives; a bare dict lookup later would not."""
        with pytest.raises(KeyError):
            scb.Measurement._resolve_meas_key(SIMULATED_FILE, "2W_doppler")


# endregion                                          #
# ---------------------------------------------------#
