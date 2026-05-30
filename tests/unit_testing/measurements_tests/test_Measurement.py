# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import spiceypy as spice
import pytest
import numpy as np

km, sec, kg, hr = scb.Units.get_units(["km", "sec", "kg", "hr"])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def epochs():
    return scb.EpochArray.interval(
        "2028 FEB 08 13:31:08.00",
        "2028 FEB 09 05:31:08.00",
        scb.ArrayWUnits(1, hr),
        sys="TDB",
    )


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
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), Orbiter_area, Orbiter_cr_srp
    )


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
        [km, km, km, km * sec**-1, km * sec**-1, km * sec**-1],
    )


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
        "scft_transpd_delay_s",
    ]

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
        delta=np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]),
    )

    assert 0.002 < light_time < 0.05, "Light time outside physical bounds"


def test_compute_RLT_SUN_lt(GS1):
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


def test_compute_RLT_BODIES_lt(GS1):
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


def test_compute_RLT_lt(GS1):
    pytest.skip('MOVE TO INTEGRATION, NEED S/C SPK LOADED')
    # meas = scb.Measurement(name="Test", instrument=GS1)

    # t0 = 886901468.24  # known working epoch
    # light_time = 0.02
    # t1, t2, t3 = t0, t0 + light_time, t0 + 2 * light_time

    # ssb_to_gs_t1 = spice.spkezr("DSS-55", t1, "J2000", "NONE", "SSB")[0]
    # sun_to_sc_t2 = spice.spkezr("-1000", t2, "J2000", "NONE", "10")[0]
    # sun_to_gs_t1 = spice.spkezr("DSS-55", t1, "J2000", "NONE", "10")[0]

    # delta = np.zeros(6)

    # # ── Case 1: RLT-SUN only ──
    # dt_sun, r_vec_sun = meas._compute_RLT_lt(
    #     components="RLT-SUN",
    #     r1_r2_vec=sun_to_gs_t1,
    #     r2_r3_vec=sun_to_sc_t2,
    # )
    # assert dt_sun > 0
    # assert r_vec_sun.shape == (3,)
    # assert 1e-6 < dt_sun < 1e-3

    # # ── Case 2: RLT-BODIES only ──
    # dt_bodies, r_vec_bodies = meas._compute_RLT_lt(
    #     components="RLT-BODIES",
    #     t2_t1=t1,
    #     t3_t2=t2,
    #     sc_spice_id="-1000",
    #     ssb_to_gs_t_31_RLT=ssb_to_gs_t1,
    #     way="up",
    #     delta=delta,
    # )
    # assert dt_bodies > 0
    # assert np.all(r_vec_bodies == 0)
    # assert 1e-12 < dt_bodies < 1e-4

    # # ── Case 3: RLT combined == sum of parts ──
    # dt_rlt, _ = meas._compute_RLT_lt(
    #     components="RLT",
    #     r1_r2_vec=sun_to_gs_t1,
    #     r2_r3_vec=sun_to_sc_t2,
    #     t2_t1=t1,
    #     t3_t2=t2,
    #     sc_spice_id="-1000",
    #     ssb_to_gs_t_31_RLT=ssb_to_gs_t1,
    #     way="up",
    #     delta=delta,
    # )
    # assert np.isclose(dt_rlt, dt_sun + dt_bodies, rtol=1e-10)
    # assert dt_rlt > dt_sun
    # assert dt_rlt > dt_bodies

    # # ── Case 4: invalid component → zero ──
    # dt_none, _ = meas._compute_RLT_lt(components="INVALID")
    # assert dt_none == 0

    # # ── Case 5: identical vectors → r32_r21 == 0, should not crash ──
    # dt_zero, _ = meas._compute_RLT_lt(
    #     components="RLT-SUN",
    #     r1_r2_vec=sun_to_gs_t1,
    #     r2_r3_vec=sun_to_gs_t1,
    # )
    # assert np.isfinite(dt_zero)


def test_compute_precise_RTLT(GS1, sc):
    pytest.skip('MOVE TO INTEGRATION, NEED S/C SPK LOADED')
    # meas = scb.Measurement(name="Test", instrument=GS1)

    # t3_et = scb.SpiceManager.cal2et("2029 APR 01 09:04:07.350")

    # electronic_delays_dict = {
    #     "rcv_tt_dly": 0.0,
    #     "trn_tt_dly": 0.0,
    #     "array_delay": 0.0,
    #     "dl_zheight_corr": 0.0,
    #     "ul_zheight_corr": 0.0,
    #     "scft_transpd_delay_s": 0.0,
    # }

    # solar_corona_dict = {
    #     "ul_freq": 0.0,
    #     "M2": 0.0,
    # }

    # delta_partials = scb.ArrayWUnits(np.zeros(6), km)  # adjust to match your type

    # rtlt, rtlt_parts, t2, t1 = meas._compute_precise_RTLT(
    #     sc_spice_id="-1000",
    #     gs_spice_id="DSS-55",
    #     tt_et=t3_et,
    #     electronic_delays_dict=electronic_delays_dict,
    #     solar_corona_dict=solar_corona_dict,
    #     delta_partials=delta_partials,
    #     gs_delta_awf=None,
    # )

    # # ── Output structure ──
    # assert rtlt_parts.shape == (17, 1)
    # assert np.isclose(rtlt, np.sum(rtlt_parts), rtol=1e-10)  # rtlt == sum of parts

    # # ── Time ordering: t1 < t2 < t3 ──
    # assert t1 < t2 < t3_et, "Signal must travel forward in time"

    # lt_dwn = rtlt_parts[0][0]  # CN down-leg
    # lt_up = rtlt_parts[2][0]  # CN up-leg
    # assert 0.01 < lt_dwn < 0.1, f"Down-leg light time out of range: {lt_dwn}"
    # assert 0.01 < lt_up < 0.1, f"Up-leg light time out of range: {lt_up}"
    # assert 0.02 < rtlt < 0.2, f"RTLT out of physical range: {rtlt}"

    # # ── GR delays are small but non-zero ──
    # dt_GR_dwn = rtlt_parts[1][0]
    # dt_GR_up = rtlt_parts[3][0]
    # assert dt_GR_dwn > 0, "GR down-leg delay must be positive"
    # assert dt_GR_up > 0, "GR up-leg delay must be positive"
    # assert dt_GR_dwn < lt_dwn, "GR delay must be smaller than Newtonian"
    # assert dt_GR_up < lt_up, "GR delay must be smaller than Newtonian"

    # # ── Electronic delays zero ──
    # assert rtlt_parts[15][0] == 0.0  # dt_tau_D
    # assert rtlt_parts[16][0] == 0.0  # dt_tau_U

    # # ── Solar corona zero (we passed zeros) ──
    # assert rtlt_parts[12][0] == 0.0  # dt_SC_dwn
    # assert rtlt_parts[14][0] == 0.0  # dt_SC_up


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
            "test", target=sc, epochs=epochs, observed_meas=obs_mock
        )
