# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ========================================#
#  Unit Tests for the DopplerReal class  #
# ========================================#
import src.scarabaeus as scb
import numpy as np
import pytest
import sys
from types import SimpleNamespace

kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def computed_meas(trk_data_file):
    """
    Fixture to create a sample computed mass for testing.
    """
    trk_data = scb.Utils.load_json(str(trk_data_file))

    computed_meas_dict = {
        # Doppler auxiliary data
        "Tc": trk_data["aux_SFDU_16"]["Tc"],
        "M2_num": trk_data["aux_SFDU_16"]["M2_num"],
        "M2_den": trk_data["aux_SFDU_16"]["M2_den"],
        "rcv_time_tag_delay": trk_data["2W_doppler"]["rcv_time_tag_delay"],
        "transmit_time_tag_delay": trk_data["2W_doppler"]["transmit_time_tag_delay"],
        "outlier_flag_doppler": trk_data["aux_SFDU_16"]["outlier_flag"],
        "year_SFDU16": trk_data["aux_SFDU_16"]["year"],
        "doy_SFDU16": trk_data["aux_SFDU_16"]["doy"],
        "sec_SFDU16": trk_data["aux_SFDU_16"]["sec"],
        # Sequential-Ranging auxiliary data
        "RU": (((np.array(trk_data["aux_SFDU_7"]["exc_scalar_den"])
                / np.array(trk_data["aux_SFDU_7"]["exc_scalar_num"]))
                / 16)** -1).tolist(),
        "M": trk_data["aux_SFDU_7"]["rng_modulo"],
        "rng_type": trk_data["aux_SFDU_7"]["rng_type"],
        "outlier_flag_sranging": trk_data["aux_SFDU_7"]["outlier_flag"],
        "ul_stn_cal": trk_data["aux_SFDU_7"]["ul_stn_cal"],
        "dl_stn_cal": trk_data["aux_SFDU_7"]["dl_stn_cal"],
        "ul_freq": trk_data["aux_SFDU_7"]["ul_freq"],
        "year_SFDU7": trk_data["aux_SFDU_7"]["year"],
        "doy_SFDU7": trk_data["aux_SFDU_7"]["doy"],
        "sec_SFDU7": trk_data["aux_SFDU_7"]["sec"],
        # Delays auxiliary data
        "ul_zheight_corr": trk_data["aux_SFDU_2"]["ul_zheight_corr"],
        "ul_cal_freq": trk_data["aux_SFDU_2"]["ul_cal_freq"],
        "dl_zheight_corr": trk_data["aux_SFDU_3"]["dl_zheight_corr"],
        # Ramp table auxiliary (from SFDU_9)
        "ramp_sec": trk_data["aux_SFDU_9"]["sec"],
        "ramp_day": trk_data["aux_SFDU_9"]["doy"],
        "ramp_freq": trk_data["aux_SFDU_9"]["ramp_freq"],
        "ramp_rate": trk_data["aux_SFDU_9"]["ramp_rate"],
        "ramp_type": trk_data["aux_SFDU_9"]["ramp_type"],
    }
    return computed_meas_dict


@pytest.fixture
def state_dummy():
    state_dummy = [
        ("position", 3, "estimated", "dynamic", None, None),
        (   "velocity",
            3,
            "estimated",
            "dynamic",
            None,
            None,),
        (   "range_bias_1",
            1,
            "estimated",
            "static",
            scb.GroundStation("DSS-14"),
            None,),]
    return state_dummy


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(computed_meas, state_dummy):
    """
    Verifies that object is constructed correctly.
    """
    name = "GS1 Doppler Model"
    GS1 = scb.GroundStation("DSS-14")
    doppler_sigma = scb.ArrayWUnits(1e-3, sec)

    DopplerReal_GS1 = scb.DopplerReal(
        name,
        GS1,
        sigma=doppler_sigma,
        computed_measurements_dict=computed_meas,
        state_definition=state_dummy,)

    assert DopplerReal_GS1._name == name
    assert DopplerReal_GS1._instrument == GS1
    assert DopplerReal_GS1._sigma == doppler_sigma
    assert DopplerReal_GS1._computed_measurements_dict == computed_meas
    assert DopplerReal_GS1._state_definition == state_dummy


def test_computed_measurements_uses_each_row_delays(computed_meas, state_dummy, monkeypatch):
    """
    Verifies that each count interval passes the electronic delays of its own row to
    the RTLT driver, at both ends of the count.
    """
    aux = dict(
        computed_meas,
        rcv_time_tag_delay=[1e-6, 2e-6, 3e-6],
        transmit_time_tag_delay=[4e-6, 5e-6, 6e-6],
        ul_zheight_corr=[1e-8, 2e-8, 3e-8],
        dl_zheight_corr=[4e-8, 5e-8, 6e-8],
        array_delay=np.array([7e-8, 8e-8, 9e-8]),
        scft_transpd_delay=0.0,
        scft_transpd_delay_s=0.0,
    )
    doppler = scb.DopplerReal(
        "GS1 Doppler Model",
        scb.GroundStation("DSS-14"),
        sigma=scb.ArrayWUnits(1e-3, sec),
        computed_measurements_dict=aux,
        state_definition=state_dummy,
    )
    target = scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )
    epochs = scb.EpochArray(np.array([576000000.0, 576000600.0, 576001200.0]), sys="TDB")

    # Record the delays and stop once the last row has been sent, before the ramp tables
    seen = []

    class _Stop(Exception):
        pass

    def _record(**kwargs):
        seen.append(kwargs["electronic_delays_dict"])
        if len(seen) == 6:
            raise _Stop
        # The parts vector must be a real 17-element decomposition, not None:
        # this branch captures it for the analytical-partials endpoint cache
        # (GR at [1]/[3], solar corona at [12]/[14]).
        return 0.0, np.zeros(17), 0.0, 0.0

    monkeypatch.setattr(scb.Measurement, "_compute_precise_RTLT", staticmethod(_record))
    # _Stop is the test's own early exit. DopplerReal wraps the RTLT call in a
    # broad `except Exception` that re-raises as RuntimeError with the failing
    # measurement named, so the sentinel arrives wrapped; accept either.
    with pytest.raises((_Stop, RuntimeError)):
        doppler.computed_measurements(target, epoch_array=epochs)

    for key in ("rcv_tt_dly", "trn_tt_dly", "ul_zheight_corr", "dl_zheight_corr", "array_delay"):
        sent = [d[key] for d in seen]
        rows = [sent[0], sent[2], sent[4]]
        assert sent[1::2] == rows, key
        assert len(set(rows)) == 3, key


@pytest.mark.parametrize(
    "frame, solver",
    [("barycentric", "_compute_precise_RTLT"), ("geocentric", "_compute_geocentric_RTLT")],
)
def test_computed_measurements_solves_light_time_in_the_chosen_frame(
    computed_meas, state_dummy, monkeypatch, frame, solver
):
    """
    Verifies that ``light_time_frame`` selects the RTLT driver at both ends of the count.
    """
    doppler = scb.DopplerReal(
        "GS1 Doppler Model",
        scb.GroundStation("DSS-14"),
        sigma=scb.ArrayWUnits(1e-3, sec),
        computed_measurements_dict=dict(
            computed_meas, scft_transpd_delay=0.0, scft_transpd_delay_s=0.0
        ),
        state_definition=state_dummy,
        light_time_frame=frame,
    )
    target = scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )
    epochs = scb.EpochArray(np.array([576000000.0, 576000600.0, 576001200.0]), sys="TDB")

    calls = {"_compute_precise_RTLT": 0, "_compute_geocentric_RTLT": 0}

    class _Stop(Exception):
        pass

    def _recorder(name):
        def _record(**kwargs):
            calls[name] += 1
            if sum(calls.values()) == 2:
                raise _Stop
            return 0.0, np.zeros(17), 0.0, 0.0

        return staticmethod(_record)

    for name in calls:
        monkeypatch.setattr(scb.Measurement, name, _recorder(name))
    with pytest.raises((_Stop, RuntimeError)):
        doppler.computed_measurements(target, epoch_array=epochs)

    assert calls[solver] == 2
    assert sum(calls.values()) == 2


def test_unknown_light_time_frame_is_refused(computed_meas):
    with pytest.raises(scb.Measurement.LightTimeFrameError):
        scb.DopplerReal(
            "dop", scb.GroundStation("DSS-14"), sigma=scb.ArrayWUnits(1e-3, sec),
            computed_measurements_dict=computed_meas, light_time_frame="topocentric",
        )


def test_bad_light_time_settings_are_refused(computed_meas):
    with pytest.raises(scb.Measurement.LightTimeSettingsError, match="max_iter"):
        scb.DopplerReal(
            "dop", scb.GroundStation("DSS-14"), sigma=scb.ArrayWUnits(1e-3, sec),
            computed_measurements_dict=computed_meas, light_time_settings={"max_iter": 20},
        )


def test_light_time_settings_reach_the_solver(computed_meas, state_dummy, monkeypatch):
    """
    Verifies that both ends of the count solve the light time with the model's
    settings, completed with the defaults.
    """
    doppler = scb.DopplerReal(
        "GS1 Doppler Model",
        scb.GroundStation("DSS-14"),
        sigma=scb.ArrayWUnits(1e-3, sec),
        computed_measurements_dict=dict(
            computed_meas, scft_transpd_delay=0.0, scft_transpd_delay_s=0.0
        ),
        state_definition=state_dummy,
        light_time_settings={"tolerance": 1e-11},
    )
    target = scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )
    epochs = scb.EpochArray(np.array([576000000.0, 576000600.0, 576001200.0]), sys="TDB")

    seen = []

    class _Stop(Exception):
        pass

    def _record(**kwargs):
        seen.append(kwargs["light_time_settings"])
        if len(seen) == 2:
            raise _Stop
        return 0.0, np.zeros(17), 0.0, 0.0

    monkeypatch.setattr(scb.Measurement, "_compute_precise_RTLT", staticmethod(_record))
    with pytest.raises((_Stop, RuntimeError)):
        doppler.computed_measurements(target, epoch_array=epochs)

    expected = dict(scb.Measurement.LIGHT_TIME_SETTINGS, tolerance=1e-11)
    assert seen == [expected, expected]


def test_write_aux_data_accepts_list_valued_aux():
    """
    _write_aux_data must not assume the aux values are ndarrays.

    _read_aux builds ul_freq and friends as ndarrays (``[...] + np.max(...)``
    broadcasts), but default_doppler_aux builds them as plain lists. Calling
    .tolist() on the stored value therefore worked only for a dict that came
    from a real file, and writing simulated Doppler from the default aux dict
    died with ``AttributeError: 'list' object has no attribute 'tolist'``.
    """
    n = 4
    keys = (
        "ul_zheight_corr", "dl_zheight_corr", "transmit_time_tag_delay",
        "rcv_time_tag_delay", "array_delay", "Tc", "M2_num", "M2_den",
        "outlier_flag_doppler", "year_SFDU16", "doy_SFDU16", "sec_SFDU16",
        "ramp_sec", "ramp_day", "ramp_freq", "ramp_rate", "ramp_type",
    )
    aux = {key: [0] * n for key in keys}
    # The five that used to be .tolist()-ed, as plain lists.
    for key in ("ul_freq", "ul_stn_cal", "dl_stn_cal",
                "exc_scalar_num", "exc_scalar_den"):
        aux[key] = [1.0] * n

    model = scb.DopplerReal(
        "dop", scb.GroundStation("DSS-14"), sigma=scb.ArrayWUnits(1e-3, sec),
        computed_measurements_dict=aux,
    )
    dict_meas = {
        "year": [1999] * n, "doy": [123] * n, "sec": [456.789] * n,
        "meas": [1.0] * n, "spice_id": -62, "outlier_flag": [0.0] * n,
    }
    dict_DB = {"name": "sim", "meas_sim": dict_meas}
    model._write_aux_data(dict_DB, dict_meas)

    assert dict_DB["2W_sranging"]["ul_freq"] == [1.0] * n
    # Doppler has a real aux_SFDU_16 block, so unlike ranging it never touches
    # the measurement block's time tags.
    assert dict_meas["year"] == [1999] * n
    assert dict_meas["sec"] == [456.789] * n
    assert len(dict_DB["aux_SFDU_16"]["year"]) == n


def _aux_model():
    """A Doppler model with no tracking data, to build simulated aux data from."""
    return scb.DopplerReal("dop", scb.GroundStation("DSS-14"), sigma=scb.ArrayWUnits(1e-3, sec))


@pytest.mark.parametrize("N", [1, 2, 5, 10, 15, 20, 50])
def test_default_aux_arrays_are_internally_consistent(N):
    """
    One length for the per-observation arrays, one for the ramp arrays.

    This dict used to carry three different lengths: ramp_sec/ramp_day off a
    hard-coded ten-step grid, year/doy/sec off a fixed 60 s grid regardless of
    N, and everything else N long. Only N == 10 on a span that was a multiple of
    600 s lined up. Below ten, RampTableManager.integrate indexes ramp_freq with
    an index derived from ramp_sec and raised IndexError.
    """
    span = 900.0
    t_0 = 661300000.0
    n_fit = int(span // 60.0)        # 60 s count intervals that fit the span
    n_expected = min(N, n_fit)
    sp = scb.EpochArray(np.array([t_0, t_0 + span]), "TDB", "AWU")

    if N > n_fit:
        # The spacing is fixed by the data type, so the count gives way.
        with pytest.warns(RuntimeWarning, match="Reduced the observation count"):
            aux = _aux_model().default_doppler_aux(sp[0], sp[-1], N)
    else:
        aux = _aux_model().default_doppler_aux(sp[0], sp[-1], N)

    ramp_keys = {"ramp_sec", "ramp_day", "ramp_freq", "ramp_rate", "ramp_type", "ramp_year"}
    n_ramp = len(aux["ramp_sec"])
    for key in ramp_keys:
        assert len(aux[key]) == n_ramp, f"{key} disagrees with the ramp grid"

    for key, value in aux.items():
        if key in ramp_keys or not isinstance(value, (list, np.ndarray)):
            continue
        assert len(value) == n_expected, (
            f"{key} is {len(value)} long, expected {n_expected}"
        )


def test_default_aux_count_interval_is_the_dsn_standard_60_s():
    """
    Tc is 60 s by default, fixed, and the sample spacing matches it.

    60 s is the standard DSN two-way Doppler count time, the default
    count_interval. When the requested count does not fit the span the count is
    reduced instead -- stretching Tc would misdescribe the data, since
    consecutive counts would then overlap or leave gaps.
    """
    t_0 = 661300000.0
    span = 910.0                      # deliberately not a multiple of 60
    sp = scb.EpochArray(np.array([t_0, t_0 + span]), "TDB", "AWU")

    aux = _aux_model().default_doppler_aux(sp[0], sp[-1], 15)
    assert aux["Tc"] == [60.0] * 15
    # The time tags are 60 s apart, to within SPICE's own ET->UTC granularity.
    # The grid steps exactly 60.0 s of ET, but the tags are the station-time
    # strings spice.et2utc emits, and successive calls can disagree in the 7th
    # decimal (...816541195 then ...816541076 at this epoch) even though
    # str2et inverts each one exactly. So ~120 ns, not 0, and it is SPICE's
    # rounding rather than a drift we could remove -- the file still reloads to
    # the originating epochs bit-exactly, which is what has to hold.
    np.testing.assert_allclose(np.diff(aux["sec_SFDU16"]), 60.0, rtol=0, atol=1e-6)

    # 16 counts do not fit in 910 s, so the count drops to 15 and Tc stays 60.
    with pytest.warns(RuntimeWarning, match="Reduced the observation count"):
        aux = _aux_model().default_doppler_aux(sp[0], sp[-1], 16)
    assert len(aux["Tc"]) == 15
    assert aux["Tc"] == [60.0] * 15


def test_default_aux_ramp_table_covers_the_uplink_transmit_time():
    """
    The ramp grid starts before the first observation by more than a round trip,
    because the ramp is queried at t1 = t3 - RTLT.
    """
    t_0 = 661300000.0
    sp = scb.EpochArray(np.array([t_0, t_0 + 900.0]), "TDB", "AWU")
    aux = _aux_model().default_doppler_aux(sp[0], sp[-1], 5)
    table = [aux["ramp_sec"], aux["ramp_day"], aux["ramp_freq"],
             aux["ramp_rate"], aux["ramp_type"]]
    t3 = float(aux["sec_SFDU16"][0])
    doy = aux["doy_SFDU16"][0]
    f_int, _ = scb.RampTableManager.integrate(table, t3 - 1400.0, t3, doy, doy)
    assert f_int > 0.0


def _doppler(computed_meas, **kwargs):
    return scb.DopplerReal(
        "dop", scb.GroundStation("DSS-14"), sigma=scb.ArrayWUnits(1e-3, sec),
        computed_measurements_dict=computed_meas, **kwargs,
    )


def test_partials_default_to_analytical(computed_meas):
    """
    With no partials_method given, the model uses the Moyer analytical partials.
    """
    model = _doppler(computed_meas)
    assert model._partials_method == "analytical"


@pytest.mark.parametrize("method", ["analytical", "fd"])
def test_a_supported_partials_method_is_kept(computed_meas, method):
    model = _doppler(computed_meas, partials_method=method)
    assert model._partials_method == method


def test_the_removed_ideal_partials_are_refused(computed_meas):
    with pytest.raises(ValueError, match="Unknown partials_method 'ideal'"):
        _doppler(computed_meas, partials_method="ideal")


def test_analytical_partials_of_srp_and_maneuver_are_zero(computed_meas, monkeypatch):
    """
    eta_srp and dv_man move the observable only through the trajectory, which the
    filter maps through the STM, so the default partials give zero columns for them
    instead of raising.
    """
    model = _doppler(
        computed_meas,
        state_definition=[
            ("eta_srp", 1, "estimated", "static", None, None),
            ("dv_man1", 3, "estimated", "static", None, None),
        ],
    )
    monkeypatch.setattr(model, "_ensure_partials_cache", lambda *args: None)

    rows = model.compute_partials(SimpleNamespace(name="-64", spice_id=-64), [None, None])

    assert len(rows) == 2
    for row in rows:
        np.testing.assert_array_equal(row, np.zeros(4))


# ---------------------------------------#
# region  Estimated solar-corona scale   #
# ---------------------------------------#
# The light-time solver stubbed below returns 1000 s plus a corona delay that
# grows by _CORONA_SLOPE per second of reception time, and the ramp integration
# counts 7.2 GHz between the two transmit times. The Doppler then moves with the
# corona scale κ by M2·f_T·κ·slope, the change of the corona over the count.
_CORONA_SLOPE = 1e-10  # [s/s]
_CORONA_EPOCHS = np.array([576000000.0, 576000600.0, 576001200.0])


def _corona_state(value):
    """A state estimating the solar-corona scale at ``value``."""
    return [("solar_corona_scale", 1, "estimated", "parameter", None,
             scb.ArrayWFrame(np.array([value]), scb.Units.get_units("unitless"), None))]


def _corona_doppler(computed_meas, monkeypatch, state_definition=None, partials_method=None):
    # The analytical rows are pre-scaled by c/ul_freq, so ul_freq must not be zero.
    model = _doppler(
        dict(computed_meas, scft_transpd_delay=0.0, scft_transpd_delay_s=0.0,
             ul_freq=[7.2e9] * len(computed_meas["ul_freq"])),
        state_definition=state_definition, partials_method=partials_method,
    )

    def solver(**kwargs):
        corona = 1e-7 + _CORONA_SLOPE * (kwargs["tt_et"] - _CORONA_EPOCHS[0])
        parts = np.zeros(17)
        parts[12], parts[14] = 0.4 * corona, 0.6 * corona
        return 1000.0 + corona, parts, kwargs["tt_et"] - 500.0, kwargs["tt_et"] - 1000.0

    mod = sys.modules[type(model).__module__]
    monkeypatch.setattr(model, "_rtlt_solver", lambda: solver)
    monkeypatch.setattr(mod.RampTableManager, "integrate", staticmethod(
        lambda table, t1s, t1e, d1s, d1e, *args, t1_q_sec_l=0.0, t2_q_sec_l=0.0, **kwargs: (
            ((t1e - t1s) + (t2_q_sec_l - t1_q_sec_l) + (d1e - d1s) * 86400.0) * 7.2e9,
            {"f_iq1": 7.2e9, "f_iq2": 7.2e9, "n_segments": 1})))
    return model


def _corona_target():
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )


def _corona_epochs():
    return scb.EpochArray(_CORONA_EPOCHS, sys="TDB")


def _corona_count(model):
    *_, meas = model.computed_measurements(_corona_target(), epoch_array=_corona_epochs())
    return np.ravel(meas.quantity.values)


def _M2(computed_meas):
    n = _CORONA_EPOCHS.size
    return np.asarray(computed_meas["M2_num"][:n], float) / np.asarray(computed_meas["M2_den"][:n], float)


def test_the_corona_scale_scales_the_corona_delay_of_both_endpoints(
    computed_meas, monkeypatch
):
    """
    An estimated scale multiplies the corona delay inside both endpoint light
    times. It used to have a partial and no effect on the computed Doppler. The
    cached corona stays unscaled, being the partial of the light time by κ.
    """
    nominal = _corona_count(_corona_doppler(computed_meas, monkeypatch))
    model = _corona_doppler(computed_meas, monkeypatch, _corona_state(1.5))

    scaled = _corona_count(model)

    np.testing.assert_allclose(model._last_rtlt_s, 1000.0 + 1.5 * model._last_sc_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model._last_rtlt_e, 1000.0 + 1.5 * model._last_sc_e, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        scaled - nominal, 0.5 * _M2(computed_meas) * 7.2e9 * _CORONA_SLOPE, atol=1e-4)


def test_a_corona_scale_of_one_leaves_the_doppler_unchanged(computed_meas, monkeypatch):
    nominal = _corona_doppler(computed_meas, monkeypatch)
    unit = _corona_doppler(computed_meas, monkeypatch, _corona_state(1.0))

    np.testing.assert_array_equal(_corona_count(unit), _corona_count(nominal))
    np.testing.assert_array_equal(unit._last_rtlt_s, nominal._last_rtlt_s)
    np.testing.assert_array_equal(unit._last_rtlt_e, nominal._last_rtlt_e)


def test_analytical_partial_of_the_corona_scale_matches_the_finite_difference(
    computed_meas, monkeypatch
):
    """
    (M2/Tc)(f_Te·ΔSC_e − f_Ts·ΔSC_s) against a central difference of κ through
    the forward model, both M2·f_T·slope here.
    """
    rows = {}
    for method in ("analytical", "fd"):
        model = _corona_doppler(computed_meas, monkeypatch, _corona_state(1.0),
                                partials_method=method)
        rows[method] = np.asarray(model.compute_partials(_corona_target(), _corona_epochs()))

    expected = (_M2(computed_meas) * 7.2e9 * _CORONA_SLOPE)[:, None]
    np.testing.assert_allclose(rows["fd"], expected, atol=1e-4)
    np.testing.assert_allclose(rows["analytical"], rows["fd"], atol=1e-4)


# endregion                              #
# ---------------------------------------#


# ---------------------------------------#
# region PPN gamma
# ---------------------------------------#
def _gamma_state(value):
    """A state estimating the PPN γ at ``value``."""
    return [("gamma_rel", 1, "estimated", "parameter", None,
             scb.ArrayWFrame(np.array([value]), scb.Units.get_units("unitless"), None))]


def _gamma_doppler(computed_meas, monkeypatch, state_definition=None, partials_method=None):
    """
    The corona Doppler with a light time of 1000 s + (1+γ)·slope·(t − t0), so
    the count moves by M2·f_T·slope per unit of γ; it records each γ it gets.
    """
    model = _corona_doppler(computed_meas, monkeypatch, state_definition, partials_method)
    model.solver_gammas = []

    def solver(**kwargs):
        model.solver_gammas.append(kwargs["gamma"])
        delay = (1.0 + kwargs["gamma"]) * _CORONA_SLOPE * (kwargs["tt_et"] - _CORONA_EPOCHS[0])
        return 1000.0 + delay, np.zeros(17), kwargs["tt_et"] - 500.0, kwargs["tt_et"] - 1000.0

    monkeypatch.setattr(model, "_rtlt_solver", lambda: solver)
    return model


def test_the_estimated_gamma_reaches_both_endpoint_light_times(computed_meas, monkeypatch):
    """The γ iterate replaces constants.gamma_GR in the light-time solver."""
    nominal = _gamma_doppler(computed_meas, monkeypatch)
    model = _gamma_doppler(computed_meas, monkeypatch, _gamma_state(1.5))

    shifted = _corona_count(model) - _corona_count(nominal)

    assert set(nominal.solver_gammas) == {1.0}
    assert set(model.solver_gammas) == {1.5}
    np.testing.assert_allclose(shifted, 0.5 * _M2(computed_meas) * 7.2e9 * _CORONA_SLOPE, atol=1e-4)


def test_fd_partial_of_gamma_differences_the_light_time(computed_meas, monkeypatch):
    model = _gamma_doppler(computed_meas, monkeypatch, _gamma_state(1.0), partials_method="fd")

    rows = np.asarray(model.compute_partials(_corona_target(), _corona_epochs()))

    np.testing.assert_allclose(rows, (_M2(computed_meas) * 7.2e9 * _CORONA_SLOPE)[:, None], atol=1e-4)
    assert model._gamma_rel_offset is None


def test_a_malformed_gamma_is_refused(computed_meas, monkeypatch):
    model = _gamma_doppler(computed_meas, monkeypatch, _gamma_state(np.inf))
    with pytest.raises(scb.Measurement.GammaRelError):
        _corona_count(model)


# endregion                              #
# ---------------------------------------#


# ---------------------------------------#
# region  Radiometric defaults           #
# ---------------------------------------#
def _aux_span():
    t_0 = 661300000.0
    return scb.EpochArray(np.array([t_0, t_0 + 900.0]), "TDB", "AWU")


def test_the_count_interval_starts_at_the_class_default():
    model = _aux_model()
    assert float(model.count_interval.values) == float(scb.DopplerReal.COUNT_INTERVAL.values) == 60.0


def test_the_count_interval_sets_tc_and_the_sample_spacing():
    """A user count interval reaches both Tc and the grid the tags sit on."""
    model = _aux_model()
    model.count_interval = scb.ArrayWUnits(2, scb.Units.get_units("min"))
    assert float(model.count_interval.convert_to(sec).values) == 120.0

    sp = _aux_span()
    aux = model.default_doppler_aux(sp[0], sp[-1], 5)
    assert aux["Tc"] == [120.0] * 5
    np.testing.assert_allclose(np.diff(aux["sec_SFDU16"]), 120.0, rtol=0, atol=1e-6)

    # By default the span is filled: 900 s holds seven whole 120 s counts.
    assert len(model.default_doppler_aux(sp[0], sp[-1])["Tc"]) == 7


def test_a_bare_count_interval_is_taken_in_seconds():
    model = _aux_model()
    model.count_interval = 30
    assert float(model.count_interval.values) == 30.0


@pytest.mark.parametrize("value", [
    0.0, -60.0, np.nan, True, np.array([60.0, 60.0]), scb.ArrayWUnits(60.0, km),
])
def test_a_bad_count_interval_is_refused(value):
    with pytest.raises(scb.Measurement.RadiometricDefaultError):
        _aux_model().count_interval = value


def test_the_model_defaults_reach_the_simulated_doppler_aux():
    model = _aux_model()
    model.transponder_delay = 2.0e-5
    model.ul_station_cal = 1.0
    model.dl_station_cal = 2.0
    model.ramp_rate = 0.5
    model.ramp_type = 2
    model.ramp_entries = 4
    model.rcv_time_tag_delay = 1.0e-3

    sp = _aux_span()
    aux = model.default_doppler_aux(sp[0], sp[-1], 5)

    assert aux["scft_transpd_delay"] == 2.0e-5
    X = scb.Measurement.DSN_BANDS["X"]
    assert aux["scft_transpd_delay_s"] == scb.Measurement.transponder_delay_seconds(
        2.0e-5, X.ru_reference_frequency.values, X.range_unit)
    assert aux["ul_stn_cal"] == [1.0] * 5
    assert aux["dl_stn_cal"] == [2.0] * 5
    assert aux["rcv_time_tag_delay"] == [1.0e-3] * 5
    assert aux["ramp_rate"] == [0.5] * 4
    assert aux["ramp_type"] == [2] * 4
    assert len(aux["ramp_sec"]) == 4
    # The ramp seconds are SPICE time-tag strings, good to about a microsecond.
    ramp_dt = np.asarray(aux["ramp_sec"]) - aux["ramp_sec"][0]
    np.testing.assert_allclose(np.asarray(aux["ramp_freq"]) - aux["ramp_freq"][0], 0.5 * ramp_dt,
                               rtol=0, atol=1e-5)


def test_the_default_aux_reproduces_the_class_defaults():
    """Untouched, the model simulates the values the class attributes give."""
    sp = _aux_span()
    aux = _aux_model().default_doppler_aux(sp[0], sp[-1], 5)
    M = scb.Measurement
    assert aux["Tc"] == [60.0] * 5
    assert aux["scft_transpd_delay"] == M.TRANSPONDER_DELAY
    assert aux["ul_stn_cal"] == [M.UL_STATION_CAL] * 5
    assert aux["dl_stn_cal"] == [M.DL_STATION_CAL] * 5
    assert aux["rcv_time_tag_delay"] == [float(M.RCV_TIME_TAG_DELAY.values)] * 5
    assert aux["ramp_rate"] == [float(M.RAMP_RATE.values)] * M.RAMP_ENTRIES
    assert aux["ramp_type"] == [M.RAMP_TYPE] * M.RAMP_ENTRIES
    assert aux["ul_freq"] == [M.DSN_BANDS["X"].uplink_frequency.values] * 5


def test_the_model_dsn_bands_set_the_simulated_uplink():
    """A band set on one model changes its aux data, not the class or other models."""
    import dataclasses

    model = _aux_model()
    X = model.dsn_bands["X"]
    model.dsn_bands = {
        **model.dsn_bands,
        "x": dataclasses.replace(X, uplink_frequency=scb.ArrayWUnits(7.18e9, 1 / sec)),
    }

    sp = _aux_span()
    aux = model.default_doppler_aux(sp[0], sp[-1], 5)
    assert aux["ul_freq"] == [7.18e9] * 5
    assert aux["ramp_freq"][0] == 7.18e9

    assert scb.Measurement.DSN_BANDS["X"].uplink_frequency.values == 7.1523135969977e09
    assert _aux_model().dsn_bands["X"] is scb.Measurement.DSN_BANDS["X"]


def test_the_solar_corona_coefficient_reaches_the_light_time_solver(computed_meas, monkeypatch):
    model = _corona_doppler(computed_meas, monkeypatch)
    model.solar_corona_coefficient = 2.0e-19
    solver = model._rtlt_solver()
    seen = []

    def spy(**kwargs):
        seen.append(kwargs["solar_corona_dict"]["coefficient"])
        return solver(**kwargs)

    monkeypatch.setattr(model, "_rtlt_solver", lambda: spy)
    _corona_count(model)
    assert seen and set(seen) == {2.0e-19}


# endregion                              #
# ---------------------------------------#
