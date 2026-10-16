# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================================#
#  Unit Tests for the SequentialRangingReal class  #
# ==================================================#
import src.scarabaeus as scb
import pytest
import numpy as np
import sys
import warnings
from types import SimpleNamespace


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def instrument():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")


@pytest.fixture
def computed_meas_dict(trk_data_file):
    """
    Fixture to create a sample computed measurements dictionary for testing.
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


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(instrument, computed_meas_dict):
    """
    Verifies that object is constructed correctly.
    """
    name = "GS1 Sequential Ranging Model"
    sranging_sigma = scb.ArrayWUnits(3, None)
    SequentialRangingReal_GS1 = scb.SequentialRangingReal(
                                name,
                                instrument,
                                sigma=sranging_sigma,
                                computed_measurements_dict=computed_meas_dict,)

    assert SequentialRangingReal_GS1._name == name
    assert SequentialRangingReal_GS1._instrument == instrument
    assert SequentialRangingReal_GS1._sigma == sranging_sigma



def test_unknown_light_time_frame_is_refused(instrument):
    with pytest.raises(scb.Measurement.LightTimeFrameError):
        scb.SequentialRangingReal(
            "sr", instrument, sigma=scb.ArrayWUnits(3, None), light_time_frame="topocentric"
        )


def test_bad_light_time_settings_are_refused(instrument):
    with pytest.raises(scb.Measurement.LightTimeSettingsError, match="tolerance"):
        scb.SequentialRangingReal(
            "sr", instrument, sigma=scb.ArrayWUnits(3, None),
            light_time_settings={"tolerance": 0.0},
        )


def test_light_time_settings_reach_the_solver(instrument, monkeypatch):
    """
    Verifies that the light time is solved with the model's settings, completed
    with the defaults.
    """
    kg, km = scb.Units.get_units(["kg", "km"])
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(
            _aux_for_write(3), RU=[1.0] * 3, M2_num=[880] * 3, M2_den=[749] * 3,
            scft_transpd_delay=0.0, scft_transpd_delay_s=0.0,
        ),
        light_time_settings={"max_iterations": 20},
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
        raise _Stop

    monkeypatch.setattr(scb.Measurement, "_compute_precise_RTLT", staticmethod(_record))
    with pytest.raises((_Stop, RuntimeError)):
        model.computed_measurements(target, epoch_array=epochs)

    assert seen == [dict(scb.Measurement.LIGHT_TIME_SETTINGS, max_iterations=20)]


def _aux_for_write(n):
    """Minimal computed_measurements_dict covering every key _write_aux_data reads."""
    per_obs = [
        "M", "M2_num", "M2_den", "rng_type", "version_num",
        "outlier_flag_sranging", "ul_freq", "ul_zheight_corr", "dl_zheight_corr",
        "transmit_time_tag_delay", "rcv_time_tag_delay", "array_delay",
        "ul_stn_cal", "dl_stn_cal", "exc_scalar_num", "exc_scalar_den",
    ]
    aux = {key: [0] * n for key in per_obs}
    # Times for the pass this aux dict describes -- deliberately not the epochs
    # the tests below simulate.
    aux.update(year_SFDU7=[2020] * n, doy_SFDU7=[350] * n, sec_SFDU7=[1.0] * n)
    aux.update(ramp_sec=[0.0], ramp_day=[350], ramp_freq=[7.15e9],
               ramp_rate=[0.0], ramp_type=[1])
    return aux


def _meas_block(n):
    """The measurement block write_observed_measurements builds before _write_aux_data."""
    return {
        "year": [1999] * n, "doy": [123] * n, "sec": [456.789] * n,
        "meas": [1.0] * n, "spice_id": -62, "outlier_flag": [0.0] * n,
    }


def test_write_aux_data_keeps_the_simulated_time_tags(instrument):
    """
    _write_aux_data must not retag the measurements with the aux dict's epochs.

    Sequential ranging has no separate SFDU-7 block: its aux fields live in the
    measurement block, so _write_aux_data writes into the same dict that
    write_observed_measurements has already filled with year/doy/sec converted
    from the epochs actually simulated. It used to overwrite them with
    year_SFDU7/doy_SFDU7/sec_SFDU7, which describe whatever pass was read -- and
    observed_measurements rebuilds its epochs from those very columns, so a
    simulated file came back with its measurements paired to the wrong times.
    """
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=_aux_for_write(5),
    )
    # One simulated epoch against a five-observation aux dict.
    dict_meas = _meas_block(1)
    model._write_aux_data({"name": "sim", "meas_sim": dict_meas}, dict_meas)

    assert dict_meas["year"] == [1999]
    assert dict_meas["doy"] == [123]
    assert dict_meas["sec"] == [456.789]
    # And it must not inherit the aux dict's five-long flag array either.
    assert len(dict_meas["outlier_flag"]) == 1


def test_write_aux_data_persists_edited_flags_when_lengths_agree(instrument):
    """
    The aux outlier-flag copy still reaches the file when it describes these
    observations -- MeasurementSpec.flag keeps it in step with
    ObsData.meas_outliers, so an edited spec written back out saves the
    post-edit flags rather than fresh zeros.
    """
    n = 5
    aux = _aux_for_write(n)
    aux["outlier_flag_sranging"] = [1, 0, 0, 0, 0]
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=aux,
    )
    dict_meas = _meas_block(n)
    model._write_aux_data({"name": "sim", "meas_sim": dict_meas}, dict_meas)
    assert list(dict_meas["outlier_flag"]) == [1, 0, 0, 0, 0]


@pytest.mark.parametrize("N", [1, 2, 5, 10, 20, 50])
def test_default_aux_arrays_are_internally_consistent(N):
    """
    One length for the per-observation arrays, one for the ramp arrays.

    These used to be sized independently: ramp_sec/ramp_day came off a
    hard-coded ten-step grid while ramp_freq/ramp_rate/ramp_type were N long, so
    only N == 10 produced a self-consistent dict. Below it,
    RampTableManager.integrate indexes ramp_freq with an index derived from
    ramp_sec and raised IndexError; above it the extra entries were dead.
    """
    import numpy as np

    span = 900.0
    t_0 = 661300000.0
    sp = scb.EpochArray(np.array([t_0, t_0 + span]), "TDB", "AWU")
    aux = scb.SequentialRangingReal.default_seqranging_aux(sp[0], sp[-1], N)

    ramp_keys = {"ramp_sec", "ramp_day", "ramp_freq", "ramp_rate", "ramp_type", "ramp_year"}
    n_ramp = len(aux["ramp_sec"])
    for key in ramp_keys:
        assert len(aux[key]) == n_ramp, f"{key} disagrees with the ramp grid"

    for key, value in aux.items():
        if key in ramp_keys or not isinstance(value, (list, np.ndarray)):
            continue
        assert len(value) == N, f"{key} is {len(value)} long, expected N={N}"


@pytest.mark.parametrize("N", [1, 5, 10, 50])
def test_default_aux_ramp_table_covers_the_uplink_transmit_time(N):
    """
    The ramp grid must start before the first observation by more than a
    round-trip light time.

    The ramp is queried at the uplink transmit time t1 = t3 - RTLT, so a table
    that merely spanned the observations failed on the first one with
    RampCoverageError -- the same gap the real tracking passes hit.
    """
    import numpy as np

    t_0 = 661300000.0
    sp = scb.EpochArray(np.array([t_0, t_0 + 900.0]), "TDB", "AWU")
    aux = scb.SequentialRangingReal.default_seqranging_aux(sp[0], sp[-1], N)
    table = [aux["ramp_sec"], aux["ramp_day"], aux["ramp_freq"],
             aux["ramp_rate"], aux["ramp_type"]]

    # First observation, with its uplink transmit time one Mars one-way light
    # time earlier.
    t3 = float(aux["sec_SFDU7"][0])
    doy = aux["doy_SFDU7"][0]
    f_int, _ = scb.RampTableManager.integrate(table, t3 - 1400.0, t3, doy, doy)
    assert f_int > 0.0


def test_the_read_installs_the_aux_data_on_the_model(instrument, trk_data_file):
    """
    observed_measurements owns the aux data, so every caller gets it.

    The aux data had two owners: ObsData.aux_dict, which the read returns, and
    computed_measurements_dict, which computed_measurements actually reads. The
    only thing connecting them was one line in MeasurementSpec.build, so a spec
    assembled any other way -- from_models, from_dict, from_dicts, or the models
    driven directly -- reached computed_measurements with the attribute still
    None and failed with "TypeError: 'NoneType' object is not subscriptable".

    This is also the only test that drives trk_data_file through the real read,
    so it guards the three things that file used to get wrong: seconds of day
    past the end of the day, a ramp table that did not cover the observations,
    and ranging measurements that were all no-data sentinels.
    """
    km = scb.Units.get_units(["km"])
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None)
    )
    assert model.computed_measurements_dict is None

    obs = model.observed_measurements(
        file_name=str(trk_data_file),
        meas_name="2W_sranging",
        units=km / km,
        sc_transponder_delay_RU=0.0,
    )

    assert obs.aux_dict is not None
    assert model.computed_measurements_dict is obs.aux_dict, (
        "the read installs the aux data it produced on the model, so no caller "
        "has to copy it across"
    )
    # Real measurements, so nothing is flagged as a no-data sentinel.
    assert int(np.sum(obs.meas_outliers)) == 0


def test_a_read_without_aux_data_does_not_clear_an_explicit_dict(instrument):
    """
    A dict supplied by hand survives a read that produces no aux data.

    The install is guarded on None so a model set up for simulation with a
    default_*_aux dict is not emptied by reading a file whose measurement type
    carries no aux block.
    """
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=_aux_for_write(4),
    )
    before = model.computed_measurements_dict
    # _read_aux returns None on the base, which is what a no-aux type does.
    assert scb.SequentialRangingReal.__mro__[1]._read_aux(model, {}, "x") is None
    assert model.computed_measurements_dict is before


def test_the_misspelled_transponder_option_is_refused(instrument, trk_data_file):
    """
    The old misspelling ``sc_transpoder_delay_RU`` (no "n") is gone.

    The option is a real delay, so a keyword the model quietly ignored would
    give a wrong observable; the old spelling has to fail rather than fall back
    to the default.
    """
    km = scb.Units.get_units(["km"])
    model = scb.SequentialRangingReal("sr", instrument, sigma=scb.ArrayWUnits(3, None))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        model.observed_measurements(
            file_name=str(trk_data_file), meas_name="2W_sranging", units=km / km,
            sc_transpoder_delay_RU=16529.7,
        )


def test_an_unrecognised_read_option_still_raises(instrument, trk_data_file):
    """
    A real typo is not swallowed.

    The option moved off the base signature into **read_kwargs, so the check
    that it is a name this model knows now happens in _read_aux. A model that
    reads auxiliary data names its options explicitly and takes no **kwargs, so
    an unknown one is still a loud failure rather than a silently ignored delay.
    """
    km = scb.Units.get_units(["km"])
    model = scb.SequentialRangingReal("sr", instrument, sigma=scb.ArrayWUnits(3, None))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        model.observed_measurements(
            file_name=str(trk_data_file), meas_name="2W_sranging", units=km / km,
            sc_transpnder_delay_RU=1.0,
        )


def test_partials_default_to_analytical(instrument):
    """
    With no partials_method given, the model uses the Moyer analytical partials.
    """
    model = scb.SequentialRangingReal("sr", instrument, sigma=scb.ArrayWUnits(3, None))
    assert model._partials_method == "analytical"


@pytest.mark.parametrize("method", ["analytical", "fd"])
def test_a_supported_partials_method_is_kept(instrument, method):
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None), partials_method=method
    )
    assert model._partials_method == method


def test_the_removed_ideal_partials_are_refused(instrument):
    with pytest.raises(ValueError, match="Unknown partials_method 'ideal'"):
        scb.SequentialRangingReal(
            "sr", instrument, sigma=scb.ArrayWUnits(3, None), partials_method="ideal"
        )


def test_analytical_partials_of_srp_and_maneuver_are_zero(
    instrument, computed_meas_dict, monkeypatch
):
    """
    eta_srp and dv_man move the observable only through the trajectory, which the
    filter maps through the STM, so the default partials give zero columns for them
    instead of raising.

    The light-time geometry is stubbed: these columns must not depend on it.
    """
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(computed_meas_dict, RU=[1.0, 1.0]),
        state_definition=[
            ("eta_srp", 1, "estimated", "static", None, None),
            ("dv_man1", 3, "estimated", "static", None, None),
        ],
    )
    # The model reaches SpiceManager through its own import, which may be a
    # different module tree from src.scarabaeus.
    mod = sys.modules[type(model).__module__]
    spice = mod.SpiceManager
    rel = mod.ArrayWUnits(
        np.array([1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]),
        [spice._spc_pos_units] * 3 + [spice._spc_vel_units] * 3,
    )
    monkeypatch.setattr(spice, "get_state", lambda **kwargs: rel)
    monkeypatch.setattr(model, "_ensure_partials_cache", lambda *args: None)
    monkeypatch.setattr(model, "_light_time_velocity", lambda *args: np.zeros(3))
    model._last_f_t1 = np.full(2, 7.2e9)
    epochs = [SimpleNamespace(times=SimpleNamespace(values=t)) for t in (6.6e8, 6.6e8 + 60.0)]

    rows = model.compute_partials(SimpleNamespace(name="-64", spice_id=-64), epochs)

    assert len(rows) == 2
    for row in rows:
        np.testing.assert_array_equal(row, np.zeros(4))


def _per_pass_partials_model(instrument, monkeypatch, state_definition=None):
    """
    A model with stubbed geometry and, unless a state definition is given, a
    two-pass range_bias_RU of its station.
    """
    if state_definition is None:
        state_definition = [
            ("range_bias_RU_DSS14", 2, "estimated", "parameter", instrument, None)]
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(_aux_for_write(2), RU=[1.0, 1.0]),
        state_definition=state_definition,
    )
    mod = sys.modules[type(model).__module__]
    spice = mod.SpiceManager
    rel = mod.ArrayWUnits(
        np.array([1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0]),
        [spice._spc_pos_units] * 3 + [spice._spc_vel_units] * 3,
    )
    monkeypatch.setattr(spice, "get_state", lambda **kwargs: rel)
    monkeypatch.setattr(model, "_ensure_partials_cache", lambda *args: None)
    monkeypatch.setattr(model, "_light_time_velocity", lambda *args: np.zeros(3))
    model._last_f_t1 = np.full(2, 7.2e9)
    times = (6.6e8, 6.6e8 + 3600.0)
    epochs = [SimpleNamespace(times=SimpleNamespace(values=t)) for t in times]
    return model, times, epochs


def test_analytical_partials_of_a_per_pass_bias_pick_the_epoch_pass(
    instrument, monkeypatch
):
    """
    A range_bias_RU holding one value per pass has a partial of 1 RU/RU in the
    slot of the epoch's pass only, so passes of one station get their own bias.
    """
    model, times, epochs = _per_pass_partials_model(instrument, monkeypatch)
    model.set_pass_epochs([[times[0]], [times[1]]])

    rows = model.compute_partials(SimpleNamespace(name="-64", spice_id=-64), epochs)

    np.testing.assert_allclose(np.asarray(rows), [[1.0, 0.0], [0.0, 1.0]])


def test_analytical_partials_of_a_per_pass_bias_need_the_pass_mapping(
    instrument, monkeypatch
):
    model, _, epochs = _per_pass_partials_model(instrument, monkeypatch)
    with pytest.raises(scb.Measurement.MissingPassIndexError):
        model.compute_partials(SimpleNamespace(name="-64", spice_id=-64), epochs)


def test_a_per_pass_bias_is_added_to_the_epochs_of_its_pass(instrument, monkeypatch):
    """
    The computed range carries the bias of the epoch's pass. It used to squeeze
    the whole bias into one float, which fails for more than one value.

    The light time and the ramp integration are stubbed to give a zero range, so
    the computed range is the bias alone.
    """
    kg, km = scb.Units.get_units(["kg", "km"])
    n = 3
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(
            _aux_for_write(n), RU=[1.0] * n, M=[2**20] * n, M2_num=[880] * n,
            M2_den=[749] * n, scft_transpd_delay=0.0, scft_transpd_delay_s=0.0,
        ),
        meas_bias=np.array([3.0, 5.0]),
    )
    target = scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )
    times = np.array([576000000.0, 576000600.0, 576001200.0])
    epochs = scb.EpochArray(times, sys="TDB")
    model.set_pass_epochs([times[:1], times[1:]])

    mod = sys.modules[type(model).__module__]
    monkeypatch.setattr(
        model, "_rtlt_solver",
        lambda: lambda **kwargs: (1000.0, np.zeros(15), kwargs["tt_et"] - 500.0,
                                  kwargs["tt_et"] - 1000.0),
    )
    monkeypatch.setattr(
        mod.RampTableManager, "integrate",
        staticmethod(lambda *args, **kwargs: (0.0, {"f_iq1": 7.2e9, "n_segments": 1})),
    )

    *_, meas = model.computed_measurements(target, epoch_array=epochs)

    np.testing.assert_allclose(np.ravel(meas.quantity.values), [3.0, 5.0, 5.0])


def test_residuals_are_wrapped_into_plus_minus_half_the_modulo(instrument):
    """
    O-C is the difference closest to zero, in [-M/2, M/2), with M per epoch.

    O and C are both reported modulo M, so a plain O-C of a point whose O and C
    sit on opposite sides of a wrap jumped by a whole modulus. C can also exceed
    M, since the bias is added after the modulo.
    """
    M = [1000.0, 1000.0, 1000.0, 1000.0, 2**20]
    n = len(M)
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(
            _aux_for_write(n), RU=[1.0] * n, M=M,
            scft_transpd_delay=0.0, scft_transpd_delay_s=0.0,
        ),
    )
    unitless = scb.Units.get_units("unitless")
    J2000 = scb.Frame.generate_common_frames()[0]
    observed = scb.ArrayWFrame(np.array([998.0, 2.0, 10.0, 0.0, 3.0]), unitless, J2000)
    computed = scb.ArrayWFrame(np.array([3.0, 999.0, 1005.0, 500.0, 2**20 - 2.0]),
                               unitless, J2000)

    residuals = np.ravel(model.residuals(observed, computed).quantity.values)

    np.testing.assert_allclose(residuals, [-5.0, 3.0, 5.0, -500.0, 5.0])
    assert np.all(residuals >= -np.array(M) / 2) and np.all(residuals < np.array(M) / 2)


def _electronic_delay_state(station, spacecraft, values=(0.0, 0.0, 0.0)):
    """The station's uplink and downlink delays and the spacecraft's transponder delay [s]."""
    sec = scb.Units.get_units(["sec"])
    names = ("uplink_delay_bias_DSS14", "transponder_delay_bias", "downlink_delay_bias_DSS14")
    return [(name, 1, "estimated", "parameter", owner,
             scb.ArrayWFrame(np.array([value]), sec, None))
            for name, owner, value in zip(names, (station, spacecraft, station), values)]


@pytest.mark.parametrize("apply_tracking_delays", [True, False])
def test_estimated_electronic_delays_reach_the_light_time(
    instrument, monkeypatch, apply_tracking_delays
):
    """
    The station's t1 and t3 delays add to the uplink and downlink z-height
    corrections and the spacecraft's t2 delay to the transponder delay, the
    three terms that lengthen the light time without moving t3 (Moyer 11-7).
    They apply whether or not the tracking data's own delays are.
    """
    kg, km = scb.Units.get_units(["kg", "km"])
    n = 2
    target = scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2),
        scb.ArrayWUnits(1.5, None),
    )
    model = scb.SequentialRangingReal(
        "sr", instrument, sigma=scb.ArrayWUnits(3, None),
        computed_measurements_dict=dict(
            _aux_for_write(n), RU=[1.0] * n, M=[2**20] * n, M2_num=[880] * n,
            M2_den=[749] * n, scft_transpd_delay=0.0, scft_transpd_delay_s=0.0,
        ),
        state_definition=_electronic_delay_state(instrument, target, (1e-6, 2e-6, 3e-6)),
        electronic_delays=apply_tracking_delays,
    )
    seen = []

    def solver(**kwargs):
        seen.append(dict(kwargs["electronic_delays_dict"]))
        return 1000.0, np.zeros(15), kwargs["tt_et"] - 500.0, kwargs["tt_et"] - 1000.0

    mod = sys.modules[type(model).__module__]
    monkeypatch.setattr(model, "_rtlt_solver", lambda: solver)
    monkeypatch.setattr(
        mod.RampTableManager, "integrate",
        staticmethod(lambda *args, **kwargs: (0.0, {"f_iq1": 7.2e9, "n_segments": 1})),
    )

    model.computed_measurements(
        target, epoch_array=scb.EpochArray(np.array([576000000.0, 576000600.0]), sys="TDB"))

    assert len(seen) == n
    for delays in seen:
        assert delays["ul_zheight_corr"] == pytest.approx(1e-6, rel=1e-12)
        assert delays["scft_transpd_delay_s"] == pytest.approx(2e-6, rel=1e-12)
        assert delays["dl_zheight_corr"] == pytest.approx(3e-6, rel=1e-12)
        # The time-tag delays, which also move t3, are left alone.
        assert delays["trn_tt_dly"] == 0.0 and delays["rcv_tt_dly"] == 0.0


def test_analytical_partials_of_electronic_delays_are_RU_f_T1(instrument, monkeypatch):
    """
    Each delay this station and spacecraft own lengthens the light time one for
    one, so its row is RU·f_T1 RU per second; another station's delay and another
    spacecraft's transponder delay have none.
    """
    kg, sec = scb.Units.get_units(["kg", "sec"])
    target = scb.Spacecraft("Tracked", -64, scb.ArrayWUnits(2000.0, kg))
    others = [
        ("downlink_delay_bias_DSS24", 1, "estimated", "parameter",
         scb.GroundStation("DSS-24"), scb.ArrayWFrame(np.zeros(1), sec, None)),
        ("transponder_delay_bias", 1, "estimated", "parameter",
         scb.Spacecraft("Other", -1001, scb.ArrayWUnits(2000.0, kg)),
         scb.ArrayWFrame(np.zeros(1), sec, None)),
    ]
    model, _, epochs = _per_pass_partials_model(
        instrument, monkeypatch, _electronic_delay_state(instrument, target) + others)

    rows = model.compute_partials(target, epochs)

    np.testing.assert_allclose(np.asarray(rows), [[7.2e9] * 3 + [0.0] * 2] * 2, rtol=1e-12)


# ---------------------------------------------------#
# region  Station elevation for the media partials   #
# ---------------------------------------------------#
# _get_elevation_rad feeds the troposphere mapping partials. A SPICE failure
# must not be passed off as a horizon elevation, and an elevation at or below
# the horizon must be reported rather than used silently.
def _elevation_model():
    model = scb.SequentialRangingReal.__new__(scb.SequentialRangingReal)
    model._name = "ranging"
    return model


@pytest.fixture
def elevation(monkeypatch):
    """Make SpiceManager.get_elevation_angle return, or raise, a chosen value."""
    rad = scb.Units.get_units("rad")

    def install(result):
        def fake(cls, target, et, station="DSS-14", abcorr="LT+S"):
            if isinstance(result, Exception):
                raise result
            return scb.ArrayWUnits(result, rad)
        monkeypatch.setattr(scb.SpiceManager, "get_elevation_angle", classmethod(fake))
        monkeypatch.setattr(scb.SpiceManager, "et2utc", classmethod(lambda cls, et: "UTC"))

    return install


def test_elevation_above_horizon_is_returned(elevation):
    elevation(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _elevation_model()._get_elevation_rad("SC", 0.0, "DSS-24") == pytest.approx(0.5)


def test_spice_failure_raises_instead_of_returning_horizon(elevation):
    elevation(ValueError("no TOPO frame"))
    with pytest.raises(RuntimeError, match="DSS-24_TOPO") as info:
        _elevation_model()._get_elevation_rad("SC", 0.0, "DSS-24")
    assert "no TOPO frame" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.parametrize("E", [0.0, -np.radians(3.0)])
def test_below_horizon_warns(elevation, E):
    elevation(E)
    with pytest.warns(UserWarning, match="below the horizon"):
        assert _elevation_model()._get_elevation_rad("SC", 0.0, "DSS-24") == pytest.approx(E)


# endregion                                          #
# ---------------------------------------------------#
