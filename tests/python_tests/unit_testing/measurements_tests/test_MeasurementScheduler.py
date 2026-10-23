# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================================#
#  Unit Tests for the MeasurementScheduler class   #
# ==================================================#
import datetime as dt

import numpy as np
import pytest

import src.scarabaeus as scb

MS = scb.MeasurementScheduler
kg, km, sec, minute, hr, day, deg, rad = scb.Units.get_units(["kg", "km", "sec", "min", "hr", "day", "deg", "rad"])
AWU = scb.ArrayWUnits

DAY = dict(start="2030 JAN 01 00:00:00", end="2030 JAN 02 00:00:00")


# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def sched():
    """Scheduler with a one-day default window and no target (windows only)."""
    return MS(cadence=AWU(1, minute), **DAY)


@pytest.fixture
def sc():
    return scb.Spacecraft(
        "Orbiter", -1000, scb.ArrayWUnits(2000.0, kg), scb.ArrayWUnits(1e-06, km**2), scb.ArrayWUnits(1.5, None)
    )


# endregion

# ---------------------------#
# region    Argument parsing #
# ---------------------------#


@pytest.mark.parametrize(
    "span, expected",
    [
        (AWU(600, sec), 600.0),
        (AWU(30, minute), 1800.0),
        (AWU(1.5, hr), 5400.0),
        (AWU(1, day), 86400.0),
        (AWU([2.0], hr), 7200.0),  # a one-element array is still one span
    ],
)
def test_to_seconds(span, expected):
    assert MS._to_seconds(span) == pytest.approx(expected)


@pytest.mark.parametrize(
    "span, match",
    [
        ("30min", "ArrayWUnits time"),  # strings are refused
        (600, "ArrayWUnits time"),  # so are bare seconds
        (45.0, "ArrayWUnits time"),
        (dt.timedelta(minutes=2), "ArrayWUnits time"),  # and timedeltas
        (None, "ArrayWUnits time"),
        (AWU(6, km), "time units"),  # not a time unit
        (AWU(5.0, None), "time units"),  # unitless
        (AWU([1.0, 2.0], hr), "single time span"),
        (AWU(-5.0, sec), "positive"),
        (AWU(0.0, sec), "positive"),
        (AWU(float("nan"), sec), "positive"),
    ],
)
def test_to_seconds_rejects(span, match):
    with pytest.raises(MS.InvalidScheduleError, match=match):
        MS._to_seconds(span)


def test_to_seconds_allow_zero():
    assert MS._to_seconds(AWU(0, sec), "offset", allow_zero=True) == 0.0


@pytest.mark.parametrize(
    "kwargs, label",
    [
        (dict(cadence="1min"), "cadence"),
        (dict(cadence=60), "cadence"),
        (dict(pass_gap="10min"), "pass_gap"),
        (dict(pass_gap=AWU(10, km)), "pass_gap"),
    ],
)
def test_constructor_requires_awu_time(kwargs, label):
    with pytest.raises(MS.InvalidScheduleError, match=f"'{label}'"):
        MS(**DAY, **kwargs)


@pytest.mark.parametrize(
    "kwargs, label",
    [
        (dict(cadence="10min"), "cadence"),
        (dict(cadence=600.0), "cadence"),
        (dict(pass_gap="10min"), "pass_gap"),
        (dict(pass_gap=dt.timedelta(minutes=10)), "pass_gap"),
    ],
)
def test_resolve_window_requires_awu_time(sched, kwargs, label):
    with pytest.raises(MS.InvalidScheduleError, match=f"'{label}'"):
        sched.resolve_window(**kwargs)


@pytest.mark.parametrize(
    "hours, expected",
    [
        ("08:00-12:00", ((8.0, 12.0),)),
        ("8 to 12 UTC", ((8.0, 12.0),)),
        ((8, 12), ((8.0, 12.0),)),
        (("08:30", "12:00"), ((8.5, 12.0),)),
        ("22:00-04:00", ((22.0, 4.0),)),
        ("02:00-04:00, 14:00-16:00", ((2.0, 4.0), (14.0, 16.0))),
        ("02:00-04:00; 14:00-16:00", ((2.0, 4.0), (14.0, 16.0))),
        (["02:00-04:00", "14:00-16:00"], ((2.0, 4.0), (14.0, 16.0))),
        ([(2, 4), (14, 16)], ((2.0, 4.0), (14.0, 16.0))),
        ([(2, 4), (14, 16), (20, 22)], ((2.0, 4.0), (14.0, 16.0), (20.0, 22.0))),
        (None, None),
    ],
)
def test_to_hours(hours, expected):
    assert MS._to_hours(hours) == expected


@pytest.mark.parametrize("hours", ["25:00-26:00", "08:00-08:00", "8-", "morning", "", [(2, 4, 6)], 8])
def test_to_hours_rejects(hours):
    with pytest.raises(MS.InvalidScheduleError):
        MS._to_hours(hours)


def test_to_slots():
    assert MS._to_slots(AWU(6, hr), AWU(5, minute), None) == (21600.0, 300.0, 0.0)
    assert MS._to_slots(AWU(3600, sec), AWU(600, sec), AWU(30, minute)) == (3600.0, 600.0, 1800.0)
    assert MS._to_slots(AWU(1, day), AWU(1, hr), AWU(0, sec)) == (86400.0, 3600.0, 0.0)
    assert MS._to_slots(None, None, None) == (None, None, 0.0)


@pytest.mark.parametrize(
    "pass_cadence, pass_duration, pass_offset, match",
    [
        (AWU(6, hr), None, None, "pass_duration=..."),  # pass_cadence without pass_duration
        (None, AWU(5, minute), None, "pass_cadence=..."),  # pass_duration without pass_cadence
        (None, None, AWU(1, hr), "pass_cadence=..."),  # pass_offset without pass_cadence
        (AWU(1, hr), AWU(2, hr), None, "overlap"),  # overlapping slots
        (AWU(0, hr), AWU(0, hr), None, "positive"),  # zero period
        (AWU(6, hr), AWU(5, minute), AWU(-1, hr), "non-negative"),  # negative offset
    ],
)
def test_to_slots_rejects(pass_cadence, pass_duration, pass_offset, match):
    with pytest.raises(MS.InvalidScheduleError, match=match):
        MS._to_slots(pass_cadence, pass_duration, pass_offset)


@pytest.mark.parametrize(
    "pass_cadence, pass_duration, pass_offset, label",
    [
        ("6h", AWU(5, minute), None, "pass_cadence"),  # strings are refused
        (AWU(6, hr), 300.0, None, "pass_duration"),  # so are bare seconds
        (AWU(6, hr), AWU(5, minute), dt.timedelta(hours=1), "pass_offset"),  # and timedeltas
        (AWU(6, km), AWU(5, minute), None, "pass_cadence"),  # not a time unit
        (AWU(6, hr), AWU(5.0, None), None, "pass_duration"),  # unitless
    ],
)
def test_to_slots_requires_awu_time(pass_cadence, pass_duration, pass_offset, label):
    with pytest.raises(MS.InvalidScheduleError, match=f"'{label}'"):
        MS._to_slots(pass_cadence, pass_duration, pass_offset)


def test_to_seconds_bare_seconds():
    # The static grid helpers also take a bare number as seconds.
    assert MS._to_seconds(600, bare_seconds=True) == 600.0
    assert MS._to_seconds(45.5, bare_seconds=True) == 45.5
    assert MS._to_seconds(np.float64(60.0), bare_seconds=True) == 60.0
    assert MS._to_seconds(AWU(2, minute), bare_seconds=True) == 120.0


@pytest.mark.parametrize("span", ["600", True, None, dt.timedelta(minutes=2)])
def test_to_seconds_bare_seconds_rejects(span):
    with pytest.raises(MS.InvalidScheduleError, match="or a number of seconds"):
        MS._to_seconds(span, bare_seconds=True)


@pytest.mark.parametrize(
    "mask, expected",
    [
        (10.0, 10.0),  # a bare number is degrees
        (0, 0.0),
        (AWU(30.0, deg), 30.0),
        (AWU(np.pi / 6, rad), 30.0),
        (1.0, 1.0),  # whole degrees are not ambiguous
        (2.5, 2.5),  # nor is anything above pi/2
        (AWU(0.5, deg), 0.5),  # a sub-degree mask with its units
    ],
)
def test_to_degrees(mask, expected):
    assert MS._to_degrees(mask) == pytest.approx(expected)


@pytest.mark.parametrize(
    "mask",
    [
        AWU(2.0, rad),  # 114.6 deg
        100.0,
        -5.0,
        AWU(10.0, km),  # not an angle
        "ten",
        None,
        0.17,  # 0.17 deg, or 10 deg in radians
        np.pi / 2,
    ],
)
def test_to_degrees_rejects(mask):
    with pytest.raises(MS.InvalidScheduleError, match="'elevation_mask'"):
        MS._to_degrees(mask)


def test_elevation_mask_property_is_degrees():
    for mask, expected in [(None, 10.0), (15.0, 15.0), (AWU(np.pi / 18, rad), 10.0)]:
        kwargs = {} if mask is None else dict(elevation_mask=mask)
        default = MS(**DAY, **kwargs).elevation_mask
        assert isinstance(default, AWU)
        assert default.convert_to(deg).values == pytest.approx(expected)


@pytest.mark.parametrize("mask", [AWU(2.0, rad), 95.0, AWU(10.0, km)])
def test_constructor_rejects_elevation_mask(mask):
    with pytest.raises(MS.InvalidScheduleError, match="'elevation_mask'"):
        MS(elevation_mask=mask, **DAY)


@pytest.mark.parametrize("request_name", ["visibility", "visible_epochs", "passes", "schedule"])
def test_request_rejects_elevation_mask(sched, sc, request_name):
    args = ("DSS-14", "range") if request_name == "schedule" else ("DSS-14",)
    with pytest.raises(MS.InvalidScheduleError, match="'elevation_mask'"):
        getattr(sched, request_name)(*args, elevation_mask=AWU(2.0, rad), target=sc)


def test_request_elevation_mask_reaches_station_in_degrees(sched, sc, monkeypatch):
    seen = []

    def station_visibility(self, target, epochs, elevation_mask=10.0, **kwargs):
        seen.append(elevation_mask)
        return np.ones(epochs.size, dtype=bool)

    monkeypatch.setattr(scb.GroundStation, "station_visibility", station_visibility)
    station = scb.GroundStation("DSS-14")
    sched.visibility(station, elevation_mask=AWU(np.pi / 6, rad), target=sc)
    sched.visible_epochs(station, elevation_mask=20.0, target=sc)
    sched.passes(station, target=sc)  # scheduler default
    MS.visibility_on(station, sc, sched.epochs, elevation_mask=AWU(15.0, deg))
    assert seen == pytest.approx([30.0, 20.0, 10.0, 15.0])


def test_sub_degree_elevation_mask_reaches_station(sc, monkeypatch):
    # Resolved to a bare 0.5 and resolved again, it would be refused as ambiguous
    seen = []

    def station_visibility(self, target, epochs, elevation_mask=10.0, **kwargs):
        seen.append(elevation_mask)
        return np.ones(epochs.size, dtype=bool)

    monkeypatch.setattr(scb.GroundStation, "station_visibility", station_visibility)
    station = scb.GroundStation("DSS-14")
    sched = MS(cadence=AWU(1, minute), elevation_mask=AWU(0.5, deg), **DAY)
    sched.visibility(station, target=sc)
    sched.visible_epochs(station, target=sc)
    sched.passes(station, elevation_mask=AWU(0.25, deg), target=sc)
    assert seen == pytest.approx([0.5, 0.5, 0.25])


# endregion

# -----------------------------#
# region    Recurring windows  #
# -----------------------------#


def test_cadence_units(sched):
    assert sched.resolve_window(cadence=AWU(10, minute)).size == sched.resolve_window(cadence=AWU(600, sec)).size == 144
    assert sched.cadence.convert_to(sec).values == pytest.approx(60.0)


def test_slot_grid(sched):
    """pass_cadence/pass_duration: 4 slots of 5 one-minute epochs, one slot per pass."""
    window = sched.resolve_window(pass_cadence=AWU(6, hr), pass_duration=AWU(5, minute))
    et = np.atleast_1d(window.epochs.et)
    assert window.size == 20
    np.testing.assert_allclose(et.reshape(4, 5)[:, 0] - window.start_et, [0.0, 21600.0, 43200.0, 64800.0])
    np.testing.assert_allclose(np.diff(et.reshape(4, 5), axis=1), 60.0)

    mask, passes = window.select(np.ones(window.size, dtype=bool))
    assert len(passes) == 4 and mask.all()
    # passes_per_day still counts the slots of each UTC day
    _, first = sched.resolve_window(
        pass_cadence=AWU(6, hr), pass_duration=AWU(5, minute), passes_per_day=1
    ).select(np.ones(20, dtype=bool))
    assert len(first) == 1


def test_slot_grid_period_not_dividing_a_day(sched):
    """One image every 5 h, first one 1 h after start: 01, 06, 11, 16, 21 h."""
    window = sched.resolve_window(
        cadence=AWU(10, minute), pass_cadence=AWU(5, hr), pass_duration=AWU(10, minute), pass_offset=AWU(1, hr)
    )
    hours = (np.atleast_1d(window.epochs.et) - window.start_et) / 3600.0
    np.testing.assert_allclose(hours, [1.0, 6.0, 11.0, 16.0, 21.0])


def test_slot_duration_not_multiple_of_cadence(sched):
    """A 25 min slot at a 10 min cadence holds the epochs at +0, +10 and +20 min."""
    window = sched.resolve_window(cadence=AWU(10, minute), pass_cadence=AWU(1, hr), pass_duration=AWU(25, minute))
    assert window.size == 3 * 24


def test_slot_clipped_at_end():
    """The last slot stops at the (excluded) end of the window."""
    window = MS().resolve_window(
        start="2030 JAN 01 00:00:00",
        end="2030 JAN 01 22:00:00",
        cadence=AWU(1, hr),
        pass_cadence=AWU(7, hr),
        pass_duration=AWU(2, hr),
    )
    hours = (np.atleast_1d(window.epochs.et) - window.start_et) / 3600.0
    np.testing.assert_allclose(hours, [0.0, 1.0, 7.0, 8.0, 14.0, 15.0, 21.0])


def test_multi_range_hours(sched):
    window = sched.resolve_window(cadence=AWU(1, hr), hours="02:00-04:00, 14:00-16:00")
    hours = (np.atleast_1d(window.epochs.et) - window.start_et) / 3600.0
    np.testing.assert_allclose(hours, [2.0, 3.0, 14.0, 15.0])
    assert "02:00-04:00 & 14:00-16:00 h UTC" in window.recurrence


def test_multi_range_hours_with_wrap(sched):
    window = sched.resolve_window(cadence=AWU(1, hr), hours=[(22, 2), (10, 12)])
    hours = (np.atleast_1d(window.epochs.et) - window.start_et) / 3600.0
    np.testing.assert_allclose(hours, [0.0, 1.0, 10.0, 11.0, 22.0, 23.0])


def test_slots_filtered_by_hours(sched):
    """Slots at 00, 06, 12, 18 h of 1 h at 30 min; hours keeps the morning ones."""
    window = sched.resolve_window(
        cadence=AWU(30, minute), pass_cadence=AWU(6, hr), pass_duration=AWU(1, hr), hours="00:00-12:00"
    )
    hours = (np.atleast_1d(window.epochs.et) - window.start_et) / 3600.0
    np.testing.assert_allclose(hours, [0.0, 0.5, 6.0, 6.5])


def test_recurrence_text(sched):
    window = sched.resolve_window(
        pass_cadence=AWU(6, hr),
        pass_duration=AWU(5, minute),
        pass_offset=AWU(30, minute),
        weekdays="Tue",
        passes_per_day=2,
    )
    assert window.recurrence == "5min every 6h from start + 30min, on Tue, 2 pass(es)/day"
    assert "5min every 6h" in str(window) and "(20 epochs)" in str(window)
    assert sched.resolve_window().recurrence == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        # slots at 02:00, 08:00, ... miss the hours
        dict(pass_cadence=AWU(6, hr), pass_duration=AWU(30, minute), pass_offset=AWU(2, hr), hours="00:00-01:00"),
        # first slot after the end
        dict(pass_cadence=AWU(6, hr), pass_duration=AWU(30, minute), pass_offset=AWU(2, day)),
        dict(hours="02:00-03:00", weekdays="Mon"),  # 2030 JAN 01 is a Tuesday
    ],
)
def test_empty_recurrence_raises(sched, kwargs):
    with pytest.raises(MS.InvalidScheduleError):
        sched.resolve_window(**kwargs)


def test_pass_gap(sched):
    assert sched.resolve_window(pass_gap=AWU(10, minute)).pass_gap_sec == 600.0
    assert sched.resolve_window().pass_gap_sec == 90.0  # 1.5 cadences
    assert sched.pass_gap is None
    gap = MS(pass_gap=AWU(1, hr)).pass_gap
    assert isinstance(gap, AWU) and gap.convert_to(sec).values == pytest.approx(3600.0)
    window_gap = sched.resolve_window(pass_gap=AWU(10, minute)).pass_gap
    assert isinstance(window_gap, AWU) and window_gap.convert_to(sec).values == pytest.approx(600.0)


def test_build_epochs_awu_matches_bare_seconds():
    bare = MS.build_epochs(0.0, 7200.0, 600.0)
    typed = MS.build_epochs(scb.EpochArray(0.0, sys="TDB"), scb.EpochArray(7200.0, sys="TDB"), AWU(10, minute))
    assert typed.size == 12
    np.testing.assert_allclose(typed.et, bare.et)

    bare = MS.build_epochs(0.0, 86400.0, 60.0, 21600.0, 300.0, 1800.0)
    typed = MS.build_epochs(
        0.0, 86400.0, AWU(1, minute), pass_cadence=AWU(6, hr), pass_duration=AWU(5, minute), pass_offset=AWU(30, minute)
    )
    assert typed.size == 4 * 5
    assert typed.et[0] == pytest.approx(1800.0)
    np.testing.assert_allclose(typed.et, bare.et)


def test_build_epochs_epoch_array_bounds():
    # An EpochArray start is taken at its first epoch, an end at its last.
    span = scb.EpochArray(np.array([0.0, 1800.0, 3600.0]), sys="TDB")
    np.testing.assert_allclose(MS.build_epochs(span, span, AWU(20, minute)).et, [0.0, 1200.0, 2400.0])


@pytest.mark.parametrize(
    "kwargs, label",
    [
        (dict(cadence=AWU(1, km)), "cadence"),  # not a time unit
        (dict(cadence="1min"), "cadence"),
        (dict(cadence=0.0), "cadence"),
        (dict(cadence=60.0, pass_cadence=AWU(6, hr)), "pass_duration"),  # pass_cadence without pass_duration
        (dict(cadence=60.0, pass_cadence=AWU(6, km), pass_duration=300.0), "pass_cadence"),
        (dict(cadence=60.0, pass_cadence=AWU(6, hr), pass_duration=300.0, pass_offset=-1.0), "pass_offset"),
    ],
)
def test_build_epochs_rejects(kwargs, label):
    with pytest.raises(MS.InvalidScheduleError, match=f"'{label}'"):
        MS.build_epochs(0.0, 86400.0, **kwargs)


def test_split_passes_awu_matches_bare_seconds():
    epochs = scb.EpochArray(np.array([0.0, 60.0, 120.0, 3600.0, 3660.0]), sys="TDB")
    bare = MS.split_passes(epochs, 600.0)
    typed = MS.split_passes(epochs, AWU(10, minute))
    assert [p.size for p in typed] == [p.size for p in bare] == [3, 2]
    for p_typed, p_bare in zip(typed, bare):
        np.testing.assert_allclose(p_typed.et, p_bare.et)
    assert len(MS.split_passes(epochs, AWU(2, hr))) == 1


@pytest.mark.parametrize("gap", [AWU(10, km), "10min", 0.0, AWU([1.0, 2.0], minute)])
def test_split_passes_rejects(gap):
    epochs = scb.EpochArray(np.array([0.0, 60.0]), sys="TDB")
    with pytest.raises(MS.InvalidScheduleError, match="'gap'"):
        MS.split_passes(epochs, gap)


# endregion

# ------------------------------#
# region    Measurement types   #
# ------------------------------#


@pytest.mark.parametrize(
    "measurement_type, expected",
    [
        ("RangeIdeal", "RangeIdeal"),
        ("rangeideal", "RangeIdeal"),
        ("Doppler_Ideal", "DopplerIdeal"),
        (scb.DopplerIdeal, "DopplerIdeal"),
        (scb.CentroidingIdeal, "CentroidingIdeal"),
        ("range", "RangeIdeal"),
        ("Range-Rate", "RangeRateIdeal"),
        ("range_rate", "RangeRateIdeal"),
        ("doppler", "DopplerIdeal"),
        ("DOR", "DiffOneWayRangeIdeal"),
        ("opnav", "CentroidingIdeal"),
        ("centroiding", "CentroidingIdeal"),
        ("RA_DEC", "AngularIdeal"),
    ],
)
def test_resolve_measurement_type(measurement_type, expected):
    assert MS.resolve_measurement_type(measurement_type) == expected


@pytest.mark.parametrize("measurement_type", ["DopplerReal", "doppler_real", scb.DopplerReal, "SequentialRangingReal", "seq_ranging"])
def test_real_models_refused(measurement_type):
    with pytest.raises(MS.RealModelNotSupportedError, match="Ideal"):
        MS.resolve_measurement_type(measurement_type)


def test_unknown_measurement_type():
    with pytest.raises(MS.UnknownMeasurementTypeError, match="RangeIdeal"):
        MS.resolve_measurement_type("lidar")


def test_every_model_class_has_units_and_family():
    for name in MS.MODEL_CLASSES:
        assert name in MS.OBSERVATION_UNITS
        assert (name in MS.RADIOMETRIC_TYPES) != (name in MS.OPTICAL_TYPES)
    assert set(MS.MEASUREMENT_TYPE_ALIASES.values()) <= set(MS.MODEL_CLASSES)


@pytest.mark.parametrize("measurement_type", ["range", "RangeIdeal"])
def test_build_model_names_by_class(measurement_type):
    model = MS.build_model(measurement_type, "DSS-14")
    assert type(model).__name__ == "RangeIdeal"
    assert model.name == "DSS-14 RangeIdeal"
    assert MS._slug(model.name) == "dss_14_range_ideal"


def test_build_model_real_refused():
    with pytest.raises(MS.RealModelNotSupportedError):
        MS.build_model("DopplerReal", "DSS-14")


def test_build_model_observer_checks(sc):
    with pytest.raises(MS.ObserverTypeError):
        MS.build_model("RangeIdeal", sc)
    with pytest.raises(MS.ObserverTypeError):
        MS.build_model("CentroidingIdeal", "DSS-14")


def test_noisy_without_sigma_is_the_models_missing_sigma_error(sched, sc):
    """The scheduler raises the measurement models' own MissingSigmaError."""
    assert MS.MissingSigmaError is scb.Measurement.MissingSigmaError
    with pytest.raises(scb.Measurement.MissingSigmaError):
        sched.schedule("DSS-14", "RangeIdeal", target=sc, noisy=True)


def test_slug():
    assert MS._slug("NavCam CentroidingIdeal") == "nav_cam_centroiding_ideal"
    assert MS._slug("DSS-43 range_rate phase 2") == "dss_43_range_rate_phase_2"


# endregion

# ------------------------------#
# region    Edge tolerance      #
# ------------------------------#


def test_the_edge_tolerance_starts_at_the_class_default(sched):
    assert MS.EDGE_TOLERANCE.convert_to(sec).values == pytest.approx(0.01)
    assert isinstance(sched.edge_tolerance, AWU)
    assert sched.edge_tolerance.convert_to(sec).values == pytest.approx(0.01)
    assert sched.window.edge_tolerance_sec == pytest.approx(0.01)
    assert MS.Window(0.0, 600.0, 60.0, 90.0).edge_tolerance_sec == pytest.approx(0.01)


def test_build_epochs_takes_the_edge_tolerance():
    # 600 s is 5 ms short of the end: inside the default slack, so it counts as the end
    assert MS.build_epochs(0.0, 600.005, 60.0).size == 10
    assert MS.build_epochs(0.0, 600.005, 60.0, edge_tolerance=0.0).size == 11
    assert MS.build_epochs(0.0, 600.005, AWU(1, minute), edge_tolerance=AWU(0, sec)).size == 11
    assert MS.build_epochs(0.0, 600.005, 60.0, edge_tolerance=AWU(1e-3, sec)).size == 11


def test_the_edge_tolerance_rebuilds_the_default_window():
    sched = MS(cadence=AWU(1, minute), start="2030 JAN 01 00:00:00", end="2030 JAN 01 00:10:00.005")
    assert sched.window.size == 10
    sched.edge_tolerance = AWU(0, sec)
    assert sched.edge_tolerance.convert_to(sec).values == 0.0
    assert sched.window.edge_tolerance_sec == 0.0 and sched.window.size == 11
    assert sched.resolve_window().edge_tolerance_sec == 0.0
    sched.edge_tolerance = AWU(0.5, minute)
    assert sched.window.edge_tolerance_sec == pytest.approx(30.0)
    assert MS(cadence=AWU(1, minute), **DAY).edge_tolerance.convert_to(sec).values == pytest.approx(0.01)


def test_the_edge_tolerance_without_a_default_window():
    sched = MS(cadence=AWU(1, minute))
    assert sched.window is None
    sched.edge_tolerance = AWU(1, sec)
    assert sched.window is None
    assert sched.resolve_window(**DAY).edge_tolerance_sec == pytest.approx(1.0)


@pytest.mark.parametrize("value", [AWU(-1, sec), 0.01, AWU(1, km), "1s", AWU([1.0, 2.0], sec), AWU(np.nan, sec)])
def test_a_bad_edge_tolerance_is_refused(sched, value):
    with pytest.raises(MS.InvalidScheduleError, match="'edge_tolerance'"):
        sched.edge_tolerance = value
    assert sched.edge_tolerance.convert_to(sec).values == pytest.approx(0.01)


# endregion
