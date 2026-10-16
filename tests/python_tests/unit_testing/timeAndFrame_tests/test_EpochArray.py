# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the EpochArray class  #
# =======================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.timeAndFrame.EpochArray import EpochArray
import pytest
import numpy as np
from fractions import Fraction

km, kg, sec, hr, day = scb.Units.get_units(["km", "kg", "sec", "hr", "day"])
J2000 = scb.Frame("J2000")


# --------------------#
# region    Fixtures #
# --------------------#
# setup
@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB")


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(epoch):
    """
    Verifies that object is constructed correctly.

    """
    pytest.skip(
        "See system_testing/integration_testing/test_times.py for other EpochArray tests."
    )


def test_split_fractional():
    """
    Verifies splitFractional operation.

    """
    fractional = 123.456
    split = EpochArray._split_fractional_part(fractional)
    assert split[0] == pytest.approx(123.0, 1e-6)
    assert split[1] == pytest.approx(0.456, 1e-6)


class TestTwoPartEpoch:
    """(high, low) epoch helpers, checked against exact rational arithmetic."""

    @pytest.mark.parametrize(
        "a, b",
        [(8.4e8 + 0.123456789, 32.184001657), (1.0, 1e-17), (-3.5, 3.5), (8.4e8, -1234.56789)],
    )
    def test_two_sum_is_exact(self, a, b):
        s, e = EpochArray._two_sum(a, b)
        assert s == a + b
        assert Fraction(s) + Fraction(e) == Fraction(a) + Fraction(b)

    def test_add_duration_keeps_the_rounding(self):
        """A plain float sum near 8.4e8 s loses up to ~60 ns; the pair keeps it to < 1e-20 s."""
        rng = np.random.default_rng(0)
        worst_float = 0.0
        for _ in range(1000):
            t_h = 8.4e8 + rng.uniform(0.0, 86400.0)
            t_l = rng.uniform(-0.5, 0.5) * np.spacing(t_h)
            dt = rng.uniform(-3000.0, 3000.0)
            exact = Fraction(t_h) + Fraction(t_l) + Fraction(dt)

            h, l = EpochArray._add_duration(t_h, t_l, dt)
            assert abs(float(Fraction(h) + Fraction(l) - exact)) < 1e-20
            assert abs(l) <= np.spacing(h) / 2  # h is the nearest float
            worst_float = max(worst_float, abs(float(Fraction(t_h + dt) - exact)))
        assert worst_float > 1e-8


class TestSystemValidation:
    def test_invalid_system(self):
        """
        Verifies that constructing an EpochArray with an unrecognized time
        system raises a ValueError.
        """
        with pytest.raises(ValueError, match="Received unknown time system"):
            scb.EpochArray("2000 JAN 01 00:00:00.000", "NOT_A_SYSTEM")

    def test_sclk_without_sc_id(self):
        """
        Verifies that constructing an EpochArray with system="SCLK" but no
        associated spacecraft ID raises an error.
        """
        with pytest.raises(Exception, match="Cannot define the requested epoch in SCLK"):
            scb.EpochArray(0.0, "SCLK")


class TestTimesProperty:
    @pytest.mark.parametrize(
        "representation, expected_substring",
        [
            ("DOY", "1999-365"),
            ("ISOC", "1999-12-31"),
            ("ISOD", "1999-365"),
            ("JDTDB", "2451544.5"),
        ],
    )
    def test_representations(self, epoch, representation, expected_substring):
        """
        Verifies that the DOY, ISOC, ISOD, and JDTDB representations
        return the correctly formatted value for a non-SCLK epoch.
        """
        epoch._representation = representation
        assert expected_substring in str(epoch.times)

    @pytest.mark.parametrize(
        "representation",
        ["CAL", "DOY", "ISOC", "ISOD", "JDTDB", "JED", "JDTDT", "JDUTC"],
    )
    def test_sclk_representation_errors(self, epoch, representation):
        """
        Verifies that requesting any non-AWU/NUM representation for an
        SCLK-system epoch raises the internal SCLKError, since SCLK ticks
        can't be converted to those formats. A real SCLK EpochArray can't
        be constructed in this environment (no SCLK kernel in the locked
        generic kernel set, confirmed via a diagnostic run), so [system]
        is set directly to exercise just this branch.
        """
        epoch._system = "SCLK"
        epoch._representation = representation
        with pytest.raises(Exception, match="Cannot convert from SCLK ticks"):
            epoch.times

    def test_sclk_awu_attempts_conversion(self, epoch):
        """
        Verifies that requesting the AWU representation for an SCLK-system
        epoch reaches the tick-to-time conversion call rather than being
        blocked by an early representation check (unlike CAL/DOY/etc.). A
        real SCLK EpochArray can't be constructed in this environment (no
        SCLK kernel in the locked generic kernel set), so this only
        confirms the conversion is attempted, not a successful return.
        """
        epoch._system = "SCLK"
        epoch._sc_id = -64
        epoch._representation = "AWU"
        with pytest.raises(Exception):
            epoch.times

    @pytest.mark.parametrize(
        "system, times, sc_id",
        [
            ("TT", "2000 JAN 01 00:00:00.000", None),
            ("TT", ["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], None),
            ("SCLK", "2000 JAN 01 00:00:00.000", -64),
            ("SCLK", ["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], -64),
        ],
        ids=[
            "Non-SCLK Scalar",
            "Non-SCLK Array",
            "SCLK Scalar Attempts Conversion",
            "SCLK Array Attempts Conversion",
        ],
    )
    def test_in_sys(self, system, times, sc_id):
        """
        Verifies that _times_in_sys converts a non-SCLK system correctly
        for both scalar and array epochs, and that it reaches the real
        SCLK tick-to-time conversion call (spice.sce2c) for both scalar
        and array SCLK epochs. A real SCLK EpochArray can't be constructed
        in this environment (no SCLK kernel in the locked generic kernel
        set), so the SCLK cases only confirm the conversion is attempted.
        """
        epoch = scb.EpochArray(times, "TDB")
        epoch._system = system
        epoch._sc_id = sc_id
        epoch._representation = "AWU"

        if system == "SCLK":
            with pytest.raises(Exception):
                epoch.times
        else:
            result = epoch.times
            assert result.values is not None


@pytest.mark.parametrize(
    "args, kwargs, error, match",
    [
        (
            ("2000 JAN 01 00:00:00.000", "UTC"),
            {"rep": "NUM"},
            ValueError,
            "No numerical representation of UTC",
        ),
        (
            ("2000 JAN 01 00:00:00.000", "TT"),
            {},
            TypeError,
            "String input not supported for",
        ),
        (
            (1000.0, "SCLK"),
            {"sc_id": -64},
            Exception,
            None,
        ),
        (
            ([None], "TDB"),
            {},
            TypeError,
            "Unsupported time input type",
        ),
    ],
    ids=[
        "UTC + NUM Representation",
        "String Input Unsupported System",
        "SCLK Attempts Conversion",
        "Unsupported Element Type",
    ],
)
def test_validate_and_set_to_et_errors(args, kwargs, error, match):
    """
    Verifies that constructing an EpochArray raises the appropriate
    exception for: UTC combined with NUM representation, a string
    time value for a system that doesn't support string input, a
    time element that is neither a string nor a float/int, and that a
    numerical SCLK time reaches the real tick-to-time conversion call.
    A real SCLK EpochArray can't be constructed in this environment
    (no SCLK kernel in the locked generic kernel set), so that case
    only confirms the conversion is attempted, not a successful
    return.
    """
    with pytest.raises(error, match=match):
        scb.EpochArray(*args, **kwargs)


class TestRepValidation:
    @pytest.mark.parametrize(
        "rep, error, match",
        [
            ("NOT_A_REP", Exception, "Unrecognized time representation"),
            (123, TypeError, "Received unsupported representation input"),
        ],
        ids=["Unrecognized Representation", "Non-String Representation"],
    )
    def test_input_errors(self, rep, error, match):
        """
        Verifies that constructing an EpochArray with an unrecognized
        representation or a non-string representation raises the
        appropriate exception.
        """
        with pytest.raises(error, match=match):
            scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB", rep=rep)

    @pytest.mark.parametrize(
        "spice_error, expected_error, match",
        [
            ("SPICE(MISSINGTIMEINFO): no LSK loaded", Exception, "leapsecond kernel"),
            ("SPICE(SOMEOTHERERROR): unrelated", Exception, "SOMEOTHERERROR"),
        ],
        ids=["Missing LSK", "Other et2utc Error"],
    )
    def test_lsk_check_errors(self, mocker, spice_error, expected_error, match):
        """
        Verifies that _validate_and_set_rep raises a descriptive error when
        its LSK sanity check fails due to a missing leapsecond kernel, and
        re-raises the original exception for any other failure. LSK is
        always loaded in this environment, so spice.et2utc is mocked to
        force both branches.
        """
        mocker.patch("spiceypy.et2utc", side_effect=Exception(spice_error))
        with pytest.raises(expected_error, match=match):
            scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB", rep="CAL")


def test_properties():
    """
    Verifies that the prec, disp_len, and sc_id properties return the
    values given at construction.
    """
    e = scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB", prec=10, disp_len=20, sc_id=-64)
    assert e.prec == 10
    assert e.disp_len == 20
    assert e.sc_id == -64


class TestItemAccess:
    def test_getitem_scalar_raises(self, epoch):
        """
        Verifies that indexing a 0-dimensional (scalar) EpochArray raises
        an IndexError.
        """
        with pytest.raises(IndexError, match="too many indices"):
            epoch[0]

    def test_setitem_raises(self, epoch):
        """
        Verifies that item assignment raises the intended "not supported"
        exception (rather than a generic Python TypeError about argument
        count, which happened before __setitem__'s missing [value]
        parameter was fixed).
        """
        with pytest.raises(Exception, match="__setitem__ currently not supported"):
            epoch[0] = 5


class TestArithmeticErrors:
    @pytest.mark.parametrize(
        "other, error, match",
        [
            (awu(5.0, kg), ValueError, "Cannot add an ArrayWUnits with"),
            (5, TypeError, "unsupported operand type"),
        ],
        ids=["Non-Time Dimension", "Non-ArrayWUnits Operand"],
    )
    def test_add_errors(self, epoch, other, error, match):
        """
        Verifies that adding an ArrayWUnits with a non-Time dimension
        raises a ValueError, and adding a non-ArrayWUnits object raises a
        TypeError.
        """
        with pytest.raises(error, match=match):
            epoch + other

    @pytest.mark.parametrize(
        "other, error, match",
        [
            (awu(5.0, kg), ValueError, "Cannot subtract an ArrayWUnits with"),
            ("not valid", TypeError, "unsupported operand type"),
        ],
        ids=["Non-Time Dimension", "Unsupported Operand Type"],
    )
    def test_sub_errors(self, epoch, other, error, match):
        """
        Verifies that subtracting an ArrayWUnits with a non-Time dimension
        raises a ValueError, and subtracting an unsupported type (neither
        EpochArray nor ArrayWUnits) raises a TypeError.
        """
        with pytest.raises(error, match=match):
            epoch - other

    @pytest.mark.parametrize(
        "op",
        [
            lambda e: e * 2,
            lambda e: 2 * e,
            lambda e: e / 2,
            lambda e: 2 / e,
            lambda e: e**2,
        ],
        ids=["__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__pow__"],
    )
    def test_unsupported_operators(self, epoch, op):
        """
        Verifies that multiplication, reverse multiplication, division,
        reverse division, and exponentiation are all undefined for
        EpochArray and raise an ArithmeticError.
        """
        with pytest.raises(ArithmeticError):
            op(epoch)


class TestEquality:
    def test_non_epocharray_error(self, epoch):
        """
        Verifies that comparing an EpochArray to a non-EpochArray object
        raises a TypeError.
        """
        with pytest.raises(TypeError, match="Equality is not defined for EpochArray"):
            epoch == "not an epoch"

    @pytest.mark.parametrize(
        "other_time, other_system, expected",
        [
            ("2000 JAN 01 00:00:00.000", "TDB", True),
            ("2000 JAN 02 00:00:00.000", "TDB", False),
        ],
        ids=["Same Time and System", "Different Time"],
    )
    def test_eq_and_ne(self, epoch, other_time, other_system, expected):
        """
        Verifies that two EpochArrays compare equal only when both their
        times and systems match, and that __ne__ is the exact opposite.
        """
        other = scb.EpochArray(other_time, other_system)
        assert (epoch == other) == expected
        assert (epoch != other) == (not expected)

    def test_different_system(self, epoch):
        """
        Verifies that two EpochArrays with the same displayed time but
        different systems compare unequal. Uses ET, which is numerically
        synonymous with TDB (no conversion applied), so the two epochs'
        displayed times match exactly and only the system differs --
        isolating this branch from the earlier time-mismatch check.
        """
        other = scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB")
        other._system = "ET"
        assert (epoch == other) is False
        assert (epoch != other) is True


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ValueError),
        ({"sys": "TT"}, ("TT", "AWU")),
        ({"rep": "CAL"}, ("TDB", "CAL")),
    ],
    ids=["No Args Error", "System Only", "Representation Only"],
)
def test_to(epoch, kwargs, expected):
    """
    Verifies that to() raises a ValueError when given neither a new
    system nor representation, keeps the current system when only a
    new representation is given, and keeps the current representation
    when only a new system is given.
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected, match="Received neither a new time system"):
            epoch.to(**kwargs)
    else:
        result = epoch.to(**kwargs)
        assert (result.system, result._representation) == expected


class TestDictConversion:
    def test_to_dict(self, epoch):
        """
        Verifies that to_dict serializes the internal ET value, shape,
        system, representation, and spacecraft ID into a plain dict.
        """
        assert epoch.to_dict() == {
            "et": [-43200.0],
            "shape": [],
            "system": "TDB",
            "representation": "AWU",
            "sc_id": None,
        }

    @pytest.mark.parametrize(
        "times, override_system",
        [
            ("2000 JAN 01 00:00:00.000", None),
            (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], None),
            ("2000 JAN 01 00:00:00.000", "TT"),
        ],
        ids=["Scalar Roundtrip", "Array Roundtrip", "Non-ET System Restored"],
    )
    def test_from_dict(self, times, override_system):
        """
        Verifies that from_dict reconstructs an EpochArray from a dict
        produced by to_dict, for scalar and array shapes, and correctly
        restores a non-ET display system.
        """
        original = scb.EpochArray(times, "TDB")
        d = original.to_dict()
        if override_system:
            d["system"] = override_system

        reconstructed = scb.EpochArray.from_dict(d)

        assert np.array_equal(reconstructed._times, original._times)
        assert reconstructed.system == (override_system or "TDB")


@pytest.mark.parametrize(
    "epoch_times, units, expected",
    [
        (["2000 JAN 01 00:00:00.000", "2000 JAN 01 00:00:00.000"], None, "non_chron"),
        ("2000 JAN 01 00:00:00.000", None, "scalar"),
        (["2000 JAN 02 00:00:00.000", "2000 JAN 01 00:00:00.000"], None, "non_chron"),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], 123, TypeError),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], kg, ValueError),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], sec, 86400.0),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], hr, 24.0),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], "hr", 24.0),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], None, 86400.0),
    ],
    ids=[
        "Identical Times Non-Chronological",
        "Scalar Epoch Error",
        "Reversed Times Non-Chronological",
        "Invalid Units Type",
        "Wrong Dimension",
        "Units Object (sec)",
        "Units Object (hr)",
        "String Units",
        "Default Units",
    ],
)
def test_duration(epoch_times, units, expected):
    """
    Verifies that duration raises for a scalar epoch, a
    non-chronological epoch, an invalid units type, or a non-time
    units dimension; and correctly computes elapsed time in seconds
    (default and explicit) and in an explicitly requested unit given
    as either a Units object or a string.
    """
    e = scb.EpochArray(epoch_times, "TDB")

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            e.duration(units) if units is not None else e.duration()
    elif expected == "scalar":
        with pytest.raises(Exception, match="Cannot define duration for scalar"):
            e.duration()
    elif expected == "non_chron":
        with pytest.raises(Exception, match="Cannot define duration for non-chronological"):
            e.duration()
    else:
        result = e.duration() if units is None else e.duration(units)
        assert result.values == pytest.approx(expected)


@pytest.mark.parametrize(
    "epoch_times, precision, expected",
    [
        ("2000 JAN 01 00:00:00.000", 13, "success"),
        ("2000 JAN 01 00:00:00.000", 5, "success"),
        ("2000 JAN 01 00:00:00.000", 1, "truncation"),
        (["2000 JAN 01 00:00:00.000", "2000 JAN 02 00:00:00.000"], 13, ValueError),
    ],
    ids=["Default Precision", "Custom Precision", "Truncation", "Array Error"],
)
def test_enforce_precision(epoch_times, precision, expected):
    """
    Verifies that enforce_precision formats the internal ET value as a
    fixed-length decimal string with the requested number of digits
    after the decimal point (padding when the natural decimal is
    shorter, truncating/rounding when it's longer), and raises for a
    non-scalar EpochArray.
    """
    e = scb.EpochArray(epoch_times, "TDB")

    if expected == "success":
        result = e.enforce_precision(precision=precision)
        _, decimal_part = result.split(".")
        # the padding loop counts the leading "." towards [precision], so the
        # actual digit count after the decimal point is precision - 1
        assert len(decimal_part) == precision - 1
        assert float(result) == pytest.approx(e._times)
    elif expected == "truncation":
        result = e.enforce_precision(precision=precision)
        assert float(result) == pytest.approx(e._times, abs=1.0)
    else:
        with pytest.raises(expected, match="only defined for scalar EpochArray"):
            e.enforce_precision(precision=precision)


# raw float TDB values (not strings) so construction needs no SPICE call and
# is therefore safe at module/collection time, before kernels are loaded by
# the module-scoped `setup` fixture in conftest.py
_interval_start = scb.EpochArray(-43200.0, "TDB")
_interval_end = scb.EpochArray(43200.0, "TDB")


class TestInterval:
    @pytest.mark.parametrize(
        "start, end, kwargs, error, match",
        [
            (_interval_start, _interval_end, {"dt": 5}, TypeError, "Received invalid dt value"),
            (_interval_start, _interval_end, {"n_epochs": "5"}, TypeError, "Received invalid n_epochs value"),
            (
                _interval_start, _interval_end,
                {"dt": awu(1, hr), "n_epochs": 5},
                ValueError, "Cannot define interval in terms of both",
            ),
            (_interval_start, _interval_end, {"dt": awu(-1, hr)}, ValueError, "time step <= 0"),
            (_interval_start, _interval_end, {"dt": awu(0, hr)}, ValueError, "time step <= 0"),
            (_interval_start, _interval_end, {"n_epochs": 0}, ValueError, "0 or less epochs"),
            (_interval_start, _interval_end, {"n_epochs": -3}, ValueError, "0 or less epochs"),
            ([1, 2, 3], _interval_end, {"dt": awu(1, hr)}, TypeError, "Received invalid start value"),
            (_interval_start, [1, 2, 3], {"dt": awu(1, hr)}, TypeError, "Received invalid end value"),
            (0.0, 86400.0, {"dt": awu(1, hr)}, ValueError, "Neither start nor end contain time system"),
        ],
        ids=[
            "Invalid dt Type",
            "Invalid n_epochs Type",
            "Both dt And n_epochs",
            "Negative dt",
            "Zero dt",
            "Zero n_epochs",
            "Negative n_epochs",
            "Invalid Start Type",
            "Invalid End Type",
            "No System Information",
        ],
    )
    def test_errors(self, start, end, kwargs, error, match):
        """
        Verifies that interval() raises the appropriate exception for an
        invalid dt/n_epochs type, both or neither of dt/n_epochs given, a
        non-positive dt or n_epochs, an invalid start/end bound type, and
        bounds that carry no time system information.
        """
        with pytest.raises(error, match=match):
            scb.EpochArray.interval(start, end, **kwargs)

    def test_missing_dt_and_n_epochs_error(self):
        """
        Verifies that interval() raises when given neither dt nor
        n_epochs (a plain Python TypeError from passing None as np's
        array-length argument, since this combination isn't explicitly
        validated).
        """
        with pytest.raises(TypeError):
            scb.EpochArray.interval(_interval_start, _interval_end)

    @pytest.mark.parametrize(
        "start, end, kwargs, expected_system, expected_size",
        [
            (_interval_start, _interval_end, {"dt": awu(6, hr)}, "TDB", 4),
            (_interval_start, _interval_end, {"n_epochs": 5}, "TDB", 5),
            (_interval_start._times, _interval_end, {"dt": awu(12, hr)}, "TDB", 2),
            (_interval_start, _interval_end._times, {"dt": awu(12, hr)}, "TDB", 2),
            (_interval_start, _interval_end, {"dt": awu(12, hr), "sys": "TT"}, "TT", 2),
            (
                "2000 JAN 01 00:00:00.000",
                "2000 JAN 02 00:00:00.000",
                {"dt": awu(12, hr), "sys": "TDB"},
                "TDB",
                2,
            ),
        ],
        ids=[
            "dt-Based (Both EpochArray)",
            "n_epochs-Based (Both EpochArray)",
            "Start Float, System From End",
            "End Float, System From Start",
            "Explicit System Override",
            "String Bounds",
        ],
    )
    def test_build(self, start, end, kwargs, expected_system, expected_size):
        """
        Verifies that interval() correctly builds a step- or count-defined
        interval whether both bounds are EpochArray objects (the primary
        documented use case), only one is (inheriting the other's time
        system), or an explicit system is given to override it.
        """
        result = scb.EpochArray.interval(start, end, **kwargs)
        assert result.system == expected_system
        assert result.size == expected_size
