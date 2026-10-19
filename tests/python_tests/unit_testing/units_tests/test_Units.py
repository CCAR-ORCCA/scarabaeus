# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================#
#  Unit Tests for the Units class  #
# ==================================#
import src.scarabaeus as scb
import pytest
import numpy as np

from scarabaeus.units.Units import _vandv_log

(
    N, kg, sec, km, dam, m, mm, mu, rad, mrad, deg, g, hr,
    microsec, msec, min_, day, Hz, lsec, lmin, LD, AU, arcsec, arcmin,
) = scb.Units.get_units(
    [
        "N", "kg", "sec", "km", "dam", "m", "mm", "mu", "rad", "mrad", "deg", "g", "hr",
        "microsec", "msec", "min", "day", "Hz", "lsec", "lmin", "LD", "AU", "arcsec", "arcmin",
    ]
)

# --------------------#
# region    Fixtures #
# --------------------#

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
@pytest.mark.parametrize(
    "dimensions, scales, expected",
    [
        (scb.Dimensions([0, 0, 0, 0]), [0, 0, 0, 0], "unitless"),
        (scb.Dimensions([1, 0, 0, 0]), [1, 0, 0, 0], "g"),
        (scb.Dimensions([0, 1, 0, 0]), [0, 1, 0, 0], "m"),
        (scb.Dimensions([0, 0, 1, 0]), [0, 0, 1, 0], "sec"),
        (scb.Dimensions([0, 0, 0, 1]), [0, 0, 0, 1], "rad"),
        (scb.Dimensions([1, 1, 1, 1]), [1, 1, 1, 1], "g*m*sec*rad"),
    ],
    ids=["Unitless", "Base Mass", "Base Length", "Base Time", "Base Angle", "Compound"],
)
def test_initialization(dimensions, scales, expected):
    """
    Verifies that the Units class initializes as expected when given Dimensions
    and scales inputs.
    """
    # construct unit
    unit = scb.Units(dimensions, scales)

    # verify that name is correct
    assert unit.name == expected
    assert repr(unit) == expected

    # verify that [unitless] flag is set properly
    if all(dimensions.powers == [0, 0, 0, 0]):
        assert unit.unitless == True
    else:
        assert unit.unitless == False


class TestSetterValidation:
    def test_dimensions(self):
        """
        Verifies that assigning a [dimensions] value that is not a
        Dimensions instance raises a TypeError.
        """
        unit = scb.Units(scb.Dimensions([1, 0, 0, 0]), [1, 0, 0, 0])
        with pytest.raises(TypeError):
            unit.dimensions = "not a dimensions object"

    @pytest.mark.parametrize(
        "scales",
        ["bad", ["a", "b", "c", "d"]],
        ids=["bad scales type", "bad scales dtype"],
    )
    def test_scales(self, scales):
        """
        Verifies that assigning an invalid [scales] value (bad type or
        non-numeric dtype) raises a TypeError.
        """
        unit = scb.Units(scb.Dimensions([1, 0, 0, 0]), [1, 0, 0, 0])
        with pytest.raises(TypeError):
            unit.scales = scales


class TestDictConversion:
    @pytest.mark.parametrize(
        "unit",
        [kg, km, N, scb.Units(scb.Dimensions([0, 0, 0, 0]), [0, 0, 0, 0])],
        ids=["Base Unit", "Scaled Unit", "Compound Unit", "Unitless"],
    )
    def test_to_dict(self, unit):
        """
        Verifies that to_dict serializes a Units object into a JSON-friendly
        dict containing its integer powers, float scales, and name.
        """
        result = unit.to_dict()

        assert result["powers"] == [int(p) for p in unit.dimensions.powers]
        assert result["scales"] == [float(s) for s in unit.scales]
        assert result["name"] == unit.name
        assert all(isinstance(p, int) for p in result["powers"])
        assert all(isinstance(s, float) for s in result["scales"])

    @pytest.mark.parametrize(
        "unit",
        [kg, km, N, scb.Units(scb.Dimensions([0, 0, 0, 0]), [0, 0, 0, 0])],
        ids=["Base Unit", "Scaled Unit", "Compound Unit", "Unitless"],
    )
    def test_from_dict(self, unit):
        """
        Verifies that from_dict reconstructs a Units object equivalent to
        the original from a dict produced by to_dict, round-tripping
        powers and scales correctly.
        """
        reconstructed = scb.Units.from_dict(unit.to_dict())

        assert reconstructed == unit
        assert np.array_equal(reconstructed.dimensions.powers, unit.dimensions.powers)
        assert np.array_equal(reconstructed.scales, unit.scales)


@pytest.mark.parametrize(
    "powers, scales, expected",
    [
        ([0, 0, 1, 0], [0, 0, 3600, 0], "hr"),
        ([1, 0, 0, 0], [1, 0, 0, 0], "g"),
        ([0, 1, 0, 0], [0, 1e3, 0, 0], "km"),
        ([0, 2, 0, 0], [0, 1e6, 0, 0], "km^2"),
        ([0, 0.5, 0, 0], [0, 31.622776601683793, 0, 0], "km^0.5"),
        ([0, 1, -1, 0], [0, 1e3, 1, 0], "km/sec"),
        ([0, -2, 0, 0], [0, 1e6, 0, 0], "km^-2"),
        ([1, 1, -2, 0], [1e3, 1e3, 1, 0], "(kg*km)/sec^2"),
        ([0, 1, -1, -1], [0, 1e3, 1, 1], "km/(sec*rad)"),
        ([0, -1, 0, 0], [0, 1e3, 0, 0], "[(1.00E+03 m^-1.0)]"),
        ([0, 2, -2, 0], [0, 1e3, 1, 0], "[(1.00E+03 m^2.0) (1.00E+00 sec^-2.0)]"),
    ],
    ids=[
        "named unit match",
        "base unit no prefix",
        "prefixed unit unity power",
        "prefixed unit integer power",
        "fractional power",
        "numerator and denominator",
        "denominator only, power != 1",
        "numerator and denominator, parens on numerator",
        "numerator and denominator, parens on denominator",
        "fallback: unnameable denominator-only unity power",
        "fallback: scale/power mismatch can't resolve a prefix",
    ],
)
def test_name_casting(powers, scales, expected):
    """
    Verifies that _name_casting resolves named units directly, builds
    compound names for prefixed/unprefixed base units across integer
    and fractional powers and numerator/denominator combinations, and
    falls back to a bracketed description when a compound name can't
    be resolved (either because the denominator-only unity-power case
    has no exponent to negate, or because the scale doesn't correspond
    to any known metric prefix).
    """
    unit = scb.Units(scb.Dimensions([0, 0, 0, 0]), [0, 0, 0, 0])
    result = unit._name_casting(np.array(powers, dtype=float), np.array(scales, dtype=float))
    assert result == expected


class TestGetUnits:
    def test_easter_egg(self, mocker, capsys):
        """
        Verifies that get_units occasionally prints one of the V&V log
        messages when the random draw falls below the trigger threshold.
        """
        mocker.patch("random.random", return_value=0.0)
        scb.Units.get_units("km")

        captured = capsys.readouterr()
        assert captured.out.strip() in [msg.strip() for msg in _vandv_log]

    @pytest.mark.parametrize(
        "requested_units, expected_units",
        [
            ("km", km),
            (["km", "sec", "mrad", "N"], [km, sec, mrad, N]),
            ("common", [kg, km, sec, rad, deg, N, mu]),
            ("base", [g, m, sec, rad]),
            ("time set", [sec, microsec, msec, min_, hr, day, Hz]),
            ("length set", [km, lsec, lmin, LD, AU]),
            ("angular set", [rad, deg, arcsec, arcmin]),
            ("", ValueError),
            ("km/sec", Exception),
            ("kday", ValueError),
            ("xsec", ValueError),
            ("banana", ValueError),
        ],
        ids=[
            "Single Request",
            "Multiple Requests",
            "Common Set",
            "Base Set",
            "Time Set",
            "Length Set",
            "Angular Set",
            "Empty Unit String",
            "Compound Unit Not Supported",
            "Unit Incompatible With Prefixes",
            "Unrecognized Prefix",
            "Unrecognized Unit",
        ],
    )
    def test_requests(self, requested_units, expected_units):
        """
        Verifies that the get_units class method functions correctly for
        single/multiple requests and for each named unit-set shortcut
        (base, common, time, length, angular), that an empty unit string
        raises a ValueError, that a compound unit request (containing "/",
        "*", or "^") raises an Exception since that isn't implemented yet,
        that prefixing a unit that doesn't allow prefixes (e.g. "kday")
        raises a ValueError, that an unrecognized metric prefix (e.g.
        "xsec") raises a ValueError, and that a completely unrecognized
        unit (e.g. "banana") raises a ValueError.
        """
        if isinstance(expected_units, type) and issubclass(expected_units, Exception):
            with pytest.raises(expected_units):
                scb.Units.get_units(requested_units)
        else:
            assert np.array_equal(scb.Units.get_units(requested_units), expected_units)


class TestMathOperations:

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (m, m, m**2),
            (dam**3 * sec**-2, km**-2, mm / sec**2),
            # a
            (
                np.array([[km, N, kg], [1, 1, 1]]),
                # b
                np.array([[sec**-2, sec**2 * kg**-1, m**2], [m, m, m]]),
                # expected
                np.array([[km / sec**2, m, kg * m**2], [m, m, m]]),
            ),
            (kg, np.eye(3), np.array([[kg, 0, 0], [0, kg, 0], [0, 0, kg]])),
        (km, [1.0, 1.0, 1.0], np.array([km, km, km])),
        (km, [[1.0, 1.0], [1.0, 1.0]], np.array([[km, km], [km, km]])),
        (
            km**-1,
            m**-1,
            scb.Units(scb.Dimensions([0, -2, 0, 0]), [0, 1000.0, 0, 0]),
        ),
        (kg, "not a unit or number", TypeError),
        ],
        ids=["Scalar Units", "Scalar Compound", "Matrix Units", "Unit Times Matrix",
             "Unit Times 1D List", "Unit Times 2D List", "Same Dim Negative Power",
             "Invalid Type"],
    )
    def test_multiplication(self, a, b, expected):
        """
        Verifies that the multiplication operator functions as expected,
        and that multiplying by an unsupported type raises a TypeError.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a * b
        elif isinstance(a, scb.Units) and isinstance(b, scb.Units):
            assert (a * b) == expected
        else:
            assert ((a * b) == expected).all()

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (m**2, m, m),
            (mm / sec**2, km**-2, mm**3 * sec**-2),
            # a
            (
                np.array([[km / sec**2, m, kg * m**2], [m, m, m]]),
                # b
                np.array([[km, N, kg], [1, 1, 1]]),
                # expected
                np.array([[sec**-2, sec**2 * kg**-1, m**2], [m, m, m]]),
            ),
            (
                scb.Units(scb.Dimensions([0, 2, 0, 0]), [0, 0, 0, 0]),
                scb.Units(scb.Dimensions([0, 1, 0, 0]), [0, 0, 0, 0]),
                scb.Units(scb.Dimensions([0, 1, 0, 0]), [0, 0, 0, 0]),
            ),
            (kg, "not a unit or number", TypeError),
        ],
        ids=["Scalar Units", "Scalar Compound", "Matrix Units", "Both Scales Zero",
             "Invalid Type"],
    )
    def test_division(self, a, b, expected):
        """
        Verifies that the division operator functions as expected, and
        that dividing by an unsupported type raises a TypeError.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a / b
        elif isinstance(a, scb.Units) and isinstance(b, scb.Units):
            assert (a / b) == expected
        else:
            assert ((a / b) == expected).all()

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (1.0, sec, sec**-1),
            (1, km, km**-1),
            (np.array([m, km]), sec, np.array([m / sec, km / sec])),
            ([m, km], sec, [m / sec, km / sec]),
            ([m, "not a unit or number"], sec, TypeError),
            ("not a unit, array, or list", sec, TypeError),
        ],
        ids=[
            "Float Div Unit",
            "Int Div Unit",
            "Array Div Unit",
            "List Div Unit",
            "Invalid Element",
            "Invalid Type",
        ],
    )
    def test_rtruediv(self, a, b, expected):
        """
        Verifies that the reverse division operator functions correctly
        when a Unit appears on the right-hand side (scalar / unit,
        ndarray / unit, list / unit), and that an invalid array element
        or an unsupported [other] type each raise a TypeError.

        Calls __rtruediv__ directly rather than via the ``/`` operator:
        for an ndarray of objects, Python/NumPy dispatch the division
        elementwise to each element's own __truediv__ instead of
        invoking Units.__rtruediv__, which would silently skip the
        array-handling branch this test targets.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                b.__rtruediv__(a)
            return

        result = b.__rtruediv__(a)
        if isinstance(result, scb.Units):
            assert result == expected
        elif isinstance(result, list):
            assert result == expected
        else:
            assert (result == expected).all()

    @pytest.mark.parametrize(
        "to_raise, raised_to, expected",
        [
            (kg, 2, kg**2),
            (m / sec, 2, m**2 / sec**2),
            (kg, "not an int or float", TypeError),
        ],
        ids=["Scalar", "Non-Hom Matrix", "Invalid Type"],
    )
    def test_power(self, to_raise, raised_to, expected):
        """
        Verifies that the power operator functions as expected, and
        that raising a Unit to a non-numeric power raises a TypeError.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                to_raise**raised_to
        else:
            assert to_raise**raised_to == expected

    @pytest.mark.parametrize(
        "unit, expected",
        [
            (m**2, m),
            (km**2, km),
            ((km / sec)**2, km / sec),
        ],
        ids=["Scalar", "Scaled Unit", "Compound"],
    )
    def test_sqrt(self, unit, expected):
        """
        Verifies that sqrt returns the unit raised to the 0.5 power.
        """
        assert unit.sqrt() == expected

    @pytest.mark.parametrize(
        "a, b, case",
        [
            (kg, kg, "a = b"),
            (km / sec, km / sec, "a = b"),
            (sec, hr, "a != b"),
            (kg, m, "a != b"),
            (kg, "not a unit", "error"),
        ],
        ids=[
            "same unit", "equal compound units", "same type diff unit", "different types",
            "non-Units comparison",
        ],
    )
    def test_equal(self, a, b, case):
        """
        Verifies that the equality operator functions as expected, and
        that comparing to a non-Units object raises a TypeError.
        """
        match case:
            case "a = b":
                # ensure that a is to b
                assert a == b

            case "a != b":
                assert a != b

            case "error":
                with pytest.raises(TypeError):
                    a == b

    @pytest.mark.parametrize(
        "a, b, case",
        [
            (kg, kg, "a == b"),
            (m**2, km**2, "a != b"),
            (sec, hr, "a != b"),
            (kg, m, "a != b"),
        ],
        ids=[
            "same unit",
            "same type diffrent dim",
            "same type diff unit",
            "diffrent types",
        ],
    )
    def test_not_equal(self, a, b, case):
        """
        Verifies that the inequality operator functions as expected.
        """
        match case:
            case "a != b":
                # ensure that a is to b
                assert a != b

            case "a == b":
                assert (a != b) == False


@pytest.mark.parametrize(
    "option, value",
    [("verbose", True), ("unit_conventions", ["mg", "Gm", "ns", "deg"])],
    ids=["Verbose", "Unit Conventions"],
)
def test_print_options(option, value):
    """
    Verifies that unit printing options are respected.
    """
    # NOTE: need to do these with eval
    # scb.Units.set_print_options(verbose = value)
    # scb.Units.set_print_options(unit_conventions = value)
    pytest.skip("Not implemented in Units yet")


def test_disp_named_units(capsys):
    """
    Verifies that disp_named_units prints a formatted table containing
    the expected header and known unit names, and returns None.
    """
    result = scb.Units.disp_named_units()

    assert result is None

    captured = capsys.readouterr()
    assert "List of Named Units" in captured.out
    assert '"g"' in captured.out
    assert '"m"' in captured.out
    assert '"sec"' in captured.out
    assert '"N"' in captured.out
