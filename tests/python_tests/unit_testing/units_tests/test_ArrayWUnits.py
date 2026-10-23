# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
import src.scarabaeus as scb
import pytest
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import numpy as np
import warnings


Gm, km, m, mm, kg, g, sec, day, rad, mrad, deg, N, mN, microm, unitless = scb.Units.get_units(['Gm','km', 'm', 'mm', 'kg', 'g', 
                                                                             'sec', 'day','rad', 'mrad', 'deg', 
                                                                             'N', 'mN', 'microm','unitless'])

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
class TestInitialization:
    @pytest.mark.parametrize(
        'values, units, error',
        [
            ({'a': 1}, km, TypeError),
            (np.array(['a', 'b']), km, TypeError),
            (np.array([]), km, Exception),
            (1, 'not a unit', TypeError),
            ([1, 2], [km, 'bad'], TypeError),
        ],
        ids=[
            'Bad Values Type',
            'Non-Numeric Values',
            'Empty Values Array',
            'Bad Units Type',
            'Invalid Units List Element',
        ],
    )
    def test_errors(self, values, units, error):
        """
        Verifies that constructing an ArrayWUnits object raises the
        appropriate exception for a bad [values] type, non-numeric values,
        an empty values array, a bad [units] type, or a units list
        containing a non-Units, non-None element.
        """
        with pytest.raises(error):
            awu(values, units)

    @pytest.mark.parametrize(
        'values, units, expected_units',
        [
            ([1, 2, 3], [km, None, km], [km, unitless, km]),
            ([1, 2], [None, None], [unitless, unitless]),
        ],
        ids=['Mixed With Real Units', 'All None'],
    )
    def test_units_with_none(self, values, units, expected_units):
        """
        Verifies that None elements in a [units] list are rebuilt as
        unitless Units objects, preserving the position of the real units
        around them.
        """
        a = awu(values, units)
        assert np.array_equal(a.values, values)

        if a.homogeneous_units:
            result_units = [a.units] * len(expected_units)
        else:
            result_units = list(np.array(a.units).flatten())
        assert result_units == expected_units


@pytest.mark.parametrize(
    'new_values, expected',
    [
        ([4, 5, 6], [4, 5, 6]),
        (42, 42),
    ],
    ids=['List', 'Scalar'],
)
def test_values_setter_conditions_non_ndarray(new_values, expected):
    """
    Verifies that directly assigning a non-ndarray (list or scalar) to
    the [values] property converts it to a numpy array, matching the
    conditioning __init__ already performs for its own values argument.
    """
    a = awu([1, 2, 3], kg)
    a.values = new_values
    assert np.array_equal(a.values, expected)


class TestGetItem:
    def test_scalar_shape(self):
        """
        Verifies that indexing a true 0D scalar ArrayWUnits (shape == ())
        returns the object itself for index 0, and raises an IndexError
        for any other index.
        """
        a = awu(np.array(5.0), kg)
        assert a[0] is a
        with pytest.raises(IndexError):
            a[1]

    def test_non_homogeneous(self):
        """
        Verifies that indexing a non-homogeneous ArrayWUnits returns the
        value and its own unit at that index, not the whole units array.
        """
        a = awu([1, 2, 3], [kg, km, kg])
        result = a[1]
        assert result.values == 2
        assert result.units == km


class TestSetItem:
    @pytest.mark.parametrize(
        'self_awu, key, awu_in, expected_values, expected_units',
        [
            (awu([1, 2, 3], kg), 0, awu(99, kg), [99, 2, 3], [kg, kg, kg]),
            (awu([1, 2, 3], [kg, km, kg]), 1, awu(50, km), [1, 50, 3], [kg, km, kg]),
            (awu([1, 2, 3], [kg, km, kg]), 1, awu(50, kg), [1, 50, 3], [kg, kg, kg]),
            (awu([1, 2, 3], kg), 0, awu(50, km), [50, 2, 3], [km, kg, kg]),
        ],
        ids=[
            'Homogeneous Same Units',
            'Non-Hom Matching Slot',
            'Non-Hom Differing Slot',
            'Homogeneous Differing Units',
        ],
    )
    def test_setitem(self, self_awu, key, awu_in, expected_values, expected_units):
        """
        Verifies that setitem updates only the values in place when the
        target slot already shares units with the input, and correctly
        updates both values and units otherwise -- including demoting a
        homogeneous object to non-homogeneous without corrupting the
        units of untouched elements.
        """
        self_awu[key] = awu_in
        assert np.array_equal(self_awu.values, expected_values)

        if self_awu.homogeneous_units:
            result_units = [self_awu.units] * len(expected_values)
        else:
            result_units = list(np.array(self_awu.units).flatten())
        assert result_units == expected_units

    @pytest.mark.parametrize(
        'awu_in, error',
        [
            ('not an awu', TypeError),
        ],
        ids=['Invalid Type'],
    )
    def test_errors(self, awu_in, error):
        """
        Verifies that setitem raises a TypeError when the input is not an
        ArrayWUnits object.
        """
        a = awu([1, 2, 3], kg)
        with pytest.raises(error):
            a[0] = awu_in

    def test_non_numerical_values_error(self):
        """
        Verifies that setitem raises a TypeError when the input
        ArrayWUnits' values are non-numerical. This state can't occur
        through normal construction (the values setter always guarantees
        numerical elements), so it's exercised here by directly
        overwriting the private attribute.
        """
        a = awu([1, 2, 3], kg)
        awu_in = awu(99, kg)
        awu_in._values = np.array(['not', 'numeric'])

        with pytest.raises(TypeError):
            a[0] = awu_in


@pytest.mark.parametrize(
        'to_convert, convert_to, expected',
        [(awu(1, km)               , mm             , awu(1e6, mm)               ),    # simplest conversion,
         (awu(1, rad)              , deg            , awu(180.0 / np.pi, deg)    ),    # convert between non-base 10
         (awu(1, m**2)             , km**2          , awu(1e-6, km**2)           ),    # squaring
         (awu(1, km**3)            , mm**3          , awu(1e18, mm**3)           ),    # cubing
         (awu(1, kg*Gm**-2*mrad**3), g*km**-2*rad**3, awu(1e-18, g*km**-2*rad**3)),    # compound units
         (awu(1, km/day**2)        , m/sec**2       , awu(1e3/86400**2, m/sec**2)),    # compound units with base and non-base 10 conversions
         (awu([[1  , 2   , 3  ],
               [4  , 5e-3, 6  ],
               [7  , 8   , 9  ]], Gm),     # to_convert
               km,                         # convert_to
          awu([[1e6, 2e6 , 3e6],           # expected
               [4e6, 5e3 , 6e6],
               [7e6, 8e6 , 9e6]], km)                                            ),    # homogeneous matrix conversion
         (awu([0.1, 1000 , 5           ], 
              [kg , m**2 , kg*m*sec**-2]), # to_convert
              [g, km**2, mN],              # convert_to
          awu([1e2, 1e-3 , 5e3         ],  # expected
              [g  , km**2, mN          ])                                        ),    # non-homogeneous matrix conversion
         (awu(1, deg)              , 'not a unit'   , None                       ),    # can't convert to a non-unit
         (awu(1, sec)              , kg*sec         , None                       ),    # can't convert to a unit with different dimensions
         (awu([0.1, 1000 , 5           ],
              [kg , m**2 , kg*m*sec**-2]), # to_convert
              [g, sec, mN],                # convert_to
              None                       )],  # non-homogeneous: one element has incompatible dimensions
        ids = ['Basic Scalar', 'Non-Base 10 Scalar', 'Squared Scalar', 'Cubed Scalar', 'Compound Scalar', 'Multi-Base Comp. Scalar',
               'Homogeneous Matrix', 'Non-Hom Matrix', 'Non-Unit Argument', 'Incompatible Conversion', 'Non-Hom Incompatible Conversion']
)
def test_convert_to(request, to_convert, convert_to, expected):
        """
            Verifies that unit conversions function properly.

            Checks that a given AWU is converted to the requested units and back again correctly.
            This ensures that there is no precision loss when moving between units.
        """
        test_id = request.node.callspec.id      # current parameterized test
        # improper argument passed
        if test_id in ('Non-Unit Argument', 'Incompatible Conversion', 'Non-Hom Incompatible Conversion'):
            # should raise an error when called
            with pytest.raises(Exception):
                to_convert.convert_to(convert_to)
        
        # proper argument passed
        else:
            # verify conversion to
            converted = to_convert.convert_to(convert_to)
            if converted.homogeneous_units:
                # homogeneous AWU's
                assert np.array_equal(converted.values, expected.values)
                # and back
                converted_back = converted.convert_to(to_convert.units)
                assert np.array_equal(converted_back , to_convert)
            else:
                # non-homogeneous AWU's
                assert np.array_equal(converted.values , expected.values)
                assert all(converted.units  == expected.units)


class TestGetValueInTargetUnits:
    @pytest.mark.parametrize(
        "quantity, target_units, case",
        [
            (awu(1.0, km), m, "success"),
            (awu(1.0, km), sec, "failure"),
            (awu(3.0, km), km, "direct_match"),
            (5.0, unitless, "plain_unitless_value"),
        ],
        ids=[
            "Compatible Units Auto-Converts",
            "Incompatible Units Raise ValueError",
            "Direct Unit Match",
            "Plain Unitless Value",
        ],
    )
    def test_conversion(self, quantity, target_units, case):
        """
        Verifies that get_value_in_target_units warns and auto-converts
        when the quantity's units differ from the target but are
        dimensionally compatible, raises a ValueError with the expected
        message when conversion is not possible, returns the value
        directly when units already match, and returns the quantity as-is
        when it's already a plain unitless float/int.
        """
        match case:
            case "success":
                with pytest.warns(UserWarning, match="Trying automatic conversion"):
                    result = awu.get_value_in_target_units(quantity, target_units)
                assert np.array_equal(result, quantity.convert_to(target_units).values)

            case "failure":
                with pytest.warns(UserWarning, match="Trying automatic conversion"):
                    with pytest.raises(ValueError, match="Automatic conversion attempted but failed."):
                        awu.get_value_in_target_units(quantity, target_units)

            case "direct_match":
                result = awu.get_value_in_target_units(quantity, target_units)
                assert result == quantity.values

            case "plain_unitless_value":
                result = awu.get_value_in_target_units(quantity, target_units)
                assert result == quantity

    def test_mismatch_after_conversion(self, mocker):
        """
        Verifies that get_value_in_target_units raises a ValueError when
        convert_to succeeds but its result's units still don't match the
        requested target. convert_to's real implementation always
        constructs its result with the exact requested target units, so
        this branch can't occur through normal use -- it's exercised here
        by mocking convert_to's return value.
        """
        quantity = awu(1.0, km)
        mocker.patch.object(quantity, "convert_to", return_value=awu(1.0, sec))

        with pytest.warns(UserWarning, match="Trying automatic conversion"):
            with pytest.raises(ValueError, match="Automatic conversion attempted but failed."):
                awu.get_value_in_target_units(quantity, m)


class TestValuesIn:
    """
        values_in strips a quantity to bare numbers in the target units, the
        way public methods hand their input to a numeric core.
    """

    def test_none_and_bare_numbers(self):
        """
            Verifies that None, an optional argument left out, passes through,
            and that bare numbers are taken to be in the target units and come
            back as a new float array.
        """
        values = [1, 2, 3]
        result = awu.values_in(values, km)

        assert awu.values_in(None, km) is None
        assert result.dtype == float
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        'quantity, units, expected',
        [
            (awu([1.0, 2.0], km)                , m             , [1000.0, 2000.0]),
            (awu([1.0, 2.0], [km, sec])         , [m, sec]      , [1000.0, 2.0]),
            (awu([[1.0, 2.0], [3.0, 4.0]], m)   , [km, m]       , [[1e-3, 2e-3], [3.0, 4.0]]),
            (awu(2.0, deg)                      , deg           , 2.0),
        ],
        ids=['One Unit', 'One Unit Per Element', 'One Unit Per Row', 'Scalar'],
    )
    def test_conversion(self, quantity, units, expected):
        """
            Verifies that elements in other units of the same dimension are
            converted without a warning, for one target unit, one per element
            and one per row.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = awu.values_in(quantity, units)

        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_matching_units_keep_their_numbers(self):
        """
            Verifies that elements already in their target units are copied,
            not converted, so they are unchanged to the last bit and the input
            is not aliased.
        """
        values = np.array([0.1, 1.0 / 3.0, np.pi])
        quantity = awu(values, km)
        result = awu.values_in(quantity, km)

        assert np.array_equal(result, values)
        result[0] = 0.0
        assert quantity.values[0] == 0.1

    def test_framed_quantity(self):
        """
            Verifies that an ArrayWFrame gives the numbers of its quantity.
        """
        quantity = scb.ArrayWFrame(awu([1.0, 2.0, 3.0], m), scb.Frame("J2000"))

        np.testing.assert_allclose(awu.values_in(quantity, km), [1e-3, 2e-3, 3e-3], rtol=1e-15)

    @pytest.mark.parametrize(
        'quantity, units',
        [
            (awu([1.0, 2.0, 3.0], km)   , sec),
            (awu([1.0, 2.0, 3.0], km)   , [km, km]),
        ],
        ids=['Different Dimensions', 'Units Match Neither Elements Nor Rows'],
    )
    def test_errors(self, quantity, units):
        """
            Verifies that a ValueError is raised for target units of another
            dimension and for target units that fit the quantity's shape
            neither per element nor per row.
        """
        with pytest.raises(ValueError):
            awu.values_in(quantity, units)


class TestMathOperations:
    """
        Collected tests for all mathematical operations for ArrayWUnits.
    """
    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu(1        , kg), awu(1        , kg), awu(2        , kg)),
             (awu([1, 2, 3], kg), awu([1, 2, 3], kg), awu([2, 4, 6], kg)),
             (awu(1        , kg), awu(1        , km), Exception        ),
             (awu([1, 2, 3], kg), awu([1, 2, 3], km), Exception        ),
             (awu(1        , kg), 'not awu'         , TypeError        ),
             (awu([1, 2, 3], kg), awu([1, 2], kg)   , Exception        ),
             (awu([1, 2]   , [kg, km]), awu([1, 2]  , [kg, sec]), ValueError)],
            ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix',
                   'Invalid Type', 'Shape Mismatch', 'Non-Hom Dimension Mismatch']
    )
    def test_addition(self, a, b, expected):
        """
            Verifies that addition operations perform as expected, and
            that a non-ArrayWUnits operand, mismatched shapes, or a
            non-homogeneous dimension mismatch each raise the
            appropriate exception.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a + b
        elif a.size == 1:
            assert a + b == expected
        else:
            assert all(a + b == expected)
    
    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu(2        , kg), awu(1        , kg), awu(1        , kg)),
             (awu([2, 4, 6], kg), awu([1, 2, 3], kg), awu([1, 2, 3], kg)),
             (awu(2        , kg), awu(1        , km), None              ),
             (awu([2, 4, 6], kg), awu([1, 2, 3], km), None              )],
            ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix']
    )
    def test_subtraction(self, a, b, expected):
        """
            Verifies that subtraction operations perform as expected.
        """
        if expected is not None:
            if a.size == 1:
                assert a - b == expected
            else:
                assert all(a - b == expected)
        
        else:
            # can't perform subtraction between two different units
            with pytest.raises(Exception):
                a - b

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu(2        , kg)           , awu(2        , kg)               , awu(4        , kg**2             )),
             (awu([1, 2, 3], kg)           , awu([1, 2, 3], kg)               , awu([1, 4, 9], kg**2             )),
             (awu(1        , kg)           , awu(1        , km)               , awu(1        , kg*km             )),
             (awu([1, 2, 3], kg)           , awu([1, 2, 3], km)               , awu([1, 4, 9], kg*km             )),
             (awu(1        , kg*km*sec**-2), awu(1        , km**2*kg)         , awu(1        , km**3*kg**2/sec**2)),
             (awu([1, 2]   , [kg, km])     , awu([3, 4]   , [km**-3, sec**-1]), awu([3, 8]   , [kg/km**3, km/sec])),
             (awu(1, m**3*sec**-2)         , awu(1, km**-2)                   , awu(1        , microm/sec**2               )),
             (awu(1        , kg)           , 'not valid'                      , TypeError                                  )],
             ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix', 'Compound Units', 'Comp Units Matrix',
                    'Same Dims', 'Invalid Type']
    )
    def test_multiplication(self, a, b, expected):
        """
            Verifies that the multiplication operator functions as expected,
            and that multiplying by an unsupported type raises a TypeError.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a * b
        elif a.size == 1:
            assert (a * b) == expected
        else:
            assert all((a * b) == expected)
    
    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu(2        , kg)           , awu(2        , kg)               , awu(1        , None              )),
             (awu([1, 2, 3], kg)           , awu([1, 2, 3], kg)               , awu([1, 1, 1], None              )),
             (awu(1        , kg)           , awu(1        , km)               , awu(1        , kg/km             )),
             (awu([1, 2, 3], kg)           , awu([1, 2, 3], km)               , awu([1, 1, 1], kg/km             )),
             (awu(1        , kg*km*sec**-2), awu(1        , km**2*kg)         , awu(1        , sec**-2*km**-1    )),
             (awu([1, 2]   , [kg, km])     , awu([1, 2]   , [km**-3, sec**-1]), awu([1, 1]   , [kg*km**3, km*sec]))],
             ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix', 'Compound Units', 'Comp Units Matrix']
    )
    def test_division(self, a, b, expected):
        """
            Verifies that the division operator functions as expected.
        """
        
        if a.size == 1:
            assert (a / b) == expected
        else:
            assert all((a / b) == expected)

    @pytest.mark.parametrize(
            'a, b, expected',
            [(5  , awu(2, kg)      , awu(2.5      , kg**-1        )),
             (1.0, awu([1, 2], kg) , awu([1, 0.5] , kg**-1        ))],
             ids = ['Int Div AWU', 'Float Div AWU Matrix']
    )
    def test_rtruediv(self, a, b, expected):
        """
            Verifies that the reverse true-division operator functions
            correctly when a plain int or float appears on the left of
            an ArrayWUnits object (e.g. ``5 / awu(2, kg)``).
        """
        result = a / b
        if result.size == 1:
            assert result == expected
        else:
            assert all(result == expected)

    @pytest.mark.parametrize(
            'to_raise, raised_to, expected',
            [(awu(5        , kg           ),         2, awu(25                , kg**2                 )),
             (awu([1, 2, 3], kg           ),        -2, awu([1, 0.25, 3.0**-2], kg**-2                )),
             (awu([1, 2, 3], [kg, km, sec]),         2, awu([1,    4,       9], [kg**2, km**2, sec**2])),
             (awu(5        , kg           ), 'not a number', TypeError                                 )],
             ids = ['Scalar', 'Matrix to Scalar', 'Non-Hom Matrix', 'Invalid Exponent Type']
    )
    def test_power(self, to_raise, raised_to, expected):
        """
            Verifies that the power operator functions as expected, and
            that a non-int/float exponent raises a TypeError.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                to_raise**raised_to
        elif expected.size == 1:
            assert (to_raise**raised_to) == expected

        else:
            assert all((to_raise**raised_to) == expected)

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(4, kg**2)               , awu(2      , kg)          ),
             (awu([4, 9, 16], kg**2)      , awu([2, 3, 4], kg)        )],
             ids = ['Scalar', 'Matrix']
    )
    def test_sqrt(self, a, expected):
        """
            Verifies that sqrt returns the same result as raising to
            the 0.5 power.
        """
        if expected.size == 1:
            assert a.sqrt() == expected
        else:
            assert all(a.sqrt() == expected)

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu([1, 2, 3], kg), awu([4, 5, 6]      , kg), awu(32, kg**2)),
             (awu([[1, 2, 3]], kg), awu([[4], [5], [6]], kg), awu(32, kg**2)),
             (awu([[1, 0, 0],                   # a
                   [0, 2, 0],
                   [0, 0, 3]]      , kg    ),
              awu([[4],  [5],  [6]], kg    ),   # b
              awu([[4], [10], [18]], kg**2)),   # expected
             (awu([[1, 0, 0],                   # a
                   [0, 2, 0],
                   [0, 0, 3]]  , kg    ),
              awu([[4, 0, 0],                   # b
                   [0, 5, 0],
                   [0, 0, 6]]  , kg    ),
              awu([[4,  0,  0],                 # expectd
                   [0, 10,  0],
                   [0, 0 , 18]], kg**2)),
             (awu([[1], [2], [3]], kg), awu([4, 5], km),
              awu([[4, 5], [8, 10], [12, 15]], kg*km)),  # column @ row-vector (outer product)
             (awu([1, 2, 3], kg), 'not awu', TypeError),
             (awu([1, 2, 3], kg), awu([1, 2], kg), Exception),
             (awu([[1, 2, 3], [4, 5, 6]], kg), awu([[1, 2], [3, 4]], kg), Exception)],
             ids = ['Hom Row * Hom Row', 'Hom Row * Hom Column', 'Hom Mat * Hom Column',
                    'Hom Mat * Hom Mat', 'Column * Row Vector', 'Invalid Type',
                    'Row Vector Shape Mismatch', 'Matrix Shape Mismatch']
    )
    def test_matrix_mulitplication(self, a, b, expected, request):
        """
            Verifies that the matrix multiplication operator functions as expected,
            that non-ArrayWUnits operands raise a TypeError, and that mismatched
            shapes (for both the row-vector dot-product case and the general
            matrix case) raise an Exception.
        """
        if request.node.callspec.id == 'Hom Row * Hom Column': pytest.skip('SKIP FOR NOW')
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a @ b
        elif (a @ b).size == 1:
            assert a @ b == expected, f'expected {expected}, got {a @ b}'
        else:
            assert all(a @ b == expected)

    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(1, kg)        , awu(1, kg)        , 'a = b'),
             (awu([1, 2, 3], kg), awu([1, 2, 3], kg), 'a = b'),
             (awu(1        , km), 5                 , 'error')],
             ids = ['Same Units', 'Same Units Matrix', 'Not AWU']
    )
    def test_equal(self, a, b, case):
        """
            Verifies that the equals operator functions as expected.
        """
        match case:
            case 'a = b':
                # ensure that a is to b
                if a.size == 1:
                    assert a == b
                else:
                    assert all(a == b)
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a == b

    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(1, kg)        , awu(2, kg)        , 'a != b'),
             (awu([1, 2, 3], kg), awu([1, 1, 1], kg), 'a != b'),
             (awu(1        , kg), awu(2        , km), 'a != b'),
             (awu([1, 2, 3], kg), awu([1, 1, 1], km), 'a != b'),
             (awu(1        , km), 5                 , 'error' )],
             ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix', 'Not AWU']
    )
    def test_not_equal(self, a, b, case):
        """
            Verifies that the not-equal operator functions as expected.
        """
        match case:
            case 'a != b':
                # ensure that a is not equal to b
                if a.size == 1:
                    assert a != b
                else:
                    assert any(a != b)
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a != b

    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(2, kg), awu(1, kg) , 'a > b'),
             (awu(1, kg), awu(1, rad), 'error')],
             ids = ['Same Units', 'Different Units']
    )
    def test_greater_than(self, a, b, case):
        """
            Verifies that the greater than inequality functions as expected.
        """
        match case:
            case 'a > b':
                # ensure that a is greater than b
                assert a > b
                # and that b is not greater than a
                assert not b > a
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a > b

    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(2, kg), awu(1, kg) , 'a > b'),
             (awu(1, kg), awu(1, kg) , 'a = b'),
             (awu(1, kg), awu(1, rad), 'error')],
             ids = ['Greater Than (Same Units)', 'Equal To (Same Units)', 'Different Units']
    )
    def test_greater_or_equal(self, a, b, case):
        """
            Verifies that the greater than or equal to inequality functions as expected.
        """
        match case:
            case 'a > b':
                # ensure that a is greater than b
                assert a >= b
                # and that b is not greater than a
                assert not b >= a
            
            case 'a = b':
                # ensure that a is equal to b by ge
                assert a >= b
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a >= b

    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(1, kg), awu(2, kg), 'a < b'),
             (awu(1, kg), awu(2, rad), 'error')],
             ids = ['Same Units', 'Different Units']
    )
    def test_less_than(self, a, b, case):
        """
            Verifies that the less than inequality functions as expected.
        """
        match case:
            case 'a < b':
                # ensure that a is greater than b
                assert a < b
                # and that b is not greater than a
                assert not b < a
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a < b
        
    @pytest.mark.parametrize(
            'a, b, case',
            [(awu(1, kg), awu(2, kg) , 'a < b'),
             (awu(1, kg), awu(1, kg) , 'a = b'),
             (awu(1, kg), awu(1, rad), 'error')],
             ids = ['Less Than (Same Units)', 'Equal To (Same Units)', 'Different Units']
    )
    def test_less_or_equal(self, a, b, case):
        """
            Verifies that the less than or equal to inequality functions as expected.
        """
        match case:
            case 'a < b':
                # ensure that a is greater than b
                assert a <= b
                # and that b is not greater than a
                assert not b <= a
            
            case 'a = b':
                # ensure that a is equal to b by ge
                assert a <= b
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a <= b

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu([1, 2, 3], kg)           , awu([4, 0, 1], kg)               , awu([2, 11, -8], kg**2             )),
             (awu([1, 2, 3], kg)           , awu([4, 0, 1], km)               , awu([2, 11, -8], kg*km             )),
             (awu([1, 2, 3] , kg*km*sec**-2), awu([4, 0, 1] , km**2*kg)       , awu([2, 11, -8]        , km**3*kg**2/sec**2)),
             (awu([1, 2, 3], kg)            , 'not awu'                       , TypeError                                ),
             (awu([1, 2, 3], [kg, km, sec]) , awu([1, 2, 3], kg)              , Exception                                ),
             (awu([[1, 2, 3], [4, 5, 6]], kg), awu([1, 2, 3], kg)             , Exception                                )],
             ids = [ 'Same Units Matrix', 'Diff Units Matrix',  'Comp Units Matrix',
                     'Invalid Type', 'Non-Homogeneous', 'Non-Vector']
    )
    def test_cross_product(self,a,b, expected):
        """
            Verfies that the cross-product operator functions as expected.

            Limits:
            - Cross-product operation is only defined between two ArrayWUnits objects.
            - Cross-product operation is only defined for ArrayWUnits with homogeneous units.
            - Cross-product operation is only defined for vectorial ArrayWUnits objects.

        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.cross(a, b)
        else:
            assert awu.cross(a,b) == expected
        

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(1, kg)                              , awu(1, kg)                             ),
             (awu([1, 2, 3], km)                      , awu([[1],[2],[3]], km)                 ),
             (awu([[1],[2],[3]], km)                   , awu([1, 2, 3], km)                    ),
             (awu([[1, 2, 3],[4,5,6],[7,8,9]], km/sec), awu([[1,4,7],[2,5,8],[3,6,9]], km/sec)),
             (awu([1, 2, 3], [km, kg, km])             , awu([[1],[2],[3]], [[km],[kg],[km]])   ),
             (awu([[1],[2],[3]], [[km],[kg],[km]])     , awu([1, 2, 3], [km, kg, km])            ),
             (awu([[1, 2, 3],[4, 5, 6]], [[km, kg, sec],[rad, km, kg]]),
              awu([[1, 4],[2, 5],[3, 6]], [[km, rad],[kg, km],[sec, kg]])                        ),],

             ids = ['Scaler', '1D Matrix', '2D Column Vector', '2D Matrix',
                    'Non-Hom 1D Vector', 'Non-Hom Column Vector', 'Non-Hom 2D Matrix']
    )
    def test_transpose(self, a, expected):
        """
            Verfies that the transpose operator functions as expected.

            Limits:
            - Transpose operation defined only for scalar, 1D, and 2D ArrayWunits instances.
        """
        assert awu.transpose(a) == expected

    def test_transpose_raises_for_3d(self):
        """
        Verifies that a ValueError is raised when transpose is called on a 3D array.
        """
        with pytest.raises(ValueError):
            awu(np.ones((2, 2, 2)), km).transpose()

    @pytest.mark.parametrize(
        'a, b, axis, case',
        [
            (awu([1, 2, 3], km), awu([4, 5], km), None, "homogeneous"),
            (awu([1, 2], [km, kg]), awu([4, 5], km), None, "non_homogeneous_self"),
            (awu([1, 2, 3], km), awu([4, 5], [km, kg]), None, "non_homogeneous_other"),
            (awu([[1, 2], [3, 4]], km), awu([1, 2, 3], km), 0, ValueError),
            (awu([1, 2, 3], km), [1, 2, 3], None, TypeError),
        ],
        ids=[
            'Homogeneous',
            'Non-Homogeneous Self',
            'Non-Homogeneous Other',
            'Incompatible Shapes',
            'Invalid Other Type',
        ],
    )
    def test_append(self, a, b, axis, case):
        """
        Verifies that append correctly combines homogeneous and
        non-homogeneous ArrayWUnits objects, and raises the
        appropriate exception for incompatible shapes or a
        non-ArrayWUnits [other].
        """
        if isinstance(case, type) and issubclass(case, Exception):
            with pytest.raises(case):
                a.append(b, axis)
            return

        result = a.append(b, axis)
        match case:
            case "homogeneous":
                assert np.array_equal(result.values, [1, 2, 3, 4, 5])
                assert result.units == km

            case "non_homogeneous_self":
                assert np.array_equal(result.values, [1, 2, 4, 5])
                assert list(result.units) == [km, kg, km, km]

            case "non_homogeneous_other":
                assert np.array_equal(result.values, [1, 2, 3, 4, 5])
                assert list(result.units) == [km, km, km, km, kg]

    @pytest.mark.parametrize(
        'a, expected',
        [
            (awu(np.array(5.0), km),               (1, 1)),
            (awu(5, km),                           (1, 1)),
            (awu([1, 2, 3], km),                   (1, 3)),
            (awu([[1, 2], [3, 4], [5, 6]], km),    (3, 2)),
        ],
        ids=['true 0D scalar', '1-element 1D', '1D', '2D']
    )
    def test_get_shape(self, a, expected):
        """
        Verifies that _get_shape returns the correct (n, m) tuple for
        a true 0D scalar (shape == ()), a 1-element 1D array (int/float
        construction always yields shape (1,), never a true 0D shape),
        a 1D array, and a 2D array.
        """
        assert a._get_shape() == expected

    @pytest.mark.parametrize(
        'a, b, expected',
        [
            (awu([[1, 2], [3, 4]], km), awu([[5, 6], [7, 8]], km), True),
            (awu([1, 2], km), awu([[1, 2], [3, 4], [5, 6]], km), True),
            (awu([[1, 2], [3, 4], [5, 6]], km), awu([1, 2], km), True),
            (awu([[1, 2], [3, 4]], km), awu([1, 2, 3], km), Exception),
        ],
        ids=[
            'Same Shape',
            'Self is 1D Subarray',
            'Other is 1D Subarray',
            'Incompatible Shapes',
        ],
    )
    def test_are_shapes_compatible(self, a, b, expected):
        """
        Verifies that _are_shapes_compatible returns True for matching
        shapes and for the two broadcastable 1D-subarray cases, and
        raises an Exception when shapes are genuinely incompatible.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a._are_shapes_compatible(b)
        else:
            assert a._are_shapes_compatible(b) == expected

    @pytest.mark.parametrize(
                'a, expected',
                [
                (awu([[1, 2],[3, 4]], km)             , awu([[[-2, 1],[1.5, -0.5]]], km             )),
                (awu([[1, 0, 2],[0, 1, 1],[1, 0, 1]] , km/sec) , awu([[[1, 0, -2],[1, 1, -1],[-1, 0, 1]]], km/sec)),
                (awu([[1, 2],[3, 4]], [[km, kg],[kg, km]])     , ValueError                                 ),
                (awu([[1, 2, 3],[4, 5, 6]], km)                , ValueError                                 ),],

                ids = [ '2x2',  '3x3', 'Non-Homogeneous', 'Non-Square']
        )
    def test_inverse(self,a, expected):
        """
            verifies that the inverse operator functions as expected.

            Limits:
            - Inverse operation is only defined for ArrayWUnits objects with homogeneous units.
            - Inverse operation is only defined for square matrices.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.inverse(a)
        else:
            assert awu.inverse(a) == expected

    @pytest.mark.parametrize(
                'a, expected',
                [(awu([[1, 2, 3], [4, 5, 6]], km)     , awu([[-17/18,  4/9], [-1/9,  1/9], [13/18, -2/9]], km             )),
                (awu([[3], [4]], kg)                  , awu([[0.12, 0.16]], kg             )),
                (awu([[1, 2],[3, 4]], m/sec)             , awu([[[-2, 1],[1.5, -0.5]]], m/sec            )),
                (awu([[1, 2], [2, 4]]  , km/sec) , awu([[[1, 0, -2],[1, 1, -1],[-1, 0, 1]]], km/sec)),
                (awu([1, 2, 3], [kg, km, rad])   , Exception),],

                ids = [ 'Rectangular Matrix',  'Vector', 'Square Matrix', 'Singular Matrix', 'Non-Homogeneous']
        )
    def test_pseudo_inverse(self,a, expected):
        """
            Verfies that the pseudo-inverse operator functions as expected.

            Limits:
            - Pseudo-inverse operation is only defined for ArrayWUnits objects with homogeneous units.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.pseudo_inverse(a)
        else:
            assert awu.pseudo_inverse(a) == expected


    @pytest.mark.parametrize(
            'to_sum, expected',
            [(awu(1        , kg           ), awu(1, kg)),
             (awu([1, 2, 3], kg           ), awu(6, kg)),
             (awu([1, 2, 3], [km, kg, rad]), 'error'   )],
             ids = ['Single', 'Matrix', 'Non-Homogeneous']
    )
    def test_summation(self, to_sum, expected):
        """
            Verifies that the summation operator functions as expected.
        """
        if isinstance(expected, str):
            # can't sum non-homogeneous ArrayWUnits
            with pytest.raises(Exception):
                to_sum.summation()
        else:
            assert to_sum.summation() == expected
    
    @pytest.mark.parametrize(
            'to_norm, expected',
            [(awu([1, 2, 3]  , kg), awu(np.sqrt(14), kg)),
             (awu([[1, 2, 3] ,
                   [4, 5, 6] ,
                   [7, 8, 9]], kg), 
             awu(np.sqrt(285)           , kg)),
             (awu([1, 2], [kg, km]), awu(np.sqrt(5), unitless))],
             ids = ['Vector Norm', 'Matrix Norm', 'Non-Homogeneous Units']
    )
    def test_norm(self, to_norm, expected):
        """
            Verifies that the norm operator functions as expected.
        """
        if not isinstance(expected, str):
            assert to_norm.norm() == expected
        else:
            # can't norm non-homogeneous units
            with pytest.raises(Exception):
                to_norm.norm()

    @pytest.mark.parametrize(
                'a, expected',
                [(awu([3, 4], kg)     , awu([3/5, 4/5], kg            )),
                (awu([1, 2, 2], km)   , awu([[1/3, 2/3, 2/3]], km             )),
                (awu([[1, 2], [3, 4]], km)      , ValueError                  ),
                (awu([1, 2, 3], [km, kg, sec])  , ValueError                  ),],
                ids = [ '2x1 vector',  '3x1 vector', 'Non-1D Matrix', 'Non-Homogeneous']
        )
    def test_unitary(self,a, expected):
        """
            Verfies that the unitary operator functions as expected.

            Limits:
            - Cannot unitize a non-1-D ArrayWUnits vector.
            - Cannot unitize an ArrayWUnits objects with non-homogeneous units.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.unitary(a)
        else:
            assert awu.unitary(a) == expected

    @pytest.mark.parametrize(
        'a, b, expected',
        [
            (awu([1, 0, 0], km), awu([0, 1, 0], km), np.pi / 2),
            (awu([1, 0, 0], km), awu([2, 0, 0], km), 0.0),
            (awu([1, 0, 0], km), awu([-1, 0, 0], km), np.pi),
            (awu([1, 2, 3], [kg, km, sec]), awu([1, 0, 0], km), ValueError),
            (awu([1, 0, 0], km), awu([1, 0], km), Exception),
            (awu([1, 0, 0], km), [1, 0, 0], TypeError),
            (awu([1, 0, 0], km), awu([1, 0, 0], km), "non_units_flat_element"),
        ],
        ids=[
            'Perpendicular',
            'Parallel',
            'Anti-Parallel',
            'Non-Homogeneous Self',
            'Shape Mismatch',
            'Invalid Other Type',
            'Non-Units Flat Element',
        ],
    )
    def test_angle_between(self, a, b, expected):
        """
        Verifies that angle_between computes the correct angle (in
        radians) between two homogeneous-unit vectors, that invalid
        inputs (non-homogeneous self, mismatched shapes, or a
        non-ArrayWUnits [other]) each raise the appropriate exception,
        and that it implicitly returns None when the internal _units
        array's first element is not a Units instance -- a state that
        can't occur through normal construction (the units setter
        always guarantees Units elements), so it's exercised here by
        directly overwriting the private attribute.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a.angle_between(b)
        elif expected == "non_units_flat_element":
            a._units = np.array(42)
            assert a.angle_between(b) is None
        else:
            result = a.angle_between(b)
            assert np.isclose(result.values, expected, atol=1e-12)
            assert result.units == rad


    @pytest.mark.parametrize(
    'a, expected',
    [
        (awu(1, kg),                               awu(np.e, kg)),
        (awu([1, 2, 3], km),                       awu([np.e, np.e**2, np.e**3], km)),
        (awu([[1, 2], [3, 4]], km/sec),             awu([[np.e, np.e**2], [np.e**3, np.e**4]], km/sec)),
        (awu([1, 2, 3], [kg, km, sec]),             awu([np.e, np.e**2, np.e**3], [kg, km, sec])),
    ],
    ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous']
    )
    def test_exp(self, a, expected):
        """
            Verifies that the exponential function works element-wise
            and preserves units.
        """
        if a.size == 1:
            assert np.isclose(awu.exp(a).values, expected.values, atol=1e-14)
            assert awu.exp(a).units == expected.units
        else:
            res = awu.exp(a)
            assert np.allclose(res.values, expected.values, atol=1e-14)
            assert np.array_equal(res.units, expected.units)

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(1, kg)                                , awu(2, kg            )),
             (awu([1, 2,3], km)                         , awu([2,4,8], km             )),
             (awu([[1, 2, 3],[4,5,6],[7,8,9]] , km/sec) , awu([[2,4,8],[16,32,64],[128,256,512]], km/sec)),
             (awu([1, 2,3], [kg, km, sec])                         , awu([2,4,8], [kg, km, sec]             ))],

             ids = [ 'Scaler', '1D Matrix',  '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_exp2(self, a, expected):
        """
            verifies that the exp2 function works as expected.
        """
        if a.size == 1:
            assert awu.exp2(a) == expected
        else:
            assert all(awu.exp2(a) == expected)

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(1, kg)                         , awu(0, kg                  )),
             (awu([1, 10,100], km)               , awu([0,1,2], km            )),
             (awu([[1, 10],[100,1000]] , km/sec) , awu([[0,1],[2,3]], km/sec  )),
             (awu([1, 10, 1000], [kg, km, sec])  , awu([0,1,3], [kg, km, sec] ))],

             ids = [ 'Scaler', '1D Matrix',  '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log10(self, a, expected):
        """
            Verifies that the log10 function works as expected.
        """
        if a.size == 1:
            assert awu.log10(a) == expected
        else:
            assert all(awu.log10(a) == expected)

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(2, kg)                                        , awu(1, kg            )),
             (awu([2,4,8], km)                                  , awu([1, 2,3], km             )),
             (awu([[2,4,8],[16,32,64],[128,256,512]] , km/sec)  , awu([[1, 2, 3],[4,5,6],[7,8,9]], km/sec)),
             (awu([2, 4,8], [kg, km, sec])                      , awu([1,2,3], [kg, km, sec]             ))],

             ids = [ 'Scaler', '1D Matrix',  '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log2(self, a, expected):
        """
            Verifies that the log2 function works as expected.
        """
        if a.size == 1:
            assert awu.log2(a) == expected
        else:
            assert all(awu.log2(a) == expected)

    @pytest.mark.parametrize(
        'a, expected',
        [   (awu(1, kg)                                                                                    ,awu(0, kg)),                                                                       
            (awu([1, np.e, np.e**2], km)                                                                   ,awu([0, 1, 2], km)), 
            (awu([[1, np.e, np.e**2], [np.e**3, np.e**4, np.e**5], [np.e**6, np.e**7, np.e**8]], km/sec)   ,awu([[0, 1, 2], [3, 4, 5], [6, 7, 8]], km/sec)),
            (awu([1, np.e, np.e**2], [kg, km, sec])                                                        ,awu([0, 1, 2], [kg, km, sec]))  
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log(self, a, expected):
        """
            Verifies that the natural logarithm function works as expected.
        """
        if a.size == 1:
            assert awu.log(a) == expected
        else:
            assert np.array_equal(awu.log(a) , expected)

    @pytest.mark.parametrize(
        'a, expected',
        [
            (awu(0, rad)                                      , awu(0,unitless )),  
            (awu(np.pi/2, rad)                                , awu(1,unitless )),  
            (awu(np.pi, rad)                                  , awu(0,unitless )),  
            (awu([0, np.pi/2, np.pi], rad)                    , awu([0, 1, 0],unitless)),
            (awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad)   , awu([[0, np.sqrt(2)/2], [1, np.sqrt(2)/2]],unitless)),
            (awu([1, 2, 3], [kg, km, rad])                    , Exception),
            (awu(1, km)                                       , ValueError)],
        ids=['Scalar 0', 'Scalar pi/2', 'Scalar pi', '1D Array', '2D Array', 'Non-Homogeneous', 'Non-Radian Units']
    )
    def test_sin(self, a, expected):
        """
            verifies that the sine function works as expected.

            Limits:
            - Sine function is only defined for ArrayWUnits instances with homogeneous units
            - Sine function only accepts an input in radians
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.sin(a)
        elif a.size == 1:
            assert np.isclose(awu.sin(a).values, expected.values, atol=1e-14)
        else:
            res = awu.sin(a)
            assert np.allclose(res.values, expected.values, atol=1e-14)
            assert np.array_equal(res.units, expected.units)
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awu(0, rad)                                      , awu(1, unitless )),  
            (awu(np.pi/2, rad)                                , awu(0, unitless )),  
            (awu(np.pi, rad)                                  , awu(-1, unitless )),  
            (awu([0, np.pi/2, np.pi], rad)                    , awu([1, 0, -1], unitless)),
            (awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad)  , awu([[1, np.sqrt(2)/2], [0, -np.sqrt(2)/2]], unitless)),
            (awu([1, 2, 3], [kg, km, rad])                   , ValueError),
            (awu(1, km)                                      , ValueError)
        ],
        ids=['Scalar 0', 'Scalar pi/2', 'Scalar pi', '1D Array', '2D Array', 'Non-Homogeneous', 'Non-Radian Units']
    )
    def test_cos(self, a, expected):
        """
            verifies that the cosine function works as expected.

            Limits:
            - Cosine function is only defined for ArrayWUnits instances with homogeneous units
            - Cosine function only accepts an input in radians
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.cos(a)
        elif a.size == 1:
            assert np.isclose(awu.cos(a).values, expected.values, atol=1e-14)
        else:
            res = awu.cos(a)
            assert np.allclose(res.values, expected.values, atol=1e-14)
            assert np.array_equal(res.units, expected.units)


    @pytest.mark.parametrize(
        'a, expected',
        [
            (awu(0, rad)                                      , awu(0, unitless )),  
            (awu(np.pi/4, rad)                                , awu(1, unitless )),  
            (awu(-np.pi/4, rad)                               , awu(-1, unitless )),  
            (awu([0, np.pi/4, -np.pi/4], rad)                , awu([0, 1, -1], unitless)),
            (awu([[0, np.pi/6], [np.pi/4, np.pi/3]], rad)    , awu([[0, 1/np.sqrt(3)], [1, np.sqrt(3)]], unitless)),
            (awu([1, 2, 3], [kg, km, rad])                    , TypeError),
            (awu(1, km)                                       , ValueError)
        ],
        ids=['Scalar 0', 'Scalar pi/4', 'Scalar -pi/4', '1D Array', '2D Array', 'Non-Homogeneous', 'Non-Radian Units']
    )
    def test_tan(self, a, expected):
        """
            verifies that the tangent function works as expected.

            Limits:
            - Tangent function is only defined for ArrayWUnits instances with homogeneous units
            - Tangent function only accepts an input in radians
            - Tangent is undefined at odd multiples of pi/2 (not tested here)

        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                awu.tan(a)
        elif a.size == 1:
            assert np.isclose(awu.tan(a).values, expected.values, atol=1e-14)
        else:
            res = awu.tan(a)
            assert np.allclose(res.values, expected.values, atol=1e-14)
            assert np.array_equal(res.units, expected.units)



class TestAltConstruction:
    """
        Collected tests for alternative construction methods for ArrayWUnits.
    """

    @pytest.mark.parametrize("shape", 
                             [5,
                            [2, 3],
                            [4, 1]],
                            ids=['Vector', '2D Matrix', '2D Vector'])
    def test_empty(self,shape):
        """
            Verfies tha we can initializes a new ArrayWUnits object of the given shape without 
            initializing value entries. Units are set as unitless.
        """
        a = awu.empty(shape)

        assert isinstance(a, awu)
        assert a.shape == (shape,) if isinstance(shape, int) else shape
        assert a.units == unitless
    
    def test_atleast1d(self):
        """
            DESC
        """
        pytest.skip()

    @pytest.mark.parametrize(
        'a, expected',
        [
            # scalar → 2D
            (awu(3, kg)                         , awu([[3]], [[kg]])),

            # 1D homogeneous → 2D
            (awu([1, 2, 3], km)                 , awu([[1, 2, 3]], [[km, km, km]])),

            # 2D homogeneous → unchanged
            (awu([[1, 2], [3, 4]], sec)         , awu([[1, 2], [3, 4]], [[sec, sec], [sec, sec]])),

            # 1D non-homogeneous → 2D
            (awu([1, 2, 3], [kg, km, sec])      , awu([[1, 2, 3]], [[kg, km, sec]])),

            # invalid argument type passed explicitly
            (awu(3, kg)                         , TypeError),
        ],
        ids=['Scalar', '1D homogeneous', '2D homogeneous', '1D non-homogeneous', 'Invalid Argument Type']
    )
    def test_atleast2d(self, a, expected):
        """
        Verifies that the atleast_2d operator functions as expected.

        Limits:
        - Input must be an ArrayWUnits object
        - Preserves values and units
        - Non-homogeneous units are supported
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                a.atleast_2d('not an awu')
        else:
            assert awu.atleast_2d(a) == expected

    def test_from_row_units(self):
        """
            Verifies that from_row_units gives every element of a row that
            row's unit, and that the unit array is new, so one element can be
            changed without touching its row.
        """
        values = np.arange(12.0).reshape(6, 2)
        row_units = (km,) * 3 + (km / sec,) * 3
        a = awu.from_row_units(values, row_units)

        assert np.array_equal(a.values, values)
        for row, unit in enumerate(row_units):
            assert all(element == unit for element in a.units[row])

        a.units[0, 1] = m
        assert a.units[0, 0] == km
        assert a.units[1, 1] == km

    def test_atleast3d(self):
        """
            DESC
        """
        pytest.skip()

    def test_eye(self):
        """
            DESC
        """
        pytest.skip()


class TestRepr:
    """
    Tests for __repr__ (and __str__) across all shape/unit branches.

    Branch map
    ----------
    A  Scalar                           size == 1
    B  Row vector                       1D → (1, n) after atleast_2d
    C  Column vector, val_cnt == 3      first row opens units immediately
    C  Column vector, val_cnt == 4      deferred first row; all intermediates hit
    C  Column vector, val_cnt >= 5      adds the plain else-intermediate before units start
    D  Matrix, val_cnt == 3             same logic as C but for 2D
    D  Matrix, val_cnt == 4             all intermediates in the matrix path
    D  Matrix, val_cnt >= 5             plain else-intermediate in matrix path
    E  Non-homogeneous                  homogeneous_units is False
    """

    @pytest.mark.parametrize(
        'a, expected_substr',
        [
            # A — scalar
            (awu(5, km),
             'km'),

            # B — row vector (1D → (1,n))
            (awu([1, 2, 3], km),
             '...'),

            # C — column vector, val_cnt == 3
            (awu([[1], [2], [3]], km),
             '⋮'),

            # C — column vector, val_cnt == 4 (deferred first row; start/middle/last branches)
            (awu([[1], [2], [3], [4]], km),
             '⋮'),

            # C — column vector, val_cnt == 5 (plain else-intermediate before units start)
            (awu([[1], [2], [3], [4], [5]], km),
             '⋮'),

            # D — matrix, val_cnt (cols) == 3  →  (3, 3)
            (awu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], km),
             '⋱'),

            # D — matrix, val_cnt (cols) == 4  →  (4, 4)
            (awu([[1, 2, 3, 4], [5, 6, 7, 8],
                  [9, 10, 11, 12], [13, 14, 15, 16]], km),
             '⋱'),

            # D — matrix, val_cnt (cols) == 5  →  (5, 5), adds plain else-intermediate
            (awu([[1, 2, 3, 4, 5]] * 5, km),
             '⋱'),

            # E — non-homogeneous
            (awu([1, 2], [km, sec]),
             'non-homogeneous'),
        ],
        ids=[
            'scalar',
            'row vector',
            'col vector val_cnt=3',
            'col vector val_cnt=4',
            'col vector val_cnt=5+',
            'matrix val_cnt=3',
            'matrix val_cnt=4',
            'matrix val_cnt=5+',
            'non-homogeneous',
        ]
    )
    def test_repr(self, a, expected_substr):
        """
        Verifies that __repr__ runs without error and contains the expected
        substring for each structural branch.
        """
        result = repr(a)
        assert isinstance(result, str) and len(result) > 0
        assert expected_substr in result
