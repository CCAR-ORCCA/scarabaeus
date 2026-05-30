# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest
import numpy as np

N, kg, sec, km, dam, m, mm, mu, rad, mrad, deg, g, hr = scb.Units.get_units(['N', 'kg', 'sec', 'km', 'dam', 'm', 'mm', 'mu', 'rad', 'mrad', 'deg', 'g', 'hr'])

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
@pytest.mark.parametrize(
    'dimensions, scales, expected',
    [(scb.Dimensions([0, 0, 0, 0]), [0, 0, 0, 0], 'unitless'     ),
     (scb.Dimensions([1, 0, 0, 0]), [1, 0, 0, 0], 'g'            ),
     (scb.Dimensions([0, 1, 0, 0]), [0, 1, 0, 0], 'm'            ),
     (scb.Dimensions([0, 0, 1, 0]), [0, 0, 1, 0], 'sec'          ),
     (scb.Dimensions([0, 0, 0, 1]), [0, 0, 0, 1], 'rad'          ),
     (scb.Dimensions([1, 1, 1, 1]), [1, 1, 1, 1], 'g*m*sec*rad')],
    ids = ['Unitless', 'Base Mass', 'Base Length', 'Base Time', 'Base Angle', 'Compound']
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

    # verify that [unitless] flag is set properly
    if all(dimensions.powers == [0, 0, 0, 0]):
        assert unit.unitless == True
    else:
        assert unit.unitless == False

@pytest.mark.parametrize(
        'requested_units, expected_units',
        [('km', km),
         (['km', 'sec', 'mrad', 'N'], [km, sec, mrad, N]),
         ('common', [kg, km, sec, rad, deg, N, mu])],
         ids = ['Single Request', 'Multiple Requests', 'Unit Set Request']
)
def test_get_units(requested_units, expected_units):
    """
        Verifies that the get_units class method functions correctly.
    """
    assert np.array_equal(scb.Units.get_units(requested_units) , expected_units)

class TestMathOperations():

    @pytest.mark.parametrize(
            'a, b, expected',
            [(m, m, m**2),
             (dam**3*sec**-2, km**-2, mm/sec**2),
            # a
            (np.array([[km       , N            , kg     ],
                       [1        , 1            , 1      ]]),
            # b
            np.array([[sec**-2  , sec**2*kg**-1, m**2   ],
                      [m        , m            , m      ]]), 
            # expected
            np.array([[km/sec**2, m            , kg*m**2],
                      [m        , m            , m      ]])),
            (kg, np.eye(3), np.array([[kg, 0 , 0 ],
                                    [0 , kg, 0 ],
                                    [0 , 0 , kg]]))],
            ids = ['Scalar Units', 'Scalar Compound', 'Matrix Units', 'Unit Times Matrix']
    )
    def test_multiplication(self, a, b, expected):
        """
            Verifies that the multiplication operator functions as expected.
        """
        if isinstance(a, scb.Units) and isinstance(b, scb.Units):
            assert (a * b) == expected
        else:
            assert ((a * b) == expected).all()

    @pytest.mark.parametrize(
            'a, b, expected',
            [(m**2, m, m),
             ( mm/sec**2, km**-2, mm**3*sec**-2),
            # a
            (np.array([[km/sec**2, m            , kg*m**2],
                      [m        , m            , m      ]]),
            #b
            np.array([[km       , N            , kg     ],
                       [1        , 1            , 1      ]]),
            # expected 
            np.array([[sec**-2  , sec**2*kg**-1, m**2   ],
                      [m        , m            , m      ]]))],

            ids = ['Scalar Units', 'Scalar Compound', 'Matrix Units']
    )
    def test_division(self, a, b, expected):
        """
            Verifies that the division operator functions as expected.
        """
        if isinstance(a, scb.Units) and isinstance(b, scb.Units):
            assert (a / b) == expected
        else:
            assert ((a / b) == expected).all()

    @pytest.mark.parametrize(
            'to_raise, raised_to, expected',
            [(kg,  2,  kg**2 ),
             (m/sec, 2, m**2/sec**2)],
             ids = ['Scalar', 'Non-Hom Matrix']
    )
    def test_power(self, to_raise, raised_to, expected):
        """
            Verifies that the power operator functions as expected.
        """

        assert to_raise**raised_to == expected

    @pytest.mark.parametrize(
        'a, b, case',
        [(kg,  kg, 'a = b' ),
         (sec, hr, 'a != b'),
         (kg, m ,'a != b' )],
        ids = ['same unit', 'same type diff unit', 'different types']
    )
    def test_equal(self, a, b, case):
        """
            Verifies that the equality operator functions as expected.
        """
        match case:
            case 'a = b':
                # ensure that a is to b
                assert a == b
            
            case 'error':
                assert a != b
    
    @pytest.mark.parametrize(
        'a, b, case',
        [(kg,  kg, 'a == b' ),
         (m**2, km**2 , 'a != b' ),
         (sec, hr, 'a != b'),
         (kg, m ,'a != b' )],
        ids = ['same unit', 'same type diffrent dim', 'same type diff unit', 'diffrent types']
    )
    def test_not_equal(self, a, b, case):
        """
            Verifies that the inequality operator functions as expected.
        """
        match case:
            case 'a != b':
                # ensure that a is to b
                assert a != b
            
            case 'a == b':
                assert (a != b) == False

@pytest.mark.parametrize(
        'option, value',
        [('verbose', True),
         ('unit_conventions', ['mg', 'Gm', 'ns', 'deg'])],
        ids = ['Verbose', 'Unit Conventions']
)
def test_print_options(option, value):
    """
        Verifies that unit printing options are respected.
    """
    # NOTE: need to do these with eval
    # scb.Units.set_print_options(verbose = value)
    # scb.Units.set_print_options(unit_conventions = value)
    pytest.skip('Not implemented in Units yet')