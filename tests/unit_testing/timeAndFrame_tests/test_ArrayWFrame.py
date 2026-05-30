# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
from scarabaeus import ArrayWFrame as awf
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import numpy as np
import pytest
import os

km, kg, sec, hr,day, m, rad, unitless = scb.Units.get_units(['km', 'kg', 'sec', 'hr', 'day', 'm', 'rad', 'unitless'])
# load spice kernels
furnshKernelFilename = os.getcwd() + "/data/kernels/locked/locked_generic.tm"
scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)

# generate common frames
(J2000,ITRF93,ECLIPJ2000,IAUEARTH) = scb.Frame.generate_common_frames()
#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization():
    """
        Verifies that object is constructed correctly.
    """
    arrayWFrame = awf(50.0, km,  J2000)

    assert arrayWFrame._shape == (1,)
    assert arrayWFrame._frame == J2000
    assert arrayWFrame._quantity == awu(50.0, km)
 

class TestMathOperations:
        
    @pytest.mark.parametrize(
            'a, b, expected',
            [(awf(50.0, km,  J2000),                                             awf(20.0, km,  J2000),                          awf(70.0, km,  J2000)),
             (awf(20.0, km,  ITRF93),                                            awf(200.0, km,  ITRF93),                         awf(220, km,  ITRF93)),
             (awf(awu([100.0, 20, 43],m/sec),  J2000),                           awf(20.0, m/sec,  J2000),                       awf(awu([120,40,63], m/sec),  J2000)),
             (awf(awu([100.0, 20, 43],kg),  ECLIPJ2000),                         awf(awu([30,40,50], kg),  ECLIPJ2000),          awf(awu([130,60,93], kg),  ECLIPJ2000)),
             (awf(awu([100.0, 20, 43],[kg, m ,sec]),  ECLIPJ2000),               awf(awu([30,40,50], [kg, m, sec]), ECLIPJ2000), awf(awu([130,60,93],[kg, m, sec]),  ECLIPJ2000))],
            ids = ['Scaler J2000','Scaler ITRF93', ' Matrix + scaler', ' Matrix same unit', 'Diff Units Matrix']
    )
    def test_addition(self, a, b, expected):
        """
            Verifies that addition works correctly.
        """
        if a._shape == (1,):
            assert a + b == expected
        else:
            assert all(a + b == expected)



    @pytest.mark.parametrize(
            'a, expected',
            [(awf(50.0, km,  J2000),                                             awf(-50.0, km,  J2000)),                        
             (awf(20.0, km,  ITRF93),                                            awf(-20.0, km,  ITRF93)),                       
             (awf(awu([100.0, 20, 43],m/sec),  J2000),                           awf([-100.0, -20, -43], m/sec,  J2000)),                
             (awf(awu([-100.0, -20, -43],kg),  ECLIPJ2000),                      awf(awu([100.0, 20, 43], kg),  ECLIPJ2000)),        
             (awf(awu([100.0, 20, 43],[kg, m ,sec]),  ECLIPJ2000),               awf(awu([-100.0, -20, -43], [kg, m, sec]), ECLIPJ2000))],
            ids = ['Scaler J2000','Scaler ITRF93', ' Matrix + scaler', ' Matrix same unit', 'Diff Units Matrix']
    )
    def test_negative(self, a, expected):
        """
            Verifies that negation works correctly.
        """
        if a._shape == (1,):
            assert -a == expected
        else:
            assert all(-a == expected)

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awf(5.0, km,  J2000),                       2,      awf(25.0, km**2,  J2000)),
             (awf(2.0, km,  ITRF93),                      2,        awf(4, km**2,  ITRF93)),
             (awf(awu([10.0, 20, 3],m/sec),  J2000),      2,      awf(awu([100,400,9], m**2/sec**2),  J2000)),
             ],
            ids = ['Scaler J2000','Scaler ITRF93', ' Matrix + scaler']
    )
    def test_power(self, a, b, expected):
        """
            Verifies that power works correctly.
        """
        assert a**b == expected

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awf(50.0, km,  J2000),                                 awf(20.0, km,  J2000),                         awf(1000.0, km**2,  J2000)),
             (awf(20.0, km,  ITRF93),                                awf(200.0, kg,  ITRF93),                       awf(4000, km*kg,  ITRF93)),
             (awf(awu([100.0, 20, 43],m/sec),  J2000),               awf(20.0, m,  J2000),                          awf(awu([2000,400,860], m**2/sec),  J2000)),
             (awf(awu([100.0, 20, 4],kg),  ECLIPJ2000),             awf(awu([30,40,50], kg),  ECLIPJ2000),          awf(awu([3000,800,200], kg**2),  ECLIPJ2000)),
             (awf(awu([100.0, 20, 4],[kg, m ,sec]),  ECLIPJ2000),   awf(awu([30,40,50], [km, m, m]), ECLIPJ2000),   awf(awu([3000,800,200],[kg*km, m**2, sec*m]),  ECLIPJ2000))],
            ids = ['Same unit J2000','Diff units ITRF93', ' Matrix & scaler', ' Matrix same unit', 'Diff Units Matrix']
    )
    def test_multiplication(self, a, b, expected):

        """
            Verifies that multiplication works correctly.
        """
        if a._shape == (1,):
            assert a * b == expected
        else:
            assert all(a * b == expected)

    @pytest.mark.parametrize(
            'a, b, expected',
            [(awf(awu([1, 2, 3], kg), J2000),  awf(awu([4, 5, 6]      , kg), J2000), awf(awu(32, kg**2), J2000)),
             (awf(awu([[1, 2, 3]], kg), J2000), awf(awu([[4], [5], [6]], kg), J2000), awf(awu(32, kg**2), J2000)),
             (awf(awu([[1, 0, 0],                   # a
                       [0, 2, 0],
                       [0, 0, 3]]      , kg    ), J2000),
              awf(awu([[4],  [5],  [6]], kg    ), J2000),   # b
              awf(awu([[4], [10], [18]], kg**2), J2000)),   # expected
             (awf(awu([[1, 0, 0],                   # a
                   [0, 2, 0],
                   [0, 0, 3]]  , kg    ), ITRF93),
              awf(awu([[4, 0, 0],                   # b
                   [0, 5, 0],
                   [0, 0, 6]]  , kg    ), ITRF93),
              awf(awu([[4,  0,  0],                 # expectd
                   [0, 10,  0],
                   [0, 0 , 18]], kg**2), ITRF93)),],
             ids = ['Hom Row * Hom Row', 'Hom Row * Hom Column', 'Hom Mat * Hom Column',
                    'Hom Mat * Hom Mat']
    )
    def test_matrix_multiplication(self, a, b, expected, request):
        """
            Verifies that matrix multiplication works correctly.
        """
        if request.node.callspec.id == 'Hom Row * Hom Column': pytest.skip('SKIP FOR NOW')

        if (a@b)._quantity.size ==1 :
            assert a @ b == expected
        else:
            assert all(a @ b == expected)

        
    
    @pytest.mark.parametrize(
        'a, b, expected',
        [   (awf(50.0, km,  J2000),                                             awf(20.0, km,  J2000),                          awf(30.0, km,  J2000)),
            (awf(20.0, km,  ITRF93),                                            awf(200.0, km,  ITRF93),                        awf(-180, km,  ITRF93)),
            (awf(awu([100.0, 20, 43],m/sec),  J2000),                           awf(20.0, m/sec,  J2000),                       awf(awu([80,0,23], m/sec),  J2000)),
            (awf(awu([100.0, 20, 83],kg),  ECLIPJ2000),                         awf(awu([30,40,50], kg),  ECLIPJ2000),          awf(awu([70,-20,33], kg),  ECLIPJ2000)),
            (awf(awu([100.0, 20, 83],[kg, m ,sec]),  ECLIPJ2000),               awf(awu([30,40,50], [kg, m, sec]), ECLIPJ2000), awf(awu([70,-20,33],[kg, m, sec]),  ECLIPJ2000))],
        ids = ['Scaler J2000','Scaler ITRF93', ' Matrix + scaler', ' Mtrix same unit', 'Diff Units Matrix']
    )
    def test_subtraction(self, a, b, expected):
        """
            Verifies that subtraction works correctly.
        """
        if a._shape == (1,):
            assert a - b == expected
        else:
            assert all(a - b == expected)
    
    @pytest.mark.parametrize(
        'a, b, expected',
        [(awf(awu(2        , kg), J2000),           awf(awu(2        , kg), J2000),                awf(awu(1        , None              ), J2000)),
         (awf(awu([1, 2, 3], kg), J2000),           awf(awu([1, 2, 3], kg), J2000),                awf(awu([1, 1, 1], None              ), J2000)),
         (awf(awu(1        , kg), ITRF93),          awf(awu(1        , km), ITRF93),               awf(awu(1        , kg/km             ), ITRF93)),
         (awf(awu([1, 2, 3], kg), ITRF93),          awf(awu([1, 2, 3], km), ITRF93),               awf(awu([1, 1, 1], kg/km             ), ITRF93)),
         (awf(awu(1        , kg*km*sec**-2),J2000), awf(awu(1        , km**2*kg), J2000),          awf(awu(1        , sec**-2*km**-1    ), J2000)),
         (awf(awu([1, 2]   , [kg, km]), J2000),     awf(awu([1, 2]   , [km**-3, sec**-1]), J2000), awf(awu([1, 1]   , [kg*km**3, km*sec]), J2000))],
        ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix', 'Compound Units', 'Comp Units Matrix']
)
    def test_division(self, a, b, expected):
        """
            Verifies that division works correctly.
        """
        if a._shape == (1,):
            assert (a / b) == expected
        else:
            assert all((a / b) == expected)

    @pytest.mark.parametrize(
        'a, b, case',
        [
            (awf(awu(1, kg), J2000),          awf(awu(1, kg), J2000),          'a = b'),
            (awf(awu([1,2,3], kg), J2000),    awf(awu([1,2,3], kg), J2000),    'a = b'),
            (awf(awu(1, km), J2000),          5,                               'error')
        ],
        ids=['Same Units', 'Same Units Matrix', 'Not AWF']
    )
    def test_equality(self, a, b, case):
        """
            Verifies that equality comparison works correctly.
        """
        match case:
            case 'a = b':
                # ensure that a is to b
                if a._shape == (1,):
                    assert a == b
                else:
                    assert all(a == b)
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a == b
    
    @pytest.mark.parametrize(
    'a, b, case',
    [
        (awf(awu(1, kg), J2000),           awf(awu(2, kg), J2000),           'a != b'),
        (awf(awu([1,2,3], kg), J2000),     awf(awu([1,1,1], kg), J2000),     'a != b'),
        (awf(awu(1, kg), J2000),           awf(awu(2, km), J2000),           'a != b'),
        (awf(awu([1,2,3], kg), J2000),     awf(awu([1,1,1], km), J2000),     'a != b'),
        (awf(awu(1, km), J2000),           5,                                'error')
    ],
    ids=[
        'Same Units',
        'Same Units Matrix',
        'Different Units',
        'Diff Units Matrix',
        'Not AWF'
    ]
)
    def test_inequality(self, a, b, case):
        """
            Verifies that inequality comparison works correctly.
        """
        match case:
            case 'a != b':
                # ensure that a is not equal to b
                if a._shape == (1,):
                    assert a != b
                else:
                    assert any(a != b)
            
            case 'error':
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a != b
    
    @pytest.mark.parametrize(
    'a, b, case',
    [
        (awf(awu(2, kg), J2000), awf(awu(1, kg), J2000), 'a > b'),
        (awf(awu(1, kg), J2000), awf(awu(1, rad), J2000), 'error')
    ],
    ids=['Same Units', 'Different Units']
    )
    def test_greater_than(self, a, b, case):
        """
            Verifies that greater than comparison works correctly.
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
        [
            (awf(awu(2, kg), J2000), awf(awu(1, kg), J2000), 'a > b'),
            (awf(awu(1, kg), J2000), awf(awu(1, kg), J2000), 'a = b'),
            (awf(awu(1, kg), J2000), awf(awu(1, rad), J2000), 'error')
        ],
        ids=[
            'Greater Than (Same Units)',
            'Equal To (Same Units)',
            'Different Units'
        ]
    ) 
    def test_greater_than_equal(self, a, b, case):
        """
            Verifies that greater than or equal comparison works correctly.
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
    [
        (awf(awu(1, kg), J2000), awf(awu(2, kg), J2000), 'a < b'),
        (awf(awu(1, kg), J2000), awf(awu(2, rad), J2000), 'error')
    ],
    ids=['Same Units', 'Different Units']
    )
    def test_less_than(self, a, b, case):
        """
            Verifies that less than comparison works correctly.
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
        [
            (awf(awu(1, kg), J2000), awf(awu(2, kg), J2000), 'a < b'),
            (awf(awu(1, kg), J2000), awf(awu(1, kg), J2000), 'a = b'),
            (awf(awu(1, kg), J2000), awf(awu(1, rad), J2000), 'error')
        ],
        ids=[
            'Less Than (Same Units)',
            'Equal To (Same Units)',
            'Different Units'
        ]
    )
    def test_less_than_equal(self, a, b, case):
        """
            Verifies that less than or equal comparison works correctly.
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
        [
            (awf(awu([1, 2, 3], kg), J2000),
            awf(awu([4, 0, 1], kg), J2000),
            awf(awu([2, 11, -8], kg**2), J2000)),

            (awf(awu([1, 2, 3], kg), J2000),
            awf(awu([4, 0, 1], km), J2000),
            awf(awu([2, 11, -8], kg*km), J2000)),

            (awf(awu([1, 2, 3], kg*km*sec**-2), J2000),
            awf(awu([4, 0, 1], km**2*kg), J2000),
            awf(awu([2, 11, -8], km**3*kg**2/sec**2), J2000)),
        ],
        ids=['Same Units Matrix', 'Diff Units Matrix', 'Comp Units Matrix']
    )
    def test_cross_product(self,a,b, expected):
        """
            Verfies that the cross-product operator functions as expected.

            Limits:
            - Cross-product operation is only defined between two ArrayWUnits objects.
            - Cross-product operation is only defined for ArrayWUnits with homogeneous units.
            - Cross-product operation is only defined for vectorial ArrayWUnits objects.

        """
        assert awf.cross(a,b) == expected  
    

    @pytest.mark.parametrize(
        'to_sum, expected',
        [
            (awf(awu(1, kg), J2000), awf(awu(1, kg), J2000)),
            (awf(awu([1, 2, 3], kg), J2000), awf(awu(6, kg), J2000)),
            (awf(awu([1, 2, 3], [km, kg, rad]), J2000), 'error')
        ],
        ids=['Single', 'Matrix', 'Non-Homogeneous']
    )
    def test_summation(self, to_sum, expected):
        """
        Verifies that the summation operator functions as expected for ArrayWFrame objects.

        Limits:
        - Only homogeneous units can be summed
        - Frame is preserved in the result
        """

        if isinstance(expected, str):
            # Non-homogeneous AWU within AWF → cannot sum
            with pytest.raises(Exception):
                to_sum.summation()
        else:
            assert to_sum.summation() == expected

    @pytest.mark.parametrize(
    'to_norm, expected',
    [
        (awf(awu([1, 2, 3], kg), J2000), awf(awu(np.sqrt(14), kg), J2000)),
        (awf(awu([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], kg), J2000), awf(awu(np.sqrt(285), kg), J2000)),
        (awf(awu([1, 2], [kg, km]), J2000), awf(awu(np.sqrt(5), unitless), J2000))
    ],
    ids=['Vector Norm', 'Matrix Norm', 'Non-Homogeneous Units']
    )
    def test_norm(self, to_norm, expected):
        """
            Verifies that norm calculation works correctly.
        """
        if not isinstance(expected, str):
            assert to_norm.norm() == expected
        else:
            # can't norm non-homogeneous units
            with pytest.raises(Exception):
                to_norm.norm()
    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu([3, 4], kg), J2000), awf(awu([3/5, 4/5], kg), J2000)),
            (awf(awu([1, 2, 2], km), J2000), awf(awu([[1/3, 2/3, 2/3]], km), J2000))
        ],
        ids=['2x1 vector', '3x1 vector']
    ) 
    def test_unitary(self, a, expected):
        """
            Verifies that unitary vector calculation works correctly.

            Limits:
            - Cannot unitize a non-1-D ArrayWUnits vector.
            - Cannot unitize an ArrayWUnits objects with non-homogeneous units.
        """
        assert a.unitary() == expected

     
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(1, kg), J2000),
            awf(awu(1, kg), J2000)),

            (awf(awu([1, 2, 3], km), J2000),
            awf(awu([[1], [2], [3]], km), J2000)),

            (awf(awu([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], km/sec), J2000),
            awf(awu([[1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9]], km/sec), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix']
    )
    def test_transpose(self, a, expected):
        """
        Verifies that the transpose operator functions as expected
        for ArrayWFrame objects.

        Limits:
        - Transpose operation defined only for scalar, 1D, and 2D ArrayWFrame instances
        - Frame is preserved
        - Units are preserved
        """
        assert a.transpose() == expected

    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu([[1, 2],[3, 4]], km), J2000),
            awf(awu([[[-2, 1],[1.5, -0.5]]], km), J2000)),

            (awf(awu([[1, 0, 2],
                    [0, 1, 1],
                    [1, 0, 1]], km/sec), J2000),
            awf(awu([[[1, 0, -2],
                    [1, 1, -1],
                    [-1, 0, 1]]], km/sec), J2000)),
        ],
        ids=['2x2', '3x3']
    )
    def test_inverse(self, a, expected):
        """
        Verifies that the inverse operator functions as expected for ArrayWFrame objects.

        Limits:
        - Only defined for square matrices
        - Only defined for homogeneous units
        - Frame is preserved in the result
        """

        assert a.inverse() == expected
    
    @pytest.mark.parametrize(
    'a, expected',
    [
        (awf(awu([[1, 2, 3],
                  [4, 5, 6]], km), J2000),
         awf(awu([[-17/18,  4/9],
                  [-1/9,   1/9],
                  [13/18, -2/9]], km), J2000)),

        (awf(awu([[3],
                  [4]], kg), J2000),
         awf(awu([[0.12, 0.16]], kg), J2000)),

        (awf(awu([[1, 2],
                  [3, 4]], m/sec), J2000),
         awf(awu([[[-2, 1],
                   [1.5, -0.5]]], m/sec), J2000)),

        (awf(awu([[1, 2],
                  [2, 4]], km/sec), J2000),
         awf(awu([[[1, 0, -2],
                   [1, 1, -1],
                   [-1, 0, 1]]], km/sec), J2000)),
    ],
    ids=['Rectangular Matrix', 'Vector', 'Square Matrix', 'Singular Matrix']
    )
    def test_pseudo_inverse(self, a, expected):
        """
        Verifies that the pseudo-inverse operator functions as expected for ArrayWFrame objects.

        Limits:
        - Only defined for ArrayWFrame objects with homogeneous units
        - Frame is preserved in the result
        """

        assert a.pseudo_inverse() == expected

        
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(1, kg), J2000),
            awf(awu(np.e, kg), J2000)),

            (awf(awu([1, 2, 3], km), J2000),
            awf(awu([np.e, np.e**2, np.e**3], km), J2000)),

            (awf(awu([[1, 2], [3, 4]], km/sec), J2000),
            awf(awu([[np.e, np.e**2], [np.e**3, np.e**4]], km/sec), J2000)),

            (awf(awu([1, 2, 3], [kg, km, sec]), J2000),
            awf(awu([np.e, np.e**2, np.e**3], [kg, km, sec]), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous']
    )
    def test_exp(self, a, expected):
        """
            Verifies that the exponential function works element-wise
            for ArrayWFrame objects, preserving both units and frame.
        """
        res = awf.exp(a)

        assert res == expected



    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(1, kg), J2000),
            awf(awu(2, kg), J2000)),

            (awf(awu([1, 2, 3], km), J2000),
            awf(awu([2, 4, 8], km), J2000)),

            (awf(awu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], km/sec), J2000),
            awf(awu([[2, 4, 8], [16, 32, 64], [128, 256, 512]], km/sec), J2000)),

            (awf(awu([1, 2, 3], [kg, km, sec]), J2000),
            awf(awu([2, 4, 8], [kg, km, sec]), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_exp2_awf(self, a, expected):
        """
            Verifies that the exp2 function works as expected for ArrayWFrame objects.

        """
        if a._shape == (1,):
            assert awf.exp2(a) == expected
        else:
            assert all(awf.exp2(a) == expected)


    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(1, kg), J2000),
            awf(awu(0, kg), J2000)),

            (awf(awu([1, 10, 100], km), J2000),
            awf(awu([0, 1, 2], km), J2000)),

            (awf(awu([[1, 10], [100, 1000]], km/sec), J2000),
            awf(awu([[0, 1], [2, 3]], km/sec), J2000)),

            (awf(awu([1, 10, 1000], [kg, km, sec]), J2000),
            awf(awu([0, 1, 3], [kg, km, sec]), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log10(self, a, expected):
        """
            Verifies that the log10 function works as expected for ArrayWFrame objects.
        """
        if a._shape == (1,):
            assert awf.log10(a) == expected
        else:
            assert all(awf.log10(a) == expected)

    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(2, kg), J2000),
            awf(awu(1, kg), J2000)),

            (awf(awu([2, 4, 8], km), J2000),
            awf(awu([1, 2, 3], km), J2000)),

            (awf(awu([[2, 4, 8],
                    [16, 32, 64],
                    [128, 256, 512]], km/sec), J2000),
            awf(awu([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], km/sec), J2000)),

            (awf(awu([2, 4, 8], [kg, km, sec]), J2000),
            awf(awu([1, 2, 3], [kg, km, sec]), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log2(self, a, expected):
        """
            Verifies that the log2 function works as expected for ArrayWFrame objects.

        """
        if a._shape == (1,):
            assert awf.log2(a) == expected
        else:
            assert all(awf.log2(a) == expected)

    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(1, kg), J2000),
            awf(awu(0, kg), J2000)),

            (awf(awu([1, np.e, np.e**2], km), J2000),
            awf(awu([0, 1, 2], km), J2000)),

            (awf(awu([[1, np.e, np.e**2],
                    [np.e**3, np.e**4, np.e**5],
                    [np.e**6, np.e**7, np.e**8]], km/sec), J2000),
            awf(awu([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]], km/sec), J2000)),

            (awf(awu([1, np.e, np.e**2], [kg, km, sec]), J2000),
            awf(awu([0, 1, 2], [kg, km, sec]), J2000)),
        ],
        ids=['Scalar', '1D Matrix', '2D Matrix', 'Non-Homogeneous 1D']
    )
    def test_log(self, a, expected):
        """
            Verifies that the natural logarithm function works as expected
            for ArrayWFrame objects.

        """
        if a._shape == (1,):
            assert awf.log(a) == expected
        else:
            assert np.array_equal(awf.log(a), expected)

    
    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(0, rad), J2000)                                      , awf(awu(0,unitless), J2000)),  
            (awf(awu(np.pi/2, rad), J2000)                                , awf(awu(1,unitless), J2000)),  
            (awf(awu(np.pi, rad), J2000)                                  , awf(awu(0,unitless), J2000)),  
            (awf(awu([0, np.pi/2, np.pi], rad), J2000)                    , awf(awu([0, 1, 0],unitless), J2000)),
            (awf(awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad), J2000)   , awf(awu([[0, np.sqrt(2)/2], [1, np.sqrt(2)/2]],unitless), J2000))
        ],
        ids=['Scalar 0', 'Scalar pi/2', 'Scalar pi', '1D Array', '2D Array']
    )
    def test_sin(self, a, expected):
        """
            verifies that the sine function works as expected.

            Limits:
            - Sine function is only defined for ArrayWUnits instances with homogeneous units
            - Sine function only accepts an input in radians
        """
        if a._shape == (1,):
            assert np.isclose(awf.sin(a)._quantity.values, expected._quantity.values, atol=1e-14)
            assert a._frame == expected._frame
            assert awf.sin(a)._quantity.units == expected._quantity.units
        else:
            res = awf.sin(a)
            assert np.allclose(res._quantity.values, expected._quantity.values, atol=1e-14)
            assert res.frame == expected.frame
            assert np.array_equal(res._quantity.units, expected._quantity.units)


    @pytest.mark.parametrize(
            'a, expected',
            [
                (awf(awu(0, rad), J2000)                                      , awf(awu(1, unitless), J2000)),  
                (awf(awu(np.pi/2, rad), J2000)                                , awf(awu(0, unitless), J2000)),  
                (awf(awu(np.pi, rad), J2000)                                  , awf(awu(-1, unitless), J2000)),  
                (awf(awu([0, np.pi/2, np.pi], rad), J2000)                    , awf(awu([1, 0, -1], unitless), J2000)),
                (awf(awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad), J2000)  , awf(awu([[1, np.sqrt(2)/2], [0, -np.sqrt(2)/2]], unitless), J2000))
            ],
            ids=['Scalar 0', 'Scalar pi/2', 'Scalar pi', '1D Array', '2D Array']
        )
    def test_cos(self, a, expected):
        """
            verifies that the cosine function works as expected.

            Limits:
            - Cosine function is only defined for ArrayWUnits instances with homogeneous units
            - Cosine function only accepts an input in radians
        """
        if a._shape == (1,):
            assert np.isclose(awf.cos(a)._quantity.values, expected._quantity.values, atol=1e-14)
            assert a._frame == expected._frame
            assert awf.cos(a)._quantity.units == expected._quantity.units
        else:
            res = awf.cos(a)
            assert np.allclose(res._quantity.values, expected._quantity.values, atol=1e-14)
            assert res.frame == expected.frame
            assert np.array_equal(res._quantity.units, expected._quantity.units)

    @pytest.mark.parametrize(
        'a, expected',
        [
            (awf(awu(0, rad), J2000)                                      , awf(awu(0, unitless), J2000)),  
            (awf(awu(np.pi/4, rad), J2000)                                 , awf(awu(1, unitless), J2000)),  
            (awf(awu(-np.pi/4, rad), J2000)                                , awf(awu(-1, unitless), J2000)),  
            (awf(awu([0, np.pi/4, -np.pi/4], rad), J2000)                  , awf(awu([0, 1, -1], unitless), J2000)),
            (awf(awu([[0, np.pi/6], [np.pi/4, np.pi/3]], rad), J2000)      , awf(awu([[0, 1/np.sqrt(3)], [1, np.sqrt(3)]], unitless), J2000))
        ],
        ids=['Scalar 0', 'Scalar pi/4', 'Scalar -pi/4', '1D Array', '2D Array']
    )
    def test_tan(self, a, expected):
        """
            Verifies that tangent function works correctly.

            Limits:
            - Tangent function is only defined for ArrayWUnits instances with homogeneous units
            - Tangent function only accepts an input in radians
            - Tangent is undefined at odd multiples of pi/2 (not tested here)
            
        """
        if a._shape == (1,):
            assert np.isclose(awf.tan(a)._quantity.values, expected._quantity.values, atol=1e-14)
            assert a._frame == expected._frame
            assert awf.tan(a)._quantity.units == expected._quantity.units
        else:
            res = awf.tan(a)
            assert np.allclose(res._quantity.values, expected._quantity.values, atol=1e-14)
            assert res.frame == expected.frame
            assert np.array_equal(res._quantity.units, expected._quantity.units)
    
@pytest.mark.parametrize(
        'a, b, c, expected',
        [
            (awf(awu([1,2,3], kg), J2000),  ITRF93,  scb.EpochArray([1096804869.1823437],'TDB'),  awf(awu([1.3940419390786, 1.7427464275878, 3.0032452382748], kg),  ITRF93)),
            (awf(awu([1,2,3], m), ITRF93),  J2000,   scb.EpochArray([1096804869.1823437],'TDB'),  awf(awu([0.5601165549571, 2.1672078360427, 2.998246094012 ], m),   J2000)),
            (awf(awu([1,2,3], kg), J2000), J2000,    scb.EpochArray([1096804869.1823437],'TDB'),   awf(awu([1,2,3], kg), J2000))
        ],
        ids=[
            'ITRF93 to J2000',
            'J2000 to ITRF93',
            'J2000 to J2000'
        ]
)
def test_convertor(a, b,c, expected):
    """
        Verifies conversion operation.

        Limits:
        - only works with TDB epochs

    """
    a.convert_to(b,c)
    assert all(np.isclose(a._quantity.values , expected._quantity.values ,atol=1e-15))
    assert a._frame == expected._frame
    assert np.array_equal(a._quantity.units, expected._quantity.units)


