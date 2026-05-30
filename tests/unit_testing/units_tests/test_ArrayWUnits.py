# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

# # create all construction instance ID's and defaults (testing with different units for each instance as well)
# inst_ids  = ['single int', 'single float',
#              'array (homo units)', 'array (hetero units)',
#              'single (unitless)', 'array (unitless)']

# # valid singular input constructions
# single_int_def   = {'values' : 1                    , 'units' : km     , 'physical_type' : None}
# single_flt_def   = {'values' : 1.1                  , 'units' : kg     , 'physical_type' : None}

# # valid n-size input constructions
# non_hom = td.Generic.test_non_hom_units_array

# arr_hom_def      = {'values' : np.array([1, 1.1, 0]), 'units' : sec    , 'physical_type' : None}
# arr_non_hom_def  = {'values' : np.array([1, 1.1, 0]), 'units' : non_hom, 'physical_type' : None}

# # valid single and n-size inputs with no units
# unitless = scb.UnitsArray(None, None)
# mat2x2   = np.array([[1, 1.1], [2, 2.2]])

# sngle_untlss_def = {'values' : 1                    , 'units' : unitless, 'physical_type' : None}
# arr_untlss_def   = {'values' : mat2x2               , 'units' : unitless, 'physical_type' : None}

# # consolidate instance values
# inst_defs = (single_int_def, single_flt_def,
#              arr_hom_def, arr_non_hom_def,
#              sngle_untlss_def, arr_untlss_def)

# #---------------------------#
# #          Fixtures         #
# #---------------------------#
# @pytest.fixture(params = inst_defs, ids = inst_ids)
# def init_arr_w_units(request):
#     """
#         Fixture for a ArrayWUnits object. Tests the ArrayWUnits constructor.
        
#         Returns a factory method with provided values as default arguments.
#     """
#     def make_class(       # default values
#             values        = request.param['values'],
#             units         = request.param['units'],
#             physical_type = request.param['physical_type']):

#          return scb.ArrayWUnits(values, units, physical_type)

#     return make_class

# @pytest.fixture
# def single_arr_w_units():
#     """
#         Non-parametrized fixture to be used by tests that don't require
#         multiple instances.
#     """
#     def make_class(       # default values
#             values        = single_int_def['values'],
#             units         = single_int_def['units'],
#             physical_type = single_int_def['physical_type']):

#          return scb.ArrayWUnits(values, units, physical_type)

#     return make_class
# #---------------------------#
# #           Tests           #
# #---------------------------#
# def test_typing(init_arr_w_units):
#     """
#         Verify initialization works.
        
#         Test will fail if:
#           1) The class fails to initialize given the default inputs
#     """
#     assert isinstance(init_arr_w_units(), scb.ArrayWUnits)

# def test_properties(request, init_arr_w_units):
#     """
#         Verify all properties exist and are defined correctly using the TestWide prop_checker
#         method.
        
#         Test will fail if:
#           1) all properties aren't defined in the 'checks' dict (unless disabled by
#              the 'ignore_prop_count_check' argument)
#           2) any property doesn't match the expected value
        
#         If you want to skip one, set its value as "SKIP". This will still check the properties
#         you have defined (printed in the TEST RESULTS terminal) while not letting the entire
#         test_properties() test pass. Instead it will skip, making it obvious in the testing
#         window that this isn't complete.
#     """
#     #---------------------------------------------------------#
#     # define all properties/expected values per instance here #
#     #---------------------------------------------------------#
#     ## single int instance ##
#     single_int_expect = {
#         'values'        : single_int_def['values'],
#         'units'         : single_int_def['units'],
#         'shape'         : (),
#         'size'          : 1,
#         'physical_type' : 'Position'
#     }

#     ## single float instance ##
#     single_flt_expect = {
#         'values'        : single_flt_def['values'],
#         'units'         : single_flt_def['units'],
#         'shape'         : (),
#         'size'          : 1,
#         'physical_type' : 'Mass'
#     }

#     ## array with homogeneous units instance ##
#     arr_homo_expect = {
#         'values'        : arr_hom_def['values'],
#         'units'         : arr_hom_def['units'],
#         'shape'         : (3,),
#         'size'          : 3,
#         'physical_type' : 'Time'
#     }

#     ## array with heterogeneous units instance ##
#     arr_hetero_expect = {
#         'values'        : arr_non_hom_def['values'],
#         'units'         : arr_non_hom_def['units'],
#         'shape'         : (3,),
#         'size'          : 3,
#         'physical_type' : None      # should not have a physical type
#     }

#     ## single unitless instance ##
#     single_unitless_expect = {
#         'values'        : sngle_untlss_def['values'],
#         'units'         : sngle_untlss_def['units'],
#         'shape'         : (),
#         'size'          : 1,
#         'physical_type' : None
#     }

#     ## array unitless instance ##
#     arr_unitless_expect = {
#         'values'        : arr_untlss_def['values'],
#         'units'         : arr_untlss_def['units'],
#         'shape'         : (2,2),
#         'size'          : 4,
#         'physical_type' : None
#     }

#     # consolidate expected values for all instances
#     instance_checks = [single_int_expect, single_flt_expect,
#                        arr_homo_expect, arr_hetero_expect,
#                        single_unitless_expect, arr_unitless_expect]

#     #-------------------------------------------------------------------#
#     # check properties against their expected values for every instance #
#     #-------------------------------------------------------------------#
#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # find the expected values corresponding to the current construction instance
#     for i, id in enumerate(inst_ids):
#         if node_name == f'test_properties[{id}]':
#             # test the expected values
#             TestWide.prop_checker(init_arr_w_units(), instance_checks[i])

# def test_exceptions(request, init_arr_w_units):
#     """
#         Verify all exceptions are raised correctly using the TestWide except_checker
#         method.
        
#         Test will fail if:
#           1) the given incorrect inputs do not raise an exception
#           2) given an error message, a matching error message is not raised, even if the exception itself is
        
#         If you want to test that an exception is raised but don't care about the message
#         attached to it, define the corresponding key with a string of "ANY".
        
#         For readability, even while using the "ANY" input, define the error message key
#         as a variable with a name that describes the error to check and pass it to the
#         exceptions dictionary. Also explain the error with a comment above the variable
#         definition.
#     """
#     #---------------------------------#
#     # define errors for all instances #
#     #---------------------------------#
#     # values must be an int, float, or np.ndarry
#     vals_type_err = 'Input [values] must be of type float, int, or numpy.ndarray'

#     # units must be a UnitsArray
#     unit_type_err = 'Input [units] must be of type UnitsArray or None for unitless array'
    
#     all_excepts = {
#         vals_type_err : ('values', 'NOT THE RIGHT DATA TYPE'),
#         unit_type_err : ('units' , 'NOT THE RIGHT DATA TYPE')
#     }

#     ## applies to all array instances ##
#     # all elements must be of type int or float
#     non_num_err   = 'All elements in input [values] must be of type float or int'
#     non_num_input = ('values', np.array(['must be', 'numbers']))

#     # array must be non-empty
#     empty_err     = 'Input [values] must be a non-empty numpy.ndarray instance'
#     empty_input   = ('values', np.array([]))
    
#     #----------------------------#
#     # define errors per instance #
#     #----------------------------#
#     ## single int instance ##
#     sngle_int_excepts  = None

#     ## single float instance ##
#     sngle_flt_excepts = None

#     ## array with homogeneous units instance ##
#     arr_homo_excepts = {
#         non_num_err : non_num_input,
#         empty_err   : empty_input
#     }

#     ## array with heterogeneous units instance ##
#     arr_hetero_excepts = {
#         non_num_err : non_num_input,
#         empty_err   : empty_input
#     }

#     ## single unitless instance ##
#     sngle_unitless_excepts = None

#     ## array unitless instance ##
#     arr_unitless_excepts = {
#         non_num_err : non_num_input,
#         empty_err   : empty_input
#     }
    
#     # consolidate exception dictionaries for all instances
#     instance_excepts = [sngle_int_excepts, sngle_flt_excepts,
#                         arr_homo_excepts, arr_hetero_excepts,
#                         sngle_unitless_excepts, arr_unitless_excepts]
    
#     #---------------------------------------------------------------#
#     # check that exceptions are raised with matching error messages #
#     #---------------------------------------------------------------#
#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # perform check corresponding to the current construction instance
#     for i, id in enumerate(inst_ids):
#         if node_name == f'test_properties[{id}]':
#             # create exceptions dictionary
#             exceptions = all_excepts | instance_excepts[i]

#             # verify exceptions
#             TestWide.except_checker(init_arr_w_units, exceptions)

# # NOTE: test for units_arithmetic() is not necessary because it's functionality can be verified by
# #       operator overloading tests

# # overloaded methods
# def test_repr(request, init_arr_w_units):
#     """
#         Tests that the __repr__ operator has been overloaded correctly.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # create test object
#     test_arr_w_units = init_arr_w_units()

#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # define expected values per instance
#     sngle_int_expct    = '1.0 km'
#     sngle_flt_expct    = '1.1 kg'
#     hom_arr_expct      = f'Units:\n{arr_hom_def['units']}\nValues:\n{arr_hom_def["values"]}'
#     non_hom_arr_expct  = "['1.0 kg' '1.1 rad' '0.0 kg*rad^-1']"
#     sngle_untlss_expct = '1.0 Units Array (Unitless)'
#     array_untlss_expct = f'Units:\nUnits Array (Unitless)\nValues:\n{arr_untlss_def["values"]}'

#     # consolidate
#     expct_reprs = [sngle_int_expct, sngle_flt_expct, hom_arr_expct, non_hom_arr_expct, sngle_untlss_expct, array_untlss_expct]

#     #-------------#
#     # test method #
#     #-------------#
#     # call method
#     repr = test_arr_w_units.__repr__()

#     # verify
#     for i, id in enumerate(inst_ids):
#         if node_name == f'test_repr[{id}]':
#             assert repr == expct_reprs[i]

# # NOTE: the overloading for __str__ just calls __repr__ -> test_repr() also validates __str__

# def test_getitem(request, init_arr_w_units):
#     """
#         Tests that the __getitem__ operator has been overloaded correctly.
#     """
#     pytest.skip('NOT YET IMPLEMENTED')

# def test_setitem(request, init_arr_w_units):
#     """
#         Tests that the __setitem__ operator has been overloaded correctly.
#     """
#     pytest.skip('NOT YET IMPLEMENTED')

# def test_add(request, init_arr_w_units):
#     """
#         Tests that the __add__ operator has been overloaded correctly.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # create test objects
#     test_arr_w_units = init_arr_w_units()

#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # define expected values per instance
#     sngle_int_expct    = scb.ArrayWUnits(2                             , single_int_def['units'])
#     sngle_flt_expct    = scb.ArrayWUnits(2.1                           , single_flt_def['units'])
#     hom_arr_expct      = scb.ArrayWUnits(np.array([2, 2.1, 1])         , arr_hom_def['units'])
#     non_hom_arr_expct  = scb.ArrayWUnits(np.array([2, 2.1, 1])         , arr_non_hom_def['units'])
#     sngle_untlss_expct = scb.ArrayWUnits(2                             , sngle_untlss_def['units'])
#     array_untlss_expct = scb.ArrayWUnits(np.array([[2, 2.1], [3, 3.2]]), arr_untlss_def['units'])

#     # consolidate
#     expct_adds = [sngle_int_expct, sngle_flt_expct, hom_arr_expct, non_hom_arr_expct, sngle_untlss_expct, array_untlss_expct]

#     #-------------#
#     # test method #
#     #-------------#
#     for i, id in enumerate(inst_ids):
#         match node_name:
#             case 'test_add[single int]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(1, single_int_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
#             case 'test_add[single float]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(1, single_flt_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
#             case 'test_add[array (homo units)]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(np.array([1, 1, 1]), arr_hom_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
#             case 'test_add[array (hetero units)]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(np.array([1, 1, 1]), arr_non_hom_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
#             case 'test_add[single (unitless)]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(1, sngle_untlss_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
#             case 'test_add[array (unitless)]':
#                 # create an ArrayWUnits with matching units
#                 arr2add = scb.ArrayWUnits(np.array([[1, 1], [1, 1]]), arr_untlss_def['units'])
#                 # call the method
#                 add = test_arr_w_units.__add__(arr2add)
        
#         # verify
#         if node_name == f'test_add[{id}]':
#             try:
#                 # see if equality is iterable or not
#                 iter(add == expct_adds[i])
#             except:
#                 # not iterable -> single value to verify
#                 assert add == expct_adds[i]
#             else:
#                 # iterable -> array of values to verify
#                 assert (add == expct_adds[i]).all()

#     #-------------------------------------#
#     # additionally, test input validation #
#     #-------------------------------------#
#     # must only add ArrayWUnits objects
#     non_arrwun_err = 'Addition operation is only defined between two ArrayWUnits instances'

#     # must have the same unit dimensions
#     diff_dim_err   = 'Addition operator is only defined for units objects with the same unit dimensions'
#     bad_dim_arrwun = scb.ArrayWUnits(0, km*sec*kg)

#     # define exception dictionary
#     add_exceptions = {
#         non_arrwun_err : ('other', 0),
#         diff_dim_err   : ('other', bad_dim_arrwun)
#     }

#     # verify exceptions are raised
#     TestWide.except_checker(init_arr_w_units, add_exceptions, '__add__', True)

# def test_cross(single_arr_w_units):
#     """
#         Tests the cross method.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # create test objects
#     km_vec   = single_arr_w_units(np.array([1, 2, 3]), km)
#     sec_vec  = single_arr_w_units(np.array([4, 5, 6]), sec)

#     # expected cross product
#     expected = single_arr_w_units(np.array([-3, 6, -3]), km*sec)

#     #-------------#
#     # test method #
#     #-------------#
#     # call method to be tested
#     crossed = km_vec.cross(sec_vec)

#     # assert that method returns expected vector
#     assert (crossed == expected).all()

#     #-------------------------------------#
#     # additionally, test input validation #
#     #-------------------------------------#
#     # can only cross with another ArrayWUnits
#     non_arrwun_err = 'Cross-product operation is only defined between two ArrayWUnits objects'

#     # can only cross vectors
#     non_vec_err    = 'Cross-product operation is only defined for vectorial ArrayWUnits objects'

#     # can only cross vectors with homogeneous units
#     non_hom_un_err = 'Cross-product operation is only defined for ArrayWUnits with homogeneous units'

#     # define exception dictionary
#     cross_exceptions = {
#         non_arrwun_err : ('other', 0),
#         non_vec_err    : ('other', scb.ArrayWUnits(np.eye(3), km)),
#         non_hom_un_err : ('other', scb.ArrayWUnits(np.array([1, 2, 3]), non_hom))
#     }

#     # verify exceptions are raised
#     TestWide.except_checker(single_arr_w_units, cross_exceptions, 'cross')

# def test_summation(single_arr_w_units):
#     """
#         Tests the summation method.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # units as [cm, m, km]
#     cm_m_km_powers = np.array([[0, 1, 0, 0]   , [0, 1, 0, 0]   , [0, 1, 0, 0]])   # length, length, length
#     cm_m_km_scales = np.array([[0, 1e-5, 0, 0], [0, 1e-3, 0, 0], [0, 1, 0, 0]])   # kmE-5 , kmE-3 , km

#     cm_m_km        = scb.UnitsArray(cm_m_km_powers, cm_m_km_scales)

#     # create test object
#     test_arr_w_un  = single_arr_w_units(np.array([1, 1, 1]), cm_m_km)

#     # expected value
#     expected_sum   = scb.ArrayWUnits(1e-5 + 1e-3 + 1, km)

#     #-------------#
#     # test method #
#     #-------------#
#     # call method to be tested
#     summed = test_arr_w_un.summation()

#     # assert that method returns expected vector
#     assert summed == expected_sum

# def test_norm(request, init_arr_w_units):
#     """
#         Tests the norm method.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # create test object
#     test_arr_w_units = init_arr_w_units()

#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # define expected values per instance
#     sngle_int_expct    = scb.ArrayWUnits(1, single_int_def['units'])
#     sngle_flt_expct    = scb.ArrayWUnits(1.1, single_flt_def['units'])
#     hom_arr_expct      = scb.ArrayWUnits(1.4866068747318506, arr_hom_def['units'])
#     non_hom_arr_expct  = scb.ArrayWUnits(1.4866068747318506, None)   # will still normalize, but will remove units
#     sngle_untlss_expct = scb.ArrayWUnits(1, sngle_untlss_def['units'])
#     array_untlss_expct = scb.ArrayWUnits(3.3241540277189325, arr_untlss_def['units'])

#     # consolidate
#     expct_norms = [sngle_int_expct, sngle_flt_expct, hom_arr_expct, non_hom_arr_expct, sngle_untlss_expct, array_untlss_expct]

#     #-------------#
#     # test method #
#     #-------------#
#     # call method
#     norm = test_arr_w_units.norm()

#     # verify
#     for i, id in enumerate(inst_ids):
#         if node_name == f'test_norm[{id}]':
#             assert norm == expct_norms[i]
    

# def test_unitary(request, init_arr_w_units):
#     """
#         Tests the unitary method.
#     """
#     #----------------------#
#     # method testing setup #
#     #----------------------#
#     # create test object
#     test_arr_w_units = init_arr_w_units()

#     # get the name of the construction instance being tested
#     node_name = request.node.name

#     # define expected values per instance
#     sngle_int_expct    = td.Generic.sngle_int['unitary']
#     sngle_flt_expct    = td.Generic.sngle_flt['unitary']
#     hom_arr_expct      = td.Generic.hom_arr['unitary']
#     non_hom_arr_expct  = td.Generic.non_hom_arr['unitary']
#     sngle_untlss_expct = td.Generic.sngle_untlss['unitary']
#     array_untlss_expct = td.Generic.arr_untlss['unitary']

#     # consolidate
#     expct_units = [sngle_int_expct, sngle_flt_expct, hom_arr_expct, non_hom_arr_expct, sngle_untlss_expct, array_untlss_expct]

#     #-------------#
#     # test method #
#     #-------------#
#     for i, id in enumerate(inst_ids):
#         # skip the method for inputs that should not work
#         if node_name == f'test_unitary[{id}]':
#             if isinstance(expct_units[i], scb.ArrayWUnits):
#                 # expects a valid input, call method
#                 unit = test_arr_w_units.unitary()
#             elif expct_units[i] != 'SHOULD THROW ERROR':
#                 # does not expect a valid input, match 
#                 unit = 'SHOULD THROW ERROR'
            
#             # verify that the expected result is returned
#             assert unit == expct_units[i]
    
#     #-------------------------------------#
#     # additionally, test input validation #
#     #-------------------------------------#

Gm, km, m, mm, kg, g, sec, day, rad, mrad, deg, N, mN, microm, unitless = scb.Units.get_units(['Gm','km', 'm', 'mm', 'kg', 'g', 
                                                                             'sec', 'day','rad', 'mrad', 'deg', 
                                                                             'N', 'mN', 'microm','unitless'])
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu

import pytest
import numpy as np

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
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
         (awu(1, sec)              , kg*sec         , None                       )],   # can't convert to a unit with different dimensions
        ids = ['Basic Scalar', 'Non-Base 10 Scalar', 'Squared Scalar', 'Cubed Scalar', 'Compound Scalar', 'Multi-Base Comp. Scalar',
               'Homogeneous Matrix', 'Non-Hom Matrix', 'Non-Unit Argument', 'Incompatible Conversion']
)
def test_convert_to(request, to_convert, convert_to, expected):
        """
            Verifies that unit conversions function properly.

            Checks that a given AWU is converted to the requested units and back again correctly.
            This ensures that there is no precision loss when moving between units.
        """
        test_id = request.node.callspec.id      # current parameterized test
        # improper argument passed
        if (test_id == 'Non-Unit Argument') or (test_id == 'Incompatible Conversion'):
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

class TestMathOperations:
    """
        Collected tests for all mathematical operations for ArrayWUnits.
    """
    @pytest.mark.parametrize(
            'a, b, expected',
            [(awu(1        , kg), awu(1        , kg), awu(2        , kg)),
             (awu([1, 2, 3], kg), awu([1, 2, 3], kg), awu([2, 4, 6], kg)),
             (awu(1        , kg), awu(1        , km), None              ),
             (awu([1, 2, 3], kg), awu([1, 2, 3], km), None              )],
            ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix']
    )
    def test_addition(self, a, b, expected):
        """
            Verifies that addition operations perform as expected.
        """
        if expected is not None:
            if a.size == 1:
                assert a + b == expected
            else:
                assert all(a + b == expected)
        
        else:
            # can't perform addition between two different units
            with pytest.raises(Exception):
                a + b
    
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
             (awu(1, m**3*sec**-2)         , awu(1, km**-2)                   , awu(1        , microm/sec**2               ))],
             ids = ['Same Units', 'Same Units Matrix', 'Different Units', 'Diff Units Matrix', 'Compound Units', 'Comp Units Matrix', 
                    'Same Dims']
    )
    def test_multiplication(self, a, b, expected):
        """
            Verifies that the multiplication operator functions as expected.
        """
        if a.size == 1:
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
            'to_raise, raised_to, expected',
            [(awu(5        , kg           ),         2, awu(25                , kg**2                 )),
             (awu([1, 2, 3], kg           ),        -2, awu([1, 0.25, 3.0**-2], kg**-2                )),
             (awu([1, 2, 3], [kg, km, sec]),         2, awu([1,    4,       9], [kg**2, km**2, sec**2]))],
             ids = ['Scalar', 'Matrix to Scalar', 'Non-Hom Matrix']
    )
    def test_power(self, to_raise, raised_to, expected):
        """
            Verifies that the power operator functions as expected.
        """
        if expected.size == 1:
            assert (to_raise**raised_to) == expected

        else:
            assert all((to_raise**raised_to) == expected)

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
                   [0, 0 , 18]], kg**2))],
             ids = ['Hom Row * Hom Row', 'Hom Row * Hom Column', 'Hom Mat * Hom Column',
                    'Hom Mat * Hom Mat']
    )
    def test_matrix_mulitplication(self, a, b, expected, request):
        """
            Verifies that the matrix multiplication operator functions as expected.
        """
        if request.node.callspec.id == 'Hom Row * Hom Column': pytest.skip('SKIP FOR NOW')
        if (a @ b).size == 1:
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
             (awu([1, 2, 3] , kg*km*sec**-2), awu([4, 0, 1] , km**2*kg)       , awu([2, 11, -8]        , km**3*kg**2/sec**2)),],
             ids = [ 'Same Units Matrix', 'Diff Units Matrix',  'Comp Units Matrix']
    )
    def test_cross_product(self,a,b, expected):
        """
            Verfies that the cross-product operator functions as expected.

            Limits:
            - Cross-product operation is only defined between two ArrayWUnits objects.
            - Cross-product operation is only defined for ArrayWUnits with homogeneous units.
            - Cross-product operation is only defined for vectorial ArrayWUnits objects.

        """
        assert awu.cross(a,b) == expected
        

    @pytest.mark.parametrize(
            'a, expected',
            [(awu(1, kg)                    , awu(1, kg            )),
             (awu([1, 2,3], km)             , awu([[1],[2],[3]], km             )),
             (awu([[1, 2, 3],[4,5,6],[7,8,9]] , km/sec) , awu([[1,4,7],[2,5,8],[3,6,9]], km/sec)),],

             ids = [ 'Scaler', '1D Matrix',  '2D Matrix']
    )

    def test_transpose(self, a, expected):
        """
            Verfies that the transpose operator functions as expected.

            Limits:
            - Transpose operation defined only for scalar, 1D, and 2D ArrayWunits instances.
        """
        assert awu.transpose(a) == expected

    @pytest.mark.parametrize(
                'a, expected',
                [
                (awu([[1, 2],[3, 4]], km)             , awu([[[-2, 1],[1.5, -0.5]]], km             )),
                (awu([[1, 0, 2],[0, 1, 1],[1, 0, 1]] , km/sec) , awu([[[1, 0, -2],[1, 1, -1],[-1, 0, 1]]], km/sec)),],

                ids = [ '2x2',  '3x3']
        )
    def test_inverse(self,a, expected):
        """
            verifies that the inverse operator functions as expected.

            Limits:
            - Inverse operation is only defined for ArrayWUnits objects with homogeneous units.
            - Inverse operation is only defined for square matrices.
        """
        assert awu.inverse(a) == expected

    @pytest.mark.parametrize(
                'a, expected',
                [(awu([[1, 2, 3], [4, 5, 6]], km)     , awu([[-17/18,  4/9], [-1/9,  1/9], [13/18, -2/9]], km             )),
                (awu([[3], [4]], kg)                  , awu([[0.12, 0.16]], kg             )),
                (awu([[1, 2],[3, 4]], m/sec)             , awu([[[-2, 1],[1.5, -0.5]]], m/sec            )),
                (awu([[1, 2], [2, 4]]  , km/sec) , awu([[[1, 0, -2],[1, 1, -1],[-1, 0, 1]]], km/sec)),],

                ids = [ 'Rectangular Matrix',  'Vector', 'Square Matrix', 'Singular Matrix']
        )
    def test_pseudo_inverse(self,a, expected):
        """
            Verfies that the pseudo-inverse operator functions as expected.

            Limits:
            - Pseudo-inverse operation is only defined for ArrayWUnits objects with homogeneous units.
        """
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
                (awu([1, 2, 2], km)   , awu([[1/3, 2/3, 2/3]], km             )),],
                ids = [ '2x1 vector',  '3x1 vector']
        )
    def test_unitary(self,a, expected):
        """
            Verfies that the unitary operator functions as expected.

            Limits:
            - Cannot unitize a non-1-D ArrayWUnits vector.
            - Cannot unitize an ArrayWUnits objects with non-homogeneous units.
        """

        assert awu.unitary(a) == expected
        
    
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
            (awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad)   , awu([[0, np.sqrt(2)/2], [1, np.sqrt(2)/2]],unitless))],
        ids=['Scalar 0', 'Scalar pi/2', 'Scalar pi', '1D Array', '2D Array']
    )
    def test_sin(self, a, expected):
        """
            verifies that the sine function works as expected.

            Limits:
            - Sine function is only defined for ArrayWUnits instances with homogeneous units
            - Sine function only accepts an input in radians
        """
        if a.size == 1:
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
            (awu([[0, np.pi/4], [np.pi/2, 3*np.pi/4]], rad)  , awu([[1, np.sqrt(2)/2], [0, -np.sqrt(2)/2]], unitless))
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
        if a.size == 1:
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
            (awu([[0, np.pi/6], [np.pi/4, np.pi/3]], rad)    , awu([[0, 1/np.sqrt(3)], [1, np.sqrt(3)]], unitless))
        ],
        ids=['Scalar 0', 'Scalar pi/4', 'Scalar -pi/4', '1D Array', '2D Array']
    )
    def test_tan(self, a, expected):
        """
            verifies that the tangent function works as expected.

            Limits:
            - Tangent function is only defined for ArrayWUnits instances with homogeneous units
            - Tangent function only accepts an input in radians
            - Tangent is undefined at odd multiples of pi/2 (not tested here)
            
        """
        if a.size == 1:
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
        ],
        ids=['Scalar', '1D homogeneous', '2D homogeneous', '1D non-homogeneous']
    )
    def test_atleast2d(self, a, expected):
        """
        Verifies that the atleast_2d operator functions as expected.

        Limits:
        - Input must be an ArrayWUnits object
        - Preserves values and units
        - Non-homogeneous units are supported
        """
        assert awu.atleast_2d(a) == expected

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






# # """
# # Unit Tests for the ArrayWUnits class.
# # """
# # # Versioning
# # __verison__ = "0.0.0"
# # __author__ = "Jacopo Villa"

# # # Imports
# # import sys
# # import pytest
# # import numpy as np
# # import random
# # from hypothesis import given, strategies as st
# # from hypothesis.strategies import data
# # from hypothesis.extra import numpy as hnp

# # sys.path.append("./src/")
# # from scarabaeus import UnitsArray, UnitsDatabase, ArrayWUnits

# # # -----------------
# # # SETUP
# # # -----------------
# # # Names of unit dimensions
# # units_dim = UnitsDatabase._unitDim

# # # Number of unit dimensions
# # num_units_dim = len(UnitsDatabase._unitDim)

# # # Query the single-dimensional units corresponding to each unit dimension
# # single_dim_units = [UnitsDatabase.get_units_of_dim(dim_iter) for dim_iter in units_dim]

# # # Get dictionary of power arrays associated with all units
# # powers_dict = UnitsDatabase.get_all_powers_arrays()

# # # Get dictionary of scale arrays associated with all units
# # scales_dict = UnitsDatabase.get_all_scales_arrays()

# # # Min/max units power to test
# # min_units_power = -10
# # max_units_power = 10

# # # Min/max units scale to test
# # min_units_scale = 1e-08
# # max_units_scale = 1e08

# # # Min/max numerical value to test
# # min_value = -1e08
# # max_value = 1e08

# # # Min/max number of array dimensions to test
# # min_arr_dim = 0
# # max_arr_dim = 3

# # # Min/max number of elements (i.e., array size), along each dimension, to test
# # min_arr_size = 1
# # max_arr_size = 10

# # # Assertion tolerance
# # # TODO: Understand how to define the tolarance, and if this tolerance-based methodology
# # # is the most appropriate
# # tol_assert = 1e-15


# # # -----------------
# # # HELPER FUNCTIONS
# # # -----------------
# # def gen_random_array_shape(data):
# #     # Draw a shape for a Numpy array
# #     return data.draw(
# #         hnp.array_shapes(
# #             min_dims=min_arr_dim,
# #             max_dims=max_arr_dim,
# #             min_side=min_arr_size,
# #             max_side=max_arr_size,
# #         )
# #     )


# # def gen_random_numpy_array_float(data, shape):
# #     # Draw a random Numpy array of floating points for the given shape
# #     return data.draw(
# #         hnp.arrays(np.floating, shape=shape, elements=st.floats(min_value, max_value))
# #     )


# # def gen_random_array_given_shape(data, shape):
# #     # Draw numerical arrays for the two operands
# #     return gen_random_numpy_array_float(data, shape)


# # def gen_random_powers_array(data, shape):
# #     # Draw the powers array
# #     powers = data.draw(
# #         hnp.arrays(
# #             np.integer,
# #             shape=shape,
# #             elements=st.integers(min_units_power, max_units_power),
# #         )
# #     )

# #     # Draw a boolean mask to randomly generate zero powers in the power (and therefore scale) array
# #     mask_nonzero_powers = data.draw(hnp.arrays(bool, shape))
# #     return powers * mask_nonzero_powers, mask_nonzero_powers


# # def gen_random_units_from_database(data):
# #     # Randomly choose a unit from each group of single-dimensional units, for each dimension
# #     return [random.choice(elem) for elem in single_dim_units]


# # def gen_random_scales(data, shape, mask_nonzero):
# #     # Generate a scale array

# #     # NOTE: While the unit powers can be arbitrary, and hence are simple to randomize,the unit scale factors should correspond to
# #     # the units existing in UnitsDatabase, otherwise, there would be no unit to represent such a scale factor.
# #     # Thus, in this implementation the scale factors are randomly sampled by single-dimensional units from the database.
# #     # Single-dimensional units are those with a single non-zero power in the power array, such as "m", "km", and
# #     # "AU", which all represent "length". In contrast, "Newton" is not a single-dimensional unit as it represents "mass*length/time**2".
# #     # TODO: This function can probably be optimized.

# #     # Initialize scales array
# #     scales_flat = np.zeros((int(np.prod(shape[:-1])), shape[-1]))

# #     # Loop over units elements
# #     for i in range(scales_flat.shape[0]):
# #         # Randomly choose a unit from each group of single-dimensional units, for each dimension
# #         units_slice = gen_random_units_from_database(data)

# #         # Assemble the scale vector corresponding to each unit dimension
# #         scales_slice = np.sum([scales_dict[unit] for unit in units_slice], axis=0)

# #         # Assign the scales for this slice
# #         scales_flat[i] = scales_slice

# #     # Reshape scale arrays
# #     scales = scales_flat.reshape(shape)

# #     # Apply nonzero-power mask to the scale arrays as well
# #     scales[mask_nonzero == 0] = 0.0
# #     return scales


# # # -----------------
# # # TESTS
# # # -----------------
# # # TODO: Test power
# # # TODO: Add case where one operand has homogeneous units and the other has heterogeneous units
# # # TODO: Add case with broadcasting operations
# # # TODO: Test dot product
# # # TODO: Test cross product
# # # TODO: Test norm
# # # TODO: Test negation property in multiplication
# # # TODO: Test inverse element in multiplication
# # # TODO: Avoid unit conversion (in ArrayWUnits.match_unit_scales()) when identity operation


# # @given(data())
# # def test_array_w_units_addition(data):
# #     # Generate random Numpy arrays with the same shape
# #     shape = gen_random_array_shape(data)
# #     values_a = gen_random_array_given_shape(data, shape)
# #     values_b = gen_random_array_given_shape(data, shape)
# #     values_c = gen_random_array_given_shape(data, shape)

# #     # Add one axis to the powers and scale arrays (since the last axis is used to describe powers and scales for each units dimension)
# #     shape_powers_and_scales = shape + (num_units_dim,)

# #     # Draw units power arrays. Note that, for addition, the powers of the two operands must match. Here, it is assumed that this is the case.
# #     powers, mask_nonzero_powers = gen_random_powers_array(data, shape_powers_and_scales)

# #     # Draw scale arrays where the elements of self are multiples of the elements of other
# #     # scales_a, scales_b, scales_c = gen_3_random_scales_multiples(powers, mask_nonzero_powers)
# #     scales_a = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers)
# #     scales_b = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers)
# #     scales_c = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers)

# #     # Instantiate UnitsArray objects using the above power and scale arrays
# #     units_array_a = UnitsArray(powers, scales_a)
# #     units_array_b = UnitsArray(powers, scales_b)
# #     units_array_c = UnitsArray(powers, scales_c)

# #     # Instantiate ArrayWUnits operands using the above numerical arrays and unit arrays
# #     awu_a = ArrayWUnits(values_a, units_array_a)
# #     awu_b = ArrayWUnits(values_b, units_array_b)
# #     awu_c = ArrayWUnits(values_c, units_array_c)

# #     # Instantiate a very small ArrayWUnits
# #     eps_addition = 1e-10
# #     values_eps = eps_addition * np.ones(shape)
# #     awu_eps = ArrayWUnits(values_eps, units_array_a)

# #     # Instantiate the identity vector of addition, for awu_a. This is to test the identity element of addition.
# #     awu_a_zeros = ArrayWUnits(np.zeros(values_a.shape), units_array_a)

# #     # NOTE: These assertion makes use of a threshold to account for roundoff errors
# #     # Test commutativity
# #     # assert awu_a + awu_b == awu_b + awu_a
# #     sum1 = awu_a + awu_b
# #     sum2 = awu_b + awu_a

# #     assert np.all(((sum1 - sum2) / (sum1 + awu_eps)).values < tol_assert)

# #     # denominator = np.max(np.absolute([awu_a.values, awu_b.values]), axis=0)
# #     # rel_err_ab = np.where(denominator != 0, np.absolute((sum1.values-sum2.values) / denominator), 0)
# #     # assert np.all(rel_err_ab < tol_assert)

# #     # assert np.all(np.isclose(sum1.values, sum2.values, rtol=tol_assert))
# #     # assert np.all(np.isclose(sum1.values, sum2.values, rtol=tol_assert))

# #     # Test associativity
# #     # assert (awu_a + awu_b) + awu_c == awu_a + (awu_b + awu_c)
# #     sum1 = (awu_a + awu_b) + awu_c
# #     sum2 = awu_a + (awu_b + awu_c)
# #     denominator = np.max(
# #         np.absolute([awu_a.values, awu_b.values, awu_c.values]), axis=0
# #     )
# #     rel_err_abc = np.where(
# #         denominator != 0, np.absolute((sum1.values - sum2.values) / denominator), 0
# #     )
# #     assert np.all(rel_err_abc < tol_assert)
# #     # assert np.all(np.isclose(sum1.values, sum2.values, rtol=tol_assert))

# #     # Test identity
# #     # assert awu_a + awu_a_zeros == awu_a
# #     awu_a_plus_zeros = awu_a + awu_a_zeros
# #     assert awu_a_plus_zeros == awu_a
# #     # assert np.all(np.isclose(awu_a_plus_zeros.values, awu_a.values, rtol=tol_assert))


# # @given(data())
# # def test_array_w_units_multiplication(data):
# #     # Generate random Numpy arrays with the same shape
# #     shape = gen_random_array_shape(data)
# #     values_a = gen_random_array_given_shape(data, shape)
# #     values_b = gen_random_array_given_shape(data, shape)
# #     values_c = gen_random_array_given_shape(data, shape)
# #     values_d = gen_random_array_given_shape(data, shape)

# #     # Add one axis to the powers and scale arrays (since the last axis is used to describe powers and scales for each units dimension)
# #     shape_powers_and_scales = shape + (num_units_dim,)

# #     # Draw units power arrays. Note that, for multiplication, the powers of the two operands can be different in general.
# #     powers_a, mask_nonzero_powers_a = gen_random_powers_array(
# #         data, shape_powers_and_scales
# #     )
# #     powers_b, mask_nonzero_powers_b = gen_random_powers_array(
# #         data, shape_powers_and_scales
# #     )
# #     powers_c, mask_nonzero_powers_c = gen_random_powers_array(
# #         data, shape_powers_and_scales
# #     )

# #     # Draw scale arrays where the elements of self are multiples of the elements of other
# #     scales_a = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers_a)
# #     scales_b = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers_b)
# #     scales_c = gen_random_scales(data, shape_powers_and_scales, mask_nonzero_powers_c)

# #     # Instantiate UnitsArray objects using the above power and scale arrays
# #     units_array_a = UnitsArray(powers_a, scales_a)
# #     units_array_b = UnitsArray(powers_b, scales_b)
# #     units_array_c = UnitsArray(powers_c, scales_c)

# #     # Instantiate ArrayWUnits operands using the above numerical arrays and unit arrays
# #     awu_a = ArrayWUnits(values_a, units_array_a)
# #     awu_b = ArrayWUnits(values_b, units_array_b)
# #     awu_c = ArrayWUnits(values_c, units_array_c)

# #     # Instantiate an ArrayWUnits object whose powers are the same as awu_a.
# #     # This is to test the distributivity property of multiplication
# #     awu_d = ArrayWUnits(values_d, units_array_b)

# #     # Instantiate the identity and zero vectors of multiplication, for awu_a. This is to test the identity element of multiplication
# #     # and the property of zero.
# #     awu_a_ones = ArrayWUnits(np.ones(values_a.shape), units_array_a)
# #     awu_a_zeros = ArrayWUnits(np.zeros(values_a.shape), units_array_a)

# #     # NOTE: These assertion makes use of a threshold to account for roundoff errors
# #     # Test commutativity
# #     # assert awu_a * awu_b == awu_b * awu_a
# #     prod1 = awu_a * awu_b
# #     prod2 = awu_b * awu_a
# #     denominator = np.max(np.absolute([awu_a.values, awu_b.values]), axis=0)
# #     rel_err_ab = np.where(
# #         denominator != 0, np.absolute((prod1.values - prod2.values) / denominator), 0
# #     )
# #     assert np.all(rel_err_ab < tol_assert)
# #     # assert np.all(np.isclose(prod1.values, prod2.values, rtol=tol_assert))

# #     # Test associativity
# #     # assert (awu_a * awu_b) * awu_c == awu_a * (awu_b * awu_c)
# #     prod1 = (awu_a * awu_b) * awu_c
# #     prod2 = awu_a * (awu_b * awu_c)
# #     rel_err_abc = np.absolute(
# #         (prod1.values - prod2.values)
# #         / np.max(np.absolute([awu_a.values, awu_b.values, awu_c.values]))
# #     )
# #     assert np.all(rel_err_abc < tol_assert)
# #     # assert np.all(np.isclose(prod1.values, prod2.values, rtol=tol_assert))

# #     # Test distributivity
# #     # assert awu_a * (awu_b + awu_c) == awu_a * awu_b + awu_a * awu_c
# #     prod1 = awu_a * (awu_b + awu_d)
# #     prod2 = awu_a * awu_b + awu_a * awu_d
# #     rel_err_abd = np.absolute(
# #         (prod1.values - prod2.values)
# #         / np.max(np.absolute([awu_a.values, awu_b.values, awu_d.values]))
# #     )
# #     assert np.all(rel_err_abd < tol_assert)
# #     # assert np.all(np.isclose(prod1.values, prod2.values, rtol=tol_assert))

# #     # Test identity
# #     # assert awu_a * awu_a_ones == awu_a
# #     awu_a_times_ones = awu_a * awu_a_ones
# #     awu_a_times_one_scalar = awu_a * 1.0
# #     rel_err_a = np.absolute(
# #         (awu_a_times_ones.values - awu_a.values) / np.absolute([awu_a.values])
# #     )
# #     rel_err_a_scalar = np.absolute(
# #         (awu_a_times_one_scalar.values - awu_a.values) / np.absolute([awu_a.values])
# #     )
# #     assert np.all(rel_err_a < tol_assert)
# #     assert np.all(rel_err_a_scalar < tol_assert)
# #     # assert np.all(np.isclose(awu_a_times_ones.values, awu_a.values, rtol=tol_assert))
# #     # assert np.all(np.isclose(awu_a_times_one_scalar.values, awu_a.values, rtol=tol_assert))

# #     # Test property of zero
# #     # assert awu_a * awu_a_zeros = awu_a_zeros
# #     awu_a_times_zeros = awu_a * awu_a_zeros
# #     awu_a_times_zero_scalar = awu_a * 0.0
# #     assert np.all(awu_a_times_zeros.values == 0)
# #     assert np.all(awu_a_times_zero_scalar.values == 0)
# #     # assert np.all(np.isclose(awu_a_times_zeros.values, awu_a_zeros.values, rtol=tol_assert))
# #     # assert np.all(np.isclose(awu_a_times_zero_scalar.values, awu_a_zeros.values, rtol=tol_assert))

# # # @given(data())
# # # def test_array_w_units_power(data):
# # #     # Generate random Numpy arrays with the same shape
# # #     shape = gen_random_array_shape(data)
# # #     values_a = gen_random_array_given_shape(data, shape)
# # #     values_b = gen_random_array_given_shape(data, shape)
# # #     values_c = gen_random_array_given_shape(data, shape)
# # #     values_d = gen_random_array_given_shape(data, shape)

# # #     # Add one axis to the powers and scale arrays (since the last axis is used to describe powers and scales for each unit's dimension)
# # #     shape_powers_and_scales = shape + (num_units_dim,)

    
# # #     # NOTE: These assertion makes use of a threshold to account for roundoff errors

# # #     # Test positive and negative integer exponents

# # #     # Test exponentiation to 0

# # #     # Test law of indices

# # def test_array_w_units_equality_and_inequality():
# #     pytest.skip("NOT IMPLEMENTED")
# #     # NOTE: this is all of the testing I used when re-doing the equality and inequality operators,
# #     #       leaving it here for now to help with testing later
# #     # (
# #     #     kg,
# #     #     km,
# #     #     sec,
# #     #     rad,
# #     #     meter,
# #     #     AU,
# #     #     min,
# #     #     hour,
# #     #     day,
# #     #     deg,
# #     #     Newton,
# #     # ) = scb.UnitsArray.generate_common_units()

# #     # meter2 = meter * meter
# #     # km_hour2 = km / hour**2

# #     # one_units = scb.UnitsArray.from_array(np.array([[meter2, km_hour2], [km, kg]]))
# #     # one = scb.ArrayWUnits(np.array([[1, 2], [3, 4]]), one_units)

# #     # two_units = scb.UnitsArray.from_array(np.array([[AU, km_hour2], [rad, kg]]))
# #     # two = scb.ArrayWUnits(np.array([[1, 1], [3, 4]]), two_units)

# #     # # print(np.equal(one.units, two.units))
# #     # # print(one.units.get_names())

# #     # # print(one == two)

# #     # unitless1 = scb.ArrayWUnits(1, None)
# #     # unitless2 = scb.ArrayWUnits(5, None)
# #     # barg = scb.ArrayWUnits(1, kg)
# #     # oneMat = scb.ArrayWUnits(np.array([1, 2, 3]), kg)
# #     # twoMat = scb.ArrayWUnits(np.array([1, 1, 3]), scb.UnitsArray.from_name("kg"))

# #     # darg = scb.ArrayWUnits(1, kg)
# #     # print(barg == darg)
# #     # print(barg != darg)
# #     # print(one == two)
# #     # print(one != two)