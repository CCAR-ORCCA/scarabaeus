# """
# Unit Tests for the Body class.
# """
# # Versioning
# __version__ = "0.0.0"
# __author__ = "Mohamed Salem"

# # Imports
# import sys
# import os

# import pytest
# import numpy as np

# sys.path.append("./src/")
# import scarabaeus as scb

# #---------------------------#
# #           Setup           #
# #---------------------------#
# # Perform necessary setup here
# # Generate common units
# kg, km, sec, rad, meter, AU, min, hour, day, deg, Newton = scb.UnitsArray.generate_common_units()

# # ----------------------------------
# #          initiate Spice manager spiceMgr
# # ----------------------------------
# spiceMgr = scb.SpiceManager()


# #   Import SPICE kernels
# path = os.getcwd()+'/Kernels'
# print(path)
# kernelFiles = [
#     path+'/cas00084.tsc',
#     path+'/naif0012.tls',
#     path+'/de432s.bsp',
#     path+'/pck00008.tpc',
#     path+'/gm_de431.tpc']
# spiceMgr.load_kernel(kernelFiles)

# #---------------------------#
# #           Tests           #
# #---------------------------#
    
    
# @pytest.fixture
# def Test_ideal_Case():
#     return [
#    'Earth', 
#    scb.ArrayWUnits(np.array([-9.58220572e+03, 7.94223775e+05, 7.87132144e+05]), km),
#    scb.ArrayWUnits(887112000,sec),
#    'J2000',
#    body,
#    spiceMgr
#    ]

# def SRP_values(args):
#     return scb.SolarRadiationPressure(args[0], args[1], args[2], args[3], args[4], args[5]).compute_acceleration().values

# def test_passing_position_in_meter(Test_ideal_Case):
#     Test_ideal_Case[1] = scb.ArrayWUnits(np.array([-9.58220572e+06, 7.94223775e+08, 7.87132144e+08]), meter)
#     np.testing.assert_array_almost_equal(
#         scb.SolarRadiationPressure(Test_ideal_Case[0], Test_ideal_Case[1], Test_ideal_Case[2], Test_ideal_Case[3], Test_ideal_Case[4], Test_ideal_Case[5]).compute_acceleration().values, 
#         np.array([-3.44718932e-12, 2.55008237e-12, 1.11866483e-12]),
#         decimal = 11                
#     )



# def test_passing_epoch_in_sec(Test_ideal_Case):
#     Test_ideal_Case[2] = scb.ArrayWUnits(14785200.0,min)
#     np.testing.assert_array_almost_equal(
        
#         scb.SolarRadiationPressure(Test_ideal_Case[0], Test_ideal_Case[1], Test_ideal_Case[2], Test_ideal_Case[3], Test_ideal_Case[4], Test_ideal_Case[5]).compute_acceleration().values, 
#         np.array([-3.44718932e-12, 2.55008237e-12, 1.11866483e-12]),
#         decimal = 11                
#     )

# def test_passing_non_string_origin_name(Test_ideal_Case):
#     Test_ideal_Case[0] = 5
#     np.testing.assert_raises(
#         Exception, 
#         SRP_values, 
#         Test_ideal_Case
#     )

    

# def test_passing_undefined_origin_name(Test_ideal_Case):
#     Test_ideal_Case[0] = 'undefined'
#     np.testing.assert_raises(
#         Exception, 
#         SRP_values, 
#         Test_ideal_Case
#     )           
    
# def test_passing_non_string_ref_frame(Test_ideal_Case):
#     Test_ideal_Case[0] = 555
#     np.testing.assert_raises(
#         Exception, 
#         SRP_values, 
#         Test_ideal_Case
#     )   
    
# def test_passing_undefined_ref_frame(Test_ideal_Case):
#     Test_ideal_Case[0] = 'undefined'
#     np.testing.assert_raises(
#         Exception, 
#         SRP_values, 
#         Test_ideal_Case
#     ) 


    
# # #to Run , "python -m pytest test\test_SolarRadiationPressure.py -v"