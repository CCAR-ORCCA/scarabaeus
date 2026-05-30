# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest
import spiceypy as spice
import os

(J2000,ITRF93,ECLIPJ2000,IAUEARTH) = scb.Frame.generate_common_frames()
# load spice kernels
furnshKernelFilename = os.getcwd() + "/data/kernels/locked/locked_generic.tm"
scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)
km, kg, sec, hr,day, m, rad, unitless = scb.Units.get_units(['km', 'kg', 'sec', 'hr','day', 'm', 'rad', 'unitless'])
#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_skbrief():
    """
        Verifies that ID and interval information are extracted from the given C-kernel.
    """
    pytest.skip()

def test_get_pos():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_pos_core():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_vel():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_vel_core():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_state():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_state_precise():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_status_antenna():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_elevation_angle():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_sep_angle():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_lighttime():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_STMs():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_propagator_settings():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_attitude():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()

def test_get_xfrm():
    """
        Verifies that the correct DCM is returned.
    """
    pytest.skip()


    

# #---------------------------#
# #           Setup           #
# #---------------------------#
# # Perform necessary setup here
# # generate units
# km  = scb.UnitsArray.from_name("km")
# sec = scb.UnitsArray.from_name("s")
# rad = scb.UnitsArray.from_name("rad")

# # get data paths from TruthData
# (furnsh_kernel, loading_kernel, spk_filename) = td.DataPaths.get_data_paths(td.DataPaths, 'SpiceManager')

# # clearing the kernel pool before the test to eliminate any load errors
# spiceypy.kclear()

# # load a set of kernels using the list contained in the meta-kernel (MK)
# scb.SpiceManager.load_kernel_from_mkfile(furnsh_kernel)

# # define the bodies for the test
# target         = scb.Body("TEST_target", spice_id = -9998)
# observer       = scb.GroundStation("TEST_obs", spice_id = -9999)

# # define the reference frame and origin body for the test
# ref_frame_name = "J2000"
# origin_body    = "EARTH"

# # get position and velocity arrays defining the trajectory data points
# pos1 = td.Generic.r
# vel1 = td.Generic.v
# pos2 = td.Generic.r * 0
# vel2 = td.Generic.v * 0

# # define epochs corresponding to the trajectory data points
# epochs = scb.EpochArray(scb.ArrayWUnits(np.array([0.0, 1.0, 2.0, 3.0]), sec), "TDB")

# # initialize the trajectories
# target_trajectory   = scb.Trajectory("TEST_target.bsp", pos1, vel1, epochs, ref_frame_name, origin_body, target)
# observer_trajectory = scb.Trajectory("TEST_obs.bsp", pos2, vel2, epochs, ref_frame_name, origin_body, observer)

# #---------------------------#
# #          Fixtures         #
# #---------------------------#
# # Create any necessary fixtures here
# @pytest.fixture
# def init_spc_manager():
#     """
#         Fixture for a SpiceManager object.

#         Returns a factory method with provided values as default arguments.
#     """
#     def make_class():
#         # no constructor to define defaults with
#         return scb.SpiceManager()
    
#     return make_class

# #---------------------------#
# #           Tests           #
# #---------------------------#
# def test_typing(init_spc_manager):
#     """
#         Verify initialization works.

#         Test will fail if:
#           1) The class fails to initialize given the default inputs
#     """
#     assert isinstance(init_spc_manager(), scb.SpiceManager)

# def test_properties(init_spc_manager):
#     """
#         Verify all properties exist and are defined correctly using the TestWide prop_checker
#         method.

#         Test will fail if:
#           1) all properties aren't defined in the 'checks' dict (unless disabled by 
#             the 'ignore_prop_count_check' argument)
#           2) any property doesn't match the expected value
        
#         If you want to skip one, set its value as "SKIP". This will still check the properties
#         you have defined (printed in the TEST RESULTS terminal) while not letting the entire 
#         test_properties() test pass. Instead it will skip, making it obvious in the testing 
#         window that this isn't complete.
#     """
#     #------------------------------------------------------#
#     # define all properties and their expected values here #
#     #------------------------------------------------------#
#     checks = {
#         "spc_pos_units"       : km,         # hard coded in SpiceManager
#         "spc_time_units"      : sec,        # hard coded in SpiceManager
#         "spc_angle_units"     : rad,        # hard coded in SpiceManager
#         "spc_vel_units"       : km/sec,     # hard coded in SpiceManager
#         "kernel_list"         : [],         # should be an empty list since we haven't done anything with it yet
#         "kernel_folder"       : td.DataPaths.get_data_truths("SpiceManager"),
#         "poly_interp_deg"     : 3,          # hard coded in SpiceManager
#         "poly_interp_par_deg" : 1           # hard coded in SpiceManager
#     }

#     #------------------------------------------------#
#     # check properties against their expected values #
#     #------------------------------------------------#
#     TestWide.prop_checker(init_spc_manager(), checks, ignore_prop_count_check = True)

# def test_exceptions(init_spc_manager):
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
#     #--------------------------------------------------------#
#     # define error messages and any needed error inputs here #
#     #--------------------------------------------------------#
#     # no errors to define
    
#     #-----------------------------#
#     # define exception dictionary #
#     #-----------------------------#
#     exceptions = {
#         # no exceptions to check
#     }

#     #---------------------------------------------------------------#
#     # check that exceptions are raised with matching error messages #
#     #---------------------------------------------------------------#
#     TestWide.except_checker(init_spc_manager, exceptions)

# def test_print_kernels(capsys):
#     """
#         Test that the print_kernels function prints the expected
#         list of loaded kernels.
#     """
#     # print_kernels() call prints list of all loaded kernels
#     scb.SpiceManager.print_kernels()

#     # capture the printed block
#     captured = capsys.readouterr()

#     # compare captured print to the expected block as defined by TruthData
#     assert captured.out == td.Generic.loaded_kernel_print_block

# # NOTE: test_clear_kernels() should appear here by the order of method definitions in 
# #       SpiceManager, but it has been placed at the end due to it needing to undo the 
# #       setup necessary for the rest of the tests

# def test_get_state():
#     """
#         Test that the get_state function returns the correct state
#         of a target body given some origin.
#     """
#     # check that the state is the same as the first state defined by pos1 and vel1
#     scarabaeus_state = scb.SpiceManager.get_state(target.name, epochs[0].time.values.tolist(), 'J2000', observer.name)

#     # get the values and units from the first position and velocity, then create a state using those values
#     first_pos_and_vel_vals = np.array(pos1[0].values.tolist() + vel1[0].values.tolist())
#     expected_state         = scb.ArrayWUnits(first_pos_and_vel_vals, scb.UnitsArray.from_array(np.array([km, km, km, km/sec, km/sec, km/sec])))

#     assert (scarabaeus_state == expected_state).all()

# def test_get_pos():
#     """
#         Test that the get_pos function returns the correct position
#         of a target body given some origin.
#     """
#     scb_pos = scb.SpiceManager.get_pos("TEST_target", epochs[0].time.values.tolist(), "J2000", "TEST_obs")
    
#     assert (scb_pos == pos1[0, :]).all()

# def test_get_vel():
#     """
#         Test that the get_pos function returns the correct position
#         of a target body given some origin.
#     """
#     scb_pos = scb.SpiceManager.get_vel("TEST_target", epochs[0].time.values.tolist(), "J2000", "TEST_obs")
    
#     assert (scb_pos == vel1[0, :]).all()

# def test_get_parameters():
#     """
#         Test that the get_parameters function returns the expected
#         parameters given a target, epoch time, reference frame, and origin.
#     """
#     pytest.skip("NOT IMPLEMENTED")

# def test_get_attitude():
#     """
#         Test that the ckgpav wrapper returns the same value
#         as the spiceypy version.
#     """
#     pytest.skip("NOT IMPLEMENTED - NEED TRUTH DATA")

# def test_get_observer_target_visibility_windows():
#     pytest.skip("NOT IMPLEMENTED")

# def test_str2et():
#     """
#         Test that the str2et function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.str2et(td.Generic.datetime_string)
#     spc_time = spiceypy.str2et(td.Generic.datetime_string)

#     # compare
#     assert scb_time == spc_time

# def test_utc2et():
#     """
#         Test that the utc2et function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.utc2et(td.Generic.datetime_string)
#     spc_time = spiceypy.utc2et(td.Generic.datetime_string)

#     # compare
#     assert scb_time == spc_time

# def test_utc2tdb():
#     """
#         Test that the utc2tdb function wrapper returns the same value
#         as the spiceypy version.
#     """
#     pytest.skip("THIS FUNCTION WRAPS THE SAME SPICE FUNCTION AS utc2et, CAN'T CHECK AGAINST SPICEYPY")

# def test_sclk2et():
#     """
#         Test that the sclk2et function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.sclk2et(td.Cassini.spice_id, td.Cassini.sc_clock_time)
#     spc_time = spiceypy.scs2e(td.Cassini.spice_id, td.Cassini.sc_clock_time)

#     # compare
#     assert scb_time == spc_time

# def test_et2utc():
#     """
#         Test that the et2utc function wrapper returns the same value
#         as the spiceypy version.
#     """
#     pytest.skip("SKIPPING UNTIL WE DECIDE TO EITHER CHOP SPICEYPY'S OUTPUT STR TO SAME SIG FIGS AS SPICEMANAGER OR INCREASE SPICEMANAGER'S OUTPUT SIG FIGS")

#     ephem_time = td.Cassini.ephem_time
#     scb_et2utc = scb.SpiceManager.et2utc(epoch_time)
#     spc_et2utc = spiceypy.et2utc([epoch_time], 'ISOC', 35, 35)[0]
#     assert spc_et2utc == pytest.approx(scb_et2utc, TestWide.default_tol)
#     # np.testing.assert_equal(scarabaeus_et2utc, spice_et2utc)

# def test_et2sclk():
#     """
#         Test that the et2sclk function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.et2sclk(td.Cassini.spice_id, td.Cassini.ephem_time, 30)
#     spc_time = spiceypy.sce2s(td.Cassini.spice_id, td.Cassini.ephem_time, 30)

#     # compare
#     assert scb_time == spc_time

# def test_e2jd():
#     """
#         Test that the e2jd function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.et2jd(td.Cassini.ephem_time)
#     spc_time = spiceypy.et2utc([td.Cassini.ephem_time], 'J', 27, 27)[0]

#     # compare
#     assert scb_time == spc_time

# def test_et2cal():
#     """
#        Test that the et2cal function wrapper returns the same value
#        as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.et2cal(td.Cassini.ephem_time)
#     spc_time = spiceypy.et2utc([td.Cassini.ephem_time], 'C', 30, 50)[0]

#     # compare
#     assert scb_time == spc_time

# def test_cal2et():
#     """
#         Test that the cal2et function wrapper returns the same value
#         as the spiceypy version.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.cal2et(td.Cassini.calendar_time)
#     spc_time = spiceypy.str2et(td.Cassini.calendar_time)

#     # compare
#     assert scb_time == spc_time

# def test_jd2et():
#     """
#         Test that the conversion is correct and that the leapsecond is applied 
#         properly.
#     """
#     # get times returned by Scarabaeus and SPICE
#     scb_time = scb.SpiceManager.jd2et(td.Cassini.jd_time)
#     spc_time = spiceypy.unitim(td.Cassini.jd_time, 'JDTDB', 'TDB') + spiceypy.deltet(spiceypy.unitim(td.Cassini.jd_time, 'JDTDB', 'TDB'),'ET')

#     # compare
#     assert scb_time == spc_time

# def test_get_xfrm():
#     """
#         Test that the wrapper funtion returns the same value
#         as the spiceypy version.
#     """
#     # define from and to frame
#     from_frame = 'ECLIPJ2000'
#     to_frame   = 'J2000'

#     # get frames returned by Scarabaeus and SPICE
#     scb_xfrm   = scb.SpiceManager.get_xfrm(from_frame, to_frame, td.Cassini.ephem_time)
#     spc_xfrm   = spiceypy.pxform(from_frame, to_frame, td.Cassini.ephem_time)

#     # compare
#     assert (scb_xfrm == spc_xfrm).all()

# # NOTE: commenting these two out because the methods they're testing don't exist in SpiceManager right now
# #       will delete once we've decided if they need to be re-added or not
# # def test_get_mu():
# #     """
# #         Test the wrapper function returns the correct value of mu for 
# #         a body.
# #     """
# #     body = 'EARTH' # Earth

# #     scb_mu = scb.SpiceManager.get_mu(body).values.tolist()
# #     spc_mu = spiceypy.bodvrd(body, 'GM', 1)[1]

# #     np.testing.assert_equal(scb_mu, spc_mu)

# # def test_get_radii():
# #     """
# #     Test that the wrapper function correctly returns the body radius.
# #     """
# #     body = 'EARTH'

# #     scb_radius = scb.SpiceManager.get_radii(body).values.tolist()
# #     spc_radius = spiceypy.bodvrd(body, 'RADII', 3)[1][1]

# #     np.testing.assert_equal(scb_radius, spc_radius)

# def test_matrix_times_vector():
#     """
#         Test that the wrapper function returns the same value
#         as the spiceypy version
#     """
#     pytest.skip("NOT IMPLEMENTED")

# def test_get_id_from_string():
#     """
#         Test that the wrapper function returns the same value
#         as the spiceypy version.
#     """
#     # get ID's returned by Scarabaeus and SPICE
#     scb_spice_id = scb.SpiceManager.get_id_from_string("EARTH")
#     spc_spice_id = spiceypy.bods2c("EARTH")

#     # compare
#     assert scb_spice_id == spc_spice_id

# def test_get_frame_w_spice_id():
#     """
#         Test that the wrapper function returns the same value
#         as the spiceypy version.
#     """
#     # get frames returned by Scarabaeus and SPICE
#     scb_frm = scb.SpiceManager.get_frame_w_spice_id(399)
#     spc_frm = spiceypy.cidfrm(399)

#     # compare
#     assert scb_frm == spc_frm

# def test_get_intervals():
#     """
#         Test that the wrapper function returns the same value
#         as the spiceypy version.
#     """
#     pytest.skip("NOT IMPLEMENTED")
#     spk_file = ""
#     object_id = ""
#     np.testing.assert_equal(
#         scb.SpiceManager.get_intervals(spk_file, object_id),
#         []
#     )

# def test_load_kernel():
#     """
#         Test the functions furnishes the kernels in the list provided.

#         Since we load a metakernel in the setup of this test script
#         we want to load another kernel that is not in that list. To 
#         check if it is loaded, we get the count of all kernels loaded
#         before we call load_kernel(), then we call the function to 
#         load the kernel. Finally we assert that the total number of
#         kernels in the pool has increased by 1.
#     """
#     # save how many kernels are currently loaded using spice
#     kernel_count_before_load = spiceypy.ktotal("ALL")

#     # load another kernel
#     scb.SpiceManager.load_kernel([loading_kernel])

#     # save how many kernels are loaded now
#     kernel_count_after_load = spiceypy.ktotal("ALL")

#     # ensure that the number of kernels after loading is one more than the number before loading
#     assert kernel_count_after_load == (kernel_count_before_load + 1)

# def test_load_kernel_from_mkfile():
#     """
#         Test that the function furnishes all the kernels specified
#         in the meta-kernel file.

#         For this test we need to unload the kernel pool beause we
#         have already furnished from a metakernel for other tests.

#         Steps: 
#         - Clear kernel pool.
#         - Count number of kernels in pool (should be 0) = a
#         - Furnish from meta-kernel.
#         - Count number of kernels in pool (should be n) = b
#         - Assert that that a + b = b
#     """
#     # clear kernel pool
#     spiceypy.kclear()

#     # count number of kernels in pool
#     kernel_count_cleared = spiceypy.ktotal('ALL')

#     # furnish from meta-kernel
#     scb.SpiceManager.load_kernel_from_mkfile(furnsh_kernel)

#     # get pool count again
#     kernel_count_furnshed = spiceypy.ktotal('ALL')

#     # assert
#     assert (kernel_count_cleared + kernel_count_furnshed == kernel_count_furnshed)

# def test_unload_kernel_from_pool():
#     pytest.skip("NOT IMPLEMENTED")

# def test_check_kernel_status_in_pool():
#     pytest.skip("NOT IMPLEMENTED")

# def test_increase_kernel_priority():
#     pytest.skip("NOT IMPLEMENTED")

# def test_def_new_body():
#     """
#         Test that a new body is created from a given SPICE id.
#     """
#     # define new test body's properties
#     test_name = "SPICE_TEST_BODY"
#     test_id   = -12345

#     # add the new body using Scarabaeus
#     scb.SpiceManager.def_new_body(test_name, test_id)

#     # find the body in spiceypy by name and also check that the id was added correctly
#     spc_spice_id = spiceypy.bodn2c(test_name)

#     # compare the ID's (ID won't be found if the names don't match so don't need to test that)
#     assert test_id == spc_spice_id

# def test_write_spk_segment_type9():
#     """
#         Test that we can write an spk_segment to a file correctly.
        
#         We delete one of the bsp files we made as part of the setup.
#         Steps:
#         - Delete bsp file
#         - Count files with that name in the pool (should be n-1) = a 
#         - Write a new bsp file with same name
#         - Count files with that name in the pool (should be n) = b
#         - Assert b - a = 1 
#     """
#     pytest.skip("NEED TO LOOK MORE INTO HOW TO TEST THIS")
#     body_id = target.spice_id
#     origin_id = observer.spice_id
#     reference_frame = "J2000"
#     epoch_array = epochs.time.values.tolist()
#     degree_of_polynomial = 3 # This is just the default we use
#     n_states = len(epoch_array)
#     state_array = np.concatenate((pos1.values, pos2.values),axis=1).tolist()

#     # Delete the target bsp file
#     spiceypy.dvpool(spk_filename)
#     os.remove(spk_filename)
#     count_before_writing = spiceypy.ktotal('ALL') # Count
#     scb.SpiceManager.write_spk_segment_type9(
#         spk_filename,
#         body_id,
#         origin_id,
#         reference_frame,
#         epoch_array,
#         degree_of_polynomial,
#         n_states,
#         state_array
#     )

#     count_after_writing = spiceypy.ktotal('ALL') # Count
    
#     np.testing.assert_equal(count_after_writing - count_before_writing, 1)
    
# def test_name_to_id():
#     pytest.skip("NOT IMPLEMENTED")
#     # TODO: combine both of the following commented functions into one
#     def test_name_to_id_in_pool():
#         """
#         Test the function returns the correct id for a given 
#         body name.
#         """
#         name = 'EARTH'
#         np.testing.assert_equal(
#             scb.SpiceManager.name_to_ID(name),
#             399
#         )

#     def test_name_to_id_not_in_pool():
#         """
#         Test the function returns None for a name not in
#         the pool.
#         """
#         name = 'JAY_MCCOMET'
#         np.testing.assert_equal(
#             scb.SpiceManager.name_to_ID(name),
#             None
#         )

# def test_split_epoch_sequence_by_periods():
#     pytest.skip("NOT IMPLEMENTED")


# # NOTE: this test has been placed at the end due to it needing to undo the 
# #       setup necessary for the rest of the tests
# def test_clear_kernels(capsys):
#     """
#         Test that the clear_kernels function clears the kernel list.
#     """
#     # clear the kernels
#     scb.SpiceManager.clear_kernels()

#     # verify that the kernels are empty using a similar method to test_print_kernels
#     scb.SpiceManager.print_kernels()
#     captured = capsys.readouterr()

#     assert captured.out == td.Generic.no_kernels_loaded_print_block

# # teardown?
# # os.remove(spk_filename)
# # os.remove(os.getcwd() + "/data/Kernels/TEST_obs.bsp")