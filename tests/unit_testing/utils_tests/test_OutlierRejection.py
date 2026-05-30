# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb

import pytest

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
    pytest.skip()





# #---------------------------#
# #          Imports          #
# #---------------------------#
# import pytest

# import sys
# sys.path.append('./test/')
# import TruthData as td
# from TestWide import TestWide

# sys.path.append('./src/')
# import scarabaeus as scb
# import numpy as np
# import matplotlib.pyplot as plt

# #---------------------------#
# #           Setup           #
# #---------------------------#
# # get truth data
# # TODO: add these to TruthData.Generic
# # NOTE: should use np.sin() for this instead of creating a bunch of points manually
# fig, sub_ax = plt.subplots(1,1,figsize=(8, 8),sharex=True)
# test_x      = np.array([0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0, 6.33])
# test_y      = np.array([0.0, 0.33, 0.62, 0.84, 0.98, 0.99, 0.91, 0.74, 0.5, 0.24, -0.03, -0.29, -0.48, -0.6, -0.64, -0.59, -0.46, -0.25, -0.01, 0.25])
# test_marker = "."
# test_label  = "Sine Wave"

# # get expected values
# # TODO: add this to TruthData.Generic as well
# expected_canvas  = sub_ax.figure.canvas
# expected_scatter = sub_ax.scatter(test_x, test_y, marker = test_marker, label = test_label) # <- needs some tweaking

# #---------------------------#
# #          Fixtures         #
# #---------------------------#

# @pytest.fixture
# def init_outlier_rejection():
#     """
#         Fixture for a OutlierRejection object. Tests the OutlierRejection constructor.
        
#         Returns a factory method with provided values as default arguments.
#     """
#     def make_class(# default values
            
#             ax     = sub_ax,
#             x      = test_x,
#             y      = test_y,
#             marker = test_marker,
#             label  = test_label):

#          return scb.OutlierRejection(ax, x, y, marker, label)

#     return make_class

# #---------------------------#
# #           Tests           #
# #---------------------------#
# def test_typing(init_outlier_rejection):
#     """
#         Verify initialization works.
        
#         Test will fail if:
#           1) The class fails to initialize given the default inputs
#     """
#     assert isinstance(init_outlier_rejection(), scb.OutlierRejection)

# def test_properties(init_outlier_rejection):
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
#     #------------------------------------------------------#
#     # define all properties and their expected values here #
#     #------------------------------------------------------#
#     checks = {
#             '_ax'             : sub_ax,
#             '_x'              : test_x,
#             '_y'              : test_y,
#             '_marker'         : test_marker,
#             '_label'          : test_label,
#             '_canvas'         : "SKIP",
#             '_scatter'        : "SKIP",
#             '_lasso'          : "SKIP",  # not sure how this works but need to have a proper expected value
#             '_removed_indices': "SKIP",      # set to [] in the constructor
#     }
    
#     #------------------------------------------------#
#     # check properties against their expected values #
#     #------------------------------------------------#
#     TestWide.prop_checker(init_outlier_rejection(), checks, True)

# def test_exceptions(init_outlier_rejection):
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
#     ax_err     = 'ax must be a matplotlib axes object'
#     x_err      = "x must be a ndarray"
#     y_arr      = 'y must be a ndarray'
#     marker_arr = 'marker must be a string'
#     label_err  = 'label must be a string'

#     #-----------------------------#
#     # define exception dictionary #
#     #-----------------------------#
#     exceptions = {
#         ax_err     : ('ax', 0),
#         x_err      : ('x', 0),
#         y_arr      : ('y', 0),
#         marker_arr : ('marker', 0),
#         label_err  : ('label', 0)
#     }
    
#     #---------------------------------------------------------------#
#     # check that exceptions are raised with matching error messages #
#     #---------------------------------------------------------------#
#     TestWide.except_checker(init_outlier_rejection, exceptions)

# def test_onselect(init_outlier_rejection):
#     # ----------------------#
#     # Method Testing Setup  #
#     # ----------------------#
#     # Initialize the test object by calling init_outlier_rejection
#     test_obj = init_outlier_rejection()

#     # Selection area
#     selection_coords = np.array([
#         [1, 1,]  ,  # Top-left corner
#         [2, 1],    # Top-right corner
#         [2, 0.8],   # Bottom-right corner
#         [1,0.8],   # Bottom-left corner
#         [1, 0.84]     # Closing the rectangle
#     ])

#     # Call the onselect method
#     test_obj.onselect(selection_coords)

#     # Calculate expected_removed_indices based on correct logic (x between 1.0 and 3.0)
#     expected_removed_indices = [4,5]

#     # Assert that the indices match
#     assert set(test_obj.removed_indices) == set(expected_removed_indices)
