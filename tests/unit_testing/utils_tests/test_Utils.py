# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest

#--------------------#
# region    Fixtures #
#--------------------#

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
@pytest.mark.parametrize(
    "a, expected", 
    [(np.array([[1, 2, 3], [1,2,3]]), True),
     (np.array([[1, 1, 1]]), True),
     (np.array([[1, 1, 1], [2,2,2]]), False)],
    ids=["2x3 array", "1x3 array", "2x3 array with different rows"]
)
def test_equal_along_rows(a, expected):
    """
        Verifies equality along rows of the given array.
    """
    assert scb.Utils.equal_along_rows(a) == expected

@pytest.mark.parametrize(
    "a, expected", 
    [(np.array([[1, 2, 3], [1,2,3]]), True),
     (np.array([[1, 1, 1], [2,2,2]]),  False),
     (np.array([[1, 2, 3]]),  True)],
    ids=["equal arrays", "arrays with different rows", "arrays with different rows"]
)
def test_sclices_equal(a, expected):
    """
        Verifies that all slices along the first axis of the given array are equal.
    """
    assert scb.Utils.check_slices_equal(a) == expected

@pytest.mark.parametrize(
    "a, expected", 
    [(np.array([[1, 2, 3], [4,5,6]]), 6),
     (np.array([[3, 2, 2], [1,1,1]]), 1)],
    ids=["2x3 array", "2x3 array 2"]
)
def test_max_of_last_slice(a, expected):
    """
        Finds the maximum value along the last slice of the given array.
    """
    pytest.skip('NOT SURE HOW MAXOFLASTSLICE IS SUPPOSED TO FUNCTION -> CANT TEST RIGHT NOW')
    # assert scb.Utils.max_of_last_slice(a) == expected