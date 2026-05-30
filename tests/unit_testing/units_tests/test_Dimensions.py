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
def test_initialization():
    """
        Verifies that object is constructed correctly.
    """
    Dim = scb.Dimensions([0,1,-1,0])
    assert np.array_equal(Dim.powers, [0,1,-1,0])
    assert Dim.name == 'Velocity'

@pytest.mark.parametrize(
    'a, b, case',
    [(scb.Dimensions([0,1,0,0]),  scb.Dimensions([0,1,0,0]), 'a = b' ),
     (scb.Dimensions([1,1,0,0]),  scb.Dimensions([1,1,0,0]), 'a = b'),
     (scb.Dimensions([0,1,1,0]),  scb.Dimensions([0,1,0,0]), 'a != b')],
    ids = ['same dim', 'same dim mixed unit', 'different dim']
)
def test_equality(a, b, case):
    """
        Verifies that the equality operator functions as expected.
    """
    match case:
        case 'a = b':
            # ensure that a is to b
            assert a == b
        
        case 'a != b':
            assert a != b

@pytest.mark.parametrize(
    'a, b, case',
    [(scb.Dimensions([0,1,0,0]),  scb.Dimensions([0,3,0,0]), 'a != b' ),
     (scb.Dimensions([1,2,0,0]),  scb.Dimensions([1,1,0,0]), 'a != b'),
    ],
    ids = ['same unit', 'same unit mixed']
)
def test_inequality(a, b, case):
    """
        Verifies that the inequality operator functions as expected.
    """
    match case:
        case 'a != b':
            # ensure that a is to b
            assert a != b
        
        case 'error':
            # can't examine inequalities between two different units
            with pytest.raises(Exception):
                a != b
