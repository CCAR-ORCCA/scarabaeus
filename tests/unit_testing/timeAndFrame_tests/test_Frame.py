# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import os
import pytest

(J2000,ITRF93,ECLIPJ2000,IAUEARTH) = scb.Frame.generate_common_frames()
km, kg, sec, hr,day, m, rad, unitless = scb.Units.get_units(['km', 'kg', 'sec', 'hr','day', 'm', 'rad', 'unitless'])

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
    Frame_1 = scb.Frame("J2000")

    print(Frame_1._origin)
    assert Frame_1._origin == 0
    assert Frame_1._name == 'J2000'
    assert Frame_1._frame_class == 1
    assert Frame_1._class_id == 1

@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (J2000, ITRF93, scb.EpochArray(1096804869.182,'TDB'), np.array([[ 9.7694412361129e-01,  2.1346994242815e-01, -3.2807043600831e-03],[-2.1346855355350e-01,  9.7694963062890e-01,  7.7191827746788e-04],[ 3.3698642630361e-03, -5.3793810696780e-05,  9.9999432054441e-01]])),
        (J2000, J2000, scb.EpochArray(1096804869.182,'TDB'), np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

    ],
    ids=["J2000 to ITRF93", "J2000 to J2000"]
)
def test_get_DCM(a,b,c,expected):
    """
        Verifies that the correct DCM is returned.
    """
    assert np.allclose(a.get_DCM(a, b, c), expected)

@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (J2000, ITRF93, scb.EpochArray(1096804869.182,'TDB'), scb.ArrayWUnits([ 1.4824386025344e+08 , -7.7148149059754e+06 , 1.0964030655836e+07], km)),
        (J2000, J2000, scb.EpochArray(1096804869.182,'TDB'), scb.ArrayWUnits([0,0,0], km))

    ],
    ids=["J2000 to ITRF93", "J2000 to J2000"]
)
def test_get_translation(a,b,c,expected):
    """
        Verifies that the correct translation is returned.
    """
    assert np.allclose((a.get_relative_pos(a, b, c)).quantity.values, expected.values)
    assert (a.get_relative_pos(a, b, c)).quantity.units == expected.units

@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (J2000, ITRF93, scb.EpochArray(1096804869.182,'TDB'), scb.ArrayWUnits([  -562.5883813609161, -10782.705827176937   ,   11.5618578354427] , km/sec)),
        (J2000, J2000, scb.EpochArray(1096804869.182,'TDB'), scb.ArrayWUnits([0,0,0],  km/sec))

    ],
    ids=["J2000 to ITRF93", "J2000 to J2000"]
)
def test_get_translation_velocity(a,b,c,expected):
    """
        Verifies that the correct translation velocity is returned.
    """
    assert np.allclose((a.get_relative_vel(a, b, c)).quantity.values, expected.values)
    assert (a.get_relative_vel(a, b, c)).quantity.units == expected.units

@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (J2000, ITRF93, scb.EpochArray(1096804869.182,'TDB'), [
    [ 9.7694412361129e-01,  2.1346994242815e-01, -3.2807043600831e-03,  1.4824386025344e+08],
    [-2.1346855355350e-01,  9.7694963062890e-01,  7.7191827746788e-04, -7.7148149059754e+06],
    [ 3.3698642630361e-03, -5.3793810696780e-05,  9.9999432054441e-01,  1.0964030655836e+07],
    [ 0.0,                  0.0,                  0.0,                  1.0]
        ]),
        (J2000, J2000, scb.EpochArray(1096804869.182,'TDB'), [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],[0., 0., 0., 1.]])

    ],
    ids=["J2000 to ITRF93", "J2000 to J2000"]
)
def test_get_transformation(a,b,c,expected):
    """
        Verifies that the correct transformation is returned.
    """
    assert np.allclose(a.get_transformation(a, b, c)[0].values, np.array(expected))
    # assert (a.get_transformation(a, b, c)).units == expected.units
