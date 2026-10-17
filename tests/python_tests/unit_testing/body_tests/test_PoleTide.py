# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Pole-tide station displacement, checked against IERS Conventions (2010)
eq. 7.26, which quotes the effect in millimetres per arcsecond of pole offset.
Those rounded coefficients are independent of the class's constants."""
from dataclasses import replace

import numpy as np
import pytest
import spiceypy as spice
import scarabaeus as scb

ARCSEC = np.pi / 180.0 / 3600.0
m, km = scb.Units.get_units(["m", "km"])


@pytest.fixture(scope="module", autouse=True)
def kernels():
    spice.kclear()
    spice.furnsh("data/kernels/locked/lsk/naif0012.tls")
    spice.furnsh("data/kernels/locked/pck/pck00010.tpc")
    yield
    spice.kclear()


def station_axes(lat_deg, lon_deg, radius=6371.0):
    phi, lam = np.radians(lat_deg), np.radians(lon_deg)
    position = radius * np.array([np.cos(phi) * np.cos(lam),
                                  np.cos(phi) * np.sin(lam), np.sin(phi)])
    up = position / radius
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    return position, up, np.cross(up, east), east


@pytest.mark.parametrize("lat, lon", [(35.4, -116.9), (-35.4, 149.0),
                                      (40.4, -4.2), (5.0, 30.0), (-60.0, 200.0)])
def test_matches_iers_eq_7_26(lat, lon):
    tide = scb.PoleTide(pole_x=0.2 * ARCSEC, pole_y=0.35 * ARCSEC)
    position, up, north, east = station_axes(lat, lon)
    delta = tide.displacement(position).values * 1e6  # mm

    mean_x, mean_y = tide.mean_pole
    m1 = (tide.pole_x - mean_x) / ARCSEC
    m2 = -(tide.pole_y - mean_y) / ARCSEC
    colat, lam = np.radians(90.0 - lat), np.radians(lon)
    s_r = -33.0 * np.sin(2 * colat) * (m1 * np.cos(lam) + m2 * np.sin(lam))
    s_theta = -9.0 * np.cos(2 * colat) * (m1 * np.cos(lam) + m2 * np.sin(lam))
    s_lambda = 9.0 * np.cos(colat) * (m1 * np.sin(lam) - m2 * np.cos(lam))
    # IERS rounds its coefficients to two digits: allow 3% plus 0.02 mm.
    for ours, iers in ((delta @ up, s_r), (delta @ north, -s_theta), (delta @ east, s_lambda)):
        assert ours == pytest.approx(iers, rel=0.03, abs=0.02)


def test_mean_pole_is_the_conventional_one():
    mean_x, mean_y = scb.PoleTide(pole_x=0.0, pole_y=0.0).mean_pole
    assert mean_x / ARCSEC * 1e3 == pytest.approx(41.8, abs=0.1)
    assert mean_y / ARCSEC * 1e3 == pytest.approx(292.7, abs=0.1)


def test_no_displacement_at_the_mean_pole_and_no_rate():
    tide = scb.PoleTide(pole_x=0.0, pole_y=0.0)
    at_mean = scb.PoleTide(*tide.mean_pole)
    position = station_axes(40.4, -4.2)[0]
    np.testing.assert_allclose(at_mean.state_displacement(position, 0.0).values, 0.0, atol=1e-18)
    state = tide.state_displacement(position, 1.0e6, "ITRF93").values
    assert np.linalg.norm(state[:3]) > 1e-6  # millimetres, in km
    np.testing.assert_array_equal(state[3:], 0.0)


def test_rejects_an_unknown_body_and_the_axis():
    with pytest.raises(scb.PoleTide.UnknownPoleTideResponseError):
        scb.PoleTide(pole_x=0.0, pole_y=0.0, central_body="MARS")
    with pytest.raises(scb.PoleTide.UndefinedVerticalError):
        scb.PoleTide(pole_x=0.0, pole_y=0.0).displacement([0.0, 0.0, 6357.0])


def test_love_and_pole_partials_are_exact_derivatives():
    """Both Jacobians are the displacement's derivatives, and exact: the
    displacement is linear in (h2, l2) and, separately, in the pole offset."""
    tide = scb.PoleTide(pole_x=0.2 * ARCSEC, pole_y=0.35 * ARCSEC)
    position = station_axes(35.4, -116.9)[0]
    love = tide.state_displacement_partials(position, 1.0e6, "ITRF93").values
    pole = tide.state_displacement_pole_partials(position, 1.0e6, "ITRF93").values
    assert love.shape == pole.shape == (6, 2)
    np.testing.assert_array_equal(love[3:], 0.0)
    np.testing.assert_array_equal(pole[3:], 0.0)
    base = tide.displacement(position).values
    mean_x, mean_y = tide.mean_pole
    np.testing.assert_allclose(love[:3] @ [tide.love_h2, tide.shida_l2], base,
                               rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(pole[:3] @ [tide.pole_x - mean_x, tide.pole_y - mean_y],
                               base, rtol=1e-12, atol=1e-18)
    for jacobian, names, step in ((love, ("love_h2", "shida_l2"), 0.1),
                                  (pole, ("pole_x", "pole_y"), ARCSEC)):
        for column, name in enumerate(names):
            moved = replace(tide, **{name: getattr(tide, name) + step})
            np.testing.assert_allclose((moved.displacement(position).values - base) / step,
                                       jacobian[:3, column], rtol=1e-8, atol=1e-18)
    # h2 is only radial and l2 only horizontal.
    up = position / np.linalg.norm(position)
    np.testing.assert_allclose(np.cross(love[:3, 0], up), 0.0, atol=1e-18)
    assert love[:3, 1] @ up == pytest.approx(0.0, abs=1e-18)


def test_results_carry_units_and_a_position_in_metres_is_converted():
    tide = scb.PoleTide(pole_x=0.2 * ARCSEC, pole_y=0.35 * ARCSEC)
    position = station_axes(35.4, -116.9)[0]
    in_m = scb.ArrayWUnits(position * 1e3, m)
    displacement = tide.displacement(in_m)
    assert isinstance(displacement, scb.ArrayWUnits) and displacement.units == km
    np.testing.assert_allclose(displacement.values, tide.displacement(position).values,
                               rtol=1e-14, atol=0)
    for name, rows in (("state_displacement", tide.STATE_UNITS),
                       ("state_displacement_partials", tide.STATE_UNITS),
                       ("state_displacement_pole_partials", tide.POLE_PARTIAL_UNITS)):
        result = getattr(tide, name)(in_m)
        np.testing.assert_allclose(result.values, getattr(tide, f"_{name}")(position),
                                   rtol=1e-14, atol=0)
        for units, row in zip(np.broadcast_to(result.units, result.shape), rows):
            assert all(unit == row for unit in np.atleast_1d(units))
