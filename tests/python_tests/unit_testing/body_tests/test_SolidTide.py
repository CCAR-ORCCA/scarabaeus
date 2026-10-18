# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Solid-tide station displacement, checked against tidal physics.

The reference values here come from the shape of a tidal bulge and from a
closed-form amplitude worked out in the test, not from re-running the class's
own formula, so an error in the implementation cannot make the test agree with
itself.
"""
import numpy as np
import pytest
import spiceypy as spice
import scarabaeus as scb

ET = 1.0e6
RADIUS = 6378.1366


@pytest.fixture(scope="module", autouse=True)
def kernels():
    spice.kclear()
    spice.furnsh("data/kernels/locked/lsk/naif0012.tls")
    spice.furnsh("data/kernels/locked/pck/pck00010.tpc")
    spice.furnsh("data/kernels/locked/spk/de432s.bsp")
    yield
    spice.kclear()


def moon_direction(frame="IAU_EARTH", epoch=ET):
    position = np.asarray(spice.spkpos("MOON", epoch, frame, "NONE", "EARTH")[0])
    return position / np.linalg.norm(position), np.linalg.norm(position)


def test_bulge_shape_is_the_classic_factor_of_minus_two():
    """A tidal bulge lifts the sub-body point twice as far as it drops the limb.

    The degree-2 radial response goes as 3cos^2/2 - 1/2, which is +1 under the
    tide raiser and -1/2 ninety degrees away. That ratio is the signature of a
    tide and is independent of every constant in the model.
    """
    tide = scb.SolidTide(tide_bodies=("MOON",))
    direction, _ = moon_direction()
    sub_body = direction * RADIUS
    limb = np.cross(direction, [0.0, 0.0, 1.0])
    limb = limb / np.linalg.norm(limb) * RADIUS

    radial_sub = tide.displacement(sub_body, ET, "IAU_EARTH").values @ direction
    radial_limb = tide.displacement(limb, ET, "IAU_EARTH").values @ (limb / RADIUS)
    assert radial_sub / radial_limb == pytest.approx(-2.0, rel=1e-9)


def test_sub_lunar_amplitude_matches_the_closed_form():
    """At the sub-lunar point the radial term reduces to a one-line formula."""
    tide = scb.SolidTide(tide_bodies=("MOON",))
    direction, distance = moon_direction()
    gm_moon = float(scb.constants.MOON.GM.values)
    gm_earth = float(scb.constants.EARTH.GM.values)
    expected = (gm_moon / gm_earth) * (RADIUS**4 / distance**3) * tide.love_h2
    actual = tide.displacement(direction * RADIUS, ET, "IAU_EARTH").values @ direction
    assert actual == pytest.approx(expected, rel=1e-9)
    # Roughly 22 cm, the textbook lunar radial tide.
    assert 0.15 < expected * 1e3 < 0.30


def test_horizontal_term_vanishes_under_the_tide_raiser_and_at_the_limb():
    """The Shida term goes as cos(psi) sin(psi), so it is zero at 0 and 90 deg."""
    tide = scb.SolidTide(tide_bodies=("MOON",))
    direction, _ = moon_direction()
    for point in (direction * RADIUS,
                  np.cross(direction, [0., 0., 1.]) /
                  np.linalg.norm(np.cross(direction, [0., 0., 1.])) * RADIUS):
        unit = point / np.linalg.norm(point)
        delta = tide.displacement(point, ET, "IAU_EARTH").values
        horizontal = delta - (delta @ unit) * unit
        assert np.linalg.norm(horizontal) < 1e-15


def test_amplitude_scales_with_the_love_and_shida_numbers():
    direction, _ = moon_direction()
    point = (direction + np.array([0.0, 0.0, 0.7]))
    point = point / np.linalg.norm(point) * RADIUS
    unit = point / RADIUS
    base = scb.SolidTide(tide_bodies=("MOON",))
    doubled = scb.SolidTide(tide_bodies=("MOON",), love_h2=2 * base.love_h2,
                                 shida_l2=2 * base.shida_l2)
    np.testing.assert_allclose(doubled.displacement(point, ET, "IAU_EARTH").values,
                               2 * base.displacement(point, ET, "IAU_EARTH").values,
                               rtol=1e-12)
    # Radial responds to h2 alone, horizontal to l2 alone.
    only_h = scb.SolidTide(tide_bodies=("MOON",), shida_l2=0.0)
    delta = only_h.displacement(point, ET, "IAU_EARTH").values
    assert np.linalg.norm(delta - (delta @ unit) * unit) < 1e-15


def test_moon_and_sun_contributions_add():
    point = np.array([RADIUS * 0.6, RADIUS * 0.5, RADIUS * 0.62])
    point = point / np.linalg.norm(point) * RADIUS
    both = scb.SolidTide().displacement(point, ET, "IAU_EARTH").values
    moon = scb.SolidTide(tide_bodies=("MOON",)).displacement(point, ET, "IAU_EARTH").values
    sun = scb.SolidTide(tide_bodies=("SUN",)).displacement(point, ET, "IAU_EARTH").values
    np.testing.assert_allclose(both, moon + sun, rtol=1e-12, atol=1e-18)

    # The Sun raises roughly half the Moon's tide. Compare the two at their own
    # sub-body points: the tide-raising strength goes as GM/R^3, and the ratio
    # of vector norms at some arbitrary third point is not that quantity.
    amplitudes = {}
    for body, model in (("MOON", moon), ("SUN", sun)):
        direction, distance = moon_direction("IAU_EARTH") if body == "MOON" else (
            lambda p: (p / np.linalg.norm(p), np.linalg.norm(p)))(
            np.asarray(spice.spkpos("SUN", ET, "IAU_EARTH", "NONE", "EARTH")[0]))
        single = scb.SolidTide(tide_bodies=(body,))
        amplitudes[body] = single.displacement(direction * RADIUS, ET, "IAU_EARTH").values @ direction
    ratio = amplitudes["SUN"] / amplitudes["MOON"]
    assert ratio == pytest.approx(0.485, abs=0.02)
    assert amplitudes["MOON"] * 1e3 == pytest.approx(0.217, abs=0.01)
    assert amplitudes["SUN"] * 1e3 == pytest.approx(0.105, abs=0.01)


def test_displacement_magnitude_is_decimetre_scale_over_a_day():
    """Radial swing of a few tenths of a metre, horizontal under a tenth."""
    tide = scb.SolidTide()
    point = np.array([4849.3, -360.4, 4114.8])       # a mid-latitude site
    unit = point / np.linalg.norm(point)
    epochs = ET + np.arange(0.0, 86400.0, 600.0)
    delta = np.array([tide.displacement(point, t, "IAU_EARTH").values for t in epochs]) * 1e3
    radial = delta @ unit
    horizontal = np.linalg.norm(delta - radial[:, None] * unit, axis=1)
    assert 0.15 < np.ptp(radial) < 0.60
    assert horizontal.max() < 0.12


def test_state_displacement_rate_matches_a_tight_finite_difference():
    tide = scb.SolidTide()
    point = np.array([4849.3, -360.4, 4114.8])
    state = tide.state_displacement(point, ET, "IAU_EARTH").values
    np.testing.assert_allclose(state[:3], tide.displacement(point, ET, "IAU_EARTH").values, rtol=0)
    reference = (tide.displacement(point, ET + 1.0, "IAU_EARTH").values
                 - tide.displacement(point, ET - 1.0, "IAU_EARTH").values) / 2.0
    np.testing.assert_allclose(state[3:], reference, rtol=1e-4, atol=1e-14)
    # Tens of microns per second: small, but the same order as the frame term
    # the state transformation adds, and Doppler noise is quoted near 1 mm/s.
    assert 1e-9 < np.linalg.norm(state[3:]) < 1e-7


def test_permanent_tide_option_removes_the_iers_zero_frequency_term():
    point = np.array([4849.3, -360.4, 4114.8])
    unit = point / np.linalg.norm(point)
    kept = scb.SolidTide()
    removed = scb.SolidTide(remove_permanent_tide=True)
    difference = (kept.displacement(point, ET, "IAU_EARTH").values
                  - removed.displacement(point, ET, "IAU_EARTH").values) * 1e3
    sin_phi = unit[2]
    legendre = 1.5 * sin_phi**2 - 0.5
    assert difference @ unit == pytest.approx((-0.1206 + 0.0001 * legendre) * legendre,
                                              rel=1e-9)
    # Time independent, which is what "permanent" means.
    later = (kept.displacement(point, ET + 43200.0, "IAU_EARTH").values
             - removed.displacement(point, ET + 43200.0, "IAU_EARTH").values) * 1e3
    np.testing.assert_allclose(difference, later, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"rate_step": 0.0}, {"rate_step": -1.0}, {"tide_bodies": ()},
    {"love_h2": float("nan")},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        scb.SolidTide(**kwargs)


def test_displacement_at_the_body_centre_is_rejected():
    with pytest.raises(ValueError, match="body centre"):
        scb.SolidTide().displacement(np.zeros(3), ET, "IAU_EARTH")


# --------------------------------------------------------------------------- #
# Integration with GroundStation                                              #
# --------------------------------------------------------------------------- #
def test_station_nominal_state_carries_the_displacement():
    point = np.array([4849.3, -360.4, 4114.8])
    station = scb.GroundStation("TIDE_GS", 399099, body_fixed_frame="IAU_EARTH",
                                body_fixed_position=point)
    plain = station.nominal_state(ET, "IAU_EARTH").quantity.values
    np.testing.assert_allclose(plain, np.r_[point, np.zeros(3)], atol=1e-12)

    tide = scb.SolidTide()
    station.surface_displacement = tide
    moved = station.nominal_state(ET, "IAU_EARTH").quantity.values
    np.testing.assert_allclose(moved, plain + tide.state_displacement(
        point, ET, "IAU_EARTH").values, rtol=0, atol=1e-15)
    # The station now has a body-fixed velocity it did not have before.
    assert np.linalg.norm(moved[3:]) > 0.0

    # In an inertial frame the displacement is rotated, not dropped.
    inertial = station.nominal_state(ET, "J2000").quantity.values
    expected = spice.sxform("IAU_EARTH", "J2000", ET) @ moved
    np.testing.assert_allclose(inertial, expected, rtol=0, atol=1e-15)


def test_station_partials_are_taken_about_the_displaced_point():
    """The tide is nominal geometry, so estimated partials linearise about it."""
    point = np.array([4849.3, -360.4, 4114.8])
    station = scb.GroundStation("TIDE_GS2", 399098, body_fixed_frame="IAU_EARTH",
                                body_fixed_position=point)
    plain = station.rotation_partials(ET, "IAU_EARTH").values
    station.surface_displacement = scb.SolidTide()
    moved = station.rotation_partials(ET, "IAU_EARTH").values
    # d/d(rotation) = -[r x], so the blocks differ by the skew of the tide.
    delta = station.nominal_state(ET, "IAU_EARTH").quantity.values[:3] - point
    expected = plain[:3] - np.array([[0.0, -delta[2], delta[1]],
                                     [delta[2], 0.0, -delta[0]],
                                     [-delta[1], delta[0], 0.0]])
    np.testing.assert_allclose(moved[:3], expected, rtol=0, atol=1e-12)


def test_surface_displacement_rejects_an_object_without_the_hook():
    station = scb.GroundStation("TIDE_GS3", 399097)
    with pytest.raises(TypeError, match="state_displacement"):
        station.surface_displacement = object()
    station.surface_displacement = None
    assert station.surface_displacement is None


# --------------------------------------------------------------------------- #
# Body neutrality                                                             #
# --------------------------------------------------------------------------- #
def test_a_body_with_no_known_response_must_be_given_one():
    """Earth's Love numbers must never be silently applied to another planet.

    The tidal response varies by a factor of several between bodies -- Mars is
    near h2 = 0.17 against Earth's 0.61 -- so a default would be a large
    silent error, not an approximation.
    """
    with pytest.raises(scb.SolidTide.UnknownTidalResponseError, match="MARS"):
        scb.SolidTide(central_body="MARS")
    with pytest.raises(scb.SolidTide.UnknownTidalResponseError):
        scb.SolidTide(central_body="MARS", love_h2=0.17)     # still missing l2
    mars = scb.SolidTide(central_body="MARS", love_h2=0.17, shida_l2=0.02,
                         love_k={(2, 0): 0.169}, tide_bodies=("SUN", "PHOBOS"))
    assert (mars.love_h2, mars.shida_l2, mars.tide_bodies) == (
        0.17, 0.02, ("SUN", "PHOBOS"))


def test_earth_defaults_come_from_the_nominal_table():
    tide = scb.SolidTide()
    nominal = scb.SolidTide.NOMINAL_RESPONSE["EARTH"]
    assert nominal is scb.constants.EARTH_SOLID_TIDE
    assert tide.love_h2 == nominal.love_h2 == 0.6078
    assert tide.shida_l2 == nominal.shida_l2 == 0.0847
    assert tide.tide_bodies == nominal.tide_bodies
    assert tide.love_k == nominal.love_k


def test_permanent_tide_removal_is_earth_only():
    """IERS eq. 7.14 is an Earth fit; there is no equivalent elsewhere."""
    with pytest.raises(scb.SolidTide.UnknownTidalResponseError, match="7.14"):
        scb.SolidTide(central_body="MARS", love_h2=0.17, shida_l2=0.02,
                      love_k={(2, 0): 0.169}, tide_bodies=("SUN",),
                      remove_permanent_tide=True)


# --------------------------------------------------------------------------- #
# The geopotential side                                                       #
# --------------------------------------------------------------------------- #
def test_geopotential_coefficients_match_an_independent_evaluation():
    """IERS eq. 6.6 step 1, written out here rather than reusing the class."""
    tide = scb.SolidTide(tide_bodies=("MOON",))
    actual = tide.geopotential_coefficients(ET, "IAU_EARTH")
    gm_earth = float(scb.constants.EARTH.GM.values)
    gm_moon = float(scb.constants.MOON.GM.values)
    r = np.asarray(spice.spkpos("MOON", ET, "IAU_EARTH", "NONE", "EARTH")[0])
    d = np.linalg.norm(r)
    s = r[2] / d
    c = np.sqrt(1 - s**2)
    lon = np.arctan2(r[1], r[0])
    legendre = {0: np.sqrt(5.) * (3 * s**2 - 1) / 2,
                1: np.sqrt(5. / 3.) * 3 * s * c,
                2: np.sqrt(5. / 12.) * 3 * c**2}
    for order in (0, 1, 2):
        term = (tide.love_k[(2, order)] / 5.) * (gm_moon / gm_earth) * (RADIUS / d)**3 \
            * legendre[order] * np.exp(-1j * order * lon)
        assert actual[(2, order)][0] == pytest.approx(term.real, rel=1e-10, abs=1e-20)
        assert actual[(2, order)][1] == pytest.approx(-term.imag, rel=1e-10, abs=1e-20)


def test_geopotential_perturbation_is_linear_in_each_love_number():
    from dataclasses import replace
    base = scb.SolidTide()
    doubled = replace(base, love_k={k: 2 * v for k, v in base.love_k.items()})
    a = base.geopotential_coefficients(ET, "IAU_EARTH")
    b = doubled.geopotential_coefficients(ET, "IAU_EARTH")
    # Degrees 2 and 3, every order: the table drives the resolution.
    assert set(a) == {(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)}
    for key in a:
        assert b[key][0] == pytest.approx(2 * a[key][0], rel=1e-12)
        assert b[key][1] == pytest.approx(2 * a[key][1], rel=1e-12)
    # An m=0 term has no sine component.
    assert a[(2, 0)][1] == pytest.approx(0.0, abs=1e-22)
    assert a[(3, 0)][1] == pytest.approx(0.0, abs=1e-22)
    # Degree 3 is orders of magnitude below degree 2.
    assert abs(a[(3, 0)][0]) < 0.01 * abs(a[(2, 0)][0])


def test_normalized_legendre_matches_the_closed_forms():
    """The general recursion must reproduce the degree-2 and 3 closed forms."""
    for sine in (0.0, 0.3, -0.7, 0.95):
        cosine = np.sqrt(1 - sine**2)
        actual = scb.SolidTide._normalized_legendre(3, sine)
        expected = {
            (2, 0): np.sqrt(5) * (3 * sine**2 - 1) / 2,
            (2, 1): np.sqrt(5 / 3) * 3 * sine * cosine,
            (2, 2): np.sqrt(5 / 12) * 3 * cosine**2,
            (3, 0): np.sqrt(7) * (5 * sine**3 - 3 * sine) / 2,
            (3, 1): np.sqrt(7 / 6) * 1.5 * (5 * sine**2 - 1) * cosine,
            (3, 2): np.sqrt(7 / 60) * 15 * sine * cosine**2,
            (3, 3): np.sqrt(7 / 360) * 15 * cosine**3,
        }
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_geopotential_perturbation_is_about_1e_minus_5_of_c20():
    """Roughly 5e-9 in normalized C20, 1e-5 of the nominal field."""
    delta = scb.SolidTide().geopotential_coefficients(ET, "IAU_EARTH")[(2, 0)][0]
    assert 1e-9 < abs(delta) < 2e-8
    assert 1e-6 < abs(delta / -4.8417e-4) < 1e-4


def test_potential_and_displacement_love_numbers_are_distinct():
    """h2 moves the surface, k2 changes the field. Confusing them is a
    factor-of-two error, so they must not share a value or a code path."""
    from dataclasses import replace
    tide = scb.SolidTide()
    assert tide.love_k[(2, 0)] != tide.love_h2
    assert tide.love_k[(2, 0)] == pytest.approx(0.29525)
    # IERS gives k per order, not one number.
    assert tide.love_k[(2, 0)] != tide.love_k[(2, 2)]
    point = np.array([4849.3, -360.4, 4114.8])
    # Changing k must not move the surface; changing h2 must not move the field.
    other_k = replace(tide, love_k={k: 0.5 for k in tide.love_k})
    np.testing.assert_allclose(other_k.displacement(point, ET, "IAU_EARTH").values,
                               tide.displacement(point, ET, "IAU_EARTH").values, rtol=0, atol=0)
    other_h = scb.SolidTide(love_h2=0.9)
    assert (other_h.geopotential_coefficients(ET, "IAU_EARTH")[(2, 0)]
            == tide.geopotential_coefficients(ET, "IAU_EARTH")[(2, 0)])


# ---------------------------------------------------------------------- #
# The complete IERS Conventions (2010) routine, model="IERS2010"        #
# ---------------------------------------------------------------------- #

def _tt_centuries(year, month, day, hours, tai_minus_utc):
    """T as DEHANTTIDEINEL forms it: CAL2JD plus DAT plus 32.184 s."""
    a = (14 - month) // 12
    y, m = year + 4800 - a, month + 12 * a - 3
    day_number = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return (((day_number - 0.5) - 2451545.0) + hours / 24.0) / 36525.0 + (
        tai_minus_utc + 32.184) / (86400.0 * 36525.0)


@pytest.mark.parametrize("station, sun, moon, date, expected", [
    ((4075578.385, 931852.890, 4801570.154),
     (137859926952.015, 54228127881.4350, 23509422341.6960),
     (-179996231.920342, -312468450.131567, -169288918.592160), (2009, 4, 13, 34.0),
     (0.7700420357108125891e-01, 0.6304056321824967613e-01, 0.5516568152597246810e-01)),
    ((1112189.660, -4842955.026, 3985352.284),
     (-54537460436.2357, 130244288385.279, 56463429031.5996),
     (300396716.912, 243238281.451, 120548075.939), (2012, 7, 13, 35.0),
     (-0.2036831479592075833e-01, 0.5658254776225972449e-01, -0.7597679676871742227e-01)),
    ((1112200.5696, -4842957.8511, 3985345.9122),
     (100210282451.6279, 103055630398.3160, 56855096480.4475),
     (369817604.4348, 1897917.5258, 120804980.8284), (2015, 7, 15, 36.0),
     (.00509570869172363845, .0828663025983528700, -.0636634925404189617)),
], ids=["2009", "2012", "2015"])
def test_iers2010_routine_reproduces_the_published_test_cases(station, sun, moon, date, expected):
    """The test cases shipped in the IERS DEHANTTIDEINEL source, to 1e-15 m."""
    year, month, day, tai_minus_utc = date
    actual = scb.SolidTide.iers2010_displacement(np.array(station), 0.0,
                                                 _tt_centuries(year, month, day, 0.0, tai_minus_utc),
                                                 np.array(sun), np.array(moon))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_iers2010_model_adds_centimetre_level_terms_to_degree2():
    """The complete routine differs from the degree-2 in-phase term by its
    small terms: more than a millimetre, well under two centimetres."""
    point = np.array([4849.3, -360.4, 4114.8])
    for epoch in ET + np.arange(0.0, 86400.0, 7200.0):
        degree2 = scb.SolidTide().displacement(point, epoch, "IAU_EARTH").values
        complete = scb.SolidTide(model="IERS2010").displacement(point, epoch, "IAU_EARTH").values
        assert np.linalg.norm(complete - degree2) < 2e-5
    # Over a day the K1 correction alone swings by about a centimetre.
    swing = [np.linalg.norm(scb.SolidTide(model="IERS2010").displacement(point, e, "IAU_EARTH").values
                            - scb.SolidTide().displacement(point, e, "IAU_EARTH").values)
             for e in ET + np.arange(0.0, 86400.0, 3600.0)]
    assert max(swing) > 2e-6


def test_iers2010_model_is_earths_with_its_own_numbers():
    with pytest.raises(scb.SolidTide.UnknownModelError):
        scb.SolidTide(model="degree3")
    with pytest.raises(scb.SolidTide.UnknownModelError):
        scb.SolidTide(model="IERS2010", love_h2=0.60)
    with pytest.raises(scb.SolidTide.UnknownModelError):
        scb.SolidTide(model="IERS2010", central_body="MARS", love_h2=0.17,
                      shida_l2=0.03, tide_bodies=("SUN",))
    assert scb.SolidTide(model="IERS2010").love_h2 == 0.6078
