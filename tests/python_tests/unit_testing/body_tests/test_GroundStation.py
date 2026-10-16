# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================#
#  Unit Tests for the GroundStation class  #
# ==========================================#
"""
GroundStation represents a DSN or user-defined ground station, providing
state queries and communication modeling between the station and a
spacecraft.

Tests
-----
test_initialization
    Verifies correct attribute assignment for construction with and
    without an explicit SPICE ID.
test_barycentric_transformation_*
    The geocentric-to-barycentric station transformation: off by default,
    identical to the one the light-time observables already apply, about
    16 cm inward, position only, and independent of the output axes.
test_body_fixed_velocity_*
    Plate motion of explicit coordinates: moved from their epoch at every
    epoch, the epoch required, kernel stations refused, and a tide attached
    on top still visible to the Love-number partials.
test_constructor_keywords_match_the_setters
    Tides, the barycentric transformation and an orientation correction
    given at construction build the same station as the property setters.
test_epoch_array_and_frame_inputs_match_seconds_and_names
    Every epoch/frame query gives the same result for an EpochArray and a
    Frame as for TDB seconds and a frame name.
test_displacement_state_is_the_term_nominal_state_adds
    The attached displacement the kernel-reading models add themselves:
    zero with none attached, the gap between the nominal and undisplaced
    states body-fixed, and that gap rotated in other axes.
test_tide_correction_moves_the_tide_to_the_estimated_numbers
    Zero with no tide, no estimate or the model's own numbers; otherwise
    the Love partials times the offset, which places the station where a
    tide built with the estimated numbers does.
test_tide_correction_shifts_every_tide_from_the_reference_numbers
    A solid and a pole tide keep their different nominal numbers: the
    estimate is read as the reference tide's, every tide shifts by the same
    amount, and the partials are the sum of both models'.
test_pole_tide_partials_move_the_pole_tide_with_the_pole
    Zero with no pole tide; otherwise the Jacobian times a pole delta places
    the station where a pole tide built at the moved pole does.
test_inertial_rotation_partials_turn_the_station_about_j2000_axes
    The Jacobian times a small J2000 rotation matches the nominal station
    rotated by that exact rotation matrix, in J2000 and in ITRF93, and
    differs from the body-fixed rotation_partials.
test_iau_pole_correction_is_zero_without_an_override
    No estimated pole, meridian or spin rate leaves the station where its
    kernel or coordinates put it, in J2000 and body-fixed axes.
test_iau_pole_partials_match_differences_of_the_correction
    Each of the four IAU columns is the central difference of
    iau_pole_correction, about given values and about the unset model, for
    a Mars lander and an Earth station, in J2000 and body-fixed axes.
test_iau_pole_override_turns_a_mars_lander_with_the_estimated_rotation
    On Mars, whose IAU frame is the secular model, the moved lander is the
    estimated rotation applied to its body-fixed coordinates, and moves with
    that rotation's time derivative.
"""

import numpy as np
import pytest
import spiceypy as spice

import src.scarabaeus as scb

#: An epoch inside every kernel below, TDB seconds past J2000 (2025).
EPOCH = 8.0e8

m, sec, mrad, deg, day = scb.Units.get_units(["m", "sec", "mrad", "deg", "day"])

# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(scope="module")
def kernels():
    """Ephemeris, Earth orientation, the IAU constants and the DSN station kernel."""
    spice.kclear()
    for kernel in ("lsk/naif0012.tls", "spk/de432s.bsp", "pck/pck00010.tpc",
                   "spk/earth_200101_990628_predict.bpc",
                   "spk/earthstns_itrf93_201023.bsp"):
        spice.furnsh(f"data/kernels/locked/{kernel}")
    yield
    spice.kclear()


def dss35(transformed):
    """DSS 35 from the station kernel, with or without the transformation."""
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")
    station.barycentric_transformation = transformed
    return station


def bare(quantity):
    """Numbers of an ArrayWFrame or ArrayWUnits result, in its own units."""
    quantity = getattr(quantity, "quantity", quantity)
    return getattr(quantity, "values", quantity)


def element_units(quantity):
    """Units of every element of an ArrayWUnits, as an object array."""
    units = np.empty(quantity.shape, dtype=object)
    units[...] = quantity.units
    return units

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
@pytest.mark.parametrize(
    "init_args",
    [("DSS-14", 399014), ("DSS-24", None)],
    ids=["with ID", "Without ID"],
)
def test_initialization(init_args):
    """
    Verifies that object is constructed correctly for all valid initialization configurations.
    """
    gs = scb.GroundStation(*init_args)

    assert gs.name == init_args[0]
    if init_args[1] is not None:
        assert gs.spice_id == init_args[1]


def test_barycentric_transformation_is_off_by_default_and_validated():
    station = scb.GroundStation("DSS-35", 399035)
    assert station.barycentric_transformation is False
    with pytest.raises(TypeError):
        station.barycentric_transformation = 1


def test_barycentric_transformation_matches_the_light_time_model(kernels):
    """The light-time observables carry the same transformation (Moyer eq.
    4-10) in their own code path; the two must give the same station."""
    ours = bare(dss35(True).nominal_state(EPOCH, "J2000"))[:3]
    light_time = scb.Measurement._compute_GS_in_SSB_space_time_RLT("399035", EPOCH)[:3]
    assert np.max(np.abs(ours - light_time)) < 1e-12


def test_barycentric_transformation_moves_the_station_about_16_cm_inward(kernels):
    """The scale term, 2.5e-8, is 16 cm at the Earth's radius; the velocity
    term adds at most 3 cm, so the shift is 12-20 cm and points nearly at the
    geocentre."""
    plain = bare(dss35(False).nominal_state(EPOCH, "J2000"))
    moved = bare(dss35(True).nominal_state(EPOCH, "J2000"))
    shift = moved[:3] - plain[:3]
    assert 0.12e-3 < np.linalg.norm(shift) < 0.20e-3
    cosine = -shift @ plain[:3] / (np.linalg.norm(shift) * np.linalg.norm(plain[:3]))
    assert cosine > 0.95
    # Position only: the velocity and the body-fixed state are untouched.
    np.testing.assert_array_equal(moved[3:], plain[3:])
    np.testing.assert_array_equal(bare(dss35(True).nominal_state(EPOCH)),
                                  bare(dss35(False).nominal_state(EPOCH)))


def test_barycentric_transformation_is_the_same_in_any_axes(kernels):
    """Built from a scale and a dot product, the transformation commutes with
    a rotation of the output axes."""
    j2000 = bare(dss35(True).nominal_state(EPOCH, "J2000"))[:3]
    ecliptic = bare(dss35(True).nominal_state(EPOCH, "ECLIPJ2000"))[:3]
    rotated = spice.pxform("J2000", "ECLIPJ2000", EPOCH) @ j2000
    assert np.max(np.abs(ecliptic - rotated)) < 1e-12


#: A DSS 35-like explicit station, km, and a 5 cm/yr plate velocity, km/s.
POSITION = np.array([-4461.1, 2682.3, -3674.2])
VELOCITY = np.array([-0.0335, -0.0041, 0.0392]) / 1e3 / (365.25 * 86400.0)
#: 2003 JAN 01 00:00:00 TDB.
POSITION_EPOCH = 94651200.0


def moving_station():
    return scb.GroundStation("DSS-35", 399035, body_fixed_position=POSITION,
                             body_fixed_velocity=VELOCITY, position_epoch=POSITION_EPOCH)


def test_body_fixed_velocity_moves_the_station_from_its_position_epoch():
    station = moving_station()
    for epoch in (POSITION_EPOCH, EPOCH, 9.2e8):
        state = bare(station.nominal_state(epoch))
        np.testing.assert_array_equal(state[:3], POSITION + VELOCITY * (epoch - POSITION_EPOCH))
        np.testing.assert_array_equal(state[3:], VELOCITY)
    km, sec = scb.Units.get_units(["km", "sec"])
    np.testing.assert_array_equal(station.body_fixed_velocity.values, VELOCITY)
    assert station.body_fixed_velocity.units == km / sec
    # Without a velocity the explicit station stays where it is.
    fixed = scb.GroundStation("DSS-35", 399035, body_fixed_position=POSITION)
    np.testing.assert_array_equal(bare(fixed.nominal_state(EPOCH)), np.r_[POSITION, 0, 0, 0])


def test_body_fixed_velocity_needs_a_position_epoch_and_explicit_coordinates():
    station = moving_station()
    station.position_epoch = None
    with pytest.raises(scb.GroundStation.MissingPositionEpochError):
        station.nominal_state(EPOCH)
    kernel_station = scb.GroundStation("DSS-35", 399035, body_fixed_velocity=VELOCITY)
    with pytest.raises(scb.GroundStation.VelocityWithoutPositionError):
        kernel_station.nominal_state(EPOCH)
    with pytest.raises(ValueError):
        station.body_fixed_velocity = [1.0, 2.0]
    with pytest.raises(ValueError):
        station.position_epoch = float("nan")


def test_body_fixed_velocity_keeps_the_tide_visible_to_its_partials(kernels):
    """Plate motion is the station's own; the tide stays the surface
    displacement, so its Love-number partials are not hidden behind it."""
    station = moving_station()
    station.surface_displacement = scb.SolidTide(model="IERS2010")
    moved = station._undisplaced_state(EPOCH)[:3]
    expected = scb.SolidTide(model="IERS2010").state_displacement(moved, EPOCH, "ITRF93")
    np.testing.assert_allclose(bare(station.nominal_state(EPOCH))
                               - station._undisplaced_state(EPOCH),
                               bare(expected), rtol=0, atol=1e-12)
    assert np.all(np.abs(bare(station.tide_partials(EPOCH))[:3]) > 0)


def test_constructor_keywords_match_the_setters(kernels):
    tides = [scb.SolidTide(model="IERS2010"), scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    correction = scb.BodyOrientation.earth_correction(delta_ut1=1e-3)
    built = scb.GroundStation(
        "DSS-35", 399035, body_fixed_position=POSITION, body_fixed_velocity=VELOCITY,
        position_epoch=POSITION_EPOCH, surface_displacement=tides,
        barycentric_transformation=True, body_orientation=correction)
    configured = moving_station()
    configured.surface_displacement = tides
    configured.barycentric_transformation = True
    configured.body_orientation = correction
    assert built.surface_displacement is tides
    assert built.body_orientation is correction
    for frame in ("ITRF93", "J2000"):
        np.testing.assert_array_equal(bare(built.nominal_state(EPOCH, frame)),
                                      bare(configured.nominal_state(EPOCH, frame)))
    cylindrical = scb.GroundStation.from_cylindrical(
        "DSS-35", 399035, longitude=2.6, cylindrical_radius=5205.3, z=-3674.2,
        surface_displacement=tides, barycentric_transformation=True)
    assert cylindrical.surface_displacement is tides
    assert cylindrical.barycentric_transformation is True
    with pytest.raises(TypeError):
        scb.GroundStation("DSS-35", 399035, barycentric_transformation=1)
    with pytest.raises(TypeError):
        scb.GroundStation("DSS-35", 399035, surface_displacement=object())


def test_epoch_array_and_frame_inputs_match_seconds_and_names(kernels):
    epoch = scb.EpochArray(EPOCH, sys="TDB")
    seconds = float(np.ravel(epoch.et)[0])
    j2000, itrf = scb.Frame("J2000"), scb.Frame("ITRF93")
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame=itrf,
                                surface_displacement=scb.SolidTide(model="IERS2010"),
                                body_orientation=scb.BodyOrientation.earth_correction(delta_ut1=1e-3))
    assert station.body_fixed_frame == "ITRF93"
    location = np.array([1e-3, -2e-3, 3e-3])
    pairs = [
        (station.nominal_state(seconds, "J2000"), station.nominal_state(epoch, j2000)),
        (station.state_correction(seconds, "J2000", location=location, location_frame="ITRF93"),
         station.state_correction(epoch, j2000, location=location, location_frame=itrf)),
        (station.location_partials(seconds, "J2000", location_frame="ITRF93"),
         station.location_partials(epoch, j2000, location_frame=itrf)),
        (station.rotation_partials(seconds, "J2000"), station.rotation_partials(epoch, j2000)),
        (station.inertial_rotation_partials(seconds, "J2000"),
         station.inertial_rotation_partials(epoch, j2000)),
        (station.tide_partials(seconds, "J2000"), station.tide_partials(epoch, j2000)),
        (station.eop_displacement(seconds, "J2000", as_state=True),
         station.eop_displacement(epoch, j2000, as_state=True)),
        *zip(station.cylindrical_basis(seconds, "ITRF93"), station.cylindrical_basis(epoch, itrf)),
    ]
    for by_name, by_object in pairs:
        np.testing.assert_array_equal(bare(by_object), bare(by_name))
    moving = moving_station()
    moving.position_epoch = scb.EpochArray(POSITION_EPOCH, sys="TDB")
    assert moving.position_epoch == pytest.approx(POSITION_EPOCH, abs=1e-6)


def test_displacement_state_is_the_term_nominal_state_adds(kernels):
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")
    np.testing.assert_array_equal(bare(station.displacement_state(EPOCH)), np.zeros(6))
    station.surface_displacement = [scb.SolidTide(model="IERS2010"),
                                    scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    body_fixed = bare(station.displacement_state(EPOCH))
    np.testing.assert_allclose(body_fixed,
                               bare(station.nominal_state(EPOCH))
                               - station._undisplaced_state(EPOCH),
                               rtol=0, atol=1e-12)
    assert np.linalg.norm(body_fixed[:3]) > 1e-5
    np.testing.assert_allclose(bare(station.displacement_state(EPOCH, "J2000")),
                               spice.sxform("ITRF93", "J2000", EPOCH) @ body_fixed,
                               rtol=0, atol=1e-15)


def test_tide_correction_moves_the_tide_to_the_estimated_numbers(kernels):
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")
    np.testing.assert_array_equal(bare(station.tide_correction(EPOCH, love_h2=0.7)), np.zeros(6))
    tide = scb.SolidTide()
    station.surface_displacement = tide
    h2, l2, dh, dl = tide.love_h2, tide.shida_l2, 0.05, -0.02
    for no_change in ({}, {"love_h2": h2, "shida_l2": l2}):
        np.testing.assert_array_equal(bare(station.tide_correction(EPOCH, **no_change)),
                                      np.zeros(6))
    for frame in ("ITRF93", "J2000"):
        correction = bare(station.tide_correction(EPOCH, frame, love_h2=h2 + dh, shida_l2=l2 + dl))
        np.testing.assert_allclose(correction,
                                   bare(station.tide_partials(EPOCH, frame)) @ [dh, dl],
                                   rtol=0, atol=1e-15)
    estimated = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93",
                                  surface_displacement=scb.SolidTide(love_h2=h2 + dh,
                                                                     shida_l2=l2 + dl))
    correction = station.tide_correction(EPOCH, "ITRF93", love_h2=h2 + dh, shida_l2=l2 + dl)
    assert np.linalg.norm(bare(correction)[:3]) > 1e-7
    np.testing.assert_allclose(bare(station.nominal_state(EPOCH) + correction),
                               bare(estimated.nominal_state(EPOCH)), rtol=0, atol=1e-11)


def test_tide_correction_shifts_every_tide_from_the_reference_numbers(kernels):
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")
    assert station.tide_reference is None
    solid, pole = scb.SolidTide(), scb.PoleTide(pole_x=1e-6, pole_y=2e-6)
    station.surface_displacement = [solid, pole]
    assert station.tide_reference is solid
    assert pole.love_h2 != solid.love_h2 and pole.shida_l2 != solid.shida_l2
    # The reference numbers leave both tides where they are.
    np.testing.assert_array_equal(bare(station.tide_correction(
        EPOCH, love_h2=solid.love_h2, shida_l2=solid.shida_l2)), np.zeros(6))
    parts = [bare(scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93",
                                    surface_displacement=model).tide_partials(EPOCH))
             for model in (solid, pole)]
    np.testing.assert_allclose(bare(station.tide_partials(EPOCH)), parts[0] + parts[1],
                               rtol=0, atol=1e-18)
    assert np.all(np.abs(parts[1][:3]) > 0)
    dh, dl = 0.05, -0.02
    shifted = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93",
                                surface_displacement=[
                                    scb.SolidTide(love_h2=solid.love_h2 + dh,
                                                  shida_l2=solid.shida_l2 + dl),
                                    scb.PoleTide(pole_x=1e-6, pole_y=2e-6,
                                                 love_h2=pole.love_h2 + dh,
                                                 shida_l2=pole.shida_l2 + dl)])
    correction = station.tide_correction(EPOCH, "ITRF93", love_h2=solid.love_h2 + dh,
                                         shida_l2=solid.shida_l2 + dl)
    np.testing.assert_allclose(bare(station.nominal_state(EPOCH) + correction),
                               bare(shifted.nominal_state(EPOCH)), rtol=0, atol=1e-11)


def test_pole_tide_partials_move_the_pole_tide_with_the_pole(kernels):
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93",
                                surface_displacement=scb.SolidTide())
    np.testing.assert_array_equal(bare(station.pole_tide_partials(EPOCH)), np.zeros((6, 2)))
    station.surface_displacement = [scb.SolidTide(), scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    delta = np.array([3e-7, -4e-7])
    moved = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93",
                              surface_displacement=[
                                  scb.SolidTide(),
                                  scb.PoleTide(pole_x=1e-6 + delta[0], pole_y=2e-6 + delta[1])])
    for frame in ("ITRF93", "J2000"):
        change = bare(station.pole_tide_partials(EPOCH, frame)) @ delta
        assert np.linalg.norm(change[:3]) > 1e-7
        np.testing.assert_allclose(bare(station.nominal_state(EPOCH, frame)) + change,
                                   bare(moved.nominal_state(EPOCH, frame)), rtol=0, atol=1e-11)


def test_inertial_rotation_partials_turn_the_station_about_j2000_axes(kernels):
    station = scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")
    omega = np.array([3e-7, -2e-7, 1e-7])
    to_j2000 = spice.sxform("ITRF93", "J2000", EPOCH)
    inertial = to_j2000 @ bare(station.nominal_state(EPOCH))
    turn = spice.axisar(omega / np.linalg.norm(omega), np.linalg.norm(omega))
    moved = np.r_[turn @ inertial[:3], turn @ inertial[3:]] - inertial
    # Second-order terms are |omega|^2 * 6400 km, below 1e-9 km.
    np.testing.assert_allclose(bare(station.inertial_rotation_partials(EPOCH)) @ omega, moved,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(bare(station.inertial_rotation_partials(EPOCH, "ITRF93")) @ omega,
                               np.linalg.solve(to_j2000, moved), rtol=0, atol=1e-9)
    # The same axes held in the body move the station identically at this
    # instant, but turn with the Earth, so the velocity rows differ.
    body_fixed = bare(station.rotation_partials(EPOCH, "J2000")) @ (to_j2000[:3, :3].T @ omega)
    np.testing.assert_allclose(body_fixed[:3], moved[:3], rtol=0, atol=1e-9)
    assert not np.allclose(body_fixed[3:], moved[3:], rtol=0, atol=1e-12)


def mars_lander():
    """A lander given by IAU_MARS coordinates, with no station kernel."""
    return scb.GroundStation("LANDER", -999001, central_body="MARS",
                             body_fixed_frame="IAU_MARS",
                             body_fixed_position=np.array([3000., 1000., 1500.]))


def iau_station(body):
    if body == "MARS":
        return mars_lander()
    return scb.GroundStation("DSS-35", 399035, body_fixed_frame="ITRF93")


@pytest.mark.parametrize("body", ["MARS", "EARTH"])
def test_iau_pole_correction_is_zero_without_an_override(kernels, body):
    station = iau_station(body)
    for frame in ("J2000", station.body_fixed_frame):
        np.testing.assert_allclose(bare(station.iau_pole_correction(EPOCH, frame)), np.zeros(6),
                                   rtol=0, atol=1e-12)
        assert station.iau_pole_partials(EPOCH, frame).shape == (6, 4)


@pytest.mark.parametrize("body", ["MARS", "EARTH"])
@pytest.mark.parametrize("given", [False, True], ids=["unset", "given"])
def test_iau_pole_partials_match_differences_of_the_correction(kernels, body, given):
    station = iau_station(body)
    coefficients = scb.BodyOrientation.iau_pole_coefficients(body)
    secular, _ = scb.BodyOrientation.iau_angles(EPOCH, None, coefficients)
    # The secular model's own numbers: an unset parameter is linearised there.
    reference = {"pole_ra": secular[0], "pole_dec": secular[1],
                 "pole_pm": coefficients[2][1], "pole_w0": np.deg2rad(coefficients[2][0])}
    values = ({"pole_ra": secular[0] + 1e-3, "pole_dec": secular[1] - 2e-3,
               "pole_pm": coefficients[2][1] + 1e-6,
               "pole_w0": np.deg2rad(coefficients[2][0]) + 1e-3} if given else {})
    steps = {"pole_ra": 1e-7, "pole_dec": 1e-7, "pole_pm": 1e-9, "pole_w0": 1e-7}
    for frame in ("J2000", station.body_fixed_frame):
        jacobian = bare(station.iau_pole_partials(EPOCH, frame, **values))
        # Inertial, since a body-fixed velocity is zero but its round-off is not.
        state = np.abs(bare(station.nominal_state(EPOCH, "J2000")))
        for column, name in enumerate(scb.BodyOrientation.IAU_POLE_PARAMETERS):
            centre = values.get(name, reference[name])
            plus = dict(values, **{name: centre + steps[name]})
            minus = dict(values, **{name: centre - steps[name]})
            difference = bare(station.iau_pole_correction(EPOCH, frame, **plus)
                              - station.iau_pole_correction(EPOCH, frame, **minus)) / (2 * steps[name])
            assert np.abs(difference[:3]).max() > 1.
            # W is ~5e4 rad, so the PM and W0 differences carry ~1e-5
            # cancellation; the floor is the round-off of the state itself,
            # which is all a body-fixed velocity row of a spin column holds.
            for rows in (slice(0, 3), slice(3, 6)):
                floor = 1e-15 * state[rows].max() / steps[name]
                np.testing.assert_allclose(jacobian[rows, column], difference[rows], rtol=0,
                                           atol=1e-4 * np.abs(difference[rows]).max() + floor)


def test_iau_pole_override_turns_a_mars_lander_with_the_estimated_rotation(kernels):
    station = mars_lander()
    coefficients = scb.BodyOrientation.iau_pole_coefficients("MARS")
    pole = {"pole_ra": 0.33, "pole_dec": 0.95, "pole_pm": 350.89, "pole_w0": 3.0}

    def rotation(epoch):
        angles, _ = scb.BodyOrientation.iau_angles(epoch, None, coefficients, **pole)
        return scb.BodyOrientation.iau_matrix(angles)

    body_fixed = bare(station.nominal_state(EPOCH, "IAU_MARS"))
    moved = (spice.sxform("IAU_MARS", "J2000", EPOCH) @ body_fixed
             + bare(station.iau_pole_correction(EPOCH, "J2000", **pole)))
    np.testing.assert_allclose(moved[:3], rotation(EPOCH) @ body_fixed[:3], rtol=0, atol=1e-7)
    rate = (rotation(EPOCH + 1.) - rotation(EPOCH - 1.)) @ body_fixed[:3] / 2.
    np.testing.assert_allclose(moved[3:], rate, rtol=0, atol=1e-8)


def test_states_are_framed_in_km_and_km_per_second(kernels):
    station = dss35(False)
    station.surface_displacement = scb.PoleTide(pole_x=1e-6, pole_y=2e-6)
    station.earth_orientation = scb.BodyOrientation(rotation_vector=[1e-8, 2e-8, 3e-8],
                                                    body_fixed_frame="ITRF93")
    for frame in ("J2000", "ITRF93", scb.Frame("ECLIPJ2000")):
        for state in (station.nominal_state(EPOCH, frame),
                      station.state_correction(EPOCH, frame, location=[1e-3, 0., 0.]),
                      station.displacement_state(EPOCH, frame),
                      station.tide_correction(EPOCH, frame),
                      station.iau_pole_correction(EPOCH, frame),
                      station.eop_displacement(EPOCH, frame, as_state=True)):
            assert isinstance(state, scb.ArrayWFrame)
            assert state.frame == (frame if isinstance(frame, scb.Frame) else scb.Frame(frame))
            for unit, row in zip(element_units(state.quantity), scb.GroundStation.STATE_UNITS):
                assert unit == row
    position = station.eop_displacement(EPOCH)
    assert position.quantity.shape == (3,)
    assert all(unit == scb.GroundStation.STATE_UNITS[0]
               for unit in element_units(position.quantity))


def test_jacobians_carry_the_units_of_their_rows_and_columns(kernels):
    station = dss35(False)
    station.surface_displacement = [scb.SolidTide(model="IERS2010"),
                                    scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    jacobians = [(station.location_partials(EPOCH), station.LOCATION_PARTIAL_UNITS),
                 (station.tide_partials(EPOCH), station.STATE_UNITS),
                 (station.pole_tide_partials(EPOCH), station.ROTATION_PARTIAL_UNITS),
                 (station.rotation_partials(EPOCH), station.ROTATION_PARTIAL_UNITS),
                 (station.inertial_rotation_partials(EPOCH), station.ROTATION_PARTIAL_UNITS)]
    for jacobian, rows in jacobians:
        assert isinstance(jacobian, scb.ArrayWUnits)
        for units, row in zip(element_units(jacobian), rows):
            assert all(unit == row for unit in units)
    _, derivative = station.cylindrical_basis(EPOCH)
    assert all(unit == scb.Units.get_units(["unitless"]) / sec
               for unit in element_units(derivative).flat)
    units = element_units(station.iau_pole_partials(EPOCH))
    pm = scb.BodyOrientation.IAU_POLE_PARAMETERS.index("pole_pm")
    for row, (angle, rate) in enumerate(zip(station.ROTATION_PARTIAL_UNITS,
                                            station.POLE_PM_PARTIAL_UNITS)):
        assert units[row, pm] == rate
        assert all(unit == angle for column, unit in enumerate(units[row]) if column != pm)


def test_public_results_are_their_numeric_cores_unchanged(kernels):
    station = dss35(False)
    station.surface_displacement = [scb.SolidTide(model="IERS2010"),
                                    scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    for name in ("nominal_state", "displacement_state", "tide_correction", "tide_partials",
                 "pole_tide_partials", "location_partials", "rotation_partials",
                 "inertial_rotation_partials", "iau_pole_correction", "iau_pole_partials"):
        np.testing.assert_array_equal(bare(getattr(station, name)(EPOCH, "J2000")),
                                      getattr(station, f"_{name}")(EPOCH, "J2000"))
    np.testing.assert_array_equal(
        bare(station.state_correction(EPOCH, location=[1e-3, 2e-3, 0.],
                                      rotation_vector=[1e-8, 0., 0.])),
        station._state_correction(EPOCH, location=[1e-3, 2e-3, 0.],
                                  rotation_vector=[1e-8, 0., 0.]))


def test_inputs_with_units_are_converted_to_the_numbers_they_mean(kernels):
    station = dss35(False)
    location = np.array([1e-3, -2e-3, 5e-4])
    rotation = np.array([1e-8, -2e-8, 3e-8])
    in_m = scb.ArrayWUnits(location * 1e3, m)
    in_mrad = scb.ArrayWUnits(rotation * 1e3, mrad)
    np.testing.assert_allclose(
        bare(station.state_correction(EPOCH, location=in_m, rotation_vector=in_mrad)),
        bare(station.state_correction(EPOCH, location=location, rotation_vector=rotation)),
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(bare(station.rotation_partials(EPOCH, location=in_m)),
                               bare(station.rotation_partials(EPOCH, location=location)),
                               rtol=1e-13, atol=0)
    np.testing.assert_allclose(bare(station.location_partials(EPOCH, rotation_vector=in_mrad)),
                               bare(station.location_partials(EPOCH, rotation_vector=rotation)),
                               rtol=1e-13, atol=0)
    lander = mars_lander()
    pole = {"pole_ra": 0.33, "pole_dec": 0.95, "pole_pm": 350.89, "pole_w0": 3.0}
    with_units = {"pole_ra": scb.ArrayWUnits(np.degrees(0.33), deg),
                  "pole_dec": scb.ArrayWUnits(950., mrad),
                  "pole_pm": scb.ArrayWUnits(350.89 / 86400., deg / sec),
                  "pole_w0": scb.ArrayWUnits(np.degrees(3.0), deg)}
    np.testing.assert_allclose(bare(lander.iau_pole_correction(EPOCH, **with_units)),
                               bare(lander.iau_pole_correction(EPOCH, **pole)),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(bare(lander.iau_pole_partials(EPOCH, **with_units)),
                               bare(lander.iau_pole_partials(EPOCH, **pole)),
                               rtol=1e-12, atol=1e-15)


def test_a_framed_location_is_read_in_its_own_axes(kernels):
    station = dss35(False)
    location = np.array([1e-3, -2e-3, 5e-4])
    framed = scb.ArrayWFrame(scb.ArrayWUnits(location * 1e3, m), scb.Frame("ECLIPJ2000"))
    np.testing.assert_allclose(
        bare(station.state_correction(EPOCH, location=framed)),
        bare(station.state_correction(EPOCH, location=location, location_frame="ECLIPJ2000")),
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        bare(station.rotation_partials(EPOCH, location=framed)),
        bare(station.rotation_partials(EPOCH, location=location, location_frame="ECLIPJ2000")),
        rtol=1e-13, atol=0)


# NOTE: station_visibility and elevation are tested in integration tests
# endregion Tests #
