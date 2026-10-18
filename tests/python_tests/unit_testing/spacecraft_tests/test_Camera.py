# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ===================================#
#  Unit Tests for the Camera class  #
# ===================================#

import src.scarabaeus as scb
import numpy as np
import pytest

km, deg, rad, kg = scb.Units.get_units(["km", "deg", "rad", "kg"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def fov_angular():
    return scb.ArrayWUnits(np.array([np.deg2rad(2.5), np.deg2rad(2.5)]), rad)


@pytest.fixture
def focal_length():
    return scb.ArrayWUnits(150 * 10**-6, km)


@pytest.fixture
def ifov_angle():
    return scb.ArrayWUnits(np.array([25e-6, 25e-6]), rad)


@pytest.fixture
def camera(fov_angular, focal_length, ifov_angle):
    return scb.Camera(
        "Unit Test Camera",
        -12345,
        J2000,
        fov_angular,
        (2000, 2000),
        focal_length,
        ifov_angle,
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(camera, fov_angular, focal_length, ifov_angle):
    """
    Verifies correct attribute assignment on construction and all
    read-only properties: associated_body_spice_id, fov_angular, ifov,
    focal_length, and max_sample_line.
    """
    assert camera.name == "Unit Test Camera"
    assert camera._spice_id == -12345
    assert camera._fov_pixels == (2000, 2000)
    assert camera._ifov_angle is ifov_angle

    # associated_body_spice_id
    assert camera.associated_body_spice_id == -12345

    # camera_frame
    assert camera.camera_frame is J2000

    # fov_angular
    assert camera.fov_angular is fov_angular

    # focal_length
    assert camera.focal_length is focal_length

    # ifov — computed from focal_length and ifov_angle
    assert isinstance(camera.ifov, scb.ArrayWUnits)
    assert camera.ifov.values.shape == (2,)

    # ifov setter: None derives from fov_angular/fov_pixels; a given value is stored as-is
    camera.ifov = None
    assert camera.ifov.values == pytest.approx((fov_angular[0] / camera._fov_pixels[0]).values)
    custom_ifov = scb.ArrayWUnits(np.array([1.0, 2.0]), rad)
    camera.ifov = custom_ifov
    assert camera.ifov is custom_ifov

    # max_sample_line — fov_pixels / 2 + zero pp_offset = [1000, 1000]
    assert isinstance(camera.max_sample_line, scb.ArrayWUnits)
    assert camera.max_sample_line.values.shape == (2,)
    assert np.allclose(camera.max_sample_line.values, [1000.0, 1000.0])

    # calibration_matrix — 2x3 pinhole intrinsics matrix
    assert isinstance(camera.calibration_matrix, np.ndarray)
    assert camera.calibration_matrix.shape == (2, 3)


@pytest.mark.parametrize(
    "rotation_matrix, expected",
    [
        (None, np.eye(3)),
        (
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
    ],
    ids=["Default Identity", "Custom Rotation"],
)
def test_attitude_matrix(camera, rotation_matrix, expected):
    """
    Verifies that attitude_matrix returns identity when unset and stores
    any 3x3 matrix assigned via the setter.
    """
    if rotation_matrix is not None:
        camera.attitude_matrix = rotation_matrix
    assert np.allclose(camera.attitude_matrix, expected)


class TestPpOffset:
    """Groups pp_offset default, valid-input, and error cases."""

    def test_default_is_zero(self, camera):
        """pp_offset defaults to [0, 0] km when None is passed at construction."""
        assert isinstance(camera.pp_offset, scb.ArrayWUnits)
        assert np.allclose(camera.pp_offset.values, [0.0, 0.0])

    def test_valid_km_input(self, fov_angular, focal_length, ifov_angle):
        """A valid km ArrayWUnits is accepted and stored unchanged."""
        offset = scb.ArrayWUnits(np.array([0.001, 0.002]), km)
        cam = scb.Camera(
            "Cam", -1, J2000, fov_angular, (2000, 2000),
            focal_length, ifov_angle, pp_offset=offset,
        )
        assert np.allclose(cam.pp_offset.values, [0.001, 0.002])

    def test_wrong_units_raises(self, camera):
        """Setting pp_offset in non-km units raises ValueError."""
        with pytest.raises(ValueError):
            camera.pp_offset = scb.ArrayWUnits(np.array([1.0, 1.0]), rad)

    def test_non_homogeneous_units_raises(self, camera):
        """Setting pp_offset to a non-homogeneous ArrayWUnits (mixed units
        across elements) raises ValueError with a distinct message from the
        homogeneous-but-wrong-units case."""
        with pytest.raises(ValueError, match="matching units"):
            camera.pp_offset = scb.ArrayWUnits([1.0, 2.0], [km, kg])

    def test_wrong_type_raises(self, camera):
        """Setting pp_offset to a non-ArrayWUnits type raises TypeError."""
        with pytest.raises(TypeError):
            camera.pp_offset = [0.0, 0.0]


# NOTE: target_in_fov and other SPICE-dependent methods are tested in integration tests