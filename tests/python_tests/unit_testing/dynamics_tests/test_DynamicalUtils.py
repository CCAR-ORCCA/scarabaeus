# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Unit Tests for the DynamicalUtils class  #
# ============================================#
from types import SimpleNamespace

import src.scarabaeus as scb
from scarabaeus import SpiceManager
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.dynamics.DynamicalUtils import DynamicalUtils
import pytest
import numpy as np

km, sec = scb.Units.get_units(["km", "sec"])

SPICE_ID = -1357
R0 = np.array([6800.0, 1200.0, 2400.0])
V0 = np.array([-1.4, 6.9, 2.1])
A_FRAME = np.array([3e-9, -2e-9, 5e-9])
EPOCH = 1000.0
MOON = np.array([3.8e5, 2.0e4, -1.5e4, -0.05, 0.98, 0.1])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def sun2body():
    """Heliocentric spacecraft position, km (roughly Earth's in January)."""
    return np.array([-2.6e7, 1.33e8, 5.77e7])


@pytest.fixture
def sun2body_vel():
    """Heliocentric spacecraft velocity, km/s."""
    return np.array([-29.8, -5.0, -2.2])


@pytest.fixture(autouse=True)
def fresh_memo(monkeypatch):
    """Each test starts with an empty nadir memo, so none depends on another."""
    monkeypatch.setattr(DynamicalUtils, "_nadir_memo", (None, None))


def central_difference(f, x, h):
    """``(3, 3, 3)`` derivative of the DCM columns: ``D[j][:, k] = d col_j / d x_k``."""
    D = np.zeros((3, 3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        D[:, :, k] = ((f(x + e) - f(x - e)) / (2 * h)).T
    return D


@pytest.fixture
def sc():
    """Body the acceleration is evaluated for; only its SPICE ID is read."""
    return SimpleNamespace(spice_id=SPICE_ID)


@pytest.fixture
def spice(monkeypatch):
    """SPICE stand-in that counts its lookups: the Moon's state and a fixed rotation."""
    calls = {"get_state": [], "get_xfrm": []}
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def get_state(target, epoch, frame, origin):
        calls["get_state"].append((target, epoch, frame, origin))
        return MOON.copy()

    def get_xfrm(source, target, epoch):
        calls["get_xfrm"].append((source, target, epoch))
        return rotation

    monkeypatch.setattr(SpiceManager, "get_state", staticmethod(get_state))
    monkeypatch.setattr(SpiceManager, "get_xfrm", staticmethod(get_xfrm))
    return SimpleNamespace(calls=calls, rotation=rotation)


def model(frame="inertial", body_fixed_frame=None, relative_body=None, memo=True):
    """The attributes DynamicalUtils reads off a frame-expressed acceleration."""
    frame, body_fixed_frame, relative_body = DynamicalUtils.resolve_frame(
        frame, body_fixed_frame, relative_body
    )
    settings = SimpleNamespace(
        frame=frame,
        body_fixed_frame=body_fixed_frame,
        relative_body=relative_body,
        ref_frame=SimpleNamespace(name="J2000"),
        origin_name="EARTH",
    )
    if memo:
        settings._frame_memo = {}
    return settings


def state_dict(v=V0):
    """Propagated state: only the velocity is read."""
    return {("velocity", SPICE_ID): np.asarray(v, dtype=float)}


def fd_jacobian(f, x, h):
    """Central-difference Jacobian of ``f`` at ``x``."""
    columns = []
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        columns.append((f(x + step) - f(x - step)) / (2.0 * h))
    return np.column_stack(columns)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestVectorAlgebra:
    """The small vector helpers agree with their NumPy definitions."""

    def test_cross_matches_numpy(self):
        """The written-out cross product is ``np.cross``."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            a, b = rng.normal(size=3), rng.normal(size=3)
            np.testing.assert_allclose(DynamicalUtils.cross(a, b), np.cross(a, b), rtol=1e-14)

    def test_skew_is_the_cross_product_matrix(self):
        """``skew(a) @ b`` is ``a x b``."""
        a, b = np.array([0.3, -1.2, 2.0]), np.array([1.1, 0.4, -0.7])
        np.testing.assert_allclose(DynamicalUtils.skew(a) @ b, np.cross(a, b), rtol=1e-14)

    def test_body_vector_jacobian_sums_the_axis_derivatives(self):
        """A body vector turns with the axes it is made of, one or many at a time."""
        rng = np.random.default_rng(4)
        D = rng.normal(size=(3, 3, 3))
        e = np.array([0.2, -0.5, 1.0])
        np.testing.assert_allclose(
            DynamicalUtils.body_vector_jacobian(e, D), sum(e[j] * D[j] for j in range(3))
        )
        stack = np.stack([e, 2 * e])
        J = DynamicalUtils.body_vector_jacobian(stack, D)
        assert J.shape == (2, 3, 3)
        np.testing.assert_allclose(J[1], 2 * J[0])


class TestModes:
    """Mode lookup, the fixed attitude and where the velocity comes from."""

    def test_unknown_mode_is_refused(self):
        with pytest.raises(DynamicalUtils.InvalidAttitudeModeError):
            DynamicalUtils.attitude_mode(SimpleNamespace(attitudeMode="spinning"))

    @pytest.mark.parametrize("mode", DynamicalUtils.ATTITUDE_MODES)
    def test_every_supported_mode_is_accepted(self, mode):
        assert DynamicalUtils.attitude_mode(SimpleNamespace(attitudeMode=mode)) == mode

    @pytest.mark.parametrize(
        "mode, expected",
        [("nadirPointingSun", True), ("inertialFixedAttitude", False), ("CK", False)],
    )
    def test_only_the_nadir_attitude_depends_on_velocity(self, mode, expected):
        assert DynamicalUtils.depends_on_velocity(SimpleNamespace(attitudeMode=mode)) is expected

    def test_fixed_dcm_is_a_float_copy(self):
        """An integer matrix comes back as float, and editing it leaves the spacecraft alone."""
        body = SimpleNamespace(fixedInertialAttitude=np.eye(3, dtype=int))
        dcm = DynamicalUtils.fixed_dcm(body)
        assert dcm.dtype == float
        dcm[0, 0] = 5.0
        assert body.fixedInertialAttitude[0, 0] == 1

    @pytest.mark.parametrize("value", [None, np.eye(2), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                             ids=["missing", "2x2", "list"])
    def test_fixed_dcm_must_be_a_3x3_array(self, value):
        with pytest.raises(DynamicalUtils.InvalidFixedAttitudeError):
            DynamicalUtils.fixed_dcm(SimpleNamespace(fixedInertialAttitude=value))

    def test_velocity_is_read_by_spacecraft_first(self):
        """The spacecraft's own entry wins over a bare one, and units are dropped."""
        body = SimpleNamespace(spice_id=-64)
        state = {
            ("velocity", -64): awu(np.array([1.0, 2.0, 3.0]), km / sec),
            "velocity": np.zeros(3),
        }
        np.testing.assert_array_equal(DynamicalUtils.velocity_from_state(state, body), [1, 2, 3])
        np.testing.assert_array_equal(
            DynamicalUtils.velocity_from_state({"velocity": np.ones(3)}, body), np.ones(3)
        )

    def test_missing_velocity_is_refused(self):
        with pytest.raises(DynamicalUtils.MissingVelocityError):
            DynamicalUtils.velocity_from_state({}, SimpleNamespace(spice_id=-64))


class TestNadirPointingSun:
    """
    The Sun-pointed attitude: -X at the Sun, Y along the heliocentric velocity.

    Its derivatives feed the analytical partials of every radiation model, so
    they are checked against finite differences of the DCM itself.
    """

    def test_axes(self, sun2body, sun2body_vel):
        """A proper rotation, with -X at the Sun and Y on the velocity side."""
        dcm, _, _ = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel)

        np.testing.assert_allclose(dcm.T @ dcm, np.eye(3), atol=1e-14)
        assert np.linalg.det(dcm) == pytest.approx(1.0)
        np.testing.assert_allclose(dcm[:, 0], sun2body / np.linalg.norm(sun2body), rtol=1e-14)
        assert dcm[:, 1] @ sun2body_vel > 0.0
        assert abs(dcm[:, 2] @ sun2body_vel) < 1e-12

    def test_no_derivatives_unless_asked(self, sun2body, sun2body_vel):
        _, D_r, D_v = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel)
        assert D_r is None and D_v is None

    def test_position_derivative_matches_finite_differences(self, sun2body, sun2body_vel):
        _, D_r, _ = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, True)
        D_fd = central_difference(
            lambda s: DynamicalUtils.nadirPointingSun(s, sun2body_vel)[0], sun2body, 1e3
        )
        np.testing.assert_allclose(D_r, D_fd, rtol=1e-6, atol=1e-6 * np.abs(D_fd).max())

    def test_velocity_derivative_matches_finite_differences(self, sun2body, sun2body_vel):
        """X ignores the velocity, so its derivative is exactly zero."""
        _, _, D_v = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, True)
        D_fd = central_difference(
            lambda v: DynamicalUtils.nadirPointingSun(sun2body, v)[0], sun2body_vel, 1e-4
        )
        np.testing.assert_allclose(D_v, D_fd, rtol=1e-6, atol=1e-6 * np.abs(D_fd).max())
        assert not D_v[0].any()

    def test_repeat_call_is_served_from_the_memo(self, sun2body, sun2body_vel):
        """The same inputs return the same arrays, locked so no caller can corrupt them."""
        first = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, True)
        second = DynamicalUtils.nadirPointingSun(sun2body.copy(), sun2body_vel.copy(), True)

        assert all(a is b for a, b in zip(first, second))
        for array in first:
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array[0] = 0.0

    def test_new_inputs_are_not_served_from_the_memo(self, sun2body, sun2body_vel):
        first, _, _ = DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, True)
        moved, _, _ = DynamicalUtils.nadirPointingSun(sun2body + 1e5, sun2body_vel, True)
        assert not np.allclose(first, moved, rtol=0.0, atol=1e-12)


class TestRTN:
    """
    The radial / along-track / cross-track triad of an orbit.

    Its derivatives feed the RTN partials of the empirical accelerations, so
    they are checked against finite differences of the DCM itself.
    """

    def test_axes(self):
        """A proper rotation, with R on the radius, N on the angular momentum."""
        dcm, _, _ = DynamicalUtils.rtn(R0, V0)
        h = np.cross(R0, V0)

        np.testing.assert_allclose(dcm.T @ dcm, np.eye(3), atol=1e-14)
        assert np.linalg.det(dcm) == pytest.approx(1.0)
        np.testing.assert_allclose(dcm[:, 0], R0 / np.linalg.norm(R0), rtol=1e-14)
        np.testing.assert_allclose(dcm[:, 2], h / np.linalg.norm(h), rtol=1e-14)
        assert dcm[:, 1] @ V0 > 0.0

    def test_no_derivatives_unless_asked(self):
        _, D_r, D_v = DynamicalUtils.rtn(R0, V0)
        assert D_r is None and D_v is None

    def test_position_derivative_matches_finite_differences(self):
        _, D_r, _ = DynamicalUtils.rtn(R0, V0, True)
        D_fd = central_difference(lambda r: DynamicalUtils.rtn(r, V0)[0], R0, 1e-2)
        np.testing.assert_allclose(D_r, D_fd, rtol=1e-6, atol=1e-6 * np.abs(D_fd).max())

    def test_velocity_derivative_matches_finite_differences(self):
        """R ignores the velocity, so its derivative is exactly zero."""
        _, _, D_v = DynamicalUtils.rtn(R0, V0, True)
        D_fd = central_difference(lambda v: DynamicalUtils.rtn(R0, v)[0], V0, 1e-5)
        np.testing.assert_allclose(D_v, D_fd, rtol=1e-6, atol=1e-6 * np.abs(D_fd).max())
        assert not D_v[0].any()

    @pytest.mark.parametrize(
        "r, v",
        [
            (np.zeros(3), np.array([1.0, 0.0, 0.0])),
            (np.array([7000.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])),
        ],
        ids=["zero-radius", "radial-velocity"],
    )
    def test_degenerate_geometry_is_refused(self, r, v):
        with pytest.raises(DynamicalUtils.DegenerateGeometryError):
            DynamicalUtils.rtn(r, v)


class TestResolveFrame:
    """Frame settings are checked once, at build."""

    def test_default_is_inertial(self):
        assert DynamicalUtils.resolve_frame() == ("inertial", None, None)

    @pytest.mark.parametrize(
        "given, expected",
        [("rtn", "RTN"), ("Inertial", "inertial"), (" BODY_FIXED ", "body_fixed")],
    )
    def test_frame_is_matched_without_regard_to_case(self, given, expected):
        frame, _, _ = DynamicalUtils.resolve_frame(given, "IAU_EARTH")
        assert frame == expected

    def test_unknown_frame_is_refused(self):
        with pytest.raises(DynamicalUtils.InvalidFrameError, match="lvlh"):
            DynamicalUtils.resolve_frame("lvlh")

    def test_body_fixed_needs_a_frame_name(self):
        with pytest.raises(DynamicalUtils.MissingBodyFixedFrameError):
            DynamicalUtils.resolve_frame("body_fixed")

    def test_frame_and_body_objects_are_reduced_to_names(self):
        """A Frame or Body is stored by name, as SPICE wants it."""
        resolved = DynamicalUtils.resolve_frame(
            "body_fixed",
            SimpleNamespace(name="IAU_EARTH"),
            SimpleNamespace(name="MOON"),
        )
        assert resolved == ("body_fixed", "IAU_EARTH", "MOON")


class TestFrameMatrix:
    """The rotation from the component frame into the propagation frame."""

    def test_inertial_components_are_left_alone(self, sc):
        """No rotation is built, and the values come back untouched."""
        inertial = model()
        assert DynamicalUtils.frame_matrix(inertial, R0, state_dict(), EPOCH, sc) is None
        assert (
            DynamicalUtils.from_frame(inertial, A_FRAME, R0, state_dict(), EPOCH, sc)
            is A_FRAME
        )

    def test_rtn_triad_is_the_orbit_triad(self, sc):
        """A proper rotation with R along the radius and N along the angular momentum."""
        T = DynamicalUtils.frame_matrix(model("RTN"), R0, state_dict(), EPOCH, sc)
        h = np.cross(R0, V0)

        np.testing.assert_allclose(T.T @ T, np.eye(3), atol=1e-14)
        assert np.linalg.det(T) == pytest.approx(1.0)
        np.testing.assert_allclose(T[:, 0], R0 / np.linalg.norm(R0), rtol=1e-14)
        np.testing.assert_allclose(T[:, 2], h / np.linalg.norm(h), rtol=1e-14)

    def test_body_fixed_rotation_comes_from_spice(self, sc, spice):
        bf = model("body_fixed", "IAU_EARTH")
        a = DynamicalUtils.from_frame(bf, A_FRAME, R0, state_dict(), EPOCH, sc)

        np.testing.assert_allclose(a, spice.rotation @ A_FRAME, rtol=1e-14)
        assert spice.calls["get_xfrm"] == [("IAU_EARTH", "J2000", EPOCH)]

    def test_body_fixed_rotation_needs_an_epoch(self, sc, spice):
        with pytest.raises(DynamicalUtils.MissingEpochError, match="IAU_EARTH"):
            DynamicalUtils.frame_matrix(
                model("body_fixed", "IAU_EARTH"), R0, state_dict(), None, sc
            )

    def test_relative_body_moves_the_triad(self, sc, spice):
        """The triad is the orbit about the Moon: its state is subtracted first."""
        T = DynamicalUtils.frame_matrix(
            model("RTN", relative_body="MOON"), R0, state_dict(), EPOCH, sc
        )
        expected, _, _ = DynamicalUtils.rtn(R0 - MOON[:3], V0 - MOON[3:])

        np.testing.assert_allclose(T, expected, rtol=1e-14)
        assert spice.calls["get_state"] == [("MOON", EPOCH, "J2000", "EARTH")]

    def test_relative_body_needs_an_epoch(self, sc, spice):
        with pytest.raises(DynamicalUtils.MissingEpochError, match="MOON"):
            DynamicalUtils.frame_matrix(
                model("RTN", relative_body="MOON"), R0, state_dict(), None, sc
            )

    def test_missing_velocity_is_refused(self, sc):
        with pytest.raises(DynamicalUtils.MissingVelocityError):
            DynamicalUtils.frame_matrix(model("RTN"), R0, {}, EPOCH, sc)


class TestFrameMemo:
    """Each SPICE lookup happens once per epoch when the model carries a memo."""

    def test_same_epoch_is_served_from_the_memo(self, sc, spice):
        bf = model("body_fixed", "IAU_EARTH")
        rel = model("RTN", relative_body="MOON")
        for _ in range(3):
            DynamicalUtils.frame_matrix(bf, R0, state_dict(), EPOCH, sc)
            DynamicalUtils.frame_matrix(rel, R0, state_dict(), EPOCH, sc)

        assert len(spice.calls["get_xfrm"]) == 1
        assert len(spice.calls["get_state"]) == 1

    def test_new_epoch_is_looked_up_again(self, sc, spice):
        bf = model("body_fixed", "IAU_EARTH")
        DynamicalUtils.frame_matrix(bf, R0, state_dict(), EPOCH, sc)
        DynamicalUtils.frame_matrix(bf, R0, state_dict(), EPOCH + 60.0, sc)

        assert [call[2] for call in spice.calls["get_xfrm"]] == [EPOCH, EPOCH + 60.0]
        assert bf._frame_memo["rotation"][0] == EPOCH + 60.0

    def test_model_without_a_memo_looks_up_every_call(self, sc, spice):
        bf = model("body_fixed", "IAU_EARTH", memo=False)
        DynamicalUtils.frame_matrix(bf, R0, state_dict(), EPOCH, sc)
        DynamicalUtils.frame_matrix(bf, R0, state_dict(), EPOCH, sc)

        assert len(spice.calls["get_xfrm"]) == 2


class TestFramePartials:
    """Only the RTN triad depends on the state, so only it adds partials."""

    @pytest.mark.parametrize("frame", ["inertial", "body_fixed"])
    def test_epoch_only_frames_have_no_state_partials(self, frame, sc, spice):
        da_dr, da_dv = DynamicalUtils.frame_partials(
            model(frame, "IAU_EARTH"), A_FRAME, R0, state_dict(), EPOCH, sc
        )
        assert not da_dr.any() and not da_dv.any()

    @pytest.mark.parametrize("relative_body", [None, "MOON"])
    def test_rtn_partials_match_finite_differences(self, relative_body, sc, spice):
        """The ephemeris offset of a relative body drops out of the derivative."""
        rtn = model("RTN", relative_body=relative_body)
        da_dr, da_dv = DynamicalUtils.frame_partials(
            rtn, A_FRAME, R0, state_dict(), EPOCH, sc
        )

        def a_of_r(r):
            return DynamicalUtils.from_frame(rtn, A_FRAME, r, state_dict(), EPOCH, sc)

        def a_of_v(v):
            return DynamicalUtils.from_frame(rtn, A_FRAME, R0, state_dict(v), EPOCH, sc)

        fd_r = fd_jacobian(a_of_r, R0, 1e-2)
        fd_v = fd_jacobian(a_of_v, V0, 1e-5)
        np.testing.assert_allclose(da_dr, fd_r, rtol=1e-6, atol=1e-6 * np.abs(fd_r).max())
        np.testing.assert_allclose(da_dv, fd_v, rtol=1e-6, atol=1e-6 * np.abs(fd_v).max())


# endregion Tests #
# --------------#
