# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Unit Tests for the FirstOrderGaussMarkov class  #
# ================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

SPICE_ID = -1357
R0 = np.array([6800.0, 1200.0, 2400.0])
V0 = np.array([-1.4, 6.9, 2.1])
A0 = np.array([3e-9, -2e-9, 5e-9])
BETA = np.array([1 / 3600.0, 1 / 1800.0, 1 / 900.0])
EPOCH = 1000.0

# --------------------#
# region    Fixtures  #
# --------------------#


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc():
    """Spacecraft the empirical acceleration is attached to."""
    return scb.Spacecraft(
        "FOGM_SC", SPICE_ID, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )


def framed(origin, sc, frame, **kwargs):
    """Model with its components in ``frame``, body-fixed to the Earth."""
    if frame == "body_fixed":
        kwargs.setdefault("body_fixed_frame", "IAU_EARTH")
    return scb.FirstOrderGaussMarkov(
        origin, sc, J2000, default_beta=BETA, frame=frame, **kwargs
    )


def state_dict(a=A0, v=V0):
    """Propagated state seen by the model: the velocity and the GM state."""
    return {
        ("velocity", SPICE_ID): np.asarray(v, dtype=float),
        ("a_fogm", SPICE_ID): np.asarray(a, dtype=float),
    }


def rtn_basis(r, v):
    """Radial, along-track and cross-track unit vectors, as columns."""
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    return np.column_stack((r_hat, np.cross(n_hat, r_hat), n_hat))


def fd_jacobian(f, x, h):
    """Central-difference Jacobian of ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    columns = []
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        columns.append((f(x + step) - f(x - step)) / (2.0 * h))
    return np.column_stack(columns)


def relative_error(analytic, numeric):
    """Largest entry-wise difference, scaled by the size of the Jacobian."""
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)
    return np.abs(analytic - numeric).max() / np.abs(numeric).max()


# --------------------#
# endregion Fixtures  #
# --------------------#

# --------------------#
# region       Tests  #
# --------------------#


class TestFrames:
    def test_defaults_to_inertial(self, origin, sc):
        """Without a frame the GM state is the applied acceleration itself."""
        fogm = scb.FirstOrderGaussMarkov(origin, sc, J2000)

        assert fogm.frame == "inertial"
        assert fogm.body_fixed_frame is None
        assert fogm.relative_body is None
        np.testing.assert_array_equal(
            fogm.compute_acceleration(R0, state_dict(), EPOCH, sc), A0
        )

    def test_unknown_frame_is_reported(self, origin, sc):
        """The RTN frame is spelled 'RTN'; the retired 'lvlh' name is refused."""
        with pytest.raises(scb.DynamicalUtils.InvalidFrameError):
            framed(origin, sc, "lvlh")

    def test_body_fixed_frame_needs_a_frame_name(self, origin, sc):
        """A body-fixed model must say which body-fixed frame it lives in."""
        with pytest.raises(scb.DynamicalUtils.MissingBodyFixedFrameError):
            framed(origin, sc, "body_fixed", body_fixed_frame=None)

    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_depends_on_velocity(self, origin, sc, frame, expected):
        """The RTN triad is built from the velocity; the other frames are not."""
        fogm = framed(origin, sc, frame)

        assert fogm.frame == frame
        assert fogm.depends_on_velocity is expected

    def test_rtn_turns_the_state_onto_rtn(self, origin, sc):
        """The GM components are laid along R, S and W."""
        accel = framed(origin, sc, "RTN").compute_acceleration(
            R0, state_dict(), EPOCH, sc
        )

        assert relative_error(accel, rtn_basis(R0, V0) @ A0) < 1e-14

    def test_rtn_about_another_body(self, origin, sc):
        """With a relative body the triad is built about that body."""
        accel = framed(origin, sc, "RTN", relative_body="MOON").compute_acceleration(
            R0, state_dict(), EPOCH, sc
        )

        moon = scb.SpiceManager.get_state("MOON", EPOCH, J2000, origin.name)
        offset = np.asarray(moon.values, dtype=float).ravel()
        expected = rtn_basis(R0 - offset[:3], V0 - offset[3:6]) @ A0
        assert relative_error(accel, expected) < 1e-12

    def test_body_fixed_turns_with_the_body(self, origin, sc):
        """A body-fixed GM state is rotated out of the body frame at the epoch."""
        accel = framed(origin, sc, "body_fixed").compute_acceleration(
            R0, state_dict(), EPOCH, sc
        )

        rotation = np.asarray(
            scb.SpiceManager.get_xfrm("IAU_EARTH", "J2000", EPOCH), dtype=float
        )
        assert relative_error(accel, rotation @ A0) < 1e-14

    def test_rtn_needs_a_velocity(self, origin, sc):
        """Without a velocity there is no RTN triad to rotate by."""
        fogm = framed(origin, sc, "RTN")

        with pytest.raises(scb.DynamicalUtils.MissingVelocityError):
            fogm.compute_acceleration(R0, {("a_fogm", SPICE_ID): A0}, EPOCH, sc)

    @pytest.mark.parametrize("relative_body", [None, "MOON"])
    def test_rtn_state_partials_match_finite_differences(
        self, origin, sc, relative_body
    ):
        """da/dr and da/dv carry the derivatives of the RTN triad."""
        fogm = framed(origin, sc, "RTN", relative_body=relative_body)

        by_position = fogm._compute_partial_by_position(R0, state_dict(), EPOCH, sc)
        by_velocity = fogm._compute_partial_by_velocity(R0, state_dict(), EPOCH, sc)
        # 1 km is small against the 7000 km orbit that sets the triad.
        numeric_r = fd_jacobian(
            lambda r: fogm.compute_acceleration(r, state_dict(), EPOCH, sc), R0, 1.0
        )
        numeric_v = fd_jacobian(
            lambda v: fogm.compute_acceleration(R0, state_dict(v=v), EPOCH, sc),
            V0,
            1e-6,
        )

        assert relative_error(by_position, numeric_r) < 1e-6
        assert relative_error(by_velocity, numeric_v) < 1e-6

    @pytest.mark.parametrize("frame", ["inertial", "body_fixed"])
    def test_other_frames_have_no_state_partials(self, origin, sc, frame):
        """The inertial and body-fixed rotations do not depend on the state."""
        fogm = framed(origin, sc, frame)

        assert np.all(fogm._compute_partial_by_position(R0, state_dict(), EPOCH, sc) == 0)
        assert np.all(fogm._compute_partial_by_velocity(R0, state_dict(), EPOCH, sc) == 0)

    @pytest.mark.parametrize("frame", ["inertial", "RTN", "body_fixed"])
    def test_state_partial_is_the_rotation(self, origin, sc, frame):
        """da/da_fogm is the frame rotation, the identity in the inertial frame."""
        fogm = framed(origin, sc, frame)

        analytic = fogm._compute_partial_by_acceleration_state(
            R0, state_dict(), EPOCH, sc
        )
        numeric = fd_jacobian(
            lambda a: fogm.compute_acceleration(R0, state_dict(a=a), EPOCH, sc),
            A0,
            1e-10,
        )

        assert relative_error(analytic, numeric) < 1e-8
        np.testing.assert_allclose(analytic.T @ analytic, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    def test_drift_acts_on_the_frame_components(self, origin, sc, frame):
        """The frame rotates what is applied, not how the components decay."""
        rotated = framed(origin, sc, frame)

        np.testing.assert_array_equal(
            rotated._compute_acceleration_state_derivative(R0, state_dict(), EPOCH, sc),
            -BETA * A0,
        )


# --------------------#
# endregion Tests     #
# --------------------#

# ------------------------------------------------------------------#
# region  ForceModelTranslation and propagation                      #
# ------------------------------------------------------------------#
R_PROP = np.array([7000.0, 0.0, 1000.0])
V_PROP = np.array([0.0, 7.3, 0.5])
PROP_END = 3000.0


def fogm_force_model(sc, **kwargs):
    """ForceModelTranslation configured with a Gauss-Markov acceleration."""
    return scb.ForceModelTranslation(
        primary_body=sc, first_order_gauss_markov=True, fogm_beta=BETA, **kwargs
    )


def fogm_state(origin, sc, a=A0, r=R_PROP, v=V_PROP):
    """State carrying the GM acceleration as a dynamic parameter."""
    return scb.StateArray(
        epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
        origin=origin,
        state=(
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.asarray(r, dtype=float), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.asarray(v, dtype=float), km / sec, J2000))
            .param(
                "a_fogm",
                sc,
                scb.ArrayWFrame(np.asarray(a, dtype=float), km / sec**2, J2000),
                dynamics="dynamic",
            )
        ),
    )


def propagate(origin, sc, force_model, x0):
    """Propagate one arc from ``[r, v, a_fogm]`` and return the final state and STM."""
    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=fogm_state(origin, sc, a=x0[6:9], r=x0[:3], v=x0[3:6]),
        tspan=scb.EpochArray(np.linspace(0.0, PROP_END, 4), sys="TDB", rep="NUM"),
        force_models=force_model,
        display_progress=False,
    )
    propagator.propagate()

    propagated = propagator.propagated_state_array
    final = np.concatenate(
        [np.asarray(component[5].quantity.values)[-1] for component in propagated.state]
    )
    stm = np.asarray(propagator.STM.quantity.values)
    size = int(round(stm.shape[-1] ** 0.5))
    return final, stm[-1].reshape(size, size)


class TestForceModel:
    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_asks_for_the_velocity_partial(
        self, origin, sc, frame, expected
    ):
        """The RTN triad needs the da/dv block; the other frames do not."""
        kwargs = {"fogm_body_fixed_frame": "IAU_EARTH"} if frame == "body_fixed" else {}
        force_model = fogm_force_model(sc, fogm_frame=frame, **kwargs)

        assert force_model._velocity_partial_active is expected
        force_model.initialize_force_models(fogm_state(origin, sc), J2000, 0.0)
        assert force_model._velocity_partial_active is expected

    @pytest.mark.parametrize(
        "frame, label",
        [
            ("inertial", "FirstOrderGaussMarkov"),
            ("RTN", "FirstOrderGaussMarkov(frame=RTN)"),
        ],
    )
    def test_frame_reaches_the_model(self, origin, sc, frame, label):
        """The frame is handed to the built model and named in the force list."""
        force_model = fogm_force_model(sc, fogm_frame=frame)
        force_model.initialize_force_models(fogm_state(origin, sc), J2000, 0.0)

        assert force_model.fogm_frame == frame
        assert force_model.force_models["FirstOrderGaussMarkov"].frame == frame
        assert label in force_model.list_forces()

    def test_frame_serialization_round_trip(self, origin, sc):
        """The frame, body-fixed frame and relative body survive to_dict / from_dict."""
        force_model = fogm_force_model(
            sc,
            fogm_frame="body_fixed",
            fogm_body_fixed_frame="IAU_EARTH",
            fogm_relative_body="MOON",
        )

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc)
        restored.initialize_force_models(fogm_state(origin, sc), J2000, 0.0)

        rebuilt = restored.force_models["FirstOrderGaussMarkov"]
        assert restored.fogm_frame == "body_fixed"
        assert rebuilt.frame == "body_fixed"
        assert rebuilt.body_fixed_frame == "IAU_EARTH"
        assert rebuilt.relative_body == "MOON"

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    def test_stm_matches_re_propagation(self, origin, sc, frame):
        """The rotated GM partials and, in RTN, the triad's da/dr and da/dv reach the STM."""
        kwargs = {"fogm_body_fixed_frame": "IAU_EARTH"} if frame == "body_fixed" else {}
        # Large enough that leaving out the triad's partials would fail the
        # comparison.
        x0 = np.concatenate((R_PROP, V_PROP, [1e-5, -5e-6, 7e-6]))
        _, stm = propagate(origin, sc, fogm_force_model(sc, fogm_frame=frame, **kwargs), x0)

        steps = [1e-3] * 3 + [1e-6] * 3 + [1e-9] * 3
        numeric = np.zeros((9, 9))
        for j, step in enumerate(steps):
            ends = []
            for sign in (1.0, -1.0):
                x = x0.copy()
                x[j] += sign * step
                end, _ = propagate(
                    origin, sc, fogm_force_model(sc, fogm_frame=frame, **kwargs), x
                )
                ends.append(end)
            numeric[:, j] = (ends[0] - ends[1]) / (2.0 * step)

        assert relative_error(stm[:6, :], numeric[:6]) < 1e-7
        assert relative_error(stm[6:, 6:], numeric[6:, 6:]) < 1e-7


# endregion                                                          #
# ------------------------------------------------------------------#
