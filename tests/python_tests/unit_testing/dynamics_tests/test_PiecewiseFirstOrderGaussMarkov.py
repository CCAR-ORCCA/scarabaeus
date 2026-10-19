# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================================#
#  Unit Tests for the PiecewiseFirstOrderGaussMarkov class  #
# ==========================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

BATCH_LENGTH = 1800.0
N_BATCHES = 3
T0 = 100.0

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc():
    """Spacecraft the empirical acceleration is attached to."""
    return scb.Spacecraft(
        "ORCCA", -1234, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )


@pytest.fixture
def model(origin, sc):
    """Model anchored at a plain float epoch."""
    return scb.PiecewiseFirstOrderGaussMarkov(
        origin,
        sc,
        J2000,
        t0=T0,
        batch_length=BATCH_LENGTH,
        n_batches=N_BATCHES,
    )


@pytest.fixture
def state(origin, sc):
    """State carrying one stacked acceleration per batch."""
    return scb.StateArray(
        epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
        origin=origin,
        state=(
            scb.StateDefinition()
            .position(sc, scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000))
            .velocity(sc, scb.ArrayWFrame(np.array([0.0, 7.5, 0.0]), km / sec, J2000))
            .param(
                "a_pfogm",
                sc,
                scb.ArrayWFrame(np.zeros(3 * N_BATCHES), km / sec**2, J2000),
                dynamics="dynamic",
            )
        ),
    )


def piecewise_force_model(sc, t0):
    """ForceModelTranslation configured with a piecewise Gauss-Markov model."""
    return scb.ForceModelTranslation(
        primary_body=sc,
        piecewise_first_order_gauss_markov=True,
        pfogm_batch_length=BATCH_LENGTH,
        pfogm_n_batches=N_BATCHES,
        pfogm_beta=np.array([1 / 3600.0] * 3),
        t0=t0,
    )


def leg(sc, state, force_model, t_start, t_end):
    """Propagator over one leg, standing in for one leg of a MissionSequence."""
    return scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=scb.EpochArray(np.linspace(t_start, t_end, 20), sys="TDB", rep="NUM"),
        force_models=force_model,
        display_progress=False,
    )


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
def test_initialization(model, origin, sc):
    """Verifies correct attribute assignment on construction."""
    assert model.origin is origin
    assert model.t0 == T0
    assert model.batch_length == BATCH_LENGTH
    assert model.n_batches == N_BATCHES
    assert np.allclose(model.default_beta, np.ones(3))


class TestT0Handling:
    def test_epoch_array_is_read_as_ephemeris_time(self, origin, sc):
        """An EpochArray t0 anchors the grid at its raw ephemeris time."""
        epoch = scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")

        model = scb.PiecewiseFirstOrderGaussMarkov(
            origin, sc, J2000, t0=epoch, batch_length=BATCH_LENGTH, n_batches=N_BATCHES
        )

        # .et is the TDB second, independent of the array's display system
        assert model.t0 == pytest.approx(float(np.asarray(epoch.et).ravel()[0]))

    def test_float_is_accepted(self, model):
        """Raw ephemeris seconds are accepted; Propagator binds force models with them."""
        assert isinstance(model.t0, float)
        assert model.t0 == T0

    def test_invalid_type_is_reported(self, origin, sc):
        """Anything but an EpochArray or a real number is refused."""
        with pytest.raises(
            scb.PiecewiseFirstOrderGaussMarkov.InvalidEpochError, match="EpochArray"
        ):
            scb.PiecewiseFirstOrderGaussMarkov(
                origin,
                sc,
                J2000,
                t0="2000 JAN 01",
                batch_length=BATCH_LENGTH,
                n_batches=N_BATCHES,
            )


class TestBatchIndex:
    @pytest.mark.parametrize(
        "offset, expected",
        [
            (0.0, 0),
            (0.5 * BATCH_LENGTH, 0),
            (BATCH_LENGTH, 1),
            (2 * BATCH_LENGTH, 2),
            (N_BATCHES * BATCH_LENGTH - 1e-6, N_BATCHES - 1),
        ],
        ids=["start", "inside", "boundary-1", "boundary-2", "just-inside-end"],
    )
    def test_inside_the_grid(self, model, offset, expected):
        """Interior boundaries belong to the batch that starts there."""
        assert model._get_active_batch_index(T0 + offset) == expected

    def test_at_the_closing_endpoint(self, model):
        """The grid's final epoch belongs to the last batch, not off the grid."""
        t_end = T0 + N_BATCHES * BATCH_LENGTH

        assert model._get_active_batch_index(t_end) == N_BATCHES - 1

    @pytest.mark.parametrize(
        "epoch",
        [
            T0 - 1e-3,
            T0 + N_BATCHES * BATCH_LENGTH + 1e-3,
            T0 + (N_BATCHES + 1) * BATCH_LENGTH,
        ],
        ids=["before", "just-past-end", "a-batch-past"],
    )
    def test_outside_the_grid_is_reported(self, model, epoch):
        """Epochs genuinely off the grid raise; the endpoint tolerance covers rounding only."""
        with pytest.raises(ValueError, match="outside the batch window"):
            model._get_active_batch_index(epoch)


class TestLegBinding:
    def test_configured_epoch_survives_binding(self, sc, state):
        """Binding to a leg must not move the batch grid, or batch k drifts per leg."""
        force_model = piecewise_force_model(sc, t0=0.0)

        legs = [
            leg(sc, state, force_model, start, start + BATCH_LENGTH)
            for start in (0.0, BATCH_LENGTH, 2 * BATCH_LENGTH)
        ]
        origins = [
            prop._force_models[0].force_models["PiecewiseFirstOrderGaussMarkov"].t0
            for prop in legs
        ]

        # One grid origin across every leg ...
        assert origins == [0.0, 0.0, 0.0]

        # ... so the batches tile the arc instead of restarting at each leg
        last = legs[-1]._force_models[0].force_models["PiecewiseFirstOrderGaussMarkov"]
        assert last._get_active_batch_index(0.5 * BATCH_LENGTH) == 0
        assert last._get_active_batch_index(1.5 * BATCH_LENGTH) == 1
        assert last._get_active_batch_index(2.5 * BATCH_LENGTH) == 2

        # The leg's own epoch still governs the models bound per leg
        assert legs[1]._force_models[0].t0 == BATCH_LENGTH

    def test_missing_epoch_falls_back_to_leg(self, sc, state):
        """A piecewise model with no configured t0 falls back to the leg it is bound to."""
        force_model = piecewise_force_model(sc, t0=None)

        prop = leg(sc, state, force_model, 600.0, 2400.0)
        model = prop._force_models[0].force_models["PiecewiseFirstOrderGaussMarkov"]

        assert model.t0 == 600.0


# endregion Tests     #
# --------------------#

# ------------------------------------------------------------------#
# region  Arc pinning                                                #
# ------------------------------------------------------------------#
# Batch membership is right-continuous, so an arc ending on a boundary closes on
# an epoch owned by the next batch. The integrator then meets a jump at the very
# point it has to land on and shrinks its step forever; at ephemeris-second
# magnitudes the step runs out of float resolution and the run dies.


class TestArcPinning:
    def test_holds_one_batch_for_the_whole_arc(self, model):
        """A pinned arc reports its own batch at every epoch, including its endpoint."""
        arc_start = T0 + BATCH_LENGTH
        arc_end = arc_start + BATCH_LENGTH

        assert model.pin_batch_window(arc_start, arc_end) == 1
        assert model._get_active_batch_index(arc_start) == 1
        assert model._get_active_batch_index(0.5 * (arc_start + arc_end)) == 1

        # the closing epoch would be batch 2 without the pin
        assert model._get_active_batch_index(arc_end) == 1

        model.clear_batch_window()
        assert model._get_active_batch_index(arc_end) == 2

    def test_is_refused_for_an_arc_spanning_batches(self, model):
        """An arc wider than one batch is left to the per-epoch lookup."""
        assert model.pin_batch_window(T0, T0 + 2 * BATCH_LENGTH) is None

        assert model._get_active_batch_index(T0) == 0
        assert model._get_active_batch_index(T0 + BATCH_LENGTH) == 1

    def test_leg_ending_on_a_batch_boundary_integrates(self, sc, state):
        """A leg whose end lands on a batch boundary integrates at mission epochs."""
        # Epoch magnitude matters: the float spacing near 7.5e8 s is ~1e-7, which is
        # where an unbounded step reduction runs out of room.
        arc_t0 = 7.5e8
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            piecewise_first_order_gauss_markov=True,
            pfogm_batch_length=BATCH_LENGTH,
            pfogm_n_batches=N_BATCHES,
            pfogm_beta=np.array([1 / 3600.0] * 3),
            t0=arc_t0,
        )
        state_at_t0 = scb.StateArray(
            epoch=scb.EpochArray(np.array([arc_t0]), sys="TDB", rep="NUM"),
            origin=state.origin,
            state=scb.StateDefinition.from_components(list(state.state)),
        )

        prop = scb.Propagator(
            primary_body=sc,
            state_vector=state_at_t0,
            tspan=scb.EpochArray(
                np.linspace(arc_t0, arc_t0 + BATCH_LENGTH, 20), sys="TDB", rep="NUM"
            ),
            force_models=force_model,
            display_progress=False,
            propagate_STM=True,
        )

        prop.propagate()

        assert prop.propagated_state_array is not None


# endregion                                                          #
# ------------------------------------------------------------------#

# ------------------------------------------------------------------#
# region  Frames                                                     #
# ------------------------------------------------------------------#
# Each batch's three components may be read in the RTN (RTN) frame or a
# body-fixed frame; the Gauss-Markov drift acts on those components unchanged.
SPICE_ID = -1234
R0 = np.array([6800.0, 1200.0, 2400.0])
V0 = np.array([-1.4, 6.9, 2.1])
# Inside batch 1.
EPOCH = T0 + 1.3 * BATCH_LENGTH
BETA = np.array([1 / 3600.0, 1 / 1800.0, 1 / 900.0])


def framed(origin, sc, frame, **kwargs):
    """Model with its components in ``frame``, body-fixed to the Earth."""
    if frame == "body_fixed":
        kwargs.setdefault("body_fixed_frame", "IAU_EARTH")
    return scb.PiecewiseFirstOrderGaussMarkov(
        origin,
        sc,
        J2000,
        t0=T0,
        batch_length=BATCH_LENGTH,
        n_batches=N_BATCHES,
        default_beta=BETA,
        frame=frame,
        **kwargs,
    )


def stacked(seed=5, scale=1e-9):
    """One random acceleration per batch, stacked."""
    return scale * np.random.default_rng(seed).standard_normal(3 * N_BATCHES)


def state_dict(a=None, v=V0):
    """Propagated state seen by the model: the velocity and the stacked state."""
    return {
        ("velocity", SPICE_ID): np.asarray(v, dtype=float),
        ("a_pfogm", SPICE_ID): stacked() if a is None else a,
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


class TestFrames:
    def test_unknown_frame_is_reported(self, origin, sc):
        """The RTN frame is spelled 'RTN'; the retired 'lvlh' name is refused."""
        with pytest.raises(scb.DynamicalUtils.InvalidFrameError):
            framed(origin, sc, "lvlh")

    def test_body_fixed_frame_needs_a_frame_name(self, origin, sc):
        """A body-fixed model must say which body-fixed frame it lives in."""
        with pytest.raises(
            scb.DynamicalUtils.MissingBodyFixedFrameError
        ):
            framed(origin, sc, "body_fixed", body_fixed_frame=None)

    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_depends_on_velocity(self, origin, sc, frame, expected):
        """The RTN triad is built from the velocity; the other frames are not."""
        pfogm = framed(origin, sc, frame)

        assert pfogm.frame == frame
        assert pfogm.depends_on_velocity is expected

    def test_rtn_turns_the_active_batch_onto_rtn(self, origin, sc):
        """The active batch's components are laid along R, S and W."""
        a = stacked()

        accel = framed(origin, sc, "RTN").compute_acceleration(
            R0, state_dict(a), EPOCH, sc
        )

        assert relative_error(accel, rtn_basis(R0, V0) @ a[3:6]) < 1e-14

    def test_body_fixed_turns_with_the_body(self, origin, sc):
        """A body-fixed batch is rotated out of the body frame at the epoch."""
        a = stacked()

        accel = framed(origin, sc, "body_fixed").compute_acceleration(
            R0, state_dict(a), EPOCH, sc
        )

        rotation = np.asarray(
            scb.SpiceManager.get_xfrm("IAU_EARTH", "J2000", EPOCH), dtype=float
        )
        assert relative_error(accel, rotation @ a[3:6]) < 1e-14

    def test_rtn_needs_a_velocity(self, origin, sc):
        """Without a velocity there is no RTN triad to rotate by."""
        pfogm = framed(origin, sc, "RTN")

        with pytest.raises(scb.DynamicalUtils.MissingVelocityError):
            pfogm.compute_acceleration(
                R0, {("a_pfogm", SPICE_ID): stacked()}, EPOCH, sc
            )

    @pytest.mark.parametrize("relative_body", [None, "MOON"])
    def test_rtn_state_partials_match_finite_differences(
        self, origin, sc, relative_body
    ):
        """da/dr and da/dv carry the derivatives of the RTN triad."""
        pfogm = framed(origin, sc, "RTN", relative_body=relative_body)
        a = stacked()

        by_position = pfogm._compute_partial_by_position(R0, state_dict(a), EPOCH, sc)
        by_velocity = pfogm._compute_partial_by_velocity(R0, state_dict(a), EPOCH, sc)
        numeric_r = fd_jacobian(
            lambda r: pfogm.compute_acceleration(r, state_dict(a), EPOCH, sc), R0, 1.0
        )
        numeric_v = fd_jacobian(
            lambda v: pfogm.compute_acceleration(R0, state_dict(a, v=v), EPOCH, sc),
            V0,
            1e-6,
        )

        assert relative_error(by_position, numeric_r) < 1e-6
        assert relative_error(by_velocity, numeric_v) < 1e-6

    def test_body_fixed_has_no_state_partials(self, origin, sc):
        """The body-fixed rotation depends on the epoch alone."""
        pfogm = framed(origin, sc, "body_fixed")

        assert np.all(pfogm._compute_partial_by_position(R0, state_dict(), EPOCH, sc) == 0)
        assert np.all(pfogm._compute_partial_by_velocity(R0, state_dict(), EPOCH, sc) == 0)

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    def test_state_partial_holds_the_rotation_at_the_active_batch(
        self, origin, sc, frame
    ):
        """da/da_pfogm is the frame rotation at the active batch, zero elsewhere."""
        pfogm = framed(origin, sc, frame)
        a = stacked()

        H = pfogm._compute_partial_by_acceleration_state(R0, state_dict(a), EPOCH, sc)
        numeric = fd_jacobian(
            lambda x: pfogm.compute_acceleration(R0, state_dict(x), EPOCH, sc), a, 1e-10
        )

        assert H.shape == (3, 3 * N_BATCHES)
        assert np.all(H[:, :3] == 0) and np.all(H[:, 6:] == 0)
        assert relative_error(H, numeric) < 1e-8

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    def test_drift_acts_on_the_frame_components(self, origin, sc, frame):
        """The frame rotates what is applied, not how the components decay."""
        a = stacked()
        inertial = framed(origin, sc, "inertial")
        rotated = framed(origin, sc, frame)

        np.testing.assert_array_equal(
            rotated._compute_acceleration_state_derivative(R0, state_dict(a), EPOCH, sc),
            inertial._compute_acceleration_state_derivative(
                R0, state_dict(a), EPOCH, sc
            ),
        )


def pfogm_force_model(sc, **kwargs):
    """ForceModelTranslation configured with a framed piecewise Gauss-Markov model."""
    return scb.ForceModelTranslation(
        primary_body=sc,
        piecewise_first_order_gauss_markov=True,
        pfogm_batch_length=BATCH_LENGTH,
        pfogm_n_batches=N_BATCHES,
        pfogm_beta=BETA,
        t0=T0,
        **kwargs,
    )


class TestFrameForceModel:
    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_asks_for_the_velocity_partial(
        self, sc, state, frame, expected
    ):
        """The RTN triad needs the da/dv block; the other frames do not."""
        kwargs = (
            {"pfogm_body_fixed_frame": "IAU_EARTH"} if frame == "body_fixed" else {}
        )
        force_model = pfogm_force_model(sc, pfogm_frame=frame, **kwargs)

        assert force_model._velocity_partial_active is expected
        force_model.initialize_force_models(state, J2000, 0.0)
        assert force_model._velocity_partial_active is expected

    def test_frame_reaches_the_model(self, sc, state):
        """The frame settings are handed to the built model and named in the list."""
        force_model = pfogm_force_model(
            sc, pfogm_frame="RTN", pfogm_relative_body="MOON"
        )
        force_model.initialize_force_models(state, J2000, 0.0)

        built = force_model.force_models["PiecewiseFirstOrderGaussMarkov"]
        assert force_model.pfogm_frame == "RTN"
        assert built.frame == "RTN"
        assert built.relative_body == "MOON"
        assert (
            "PiecewiseFirstOrderGaussMarkov(n_batches=3, frame=RTN)"
            in force_model.list_forces()
        )

    def test_frame_serialization_round_trip(self, sc, state):
        """The frame and body-fixed frame survive to_dict / from_dict."""
        force_model = pfogm_force_model(
            sc, pfogm_frame="body_fixed", pfogm_body_fixed_frame="IAU_EARTH"
        )

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc)
        restored.initialize_force_models(state, J2000, 0.0)

        rebuilt = restored.force_models["PiecewiseFirstOrderGaussMarkov"]
        assert restored.pfogm_frame == "body_fixed"
        assert rebuilt.frame == "body_fixed"
        assert rebuilt.body_fixed_frame == "IAU_EARTH"


# endregion                                                          #
# ------------------------------------------------------------------#
