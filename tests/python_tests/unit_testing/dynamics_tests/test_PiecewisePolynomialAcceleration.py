# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ===========================================================#
#  Unit Tests for the PiecewisePolynomialAcceleration class  #
# ===========================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

SPICE_ID = -2468
R0 = np.array([6800.0, 1200.0, 2400.0])
V0 = np.array([-1.4, 6.9, 2.1])
BATCH_LENGTH = 1800.0
N_BATCHES = 3
T0 = 100.0

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
        "PPOLY_SC", SPICE_ID, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )


def model(origin, sc, order, t0=T0, **kwargs):
    """Model on the shared test grid."""
    return scb.PiecewisePolynomialAcceleration(
        origin,
        sc,
        J2000,
        t0=t0,
        batch_length=BATCH_LENGTH,
        n_batches=N_BATCHES,
        order=order,
        **kwargs,
    )


def coefficients(order, seed=11, scale=1e-9):
    """Reproducible coefficient vector of the right length for an order."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=3 * (N_BATCHES + order))


def state_dict(p=None, v=V0):
    """Propagated-state dictionary as the force model receives it."""
    values = {("velocity", SPICE_ID): np.asarray(v, dtype=float)}
    if p is not None:
        values[("a_ppoly", SPICE_ID)] = np.asarray(p, dtype=float)
    return values


def fd_jacobian(f, x, h):
    """Central-difference Jacobian of a vector function."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(f(x))
    jac = np.zeros((f0.size, x.size))
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        jac[:, i] = (np.atleast_1d(f(x + step)) - np.atleast_1d(f(x - step))) / (2 * h)
    return jac


def framed(origin, sc, order, frame, **kwargs):
    """Model with its coefficients in ``frame``, body-fixed to the Earth."""
    if frame == "body_fixed":
        kwargs.setdefault("body_fixed_frame", "IAU_EARTH")
    return model(origin, sc, order, frame=frame, **kwargs)


def rtn_basis(r, v):
    """Radial, along-track and cross-track unit vectors, as columns."""
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    return np.column_stack((r_hat, np.cross(n_hat, r_hat), n_hat))


def relative_error(analytic, numeric):
    """Largest entry-wise difference, scaled by the size of the Jacobian."""
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)
    scale = np.abs(numeric).max()
    if scale == 0.0:
        return np.abs(analytic).max()
    return np.abs(analytic - numeric).max() / scale


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests  #
# --------------------#
class TestConfiguration:
    def test_initialization(self, origin, sc):
        """Constructor arguments are exposed unchanged."""
        pp = model(origin, sc, order=1)

        assert pp.origin is origin
        assert pp.origin_name == origin.name
        assert pp.ref_frame is J2000
        assert pp.order == 1
        assert pp.t0 == T0
        assert pp.batch_length == BATCH_LENGTH
        assert pp.n_batches == N_BATCHES
        assert pp.default_coefficients is None
        assert pp.depends_on_velocity is False

    @pytest.mark.parametrize("order, expected", [(0, 9), (1, 12)])
    def test_coefficient_count_follows_the_order(self, origin, sc, order, expected):
        """One vector per batch for a hold, one per batch boundary for a ramp."""
        pp = model(origin, sc, order=order)

        assert pp.n_coefficients == expected
        assert len(pp.coefficient_labels()) == expected

    def test_labels_name_batches_or_nodes(self, origin, sc):
        """Labels follow the stacked state layout of each order."""
        hold = model(origin, sc, order=0).coefficient_labels()
        ramp = model(origin, sc, order=1).coefficient_labels()

        assert hold[:3] == ["batch0_x", "batch0_y", "batch0_z"]
        assert hold[-1] == f"batch{N_BATCHES - 1}_z"
        assert ramp[:3] == ["node0_x", "node0_y", "node0_z"]
        assert ramp[-1] == f"node{N_BATCHES}_z"

    def test_epoch_array_t0_is_read_as_ephemeris_time(self, origin, sc):
        """An EpochArray t0 anchors the grid at its raw ephemeris time."""
        epoch = scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")

        pp = model(origin, sc, order=0, t0=epoch)

        assert pp.t0 == pytest.approx(float(np.asarray(epoch.et).ravel()[0]))


class TestValidation:
    @pytest.mark.parametrize("order", [None, 2, -1, 0.5, True, "1"])
    def test_unsupported_order_is_reported(self, origin, sc, order):
        """Only 0 and 1 are accepted; booleans and strings are not orders."""
        with pytest.raises(
            scb.PiecewisePolynomialAcceleration.InvalidOrderError, match="order"
        ):
            model(origin, sc, order=order)

    @pytest.mark.parametrize(
        "batch_length, n_batches",
        [(0.0, 3), (-10.0, 3), (None, 3), (1800.0, 0), (1800.0, 1.5), (1800.0, None)],
    )
    def test_degenerate_grid_is_reported(self, origin, sc, batch_length, n_batches):
        """The batches need a positive length and a positive whole count."""
        with pytest.raises(scb.PiecewisePolynomialAcceleration.InvalidBatchGridError):
            scb.PiecewisePolynomialAcceleration(
                origin,
                sc,
                J2000,
                t0=T0,
                batch_length=batch_length,
                n_batches=n_batches,
                order=0,
            )

    def test_invalid_t0_type_is_reported(self, origin, sc):
        """Anything but an EpochArray or a real number is refused."""
        with pytest.raises(
            scb.PiecewisePolynomialAcceleration.InvalidEpochError, match="EpochArray"
        ):
            model(origin, sc, order=0, t0="2000 JAN 01")

    def test_default_coefficients_must_match_the_order(self, origin, sc):
        """A hold-sized vector is refused by a ramp, which needs one more node."""
        with pytest.raises(
            scb.PiecewisePolynomialAcceleration.CoefficientSizeError, match="12"
        ):
            model(origin, sc, order=1, coefficients=coefficients(0))

    def test_state_coefficients_must_match_the_order(self, origin, sc):
        """A mis-sized a_ppoly state is reported when it is read."""
        pp = model(origin, sc, order=0)

        with pytest.raises(
            scb.PiecewisePolynomialAcceleration.CoefficientSizeError, match="9"
        ):
            pp.compute_acceleration(R0, state_dict(coefficients(1)), T0, sc)

    @pytest.mark.parametrize("epoch", [T0 - 1.0, T0 + N_BATCHES * BATCH_LENGTH + 1.0])
    def test_epoch_outside_the_grid_is_reported(self, origin, sc, epoch):
        """No acceleration is invented outside the batches."""
        pp = model(origin, sc, order=1, coefficients=coefficients(1))

        with pytest.raises(
            scb.PiecewisePolynomialAcceleration.EpochOutsideGridError,
            match="ppoly_n_batches",
        ):
            pp.compute_acceleration(R0, {}, epoch, sc)


class TestAcceleration:
    def test_zero_order_hold_is_constant_within_each_batch(self, origin, sc):
        """Every epoch of batch k returns c_k, and the next batch starts on c_(k+1)."""
        p = coefficients(0)
        pp = model(origin, sc, order=0)
        nodes = p.reshape(-1, 3)

        for k in range(N_BATCHES):
            start = T0 + k * BATCH_LENGTH
            for epoch in (start, start + 0.3 * BATCH_LENGTH, start + BATCH_LENGTH - 1e-3):
                assert pp.compute_acceleration(R0, state_dict(p), epoch, sc) == (
                    pytest.approx(nodes[k], abs=0.0)
                )

    def test_zero_order_hold_closes_on_the_last_batch(self, origin, sc):
        """The grid's closing epoch still belongs to the last batch."""
        p = coefficients(0)
        pp = model(origin, sc, order=0)
        t_end = T0 + N_BATCHES * BATCH_LENGTH

        acc = pp.compute_acceleration(R0, state_dict(p), t_end, sc)

        assert acc == pytest.approx(p.reshape(-1, 3)[-1], abs=0.0)

    def test_first_order_hold_passes_through_its_nodes(self, origin, sc):
        """Node j is the acceleration at t0 + j*L, including both grid ends."""
        p = coefficients(1)
        pp = model(origin, sc, order=1)
        nodes = p.reshape(-1, 3)

        for j in range(N_BATCHES + 1):
            acc = pp.compute_acceleration(R0, state_dict(p), T0 + j * BATCH_LENGTH, sc)
            assert acc == pytest.approx(nodes[j], rel=1e-12, abs=1e-24)

    def test_first_order_hold_is_linear_within_a_batch(self, origin, sc):
        """Inside batch k the acceleration is the linear blend of its two nodes."""
        p = coefficients(1)
        pp = model(origin, sc, order=1)
        nodes = p.reshape(-1, 3)

        for tau in (0.25, 0.5, 0.9):
            acc = pp.compute_acceleration(
                R0, state_dict(p), T0 + (1 + tau) * BATCH_LENGTH, sc
            )
            assert acc == pytest.approx(
                (1 - tau) * nodes[1] + tau * nodes[2], rel=1e-12, abs=1e-24
            )

    def test_first_order_hold_is_continuous_across_batches(self, origin, sc):
        """Ending batch k and starting batch k+1 give the same acceleration."""
        p = coefficients(1)
        pp = model(origin, sc, order=1)

        for k in range(1, N_BATCHES):
            boundary = T0 + k * BATCH_LENGTH

            # The boundary as the closing epoch of the batch before it ...
            pp.pin_batch_window(boundary - BATCH_LENGTH, boundary)
            left = pp.compute_acceleration(R0, state_dict(p), boundary, sc)
            pp.clear_batch_window()
            # ... and as the opening epoch of the batch after it
            right = pp.compute_acceleration(R0, state_dict(p), boundary, sc)

            assert left == pytest.approx(right, rel=1e-12, abs=1e-24)

    def test_zero_order_hold_jumps_across_batches(self, origin, sc):
        """The hold is discontinuous: each side of a boundary keeps its own batch."""
        p = coefficients(0)
        pp = model(origin, sc, order=0)
        nodes = p.reshape(-1, 3)
        boundary = T0 + BATCH_LENGTH

        pp.pin_batch_window(T0, boundary)
        left = pp.compute_acceleration(R0, state_dict(p), boundary, sc)
        pp.clear_batch_window()
        right = pp.compute_acceleration(R0, state_dict(p), boundary, sc)

        assert left == pytest.approx(nodes[0], abs=0.0)
        assert right == pytest.approx(nodes[1], abs=0.0)

    def test_state_coefficients_take_precedence_over_defaults(self, origin, sc):
        """The propagated state wins over the model's configured coefficients."""
        defaults = coefficients(1, seed=1)
        estimated = coefficients(1, seed=2)
        pp = model(origin, sc, order=1, coefficients=defaults)
        epoch = T0 + 0.4 * BATCH_LENGTH

        from_state = pp.compute_acceleration(R0, state_dict(estimated), epoch, sc)
        from_defaults = pp.compute_acceleration(R0, state_dict(), epoch, sc)

        nodes = estimated.reshape(-1, 3)
        fallback = defaults.reshape(-1, 3)
        assert from_state == pytest.approx(0.6 * nodes[0] + 0.4 * nodes[1], rel=1e-12)
        assert from_defaults == pytest.approx(
            0.6 * fallback[0] + 0.4 * fallback[1], rel=1e-12
        )

    def test_no_coefficients_means_no_acceleration(self, origin, sc):
        """With neither a state entry nor defaults the model is inert."""
        pp = model(origin, sc, order=0)

        assert np.all(pp.compute_acceleration(R0, state_dict(), T0, sc) == 0.0)


class TestPartials:
    @pytest.mark.parametrize("order", [0, 1])
    @pytest.mark.parametrize("fraction", [0.0, 0.37, 1.5, 2.99])
    def test_coefficient_partial_matches_finite_differences(
        self, origin, sc, order, fraction
    ):
        """da/dc is the interpolation matrix at every epoch of the grid."""
        p = coefficients(order)
        pp = model(origin, sc, order=order)
        epoch = T0 + fraction * BATCH_LENGTH

        analytic = pp._compute_partial_by_coefficients(R0, state_dict(p), epoch, sc)
        numeric = fd_jacobian(
            lambda x: pp.compute_acceleration(R0, state_dict(x), epoch, sc), p, 1e-10
        )

        assert analytic.shape == (3, pp.n_coefficients)
        assert relative_error(analytic, numeric) < 1e-8

    def test_first_order_partial_holds_the_hat_weights(self, origin, sc):
        """A quarter into batch 1 the nodes 1 and 2 weigh 3/4 and 1/4."""
        pp = model(origin, sc, order=1)

        partial = pp._compute_partial_by_coefficients(
            R0, state_dict(), T0 + 1.25 * BATCH_LENGTH, sc
        )

        expected = np.zeros((3, 12))
        expected[:, 3:6] = 0.75 * np.eye(3)
        expected[:, 6:9] = 0.25 * np.eye(3)
        assert partial == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("order", [0, 1])
    def test_has_no_state_partials(self, origin, sc, order):
        """The acceleration depends on time only."""
        p = coefficients(order)
        pp = model(origin, sc, order=order)
        epoch = T0 + 0.5 * BATCH_LENGTH

        assert np.all(pp._compute_partial_by_position(R0, state_dict(p), epoch, sc) == 0)
        assert np.all(pp._compute_partial_by_velocity(R0, state_dict(p), epoch, sc) == 0)


class TestFrames:
    """Coefficients given in the RTN (RTN) or a body-fixed frame."""

    # Inside batch 1, so an order-1 ramp mixes two nodes.
    EPOCH = T0 + 1.3 * BATCH_LENGTH

    def test_unknown_frame_is_reported(self, origin, sc):
        """The RTN frame is spelled 'RTN'; the retired 'lvlh' name is refused."""
        with pytest.raises(scb.DynamicalUtils.InvalidFrameError):
            model(origin, sc, order=0, frame="lvlh")

    def test_body_fixed_frame_needs_a_frame_name(self, origin, sc):
        """A body-fixed model must say which body-fixed frame it lives in."""
        with pytest.raises(
            scb.DynamicalUtils.MissingBodyFixedFrameError
        ):
            model(origin, sc, order=0, frame="body_fixed")

    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_depends_on_velocity(self, origin, sc, frame, expected):
        """The RTN triad is built from the velocity; the other frames are not."""
        pp = framed(origin, sc, 0, frame)

        assert pp.frame == frame
        assert pp.depends_on_velocity is expected

    def test_rtn_labels_name_the_axes(self, origin, sc):
        """RTN coefficients are labelled radial, along-track and cross-track."""
        hold = framed(origin, sc, 0, "RTN").coefficient_labels()
        ramp = framed(origin, sc, 1, "RTN").coefficient_labels()

        assert hold[:3] == ["batch0_R", "batch0_T", "batch0_N"]
        assert ramp[-3:] == [f"node{N_BATCHES}_{axis}" for axis in "RTN"]

    @pytest.mark.parametrize("order", [0, 1])
    def test_rtn_turns_the_interpolated_vector_onto_rtn(self, origin, sc, order):
        """The held or ramped vector is laid along the R, S and W directions."""
        p = coefficients(order)

        local = model(origin, sc, order).compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )
        accel = framed(origin, sc, order, "RTN").compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )

        assert relative_error(accel, rtn_basis(R0, V0) @ local) < 1e-14

    @pytest.mark.parametrize("order", [0, 1])
    def test_body_fixed_turns_with_the_body(self, origin, sc, order):
        """A body-fixed vector is rotated out of the body frame at the epoch."""
        p = coefficients(order)

        local = model(origin, sc, order).compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )
        accel = framed(origin, sc, order, "body_fixed").compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )

        rotation = np.asarray(
            scb.SpiceManager.get_xfrm("IAU_EARTH", "J2000", self.EPOCH), dtype=float
        )
        assert relative_error(accel, rotation @ local) < 1e-14

    def test_rtn_about_another_body(self, origin, sc):
        """With a relative body the triad is built about that body."""
        p = coefficients(1)

        local = model(origin, sc, 1).compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )
        accel = framed(origin, sc, 1, "RTN", relative_body="MOON").compute_acceleration(
            R0, state_dict(p), self.EPOCH, sc
        )

        moon = scb.SpiceManager.get_state("MOON", self.EPOCH, J2000, origin.name)
        offset = np.asarray(moon.values, dtype=float).ravel()
        expected = rtn_basis(R0 - offset[:3], V0 - offset[3:6]) @ local
        assert relative_error(accel, expected) < 1e-12

    def test_rtn_needs_a_velocity(self, origin, sc):
        """Without a velocity there is no RTN triad to rotate by."""
        pp = framed(origin, sc, 0, "RTN")

        with pytest.raises(scb.DynamicalUtils.MissingVelocityError):
            pp.compute_acceleration(
                R0, {("a_ppoly", SPICE_ID): coefficients(0)}, self.EPOCH, sc
            )

    @pytest.mark.parametrize("relative_body", [None, "MOON"])
    @pytest.mark.parametrize("order", [0, 1])
    def test_rtn_state_partials_match_finite_differences(
        self, origin, sc, order, relative_body
    ):
        """da/dr and da/dv carry the derivatives of the RTN triad."""
        p = coefficients(order)
        pp = framed(origin, sc, order, "RTN", relative_body=relative_body)

        by_position = pp._compute_partial_by_position(R0, state_dict(p), self.EPOCH, sc)
        by_velocity = pp._compute_partial_by_velocity(R0, state_dict(p), self.EPOCH, sc)
        # 1 km is small against the 7000 km orbit that sets the triad.
        numeric_r = fd_jacobian(
            lambda r: pp.compute_acceleration(r, state_dict(p), self.EPOCH, sc), R0, 1.0
        )
        numeric_v = fd_jacobian(
            lambda v: pp.compute_acceleration(R0, state_dict(p, v=v), self.EPOCH, sc),
            V0,
            1e-6,
        )

        assert relative_error(by_position, numeric_r) < 1e-6
        assert relative_error(by_velocity, numeric_v) < 1e-6

    @pytest.mark.parametrize("order", [0, 1])
    def test_body_fixed_has_no_state_partials(self, origin, sc, order):
        """The body-fixed rotation depends on the epoch alone."""
        p = coefficients(order)
        pp = framed(origin, sc, order, "body_fixed")

        assert np.all(
            pp._compute_partial_by_position(R0, state_dict(p), self.EPOCH, sc) == 0
        )
        assert np.all(
            pp._compute_partial_by_velocity(R0, state_dict(p), self.EPOCH, sc) == 0
        )

    @pytest.mark.parametrize("frame", ["RTN", "body_fixed"])
    @pytest.mark.parametrize("order", [0, 1])
    def test_coefficient_partial_matches_finite_differences(
        self, origin, sc, order, frame
    ):
        """da/dc is the interpolation matrix rotated out of the frame."""
        p = coefficients(order)
        pp = framed(origin, sc, order, frame)

        analytic = pp._compute_partial_by_coefficients(
            R0, state_dict(p), self.EPOCH, sc
        )
        numeric = fd_jacobian(
            lambda x: pp.compute_acceleration(R0, state_dict(x), self.EPOCH, sc),
            p,
            1e-10,
        )

        assert analytic.shape == (3, pp.n_coefficients)
        assert relative_error(analytic, numeric) < 1e-8


# endregion Tests     #
# --------------------#


# ------------------------------------------------------------------#
# region  ForceModelTranslation and propagation                      #
# ------------------------------------------------------------------#
# The propagated grid starts at the state epoch, 0, so an arc to 3000 s crosses
# the first batch boundary at 1800 s and stays short of the second.
PROP_END = 3000.0


def ppoly_force_model(sc, order, t0=0.0, **kwargs):
    """ForceModelTranslation configured with a piecewise polynomial acceleration."""
    return scb.ForceModelTranslation(
        primary_body=sc,
        t0=t0,
        piecewise_polynomial=True,
        ppoly_order=order,
        ppoly_batch_length=BATCH_LENGTH,
        ppoly_n_batches=N_BATCHES,
        **kwargs,
    )


R_PROP = np.array([7000.0, 0.0, 1000.0])
V_PROP = np.array([0.0, 7.3, 0.5])


def ppoly_state(origin, sc, values=None, epoch=0.0, extra=(), r=R_PROP, v=V_PROP):
    """State carrying the stacked coefficients as a static parameter."""
    definition = (
        scb.StateDefinition()
        .position(sc, scb.ArrayWFrame(np.asarray(r, dtype=float), km, J2000))
        .velocity(sc, scb.ArrayWFrame(np.asarray(v, dtype=float), km / sec, J2000))
    )
    if values is not None:
        definition = definition.param(
            "a_ppoly",
            sc,
            scb.ArrayWFrame(np.asarray(values, dtype=float), km / sec**2, J2000),
            dynamics="static",
        )
    for name, size in extra:
        definition = definition.param(
            name,
            sc,
            scb.ArrayWFrame(np.zeros(size), km / sec**2, J2000),
            dynamics="dynamic",
        )
    return scb.StateArray(
        epoch=scb.EpochArray(np.array([epoch]), sys="TDB", rep="NUM"),
        origin=origin,
        state=definition,
    )


def propagate(
    origin, sc, force_model, values=None, t_end=PROP_END, points=4, r=R_PROP, v=V_PROP
):
    """Propagate one arc and return the final state and its STM."""
    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=ppoly_state(origin, sc, values=values, r=r, v=v),
        tspan=scb.EpochArray(np.linspace(0.0, t_end, points), sys="TDB", rep="NUM"),
        force_models=force_model,
        display_progress=False,
    )
    propagator.propagate()

    propagated = propagator.propagated_state_array
    final = np.concatenate(
        (
            np.asarray(propagated.state[0][5].quantity.values)[-1],
            np.asarray(propagated.state[1][5].quantity.values)[-1],
        )
    )
    stm = np.asarray(propagator.STM.quantity.values)
    size = int(round(stm.shape[-1] ** 0.5))
    return final, stm[-1].reshape(size, size)


class TestForceModel:
    def test_reports_the_coefficient_partial(self, origin, sc):
        """ForceModelTranslation resolves the a_ppoly partial key to the model."""
        p = coefficients(1)
        force_model = ppoly_force_model(sc, order=1)
        force_model.initialize_force_models(ppoly_state(origin, sc, p), J2000, 0.0)
        epoch = 0.4 * BATCH_LENGTH

        partial = force_model._collect_partials(
            f"acceleration@{SPICE_ID}_by_a_ppoly@{SPICE_ID}",
            scb.ArrayWUnits(R0, km),
            state_dict(p),
            epoch,
            sc,
        )

        direct = force_model.force_models[
            "PiecewisePolynomialAcceleration"
        ]._compute_partial_by_coefficients(R0, state_dict(p), epoch, sc)
        assert np.asarray(partial).shape == (3, 12)
        assert np.asarray(partial) == pytest.approx(direct)

    def test_is_listed_with_its_order(self, sc):
        """The force list names the order and the batch count."""
        force_model = ppoly_force_model(sc, order=1)

        assert "PiecewisePolynomialAcceleration(order=1, n_batches=3)" in (
            force_model.list_forces()
        )

    def test_order_is_required(self, sc):
        """The order sizes the state, so it is asked for up front."""
        with pytest.raises(scb.PiecewisePolynomialAcceleration.InvalidOrderError):
            scb.ForceModelTranslation(
                primary_body=sc,
                t0=0.0,
                piecewise_polynomial=True,
                ppoly_batch_length=BATCH_LENGTH,
                ppoly_n_batches=N_BATCHES,
            )

    def test_grid_is_required(self, sc):
        """A batch grid is checked before any state is built against it."""
        with pytest.raises(scb.PiecewisePolynomialAcceleration.InvalidBatchGridError):
            scb.ForceModelTranslation(
                primary_body=sc, t0=0.0, piecewise_polynomial=True, ppoly_order=0
            )

    def test_missing_epoch_is_reported(self, origin, sc):
        """With no configured or leg t0 there is nothing to anchor the grid."""
        force_model = ppoly_force_model(sc, order=0, t0=None)

        with pytest.raises(scb.ForceModelTranslation.MissingBatchEpochError):
            force_model.initialize_force_models(ppoly_state(origin, sc), J2000)

    def test_missing_epoch_falls_back_to_leg(self, origin, sc):
        """With no configured t0 the grid starts at the leg it is bound to."""
        force_model = ppoly_force_model(sc, order=0, t0=None)

        force_model.initialize_force_models(ppoly_state(origin, sc), J2000, 600.0)

        assert force_model.force_models["PiecewisePolynomialAcceleration"].t0 == 600.0

    def test_configured_epoch_survives_leg_binding(self, origin, sc):
        """Binding to a leg does not move the batch grid."""
        force_model = ppoly_force_model(sc, order=1, t0=100.0)

        force_model.initialize_force_models(ppoly_state(origin, sc), J2000, 4000.0)

        assert force_model.force_models["PiecewisePolynomialAcceleration"].t0 == 100.0

    def test_serialization_round_trip(self, origin, sc):
        """The order, grid and default coefficients survive to_dict / from_dict."""
        p = coefficients(1)
        force_model = ppoly_force_model(sc, order=1, t0=100.0, ppoly_coefficients=p)

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc)
        restored.initialize_force_models(ppoly_state(origin, sc), J2000, 4000.0)

        assert "PiecewisePolynomialAcceleration" in restored.dynamics_mode_list
        assert restored.ppoly_order == 1
        assert restored.ppoly_batch_length == BATCH_LENGTH
        assert restored.ppoly_n_batches == N_BATCHES
        rebuilt = restored.force_models["PiecewisePolynomialAcceleration"]
        assert rebuilt.t0 == 100.0
        assert rebuilt.default_coefficients == pytest.approx(p, abs=0.0)

    @pytest.mark.parametrize("order", [0, 1])
    def test_estimated_and_fixed_coefficients_agree(self, origin, sc, order):
        """Estimating the coefficients does not change the trajectory they produce."""
        p = coefficients(order, scale=1e-8)

        estimated, _ = propagate(origin, sc, ppoly_force_model(sc, order), values=p)
        fixed, _ = propagate(
            origin, sc, ppoly_force_model(sc, order, ppoly_coefficients=p)
        )

        assert estimated == pytest.approx(fixed, abs=1e-9)

    @pytest.mark.parametrize("frame", ["inertial", "RTN"])
    @pytest.mark.parametrize(
        "order, indices",
        [
            # c0_x, c1_y, and c2_z of a batch the arc never reaches
            (0, (0, 4, 8)),
            # n0_x, n1_y, n2_z ramping into batch 1, and n3_y it never reaches
            (1, (0, 4, 8, 10)),
        ],
    )
    def test_propagation_maps_coefficients_into_the_stm(
        self, origin, sc, order, indices, frame
    ):
        """Across a batch boundary the STM's parameter block matches re-propagation."""
        p = coefficients(order, scale=1e-8)
        nominal, stm = propagate(
            origin, sc, ppoly_force_model(sc, order, ppoly_frame=frame), values=p
        )

        assert stm.shape == (6 + 3 * (N_BATCHES + order),) * 2

        step = 1e-10
        for index in indices:
            shifted = p.copy()
            shifted[index] += step
            moved, _ = propagate(
                origin, sc, ppoly_force_model(sc, order, ppoly_frame=frame), values=shifted
            )
            numeric = (moved - nominal) / step

            assert relative_error(stm[:6, 6 + index], numeric) < 1e-4

    def test_rtn_state_block_matches_re_propagation(self, origin, sc):
        """The RTN triad's da/dr and da/dv reach the position-velocity STM."""
        # Large enough that leaving out the triad's partials (about 1e-3 of the
        # block) would fail the comparison.
        p = np.tile([1e-5, -5e-6, 7e-6], N_BATCHES + 1)

        def force_model():
            return ppoly_force_model(sc, 1, ppoly_coefficients=p, ppoly_frame="RTN")

        _, stm = propagate(origin, sc, force_model())

        x0 = np.concatenate((R_PROP, V_PROP))
        steps = [1e-3] * 3 + [1e-6] * 3
        numeric = np.zeros((6, 6))
        for j, step in enumerate(steps):
            ends = []
            for sign in (1.0, -1.0):
                x = x0.copy()
                x[j] += sign * step
                end, _ = propagate(origin, sc, force_model(), r=x[:3], v=x[3:])
                ends.append(end)
            numeric[:, j] = (ends[0] - ends[1]) / (2.0 * step)

        assert relative_error(stm[:6, :6], numeric) < 1e-7

    @pytest.mark.parametrize(
        "frame, expected", [("inertial", False), ("RTN", True), ("body_fixed", False)]
    )
    def test_only_rtn_asks_for_the_velocity_partial(self, origin, sc, frame, expected):
        """The RTN triad needs the da/dv block; the other frames do not."""
        kwargs = {"ppoly_body_fixed_frame": "IAU_EARTH"} if frame == "body_fixed" else {}
        force_model = ppoly_force_model(sc, order=0, ppoly_frame=frame, **kwargs)

        assert force_model._velocity_partial_active is expected
        force_model.initialize_force_models(ppoly_state(origin, sc), J2000, 0.0)
        assert force_model._velocity_partial_active is expected

    def test_frame_reaches_the_model(self, origin, sc):
        """The frame settings are handed to the built model and named in the list."""
        force_model = ppoly_force_model(
            sc, order=1, ppoly_frame="body_fixed", ppoly_body_fixed_frame="IAU_EARTH"
        )
        force_model.initialize_force_models(ppoly_state(origin, sc), J2000, 0.0)

        built = force_model.force_models["PiecewisePolynomialAcceleration"]
        assert force_model.ppoly_frame == "body_fixed"
        assert built.frame == "body_fixed"
        assert built.body_fixed_frame == "IAU_EARTH"
        assert (
            "PiecewisePolynomialAcceleration(order=1, n_batches=3, frame=body_fixed)"
            in force_model.list_forces()
        )

    def test_unknown_frame_is_reported_at_build(self, origin, sc):
        """An unknown frame stops the build rather than defaulting to inertial."""
        force_model = ppoly_force_model(sc, order=0, ppoly_frame="lvlh")

        with pytest.raises(scb.DynamicalUtils.InvalidFrameError):
            force_model.initialize_force_models(ppoly_state(origin, sc), J2000, 0.0)

    def test_frame_serialization_round_trip(self, origin, sc):
        """The frame and relative body survive to_dict / from_dict."""
        force_model = ppoly_force_model(
            sc, order=0, ppoly_frame="RTN", ppoly_relative_body="MOON"
        )

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc)
        restored.initialize_force_models(ppoly_state(origin, sc), J2000, 0.0)

        rebuilt = restored.force_models["PiecewisePolynomialAcceleration"]
        assert restored.ppoly_frame == "RTN"
        assert rebuilt.frame == "RTN"
        assert rebuilt.relative_body == "MOON"
        assert restored._velocity_partial_active

    def test_shares_the_propagator_breakpoints_with_pfogm(self, origin, sc):
        """Both batch-grid models are cut at, and pinned to, their own boundaries."""
        force_model = ppoly_force_model(
            sc,
            order=0,
            ppoly_coefficients=coefficients(0, scale=1e-8),
            piecewise_first_order_gauss_markov=True,
            pfogm_batch_length=1200.0,
            pfogm_n_batches=3,
            pfogm_beta=np.array([1 / 3600.0] * 3),
        )
        propagator = scb.Propagator(
            primary_body=sc,
            state_vector=ppoly_state(origin, sc, extra=[("a_pfogm", 9)]),
            tspan=scb.EpochArray(np.linspace(0.0, PROP_END, 4), sys="TDB", rep="NUM"),
            force_models=force_model,
            display_progress=False,
        )

        assert [type(m).__name__ for m in propagator._batch_grid_models()] == [
            "PiecewiseFirstOrderGaussMarkov",
            "PiecewisePolynomialAcceleration",
        ]

        propagator.propagate()

        # the pins belong to the arc loop only
        for batch_model in propagator._batch_grid_models():
            assert batch_model._pinned_batch is None

    @pytest.mark.parametrize("order", [0, 1])
    def test_leg_ending_on_a_batch_boundary_integrates(self, origin, sc, order):
        """A leg whose end lands on a batch boundary integrates at mission epochs."""
        # The float spacing near 7.5e8 s is ~1e-7, where an integrator chasing
        # the hold's jump onto its own endpoint runs out of room.
        arc_t0 = 7.5e8
        force_model = ppoly_force_model(
            sc, order, t0=arc_t0, ppoly_coefficients=coefficients(order, scale=1e-8)
        )

        prop = scb.Propagator(
            primary_body=sc,
            state_vector=ppoly_state(origin, sc, epoch=arc_t0),
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
