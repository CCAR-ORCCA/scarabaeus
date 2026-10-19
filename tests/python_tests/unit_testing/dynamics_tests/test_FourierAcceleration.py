# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================================#
#  Unit Tests for the FourierAcceleration class     #
# ==================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

SPICE_ID = -1357
R0 = np.array([6800.0, 1200.0, 2400.0])
V0 = np.array([-1.4, 6.9, 2.1])
EPOCH = 12345.0
PERIOD = 5400.0

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
        "FOURIER_SC", SPICE_ID, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )


def model(origin, sc, degree=1, frame="inertial", angle="time", **kwargs):
    """FourierAcceleration with the phase settings a harmonic series needs."""
    kwargs.setdefault("period", PERIOD)
    kwargs.setdefault("t0", 0.0)
    return scb.FourierAcceleration(
        origin, sc, J2000, degree=degree, frame=frame, angle=angle, **kwargs
    )


def state_dict(coefficients=None, r=R0, v=V0):
    """Propagated-state dictionary as the force model receives it."""
    values = {("velocity", SPICE_ID): v}
    if coefficients is not None:
        values[("a_fourier", SPICE_ID)] = np.asarray(coefficients, dtype=float)
    return values


def coefficients(degree, seed=7, scale=1e-9):
    """Reproducible coefficient vector of the right length for a degree."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=3 * (2 * degree + 1))


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


def relative_error(analytic, numeric):
    """Largest entry-wise difference, scaled by the size of the Jacobian."""
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)
    scale = np.abs(numeric).max()
    if scale == 0.0:
        return np.abs(analytic).max()
    return np.abs(analytic - numeric).max() / scale


# Frames and phases exercised by the partial checks. "time" in an inertial or
# body-fixed frame is the case with no state dependence at all; the others each
# switch on one of the two mechanisms that create one.
STATE_CASES = [
    ("inertial", "time"),
    ("inertial", "argument_of_latitude"),
    ("inertial", "solar_longitude"),
    ("RTN", "time"),
    ("RTN", "argument_of_latitude"),
    ("RTN", "solar_longitude"),
]


# --------------------------#
# region  Configuration     #
# --------------------------#


class TestConfiguration:
    def test_initialization(self, origin, sc):
        """Verifies correct attribute assignment on construction."""
        fourier = model(origin, sc, degree=2, frame="RTN", angle="time")

        assert fourier.degree == 2
        assert fourier.n_terms == 5
        assert fourier.n_coefficients == 15
        assert fourier.frame == "RTN"
        assert fourier.angle == "time"
        assert fourier.period == PERIOD
        assert fourier.t0 == 0.0
        assert fourier.origin is origin
        assert fourier.origin_name == origin.name
        assert fourier.ref_frame == J2000

    def test_a_constant_series_needs_no_phase_settings(self, origin, sc):
        """Degree 0 has no phase, so a period and an epoch are not required."""
        fourier = scb.FourierAcceleration(origin, sc, J2000, degree=0)

        assert fourier.n_coefficients == 3
        assert fourier.period is None
        assert fourier.t0 is None

    @pytest.mark.parametrize(
        "degree, expected",
        [(0, 3), (1, 9), (2, 15), (4, 27)],
    )
    def test_coefficient_count_follows_the_degree(self, origin, sc, degree, expected):
        """A degree-N series carries 3*(2N+1) coefficients, labelled per axis."""
        fourier = model(origin, sc, degree=degree, frame="RTN")

        assert fourier.n_coefficients == expected
        labels = fourier.coefficient_labels()
        assert len(labels) == expected
        assert labels[: 2 * degree + 1] == ["A0_R"] + [
            f"{letter}{k}_R" for k in range(1, degree + 1) for letter in ("A", "B")
        ]
        assert labels[-1].endswith("_N")

    def test_epoch_array_t0_is_read_as_ephemeris_time(self, origin, sc):
        """An EpochArray t0 anchors the phase at its raw ephemeris epoch."""
        epochs = scb.EpochArray(np.array([EPOCH, EPOCH + 60.0]), sys="TDB", rep="NUM")
        fourier = model(origin, sc, t0=epochs)

        assert fourier.t0 == pytest.approx(float(np.asarray(epochs.et).ravel()[0]))


class TestValidation:
    def test_unknown_frame_is_reported(self, origin, sc):
        """An unsupported frame is refused, listing the ones that exist."""
        with pytest.raises(scb.DynamicalUtils.InvalidFrameError, match="RTN"):
            model(origin, sc, frame="lvlh")

    def test_unknown_angle_is_reported(self, origin, sc):
        """An unsupported phase angle is refused, pointing at the alternative."""
        with pytest.raises(
            scb.FourierAcceleration.InvalidAngleError, match="argument_of_latitude"
        ):
            model(origin, sc, angle="true_anomaly")

    @pytest.mark.parametrize("degree", [-1, 1.5])
    def test_negative_degree_is_reported(self, origin, sc, degree):
        """A degree that is not a non-negative integer is refused."""
        with pytest.raises(scb.FourierAcceleration.InvalidDegreeError):
            model(origin, sc, degree=degree)

    @pytest.mark.parametrize("period", [None, 0.0, -10.0])
    def test_harmonic_series_needs_a_period(self, origin, sc, period):
        """A time-keyed harmonic series without a usable period is refused."""
        with pytest.raises(scb.FourierAcceleration.MissingPeriodError):
            scb.FourierAcceleration(
                origin, sc, J2000, degree=1, angle="time", period=period, t0=0.0
            )

    def test_harmonic_series_needs_a_phase_epoch(self, origin, sc):
        """A time-keyed harmonic series without a phase origin is refused."""
        with pytest.raises(
            scb.FourierAcceleration.MissingEpochError, match="MissionSequence"
        ):
            scb.FourierAcceleration(
                origin, sc, J2000, degree=1, angle="time", period=PERIOD, t0=None
            )

    def test_body_fixed_frame_needs_a_frame_name(self, origin, sc):
        """A body-fixed series must say which body-fixed frame it lives in."""
        with pytest.raises(scb.DynamicalUtils.MissingBodyFixedFrameError):
            model(origin, sc, frame="body_fixed")

    def test_invalid_t0_type_is_reported(self, origin, sc):
        """A t0 that is neither an EpochArray nor a real number is refused."""
        with pytest.raises(scb.FourierAcceleration.InvalidEpochError):
            model(origin, sc, t0="2028 Feb 8 00:00:00")

    def test_coefficient_state_must_match_the_degree(self, origin, sc):
        """A mis-sized a_fourier state names the size the degree implies."""
        fourier = model(origin, sc, degree=1)

        with pytest.raises(scb.FourierAcceleration.CoefficientSizeError, match="9"):
            fourier.compute_acceleration(R0, state_dict(np.zeros(6)), EPOCH, sc)


# --------------------------#
# region  Acceleration      #
# --------------------------#


class TestAcceleration:
    def test_series_reproduces_its_own_definition(self, origin, sc):
        """The acceleration equals the explicit sum of its terms."""
        degree = 3
        fourier = model(origin, sc, degree=degree)
        p = coefficients(degree)

        accel = fourier.compute_acceleration(R0, state_dict(p), EPOCH, sc)

        theta = 2.0 * np.pi * EPOCH / PERIOD
        block = p.reshape(3, 2 * degree + 1)
        expected = np.array(
            [
                block[axis, 0]
                + sum(
                    block[axis, 2 * k - 1] * np.cos(k * theta)
                    + block[axis, 2 * k] * np.sin(k * theta)
                    for k in range(1, degree + 1)
                )
                for axis in range(3)
            ]
        )

        assert accel == pytest.approx(expected, rel=1e-12)

    def test_a_constant_series_ignores_epoch_and_position(self, origin, sc):
        """Degree 0 in an inertial frame is genuinely constant."""
        fourier = scb.FourierAcceleration(origin, sc, J2000, degree=0)
        p = np.array([1e-9, -2e-9, 3e-10])

        here = fourier.compute_acceleration(R0, state_dict(p), EPOCH, sc)
        elsewhere = fourier.compute_acceleration(
            3.0 * R0, state_dict(p, r=3.0 * R0), EPOCH + 1e5, sc
        )

        assert here == pytest.approx(p)
        assert elsewhere == pytest.approx(p)

    def test_time_phase_repeats_after_one_period(self, origin, sc):
        """A time-keyed series is periodic in the configured period."""
        fourier = model(origin, sc, degree=2)
        p = coefficients(2)

        now = fourier.compute_acceleration(R0, state_dict(p), EPOCH, sc)
        later = fourier.compute_acceleration(R0, state_dict(p), EPOCH + PERIOD, sc)
        between = fourier.compute_acceleration(R0, state_dict(p), EPOCH + PERIOD / 3, sc)

        assert later == pytest.approx(now, rel=1e-10)
        # An empirical acceleration is ~1e-9 km/s^2, well under np.allclose's
        # default atol, so the separation is measured against its own size.
        assert np.linalg.norm(between - now) > 0.1 * np.linalg.norm(now)

    def test_rtn_axes_are_radial_along_track_and_cross_track(self, origin, sc):
        """Each RTN coefficient drives its own basis direction."""
        fourier = scb.FourierAcceleration(origin, sc, J2000, degree=0, frame="RTN")

        r_hat = R0 / np.linalg.norm(R0)
        n_hat = np.cross(R0, V0) / np.linalg.norm(np.cross(R0, V0))
        t_hat = np.cross(n_hat, r_hat)

        for index, direction in enumerate((r_hat, t_hat, n_hat)):
            p = np.zeros(3)
            p[index] = 2e-9
            accel = fourier.compute_acceleration(R0, state_dict(p), EPOCH, sc)
            assert accel == pytest.approx(2e-9 * direction)

    def test_body_fixed_coefficients_rotate_with_the_body(self, origin, sc):
        """A body-fixed series turns in inertial space while keeping its magnitude."""
        fourier = scb.FourierAcceleration(
            origin, sc, J2000, degree=0, frame="body_fixed", body_fixed_frame="IAU_EARTH"
        )
        p = np.array([1e-9, 0.0, 0.0])

        now = fourier.compute_acceleration(R0, state_dict(p), 0.0, sc)
        quarter_day = fourier.compute_acceleration(R0, state_dict(p), 21600.0, sc)

        assert np.linalg.norm(now) == pytest.approx(1e-9, rel=1e-9)
        assert np.linalg.norm(quarter_day) == pytest.approx(1e-9, rel=1e-9)
        # Six hours of Earth rotation turns the vector by close to a right angle,
        # so the directions are compared rather than the raw 1e-9 components.
        cosine = now @ quarter_day / (np.linalg.norm(now) * np.linalg.norm(quarter_day))
        assert abs(cosine) < 0.05

    def test_argument_of_latitude_is_measured_from_the_ascending_node(self, origin, sc):
        """The phase is zero at the node and a quarter turn later at the pole side."""
        # Orbit in the x-z plane inclined 90 deg: the ascending node is +x.
        at_node = np.array([7000.0, 0.0, 0.0])
        northbound = np.array([0.0, 0.0, 7.5])
        fourier = model(origin, sc, degree=1, angle="argument_of_latitude")

        u_node = fourier._argument_of_latitude(
            at_node, state_dict(v=northbound), EPOCH, sc
        )[0]
        u_pole = fourier._argument_of_latitude(
            np.array([0.0, 0.0, 7000.0]),
            state_dict(v=np.array([-7.5, 0.0, 0.0])),
            EPOCH,
            sc,
        )[0]

        assert u_node == pytest.approx(0.0, abs=1e-12)
        assert u_pole == pytest.approx(np.pi / 2, rel=1e-12)

    def test_equatorial_orbit_has_no_argument_of_latitude(self, origin, sc):
        """An orbit with no ascending node is refused rather than silently wrong."""
        fourier = model(origin, sc, degree=1, angle="argument_of_latitude")
        equatorial = state_dict(coefficients(1), v=np.array([0.0, 7.5, 0.0]))

        with pytest.raises(
            scb.FourierAcceleration.DegenerateGeometryError, match="equatorial"
        ):
            fourier.compute_acceleration(
                np.array([7000.0, 0.0, 0.0]), equatorial, EPOCH, sc
            )

    @pytest.mark.parametrize(
        "frame, angle", [("RTN", "time"), ("inertial", "argument_of_latitude")]
    )
    def test_missing_velocity_is_reported(self, origin, sc, frame, angle):
        """A state-dependent frame or phase with no velocity is refused."""
        fourier = model(origin, sc, degree=1, frame=frame, angle=angle)
        without_velocity = {("a_fourier", SPICE_ID): coefficients(1)}

        with pytest.raises(scb.DynamicalUtils.MissingVelocityError):
            fourier.compute_acceleration(R0, without_velocity, EPOCH, sc)

    def test_state_coefficients_take_precedence_over_defaults(self, origin, sc):
        """The propagated state wins over the model's configured coefficients."""
        default = np.array([1e-9, 0.0, 0.0])
        estimated = np.array([0.0, 5e-9, 0.0])
        fourier = scb.FourierAcceleration(
            origin, sc, J2000, degree=0, coefficients=default
        )

        assert fourier.compute_acceleration(R0, state_dict(), EPOCH, sc) == pytest.approx(
            default
        )
        assert fourier.compute_acceleration(
            R0, state_dict(estimated), EPOCH, sc
        ) == pytest.approx(estimated)

    def test_no_coefficients_means_no_acceleration(self, origin, sc):
        """Without a state entry or a default the model contributes nothing."""
        fourier = scb.FourierAcceleration(origin, sc, J2000, degree=0)

        assert fourier.compute_acceleration(R0, state_dict(), EPOCH, sc) == pytest.approx(
            np.zeros(3)
        )


# --------------------------#
# region  Partials          #
# --------------------------#


class TestPartials:
    @pytest.mark.parametrize("degree", [0, 1, 3])
    @pytest.mark.parametrize("frame, angle", STATE_CASES)
    def test_coefficient_partial_matches_finite_differences(
        self, origin, sc, degree, frame, angle
    ):
        """The design matrix is checked against differencing the acceleration."""
        fourier = model(origin, sc, degree=degree, frame=frame, angle=angle)
        p = coefficients(degree)

        analytic = fourier._compute_partial_by_coefficients(
            R0, state_dict(p), EPOCH, sc
        )
        numeric = fd_jacobian(
            lambda q: fourier.compute_acceleration(R0, state_dict(q), EPOCH, sc),
            p,
            1e-12,
        )

        assert analytic.shape == (3, 3 * (2 * degree + 1))
        assert relative_error(analytic, numeric) < 1e-9

    @pytest.mark.parametrize("degree", [0, 1, 3])
    @pytest.mark.parametrize("frame, angle", STATE_CASES)
    def test_position_partial_matches_finite_differences(
        self, origin, sc, degree, frame, angle
    ):
        """da/dr is checked against differencing the acceleration in position."""
        fourier = model(origin, sc, degree=degree, frame=frame, angle=angle)
        p = coefficients(degree)

        analytic = fourier._compute_partial_by_position(R0, state_dict(p), EPOCH, sc)
        # 1 km: small against the 7000 km orbit that sets the RTN triad, and large
        # enough to move the Sun direction above round-off on its 1.5e8 km baseline.
        numeric = fd_jacobian(
            lambda r: fourier.compute_acceleration(r, state_dict(p, r=r), EPOCH, sc),
            R0,
            1.0,
        )

        assert analytic.shape == (3, 3)
        assert relative_error(analytic, numeric) < 1e-6

    @pytest.mark.parametrize("degree", [0, 1, 3])
    @pytest.mark.parametrize("frame, angle", STATE_CASES)
    def test_velocity_partial_matches_finite_differences(
        self, origin, sc, degree, frame, angle
    ):
        """da/dv is checked against differencing the acceleration in velocity."""
        fourier = model(origin, sc, degree=degree, frame=frame, angle=angle)
        p = coefficients(degree)

        analytic = fourier._compute_partial_by_velocity(R0, state_dict(p), EPOCH, sc)
        numeric = fd_jacobian(
            lambda v: fourier.compute_acceleration(R0, state_dict(p, v=v), EPOCH, sc),
            V0,
            1e-6,
        )

        assert analytic.shape == (3, 3)
        assert relative_error(analytic, numeric) < 1e-6

    @pytest.mark.parametrize("degree", [0, 2])
    def test_inertial_time_series_has_no_state_partials(self, origin, sc, degree):
        """The common configuration has exactly zero position and velocity blocks."""
        fourier = model(origin, sc, degree=degree)
        p = coefficients(degree)

        assert np.array_equal(
            fourier._compute_partial_by_position(R0, state_dict(p), EPOCH, sc),
            np.zeros((3, 3)),
        )
        assert np.array_equal(
            fourier._compute_partial_by_velocity(R0, state_dict(p), EPOCH, sc),
            np.zeros((3, 3)),
        )

    @pytest.mark.parametrize("frame, angle", STATE_CASES)
    def test_coefficient_partial_does_not_depend_on_the_coefficients(
        self, origin, sc, frame, angle
    ):
        """The acceleration is linear in the coefficients, so its partial is not."""
        fourier = model(origin, sc, degree=2, frame=frame, angle=angle)

        small = fourier._compute_partial_by_coefficients(
            R0, state_dict(coefficients(2, scale=1e-12)), EPOCH, sc
        )
        large = fourier._compute_partial_by_coefficients(
            R0, state_dict(coefficients(2, seed=11, scale=1e-5)), EPOCH, sc
        )

        assert small == pytest.approx(large, rel=1e-12)


# --------------------------#
# region  Force model       #
# --------------------------#


def fourier_force_model(sc, degree=1, frame="RTN", angle="argument_of_latitude",
                        t0=0.0, **kwargs):
    """ForceModelTranslation configured with a Fourier empirical acceleration."""
    return scb.ForceModelTranslation(
        primary_body=sc,
        t0=t0,
        fourier_acceleration=True,
        fourier_degree=degree,
        fourier_frame=frame,
        fourier_angle=angle,
        fourier_period=PERIOD,
        **kwargs,
    )


def fourier_state(origin, sc, degree=1, values=None, r=None, v=None):
    """State carrying the stacked Fourier coefficients as a static parameter."""
    definition = (
        scb.StateDefinition()
        .position(
            sc,
            scb.ArrayWFrame(
                np.array([7000.0, 0.0, 1000.0]) if r is None else r, km, J2000
            ),
        )
        .velocity(
            sc,
            scb.ArrayWFrame(
                np.array([0.0, 7.3, 0.5]) if v is None else v, km / sec, J2000
            ),
        )
    )
    if values is not None:
        definition = definition.param(
            "a_fourier",
            sc,
            scb.ArrayWFrame(np.asarray(values, dtype=float), km / sec**2, J2000),
            dynamics="static",
        )
    return scb.StateArray(
        epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
        origin=origin,
        state=definition,
    )


def propagate(origin, sc, force_model, values=None, r=None, v=None,
              t_end=900.0, points=4):
    """Propagate one short arc and return the final state and its STM."""
    state = fourier_state(origin, sc, values=values, r=r, v=v)
    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=state,
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
        """ForceModelTranslation resolves the a_fourier partial key to the model."""
        p = coefficients(1)
        force_model = fourier_force_model(sc)
        state = fourier_state(origin, sc, values=p)
        force_model.initialize_force_models(state, J2000, 0.0)

        partial = force_model._collect_partials(
            f"acceleration@{SPICE_ID}_by_a_fourier@{SPICE_ID}",
            scb.ArrayWUnits(R0, km),
            state_dict(p),
            EPOCH,
            sc,
        )

        assert np.asarray(partial).shape == (3, 9)
        direct = force_model.force_models[
            "FourierAcceleration"
        ]._compute_partial_by_coefficients(R0, state_dict(p), EPOCH, sc)
        assert np.asarray(partial) == pytest.approx(direct)

    @pytest.mark.parametrize(
        "frame, angle, expected",
        [
            ("inertial", "time", False),
            ("RTN", "time", True),
            ("inertial", "argument_of_latitude", True),
            ("RTN", "argument_of_latitude", True),
            ("inertial", "solar_longitude", True),
        ],
    )
    def test_velocity_block_is_requested_only_when_needed(self, sc, frame, angle, expected):
        """da/dv is raised for a velocity-dependent frame or phase, and only then."""
        force_model = fourier_force_model(sc, frame=frame, angle=angle)

        assert force_model._velocity_partial_active is expected

    def test_estimated_and_fixed_coefficients_agree(self, origin, sc):
        """Estimating the coefficients does not change the trajectory they produce."""
        p = coefficients(1, scale=1e-8)

        estimated, _ = propagate(origin, sc, fourier_force_model(sc), values=p)
        fixed, _ = propagate(
            origin, sc, fourier_force_model(sc, fourier_coefficients=p)
        )

        assert estimated == pytest.approx(fixed, abs=1e-9)

    def test_propagation_maps_coefficients_into_the_stm(self, origin, sc):
        """The STM's parameter block matches re-propagating with a shifted coefficient."""
        p = coefficients(1, scale=1e-8)
        nominal, stm = propagate(origin, sc, fourier_force_model(sc), values=p)

        assert stm.shape == (15, 15)

        step = 1e-11
        for index in (0, 4, 7):
            shifted = p.copy()
            shifted[index] += step
            moved, _ = propagate(origin, sc, fourier_force_model(sc), values=shifted)
            numeric = (moved - nominal) / step

            assert relative_error(stm[:6, 6 + index], numeric) < 1e-4

    def test_state_transition_matrix_matches_finite_differences(self, origin, sc):
        """The 6x6 block absorbs the RTN frame and phase partials."""
        # Large enough that the empirical term, not the central field, sets the
        # difference between the analytic and numerical blocks.
        p = np.array([3e-6, 1e-6, -2e-6, 8e-6, 1e-6, 3e-6, 2e-6, 1e-6, 1e-6])

        def run(x):
            r, v = np.split(np.asarray(x, dtype=float), 2)
            return propagate(
                origin, sc, fourier_force_model(sc, fourier_coefficients=p), r=r, v=v
            )

        x0 = np.concatenate((np.array([7000.0, 0.0, 1000.0]), np.array([0.0, 7.3, 0.5])))
        _, stm = run(x0)

        numeric = np.zeros((6, 6))
        for index in range(6):
            step = np.zeros(6)
            step[index] = 1e-4 if index < 3 else 1e-7
            plus, _ = run(x0 + step)
            minus, _ = run(x0 - step)
            numeric[:, index] = (plus - minus) / (2 * step[index])

        assert relative_error(stm[:6, :6], numeric) < 1e-5

    def test_configured_epoch_survives_leg_binding(self, origin, sc):
        """Binding to a leg does not move the phase origin."""
        force_model = fourier_force_model(sc, frame="inertial", angle="time", t0=100.0)
        state = fourier_state(origin, sc, values=coefficients(1))

        force_model.initialize_force_models(state, J2000, 4000.0)

        assert force_model.force_models["FourierAcceleration"].t0 == 100.0

    def test_serialization_round_trip(self, sc):
        """The Fourier settings survive to_dict / from_dict."""
        force_model = fourier_force_model(
            sc, degree=2, frame="RTN", angle="time", fourier_coefficients=coefficients(2)
        )

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc, t0=0.0)

        assert "FourierAcceleration" in restored.dynamics_mode_list
        assert restored.fourier_degree == 2
        assert restored.fourier_frame == "RTN"
        assert restored.fourier_angle == "time"
        assert restored.fourier_period == PERIOD

    def test_configured_epoch_survives_serialization(self, origin, sc):
        """A reloaded model keeps its phase origin instead of taking the leg's."""
        force_model = fourier_force_model(sc, frame="inertial", angle="time", t0=100.0)
        state = fourier_state(origin, sc, values=coefficients(1))

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), sc)
        restored.initialize_force_models(state, J2000, 4000.0)

        assert restored.force_models["FourierAcceleration"].t0 == 100.0


# --------------------------------#
# region  Solar longitude phase   #
# --------------------------------#
"""
McMahon and Scheeres expand the solar radiation pressure force as a Fourier
series in the *solar longitude* — the longitude of the Sun in the body-fixed
frame — so its frequency is the body's rotation relative to the Sun rather than
anything the caller supplies.

Tests
-----
test_solar_longitude_reproduces_the_body_frame_sun_direction
    lambda_s and delta_s rebuild the Sun unit vector exactly, which is
    McMahon Eq. (2.21).
test_solar_longitude_advances_with_the_body_rotation
    One body rotation is one cycle of the phase, per Eq. (2.26).
test_rtn_solar_longitude_is_the_sun_relative_in_plane_angle
    In RTN the phase is the CODE/Bern (u - u0), up to the documented sign.
test_solar_longitude_needs_no_period
    Nothing is supplied by the caller; the geometry sets the frequency.
test_sun_along_the_frame_pole_is_reported
    No longitude exists about an axis the Sun lies on.
"""


def sun_direction(fourier, position, epoch):
    """Unit vector from the spacecraft to the Sun, in the propagation frame."""
    return fourier._sun_direction(position, epoch)[0]


class TestSolarLongitudePhase:
    def test_solar_longitude_reproduces_the_body_frame_sun_direction(self, origin, sc):
        """lambda_s and delta_s rebuild the Sun unit vector exactly."""
        fourier = model(
            origin, sc, degree=1, frame="body_fixed", angle="solar_longitude",
            body_fixed_frame="IAU_EARTH",
        )

        lam = fourier._solar_longitude(R0, state_dict(), EPOCH, sc)[0]
        u_hat = sun_direction(fourier, R0, EPOCH)
        basis = scb.DynamicalUtils.frame_matrix(fourier, R0, state_dict(), EPOCH, sc)
        e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2]
        delta = np.arcsin(np.clip(u_hat @ e3, -1.0, 1.0))

        rebuilt = (
            np.cos(delta) * np.cos(lam) * e1
            + np.cos(delta) * np.sin(lam) * e2
            + np.sin(delta) * e3
        )

        assert rebuilt == pytest.approx(u_hat, abs=1e-12)

    def test_solar_longitude_advances_with_the_body_rotation(self, origin, sc):
        """One body rotation is one cycle of the phase."""
        fourier = model(
            origin, sc, degree=1, frame="body_fixed", angle="solar_longitude",
            body_fixed_frame="IAU_EARTH",
        )

        sidereal_day = 86164.0905
        epochs = EPOCH + np.linspace(0.0, sidereal_day, 9)
        phases = np.unwrap(
            [fourier._solar_longitude(R0, state_dict(), float(t), sc)[0] for t in epochs]
        )
        revolutions = (phases[-1] - phases[0]) / (-2.0 * np.pi)

        # Retrograde, because lambda_s = lambda_s0 - phi. Just short of a full turn:
        # the Sun itself moves about a degree a day.
        assert revolutions == pytest.approx(1.0, abs=0.01)

    def test_rtn_solar_longitude_is_the_sun_relative_in_plane_angle(self, origin, sc):
        """In RTN the phase is the CODE/Bern (u - u0), up to the documented sign."""
        solar = model(origin, sc, degree=1, frame="RTN", angle="solar_longitude")
        orbital = model(origin, sc, degree=1, frame="RTN", angle="argument_of_latitude")

        lam = solar._solar_longitude(R0, state_dict(), EPOCH, sc)[0]
        u = orbital._argument_of_latitude(R0, state_dict(), EPOCH, sc)[0]

        # Argument of latitude of the Sun, projected into the orbit plane.
        h_hat = np.cross(R0, V0) / np.linalg.norm(np.cross(R0, V0))
        sun_in_plane = sun_direction(solar, R0, EPOCH)
        sun_in_plane = sun_in_plane - (sun_in_plane @ h_hat) * h_hat
        sun_in_plane /= np.linalg.norm(sun_in_plane)
        node = np.cross(np.array([0.0, 0.0, 1.0]), h_hat)
        node /= np.linalg.norm(node)
        u_sun = np.arctan2(sun_in_plane @ np.cross(h_hat, node), sun_in_plane @ node)

        bern = np.arctan2(np.sin(u - u_sun), np.cos(u - u_sun))

        assert lam == pytest.approx(-bern, abs=1e-9)

    def test_solar_longitude_needs_no_period(self, origin, sc):
        """Nothing is supplied by the caller; the geometry sets the frequency."""
        fourier = scb.FourierAcceleration(
            origin, sc, J2000, degree=2, frame="RTN", angle="solar_longitude"
        )

        assert fourier.period is None
        assert fourier.compute_acceleration(
            R0, state_dict(coefficients(2)), EPOCH, sc
        ).shape == (3,)

    def test_sun_along_the_frame_pole_is_reported(self, origin, sc):
        """No longitude exists about an axis the Sun lies on."""
        fourier = model(origin, sc, degree=1, frame="RTN", angle="solar_longitude")

        # An orbit whose cross-track axis points at the Sun. The Sun direction is
        # itself a function of position, so the radius is iterated until it is
        # perpendicular to the direction seen *from that radius*.
        r = R0
        for _ in range(4):
            u_hat = sun_direction(fourier, r, EPOCH)
            r = np.cross(u_hat, np.array([0.0, 0.0, 1.0]))
            r = 7000.0 * r / np.linalg.norm(r)
        u_hat = sun_direction(fourier, r, EPOCH)
        v = np.cross(u_hat, r)
        v = 7.5 * v / np.linalg.norm(v)

        # The construction has to actually be degenerate for the guard to be the
        # thing under test.
        h_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
        assert abs(abs(h_hat @ u_hat) - 1.0) < 1e-12

        with pytest.raises(
            scb.FourierAcceleration.DegenerateGeometryError, match="solar latitude"
        ):
            fourier.compute_acceleration(r, state_dict(coefficients(1), v=v), EPOCH, sc)


# --------------------------------#
# region  RTN about another body #
# --------------------------------#
"""
The RTN triad and the orbital phase angles can be referred to a body other
than the propagation origin — a lunar RTN frame inside an Earth-centred arc.
Both move together: they describe the same orbit.

Tests
-----
test_relative_body_moves_the_rtn_triad
    The triad follows the chosen body, not the propagation origin.
test_relative_body_anchors_the_orbital_phase
    The argument of latitude is measured about the same body.
test_relative_body_partials_match_finite_differences
    The Jacobians are unchanged in form, since the offset is an ephemeris
    quantity and drops out of the derivative.
"""


class TestRTNAboutAnotherBody:
    def test_relative_body_moves_the_rtn_triad(self, origin, sc):
        """The triad follows the chosen body, not the propagation origin."""
        about_origin = scb.FourierAcceleration(
            origin, sc, J2000, degree=0, frame="RTN"
        )
        about_moon = scb.FourierAcceleration(
            origin, sc, J2000, degree=0, frame="RTN", relative_body="MOON"
        )
        radial_only = np.array([1e-9, 0.0, 0.0])

        a_origin = about_origin.compute_acceleration(
            R0, state_dict(radial_only), EPOCH, sc
        )
        a_moon = about_moon.compute_acceleration(R0, state_dict(radial_only), EPOCH, sc)

        # Both are purely radial, but about different centres.
        assert np.linalg.norm(a_origin) == pytest.approx(1e-9, rel=1e-9)
        assert np.linalg.norm(a_moon) == pytest.approx(1e-9, rel=1e-9)

        moon = scb.SpiceManager.get_state("MOON", EPOCH, J2000, origin.name)
        offset = np.asarray(moon.values, dtype=float).ravel()
        expected = (R0 - offset[:3]) / np.linalg.norm(R0 - offset[:3])

        assert a_moon / 1e-9 == pytest.approx(expected, abs=1e-9)
        assert not np.allclose(a_origin, a_moon, atol=0.0, rtol=1e-3)

    def test_relative_body_anchors_the_orbital_phase(self, origin, sc):
        """The argument of latitude is measured about the same body."""
        about_origin = model(origin, sc, degree=1, angle="argument_of_latitude")
        about_moon = model(
            origin, sc, degree=1, angle="argument_of_latitude", relative_body="MOON"
        )

        u_origin = about_origin._argument_of_latitude(R0, state_dict(), EPOCH, sc)[0]
        u_moon = about_moon._argument_of_latitude(R0, state_dict(), EPOCH, sc)[0]

        assert not np.isclose(u_origin, u_moon)

    @pytest.mark.parametrize(
        "frame, angle",
        [("RTN", "time"), ("RTN", "argument_of_latitude"), ("RTN", "solar_longitude")],
    )
    def test_relative_body_partials_match_finite_differences(self, origin, sc, frame, angle):
        """The Jacobians hold when the triad is referred to another body."""
        fourier = model(
            origin, sc, degree=1, frame=frame, angle=angle, relative_body="MOON"
        )
        p = coefficients(1)

        for partial, wrt, step, arg in (
            (fourier._compute_partial_by_position, R0, 1e-2, "r"),
            (fourier._compute_partial_by_velocity, V0, 1e-6, "v"),
        ):
            analytic = partial(R0, state_dict(p), EPOCH, sc)
            numeric = fd_jacobian(
                lambda x: fourier.compute_acceleration(
                    x if arg == "r" else R0,
                    state_dict(p, **{arg: x}),
                    EPOCH,
                    sc,
                ),
                wrt,
                step,
            )
            assert relative_error(analytic, numeric) < 1e-5
