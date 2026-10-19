# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =====================================================#
#  Unit Tests for the AntennaRadiationPressure class  #
# =====================================================#
import json

import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.dynamics.AntennaRadiationPressure import AntennaRadiationPressure
import pytest
import numpy as np

km, kg, sec, m, deg = scb.Units.get_units(["km", "kg", "sec", "m", "deg"])
W = kg * m**2 * sec**-3
J2000 = scb.Frame("J2000")

C_MPS = 299792458.0
MASS = 1390.0
SC_ID = -64
EPOCH = 7.0e8  # TDB seconds past J2000

# Heliocentric state near 0.97 AU.
R0 = np.array([1.2e8, 8.0e7, 1.0e7])
V0 = np.array([-15.0, 24.0, 1.0])


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def sun():
    """Heliocentric origin: the Sun-spacecraft vector is the position itself."""
    return scb.CelestialBody.from_constants("Sun")


def fixed_sc(dcm=None):
    """Spacecraft held at a fixed inertial attitude (identity by default)."""
    return scb.Spacecraft(
        "ORX", SC_ID, tot_mass=awu(MASS, kg), attitudeMode="inertialFixedAttitude",
        fixedInertialAttitude=np.eye(3) if dcm is None else dcm,
    )


def nadir_sc():
    """Spacecraft with -X at the Sun and Y along the heliocentric velocity."""
    return scb.Spacecraft(
        "ORX", SC_ID, tot_mass=awu(MASS, kg), attitudeMode="nadirPointingSun"
    )


def emitter(name="HGA", power=100.0, hpbw=1.02, boresight=(0.0, 0.0, 1.0), **extra):
    return {"name": name, "power": awu(power, W), "hpbw": awu(hpbw, deg),
            "boresight": list(boresight), **extra}


def osiris_rex_emitters():
    """High- and low-gain antennas on opposite faces, as on OSIRIS-REx."""
    return [
        emitter("HGA", 100.0, 1.02, (0.0, 0.0, 1.0)),
        emitter("LGA", 60.0, 70.6, (0.2, -0.3, -1.0)),
    ]


def state(eta=None, velocity=V0):
    st = {("velocity", SC_ID): velocity}
    if eta is not None:
        st[("eta_antenna", SC_ID)] = eta
    return st


def recoil(power, hpbw_deg):
    """Recoil acceleration magnitude of one emitter, km/s^2."""
    return power * np.cos(np.radians(hpbw_deg) / 4.0) ** 2 / (MASS * C_MPS) * 1e-3


def jacobian_fd(f, x, h):
    """Central-difference Jacobian of ``f`` at ``x``."""
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (f(x + e) - f(x - e)) / (2 * h)
    return J


def random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestAcceleration:
    """The recoil of each emitter against the paper's formula."""

    def test_single_emitter_matches_the_formula(self, sun):
        """-P cos^2(hpbw/4) / (m c) along the boresight: about 2.4e-13 km/s^2 for 100 W."""
        sc = fixed_sc()
        model = AntennaRadiationPressure(sun, sc, J2000, [emitter()])
        acc = model.compute_acceleration(R0, state(), EPOCH, sc)

        np.testing.assert_allclose(acc, [0.0, 0.0, -recoil(100.0, 1.02)], rtol=1e-14)
        assert acc[2] == pytest.approx(-100.0 / (MASS * C_MPS) * 1e-3, rel=1e-4)

    def test_wide_beam_loses_about_nine_percent(self, sun):
        """A 70.6 degree low-gain beam keeps cos^2(17.65 deg) = 0.908 of its momentum."""
        sc = fixed_sc()
        wide = AntennaRadiationPressure(sun, sc, J2000, [emitter(hpbw=70.6)])
        narrow = AntennaRadiationPressure(sun, sc, J2000, [emitter(hpbw=0.0)])
        ratio = (wide.compute_acceleration(R0, state(), EPOCH, sc)[2]
                 / narrow.compute_acceleration(R0, state(), EPOCH, sc)[2])
        assert ratio == pytest.approx(0.9081, abs=1e-4)

    def test_emitters_add_and_turn_with_the_attitude(self, sun):
        """Boresights are unit vectors in the spacecraft frame, rotated to inertial."""
        dcm = random_rotation(5)
        model = AntennaRadiationPressure(sun, fixed_sc(dcm), J2000, osiris_rex_emitters())
        acc = model.compute_acceleration(R0, state(), EPOCH, model.spacecraft)

        lga = np.array([0.2, -0.3, -1.0]) / np.linalg.norm([0.2, -0.3, -1.0])
        body = -(recoil(100.0, 1.02) * np.array([0.0, 0.0, 1.0]) + recoil(60.0, 70.6) * lga)
        np.testing.assert_allclose(acc, dcm @ body, rtol=1e-13)

    def test_scale_factor(self, sun):
        """eta scales the acceleration and its partials; the eta partial is the eta = 1 value."""
        sc = nadir_sc()
        model = AntennaRadiationPressure(sun, sc, J2000, osiris_rex_emitters())
        unscaled = model.compute_acceleration(R0, state(), EPOCH, sc)
        J_unscaled = model._compute_partial_by_velocity(R0, state(), EPOCH, sc)

        np.testing.assert_allclose(
            model.compute_acceleration(R0, state(eta=0.5), EPOCH, sc), 0.5 * unscaled, rtol=1e-14
        )
        np.testing.assert_allclose(
            model._compute_partial_by_velocity(R0, state(eta=0.5), EPOCH, sc),
            0.5 * J_unscaled, rtol=1e-14,
        )
        for eta in (0.0, 0.5, 2.0):
            np.testing.assert_allclose(
                model._compute_partial_by_eta(R0, state(eta=eta), EPOCH, sc), unscaled, rtol=1e-14
            )


class TestSchedule:
    """Emitters transmit only inside their on-intervals."""

    def test_intervals_are_closed_at_start_and_open_at_end(self, sun):
        start, end = EPOCH - 100.0, EPOCH + 100.0
        emitters = [
            emitter("ALWAYS"),
            emitter("WINDOW", on_intervals=[(start, end)]),
            emitter("NEVER", on_intervals=[]),
        ]
        model = AntennaRadiationPressure(sun, fixed_sc(), J2000, emitters)

        assert model.emitter_names == ["ALWAYS", "WINDOW", "NEVER"]
        np.testing.assert_array_equal(model.active_emitters(start - 1.0), [True, False, False])
        np.testing.assert_array_equal(model.active_emitters(start), [True, True, False])
        np.testing.assert_array_equal(model.active_emitters(end), [True, False, False])

    def test_epoch_array_bounds(self, sun):
        """Interval bounds may be EpochArrays; they are stored as TDB seconds."""
        start = scb.EpochArray("2022 MAR 07 12:00:00.000", "UTC")
        end = scb.EpochArray("2022 MAR 07 13:00:00.000", "UTC")
        model = AntennaRadiationPressure(
            sun, fixed_sc(), J2000, [emitter(on_intervals=[(start, end)])]
        )
        t0, t1 = model.emitters[0]["on_intervals"][0]
        assert t1 - t0 == pytest.approx(3600.0, abs=1e-6)
        assert model.active_emitters(t0)[0] and not model.active_emitters(t0 - 1e-3)[0]

    @pytest.mark.parametrize("make_sc", [fixed_sc, nadir_sc], ids=["fixed", "nadir"])
    def test_silent_spacecraft_feels_nothing(self, sun, make_sc):
        """With every emitter off the acceleration and all partials are zero."""
        sc = make_sc()
        model = AntennaRadiationPressure(
            sun, sc, J2000, [emitter(on_intervals=[(EPOCH + 1.0, EPOCH + 2.0)])]
        )
        for method in ("compute_acceleration", "_compute_partial_by_position",
                       "_compute_partial_by_velocity", "_compute_partial_by_eta"):
            assert not np.any(getattr(model, method)(R0, state(), EPOCH, sc))


class TestPartials:
    """Analytical partials against central differences of the acceleration."""

    def test_fixed_attitude_has_no_state_partials(self, sun):
        sc = fixed_sc(random_rotation(5))
        model = AntennaRadiationPressure(sun, sc, J2000, osiris_rex_emitters())
        assert not model._compute_partial_by_position(R0, state(), EPOCH, sc).any()
        assert not model._compute_partial_by_velocity(R0, state(), EPOCH, sc).any()

    def test_nadir_attitude(self, sun):
        """The boresights turn with the Sun direction and the heliocentric velocity."""
        sc = nadir_sc()
        model = AntennaRadiationPressure(sun, sc, J2000, osiris_rex_emitters())
        acc_r = lambda r: model.compute_acceleration(r, state(), EPOCH, sc)  # noqa: E731
        acc_v = lambda v: model.compute_acceleration(R0, state(velocity=v), EPOCH, sc)  # noqa: E731

        for J, J_fd in (
            (model._compute_partial_by_position(R0, state(), EPOCH, sc), jacobian_fd(acc_r, R0, 100.0)),
            (model._compute_partial_by_velocity(R0, state(), EPOCH, sc), jacobian_fd(acc_v, V0, 1e-3)),
        ):
            assert np.abs(J_fd).max() > 0.0
            np.testing.assert_allclose(J, J_fd, rtol=0.0, atol=1e-6 * np.abs(J_fd).max())


class TestDefinition:
    """Properties, validation and serialization of the emitters."""

    @pytest.mark.parametrize("make_sc, expected", [(fixed_sc, False), (nadir_sc, True)],
                             ids=["fixed", "nadir"])
    def test_properties(self, sun, make_sc, expected):
        model = AntennaRadiationPressure(sun, make_sc(), J2000, osiris_rex_emitters())
        assert model.param_name == "eta_antenna" == AntennaRadiationPressure.PARAMETER_NAME
        assert model.depends_on_velocity is expected
        assert model.emitter_names == ["HGA", "LGA"]
        assert model.origin_name == "Sun"

    def test_json_round_trip(self, sun):
        emitters = osiris_rex_emitters()
        emitters[1]["on_intervals"] = [(EPOCH - 10.0, EPOCH + 10.0), (EPOCH + 50.0, EPOCH + 60.0)]
        written = AntennaRadiationPressure.emitters_to_dict(emitters)
        read_back = json.loads(json.dumps(written))

        sc = nadir_sc()
        original = AntennaRadiationPressure(sun, sc, J2000, emitters)
        loaded = AntennaRadiationPressure(sun, sc, J2000, read_back)
        assert [e["name"] for e in loaded.to_dict()] == ["HGA", "LGA"]
        assert loaded.to_dict()[1]["on_intervals"] == written[1]["on_intervals"]
        for epoch in (EPOCH, EPOCH + 30.0, EPOCH + 55.0):
            np.testing.assert_allclose(
                loaded.compute_acceleration(R0, state(), epoch, sc),
                original.compute_acceleration(R0, state(), epoch, sc),
                rtol=1e-14,
            )

    @pytest.mark.parametrize(
        "change",
        [
            {"name": None},
            {"power": 100.0},
            {"power": awu(-1.0, W)},
            {"power": awu(100.0, m)},
            {"power": awu(np.array([1.0, 2.0]), W)},
            {"hpbw": awu(360.0, deg)},
            {"hpbw": awu(-1.0, deg)},
            {"boresight": [0.0, 0.0, 0.0]},
            {"boresight": [1.0, 0.0]},
            {"on_intervals": [(EPOCH + 1.0, EPOCH)]},
            {"on_intervals": [(EPOCH, EPOCH)]},
            {"on_intervals": [EPOCH]},
        ],
        ids=[
            "name-missing", "power-no-units", "power-negative", "power-length",
            "power-two-values", "hpbw-360", "hpbw-negative", "boresight-zero",
            "boresight-2d", "interval-reversed", "interval-empty", "interval-not-pair",
        ],
    )
    def test_malformed_emitter_is_refused(self, sun, change):
        bad = emitter()
        for key, value in change.items():
            if value is None:
                del bad[key]
            else:
                bad[key] = value
        with pytest.raises(AntennaRadiationPressure.InvalidEmitterError):
            AntennaRadiationPressure(sun, fixed_sc(), J2000, [emitter("OK"), bad])

    @pytest.mark.parametrize("emitters", [[], emitter(), None], ids=["empty", "dict", "none"])
    def test_emitters_must_be_a_list(self, sun, emitters):
        with pytest.raises(AntennaRadiationPressure.InvalidEmitterError):
            AntennaRadiationPressure(sun, fixed_sc(), J2000, emitters)


# endregion Tests #
# --------------#
