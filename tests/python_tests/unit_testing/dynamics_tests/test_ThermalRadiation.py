# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the ThermalRadiation class  #
# =============================================#
import json

import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.dynamics.ThermalRadiation import ThermalRadiation
from scipy.interpolate import CubicSpline
import pytest
import numpy as np

km, kg, sec, m, deg, AU = scb.Units.get_units(["km", "kg", "sec", "m", "deg", "AU"])
J2000 = scb.Frame("J2000")

AU_KM = 149597870.7
SIGMA = 5.670374419e-8  # W / (m^2 K^4), CODATA 2018
C_MPS = 299792458.0
MASS = 1390.0
SC_ID = -64
EPOCH = 7.0e8  # TDB seconds past J2000, inside de430


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


def constant_table(T, units="K", **extra):
    """A table at one temperature for every angle, at 1 AU."""
    return {
        "angles": awu(np.array([0.0, 180.0]), deg),
        "distances": awu(np.array([1.0]), AU),
        "temperatures": np.array([T, T]),
        "temperature_units": units,
        **extra,
    }


def surface(name, normal, table, area=2.0, emissivity=0.8, shadow=None):
    """One surface; ``shadow`` names the table used when it faces away from the Sun."""
    s = {"name": name, "area": awu(area, m**2), "emissivity": emissivity,
         "normal": normal, "temperature": table}
    if shadow is not None:
        s["shadow_temperature"] = shadow
    return s


def varying_model():
    """
    Tables that change with both angle and distance, a lit/shadow pair among
    them, so every term of the partials is exercised.
    """
    angles = np.arange(0.0, 181.0, 30.0)
    dists = np.array([0.8, 0.9, 1.1, 1.3])
    temps = np.array([[250.0 + 100.0 * np.cos(np.radians(a)) + 20.0 * np.sin(np.radians(2 * a))
                       - 60.0 * (d - 1.0) for a in angles] for d in dists])
    table = lambda offset, **extra: {  # noqa: E731
        "angles": awu(angles, deg), "distances": awu(dists, AU),
        "temperatures": temps + offset, "temperature_units": "K", **extra,
    }
    return {
        "temperature_tables": {
            "BUS": table(0.0),
            "HGA_LIT": table(40.0, angle_reference=[0.0, 0.0, 1.0]),
            "HGA_DARK": table(-120.0, angle_reference=[0.0, 0.0, 1.0]),
        },
        "surfaces": [
            surface("PX", [1.0, 0.0, 0.0], "BUS", area=4.0),
            surface("MX", [-1.0, 0.2, 0.0], "BUS", area=4.0),
            surface("PY", [0.1, 1.0, -0.3], "BUS", area=3.0, emissivity=0.6),
            surface("HGA1", [0.3, 0.1, 1.0], "HGA_LIT", area=0.5, emissivity=0.9, shadow="HGA_DARK"),
            surface("HGA2", [-0.3, -0.5, 1.0], "HGA_LIT", area=0.5, emissivity=0.9, shadow="HGA_DARK"),
        ],
    }


def one_surface_model(T=300.0, units="K", normal=(1.0, 0.0, 0.0)):
    return {"temperature_tables": {"T": constant_table(T, units)},
            "surfaces": [surface("S", list(normal), "T")]}


# Heliocentric state near 0.97 AU, away from every distance knot of varying_model.
R0 = np.array([1.2e8, 8.0e7, 1.0e7])
V0 = np.array([-15.0, 24.0, 1.0])


def state(eta=None, velocity=V0):
    st = {("velocity", SC_ID): velocity}
    if eta is not None:
        st[("eta_thermal", SC_ID)] = eta
    return st


def jacobian_fd(f, x, h):
    """Central-difference Jacobian of ``f`` at ``x``."""
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (f(x + e) - f(x - e)) / (2 * h)
    return J


def assert_same_definition(a, b):
    """Nested dicts and lists equal, numbers to rounding: loading re-normalizes vectors."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            assert_same_definition(a[key], b[key])
    elif isinstance(a, list) and a and isinstance(a[0], (dict, str)):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_same_definition(x, y)
    elif isinstance(a, (list, float)):
        np.testing.assert_allclose(a, b, rtol=1e-15)
    else:
        assert a == b


def random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestAcceleration:
    """The recoil of each surface against the paper's formula."""

    def test_single_surface_matches_the_formula(self, sun):
        """-(2/3) sigma A eps T^4 / (m c) along the normal, in km/s^2."""
        normal = np.array([0.6, 0.0, 0.8])
        model = ThermalRadiation(sun, fixed_sc(), J2000, one_surface_model(normal=normal))
        acc = model.compute_acceleration(R0, state(), EPOCH, model.spacecraft)

        expected = -(2 / 3) * SIGMA * 2.0 * 0.8 * 300.0**4 / (MASS * C_MPS) * normal * 1e-3
        np.testing.assert_allclose(acc, expected, rtol=1e-12)

    def test_celsius_table_is_offset_to_kelvin(self, sun):
        """26.85 degC is 300 K."""
        sc = fixed_sc()
        kelvin = ThermalRadiation(sun, sc, J2000, one_surface_model(300.0, "K"))
        celsius = ThermalRadiation(sun, sc, J2000, one_surface_model(26.85, "degC"))
        np.testing.assert_allclose(
            celsius.surface_temperatures(R0, state(), EPOCH, sc), [300.0], rtol=1e-12
        )
        np.testing.assert_allclose(
            celsius.compute_acceleration(R0, state(), EPOCH, sc),
            kelvin.compute_acceleration(R0, state(), EPOCH, sc),
            rtol=1e-12,
        )

    def test_scale_factor(self, sun):
        """eta scales the acceleration and its partials; the eta partial is the eta = 1 value."""
        sc = nadir_sc()
        model = ThermalRadiation(sun, sc, J2000, varying_model())
        unscaled = model.compute_acceleration(R0, state(), EPOCH, sc)
        J_unscaled = model._compute_partial_by_position(R0, state(), EPOCH, sc)

        np.testing.assert_allclose(
            model.compute_acceleration(R0, state(eta=0.5), EPOCH, sc), 0.5 * unscaled, rtol=1e-14
        )
        np.testing.assert_allclose(
            model._compute_partial_by_position(R0, state(eta=0.5), EPOCH, sc),
            0.5 * J_unscaled, rtol=1e-14,
        )
        for eta in (0.0, 0.5, 2.0):
            np.testing.assert_allclose(
                model._compute_partial_by_eta(R0, state(eta=eta), EPOCH, sc), unscaled, rtol=1e-14
            )

    def test_another_spacecraft_scale_factor_is_ignored(self, sun):
        sc = fixed_sc()
        model = ThermalRadiation(sun, sc, J2000, varying_model())
        other = {**state(), ("eta_thermal", SC_ID - 1): 0.0}
        np.testing.assert_array_equal(
            model.compute_acceleration(R0, other, EPOCH, sc),
            model.compute_acceleration(R0, state(), EPOCH, sc),
        )

    def test_earth_origin_matches_heliocentric(self, sun):
        """
        The Sun geometry read from SPICE, or from the ephemeris cache when the
        span is given, gives the acceleration a heliocentric origin gives.
        """
        earth = scb.CelestialBody.from_constants("Earth")
        sc = fixed_sc(random_rotation(7))
        r_earth = np.array([7000.0, 1200.0, -300.0])
        sun2earth = scb.SpiceManager.get_pos_values(
            scb.constants.EARTH.spice_name, EPOCH, J2000, "SUN"
        )

        helio = ThermalRadiation(sun, sc, J2000, varying_model())
        expected = helio.compute_acceleration(sun2earth + r_earth, state(), EPOCH, sc)
        for t0, tf in ((None, None), (EPOCH - 3600.0, EPOCH + 3600.0)):
            geo = ThermalRadiation(earth, sc, J2000, varying_model(), t0=t0, tf=tf)
            np.testing.assert_allclose(
                geo.compute_acceleration(r_earth, state(), EPOCH, sc), expected, rtol=1e-9
            )


class TestTemperatures:
    """Which table a surface reads, and where in it."""

    # Heliocentric position on +X with velocity on +Y: the nadir attitude is the
    # identity, so both attitudes below put the Sun on body -X.
    R_ON_X = np.array([AU_KM, 0.0, 0.0])
    V_ON_Y = np.array([0.0, 30.0, 0.0])

    @pytest.mark.parametrize("make_sc", [fixed_sc, nadir_sc], ids=["fixed", "nadir"])
    def test_lit_and_shadow_tables(self, sun, make_sc):
        """
        The face toward the Sun reads the lit table, the one away the shadow
        table, and the hotter Sun face pushes the spacecraft away from the Sun.
        """
        model_def = {
            "temperature_tables": {"HOT": constant_table(350.0), "COLD": constant_table(150.0)},
            "surfaces": [
                surface("SUNWARD", [-1.0, 0.0, 0.0], "HOT", shadow="COLD"),
                surface("ANTISUN", [1.0, 0.0, 0.0], "HOT", shadow="COLD"),
            ],
        }
        sc = make_sc()
        model = ThermalRadiation(sun, sc, J2000, model_def)
        st = state(velocity=self.V_ON_Y)

        np.testing.assert_allclose(
            model.surface_temperatures(self.R_ON_X, st, EPOCH, sc), [350.0, 150.0], rtol=1e-12
        )
        acc = model.compute_acceleration(self.R_ON_X, st, EPOCH, sc)
        assert acc[0] > 0.0
        np.testing.assert_allclose(acc[1:], 0.0, atol=1e-14 * acc[0])

    def test_spline_hits_the_knots_and_matches_scipy_between(self, sun):
        """
        Each surface's off-Sun angle is measured from its normal; the table
        is a spline with zero slope at 0 and 180 degrees.
        """
        knots = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        temps = np.array([400.0, 360.0, 250.0, 180.0, 170.0])
        probe = np.array([0.0, 22.5, 45.0, 90.0, 112.0, 135.0, 180.0])
        model_def = {
            "temperature_tables": {"T": {
                "angles": awu(knots, deg), "distances": awu(np.array([1.0]), AU),
                "temperatures": temps, "temperature_units": "K",
            }},
            # Normal at angle a from the Sun direction, body -X.
            "surfaces": [
                surface(f"A{a:g}", [-np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0], "T")
                for a in probe
            ],
        }
        sc = nadir_sc()
        model = ThermalRadiation(sun, sc, J2000, model_def)
        T = model.surface_temperatures(self.R_ON_X, state(velocity=self.V_ON_Y), EPOCH, sc)

        clamped = CubicSpline(np.radians(knots), temps, bc_type=((1, 0.0), (1, 0.0)))
        np.testing.assert_allclose(T, clamped(np.radians(probe)), rtol=1e-9)
        np.testing.assert_allclose(T[[0, 2, 3, 5, 6]], temps, rtol=1e-9)

    def test_angle_reference_replaces_the_normal(self, sun):
        """With an angle reference every surface reads the table at the same angle."""
        model_def = {
            "temperature_tables": {"T": {
                "angles": awu(np.array([0.0, 90.0, 180.0]), deg),
                "distances": awu(np.array([1.0]), AU),
                "temperatures": np.array([300.0, 200.0, 100.0]),
                "temperature_units": "K",
                "angle_reference": [0.0, 1.0, 0.0],
            }},
            "surfaces": [surface("PX", [1.0, 0.0, 0.0], "T"), surface("MX", [-1.0, 0.0, 0.0], "T")],
        }
        sc = fixed_sc()
        model = ThermalRadiation(sun, sc, J2000, model_def)
        np.testing.assert_allclose(
            model.surface_temperatures(self.R_ON_X, state(), EPOCH, sc), [200.0, 200.0], rtol=1e-9
        )

    @pytest.mark.parametrize("dist_au, expected", [(1.0, 300.0), (0.75, 350.0), (0.25, 400.0), (3.0, 200.0)],
                             ids=["mid", "quarter", "clamped-near", "clamped-far"])
    def test_linear_in_distance_and_held_at_the_edges(self, sun, dist_au, expected):
        model_def = {
            "temperature_tables": {"T": {
                "angles": awu(np.array([0.0, 180.0]), deg),
                "distances": awu(np.array([0.5, 1.5]), AU),
                "temperatures": np.array([[400.0, 400.0], [200.0, 200.0]]),
                "temperature_units": "K",
            }},
            "surfaces": [surface("S", [-1.0, 0.0, 0.0], "T")],
        }
        sc = fixed_sc()
        model = ThermalRadiation(sun, sc, J2000, model_def)
        T = model.surface_temperatures(np.array([dist_au * AU_KM, 0.0, 0.0]), state(), EPOCH, sc)
        np.testing.assert_allclose(T, [expected], rtol=1e-12)


class TestPartials:
    """Analytical partials against central differences of the acceleration."""

    @staticmethod
    def check(model, sc):
        acc_r = lambda r: model.compute_acceleration(r, state(), EPOCH, sc)  # noqa: E731
        acc_v = lambda v: model.compute_acceleration(R0, state(velocity=v), EPOCH, sc)  # noqa: E731

        J_r = model._compute_partial_by_position(R0, state(), EPOCH, sc)
        J_r_fd = jacobian_fd(acc_r, R0, 100.0)
        assert np.abs(J_r_fd).max() > 0.0
        np.testing.assert_allclose(J_r, J_r_fd, rtol=0.0, atol=1e-6 * np.abs(J_r_fd).max())

        J_v = model._compute_partial_by_velocity(R0, state(), EPOCH, sc)
        J_v_fd = jacobian_fd(acc_v, V0, 1e-3)
        scale = max(np.abs(J_v_fd).max(), 1e-30)
        np.testing.assert_allclose(J_v, J_v_fd, rtol=0.0, atol=1e-6 * scale)
        return J_v

    def test_fixed_attitude(self, sun):
        """Temperatures change with distance and angle; the velocity partial is zero."""
        dcm = random_rotation(11)
        sc = fixed_sc(dcm)
        model = ThermalRadiation(sun, sc, J2000, varying_model())

        # Keep every plate clear of the terminator, where the tables switch.
        u_sun = -R0 / np.linalg.norm(R0)
        normals_b = np.array([s["normal"] for s in model.thermal_model["surfaces"]])
        assert np.abs(normals_b @ dcm.T @ u_sun).min() > 0.1

        J_v = self.check(model, sc)
        assert not J_v.any()

    def test_nadir_attitude(self, sun):
        """
        The normals turn with the Sun direction and the velocity. The Sun stays
        on body -X, so every off-Sun angle is constant: the angle term of dT/dr
        and the turning of its reference cancel, and only the distance and the
        normal rotation remain.
        """
        sc = nadir_sc()
        model = ThermalRadiation(sun, sc, J2000, varying_model())
        J_v = self.check(model, sc)
        assert np.abs(J_v).max() > 0.0


class TestDefinition:
    """Properties, validation and serialization of the thermal model."""

    @pytest.mark.parametrize("make_sc, expected", [(fixed_sc, False), (nadir_sc, True)],
                             ids=["fixed", "nadir"])
    def test_properties(self, sun, make_sc, expected):
        model = ThermalRadiation(sun, make_sc(), J2000, varying_model())
        assert model.param_name == "eta_thermal" == ThermalRadiation.PARAMETER_NAME
        assert model.depends_on_velocity is expected
        assert model.surface_names == ["PX", "MX", "PY", "HGA1", "HGA2"]
        assert model.origin_name == "Sun"

    def test_json_file_round_trip(self, sun, tmp_path):
        """What model_to_dict writes, a file path reads back to the same model."""
        written = ThermalRadiation.model_to_dict(varying_model())
        path = tmp_path / "thermal.json"
        path.write_text(json.dumps(written))

        sc = nadir_sc()
        original = ThermalRadiation(sun, sc, J2000, varying_model())
        for source in (path, str(path)):
            loaded = ThermalRadiation(sun, sc, J2000, source)
            assert_same_definition(loaded.to_dict(), written)
            np.testing.assert_allclose(
                loaded.compute_acceleration(R0, state(), EPOCH, sc),
                original.compute_acceleration(R0, state(), EPOCH, sc),
                rtol=1e-14,
            )

    def _set(path, value):
        """Mutator that sets ``model[path...] = value``."""
        def mutate(model):
            target = model
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        return mutate

    def _drop(path):
        """Mutator that deletes ``model[path...]``."""
        def mutate(model):
            target = model
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
        return mutate

    Model = ThermalRadiation.InvalidThermalModelError
    Table = ThermalRadiation.InvalidTemperatureTableError
    Surface = ThermalRadiation.InvalidSurfaceError
    Unknown = ThermalRadiation.UnknownTemperatureTableError

    @pytest.mark.parametrize(
        "mutate, error",
        [
            (_drop(["surfaces"]), Model),
            (_set(["temperature_tables"], {}), Model),
            (_set(["surfaces"], []), Model),
            (_drop(["temperature_tables", "BUS", "distances"]), Table),
            (_set(["temperature_tables", "BUS", "angles"], [0.0, 90.0]), Table),
            (_set(["temperature_tables", "BUS", "angles"], awu(np.linspace(0, 1, 7), km)), Table),
            (_set(["temperature_tables", "BUS", "angles"], awu(np.linspace(180, 0, 7), deg)), Table),
            (_set(["temperature_tables", "BUS", "angles"], awu(np.linspace(0, 200, 7), deg)), Table),
            (_set(["temperature_tables", "BUS", "distances"], awu(np.array([0.8, 0.9, 1.1, 1.3]), sec)), Table),
            (_set(["temperature_tables", "BUS", "distances"], awu(np.array([0.0, 0.9, 1.1, 1.3]), AU)), Table),
            (_set(["temperature_tables", "BUS", "temperatures"], np.full((4, 6), 300.0)), Table),
            (_set(["temperature_tables", "BUS", "temperatures"], np.full((4, 7), -1.0)), Table),
            (_set(["temperature_tables", "BUS", "temperature_units"], "F"), Table),
            (_set(["temperature_tables", "BUS", "angle_reference"], [0.0, 0.0, 0.0]), Table),
            (_drop(["surfaces", 0, "area"]), Surface),
            (_set(["surfaces", 0, "area"], 4.0), Surface),
            (_set(["surfaces", 0, "area"], awu(-4.0, m**2)), Surface),
            (_set(["surfaces", 0, "area"], awu(4.0, m)), Surface),
            (_set(["surfaces", 0, "emissivity"], 1.5), Surface),
            (_set(["surfaces", 0, "emissivity"], "high"), Surface),
            (_set(["surfaces", 0, "normal"], [0.0, 0.0, 0.0]), Surface),
            (_set(["surfaces", 0, "normal"], [1.0, 0.0]), Surface),
            (_set(["surfaces", 0, "temperature"], "RADIATOR"), Unknown),
            (_set(["surfaces", 3, "shadow_temperature"], "RADIATOR"), Unknown),
        ],
        ids=[
            "no-surfaces", "no-tables", "empty-surfaces", "table-missing-key",
            "angles-no-units", "angles-length", "angles-decreasing", "angles-past-180",
            "distances-time", "distance-zero", "temps-shape", "temps-below-zero",
            "temps-units", "reference-zero", "area-missing", "area-no-units",
            "area-negative", "area-length", "emissivity-high", "emissivity-text",
            "normal-zero", "normal-2d", "unknown-table", "unknown-shadow-table",
        ],
    )
    def test_malformed_model_is_refused(self, sun, mutate, error):
        model_def = varying_model()
        mutate(model_def)
        with pytest.raises(error):
            ThermalRadiation(sun, fixed_sc(), J2000, model_def)

    def test_non_dict_model_is_refused(self, sun):
        with pytest.raises(ThermalRadiation.InvalidThermalModelError):
            ThermalRadiation(sun, fixed_sc(), J2000, [varying_model()])


# endregion Tests #
# --------------#
