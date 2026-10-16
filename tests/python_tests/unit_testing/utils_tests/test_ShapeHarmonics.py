# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Unit Tests for the ShapeHarmonics utility      #
# ================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import json
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

G = float(scb.Constants.G.values)
GM = 4.463e-4  # km^3/s^2, 433 Eros

# --------------------#
# region    Fixtures #
# --------------------#


def cube_mesh(half_width: float = 1.0):
    """Axis-aligned cube centred on the origin, wound outward."""
    h = half_width
    vertices = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def icosphere(subdivisions: int = 2):
    """Unit sphere from a subdivided icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    vertices = [np.array(v, dtype=float) / np.linalg.norm(v) for v in vertices]

    for _ in range(subdivisions):
        midpoints, new_faces = {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                m = vertices[i] + vertices[j]
                vertices.append(m / np.linalg.norm(m))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    return np.array(vertices), np.array(faces, dtype=np.int64)


@pytest.fixture
def asteroid_mesh():
    """Elongated, lumpy body centred on its centre of mass, Eros-like."""
    unit_vertices, faces = icosphere(2)
    x, y, z = unit_vertices.T
    lumps = (
        1.0
        + 0.16 * np.exp(-6.0 * ((x - 0.8) ** 2 + y**2))
        - 0.12 * np.exp(-8.0 * ((x + 0.7) ** 2 + (z - 0.5) ** 2))
    )
    vertices = unit_vertices * np.array([17.2, 6.05, 5.85]) * lumps[:, None]
    vertices = vertices - scb.ShapeHarmonics.center_of_mass(vertices, faces)
    return vertices, faces


@pytest.fixture
def asteroid_body():
    """433 Eros, from the NEAR Shoemaker numbers."""
    return scb.CelestialBody(
        name="EROS",
        mass=awu(GM / G, kg),
        mean_radius=awu(8.42, km),
        grav_param=awu(GM, km**3 / sec**2),
        base_frame=J2000,
        spice_id=2000433,
    )


@pytest.fixture
def coefficients(asteroid_mesh):
    """Degree-12 coefficients of the test asteroid."""
    return scb.ShapeHarmonics.from_shape(asteroid_mesh, degree=12)


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
class TestFromShape:
    """Correctness of the coefficients that from_shape computes."""

    def test_c00_is_unity(self, coefficients):
        """The coefficients are normalised the way the loader expects."""
        Cnm, _, _ = coefficients

        assert Cnm[0, 0] == pytest.approx(1.0, rel=1e-12)

    def test_degree_one_terms_vanish_for_centred_body(self, coefficients):
        """Centring on the centre of mass kills the dipole, as it must."""
        Cnm, Snm, _ = coefficients

        for value in (Cnm[1, 0], Cnm[1, 1], Snm[1, 1]):
            assert abs(value) < 1e-12

    def test_degree_one_terms_survive_offset_body(self, asteroid_mesh):
        """Skipping the centring leaves the offset in the degree-1 terms."""
        vertices, faces = asteroid_mesh
        offset = np.array([2.0, 0.0, 0.0])

        Cnm, _, radius = scb.ShapeHarmonics.from_shape(
            (vertices + offset, faces), degree=2, center=False
        )

        # C10/C11 encode the centre-of-mass offset in units of the reference radius
        assert abs(Cnm[1, 1]) > 1e-3

    def test_cube_degree_two_terms_vanish(self):
        """A cube is inertially isotropic, so its quadrupole is zero."""
        Cnm, Snm, _ = scb.ShapeHarmonics.from_shape(cube_mesh(), degree=4)

        for m in range(3):
            assert abs(Cnm[2, m]) < 1e-14
            assert abs(Snm[2, m]) < 1e-14

        # the first shape signature of a cube shows up at degree 4
        assert abs(Cnm[4, 0]) > 1e-4

    def test_degree_power_summarises_the_spectrum(self, coefficients):
        """The per-degree RMS power matches its definition."""
        Cnm, Snm, _ = coefficients

        power = scb.ShapeHarmonics.degree_power(Cnm, Snm)

        assert power.shape == (13,)
        assert power[0] == pytest.approx(1.0, rel=1e-12)  # C00 alone
        expected_2 = np.sqrt(np.sum(Cnm[2] ** 2 + Snm[2] ** 2))
        assert power[2] == pytest.approx(expected_2, rel=1e-12)

        # a real body's spectrum decays; degree 12 must sit well below degree 2
        assert power[12] < power[2]


class TestFieldAccuracy:
    """Agreement between the harmonic expansion and the polyhedral ground truth."""

    def test_reproduces_the_polyhedral_field(self, asteroid_mesh, asteroid_body, tmp_path):
        """The expansion converges on the shape model's own field with distance.

        The two models share no code — solid angles and edge integrals against a
        symbolic Legendre recursion — so agreement pins both.
        """
        path = str(tmp_path / "asteroid.json")
        record = scb.ShapeHarmonics.to_coefficient_file(
            asteroid_mesh, path, degree=12, gm=GM, body_name="EROS"
        )

        polyhedron = scb.PolyhedralGravity(asteroid_body, J2000, asteroid_mesh, gm=GM)
        harmonics = scb.SphericalHarmonicsGravity(
            12, path, asteroid_body, asteroid_body, True, 12, J2000
        )

        radius = record["ReferenceRadius"]
        point = np.array([0.6, 0.7, 0.39])
        point = point / np.linalg.norm(point) * (3.0 * radius)

        a_polyhedron = polyhedron.compute_acceleration(point, {}, 0.0, None)
        a_harmonics = harmonics.compute_acceleration(point, {}, 0.0, None)

        relative = np.linalg.norm(a_harmonics - a_polyhedron) / np.linalg.norm(a_polyhedron)
        assert relative < 1e-8

    def test_truncation_error_falls_with_distance(self, asteroid_mesh, asteroid_body, tmp_path):
        """Higher up, the truncated expansion gets monotonically better."""
        path = str(tmp_path / "asteroid.json")
        record = scb.ShapeHarmonics.to_coefficient_file(
            asteroid_mesh, path, degree=12, gm=GM
        )
        radius = record["ReferenceRadius"]

        polyhedron = scb.PolyhedralGravity(asteroid_body, J2000, asteroid_mesh, gm=GM)
        harmonics = scb.SphericalHarmonicsGravity(
            12, path, asteroid_body, asteroid_body, True, 12, J2000
        )

        errors = []
        direction = np.array([0.6, 0.7, 0.39])
        direction = direction / np.linalg.norm(direction)
        for factor in (1.2, 2.0, 4.0):
            point = direction * (factor * radius)
            a_polyhedron = polyhedron.compute_acceleration(point, {}, 0.0, None)
            a_harmonics = harmonics.compute_acceleration(point, {}, 0.0, None)
            errors.append(
                np.linalg.norm(a_harmonics - a_polyhedron) / np.linalg.norm(a_polyhedron)
            )

        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-10


class TestCoefficientFile:
    """Behavior of to_coefficient_file and the files it writes."""

    def test_written_file_loads_into_gravity_model(
        self, asteroid_mesh, asteroid_body, tmp_path
    ):
        """The file round-trips through SphericalHarmonicsGravity."""
        path = str(tmp_path / "asteroid.json")
        scb.ShapeHarmonics.to_coefficient_file(asteroid_mesh, path, degree=8, gm=GM)

        model = scb.SphericalHarmonicsGravity(
            8, path, asteroid_body, asteroid_body, True, 8, J2000
        )

        assert model.order == 8
        assert model.mu == pytest.approx(GM)
        assert model.Cnm[0, 0] == pytest.approx(1.0, rel=1e-12)

    def test_records_expected_metadata(self, asteroid_mesh, tmp_path):
        """Reference radius, GM, degree and term count are all recorded."""
        path = str(tmp_path / "asteroid.json")
        scb.ShapeHarmonics.to_coefficient_file(
            asteroid_mesh, path, degree=6, gm=GM, body_name="EROS", source="unit test"
        )

        with open(path) as handle:
            record = json.load(handle)

        assert record["BodyName"] == "EROS"
        assert record["MaxDegree"] == 6
        assert record["Normalized"] is True
        assert record["GravitationalParameter"] == pytest.approx(GM)
        assert record["source"][0]["srs"] == "unit test"

        # one entry per (degree, order) pair with n >= m
        assert len(record["Coefficients"]) == 7 * 8 // 2

        # the reference radius is the circumscribing radius of the mesh
        vertices, _ = asteroid_mesh
        assert record["ReferenceRadius"] == pytest.approx(
            float(np.max(np.linalg.norm(vertices, axis=1))), rel=1e-12
        )

    def test_density_sets_the_gravitational_parameter(self, asteroid_mesh, tmp_path):
        """Giving a density derives GM through the mesh volume."""
        vertices, faces = asteroid_mesh
        density = 2670e9  # kg/km^3, Eros

        path = str(tmp_path / "asteroid.json")
        record = scb.ShapeHarmonics.to_coefficient_file(
            asteroid_mesh, path, degree=4, density=density
        )

        volume = scb.ShapeHarmonics.volume(vertices, faces)
        assert record["GravitationalParameter"] == pytest.approx(G * density * volume)

    def test_unnormalized_coefficients_are_available(self, asteroid_mesh, tmp_path):
        """The raw convention is written when asked for."""
        path = str(tmp_path / "asteroid.json")
        record = scb.ShapeHarmonics.to_coefficient_file(
            asteroid_mesh, path, degree=4, gm=GM, normalized=False
        )

        assert record["Normalized"] is False

        # normalised and raw differ term by term but both start at C00 = 1
        normalized, _, _ = scb.ShapeHarmonics.from_shape(asteroid_mesh, degree=4)
        raw, _, _ = scb.ShapeHarmonics.from_shape(
            asteroid_mesh, degree=4, normalized=False
        )
        assert raw[0, 0] == pytest.approx(1.0, rel=1e-12)
        assert abs(raw[2, 0] - normalized[2, 0]) > 1e-6

    def test_accepts_shape_file_input(self, asteroid_mesh, tmp_path):
        """A shape file is accepted directly, like the force model."""
        vertices, faces = asteroid_mesh
        obj_path = tmp_path / "asteroid.obj"
        with open(obj_path, "w") as handle:
            for vertex in vertices:
                handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            for tri in faces:
                handle.write("f " + " ".join(str(i + 1) for i in tri) + "\n")

        from_file, _, radius_file = scb.ShapeHarmonics.from_shape(str(obj_path), degree=6)
        from_arrays, _, radius_arrays = scb.ShapeHarmonics.from_shape(
            asteroid_mesh, degree=6
        )

        assert radius_file == pytest.approx(radius_arrays, rel=1e-9)
        assert np.allclose(from_file, from_arrays, rtol=1e-9, atol=1e-14)


class TestErrorHandling:
    """Invalid inputs to from_shape / to_coefficient_file are reported."""

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"gm": GM, "density": 2670e9}],
        ids=["neither", "both"],
    )
    def test_gm_or_density_but_not_both(self, asteroid_mesh, tmp_path, kwargs):
        """The two ways of setting field strength are mutually exclusive."""
        with pytest.raises(scb.ShapeHarmonics.ShapeHarmonicsError, match="exactly one"):
            scb.ShapeHarmonics.to_coefficient_file(
                asteroid_mesh, str(tmp_path / "x.json"), degree=2, **kwargs
            )

    def test_negative_degree(self, asteroid_mesh):
        """A nonsense degree is refused."""
        with pytest.raises(scb.ShapeHarmonics.ShapeHarmonicsError, match="degree"):
            scb.ShapeHarmonics.from_shape(asteroid_mesh, degree=-1)

    def test_inward_wound_mesh(self):
        """A flipped mesh is caught rather than yielding negative coefficients."""
        vertices, faces = cube_mesh()

        with pytest.raises(scb.ShapeHarmonics.ShapeHarmonicsError, match="wound inwards"):
            scb.ShapeHarmonics.from_shape((vertices, faces[:, ::-1].copy()), degree=2)


# endregion Tests     #
# --------------------#
