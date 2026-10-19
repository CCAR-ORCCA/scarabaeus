# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ================================================#
#  Unit Tests for the PolyhedralGravity class     #
# ================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import copy
import pickle
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

G = float(scb.Constants.G.values)
DENSITY = 2.0e12  # kg/km^3, i.e. 2 g/cm^3

# --------------------#
# region    Fixtures #
# --------------------#


def cube_mesh(half_width: float = 0.5):
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
            [0, 3, 2], [0, 2, 1],  # -z
            [4, 5, 6], [4, 6, 7],  # +z
            [0, 1, 5], [0, 5, 4],  # -y
            [1, 2, 6], [1, 6, 5],  # +x
            [2, 3, 7], [2, 7, 6],  # +y
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        dtype=np.int64,
    )
    return vertices, faces


def octahedron_mesh(a=1.0, b=0.6, c=0.4):
    """Triaxial octahedron: a cheap stand-in for an elongated body."""
    vertices = np.array(
        [[a, 0, 0], [-a, 0, 0], [0, b, 0], [0, -b, 0], [0, 0, c], [0, 0, -c]],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


@pytest.fixture
def origin():
    """Origin body the shape belongs to."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def cube(origin):
    """Unit cube with a rocky density."""
    vertices, faces = cube_mesh()
    return scb.PolyhedralGravity(origin, J2000, vertices, faces, density=DENSITY)


@pytest.fixture
def octahedron(origin):
    """Elongated body, where the field is least sphere-like."""
    vertices, faces = octahedron_mesh()
    return scb.PolyhedralGravity(origin, J2000, vertices, faces, density=DENSITY)


@pytest.fixture
def outside_points():
    """Field points comfortably outside either test body."""
    return [
        np.array([3.0, 1.0, -2.0]),
        np.array([0.9, 0.0, 0.0]),
        np.array([-1.5, 2.5, 0.4]),
    ]


def closed_form_solid_angle(r1, r2, r3):
    """
    Signed solid angle of a triangle by van Oosterom & Strackee.

    Independent of the ported recursion, so it can referee it.
    """
    n1, n2, n3 = (np.linalg.norm(r) for r in (r1, r2, r3))
    numerator = np.dot(r1, np.cross(r2, r3))
    denominator = (
        n1 * n2 * n3
        + np.dot(r1, r2) * n3
        + np.dot(r1, r3) * n2
        + np.dot(r2, r3) * n1
    )
    return 2.0 * np.arctan2(numerator, denominator)


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
class TestSolidAngle:
    def test_recursion_matches_the_closed_form(self, cube, outside_points):
        """The ported MEX recursion reproduces the van Oosterom-Strackee angle.

        This is the check that the port is faithful: the reference
        implementation's half-angle recursion and the closed form used in
        production are independent evaluations of the same angle, and they agree
        wherever the recursion is well conditioned.
        """
        core = cube._rust_core
        vertices, faces = cube.vertices, cube.faces

        for point in outside_points + [np.array([0.2, 0.1, -0.05])]:
            recursion = np.array(core.facet_solid_angles_recursion(point))
            production = np.array(core.facet_solid_angles(point))
            reference = np.array(
                [closed_form_solid_angle(*(vertices[i] - point for i in tri))
                 for tri in faces]
            )

            assert np.allclose(recursion, reference, atol=1e-12)
            assert np.allclose(production, reference, atol=1e-15)

    def test_is_zero_outside_and_four_pi_inside(self, cube, outside_points):
        """The summed solid angle separates interior from exterior exactly."""
        for point in outside_points:
            assert cube._rust_core.solid_angle(point) == pytest.approx(0.0, abs=1e-10)

        for point in (np.zeros(3), np.array([0.2, -0.1, 0.3])):
            assert cube._rust_core.solid_angle(point) == pytest.approx(
                4.0 * np.pi, rel=1e-12
            )

    def test_containment_follows_the_solid_angle(self, cube):
        """contains() agrees with the geometry for points in and out of the body."""
        assert cube.contains(np.zeros(3))
        assert cube.contains(np.array([0.49, 0.0, 0.0]))
        assert not cube.contains(np.array([0.51, 0.0, 0.0]))
        assert not cube.contains(np.array([10.0, 10.0, 10.0]))

    def test_distant_facets_keep_their_solid_angle(self, cube):
        """Solid angles stay accurate far from the body, where the source recursion did not.

        The reference implementation ends with ``atan2(1 - cos w, sin w)``; for a
        distant facet ``cos w -> 1`` and the subtraction cancels, so the angles
        decay into noise. The closed form used in production has no such term.
        """
        magnitudes = []
        for r in (100.0, 1e3, 1e4):
            point = np.array([r, 0.0, 0.0])
            angles = np.array(cube._rust_core.facet_solid_angles(point))

            # exterior point: the signed angles must cancel exactly ...
            assert angles.sum() == pytest.approx(0.0, abs=1e-14)
            magnitudes.append(np.abs(angles).sum())

        # ... while the magnitudes keep falling as 1/r^2 instead of collapsing
        # into noise, which is the failure the recursion exhibited
        for coarse, fine in zip(magnitudes, magnitudes[1:]):
            assert coarse / fine == pytest.approx(100.0, rel=2e-2)


def test_center_of_mass_locates_the_centroid(origin, cube):
    """The centroid comes back where the mesh actually sits."""
    assert np.allclose(cube.center_of_mass(), np.zeros(3), atol=1e-14)

    offset = np.array([0.3, -0.2, 0.1])
    vertices, faces = cube_mesh()
    shifted = scb.PolyhedralGravity(
        origin, J2000, (vertices + offset, faces), density=DENSITY
    )

    # an off-centre mesh carries a dipole, so this is worth being able to check
    assert np.allclose(shifted.center_of_mass(), offset, rtol=1e-12)


def test_volume_of_a_cube(cube, octahedron):
    """The divergence-theorem volume recovers the analytic one."""
    assert cube.volume() == pytest.approx(1.0, rel=1e-14)

    # octahedron with semi-axes a, b, c has volume 4/3 a b c
    assert octahedron.volume() == pytest.approx(
        4.0 / 3.0 * 1.0 * 0.6 * 0.4, rel=1e-14
    )


def test_far_field_approaches_a_point_mass(cube):
    """The field converges on GM/r^2 as the body shrinks with distance."""
    gm = cube.gm

    errors = []
    for r in (5.0, 50.0, 500.0):
        point = np.array([r, 0.0, 0.0])
        acc = np.linalg.norm(cube.compute_acceleration(point, {}, 0.0, None))
        errors.append(abs(acc - gm / r**2) / (gm / r**2))

    # A cube's leading departure from a point mass is its quadrupole, which
    # falls off as 1/r^2 relative, so each ten-fold step out buys ~two decades.
    assert errors[0] < 1e-3
    assert errors[1] < errors[0] / 50.0
    assert errors[2] < errors[1] / 50.0


def test_acceleration_points_toward_the_body(cube, octahedron, outside_points):
    """Gravity attracts."""
    for model in (cube, octahedron):
        for point in outside_points:
            acc = model.compute_acceleration(point, {}, 0.0, None)
            assert np.dot(acc, point) < 0.0


def test_acceleration_matches_the_potential_gradient(octahedron, outside_points):
    """The acceleration is the gradient of the returned potential."""
    h = 1e-6
    for point in outside_points:
        analytic = octahedron.compute_acceleration(point, {}, 0.0, None)
        numeric = np.array(
            [
                (
                    octahedron.compute_potential(point + h * e)
                    - octahedron.compute_potential(point - h * e)
                )
                / (2.0 * h)
                for e in np.eye(3)
            ]
        )
        assert np.allclose(numeric, analytic, rtol=1e-5)


class TestJacobian:
    def test_matches_the_acceleration_gradient(self, octahedron, outside_points):
        """The gravity gradient is the Jacobian of the returned acceleration."""
        h = 1e-6
        for point in outside_points:
            analytic = octahedron._compute_partial_by_position(point, {}, 0.0, None)
            numeric = np.column_stack(
                [
                    (
                        octahedron.compute_acceleration(point + h * e, {}, 0.0, None)
                        - octahedron.compute_acceleration(point - h * e, {}, 0.0, None)
                    )
                    / (2.0 * h)
                    for e in np.eye(3)
                ]
            )
            # Central differences on a ~1e-10 quantity: the FD step, not the
            # model, sets the achievable agreement, and entries that are
            # analytically zero come back as FD noise rather than zero.
            assert np.allclose(
                numeric, analytic, rtol=1e-4, atol=1e-8 * np.max(np.abs(analytic))
            )

    def test_is_symmetric(self, octahedron, outside_points):
        """A potential field has a symmetric Hessian.

        Only to machine precision, not bit-for-bit: the individual edge dyads
        ``n_A n_A12^T + n_B n_B21^T`` are not symmetric, and the symmetry of the
        total emerges from the sum over edges.
        """
        for point in outside_points:
            jac = octahedron._compute_partial_by_position(point, {}, 0.0, None)
            assert np.allclose(jac, jac.T, rtol=1e-12, atol=1e-30)

    def test_trace_satisfies_poisson(self, cube):
        """trace of the gradient matrix is 0 outside and -4 pi G rho inside."""
        outside = np.array([3.0, 1.0, -2.0])
        _, _, _, laplacian, _ = cube._rust_core.compute(outside)
        jac = cube._compute_partial_by_position(outside, {}, 0.0, None)

        # both are zero to round-off, judged against the size of the entries the
        # trace is formed from rather than an absolute floor
        scale = np.max(np.abs(jac))
        assert abs(np.trace(jac) - laplacian) < 1e-9 * scale
        assert abs(laplacian) < 1e-9 * scale

        inside = np.array([0.1, -0.05, 0.2])
        _, _, _, laplacian_in, _ = cube._rust_core.compute(inside)
        jac_in = cube._compute_partial_by_position(inside, {}, 0.0, None)
        assert laplacian_in == pytest.approx(-4.0 * np.pi * G * DENSITY, rel=1e-12)
        assert np.trace(jac_in) == pytest.approx(laplacian_in, rel=1e-12)


class TestErrorHandling:
    def test_inward_winding_gives_a_negative_volume(self, origin):
        """A flipped mesh is detectable rather than silently wrong."""
        vertices, faces = cube_mesh()
        flipped = faces[:, ::-1].copy()

        model = scb.PolyhedralGravity(origin, J2000, vertices, flipped, density=DENSITY)

        assert model.volume() == pytest.approx(-1.0, rel=1e-14)

        # and building from GM refuses it rather than returning a negative density
        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="wound inwards"):
            scb.PolyhedralGravity(origin, J2000, vertices, flipped, gm=1.0e-9)

    def test_open_mesh_is_reported(self, origin):
        """A hole in the surface is refused, naming the offending edge."""
        vertices, faces = cube_mesh()
        holed = faces[:-1]  # drop one facet, leaving three unpaired edges

        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="closed surface"):
            scb.PolyhedralGravity(origin, J2000, vertices, holed, density=DENSITY)

    def test_bad_facet_index_is_reported(self, origin):
        """An out-of-range vertex index is refused."""
        vertices, faces = cube_mesh()
        faces = faces.copy()
        faces[0, 0] = 99

        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="outside the"):
            scb.PolyhedralGravity(origin, J2000, vertices, faces, density=DENSITY)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"density": DENSITY, "gm": 1.0e-9}],
        ids=["neither", "both"],
    )
    def test_density_or_gm_but_not_both(self, origin, kwargs):
        """The two ways of setting the field strength are mutually exclusive."""
        vertices, faces = cube_mesh()

        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="exactly one"):
            scb.PolyhedralGravity(origin, J2000, vertices, faces, **kwargs)

    def test_body_fixed_frame_needs_an_epoch(self, origin):
        """Rotating out of a body-fixed frame without an epoch is refused."""
        vertices, faces = cube_mesh()
        model = scb.PolyhedralGravity(
            origin,
            J2000,
            vertices,
            faces,
            density=DENSITY,
            body_fixed_frame="IAU_EARTH",
        )

        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="epoch is required"):
            model.compute_potential(np.array([3.0, 0.0, 0.0]))

    def test_missing_mesh_is_reported(self):
        """Enabling the model without a mesh is refused at construction."""
        spacecraft = scb.Spacecraft(
            "ORCCA", -1234, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
        )

        with pytest.raises(ValueError, match="polyhedron_shape"):
            scb.ForceModelTranslation(
                primary_body=spacecraft, polyhedral_gravity=True
            )

    def test_missing_shape_file_is_reported(self, origin, tmp_path):
        """A path that is not there is refused by name."""
        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="not found"):
            scb.PolyhedralGravity(
                origin, J2000, str(tmp_path / "absent.obj"), density=DENSITY
            )

    def test_unsupported_shape_format_is_reported(self, origin, tmp_path):
        """An unreadable format says so, and says what is supported."""
        path = tmp_path / "shape.ply"
        path.write_text("ply\n")

        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="unsupported"):
            scb.PolyhedralGravity(origin, J2000, str(path), density=DENSITY)

    def test_no_shape_at_all_is_reported(self, origin):
        """Forgetting the shape entirely is refused with the options listed."""
        with pytest.raises(scb.PolyhedralGravity.ShapeModelError, match="no shape"):
            scb.PolyhedralGravity(origin, J2000, density=DENSITY)


def test_potential_is_linear_in_density(origin):
    """Doubling the density doubles potential and acceleration."""
    vertices, faces = octahedron_mesh()
    point = np.array([2.0, 0.5, -0.3])

    single = scb.PolyhedralGravity(origin, J2000, vertices, faces, density=DENSITY)
    double = scb.PolyhedralGravity(origin, J2000, vertices, faces, density=2 * DENSITY)

    assert double.compute_potential(point) == pytest.approx(
        2.0 * single.compute_potential(point), rel=1e-14
    )
    assert np.allclose(
        double.compute_acceleration(point, {}, 0.0, None),
        2.0 * single.compute_acceleration(point, {}, 0.0, None),
        rtol=1e-14,
    )


def test_density_partial_is_exact(octahedron):
    """The density partial matches a finite difference to round-off."""
    point = np.array([2.0, 0.5, -0.3])
    analytic = octahedron._compute_partial_by_rho_poly(point, {}, 0.0, None)

    h = DENSITY * 1e-6
    base = octahedron.density
    octahedron.density = base + h
    plus = octahedron.compute_acceleration(point, {}, 0.0, None)
    octahedron.density = base - h
    minus = octahedron.compute_acceleration(point, {}, 0.0, None)
    octahedron.density = base

    assert np.allclose((plus - minus) / (2.0 * h), analytic, rtol=1e-9)


def test_batch_matches_single_evaluations(octahedron, outside_points):
    """The batched entry point agrees with per-point calls."""
    points = np.array(outside_points)
    batched = np.array(
        octahedron._rust_core.compute_acceleration_batch(points)
    ).reshape(-1, 3)

    for i, point in enumerate(outside_points):
        assert np.array_equal(
            batched[i], np.array(octahedron._rust_core.compute_acceleration(point))
        )


def test_gm_round_trips_through_the_mesh_volume(origin, cube):
    """Building from GM reproduces the density that would give that GM."""
    vertices, faces = cube_mesh()

    from_gm = scb.PolyhedralGravity(origin, J2000, vertices, faces, gm=cube.gm)

    assert from_gm.density == pytest.approx(DENSITY, rel=1e-12)
    assert from_gm.gm == pytest.approx(cube.gm, rel=1e-12)


def test_rotating_the_mesh_rotates_the_field(origin):
    """The field is frame-covariant, which the body-fixed rotation relies on."""
    vertices, faces = octahedron_mesh()

    # a rotation about z by 30 degrees
    angle = np.deg2rad(30.0)
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    plain = scb.PolyhedralGravity(origin, J2000, vertices, faces, density=DENSITY)
    rotated = scb.PolyhedralGravity(
        origin, J2000, vertices @ R.T, faces, density=DENSITY
    )

    point = np.array([2.0, 0.5, -0.3])
    acc_plain = plain.compute_acceleration(point, {}, 0.0, None)
    acc_rotated = rotated.compute_acceleration(R @ point, {}, 0.0, None)

    assert np.allclose(acc_rotated, R @ acc_plain, rtol=1e-12)


def test_model_survives_deepcopy_and_pickle(cube):
    """The Propagator deep-copies force models, so the kernel must rebuild."""
    point = np.array([3.0, 1.0, -2.0])
    expected = cube.compute_acceleration(point, {}, 0.0, None)

    for clone in (copy.deepcopy(cube), pickle.loads(pickle.dumps(cube))):
        assert np.array_equal(
            clone.compute_acceleration(point, {}, 0.0, None), expected
        )
        assert clone.density == cube.density


def test_force_model_translation_wiring(origin):
    """The model is reachable through ForceModelTranslation."""
    vertices, faces = cube_mesh()
    spacecraft = scb.Spacecraft(
        "ORCCA", -1234, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )

    force_model = scb.ForceModelTranslation(
        primary_body=spacecraft,
        polyhedral_gravity=True,
        polyhedron_shape=(vertices, faces),
        polyhedron_density=DENSITY,
    )

    # A shape model already carries the central body's whole field, so the
    # default point mass must not be stacked on top of it.
    assert force_model.dynamics_mode_list == ["PolyhedralGravity"]

    state = scb.StateArray(
        epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
        origin=origin,
        state=(
            scb.StateDefinition()
            .position(spacecraft, scb.ArrayWFrame(np.array([3.0, 0.0, 0.0]), km, J2000))
            .velocity(
                spacecraft,
                scb.ArrayWFrame(np.array([0.0, 1.0e-4, 0.0]), km / sec, J2000),
            )
        ),
    )
    force_model.initialize_force_models(state, J2000, t0=0.0, tf=100.0)

    assert "PolyhedralGravity" in force_model.force_models
    assert "PointMassGravity" not in force_model.force_models


# ------------------------------------------------------------------#
# region  Shape input forms                                          #
# ------------------------------------------------------------------#
# A shape arrives in whatever form the caller already has: a published file, a
# mesh object from another library, or arrays. All of them have to land on the
# same field.


@pytest.fixture
def cube_obj_file(tmp_path):
    """The test cube written as a Wavefront .obj."""
    vertices, faces = cube_mesh()
    path = tmp_path / "cube.obj"
    with open(path, "w") as handle:
        handle.write("# unit cube\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        handle.write("vn 0 0 1\n")
        for tri in faces:
            # exercise the 'v//vn' record form as well as the plain one
            handle.write("f " + " ".join(f"{i + 1}//1" for i in tri) + "\n")
    return path


@pytest.fixture
def cube_table_file(tmp_path):
    """The test cube written as a PDS-style vertex/facet table."""
    vertices, faces = cube_mesh()
    path = tmp_path / "cube.tab"
    with open(path, "w") as handle:
        handle.write(f"{len(vertices)} {len(faces)}\n")
        for i, vertex in enumerate(vertices, start=1):
            handle.write(f"{i} {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for i, tri in enumerate(faces, start=1):
            handle.write(f"{i} {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    return path


class TestShapeInputForms:
    def test_from_an_obj_file(self, origin, cube, cube_obj_file):
        """A Wavefront .obj reproduces the field built from arrays."""
        from_file = scb.PolyhedralGravity(
            origin, J2000, str(cube_obj_file), density=DENSITY
        )

        assert from_file.n_facets == cube.n_facets
        assert from_file.volume() == pytest.approx(cube.volume(), rel=1e-14)

        point = np.array([3.0, 1.0, -2.0])
        assert np.allclose(
            from_file.compute_acceleration(point, {}, 0.0, None),
            cube.compute_acceleration(point, {}, 0.0, None),
            rtol=1e-14,
        )

    def test_from_a_vertex_facet_table(self, origin, cube, cube_table_file):
        """A PDS-style table reproduces the field built from arrays."""
        from_file = scb.PolyhedralGravity(
            origin, J2000, str(cube_table_file), density=DENSITY
        )

        point = np.array([3.0, 1.0, -2.0])
        assert np.allclose(
            from_file.compute_acceleration(point, {}, 0.0, None),
            cube.compute_acceleration(point, {}, 0.0, None),
            rtol=1e-14,
        )

    def test_from_a_mesh_object(self, origin, cube):
        """Anything exposing vertices and faces drops straight in."""
        vertices, faces = cube_mesh()
        mesh = type("Mesh", (), {"vertices": vertices, "faces": faces})()

        from_mesh = scb.PolyhedralGravity(origin, J2000, mesh, density=DENSITY)

        point = np.array([3.0, 1.0, -2.0])
        assert np.allclose(
            from_mesh.compute_acceleration(point, {}, 0.0, None),
            cube.compute_acceleration(point, {}, 0.0, None),
            rtol=1e-14,
        )

    def test_from_a_vertices_faces_pair(self, origin, cube):
        """A (vertices, faces) pair is accepted as one argument."""
        pair = cube_mesh()

        from_pair = scb.PolyhedralGravity(origin, J2000, pair, density=DENSITY)

        assert from_pair.n_facets == cube.n_facets

    def test_scale_converts_the_file_units(self, origin, cube_obj_file):
        """A mesh published in another unit is scaled on the way in."""
        scaled = scb.PolyhedralGravity(
            origin, J2000, str(cube_obj_file), density=DENSITY, scale=2.0
        )

        # a cube twice as wide encloses eight times the volume
        assert scaled.volume() == pytest.approx(8.0, rel=1e-14)

    def test_obj_triangulates_polygon_faces(self, origin, cube, tmp_path):
        """Quad faces are fan-triangulated rather than refused."""
        vertices, _ = cube_mesh()
        path = tmp_path / "quads.obj"
        quads = [
            [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        with open(path, "w") as handle:
            for vertex in vertices:
                handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            for quad in quads:
                handle.write("f " + " ".join(str(i + 1) for i in quad) + "\n")

        from_quads = scb.PolyhedralGravity(origin, J2000, str(path), density=DENSITY)

        assert from_quads.n_facets == 12  # six quads -> twelve triangles
        assert from_quads.volume() == pytest.approx(1.0, rel=1e-14)

        point = np.array([3.0, 1.0, -2.0])
        assert np.allclose(
            from_quads.compute_acceleration(point, {}, 0.0, None),
            cube.compute_acceleration(point, {}, 0.0, None),
            rtol=1e-14,
        )

    def test_force_model_accepts_a_shape_file(self, cube_obj_file, origin):
        """The force model accepts a shape file directly, as the wrapper does."""
        spacecraft = scb.Spacecraft(
            "ORCCA", -1234, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
        )

        force_model = scb.ForceModelTranslation(
            primary_body=spacecraft,
            polyhedral_gravity=True,
            polyhedron_shape=str(cube_obj_file),
            polyhedron_density=DENSITY,
        )
        state = scb.StateArray(
            epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
            origin=origin,
            state=(
                scb.StateDefinition()
                .position(spacecraft, scb.ArrayWFrame(np.array([3.0, 0.0, 0.0]), km, J2000))
                .velocity(
                    spacecraft,
                    scb.ArrayWFrame(np.array([0.0, 1.0e-4, 0.0]), km / sec, J2000),
                )
            ),
        )
        force_model.initialize_force_models(state, J2000, t0=0.0, tf=100.0)

        assert force_model.force_models["PolyhedralGravity"].n_facets == 12


# endregion                                                          #
# ------------------------------------------------------------------#


# ------------------------------------------------------------------#
# region  Serialization                                              #
# ------------------------------------------------------------------#
# A ForceModelTranslation file used to record only the polyhedral flag, so a
# reloaded model silently fell back to a point mass.


@pytest.fixture
def spacecraft():
    return scb.Spacecraft(
        "ORCCA", -1234, awu(100.0, kg), awu(1e-06, km**2), awu(1.5, None)
    )


def bound_polyhedron(force_model, spacecraft, origin):
    """Bind the force model to a state and return its PolyhedralGravity."""
    state = scb.StateArray(
        epoch=scb.EpochArray(np.array([0.0]), sys="TDB", rep="NUM"),
        origin=origin,
        state=(
            scb.StateDefinition()
            .position(spacecraft, scb.ArrayWFrame(np.array([3.0, 0.0, 0.0]), km, J2000))
            .velocity(
                spacecraft,
                scb.ArrayWFrame(np.array([0.0, 1.0e-4, 0.0]), km / sec, J2000),
            )
        ),
    )
    force_model.initialize_force_models(state, J2000, t0=0.0, tf=100.0)
    return force_model.force_models["PolyhedralGravity"]


class TestSerialization:
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    def test_mesh_survives_serialization(self, origin, spacecraft, scale):
        """A reloaded model rebuilds the same mesh, not a point mass."""
        vertices, faces = cube_mesh()
        force_model = scb.ForceModelTranslation(
            primary_body=spacecraft,
            polyhedral_gravity=True,
            polyhedron_shape=(vertices, faces),
            polyhedron_density=DENSITY,
            polyhedron_scale=scale,
        )

        restored = scb.ForceModelTranslation.from_dict(force_model.to_dict(), spacecraft)

        assert restored.dynamics_mode_list == ["PolyhedralGravity"]
        original = bound_polyhedron(force_model, spacecraft, origin)
        reloaded = bound_polyhedron(restored, spacecraft, origin)
        np.testing.assert_array_equal(reloaded.vertices, original.vertices)
        np.testing.assert_array_equal(reloaded.faces, original.faces)
        assert reloaded.density == original.density

    def test_shape_file_is_recorded_by_path(self, origin, spacecraft, cube_obj_file):
        """A shape file is written as its path, like a gravity-field file."""
        force_model = scb.ForceModelTranslation(
            primary_body=spacecraft,
            polyhedral_gravity=True,
            polyhedron_shape=str(cube_obj_file),
            polyhedron_density=DENSITY,
        )

        data = force_model.to_dict()
        restored = scb.ForceModelTranslation.from_dict(data, spacecraft)

        assert data["polyhedron"]["path"] == str(cube_obj_file)
        assert "vertices" not in data["polyhedron"]
        assert bound_polyhedron(restored, spacecraft, origin).n_facets == 12

    def test_file_without_the_shape_is_reported(self, spacecraft):
        """An older file with only the flag warns rather than passing silently."""
        vertices, faces = cube_mesh()
        data = scb.ForceModelTranslation(
            primary_body=spacecraft,
            polyhedral_gravity=True,
            polyhedron_shape=(vertices, faces),
            polyhedron_density=DENSITY,
        ).to_dict()
        del data["polyhedron"]

        with pytest.warns(UserWarning, match="polyhedral gravity"):
            restored = scb.ForceModelTranslation.from_dict(data, spacecraft)

        assert "PolyhedralGravity" not in restored.dynamics_mode_list


# endregion                                                          #
# ------------------------------------------------------------------#


# endregion Tests     #
# --------------------#
