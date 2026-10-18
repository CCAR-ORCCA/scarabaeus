// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Number of edges bounding one facet; every facet is a triangle.
const NUM_EDGES_PER_FACET: usize = 3;

//============================================================================
// Small vector helpers
//============================================================================
#[inline]
fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn scale(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Normalise, returning `None` for a degenerate (zero-length) vector.
#[inline]
fn unit(a: &[f64; 3]) -> Option<[f64; 3]> {
    let n = norm(a);
    if n == 0.0 || !n.is_finite() {
        None
    } else {
        Some(scale(a, 1.0 / n))
    }
}

/// Outer product `a b^T`, row-major.
#[inline]
fn outer(a: &[f64; 3], b: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i] * b[j];
        }
    }
    m
}

/// Matrix-vector product for a 3x3 row-major matrix.
#[inline]
fn matvec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

//============================================================================
// Precomputed edge record
//============================================================================
/// One undirected edge of the mesh, with the dyad that weights its line
/// integral.
///
/// `e_dyad` is the edge dyad
/// :math:`E_e = \hat n_A \hat n_{A,12}^T + \hat n_B \hat n_{B,21}^T`,
/// where `A` and `B` are the two facets sharing the edge, `n` is a facet's
/// outward normal, and `n_{A,12}` is the in-plane normal of `A` pointing away
/// from the facet across this edge. It depends only on geometry, so it is
/// built once at construction.
struct Edge {
    /// Index of the edge's first vertex.
    v1: usize,
    /// Index of the edge's second vertex.
    v2: usize,
    /// Edge dyad `E_e`.
    e_dyad: [[f64; 3]; 3],
}

//============================================================================
// Polyhedron Gravity Core
//============================================================================
/// Exterior gravity of a constant-density polyhedron.
///
/// Evaluates the potential, acceleration, gravity-gradient matrix and
/// Laplacian of a homogeneous polyhedral body by the closed-form surface and
/// line integrals of Werner & Scheeres (1997), summing one solid-angle term
/// per triangular facet and one line term per edge:
///
/// ```text
/// U       =  0.5 G rho (sum_e r_e . E_e r_e L_e  -  sum_f r_f . F_f r_f w_f)
/// grad U  = -G rho (sum_e E_e r_e L_e)  +  G rho (sum_f F_f r_f w_f)
/// grad2 U =  G rho (sum_e E_e L_e)      -  G rho (sum_f F_f w_f)
/// lap U   = -G rho sum_f w_f
/// ```
///
/// The facet dyads `F_f`, the edge dyads `E_e` and the mesh topology are
/// fixed by the shape, so they are built once here; only the per-evaluation
/// solid angles `w_f` and edge factors `L_e` depend on the field point.
///
/// The Laplacian doubles as a containment test: `sum_f w_f` is zero outside
/// the body and `4 pi` inside it (so `lap U` is Poisson's `-4 pi G rho`),
/// which is what makes the exterior series exact rather than truncated the
/// way a spherical-harmonic expansion is.
///
/// The gravitational constant is supplied by the caller (Scarabaeus passes
/// `Constants.G`), so this kernel stays purely geometric.

#[pyclass]
#[derive(Clone)]
pub struct PolyhedronGravityCore {
    /// Vertex coordinates, km.
    vertices: Vec<[f64; 3]>,
    /// Facet vertex indices, three per facet, in winding order.
    faces: Vec<[usize; 3]>,
    /// Outward unit normal of each facet.
    face_normals: Vec<[f64; 3]>,
    /// Facet dyads `F_f = n_f n_f^T`.
    face_dyads: Vec<[[f64; 3]; 3]>,
    /// Undirected edges with their precomputed dyads.
    edges: Vec<Edge>,
    /// Bulk density, kg/km^3.
    density: f64,
    /// Gravitational constant, km^3 kg^-1 s^-2.
    ///
    /// Supplied by the caller rather than fixed here, so the value tracks
    /// `scarabaeus.Constants.G` and a run reproducing the reference MEX
    /// implementation can pass its older CODATA figure instead.
    g_const: f64,
}

impl Clone for Edge {
    fn clone(&self) -> Self {
        Edge {
            v1: self.v1,
            v2: self.v2,
            e_dyad: self.e_dyad,
        }
    }
}

impl PolyhedronGravityCore {
    /// Solid angle subtended at `field_point` by facet `f`.
    ///
    /// Van Oosterom & Strackee's closed form,
    /// `w = 2 atan2(r1.(r2 x r3), R1 R2 R3 + (r1.r2) R3 + (r1.r3) R2 + (r2.r3) R1)`,
    /// which is the same angle the reference implementation's recursion
    /// computes but stays accurate when the facet is far away.
    ///
    /// The recursion it replaces (kept as
    /// [`Self::facet_solid_angle_recursion`]) finishes with
    /// `atan2(1 - cos w, sin w)`; for a distant facet `cos w -> 1`, so
    /// `1 - cos w` cancels catastrophically and the angle degrades to noise
    /// past roughly fifty body radii. Here the numerator shrinks with the
    /// facet while the denominator grows like `R^3`, so `atan2` stays exact.
    /// The two agree to ~1e-13 wherever the recursion is well conditioned.
    fn facet_solid_angle(&self, f: usize, field_point: &[f64; 3]) -> f64 {
        let face = self.faces[f];
        let r1 = sub(&self.vertices[face[0]], field_point);
        let r2 = sub(&self.vertices[face[1]], field_point);
        let r3 = sub(&self.vertices[face[2]], field_point);

        let n1 = norm(&r1);
        let n2 = norm(&r2);
        let n3 = norm(&r3);
        if n1 == 0.0 || n2 == 0.0 || n3 == 0.0 {
            // Field point sits on a vertex: the angle is undefined there.
            return f64::NAN;
        }

        let numerator = dot(&r1, &cross(&r2, &r3));
        let denominator = n1 * n2 * n3
            + dot(&r1, &r2) * n3
            + dot(&r1, &r3) * n2
            + dot(&r2, &r3) * n1;

        2.0 * numerator.atan2(denominator)
    }

    /// Solid angle by the reference implementation's half-angle recursion.
    ///
    /// The MEX routine's own method, kept so the port can be checked against
    /// its source: it walks the facet's three vertices, accumulating the
    /// rotation `[-cos S, sin S; -sin S, -cos S]` per vertex, and reads the
    /// signed angle off the accumulated pair, taking the sign from the last
    /// vertex's geometry. Superseded for production use by
    /// [`Self::facet_solid_angle`], which is the same angle without the
    /// far-field cancellation.
    fn facet_solid_angle_recursion(&self, f: usize, field_point: &[f64; 3]) -> f64 {
        let face = self.faces[f];
        let p = [
            self.vertices[face[0]],
            self.vertices[face[1]],
            self.vertices[face[2]],
        ];

        // cos/sin accumulator, seeded at (1, 0) as in the source
        let mut cos_sin_wf = [1.0_f64, 0.0_f64];

        // carried out of the loop: the sign test uses the last vertex
        let mut rj_last = [0.0_f64; 3];
        let mut n_hat_sign_last = [0.0_f64; 3];

        for gg in 0..NUM_EDGES_PER_FACET {
            // Middle vertex j, with its neighbours i (previous) and k (next).
            let (pj, pi, pk) = match gg {
                0 => (p[0], p[2], p[1]),
                1 => (p[1], p[0], p[2]),
                _ => (p[2], p[1], p[0]),
            };

            let rj = sub(&pj, field_point);
            let ri = sub(&pi, field_point);
            let rk = sub(&pk, field_point);

            let rj_norm = norm(&rj);
            if rj_norm == 0.0 {
                // Field point sits on a vertex: the angle is undefined there.
                return f64::NAN;
            }
            let rj_hat = scale(&rj, 1.0 / rj_norm);

            // In-plane directions from j towards i and k, obtained by
            // rotating each face-diagonal cross product back onto rj.
            let rj_ri_cross = cross(&rj, &ri);
            let rj_rk_cross = cross(&rj, &rk);
            let sji = cross(&rj_ri_cross, &rj);
            let sjk = cross(&rj_rk_cross, &rj);

            let (sji_hat, sjk_hat) = match (unit(&sji), unit(&sjk)) {
                (Some(a), Some(b)) => (a, b),
                // Degenerate: the field point is collinear with an edge.
                _ => return f64::NAN,
            };

            let n_hat_sign = cross(&sjk_hat, &sji_hat);

            let cos_s = dot(&sji_hat, &sjk_hat);
            let sin_s = dot(&rj_hat, &n_hat_sign).abs();

            // cos_sin_wf = [-cos_S, sin_S; -sin_S, -cos_S] * cos_sin_wf
            let c = cos_sin_wf[0];
            let s = cos_sin_wf[1];
            cos_sin_wf[0] = -cos_s * c + sin_s * s;
            cos_sin_wf[1] = -sin_s * c - cos_s * s;

            rj_last = rj;
            n_hat_sign_last = n_hat_sign;
        }

        let sign_dot = dot(&rj_last, &n_hat_sign_last);
        let sign_omega = if sign_dot > 0.0 {
            1.0
        } else if sign_dot < 0.0 {
            -1.0
        } else {
            0.0
        };

        sign_omega * 2.0 * (1.0 - cos_sin_wf[0]).atan2(cos_sin_wf[1])
    }

    /// Accumulate the facet and edge sums for one field point.
    ///
    /// Returns `(r_e.E r_e L, E r_e L, E L, r_f.F r_f w, F r_f w, F w, sum w)`
    /// — every term the potential, acceleration, gradient matrix and
    /// Laplacian are assembled from.
    #[allow(clippy::type_complexity)]
    fn accumulate(
        &self,
        field_point: &[f64; 3],
    ) -> (
        f64,
        [f64; 3],
        [[f64; 3]; 3],
        f64,
        [f64; 3],
        [[f64; 3]; 3],
        f64,
    ) {
        let mut r_e_l = 0.0_f64;
        let mut erl = [0.0_f64; 3];
        let mut el = [[0.0_f64; 3]; 3];

        let mut r_f_omega = 0.0_f64;
        let mut fr_omega = [0.0_f64; 3];
        let mut f_omega = [[0.0_f64; 3]; 3];
        let mut omega_total = 0.0_f64;

        // ---- surface integrals: one solid angle per facet ----
        for f in 0..self.faces.len() {
            let omega = self.facet_solid_angle(f, field_point);
            omega_total += omega;

            // r_f: field point to any point on the facet (its first vertex)
            let rf = sub(&self.vertices[self.faces[f][0]], field_point);
            let dyad = &self.face_dyads[f];
            let f_rf = matvec(dyad, &rf);

            r_f_omega += dot(&rf, &f_rf) * omega;

            for i in 0..3 {
                fr_omega[i] += f_rf[i] * omega;
                for j in 0..3 {
                    f_omega[i][j] += dyad[i][j] * omega;
                }
            }
        }

        // ---- line integrals: one log term per edge ----
        for edge in &self.edges {
            let v1 = self.vertices[edge.v1];
            let v2 = self.vertices[edge.v2];

            let a_vec = sub(field_point, &v1);
            let b_vec = sub(field_point, &v2);
            let e_vec = sub(&v2, &v1);

            let a = norm(&a_vec);
            let b = norm(&b_vec);
            let e = norm(&e_vec);

            // L_e = ln((a + b + e) / (a + b - e))
            let l = ((a + b + e) / (a + b - e)).ln();

            // r_e: field point to any point on the edge (its first vertex)
            let re = sub(&v1, field_point);
            let e_re = matvec(&edge.e_dyad, &re);

            r_e_l += dot(&re, &e_re) * l;

            for i in 0..3 {
                erl[i] += e_re[i] * l;
                for j in 0..3 {
                    el[i][j] += edge.e_dyad[i][j] * l;
                }
            }
        }

        (r_e_l, erl, el, r_f_omega, fr_omega, f_omega, omega_total)
    }
}

/// methods exposed to python
#[pymethods]
impl PolyhedronGravityCore {
    /// class constructor
    ///
    /// Builds the facet normals, the edge list and the geometry-only dyads.
    #[new]
    #[pyo3(signature = (vertices, faces, density, gravitational_constant))]
    pub fn __new__(
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
        density: f64,
        gravitational_constant: f64,
    ) -> PyResult<Self> {
        let v = vertices.as_array();
        let f = faces.as_array();

        if v.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vertices must have shape (N, 3); received (
                {}, {})",
                v.nrows(),
                v.ncols()
            )));
        }
        if f.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "faces must have shape (M, 3) — every facet is a triangle; received ({}, {})",
                f.nrows(),
                f.ncols()
            )));
        }
        if v.nrows() < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a closed polyhedron needs at least 4 vertices",
            ));
        }
        if f.nrows() < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a closed polyhedron needs at least 4 facets",
            ));
        }

        let n_vertices = v.nrows();
        let vertices: Vec<[f64; 3]> = (0..n_vertices)
            .map(|i| [v[[i, 0]], v[[i, 1]], v[[i, 2]]])
            .collect();

        let mut faces: Vec<[usize; 3]> = Vec::with_capacity(f.nrows());
        for i in 0..f.nrows() {
            let mut tri = [0usize; 3];
            for j in 0..3 {
                let idx = f[[i, j]];
                if idx < 0 || (idx as usize) >= n_vertices {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "facet {} references vertex {}, outside the {} supplied vertices \
                         (indices are 0-based)",
                        i, idx, n_vertices
                    )));
                }
                tri[j] = idx as usize;
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "facet {} repeats a vertex: {:?}",
                    i, tri
                )));
            }
            faces.push(tri);
        }

        // ---- facet normals and dyads ----
        let mut face_normals = Vec::with_capacity(faces.len());
        let mut face_dyads = Vec::with_capacity(faces.len());
        for (i, tri) in faces.iter().enumerate() {
            let edge_1 = sub(&vertices[tri[1]], &vertices[tri[0]]);
            let edge_2 = sub(&vertices[tri[2]], &vertices[tri[0]]);
            let n = match unit(&cross(&edge_1, &edge_2)) {
                Some(n) => n,
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "facet {} is degenerate: its vertices are collinear",
                        i
                    )));
                }
            };
            face_dyads.push(outer(&n, &n));
            face_normals.push(n);
        }

        // ---- edges, keyed on the unordered vertex pair ----
        // Each interior edge is walked by exactly two facets, in opposite
        // directions for a consistently wound closed surface. The traversal
        // direction is what orients the in-plane normal.
        let mut edge_map: HashMap<(usize, usize), Vec<(usize, i32)>> = HashMap::new();
        for (fi, tri) in faces.iter().enumerate() {
            for k in 0..NUM_EDGES_PER_FACET {
                let a = tri[k];
                let b = tri[(k + 1) % NUM_EDGES_PER_FACET];
                let key = if a < b { (a, b) } else { (b, a) };
                let direction = if a < b { 1 } else { -1 };
                edge_map.entry(key).or_default().push((fi, direction));
            }
        }

        // Sorted so the edge order is deterministic across runs, which keeps
        // the summation order — and therefore the last bits of the result —
        // reproducible.
        let mut keys: Vec<(usize, usize)> = edge_map.keys().copied().collect();
        keys.sort_unstable();

        let mut edges = Vec::with_capacity(keys.len());
        for key in keys {
            let adjacency = &edge_map[&key];
            if adjacency.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "edge ({}, {}) is shared by {} facet(s); a closed surface needs exactly 2. \
                     Check the mesh for holes or duplicated facets.",
                    key.0,
                    key.1,
                    adjacency.len()
                )));
            }

            let (v1, v2) = key;
            let mut e_dyad = [[0.0_f64; 3]; 3];

            for &(facet, direction) in adjacency.iter() {
                let n_hat = face_normals[facet];

                // Edge line in this facet's traversal direction.
                let edge_line = if direction == 1 {
                    sub(&vertices[v2], &vertices[v1])
                } else {
                    sub(&vertices[v1], &vertices[v2])
                };

                // In-plane normal pointing out of the facet across this edge:
                // n_12 = -(n x edge_line) / |n x edge_line|
                let cross_hat_edge = cross(&n_hat, &edge_line);
                let n_12 = match unit(&cross_hat_edge) {
                    Some(u) => scale(&u, -1.0),
                    None => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "edge ({}, {}) is parallel to facet {}'s normal",
                            v1, v2, facet
                        )));
                    }
                };

                let contribution = outer(&n_hat, &n_12);
                for i in 0..3 {
                    for j in 0..3 {
                        e_dyad[i][j] += contribution[i][j];
                    }
                }
            }

            edges.push(Edge { v1, v2, e_dyad });
        }

        Ok(Self {
            vertices,
            faces,
            face_normals,
            face_dyads,
            edges,
            density,
            g_const: gravitational_constant,
        })
    }

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!(
            "PolyhedronGravityCore | vertices: {} | facets: {} | edges: {} | rho: {:e} kg/km^3",
            self.vertices.len(),
            self.faces.len(),
            self.edges.len(),
            self.density
        )
    }

    /// shallow/deep copy support so owning Python objects stay copyable
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    /// Number of vertices in the mesh.
    #[getter]
    fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangular facets in the mesh.
    #[getter]
    fn n_facets(&self) -> usize {
        self.faces.len()
    }

    /// Number of unique edges in the mesh.
    #[getter]
    fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Bulk density in kg/km^3.
    #[getter]
    fn density(&self) -> f64 {
        self.density
    }

    #[setter]
    fn set_density(&mut self, density: f64) {
        self.density = density;
    }

    /// Enclosed volume in km^3, by the divergence theorem over the facets.
    ///
    /// Signed: a mesh wound inwards returns a negative volume, which is the
    /// cheapest way to catch a flipped facet convention.
    fn volume(&self) -> f64 {
        let mut total = 0.0;
        for tri in &self.faces {
            let a = self.vertices[tri[0]];
            let b = self.vertices[tri[1]];
            let c = self.vertices[tri[2]];
            total += dot(&a, &cross(&b, &c)) / 6.0;
        }
        total
    }

    /// Total mass in kg, `density * volume`.
    fn mass(&self) -> f64 {
        self.density * self.volume()
    }

    /// Centre of mass in the mesh's own frame, km.
    ///
    /// Decomposes the body into tetrahedra on the origin — one per facet — and
    /// takes the signed-volume-weighted mean of their centroids. Uniform
    /// density, so this is the centroid.
    ///
    /// Worth checking on any shape model: a mesh whose centroid is not at the
    /// origin carries a dipole term, so the field it produces will not match a
    /// point mass placed at the origin no matter how far away you go.
    fn center_of_mass(&self) -> [f64; 3] {
        let mut weighted = [0.0_f64; 3];
        let mut total_volume = 0.0_f64;

        for tri in &self.faces {
            let a = self.vertices[tri[0]];
            let b = self.vertices[tri[1]];
            let c = self.vertices[tri[2]];

            // signed volume of the tetrahedron (origin, a, b, c)
            let volume = dot(&a, &cross(&b, &c)) / 6.0;
            total_volume += volume;

            // its centroid is the mean of its four corners, one of which is
            // the origin
            for i in 0..3 {
                weighted[i] += volume * (a[i] + b[i] + c[i]) / 4.0;
            }
        }

        if total_volume == 0.0 {
            return [f64::NAN; 3];
        }
        scale(&weighted, 1.0 / total_volume)
    }

    /// Outward unit normal of each facet, flattened row-major (M x 3).
    fn face_normals(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.face_normals.len() * 3);
        for n in &self.face_normals {
            out.extend_from_slice(n);
        }
        out
    }

    /// Potential, acceleration, gravity gradient and Laplacian at one point.
    ///
    /// Parameters
    /// ----------
    /// position : (3,) array
    ///     Field point in the body-fixed frame, km.
    ///
    /// Returns
    /// -------
    /// tuple
    ///     `(U, grad_U, grad2_U, laplacian, omega)` with `U` in km^2/s^2,
    ///     `grad_U` in km/s^2 (3 entries), `grad2_U` in 1/s^2 (9 entries,
    ///     row-major), `laplacian` in 1/s^2, and `omega` the summed signed
    ///     solid angle in steradians (0 outside the body, 4 pi inside).
    #[pyo3(signature = (position))]
    fn compute(
        &self,
        position: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(f64, [f64; 3], [f64; 9], f64, f64)> {
        let p = position.as_array();
        if p.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "position must have 3 entries; received {}",
                p.len()
            )));
        }
        let field_point = [p[0], p[1], p[2]];

        let (r_e_l, erl, el, r_f_omega, fr_omega, f_omega, omega) =
            self.accumulate(&field_point);

        let g_rho = self.g_const * self.density;

        // U = 0.5 G rho (r.E r L) - 0.5 G rho (r.F r w)
        let potential = 0.5 * g_rho * r_e_l - 0.5 * g_rho * r_f_omega;

        let mut grad = [0.0_f64; 3];
        let mut grad2 = [0.0_f64; 9];
        for i in 0..3 {
            // grad U = -G rho E r L + G rho F r w
            grad[i] = -g_rho * erl[i] + g_rho * fr_omega[i];
            for j in 0..3 {
                // grad2 U = G rho E L - G rho F w
                grad2[3 * i + j] = g_rho * el[i][j] - g_rho * f_omega[i][j];
            }
        }

        // lap U = -G rho sum(w)
        let laplacian = -g_rho * omega;

        Ok((potential, grad, grad2, laplacian, omega))
    }

    /// Gravitational potential at one point, km^2/s^2.
    #[pyo3(signature = (position))]
    fn compute_potential(&self, position: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        Ok(self.compute(position)?.0)
    }

    /// Gravitational acceleration at one point, km/s^2.
    #[pyo3(signature = (position))]
    fn compute_acceleration(
        &self,
        position: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<[f64; 3]> {
        Ok(self.compute(position)?.1)
    }

    /// Gravity-gradient matrix at one point, 1/s^2, flattened row-major.
    #[pyo3(signature = (position))]
    fn compute_jacobian(&self, position: PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 9]> {
        Ok(self.compute(position)?.2)
    }

    /// Summed signed solid angle in steradians: 0 outside, 4 pi inside.
    ///
    /// Cheaper than a ray cast and exact for a closed mesh, so it is the
    /// natural containment test for a shape model.
    #[pyo3(signature = (position))]
    fn solid_angle(&self, position: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        Ok(self.compute(position)?.4)
    }

    /// True when the field point lies inside the body.
    ///
    /// Uses the solid angle: it is `4 pi` strictly inside and `0` strictly
    /// outside, so half of `4 pi` separates the two without ambiguity.
    #[pyo3(signature = (position))]
    fn contains(&self, position: PyReadonlyArray1<'_, f64>) -> PyResult<bool> {
        let omega = self.compute(position)?.4;
        Ok(omega.abs() > std::f64::consts::PI * 2.0)
    }

    /// Signed solid angle each facet subtends at a field point, steradians.
    ///
    /// Exposed for validation: the per-facet angles are what the surface sum
    /// is built from, and they can be checked one at a time against a closed
    /// form without going through the whole potential.
    #[pyo3(signature = (position))]
    fn facet_solid_angles(
        &self,
        position: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        let p = position.as_array();
        if p.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "position must have 3 entries; received {}",
                p.len()
            )));
        }
        let field_point = [p[0], p[1], p[2]];
        Ok((0..self.faces.len())
            .map(|f| self.facet_solid_angle(f, &field_point))
            .collect())
    }

    /// Per-facet solid angles by the reference implementation's recursion.
    ///
    /// Exposed only to check the port against its source; production sums use
    /// [`Self::facet_solid_angles`], which is the same angle evaluated in a
    /// form that survives the far field.
    #[pyo3(signature = (position))]
    fn facet_solid_angles_recursion(
        &self,
        position: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        let p = position.as_array();
        if p.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "position must have 3 entries; received {}",
                p.len()
            )));
        }
        let field_point = [p[0], p[1], p[2]];
        Ok((0..self.faces.len())
            .map(|f| self.facet_solid_angle_recursion(f, &field_point))
            .collect())
    }

    /// Accelerations for many field points, flattened row-major (N x 3).
    ///
    /// One call per trajectory sample amortises the Python boundary, which
    /// dominates when a shape model is sampled over a grid or an arc.
    #[pyo3(signature = (positions))]
    fn compute_acceleration_batch(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Vec<f64>> {
        let p = positions.as_array();
        if p.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "positions must have shape (N, 3); received ({}, {})",
                p.nrows(),
                p.ncols()
            )));
        }

        let mut out = Vec::with_capacity(p.nrows() * 3);
        for i in 0..p.nrows() {
            let field_point = [p[[i, 0]], p[[i, 1]], p[[i, 2]]];
            let (_, erl, _, _, fr_omega, _, _) = self.accumulate(&field_point);
            let g_rho = self.g_const * self.density;
            for k in 0..3 {
                out.push(-g_rho * erl[k] + g_rho * fr_omega[k]);
            }
        }
        Ok(out)
    }

    /// The gravitational constant this core was built with, km^3 kg^-1 s^-2.
    #[getter]
    fn gravitational_constant(&self) -> f64 {
        self.g_const
    }
}
