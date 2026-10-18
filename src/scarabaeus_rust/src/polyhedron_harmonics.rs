// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

use numpy::{PyReadonlyArray2};
use pyo3::prelude::*;

/// Number of trinomial terms of exactly degree `n`.
#[inline]
fn coeff_count(n: usize) -> usize {
    (n + 1) * (n + 2) / 2
}

/// Linear storage index of the trinomial term `X^i Y^j Z^k` within degree `n`.
///
/// Terms are stored with `i` descending and `k` ascending, so the block for a
/// given `i` starts after every block with a larger `i`. That closed form
/// replaces the hash lookup the reference implementation performs per term,
/// which is the inner-most operation of the whole routine.
#[inline]
fn term_index(n: usize, i: usize, k: usize) -> usize {
    let d = n - i;
    d * (d + 1) / 2 + k
}

/// A homogeneous trinomial in `(X, Y, Z)`: a degree plus its dense coefficients.
#[derive(Clone)]
struct Trinomial {
    degree: usize,
    data: Vec<f64>,
}

impl Trinomial {
    fn zeros(degree: usize) -> Self {
        Self {
            degree,
            data: vec![0.0; coeff_count(degree)],
        }
    }

    fn scale(&mut self, factor: f64) {
        for value in self.data.iter_mut() {
            *value *= factor;
        }
    }
}

/// Per-degree `(i, j, k)` exponent tables, built once for the whole run.
struct TermTables {
    terms: Vec<Vec<(usize, usize, usize)>>,
    /// `i! j! k! / (n + 3)!` per term: the volume integral of a monomial over
    /// the unit simplex, so integrating a trinomial is one dot product.
    mixing: Vec<Vec<f64>>,
}

impl TermTables {
    fn new(max_degree: usize) -> Self {
        let mut factorials = vec![1.0_f64; max_degree + 4];
        for i in 1..factorials.len() {
            factorials[i] = factorials[i - 1] * i as f64;
        }

        let mut terms = Vec::with_capacity(max_degree + 1);
        let mut mixing = Vec::with_capacity(max_degree + 1);

        for n in 0..=max_degree {
            let mut degree_terms = Vec::with_capacity(coeff_count(n));
            let mut degree_mixing = Vec::with_capacity(coeff_count(n));
            let denominator = factorials[n + 3];

            for i in (0..=n).rev() {
                for k in 0..=(n - i) {
                    let j = n - i - k;
                    degree_terms.push((i, j, k));
                    degree_mixing
                        .push(factorials[i] * factorials[j] * factorials[k] / denominator);
                }
            }

            terms.push(degree_terms);
            mixing.push(degree_mixing);
        }

        Self { terms, mixing }
    }

    /// Product of two trinomials, degrees adding.
    fn multiply(&self, left: &Trinomial, right: &Trinomial) -> Trinomial {
        let degree = left.degree + right.degree;
        let mut result = Trinomial::zeros(degree);

        for (index_l, &(i_l, _, k_l)) in self.terms[left.degree].iter().enumerate() {
            let coeff_l = left.data[index_l];
            if coeff_l == 0.0 {
                continue;
            }
            for (index_r, &(i_r, _, k_r)) in self.terms[right.degree].iter().enumerate() {
                let coeff_r = right.data[index_r];
                if coeff_r == 0.0 {
                    continue;
                }
                let index = term_index(degree, i_l + i_r, k_l + k_r);
                result.data[index] += coeff_l * coeff_r;
            }
        }
        result
    }

    /// Integral of a trinomial over the simplex, up to the simplex volume.
    fn integrate(&self, tri: &Trinomial) -> f64 {
        self.mixing[tri.degree]
            .iter()
            .zip(tri.data.iter())
            .map(|(m, c)| m * c)
            .sum()
    }
}

/// Recursion factors for the associated Legendre functions.
///
/// Two conventions: fully normalised (what Scarabaeus stores) and raw. They
/// differ only in these tables, so the recursion below is written once.
struct RecursionFactors {
    diagonal: Vec<f64>,
    subdiagonal: Vec<f64>,
    vertical_1: Vec<Vec<f64>>,
    vertical_2: Vec<Vec<f64>>,
}

impl RecursionFactors {
    fn new(n_degree: usize, normalized: bool) -> Self {
        let mut diagonal = vec![0.0; n_degree + 1];
        let mut subdiagonal = vec![0.0; n_degree + 1];
        let mut vertical_1 = vec![vec![0.0; n_degree + 1]; n_degree + 1];
        let mut vertical_2 = vec![vec![0.0; n_degree + 1]; n_degree + 1];

        if normalized {
            if n_degree >= 1 {
                diagonal[1] = 1.0 / 3.0_f64.sqrt();
            }
            for n in 2..=n_degree {
                let nf = n as f64;
                diagonal[n] = (2.0 * nf - 1.0) / (2.0 * nf * (2.0 * nf + 1.0)).sqrt();
            }
            for n in 0..=n_degree {
                let nf = n as f64;
                subdiagonal[n] = (2.0 * nf - 1.0) / (2.0 * nf + 1.0).sqrt();
            }
            for n in 2..=n_degree {
                let nf = n as f64;
                for m in 0..n.saturating_sub(1) {
                    let mf = m as f64;
                    let denominator = (2.0 * nf + 1.0) * (nf + mf) * (nf - mf);
                    vertical_1[n][m] = (2.0 * nf - 1.0) * ((2.0 * nf - 1.0) / denominator).sqrt();
                    vertical_2[n][m] =
                        (((2.0 * nf - 3.0) * (nf + mf - 1.0) * (nf - mf - 1.0)) / denominator)
                            .sqrt();
                }
            }
        } else {
            if n_degree >= 1 {
                diagonal[1] = 1.0;
            }
            for n in 2..=n_degree {
                diagonal[n] = 1.0 / (2.0 * n as f64);
            }
            for n in 0..=n_degree {
                subdiagonal[n] = 1.0;
            }
            for n in 2..=n_degree {
                let nf = n as f64;
                for m in 0..n.saturating_sub(1) {
                    let mf = m as f64;
                    vertical_1[n][m] = (2.0 * nf - 1.0) / (nf + mf);
                    vertical_2[n][m] = (nf - mf - 1.0) / (nf + mf);
                }
            }
        }

        Self {
            diagonal,
            subdiagonal,
            vertical_1,
            vertical_2,
        }
    }
}

/// Accumulate one simplex (origin + triangle) into the coefficient arrays.
///
/// The body is decomposed into tetrahedra on the origin; each contributes its
/// exact harmonic coefficients, obtained by running the Legendre recursion
/// symbolically in the simplex's barycentric coordinates and integrating the
/// resulting trinomial term by term.
#[allow(clippy::too_many_arguments)]
fn accumulate_one_simplex(
    n_degree: usize,
    cnm: &mut [f64],
    snm: &mut [f64],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    factors: &RecursionFactors,
    tables: &TermTables,
    ref_radius: f64,
) {
    let stride = n_degree + 1;

    // six times the tetrahedron volume: the simplex's Jacobian
    let overall_factor = v0[0] * (v1[1] * v2[2] - v2[1] * v1[2])
        + v1[0] * (v2[1] * v0[2] - v2[2] * v0[1])
        + v2[0] * (v0[1] * v1[2] - v1[1] * v0[2]);
    if overall_factor == 0.0 {
        return;
    }

    // work in units of the reference radius
    let x = [v0[0] / ref_radius, v1[0] / ref_radius, v2[0] / ref_radius];
    let y = [v0[1] / ref_radius, v1[1] / ref_radius, v2[1] / ref_radius];
    let z = [v0[2] / ref_radius, v1[2] / ref_radius, v2[2] / ref_radius];

    // The Cartesian coordinates as linear trinomials in the barycentric
    // coordinates of the simplex.
    let mut x_tri = Trinomial::zeros(1);
    let mut y_tri = Trinomial::zeros(1);
    let mut z_tri = Trinomial::zeros(1);
    for corner in 0..3 {
        let index = term_index(1, if corner == 0 { 1 } else { 0 }, if corner == 2 { 1 } else { 0 });
        x_tri.data[index] = x[corner];
        y_tri.data[index] = y[corner];
        z_tri.data[index] = z[corner];
    }

    // r^2 = x^2 + y^2 + z^2, quadratic in the same coordinates
    let mut r_squared = Trinomial::zeros(2);
    r_squared.data[term_index(2, 2, 0)] = x[0] * x[0] + y[0] * y[0] + z[0] * z[0];
    r_squared.data[term_index(2, 0, 0)] = x[1] * x[1] + y[1] * y[1] + z[1] * z[1];
    r_squared.data[term_index(2, 0, 2)] = x[2] * x[2] + y[2] * y[2] + z[2] * z[2];
    r_squared.data[term_index(2, 1, 0)] = 2.0 * (x[0] * x[1] + y[0] * y[1] + z[0] * z[1]);
    r_squared.data[term_index(2, 0, 1)] = 2.0 * (x[1] * x[2] + y[1] * y[2] + z[1] * z[2]);
    r_squared.data[term_index(2, 1, 1)] = 2.0 * (x[2] * x[0] + y[2] * y[0] + z[2] * z[0]);

    let mut vertical_c = [Trinomial::zeros(0), Trinomial::zeros(0), Trinomial::zeros(0)];
    let mut vertical_s = [Trinomial::zeros(0), Trinomial::zeros(0), Trinomial::zeros(0)];
    let mut diagonal_c = Trinomial::zeros(0);
    let mut diagonal_s = Trinomial::zeros(0);

    // Rolling slots for the two-term vertical recurrence. They rotate only in
    // the general branch and carry across columns, as in the reference.
    let (mut prior2, mut prior1, mut present) = (0usize, 1usize, 2usize);

    for m in 0..=n_degree {
        for n in m..=n_degree {
            if n == m {
                // sectoral term: shift the previous column's diagonal
                if m == 0 {
                    vertical_c[prior2] = Trinomial::zeros(0);
                    vertical_c[prior2].data[0] = overall_factor;
                    vertical_s[prior2] = Trinomial::zeros(0);
                } else if m == 1 {
                    let factor = overall_factor * factors.diagonal[1];
                    vertical_c[prior2] = Trinomial::zeros(1);
                    vertical_s[prior2] = Trinomial::zeros(1);
                    for corner in 0..3 {
                        let index = term_index(
                            1,
                            if corner == 0 { 1 } else { 0 },
                            if corner == 2 { 1 } else { 0 },
                        );
                        vertical_c[prior2].data[index] = x[corner] * factor;
                        vertical_s[prior2].data[index] = y[corner] * factor;
                    }
                } else {
                    let mut c = tables.multiply(&x_tri, &diagonal_c);
                    let cs = tables.multiply(&y_tri, &diagonal_s);
                    for (value, subtrahend) in c.data.iter_mut().zip(cs.data.iter()) {
                        *value -= subtrahend;
                    }
                    c.scale(factors.diagonal[m]);

                    let mut s = tables.multiply(&y_tri, &diagonal_c);
                    let sc = tables.multiply(&x_tri, &diagonal_s);
                    for (value, addend) in s.data.iter_mut().zip(sc.data.iter()) {
                        *value += addend;
                    }
                    s.scale(factors.diagonal[m]);

                    vertical_c[prior2] = c;
                    vertical_s[prior2] = s;
                }

                cnm[n * stride + m] += tables.integrate(&vertical_c[prior2]);
                snm[n * stride + m] += tables.integrate(&vertical_s[prior2]);

                diagonal_c = vertical_c[prior2].clone();
                diagonal_s = vertical_s[prior2].clone();
            } else if n == m + 1 {
                // first term below the diagonal
                let mut c = tables.multiply(&vertical_c[prior2], &z_tri);
                let mut s = tables.multiply(&vertical_s[prior2], &z_tri);
                c.scale(factors.subdiagonal[n]);
                s.scale(factors.subdiagonal[n]);
                vertical_c[prior1] = c;
                vertical_s[prior1] = s;

                cnm[n * stride + m] += tables.integrate(&vertical_c[prior1]);
                snm[n * stride + m] += tables.integrate(&vertical_s[prior1]);
            } else {
                // two-term vertical recurrence
                let mut c1 = tables.multiply(&vertical_c[prior1], &z_tri);
                c1.scale(factors.vertical_1[n][m]);
                let mut c2 = tables.multiply(&vertical_c[prior2], &r_squared);
                c2.scale(factors.vertical_2[n][m]);
                for (value, subtrahend) in c1.data.iter_mut().zip(c2.data.iter()) {
                    *value -= subtrahend;
                }

                let mut s1 = tables.multiply(&vertical_s[prior1], &z_tri);
                s1.scale(factors.vertical_1[n][m]);
                let mut s2 = tables.multiply(&vertical_s[prior2], &r_squared);
                s2.scale(factors.vertical_2[n][m]);
                for (value, subtrahend) in s1.data.iter_mut().zip(s2.data.iter()) {
                    *value -= subtrahend;
                }

                vertical_c[present] = c1;
                vertical_s[present] = s1;

                cnm[n * stride + m] += tables.integrate(&vertical_c[present]);
                snm[n * stride + m] += tables.integrate(&vertical_s[present]);

                let rotate = prior2;
                prior2 = prior1;
                prior1 = present;
                present = rotate;
            }
        }
    }
}

/// Spherical harmonic coefficients of a constant-density polyhedron.
///
/// Werner's exact recursion, summed over the tetrahedra formed by each facet
/// and the origin. Exact for the polyhedron: the only approximation is the
/// polyhedron itself, plus the truncation at `n_degree`.
///
/// The mesh is validated and stored at construction; `compute` runs the
/// recursion for a given degree, reference radius and normalisation.
#[pyclass]
#[derive(Clone)]
pub struct PolyhedronHarmonics {
    /// Vertex coordinates in the body-fixed frame, km.
    vertices: Vec<[f64; 3]>,
    /// Facet vertex indices, three per facet.
    faces: Vec<[usize; 3]>,
}

/// methods exposed to python
#[pymethods]
impl PolyhedronHarmonics {
    /// class constructor
    ///
    /// Parameters
    /// ----------
    /// vertices : (N, 3) array
    ///     Vertex coordinates in the body-fixed frame, km. Should be centred on
    ///     the body's centre of mass, or the degree-1 terms will not vanish.
    /// faces : (M, 3) array
    ///     Zero-based triangular facet indices.
    #[new]
    #[pyo3(signature = (vertices, faces))]
    pub fn __new__(
        vertices: PyReadonlyArray2<'_, f64>,
        faces: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Self> {
        let v = vertices.as_array();
        let f = faces.as_array();

        if v.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vertices must have shape (N, 3); received ({}, {})",
                v.nrows(),
                v.ncols()
            )));
        }
        if f.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "faces must have shape (M, 3); received ({}, {})",
                f.nrows(),
                f.ncols()
            )));
        }

        let n_vertices = v.nrows();
        let vertices: Vec<[f64; 3]> = (0..n_vertices)
            .map(|i| [v[[i, 0]], v[[i, 1]], v[[i, 2]]])
            .collect();

        let mut faces: Vec<[usize; 3]> = Vec::with_capacity(f.nrows());
        for row in 0..f.nrows() {
            let mut tri = [0usize; 3];
            for c in 0..3 {
                let index = f[[row, c]];
                if index < 0 || (index as usize) >= n_vertices {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "facet {} references vertex {}, outside the {} supplied vertices",
                        row, index, n_vertices
                    )));
                }
                tri[c] = index as usize;
            }
            faces.push(tri);
        }

        Ok(Self { vertices, faces })
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

    /// Spherical harmonic coefficients up to `n_degree`.
    ///
    /// The returned coefficients are **unnormalised by mass** — they carry the
    /// body's volume, so `C[0][0]` is the volume in units of `ref_radius^3` times
    /// the simplex Jacobian convention. Dividing the whole set by `C[0][0]` gives
    /// the usual convention with `C00 = 1`, which is what the Python caller does.
    ///
    /// Parameters
    /// ----------
    /// n_degree : int
    ///     Maximum degree and order to compute.
    /// ref_radius : float
    ///     Normalising radius, km.
    /// normalized : bool
    ///     Fully normalised coefficients when true, raw when false.
    ///
    /// Returns
    /// -------
    /// tuple
    ///     `(Cnm, Snm)`, each a flattened `(n_degree + 1)^2` row-major array.
    #[pyo3(signature = (n_degree, ref_radius, normalized=true))]
    fn compute(
        &self,
        n_degree: usize,
        ref_radius: f64,
        normalized: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if ref_radius <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ref_radius must be positive; received {}",
                ref_radius
            )));
        }

        // Intermediates reach two degrees above the target: the vertical
        // recurrence multiplies a degree-n trinomial by r^2.
        let tables = TermTables::new(n_degree + 2);
        let factors = RecursionFactors::new(n_degree, normalized);

        let stride = n_degree + 1;
        let mut cnm = vec![0.0_f64; stride * stride];
        let mut snm = vec![0.0_f64; stride * stride];

        for tri in self.faces.iter() {
            accumulate_one_simplex(
                n_degree,
                &mut cnm,
                &mut snm,
                self.vertices[tri[0]],
                self.vertices[tri[1]],
                self.vertices[tri[2]],
                &factors,
                &tables,
                ref_radius,
            );
        }

        Ok((cnm, snm))
    }
}
