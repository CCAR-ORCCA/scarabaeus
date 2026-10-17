// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC
use num_dual::{Dual64, DualNum};
use numpy::ndarray::ArrayView2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

//============================================================================
// N-Plate SRP Core
//============================================================================
/// N-plate SRP acceleration and partials.
///
/// Computational core of `nPlateSRP`: the per-plate illumination sums for the
/// acceleration and its position / velocity Jacobians, including the SC-fixed
/// <-> inertial mapping. Attitude resolution, the Sun ephemeris and the
/// eta_srp / mass lookup stay on the Python side, which passes in the
/// Sun->body vector (inertial), the body->inertial DCM (and its state
/// derivatives when the attitude follows the state), the effective
/// eta_srp / mass, and the plate normals (SC-fixed) for each evaluation.
/// A double-sided plate lit from behind acts as a plate with the opposite
/// normal (both faces share the optical coefficients).
///
/// The acceleration kernel is generic over `num_dual::DualNum<f64>` (`f64` in
/// production, `Dual64` for `compute_partials_ad`), so the dual-number partials
/// differentiate the very expressions that compute the acceleration.
#[pyclass]
#[derive(Clone)]
pub struct NPlateSRPCore {
    pub p_srp: f64,
    pub areas: Vec<f64>,
    pub diff_ref: Vec<f64>,
    pub spec_ref: Vec<f64>,
    pub double_sided: Vec<bool>,
}

/// methods exposed to python
#[pymethods]
impl NPlateSRPCore {
    /// class constructor
    #[new]
    #[pyo3(signature = (p_srp, areas, diff_ref, spec_ref, double_sided=None))]
    pub fn __new__(
        p_srp: f64,
        areas: PyReadonlyArray1<'_, f64>,
        diff_ref: PyReadonlyArray1<'_, f64>,
        spec_ref: PyReadonlyArray1<'_, f64>,
        double_sided: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let areas: Vec<f64> = areas.as_array().to_vec();
        let diff_ref: Vec<f64> = diff_ref.as_array().to_vec();
        let spec_ref: Vec<f64> = spec_ref.as_array().to_vec();
        if areas.len() != diff_ref.len() || areas.len() != spec_ref.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "areas ({}), diff_ref ({}), and spec_ref ({}) must have equal lengths",
                areas.len(),
                diff_ref.len(),
                spec_ref.len()
            )));
        }
        let double_sided = double_sided.unwrap_or_else(|| vec![false; areas.len()]);
        if double_sided.len() != areas.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "double_sided ({}) must have one flag per plate ({})",
                double_sided.len(),
                areas.len()
            )));
        }
        Ok(Self {
            p_srp,
            areas,
            diff_ref,
            spec_ref,
            double_sided,
        })
    }

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!(
            "NPlateSRPCore | plates: {} | P_SRP: {:e}",
            self.areas.len(),
            self.p_srp
        )
    }

    /// shallow/deep copy support so owning Python objects stay copyable
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    /// SRP acceleration in the inertial frame [km/s^2].
    ///
    /// `sun2body` is the Sun->spacecraft vector (inertial) [km], `dcm_b2i` the
    /// SC-fixed -> inertial DCM, `normals` the (N, 3) SC-fixed plate outward
    /// normals (not necessarily unit).
    fn compute_acceleration(
        &self,
        sun2body: [f64; 3],
        dcm_b2i: PyReadonlyArray2<'_, f64>,
        eta_srp: f64,
        mass: f64,
        normals: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<[f64; 3]> {
        let m = to_mat3(&dcm_b2i)?;
        self.accel(&m, sun2body, eta_srp, mass, &normals.as_array())
    }

    /// Inertial position Jacobian da/dr [1/s^2], row-major, analytical.
    ///
    /// With the attitude fixed, da/dr = -DCM @ dT/ds @ DCM^T, with
    /// s = -DCM^T sun2body the SC-fixed body->Sun vector and T the SC-fixed
    /// acceleration. When the attitude follows the position (nadir pointing),
    /// `dcm_dr` is the (3, 3, 3) derivative dcm_dr[j, i, k] = dDCM[i, j]/dr_k
    /// and the attitude terms of `attitude_jacobian` are added.
    #[pyo3(signature = (sun2body, dcm_b2i, eta_srp, mass, normals, dcm_dr=None))]
    fn compute_jacobian(
        &self,
        sun2body: [f64; 3],
        dcm_b2i: PyReadonlyArray2<'_, f64>,
        eta_srp: f64,
        mass: f64,
        normals: PyReadonlyArray2<'_, f64>,
        dcm_dr: Option<PyReadonlyArray3<'_, f64>>,
    ) -> PyResult<[f64; 9]> {
        let m = to_mat3(&dcm_b2i)?;
        let s = rotate_t(&m, [-sun2body[0], -sun2body[1], -sun2body[2]]);
        let g = self.prep(s, eta_srp, mass, &normals.as_array())?;
        let jac_bf = body_jacobian(&g);

        // -DCM @ dT/ds @ DCM^T  (ds/dr = -DCM^T)
        let mut out = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut v = 0.0_f64;
                for a in 0..3 {
                    for b in 0..3 {
                        v += m[i][a] * jac_bf[a][b] * m[j][b];
                    }
                }
                out[i * 3 + j] = -v;
            }
        }

        if let Some(d) = dcm_dr {
            let d = to_dcm_derivative(&d)?;
            let att = attitude_jacobian(&m, &body_accel(&g), &jac_bf, sun2body, &d);
            for (o, a) in out.iter_mut().zip(att) {
                *o += a;
            }
        }
        Ok(out)
    }

    /// Inertial velocity Jacobian da/dv [1/s], row-major, analytical.
    ///
    /// The acceleration depends on the velocity only through the attitude, so
    /// this is the `attitude_jacobian` of `dcm_dv`, the (3, 3, 3) derivative
    /// dcm_dv[j, i, k] = dDCM[i, j]/dv_k (nadir pointing).
    fn compute_jacobian_by_velocity(
        &self,
        sun2body: [f64; 3],
        dcm_b2i: PyReadonlyArray2<'_, f64>,
        eta_srp: f64,
        mass: f64,
        normals: PyReadonlyArray2<'_, f64>,
        dcm_dv: PyReadonlyArray3<'_, f64>,
    ) -> PyResult<[f64; 9]> {
        let m = to_mat3(&dcm_b2i)?;
        let s = rotate_t(&m, [-sun2body[0], -sun2body[1], -sun2body[2]]);
        let g = self.prep(s, eta_srp, mass, &normals.as_array())?;
        let d = to_dcm_derivative(&dcm_dv)?;
        Ok(attitude_jacobian(
            &m,
            &body_accel(&g),
            &body_jacobian(&g),
            sun2body,
            &d,
        ))
    }

    /// Inertial Jacobians da/dr [1/s^2] and da/dv [1/s] (row-major) and
    /// da/deta_srp [km/s^2], via forward-mode automatic differentiation (dual
    /// numbers) of `compute_acceleration`. Each position component is seeded
    /// in turn (d sun2body / d r = I), together with the DCM through `dcm_dr`
    /// when the attitude follows the position; each velocity component seeds
    /// only the DCM, through `dcm_dv` (da/dv = 0 without it); then eta_srp.
    ///
    /// Independent of the analytical `compute_jacobian`,
    /// `compute_jacobian_by_velocity` and the `a / eta` partial; the pairs
    /// must agree. Normals, mass and the Sun position are held fixed, and the
    /// DCM derivatives are taken as given.
    #[pyo3(signature = (sun2body, dcm_b2i, eta_srp, mass, normals, dcm_dr=None, dcm_dv=None))]
    fn compute_partials_ad(
        &self,
        sun2body: [f64; 3],
        dcm_b2i: PyReadonlyArray2<'_, f64>,
        eta_srp: f64,
        mass: f64,
        normals: PyReadonlyArray2<'_, f64>,
        dcm_dr: Option<PyReadonlyArray3<'_, f64>>,
        dcm_dv: Option<PyReadonlyArray3<'_, f64>>,
    ) -> PyResult<([f64; 9], [f64; 9], [f64; 3])> {
        let c = to_mat3(&dcm_b2i)?;
        let dcm_dr = dcm_dr.map(|d| to_dcm_derivative(&d)).transpose()?;
        let dcm_dv = dcm_dv.map(|d| to_dcm_derivative(&d)).transpose()?;
        let normals = normals.as_array();
        let eta = Dual64::from(eta_srp);
        let mut jac_r = [0.0_f64; 9];
        let mut jac_v = [0.0_f64; 9];
        for k in 0..3 {
            // seed the k-th position component (derivative = 1), others constant
            let mut p = sun2body.map(Dual64::from);
            p[k] = p[k].derivative();
            let a = self.accel(&seed_dcm(&c, dcm_dr.as_ref(), k), p, eta, mass, &normals)?;
            for i in 0..3 {
                jac_r[3 * i + k] = a[i].eps; // d(a_i)/d(r_k)
            }
            if let Some(d) = dcm_dv.as_ref() {
                let p = sun2body.map(Dual64::from);
                let a = self.accel(&seed_dcm(&c, Some(d), k), p, eta, mass, &normals)?;
                for i in 0..3 {
                    jac_v[3 * i + k] = a[i].eps; // d(a_i)/d(v_k)
                }
            }
        }
        let p = sun2body.map(Dual64::from);
        let a = self.accel(&seed_dcm(&c, None, 0), p, eta.derivative(), mass, &normals)?;
        Ok((jac_r, jac_v, a.map(|x| x.eps)))
    }
}

/// Per-evaluation plate geometry shared by the acceleration and Jacobian.
struct PlateGeometry<T> {
    n: usize,
    s: [T; 3],
    r: T,
    nh: Vec<[f64; 3]>,
    n_dot_s: Vec<T>,
    a_i: Vec<T>,
    b_i: Vec<T>,
    d_i: Vec<T>,
}

/// rust-only methods
impl NPlateSRPCore {
    /// Inertial acceleration: s = DCM^T (-sun2body), the SC-fixed
    /// T = T1 + T2 + T3 (`body_accel`), then DCM @ T.
    fn accel<T: DualNum<f64> + Copy>(
        &self,
        m: &[[T; 3]; 3],
        sun2body: [T; 3],
        eta_srp: T,
        mass: f64,
        normals: &ArrayView2<f64>,
    ) -> PyResult<[T; 3]> {
        let s = rotate_t(m, [-sun2body[0], -sun2body[1], -sun2body[2]]);
        let g = self.prep(s, eta_srp, mass, normals)?;
        Ok(rotate(m, body_accel(&g)))
    }

    /// Unit normals, illumination geometry, and per-plate coefficients.
    fn prep<T: DualNum<f64> + Copy>(
        &self,
        s: [T; 3],
        eta_srp: T,
        mass: f64,
        normals: &ArrayView2<f64>,
    ) -> PyResult<PlateGeometry<T>> {
        let n = self.areas.len();
        if normals.nrows() != n || normals.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "normals must be ({n}, 3); got {:?}",
                normals.shape()
            )));
        }

        let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();

        let mut nh: Vec<[f64; 3]> = Vec::with_capacity(n);
        let mut n_dot_s: Vec<T> = Vec::with_capacity(n);
        for i in 0..n {
            let (nx, ny, nz) = (normals[[i, 0]], normals[[i, 1]], normals[[i, 2]]);
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            let mut h = [nx / norm, ny / norm, nz / norm];
            let mut p = T::from(h[0]) * s[0] + T::from(h[1]) * s[1] + T::from(h[2]) * s[2];
            // a double-sided plate lit from behind: the back face, normal -h
            if self.double_sided[i] && p.re() < 0.0 {
                h = [-h[0], -h[1], -h[2]];
                p = -p;
            }
            n_dot_s.push(p);
            nh.push(h);
        }

        // C_i = P_SRP (eta/m) A_i, then the illuminated-plate coefficients
        let c0 = T::from(self.p_srp) * eta_srp / T::from(mass);
        let zero = T::from(0.0);
        let mut a_i: Vec<T> = Vec::with_capacity(n);
        let mut b_i: Vec<T> = Vec::with_capacity(n);
        let mut d_i: Vec<T> = Vec::with_capacity(n);
        for i in 0..n {
            if n_dot_s[i].re() > 0.0 {
                let c = c0 * T::from(self.areas[i]);
                a_i.push(-c * T::from(1.0 - self.spec_ref[i]));
                b_i.push(T::from(-2.0 / 3.0 * self.diff_ref[i]) * c);
                d_i.push(T::from(-2.0 * self.spec_ref[i]) * c);
            } else {
                a_i.push(zero);
                b_i.push(zero);
                d_i.push(zero);
            }
        }

        Ok(PlateGeometry {
            n,
            s,
            r,
            nh,
            n_dot_s,
            a_i,
            b_i,
            d_i,
        })
    }
}

/// SC-fixed acceleration T = T1 + T2 + T3 [km/s^2]:
/// T1 = (sum_i A_i (n.s) / r^4) s,
/// T2 = sum_i B_i (n.s) n_i / r^3,  T3 = sum_i D_i (n.s)^2 n_i / r^4.
fn body_accel<T: DualNum<f64> + Copy>(g: &PlateGeometry<T>) -> [T; 3] {
    let r2 = g.r * g.r;
    let r3 = r2 * g.r;
    let r4 = r2 * r2;

    let zero = T::from(0.0);
    let mut sap = zero;
    let mut sum2 = [zero; 3];
    let mut sum3 = [zero; 3];
    for i in 0..g.n {
        let p = g.n_dot_s[i];
        sap = sap + g.a_i[i] * p;
        let b_p = g.b_i[i] * p;
        let d_p2 = g.d_i[i] * p * p;
        for j in 0..3 {
            let h = T::from(g.nh[i][j]);
            sum2[j] = sum2[j] + b_p * h;
            sum3[j] = sum3[j] + d_p2 * h;
        }
    }

    let mut acc_bf = [zero; 3];
    for j in 0..3 {
        acc_bf[j] = sap / r4 * g.s[j] + sum2[j] / r3 + sum3[j] / r4;
    }
    acc_bf
}

/// SC-fixed Jacobian dT/ds [1/s^2] of `body_accel`.
fn body_jacobian(g: &PlateGeometry<f64>) -> [[f64; 3]; 3] {
    let r2 = g.r * g.r;
    let r3 = r2 * g.r;
    let r4 = r2 * r2;
    let r5 = r4 * g.r;
    let r6 = r4 * r2;

    // vecAn = sum_i A_i n_i ; SAp = sum_i A_i (n.s)
    // MB = sum_i B_i n_i n_i^T ; vecBpn = sum_i B_i (n.s) n_i
    // MDp = sum_i D_i (n.s) n_i n_i^T ; vecDp2n = sum_i D_i (n.s)^2 n_i
    let mut vec_an = [0.0_f64; 3];
    let mut sap = 0.0_f64;
    let mut mb = [[0.0_f64; 3]; 3];
    let mut mdp = [[0.0_f64; 3]; 3];
    let mut vec_bpn = [0.0_f64; 3];
    let mut vec_dp2n = [0.0_f64; 3];
    for i in 0..g.n {
        let h = g.nh[i];
        let p = g.n_dot_s[i];
        sap += g.a_i[i] * p;
        for a in 0..3 {
            vec_an[a] += g.a_i[i] * h[a];
            vec_bpn[a] += g.b_i[i] * p * h[a];
            vec_dp2n[a] += g.d_i[i] * p * p * h[a];
            for b in 0..3 {
                mb[a][b] += g.b_i[i] * h[a] * h[b];
                mdp[a][b] += g.d_i[i] * p * h[a] * h[b];
            }
        }
    }

    let mut jac_bf = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let eye = if a == b { sap } else { 0.0 };
            let dt1 = (g.s[a] * vec_an[b] + eye) / r4 - 4.0 * sap / r6 * g.s[a] * g.s[b];
            let dt2 = mb[a][b] / r3 - 3.0 / r5 * vec_bpn[a] * g.s[b];
            let dt3 = 2.0 / r4 * mdp[a][b] - 4.0 / r6 * vec_dp2n[a] * g.s[b];
            jac_bf[a][b] = dt1 + dt2 + dt3;
        }
    }
    jac_bf
}

/// Attitude part of da/dq, row-major, for a DCM that depends on q (the
/// position or the velocity), with d[j][i][k] = dDCM[i, j]/dq_k:
///
///   da_i/dq_k = sum_j T_j dDCM[i, j]/dq_k + (DCM @ dT/ds @ ds/dq)_ik,
///   ds_b/dq_k = -sum_c dDCM[c, b]/dq_k sun2body_c,
///
/// i.e. the rotating SC-fixed acceleration plus the Sun direction moving in
/// the rotating SC frame.
fn attitude_jacobian(
    m: &[[f64; 3]; 3],
    acc_bf: &[f64; 3],
    jac_bf: &[[f64; 3]; 3],
    sun2body: [f64; 3],
    d: &[[[f64; 3]; 3]; 3],
) -> [f64; 9] {
    let mut ds = [[0.0_f64; 3]; 3];
    for b in 0..3 {
        for k in 0..3 {
            let mut v = 0.0_f64;
            for c in 0..3 {
                v += d[b][c][k] * sun2body[c];
            }
            ds[b][k] = -v;
        }
    }
    let mut mj = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for b in 0..3 {
            for a in 0..3 {
                mj[i][b] += m[i][a] * jac_bf[a][b];
            }
        }
    }
    let mut out = [0.0_f64; 9];
    for i in 0..3 {
        for k in 0..3 {
            let mut v = 0.0_f64;
            for j in 0..3 {
                v += acc_bf[j] * d[j][i][k] + mj[i][j] * ds[j][k];
            }
            out[i * 3 + k] = v;
        }
    }
    out
}

/// Dual DCM with the k-th column of the derivative seeded:
/// eps[i][j] = d[j][i][k], or zero without a derivative.
fn seed_dcm(c: &[[f64; 3]; 3], d: Option<&[[[f64; 3]; 3]; 3]>, k: usize) -> [[Dual64; 3]; 3] {
    let mut out = [[Dual64::from(0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = Dual64::new(c[i][j], d.map_or(0.0, |d| d[j][i][k]));
        }
    }
    out
}

/// Read the SC-fixed -> inertial DCM into a fixed array.
fn to_mat3(m: &PyReadonlyArray2<'_, f64>) -> PyResult<[[f64; 3]; 3]> {
    let v = m.as_array();
    if v.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dcm_b2i must be (3, 3); got {:?}",
            v.shape()
        )));
    }
    Ok([
        [v[[0, 0]], v[[0, 1]], v[[0, 2]]],
        [v[[1, 0]], v[[1, 1]], v[[1, 2]]],
        [v[[2, 0]], v[[2, 1]], v[[2, 2]]],
    ])
}

/// Read a (3, 3, 3) DCM derivative, d[j][i][k] = dDCM[i, j]/dq_k.
fn to_dcm_derivative(d: &PyReadonlyArray3<'_, f64>) -> PyResult<[[[f64; 3]; 3]; 3]> {
    let v = d.as_array();
    if v.shape() != [3, 3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "DCM derivative must be (3, 3, 3); got {:?}",
            v.shape()
        )));
    }
    let mut out = [[[0.0_f64; 3]; 3]; 3];
    for j in 0..3 {
        for i in 0..3 {
            for k in 0..3 {
                out[j][i][k] = v[[j, i, k]];
            }
        }
    }
    Ok(out)
}

/// m @ v
fn rotate<T: DualNum<f64> + Copy>(m: &[[T; 3]; 3], v: [T; 3]) -> [T; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// m^T @ v
fn rotate_t<T: DualNum<f64> + Copy>(m: &[[T; 3]; 3], v: [T; 3]) -> [T; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}
