// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC
use numpy::ndarray::{Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

use num_dual::{Dual64, DualNum};

//============================================================================
// Spherical Harmonics Core
//============================================================================
/// Body-fixed spherical-harmonics kernels (normalized coefficients only).
///
/// This is the computational core of `SphericalHarmonicsGravity`: the
/// Pines/Eckman B_nm recurrence, the acceleration sum, the position Jacobian,
/// and the partials with respect to the C_nm / S_nm coefficients.  Everything
/// frame- or SPICE-related (rotations, ephemerides, estimated-parameter
/// handling) stays on the Python side, which passes in the body-fixed
/// position and the effective mu / C_nm / S_nm for each evaluation.
///
/// All per-(n, m) normalization factors — which depend only on degree/order,
/// never on position — are precomputed once in the constructor and reused on
/// every evaluation, so the hot loops do array lookups instead of tens of
/// thousands of `sqrt` calls per RHS step.
///
/// The `*_rf` methods additionally rotate the body-fixed result into the
/// reference frame (given the body-fixed -> reference-frame DCM) so the Python
/// side does not pay numpy overhead for those 3x3 products.
///
/// Floating-point expressions intentionally mirror the Python implementation
/// term-by-term (same association order, `powf` for `**`) so results match
/// the pure-Python reference to machine precision.
///
/// The B_nm recurrence and the acceleration sum are generic over
/// `num_dual::DualNum<f64>` (`f64` in production, `Dual64` for the `*_ad_*`
/// checks), so the dual-number partials differentiate the very expressions
/// that compute the acceleration.
#[pyclass]
#[derive(Clone)]
pub struct SphericalHarmonicsCore {
    pub order: usize,
    pub r_ref: f64,

    // Precomputed B_nm recurrence factors (indexed [n] / [n][m]), size order+3.
    bnm_diag: Vec<f64>,   // sectorial (n == m) norm factor, valid for n >= 1
    bnm_st1: Array2<f64>, // off-diagonal sqrt_term_1
    bnm_st2: Array2<f64>, // off-diagonal sqrt_term_2 (valid for n >= 2)

    // Precomputed acceleration / C-S-partial factors (indexed [n][m], size order+1).
    g1: Array2<f64>,
    g2: Array2<f64>,
    g3: Array2<f64>,

    // Precomputed Jacobian factors (indexed [n][m], size order+1).
    s1: Array2<f64>,
    s2: Array2<f64>,
    s3: Array2<f64>,
    s4: Array2<f64>,
    s5: Array2<f64>,
    s6: Array2<f64>,
}

/// methods exposed to python
#[pymethods]
impl SphericalHarmonicsCore {
    /// class constructor — precomputes all degree/order-only factors.
    #[new]
    #[pyo3(signature = (order, r_ref))]
    pub fn __new__(order: usize, r_ref: f64) -> Self {
        let n_size = order + 3;

        // --- B_nm recurrence factors ---
        let mut bnm_diag = vec![0.0_f64; n_size];
        for n in 1..n_size {
            let nf = n as f64;
            bnm_diag[n] = ((1.0 + Self::kron(1, n)) * (2.0 * nf + 1.0) / (2.0 * nf)).sqrt();
        }
        let mut bnm_st1 = Array2::<f64>::zeros((n_size, n_size));
        let mut bnm_st2 = Array2::<f64>::zeros((n_size, n_size));
        for m in 0..n_size {
            for n in (m + 1)..n_size {
                let nf = n as f64;
                let nm_sq = (n * n - m * m) as f64;
                bnm_st1[[n, m]] = ((4.0 * nf * nf - 1.0) / nm_sq).sqrt();
                if n >= 2 {
                    let m_sq = (m * m) as f64;
                    bnm_st2[[n, m]] = ((2.0 * nf + 1.0) * ((nf - 1.0) * (nf - 1.0) - m_sq)
                        / ((2.0 * nf - 3.0) * nm_sq))
                        .sqrt();
                }
            }
        }

        // --- acceleration / partial / Jacobian factors ---
        let sz = order + 1;
        let mut g1 = Array2::<f64>::zeros((sz, sz));
        let mut g2 = Array2::<f64>::zeros((sz, sz));
        let mut g3 = Array2::<f64>::zeros((sz, sz));
        let mut s1 = Array2::<f64>::zeros((sz, sz));
        let mut s2 = Array2::<f64>::zeros((sz, sz));
        let mut s3 = Array2::<f64>::zeros((sz, sz));
        let mut s4 = Array2::<f64>::zeros((sz, sz));
        let mut s5 = Array2::<f64>::zeros((sz, sz));
        let mut s6 = Array2::<f64>::zeros((sz, sz));
        for n in 0..=order {
            let nf = n as f64;
            for m in 0..=n {
                let mf = m as f64;
                let (a, b, c) = Self::g_ext_calc(nf, mf, m);
                g1[[n, m]] = a;
                g2[[n, m]] = b;
                g3[[n, m]] = c;

                let d0 = Self::kron(0, m);
                let d1 = Self::kron(1, m);
                let d2 = Self::kron(2, m);
                s1[[n, m]] = (0.5
                    * (2.0 - d0)
                    * (2.0 * nf + 1.0)
                    * (nf + mf + 4.0)
                    * (nf + mf + 3.0)
                    * (nf + mf + 2.0)
                    * (nf + mf + 1.0)
                    / (2.0 * nf + 5.0))
                    .sqrt();
                s2[[n, m]] = ((2.0 * nf + 1.0)
                    * (nf + mf + 2.0)
                    * (nf + mf + 1.0)
                    * (nf - mf + 2.0)
                    * (nf - mf + 1.0)
                    / (2.0 * nf + 5.0))
                    .sqrt();
                s3[[n, m]] = ((2.0 - d0)
                    * (2.0 * nf + 1.0)
                    * (nf - mf + 4.0)
                    * (nf - mf + 3.0)
                    * (nf - mf + 2.0)
                    * (nf - mf + 1.0)
                    / ((2.0 - d2) * (2.0 * nf + 5.0)))
                    .sqrt();
                s4[[n, m]] = (0.5
                    * (2.0 - d0)
                    * (2.0 * nf + 1.0)
                    * (nf + mf + 3.0)
                    * (nf + mf + 2.0)
                    * (nf + mf + 1.0)
                    * (nf - mf + 1.0)
                    / (2.0 * nf + 5.0))
                    .sqrt();
                s5[[n, m]] = ((2.0 - d0)
                    * (2.0 * nf + 1.0)
                    * (nf + mf + 1.0)
                    * (nf - mf + 3.0)
                    * (nf - mf + 2.0)
                    * (nf - mf + 1.0)
                    / ((2.0 - d1) * (2.0 * nf + 5.0)))
                    .sqrt();
                s6[[n, m]] = ((2.0 * nf + 1.0)
                    * (nf + mf + 2.0)
                    * (nf + mf + 1.0)
                    * (nf - mf + 2.0)
                    * (nf - mf + 1.0)
                    / (2.0 * nf + 5.0))
                    .sqrt();
            }
        }

        Self {
            order,
            r_ref,
            bnm_diag,
            bnm_st1,
            bnm_st2,
            g1,
            g2,
            g3,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
        }
    }

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!(
            "SphericalHarmonicsCore | order: {} | r_ref: {} km",
            self.order, self.r_ref
        )
    }

    /// shallow/deep copy support so owning Python objects stay copyable
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    /// Normalized exterior B_nm matrices for a body-fixed position [km].
    /// Returns (real, imag), each of shape (order+3, order+3).
    fn compute_bnm<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
        let (br, bi) = self.bnm(r_bf);
        (br.into_pyarray(py), bi.into_pyarray(py))
    }

    /// Acceleration in the body-fixed frame [km/s^2].
    /// Returns ([ax, ay, az], bnm_real, bnm_imag).
    fn compute_acceleration_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        [f64; 3],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        let (br, bi) = self.bnm(r_bf);
        let acc = self.accel(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        Ok((acc, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// Acceleration rotated into the reference frame [km/s^2].
    /// `r_b2rf` is the body-fixed -> reference-frame DCM.
    /// Returns ([ax, ay, az]_rf, bnm_real, bnm_imag).
    fn compute_acceleration_rf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        r_b2rf: PyReadonlyArray2<'py, f64>,
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        [f64; 3],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        let (br, bi) = self.bnm(r_bf);
        let acc_bf = self.accel(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        let r = Self::to_mat3(&r_b2rf);
        let acc_rf = Self::rotate_vec(&r, acc_bf);
        Ok((acc_rf, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// Fused acceleration + da/dr on a single B_nm pass, body-fixed frame.
    /// The jacobian uses `mu_jac`/`cnm_jac`/`snm_jac` when given (nominal
    /// coefficients while the acceleration uses estimated ones), otherwise
    /// the acceleration's values.
    /// Returns ([ax, ay, az], [d00..d22], bnm_real, bnm_imag).
    #[pyo3(signature = (r_bf, mu, cnm, snm, mu_jac=None, cnm_jac=None, snm_jac=None))]
    fn compute_accel_and_jacobian_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
        mu_jac: Option<f64>,
        cnm_jac: Option<PyReadonlyArray2<'py, f64>>,
        snm_jac: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<(
        [f64; 3],
        [f64; 9],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        if let (Some(cj), Some(sj)) = (&cnm_jac, &snm_jac) {
            self.check_coeff_dims(cj, sj)?;
        }
        let (br, bi) = self.bnm(r_bf);
        let acc = self.accel(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        let jac = self.jacobian_with(&br, &bi, mu, &cnm, &snm, mu_jac, &cnm_jac, &snm_jac);
        Ok((acc, jac, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// Fused acceleration + da/dr, both rotated into the reference frame.
    /// `r_b2rf` is the body-fixed -> reference-frame DCM. Acceleration is
    /// rotated as R @ a; the Jacobian as R @ J @ R^T.
    /// Returns ([ax, ay, az]_rf, [d00..d22]_rf, bnm_real, bnm_imag).
    #[pyo3(signature = (r_bf, r_b2rf, mu, cnm, snm, mu_jac=None, cnm_jac=None, snm_jac=None))]
    fn compute_accel_and_jacobian_rf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        r_b2rf: PyReadonlyArray2<'py, f64>,
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
        mu_jac: Option<f64>,
        cnm_jac: Option<PyReadonlyArray2<'py, f64>>,
        snm_jac: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<(
        [f64; 3],
        [f64; 9],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        if let (Some(cj), Some(sj)) = (&cnm_jac, &snm_jac) {
            self.check_coeff_dims(cj, sj)?;
        }
        let (br, bi) = self.bnm(r_bf);
        let acc_bf = self.accel(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        let jac_bf = self.jacobian_with(&br, &bi, mu, &cnm, &snm, mu_jac, &cnm_jac, &snm_jac);
        let r = Self::to_mat3(&r_b2rf);
        let acc_rf = Self::rotate_vec(&r, acc_bf);
        let jac_rf = Self::rotate_jac(&r, &jac_bf);
        Ok((acc_rf, jac_rf, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// da/dr in the body-fixed frame [1/s^2], row-major.
    /// Returns ([d00, d01, d02, d10, d11, d12, d20, d21, d22], bnm_real, bnm_imag).
    fn compute_jacobian_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        [f64; 9],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        let (br, bi) = self.bnm(r_bf);
        let jac = self.jacobian(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        Ok((jac, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// da/dr rotated into the reference frame [1/s^2], row-major (R @ J @ R^T).
    /// Returns (jac_rf[9], bnm_real, bnm_imag).
    fn compute_jacobian_rf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        r_b2rf: PyReadonlyArray2<'py, f64>,
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        [f64; 9],
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        self.check_coeff_dims(&cnm, &snm)?;
        let (br, bi) = self.bnm(r_bf);
        let jac_bf = self.jacobian(&br, &bi, mu, &cnm.as_array(), &snm.as_array());
        let r = Self::to_mat3(&r_b2rf);
        let jac_rf = Self::rotate_jac(&r, &jac_bf);
        Ok((jac_rf, br.into_pyarray(py), bi.into_pyarray(py)))
    }

    /// da/dr in the body-fixed frame [1/s^2], row-major, via forward-mode
    /// automatic differentiation of the acceleration (dual numbers).
    ///
    /// Independent of the analytical `compute_jacobian_bf`; the two must agree,
    /// which verifies the hand-coded position-partial (`s1..s6`) formulas.
    fn compute_jacobian_ad_bf<'py>(
        &self,
        r_bf: [f64; 3],
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<[f64; 9]> {
        self.check_coeff_dims(&cnm, &snm)?;
        let c = cnm.as_array().mapv(Dual64::from);
        let s = snm.as_array().mapv(Dual64::from);
        let mut jac = [0.0_f64; 9];
        for k in 0..3 {
            // seed the k-th position component (derivative = 1), others constant
            let mut r = r_bf.map(Dual64::from);
            r[k] = r[k].derivative();
            let (br, bi) = self.bnm(r);
            let a = self.accel(&br, &bi, Dual64::from(mu), &c.view(), &s.view());
            jac[k] = a[0].eps; // d(a_x)/d(r_k)
            jac[3 + k] = a[1].eps; // d(a_y)/d(r_k)
            jac[6 + k] = a[2].eps; // d(a_z)/d(r_k)
        }
        Ok(jac)
    }

    /// da/dGM in the reference frame [km/s^2 per km^3/s^2], via forward-mode
    /// automatic differentiation (dual numbers) with GM as the seed.
    ///
    /// Differentiates the `compute_acceleration_rf` expressions and, when
    /// `r_origin_sph` is given (the harmonics body is not the propagation
    /// origin), the indirect term `-GM r_o / |r_o|^3` that
    /// `SphericalHarmonicsGravity.compute_acceleration` subtracts.
    /// Independent of the analytical `a / GM` partial; the two must agree.
    #[pyo3(signature = (r_bf, r_b2rf, mu, cnm, snm, r_origin_sph=None))]
    fn compute_partial_by_mu_ad_rf<'py>(
        &self,
        r_bf: [f64; 3],
        r_b2rf: PyReadonlyArray2<'py, f64>,
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
        r_origin_sph: Option<[f64; 3]>,
    ) -> PyResult<[f64; 3]> {
        self.check_coeff_dims(&cnm, &snm)?;
        let c = cnm.as_array().mapv(Dual64::from);
        let s = snm.as_array().mapv(Dual64::from);
        let mu_d = Dual64::from(mu).derivative();
        let (br, bi) = self.bnm(r_bf.map(Dual64::from));
        let acc_bf = self.accel(&br, &bi, mu_d, &c.view(), &s.view());
        let r = Self::to_mat3(&r_b2rf).map(|row| row.map(Dual64::from));
        let mut acc = Self::rotate_vec(&r, acc_bf);
        if let Some(ro) = r_origin_sph {
            let r_norm = (ro[0] * ro[0] + ro[1] * ro[1] + ro[2] * ro[2]).sqrt();
            if r_norm > 0.0 {
                let r_norm3 = Dual64::from(r_norm.powf(3.0));
                for i in 0..3 {
                    acc[i] = acc[i] - mu_d * Dual64::from(ro[i]) / r_norm3;
                }
            }
        }
        Ok(acc.map(|a| a.eps))
    }

    /// da/dC_nm and da/dS_nm in the body-fixed frame [km/s^2], each of shape
    /// (3, order+1, order+1), via forward-mode automatic differentiation
    /// (dual numbers): one acceleration pass per coefficient, seeding it alone.
    ///
    /// Independent of the analytical `compute_partial_by_c_bf` /
    /// `compute_partial_by_s_bf`; the two must agree. S_n0 is left at zero,
    /// as it is there.
    fn compute_partial_by_cs_ad_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
        self.check_coeff_dims(&cnm, &snm)?;
        let mut c = cnm.as_array().mapv(Dual64::from);
        let mut s = snm.as_array().mapv(Dual64::from);
        // B_nm does not depend on the coefficients: one recurrence serves all seeds.
        let (br, bi) = self.bnm(r_bf.map(Dual64::from));
        let mu_d = Dual64::from(mu);
        let size = self.order + 1;
        let mut d_c = Array3::<f64>::zeros((3, size, size));
        let mut d_s = Array3::<f64>::zeros((3, size, size));
        for n in 0..=self.order {
            for m in 0..=n {
                c[[n, m]].eps = 1.0;
                let a = self.accel(&br, &bi, mu_d, &c.view(), &s.view());
                c[[n, m]].eps = 0.0;
                for i in 0..3 {
                    d_c[[i, n, m]] = a[i].eps;
                }

                if m == 0 {
                    continue;
                }
                s[[n, m]].eps = 1.0;
                let a = self.accel(&br, &bi, mu_d, &c.view(), &s.view());
                s[[n, m]].eps = 0.0;
                for i in 0..3 {
                    d_s[[i, n, m]] = a[i].eps;
                }
            }
        }
        Ok((d_c.into_pyarray(py), d_s.into_pyarray(py)))
    }

    /// Pole partials in the reference frame via forward-mode automatic
    /// differentiation (dual numbers) of the whole rotated acceleration, the
    /// IAU pole rotation included: da/d(pole_ra) and da/d(pole_dec)
    /// [km/s^2 per rad], da/d(pole_pm) [km/s^2 per deg/day].
    ///
    /// The body-fixed -> reference-frame DCM is rebuilt from the pole angles as
    /// `SphericalHarmonicsGravity._get_R_b2rf_effective` builds it when the
    /// pole is estimated: `m_j2rf @ Rz(-(90+RA)) @ Rx(-(90-DEC)) @ Rz(-W)` with
    /// `W = w0 + pm_rate * d_days` [deg]. `r_rel_rf` is the position relative
    /// to the harmonics body in the reference frame [km]. Independent of the
    /// analytical `_compute_partial_by_pole_*`; the two must agree.
    ///
    /// Returns (a_rf, [da/dRA, da/dDEC, da/dPM]); the acceleration value lets
    /// the caller confirm this rotation is the one the model uses.
    #[pyo3(signature = (r_rel_rf, m_j2rf, ra, dec, w0, pm_rate, d_days, mu, cnm, snm))]
    fn compute_partial_by_pole_ad_rf<'py>(
        &self,
        r_rel_rf: [f64; 3],
        m_j2rf: PyReadonlyArray2<'py, f64>,
        ra: f64,
        dec: f64,
        w0: f64,
        pm_rate: f64,
        d_days: f64,
        mu: f64,
        cnm: PyReadonlyArray2<'py, f64>,
        snm: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<([f64; 3], [[f64; 3]; 3])> {
        self.check_coeff_dims(&cnm, &snm)?;
        let c = cnm.as_array().mapv(Dual64::from);
        let s = snm.as_array().mapv(Dual64::from);
        let m_j2rf = Self::to_mat3(&m_j2rf).map(|row| row.map(Dual64::from));
        let r_rel = r_rel_rf.map(Dual64::from);
        let mut acc = [0.0_f64; 3];
        let mut partials = [[0.0_f64; 3]; 3];
        for k in 0..3 {
            // seed RA, DEC or the prime-meridian rate (derivative = 1)
            let mut p = [Dual64::from(ra), Dual64::from(dec), Dual64::from(pm_rate)];
            p[k] = p[k].derivative();
            let w_deg = Dual64::from(w0) + p[2] * Dual64::from(d_days);
            let r_b2rf = Self::mat_mul(&m_j2rf, &Self::iau_b2j(p[0], p[1], w_deg));
            let r_bf = Self::rotate_vec(&Self::transpose(&r_b2rf), r_rel);
            let (br, bi) = self.bnm(r_bf);
            let acc_bf = self.accel(&br, &bi, Dual64::from(mu), &c.view(), &s.view());
            let a = Self::rotate_vec(&r_b2rf, acc_bf);
            acc = a.map(|x| x.re);
            partials[k] = a.map(|x| x.eps); // d(a_rf)/d(p_k)
        }
        Ok((acc, partials))
    }

    /// da/dC_nm in the body-fixed frame [km/s^2], shape (3, order+1, order+1).
    /// Returns (partials, bnm_real, bnm_imag).
    fn compute_partial_by_c_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
    ) -> (
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    ) {
        let (br, bi) = self.bnm(r_bf);
        let partials = self.partial_by_c(&br, &bi, mu);
        (
            partials.into_pyarray(py),
            br.into_pyarray(py),
            bi.into_pyarray(py),
        )
    }

    /// da/dS_nm in the body-fixed frame [km/s^2], shape (3, order+1, order+1).
    /// Returns (partials, bnm_real, bnm_imag).
    fn compute_partial_by_s_bf<'py>(
        &self,
        py: Python<'py>,
        r_bf: [f64; 3],
        mu: f64,
    ) -> (
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    ) {
        let (br, bi) = self.bnm(r_bf);
        let partials = self.partial_by_s(&br, &bi, mu);
        (
            partials.into_pyarray(py),
            br.into_pyarray(py),
            bi.into_pyarray(py),
        )
    }
}

/// rust-only methods
impl SphericalHarmonicsCore {
    /// Kronecker delta as f64
    fn kron(i: usize, j: usize) -> f64 {
        if i == j { 1.0 } else { 0.0 }
    }

    /// Read a 3x3 body-fixed -> reference-frame DCM into a fixed array.
    fn to_mat3(r: &PyReadonlyArray2<'_, f64>) -> [[f64; 3]; 3] {
        let v = r.as_array();
        [
            [v[[0, 0]], v[[0, 1]], v[[0, 2]]],
            [v[[1, 0]], v[[1, 1]], v[[1, 2]]],
            [v[[2, 0]], v[[2, 1]], v[[2, 2]]],
        ]
    }

    /// r @ v for a 3-vector (matches numpy `R @ v` term order).
    fn rotate_vec<T: DualNum<f64> + Copy>(r: &[[T; 3]; 3], v: [T; 3]) -> [T; 3] {
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }

    /// a @ b for 3x3 matrices.
    fn mat_mul<T: DualNum<f64> + Copy>(a: &[[T; 3]; 3], b: &[[T; 3]; 3]) -> [[T; 3]; 3] {
        let mut out = [[T::from(0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        out
    }

    /// m^T for a 3x3 matrix.
    fn transpose<T: DualNum<f64> + Copy>(m: &[[T; 3]; 3]) -> [[T; 3]; 3] {
        [
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ]
    }

    /// Body-fixed -> J2000 DCM from the IAU pole angles (RA, DEC [rad], W [deg]),
    /// the transpose of the NAIF J2000 -> body 3-1-3 sequence
    /// `Rz(W) @ Rx(90-DEC) @ Rz(90+RA)` of passive rotations.
    /// Mirrors `SphericalHarmonicsGravity._build_R_b2j_iau` (fed the state's
    /// radians through `math.degrees`); generic so the pole can be seeded.
    fn iau_b2j<T: DualNum<f64> + Copy>(ra: T, dec: T, w_deg: T) -> [[T; 3]; 3] {
        let rad_to_deg = T::from(180.0 / std::f64::consts::PI);
        let deg_to_rad = T::from(std::f64::consts::PI / 180.0);
        let phi = (T::from(90.0) + ra * rad_to_deg) * deg_to_rad;
        let theta = (T::from(90.0) - dec * rad_to_deg) * deg_to_rad;
        let psi = w_deg * deg_to_rad;

        let (cp, sp) = (phi.cos(), phi.sin());
        let (ct, st) = (theta.cos(), theta.sin());
        let (cw, sw) = (psi.cos(), psi.sin());
        let (zero, one) = (T::from(0.0), T::from(1.0));

        // passive rotation matrices Ri(angle)
        let rz_phi = [[cp, sp, zero], [-sp, cp, zero], [zero, zero, one]];
        let rx_theta = [[one, zero, zero], [zero, ct, st], [zero, -st, ct]];
        let rz_psi = [[cw, sw, zero], [-sw, cw, zero], [zero, zero, one]];
        Self::transpose(&Self::mat_mul(&Self::mat_mul(&rz_psi, &rx_theta), &rz_phi))
    }

    /// R @ J @ R^T for a row-major 3x3 Jacobian (matches numpy association).
    fn rotate_jac(r: &[[f64; 3]; 3], j: &[f64; 9]) -> [f64; 9] {
        // M = R @ J
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                m[i][k] = r[i][0] * j[k] + r[i][1] * j[3 + k] + r[i][2] * j[6 + k];
            }
        }
        // out = M @ R^T  =>  out[i][l] = sum_k M[i][k] * R[l][k]
        let mut out = [0.0_f64; 9];
        for i in 0..3 {
            for l in 0..3 {
                out[i * 3 + l] = m[i][0] * r[l][0] + m[i][1] * r[l][1] + m[i][2] * r[l][2];
            }
        }
        out
    }

    /// Normalized exterior B_nm recurrence (Pines/Eckman), body-fixed frame.
    fn bnm<T: DualNum<f64> + Copy>(&self, r_bf: [T; 3]) -> (Array2<T>, Array2<T>) {
        let n_size = self.order + 3;
        let (x, y, z) = (r_bf[0], r_bf[1], r_bf[2]);
        let r_sat = (x * x + y * y + z * z).sqrt();

        let inv_r = T::from(1.0) / r_sat;
        let x_r = x * inv_r;
        let y_r = y * inv_r;
        let z_r = z * inv_r;
        let rr = T::from(self.r_ref) * inv_r;
        let rr2 = rr * rr;

        let mut br = Array2::<T>::from_elem((n_size, n_size), T::from(0.0));
        let mut bi = Array2::<T>::from_elem((n_size, n_size), T::from(0.0));

        for m in 0..n_size {
            for n in m..n_size {
                if n == m {
                    if m == 0 {
                        br[[0, 0]] = rr;
                    } else {
                        let norm_factor = T::from(self.bnm_diag[n]);
                        let (pr, pi) = (br[[n - 1, n - 1]], bi[[n - 1, n - 1]]);
                        br[[n, n]] = norm_factor * rr * (x_r * pr - y_r * pi);
                        bi[[n, n]] = norm_factor * rr * (y_r * pr + x_r * pi);
                    }
                } else {
                    let sqrt_term_1 = T::from(self.bnm_st1[[n, m]]);
                    let (pr, pi) = (br[[n - 1, m]], bi[[n - 1, m]]);
                    if n >= 2 {
                        let sqrt_term_2 = T::from(self.bnm_st2[[n, m]]);
                        br[[n, m]] =
                            sqrt_term_1 * rr * z_r * pr - sqrt_term_2 * rr2 * br[[n - 2, m]];
                        bi[[n, m]] =
                            sqrt_term_1 * rr * z_r * pi - sqrt_term_2 * rr2 * bi[[n - 2, m]];
                    } else {
                        br[[n, m]] = sqrt_term_1 * rr * z_r * pr;
                        bi[[n, m]] = sqrt_term_1 * rr * z_r * pi;
                    }
                }
            }
        }
        (br, bi)
    }

    /// Acceleration sum over the normalized expansion, body-fixed frame.
    fn accel<T: DualNum<f64> + Copy>(
        &self,
        br: &Array2<T>,
        bi: &Array2<T>,
        mu: T,
        cnm: &ArrayView2<T>,
        snm: &ArrayView2<T>,
    ) -> [T; 3] {
        let mu_by_r2 = mu / T::from(self.r_ref.powf(2.0));
        let (mut x_ddot, mut y_ddot, mut z_ddot) = (T::from(0.0), T::from(0.0), T::from(0.0));

        for n in 0..=self.order {
            for m in 0..=n {
                let g_ext_1 = T::from(self.g1[[n, m]]);
                let g_ext_2 = T::from(self.g2[[n, m]]);
                let g_ext_3 = T::from(self.g3[[n, m]]);

                let c = cnm[[n, m]];
                let s = snm[[n, m]];

                if m == 0 {
                    x_ddot = x_ddot + mu_by_r2 * g_ext_1 * (-c * br[[n + 1, m + 1]]);
                    y_ddot = y_ddot + mu_by_r2 * g_ext_1 * (-c * bi[[n + 1, m + 1]]);
                } else {
                    x_ddot = x_ddot
                        + mu_by_r2
                            * (g_ext_1 * (-c * br[[n + 1, m + 1]] - s * bi[[n + 1, m + 1]])
                                + g_ext_2 * (c * br[[n + 1, m - 1]] + s * bi[[n + 1, m - 1]]));
                    y_ddot = y_ddot
                        + mu_by_r2
                            * (g_ext_1 * (s * br[[n + 1, m + 1]] - c * bi[[n + 1, m + 1]])
                                + g_ext_2 * (s * br[[n + 1, m - 1]] - c * bi[[n + 1, m - 1]]));
                }

                z_ddot = z_ddot + mu_by_r2 * g_ext_3 * (c * br[[n + 1, m]] + s * bi[[n + 1, m]]);
            }
        }
        [x_ddot, y_ddot, z_ddot]
    }

    /// da/dr for the normalized expansion, body-fixed frame, row-major 3x3.
    fn jacobian(
        &self,
        br: &Array2<f64>,
        bi: &Array2<f64>,
        mu: f64,
        cnm: &ArrayView2<f64>,
        snm: &ArrayView2<f64>,
    ) -> [f64; 9] {
        let mu_r3 = mu / self.r_ref.powf(3.0);
        let (mut dxddot_dx, mut dzddot_dz) = (0.0_f64, 0.0_f64);
        let (mut dxddot_dy, mut dxddot_dz, mut dyddot_dz) = (0.0_f64, 0.0_f64, 0.0_f64);

        for n in 0..=self.order {
            for m in 0..=n {
                let c = cnm[[n, m]];
                let s = snm[[n, m]];

                let s1 = self.s1[[n, m]];
                let s2 = self.s2[[n, m]];
                let s3 = self.s3[[n, m]];
                let s4 = self.s4[[n, m]];
                let s5 = self.s5[[n, m]];
                let s6 = self.s6[[n, m]];

                if m == 0 {
                    dxddot_dx += mu_r3 * 0.5 * c * (s1 * br[[n + 2, 2]] - s2 * br[[n + 2, 0]]);
                    dxddot_dy += mu_r3 * 0.5 * c * s1 * bi[[n + 2, 2]];
                } else if m == 1 {
                    dxddot_dx += mu_r3
                        * 0.25
                        * (c * (s1 * br[[n + 2, 3]] - 3.0 * s2 * br[[n + 2, 1]])
                            + s * (s1 * bi[[n + 2, 3]] - s2 * bi[[n + 2, 1]]));
                    dxddot_dy += mu_r3
                        * 0.25
                        * (-s * (s1 * br[[n + 2, 3]] + s2 * br[[n + 2, 1]])
                            + c * (s1 * bi[[n + 2, 3]] - s2 * bi[[n + 2, 1]]));
                } else {
                    dxddot_dx += mu_r3
                        * 0.25
                        * (c * (s1 * br[[n + 2, m + 2]] - 2.0 * s2 * br[[n + 2, m]]
                            + s3 * br[[n + 2, m - 2]])
                            + s * (s1 * bi[[n + 2, m + 2]] - 2.0 * s2 * bi[[n + 2, m]]
                                + s3 * bi[[n + 2, m - 2]]));
                    dxddot_dy += mu_r3
                        * 0.25
                        * (-s * (s1 * br[[n + 2, m + 2]] - s3 * br[[n + 2, m - 2]])
                            + c * (s1 * bi[[n + 2, m + 2]] - s3 * bi[[n + 2, m - 2]]));
                }

                if m == 0 {
                    dxddot_dz += mu_r3 * s4 * c * br[[n + 2, 1]];
                    dyddot_dz += mu_r3 * s4 * c * bi[[n + 2, 1]];
                } else {
                    dxddot_dz += mu_r3
                        * 0.5
                        * (c * (s4 * br[[n + 2, m + 1]] - s5 * br[[n + 2, m - 1]])
                            + s * (s4 * bi[[n + 2, m + 1]] - s5 * bi[[n + 2, m - 1]]));
                    dyddot_dz += mu_r3
                        * 0.5
                        * (-s * (s4 * br[[n + 2, m + 1]] + s5 * br[[n + 2, m - 1]])
                            + c * (s4 * bi[[n + 2, m + 1]] + s5 * bi[[n + 2, m - 1]]));
                }

                dzddot_dz += mu_r3 * s6 * (c * br[[n + 2, m]] + s * bi[[n + 2, m]]);
            }
        }

        let dyddot_dy = -dxddot_dx - dzddot_dz;

        [
            dxddot_dx, dxddot_dy, dxddot_dz,
            dxddot_dy, dyddot_dy, dyddot_dz,
            dxddot_dz, dyddot_dz, dzddot_dz,
        ]
    }

    /// Jacobian with an optional distinct (mu, C, S) set (used by the fused
    /// accel+jacobian calls, where the Jacobian is evaluated at the nominal
    /// coefficients while the acceleration uses the estimated ones).
    fn jacobian_with(
        &self,
        br: &Array2<f64>,
        bi: &Array2<f64>,
        mu: f64,
        cnm: &PyReadonlyArray2<'_, f64>,
        snm: &PyReadonlyArray2<'_, f64>,
        mu_jac: Option<f64>,
        cnm_jac: &Option<PyReadonlyArray2<'_, f64>>,
        snm_jac: &Option<PyReadonlyArray2<'_, f64>>,
    ) -> [f64; 9] {
        match (cnm_jac, snm_jac) {
            (Some(cj), Some(sj)) => self.jacobian(
                br,
                bi,
                mu_jac.unwrap_or(mu),
                &cj.as_array(),
                &sj.as_array(),
            ),
            _ => self.jacobian(
                br,
                bi,
                mu_jac.unwrap_or(mu),
                &cnm.as_array(),
                &snm.as_array(),
            ),
        }
    }

    /// da/dC_nm for the normalized expansion, body-fixed frame.
    fn partial_by_c(&self, br: &Array2<f64>, bi: &Array2<f64>, mu: f64) -> Array3<f64> {
        let size = self.order + 1;
        let mut partials = Array3::<f64>::zeros((3, size, size));
        let mu_by_r2 = mu / self.r_ref.powf(2.0);

        for n in 0..=self.order {
            for m in 0..=n {
                let g_ext_1 = self.g1[[n, m]];
                let g_ext_2 = self.g2[[n, m]];
                let g_ext_3 = self.g3[[n, m]];

                if m == 0 {
                    partials[[0, n, m]] = mu_by_r2 * g_ext_1 * (-br[[n + 1, m + 1]]);
                    partials[[1, n, m]] = mu_by_r2 * g_ext_1 * (-bi[[n + 1, m + 1]]);
                } else {
                    partials[[0, n, m]] = mu_by_r2
                        * (g_ext_1 * (-br[[n + 1, m + 1]]) + g_ext_2 * (br[[n + 1, m - 1]]));
                    partials[[1, n, m]] = mu_by_r2
                        * (g_ext_1 * (-bi[[n + 1, m + 1]]) + g_ext_2 * (-bi[[n + 1, m - 1]]));
                }

                partials[[2, n, m]] = mu_by_r2 * g_ext_3 * br[[n + 1, m]];
            }
        }
        partials
    }

    /// da/dS_nm for the normalized expansion, body-fixed frame.
    fn partial_by_s(&self, br: &Array2<f64>, bi: &Array2<f64>, mu: f64) -> Array3<f64> {
        let size = self.order + 1;
        let mut partials = Array3::<f64>::zeros((3, size, size));
        let mu_by_r2 = mu / self.r_ref.powf(2.0);

        for n in 0..=self.order {
            for m in 0..=n {
                if m == 0 {
                    continue;
                }
                let g_ext_1 = self.g1[[n, m]];
                let g_ext_2 = self.g2[[n, m]];
                let g_ext_3 = self.g3[[n, m]];

                partials[[0, n, m]] =
                    mu_by_r2 * (g_ext_1 * (-bi[[n + 1, m + 1]]) + g_ext_2 * (bi[[n + 1, m - 1]]));
                partials[[1, n, m]] =
                    mu_by_r2 * (g_ext_1 * (br[[n + 1, m + 1]]) + g_ext_2 * (br[[n + 1, m - 1]]));
                partials[[2, n, m]] = mu_by_r2 * g_ext_3 * bi[[n + 1, m]];
            }
        }
        partials
    }

    /// Normalized g_ext_1/2/3 factors (used only at construction to fill the
    /// precomputed tables shared by the acceleration and C/S partials).
    fn g_ext_calc(nf: f64, mf: f64, m: usize) -> (f64, f64, f64) {
        let d0 = Self::kron(0, m);
        let d1 = Self::kron(1, m);
        let g_ext_1 = 0.5
            * (1.0 + d0)
            * ((2.0 - d0) * (2.0 * nf + 1.0) * (nf + mf + 2.0) * (nf + mf + 1.0)
                / (2.0 * (2.0 * nf + 3.0)))
                .sqrt();
        let g_ext_2 = 0.5
            * ((2.0 - d0) * (2.0 * nf + 1.0) * (nf - mf + 2.0) * (nf - mf + 1.0)
                / ((2.0 - d1) * (2.0 * nf + 3.0)))
                .sqrt();
        let g_ext_3 =
            -((2.0 * nf + 1.0) * (nf + mf + 1.0) * (nf - mf + 1.0) / (2.0 * nf + 3.0)).sqrt();
        (g_ext_1, g_ext_2, g_ext_3)
    }

    /// Validate that the coefficient matrices cover the configured order.
    fn check_coeff_dims(
        &self,
        cnm: &PyReadonlyArray2<'_, f64>,
        snm: &PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let need = self.order + 1;
        let (c, s) = (cnm.as_array(), snm.as_array());
        if c.nrows() < need || c.ncols() < need || s.nrows() < need || s.ncols() < need {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cnm/Snm must be at least ({need}, {need}) for order {}; got Cnm {:?}, Snm {:?}",
                self.order,
                c.shape(),
                s.shape()
            )));
        }
        Ok(())
    }
}
