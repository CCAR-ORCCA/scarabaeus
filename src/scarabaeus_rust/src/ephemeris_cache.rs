// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};

// Chebyshev components
pub struct ChebyRecord {
    pub t_mid:    f64,
    pub t_rad:    f64,
    pub x_coeffs: Vec<f64>,
    pub y_coeffs: Vec<f64>,
    pub z_coeffs: Vec<f64>,
}

impl ChebyRecord {
    pub fn eval_pos(&self, t: f64) -> [f64; 3] {
        [clenshaw(&self.x_coeffs, self.t_mid, self.t_rad, t),
         clenshaw(&self.y_coeffs, self.t_mid, self.t_rad, t),
         clenshaw(&self.z_coeffs, self.t_mid, self.t_rad, t)]
    }

    pub fn contains(&self, t: f64) -> bool {
        t >= self.t_mid - self.t_rad && t <= self.t_mid + self.t_rad
    }
}

/// Evalate a given set of Chebyshev coefficients using the Clenshaw algorithm
pub fn clenshaw(coeffs: &[f64], t_mid: f64, t_rad: f64, t: f64) -> f64 {
    let s: f64 = (t - t_mid) / t_rad;
    let n: usize = coeffs.len();
    if n == 0 { return 0.0; }
    if n == 1 { return coeffs[0]; }

    let mut b_prev: f64 = 0.0;
    let mut b_curr: f64 = 0.0;
    for &c in coeffs[1..].iter().rev() {
        let b_next: f64 = c + 2.0 * s * b_curr - b_prev;
        b_prev = b_curr;
        b_curr = b_next;
    }
    coeffs[0] + s * b_curr - b_prev
}

//============================================================================
// Ephemeris Cache
//============================================================================
/// Chebyshev ephemeris for a set of bodies, read out of the loaded SPKs.
///
/// Positions only: the models that use them (third-body gravity, SRP) do their
/// own dynamics.
#[pyclass]
pub struct EphemerisCache {
    pub coeffs : Vec<Vec<ChebyRecord>>,
    #[pyo3(get)]
    pub ids    : Vec<i32>,
    pub centers: Vec<i32>,
}

/// methods exposed to python
#[pymethods]
impl EphemerisCache {
    /// class constructor
    #[new]
    #[pyo3(signature = (body_coeffs, body_ids, center_ids))]
    pub fn __new__(
        body_coeffs: Vec<Vec<(f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)>>,
        body_ids   : Vec<i32>,
        center_ids : Vec<i32>,
    ) -> Self {
        let coeffs: Vec<Vec<ChebyRecord>> = body_coeffs.into_iter()
            .map(|records: Vec<(f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)>| {
                records.into_iter()
                    .map(|(t_mid, t_rad, x, y, z)| ChebyRecord {
                        t_mid, t_rad,
                        x_coeffs: x,
                        y_coeffs: y,
                        z_coeffs: z,
                    })
                    .collect()
            })
            .collect();

        Self {coeffs : coeffs,
              ids    : body_ids,
              centers: center_ids,}
    }

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!("EphemerisCache | primary: {} | bodies: {:?} | centers: {:?}",
                self.ids.first().unwrap_or(&-1),
                self.ids,
                self.centers)
    }

    /// Position of `target_id` relative to `observer_id` at time `t` [km],
    /// in the SPK native frame (J2000). Both bodies must be in the cache.
    fn get_relative_position(
        &self,
        target_id: i32,
        observer_id: i32,
        t: f64,
    ) -> PyResult<[f64; 3]> {
        let t_idx: usize = self.body_index(target_id)?;
        let o_idx: usize = self.body_index(observer_id)?;

        let t_pos: [f64; 3] = self.get_pos(t_idx, t)
            .map_err(|e: Box<dyn std::error::Error>|
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let o_pos: [f64; 3] = self.get_pos(o_idx, t)
            .map_err(|e: Box<dyn std::error::Error>|
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok([t_pos[0] - o_pos[0], t_pos[1] - o_pos[1], t_pos[2] - o_pos[2]])
    }

    /// Positions of `target_ids` relative to `observer_id` at time `t` [km],
    /// in the SPK native frame (J2000), one row per target. The batched
    /// `get_relative_position`, for models that need several bodies per call.
    fn get_relative_positions<'py>(
        &self,
        py: Python<'py>,
        target_ids: Vec<i32>,
        observer_id: i32,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let to_py = |e: Box<dyn std::error::Error>|
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string());

        let o_pos: [f64; 3] = self.get_pos(self.body_index(observer_id)?, t).map_err(to_py)?;

        let mut rel: Array2<f64> = Array2::zeros((target_ids.len(), 3));
        for (i, &target_id) in target_ids.iter().enumerate() {
            let t_pos: [f64; 3] = self.get_pos(self.body_index(target_id)?, t).map_err(to_py)?;
            for k in 0..3 {
                rel[[i, k]] = t_pos[k] - o_pos[k];
            }
        }

        Ok(rel.into_pyarray(py))
    }
}

/// rust-only methods
impl EphemerisCache {
    /// Cache index of `body_id`.
    pub fn body_index(&self, body_id: i32) -> PyResult<usize> {
        self.ids.iter().position(|&id| id == body_id)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("body {} not in cache", body_id)))
    }

    /// Get SSB-relative position of body at body_idx, resolving center chaining.
    pub fn get_pos(&self, body_idx: usize, t: f64)
        -> Result<[f64; 3], Box<dyn std::error::Error>>
    {
        let records: &Vec<ChebyRecord> = &self.coeffs[body_idx];

        let idx: usize = records
            .partition_point(|r: &ChebyRecord| r.t_mid - r.t_rad <= t)
            .saturating_sub(1);

        let raw: [f64; 3] = if idx < records.len() && records[idx].contains(t) {
            records[idx].eval_pos(t)
        } else {
            return Err(format!("no record covers epoch {}", t).into());
        };

        let center_id: i32 = self.centers[body_idx];

        if center_id == 0 {
            return Ok(raw);
        }

        match self.ids.iter().position(|&id| id == center_id) {
            Some(parent_idx) => {
                let parent: [f64; 3] = self.get_pos(parent_idx, t)?;
                Ok([
                    raw[0] + parent[0],
                    raw[1] + parent[1],
                    raw[2] + parent[2],
                ])
            }
            None => Err(format!("body {} (idx {}) has center {} but center not found in cache \
                                 - add parent body to bodies_to_cache",
                                self.ids[body_idx], body_idx, center_id).into())
        }
    }

}
