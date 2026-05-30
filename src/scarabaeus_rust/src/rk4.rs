// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Fourth Order Runge Kutta (RK4) Integrator
//! 
//! Not bound to SCB, used as stepping stone for IAS15 implementation
//! Leaving here unbound for now until we decide if it's worth keeping or not.

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, Ix1, Ix2};
use ndarray::{Array, Array1};
use itertools::izip;

pub enum StepResult {
    Success,
    Complete
}

/// Fourth-order Runge–Kutta fixed-step integrator.
///
/// Integrates a first-order ODE `dy/dt = f(t, y)` where the right-hand side
/// is provided as a callable Python function.  The step size `dt` is fixed
/// throughout the integration but its sign is adjusted automatically to match
/// the direction of integration (forward or backward in time).
#[pyclass]
pub struct RK4 {
    // #[pyo3(get, set)]
    pub dt: f64,
    // #[pyo3(get, set)]
    min_dt: f64,
    // #[pyo3(get, set)]
    pub dynamics: Py<PyAny>,
    // #[pyo3(get, set)]
    n_state : u32,
    // #[pyo3(get, set)]
    stm: bool,
}
#[pymethods]
impl RK4 {
    /// Construct an RK4 integrator.
    ///
    /// # Arguments
    /// * `dt`       – nominal step size (seconds).
    /// * `min_dt`   – minimum allowable step size (reserved for future adaptive use).
    /// * `dynamics` – Python callable with signature `f(t: float, y: list[float]) -> list[float]`.
    /// * `n_state`  – dimension of the state vector.
    /// * `stm`      – if `true`, the state vector includes the flattened STM.
    #[new]
    pub fn __new__(dt: f64, min_dt: f64, dynamics: Py<PyAny>, n_state: u32, stm: bool) -> Self {
        // exception handling
        // if ! dynamics.is_callable() {
        //     // Err(PyValueError::new_err("Dynamics must be callable"))

        // }

        // create struct for class construction
        RK4 {dt, min_dt, dynamics, n_state, stm}
    }

    /// Integrate from `t0` to `tf` starting at state `y0`.
    ///
    /// # Arguments
    /// * `t0` – initial epoch (seconds past J2000 TDB).
    /// * `tf` – final epoch (seconds past J2000 TDB).
    /// * `y0` – initial state vector.
    ///
    /// # Returns
    /// A tuple `(ts, ys)` where `ts` is a 1-D array of epoch values and `ys`
    /// is a 2-D array whose rows are the corresponding state vectors.
    pub fn integrate<'py>(&mut self, py: Python<'py>, t0: f64, tf: f64, y0: Vec<f64>) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
        // let a: bool = self.step(py, t0, y0);

        // initialize time and state
        let mut ts: Vec<f64>      = Vec::new();
        let mut ys: Vec<Vec<f64>> = Vec::new();

        ts.push(t0); ys.push(y0.clone());

        let mut t: f64      = t0;
        let mut y: Vec<f64> = y0;

        // define timestep sign
        self.dt = (self.dt.abs()).copysign(tf - t0);

        let dir:f64 = if tf > t0 {1.} else {-1.}; // direction of integration

        // integrate
        loop {
            if dir * (tf - t) <= 0. {break;}    // stop integrating at or past target

            // adjust final step to align with final time
            if dir * (t + self.dt - tf) > 0. {self.dt = tf - t;}

            // perform integration step
            self.step(py, &mut t, &mut y);
            ts.push(t); ys.push(y.clone());
            
        }

        let ts_arr: Bound<'py, PyArray1<f64>> = PyArray1::from_vec(py, ts);
        let ys_arr: Bound<'py, PyArray2<f64>> = PyArray2::from_vec2(py, &ys).unwrap();

        (ts_arr, ys_arr)
    }
}

impl RK4 {
    /// Integrator step
    fn step(&mut self, py: Python, t: &mut f64, y: &mut Vec<f64>) -> () {
        let h: f64 = self.dt;
        // let y_arr: Array1<f64> = Array::from_vec(y.to_vec());

        // call python dynamics function and multiply result by h
        let k1: Vec<f64> = self.dynamics.call1(py, (*t, y.clone())).unwrap().extract::<Vec<f64>>(py).expect("Dynamics function returned non Vec<f64> value").iter().map(|&k1| k1 * h).collect();
        let y1: Vec<f64> = y.iter().zip(k1.iter()).map(|(&yi, &k1i)| yi + k1i / 2.0).collect();

        // k2 and y2
        let k2: Vec<f64> = self.dynamics.call1(py, (*t + (h/2.), y1)).unwrap().extract::<Vec<f64>>(py).expect("Dynamics function returned non Vec<f64> value").iter().map(|&k2| k2 * h).collect();
        let y2: Vec<f64> = y.iter().zip(k2.iter()).map(|(&yi, &k2i)| yi + k2i / 2.0).collect();

        // k3 and y3
        let k3: Vec<f64> = self.dynamics.call1(py, (*t + (h/2.), y2)).unwrap().extract::<Vec<f64>>(py).expect("Dynamics function returned non Vec<f64> value").iter().map(|&k3| k3 * h).collect();
        let y3: Vec<f64> = y.iter().zip(k3.iter()).map(|(&yi, &k3i)| yi + k3i / 2.0).collect();

        // k4
        let k4: Vec<f64> = self.dynamics.call1(py, (*t + h, y3)).unwrap().extract::<Vec<f64>>(py).expect("Dynamics function returned non Vec<f64> value").iter().map(|&k4| k4 * h).collect();
        
        // y_new = y + (k1 + 2k2 + 2k3 + k4)/6
        for i in 0..y.len() {
            y[i] += (k1[i] + 2.*k2[i] + 2.*k3[i] + k4[i]) / 6.0;
        }
        // *y = izip!(&dyn_y, &k1, &k2, &k3, &k4).map(|(yi, &k1i, &k2i, &k3i, &k4i)| yi + (k1i + 2.*k2i + 2.*k3i + k4i)/6.).collect();

        // update time reference and return
        *t += h; ()
    }
}