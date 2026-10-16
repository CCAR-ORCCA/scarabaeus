// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Adder
//!
//! Minimal example class used to demonstrate Rust unit testing conventions.

use pyo3::prelude::*;

//============================================================================
// Adder
//============================================================================
/// A simple calculator that adds two numbers together.
#[pyclass]
pub struct Adder {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
}

// methods exposed to python
#[pymethods]
impl Adder {
    /// class constructor
    #[new]
    pub fn __new__(a: f64, b: f64) -> Self {
        Self { a: a, b: b }
    }

    /// Add the two stored operands together.
    ///
    /// # Returns
    /// The sum `a + b`.
    pub fn add(&self) -> f64 {
        self.a + self.b
    }
}
