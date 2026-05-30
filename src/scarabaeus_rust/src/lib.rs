// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Scarabaeus Rust Implementation
//! 
//! Rust code bound to Scarabaeus' Python front end.

use pyo3::prelude::*;

mod ias15; mod integration_event;
pub use ias15::IAS15;
pub use integration_event::IntegrationEvent;

/// bind rust structs (as classes) to Python
#[pymodule]
#[pyo3(name = "scarabaeus_rust")]
fn scarabaeus_rust<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<IAS15>()?;
    m.add_class::<IntegrationEvent>()?;

    // return
    Ok(())
}