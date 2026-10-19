// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Scarabaeus Rust Implementation
//! 
//! Rust code bound to Scarabaeus' Python front end.

use pyo3::prelude::*;

mod ias15; mod integration_event; mod adder;
pub use ias15::IAS15;
pub use integration_event::IntegrationEvent;
pub use adder::Adder;


mod ephemeris_cache;
pub use ephemeris_cache::EphemerisCache;

mod spherical_harmonics;
pub use spherical_harmonics::SphericalHarmonicsCore;

mod nplate_srp;
pub use nplate_srp::NPlateSRPCore;

mod polyhedron_gravity;
pub use polyhedron_gravity::PolyhedronGravityCore;

mod polyhedron_harmonics;
pub use polyhedron_harmonics::PolyhedronHarmonics;

/// bind rust structs (as classes) to Python
#[pymodule]
#[pyo3(name = "rust")]
fn rust<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // integrator methods
    m.add_class::<IAS15>()?;
    m.add_class::<IntegrationEvent>()?;
    m.add_class::<Adder>()?;

    // dynamics methods
    m.add_class::<EphemerisCache>()?;
    m.add_class::<SphericalHarmonicsCore>()?;
    m.add_class::<NPlateSRPCore>()?;
    m.add_class::<PolyhedronGravityCore>()?;

    // shape model -> spherical harmonics
    m.add_class::<PolyhedronHarmonics>()?;

    // return
    Ok(())
}