// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Integration Event
//! 
//! Defines an event used by IAS15 to search for during integration.
//! Note that builtins are currently unsupported. Event detection function
//! must be defined in Python and passed to this struct. 

use pyo3::{prelude::*};
use std::sync::Arc;

//============================================================================
// Macros 
//============================================================================
// macro_rules! gen_builtins_binding {
//     #[pymethods]
//     ( $( ($builtin_name: ident, $detector: fn) ),* ) = {
//         $(
//             fn $builtin_name(v: $detector) {
//                 Event.__new__()
//             }
//         )
//     }
// }

/// enum to handle event finding functions defined either in Python or Rust
// enum EventFinder {
//     PyFinder(Arc<Py<PyAny>>),
//     RustFinder(fn())
// }

//============================================================================
// Events 
//============================================================================
/// An event that IAS15 searches for during integration.
///
/// The user supplies a Python callable (`event_finder`) that maps `(t, y)` to a
/// scalar.  IAS15 calls [`evaluate`] at each accepted step; when the scalar
/// crosses zero in the requested direction the event is flagged and, if
/// `is_terminal`, integration stops.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct IntegrationEvent {
    // parameters
    event_finder   : Arc<Py<PyAny>>,
    find_increasing: bool,
    find_decreasing: bool,
    is_terminal    : bool,
    // evaluation
    prev_val: f64,
}

// methods exposed to python
#[pymethods]
impl IntegrationEvent {
    /// class constructor
    #[new]
    #[pyo3(signature = (event_finder, find_increasing, find_decreasing, 
                        is_terminal = false))]
    pub fn __new__(event_finder: Py<PyAny>, find_increasing: bool, find_decreasing: bool, is_terminal: bool) -> Self {
        Self {event_finder   : Arc::new(event_finder),
              find_increasing: find_increasing,
              find_decreasing: find_decreasing,
              is_terminal    : is_terminal,
              prev_val       : f64::NAN}
    }

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!("IAS15 Event\nfind_increasing: {}\nfind_decreasing: {}\nis_terminal: {}", 
                 self.find_increasing, self.find_decreasing, self.is_terminal)
    }
}

// rust only methods
impl IntegrationEvent {
    /// Evaluate the event at the current step and return whether it triggered.
    ///
    /// Calls [`check`] to obtain the current scalar value, then compares it
    /// against the previous value to detect a zero-crossing in the direction(s)
    /// requested by `find_increasing` / `find_decreasing`.
    ///
    /// # Arguments
    /// * `t` – current epoch (seconds past J2000 TDB).
    /// * `y` – current state vector.
    ///
    /// # Returns
    /// `true` if a zero-crossing occurred in the requested direction.
    pub fn evaluate(&mut self, t: f64, y: &Vec<f64>) -> bool {
        // call event detection function
        let current_val: f64 = self.check(t, y);

        // update on first step
        if self.prev_val.is_nan() {self.prev_val = current_val;}

        // check if crossed between the last step and this one
        let happened: bool = (self.find_increasing && current_val >= 0.0 && self.prev_val < 0.0) || 
                             (self.find_decreasing && current_val <= 0.0 && self.prev_val > 0.0);

        // update previous value and return trigger bool
        self.prev_val = current_val;
        happened
    }

    /// Invoke the Python event-finder callable and return its scalar result.
    ///
    /// # Arguments
    /// * `t` – current epoch (seconds past J2000 TDB).
    /// * `y` – current state vector.
    ///
    /// # Returns
    /// The scalar value returned by `event_finder(t, y)`.
    ///
    /// # Panics
    /// Panics if the Python callable raises an exception or does not return a
    /// `f64`-compatible value.
    pub fn check(&self, mut t: f64, y: &Vec<f64>) -> f64 {
        let event_finder_rs = |t: &mut f64, y: &[f64]| -> f64 {
            Python::attach(|py: Python<'_>| {
                self.event_finder.call1(py, (*t, y))
                        .expect("Dynamics function did not return a value.")
                        .extract::<f64>(py)
                        .expect("Dynamics function did not return a list of floats.")
            })
        };

        return event_finder_rs(&mut t, &y)
    }
}

//============================================================================
// Built-in Events 
//============================================================================
// generate alternate constructors for each built-in event
// gen_builtins_binding(Periapsis, detect_periapsis)
// gen_builtins_binding(Apoapsis, detect_apoapsis)

// //// commonly used events
// /// detect periapsis of an orbit
// fn detect_periapsis() -> () {
//     todo!()
// }

// /// detect apoapsis of an orbit
// fn detect_apoapsis() -> () {
//     todo!()
// }

// // trait Periapsis  {
// //     fn detect_periapsis() -> bool;

// //     fn periapsis_builtin(find_increasing: bool, find_decreasing: bool, 
// //                          is_terminal: bool, count: usize) -> Self;
// // }

// // impl Periapsis for Event {
// //     fn detect_periapsis() -> bool {
// //         false
// //     }

// //     fn periapsis_builtin(find_increasing: bool, find_decreasing: bool,
// //                          is_terminal: bool, count: usize) -> Self {
// //         Self {event_finder   : Arc<detect_periapsis>,
// //               find_increasing: false,
// //               find_decreasing: false,
// //               is_terminal    : false,
// //               count          : 0_usize}
// //     }
// // }