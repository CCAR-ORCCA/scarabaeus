// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC

//! Implicit integrator with Adaptive timeStepping, 15th order (IAS15)
//! 
//! Rust implementation of the "IAS15" algorithm [1] that interfaces 
//! with Python.
//! 
//! [1] H. Rein and D. S. Spiegel, "IAS15: a fast, adaptive, high-order 
//!     integrator for gravitational dynamics, accurate to machine 
//!     precision over a billion orbits," Monthly Notices of the Royal 
//!     Astronomical Society, vol. 446, no. 2, pp. 1424-1437, 2015. 
//!     Available: https://doi.org/10.1093/mnras/stu2164

use std::vec;
use pyo3::{prelude::*};

use crate::IntegrationEvent;

//============================================================================
// Constants 
//============================================================================
/// integrator constants
const SAFETY_FACTOR: &'static f64 = &0.25;
const EPSILON      : &'static f64 = &1e-10;

// Gauss Radau spacings
const H: [f64; 8] = [0.0                             , 0.0562625605369221464656521910318,  
                     0.180240691736892364987579942780, 0.352624717113169637373907769648, 
                     0.547153626330555383001448554766, 0.734210177215410531523210605558, 
                     0.885320946839095768090359771030, 0.977520613561287501891174488626];

// other constants based on H_i products, convenient for computation
const RR: [f64; 28] = [0.0562625605369221464656522, 0.1802406917368923649875799, 
                       0.1239781311999702185219278, 0.3526247171131696373739078, 
                       0.2963621565762474909082556, 0.1723840253762772723863278, 
                       0.5471536263305553830014486, 0.4908910657936332365357964, 
                       0.3669129345936630180138686, 0.1945289092173857456275408, 
                       0.7342101772154105315232106, 0.6779476166784883850575584, 
                       0.5539694854785181665356307, 0.3815854601022408941493028, 
                       0.1870565508848551485217621, 0.8853209468390957680903598, 
                       0.8290583863021736216247076, 0.7050802551022034031027798, 
                       0.5326962297259261307164520, 0.3381673205085403850889112, 
                       0.1511107696236852365671492, 0.9775206135612875018911745, 
                       0.9212580530243653554255223, 0.7972799218243951369035945, 
                       0.6248958964481178645172667, 0.4303669872307321188897259, 
                       0.2433104363458769703679639, 0.0921996667221917338008147];

const C: [f64; 21] = [-0.0562625605369221464656522, 0.0101408028300636299864818,
                      -0.2365032522738145114532321, -0.0035758977292516175949345,
                      0.0935376952594620658957485 , -0.5891279693869841488271399,
                      0.0019565654099472210769006 , -0.0547553868890686864408084,
                      0.4158812000823068616886219 , -1.1362815957175395318285885,
                      -0.0014365302363708915424460, 0.0421585277212687077072973,
                      -0.3600995965020568122897665, 1.2501507118406910258505441,
                      -1.8704917729329500633517991, 0.0012717903090268677492943,
                      -0.0387603579159067703699046, 0.3609622434528459832253398,
                      -1.4668842084004269643701553, 2.9061362593084293014237913,
                      -2.7558127197720458314421588];

const D: [f64; 21] = [0.0562625605369221464656522, 0.0031654757181708292499905,
                      0.2365032522738145114532321, 0.0001780977692217433881125,
                      0.0457929855060279188954539, 0.5891279693869841488271399,
                      0.0000100202365223291272096, 0.0084318571535257015445000,
                      0.2535340690545692665214616, 1.1362815957175395318285885,
                      0.0000005637641639318207610, 0.0015297840025004658189490,
                      0.0978342365324440053653648, 0.8752546646840910912297246,
                      1.8704917729329500633517991, 0.0000000317188154017613665,
                      0.0002762930909826476593130, 0.0360285539837364596003871,
                      0.5767330002770787313544596, 2.2485887607691597933926895,
                      2.7558127197720458314421588];

// // weights for integration of a first order differential equation
// const W: [f64; 8] = [0.03125, 
//                      0.185358154802979278540728972807180754479812609, 
//                      0.304130620646785128975743291458180383736715043, 
//                      0.376517545389118556572129261157225608762708603,
//                      0.391572167452493593082499533303669362149363727, 
//                      0.347014795634501068709955597003528601733139176,
//                      0.249647901329864963257869294715235590174262844, 
//                      0.114508814744257199342353731044292225247093225];

//============================================================================
// Structs & Other Methods
//============================================================================
/// enum to handle different input types for integration interval
#[derive(FromPyObject)]
enum TimeInterval {
    StartEnd(f64, f64),      // received start and stop times only
    Discrete(Vec<f64>)       // received discrete time points to solve along
}

impl TimeInterval {
    /// get the discrete time points if TimeInterval is a vector
    pub fn get_points(&self) -> Vec<f64> {
        match self {
            TimeInterval::Discrete(interval) => interval.to_vec(),
            _ => panic!("Cannot extract time points from received interval.")
        }
    }
}

/// integration step return
enum StepResult {
    Success,    // successful step
    Reject,     // rejected step
}
/// polynomial coefficient container
#[derive(Clone, Default)]
#[allow(non_camel_case_types)]
struct reb_dp7 {
    p0: Vec<f64>,
    p1: Vec<f64>,
    p2: Vec<f64>,
    p3: Vec<f64>,
    p4: Vec<f64>,
    p5: Vec<f64>,
    p6: Vec<f64>,
}

/// prefill a reb_reb_dp7 struct with 0 vectors of the correct size for the given state size
trait Prefill {
    fn prefill(n: usize) -> reb_dp7;
}

impl Prefill for reb_dp7 {
    fn prefill(n: usize) -> Self {
        Self {p0: vec![0.0_f64; n],
              p1: vec![0.0_f64; n],
              p2: vec![0.0_f64; n],
              p3: vec![0.0_f64; n],
              p4: vec![0.0_f64; n],
              p5: vec![0.0_f64; n],
              p6: vec![0.0_f64; n]}
    }
}

impl reb_dp7 {
    /// copy all elements of all ps of given reb_dp7 to this reb_dp7
    fn copy_buffers(&mut self, other: &reb_dp7) {
        self.p0 = other.p0.clone(); self.p1 = other.p1.clone();
        self.p2 = other.p2.clone(); self.p3 = other.p3.clone();
        self.p4 = other.p4.clone(); self.p5 = other.p5.clone();
        self.p6 = other.p6.clone();
    }

    /// set all elements of each p to zero
    fn set_ps_zero(&mut self) {
        self.p0 = vec![0.0_f64; self.p0.len()]; self.p1 = vec![0.0_f64; self.p1.len()];
        self.p2 = vec![0.0_f64; self.p2.len()]; self.p3 = vec![0.0_f64; self.p3.len()];
        self.p4 = vec![0.0_f64; self.p4.len()]; self.p5 = vec![0.0_f64; self.p5.len()];
        self.p6 = vec![0.0_f64; self.p6.len()];
    }
}

/// machine independent implementation of pow(*, 1./7.)
fn sqrt7(mut a: f64) -> f64 {
    let mut scale: f64 = 1.0;

    // loops
    while (a < 1e-7) && (a.is_normal()) {
        scale *= 0.1_f64;
        a     *= 1e7;
    }

    while (a > 1e2) && (a.is_normal()) {
        scale *= 10.0_f64;
        a     *= 1e-7;
    }

    let mut x: f64 = 1.0_f64;
    for _ in 0..20_usize {
        let x6: f64 = x * x * x * x * x * x;
        x += (a/x6 - x)/7.0_f64;
    }

    // return
    x * scale
}

/// logic to predict next integration step
fn predict_next_step(ratio: f64, n3: usize, e_prev: &reb_dp7, b_prev: &reb_dp7, e_curr: &mut reb_dp7, b_curr: &mut reb_dp7) {
    if ratio > 20.0 {
        // don't predict if step size increase is very large
        e_curr.set_ps_zero();
        b_curr.set_ps_zero();

    } else {
        // predict new B values to use at the start of the next sequence
        // predicted values from last call are saved as E. The correction
        // BD between the actual and predicted values of B is applied 
        // in advance as a correction
        let q1: f64 = ratio;
        let q2: f64 = q1 * q1; let q3: f64 = q1 * q2;
        let q4: f64 = q2 * q2; let q5: f64 = q2 * q3;
        let q6: f64 = q3 * q3; let q7: f64 = q3 * q4;

        for k in 0..n3 {
            let be0: f64 = b_prev.p0[k] - e_prev.p0[k]; let be1: f64 = b_prev.p1[k] - e_prev.p1[k];
            let be2: f64 = b_prev.p2[k] - e_prev.p2[k]; let be3: f64 = b_prev.p3[k] - e_prev.p3[k];
            let be4: f64 = b_prev.p4[k] - e_prev.p4[k]; let be5: f64 = b_prev.p5[k] - e_prev.p5[k];
            let be6: f64 = b_prev.p6[k] - e_prev.p6[k];

            // update current e
            e_curr.p0[k] = q1*(b_prev.p6[k]*7.0 + b_prev.p5[k]*6.0 +
                               b_prev.p4[k]*5.0 + b_prev.p3[k]*4.0 +
                               b_prev.p2[k]*3.0 + b_prev.p1[k]*2.0 +
                               b_prev.p0[k]);
            e_curr.p1[k] = q2*(b_prev.p6[k]*21.0 + b_prev.p5[k]*15.0 +
                               b_prev.p4[k]*10.0 + b_prev.p3[k]*6.0  +
                               b_prev.p2[k]*3.0  + b_prev.p1[k]);
            e_curr.p2[k] = q3*(b_prev.p6[k]*35.0 + b_prev.p5[k]*20.0 +
                               b_prev.p4[k]*10.0 + b_prev.p3[k]*4.0  +
                               b_prev.p2[k]);
            e_curr.p3[k] = q4*(b_prev.p6[k]*35.0 + b_prev.p5[k]*15.0 +
                               b_prev.p4[k]*5.0  + b_prev.p3[k]);
            e_curr.p4[k] = q5*(b_prev.p6[k]*21.0 + b_prev.p5[k]*6.0  + 
                               b_prev.p4[k]);
            e_curr.p5[k] = q6*(b_prev.p6[k]*7.0  + b_prev.p5[k]);
            e_curr.p6[k] = q7*b_prev.p6[k];

            // update current b
            b_curr.p0[k] = e_curr.p0[k] + be0; b_curr.p1[k] = e_curr.p1[k] + be1;
            b_curr.p2[k] = e_curr.p2[k] + be2; b_curr.p3[k] = e_curr.p3[k] + be3;
            b_curr.p4[k] = e_curr.p4[k] + be4; b_curr.p5[k] = e_curr.p5[k] + be5;
            b_curr.p6[k] = e_curr.p6[k] + be6;
        }
    }
}

/// compensated summation
pub fn add_cs(p: f64, csp: f64, inp: f64) -> (f64, f64) {
    let y      : f64 = inp - csp;
    let t      : f64 = p + y;
    let new_csp: f64 = (t - p) - y;
    (t, new_csp)
}

//============================================================================
// IAS15
//============================================================================
/// struct definition
#[pyclass]
pub struct IAS15 {
    // integrator parameters
    dt          : f64,
    min_dt      : f64,
    dynamics    : Py<PyAny>,
    n_state     : usize,
    dt_last_done: f64,

    // polynomial coefficients
    g  : reb_dp7,
    b  : reb_dp7,
    br : reb_dp7,
    csb: reb_dp7,
    e  : reb_dp7,
    er : reb_dp7,
    csx: Vec<f64>,
    csv: Vec<f64>,

    // events
    #[pyo3(get)]
    events: Vec<IntegrationEvent>,

    #[pyo3(get)]
    logged_events: Vec<(f64, Vec<f64>)>,

    #[pyo3(get)]
    event_tol: f64
}

/// methods exposed to python
#[pymethods]
impl IAS15 {
    /// class constructor
    #[new]
    #[pyo3(signature = (dt, min_dt, dynamics, n_state = 6_usize))]
    pub fn __new__(dt: f64, min_dt: f64, dynamics: Py<PyAny>, n_state: usize) -> Self {
        Self {dt           : dt,
              min_dt       : min_dt,
              dynamics     : dynamics,
              n_state      : n_state,
              dt_last_done : 0.0,
              g            : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              b            : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              e            : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              csb          : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              br           : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              er           : reb_dp7::prefill((n_state * n_state + n_state) / 2),
              csx          : vec![0.0_f64; n_state / 2],
              csv          : vec![0.0_f64; n_state / 2],
              events       : Vec::with_capacity(1),
              logged_events: Vec::with_capacity(1),
              event_tol    : 1e-3}
    }

    //// class attributes
    // constants
    #[classattr]
    const SAFETY_FACTOR: &'static f64 = &SAFETY_FACTOR;
    #[classattr]
    const EPSILON: &'static f64 = &EPSILON;

    /// overloaded print method
    fn __repr__(&self) -> String {
        format!("IAS15 Integrator | dt: {} | min_dt: {}", self.dt, self.min_dt)
    }

    /// run integration
    fn integrate(&mut self, py: Python<'_>, interval: TimeInterval, y0: Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        //// release global interpreter lock (GIL)
        let (times, states) = py.detach(|| -> (Vec<f64>, Vec<Vec<f64>>) {
            // reset integrator
            self.reset();

            // handle interval input
            let (t0, tf, interp) = match &interval {
                // given start stop times -> solve without interpolation
                TimeInterval::StartEnd(t0_in, tf_in) => (t0_in, tf_in, false),

                // given discrete time points -> interpolate solution along them
                TimeInterval::Discrete(t_arr) => (&t_arr[0], &t_arr[t_arr.len() - 1], true)
            };

            // define timestep sign
            let dir: f64 = if tf > t0 {1.0} else {-1.0};
            self.dt      = dir * self.dt.abs();

            // preallocate results 
            let mut times : Vec<f64>      = Vec::with_capacity(1000);
            let mut states: Vec<Vec<f64>> = Vec::with_capacity(1000);

            // and initialize
            let mut t: f64 = *t0; let mut y: Vec<f64> = y0;
            times.push(t)       ; states.push(y.clone());

            // integration loop
            loop {
                // save time and state before step
                let (mut t_pre, y_pre) = (t, y.clone());

                // integration step
                match self.step(&mut t, &mut y) {
                    StepResult::Success => {
                        //// successful -> perform logic for step before continuing
                        // check for events in this time step and log them
                        self.event_finder(t, y.clone(), &mut t_pre, &y_pre.clone());
                        
                        // interpolate between currently evaluated times if given time points
                        let t_post: f64 = t;
                        if interp {
                            // have time points -> interopolate
                            let (ts_interp, ys_interp) = self.interpolate(t_pre, t_post, y_pre, interval.get_points());

                            // and save to states discrete time points
                            for (t_interp, y_interp) in ts_interp.iter().zip(ys_interp) {
                                times.push(*t_interp);
                                states.push(y_interp);
                            }
                        } else {
                            // don't have time points -> just save solved states and times
                            times.push(t);
                            states.push(y.clone());
                        }

                        // check if reached final/max time, adjust step size accordingly
                        let dt_done:f64 = t_post - t_pre;
                        if self.check(t, *tf, dt_done, dir) {break};
                    }
                    StepResult::Reject => {
                        //// unsuccessful -> keep trying until successful
                        continue;
                    }
                }
            }

            // return computed states and times outside of GIl
            (times, states)
        });

        // return solution to Python inside of GIL
        (times, states)
    }

    /// add an event to check for during integration
    fn add_event(&mut self, event: PyRef<IntegrationEvent>) -> () {
        let event_rs: IntegrationEvent = event.clone();
        self.events.push(event_rs);
    }

    /// set event detection tolerance
    fn set_event_tol(&mut self, tolerance: f64) -> () {
        self.event_tol = tolerance;
    }
}

/// rust only methods
impl IAS15 {
    /// compute the integral approximation based on each Gauss Radau spacing
    fn integral_approx(&self, h_n: f64, n3: usize, dt: f64, a0: &[f64], y: &[f64]) -> Vec<f64> {
        let mut y_new: Vec<f64> = y.to_vec();
        for i in 0..n3 {
            // first part of state update
            let update_i  : f64 = ((((((((self.b.p6[i] * 7.0 * h_n / 9.0 + self.b.p5[i])*3.0* h_n/4.0 +
                                          self.b.p4[i])*5.0*h_n/7.0 + self.b.p3[i])*2.0*h_n/3.0 + 
                                          self.b.p2[i])*3.0*h_n/5.0 + self.b.p1[i])*h_n/2.0 + 
                                          self.b.p0[i])*h_n/3.0 + a0[i])*dt*h_n/2.0 + 
                                          y[i + n3])*dt*h_n;

            // second part of state update
            let update_in3: f64 = (((((((self.b.p6[i]*7.0*h_n/8.0 + self.b.p5[i])*6.0*h_n/7.0 +
                                         self.b.p4[i])*5.0*h_n/6.0 + self.b.p3[i])*4.0*h_n/5.0 + 
                                         self.b.p2[i])*3.0*h_n/4.0 + self.b.p1[i])*2.0*h_n/3.0 +
                                         self.b.p0[i])*h_n/2.0 + a0[i])*dt*h_n;

            // add both parts to state
            y_new[i]      += update_i;
            y_new[i + n3] += update_in3;
        }

        // return approximate
        y_new
    }

    // // evaluate state at time
    // fn evaluate_state(&mut self, t_next: f64, t_prev: &mut f64, dt_done: f64, y_prev: Vec<f64>) -> Vec<f64> {
    //     // reacquire GIL to access dynamics function
    //     let dynamics_rs = |t: &mut f64, y: &[f64]| -> Vec<f64> {
    //         Python::attach(|py| {
    //             self.dynamics.call1(py, (*t, y))
    //                     .expect("Dynamics function did not return a value.")
    //                     .extract::<Vec<f64>>(py)
    //                     .expect("Dynamics function did not return a list of floats.")
    //         })
    //     };

    //     // rest outside of GIL
    //     let h_curr: f64 = (t_next - *t_prev) / dt_done;

    //     let y_dot_prev: Vec<f64> = dynamics_rs(t_prev, &y_prev);

    //     let mut a0: Vec<f64> = Vec::with_capacity(y_dot_prev.len() / 2);
    //     for i in 0..(self.n_state / 2_usize) {
    //         a0[i] = y_dot_prev[i + self.n_state/2_usize];
    //     }

    //     return self.integral_approx(h_curr, self.n_state, dt_done, &a0, &y_prev)
    // }

    /// find events 
    // pub fn event_finder(&mut self, t: f64, y: Vec<f64>, t_pre: &mut f64, y_pre: &Vec<f64>) {
    pub fn event_finder(&mut self, t: f64, y: Vec<f64>, _t_pre: &mut f64, y_pre: &Vec<f64>) {
        let t_next : f64 = t;
        // let dt_done: f64 = t_next - *t_pre;
        
        // take events from memory so self is callable inside the loop
        let mut events_free: Vec<IntegrationEvent> = std::mem::take(&mut self.events);

        for event in events_free.iter_mut() {
            if event.evaluate(t_next, &y) {
                let t_new: f64 = t_next;
                let y_new: Vec<f64> = y_pre.clone();

                // let prev_val: f64 = event.check(*t_pre, &y_pre);
                // let next_val: f64 = event.check(t_next, &y.clone());
                
                // let mut t_new: f64 = t_next;
                // let mut dt_new = dt_done / 2.0;
                // let mut dt_sign = -1.0;

                // let mut y_new: Vec<f64> = y_pre.clone(); 
                // let mut tol = dt_new;

                // NOTE: having issues translating this code, seems to work without it, leaving just
                // in case a more complex event needs it

                // while tol.abs() > self.event_tol {
                //     t_new += dt_sign * dt_new;
                //     y_new = self.evaluate_state(t_new, t_pre, dt_done, y_pre.clone());
                    
                //     let new_val: f64 = event.check(t_new, &y_new);
                //     dt_new /= 2.0;

                //     if new_val * prev_val > 0.0_f64 {
                //         // both have same sign -> go toward next value
                //         dt_sign = 1.0;
                //     } else if new_val * next_val > 0.0_f64 {
                //         // both have same sign -> go toward previous value
                //         dt_sign = -1.0;
                //     } else {
                //         // event found exactly (very rare)
                //         break
                //     }
                //     tol = dt_new;
                // }

                self.logged_events.push((t_new, y_new));
            }
        }

        // return back to original place in memory
        self.events = events_free;
    }

    /// if a time vector is provided, perform interpolation at each provided time
    pub fn interpolate(&mut self, t_pre: f64, t_post: f64, y_pre: Vec<f64>, time_interval: Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        // reacquire GIL to access dynamics function
        let dynamics_rs = |t: &mut f64, y: &[f64]| -> Vec<f64> {
            Python::attach(|py| {
                self.dynamics.call1(py, (*t, y))
                        .expect("Dynamics function did not return a value.")
                        .extract::<Vec<f64>>(py)
                        .expect("Dynamics function did not return a list of floats.")
            })
        };

        //// perform rest of calculation outside of GIL
        // find first and last indices to interpolate at
        let (first_ind, final_ind) = match t_pre <= t_post {
            true  => {
                let first_ind: usize = time_interval.partition_point(|&t| t < t_pre);
                let final_ind: usize = time_interval[first_ind..].partition_point(|&t| t <= t_post) + first_ind;

                (first_ind, final_ind)
            },
            false => {
                let first_ind: usize = time_interval.iter().position(|&el| el <= t_pre).unwrap();
                let final_ind: usize = time_interval[first_ind..].iter().position(|&el| el < t_post).map(|ind| ind + first_ind).unwrap();
                (first_ind, final_ind)
            }
        };

        // call dynamics and create a0
        let mut dyn_t: f64      = t_pre;
        let y_dot    : Vec<f64> = dynamics_rs(&mut dyn_t, &y_pre);

        let mut a0   : Vec<f64> = vec![0.0_f64; y_dot.len() / 2];
        for i in 0..(self.n_state/2_usize) {
            a0[i] = y_dot[i + self.n_state/2_usize];
        }

        // calculate integral approximates at each time point
        let dt_last: f64 = t_post - t_pre;

        let mut ts_interp: Vec<f64> = Vec::with_capacity(100);
        let mut ys_interp: Vec<Vec<f64>> = Vec::with_capacity(100);
        for &t_curr in &time_interval[first_ind..final_ind] {
            let h_curr: f64 = (t_curr - t_pre) / dt_last;

            if t_curr != t_pre {
                // interpolate and save
                ts_interp.push(t_curr); 
                ys_interp.push(self.integral_approx(h_curr, self.n_state / 2_usize, dt_last, &a0, &y_pre));
            }
        }
        
        // return interpolated states and times
        (ts_interp, ys_interp)
    }

    /// ensure no time overshoot
    pub fn check(&mut self, t_curr: f64, t_max: f64, dt_done: f64, dir: f64) -> bool {
        if (t_curr + self.dt) >= t_max {
            // next step would overshoot -> more checks
            if t_curr == t_max {
                // don't need to do anything, already correct
                ()
            } else {
                // first find order of magnitude for time
                let t_scale: f64 = if (1e-12 * t_max.abs()) < 1e-200 {1e-12}    // failsafe if t_max = 0
                                   else {1e-12 * t_max.abs()};

                // now check dt
                if (t_curr - t_max).abs() < t_scale {
                    // tolerable output error, do nothing
                    ()
                } else {
                    // not there yet -> do another step
                    self.dt = t_max - t_curr;
                }
            }
        }

        // update e and b
        let ratio: f64 = self.dt / dt_done;
        let (mut e_new, mut b_new) = (self.e.clone(), self.b.clone());
        predict_next_step(ratio, self.n_state / 2, &self.e, &self.b, &mut e_new, &mut b_new);
        self.e = e_new;
        self.b = b_new;

        // at or past final time -> stop integration
        if dir*(t_max - t_curr) <= 0.0 {return true;}

        // make sure last step happens on final time
        if dir*(t_curr + self.dt - t_max) > 0.0 {self.dt = t_max - t_curr;}

        // not done yet -> keep going
        return false
    }

    /// integration step logic
    fn step(&mut self, t: &mut f64, y: &mut Vec<f64>) -> StepResult {
        // reacquire GIL to access dynamics function
        let dynamics_rs = |t: &mut f64, y: &[f64]| -> Vec<f64> {
            Python::attach(|py| {
                self.dynamics.call1(py, (*t, y))
                        .expect("Dynamics function did not return a value.")
                        .extract::<Vec<f64>>(py)
                        .expect("Dynamics function did not return a list of floats.")
            })
        };
        //// perform rest of the calculations outside of the GIL
        // call dynamics
        let y_dot : Vec<f64> = dynamics_rs(t, &y);

        // initialize vectors and save current step dt
        let dim   : usize    = self.n_state / 2usize;
        let mut a0: Vec<f64> = vec![0.0f64; dim];
        let mut at: Vec<f64> = vec![0.0f64; dim];

        self.csb.set_ps_zero();

        for i in 0..dim {
            a0[i] = y_dot[i + 3];
            at[i] = a0[i];
        }

        // calculate gs
        for k in 0..dim {
            self.g.p0[k] = self.b.p6[k]*D[15] + self.b.p5[k]*D[10] + 
                           self.b.p4[k]*D[6]  + self.b.p3[k]*D[3]  + 
                           self.b.p2[k]*D[1]  + self.b.p1[k]*D[0]  + 
                           self.b.p0[k];
            self.g.p1[k] = self.b.p6[k]*D[16] + self.b.p5[k]*D[11] + 
                           self.b.p4[k]*D[7]  + self.b.p3[k]*D[4]  + 
                           self.b.p2[k]*D[2]  + self.b.p1[k];
            self.g.p2[k] = self.b.p6[k]*D[17] + self.b.p5[k]*D[12] + 
                           self.b.p4[k]*D[8]  + self.b.p3[k]*D[5]  + 
                           self.b.p2[k];
            self.g.p3[k] = self.b.p6[k]*D[18] + self.b.p5[k]*D[13] + 
                           self.b.p4[k]*D[9]  + self.b.p3[k];
            self.g.p4[k] = self.b.p6[k]*D[19] + self.b.p5[k]*D[14] + 
                           self.b.p4[k];
            self.g.p5[k] = self.b.p6[k]*D[20] + self.b.p5[k];
            self.g.p6[k] = self.b.p6[k];
        }

        // predictor-corrector loop
        // stops if one of the following conditions is satisfied:
        //      1) better error than 1e-16
        //      2) error starts to oscillate
        //      3) more than 12 iterations
        let t_beginning    : f64 = *t;
        let mut pc_err     : f64 = 1e300;
        let mut pc_err_last: f64 = 2.0_f64;
        let mut iterations : i32 = 0_i32;
        let mut y_new: Vec<f64>  = Vec::new();

        while (pc_err >= 1e-16) && 
              (iterations <= 2 || pc_err < pc_err_last) &&
              (iterations < 12) {
            // update iteration info
            pc_err_last = pc_err;
            pc_err      = 0.0_f64;
            iterations += 1;

            // loop over interval using Gauss-Radau spacings
            for n in 1..8 {
                *t = t_beginning + self.dt * H[n];

                // estimate state
                y_new = self.integral_approx(H[n], self.n_state / 2_usize, self.dt, &a0, y);
                for i in 0..3 {
                    y_new[i] -= self.csx[i];
                    y_new[i + 3] -= self.csv[i];
                }

                // new y dot
                let y_dot_new = dynamics_rs(t, &y_new);
                for i in 0..dim {
                    at[i] = y_dot_new[i + 3];
                }

                // improve b and g coefficient values with added compensation
                match n {
                    1 => {
                        for k in 0..dim {
                            let tmp: f64   = self.g.p0[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p0[k]   = gk / RR[0];
                            let delta: f64 = self.g.p0[k] - tmp;

                            let (new_b, new_csb) = add_cs(self.b.p0[k], self.csb.p0[k], delta);
                            self.b.p0[k]   = new_b;
                            self.csb.p0[k] = new_csb;
                        }
                    }
                    2 => {
                        for k in 0..dim {
                            let tmp: f64   = self.g.p1[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p1[k]   = (gk/RR[1] - self.g.p0[k])/RR[2];
                            let delta: f64 = self.g.p1[k] - tmp;

                            let (new_b0, new_csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta * C[0]);
                            self.b.p0[k]   = new_b0;
                            self.csb.p0[k] = new_csb0;

                            let (new_b1, new_csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta);
                            self.b.p1[k]   = new_b1;
                            self.csb.p1[k] = new_csb1;
                        }
                    }
                    3 => {
                        for k in 0..dim {
                            let tmp: f64   = self.g.p2[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p2[k]   = ((gk/RR[3] - self.g.p0[k])/RR[4] - self.g.p1[k])/RR[5];
                            let delta: f64 = self.g.p2[k] - tmp;

                            let (b0, csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta*C[1]);
                            let (b1, csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta*C[2]);
                            let (b2, csb2) = add_cs(self.b.p2[k], self.csb.p2[k], delta);

                            self.b.p0[k] = b0; self.csb.p0[k] = csb0;
                            self.b.p1[k] = b1; self.csb.p1[k] = csb1;
                            self.b.p2[k] = b2; self.csb.p2[k] = csb2;
                        }
                    }
                    4 => {
                        for k in 0..dim {
                            let tmp: f64   = self.g.p3[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p3[k]   = (((gk/RR[6] - self.g.p0[k])/RR[7] - self.g.p1[k])/RR[8] - 
                                                self.g.p2[k])/RR[9];
                            let delta: f64 = self.g.p3[k] - tmp;

                            let (b0, csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta*C[3]);
                            let (b1, csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta*C[4]);
                            let (b2, csb2) = add_cs(self.b.p2[k], self.csb.p2[k], delta*C[5]);
                            let (b3, csb3) = add_cs(self.b.p3[k], self.csb.p3[k], delta);

                            self.b.p0[k] = b0; self.csb.p0[k] = csb0;
                            self.b.p1[k] = b1; self.csb.p1[k] = csb1;
                            self.b.p2[k] = b2; self.csb.p2[k] = csb2;
                            self.b.p3[k] = b3; self.csb.p3[k] = csb3;
                        }
                    }
                    5 => {
                        for k in 0..dim {
                            let tmp: f64 = self.g.p4[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p4[k] = ((((gk/RR[10] - self.g.p0[k])/RR[11] - self.g.p1[k])/RR[12] - 
                                               self.g.p2[k])/RR[13] - self.g.p3[k])/RR[14];
                            let delta: f64 = self.g.p4[k] - tmp;

                            let (b0, csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta*C[6]);
                            let (b1, csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta*C[7]);
                            let (b2, csb2) = add_cs(self.b.p2[k], self.csb.p2[k], delta*C[8]);
                            let (b3, csb3) = add_cs(self.b.p3[k], self.csb.p3[k], delta*C[9]);
                            let (b4, csb4) = add_cs(self.b.p4[k], self.csb.p4[k], delta);

                            self.b.p0[k] = b0; self.csb.p0[k] = csb0;
                            self.b.p1[k] = b1; self.csb.p1[k] = csb1;
                            self.b.p2[k] = b2; self.csb.p2[k] = csb2;
                            self.b.p3[k] = b3; self.csb.p3[k] = csb3;
                            self.b.p4[k] = b4; self.csb.p4[k] = csb4;
                        }
                    }
                    6 => {
                        for k in 0..dim {
                            let tmp: f64 = self.g.p5[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p5[k] = (((((gk/RR[15] - self.g.p0[k])/RR[16] - self.g.p1[k])/RR[17] -
                                                self.g.p2[k])/RR[18] - self.g.p3[k])/RR[19] - self.g.p4[k])/RR[20];
                            let delta: f64 = self.g.p5[k] - tmp;

                            let (b0, csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta*C[10]);
                            let (b1, csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta*C[11]);
                            let (b2, csb2) = add_cs(self.b.p2[k], self.csb.p2[k], delta*C[12]);
                            let (b3, csb3) = add_cs(self.b.p3[k], self.csb.p3[k], delta*C[13]);
                            let (b4, csb4) = add_cs(self.b.p4[k], self.csb.p4[k], delta*C[14]);
                            let (b5, csb5) = add_cs(self.b.p5[k], self.csb.p5[k], delta);

                            self.b.p0[k] = b0; self.csb.p0[k] = csb0;
                            self.b.p1[k] = b1; self.csb.p1[k] = csb1;
                            self.b.p2[k] = b2; self.csb.p2[k] = csb2;
                            self.b.p3[k] = b3; self.csb.p3[k] = csb3;
                            self.b.p4[k] = b4; self.csb.p4[k] = csb4;
                            self.b.p5[k] = b5; self.csb.p5[k] = csb5;
                        }
                    }
                    7 => {
                        let mut max_ak     : f64 = 0.0_f64;
                        let mut max_b6k_tmp: f64 = 0.0_f64;

                        for k in 0..dim {
                            let tmp: f64 = self.g.p6[k];
                            let (gk, _) = add_cs(at[k], 0.0, -a0[k]);
                            self.g.p6[k] = ((((((gk/RR[21] - self.g.p0[k])/RR[22] - self.g.p1[k])/RR[23] -
                                                 self.g.p2[k])/RR[24] - self.g.p3[k])/RR[25] - 
                                                 self.g.p4[k])/RR[26] - self.g.p5[k])/RR[27];
                            let delta: f64 = self.g.p6[k] - tmp;

                            let (b0, csb0) = add_cs(self.b.p0[k], self.csb.p0[k], delta*C[15]);
                            let (b1, csb1) = add_cs(self.b.p1[k], self.csb.p1[k], delta*C[16]);
                            let (b2, csb2) = add_cs(self.b.p2[k], self.csb.p2[k], delta*C[17]);
                            let (b3, csb3) = add_cs(self.b.p3[k], self.csb.p3[k], delta*C[18]);
                            let (b4, csb4) = add_cs(self.b.p4[k], self.csb.p4[k], delta*C[19]);
                            let (b5, csb5) = add_cs(self.b.p5[k], self.csb.p5[k], delta*C[20]);
                            let (b6, csb6) = add_cs(self.b.p6[k], self.csb.p6[k], delta);

                            self.b.p0[k] = b0; self.csb.p0[k] = csb0;
                            self.b.p1[k] = b1; self.csb.p1[k] = csb1;
                            self.b.p2[k] = b2; self.csb.p2[k] = csb2;
                            self.b.p3[k] = b3; self.csb.p3[k] = csb3;
                            self.b.p4[k] = b4; self.csb.p4[k] = csb4;
                            self.b.p5[k] = b5; self.csb.p5[k] = csb5;
                            self.b.p6[k] = b6; self.csb.p6[k] = csb6;

                            // monitor change b.p6[k] relative to at[k]. Predictor corrector scheme
                            // is converged if it's close to 0. Using global error estimate which is the 
                            // default in REBOUND
                            let ak: f64 = at[k].abs();
                            if ak.is_normal() && ak > max_ak {
                                max_ak = ak;
                            }
                            let b6k_tmp = delta.abs();
                            if b6k_tmp.is_normal() && b6k_tmp > max_b6k_tmp {
                                max_b6k_tmp = b6k_tmp;
                            }
                        }

                        // update predictor corrector error
                        pc_err = max_b6k_tmp / max_ak;
                    }
                    _ => ()
                }
            }
        }

        // set time back to initial value (will be updated below)
        *t = t_beginning;
        let dt_done: f64 = self.dt;     // find new time step

        // estimate error (given by last term in series expansion)
        // set to be the default (global) option in REBOUND
        // first determine max acceleration and max of the last term in the series, then 
        // the two are divided
        if *EPSILON > 0.0f64 {
            let mut max_ak : f64 = 0.0f64;
            let mut max_b6k: f64 = 0.0f64;

            let x2: f64 = y_new[0]*y_new[0] + y_new[1]*y_new[1] + y_new[2]*y_new[2];
            let v2: f64 = y_new[3]*y_new[3] + y_new[4]*y_new[4] + y_new[5]*y_new[5];

            if ((v2*dt_done*dt_done) / x2).abs() > 1e-16 {  // machine precision
                for k in 0..dim {
                    let ak: f64 = at[k].abs();
                    if ak.is_normal() && ak > max_ak {max_ak = ak;}

                    let b6k: f64 = self.b.p6[k].abs();
                    if b6k.is_normal() && b6k > max_b6k {max_b6k = b6k;}
                }
            }

            let integrator_error: f64 = max_b6k / max_ak;

            let dt_new = if integrator_error.is_normal() {
                // if error estimate is available increase by more educated guess
                sqrt7(EPSILON/integrator_error)*dt_done
            } else {
                // in the rare case that the error estimate doesn't give a finite number
                // e.g. when all forces accidentally cancel up to machine precision -> increase
                // time step a little
                dt_done / SAFETY_FACTOR
            };

            let dt_new: f64 = dt_new.signum() * dt_new.abs().max(self.min_dt);
            if (dt_new / dt_done).abs() < *SAFETY_FACTOR {
                self.dt = dt_new;

                if self.dt_last_done != 0.0 {
                    let ratio: f64 = self.dt / self.dt_last_done;
                    let (mut e_new, mut b_new) = (self.e.clone(), self.b.clone());
                    predict_next_step(ratio, 3, &self.er, &self.br, &mut e_new, &mut b_new);
                    self.e = e_new;
                    self.b = b_new;
                }

                return StepResult::Reject;
            }

            if dt_new.abs() > dt_done.abs() && 
               (dt_new / dt_done).abs() > 1.0 / SAFETY_FACTOR {
                self.dt = dt_done / SAFETY_FACTOR;
            } else {
                self.dt = dt_new;
            }
        }

        // update state
        for k in 0..3 {
            let (y0, csx0) = add_cs(y[k], self.csx[k], self.b.p6[k]/72.0 * dt_done*dt_done);
            let (y1, csx1) = add_cs(y0, csx0, self.b.p5[k]/56.0 * dt_done*dt_done);
            let (y2, csx2) = add_cs(y1, csx1, self.b.p4[k]/42.0 * dt_done*dt_done);
            let (y3, csx3) = add_cs(y2, csx2, self.b.p3[k]/30.0 * dt_done*dt_done);
            let (y4, csx4) = add_cs(y3, csx3, self.b.p2[k]/20.0 * dt_done*dt_done);
            let (y5, csx5) = add_cs(y4, csx4, self.b.p1[k]/12.0 * dt_done*dt_done);
            let (y6, csx6) = add_cs(y5, csx5, self.b.p0[k]/6.0  * dt_done*dt_done);
            let (y7, csx7) = add_cs(y6, csx6, a0[k]/2.0 * dt_done*dt_done);
            let (y8, csx8) = add_cs(y7, csx7, y[k + 3] * dt_done);
            y[k] = y8;
            self.csx[k] = csx8;

            let (v0, csv0) = add_cs(y[k + 3], self.csv[k], self.b.p6[k]/8.0 * dt_done);
            let (v1, csv1) = add_cs(v0, csv0, self.b.p5[k]/7.0 * dt_done);
            let (v2, csv2) = add_cs(v1, csv1, self.b.p4[k]/6.0 * dt_done);
            let (v3, csv3) = add_cs(v2, csv2, self.b.p3[k]/5.0 * dt_done);
            let (v4, csv4) = add_cs(v3, csv3, self.b.p2[k]/4.0 * dt_done);
            let (v5, csv5) = add_cs(v4, csv4, self.b.p1[k]/3.0 * dt_done);
            let (v6, csv6) = add_cs(v5, csv5, self.b.p0[k]/2.0 * dt_done);
            let (v7, csv7) = add_cs(v6, csv6, a0[k] * dt_done);
            y[k + 3] = v7;
            self.csv[k] = csv7;
        }

        // step to next time
        *t += dt_done;
        self.dt_last_done = dt_done;

        // update e an b
        self.er.copy_buffers(&self.e);
        self.br.copy_buffers(&self.b);

        // return succesful step
        StepResult::Success
    }

    /// reset integrator
    fn reset(&mut self) {
        self.g.set_ps_zero();
        self.b.set_ps_zero();
        self.e.set_ps_zero();
        self.csb.set_ps_zero();
        self.br.set_ps_zero();
        self.er.set_ps_zero();
        self.csx = vec![0.0_f64; self.n_state / 2];
        self.csv = vec![0.0_f64; self.n_state / 2];
        self.dt_last_done = 0.0;
    }
}
