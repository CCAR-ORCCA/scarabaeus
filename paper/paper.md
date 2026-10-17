---
title: 'Scarabaeus: a Modular Orbit Determination Framework for Deep-Space and Small-Body Missions'
tags:
  - Python
  - Rust
  - spacecraft navigation
  - orbit determination
  - astrodynamics
  - small-body
  - radiometric tracking
  - optical navigation
authors:
  - name: Jay W. McMahon
    orcid: 0000-0002-1847-4795
    affiliation: 1
  - name: Giovanni Fereoli
    orcid: 0000-0002-2560-1000
    corresponding: true
    affiliation: 1
  - name: Trevor N. Wolf
    orcid: 0000-0002-1406-8829
    affiliation: 1
  - name: Zachary Ellis
    affiliation: 1
  - name: Bushra Aldhanhani
    affiliation: 2
  - name: Mohamed Kuleib
    affiliation: 2
  - name: Wendy Frank
    affiliation: 3
  - name: Mattia Pugliatti
    orcid: 0000-0002-1286-2594
    affiliation: 1
  - name: Jacopo Villa
    orcid: 0009-0009-9491-8317
    affiliation: 3
  - name: Mohamed Almashjari
    affiliation: 4
  - name: Jeremy Knittel
    orcid: 0009-0004-5160-7467
    affiliation: 3
affiliations:
  - name: Aerospace Engineering Sciences, University of Colorado Boulder, Boulder, CO, United States
    index: 1
  - name: Technology Innovation Institute, Masdar City, Abu Dhabi, United Arab Emirates
    index: 2
  - name: Laboratory for Atmospheric and Space Physics, University of Colorado Boulder, Boulder, CO, United States
    index: 3
  - name: Space Missions Department, UAE Space Agency, Abu Dhabi, United Arab Emirates
    index: 4
date: 16 October 2026
bibliography: paper.bib
---

# Summary

Scarabaeus (SCB) is an open-source spacecraft orbit determination (OD) package developed by the Orbital Research Cluster for Celestial Applications (ORCCA) at the University of Colorado Boulder. An object-oriented Python front end and a Rust extension for the computationally intensive kernels form a complete end-to-end navigation environment. SCB provides high-fidelity trajectory propagation with configurable dynamical models; simulation, scheduling, ingestion, and processing of radiometric measurements (two-way coherent Doppler, sequential ranging, range, range-rate, and differential one-way ranging) and optical measurements (center-finding and landmark observations); and estimation spanning batch and sequential filters and smoothers, square-root information formulations, consider and stochastic parameters, process noise, multi-arc estimation, and Markov chain Monte Carlo sampling of the posterior. NASA's SPICE toolkit [@acton1996] supplies mission-grade time systems, reference frames, and ephemerides.

Version 2.0.0, described here, makes every layer of the framework user-extensible: force models, solve-for parameters, measurement models, and whole estimators can be defined from a user script without modifying the library. It also adds small-body gravity and non-gravitational force models, station and planetary orientation modeling, and analytical partial derivatives for the operational Deep Space Network (DSN) observables.

Development is primarily motivated by the Emirates Mission to the Asteroid Belt (EMA) [@parker2024ema], whose MBR Explorer spacecraft launches in 2028 for six main-belt asteroid flybys before rendezvous with asteroid (269) Justitia. SCB therefore emphasizes interplanetary navigation and small-body proximity operations, yet its modular architecture also suits other problems, such as cislunar navigation; GNSS observables and atmospheric drag are not yet implemented, though the custom-model interfaces ease an Earth-orbiting extension.

# Statement of Need

Spacecraft navigation poses a complex estimation problem spanning nearly every aspect of astrodynamics: high-fidelity propagation must capture gravitational and non-gravitational perturbations, heterogeneous tracking data must be modeled and processed with comparable fidelity, and OD algorithms must iteratively estimate the spacecraft state, dynamical and measurement model parameters, and associated uncertainties, producing statistically consistent estimates and covariance information.

The state of the art is well established: mature, comprehensive tools include MONTE [@monte2018], GEODYN [@nicholas2025geodyn], ODTK [@vallado2010odtk], Tudat [@gisolfi2025tudat], and GODOT [@godot]. Many other tools target specific mission classes or organizations rather than general-purpose, mission-agnostic use, and among major publicly available systems only Tudat is fully open source, highlighting the need for more community-developed, openly accessible navigation software.

SCB contributes a comprehensive OD framework that combines research flexibility with the robustness real missions need. Driven by EMA requirements, it is continuously exercised against realistic operational scenarios and real tracking data. Its audience is the astrodynamics research community and mission teams seeking a flexible, transparent, extensible open-source platform for navigation analysis.

# State of the Field

Among open-source OD frameworks, the most comprehensive alternative is **Tudat** [@gisolfi2025tudat], a mature astrodynamics and estimation toolkit with a C++ core and Python interface (TudatPy) that now supports radiometric tracking data analysis. SCB is a complementary effort, developed natively in Python and Rust and driven by an active deep-space mission's operational requirements.

While both provide high-fidelity dynamical modeling and estimation, SCB emphasizes the day-to-day needs of navigation and flight dynamics teams. Beyond OD itself, it offers robust measurement editing and solution quality control, prediction-consistency and solution-closeness metrics [@bhaskaran2004closeness], B-plane mapping and targeting, and maneuver design with local optimization of finite burns: a coherent environment for the full workflow, from data processing and state estimation to operational analysis and trajectory design.

Rather than reimplement general numerical infrastructure, SCB builds on mature open-source tools: SpiceyPy [@annex2020spiceypy] for geometry, reference frames, and ephemerides; NumPy [@harris2020numpy] and SciPy [@virtanen2020scipy] for numerical methods; the IAS15 scheme [@rein2015ias15], reimplemented in the Rust extension, for high-accuracy propagation; emcee [@foreman2013emcee] and corner [@foreman2016corner] for posterior sampling and its visualization; Sphinx [@sphinx] for documentation; and Pytest [@pytest] for testing and verification.

# Software Design

**Architecture and design principles.** SCB's object-oriented Python front end exposes user-facing classes and scripting interfaces and imports them lazily. The Rust extension, `scarabaeus.rust`, carries the performance-critical kernels: spherical-harmonic, polyhedral, and n-plate solar radiation pressure accelerations, whose partial derivatives are verified by dual-number automatic differentiation; the IAS15 integrator; and a Chebyshev ephemeris cache for third bodies. Four design principles guide the codebase: modularity through clear class responsibilities; *contextualization*, in which each datum is owned by a specific object and traceable throughout its lifecycle; *unit typing*, in which physical units attach to quantities via `ArrayWUnits` and propagate through computations; and *frame awareness*, in which vectors carry explicit frame and origin information via `ArrayWFrame`. Unit and frame awareness matter especially in navigation, where subtle inconsistencies carry major operational consequences, notably the Mars Climate Orbiter loss, traced to a metric-imperial mismatch in ground trajectory software [@mco1999].

**Extensibility.** `CustomDynamicModel`, `CustomParameter`, `CustomMeasurement`, and `CustomFilter` bind user functions into the propagator, the state vector, the measurement pipeline, and the estimation loop. Partial derivatives that the user does not supply are filled by central finite differences, so a new model can be prototyped in a few lines and later given analytical partials.

**Dynamical models.** SCB implements point-mass and N-body gravity; spherical-harmonic gravity, with several fields per spacecraft (e.g., a planet and its moons); constant-density polyhedral gravity [@werner1997polyhedron] and its exact conversion to spherical harmonics; cannonball and n-plate solar radiation pressure; thermal and antenna radiation recoil after the OSIRIS-REx small-force models [@geeraert2019smallforces]; Fourier [@mcmahon2010srp], piecewise polynomial, and piecewise first-order Gauss-Markov empirical accelerations, expressed in inertial, RTN, or body-fixed frames; and impulsive, smoothed impulsive, and finite-burn maneuvers. Every model supplies the partial derivatives used in the variational equations.

**Measurement models.** SCB supports (1) ideal range, range-rate, and differential one-way ranging measurements; (2) operational two-way coherent Doppler and sequential ranging observables; and (3) optical measurements in sample/line coordinates or right ascension and declination, for center-finding and landmark-based observations. Radiometric observables follow Moyer's formulations [@moyer2000]: round-trip light times are solved iteratively in the barycentric or geocentric frame with relativistic corrections, and the models account for ramped uplink frequencies, station and transponder delays, solar corona path delays, tropospheric and ionospheric corrections, and time system conversions. Analytical light-time partials map deviations through the reference trajectory's state transition matrices. Station positions include solid-Earth, pole, and ocean-loading displacements [@petit2010iers] and Earth orientation corrections; station locations, Earth orientation parameters, Love and Shida numbers, zenith media delays, and IAU pole parameters are estimable, all with analytical partials. `MeasurementScheduler` builds synthetic tracking from recurring passes with elevation masks, and real tracking data are preprocessed by the Deep Whale Network, a companion ORCCA library that converts TRK-2-34 files via PyTrk234 [@pytrk234] into standardized JSON with SPICE station identifiers and outlier flags.

**Filtering framework.** Four estimators are implemented: a linearized Kalman filter, a least-squares batch estimator, and sequential and batch square-root information filters [@bierman1977], with smoothing for both sequential formulations. Estimable quantities include the states of several bodies, dynamical and measurement model parameters, and consider parameters. Sequential filters offer State Noise Compensation and Dynamical Model Compensation, assignable per body with a cap on the accumulation interval, while batch estimators handle stochastic accelerations as piecewise empirical parameters. Between iterations, measurements are edited by interactive lasso selection, robust Huber and median-absolute-deviation statistics, user-defined data ranges, or user callbacks; convergence is tested on both the state correction and the normalized cost. A converged solution can be sampled with an affine-invariant ensemble sampler [@foreman2013emcee], mapped to arbitrary epochs, frames, and Keplerian elements, compared with other solutions [@bhaskaran2004closeness], and saved with its full provenance; multi-leg and multi-arc estimation is supported.

**Testing.** Scarabaeus uses Pytest for its test suite: unit tests isolating low-level issues in each class and function, integration tests verifying data passing and class interactions, and system tests running full end-to-end scenarios. These run on every merge alongside a performance check. The repository follows the REUSE specification, so every file carries machine-readable license information.

**Documentation.** Docstrings follow a project-wide style guide and feed an online Sphinx site that documents every class, including the Rust kernels and the equations they evaluate. The site also hosts guides and Jupyter Notebook tutorials, from unit and frame handling to the real-data OD case below.

# Research Impact Statement

SCB directly supports EMA [@parker2024ema], led by the UAE Space Agency, and is designed to process the mission's radiometric and optical-navigation measurements across all flight phases. Its dynamical models and partial derivatives were validated against the state-of-the-art high-fidelity propagator Copernicus [@williams2025copernicus] across over a dozen test cases with varied dynamical combinations.

Measurement and filtering implementations were exercised on real and simulated data in three validation cases [@mcmahon2026issfd]: OSIRIS-REx [@lauretta2017osirisrex] DSN tracking, Emirates Mars Mission tracking arcs from multiple DSN stations, and a fully simulated asteroid flyby jointly estimating the spacecraft state and two trajectory-correction maneuvers from combined radiometric and optical data. Version 2.0.0 revisits the OSIRIS-REx case, now a tutorial, with an operational navigation setup: two DSS-35 (Canberra) passes of two-way sequential ranging and Doppler on 2 and 4 July 2018, during cruise to Bennu, form a 53-hour arc at 0.48 AU from Earth. A batch square-root information filter estimates the heliocentric state, a solar radiation pressure scale factor, one range bias per pass, and piecewise first-order Gauss-Markov accelerations, starting from an a priori state 7.8 km and 12 mm/s away from the reconstructed OSIRIS-REx ephemeris (OD044). Ranging is modeled and fitted in range units (RU), the native unit of DSN sequential ranging [@moyer2000]: one RU spans two cycles of the S-band-equivalent uplink frequency, here 0.943 ns of round-trip light time, or 0.141 m of one-way range. After robust editing, with data weights of 7.08 RU (1 m) for ranging and 5.6 mHz (0.1 mm/s) for Doppler, the filter converges in three iterations: the pre-fit residuals of 1.4 to 2.7 $\times 10^4$ RU (2 to 4 km) in ranging and 0.6 Hz in Doppler, set by the a priori error, reduce to centered post-fit residuals of 3.6 to 4.0 RU (about 0.5 m) and 0.5 to 1.0 mHz (\autoref{fig:orex}). The estimated scale factor, $1.26 \pm 0.21$, corresponds to an effective reflectivity coefficient of 1.01. The estimate agrees with OD044 to 2.3 m along the line of sight and to 7 to 10 km in the plane of sky, with a position $\chi^2$ below its 95% bound over the whole arc. Mapped three days past the last measurement, the solution remains consistent with its covariance: the line-of-sight error reaches 37 m against a 156 m standard deviation. A third DSS-35 pass on 6 July, 46 hours after the fit and never seen by the filter, falls inside the predicted $3\sigma$ bounds at every point, with offsets of 137 RU (19 m) in ranging and 6 mHz in Doppler. Because fitting and predicting real tracking data at this level depends on the force models as much as on the observable models and the estimator, this case also implicitly validates the dynamics, verifying dynamics, measurements, and filtering together end to end.

![OSIRIS-REx two-way sequential ranging (in range units, RU) and Doppler residuals of the DSS-35 passes of 2 and 4 July 2018 (shaded). Left column: pre-fit residuals of the first iteration, against the trajectory integrated from the a priori state. Right column: post-fit residuals of the converged solution; dashed lines mark the $\pm 1\sigma$ data weights used by the filter (7.08 RU, or 1 m, for ranging; 5.6 mHz, or 0.1 mm/s, for Doppler).\label{fig:orex}](figures/orex_two_pass.png)

Future development targets small-body proximity operations (terrain-relative navigation observables, asteroid-specific parameter estimation, enhanced multi-arc filtering, and possibly coupled binary-asteroid dynamics), real Delta-DOR measurements with quasar references, and finite-burn targeting, the last two already under development [@kuleib2026maneuver].

# AI Usage Disclosure

Scarabaeus' core implementation, algorithms, and architectural design were developed by the authors. Generative AI assisted with code refactoring and with generating unit and integration tests. The paper was drafted by the authors and revised with AI assistance for JOSS-format restructuring and language editing. All technical content, examples, and claims reflect the authors' work and judgment.

# Acknowledgements

Funding for the co-development of the Emirates Mission to Explore the Asteroid Belt is provided
by the United Arab Emirates Space Agency to its knowledge partner, the University of Colorado
Boulder’s Laboratory for Atmospheric and Space Physics. The authors thank past members of the ORCCA
laboratory who contributed to earlier versions of the codebase: Ms. Annalise Cabra, Dr. Anivid
Faura-Pedros, Mr. Kian Shakerin, Dr. Chloe Long, Dr. Dahlia Baker, Dr. Matthew Givens, Dr. Spencer Boone,
Mr. Santhosh Pattamudu-Manoharan, and Mr. Lars Hinüber.

# References
