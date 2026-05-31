---
title: 'Scarabaeus: An Open-Source Tool for Interplanetary Spacecraft Navigation'
tags:
  - Python
  - Rust
  - spacecraft navigation
  - orbit determination
  - astrodynamics
  - Kalman filter
  - interplanetary
authors:
  - name: Jay W. McMahon
    affiliation: 1
  - name: Giovanni Fereoli
    affiliation: 1
  - name: Trevor N. Wolff
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
    affiliation: 1
  - name: Mohamed Almashjari
    affiliation: 4
  - name: Jeremy Knittel
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
date: 31 May 2026
bibliography: paper.bib
---

# Summary

Scarabaeus (SCB) is an open-source Python framework for interplanetary spacecraft
navigation and orbit determination (OD), developed by the Orbital Research Cluster for
Celestial Applications (ORCCA) at the University of Colorado Boulder. It provides a
unified, mission-agnostic environment for end-to-end spacecraft navigation: trajectory
propagation under a configurable suite of force models, simulation and ingestion of
radiometric and optical tracking measurements, sequential and batch orbit determination
filters, and maneuver planning. SCB interfaces with NASA's SPICE toolkit [@acton1996]
for mission-grade time, reference frame, and ephemeris management, and ships a compiled
Rust back-end for performance-critical numerical integration. The framework is designed
to be accessible to graduate students and astrodynamics researchers while remaining
capable enough for operational mission support. Its development is driven by the
Emirates Mission to the Asteroid Belt (EMA) [@ema2021], while remaining general enough
for broad interplanetary navigation use cases.

# Statement of Need

Interplanetary spacecraft navigation requires solving a coupled estimation problem:
propagating spacecraft trajectories under gravitational and non-gravitational
perturbations, processing heterogeneous tracking measurements, and iterating an orbit
determination filter to refine the spacecraft state and covariance. This problem spans
nearly every aspect of astrodynamics—dynamics, measurement modeling, estimation theory,
and mission operations—and demands consistent, mission-grade handling of time scales,
reference frames, and planetary ephemerides.

The state of the art in OD software for deep-space and Earth-orbiting applications is
well established. Mature tools include MONTE [@monte2018], GEODYN, ODTBX, ODTK, GMAT
[@gmat2022], TudatPy [@dirkx2024tudat], GODOT, CubeNav [@cubenav2023], and Orbit14.
Among these, only GMAT, ODTBX, and TudatPy are open-source, underscoring the value of
additional shared, community-supported development. SCB addresses this gap by providing,
in a single Python-native package: a library of force models and high-order numerical
integrators; measurement models for both simulated and real radiometric and optical data,
including actual Deep Space Network (DSN) and ESTRACK formats; sequential and batch orbit
determination filters validated against real mission data; maneuver planning tools; and a
SPICE-integrated time and frame management system. The primary audience is the
astrodynamics research community and mission teams requiring a flexible, auditable,
open-source navigation platform.

# State of the Field

Among the open-source tools listed above, SCB is most directly compared to GMAT and
TudatPy:

**GMAT** [@gmat2022] is NASA's open-source mission design tool with trajectory
propagation and maneuver targeting capabilities. Its OD functionality is limited and
it does not provide a modular Python API suited to research-level algorithm development.

**TudatPy** [@dirkx2024tudat] provides a well-documented Python interface to the Tudat
C++ astrodynamics library with strengths in trajectory propagation and optimization, but
limited orbit determination capabilities and no support for real DSN data ingestion.

**Basilisk** [@kenneally2020basilisk], developed at the same institution as SCB,
addresses spacecraft attitude dynamics and control system simulation rather than
navigation and OD, making the tools complementary. SCB's design was directly inspired
by Basilisk's modular, open-source approach [@mcmahon2025gnc].

**ODTBX** is an open-source MATLAB/Java tool for OD analysis developed at NASA
Goddard; its development has been inactive in recent years.

None of these tools offer a complete, Python-native OD pipeline combining real
radiometric data ingestion (DSN TRK-2-34 and ESTRACK formats), multiple filter types
(batch, sequential, square-root information with smoother), process noise models (SNC,
DMC), consider parameters, measurement editing, and SPICE-native time and frame
management throughout. Contributing a capability of this scope to any existing tool
would have required restructuring its fundamental architecture; SCB was designed from
the ground up around interplanetary mission navigation requirements, building on the
scientific Python ecosystem [@virtanen2020scipy; @harris2020numpy] rather than
re-implementing general numerical infrastructure.

# Software Design

**Python/Rust hybrid architecture.** The primary API is written in Python using an
object-oriented design. Users configure dynamics models, measurement types, and
estimation filters as Python objects, compose them into a `MissionSequence`, and
execute the navigation pipeline with minimal boilerplate. The Rust back-end is reserved
for the performance-critical integration hot path. The IAS15 integrator
[@rein2015ias15]—an implicit adaptive 15th-order method suited for long-arc deep-space
propagation, exposed to Python as PyASA—is implemented in Rust and bound via PyO3 and
maturin. The Dormand–Prince DOP853 integrator is available via SciPy [@virtanen2020scipy]
for rapid prototyping.

**SPICE integration.** All time conversions, reference frame transformations, and
ephemeris queries are handled through SpiceyPy [@annex2020spiceypy], wrapping NASA's
SPICE toolkit [@acton1996; @acton2018spice]. This ensures that time scales (UTC, TDB,
TT, SCLK), planetary ephemerides, and spacecraft orientation data are managed
consistently and to mission-grade accuracy throughout the pipeline—a requirement that
motivated deep SPICE integration from the earliest design stage.

**Measurement models and the Deep Whale Network.** SCB supports ten measurement model
classes covering two-way range, range rate, Doppler, differential one-way range (DOR),
and optical centroiding in both ideal (simulated) and real-data formulations. Real
observables follow the formulation of @moyer2000 and account for round-trip light time,
relativistic corrections, transponder delays, and solar corona path delays. Real DSN and
ESTRACK tracking data in TRK-2-34 format are pre-processed by the Deep Whale Network
(DWN), a companion Python library developed concurrently with SCB that parses native
tracking formats into a standardized JSON representation consumed by SCB. Data and
auxiliary products (kernels, media corrections, ramp tables) can be managed through a
local file system or a cloud-hosted MongoDB back-end.

**Orbit determination filters.** Four filter classes are implemented: a linearized
Kalman filter (LKF) with Rauch–Tung–Striebel (RTS) smoothing, a least-squares batch
estimator (LSB), and sequential and batch square-root information filters (SRIF and
SRIFB) [@bierman1977]. Process noise is handled through State Noise Compensation (SNC),
Dynamical Model Compensation (DMC), and a batch stochastic acceleration mode. A
`StateArray` object constructs a typed, dictionary-based representation of the state
vector that supports estimated and consider parameters, including spacecraft position and
velocity, SRP coefficients, spherical harmonics, ground station location biases, range
and Doppler biases, impulsive maneuver components, and stochastic accelerations.

**Modular composition and unit safety.** Each subsystem is a self-contained class that
can be substituted independently—swapping one filter for another or adding a force model
requires minimal code changes. The `ArrayWUnits` and `ArrayWFrame` classes propagate
physical units and reference frames through computations at runtime, preventing
unit-mismatch errors. Dynamics modules are validated against the Copernicus trajectory
tool [@copernicus2010]; measurement partials are validated against finite differencing;
and filters are validated using real mission tracking data.

# Research Impact Statement

SCB is developed in direct support of the Emirates Mission to the Asteroid Belt (EMA)
[@ema2021], a multi-target deep-space mission planned for launch in 2028. Operational
navigation analyses for EMA—including trajectory propagation, covariance analysis, and
maneuver targeting—have been performed with SCB and are documented in the literature
[@kuleib2026maneuver; @mcmahon2026issfd].

Real-data validation has been demonstrated using two independent mission datasets. OSIRIS-REx
two-way sequential ranging and Doppler measurements from DSN station DSS-35 (Canberra)
were processed against the publicly archived reference trajectory, with post-fit residuals
statistically consistent with the assigned noise levels [@mcmahon2026issfd]. A second
validation arc uses Emirates Mars Mission (EMM) tracking data through multiple DSN passes,
exercising different measurement geometries and ground-station antenna configurations.
These cases provide reproducible benchmarks for external users.

The software has been presented at three peer-reviewed conference proceedings since 2025:
the 47th AAS Guidance, Navigation and Control Conference [@mcmahon2025gnc], the AAS/AIAA
Astrodynamics Specialist Conference [@mcmahon2025astro], and the 30th International
Symposium on Space Flight Dynamics [@mcmahon2026issfd]. SCB is developed collaboratively
across four institutions in two countries, and its twelve-notebook tutorial suite—covering
basics through real-data OD—lowers the barrier to entry for new research groups.

# AI Usage Disclosure

<!-- AUTHORS: Complete this section. If no AI tools were used during software
development, documentation writing, or paper authoring, state this explicitly.
If AI tools were used, describe the tool (name and version), the nature of
assistance (e.g., code generation, documentation drafting, copy-editing),
and confirm that all AI-assisted outputs were reviewed and validated by human
authors. -->

[To be completed by authors.]

# Acknowledgements

The development of Scarabaeus was supported by the United Arab Emirates Space Agency
through its knowledge partnership with the University of Colorado Boulder's Laboratory
for Atmospheric and Space Physics. The authors thank past members of the ORCCA
laboratory who contributed to earlier versions of the codebase: Annalise Cabra, Anivid
Faura-Pedros, Kian Shakerin, Chloe Long, Dahlia Baker, Jacopo Villa, Matthew Givens,
Spencer Boone, Santhosh Pattamudu-Manoharan, and Lars Hinüber.

# References
