---
title: 'Scarabaeus: An Open-Source Tool for Interplanetary Spacecraft Navigation'
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
  - name: Trevor N. Wolff
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
    affiliation: 1
  - name: Jacopo Villa
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
date: 31 May 2026
bibliography: paper.bib
---

# Summary

Scarabaeus (SCB) is an open source spacecraft orbit determination (OD) software package developed by the Orbital Research Cluster for Celestial Applications (ORCCA) at the University of Colorado Boulder. The framework combines a modular, object oriented Python front end with a hybrid Python and Rust back end for computationally intensive routines, providing a complete end to end navigation environment. SCB supports high fidelity trajectory propagation using a configurable suite of dynamical models; simulation, ingestion, and processing of radiometric measurements, including two way coherent Doppler, sequential ranging, range and range rate, and differential one way ranging (Delta DOR); as well as optical measurements such as center finding and landmark observations. The software includes a comprehensive estimation framework spanning batch and sequential filtering, consider and stochastic parameters, multi arc estimation, square root information filters and smoothers, and multiple process noise formulations. SCB leverages NASA's SPICE toolkit [@acton1996] for mission grade handling of time systems, reference frames, and ephemerides.

Development of SCB is primarily motivated by the Emirates Mission to the Asteroid Belt (EMA) [@parker2024ema], which will launch the MBR Explorer spacecraft in 2028 to conduct six main belt asteroid flybys before rendezvous with asteroid (269) Justitia. Consequently, the software has been designed with a strong emphasis on interplanetary navigation and proximity operations around small bodies. At the same time, its modular architecture and broad measurement model support make it applicable to a wide range of navigation problems, including Earth orbiting and cislunar missions.


# Statement of Need

Spacecraft navigation requires solving a complex estimation problem that spans nearly every aspect of astrodynamics. High fidelity trajectory propagation must account for gravitational and non gravitational perturbations, while heterogeneous tracking data must be modeled and processed with comparable levels of fidelity. Orbit determination algorithms must then iteratively estimate the spacecraft state, dynamical and measurement model parameters, and associated uncertainties, ultimately producing statistically consistent state estimates and covariance information.

The state of the art in spacecraft orbit determination software is well established. Mature and comprehensive tools include MONTE [@monte2018], GEODYN [@nicholas2025geodyn], ODTK [@vallado2010odtk], Tudat [@gisolfi2025tudat], and GODOT [@godot]. While other navigation tools exist, many are developed for specific missions or organizations and are not intended as general purpose, mission agnostic frameworks. Among the major publicly available systems, Tudat is currently the only fully open source solution, highlighting the need for additional community developed and openly accessible navigation software.

SCB contributes to this ecosystem by providing a comprehensive orbit determination framework that combines the flexibility required for research with the robustness needed for real mission applications. Its development is driven by the requirements of the Emirates Mission to the Asteroid Belt, ensuring that the software is continuously exercised against realistic operational scenarios. The primary audience for SCB includes the astrodynamics research community and mission teams seeking a flexible, transparent, and extensible open source platform for navigation analysis.

# State of the Field

Among open source orbit determination frameworks, the most comprehensive alternative is **Tudat** [@gisolfi2025tudat], a mature astrodynamics and estimation toolkit built around a C++ core with a Python interface (TudatPy) that now supports radiometric tracking data analysis. SCB is best viewed as a complementary effort, developed natively in Python and Rust and driven by the operational requirements of an active deep space mission.

While both frameworks provide high fidelity dynamical modeling and estimation capabilities, SCB places particular emphasis on the day to day needs of spacecraft navigation and flight dynamics teams. Beyond orbit determination itself, the software includes tools for measurement editing and solution quality control, B plane mapping and targeting, database backed mission infrastructure to support multiple operators working simultaneously, and maneuver design capabilities including local optimization of finite burns. The goal is to provide a coherent environment in which the full navigation workflow can be performed, from data processing and state estimation to operational analysis and trajectory design.

Rather than reimplementing general numerical infrastructure, SCB builds upon a mature open source ecosystem. The software leverages SpiceyPy [@annex2020spiceypy] for geometry, reference frames, and ephemerides; SciPy [@virtanen2020scipy] and PyASA [@rein2015ias15] for numerical methods and high accuracy propagation; MongoDB [@mongodb] and PyMongo [@pymongo] for data management; Sphinx [@sphinx] for documentation; and Pytest [@pytest] for testing and verification.


# Software Design

**Architecture and design principles.** SCB combines an object oriented Python front end, which exposes user facing classes and scripting interfaces, with Python and Rust back end components reserved for performance critical tasks such as numerical propagation, dynamical model evaluation, and estimation algorithms. The codebase is guided by a small set of design principles: modularity through clear class responsibilities; *contextualization*, whereby every piece of data is owned by a specific object and remains traceable throughout its lifecycle; *unit typing*, in which physical units are attached to quantities through the `ArrayWUnits` abstraction and propagated through computations; and *frame awareness*, in which vectors carry explicit information about the reference frame and origin in which they are defined through the `ArrayWFrame` abstraction. SCB also emphasizes the reuse of mature open source libraries whenever possible.

Unit and frame awareness are particularly important in spacecraft navigation, where subtle inconsistencies can lead to significant operational consequences. A well known example is the loss of the Mars Climate Orbiter, whose failure was ultimately traced to a mismatch between metric and imperial units in ground software used for trajectory modeling [@mco1999]. By explicitly associating units, reference frames, and origins with numerical quantities, SCB aims to reduce the likelihood of such errors and improve the transparency and safety of navigation analyses.


**Dynamical models.** SCB implements point mass and N body gravity, spherical harmonic gravity, cannonball and N plate solar radiation pressure, impulsive and finite burn maneuvers, and stochastic acceleration models. These models have been validated against the Copernicus trajectory design and analysis tool [@williams2025copernicus]. Future releases will place greater emphasis on the modeling of coupled binary asteroid dynamical environments, motivated by applications to small body missions. In addition to state propagation, all dynamical models provide the variational equations and partial derivatives required for orbit determination, filtering, smoothing, and covariance analysis.

**Measurement models and the Deep Whale Network.** SCB supports range, range rate, two way coherent Doppler, sequential ranging, differential one way ranging (Delta DOR), and optical measurements including center finding and landmark observations. Operational models are currently available for two way coherent Doppler and sequential ranging, while Delta DOR support is under active development. Radiometric observables follow the formulations of Moyer [@moyer2000], with round trip light times computed iteratively from SPICE based ground station and spacecraft states. The measurement models account for relativistic light time corrections, antenna and transponder delays, solar corona path delays, and time system conversions, while tropospheric and ionospheric effects are applied as measurement corrections. Real tracking data are preprocessed by the Deep Whale Network (DWN), a companion ORCCA Python library that parses TRK 2 34 Tracking and Navigation Files using PyTrk234 [@pytrk234] into a standardized JSON representation, while preserving the original observables and appending metadata such as SPICE station identifiers and outlier flags.

**Filtering framework.** Estimation is organized around `FilterOD` (the abstract
estimation workflow), `FilterDataManager` (a measurement aggregator), and `SolutionOD` (a
results container), with a typed `StateArray` that tags each state element as static or
dynamic and as estimated or considered. Four filters are implemented—a linearized Kalman
filter, a least-squares batch estimator, and sequential and batch square-root information
filters [@bierman1977], with Rauch–Tung–Striebel smoothing for the sequential cases.
Estimable parameters span the spacecraft state, SRP coefficient, asteroid ephemeris and
gravity, station locations, measurement biases, impulsive maneuvers, and stochastic
accelerations. Process noise uses State Noise Compensation, with Dynamical Model
Compensation and a batch stochastic mode, and outlier rejection is supported between
iterations.

**Verification and documentation.** TODO ZACK

# Research Impact Statement

SCB is developed in direct support of the Emirates Mission to the Asteroid Belt
(EMA) [@parker2024ema], led by the UAE Space Agency, and is designed to process the
mission's radiometric and optical-navigation measurements across all flight phases. Its
dynamical models, together with their associated partial derivatives, have been validated
against the state-of-the-art high-fidelity propagator Copernicus [@williams2025copernicus]
across more than a dozen test cases spanning a range of dynamical combinations. A
representative comparison is shown in \autoref{fig:copernicus} for Case 8, a two-body
problem with the Sun as the central body augmented by an impulsive maneuver.

The measurement and filtering implementations have been exercised against both real and
simulated data in three validation cases [@mcmahon2026issfd]: (i) two-way sequential
ranging and Doppler from OSIRIS-REx DSN passes (DSS-35, Canberra, July 2018), processed
against the publicly archived reference trajectory; (ii) Emirates Mars Mission (EMM)
tracking arcs collected through multiple DSN stations; and (iii) a fully simulated asteroid
flyby that jointly estimates the spacecraft state and two trajectory-correction maneuvers
from combined radiometric and optical data. In each case the post-fit residuals are
centered and statistically consistent with the assigned measurement noise
(\autoref{fig:residuals}), indicating that systematic effects are absorbed into the
estimated parameters.

![Comparison of SCB against Copernicus for Case 8: a two-body problem with the Sun as the
central body and an impulsive maneuver.\label{fig:copernicus}](figures/picture1.png)

![Example post-fit two-way sequential range and Doppler residuals from an OSIRIS-REx DSN
tracking pass (DSS-35, Canberra), estimated with the least-squares batch filter. Residuals
are centered and statistically consistent with the assigned measurement noise.\label{fig:residuals}](figures/picture2.png)

SCB has been presented in several conference proceedings [@mcmahon2026issfd]. While the
software is already well suited to interplanetary navigation, future development will focus
on improved models for small-body proximity operations. This work will introduce
new terrain-relative navigation observables, estimation of asteroid-specific parameters,
enhanced multi-arc filtering, and, potentially, support for binary asteroid systems. Real
Delta-DOR measurements using quasar references are also under development, alongside
finite-burn targeting capabilities [@kuleib2026maneuver].

# AI Usage Disclosure

The core software implementation, algorithms, and architectural design of Scarabaeus
were developed by the authors. Generative AI has been used to assist with code refactoring
and generation of unit and integration tests. This paper was drafted by the authors and
subsequently revised with AI assistance for restructuring according to JOSS format requirements
and language editing. All technical content, examples, and claims reflect the authors'
work and judgment.

# Acknowledgements

The development of Scarabaeus was supported by the United Arab Emirates Space Agency
through its knowledge partnership with the University of Colorado Boulder's Laboratory
for Atmospheric and Space Physics. The authors thank past members of the ORCCA
laboratory who contributed to earlier versions of the codebase: Annalise Cabra, Anivid
Faura-Pedros, Kian Shakerin, Chloe Long, Dahlia Baker, Matthew Givens, Spencer Boone,
Santhosh Pattamudu-Manoharan, and Lars Hinüber.

# References
