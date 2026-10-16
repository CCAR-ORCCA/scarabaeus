.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        Scarabaeus roadmap.
    :keywords:
        Scarabaeus, roadmap, future work

.. _integration-testing:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

==========================
:gold:`Scarabaeus Roadmap`
==========================
Last revised on 2026 OCT 16 by G. Fereoli.

Delta-DOR
---------
- Delta-DOR measurement model with analytical partials, estimated alongside DSN range
  and Doppler

Finite-Burn Optimization
------------------------
- Use Scarabaeus as a local targeting and optimization framework for finite burns

Optical Navigation
------------------
- Further optical navigation models with analytical partials
- Map-based landmark and catalogue navigation
- Map-free visual odometry and feature tracking

Rust Performance
----------------
- Move the measurement models (radiometric and optical) and their partials to Rust
- Move the filters (sequential and batch) to Rust

Measurement and Filtering V&V
-----------------------------
- Document V&V of measurement models (DSN range, Doppler, range and range-rate) and
  analytical partial derivatives
- Compare covariance analysis results to state-of-the-art OD software
- Document data processing and filter results for multiple missions

Binary Asteroid Simulator
-------------------------
- General-use binary asteroid simulation environment
- Hera mission operational scenarios

Completed in 2.0.0
------------------
- Spherical-harmonics, n-plate SRP and polyhedron gravity kernels, and the third-body
  ephemeris cache, moved to Rust
- Analytical partial derivatives of the real DSN radiometric models, including
  tropospheric and ionospheric zenith delays, so atmospheric parameters are estimated
  alongside the spacecraft state
- Geocentric light-time option for the real radiometric models
- Station location, Earth orientation, tide and IAU pole parameters estimable with
  analytical partials
- Polyhedral shape-model gravity, conversion of a shape model to spherical-harmonic
  coefficients, and several spherical-harmonic fields per spacecraft
