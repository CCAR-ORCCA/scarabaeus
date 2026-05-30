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

=========================
:gold:`Scarabaeus Roadmap`
=========================
Last revised on 2026 MAY 27 by G. Fereoli.

Performance Improvements
---------------------------
- Move performance-critical components from Python to Rust
- Improve propagation, filtering, and Monte Carlo scalability

Measurement and Filtering V&V
-------------------------------
- Document V&V of measurement models (DSN range, Doppler, range and range-rate) and analytical partial derivatives
- Compare covariance analysis results to state-of-the-art OD software
- Document data processing and filter results for multiple missions

Finite-Burn Targeting
------------------------
- Use Scarabaeus as a local targeting and optimization framework

Delta-DOR Implementation
----------------------
- Full Delta-DOR measurement modeling and estimation support

Terrain Relative Navigation
-------------------------------

Map-Based
~~~~~~~~~
- Landmark and catalogue-based navigation

Map-Free
~~~~~~~~
- Visual odometry and feature tracking

Proximity Operations
-----------------------
- Small-body geophysical estimation
- Gravity and rotational-state recovery
- Reference-frame switching and local-frame operations

Multi-Arc Filtering
----------------------

Binary Asteroid Simulator
-----------------------------
- General-use binary asteroid simulation environment
- Hera mission operational scenarios