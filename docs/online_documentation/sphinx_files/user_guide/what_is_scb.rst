.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

===========================
:gold:`What is Scarabaeus?`
===========================

**Scarabaeus (SCB)** is an open-source Python framework for spacecraft navigation and orbit determination (OD).
It provides a unified, mission-agnostic environment for researchers, mission designers, and spacecraft operators
who need a modular, extensible tool for interplanetary and Earth-orbital navigation.

-------------------
:gold:`The Problem`
-------------------

Spacecraft navigation requires integrating trajectory propagation, force modelling, measurement processing, and
state estimation into a single coherent workflow. Existing tools (GEODYN, ODTK, GMAT, MONTE) are either
proprietary, tightly coupled to specific agencies, or lack the modularity needed for research-grade
experimentation. Scarabaeus fills this gap: a fully open-source, Python-first framework with a compiled Rust
back-end for performance-critical components.

----------------------
:gold:`What SCB Does`
----------------------

Scarabaeus supports:

- **Trajectory propagation** — Keplerian, multi-body, SRP, spherical harmonics, finite/impulsive burns, and
  stochastic accelerations (Gauss–Markov), using a high-accuracy IAS15 Rust integrator.
- **Measurement modelling** — Radiometric (range, range-rate, Doppler) and optical (centroiding, angular)
  observables, with real DSN ramp-table and media correction support.
- **Orbit determination** — Sequential filters (SRIF, LKF) and smoothers (LSB), batch OD, parameter
  estimation, covariance analysis, and measurement editing.
- **SPICE integration** — Unified time, reference-frame, and ephemeris management via SpiceyPy.
- **Units safety** — All physical quantities carry attached units, enforced at every calculation.

----------------------------
:gold:`Who Should Use SCB?`
----------------------------

Scarabaeus is designed for:

- **Researchers** developing or evaluating new navigation algorithms.
- **Mission designers** performing trade studies and trajectory/OD analysis.
- **Students** learning spacecraft navigation through hands-on, real-mission examples (e.g., OSIRIS-REx).
- **Engineers** who need a scriptable, testable alternative to commercial OD tools.

Python 3.11+ and familiarity with NumPy/SciPy are the only prerequisites. No prior OD experience is assumed
for the tutorial series.

.. seealso::

   :ref:`About Scarabaeus <about_scb_page>` — project motivation, team, and related publications.
