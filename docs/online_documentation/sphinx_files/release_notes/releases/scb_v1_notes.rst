.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Scarabaeus version 1.0 release notes
    :keywords:
        Scarabaeus, release, version 1.0

.. _scb-v1-notes:

.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=====================================
:gold:`Scarabaeus 1.0.0 Release Notes`
=====================================

.. contents::
   :local:
   :depth: 2

----

Release Notes
-------------

Core Framework
^^^^^^^^^^^^^^

- All major dynamical models implemented and validated
- Full interplanetary orbit determination capability implemented
- Rust interface implemented for performance-critical components
- Filtering framework considered feature complete:

  - Batch filtering
  - Sequential filtering
  - Smoothing
  - Multi-arc estimation
  - Stochastic parameter estimation

- Finite-burn estimation and targeting implemented

Measurement Models
^^^^^^^^^^^^^^^^^^

- Core radiometric and optical measurement models implemented
- Measurement editing and manipulation utilities implemented

Utilities
^^^^^^^^^

- File saving/export utilities implemented
- Plotting and visualization tools implemented

----

Known Issues
------------

General
^^^^^^^

- General bugs may still be discovered as the framework is under active development
  and many new capabilities are continuously being introduced

Performance
^^^^^^^^^^^

- Plotting performance and plotting-library consistency can still be improved

Real Measurement Models
^^^^^^^^^^^^^^^^^^^^^^^

- Real radiometric measurement models assume the spacecraft is outside Earth's
  sphere of influence.  Geocentric-frame corrections required are not yet fully supported.
- Partial derivatives of media corrections (tropospheric and ionospheric
  parameters) with respect to filter solve-for parameters are not yet implemented.
- Ground-station and bias partial derivatives for real radiometric measurements
  remain cumbersome and require refactoring

----

Project History
---------------

The repository was originally opened on May 10, 2022.

Over the years, the project has involved contributions from many people and has grown
to nearly 4000 commits on the ``develop`` branch.

Most active development currently happens in an internal GitLab repository owned by
the university.  As a result, the public repository only reflects a subset of the
ongoing development activity, issues, and experimental features.

Public releases are planned to occur concurrently with major development milestones
and scientific or operational achievements.
