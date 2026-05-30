.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _examples:
.. _learn:

.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

==========================
:gold:`Scarabaeus Tutorials`
==========================

Hands-on Jupyter notebooks that walk through Scarabaeus from first principles to
full orbit-determination campaigns.  Notebooks are organized in three tiers — **Basics**,
**Intermediate**, and **Advanced** — and can be run in order or consulted independently.

Before starting, see the :ref:`tutorials primer <tuts-setup>` for setup instructions and
how ancillary data (SPICE kernels, measurement files) is organised.

.. toctree::
    :hidden:
    :caption: Setup

    tuts_setup

----

Basics
------
Core Scarabaeus types: physical quantities with units and frames, epoch arrays,
SPICE kernel management, and celestial body ephemerides.

.. toctree::
    :hidden:
    :caption: Basics

    /_collections/tutorials/basics_AWU_and_AWF
    /_collections/tutorials/basics_EpochArray
    /_collections/tutorials/basics_SpiceManager_CelestialBody_Constants

.. nbgallery::

    /_collections/tutorials/basics_AWU_and_AWF
    /_collections/tutorials/basics_EpochArray
    /_collections/tutorials/basics_SpiceManager_CelestialBody_Constants

----

Intermediate
------------
Spacecraft modelling, numerical integration, trajectory representation, maneuver
planning, B-plane targeting, and measurement simulation.

.. toctree::
    :hidden:
    :caption: Intermediate

    /_collections/tutorials/intermediate_Propagator_and_Trajectory
    /_collections/tutorials/intermediate_nPlateModel_and_Attitude
    /_collections/tutorials/intermediate_Measurements
    /_collections/tutorials/intermediate_MissionSequence_and_FiniteBurn
    /_collections/tutorials/intermediate_BPlane

.. nbgallery::

    /_collections/tutorials/intermediate_Propagator_and_Trajectory
    /_collections/tutorials/intermediate_nPlateModel_and_Attitude
    /_collections/tutorials/intermediate_Measurements
    /_collections/tutorials/intermediate_MissionSequence_and_FiniteBurn
    /_collections/tutorials/intermediate_BPlane

----

Advanced — Synthetic Measurements
-----------------------------------
Full orbit-determination pipelines driven by ideal (noise-only, no systematic errors)
simulated range and range-rate data.  Covers batch and sequential filters, process noise,
RTS smoothing, multi-leg arcs with impulsive maneuvers, and consider parameters.

.. toctree::
    :hidden:
    :caption: Advanced — Synthetic Measurements

    /_collections/tutorials/advanced_IdealMSR_BatchOD
    /_collections/tutorials/advanced_IdealMSR_SequentialOD

.. nbgallery::

    /_collections/tutorials/advanced_IdealMSR_BatchOD
    /_collections/tutorials/advanced_IdealMSR_SequentialOD

----

Advanced — Real Measurements
------------------------------
End-to-end OD with real tracking data: tropospheric and ionospheric media corrections,
and a complete OSIRIS-REx orbit-determination campaign using DSN radiometric observables.

.. toctree::
    :hidden:
    :caption: Advanced — Real Measurements

    /_collections/tutorials/advanced_RealMSR_MediaCorrections
    /_collections/tutorials/advanced_RealMSR_OSIRIS_REx_OD

.. nbgallery::

    /_collections/tutorials/advanced_RealMSR_MediaCorrections
    /_collections/tutorials/advanced_RealMSR_OSIRIS_REx_OD
