.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _api:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=====================
:gold:`Scarabaeus API`
=====================
This reference manual details the modules, classes, and methods included in Scarabaeus. Each module 
contains classes relevant to itself. For more in-depth descriptions of each, 
see its respective section below.

----

Body
----
The Body module defines objects in the orbit determination process. 
This includes planets, ground stations, and spacecraft.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Body
   :maxdepth: 1

   body/body_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   body/body_index

Dynamics
----
The Dynamics module provides functionality for modeling the dynamic environment
acting on a spacecraft. This includes modeling of Keplerian accelerations, 3rd body perturbations,
solar radiation pressure, higher-order fidelity gravitational modeling (spherical harmonics, polyhedral),
and finite burn modeling.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Dynamics
   :maxdepth: 1

   dynamics/dynamics_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   dynamics/dynamics_index


Environment
----
The Environment module provides high-level representations used to define the
environment of an orbit-determination scenario. This includes state vector definitions,
trajectory containers, numerical propagators, and multi-leg mission sequences.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Environment
   :maxdepth: 1

   environment/environment_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   environment/environment_index

Finite Burn
----
The Finite Burn module provides functionality for modeling finite burn maneuvers,
including maneuver sequencing and parsing of maneuver definitions for use in trajectory
propagation and orbit determination.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Finite Burn
   :maxdepth: 1

   fb/fb_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   fb/fb_index

Guidance
----
The Guidance module provides functionality for guidance tasks such as
B-Plane targeting and maneuver planning.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Guidance
   :maxdepth: 1

   guidance/guidance_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   guidance/guidance_index

Measurements
----
The Measurements module provides functionality for both computed and observed
measurement modeling, multi-type measurement datasets, and computation of partial derivatives.
Supported observables include radiometric (range, range-rate, Doppler) and optical measurements.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Measurements
   :maxdepth: 1

   measurements/measurement_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   measurements/measurement_index

Orbit Determination
----
The Orbit Determination (OD) module implements OD and filtering algorithms,
including batch least-squares, linearized Kalman filter, square-root information filter,
and smoother variants. Also includes filter management, iteration control, and solution containers.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Orbit Determination
   :maxdepth: 1

   orbit_det/orbit_det_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   orbit_det/orbit_det_index

Spacecraft
----
The Spacecraft module provides functionalities that define various properties
of a spacecraft body, including sensors (imagers, antennas), actuators (thrusters), structural
elements, n-plate models, and power-generation models.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Spacecraft
   :maxdepth: 1

   spacecraft/sc_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   spacecraft/sc_index

Time & Frame
----
The Time and Frame module provides tools for handling reference frame and time
referencing systems. A dedicated set of classes manages reference frame definition,
transformation, and storage, as well as epoch representation and SPICE integration.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Time & Frame
   :maxdepth: 1

   time_n_frame/time_n_frame_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   time_n_frame/time_n_frame_index

Units
----
The Units module provides tools to manage both physical dimensions and
units across Scarabaeus, ensuring correct operations, clear interfaces, and continuity when
performing calculations using physical properties.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Units
   :maxdepth: 1

   units/units_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   units/units_index

Uncertainty Quantification
----
The Uncertainty Quantification module provides tools for uncertainty quantification in the
orbit determination process, including covariance matrix construction, propagation, and analysis.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Uncertainty Quantification
   :maxdepth: 1

   uq/uq_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   uq/uq_index

Utilities
----
The Utilities module provides utility tools leveraged across the entirety
of Scarabaeus, including constants definitions, noise generation, orbital element conversion,
polynomial representations, and plotting utilities.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Utilities
   :maxdepth: 2

   utils/utils_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   utils/utils_index

Scarabaeus Rust
----
The Scarabaeus Rust contains computationaly expensive algorithms written 
in Rust instead of Python and exposes them as classes and functions to 
the rest of Scarabaeus.

.. sidebar toctree
.. toctree::
   :hidden:
   :caption: Scarabaeus Rust
   :maxdepth: 1

   scb_rust/scb_rust_index

.. display toctree
.. toctree::
   :maxdepth: 1
   :class: two-column-toc

   scb_rust/scb_rust_index