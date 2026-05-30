.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. meta::
    :description lang=en:
        Scarabaeus history and motivation. Origin of the name.
    :keywords:
        Scarabaeus, history, motivation

.. _integration-testing:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=========================
:gold:`History and Motivation`
=========================
Scarabaeus (SCB) is an open-source software package for the interplanetary navigation of spacecraft.
Inspired by the successful development and use of the `Basilisk <https://hanspeterschaub.info/basilisk/>`_ framework
at the University of Colorado, SCB is developed by the `ORCCA laboratory <https://www.colorado.edu/faculty/mcmahon/orcca>`_
and aims to provide a comparably open and modular navigation tool that can be used
by any future interplanetary mission. SCB is built on a modular, object-oriented Python front-end with a Rust
back-end for computationally intensive components.

.. image:: ../../../images/scarabaeus_logo_dark.png
   :class: only-dark
   :width: 250
   :align: center

.. image:: ../../../images/scarabaeus_logo.png
   :class: only-light
   :width: 250
   :align: center

Mission Driver
--------------

The primary near-term driver for SCB is the `Emirates Mission to Explore the Asteroid Belt <https://space.gov.ae/en/projects-and-initiatives/space-exploration/emirates-mission-to-the-asteroid-belt>`_ (EMA), led by the
United Arab Emirates Space Agency, which will launch the MBR Explorer in 2028 to perform six main-belt asteroid
flybys and rendezvous with asteroid (269) Justitia. Once in orbit around Justitia, the spacecraft will conduct
a months-long remote-sensing campaign to characterize the body's composition and origins, including a
gravity-science investigation performed using a combination of ground-based navigation techniques and the
on-board scientific payload.

SCB is designed to support all phases of this mission, including the processing of radiometric tracking and
optical navigation measurements. The applicability of SCB, however, extends well beyond EMA: its modularity
and open-source license are intended to encourage external contributions and reuse on other deep-space missions
and for research purposes.

State of the Art
----------------

The current state of the art in orbit determination (OD) software for deep-space and Earth-orbiting applications
is well established. Mature tools include MONTE, GEODYN, ODTBX, ODTK, GMAT, TUDAT, GODOT, CubeNav, and Orbit14.
Among these, only GMAT, ODTBX, and TUDAT are open-source, underscoring the value of additional shared,
community-supported development across researcher groups.

Name Origin
-----------

Scarabs (*Scarabaeus satyrus*, also known as dung beetles) are known to navigate across the desert by looking at
the Milky Way. The tool has taken this name in their honor.