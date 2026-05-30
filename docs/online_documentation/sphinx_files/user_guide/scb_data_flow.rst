.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _scb-dataflow:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

============================
:gold:`Data Flow`
============================
Last revised by M. Pugliatti on 2025 FEB 4

As described in :ref:`the OD process <the_od_process>`, the DWN pre-process the actual radiometric quantities that are observed by a network of ground-stations (e.g. Deep Space Network or DSN, ESTRACK, etc.) and OpNav data. In doing so, the DWN also generate auxiliary data needed by computed models (e.g. for tropo/iono corrections, frequency data, turn-around ratios, etc.)
These former quantities are referred to as the “observed” ones , while the latter as "auxiliary" data. 
These "auxiliary" data is feeded to SCB measurement models, which implement the computed formulations from [Moyer2000] for the available measurement types. This data flow is represented in the image below, which shows how the database (DB) is going to interact with SCB and the DWN.

.. image:: ../../../images/data_flow.png
   :width: 100%

- **[Moyer2000]** T.D. Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation*, JPL/NASA, 2000.  
