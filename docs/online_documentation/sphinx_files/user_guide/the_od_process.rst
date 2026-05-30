.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC
.. _the_od_process:
.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

.. create a role that makes bolded text colored blue as well
.. raw:: html

    <style> .bold {font-weight: bold; color:rgb(2 119 189)} </style>

.. role:: bold

=======================================
:gold:`The Orbit Determination Process`
=======================================
Last revised by M. Pugliatti on 2025 FEB 4

The Deep Whale Network (DWN) pre-process the actual radiometric quantities that are observed by a network of ground-stations (e.g. Deep Space Network or DSN, ESTRACK, etc.) and OpNav data. In doing so, the DWN also generate auxiliary data needed by computed models (e.g. for tropo/iono corrections, frequency data, turn-around ratios, etc.)
These former quantities are referred to as the “observed” ones , while the latter as "auxiliary" data. 
These "auxiliary" data is feeded to SCB measurement models, which implement the computed formulations from [Moyer2000] for the available measurement types.


The estimation programs of SCB set fit the **computed** values of the observables to the **observed** values of the observables in a least squares (or lKF, SRIF, batch SRIF) sense by differentially correcting the values of the solve-for parameters. 
This process uses the **observed** -minus- **computed** residuals (R = O-C) and the partial derivatives of the **computed** values of the observables with respect to the solve-for parameter vector q calculated in SCB. 
The resulting estimated values of the solve-for parameters determine the trajectory of the spacecraft.

For this process to work correctly, a variety of tasks need to be performed to provide the filters with the necessary data in an accurate and efficient manner. This is represented by the schematic below. 


|pic1| 

.. |pic1| image:: ../../../images/OD_cartwheel.png
   :width: 100%

More details about the OD process can be found in the following sources: 

- **[Moyer2000]** T.D. Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation*, JPL/NASA, 2000.  
- **[Thornton2000]** C.L. Thornton and J.S. Border, *Radiometric Tracking Techniques for Deep-Space Navigation*, Wiley, 2000.  
- **[Montenbruck2000]** O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods, and Applications*, Springer, 2000.  
- **[Tapley2004]** B.D. Tapley, B.E. Schutz, and G.H. Born, *Statistical Orbit Determination*, Elsevier, 2004.