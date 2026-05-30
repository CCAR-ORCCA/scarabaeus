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

=============================
:gold:`Scarabaeus Quickstart`
=============================

The example below propagates a spacecraft in a low-Earth orbit for 90 minutes using a
two-body gravity model. It covers the core Scarabaeus workflow: units, frames, SPICE kernels,
spacecraft definition, force model, and propagation.

.. code-block:: python

   import scarabaeus as scb
   import numpy as np

   # define units and frame
   kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])
   J2000 = scb.Frame('J2000')

   # Load SPICE kernels
   scb.SpiceManager.load_kernel_from_mkfile("path/to/metakernel.tm")

   # Define a spacecraft
   sc = scb.Spacecraft(
       name     = "Orbiter",
       spice_id = -1000,
       tot_mass = scb.ArrayWUnits(2000.0, kg),
       area     = scb.ArrayWUnits(1e-6, km**2),
       ref_coeff= 1.5,
   )

   # Define the central body
   earth = scb.CelestialBody.from_constants("EARTH")

   # Initial state (position [km], velocity [km/s])
   r0 = scb.ArrayWUnits(np.array([6778.0, 0.0, 0.0]), km)
   v0 = scb.ArrayWUnits(np.array([0.0, 7.784, 0.0]), km / sec)
   state = scb.StateArray(earth, J2000, sc, r0, v0)

   # Define epochs
   t0 = scb.EpochArray("2024-JAN-01 00:00:00.000 TDB", "TDB")
   tf = scb.EpochArray("2024-JAN-01 01:30:00.000 TDB", "TDB")

   # Build force model and propagate
   gravity = scb.PointMassGravity(earth)
   fm = scb.ForceModelTranslation([gravity])

   prop = scb.Propagator(spacecraft=sc, force_model=fm, epoch_start=t0, epoch_end=tf)
   traj = prop.propagate(state)

.. note::

   The SPICE metakernel path above is illustrative. Replace it with the path to
   your own metakernel file. See :ref:`Installation <installation>` for details.

-----------------------------------
:gold:`Next Steps`
-----------------------------------

The tutorial suite in :ref:`Examples <examples>` walks through progressively
more complex workflows:

- **Basics** — units, epochs, frames, spacecraft definition, Keplerian propagation
- **Intermediate** — perturbation models, mission sequences, measurement simulation
- **Advanced** — orbit determination with simulated and real DSN measurements,
  multi-arc OD, finite burn modelling, parameter estimation

See the :ref:`API Reference <api>` for full documentation of every class and method.
