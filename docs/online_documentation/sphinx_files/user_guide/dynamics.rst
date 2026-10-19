.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        How Scarabaeus defines a state, models the forces acting on it, and propagates it with its state transition matrix.
    :keywords:
        Scarabaeus, dynamics, force model, propagator, state transition matrix, mission sequence

.. _dynamics-concepts:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

================
:gold:`Dynamics`
================
Last revised by G. Fereoli on 2026 OCT 11

Three objects define the dynamics. A :class:`~scarabaeus.StateArray` holds what is
propagated and estimated. A :class:`~scarabaeus.ForceModelTranslation` selects the
accelerations. A :class:`~scarabaeus.Propagator` integrates the state together with its
state transition matrix (STM), which the filters need.

The state
---------
A :class:`~scarabaeus.StateDefinition` is built by chaining components and is then
attached to an epoch and an origin:

.. code-block:: python

    eta0 = scb.ArrayWFrame(scb.ArrayWUnits(np.array([1.0]), None), None)
    state = (scb.StateDefinition()
             .position(sc, r0)
             .velocity(sc, v0)
             .param("eta_srp", sc, eta0, dynamics="static"))
    x0 = scb.StateArray(epoch=t0, origin=earth, state=state)

Every component takes two modes:

- ``estimation``: ``"estimated"`` (corrected by the filter) or ``"considered"``
  (enters the covariance only).
- ``dynamics``: ``"dynamic"`` (propagated) or ``"static"`` (constant over the arc).

Dynamic components come before static ones, and estimated components come before
considered ones. Each ``.velocity()`` immediately follows its ``.position()``. The
:class:`~scarabaeus.StateDefinition` docstring lists every allowed component name.
Examples are SRP and thermal scale factors, maneuver Δv, finite-burn angles, empirical
accelerations, gravity coefficients, measurement biases, and station and Earth
orientation offsets.

Force models
------------
:class:`~scarabaeus.ForceModelTranslation` takes the spacecraft and one flag per model.
The point mass of the propagation origin is always included, unless polyhedral gravity
replaces it. Call ``list_forces()`` to print the active models.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Group
     - Flags
   * - Gravity
     - ``third_bodies`` (list of :class:`~scarabaeus.CelestialBody`); ``sph_harm`` with
       ``sph_harm_body``/``sph_harm_order``/``sph_harm_cs_file``, or ``sph_harm_bodies``
       for several fields (one dict per body); ``polyhedral_gravity`` with
       ``polyhedron_shape`` and ``polyhedron_density`` or ``polyhedron_gm``
   * - Radiation
     - ``cannonball_SRP`` (scaled by ``eta_srp``); ``nplate_SRP`` (takes precedence);
       ``thermal_radiation``; ``antenna_radiation_pressure``; ``yarkovsky_effect``
   * - Maneuvers
     - ``maneuver`` (impulsive or finite); ``smoothed_maneuvers``; ``finite_burn``
   * - Empirical
     - ``first_order_gauss_markov``, ``piecewise_first_order_gauss_markov``,
       ``fourier_acceleration``, ``piecewise_polynomial``. Each has a ``*_frame``
       option: ``"inertial"``, ``"RTN"`` or ``"body_fixed"``
   * - User-defined
     - ``custom_models``: a list of :class:`~scarabaeus.CustomDynamicModel`. Their
       :class:`~scarabaeus.CustomParameter` objects are estimated like built-in ones

.. code-block:: python

    fm = scb.ForceModelTranslation(
        primary_body=sc,
        third_bodies=scb.CelestialBody.from_constants(["SUN", "MOON"]),
        cannonball_SRP=True,
        sph_harm=True,
        sph_harm_bodies=[{"body": earth, "order": 8, "cs_file": egm_file}],
    )
    print(fm.list_forces())

The spherical-harmonic, n-plate SRP and polyhedron accelerations run in the Rust
extension. Third-body and Sun positions come from an ephemeris cache that is built once
per propagation instead of being queried from SPICE at every step. Rotational dynamics
are not modelled yet: body attitude comes from SPICE frames or from the attitude laws
of the n-plate model.

Propagation and the STM
-----------------------
The propagator integrates :math:`\dot{X} = f(X, t)` and, by default, the variational
equations

.. math::

    \dot{\Phi}(t, t_0) = A(t)\,\Phi(t, t_0), \qquad
    A(t) = \frac{\partial f}{\partial X}, \qquad
    \Phi(t_0, t_0) = I .

Static parameters have no dynamics of their own, but their STM columns record how
they move the dynamic states.

.. code-block:: python

    prop = scb.Propagator(primary_body=sc, state_vector=x0, tspan=epochs,
                          force_models=fm, integrator="DOP853")
    prop.propagate()

    traj = scb.Trajectory(state_array=prop.propagated_state_array)
    traj.write_to_spk("orbiter.bsp")

- ``integrator``: ``"DOP853"`` (default, tolerances ``atol``/``rtol`` = 2.2e-14) or
  ``"IAS15"`` (Rust). IAS15 does not propagate the STM yet, so it needs
  ``propagate_STM=False``.
- ``tspan``: an :class:`~scarabaeus.EpochArray` sets the output grid. A ``(t0, tf)``
  tuple outputs on the integrator's own steps, resampled ``n_densifier`` times.
- ``write_to_spk`` writes the SPK plus JSON sidecars with the STMs, the parameters and
  the propagator settings. The measurement models and filters read these sidecars.
  Each filter iteration writes its own reference SPK, named by ``traj_name`` and
  ``traj_dir``.

Mission sequences
-----------------
A :class:`~scarabaeus.MissionSequence` chains legs that each have their own propagator
and force model. Impulsive burns and zero-duration nodes join the legs.

.. code-block:: python

    mission = scb.MissionSequence(name="Mission")
    mission.addLeg("Cruise", prop=prop1)
    mission.addBurn("TCM-1", dV=dv, prop=prop1)
    mission.addLeg("Approach", prop=prop2)
    scenario = mission.propagate()

    traj = scb.Trajectory(leg_array=scenario)

``addFiniteBurn`` adds a thrust arc, and ``addNode`` adds an event marker. An estimated
impulsive burn needs a ``dv_man`` parameter in the next leg's state. A
:class:`~scarabaeus.MissionSequence` can be passed to any filter in place of a single
propagator.

See also
--------
- :ref:`Measurement Framework <measurements-framework>`: the models evaluated on the
  propagated reference.
- :ref:`Estimation Framework <estimation-framework>`: how the propagated reference and
  its STMs are used.
- The *Propagator and Trajectory*, *n-Plate Model and Attitude* and *Mission Sequence
  and Finite Burn* tutorials in :ref:`Examples <examples>`.
