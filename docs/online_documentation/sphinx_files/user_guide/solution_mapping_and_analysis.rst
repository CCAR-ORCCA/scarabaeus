.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        Reading, mapping, saving and comparing Scarabaeus orbit determination solutions.
    :keywords:
        Scarabaeus, SolutionOD, covariance mapping, prediction, B-plane, solution comparison

.. _solution-mapping-analysis:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=====================================
:gold:`Solution Mapping and Analysis`
=====================================
Last revised by G. Fereoli on 2026 OCT 11

``fit`` returns a :class:`~scarabaeus.SolutionOD`. It holds the histories selected in
:class:`~scarabaeus.OutputSettings` at the measurement epochs:

- ``deviation_est``, ``deviation_cumulative`` and ``state_est`` (computed when first
  read).
- ``covariance_est`` and ``covariance_consider``.
- ``deviation_smooth`` and ``covariance_smooth``, when the run was smoothed.
- ``prefits``, ``postfits`` and ``postfits_smoother``.
- Statistics and debug quantities: gains, innovation covariances, NIS.

``solution.filter`` points back to the filter, and through it to the reference
trajectory and its STMs.

Mapping to other epochs
-----------------------
A solution is mapped to any epoch through the STMs of the fitted reference. For an
update selected at epoch :math:`r`:

.. math::

    A(t, r) = \Phi(t, t_0)\,\Phi(r, t_0)^{-1}, \qquad
    \hat{X}(t) = X^*(t) + A(t, r)\,\delta x(r), \qquad
    P(t) = A(t, r)\,P(r)\,A(t, r)^T .

The mapping is linear about the reference. It adds no process noise and no new data.
The data cutoff decides which update :math:`r` is used:

.. code-block:: python

    # filtered: each epoch uses the data up to that epoch
    filtered = solution.propagate_state_covariance(epochs, use_smoothed=False, lookback_s=0.0)
    # 14-day prediction: each epoch uses the data up to 14 days earlier
    predicted = solution.propagate_state_covariance(epochs, use_smoothed=False,
                                                    lookback_s=14 * 86400.0)
    # reconstruction: every epoch uses the final update
    reconstructed = solution.propagate_state_covariance(epochs, use_smoothed=False,
                                                        use_all_data=True)

``propagate_state`` and ``propagate_covariance`` take the same keywords.
``propagate_covariance`` is the one to use for covariance analyses, which store no
deviations.

- With ``lookback_s=None`` (the default), the original convention applies: the final
  update is used inside and after the arc, and the first update before it. This is not
  a causal product.
- An explicit cutoff cannot use the smoothed history, so set ``use_smoothed=False``.
- A cutoff before the first update falls back to the a-priori state and covariance.
- ``include_consider=True`` (default) maps the consider-augmented covariance.
- On a mission sequence, each leg is mapped with its own STMs. A positive lookback and
  ``use_all_data`` across sequence events raise ``NotImplementedError``.

Reading and saving
------------------
.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Returns
   * - ``estimated_trajectory(epochs)``
     - The estimate at measurement epochs (matched to within 1 µs). Use
       ``propagate_state`` for any other epoch.
   * - ``map_state_deviation_to_epoch()``
     - Corrections mapped to their nominal initial epochs, per leg on a sequence.
   * - ``state_covariance_to_elements(epochs)``
     - Keplerian elements :math:`(a, e, i, \Omega, \omega, M)` and their covariance.
   * - ``print_stats()``
     - Residual and covariance summary.
   * - ``save(path)``, ``create_JSON(...)``
     - One JSON or pickle file with every iteration (``iter_k``) and ``final``.
       Metadata covers state labels, frames and units, the a-priori state and
       covariance, the convergence test and provenance. The final covariance is
       stored either in full or as diagonals, never both.

Analysis tools
--------------
- :class:`~scarabaeus.CovarianceMatrix`: covariance with units, frame and epoch.
  Built from sigmas (``from_list=True``) with optional ``corr_factors``; read back with
  ``get_standard_deviations`` and ``to_array``.
- ``SigmaPoints`` and ``MonteCarloSamples`` (in
  ``scarabaeus.uncertaintyQuantification``): sample-based representations, for
  nonlinear propagation that the linear mapping above cannot capture.
- :class:`~scarabaeus.OrbitalElements`: ``covariance_to_elements(grav_param, r, v, P)``
  converts a Cartesian covariance to element space with the analytical Jacobian.
- :class:`~scarabaeus.Bplane`: ``compute_parameters`` gives :math:`B\cdot T`,
  :math:`B\cdot R` and the linearised time of flight. ``target_covariance`` and
  ``propagate_covariance_to_close_approach`` map a state covariance into the B-plane.
  ``linear_targeting``, ``nonlinear_targeting`` and ``piecewise_targeting`` compute
  corrections.
- ``SolutionCloseness`` (``scarabaeus.utils.SolutionCloseness``): asks whether two
  solutions are consistent, not just how far apart they are. It returns the number of
  sigma :math:`d` at which the two error ellipsoids first touch (Bhaskaran & Rush,
  2004).

  .. code-block:: python

      from scarabaeus.utils.SolutionCloseness import SolutionCloseness
      d = SolutionCloseness.between(sol_lkf, sol_srif, states=slice(0, 3))["d"]
      D = SolutionCloseness.matrix([sol_a, sol_b, sol_c])

  Both solutions must be in the same frame and units.
- :class:`~scarabaeus.Plotting`: ``plot_postfits_scatter``,
  ``plot_postfits_histogram``, ``plot_covariance_est``, ``plot_state_errors``,
  ``plot_parameters_estimate``, ``plot_orbit_3d`` and others. Each takes a solution
  and plots its smoothed or filtered history.

See also
--------
- :ref:`Estimation Framework <estimation-framework>`: how the solution is produced.
- The *B-Plane* tutorial and the OD tutorials in :ref:`Examples <examples>`.
