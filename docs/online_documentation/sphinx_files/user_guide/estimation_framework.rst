.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        The Scarabaeus estimation framework: filters, iteration and convergence, process noise, consider parameters, editing and multi-arc OD.
    :keywords:
        Scarabaeus, orbit determination, Kalman filter, SRIF, batch least squares, process noise, multi-arc

.. _estimation-framework:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

============================
:gold:`Estimation Framework`
============================
Last revised by G. Fereoli on 2026 OCT 11

Every Scarabaeus filter solves the same linearised problem. The reference trajectory
:math:`X^*(t)` is propagated with its STM, and the filter estimates the deviation
:math:`x = X - X^*` from the residuals :math:`y = O - C`:

.. math::

    y_k = \tilde{H}_k\,\Phi(t_k, t_0)\,x_0 + \epsilon_k ,
    \qquad
    J = \sum_k \epsilon_k^T R_k^{-1} \epsilon_k
        + (x_0 - \bar{x}_0)^T \bar{P}_0^{-1} (x_0 - \bar{x}_0) .

``fit`` adds the estimated deviation to the reference, propagates the new reference,
and repeats until the deviation or the cost stops changing. The
:ref:`Dynamics <dynamics-concepts>` page covers the reference and its STM, and the
:ref:`Measurement Framework <measurements-framework>` page covers :math:`O - C` and
:math:`\tilde{H}`.

Filters
-------
All filters derive from :class:`~scarabaeus.FilterOD` and share the same constructor,
``fit`` method and :class:`~scarabaeus.SolutionOD` output.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Filter
     - Form
     - Use it for
   * - :class:`~scarabaeus.LKF`
     - Sequential, covariance
     - Process noise; RTS smoothing. A consider analysis uses the Schmidt gain
       with a Joseph update.
   * - :class:`~scarabaeus.SRIF`
     - Sequential, square-root information
     - Same capabilities as the LKF, with better numerical conditioning. Prefer it
       for long arcs or poorly conditioned problems.
   * - :class:`~scarabaeus.LSB`
     - Batch, normal equations
     - Short arcs with deterministic dynamics. No process noise and no smoother.
   * - :class:`~scarabaeus.SRIFB`
     - Batch, square-root information
     - The batch problem solved through QR factorisation.
   * - :class:`~scarabaeus.CustomFilter`
     - User-supplied
     - Your own estimator (UKF, particle, mixture, ...) running on the Scarabaeus
       reference, residuals, partials and STMs.

Setting up a run
----------------
.. code-block:: python

    P0 = scb.CovarianceMatrix([scb.ArrayWUnits(3.0, km)] * 3
                              + [scb.ArrayWUnits(3e-3, km / sec)] * 3,
                              t0, from_list=True)                 # sigmas
    Q = scb.CovarianceMatrix([scb.ArrayWUnits(1e-9, km / sec**2)] * 3,
                             t0, frame="RTN", from_list=True)
    settings = scb.FilterSettings(
        initial_covariance=P0,
        process_noise=scb.ProcessNoiseSettings(type="SNC", Q_cont=Q),
        output=scb.OutputSettings(metadata={"filter": "SRIF-SNC"}),
    )
    meas = scb.MeasurementSpec.many(
        scb.MeasurementSpec(model=range_gs1, observed_meas=obs_range, dataset_name="GS1 Range"),
        scb.MeasurementSpec(model=doppler_gs1, observed_meas=obs_dop, dataset_name="GS1 Doppler"),
    )
    srif = scb.SRIF(propagator=prop, settings=settings, measurements=meas,
                    traj_name="ref.bsp", traj_dir="kernels")
    solution, n_iters, converged = srif.fit(max_iterations=10, if_sequential_smooth=True)

``propagator`` can also be a :class:`~scarabaeus.MissionSequence`. In that case the
filter estimates every leg and carries the covariance across burns and nodes.

Convergence and iteration control
---------------------------------
``fit`` applies two tests: the RMS of the state deviation against
``convergence_threshold``, and the relative change of the normalised cost :math:`J/n`
against ``convergence_threshold_cost``. ``convergence_criterion`` combines them:

- ``"any"`` (default): either test passes.
- ``"deviation"``: the deviation test only.
- ``"cost"``: the cost test only. Use it for weakly observable problems, where the
  deviation keeps wandering along unobservable directions.
- ``"both"``: both tests pass.

``callback(filter, iteration, solution)`` runs after each iteration. Use it to
reweight or edit the data, for example with
``filt.set_measurement_sigmas({"GS1 Range": scb.ArrayWUnits(5.0, m)})``.

Process noise
-------------
Process noise is accepted by the sequential filters only. Pass one
:class:`~scarabaeus.ProcessNoiseSettings`, or a list with one entry per propagated body
(``body=``).

- ``"SNC"``: white unmodelled acceleration, :math:`Q_c` in km²/s⁴.
- ``"DMC"``: a first-order Gauss–Markov acceleration ``a_fogm`` in the state, driven by
  noise in km²/s⁶. Its decay rates :math:`\beta` come from the dynamics.

A covariance given in ``frame="RTN"`` is rotated along the reference trajectory.
``max_interval`` caps how much noise accumulates across a data gap.

Consider parameters and covariance analysis
-------------------------------------------
A component declared with ``estimation="considered"`` is not corrected. Its a-priori
uncertainty is still mapped into the estimated states through the sensitivity matrix,
in every filter. With ``covariance_analysis=True`` in :class:`~scarabaeus.FilterSettings`,
the filter makes one pass with no state update. It returns the formal covariance
that a given tracking schedule would deliver, so the observations only need to supply
epochs.

Data handling
-------------
- **Editing** (``editing_method``): ``"chi2"``, ``"lasso"`` and ``"kurtosis"`` act per
  dataset, and their ``*_all`` variants act on all data together. ``"date_range"`` and
  ``"interactive_date_range"`` remove time windows. ``editing_datasets`` and
  ``editing_kwargs`` restrict and tune the editing.
- **Weights**: ``measurement_sigmas`` in the settings, or ``set_measurement_sigmas``
  between iterations.
- **Outputs**: :class:`~scarabaeus.OutputSettings` selects which histories are kept
  (pre- and post-fit residuals, covariance, consider covariance, gains, NIS) and where
  the solution and data files are written.
- **Observability**: ``filter.observability_test()`` returns the information matrix
  :math:`\Lambda = \sum_k (H_k\Phi_{k,0})^T R_k^{-1} (H_k\Phi_{k,0})` and the stacked
  observability matrix, before any data is fitted.

Multi-arc estimation
--------------------
:class:`~scarabaeus.MultiFilterOD` runs several independent arc filters, each with its
own filter type, and then fuses the parameters they share:

.. code-block:: python

    multi = scb.MultiFilterOD([srif_arc1, srif_arc2, lsb_arc3],
                              global_param_names=["eta_srp"],
                              global_apriori_cov=P_global)
    solutions = multi.fit(max_outer_iters=5, max_inner_iters=10)

Each arc's normal equations are reduced to the global block with a Schur complement,
following NAPEOS. The reduced blocks are summed, and the combined global estimate is
fed back to the arcs until it settles. With ``subtract_arc_priors=True``, each arc's
own prior on a global parameter is removed, so ``global_apriori_cov`` is counted only
once.

See also
--------
- :ref:`Solution Mapping and Analysis <solution-mapping-analysis>`: what to do with the
  returned :class:`~scarabaeus.SolutionOD`.
- The *Ideal MSR Sequential OD*, *Ideal MSR Batch OD* and *Real MSR* tutorials in
  :ref:`Examples <examples>`.
