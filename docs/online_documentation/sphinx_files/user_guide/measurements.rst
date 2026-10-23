.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        The Scarabaeus measurement framework: radiometric and optical models, simulating and reading observations, real DSN data, scheduling and biases.
    :keywords:
        Scarabaeus, measurement model, range, Doppler, optical navigation, DSN, MeasurementSpec, tracking schedule

.. _measurements-framework:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=============================
:gold:`Measurement Framework`
=============================
Last revised by G. Fereoli on 2026 OCT 11

A measurement model predicts what an instrument sees from a reference trajectory. The
instrument and the model are separate objects: a :class:`~scarabaeus.GroundStation`,
:class:`~scarabaeus.Camera` or spacecraft observes, and the model carries the physics.
Every model gives the filter the same three things: the observations :math:`O`, the
computed values :math:`C`, and the partials :math:`\tilde{H} = \partial h/\partial X`.

Models
------
All models derive from :class:`~scarabaeus.Measurement`.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Instrument
     - Observable
   * - :class:`~scarabaeus.RangeIdeal`, :class:`~scarabaeus.RangeRateIdeal`
     - Ground station
     - Geometric range and range rate
   * - :class:`~scarabaeus.DopplerIdeal`
     - Ground station
     - Doppler shift, two- or three-way (``ground_station_2``)
   * - :class:`~scarabaeus.DiffOneWayRangeIdeal`
     - Two ground stations
     - Difference of the one-way ranges
   * - :class:`~scarabaeus.AngularIdeal`
     - Spacecraft or body
     - Right ascension and declination of a target
   * - :class:`~scarabaeus.SequentialRangingReal`, :class:`~scarabaeus.DopplerReal`
     - Ground station (SPK)
     - DSN tracking data in range units and Hz
   * - :class:`~scarabaeus.CentroidingIdeal`
     - Camera
     - Image centroid of the target (two components)
   * - :class:`~scarabaeus.CustomMeasurement`
     - Any
     - Your own :math:`h(\mathbf r, \mathbf v, p, t)`, with
       :class:`~scarabaeus.CustomParameter` objects

The *ideal* models are instantaneous geometry and are meant for simulation. The *real*
models reproduce the DSN observables (Moyer, 2000): light time solved leg by leg,
ramped frequencies (:class:`~scarabaeus.RampTableManager`), relativistic delay
(``rlt_bodies``), solar corona, electronic delays and media corrections.

Simulate, read, compute
-----------------------
.. code-block:: python

    gs = scb.GroundStation("DSS-14")
    rng = scb.RangeIdeal(name="GS1 Range", instrument=gs, sigma=scb.ArrayWUnits(1e-3, km))

    # simulate on the true trajectory, then read back as observations
    rng.write_observed_measurements(target=truth, epoch_array=epochs, noisy=True,
                                    file_name="meas_range", folder_path_override="data")
    obs = rng.observed_measurements(file_name="data/meas_range.json",
                                    meas_name="meas_ideal", units=km)

    # computed values, partials and residuals on the reference
    _, _, _, comp = rng.computed_measurements(target=ref, epoch_array=epochs, frame=J2000)
    H = np.array(rng.compute_partials(ref, epochs, J2000))
    res = rng.residuals(obs.meas_obs, comp)

``observed_measurements`` returns an :class:`~scarabaeus.ObsData`:
``(meas_time_et, meas_sec, meas_obs, meas_outliers, aux_dict)``. The filter never
calls these methods directly. It receives a :class:`~scarabaeus.MeasurementSpec`
and rebuilds the :class:`~scarabaeus.MeasurementDataSet` objects on every iteration:

.. code-block:: python

    meas = scb.MeasurementSpec.many(
        scb.MeasurementSpec(model=rng, observed_meas=obs, dataset_name="GS1 Range"),
        scb.MeasurementSpec(model=dop, observed_meas=obs_dop, dataset_name="GS1 Doppler"),
    )

For a covariance analysis, give ``epochs=`` instead of ``observed_meas``.

Tracking schedules
------------------
:class:`~scarabaeus.MeasurementScheduler` builds the models and their simulated
observations from tracking requests, keeping only the epochs where the target is
above the elevation mask:

.. code-block:: python

    scheduler = scb.MeasurementScheduler(target=truth, elevation_mask=15.0, noisy=True)
    range_dss14 = scheduler.schedule(
        "DSS-14", "RangeIdeal", sigma=scb.ArrayWUnits(1e-3, km),
        start="2029 APR 01 00:00:00", end="2029 APR 03 00:00:00",
        cadence=scb.ArrayWUnits(10, minute), hours="02:00-06:00")
    meas = scb.MeasurementSpec.many(range_dss14, doppler_dss63, opnav)

Recurring windows are set with ``weekdays``, ``hours``, ``passes_per_day``,
``pass_cadence`` and ``pass_duration``; ``schedule_many`` takes several requests at
once. Real models cannot be simulated and raise ``RealModelNotSupportedError``.

Real tracking data
------------------
``MeasurementSpec.build`` reads real tracking files directly. Each stream is a
``(file, type, station)`` tuple:

.. code-block:: python

    cfg = scb.MeasurementConfig(
        media_folder="data/media", tropo_seasonal_file_path=tropo_file,
        sigma={"SeqRange": scb.ArrayWUnits(10.0, km / km),
               "Doppler": scb.ArrayWUnits(1e-2, 1 / sec)})
    meas = scb.MeasurementSpec.build([(trk_file, "SeqRange", dss25),
                                      (trk_file, "Doppler", dss25)], config=cfg)

- :class:`~scarabaeus.MeasurementRegistry` maps the type names (``"SeqRange"``,
  ``"Doppler"``, ``"IdealRange"``, ``"IdealRangeRate"``, ``"Centroid"``) to a model
  class, a data block and units. ``register`` adds your own.
- :class:`~scarabaeus.MeasurementConfig` holds what streams share: sigmas and biases
  per type or dataset, media folders, ``sc_transponder_delay_RU`` and ``extra`` model
  keywords. A :class:`~scarabaeus.MeasurementStream` overrides any of them.
- :class:`~scarabaeus.MediaCorrections` (tropospheric and ionospheric delays) is
  attached for each pass date. A correction file that is missing or of the wrong type
  raises instead of being skipped.
- The light time is barycentric by default; ``light_time_frame="geocentric"`` is
  available for spacecraft near the Earth. ``light_time_settings`` sets its tolerance
  and iterations.

Biases and parameters
---------------------
- ``meas_bias`` sets a fixed bias. One value applies to every pass; several give one per
  pass, grouped by ``set_pass_epochs``.
- To estimate a bias, add it to the state instead (``range_bias``, ``range_bias_RU``,
  ``rangerate_bias_Hz``). A state component overrides ``meas_bias``.
- Station, Earth orientation, tide, IAU pole and zenith-delay parameters are covered in
  :ref:`Stations and Body Orientation <orientation-station-estimation>`.

Partials are analytical for every model. ``partials_method="fd"`` switches a model
to finite differences, with steps from ``eps_dict``.

See also
--------
- :ref:`Estimation Framework <estimation-framework>`: how the filters use
  :math:`O - C` and :math:`\tilde{H}`.
- The *Measurement Simulation*, *Ideal MSR* and *Real MSR* tutorials in
  :ref:`Examples <examples>`.
