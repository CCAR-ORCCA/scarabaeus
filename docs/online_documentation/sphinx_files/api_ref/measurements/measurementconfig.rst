.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

``MeasurementConfig``
=====================

Describing a measurement stream by what it is, rather than by wiring it up. A
stream is identified by its tracking file, its measurement type and its
station; the type supplies the model class, the data block and the units, and
the config supplies what every stream in a scenario shares.

``MeasurementRegistry``
-----------------------

.. autoclass:: scarabaeus.MeasurementRegistry
    :members:
    :undoc-members:
    :show-inheritance:

``MeasurementType``
-------------------

.. autoclass:: scarabaeus.MeasurementType
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: label, meas_name, model_class, real, units

``MeasurementConfig``
---------------------

.. autoclass:: scarabaeus.MeasurementConfig
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: bias, extra, frame, media_folder, sc_transponder_delay_RU, sigma, tropo_seasonal_file_path

``MeasurementStream``
---------------------

.. autoclass:: scarabaeus.MeasurementStream
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: bias, file, meas_type, media, name, sigma, station
