.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

``ObsData``
===========

What a measurement model hands back when it reads a tracking-data file: the
observation epochs, the observed quantities, the outlier flags and whatever
ancillary data the model needs to reproduce the observable, gathered into one
value on a common set of row indices.

.. autoclass:: scarabaeus.ObsData
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: meas_time_et, meas_sec, meas_obs, meas_outliers, aux_dict
