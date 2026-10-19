# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    ArrayWFrame,
    EpochArray,
)

import numpy as np 

from typing import NamedTuple

class ObsData(NamedTuple):
    """One tracking-data file's observations, as read from disk.

    The return type of :meth:`~scarabaeus.Measurement.observed_measurements`,
    and so the handoff between reading a file of observations and running orbit
    determination on them. It gathers the four things every observation set
    needs -- when, what, which rows to reject, and whatever ancillary data the
    model needs to reconstruct the observable -- so that they travel together
    and stay on a common set of row indices.

    Downstream it is held on a :class:`~scarabaeus.MeasurementSpec` leaf as its
    ``observed_meas``, from where

    * :meth:`~scarabaeus.Measurement.generate_measurement_dataset` reads the
      epochs and observed values to form residuals against the forward model,
    * the editors on :class:`~scarabaeus.MeasurementEditing` and
      :meth:`~scarabaeus.MeasurementSpec.flag` write rejections into
      ``meas_outliers``,
    * :class:`~scarabaeus.FilterOD` gates on those flags and reseeds them into
      the dataset it regenerates on each pass.

    Parameters
    ----------
    meas_time_et : EpochArray
        Observation epochs, as ephemeris time. Converted from the station time
        tag in the file, so a measurement recomputed at these epochs
        corresponds to the same spacecraft state the observation does.
    meas_sec : numpy.ndarray
        The same epochs as seconds of day, carried through unconverted from the
        file, whose time tag is a year, day-of-year and seconds triple. Kept so
        that an observation can be matched back to its source record. No
        computation consumes it -- everything that needs the epochs takes
        ``meas_time_et`` -- but it is carried through by the scripts that
        rebuild a filtered observation set, which slice it by the same mask as
        the other fields to keep the rows aligned.
    meas_obs : ArrayWFrame
        The observed quantities, with units and, where the observable is a
        vector, a frame attached. Scalar observables such as range and Doppler
        carry units only.
    meas_outliers : numpy.ndarray
        Per-observation flags, ``1`` meaning reject. Seeded from the file's own
        outlier column, and set here as well for rows carrying a no-data
        sentinel. This is the authoritative flag array downstream: editing or
        filtering writes here, not onto a derived dataset, which is what lets a
        rejection survive the dataset being regenerated on the next filter
        iteration.
    aux_dict : dict, optional
        Ancillary per-observation data that the model needs to reproduce the
        observable but that is not itself an observation -- for the DSN
        radiometric models, the uplink ramp tables, transponder ratios and
        electronic delays. Produced by the model's ``_read_aux`` and installed
        on the model by the same read, so the two cannot disagree. ``None`` for
        models that need none, which is every optical model.

    Notes
    -----
    The fields are addressed both by name and by index across the codebase
    (``observed_meas[3]`` for the flags is common in the filter and spec code),
    which is why this is a named tuple rather than a dataclass. Field *order*
    is therefore part of the interface: appending a field is safe, reordering
    or removing one is not.

    This applies to ``meas_sec`` in particular, which no computation reads and
    which therefore looks removable. Dropping it would renumber the two fields
    that positional code uses most, turning every ``[2]`` into the flags and
    every ``[3]`` into the auxiliary dict. Those reads would not raise; they
    would silently return the wrong field, so residuals would be formed against
    outlier flags. Removing it means converting every positional access in the
    same change.

    The tuple is immutable but the arrays it holds are not, and
    ``meas_outliers`` is deliberately mutated in place. A flag set through
    one reference is visible through every other reference to the same
    ``ObsData``, which is the mechanism by which edits persist across filter
    iterations rather than being rebuilt each time.

    Every row in the file is present, including rows already flagged bad.
    Nothing is dropped at read time, so ``meas_obs``,
    ``meas_outliers`` and the per-observation entries of
    ``aux_dict`` stay index-aligned, and a row rejected at any later stage
    can still be traced back to its source record.

    Examples
    --------
    .. code-block:: python

        obs = model.observed_measurements(
            file_name="pass.json", meas_name="2W_sranging", units=km / km
        )
        epochs = obs.meas_time_et            # or obs[0]
        n_rejected = int(obs.meas_outliers.sum())

        spec = MeasurementSpec.from_models([model], [obs])
    """
    meas_time_et: EpochArray
    meas_sec: np.ndarray
    meas_obs : ArrayWFrame
    meas_outliers : np.ndarray
    aux_dict : dict | None = None