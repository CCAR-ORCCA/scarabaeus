# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    Units,
    Frame,
    ArrayWUnits,
    EpochArray,
    Body,
    Spacecraft,
    GroundStation,
    Camera,
    SpiceManager,
    Measurement,
    MeasurementSpec,
    RangeIdeal,
    RangeRateIdeal,
    DopplerIdeal,
    DiffOneWayRangeIdeal,
    CentroidingIdeal,
    AngularIdeal,
)
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Serialization import format_suffix
from dataclasses import dataclass, field
import datetime
import inspect
import numbers
import os
import re
import warnings

# ------------------#
#  Generate Units  #
# ------------------#
km, sec, rad, deg, Hz = Units.get_units(["km", "sec", "rad", "deg", "Hz"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class MeasurementScheduler:
    """ Builds measurement models and their synthetic observations from
    tracking requests.

    A request is an observer (a DSN station name such as ``"DSS-14"``, a
    :class:`~scarabaeus.GroundStation`, a :class:`~scarabaeus.Camera` or a
    :class:`~scarabaeus.Spacecraft`), a measurement type and a tracking
    window: ``start``, ``end`` and ``cadence``. Every request carries its
    own window, so one scheduler holds, side by side, range from Goldstone
    every ten minutes over three days, Doppler at one minute over a single
    pass and OpNav images every hour over a week, and hands them all to a
    filter as one :class:`~scarabaeus.MeasurementSpec`. Whatever is given to
    the constructor (target, window, elevation mask, ...) is only the default
    for requests that do not set their own.

    A window can recur, with a period of its own or on the UTC calendar:

    - ``pass_cadence`` / ``pass_duration`` (and an optional ``pass_offset``),
      all ``ArrayWUnits`` times, keep slots of ``pass_duration`` starting
      every ``pass_cadence`` from ``start + pass_offset``, at any period: an
      OpNav batch of 30 minutes every 6 hours is
      ``pass_cadence=ArrayWUnits(6, hr), pass_duration=ArrayWUnits(30, min)``,
      one image per orbit ``pass_cadence=orbit_period,
      pass_duration=cadence``;
    - ``weekdays`` keeps only some days of the week;
    - ``hours`` keeps one or several UTC time-of-day ranges within each
      day, so two passes a day are ``hours="02:00-04:00, 14:00-16:00"``;
    - ``passes_per_day`` keeps the first *n* tracking passes that start on
      each UTC day.

    They combine, so "one pass every Monday from today to next month" is
    ``schedule("DSS-14", "RangeIdeal", start=date.today(), end=...,
    cadence=ArrayWUnits(10, min), weekdays="Mon", passes_per_day=1)``.

    For every request the scheduler

    1. builds the epoch grid ``[start, end)`` at the request's cadence
       (only inside the slots and days / hours kept by the recurrence),
    2. keeps the epochs at which the observer can see the target (elevation
       above the mask for a ground station, target inside the field of view
       for a camera),
    3. builds the measurement model named by the measurement type,
    4. writes the synthetic observations to disk and reads them back,
    5. returns a :class:`~scarabaeus.MeasurementSpec` ready for a filter.

    The measurement type is the name of the model class that is built:
    ``"RangeIdeal"``, ``"RangeRateIdeal"``, ``"DopplerIdeal"``,
    ``"DiffOneWayRangeIdeal"`` (radiometric, from a ground station),
    ``"CentroidingIdeal"`` (from a camera) or ``"AngularIdeal"`` (from a
    spacecraft or station). Case and separators do not matter
    (``"doppler_ideal"`` works) and the class itself can be passed. The short
    spellings ``"range"``, ``"range_rate"``, ``"doppler"``, ``"dor"``,
    ``"opnav"``, ``"angular"``, ... (:attr:`MEASUREMENT_TYPE_ALIASES`) always
    mean the ideal model. The real TRK-2-34 models (``"DopplerReal"``,
    ``"SequentialRangingReal"``) raise
    :class:`MeasurementScheduler.RealModelNotSupportedError`: their observable
    is computed from per-epoch tracking data (ramp tables, count times,
    turnaround ratios, station delays) that cannot be synthesised yet.

    Parameters
    ----------
    target : Spacecraft or Body, optional
        Default body being tracked or imaged. Its SPICE name must resolve
        over the requested windows (an SPK for it must be loaded). Requests
        can name their own target (the imaged body for OpNav). Defaults to
        ``None``.
    start : str, float or EpochArray, optional
        Default first epoch of a request. Calendar strings are UTC and
        parsed with :meth:`~scarabaeus.SpiceManager.cal2et`; floats are TDB
        seconds past J2000; a :class:`datetime.date` / :class:`datetime.datetime`
        is UTC; an :class:`~scarabaeus.EpochArray` is taken at its first
        epoch. Defaults to ``None``.
    end : str, float or EpochArray, optional
        Default end of a request, same conventions as ``start`` (an
        ``EpochArray`` is taken at its last epoch). Grids are half open:
        ``end`` itself is not included. Defaults to ``None``.
    cadence : ArrayWUnits, optional
        Default spacing between consecutive measurements, a single
        ``ArrayWUnits`` time such as ``ArrayWUnits(10, min)``. Defaults to
        ``None``.
    elevation_mask : ArrayWUnits or float, optional
        Default minimum elevation angle for ground-station observers, an
        ``ArrayWUnits`` angle such as ``ArrayWUnits(10, deg)``. A bare number
        is degrees, and one that could be radians is refused (see
        :meth:`~scarabaeus.Measurement.resolve_elevation_mask`).
        Defaults to ``10.0`` deg.
    frame : Frame, optional
        Frame in which the measurements are computed. Defaults to J2000.
    noisy : bool, optional
        Whether the generated observations carry Gaussian noise drawn from
        each model's ``sigma``. Defaults to ``False``.
    output_dir : str, optional
        Folder that receives the measurement files. ``None`` lets each model
        pick ``data/measurements/radiometric`` or ``data/measurements/optical``
        from its file name. Defaults to ``None``.
    fmt : {"json", "pickle"}, optional
        On-disk format of the measurement files. Defaults to ``"json"``.
    pass_gap : ArrayWUnits, optional
        Default gap between consecutive visible epochs that starts a new
        tracking pass, an ``ArrayWUnits`` time. ``None`` uses one and a half
        cadences of each request. Defaults to ``None``.

    Raises
    ------
    MeasurementScheduler.InvalidScheduleError
        If a default ``end`` is not after the default ``start``, the
        default cadence is not positive, or the default elevation mask is
        not an angle between 0 and 90 deg.

    See Also
    --------
    scarabaeus.MeasurementSpec : What :meth:`schedule` returns.
    scarabaeus.GroundStation.station_visibility : Elevation-based visibility.
    scarabaeus.Camera.target_in_fov : Field-of-view visibility.

    Examples
    --------
    .. code-block:: python

        import scarabaeus as scb

        km, sec, minute, hr, Hz = scb.Units.get_units(["km", "sec", "min", "hr", "Hz"])

        # Defaults shared by the requests below; every request may override them
        scheduler = scb.MeasurementScheduler(
            target=Orbiter, elevation_mask=15.0, noisy=True
        )

        # Range from Goldstone every 10 min over two days, Doppler from Madrid
        # every minute over one pass, OpNav of the Moon every hour over a week:
        # three windows, three cadences, one scheduler
        range_dss14 = scheduler.schedule(
            "DSS-14", "RangeIdeal", sigma=scb.ArrayWUnits(1e-3, km),
            start="2029 APR 01 00:00:00", end="2029 APR 03 00:00:00",
            cadence=scb.ArrayWUnits(10, minute),
        )
        doppler_dss63 = scheduler.schedule(
            "DSS-63", "DopplerIdeal", sigma=scb.ArrayWUnits(5e-3, Hz),
            start="2029 APR 01 20:00:00", end="2029 APR 02 04:00:00",
            cadence=scb.ArrayWUnits(1, minute),
            antenna_name="HGA", doppler_transmit_frequency=scb.ArrayWUnits(7.2e9, Hz),
        )
        opnav = scheduler.schedule(
            camera, "CentroidingIdeal", target=scb.Body("MOON"),
            sigma=scb.ArrayWUnits([0.5, 0.5], None),
            start="2029 APR 01 00:00:00", end="2029 APR 08 00:00:00",
            cadence=scb.ArrayWUnits(1, hr),
        )

        # Recurring windows: one Goldstone pass every Monday for a month,
        # Canberra twice a day, five OpNav images a minute apart every 6 h
        weekly = scheduler.schedule(
            "DSS-14", "RangeIdeal", sigma=scb.ArrayWUnits(1e-3, km),
            start="2029 APR 01", end="2029 MAY 01", cadence=scb.ArrayWUnits(10, minute),
            weekdays="Mon", passes_per_day=1,
        )
        twice_daily = scheduler.schedule(
            "DSS-43", "RangeRateIdeal", sigma=scb.ArrayWUnits(1e-7, km / sec),
            start="2029 APR 01", end="2029 MAY 01", cadence=scb.ArrayWUnits(1, minute),
            hours="02:00-06:00, 14:00-18:00",
        )
        batches = scheduler.schedule(
            camera, "CentroidingIdeal", target=scb.Body("MOON"),
            sigma=scb.ArrayWUnits([0.5, 0.5], None),
            start="2029 APR 01", end="2029 APR 08", cadence=scb.ArrayWUnits(1, minute),
            pass_cadence=scb.ArrayWUnits(6, hr), pass_duration=scb.ArrayWUnits(5, minute),
        )

        # Or the whole plan at once
        measurements = scheduler.schedule_many([
            ("DSS-14", "RangeIdeal", {"sigma": ..., "cadence": scb.ArrayWUnits(10, minute)}),
            ("DSS-63", "DopplerIdeal",
             {"sigma": ..., "cadence": scb.ArrayWUnits(1, minute), "antenna_name": "HGA", ...}),
            (camera, "CentroidingIdeal",
             {"target": scb.Body("MOON"), "sigma": ..., "cadence": scb.ArrayWUnits(1, hr)}),
        ], start="2029 APR 01 00:00:00", end="2029 APR 03 00:00:00")

        # Everything a filter needs
        measurements = scb.MeasurementSpec.many(range_dss14, doppler_dss63, opnav)
        range_dss14.model                                  # the RangeIdeal object
        scheduler.scheduled["DSS-14 RangeIdeal"].epochs    # the visible epochs
        scheduler.scheduled["DSS-14 RangeIdeal"].window    # its start / end / cadence / grid
    """

    WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    """Weekday spellings accepted by ``weekdays`` (case-insensitive prefixes work too)."""

    EDGE_TOLERANCE = ArrayWUnits(0.01, sec)
    """Default slack when deciding whether a scheduled grid point reaches the
    end of the schedule, a UTC midnight or an hour-of-day bound; the starting
    :attr:`edge_tolerance` of a scheduler.

    A grid at a fixed TDB cadence drifts against the UTC calendar by the
    TDB-UTC periodic terms (< 2 ms), on top of the ET float rounding.
    """

    # ---------------------#
    # region   Exceptions #
    # ---------------------#
    class UnknownMeasurementTypeError(ValueError):
        """Raised when the measurement type string is not one the scheduler knows."""

    class RealModelNotSupportedError(NotImplementedError):
        """Raised when a real (TRK-2-34) model is requested: its tracking data cannot be synthesised yet."""

    class ObserverTypeError(TypeError):
        """Raised when the observer cannot be used for the requested measurement type."""

    class UnsupportedModelArgumentError(TypeError):
        """Raised when a keyword argument is not accepted by the measurement model class."""

    class NoVisibleEpochsError(RuntimeError):
        """Raised when the observer never sees the target over the request's window."""

    class InvalidScheduleError(ValueError):
        """Raised when the start, end or cadence of a request do not define a usable grid."""

    class MissingTargetError(ValueError):
        """Raised when neither the request nor the scheduler names a target."""

    class DuplicateDatasetError(ValueError):
        """Raised when an explicit ``dataset_name`` is already scheduled."""

    # The measurement models' own error, so one except clause catches both.
    MissingSigmaError = Measurement.MissingSigmaError

    class MeasurementFileNotWrittenError(RuntimeError):
        """Raised when the measurement model produced no file to read back."""

    # endregion

    # ------------------------#
    # region   Records       #
    # ------------------------#
    @dataclass(frozen=True)
    class Window:
        """Time span and cadence of one request, with its epoch grid.

        Attributes
        ----------
        start_et : float
            First epoch, TDB seconds past J2000.
        end_et : float
            End of the span (excluded from the grid), TDB seconds past J2000.
        cadence_sec : float
            Spacing between consecutive epochs [s].
        pass_gap_sec : float
            Gap between visible epochs that starts a new pass [s].
        weekdays : tuple of int or None
            Days of the week kept in the grid, ``0`` = Monday ... ``6`` =
            Sunday (UTC calendar). ``None`` keeps every day.
        hours : tuple of (float, float) or None
            UTC time-of-day ranges ``(from, to)`` in hours kept within each
            day (their union); ``to < from`` wraps around midnight. ``None``
            keeps the whole day.
        passes_per_day : int or None
            Only the first ``passes_per_day`` tracking passes starting on
            each UTC day are scheduled. ``None`` keeps them all.
        pass_cadence_sec : float or None
            Time [s] between the starts of two tracking slots. ``None``
            means one continuous slot ``[start, end)``.
        pass_duration_sec : float or None
            Length [s] of each slot, at most ``pass_cadence_sec``.
        pass_offset_sec : float
            Delay [s] of the first slot after ``start_et``.
        edge_tolerance_sec : float or None
            Slack [s] at the end of the span, at UTC midnights and at the
            ``hours`` bounds. ``None`` takes
            :attr:`MeasurementScheduler.EDGE_TOLERANCE`.
        epochs : EpochArray
            The half-open grid ``[start, end)`` at ``cadence_sec``, TDB,
            restricted to the slots, ``weekdays`` and ``hours``.

        Raises
        ------
        MeasurementScheduler.InvalidScheduleError
            If the recurrence leaves no epoch in ``[start, end)``.
        """

        start_et: float
        end_et: float
        cadence_sec: float
        pass_gap_sec: float
        weekdays: tuple = None
        hours: tuple = None
        passes_per_day: int = None
        pass_cadence_sec: float = None
        pass_duration_sec: float = None
        pass_offset_sec: float = 0.0
        edge_tolerance_sec: float = None
        epochs: EpochArray = field(init=False, repr=False, compare=False)
        _day_starts: np.ndarray = field(init=False, repr=False, compare=False)
        _day_weekdays: np.ndarray = field(init=False, repr=False, compare=False)

        def __post_init__(self):
            if self.edge_tolerance_sec is None:
                object.__setattr__(
                    self, "edge_tolerance_sec",
                    MeasurementScheduler._to_seconds(
                        MeasurementScheduler.EDGE_TOLERANCE, "edge_tolerance", allow_zero=True
                    ),
                )
            et_values = MeasurementScheduler.build_epochs(
                self.start_et, self.end_et, self.cadence_sec,
                pass_cadence=self.pass_cadence_sec,
                pass_duration=self.pass_duration_sec,
                pass_offset=self.pass_offset_sec,
                edge_tolerance=self.edge_tolerance_sec,
            ).et
            et_values = np.atleast_1d(et_values)
            day_starts, day_weekdays = MeasurementScheduler._day_table(self.start_et, self.end_et)
            object.__setattr__(self, "_day_starts", day_starts)
            object.__setattr__(self, "_day_weekdays", day_weekdays)
            if self.weekdays is not None or self.hours is not None:
                keep = np.ones(et_values.size, dtype=bool)
                day = self.day_index(et_values)
                if self.weekdays is not None:
                    keep &= np.isin(day_weekdays[day], self.weekdays)
                if self.hours is not None:
                    # same slack as day_index so nominal HH:MM epochs fall on the right side
                    tod = np.clip(
                        (et_values + self.edge_tolerance_sec - day_starts[day])
                        / constants.hour_to_sec, 0.0, None,
                    )
                    in_hours = np.zeros(et_values.size, dtype=bool)
                    for lo, hi in self.hours:
                        in_hours |= (tod >= lo) & (tod < hi) if lo <= hi else (tod >= lo) | (tod < hi)
                    keep &= in_hours
                et_values = et_values[keep]
            if et_values.size == 0:
                raise MeasurementScheduler.InvalidScheduleError(
                    f"{MeasurementScheduler._cal(self.start_et)} -> "
                    f"{MeasurementScheduler._cal(self.end_et)} every {self.cadence_sec:g} s, "
                    f"{self.recurrence}: no epoch is left in the window."
                )
            object.__setattr__(self, "epochs", EpochArray(et_values, sys="TDB"))

        def day_index(self, et_values) -> np.ndarray:
            """Return the index of the UTC day of each epoch.

            Parameters
            ----------
            et_values : float or array_like of float
                Epochs, in TDB seconds past J2000.

            Returns
            -------
            numpy.ndarray of int
                Index of the UTC day of each epoch, with ``0`` the day of
                :attr:`start`. An epoch within ``edge_tolerance_sec`` of a
                midnight belongs to the new day.
            """
            # An epoch a hair short of midnight (rounding, TDB-UTC drift) belongs to the new day
            et_values = np.atleast_1d(np.asarray(et_values, dtype=float))
            edges = self._day_starts - self.edge_tolerance_sec
            return np.clip(np.searchsorted(edges, et_values, side="right") - 1, 0, None)

        def select(self, mask: np.ndarray) -> tuple:
            """Turn a visibility mask over :attr:`epochs` into the scheduled passes.

            Visible epochs are split into passes at gaps larger than
            ``pass_gap_sec`` and, when ``passes_per_day`` is set, only the
            first passes starting on each UTC day are kept.

            Parameters
            ----------
            mask : numpy.ndarray of bool
                Visibility of the target at each epoch of :attr:`epochs`.

            Returns
            -------
            tuple
                ``(mask, passes)``: the mask of the epochs actually scheduled
                and the list of passes (``EpochArray`` each), in time order.
            """
            mask = np.atleast_1d(np.asarray(mask, dtype=bool))
            passes = MeasurementScheduler.split_passes(self.epochs[mask], self.pass_gap_sec)
            if self.passes_per_day is None or not passes:
                return mask, passes
            days = self.day_index([float(np.atleast_1d(p.et)[0]) for p in passes])
            kept, seen = [], {}
            for single_pass, day in zip(passes, days.tolist()):
                seen[day] = seen.get(day, 0) + 1
                if seen[day] <= self.passes_per_day:
                    kept.append(single_pass)
            kept_et = np.concatenate([np.atleast_1d(p.et) for p in kept]) if kept else np.array([])
            return np.isin(np.atleast_1d(self.epochs.et), kept_et), kept

        @property
        def start(self) -> EpochArray:
            """First epoch of the window."""
            return EpochArray(self.start_et, sys="TDB")

        @property
        def end(self) -> EpochArray:
            """End of the window (excluded from the grid)."""
            return EpochArray(self.end_et, sys="TDB")

        @property
        def cadence(self) -> ArrayWUnits:
            """Spacing between consecutive epochs."""
            return ArrayWUnits(self.cadence_sec, sec)

        @property
        def pass_gap(self) -> ArrayWUnits:
            """Gap between visible epochs that starts a new pass."""
            return ArrayWUnits(self.pass_gap_sec, sec)

        @property
        def size(self) -> int:
            """Number of epochs in the grid."""
            return int(self.epochs.size)

        @property
        def recurrence(self) -> str:
            """Human-readable recurrence, empty if none.

            E.g. ``"30min every 6h, on Mon, 02:00-04:00 & 14:00-16:00 h UTC, 1 pass(es)/day"``.
            """
            span = MeasurementScheduler._format_span
            parts = []
            if self.pass_cadence_sec is not None:
                slots = f"{span(self.pass_duration_sec)} every {span(self.pass_cadence_sec)}"
                if self.pass_offset_sec:
                    slots += f" from start + {span(self.pass_offset_sec)}"
                parts.append(slots)
            if self.weekdays is not None:
                parts.append("on " + "/".join(MeasurementScheduler.WEEKDAY_NAMES[d] for d in self.weekdays))
            if self.hours is not None:
                parts.append(
                    " & ".join(
                        f"{MeasurementScheduler._hhmm(lo)}-{MeasurementScheduler._hhmm(hi)}"
                        for lo, hi in self.hours
                    ) + " h UTC"
                )
            if self.passes_per_day is not None:
                parts.append(f"{self.passes_per_day} pass(es)/day")
            return ", ".join(parts)

        def __str__(self) -> str:
            text = (
                f"{MeasurementScheduler._cal(self.start_et)} -> "
                f"{MeasurementScheduler._cal(self.end_et)} every {self.cadence_sec:g} s"
            )
            if self.recurrence:
                text += ", " + self.recurrence
            return f"{text} ({self.size} epochs)"

    @dataclass(frozen=True)
    class ScheduledMeasurement:
        """One completed request: the model, its observations and how they were made.

        Attributes
        ----------
        spec : MeasurementSpec
            Filter-ready specification (``model`` + ``observed_meas``).
        model : Measurement
            The measurement model built for this request.
        observed_meas : tuple
            Output of :meth:`~scarabaeus.Measurement.observed_measurements`.
        measurement_type : str
            Class name of the model built (``"RangeIdeal"``,
            ``"CentroidingIdeal"``, ...).
        observer : GroundStation, Camera or Spacecraft
            The instrument or body the measurements are taken from.
        target : Spacecraft or Body
            The tracked/imaged body.
        window : MeasurementScheduler.Window
            Start, end, cadence, recurrence and full epoch grid of this request.
        epochs : EpochArray
            Epochs kept after the visibility check (those in the file).
        visibility_mask : numpy.ndarray
            Boolean mask over ``window.epochs``: True where the target is visible and the epoch
            lies inside a kept pass (``passes_per_day`` may drop visible passes).
        passes : list of EpochArray
            ``epochs`` grouped into tracking passes.
        file_path : str
            Measurement file on disk.
        """

        spec: MeasurementSpec
        model: Measurement
        observed_meas: tuple
        measurement_type: str
        observer: object
        target: object
        window: "MeasurementScheduler.Window"
        epochs: EpochArray
        visibility_mask: np.ndarray
        passes: list
        file_path: str

        @property
        def dataset_name(self) -> str:
            """Dataset name of the request."""
            return self.spec.dataset_name

        @property
        def grid(self) -> EpochArray:
            """Full epoch grid of the request (before the visibility check)."""
            return self.window.epochs

        @property
        def start(self) -> EpochArray:
            """First epoch of the window."""
            return self.window.start

        @property
        def end(self) -> EpochArray:
            """Last epoch of the window."""
            return self.window.end

        @property
        def cadence(self) -> ArrayWUnits:
            """Time step of the epoch grid."""
            return self.window.cadence

    # endregion

    # -------------------------#
    # region   Type registry  #
    # -------------------------#
    # model class name (the canonical measurement type) -> model class
    MODEL_CLASSES = {
        model_cls.__name__: model_cls
        for model_cls in (
            RangeIdeal,
            RangeRateIdeal,
            DopplerIdeal,
            DiffOneWayRangeIdeal,
            CentroidingIdeal,
            AngularIdeal,
        )
    }

    # short spelling (lower case, separators dropped) -> model class name; always an ideal model
    MEASUREMENT_TYPE_ALIASES = {
        "range": "RangeIdeal",
        "rng": "RangeIdeal",
        "rangerate": "RangeRateIdeal",
        "rr": "RangeRateIdeal",
        "doppler": "DopplerIdeal",
        "dor": "DiffOneWayRangeIdeal",
        "diffonewayrange": "DiffOneWayRangeIdeal",
        "deltador": "DiffOneWayRangeIdeal",
        "centroiding": "CentroidingIdeal",
        "centroid": "CentroidingIdeal",
        "opnav": "CentroidingIdeal",
        "optical": "CentroidingIdeal",
        "angular": "AngularIdeal",
        "angles": "AngularIdeal",
        "radec": "AngularIdeal",
    }

    # Real TRK-2-34 models -> the ideal model to schedule instead. They are recognised
    # only to fail clearly: their observable needs per-epoch aux data (ramp table spanning
    # at most two days, Tc, M2, delays) that nothing synthesises yet (MR 261).
    REAL_MODEL_SUBSTITUTES = {
        "DopplerReal": "DopplerIdeal",
        "SequentialRangingReal": "RangeIdeal",
    }
    # short spelling -> real model class name
    REAL_MODEL_ALIASES = {
        "seqrange": "SequentialRangingReal",
        "seqranging": "SequentialRangingReal",
        "sequentialranging": "SequentialRangingReal",
    }

    # model class name -> units the file is read back with
    OBSERVATION_UNITS = {
        "RangeIdeal": km,
        "RangeRateIdeal": km / sec,
        "DopplerIdeal": Hz,
        "DiffOneWayRangeIdeal": sec,
        "CentroidingIdeal": None,
        "AngularIdeal": rad,
    }


    # types whose observer is a ground station
    RADIOMETRIC_TYPES = frozenset({"RangeIdeal", "RangeRateIdeal", "DopplerIdeal", "DiffOneWayRangeIdeal"})
    # types whose observer is a camera or a body carrying one
    OPTICAL_TYPES = frozenset({"CentroidingIdeal", "AngularIdeal"})

    # endregion

    # -------------------#
    # region Constructor #
    # -------------------#
    def __init__(
        self,
        target: Spacecraft | Body = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        elevation_mask: ArrayWUnits | float = 10.0,
        frame: Frame = J2000,
        noisy: bool = False,
        output_dir: str = None,
        fmt: str = "json",
        pass_gap: ArrayWUnits = None,
    ):
        self._target = target
        self._start_et = None if start is None else MeasurementScheduler._to_et(start, "start")
        self._end_et = None if end is None else MeasurementScheduler._to_et(end, "end")
        self._cadence_sec = None if cadence is None else MeasurementScheduler._to_seconds(cadence)
        self._elevation_mask = MeasurementScheduler._to_degrees(elevation_mask)
        self._frame = frame
        self._noisy = bool(noisy)
        self._output_dir = output_dir
        self._fmt = fmt
        self._pass_gap_sec = (
            None if pass_gap is None else MeasurementScheduler._to_seconds(pass_gap, "pass_gap")
        )
        self.edge_tolerance = type(self).EDGE_TOLERANCE
        if (
            self._start_et is not None
            and self._end_et is not None
            and self._end_et <= self._start_et
        ):
            raise MeasurementScheduler.InvalidScheduleError(
                f"default 'end' ({MeasurementScheduler._cal(self._end_et)}) must be after "
                f"default 'start' ({MeasurementScheduler._cal(self._start_et)})."
            )
        # A complete default window gives a default grid; partial defaults
        # are completed by each request.
        self._window = (
            self.resolve_window()
            if None not in (self._start_et, self._end_et, self._cadence_sec)
            else None
        )
        self._scheduled: dict[str, MeasurementScheduler.ScheduledMeasurement] = {}

    # endregion

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def target(self) -> Spacecraft | Body | None:
        """Default body being tracked or imaged (``None`` = set per request)."""
        return self._target

    @property
    def window(self) -> "MeasurementScheduler.Window | None":
        """Default window, or ``None`` when start, end or cadence is left to the requests."""
        return self._window

    @property
    def epochs(self) -> EpochArray | None:
        """Default epoch grid (before any visibility check), or ``None``."""
        return None if self._window is None else self._window.epochs

    @property
    def start(self) -> EpochArray | None:
        """Default first epoch of a request, or ``None``."""
        return None if self._start_et is None else EpochArray(self._start_et, sys="TDB")

    @property
    def end(self) -> EpochArray | None:
        """Default end of a request (excluded from the grid), or ``None``."""
        return None if self._end_et is None else EpochArray(self._end_et, sys="TDB")

    @property
    def cadence(self) -> ArrayWUnits | None:
        """Default spacing between consecutive measurements, or ``None``."""
        return None if self._cadence_sec is None else ArrayWUnits(self._cadence_sec, sec)

    @property
    def elevation_mask(self) -> ArrayWUnits:
        """Default minimum elevation angle for ground-station observers, in degrees."""
        return ArrayWUnits(self._elevation_mask, deg)

    @property
    def frame(self) -> Frame:
        """Frame in which the measurements are computed."""
        return self._frame

    @property
    def noisy(self) -> bool:
        """Whether generated observations carry noise."""
        return self._noisy

    @property
    def output_dir(self) -> str | None:
        """Folder receiving the measurement files (``None`` = model default)."""
        return self._output_dir

    @property
    def pass_gap(self) -> ArrayWUnits | None:
        """Default gap between visible epochs that starts a new pass (``None`` = 1.5 cadences)."""
        return None if self._pass_gap_sec is None else ArrayWUnits(self._pass_gap_sec, sec)

    @property
    def edge_tolerance(self) -> ArrayWUnits:
        """Slack when deciding whether a grid point reaches the end of a
        request, a UTC midnight or an hour-of-day bound. Starts as
        :attr:`EDGE_TOLERANCE`.

        Set a single, non-negative ``ArrayWUnits`` time. The default window,
        if any, is rebuilt with it; requests already scheduled keep the
        tolerance they were built with.

        Raises
        ------
        InvalidScheduleError
            If the value is not a single non-negative time span (see
            :meth:`_to_seconds`).
        """
        return ArrayWUnits(self._edge_tolerance_sec, sec)

    @edge_tolerance.setter
    def edge_tolerance(self, value: ArrayWUnits) -> None:
        self._edge_tolerance_sec = MeasurementScheduler._to_seconds(
            value, "edge_tolerance", allow_zero=True
        )
        if getattr(self, "_window", None) is not None:
            self._window = self.resolve_window()

    @property
    def scheduled(self) -> dict:
        """Completed requests keyed by dataset name."""
        return self._scheduled

    @property
    def last(self) -> "MeasurementScheduler.ScheduledMeasurement | None":
        """Most recently completed request, or ``None``."""
        if not self._scheduled:
            return None
        return next(reversed(self._scheduled.values()))

    @property
    def span(self) -> tuple | None:
        """Earliest start and latest end over the completed requests, or ``None``."""
        if not self._scheduled:
            return None
        windows = [rec.window for rec in self._scheduled.values()]
        return (
            EpochArray(min(w.start_et for w in windows), sys="TDB"),
            EpochArray(max(w.end_et for w in windows), sys="TDB"),
        )

    # endregion

    # ----------------------#
    # region   Resolution  #
    # ----------------------#
    @staticmethod
    def _to_et(epoch: str | float | EpochArray, label: str) -> float:
        """Convert a window bound to TDB seconds past J2000.

        Parameters
        ----------
        epoch : str or float or datetime.date or datetime.datetime or ArrayWUnits or EpochArray
            The bound: a calendar string, a date or datetime (UTC), TDB seconds
            past J2000 as a number or an ArrayWUnits time, or an EpochArray.
        label : str
            ``"start"`` or ``"end"``. It selects the first or last epoch of an
            EpochArray and names the bound in the error message.

        Returns
        -------
        float
            The bound, in TDB seconds past J2000.

        Raises
        ------
        InvalidScheduleError
            If ``epoch`` is none of the accepted types.
        """
        if isinstance(epoch, EpochArray):
            et_values = np.atleast_1d(epoch.et)
            return float(et_values[0] if label == "start" else et_values[-1])
        if isinstance(epoch, str):
            return float(SpiceManager.cal2et(epoch))
        if isinstance(epoch, datetime.datetime):
            return float(SpiceManager.cal2et(epoch.strftime("%Y-%m-%dT%H:%M:%S")))
        if isinstance(epoch, datetime.date):
            return float(SpiceManager.cal2et(epoch.strftime("%Y-%m-%dT00:00:00")))
        if isinstance(epoch, ArrayWUnits):
            return float(np.atleast_1d(epoch.convert_to(sec).values)[0])
        try:
            return float(epoch)
        except (TypeError, ValueError):
            raise MeasurementScheduler.InvalidScheduleError(
                f"'{label}' must be a calendar string, a date/datetime, TDB seconds "
                f"past J2000 or an EpochArray, got {type(epoch).__name__}."
            )

    @staticmethod
    def _to_seconds(
        span: ArrayWUnits | float,
        label: str = "cadence",
        allow_zero: bool = False,
        bare_seconds: bool = False,
    ) -> float:
        """Seconds in a single ``ArrayWUnits`` time span, checked positive (or zero if allowed).

        Seconds as bare numbers (unless ``bare_seconds``), duration strings,
        timedeltas, non-time units and arrays of several spans are refused.

        Parameters
        ----------
        span : ArrayWUnits or float
            One time span in any time unit, e.g. ``ArrayWUnits(10, min)``, or
            a bare number of seconds when ``bare_seconds``.
        label : str, optional
            Argument name quoted in the error message. Defaults to
            ``"cadence"``.
        allow_zero : bool, optional
            Accept a zero span. Defaults to ``False``.
        bare_seconds : bool, optional
            Take a bare real number as seconds, as the grid helpers
            :meth:`build_epochs` and :meth:`split_passes` do so that calls
            written before they took units keep working. Defaults to
            ``False``: the scheduler's own arguments must carry their units.

        Returns
        -------
        float
            The span in seconds.

        Raises
        ------
        InvalidScheduleError
            If ``span`` is not an ``ArrayWUnits`` (nor a number with
            ``bare_seconds``), is not in time units, holds more than one
            value, or is not finite and positive (non-negative when
            ``allow_zero``).
        """
        if bare_seconds and isinstance(span, numbers.Real) and not isinstance(span, bool):
            values = np.atleast_1d(float(span))
        elif isinstance(span, ArrayWUnits):
            try:
                values = np.atleast_1d(span.convert_to(sec).values)
            except (TypeError, ValueError):
                raise MeasurementScheduler.InvalidScheduleError(
                    f"'{label}' must be in time units, got {span.units}."
                )
        else:
            raise MeasurementScheduler.InvalidScheduleError(
                f"'{label}' must be an ArrayWUnits time such as ArrayWUnits(10, min)"
                + (" or a number of seconds" if bare_seconds else "")
                + f", got {type(span).__name__} {span!r}."
            )
        if values.size != 1:
            raise MeasurementScheduler.InvalidScheduleError(
                f"'{label}' must be a single time span, got {values.size} values."
            )
        value = float(values[0])
        if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
            raise MeasurementScheduler.InvalidScheduleError(
                f"'{label}' must be a {'non-negative' if allow_zero else 'positive'} "
                f"time span, got {value:g} s."
            )
        return value

    @staticmethod
    def _to_degrees(elevation_mask: ArrayWUnits | float) -> float:
        """Elevation mask in degrees, through :meth:`Measurement.resolve_elevation_mask`.

        Parameters
        ----------
        elevation_mask : ArrayWUnits or float
            An ``ArrayWUnits`` angle in any angle unit, or a bare number of
            degrees.

        Returns
        -------
        float
            The mask in degrees.

        Raises
        ------
        InvalidScheduleError
            If ``elevation_mask`` is not an angle, is outside 0 to 90 deg, or is
            a bare number that could be radians, from
            :class:`Measurement.ElevationMaskError`.
        """
        try:
            return Measurement.resolve_elevation_mask(elevation_mask)
        except Measurement.ElevationMaskError as err:
            raise MeasurementScheduler.InvalidScheduleError(f"'elevation_mask': {err}") from err

    @staticmethod
    def _format_span(seconds: float) -> str:
        """Return the short text of a time span in its largest whole unit.

        Parameters
        ----------
        seconds : float
            The span, in seconds.

        Returns
        -------
        str
            The span in days, hours or minutes when it is a whole number of them,
            otherwise in seconds; for example ``21600`` gives ``"6h"``.
        """
        for unit, size in (
            ("d", constants.day_to_sec),
            ("h", constants.hour_to_sec),
            ("min", constants.min_to_sec),
        ):
            if seconds >= size and float(seconds / size).is_integer():
                return f"{seconds / size:g}{unit}"
        return f"{seconds:g}s"

    @staticmethod
    def _cal(et: float) -> str:
        """Return the UTC calendar string of an epoch, to the second.

        Parameters
        ----------
        et : float
            Epoch, in TDB seconds past J2000.

        Returns
        -------
        str
            The epoch in SPICE ``"C"`` format.
        """
        return str(SpiceManager.et2utc(float(et), "C", 0))

    @staticmethod
    def _hhmm(hours: float) -> str:
        """Return a time of day as ``"HH:MM"``.

        Parameters
        ----------
        hours : float
            Time of day, in decimal hours.

        Returns
        -------
        str
            The time rounded to the minute; for example ``12.5`` gives
            ``"12:30"``.
        """
        minutes = int(round(hours * 60.0))
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @staticmethod
    def _to_weekdays(weekdays) -> tuple | None:
        """Normalise a weekday selection.

        Parameters
        ----------
        weekdays : str or int or sequence or None
            ``"weekdays"`` or ``"workdays"``, ``"weekend"`` or ``"weekends"``, a
            string of day names separated by commas, spaces or slashes, an integer
            from ``0`` (Monday) to ``6`` (Sunday), or a sequence of names and
            integers. Names are matched on their first three letters.

        Returns
        -------
        tuple of int or None
            The sorted days, ``0`` (Monday) to ``6`` (Sunday), or ``None`` when
            ``weekdays`` is ``None``.

        Raises
        ------
        InvalidScheduleError
            If an entry is not a weekday, or the selection is empty.
        """
        if weekdays is None:
            return None
        if isinstance(weekdays, str):
            lowered = weekdays.strip().lower()
            if lowered in ("weekdays", "workdays"):
                return (0, 1, 2, 3, 4)
            if lowered in ("weekend", "weekends"):
                return (5, 6)
            weekdays = re.split(r"[,\s/]+", lowered)
        elif isinstance(weekdays, (int, np.integer)):
            weekdays = [weekdays]
        days = set()
        for day in weekdays:
            if isinstance(day, (int, np.integer)) and 0 <= int(day) <= 6:
                days.add(int(day))
                continue
            text = str(day).strip().lower()[:3]
            names = [n.lower() for n in MeasurementScheduler.WEEKDAY_NAMES]
            if text not in names:
                raise MeasurementScheduler.InvalidScheduleError(
                    f"'weekdays' entry {day!r} is not a weekday name "
                    f"({', '.join(MeasurementScheduler.WEEKDAY_NAMES)}) or 0-6."
                )
            days.add(names.index(text))
        if not days:
            raise MeasurementScheduler.InvalidScheduleError("'weekdays' selects no day.")
        return tuple(sorted(days))

    @staticmethod
    def _hour_of_day(value) -> float:
        """Return a time of day in decimal hours.

        Parameters
        ----------
        value : float or str
            Time of day, in hours or as ``"HH:MM"``.

        Returns
        -------
        float
            The time of day, in decimal hours.
        """
        if isinstance(value, str):
            hh, colon, mm = value.strip().partition(":")
            return float(hh) + (float(mm) / 60.0 if colon else 0.0)
        return float(value)

    @staticmethod
    def _hour_range(item) -> tuple:
        """Return one time-of-day range as a pair of decimal hours.

        Parameters
        ----------
        item : str or sequence
            The range, as a string such as ``"08:00-12:00"`` or ``"8 to 12"``, or
            as a ``(from, to)`` pair.

        Returns
        -------
        tuple of float
            ``(from, to)``, in decimal hours.

        Raises
        ------
        ValueError
            If ``item`` does not hold exactly two times.
        """
        if isinstance(item, str):
            parts = re.split(r"\s*(?:-|to)\s*", item.strip().lower().replace("utc", "").strip())
        else:
            parts = list(item)
        if len(parts) != 2:
            raise ValueError(f"{item!r} is not a (from, to) range")
        return tuple(MeasurementScheduler._hour_of_day(part) for part in parts)

    @staticmethod
    def _to_hours(hours) -> tuple | None:
        """Normalise UTC time-of-day ranges.

        Parameters
        ----------
        hours : str or sequence or None
            One range, such as ``"08:00-12:00"`` or ``(8, 12)``, or several, such
            as ``"02:00-04:00, 14:00-16:00"``, ``["02:00-04:00", "14:00-16:00"]``
            or ``[(2, 4), (14, 16)]``.

        Returns
        -------
        tuple of tuple of float or None
            The ``(from, to)`` pairs, in decimal hours, or ``None`` when ``hours``
            is ``None``.

        Raises
        ------
        InvalidScheduleError
            If a range cannot be read, lies outside 0 to 24 h, has equal ends, or
            no range is given.
        """
        if hours is None:
            return None
        if isinstance(hours, str):
            items = [part for part in re.split(r"[,;]", hours) if part.strip()]
        else:
            try:
                items = list(hours)
            except TypeError:
                items = [hours]
            # a bare (from, to) pair of times of day is a single range
            if len(items) == 2:
                try:
                    items = [tuple(MeasurementScheduler._hour_of_day(h) for h in items)]
                except (TypeError, ValueError):
                    pass
        ranges = []
        for item in items:
            try:
                lo, hi = MeasurementScheduler._hour_range(item)
            except (TypeError, ValueError):
                raise MeasurementScheduler.InvalidScheduleError(
                    f"'hours' ranges must read like '08:00-12:00' or (8, 12), got {item!r} "
                    f"in {hours!r}."
                )
            if not (0.0 <= lo <= 24.0 and 0.0 <= hi <= 24.0) or lo == hi:
                raise MeasurementScheduler.InvalidScheduleError(
                    f"'hours' ranges must join two different times of day within 0-24 h, "
                    f"got ({lo:g}, {hi:g})."
                )
            ranges.append((lo, hi))
        if not ranges:
            raise MeasurementScheduler.InvalidScheduleError(f"'hours' selects no time of day: {hours!r}.")
        return tuple(ranges)

    @staticmethod
    def _to_slots(pass_cadence, pass_duration, pass_offset) -> tuple:
        """Check a slot recurrence and convert it to seconds.

        Parameters
        ----------
        pass_cadence : ArrayWUnits or None
            Interval at which the slots repeat. None for no slots.
        pass_duration : ArrayWUnits or None
            Length of each slot; required with ``pass_cadence``.
        pass_offset : ArrayWUnits or None
            Start of the first slot after the schedule start; None for zero.

        Returns
        -------
        pass_cadence_sec : float or None
            Cadence in seconds, None without slots.
        pass_duration_sec : float or None
            Slot length in seconds, None without slots.
        pass_offset_sec : float
            Offset in seconds, zero when not given.

        Raises
        ------
        InvalidScheduleError
            If ``pass_duration`` or ``pass_offset`` is given without
            ``pass_cadence``, ``pass_cadence`` without ``pass_duration``, a
            span is invalid (see :meth:`_to_seconds`), or the slot is longer
            than the cadence.
        """
        if pass_cadence is None:
            if pass_duration is not None or pass_offset is not None:
                raise MeasurementScheduler.InvalidScheduleError(
                    "'pass_duration' and 'pass_offset' shape the slots repeated every "
                    "'pass_cadence'; pass pass_cadence=... too (e.g. pass_cadence="
                    "ArrayWUnits(6, hr), pass_duration=ArrayWUnits(30, min))."
                )
            return None, None, 0.0
        if pass_duration is None:
            raise MeasurementScheduler.InvalidScheduleError(
                "'pass_cadence' needs the length of each slot: pass pass_duration=... "
                "(e.g. ArrayWUnits(30, min); one epoch per slot is a pass_duration of "
                "one cadence)."
            )
        pass_cadence_sec = MeasurementScheduler._to_seconds(pass_cadence, "pass_cadence")
        pass_duration_sec = MeasurementScheduler._to_seconds(pass_duration, "pass_duration")
        pass_offset_sec = (
            0.0 if pass_offset is None
            else MeasurementScheduler._to_seconds(pass_offset, "pass_offset", allow_zero=True)
        )
        if pass_duration_sec > pass_cadence_sec:
            raise MeasurementScheduler.InvalidScheduleError(
                f"'pass_duration' ({pass_duration_sec:g} s) must not exceed 'pass_cadence' "
                f"({pass_cadence_sec:g} s): the slots would overlap."
            )
        return pass_cadence_sec, pass_duration_sec, pass_offset_sec

    @staticmethod
    def _to_passes_per_day(passes_per_day) -> int | None:
        """Check a passes-per-day count.

        Parameters
        ----------
        passes_per_day : int or None
            Number of passes per day.

        Returns
        -------
        int or None
            The count, or ``None`` when ``passes_per_day`` is ``None``.

        Raises
        ------
        InvalidScheduleError
            If the count is not a positive integer.
        """
        if passes_per_day is None:
            return None
        try:
            value = int(passes_per_day)
        except (TypeError, ValueError):
            raise MeasurementScheduler.InvalidScheduleError(
                f"'passes_per_day' must be a positive integer, got {passes_per_day!r}."
            )
        if value < 1 or value != passes_per_day:
            raise MeasurementScheduler.InvalidScheduleError(
                f"'passes_per_day' must be a positive integer, got {passes_per_day!r}."
            )
        return value

    @staticmethod
    def _day_table(start_et: float, end_et: float) -> tuple:
        """Return the UTC midnights covering a window and their weekdays.

        Parameters
        ----------
        start_et : float
            Start of the window, in TDB seconds past J2000.
        end_et : float
            End of the window, in TDB seconds past J2000.

        Returns
        -------
        day_starts : numpy.ndarray
            TDB epochs of the midnights; the first is on or before ``start_et``.
        weekdays : numpy.ndarray of int
            Weekday of each day, ``0`` (Monday) to ``6`` (Sunday).
        """
        first = datetime.date.fromisoformat(
            str(SpiceManager.et2utc(float(start_et), "ISOC", 0))[:10]
        )
        n_days = int(np.ceil((end_et - start_et) / constants.day_to_sec)) + 2
        dates = [first + datetime.timedelta(days=k) for k in range(n_days)]
        day_starts = np.array(
            [float(SpiceManager.cal2et(d.strftime("%Y-%m-%dT00:00:00"))) for d in dates]
        )
        weekdays = np.array([d.weekday() for d in dates], dtype=int)
        return day_starts, weekdays

    def resolve_window(
        self,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        pass_gap: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        passes_per_day: int = None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
    ) -> "MeasurementScheduler.Window":
        """Complete a request's window with the scheduler defaults and build its grid.

        Every time span (``cadence``, ``pass_gap``, ``pass_cadence``,
        ``pass_duration``, ``pass_offset``) is a single ``ArrayWUnits`` time,
        such as ``ArrayWUnits(10, min)``.

        Parameters
        ----------
        start, end : str, float or EpochArray, optional
            Bounds of the request (class docstring conventions). ``None``
            takes the scheduler default.
        cadence : ArrayWUnits, optional
            Spacing between epochs. ``None`` takes the scheduler default.
        pass_gap : ArrayWUnits, optional
            Gap that separates two passes. ``None`` takes the scheduler
            default, itself ``None`` meaning one and a half cadences.
        weekdays : str, int or iterable, optional
            Days of the week to keep: names (``"Mon"``, ``"Mon,Thu"``,
            ``["Mon", "Fri"]``), ``"weekdays"`` / ``"weekends"`` or ``0``
            (Monday) .. ``6`` (Sunday). ``None`` keeps every day.
        hours : str, tuple or list, optional
            UTC time-of-day ranges kept within each day: one range
            ``"08:00-12:00"`` or ``(8, 12)``, several
            ``"02:00-04:00, 14:00-16:00"`` or ``[(2, 4), (14, 16)]``;
            ``"22:00-04:00"`` wraps midnight. ``None`` keeps the whole day.
        passes_per_day : int, optional
            Keep only the first ``passes_per_day`` tracking passes starting on
            each UTC day. ``None`` keeps them all.
        pass_cadence : ArrayWUnits, optional
            Time between the starts of two tracking slots, counted in elapsed
            time from ``start`` (not tied to the UTC day: use ``hours`` for
            that). Needs ``pass_duration``. ``None`` tracks over the whole
            window.
        pass_duration : ArrayWUnits, optional
            Length of each slot, at most ``pass_cadence``. Each slot holds the
            epochs ``slot start + k * cadence`` before its end, so a
            ``pass_duration`` of one cadence (or less) is one epoch per slot.
        pass_offset : ArrayWUnits, optional
            Delay of the first slot after ``start``. Defaults to ``0``.

        Returns
        -------
        MeasurementScheduler.Window

        Raises
        ------
        MeasurementScheduler.InvalidScheduleError
            If a bound or the cadence is set neither here nor on the
            scheduler, ``end`` is not after ``start``, a recurrence argument
            is malformed (``pass_cadence`` without ``pass_duration`` or the
            reverse, a time span that is not an ``ArrayWUnits`` time,
            ``pass_duration`` longer than ``pass_cadence``, ...) or the
            recurrence leaves no epoch.
        """
        start_et = self._start_et if start is None else MeasurementScheduler._to_et(start, "start")
        end_et = self._end_et if end is None else MeasurementScheduler._to_et(end, "end")
        cadence_sec = (
            self._cadence_sec if cadence is None else MeasurementScheduler._to_seconds(cadence)
        )
        missing = [
            label
            for label, value in (("start", start_et), ("end", end_et), ("cadence", cadence_sec))
            if value is None
        ]
        if missing:
            raise MeasurementScheduler.InvalidScheduleError(
                f"No {', '.join(missing)} for this request: pass it to the call "
                "or give the scheduler a default."
            )
        gap = (
            self._pass_gap_sec if pass_gap is None
            else MeasurementScheduler._to_seconds(pass_gap, "pass_gap")
        )
        if gap is None:
            gap = 1.5 * cadence_sec
        pass_cadence_sec, pass_duration_sec, pass_offset_sec = MeasurementScheduler._to_slots(
            pass_cadence, pass_duration, pass_offset
        )
        return MeasurementScheduler.Window(
            start_et,
            end_et,
            cadence_sec,
            gap,
            weekdays=MeasurementScheduler._to_weekdays(weekdays),
            hours=MeasurementScheduler._to_hours(hours),
            passes_per_day=MeasurementScheduler._to_passes_per_day(passes_per_day),
            pass_cadence_sec=pass_cadence_sec,
            pass_duration_sec=pass_duration_sec,
            pass_offset_sec=pass_offset_sec,
            edge_tolerance_sec=self._edge_tolerance_sec,
        )

    def _resolve_target(self, target: Spacecraft | Body | None) -> Spacecraft | Body:
        """Return the target of a request.

        Parameters
        ----------
        target : Spacecraft or Body or None
            Target of the request, or ``None`` for the scheduler default.

        Returns
        -------
        Spacecraft or Body
            ``target``, or the scheduler default when it is ``None``.

        Raises
        ------
        MissingTargetError
            If neither ``target`` nor a scheduler default is given.
        """
        target = self._target if target is None else target
        if target is None:
            raise MeasurementScheduler.MissingTargetError(
                "No target for this request: pass target=... to the call or "
                "give the scheduler a default target."
            )
        return target

    @staticmethod
    def build_epochs(
        start: EpochArray | float,
        end: EpochArray | float,
        cadence: ArrayWUnits | float,
        pass_cadence: ArrayWUnits | float = None,
        pass_duration: ArrayWUnits | float = None,
        pass_offset: ArrayWUnits | float = 0.0,
        edge_tolerance: ArrayWUnits | float = None,
    ) -> EpochArray:
        """Build the half-open epoch grid ``[start, end)`` at a fixed cadence.

        With ``pass_cadence`` the grid is cut into slots
        ``[s_k, s_k + pass_duration)``, ``s_k = start + pass_offset + k * pass_cadence``,
        each sampled from its own start at the cadence (so ``pass_cadence``
        need not be a multiple of the cadence).

        Parameters
        ----------
        start : EpochArray or float
            First epoch: an ``EpochArray`` (its first epoch) or TDB seconds
            past J2000. Calendar strings and dates work as for the
            scheduler's ``start``.
        end : EpochArray or float
            End of the grid (excluded), same conventions as ``start``; an
            ``EpochArray`` is taken at its last epoch.
        cadence : ArrayWUnits or float
            Spacing between epochs, an ``ArrayWUnits`` time such as
            ``ArrayWUnits(1, min)``. A bare number is seconds.
        pass_cadence : ArrayWUnits or float, optional
            Time between the starts of two slots (bare number: seconds).
            ``None`` is one slot over the whole span.
        pass_duration : ArrayWUnits or float, optional
            Length of each slot (bare number: seconds); required with
            ``pass_cadence``.
        pass_offset : ArrayWUnits or float, optional
            Delay of the first slot after ``start`` (bare number: seconds).
            Defaults to ``0``.
        edge_tolerance : ArrayWUnits or float, optional
            Slack at ``end`` and at the slot ends (bare number: seconds): a
            grid point this close to them counts as reaching them. Defaults
            to :attr:`EDGE_TOLERANCE`.

        Returns
        -------
        EpochArray
            The epochs, in TDB (possibly empty when ``pass_offset`` passes
            ``end``).

        Raises
        ------
        MeasurementScheduler.InvalidScheduleError
            If ``end`` is not after ``start``, a bound is not an epoch, or a
            span is not a single positive time (``pass_offset`` and
            ``edge_tolerance`` may be zero; see :meth:`_to_seconds`),
            including a ``pass_cadence`` given
            without ``pass_duration``.
        """
        start_et = MeasurementScheduler._to_et(start, "start")
        end_et = MeasurementScheduler._to_et(end, "end")
        cadence_sec = MeasurementScheduler._to_seconds(cadence, "cadence", bare_seconds=True)
        if end_et <= start_et:
            raise MeasurementScheduler.InvalidScheduleError(
                f"'end' ({MeasurementScheduler._cal(end_et)}) must be after "
                f"'start' ({MeasurementScheduler._cal(start_et)})."
            )
        tol = MeasurementScheduler._to_seconds(
            MeasurementScheduler.EDGE_TOLERANCE if edge_tolerance is None else edge_tolerance,
            "edge_tolerance", allow_zero=True, bare_seconds=True,
        )
        # Count the steps instead of np.arange(start, end, dt): ET rounding and
        # the TDB-UTC drift can land a whole number of steps a hair short of
        # ``end`` and produce one epoch too many. Anything within
        # ``edge_tolerance`` of ``end`` counts as ``end`` and is excluded.
        if pass_cadence is None:
            n_epochs = int(np.ceil((end_et - start_et - tol) / cadence_sec))
            return EpochArray(start_et + cadence_sec * np.arange(n_epochs), sys="TDB")
        pass_cadence_sec = MeasurementScheduler._to_seconds(pass_cadence, "pass_cadence", bare_seconds=True)
        pass_duration_sec = MeasurementScheduler._to_seconds(pass_duration, "pass_duration", bare_seconds=True)
        pass_offset_sec = MeasurementScheduler._to_seconds(
            pass_offset, "pass_offset", allow_zero=True, bare_seconds=True
        )
        first_slot = start_et + pass_offset_sec
        n_slots = max(int(np.ceil((end_et - first_slot - tol) / pass_cadence_sec)), 0)
        per_slot = max(int(np.ceil((pass_duration_sec - tol) / cadence_sec)), 1)
        slot_starts = first_slot + pass_cadence_sec * np.arange(n_slots)
        et_values = (slot_starts[:, None] + cadence_sec * np.arange(per_slot)[None, :]).ravel()
        return EpochArray(et_values[et_values < end_et - tol], sys="TDB")

    @classmethod
    def resolve_measurement_type(cls, measurement_type: str | type) -> str:
        """Map a measurement type to the class name of the model it builds.

        The explicit form is the model class name (``"RangeIdeal"``,
        ``"DopplerIdeal"``, ...) or the class itself. The short aliases of
        :attr:`MEASUREMENT_TYPE_ALIASES` (``"range"``, ``"doppler"``,
        ``"opnav"``, ...) always mean the Ideal model. Case, spaces, ``-`` and
        ``_`` are ignored.

        Parameters
        ----------
        measurement_type : str or type
            E.g. ``"DopplerIdeal"``, ``DopplerIdeal``, ``"range"``,
            ``"Range-Rate"``, ``"opnav"``, ``"RA_DEC"``.

        Returns
        -------
        str
            One of :attr:`MODEL_CLASSES`' keys.

        Raises
        ------
        MeasurementScheduler.RealModelNotSupportedError
            If the type names a Real model (``"DopplerReal"``,
            ``"SequentialRangingReal"``).
        MeasurementScheduler.UnknownMeasurementTypeError
            If the spelling is not registered.
        """
        label = measurement_type.__name__ if isinstance(measurement_type, type) else str(measurement_type)
        key = re.sub(r"[^a-z0-9]", "", label.strip().lower())
        class_names = {name.lower(): name for name in cls.MODEL_CLASSES}
        canonical = class_names.get(key) or cls.MEASUREMENT_TYPE_ALIASES.get(key)
        if canonical is not None:
            return canonical

        real_names = {name.lower(): name for name in cls.REAL_MODEL_SUBSTITUTES}
        real_name = real_names.get(key) or cls.REAL_MODEL_ALIASES.get(key)
        if real_name is not None:
            raise MeasurementScheduler.RealModelNotSupportedError(
                f"{real_name} is a Real model, which the scheduler cannot "
                "simulate yet: its per-epoch TRK-2-34 data (uplink ramp "
                "table spanning at most 2 days, count time Tc, M2 ratio, station "
                "delays) are not synthesised. Schedule "
                f"'{cls.REAL_MODEL_SUBSTITUTES[real_name]}' for simulated data."
            )
        raise MeasurementScheduler.UnknownMeasurementTypeError(
            f"Unknown measurement type '{label}'. Known types: "
            f"{sorted(cls.MODEL_CLASSES)} (aliases of the Ideal models: "
            f"{sorted(cls.MEASUREMENT_TYPE_ALIASES)})."
        )

    @staticmethod
    def resolve_observer(observer: str | GroundStation | Camera | Spacecraft):
        """Turn a station name into a :class:`~scarabaeus.GroundStation`; pass objects through.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            A DSN station name such as ``"DSS-14"`` or an already-built
            observer object.

        Returns
        -------
        GroundStation, Camera or Spacecraft
        """
        if isinstance(observer, str):
            return GroundStation(observer)
        return observer

    @classmethod
    def build_model(
        cls,
        measurement_type: str | type,
        observer,
        name: str = None,
        sigma: ArrayWUnits = None,
        **model_kwargs,
    ) -> Measurement:
        """Instantiate the measurement model for a type and observer.

        Parameters
        ----------
        measurement_type : str or type
            Model class name, class or alias accepted by
            :meth:`resolve_measurement_type`.
        observer : str, GroundStation, Camera or Spacecraft
            Station name or observer object.
        name : str, optional
            Model name. Defaults to ``"<observer> <class name>"``, e.g.
            ``"DSS-14 RangeIdeal"``.
        sigma : ArrayWUnits, optional
            Measurement standard deviation passed to the model.
        **model_kwargs
            Extra constructor arguments of the model class (``meas_bias``,
            ``state_definition``, ``sequence_definition``, ``antenna_name``,
            ``doppler_transmit_frequency``, ``ground_station_2``,
            ``observed_target``, ...).

        Returns
        -------
        Measurement
            The model instance.

        Raises
        ------
        MeasurementScheduler.RealModelNotSupportedError
            If the type names a Real model.
        MeasurementScheduler.UnknownMeasurementTypeError
            If the type is not registered.
        MeasurementScheduler.ObserverTypeError
            If the observer does not fit the measurement type (a station
            for centroiding, a camera for range, ...).
        MeasurementScheduler.UnsupportedModelArgumentError
            If a keyword in ``model_kwargs`` is not accepted by the class.
        """
        canonical = cls.resolve_measurement_type(measurement_type)
        observer = cls.resolve_observer(observer)
        model_cls = cls.MODEL_CLASSES[canonical]

        if canonical in cls.RADIOMETRIC_TYPES and not isinstance(observer, GroundStation):
            raise MeasurementScheduler.ObserverTypeError(
                f"'{canonical}' needs a GroundStation observer (or a station "
                f"name), got {type(observer).__name__}."
            )
        if canonical == "CentroidingIdeal" and not isinstance(observer, Camera):
            raise MeasurementScheduler.ObserverTypeError(
                f"'{canonical}' needs a Camera observer, got "
                f"{type(observer).__name__}."
            )
        if canonical == "AngularIdeal" and isinstance(observer, Camera):
            raise MeasurementScheduler.ObserverTypeError(
                "'AngularIdeal' takes the observing Spacecraft or GroundStation "
                "as observer, not the Camera; pass the body carrying the camera."
            )

        params = inspect.signature(model_cls.__init__).parameters
        unknown = sorted(k for k in model_kwargs if k not in params)
        if unknown:
            raise MeasurementScheduler.UnsupportedModelArgumentError(
                f"{model_cls.__name__} does not accept {unknown}; accepted "
                f"keywords: {[p for p in params if p not in ('self', 'name')]}."
            )

        if name is None:
            name = f"{cls._observer_label(observer)} {canonical}"

        # Every ideal model takes (name, <observer>) positionally.
        return model_cls(name, observer, sigma=sigma, **model_kwargs)

    @staticmethod
    def _observer_label(observer) -> str:
        """Return the short printable name of an observer.

        Parameters
        ----------
        observer : object
            The observer.

        Returns
        -------
        str
            Its ``name`` attribute, or its string form when it has none.
        """
        return str(getattr(observer, "name", observer))

    @staticmethod
    def _slug(text: str) -> str:
        """Return a file-name-safe version of a label.

        Parameters
        ----------
        text : str
            The label.

        Returns
        -------
        str
            The label in lower case, with camel-case boundaries and runs of other
            characters turned into single underscores; for example
            ``"DSS-14 RangeIdeal"`` gives ``"dss_14_range_ideal"``.
        """
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(text))
        return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

    # endregion

    # ----------------------#
    # region   Visibility  #
    # ----------------------#
    @staticmethod
    def visibility_on(
        observer,
        target: Spacecraft | Body,
        epochs: EpochArray,
        elevation_mask: ArrayWUnits | float = 10.0,
    ) -> np.ndarray:
        """Boolean visibility mask of a target from an observer at given epochs.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            Ground stations use their elevation test, cameras their FOV
            projection. Any other observer sees the target at every epoch.
        target : Spacecraft or Body
            Tracked or imaged body.
        epochs : EpochArray
            Epochs to test.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation for ground stations, an ``ArrayWUnits`` angle or
            a bare number of degrees. Defaults to ``10.0`` deg.

        Returns
        -------
        numpy.ndarray of bool
            One flag per epoch.
        """
        observer = MeasurementScheduler.resolve_observer(observer)
        if isinstance(observer, GroundStation):
            mask = observer.station_visibility(
                target, epochs, elevation_mask=MeasurementScheduler._to_degrees(elevation_mask)
            )
        elif isinstance(observer, Camera):
            mask = observer.target_in_fov(target, epochs)
        else:
            mask = np.ones(epochs.size, dtype=bool)
        return np.atleast_1d(np.asarray(mask, dtype=bool))

    def visibility(
        self,
        observer,
        elevation_mask: ArrayWUnits | float = None,
        target: Spacecraft | Body = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
    ) -> np.ndarray:
        """Boolean visibility mask of the target from an observer over a window's grid.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            Ground stations use their elevation test, cameras their FOV
            projection. Any other observer sees the target at every epoch.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation, an ``ArrayWUnits`` angle or a bare number of
            degrees; ``None`` uses the scheduler default.
        target : Spacecraft or Body, optional
            Overrides the scheduler target (e.g. the imaged body for OpNav).
        start, end, cadence, weekdays, hours, pass_cadence, pass_duration, pass_offset : optional
            Window of the request; ``None`` takes the scheduler defaults
            (see :meth:`resolve_window`).

        Returns
        -------
        numpy.ndarray of bool
            One flag per epoch of the window's grid.
        """
        window = self.resolve_window(
            start, end, cadence, weekdays=weekdays, hours=hours,
            pass_cadence=pass_cadence, pass_duration=pass_duration, pass_offset=pass_offset,
        )
        mask_deg = (
            self._elevation_mask if elevation_mask is None
            else MeasurementScheduler._to_degrees(elevation_mask)
        )
        return MeasurementScheduler.visibility_on(
            observer, self._resolve_target(target), window.epochs, ArrayWUnits(mask_deg, deg)
        )

    def elevation(
        self,
        observer,
        target: Spacecraft | Body = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
    ) -> np.ndarray:
        """Elevation of the target above a ground station's horizon over a window's grid.

        Parameters
        ----------
        observer : str or GroundStation
            Station name or ground station.
        target : Spacecraft or Body, optional
            Overrides the scheduler target.
        start, end, cadence, weekdays, hours, pass_cadence, pass_duration, pass_offset : optional
            Window of the request; ``None`` takes the scheduler defaults
            (see :meth:`resolve_window`).

        Returns
        -------
        numpy.ndarray
            Elevation in degrees, one value per epoch of the window's grid.

        Raises
        ------
        MeasurementScheduler.ObserverTypeError
            If the observer is not a ground station.
        """
        observer = MeasurementScheduler.resolve_observer(observer)
        if not isinstance(observer, GroundStation):
            raise MeasurementScheduler.ObserverTypeError(
                f"elevation() needs a ground station, got {type(observer).__name__}."
            )
        window = self.resolve_window(
            start, end, cadence, weekdays=weekdays, hours=hours,
            pass_cadence=pass_cadence, pass_duration=pass_duration, pass_offset=pass_offset,
        )
        _, elevation, _ = observer.station_visibility(
            self._resolve_target(target), window.epochs, elevation_mask=0.0, return_angles=True
        )
        return np.atleast_1d(np.asarray(elevation, dtype=float))

    def visible_epochs(
        self,
        observer,
        elevation_mask: ArrayWUnits | float = None,
        target: Spacecraft | Body = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
    ) -> EpochArray:
        """Return the epochs of a window's grid at which the observer sees the target.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            Ground stations use their elevation test, cameras their FOV
            projection. Any other observer sees the target at every epoch.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation, an ``ArrayWUnits`` angle or a bare number of
            degrees; ``None`` uses the scheduler default.
        target : Spacecraft or Body, optional
            Overrides the scheduler target (e.g. the imaged body for OpNav).
        start, end, cadence, weekdays, hours, pass_cadence, pass_duration, pass_offset : optional
            Window of the request; ``None`` takes the scheduler defaults
            (see :meth:`resolve_window`).

        Returns
        -------
        EpochArray
            The visible epochs.
        """
        window = self.resolve_window(
            start, end, cadence, weekdays=weekdays, hours=hours,
            pass_cadence=pass_cadence, pass_duration=pass_duration, pass_offset=pass_offset,
        )
        mask_deg = (
            self._elevation_mask if elevation_mask is None
            else MeasurementScheduler._to_degrees(elevation_mask)
        )
        mask = MeasurementScheduler.visibility_on(
            observer, self._resolve_target(target), window.epochs, ArrayWUnits(mask_deg, deg)
        )
        return window.epochs[mask]

    def passes(
        self,
        observer,
        elevation_mask: ArrayWUnits | float = None,
        target: Spacecraft | Body = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        pass_gap: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        passes_per_day: int = None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
    ) -> list:
        """Return the visible epochs grouped into the tracking passes that would be scheduled.

        A new pass starts whenever two consecutive visible epochs are more than
        the window's pass gap apart. With ``pass_cadence`` and ``pass_duration``
        each slot is its own pass, unless the pass gap bridges consecutive
        slots.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            Ground stations use their elevation test, cameras their FOV
            projection. Any other observer sees the target at every epoch.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation, an ``ArrayWUnits`` angle or a bare number of
            degrees; ``None`` uses the scheduler default.
        target : Spacecraft or Body, optional
            Overrides the scheduler target (e.g. the imaged body for OpNav).
        start, end, cadence, pass_gap, weekdays, hours, passes_per_day, pass_cadence, pass_duration, pass_offset : optional
            Window of the request; ``None`` takes the scheduler defaults
            (see :meth:`resolve_window`). ``passes_per_day`` keeps the first
            passes of each UTC day.

        Returns
        -------
        list of EpochArray
            One entry per pass, in time order.
        """
        window = self.resolve_window(
            start, end, cadence, pass_gap, weekdays=weekdays, hours=hours,
            passes_per_day=passes_per_day, pass_cadence=pass_cadence,
            pass_duration=pass_duration, pass_offset=pass_offset,
        )
        mask_deg = (
            self._elevation_mask if elevation_mask is None
            else MeasurementScheduler._to_degrees(elevation_mask)
        )
        mask = MeasurementScheduler.visibility_on(
            observer, self._resolve_target(target), window.epochs, ArrayWUnits(mask_deg, deg)
        )
        return window.select(mask)[1]

    @staticmethod
    def split_passes(epochs: EpochArray, gap: ArrayWUnits | float) -> list:
        """Split an epoch array into passes at gaps larger than ``gap``.

        Parameters
        ----------
        epochs : EpochArray
            Epochs in time order.
        gap : ArrayWUnits or float
            Time between consecutive epochs beyond which a new pass starts,
            an ``ArrayWUnits`` time such as ``ArrayWUnits(15, min)``. A bare
            number is seconds.

        Returns
        -------
        list of EpochArray
            One entry per pass, in TDB.

        Raises
        ------
        MeasurementScheduler.InvalidScheduleError
            If ``gap`` is not a single positive time span.
        """
        gap_sec = MeasurementScheduler._to_seconds(gap, "gap", bare_seconds=True)
        et_values = np.atleast_1d(epochs.et)
        if et_values.size == 0:
            return []
        breaks = np.where(np.diff(et_values) > gap_sec)[0] + 1
        bounds = np.concatenate(([0], breaks, [et_values.size])).astype(int)
        return [
            EpochArray(et_values[lo:hi], sys="TDB")
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ]

    # endregion

    # ----------------------#
    # region   Scheduling  #
    # ----------------------#
    def schedule(
        self,
        observer,
        measurement_type: str | type,
        sigma: ArrayWUnits = None,
        start: str | float | EpochArray = None,
        end: str | float | EpochArray = None,
        cadence: ArrayWUnits = None,
        elevation_mask: ArrayWUnits | float = None,
        target: Spacecraft | Body = None,
        name: str = None,
        dataset_name: str = None,
        file_name: str = None,
        units: Units = None,
        noisy: bool = None,
        per_pass_bias: bool = False,
        pass_gap: ArrayWUnits = None,
        weekdays=None,
        hours=None,
        passes_per_day: int = None,
        pass_cadence: ArrayWUnits = None,
        pass_duration: ArrayWUnits = None,
        pass_offset: ArrayWUnits = None,
        **model_kwargs,
    ) -> MeasurementSpec:
        """Build one measurement model and generate its observations.

        Parameters
        ----------
        observer : str, GroundStation, Camera or Spacecraft
            Station name (``"DSS-14"``), ground station, camera (for
            ``"CentroidingIdeal"``) or observing body (for
            ``"AngularIdeal"``).
        measurement_type : str or type
            Class name of the model to build: ``"RangeIdeal"``,
            ``"RangeRateIdeal"``, ``"DopplerIdeal"``,
            ``"DiffOneWayRangeIdeal"``, ``"CentroidingIdeal"`` or
            ``"AngularIdeal"`` (or the class itself). The short aliases of
            :attr:`MEASUREMENT_TYPE_ALIASES` (``"range"``, ``"doppler"``,
            ``"opnav"``, ...) always mean the Ideal model. Real models
            (``"DopplerReal"``, ``"SequentialRangingReal"``) are refused.
        sigma : ArrayWUnits, optional
            Measurement standard deviation handed to the model.
        start, end : str, float or EpochArray, optional
            Window of this request (class docstring conventions). ``None``
            takes the scheduler default.
        cadence : ArrayWUnits, optional
            Spacing between this request's measurements, e.g.
            ``ArrayWUnits(1, min)``. ``None`` takes the scheduler default.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation for this request only, an ``ArrayWUnits`` angle
            or a bare number of degrees.
        target : Spacecraft or Body, optional
            Overrides the scheduler target (the imaged body for OpNav).
        name : str, optional
            Model name. Defaults to ``"<observer> <class name>"``, e.g.
            ``"DSS-14 RangeIdeal"``.
        dataset_name : str, optional
            Dataset label on the returned spec and key in :attr:`scheduled`.
            Defaults to the model name.
        file_name : str, optional
            Measurement file name without extension. Defaults to a slug of
            the dataset name; optical types get an ``opnav_`` prefix so the
            model files them under ``data/measurements/optical``.
        units : Units, optional
            Units the file is read back with. Defaults to
            :attr:`OBSERVATION_UNITS` for the type.
        noisy : bool, optional
            Overrides the scheduler's ``noisy`` flag for this request.
        per_pass_bias : bool, optional
            When ``True``, the visible passes are registered on the model
            with :meth:`~scarabaeus.Measurement.set_pass_epochs` so a
            per-pass bias can be estimated. Defaults to ``False``.
        pass_gap : ArrayWUnits, optional
            Gap between visible epochs that starts a new pass for this
            request. ``None`` takes the scheduler default (1.5 cadences).
        weekdays : str, int or iterable, optional
            Days of the week to track on (``"Mon"``, ``"Mon,Thu"``,
            ``"weekdays"``, ``0``-``6``). Defaults to every day.
        hours : str, tuple or list, optional
            UTC time-of-day ranges to track in: ``"08:00-12:00"``,
            ``(8, 12)``, or several such as ``"02:00-04:00, 14:00-16:00"``
            or ``[(2, 4), (14, 16)]``. Defaults to the whole day.
        passes_per_day : int, optional
            Keep only the first ``passes_per_day`` passes starting on each
            UTC day ("one pass a day"). Defaults to all of them.
        pass_cadence : ArrayWUnits, optional
            Time between the starts of two tracking slots, in elapsed time
            from ``start`` (``ArrayWUnits(6, hr)``, ``ArrayWUnits(90, min)``),
            for schedules not tied to the UTC day such as OpNav images.
            Needs ``pass_duration``. Defaults to tracking over the whole
            window.
        pass_duration : ArrayWUnits, optional
            Length of each slot, at most ``pass_cadence``; a slot holds the
            epochs ``slot start + k * cadence`` before its end (one epoch when
            ``pass_duration`` is one cadence or less).
        pass_offset : ArrayWUnits, optional
            Delay of the first slot after ``start``. Defaults to ``0``.
        **model_kwargs
            Extra constructor arguments of the model class (``meas_bias``,
            ``antenna_name``, ``doppler_transmit_frequency``,
            ``ground_station_2``, ``observed_target``, ...).

        Returns
        -------
        MeasurementSpec
            Leaf spec with the model and the observations read from disk.
            The full record (window, epochs, mask, passes, file path) is
            stored in :attr:`scheduled` under ``dataset_name``.

        Raises
        ------
        MeasurementScheduler.DuplicateDatasetError
            If ``dataset_name`` is given and already in :attr:`scheduled`
            (a default name is suffixed ``(2)``, ``(3)``, ... instead).
        MeasurementScheduler.RealModelNotSupportedError
            If ``measurement_type`` names a Real model.
        MeasurementScheduler.UnknownMeasurementTypeError
            If ``measurement_type`` is not registered.
        MeasurementScheduler.InvalidScheduleError
            If the request has no start, end or cadence (neither given here
            nor as a scheduler default), its recurrence is malformed or
            leaves no epoch, or ``elevation_mask`` is not an angle between 0
            and 90 deg.
        MeasurementScheduler.MissingTargetError
            If the request has no target.
        MeasurementScheduler.MissingSigmaError
            If noisy observations are requested without a ``sigma``.
        MeasurementScheduler.NoVisibleEpochsError
            If the observer never sees the target over the window.
        MeasurementScheduler.MeasurementFileNotWrittenError
            If the model wrote no file.
        """
        canonical = MeasurementScheduler.resolve_measurement_type(measurement_type)
        observer = MeasurementScheduler.resolve_observer(observer)
        target = self._resolve_target(target)
        window = self.resolve_window(
            start, end, cadence, pass_gap, weekdays=weekdays, hours=hours,
            passes_per_day=passes_per_day, pass_cadence=pass_cadence,
            pass_duration=pass_duration, pass_offset=pass_offset,
        )
        noisy = self._noisy if noisy is None else bool(noisy)
        mask_deg = (
            self._elevation_mask if elevation_mask is None
            else MeasurementScheduler._to_degrees(elevation_mask)
        )

        model = MeasurementScheduler.build_model(
            canonical, observer, name=name, sigma=sigma, **model_kwargs
        )
        if noisy and sigma is None:
            raise MeasurementScheduler.MissingSigmaError(
                f"'{model.name}': noisy observations need a measurement 'sigma' "
                "(ArrayWUnits) to draw the noise from; pass sigma=... or schedule "
                "with noisy=False."
            )
        if dataset_name is None:
            # Same sensor scheduled again (another window / cadence): keep both records
            dataset_name = model.name
            k = 2
            while dataset_name in self._scheduled:
                dataset_name = f"{model.name} ({k})"
                k += 1
        elif dataset_name in self._scheduled:
            raise MeasurementScheduler.DuplicateDatasetError(
                f"'{dataset_name}' is already scheduled; pick another dataset_name."
            )
        if file_name is None:
            file_name = MeasurementScheduler._slug(dataset_name)
            if canonical in MeasurementScheduler.OPTICAL_TYPES and not any(
                kw in file_name for kw in ("opnav", "optical", "centroiding")
            ):
                file_name = f"opnav_{file_name}"

        ## Visibility over this request's grid, then the passes actually tracked
        # Passed on with its units: a bare 0.5 deg would be refused as ambiguous
        mask = MeasurementScheduler.visibility_on(
            observer, target, window.epochs, ArrayWUnits(mask_deg, deg)
        )
        if not mask.any():
            hint = (
                f" Consider a lower elevation mask (currently {mask_deg:g} deg)."
                if isinstance(observer, GroundStation)
                else ""
            )
            raise MeasurementScheduler.NoVisibleEpochsError(
                f"'{dataset_name}': {MeasurementScheduler._observer_label(observer)} "
                f"never sees {MeasurementScheduler._observer_label(target)} over "
                f"{window}." + hint
            )
        mask, passes = window.select(mask)
        epochs = window.epochs[mask]
        if per_pass_bias:
            model.set_pass_epochs([np.atleast_1d(p.et).tolist() for p in passes])

        ## Write the observations and read them back
        model.write_observed_measurements(
            target=target,
            epoch_array=epochs,
            frame=self._frame,
            noisy=noisy,
            file_name=file_name,
            check_visibility=False,
            folder_path_override=self._output_dir,
            fmt=self._fmt,
        )
        file_path = self._file_path(file_name)
        if not os.path.exists(file_path):
            raise MeasurementScheduler.MeasurementFileNotWrittenError(
                f"'{dataset_name}': {type(model).__name__} wrote no file at "
                f"'{file_path}'."
            )
        if units is None:
            units = MeasurementScheduler.OBSERVATION_UNITS[canonical]
        observed = model.observed_measurements(
            file_name=file_path, meas_name="meas_sim", units=units, frame=self._frame
        )

        spec = MeasurementSpec(
            model=model,
            observed_meas=observed,
            dataset_name=dataset_name,
            file_label=file_path,
        )
        self._scheduled[dataset_name] = MeasurementScheduler.ScheduledMeasurement(
            spec=spec,
            model=model,
            observed_meas=observed,
            measurement_type=canonical,
            observer=observer,
            target=target,
            window=window,
            epochs=epochs,
            visibility_mask=mask,
            passes=passes,
            file_path=file_path,
        )
        return spec

    def schedule_many(
        self, requests, skip_invisible: bool = True, **common
    ) -> MeasurementSpec:
        """Run a tracking plan of several requests and return one container spec.

        Requests of different kinds (stations and cameras, radiometric and
        optical types) with different windows and cadences go in the same
        plan.

        Parameters
        ----------
        requests : iterable of dict or tuple
            Each entry is the keyword dictionary of one :meth:`schedule` call,
            or a ``(observer, measurement_type)`` pair optionally followed by
            a keyword dictionary, e.g.
            ``("DSS-14", "RangeIdeal", {"sigma": s, "cadence": ArrayWUnits(10, min)})``.
        skip_invisible : bool, optional
            When ``True`` a request whose observer never sees the target is
            skipped with a ``RuntimeWarning`` instead of stopping the plan.
            Defaults to ``True``.
        **common
            :meth:`schedule` keywords applied to every request of the plan
            that does not set them itself (a shared ``start``/``end``, an
            ``elevation_mask``, ...).

        Returns
        -------
        MeasurementSpec
            Container built with :meth:`~scarabaeus.MeasurementSpec.many` over the
            completed requests, in plan order.
        """
        specs = []
        for request in requests:
            if isinstance(request, dict):
                kwargs = dict(request)
            else:
                request = tuple(request)
                kwargs = dict(request[2]) if len(request) > 2 else {}
                kwargs["observer"] = request[0]
                kwargs["measurement_type"] = request[1]
            kwargs = {**common, **kwargs}
            try:
                specs.append(self.schedule(**kwargs))
            except MeasurementScheduler.NoVisibleEpochsError as err:
                if not skip_invisible:
                    raise
                warnings.warn(f"{err} Skipped.", RuntimeWarning, stacklevel=2)
        return MeasurementSpec.many(*specs)

    @classmethod
    def generate(
        cls,
        target: Spacecraft | Body,
        observer,
        measurement_type: str | type,
        start: str | float | EpochArray,
        end: str | float | EpochArray,
        cadence: ArrayWUnits,
        elevation_mask: ArrayWUnits | float = 10.0,
        sigma: ArrayWUnits = None,
        frame: Frame = J2000,
        noisy: bool = False,
        output_dir: str = None,
        fmt: str = "json",
        **schedule_kwargs,
    ) -> MeasurementSpec:
        """One-call form: build a scheduler and run a single request.

        Parameters
        ----------
        target : Spacecraft or Body
            Body being tracked or imaged.
        observer : str, GroundStation, Camera or Spacecraft
            Station name or observer object.
        measurement_type : str or type
            Model class name (``"RangeIdeal"``, ...), class or alias (see
            :meth:`schedule`).
        start, end : str, float or EpochArray
            Window bounds (see the class docstring).
        cadence : ArrayWUnits
            Spacing between measurements, e.g. ``ArrayWUnits(10, min)``.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation, an ``ArrayWUnits`` angle or a bare number of
            degrees. Defaults to ``10.0`` deg.
        sigma : ArrayWUnits, optional
            Measurement standard deviation.
        frame, noisy, output_dir, fmt
            Forwarded to the constructor.
        **schedule_kwargs
            Forwarded to :meth:`schedule` (``name``, ``dataset_name``,
            ``file_name``, ``units``, ``hours``, ``pass_cadence``/``pass_duration``,
            model keywords, ...).

        Returns
        -------
        MeasurementSpec
        """
        scheduler = cls(
            target,
            start,
            end,
            cadence,
            elevation_mask=elevation_mask,
            frame=frame,
            noisy=noisy,
            output_dir=output_dir,
            fmt=fmt,
        )
        return scheduler.schedule(observer, measurement_type, sigma=sigma, **schedule_kwargs)

    def _file_path(self, file_name: str) -> str:
        """Return the path at which a measurement file is written.

        Parameters
        ----------
        file_name : str
            File name, without its suffix.

        Returns
        -------
        str
            The path :meth:`~scarabaeus.Measurement.write_observed_measurements`
            writes to: the scheduler output directory when one is set, otherwise
            ``data/measurements/optical`` for optical names and
            ``data/measurements/radiometric`` for the others, with the suffix of
            the scheduler format.
        """
        if self._output_dir is not None:
            save_dir = self._output_dir
        else:
            optical_keywords = ("centroiding", "landmarks", "opnav", "optical", "beacon")
            save_folder = (
                "measurements/optical"
                if any(kw in file_name.lower() for kw in optical_keywords)
                else "measurements/radiometric"
            )
            save_dir = os.path.join("data", save_folder)
        return os.path.join(save_dir, file_name + format_suffix(self._fmt))

    # endregion

    def __repr__(self) -> str:
        def cal(et):
            return None if et is None else MeasurementScheduler._cal(et)

        target = None if self._target is None else MeasurementScheduler._observer_label(self._target)
        cadence = None if self._cadence_sec is None else f"{self._cadence_sec:g} s"
        return (
            f"MeasurementScheduler(target={target!r}, start={cal(self._start_et)!r}, "
            f"end={cal(self._end_et)!r}, cadence={cadence}, "
            f"elevation_mask={self._elevation_mask} deg, scheduled={list(self._scheduled)})"
        )
