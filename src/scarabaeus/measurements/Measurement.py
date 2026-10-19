# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import scarabaeus as scb
from scarabaeus import (
    constants,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Body,
    Spacecraft,
    Instrument,
    SpiceManager,
    MeasurementDataSet,
    StateArray,
    StateDefinition,
    ObsData
)

import scarabaeus.utils.NumpyWrapper as np
import warnings
from scarabaeus.utils.Serialization import dump, load, format_suffix
from scipy.optimize import root_scalar
import spiceypy as spice
from tqdm import tqdm
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Tuple

# ------------------#
#  Generate Units  #
# ------------------#
J2000, ITRF93, ECLIPJ2000, IAUEARTH = scb.Frame.generate_common_frames()
km, sec, rad = Units.get_units(["km", "sec", "rad"])

# Generate common units and frames
spc_pos_units = km
spc_time_units = sec
spc_angle_units = rad
spc_vel_units = spc_pos_units / spc_time_units
spc_ad_units = spc_pos_units / spc_pos_units

J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class Measurement:
    """Abstract base class for all Scarabaeus measurement models (radiometric and optical).

    Parameters
    ----------
    name : str
        The name of the Measurement object.
    instrument : Instrument
        The instrument object linked to the measurement model (e.g. an Antenna, a Camera, etc.)
    Raises
    ------
    ValueError
        If observed and computed measurements do not share the same reference frame and
        therefore cannot be differenced to form residuals.
    ValueError
        If a single measurement model generates more than 2 components per epoch,
        which is not supported in the current version.

    See Also
    --------
    scarabaeus.RangeIdeal : Ideal two-way range measurement model.
    scarabaeus.RangeRateIdeal : Ideal two-way range-rate measurement model.
    scarabaeus.DopplerIdeal : Ideal Doppler frequency-shift measurement model.
    scarabaeus.DiffOneWayRangeIdeal : Ideal differential one-way range measurement model.
    scarabaeus.CentroidingIdeal : Ideal optical centroiding measurement model.
    scarabaeus.DopplerReal : Real DSN Doppler measurement model with RTLT corrections.
    scarabaeus.SequentialRangingReal : Real DSN sequential ranging measurement model.

    Notes
    -----
    This is the abstract interface that all subclasses implement.  Each concrete
    subclass must provide:

    * ``computed_measurements`` — evaluate the forward model on a trajectory.
    * ``observed_measurements`` — load observed data from a JSON file.
    * ``compute_partials`` — assemble the :math:`H` row (measurement partial).
    * ``generate_measurement_dataset`` — produce a
      :class:`~scarabaeus.MeasurementDataSet` ready for a filter.

    No measurement-specific formula is defined here; see subclass docstrings
    for the mathematical model of each observable.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(self, name: str, instrument: Instrument):
        self._name = name
        self._instrument = instrument
        self._trajectory = None
        # Names of the two components of a vector observable, set here so every
        # model declares them beside its ``_measurement_type``. Only meaningful
        # for a two-component model, which should name what it actually
        # measures -- ("ra", "dec"), ("sample", "line") -- rather than inherit
        # this placeholder.
        self._component_labels = ("c1", "c2")
        # Set by the subclasses that use them; None here so every model has them.
        self._sigma = None
        self._meas_bias = None
        self._reference_state_vector: StateArray | None = None
        self._epoch_to_pass_index: dict | None = None
        self._epoch_sec = None
        self._computed_measurements_dict = None
        self._media_corrections = None
        # Radiometric defaults, set through the properties so a class default
        # a subclass overrides is checked like a user value.
        cls = type(self)
        self.dsn_bands = cls.DSN_BANDS
        self.ramp_rate = cls.RAMP_RATE
        self.ramp_margin = cls.RAMP_MARGIN
        self.ramp_entries = cls.RAMP_ENTRIES
        self.ramp_type = cls.RAMP_TYPE
        self.transponder_delay = cls.TRANSPONDER_DELAY
        self.rcv_time_tag_delay = cls.RCV_TIME_TAG_DELAY
        self.ul_station_cal = cls.UL_STATION_CAL
        self.dl_station_cal = cls.DL_STATION_CAL
        # Looked up at call time, like the media_corrections setter does.
        self.solar_corona_coefficient = scb.MediaCorrections.SOLAR_CORONA_DELAY_COEFF

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError. Subclasses inherit these.
    class MissingTrajectoryError(Exception):
        """No reference trajectory is attached.

        Raised by :meth:`_stm_pair_mapping` when a deviation has to be mapped
        between two epochs: only the trajectory STMs carry that mapping.
        """

    class StationKernelRequiredError(Exception):
        """The light-time chain needs a station SPK, not table coordinates.

        Raised by :meth:`_require_station_kernel`. The Moyer light-time
        solution reads the station from the kernel pool at several epochs and
        several frames, so a station defined only by
        :attr:`~scarabaeus.GroundStation.body_fixed_position` has nothing to
        read. The geometric models compose that position themselves and accept
        such a station; write the coordinates to an SPK to use one here.
        """

    class LightTimeConvergenceError(Exception):
        """The one-way Newtonian light-time root search did not converge.

        Raised by :meth:`_compute_CN_lt` when the Brent solve on the
        light-time residual reports no convergence inside its bracket.
        """

    class LightTimeFrameError(ValueError):
        """The requested light-time frame is not one of :attr:`LIGHT_TIME_FRAMES`.

        Raised by :meth:`_resolve_light_time_frame` when a real model is built.
        """

    class LightTimeSettingsError(ValueError):
        """The light-time settings hold an unknown key or an invalid value.

        Raised by :meth:`_resolve_light_time_settings`, for the keys of
        :attr:`LIGHT_TIME_SETTINGS`.
        """

    class AuxDataError(ValueError):
        """The auxiliary dictionary of a real measurement model is unusable.

        Raised by :meth:`_per_epoch_aux` when the dictionary is missing, a
        required key is absent, or a per-epoch entry does not have one value
        per measurement epoch (records with their own cadence must first be
        aligned with :meth:`align_aux_to_epochs`), and by
        :meth:`align_aux_to_epochs` when a record holds no valid entry.
        """

    class MissingSigmaError(ValueError):
        """A measurement sigma is needed but none was given.

        Raised when noise is requested without one, when a dataset is built
        from a model with none (the sigma weights each residual), and by
        CustomMeasurement when neither ``units`` nor ``sigma`` fixes the
        observable's units. MeasurementScheduler raises this same class when
        noisy observations are scheduled without a sigma.
        """

    class SigmaUnitsError(ValueError):
        """The measurement sigma cannot be expressed in the measurement's units.

        Raised by :meth:`_require_sigma` when the sigma carries units of
        another dimension than the computed measurement (a range sigma in
        km/s, say), so the dataset's ``sigma`` column cannot be built from it.
        """

    class RLTBodyOnPathError(ValueError):
        """A relativistic light-time body lies on the signal path.

        Raised by :meth:`_compute_RLT_BODIES_lt` when a leg ends at, or runs
        through, the centre of one of its bodies -- typically a target that is
        itself listed in ``rlt_bodies`` -- where the delay of Moyer (2000),
        Eq. 8-55, diverges.
        """

    class MissingPassIndexError(Exception):
        """A per-pass bias has no epoch-to-pass mapping.

        Raised by :meth:`_bias_at_epoch` and :meth:`_bias_partials` when the
        bias, from ``meas_bias`` or a state entry, holds one value per pass
        (more than one element) but :meth:`set_pass_epochs` was never called,
        so there is no way to know which pass an epoch belongs to.
        """

    class UnsetPoleParameterError(ValueError):
        """An IAU pole component is finite-differenced without a value.

        Raised by :meth:`_finite_difference_station` when ``pole_ra``,
        ``pole_dec``, ``pole_pm`` or ``pole_w0`` is declared for a station's
        central body without a value. Unset, it leaves the secular IAU term in
        place, and that term moves with time, so there is no single value to
        step about. The analytical partials linearize about the secular model
        and need none.
        """

    class SequencePartialsError(NotImplementedError):
        """A partials method cannot follow a mission sequence.

        Raised by the real radiometric models when ``partials_method="fd"`` is
        asked for with a sequence definition. That path builds every column
        from one state definition for the whole epoch array, while a sequence
        changes the definition from leg to leg; the analytical partials switch
        it epoch by epoch through :meth:`_use_leg_state_definition`.
        """

    class ElevationMaskError(ValueError):
        """An elevation mask is not an angle from 0 to 90 degrees, or is ambiguous.

        Raised by :meth:`resolve_elevation_mask` when the mask is not a number
        or an angle with units, falls outside 0 to 90 deg, or is a bare number
        that could be radians (not a whole number of degrees, and no more than
        pi/2). MeasurementScheduler raises its ``InvalidScheduleError`` from it.
        """

    class SolarCoronaScaleError(ValueError):
        """The estimated solar-corona scale is not one finite number.

        Raised by :meth:`_solar_corona_scale` when the ``solar_corona_scale``
        entry has more than one component or a non-finite value, or when the
        state declares it more than once. The scale is a single model
        parameter shared by every link.
        """

    class GammaRelError(ValueError):
        """The estimated PPN gamma is not one finite number.

        Raised by :meth:`_gamma_rel` when the ``gamma_rel`` entry has more
        than one component or a non-finite value, or when the state declares
        it more than once. Gamma is a single parameter of the light-time
        solution shared by every link.
        """

    class RadiometricDefaultError(ValueError):
        """A radiometric default is given a value it cannot take.

        Raised by the setters of the radiometric-default properties
        (:attr:`dsn_bands`, :attr:`ramp_rate`, :attr:`ramp_margin`, ...) when
        the value is not a single finite number, has units that do not convert
        to the property's, is not a whole number where a count or a code is
        expected, or is out of range; and by :attr:`dsn_bands` when the
        mapping is empty or holds something other than :class:`DSNBand`.
        """

    # endregion Errors #
    # ------------------#

    # ---------------------------#
    # region  Auxiliary data     #
    # ---------------------------#
    @dataclass(frozen=True)
    class TrackingAuxKeys:
        """ Where a real radiometric model keeps its auxiliary data.

        The auxiliary data of a real model is the ``computed_measurements_dict``
        entry list that travels with the observations: the per-epoch values the
        forward model needs besides the ramp table, the TRK-2-34 record fields
        they are read from, and the model's copy of the outlier flags.

        Attributes
        ----------
        required : tuple of str
            Per-epoch keys the forward model needs, besides the ramp table
            (:attr:`Measurement.RAMP_AUX_KEYS`).
        optional : tuple of str
            Per-epoch keys read when present.
        record_fields : dict
            TRK-2-34 record field name -> aux key, for the fields whose names
            differ from the aux key.
        outlier_flag : str
            Key of the model's own outlier-flag copy, which
            :meth:`MeasurementSpec.flag` keeps in step with
            ``ObsData.meas_outliers``.

        References
        ----------
        .. [1] "TRK-2-34, DSN Tracking System Data Archival Format", DSN No.
            820-013, TRK-2-34, Rev. N.
        """
        required: tuple
        optional: tuple
        record_fields: dict
        outlier_flag: str

    # TrackingAuxKeys of the real radiometric models, whose auxiliary data
    # travels with the observations; None for a model that keeps none.
    AUX_KEYS = None

    RAMP_AUX_KEYS = ("ramp_sec", "ramp_day", "ramp_freq", "ramp_rate", "ramp_type")
    """Auxiliary-data keys of the ramp table every real radiometric model
    carries: start second of day, day of year, frequency, rate and type of each
    ramp segment."""

    RAMP_YEAR_KEY = "ramp_year"
    """Auxiliary-data key of the year of each ramp segment. Optional: a table
    read without it is assumed to lie in the year of the observations."""

    # endregion Auxiliary data #
    # --------------------------#

    # ------------------------------#
    # region  Radiometric defaults  #
    # ------------------------------#
    @dataclass(frozen=True)
    class DSNBand:
        """The band-dependent numbers a two-way radiometric observable needs.

        One of these per DSN frequency band, so the turnaround ratio and the
        range-unit scale are defined once per band rather than as literals in
        each measurement model. A model looks its band up in
        :attr:`Measurement.dsn_bands`.

        Attributes
        ----------
        name : str
            Band letter, ``'X'`` or ``'S'``.
        uplink_multiplier : int
            Multiple of the DSN reference frequency the uplink carrier sits at:
            749 for X band, 221 for S band.
        downlink_multiplier : int
            Same for the downlink: 880 for X band, 240 for S band.
        turnaround_num, turnaround_den : int
            The transponder turnaround ratio :math:`M_2 = f_{down}/f_{up}` as a
            fraction, 880/749 for X-up/X-down and 240/221 for S-up/S-down. Kept
            as two integers because that is how the tracking files carry it and
            because the ratio is exact.
        exciter_scalar_num, exciter_scalar_den : float
            The exciter scalars a tracking file carries, from which the range
            unit follows as ``((den / num) / 16) ** -1``. That works out to
            ``uplink_multiplier_S / (2 * uplink_multiplier)``, i.e. 221/1498 for
            X band and 1/2 for S band, so that the range unit is the same
            physical quantity in either band: the product of the range unit and
            the uplink frequency is band-invariant at 1.055181 GHz.
        uplink_frequency : ArrayWUnits
            Representative uplink carrier frequency [1/s]. The two defaults are
            one band apart in the same channel, 7.1523135969977 GHz for X band
            and 2.110362223 GHz for S band, their ratio being exactly
            749/221. A real pass carries its own, so this only seeds simulated
            auxiliary data.
        ru_reference_frequency : ArrayWUnits
            Reference frequency used to convert a spacecraft transponder delay
            from range units to seconds, as ``delay_RU / (range_unit *
            ru_reference_frequency)`` [1/s].

            This is NOT the uplink frequency, and substituting the uplink
            frequency is measurably wrong: doing so makes the delay 18% longer,
            1.4 km of two-way range, and drives the sequential-ranging residuals
            on real tracking data from 55-206 RU up to 2330-2694 RU. The X-band
            value is the one the models used as a literal and is what fits the
            data; why a downlink-scale reference is the right one here is an
            open question, and the S-band entry simply scales it by the
            downlink multipliers, 240/880, so it is unverified against data.
        """
        name: str
        uplink_multiplier: int
        downlink_multiplier: int
        turnaround_num: int
        turnaround_den: int
        exciter_scalar_num: float
        exciter_scalar_den: float
        uplink_frequency: ArrayWUnits
        ru_reference_frequency: ArrayWUnits

        @property
        def range_unit(self) -> float:
            """The range unit implied by the exciter scalars, ``RU``.

            One range unit of observed count corresponds to a round-trip phase
            of this many cycles of the uplink carrier.
            """
            return ((self.exciter_scalar_den / self.exciter_scalar_num) / 16) ** -1

    DSN_BANDS = {
        "X": DSNBand(name                   = "X",
                     uplink_multiplier      = 749,
                     downlink_multiplier    = 880,
                     turnaround_num         = 880,
                     turnaround_den         = 749,
                     exciter_scalar_num     = 221.0,
                     exciter_scalar_den     = 23968.0,
                     uplink_frequency       = ArrayWUnits(7.1523135969977e09, 1 / sec),
                     ru_reference_frequency = ArrayWUnits(8445.767679e06, 1 / sec)),
        "S": DSNBand(name                   = "S",
                     uplink_multiplier      = 221,
                     downlink_multiplier    = 240,
                     turnaround_num         = 240,
                     turnaround_den         = 221,
                     exciter_scalar_num     = 221.0,
                     exciter_scalar_den     = 7072.0,
                     uplink_frequency       = ArrayWUnits(2.110362223e09, 1 / sec),
                     ru_reference_frequency = ArrayWUnits(8445.767679e06 * 240 / 880, 1 / sec)),
    }
    """Default DSN frequency-band parameters, keyed by band letter, each a
    :class:`DSNBand`. A model starts with these as :attr:`dsn_bands`.

    Supported bands are ``'X'`` and ``'S'``. ``'X'`` holds the values the
    measurement models used as literals before they were collected here, so
    it is the default everywhere and reproduces the previous numbers exactly.

    .. [1] Moyer, T. D., Formulation for Observed and Computed Values of Deep
        Space Network Data Types for Navigation, JPL Publication 00-7,
        Monograph 2, Deep Space Communications and Navigation Series,
        Chapter 13.
    .. [2] DSN Telecommunications Link Design Handbook, 810-005, Module 201,
        Frequency and Channel Assignments.
    """

    RAMP_RATE = ArrayWUnits(0.2826, 1 / sec**2)
    """Default uplink frequency ramp rate of a simulated ramp table, the
    starting :attr:`ramp_rate` of a model.

    A representative DSN ramp slope, not a physical constant: real passes
    carry their measured rate in the ramp table.
    """

    RAMP_MARGIN = ArrayWUnits(3600.0, sec)
    """Default span a simulated ramp table extends before the first and after
    the last observation, the starting :attr:`ramp_margin` of a model.

    The uplink transmit time precedes the first tag by the round-trip light
    time, so a table that starts at the first tag raises
    :class:`~scarabaeus.measurements.RampTableManager.RampTableManager.RampCoverageError`.
    One hour clears a spacecraft out to the asteroid belt.
    """

    RAMP_ENTRIES = 12
    """Default number of entries of a simulated ramp table, the starting
    :attr:`ramp_entries` of a model."""

    RAMP_TYPE = 1
    """Default ramp type of a simulated ramp table, the SFDU 9 ``ramp_type``
    code of a ramped segment; the starting :attr:`ramp_type` of a model."""

    TRANSPONDER_DELAY = 1.0e-5
    """Default spacecraft transponder delay [RU] of simulated auxiliary data,
    the starting :attr:`transponder_delay` of a model.

    Representative, not physical: real passes carry the calibrated delay.
    """

    RCV_TIME_TAG_DELAY = ArrayWUnits(8.0e-4, sec)
    """Default receive time-tag delay of simulated auxiliary data, the
    starting :attr:`rcv_time_tag_delay` of a model.

    Representative, not physical: real passes carry the station value.
    """

    UL_STATION_CAL = 5000.0
    """Default uplink station calibration [RU] of simulated auxiliary data,
    the starting :attr:`ul_station_cal` of a model.

    Representative, not physical: real passes carry the station value.
    """

    DL_STATION_CAL = 850000.0
    """Default downlink station calibration [RU] of simulated auxiliary data,
    the starting :attr:`dl_station_cal` of a model.

    Representative, not physical: real passes carry the station value.
    """

    @staticmethod
    def _radiometric_default(name: str, value, units: Units = None, integer: bool = False,
                             minimum: float = None, strict: bool = False):
        """
        Check a value given to one of the radiometric-default properties.

        Parameters
        ----------
        name : str
            Property name, quoted in the error message.
        value : ArrayWUnits, int or float
            The value. An ArrayWUnits is converted to *units*; a bare number
            is taken to be in *units* already.
        units : Units, optional
            Units the property keeps its value in. None for a count, a code or
            a value in range units, which take bare numbers only.
        integer : bool, optional
            Whether the value must be a whole number. Defaults to False.
        minimum : float, optional
            Lower bound of the value, in *units*. None for no bound.
        strict : bool, optional
            Whether the bound itself is refused. Defaults to False.

        Returns
        -------
        int or float
            The value in *units*, an int when *integer*.

        Raises
        ------
        Measurement.RadiometricDefaultError
            If the value is not one finite number, carries units when the
            property has none or units that do not convert to *units*, is
            not whole when *integer*, or is below *minimum*.
        """
        if isinstance(value, ArrayWUnits):
            if units is None:
                raise Measurement.RadiometricDefaultError(
                    f"'{name}' takes a bare number, got {value}."
                )
            try:
                value = value.convert_to(units).values
            except (TypeError, ValueError) as err:
                raise Measurement.RadiometricDefaultError(
                    f"'{name}' must be in units of {units}, got {value.units}."
                ) from err
        if isinstance(value, (bool, np.bool_)) or np.size(value) != 1:
            raise Measurement.RadiometricDefaultError(
                f"'{name}' must be a single number, got {value!r}."
            )
        try:
            number = float(np.ravel(value)[0])
        except (TypeError, ValueError) as err:
            raise Measurement.RadiometricDefaultError(
                f"'{name}' must be a number, got {value!r}."
            ) from err
        if not np.isfinite(number):
            raise Measurement.RadiometricDefaultError(f"'{name}' must be finite, got {number}.")
        if integer:
            if not number.is_integer():
                raise Measurement.RadiometricDefaultError(
                    f"'{name}' must be a whole number, got {number:g}."
                )
            number = int(number)
        if minimum is not None and (number < minimum or (strict and number == minimum)):
            raise Measurement.RadiometricDefaultError(
                f"'{name}' must be {'above' if strict else 'at least'} {minimum:g}, "
                f"got {number:g}."
            )
        return number

    # endregion Radiometric defaults #
    # -------------------------------#

    def _require_station_kernel(self):
        """Reject a table-defined station where only a station SPK will do.

        Raises
        ------
        StationKernelRequiredError
            When this model's station carries explicit body-fixed coordinates
            instead of an ephemeris.
        """
        station = self._observer_station()
        if station is not None and station.body_fixed_position is not None:
            raise self.StationKernelRequiredError(
                f"{type(self).__name__} resolves {station.name} through the kernel pool; "
                "give the station an SPK instead of body_fixed_position, or use a "
                "geometric model (RangeIdeal, RangeRateIdeal, DopplerIdeal, "
                "DiffOneWayRangeIdeal), which composes table coordinates directly")

    # -----------------------------#
    # region  Station parameters   #
    # -----------------------------#
    # Canonical units are part of the state-parameter contract. Dimensions and
    # name patterns are read from StateDefinition, so there is only one schema.
    # Every entry is a *correction* to the loaded nominal geometry, never an
    # absolute coordinate or orientation.
    _station_parameter_units = {
        "gs_delta_location_ECEF": "km", "gs_delta_location_CYL": "km",
        "eop_delta_pole_x": "rad", "eop_delta_pole_y": "rad",
        "eop_delta_ut1": "sec", "body_orientation_delta": "rad",
        "eop_delta_celestial_pole_x": "rad", "eop_delta_celestial_pole_y": "rad",
        # Love and Shida numbers are dimensionless, so they carry no canonical
        # unit and no conversion; None selects that path in _station_parameters.
        "love_h2": None, "shida_l2": None,
        # The central body's IAU pole and prime meridian. pole_pm is read in
        # deg/day without conversion, as SphericalHarmonicsGravity and the
        # optical models read it, since StateDefinition declares it
        # dimensionless.
        "pole_ra": "rad", "pole_dec": "rad", "pole_pm": None, "pole_w0": "rad",
    }

    # Columns of GroundStation.tide_partials, in order.
    _tide_parameter_columns = {"love_h2": 0, "shida_l2": 1}

    # Estimated zenith path delays. These do not move the station, so they are
    # kept apart from the geometry parameters above; they are read from the
    # state the same way and handed to MediaCorrections. Metres, matching the
    # units the media model works in.
    _media_parameter_units = {"tropo_zenith_delay": "m", "iono_zenith_delay": "m"}

    # Estimated electronic delays, by the link end whose delay each adds to.
    # The t1 and t3 delays belong to the station, the t2 one to the spacecraft;
    # none is split by pass. They lengthen the round-trip light time only.
    _electronic_delay_link_ends = {
        "uplink_delay_bias": "t1",
        "transponder_delay_bias": "t2",
        "downlink_delay_bias": "t3",
    }

    # Earth EOP components chain through a body-fixed rotation, so they share
    # the orientation partial and differ only by the constant column below.
    _eop_correction_arguments = {
        "eop_delta_pole_x": "delta_pole_x", "eop_delta_pole_y": "delta_pole_y",
        "eop_delta_ut1": "delta_ut1",
    }

    # The pole deltas also move the pole an attached PoleTide responds to;
    # columns of GroundStation.pole_tide_partials, in order.
    _pole_tide_columns = {"eop_delta_pole_x": 0, "eop_delta_pole_y": 1}

    # Celestial pole offsets rotate the station about axes held fixed in J2000,
    # not about body-fixed ones, so they have their own Jacobian.
    _celestial_pole_arguments = {
        "eop_delta_celestial_pole_x": "delta_x", "eop_delta_celestial_pole_y": "delta_y",
    }

    @staticmethod
    def _spice_id(body):
        """Resolve a Body, numeric ID or SPICE name to its integer NAIF ID.

        Parameters
        ----------
        body : Body or GroundStation or int or str
            Anything with a ``spice_id``, else with a ``name``, else the ID or
            SPICE name itself.

        Returns
        -------
        int
            NAIF ID.

        Raises
        ------
        SpiceyError
            If the name is not one SPICE knows.
        """
        value = getattr(body, "spice_id", None)
        if value is None:
            value = getattr(body, "name", body)
        try:
            return int(value)
        except (ValueError, TypeError):
            return spice.bods2c(str(value))

    @classmethod
    def _station_parameter_kind(cls, name: str):
        """Station-geometry or tidal-response key a component name matches.

        Parameters
        ----------
        name : str
            State component name, e.g. ``"gs_delta_location_ECEF_DSS-14"``.

        Returns
        -------
        str or None
            Key of :attr:`_station_parameter_units` whose pattern matches
            ``name``, or None when none does.
        """
        allowed = StateDefinition._generate_allowed_state_dictionary()
        return next((key for key in cls._station_parameter_units
                     if allowed[key][2].match(name)), None)

    @classmethod
    def _media_parameter_kind(cls, name: str):
        """Zenith path-delay key a component name matches.

        Parameters
        ----------
        name : str
            State component name.

        Returns
        -------
        str or None
            Key of :attr:`_media_parameter_units` whose pattern matches
            ``name``, or None when none does.
        """
        allowed = StateDefinition._generate_allowed_state_dictionary()
        return next((key for key in cls._media_parameter_units
                     if allowed[key][2].match(name)), None)

    @classmethod
    def _is_station_parameter(cls, name: str) -> bool:
        """True for a station-geometry or tidal-response component name.

        Lets a model's ``_partials`` branch on the component explicitly rather
        than calling the station helper for everything and relying on it to
        return zeros. The two agree, but the explicit form makes the
        zero-fill visible where the row is assembled.

        Parameters
        ----------
        name : str
            State component name.

        Returns
        -------
        bool
            Whether :meth:`_station_parameter_kind` matches ``name``.
        """
        return cls._station_parameter_kind(name) is not None

    @classmethod
    def _is_media_parameter(cls, name: str) -> bool:
        """True for an estimated zenith path-delay component name.

        Parameters
        ----------
        name : str
            State component name.

        Returns
        -------
        bool
            Whether :meth:`_media_parameter_kind` matches ``name``.
        """
        return cls._media_parameter_kind(name) is not None

    def _media_parameters(self, station=None):
        """Estimated zenith delays in the reference state for this station.

        Parameters
        ----------
        station : GroundStation, optional
            Defaults to :meth:`_observer_station`.

        Returns
        -------
        list of dict
            Entries shaped like :meth:`_station_parameters`, in metres. Empty
            when nothing is estimated, which is the ordinary case.

        Raises
        ------
        ValueError
            For a wrong dimension or non-finite values.

        Notes
        -----
        A zenith delay is tagged per station and owned by it: the atmosphere
        above one antenna says nothing about the atmosphere above another.
        """
        station = self._observer_station() if station is None else self._as_station(station)
        entries = self._state_entries()
        if station is None or not entries:
            return []
        allowed = StateDefinition._generate_allowed_state_dictionary()
        station_id = self._spice_id(station)
        parameters = []
        for entry in entries:
            name, dimension, owner = entry[0], entry[1], entry[4]
            kind = self._media_parameter_kind(name)
            if kind is None or self._spice_id(owner) != station_id:
                continue
            if dimension != allowed[kind][1]:
                raise ValueError(f"{name} requires {allowed[kind][1]} components, got {dimension}")
            unit = Units.get_units([self._media_parameter_units[kind]])
            value = entry[5] if len(entry) > 5 else None
            quantity = getattr(value, "quantity", value)
            if quantity is None:
                values, scale = np.zeros(dimension), np.ones(dimension)
            else:
                values = np.asarray(quantity.convert_to(unit).values, dtype=float).reshape(dimension)
                scale = np.asarray(ArrayWUnits(np.ones(dimension), quantity.units)
                                   .convert_to(unit).values, dtype=float).reshape(dimension)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            override = getattr(self, "_station_override", None)
            if override is not None and override[0] == name:
                values = override[1]
            parameters.append({"name": name, "kind": kind, "values": values,
                               "scale": scale, "frame": None, "cylindrical": False})
        return parameters

    def _zenith_delays(self, station=None):
        """Estimated troposphere and ionosphere zenith delays.

        Parameters
        ----------
        station : GroundStation, optional
            Station whose delays are summed. Defaults to
            :meth:`_observer_station`.

        Returns
        -------
        tropo_zenith_delay : float
            Troposphere zenith delay, m, zero when not estimated.
        iono_zenith_delay : float
            Ionosphere zenith delay, m, zero when not estimated.

        Raises
        ------
        ValueError
            As :meth:`_media_parameters`.
        """
        delays = {"tropo_zenith_delay": 0.0, "iono_zenith_delay": 0.0}
        for parameter in self._media_parameters(station):
            delays[parameter["kind"]] += float(parameter["values"][0])
        return delays["tropo_zenith_delay"], delays["iono_zenith_delay"]

    @staticmethod
    def _as_station(candidate):
        """Return ``candidate`` when it is a ground station, else None.

        A model can be handed a spacecraft-mounted antenna as its instrument or
        second endpoint — spacecraft-to-spacecraft range and Doppler do exactly
        that. Such an observer has no body-fixed frame or central body, so it
        carries no station parameters and must not be probed for them.

        Parameters
        ----------
        candidate : object or None
            Instrument or endpoint to test.

        Returns
        -------
        GroundStation or None
            ``candidate`` when it has a ``body_fixed_frame`` and a
            ``central_body``, else None.
        """
        if candidate is not None and hasattr(candidate, "body_fixed_frame") \
                and hasattr(candidate, "central_body"):
            return candidate
        return None

    @staticmethod
    def _spice_designator(body):
        """SPICE designator for ``body``: its id where it has one, else its name.

        A ``Camera``'s ``name`` is the instrument's and means nothing to SPICE,
        but its ``spice_id`` is the body it rides on, so the id is preferred.

        Parameters
        ----------
        body : Body, Camera, str or int
            Endpoint to resolve.

        Returns
        -------
        str or object
            ``str(spice_id)`` (spkezr takes the body as a string; a bare int
            resolves to nothing), else ``name``, else ``body`` itself.
        """
        spice_id = getattr(body, "spice_id", None)
        if spice_id is not None:
            return str(spice_id)
        return getattr(body, "name", body)

    def _observer_station(self):
        """This model's ground station, or None when it observes from elsewhere.

        Returns
        -------
        GroundStation or None
            The instrument itself when it is a surface observer. An optical
            model whose instrument is a spacecraft camera has no station, and
            estimating station parameters cannot change what it sees.
        """
        return self._as_station(self.instrument)

    def _station_parameters(self, station=None):
        """Collect the reference state's correction parameters for one station.

        Parameters
        ----------
        station : GroundStation, optional
            Station to collect for. Defaults to :meth:`_observer_station`.

        Returns
        -------
        list of dict
            One entry per matching component, carrying ``name``, ``kind``,
            ``values`` in canonical units, the ``scale`` back to the declared
            units, the declared ``frame`` and whether it is ``cylindrical``.
            Empty when nothing is estimated, which is the ordinary case.
            ``values`` is None for an IAU pole component declared without a
            value.

        Raises
        ------
        ValueError
            For a wrong dimension, non-finite values, two location blocks on one
            station, or an Earth-specific EOP parameter on another body.

        Notes
        -----
        A location block belongs to the station named as its owner. Orientation
        and EOP parameters may be owned by the station or by its central body,
        where they act on every station on that body. The IAU pole
        (``pole_ra``, ``pole_dec``, ``pole_pm``, ``pole_w0``) is read only when
        owned by the central body. Parameters belonging to another station are
        left to that station's own model.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        station = self._observer_station() if station is None else self._as_station(station)
        entries = self._state_entries()
        if station is None or not entries:
            return []
        allowed = StateDefinition._generate_allowed_state_dictionary()
        station_id = self._spice_id(station)
        centre_id = self._spice_id(station.central_body)
        parameters, located = [], False
        for entry in entries:
            name, dimension, owner = entry[0], entry[1], entry[4]
            kind = self._station_parameter_kind(name)
            if kind is None:
                continue
            owner_id = self._spice_id(owner)
            is_location = kind.startswith("gs_delta_location")
            is_pole = kind in BodyOrientation.IAU_POLE_PARAMETERS
            # love_h2/shida_l2 follow the body-owned rule, like orientation.
            if owner_id != station_id and (is_location or owner_id != centre_id):
                continue
            # The IAU pole is the central body's own; a station has none.
            if is_pole and owner_id != centre_id:
                continue
            if dimension != allowed[kind][1]:
                raise ValueError(f"{name} requires {allowed[kind][1]} components, got {dimension}")
            if is_location:
                if located:
                    raise ValueError(f"Declare one Cartesian or cylindrical location "
                                     f"block per station, not both, for {station.name}")
                located = True
            if kind.startswith("eop_") and centre_id != constants.EARTH.body_center_id:
                raise ValueError("EOP parameters are Earth-specific; use "
                                 "body_orientation_delta on another body")
            unit_name = self._station_parameter_units[kind]
            unit = None if unit_name is None else Units.get_units([unit_name])
            value = entry[5] if len(entry) > 5 else None
            quantity = getattr(value, "quantity", value)
            if quantity is None and kind in self._tide_parameter_columns:
                # Unset: the reference tide's own number, so declaring the
                # parameter leaves the forward model where it was.
                reference = station.tide_reference
                values = np.full(dimension, 0.0 if reference is None
                                 else float(getattr(reference, kind)))
                scale = np.ones(dimension)
            elif quantity is None and is_pole:
                # Unset: nothing replaces the secular term. No single value
                # stands for it, since the secular angle moves with time.
                values, scale = None, np.ones(dimension)
            elif quantity is None:
                values = np.zeros(dimension)
                scale = np.ones(dimension)
            elif unit is None or getattr(quantity, "units", None) is None:
                values = np.asarray(getattr(quantity, "values", quantity),
                                    dtype=float).reshape(dimension)
                scale = np.ones(dimension)
            else:
                values = np.asarray(quantity.convert_to(unit).values, dtype=float).reshape(dimension)
                scale = np.asarray(ArrayWUnits(np.ones(dimension), quantity.units)
                                   .convert_to(unit).values, dtype=float).reshape(dimension)
            if values is not None and not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            # Set only while _finite_difference_station perturbs this component.
            override = getattr(self, "_station_override", None)
            if override is not None and override[0] == name:
                values = override[1]
            parameters.append({
                "name": name, "kind": kind, "values": values, "scale": scale,
                "frame": getattr(getattr(value, "frame", None), "name", None),
                "cylindrical": kind == "gs_delta_location_CYL",
            })
        return parameters

    def _station_location(self, parameters):
        """The estimated location block of one station.

        Parameters
        ----------
        parameters : list of dict
            Entries from :meth:`_station_parameters`.

        Returns
        -------
        location : numpy.ndarray or None
            Location offset in km, shape (3,), or None when no location is
            estimated.
        frame : str or None
            Frame the offset is declared in, None for the station's own.
        cylindrical : bool
            Whether the offset is cylindrical (spin-axis distance, longitude,
            height) rather than Cartesian.
        """
        for parameter in parameters:
            if parameter["kind"].startswith("gs_delta_location"):
                return (parameter["values"], parameter["frame"], parameter["cylindrical"])
        return None, None, False

    def _station_rotation(self, parameters, station=None):
        """Total estimated body-fixed rotation vector.

        Sums ``body_orientation_delta`` with the rotation of any Earth EOP
        deltas.

        Parameters
        ----------
        parameters : list of dict
            Entries from :meth:`_station_parameters`.
        station : GroundStation, optional
            Station the rotation applies to. Defaults to
            :meth:`_observer_station`.

        Returns
        -------
        numpy.ndarray
            Rotation vector, rad, shape (3,), in the station's body-fixed
            axes; zeros when nothing is estimated.

        Raises
        ------
        ValueError
            If ``body_orientation_delta`` declares a frame other than the
            station's body-fixed frame. The rotation is applied in that frame,
            so a differently declared one would be silently misinterpreted
            rather than rotated.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        station = self._observer_station() if station is None else self._as_station(station)
        rotation = np.zeros(3)
        eop = {}
        for parameter in parameters:
            kind = parameter["kind"]
            if kind == "body_orientation_delta":
                frame = parameter["frame"]
                if (frame is not None and station is not None
                        and frame != station.body_fixed_frame):
                    raise ValueError(
                        f"body_orientation_delta is declared in {frame} but is applied "
                        f"in the station's body-fixed frame {station.body_fixed_frame}")
                rotation = rotation + parameter["values"]
            elif kind in self._eop_correction_arguments:
                eop[self._eop_correction_arguments[kind]] = float(parameter["values"][0])
        if eop:
            rotation = rotation + BodyOrientation.earth_correction(**eop).rotation_vector
        return rotation

    def _celestial_pole_rotation(self, parameters):
        """Total estimated celestial-pole-offset rotation vector.

        Parameters
        ----------
        parameters : list of dict
            Entries from :meth:`_station_parameters`.

        Returns
        -------
        numpy.ndarray or None
            Rotation vector, rad, shape (3,), in J2000 axes, from
            :meth:`~scarabaeus.BodyOrientation.celestial_pole_correction`; None
            when no offset is estimated.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        offsets = {self._celestial_pole_arguments[parameter["kind"]]: float(parameter["values"][0])
                   for parameter in parameters
                   if parameter["kind"] in self._celestial_pole_arguments}
        return BodyOrientation.celestial_pole_correction(**offsets) if offsets else None

    def _pole_tide_delta(self, parameters):
        """Estimated pole delta an attached pole tide follows.

        Parameters
        ----------
        parameters : list of dict
            Entries from :meth:`_station_parameters`.

        Returns
        -------
        numpy.ndarray or None
            ``[delta_pole_x, delta_pole_y]`` in radians, the order of
            :meth:`~scarabaeus.GroundStation.pole_tide_partials`; None when
            neither is estimated.
        """
        delta, found = np.zeros(2), False
        for parameter in parameters:
            if parameter["kind"] in self._pole_tide_columns:
                delta[self._pole_tide_columns[parameter["kind"]]] = float(parameter["values"][0])
                found = True
        return delta if found else None

    @staticmethod
    def _iau_pole_overrides(parameters):
        """Estimated IAU pole values of the station's central body.

        Parameters
        ----------
        parameters : list of dict
            Entries from :meth:`_station_parameters`.

        Returns
        -------
        dict
            ``pole_ra``, ``pole_dec``, ``pole_w0`` (rad) and ``pole_pm``
            (deg/day) for each one declared with a value, the keywords of
            :meth:`~scarabaeus.GroundStation.iau_pole_correction`; empty when
            none is, which leaves the station where it was.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        return {parameter["kind"]: float(parameter["values"][0]) for parameter in parameters
                if parameter["kind"] in BodyOrientation.IAU_POLE_PARAMETERS
                and parameter["values"] is not None}

    def _station_state_correction(self, epoch, frame, station=None):
        """Total station state correction to subtract from a relative state.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : Frame or str
            Axes of the relative state the correction is applied to.
        station : GroundStation, optional
            Defaults to :meth:`_observer_station`.

        Returns
        -------
        numpy.ndarray or None
            Six components in km and km/s, or None when this model has no
            station or nothing corrects it.

        Notes
        -----
        The observable is formed from
        :math:`\\rho = r_{target} - r_{station}`, so a caller subtracts this
        from the nominal relative state. Estimated location and orientation
        corrections and any attached
        :attr:`~scarabaeus.GroundStation.body_orientation` are applied
        together, once, by the station itself. Estimated Love and Shida
        numbers move the attached tide from its own numbers to the estimate
        through :meth:`~scarabaeus.GroundStation.tide_correction`. Estimated
        celestial pole offsets turn the nominal station about J2000 axes
        through :meth:`~scarabaeus.GroundStation.inertial_rotation_partials`.
        Estimated terrestrial pole deltas also move an attached pole tide
        through :meth:`~scarabaeus.GroundStation.pole_tide_partials`. An
        estimated IAU pole of the central body turns the station from the
        secular orientation to the estimated one through
        :meth:`~scarabaeus.GroundStation.iau_pole_correction`.
        """
        station = self._observer_station() if station is None else self._as_station(station)
        if station is None:
            return None
        parameters = self._station_parameters(station)
        if not parameters and station.body_orientation is None:
            return None
        epoch, frame = float(epoch), getattr(frame, "name", frame)
        location, location_frame, cylindrical = self._station_location(parameters)
        correction = station._state_correction(
            epoch, frame, location=location,
            location_frame=location_frame, cylindrical=cylindrical,
            rotation_vector=self._station_rotation(parameters, station))
        tides = {parameter["kind"]: float(parameter["values"][0]) for parameter in parameters
                 if parameter["kind"] in self._tide_parameter_columns}
        if tides:
            correction = correction + station._tide_correction(epoch, frame, **tides)
        celestial = self._celestial_pole_rotation(parameters)
        if celestial is not None:
            correction = correction + station._inertial_rotation_partials(epoch, frame) @ celestial
        pole = self._pole_tide_delta(parameters)
        if pole is not None:
            correction = correction + station._pole_tide_partials(epoch, frame) @ pole
        iau_pole = self._iau_pole_overrides(parameters)
        if iau_pole:
            correction = correction + station._iau_pole_correction(epoch, frame, **iau_pole)
        return correction

    def _itrf_station_correction(self):
        """Callable giving this station's ITRF93 correction, or None.

        Returns
        -------
        callable or None
            Function of TDB seconds returning the six-component km/km-s
            correction, for the light-time solver to add to the station SPK
            state. None when nothing corrects the station, which keeps the
            uncorrected path free of extra work.

        Notes
        -----
        The light-time models read the station from its kernel rather than
        through :meth:`~scarabaeus.GroundStation.nominal_state`, so the
        correction carries everything the kernel does not: the estimated
        location and orientation corrections with any attached
        :attr:`~scarabaeus.GroundStation.body_orientation`, the attached
        :attr:`~scarabaeus.GroundStation.surface_displacement`, the move of
        the attached tide to estimated Love and Shida numbers, any
        estimated celestial pole offset, the move of an attached pole tide
        with estimated pole deltas, and any estimated IAU pole of the central
        body. This puts
        the station where :meth:`_relative_state` puts it for the ideal
        models.

        The light-time chain re-evaluates the station at several epochs while
        it iterates, so the correction is supplied as a function of epoch
        rather than as a single vector.
        """
        station = self._observer_station()
        if station is None:
            return None
        parameters = self._station_parameters(station)
        displaced = any(model is not None for model in np.atleast_1d(
            np.asarray(station.surface_displacement, dtype=object)))
        if not parameters and station.body_orientation is None and not displaced:
            return None
        # Read the state once here, not once per light-time iteration: the RTLT
        # solver calls this back several times per observation and the
        # parameters cannot change inside a single evaluation.
        corrected = bool(parameters) or station.body_orientation is not None
        location, location_frame, cylindrical = self._station_location(parameters)
        rotation = self._station_rotation(parameters, station)
        tides = {parameter["kind"]: float(parameter["values"][0]) for parameter in parameters
                 if parameter["kind"] in self._tide_parameter_columns}
        celestial = self._celestial_pole_rotation(parameters)
        pole = self._pole_tide_delta(parameters)
        iau_pole = self._iau_pole_overrides(parameters)

        def correction(epoch):
            """ITRF93 station correction at one epoch.

            Parameters
            ----------
            epoch : float
                TDB seconds past J2000.

            Returns
            -------
            numpy.ndarray
                Six components in km and km/s.
            """
            epoch = float(epoch)
            total = np.zeros(6)
            if corrected:
                total = total + station._state_correction(
                    epoch, "ITRF93", location=location, location_frame=location_frame,
                    cylindrical=cylindrical, rotation_vector=rotation)
            if displaced:
                total = total + station._displacement_state(epoch, "ITRF93")
            if tides:
                total = total + station._tide_correction(epoch, "ITRF93", **tides)
            if celestial is not None:
                total = total + station._inertial_rotation_partials(epoch, "ITRF93") @ celestial
            if pole is not None:
                total = total + station._pole_tide_partials(epoch, "ITRF93") @ pole
            if iau_pole:
                total = total + station._iau_pole_correction(epoch, "ITRF93", **iau_pole)
            return total

        return correction

    def _rtlt_station_row(self, component, t3, t1, n_dn, n_up, v_sc, v_gs, frame_name, c):
        """Round-trip light-time partial with respect to one station parameter.

        Parameters
        ----------
        component : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``.
        t3, t1 : float
            Reception and transmission epochs at the station [TDB s].
        n_dn, n_up : numpy.ndarray, shape (3,)
            Downleg and upleg unit vectors, station to spacecraft.
        v_sc : numpy.ndarray, shape (3,)
            Spacecraft velocity at the reflection time [km/s].
        v_gs : numpy.ndarray, shape (3,)
            Station velocity at ``t1`` [km/s].
        frame_name : str
            Axes of the vectors above.
        c : float
            Speed of light [km/s].

        Returns
        -------
        numpy.ndarray
            :math:`\\partial \\rho / \\partial p` [s per declared unit], shape
            ``(dimension,)``; zeros when the component does not move this
            station.

        Notes
        -----
        The station Jacobian of :meth:`_station_state_partials` is evaluated at
        each end of the light path, since Earth rotation between :math:`t_1` and
        :math:`t_3` turns a body-fixed offset into different inertial ones. Each
        end then carries its own Moyer (2000) 12-12/12-17 light-time weight from
        :meth:`rtlt_grad_station_exact`, and the change the displaced station
        makes to ET - TAI there (Eq. 2-23, :meth:`_station_clock_gradient`).

        References
        ----------
        .. [1] Moyer, T. D. (2000). Formulation for Observed and Computed Values of
            Deep Space Network Data Types for Navigation. JPL Publication 00-7.
        """
        J3 = self._station_state_partials(component, t3, frame_name)
        if J3 is None:
            return np.zeros(component[1])
        J1 = self._station_state_partials(component, t1, frame_name)
        g3, g1 = Measurement.rtlt_grad_station_exact(n_dn, n_up, v_sc, v_gs, c)
        g3 = g3 - self._station_clock_gradient(t3, frame_name)
        g1 = g1 + self._station_clock_gradient(t1, frame_name)
        return g3 @ J3[:3] + g1 @ J1[:3]

    def _relative_state(self, target, epoch, frame, station=None):
        """Target state relative to this model's observer, station corrections applied.

        Parameters
        ----------
        target : Body
            Observed body.
        epoch : EpochArray or float
            Single processing epoch, TDB seconds past J2000.
        frame : Frame or str
            Output axes.
        station : GroundStation, optional
            Observer. Defaults to :meth:`_observer_station`; None falls back to the
            instrument, which is what an optical model uses.

        Returns
        -------
        ArrayWUnits
            Six components in km and km/s.

        Notes
        -----
        A station given explicit body-fixed coordinates has no SPK of its own,
        so its geometry is composed from the central-body ephemeris and those
        coordinates. Estimated location and orientation corrections are then
        subtracted, because the observable is
        :math:`\\rho = r_{target} - r_{station}` and these corrections move the
        station. Applying them here, once, is what keeps the forward model and
        :meth:`_compute_h_tilde_station` linearized about the same geometry.
        """
        observer = self.instrument if station is None else station
        station = self._as_station(observer)
        value = float(np.asarray(getattr(getattr(epoch, "times", None), "values", epoch)).reshape(-1)[0])
        frame_name = getattr(frame, "name", frame)
        # Compose through nominal_state whenever it differs from the station
        # SPK: table coordinates, a surface displacement or the barycentric
        # transformation. Querying the SPK directly would skip a solid tide on
        # a kernel station that a table station gets -- the two must not
        # disagree. A bare kernel station reads the same geometry either way.
        composed = station is not None and (
            station.body_fixed_position is not None or station.barycentric_transformation
            or any(model is not None for model in np.atleast_1d(
                np.asarray(station.surface_displacement, dtype=object))))
        if composed:
            centre = getattr(station.central_body, "spice_id", station.central_body)
            state = np.asarray(spice.spkezr(str(getattr(target, "name", target)), value,
                                            frame_name, "NONE", str(centre))[0], dtype=float)
            state = state - station._nominal_state(value, frame_name)
            relative = ArrayWUnits(state, [km] * 3 + [km / sec] * 3)
        else:
            relative = SpiceManager.get_state(
                trgt_bdy=getattr(target, "name", target), epoch_time=value,
                reference_frame=frame_name, obsvr_bdy=self._spice_designator(observer))
        correction = self._station_state_correction(value, frame_name, station)
        if correction is None:
            return relative
        values = np.asarray(relative.values, dtype=float).reshape(-1)
        return ArrayWUnits(values - correction[:values.size], relative.units)

    def _station_state_partials(self, component, epoch, frame, station=None):
        """Sensitivity of the station's state to one state-definition component.

        Parameters
        ----------
        component : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``.
        epoch : float
            TDB seconds past J2000.
        frame : Frame or str
            Axes of the relative state the observable is formed in.
        station : GroundStation, optional
            Defaults to :meth:`_observer_station`.

        Returns
        -------
        numpy.ndarray or None
            ``(6, dimension)`` Jacobian in km and km/s per declared state unit,
            or None when the component does not move this station.

        Notes
        -----
        Analytical throughout: location corrections are linear in the station
        position, and orientation corrections are a first-order rigid rotation,
        so both have exact Jacobians. Every terrestrial Earth EOP component
        chains the same orientation partial through a constant column from
        :meth:`~scarabaeus.BodyOrientation.earth_correction_partials`; the
        celestial pole offsets do the same with the inertial rotation partial
        and :meth:`~scarabaeus.BodyOrientation.celestial_pole_correction_partials`.
        The terrestrial pole deltas add the column of an attached pole tide,
        :meth:`~scarabaeus.GroundStation.pole_tide_partials`, which is exact
        because that tide is linear in the pole. The IAU pole components take
        their column of :meth:`~scarabaeus.GroundStation.iau_pole_partials`,
        linearized about the estimated values, or about the secular model for
        a component declared without one.
        """
        station = self._observer_station() if station is None else self._as_station(station)
        if station is None:
            return None
        name = component[0]
        kind = self._station_parameter_kind(name)
        if kind is None:
            return None
        parameters = self._station_parameters(station)
        parameter = next((p for p in parameters if p["name"] == name), None)
        if parameter is None:
            return None
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        epoch, frame = float(epoch), getattr(frame, "name", frame)
        location, location_frame, cylindrical = self._station_location(parameters)
        if kind in self._tide_parameter_columns:
            # The tide is linear in these, so the column is exact. A station
            # with no tide model attached has no sensitivity and returns zeros,
            # which a filter will see as an unobservable parameter.
            jacobian = station._tide_partials(epoch, frame)[
                :, [self._tide_parameter_columns[kind]]]
        elif kind in self._celestial_pole_arguments:
            column = BodyOrientation.celestial_pole_correction_partials()[
                self._celestial_pole_arguments[kind]]
            jacobian = (station._inertial_rotation_partials(epoch, frame) @ column).reshape(6, 1)
        elif kind in BodyOrientation.IAU_POLE_PARAMETERS:
            jacobian = station._iau_pole_partials(
                epoch, frame, **self._iau_pole_overrides(parameters))[
                :, [BodyOrientation.IAU_POLE_PARAMETERS.index(kind)]]
        elif kind.startswith("gs_delta_location"):
            jacobian = station._location_partials(
                epoch, frame, cylindrical=cylindrical, location_frame=location_frame,
                rotation_vector=self._station_rotation(parameters, station))
        else:
            jacobian = station._rotation_partials(
                epoch, frame, location=location, location_frame=location_frame,
                cylindrical=cylindrical)
            if kind in self._eop_correction_arguments:
                column = BodyOrientation.earth_correction_partials()[
                    self._eop_correction_arguments[kind]]
                jacobian = (jacobian @ column).reshape(6, 1)
            if kind in self._pole_tide_columns:
                jacobian = jacobian + station._pole_tide_partials(epoch, frame)[
                    :, [self._pole_tide_columns[kind]]]
        # Back to the units the state declares, so H matches the state vector.
        return jacobian * parameter["scale"]

    #: Finite-difference perturbations for station parameters, used only by
    #: ``partials_method="fd"`` (:meth:`_compute_partials_with_fd`); every
    #: analytical path differentiates the station in closed form. They live in
    #: the same ``eps_dict`` as the spacecraft
    #: ``dx``/``dvx`` entries, keyed by StateDefinition parameter kind, so a
    #: model has one perturbation dictionary rather than two. The values are
    #: large because an RTLT solution and ramp integration lose significance in
    #: a small perturbation; they are numerical choices, not prior sigmas. The
    #: light time carries about 1e-13 s of noise per leg (SSB positions near
    #: 1.5e8 km), and a unit Love number moves a station by centimetres, so a
    #: Doppler fed by a step near one is mostly that noise. The tide and the
    #: zenith delays are exactly linear in their parameters, so their steps are
    #: as large as needed.
    station_eps_defaults = {
        "gs_delta_location_ECEF": ArrayWUnits(1.0, km),
        "gs_delta_location_CYL": ArrayWUnits(1.0, km),
        "eop_delta_pole_x": ArrayWUnits(1e-4, rad),
        "eop_delta_pole_y": ArrayWUnits(1e-4, rad),
        "eop_delta_ut1": ArrayWUnits(1.0, sec),
        "eop_delta_celestial_pole_x": ArrayWUnits(1e-4, rad),
        "eop_delta_celestial_pole_y": ArrayWUnits(1e-4, rad),
        "body_orientation_delta": ArrayWUnits(1e-4, rad),
        "love_h2": 100.0,
        "shida_l2": 100.0,
        # pole_pm in deg/day: over the ~9000 days since J2000 this turns the
        # meridian by about 2e-4 rad, like the angle steps beside it.
        "pole_ra": ArrayWUnits(1e-4, rad),
        "pole_dec": ArrayWUnits(1e-4, rad),
        "pole_pm": 1e-6,
        "pole_w0": ArrayWUnits(1e-4, rad),
        "tropo_zenith_delay": ArrayWUnits(10.0, Units.get_units(["m"])),
        "iono_zenith_delay": ArrayWUnits(10.0, Units.get_units(["m"])),
    }

    def _station_eps(self, kind: str) -> float:
        """Central-difference step for a station parameter, in canonical units.

        Parameters
        ----------
        kind : str
            StateDefinition parameter kind.

        Returns
        -------
        float
            Step in km, radians or seconds. Read from this model's ``eps_dict``
            under the same key when present, otherwise from
            :attr:`station_eps_defaults`.

        Raises
        ------
        ValueError
            For a non-positive or non-finite step.
        """
        step = (getattr(self, "eps_dict", None) or {}).get(
            kind, self.station_eps_defaults[kind])
        unit_name = {**self._station_parameter_units,
                     **self._media_parameter_units}[kind]
        # A dimensionless parameter (a Love number) has no unit to convert to.
        if hasattr(step, "convert_to") and unit_name is not None:
            step = step.convert_to(Units.get_units([unit_name])).values
        step = float(np.asarray(step).reshape(-1)[0])
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"Finite-difference step for {kind} must be positive and finite")
        return step

    def _finite_difference_station(self, component, evaluate):
        """Central-difference one station parameter through the full forward model.

        Parameters
        ----------
        component : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``.
        evaluate : callable
            No-argument evaluator returning the noiseless observable rows.

        Returns
        -------
        numpy.ndarray or None
            ``(rows, dimension)`` block, or None when the component does not
            belong to this model's station.

        Notes
        -----
        Used only by ``partials_method="fd"``, as the reference the analytical
        rows are checked against: the geometric models differentiate through
        :meth:`_compute_h_tilde_station`, the light-time models through
        :meth:`_rtlt_station_row` at the transmit and the receive epoch, and the
        zenith delays through :meth:`MediaCorrections.zenith_delay_partials`.
        The override is always cleared, including when an evaluation raises.
        """
        name, dimension = component[0], component[1]
        available = self._station_parameters() + self._media_parameters()
        parameter = next((p for p in available if p["name"] == name), None)
        if parameter is None:
            return None
        if parameter["values"] is None:
            raise self.UnsetPoleParameterError(
                f"{name} has no value to difference about; give it one, e.g. its "
                "PCK value, to use partials_method='fd'")
        step = self._station_eps(parameter["kind"])
        columns = []
        try:
            for index in range(dimension):
                outputs = []
                for sign in (1, -1):
                    values = parameter["values"].copy()
                    values[index] += sign * step
                    self._station_override = (name, values)
                    outputs.append(np.asarray(evaluate(), dtype=float).reshape(-1))
                columns.append(self._fd_difference(*outputs) / (2 * step))
        finally:
            self._station_override = None
        # Back to the declared units, matching the analytical path.
        return np.column_stack(columns) * parameter["scale"]

    def _compute_h_tilde_station(self, component, epoch, frame,
                                 h_position, h_velocity=None, station=None):
        r"""Partial of this observable with respect to a station parameter.

        Parameters
        ----------
        component : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``.
        epoch : float or array_like
            TDB seconds past J2000: one epoch shared by every row, or one
            per row.
        frame : Frame or str
            Axes ``h_position`` and ``h_velocity`` are expressed in.
        h_position : array_like
            This model's :math:`\partial h/\partial r_{target}`, shape (3,) for
            a scalar observable or (rows, 3) for a vector one.
        h_velocity : array_like, optional
            :math:`\partial h/\partial v_{target}`, same shape. Omit for an
            observable that does not depend on the relative velocity.
            Defaults to ``None``.
        station : GroundStation, optional
            Defaults to :meth:`_observer_station`.

        Returns
        -------
        numpy.ndarray
            ``(rows, dimension)`` block, zeros when the component does not move
            this station.

        Raises
        ------
        ValueError
            If ``epoch`` holds several epochs but not one per row.

        Notes
        -----
        The observable sees the station only through
        :math:`\rho = r_{target} - r_{station}`, so
        :math:`\partial h/\partial r_{station} = -\partial h/\partial r_{target}`
        and the chain rule gives the whole block from partials the model has
        already computed:

        .. math::
            \frac{\partial h}{\partial p} =
            -\left[\frac{\partial h}{\partial r},
                   \frac{\partial h}{\partial v}\right]
            \frac{\partial X_{station}}{\partial p}

        No finite differencing is involved, so no step size has to be chosen
        and the block is exact to the model's own linearization. Any model that
        can supply its position and velocity partials gets station estimation
        for free; it never needs station-specific code of its own.
        """
        dimension = component[1]
        h_position = np.atleast_2d(np.asarray(h_position, dtype=float))
        rows = h_position.shape[0]
        if h_velocity is None:
            h_velocity = np.zeros_like(h_position)
        else:
            h_velocity = np.atleast_2d(np.asarray(h_velocity, dtype=float)).reshape(rows, 3)
        values = np.asarray(getattr(getattr(epoch, "times", None), "values", epoch),
                            dtype=float).reshape(-1)
        # The station Jacobian is epoch dependent, so a multi-row block needs
        # its own epoch per row; a single epoch is shared by every row.
        if values.size == 1:
            values = np.repeat(values, rows)
        elif values.size != rows:
            raise ValueError(f"Expected {rows} epochs for a {rows}-row block, got {values.size}")
        block = np.zeros((rows, dimension))
        for index in range(rows):
            jacobian = self._station_state_partials(
                component, values[index], frame, station)
            if jacobian is None:
                return block
            block[index] = -np.concatenate(
                [h_position[index], h_velocity[index]]) @ jacobian
        return block

    # endregion Station parameters #
    # -----------------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """
        The name of the model.
        """
        return self._name

    @property
    def instrument(self) -> Instrument:
        """
        The instrument.
        """
        return self._instrument

    @property
    def default_dataset_name(self) -> str:
        """
        Dataset name used when the caller does not supply one.

        ``"<instrument>_<measurement_type>"``, which is the convention the
        radiometric datasets already read as — ``DSS-14_RangeIdeal`` — and now
        the optical ones too. For a camera-based model the instrument *is* the
        camera, so a centroiding dataset comes out as
        ``NAC_CentroidingIdeal``.

        Note that an :class:`~scarabaeus.AngularIdeal` observer is the
        spacecraft, not the camera (its name is resolved through SPICE), so a
        run that images with several cameras should pass ``dataset_name``
        explicitly to tell them apart.
        """
        instrument = getattr(self, "_instrument", None)
        instrument_name = getattr(instrument, "name", None) or "measurement"
        return f"{instrument_name}_{self._measurement_type}"

    @property
    def component_labels(self) -> tuple[str, str]:
        """
        Names of the two components of a vector observable.

        A two-component measurement is split into one scalar dataset per
        component, and these are the suffixes those datasets carry — so they
        end up as the residual keys in ``solution.json``. Each model sets them
        in its ``__init__`` beside ``_measurement_type``:
        :class:`~scarabaeus.AngularIdeal` measures ``("ra", "dec")``,
        :class:`~scarabaeus.CentroidingIdeal` ``("sample", "line")``.
        """
        return self._component_labels

    @property
    def sigma(self) -> ArrayWUnits:
        """
        The standard deviation of the measurement model.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value: ArrayWUnits) -> None:
        """
        Set the measurement standard deviation.

        Allows the sigma to be revised after the model is built, e.g. when an
        outlier rejection pass refits it from the residuals, without having to
        reconstruct the model and re-attach its media corrections.
        """
        self._sigma = value

    @property
    def epoch_sec(self) -> EpochArray:
        """Observation epochs expressed as seconds of day."""
        return self._epoch_sec

    @property
    def computed_measurements_dict(self) -> dict:
        """
        Auxiliary TRK-2-34 tracking-data dictionary needed for the generation
        of the computed measurements of a real radiometric model. See the
        ``computed_measurements_dict`` parameter of
        :class:`~scarabaeus.DopplerReal` and
        :class:`~scarabaeus.SequentialRangingReal` for the keys each requires.
        """
        return self._computed_measurements_dict

    @computed_measurements_dict.setter
    def computed_measurements_dict(self, input_dict):
        """
        Setter for auxiliary dictionary needed for the generation of the computed measurements
        """
        self._computed_measurements_dict = input_dict

    @property
    def media_corrections(self):
        """Tropospheric and ionospheric path-delay model, or ``None`` if not set."""
        return self._media_corrections

    @media_corrections.setter
    def media_corrections(self, value) -> None:
        # Looked up at call time: MediaCorrections subclasses this class, so it
        # cannot be imported at the top of this module.
        if not isinstance(value, scb.MediaCorrections):
            raise TypeError("media_corrections must be a MediaCorrections object")
        self._media_corrections = value

    @property
    def dsn_bands(self) -> types.MappingProxyType:
        """
        DSN frequency-band parameters of this model, a read-only mapping from
        band letter to :class:`DSNBand`. Starts as :attr:`DSN_BANDS`.

        The ``band`` of a real model is looked up here, for the reference
        frequency that converts the transponder delay to seconds and the range
        unit of a tracking file without exciter scalars, and so is the band of
        the ``default_*_aux`` builders, which take its turnaround ratio,
        exciter scalars and uplink frequency.

        Set a whole new mapping to change it; its keys are upper-cased.

        Examples
        --------
        .. code-block:: python

            X = model.dsn_bands["X"]
            model.dsn_bands = {
                **model.dsn_bands,
                "X": dataclasses.replace(X, uplink_frequency=scb.ArrayWUnits(7.18e9, 1 / sec)),
            }
        """
        return self._dsn_bands

    @dsn_bands.setter
    def dsn_bands(self, value: Mapping) -> None:
        if not isinstance(value, Mapping) or not value:
            raise Measurement.RadiometricDefaultError(
                f"'dsn_bands' must be a non-empty mapping of band letter to "
                f"Measurement.DSNBand, got {value!r}."
            )
        bands = {}
        for key, band in value.items():
            if not isinstance(band, Measurement.DSNBand):
                raise Measurement.RadiometricDefaultError(
                    f"'dsn_bands' entry {key!r} must be a Measurement.DSNBand, "
                    f"got {type(band).__name__}."
                )
            bands[str(key).strip().upper()] = band
        self._dsn_bands = types.MappingProxyType(bands)

    @property
    def ramp_rate(self) -> ArrayWUnits:
        """
        Uplink frequency ramp rate [Hz/s] of the ramp table the
        ``default_*_aux`` builders simulate. Starts as :attr:`RAMP_RATE`.

        Set an ArrayWUnits, converted to 1/s^2, or a bare number in Hz/s.
        """
        return ArrayWUnits(self._ramp_rate, 1 / sec**2)

    @ramp_rate.setter
    def ramp_rate(self, value: ArrayWUnits | float) -> None:
        self._ramp_rate = Measurement._radiometric_default("ramp_rate", value, 1 / sec**2)

    @property
    def ramp_margin(self) -> ArrayWUnits:
        """
        Span [s] a simulated ramp table extends before the first and after the
        last observation. Starts as :attr:`RAMP_MARGIN`; see
        :meth:`_default_aux_grids` for why it is needed.

        Set an ArrayWUnits time, or a bare, non-negative number of seconds.
        """
        return ArrayWUnits(self._ramp_margin, sec)

    @ramp_margin.setter
    def ramp_margin(self, value: ArrayWUnits | float) -> None:
        self._ramp_margin = Measurement._radiometric_default("ramp_margin", value, sec, minimum=0.0)

    @property
    def ramp_entries(self) -> int:
        """
        Number of entries of a simulated ramp table, at least 2. Starts as
        :attr:`RAMP_ENTRIES`.
        """
        return self._ramp_entries

    @ramp_entries.setter
    def ramp_entries(self, value: int) -> None:
        self._ramp_entries = Measurement._radiometric_default(
            "ramp_entries", value, integer=True, minimum=2
        )

    @property
    def ramp_type(self) -> int:
        """
        SFDU 9 ``ramp_type`` code of the segments of a simulated ramp table.
        Starts as :attr:`RAMP_TYPE`.
        """
        return self._ramp_type

    @ramp_type.setter
    def ramp_type(self, value: int) -> None:
        self._ramp_type = Measurement._radiometric_default("ramp_type", value, integer=True, minimum=0)

    @property
    def transponder_delay(self) -> float:
        """
        Spacecraft transponder delay [RU] written into simulated auxiliary
        data by the ``default_*_aux`` builders. Starts as
        :attr:`TRANSPONDER_DELAY`.

        Only simulated data: a real pass takes the ``sc_transponder_delay_RU``
        given to ``observed_measurements``, or else the delay in its tracking
        file.
        """
        return self._transponder_delay

    @transponder_delay.setter
    def transponder_delay(self, value: float) -> None:
        self._transponder_delay = Measurement._radiometric_default(
            "transponder_delay", value, minimum=0.0
        )

    @property
    def rcv_time_tag_delay(self) -> ArrayWUnits:
        """
        Receive time-tag delay [s] written into simulated auxiliary data.
        Starts as :attr:`RCV_TIME_TAG_DELAY`.

        Set an ArrayWUnits time, or a bare number of seconds.
        """
        return ArrayWUnits(self._rcv_time_tag_delay, sec)

    @rcv_time_tag_delay.setter
    def rcv_time_tag_delay(self, value: ArrayWUnits | float) -> None:
        self._rcv_time_tag_delay = Measurement._radiometric_default("rcv_time_tag_delay", value, sec)

    @property
    def ul_station_cal(self) -> float:
        """
        Uplink station calibration [RU] written into simulated auxiliary data.
        Starts as :attr:`UL_STATION_CAL`.
        """
        return self._ul_station_cal

    @ul_station_cal.setter
    def ul_station_cal(self, value: float) -> None:
        self._ul_station_cal = Measurement._radiometric_default("ul_station_cal", value)

    @property
    def dl_station_cal(self) -> float:
        """
        Downlink station calibration [RU] written into simulated auxiliary
        data. Starts as :attr:`DL_STATION_CAL`.
        """
        return self._dl_station_cal

    @dl_station_cal.setter
    def dl_station_cal(self, value: float) -> None:
        self._dl_station_cal = Measurement._radiometric_default("dl_station_cal", value)

    @property
    def solar_corona_coefficient(self) -> float:
        """
        Coefficient of the solar-corona group delay [us GHz^2 m^2] of a real
        model built with ``solar_corona=True``. Starts as
        :attr:`MediaCorrections.SOLAR_CORONA_DELAY_COEFF
        <scarabaeus.MediaCorrections.SOLAR_CORONA_DELAY_COEFF>`; see
        :meth:`MediaCorrections._compute_SC_lt
        <scarabaeus.MediaCorrections._compute_SC_lt>`.

        A fixed change of the delay model. To estimate a correction from the
        data, estimate ``solar_corona_scale`` instead.
        """
        return self._solar_corona_coefficient

    @solar_corona_coefficient.setter
    def solar_corona_coefficient(self, value: float) -> None:
        self._solar_corona_coefficient = Measurement._radiometric_default(
            "solar_corona_coefficient", value, minimum=0.0
        )

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def update_reference_state(self, state_vector: StateArray):
        """
        Attach the estimated state the model should read its parameters from.

        Call this once per iteration, before
        :meth:`~scarabaeus.Measurement.generate_measurement_dataset`. The model
        then pulls the entries belonging to this instrument, matched on its
        SPICE ID:

        * ``meas_bias_ideal_*``, as an
          :class:`~scarabaeus.units.ArrayWUnits`.
        * ``gs_delta_location_ECEF_*``, as an
          :class:`~scarabaeus.timeAndFrame.ArrayWFrame`.

        Parameters
        ----------
        state_vector : StateArray
            The current estimated state. Its layout is also recorded, so that
            the partials can build a column for every estimated parameter
            beyond position and velocity.
        """
        self._reference_state_vector = state_vector
        # Expose the full state layout so _partials() can build an H column for
        # every estimated parameter (e.g. gm, Cnm, pole) beyond pos/vel.
        if state_vector is not None and self._sequence_definition is None:
            self._state_definition = state_vector.state

    # ---------------------------------#
    #  Estimated-state / STM plumbing  #
    # ---------------------------------#
    # Shared by every model that has to build an H row against an arbitrary
    # state layout, or that maps a deviation between the two endpoints of an
    # observable through the STM (the analytical light-time partials).

    def set_trajectory(self, trajectory):
        """
        Attach the reference trajectory supplying the STMs.

        Read only by the analytical light-time partials, which map a deviation
        between the endpoints of an observable (see :meth:`_short_arc_stm`).

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory carrying the STMs. Must span every epoch the model
            evaluates.
        """
        self._trajectory = trajectory

    def _state_entries(self) -> list:
        """
        State entries to read the estimated parameter values from.

        Prefers the reference state vector set by :meth:`update_reference_state`,
        which carries the current iterate, and falls back to a state definition
        passed straight to a model constructor. Both hold the same
        ``(name, dim, flag, type, body, value)`` tuples.

        Returns
        -------
        list of tuple
            The ``(name, dim, flag, type, body, value)`` entries, or an empty list
            when the model has neither a reference state nor a state definition.
        """
        reference = getattr(self, "_reference_state_vector", None)
        if reference is not None:
            return reference.state
        return getattr(self, "_state_definition", None) or []

    def _state_layout(self):
        """
        Return ``(n_total, components)`` describing the estimated state layout.

        Returns
        -------
        n_total : int
            Total width of the H row.
        components : list of tuple
            ``(name, dim, body, start_column)`` in state order.
        """
        state_definition = getattr(self, "_state_definition", None)
        if state_definition is None:
            # Legacy fallback: position + velocity of the observing body.
            return 6, [("position", 3, None, 0), ("velocity", 3, None, 3)]

        components = []
        start = 0
        for comp_def in state_definition:
            name = comp_def[0]
            dim = int(comp_def[1])
            body = comp_def[4] if len(comp_def) > 4 else None
            components.append((name, dim, body, start))
            start += dim
        return start, components

    def _stm_pair_mapping(self, et_early: float, et_late: float) -> np.ndarray:
        """
        Return :math:`\\Phi(t_{k-1}, t_k) = \\Phi(t_{k-1},t_0)\\Phi(t_k,t_0)^{-1}`.

        Maps a deviation at the later epoch onto the earlier one, as the
        analytical light-time partials need between the endpoints of an
        observable (see :meth:`_short_arc_stm`).

        Parameters
        ----------
        et_early, et_late : float
            TDB epochs, seconds past J2000, of the earlier and later state.

        Returns
        -------
        numpy.ndarray
            ``(n, n)`` state transition matrix from the later epoch to the
            earlier one.

        Raises
        ------
        MissingTrajectoryError
            If no trajectory has been attached to the model.
        """
        if self._trajectory is None:
            raise Measurement.MissingTrajectoryError(
                f"{type(self).__name__} needs the reference trajectory STMs to map a "
                "deviation between two epochs. Attach one with set_trajectory()."
            )

        phi_early = np.asarray(
            self._trajectory.get_STM_sequence(epoch=et_early), dtype=float
        )
        phi_late = np.asarray(
            self._trajectory.get_STM_sequence(epoch=et_late), dtype=float
        )
        n_dyn = int(np.sqrt(phi_early.size))
        phi_early = phi_early.reshape(n_dyn, n_dyn)
        phi_late = phi_late.reshape(n_dyn, n_dyn)

        # Solved rather than inverted, for conditioning.
        return np.linalg.solve(phi_late.T, phi_early.T).T

    # Partials methods a real radiometric model can compute its H-tilde with.
    # Each model lists the ones it implements; DopplerReal adds "fd_endpoint",
    # which is meaningful only for a count-differenced observable.
    PARTIALS_METHODS = ("fd", "analytical")

    # Space-time frames the real models can solve the light time in, see
    # _compute_precise_RTLT and _compute_geocentric_RTLT.
    LIGHT_TIME_FRAMES = ("barycentric", "geocentric")

    LIGHT_TIME_SETTINGS = {"tolerance": 1e-12, "max_iterations": 10, "et_tai_iterations": 2}
    """Default settings of the round-trip light-time solution of the real
    measurement models, which take a dictionary with any of these keys as
    ``light_time_settings``. They apply to both solvers,
    :meth:`_compute_precise_RTLT` and :meth:`_compute_geocentric_RTLT`.

    - ``"tolerance"`` [s]: each leg (t3 to t2, then t2 to t1) is corrected
      with Moyer 2000 eq. 8-72 until the correction is no larger than this.
      1e-12 s is 0.3 mm of one-way path. The correction cannot fall much below
      the float spacing of the one-way light time: 9.1e-13 s from 4096 s
      (8.2 AU), 1.8e-12 s from 8192 s (16 AU). A tolerance under that floor
      makes every leg run to ``"max_iterations"``, so raise it for distant
      spacecraft.
    - ``"max_iterations"``: iteration cap of each leg. A leg that reaches it
      keeps its last estimate and does not raise.
    - ``"et_tai_iterations"``: evaluations of the ET - TAI series (Moyer 2000
      eq. 2-26) when the reception time is converted from TAI to ET. Each one
      is made at the ET the previous one gave. The first starts from the
      approximate series, eq. 2-27.

    .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed Values of
        Deep Space Network Data Types for Navigation. JPL Publication 00-7,
        Monograph 2, Deep Space Communications and Navigation Series,
        Sec. 8.3.6 and Eq. 2-26.
    """

    @staticmethod
    def _resolve_light_time_frame(light_time_frame: str) -> str:
        """
        Check the ``light_time_frame`` a real model was built with.

        ``"barycentric"`` solves the light time in the Solar-System barycentric
        frame (Moyer 2000, Sec. 8.3.6) and is valid for any spacecraft.
        ``"geocentric"`` solves it in the local geocentric frame (Moyer 2000,
        Sec. 8.3.1), with the Earth as the only relativistic body; it is meant
        for spacecraft near the Earth, see :meth:`_compute_geocentric_RTLT`.

        Parameters
        ----------
        light_time_frame : str
            One of :attr:`LIGHT_TIME_FRAMES`, case-insensitive.

        Returns
        -------
        str
            The frame name in lower case.

        Raises
        ------
        Measurement.LightTimeFrameError
            If the frame is not one of :attr:`LIGHT_TIME_FRAMES`.
        """
        frame = str(light_time_frame).strip().lower()
        if frame not in Measurement.LIGHT_TIME_FRAMES:
            raise Measurement.LightTimeFrameError(
                f"Unknown light_time_frame {light_time_frame!r}. Choose one of "
                f"{', '.join(Measurement.LIGHT_TIME_FRAMES)}."
            )
        return frame

    @staticmethod
    def _resolve_light_time_settings(light_time_settings: "dict | None") -> dict:
        """
        Complete and check the ``light_time_settings`` of a real model.

        Parameters
        ----------
        light_time_settings : dict or None
            Any of the keys of :attr:`LIGHT_TIME_SETTINGS`: ``"tolerance"``
            [s], a positive float, and ``"max_iterations"`` and
            ``"et_tai_iterations"``, positive integers. Missing keys take the
            default; None takes all defaults.

        Returns
        -------
        dict
            A new dictionary with every key of :attr:`LIGHT_TIME_SETTINGS`.

        Raises
        ------
        Measurement.LightTimeSettingsError
            If a key is unknown or a value is out of range.
        """
        defaults = Measurement.LIGHT_TIME_SETTINGS
        settings = dict(defaults)
        if light_time_settings is None:
            return settings
        unknown = set(light_time_settings) - set(defaults)
        if unknown:
            raise Measurement.LightTimeSettingsError(
                f"Unknown light_time_settings keys {sorted(unknown)}. Valid keys: "
                f"{', '.join(defaults)}."
            )
        settings.update(light_time_settings)
        tol = settings["tolerance"]
        if (
            isinstance(tol, bool)
            or not isinstance(tol, (int, float, np.integer, np.floating))
            or not 0 < tol < np.inf
        ):
            raise Measurement.LightTimeSettingsError(
                f"light_time_settings['tolerance'] must be a positive number of "
                f"seconds, got {tol!r}."
            )
        for key in ("max_iterations", "et_tai_iterations"):
            n = settings[key]
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
                raise Measurement.LightTimeSettingsError(
                    f"light_time_settings[{key!r}] must be a positive integer, got {n!r}."
                )
        return settings

    def _rtlt_solver(self):
        """Pick the round-trip light-time solver for this model's ``light_time_frame``.

        Returns
        -------
        callable
            :meth:`_compute_geocentric_RTLT` when ``light_time_frame`` is
            ``"geocentric"``, otherwise :meth:`_compute_precise_RTLT`, the
            barycentric solution and the default. Both are staticmethods with the
            same signature and return layout, so callers invoke the result directly.
        """
        if getattr(self, "_light_time_frame", "barycentric") == "geocentric":
            return Measurement._compute_geocentric_RTLT
        return Measurement._compute_precise_RTLT

    def _light_time_velocity(self, body_name: str, epoch: float, frame_name: str) -> np.ndarray:
        """
        Velocity of a light-time participant in the frame the light time is solved in.

        The velocity is relative to the SSB for ``"barycentric"`` and to the Earth
        for ``"geocentric"``. It is the velocity behind the :math:`\\dot{p}` terms
        of Moyer (2000) Eq. 8-60 that the analytical partials differentiate, so it
        must be the one the light-time solution uses, never a station-relative
        velocity.

        Parameters
        ----------
        body_name : str
            SPICE name of the spacecraft or ground station.
        epoch : float
            Epoch of the light-time event [ET s], ``t2`` for the spacecraft and
            ``t1`` or ``t3`` for the station.
        frame_name : str
            Axes to express the velocity in.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Velocity [km/s].

        References
        ----------
        .. [1] Moyer, T. D. (2000). Formulation for Observed and Computed Values of
            Deep Space Network Data Types for Navigation. JPL Publication 00-7.
        """
        origin = (
            "EARTH"
            if getattr(self, "_light_time_frame", "barycentric") == "geocentric"
            else "SSB"
        )
        return SpiceManager.get_state(body_name, epoch, frame_name, origin)[3:6].values

    def _station_clock_gradient(self, epoch: float, frame_name: str) -> np.ndarray:
        """
        Gradient of ET - TAI at the station with respect to the station position.

        This is the :math:`\\dot{\\mathbf{R}}_E \\cdot \\mathbf{r}_A / c^2` term of
        Moyer (2000) Eq. 2-23, with :math:`\\dot{\\mathbf{R}}_E` the Earth's SSB
        velocity and :math:`\\mathbf{r}_A` the geocentric station position:

        .. math::

            \\nabla_{\\mathbf{r}_A} (ET - TAI) = \\frac{\\dot{\\mathbf{R}}_E}{c^2}

        The barycentric light time converts the reception and transmission epochs
        through Eq. 2-23 with the station where the state puts it, so a station
        offset :math:`\\delta\\mathbf{r}` also moves the round trip by
        :math:`(\\nabla(t_1) - \\nabla(t_3)) \\cdot \\delta\\mathbf{r}`. Earth rotation
        between :math:`t_1` and :math:`t_3` keeps the two from cancelling, which
        leaves about 1e-5 of the station partial at Mars range.

        Parameters
        ----------
        epoch : float
            Reception or transmission epoch at the station [ET s].
        frame_name : str
            Axes to express the gradient in.

        Returns
        -------
        numpy.ndarray, shape (3,)
            The gradient [s/km]; zeros in the geocentric frame, where ET - TAI is
            a constant.

        References
        ----------
        .. [1] Moyer, T. D. (2000). Formulation for Observed and Computed Values of
            Deep Space Network Data Types for Navigation. JPL Publication 00-7.
        """
        if getattr(self, "_light_time_frame", "barycentric") == "geocentric":
            return np.zeros(3)
        c = float(constants.c.values)
        return SpiceManager.get_state("EARTH", epoch, frame_name, "SSB")[3:6].values / c**2

    def _gamma_position_shift(self, body_name: str, epoch: float, frame_name: str) -> np.ndarray:
        """
        Derivative of a light-time participant's position with respect to PPN :math:`\\gamma`.

        The shift comes in through the :math:`\\gamma U_E / c^2` potential scale of
        Moyer (2000) Eq. 4-10, with :math:`U_E = \\mu_\\odot / |\\mathbf{r}_{\\odot E}|`
        the Sun's potential at the Earth.

        The barycentric solution moves the station's geocentric position
        :math:`\\mathbf{r}_{GC}` into SSB space-time with
        :math:`(1 - L_C - \\gamma U_E / c^2)`, so for the station

        .. math::

            \\frac{\\partial \\mathbf{r}}{\\partial \\gamma} = -\\frac{U_E}{c^2}\\,\\mathbf{r}_{GC}

        and the spacecraft, read from a barycentric ephemeris, has none. The
        geocentric solution moves the spacecraft instead, with the inverse
        transform, so there
        :math:`\\partial \\mathbf{r} / \\partial \\gamma = +(U_E / c^2)\\,\\mathbf{r}_{SC,E}`
        and the station has none. Left out, the station term is 3e-5 of the range
        partial and 2% of the Doppler one, where Earth rotation makes it change
        faster than the Shapiro delay.

        Parameters
        ----------
        body_name : str
            SPICE name of the participant: the spacecraft or the model's ground
            station (its instrument).
        epoch : float
            Epoch of the light-time event [ET s].
        frame_name : str
            Axes to express the derivative in.

        Returns
        -------
        numpy.ndarray, shape (3,)
            :math:`\\partial \\mathbf{r} / \\partial \\gamma` [km], zeros for the
            participant the active light-time frame does not transform.

        References
        ----------
        .. [1] Moyer, T. D. (2000). Formulation for Observed and Computed Values of
            Deep Space Network Data Types for Navigation. JPL Publication 00-7.
        """
        geocentric = getattr(self, "_light_time_frame", "barycentric") == "geocentric"
        if (body_name == self.instrument.name) == geocentric:
            return np.zeros(3)
        c = float(constants.c.values)
        sun_to_earth = SpiceManager.get_state("EARTH", epoch, frame_name, "SUN")[0:3].values
        u_e = float(constants.SUN.GM.values) / float(np.linalg.norm(sun_to_earth)) / c**2
        r_geo = SpiceManager.get_state(body_name, epoch, frame_name, "EARTH")[0:3].values
        return u_e * r_geo if geocentric else -u_e * r_geo

    @staticmethod
    def _resolve_partials_method(partials_method, supported):
        """
        Settle the partials method of a real radiometric model.

        ``"analytical"``
            The Moyer partials, differentiating the same light-time solution the
            forward model produced and reusing its cached endpoints. Two orders
            of magnitude cheaper than ``"fd"`` at the same state definition, and
            the default.
        ``"fd"``
            Finite differences through the real forward model. Slow -- one full
            forward run per state component -- but it differentiates exactly the
            observable being fitted, which makes it the reference the analytical
            partials are judged against.

        Parameters
        ----------
        partials_method : str or None
            The requested method, or ``None`` when the caller did not say.
        supported : tuple of str
            The methods the model implements, its :attr:`PARTIALS_METHODS`.

        Returns
        -------
        str
            One of ``supported``.

        Raises
        ------
        ValueError
            If the model does not implement the method.
        """
        if partials_method is None:
            return "analytical"

        method = str(partials_method).strip().lower()
        # Rejected here rather than at the first compute_partials call, which
        # may come long after the model was built.
        if method not in supported:
            raise ValueError(
                f"Unknown partials_method {partials_method!r} for this model. "
                f"Choose one of {', '.join(supported)}."
            )
        return method

    # ---------------------------------------------------------------------#
    # region   Analytical (Moyer) light-time partial helpers              #
    # ---------------------------------------------------------------------#
    # Shared machinery for the analytical partials of the real radiometric
    # models. These are pure sensitivity calculations: they take a geometry
    # already solved by the forward model and return its derivative, so they
    # never re-solve a light time themselves.

    def _short_arc_stm(self, t_from: float, t_to: float) -> np.ndarray | None:
        """
        Return :math:`\\Phi(t_{from}, t_{to})`, or ``None`` when unavailable.

        The analytical light-time partials need to map a state deviation between
        the two endpoints of an observable. That is exactly what
        :meth:`_stm_pair_mapping` does, so this delegates to it rather than
        composing the STMs a second time, and takes only the dynamics 6x6 block
        the geometric partials use.

        The difference from :meth:`_stm_pair_mapping` is the failure mode, and it
        is deliberate. Here a missing trajectory is not an error: the caller then
        falls back to the free-flight form :math:`\\Phi_{rv} = \\Delta t \\cdot I`,
        which is an accurate mapping across a single count interval and lets the
        partials be evaluated with no trajectory attached at all. Returning
        ``None`` is how that fallback is signalled.

        Parameters
        ----------
        t_from, t_to : float
            TDB epochs, seconds past J2000. The returned matrix maps a deviation
            at ``t_to`` onto ``t_from``.

        Returns
        -------
        numpy.ndarray or None
            The ``(6, 6)`` dynamics block, or ``None`` when no trajectory is
            attached or its STMs do not span the requested epochs.
        """
        if getattr(self, "_trajectory", None) is None:
            return None
        try:
            return np.asarray(self._stm_pair_mapping(t_from, t_to))[0:6, 0:6]
        except Exception:
            # STMs absent or not spanning these epochs: let the caller use the
            # free-flight mapping instead of failing the whole partial.
            return None

    @staticmethod
    def rtlt_grad_pos_exact(
        n_hat_dn: np.ndarray,
        n_hat_up: np.ndarray,
        v_sc: np.ndarray,
        v_gs: np.ndarray,
        c: float,
    ) -> np.ndarray:
        """
        Exact round-trip light-time gradient with respect to the spacecraft position.

        Chains Moyer (2000) Eqs. 12-12 (downleg) and 12-17 (upleg) through the
        reflection time :math:`t_2`, holding the reception time :math:`t_3` fixed,
        so that :math:`\\rho = t_3 - t_1` and
        :math:`\\partial \\rho / \\partial \\mathbf{r}_{SC} = -\\partial t_1 / \\partial \\mathbf{r}_{SC}`
        [s/km]. With the Eq. 8-59/8-60 definitions, the spacecraft transmits on
        the downleg and the station on the upleg:

        .. math::

            \\dot{p}_{23} = \\hat{\\mathbf{r}}_{23} \\cdot \\mathbf{v}_{SC} / c
                = -\\hat{\\mathbf{n}}_{dn} \\cdot \\mathbf{v}_{SC} / c, \\qquad
            \\dot{p}_{12} = \\hat{\\mathbf{r}}_{12} \\cdot \\mathbf{v}_{GS} / c
                = \\hat{\\mathbf{n}}_{up} \\cdot \\mathbf{v}_{GS} / c

            \\dot{r}_{12} = \\hat{\\mathbf{n}}_{up} \\cdot (\\mathbf{v}_{SC} - \\mathbf{v}_{GS}) / c,
            \\qquad \\dot{r}_{12} + \\dot{p}_{12} = \\hat{\\mathbf{n}}_{up} \\cdot \\mathbf{v}_{SC} / c

        where the station terms cancel in the last sum. The gradient is then

        .. math::

            \\frac{\\partial t_2}{\\partial \\mathbf{r}_{SC}}
                = -\\frac{\\hat{\\mathbf{n}}_{dn}}{c\\,(1 - \\dot{p}_{23})}, \\qquad
            \\frac{\\partial \\rho}{\\partial \\mathbf{r}_{SC}}
                = -\\frac{(1 - \\dot{r}_{12} - \\dot{p}_{12})\\,
                  \\partial t_2 / \\partial \\mathbf{r}_{SC} - \\hat{\\mathbf{n}}_{up} / c}
                  {1 - \\dot{p}_{12}}

        Parameters
        ----------
        n_hat_dn : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the downleg,
            :math:`\\hat{\\mathbf{n}}_{dn} = (\\mathbf{r}_{SC}(t_2) - \\mathbf{r}_{GS}(t_3)) / \\rho_{dn}`.
        n_hat_up : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the upleg,
            :math:`\\hat{\\mathbf{n}}_{up} = (\\mathbf{r}_{SC}(t_2) - \\mathbf{r}_{GS}(t_1)) / \\rho_{up}`.
        v_sc : numpy.ndarray, shape (3,)
            Spacecraft velocity at :math:`t_2` [km/s], in the space-time frame the
            light time is solved in (see :meth:`_light_time_velocity`).
        v_gs : numpy.ndarray, shape (3,)
            Ground station velocity at :math:`t_1` [km/s], in the same frame.
        c : float
            Speed of light [km/s].

        Returns
        -------
        numpy.ndarray, shape (3,)
            :math:`\\partial \\rho / \\partial \\mathbf{r}_{SC}` [s/km].

        References
        ----------
        .. [1] Moyer, T. D. (2000). Formulation for Observed and Computed Values of
            Deep Space Network Data Types for Navigation. JPL Publication 00-7.
        """
        # Downleg (Eq. 12-12): ṗ₂₃ = r̂₂₃ · v_SC = −n̂_dn · v_SC  (SC is transmitter)
        p_dot_23 = -float(np.dot(n_hat_dn, v_sc)) / c
        # ∂t2/∂r_SC = r̂₂₃ / [c(1 − ṗ₂₃)] = −n̂_dn / [c(1 − p_dot_23)]
        dt2_dr = -n_hat_dn / (c * (1.0 - p_dot_23))

        # Upleg (Eq. 12-17):
        #   ṗ₁₂ = r̂₁₂ · v_GS / c   (GS is transmitter on upleg)
        #   ṙ₁₂ + ṗ₁₂ = r̂₁₂ · v_SC / c  (numerator coefficient, GS terms cancel)
        p_dot_12_GS = float(np.dot(n_hat_up, v_gs)) / c   # denominator
        r_dot_12_SC = float(np.dot(n_hat_up, v_sc)) / c   # numerator coefficient

        # ∂t1/∂r_SC = [(1 − (ṙ₁₂+ṗ₁₂)/c) · ∂t2/∂r − r̂₁₂/c] / (1 − ṗ₁₂/c)
        dt1_dr = ((1.0 - r_dot_12_SC) * dt2_dr - n_hat_up / c) / (1.0 - p_dot_12_GS)

        return -dt1_dr

    @staticmethod
    def rtlt_leg_weights(
        n_hat_dn: np.ndarray,
        n_hat_up: np.ndarray,
        v_sc: np.ndarray,
        v_gs: np.ndarray,
        c: float,
    ) -> tuple:
        """
        Sensitivity of the round-trip light time to a delay inside each leg's solution.

        Uses the Moyer (2000) 12-12/12-17 chain of :meth:`rtlt_grad_pos_exact`. A
        delay :math:`\\delta\\tau_{23}` in the downleg equation moves the reflection
        time :math:`t_2`, and the upleg carries that change to :math:`t_1`; a delay
        :math:`\\delta\\tau_{12}` in the upleg equation moves :math:`t_1` only:

        .. math::

            \\frac{\\partial \\rho}{\\partial \\tau_{23}}
                = \\frac{1 - \\dot{r}_{12} - \\dot{p}_{12}}{(1 - \\dot{p}_{23})(1 - \\dot{p}_{12})},
            \\qquad
            \\frac{\\partial \\rho}{\\partial \\tau_{12}} = \\frac{1}{1 - \\dot{p}_{12}}

        with :math:`\\dot{p}` and :math:`\\dot{r}` as defined in
        :meth:`rtlt_grad_pos_exact`. Both are 1 to within the transmitters'
        velocities over c, about 1e-4 with heliocentric velocities. They apply to
        what the iteration solves for, the geometric range and the Shapiro delay,
        and not to the media, corona and electronic delays, which are added after
        it converges.

        Parameters
        ----------
        n_hat_dn : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the downleg.
        n_hat_up : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the upleg.
        v_sc : numpy.ndarray, shape (3,)
            Spacecraft velocity at :math:`t_2` [km/s].
        v_gs : numpy.ndarray, shape (3,)
            Ground station velocity at :math:`t_1` [km/s].
        c : float
            Speed of light [km/s].

        Returns
        -------
        w_dn : float
            :math:`\\partial \\rho / \\partial \\tau_{23}`, dimensionless.
        w_up : float
            :math:`\\partial \\rho / \\partial \\tau_{12}`, dimensionless.

        See Also
        --------
        rtlt_grad_pos_exact : definitions of the inputs.
        """
        p_dot_23 = -float(np.dot(n_hat_dn, v_sc)) / c
        p_dot_12_GS = float(np.dot(n_hat_up, v_gs)) / c
        r_dot_12_SC = float(np.dot(n_hat_up, v_sc)) / c
        w_up = 1.0 / (1.0 - p_dot_12_GS)
        w_dn = (1.0 - r_dot_12_SC) * w_up / (1.0 - p_dot_23)
        return w_dn, w_up

    @staticmethod
    def rtlt_grad_station_exact(
        n_hat_dn: np.ndarray,
        n_hat_up: np.ndarray,
        v_sc: np.ndarray,
        v_gs: np.ndarray,
        c: float,
    ) -> tuple:
        """
        Exact round-trip light-time gradients with respect to the station position.

        The station enters at reception :math:`t_3` and at transmission
        :math:`t_1`, through the same Moyer (2000) 12-12/12-17 chain as
        :meth:`rtlt_grad_pos_exact`. Displacing the receiver at :math:`t_3` moves
        the reflection time :math:`t_2` (Eq. 12-12, with the station as the
        receiver), and the upleg carries that change to :math:`t_1` through its own
        :math:`\\dot{p}_{12}` denominator (Eq. 12-17); displacing the transmitter at
        :math:`t_1` enters the upleg only:

        .. math::

            \\frac{\\partial \\rho}{\\partial \\mathbf{r}_{GS}(t_3)}
                = -\\frac{(1 - \\dot{r}_{12} - \\dot{p}_{12})\\,\\hat{\\mathbf{n}}_{dn}}
                  {c\\,(1 - \\dot{p}_{23})(1 - \\dot{p}_{12})},
            \\qquad
            \\frac{\\partial \\rho}{\\partial \\mathbf{r}_{GS}(t_1)}
                = -\\frac{\\hat{\\mathbf{n}}_{up}}{c\\,(1 - \\dot{p}_{12})}

        with :math:`\\dot{p}` and :math:`\\dot{r}` as defined in
        :meth:`rtlt_grad_pos_exact`: each leg's range over c weighted by
        :meth:`rtlt_leg_weights`. Their sum is ``-rtlt_grad_pos_exact(...)``: a
        common translation of station and spacecraft leaves the light time
        unchanged. Dropping the denominators, i.e. using
        :math:`-\\hat{\\mathbf{n}}/c` on each leg, is off by the transmitters'
        velocities over c, about 1e-4 relative with heliocentric velocities.

        Parameters
        ----------
        n_hat_dn : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the downleg.
        n_hat_up : numpy.ndarray, shape (3,)
            Unit vector GS to SC on the upleg.
        v_sc : numpy.ndarray, shape (3,)
            Spacecraft velocity at :math:`t_2` [km/s].
        v_gs : numpy.ndarray, shape (3,)
            Ground station velocity at :math:`t_1` [km/s].
        c : float
            Speed of light [km/s].

        Returns
        -------
        grad_t3 : numpy.ndarray, shape (3,)
            :math:`\\partial \\rho / \\partial \\mathbf{r}_{GS}(t_3)` [s/km], in the
            frame of the input vectors.
        grad_t1 : numpy.ndarray, shape (3,)
            :math:`\\partial \\rho / \\partial \\mathbf{r}_{GS}(t_1)` [s/km], in the
            same frame.

        See Also
        --------
        rtlt_grad_pos_exact : definitions of the inputs.
        """
        # Moving the station by δr along n̂ shortens that leg's range by n̂·δr.
        w_dn, w_up = Measurement.rtlt_leg_weights(n_hat_dn, n_hat_up, v_sc, v_gs, c)
        return -w_dn * n_hat_dn / c, -w_up * n_hat_up / c

    # endregion Analytical (Moyer) light-time partial helpers #
    # ---------------------------------------------------------------------#

    def _compute_visibility_mask(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        elevation_mask: ArrayWUnits | float = 10.0,
    ):
        """
        Mark the epochs at which the instrument can see the target.

        The base implementation duck-types on the instrument:

        * ``GroundStation`` (has ``station_visibility``): elevation-angle check
          against ``elevation_mask``.
        * ``Camera`` (has ``target_in_fov``): pinhole-projection FOV check;
          ``elevation_mask`` is not used.

        Subclasses may override this method if they need custom visibility logic.

        Parameters
        ----------
        target : Spacecraft
            The observed body.
        epoch_array : EpochArray
            Epochs to check.
        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation for a ground station, normalised by
            :meth:`resolve_elevation_mask`; a bare number is in degrees. Defaults
            to ``10.0``.

        Returns
        -------
        numpy.ndarray of bool or None
            True at each visible epoch, or None when the instrument offers no
            visibility check.
        """
        # Degrees, whichever way the caller expressed it.
        elevation_mask = Measurement.resolve_elevation_mask(elevation_mask)
        if hasattr(self.instrument, "station_visibility"):
            return self.instrument.station_visibility(
                target, epoch_array, elevation_mask=elevation_mask
            )
        if hasattr(self.instrument, "target_in_fov"):
            return self.instrument.target_in_fov(target, epoch_array)
        return None

    def write_observed_measurements(
        self,
        target: Spacecraft,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        file_name: str = "ideal_measurement",
        check_visibility: bool = False,
        elevation_mask: ArrayWUnits | float = 10.0,
        folder_path_override: str = None,
        fmt: str = "json",
    ) -> None:
        """
        Generates synthetic measurements and write them as a .json file. The input of this method
        encapsulate the ones needed for the "computed_meas" method in each measurement model class.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft for which the range measurement is to be computed.

        epoch_array : EpochArray, optional
            An array of epochs (times) at which the range measurements should be computed. If
            provided, overrides ``epoch_start``, ``epoch_end``, and ``tstep``.

        epoch_start : EpochArray, optional
            The starting epoch for the range measurement computations. Required if ``epoch_array``
            is not provided.

        epoch_end : EpochArray, optional
            The ending epoch for the range measurement computations. Required if ``epoch_array``
            is not provided.

        tstep : float, optional
            The time step, in seconds, between consecutive range measurements. If `epoch_array`
            is not provided. Defaults to ``1`` second.

        frame : Frame , optional
            The reference frame in which the range computation is performed. Defaults to ``None``.

        noisy : bool , optional
            Whether to add noise to the computed range measurement. Defaults to ``False``.

        file_name : str, optional
            The filename of the JSON in which the measurement is saved, Defaults to ``'ideal_measurement'``.

        check_visibility : bool, optional
            When ``True``, epochs where the instrument cannot observe the target are removed
            before generating measurements. The specific check is instrument-dependent:
            ``GroundStation`` uses an elevation-angle test; ``Camera`` uses a FOV projection
            test. Defaults to ``False``.

        elevation_mask : ArrayWUnits or float, optional
            Minimum elevation angle used by the ``GroundStation`` visibility check. An
            :class:`~scarabaeus.units.ArrayWUnits` is converted from its own units; a bare
            number is read as degrees (see :meth:`resolve_elevation_mask`, which refuses a
            bare one that could be radians). Ignored for ``Camera`` instruments. Only relevant when
            ``check_visibility=True``. Defaults to ``10.0`` degrees.

        folder_path_override : str, optional
            If provided, the JSON file is saved directly to this path instead of the default
            ``data/measurements/radiometric`` or ``data/measurements/optical`` folder. The
            directory is created automatically if it does not exist. Defaults to ``None``.
        fmt : {"json", "pickle"}, optional
            On-disk format; selects the ``.json`` or ``.pkl`` extension appended to
            ``file_name``. Read back transparently by :meth:`observed_measurements`.
            Defaults to ``"json"``.

        Returns
        -------
        None
        """
        # Same two-form contract as computed_measurements: the three interval
        # parameters were forwarded below but then bypassed by the direct
        # epoch_array uses further down, so the documented form raised.
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        ## Apply instrument-appropriate visibility mask
        if check_visibility:
            # Resolved here for the warning below, which reports degrees rather
            # than an ArrayWUnits repr. The caller's value is passed on, not the
            # float: a resolved 0.5 deg would read as an ambiguous bare mask.
            mask_deg = Measurement.resolve_elevation_mask(elevation_mask)
            mask = self._compute_visibility_mask(target, epoch_array, elevation_mask)
            if mask is not None:
                epoch_array = epoch_array[mask]
                if epoch_array.size == 0:
                    # A station that never sees the target over this span is an
                    # ordinary outcome, not a failure: a script generating data
                    # for several instruments should skip this one and carry on
                    # rather than lose the whole run. No file is written, since
                    # an empty measurement set cannot be built -- the units are
                    # read off the measurements themselves.
                    warnings.warn(
                        f"'{self.name}': the target is never visible from this "
                        "instrument over the requested epochs, so no "
                        f"measurement file was written to '{file_name}'. Check "
                        "the geometry with GroundStation.station_visibility or "
                        "Camera.target_in_fov, and for a ground station "
                        "consider a lower elevation_mask (currently "
                        f"{mask_deg} deg).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return

        ## Generate the measurement using the computed model
        t1, t2, t3, meas = self.computed_measurements(
            target,
            epoch_array,
            epoch_start,
            epoch_end,
            tstep,
            frame,
            noisy,
        )

        ## Create an obs dictionary
        n_meas = epoch_array.size

        # Convert the epochs actually simulated into the station-time (UTC)
        # Y-DOY-SEC tags the file format carries.
        #
        # This is the inverse of what observed_measurements does on the way back
        # in: it rebuilds the epochs with SpiceManager.YDS2et (spice.str2et),
        # and et2YDS here is spice.et2utc, so the two use the same LSK and the
        # same DELTET model. The round trip is bit-exact -- verified over
        # fractional-second and post-leap-second epochs, max error 0.000 ns --
        # which is what makes a simulated file reloadable as if it were real
        # data: the model recomputes against the same spacecraft state at the
        # same epoch.
        #
        # Use .et, not .times: times is formatted in the EpochArray's display
        # system and representation, so it is only ET for a TDB/AWU array. The
        # guard this replaces (``if epoch_array.system == "TDB"``) had no else
        # branch, so any other system raised UnboundLocalError on `year` below.
        year, doy, sec = SpiceManager.et2YDS(epoch_array.et)

        # Generate measurement dictionary
        dict_meas = {
            "year": np.array(year).tolist(),
            "doy": np.array(doy).tolist(),
            "sec": np.array(sec).tolist(),
            "meas": np.array(meas.quantity.values).tolist(),
            "spice_id": np.array(self.instrument.spice_id).tolist(),
            "outlier_flag": (np.zeros((1, n_meas))[0].tolist()),
        }

        # Generate DB dictionary
        dict_DB = {
            "name": file_name,
            "meas_sim": dict_meas,  # Simulated measurement
        }

        # Add aux data as needed
        self._write_aux_data(dict_DB, dict_meas)

        # Every column in the measurement block is one value per observation, so
        # they all have to be n_meas long. A model's _write_aux_data copies its
        # columns out of computed_measurements_dict, which was filled either by
        # _read_aux (from whatever pass was read) or by a default_*_aux helper
        # (on its own internally-generated grid). Neither is guaranteed to match
        # the epochs being simulated here, and a mismatch would be written out
        # silently: observed_measurements rebuilds its epochs from the year/doy/
        # sec columns, so the file would come back with measurements paired to
        # the wrong times, or misaligned against a shorter index range. Refuse it.
        ragged = {
            key: len(value)
            for key, value in dict_meas.items()
            if isinstance(value, (list, np.ndarray)) and len(value) != n_meas
        }
        if ragged:
            raise ValueError(
                f"'{self.name}': the measurement block for '{file_name}' is "
                f"ragged -- {n_meas} epochs were simulated, but "
                + ", ".join(f"'{k}' has {n}" for k, n in sorted(ragged.items()))
                + ". Columns added by _write_aux_data come from "
                "computed_measurements_dict, which has to describe the same "
                "observations as the epochs passed here. Either pass the "
                "epoch_array those aux values belong to, or rebuild the aux "
                f"dict for these epochs (e.g. {type(self).__name__}."
                "default_* with the matching count). No file was written."
            )

        ## Determine save path
        if folder_path_override is not None:
            save_dir = folder_path_override
        else:
            optical_keywords = [
                "centroiding",
                "landmarks",
                "opnav",
                "optical",
                "beacon",
            ]
            save_folder = (
                "measurements/optical"
                if any(kw in file_name.lower() for kw in optical_keywords)
                else "measurements/radiometric"
            )
            save_dir = os.path.join("data", save_folder)

        ## Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        ## Save dictionary as JSON or pickle (extension follows ``fmt``)
        out_path = os.path.join(save_dir, file_name + format_suffix(fmt))
        dump(dict_DB, out_path, fmt=fmt, indent=4)

    def _write_aux_data(self, dict_DB, dict_meas):
        """Add this model's auxiliary data to a file being written.

        Called by :meth:`write_observed_measurements` once it has built the
        measurement block. The base model has no auxiliary data and adds nothing;
        :class:`~scarabaeus.DopplerReal` and
        :class:`~scarabaeus.SequentialRangingReal` override it.

        Parameters
        ----------
        dict_DB : dict
            The whole file being written, updated in place.
        dict_meas : dict
            Its measurement block, updated in place. Every column added must have
            one value per observation.
        """
        pass

    def _require_sigma(self, dataset_name: str = None, units: Units = None):
        """
        This model's sigma as a bare value, or a clear error if it has none.

        A dataset carries one sigma per observation, which is what weights the
        residual in the filter, so a dataset cannot be generated without one.
        Reading ``self.sigma.values`` directly (as this did) turned the two ways
        of arriving here without a sigma into ``AttributeError: 'NoneType'
        object has no attribute 'values'`` raised deep inside dataset
        generation, naming neither the model nor the dataset:

        * The sigma was never set. It is optional on every model constructor and
          in :class:`MeasurementConfig`, so ``MeasurementSpec.build`` happily
          produces a model without one and ``compute`` then fails.
        * ``MeasurementSpec.edit`` refit it and had no units to express it in,
          so it assigned a bare float. Tolerated here now, since a plain number
          is a reasonable thing to hand a model, but edit resolves units from
          the dataset first so it should not come to this.

        The bare value lands in the dataset's ``sigma`` column, which is in the
        units of the computed measurement, so a sigma carrying other units of
        the same dimension (m for a range computed in km) is converted into
        them first. Copying its bare value instead weighted the residuals by a
        sigma off by the unit ratio.

        Parameters
        ----------
        dataset_name : str, optional
            Named in the error, to say which dataset was being built.
        units : Units, optional
            The units of the computed measurement, which the sigma is returned
            in. ``None`` returns the sigma's bare value as it stands, as does a
            sigma without units.

        Returns
        -------
        float or numpy.ndarray
            The numeric sigma: one value, or one per measurement component.

        Raises
        ------
        MissingSigmaError
            If the model has no sigma.
        SigmaUnitsError
            If the sigma's units are not of the dimension of ``units``.
        """
        sigma = self.sigma
        if sigma is None:
            where = f" '{dataset_name}'" if dataset_name else ""
            raise self.MissingSigmaError(
                f"'{self.name}' has no measurement sigma, so the dataset"
                f"{where} cannot be built: the sigma is what weights each residual in the "
                "filter. Set it on the model constructor (sigma=...), through "
                "MeasurementConfig(sigma={'<type or dataset name>': ...}) when "
                "building a MeasurementSpec, or afterwards through the model's "
                "sigma setter. Note that refitting a sigma from the residuals "
                "needs residuals, which need this dataset, so an initial value "
                "is required even when it will later be re-estimated."
            )
        if isinstance(units, Units) and isinstance(getattr(sigma, "units", None), Units):
            try:
                sigma = sigma.convert_to(units)
            except ValueError as err:
                where = f" of '{dataset_name}'" if dataset_name else ""
                raise self.SigmaUnitsError(
                    f"'{self.name}' has a sigma in {sigma.units}, which cannot be "
                    f"converted to the units of the measurements{where} "
                    f"({units}): {err}"
                ) from err
        return getattr(sigma, "values", sigma)

    @staticmethod
    def _ramp_table(aux_dict: dict) -> list:
        """Ramp table of an auxiliary-data dict, as :meth:`RampTableManager.integrate` reads it.

        Parameters
        ----------
        aux_dict : dict
            Auxiliary data holding the :attr:`RAMP_AUX_KEYS` entries and,
            optionally, :attr:`RAMP_YEAR_KEY`.

        Returns
        -------
        list
            The ramp columns in :attr:`RAMP_AUX_KEYS` order, followed by
            the year column, or None in its place when the table has no years.
        """
        return [aux_dict[key] for key in Measurement.RAMP_AUX_KEYS] + [
            aux_dict.get(Measurement.RAMP_YEAR_KEY)
        ]

    @staticmethod
    def validate_aux(aux: dict, label: str = "") -> None:
        """
        Check that an auxiliary-data dictionary is internally consistent.

        An aux dict describes two things sampled on two different grids: one
        value per observation, and one entry per ramp-table segment. Everything
        in each group has to be the same length, and a value meant to be
        per-observation has to be a sequence rather than a bare number.

        Nothing checks that on the way in otherwise, and the failures it
        prevents surface far from their cause:

        * Mismatched lengths reach :meth:`RampTableManager.integrate`, which
          indexes ``ramp_freq`` with an index derived from ``ramp_sec`` and
          raised ``IndexError: index 9 is out of bounds for axis 0 with size 5``
          from inside the forward model.
        * A ragged per-observation group reaches
          :meth:`write_observed_measurements`, whose own check then refuses to
          write the file -- correct, but one step later than here.

        Parameters
        ----------
        aux : dict
            The dictionary to check, as a ``default_*_aux`` helper or
            :meth:`_read_aux` produces it.
        label : str, optional
            Named in the error, to say which dictionary was wrong.

        Raises
        ------
        TypeError
            If *aux* is not a dict. ``None`` is reported specifically, since
            that means the model was never given its auxiliary data.
        ValueError
            If the per-observation group or the ramp group is ragged. The
            message groups the keys by the length they have, since the odd one
            out is usually obvious once they are laid side by side.

        Examples
        --------
        .. code-block:: python

            aux = doppler_model.default_doppler_aux(t0, tf, N=20, band="S")
            scb.Measurement.validate_aux(aux, label="DSS-25 Range Rate")
        """
        where = f" for '{label}'" if label else ""
        if aux is None:
            raise TypeError(
                f"No auxiliary data{where}. A real radiometric model needs it "
                "to compute an observable: the ramp table, the uplink "
                "frequency, the station calibrations and the electronic "
                "delays all come from there. It is installed by "
                "observed_measurements as it reads a tracking file, or set "
                "explicitly from the model's default_doppler_aux / "
                "default_seqranging_aux for simulated data."
            )
        if not isinstance(aux, dict):
            raise TypeError(
                f"Auxiliary data{where} must be a dict, got "
                f"{type(aux).__name__}."
            )

        groups = {"per-observation": {}, "ramp-table": {}}
        for key, value in aux.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                continue  # a scalar applies to the whole pass
            group = ("ramp-table"
                     if key in (*Measurement.RAMP_AUX_KEYS, Measurement.RAMP_YEAR_KEY)
                     else "per-observation")
            groups[group].setdefault(len(value), []).append(key)

        problems = []
        for group, by_length in groups.items():
            if len(by_length) <= 1:
                continue
            spread = "; ".join(
                f"{n}: " + ", ".join(f"'{k}'" for k in sorted(keys))
                for n, keys in sorted(by_length.items())
            )
            problems.append(f"the {group} values have differing lengths -- {spread}")

        if problems:
            raise ValueError(
                f"Auxiliary data{where} is not self-consistent: "
                + "; and ".join(problems)
                + ". Every per-observation value describes the same "
                "observations and every ramp value the same ramp table, so "
                "within each group the lengths have to agree."
            )

    @staticmethod
    def resolve_elevation_mask(elevation_mask: ArrayWUnits | float) -> float:
        """
        Normalise an elevation mask to degrees.

        Parameters
        ----------
        elevation_mask : ArrayWUnits or float
            Minimum elevation angle. An :class:`~scarabaeus.units.ArrayWUnits`
            is converted from its own units, so radians work as well as
            degrees. A bare number is taken to be in degrees, which is what the
            parameter has always meant and what every default (10.0) is.

        Returns
        -------
        float
            The mask in degrees.

        Raises
        ------
        ElevationMaskError
            If the mask is not a number or an angle, or is outside 0 to 90
            degrees. Also if it is a bare number that is not a whole number of
            degrees and is no more than pi/2: 0.17 could be 0.17 deg or 10 deg
            meant in radians, and reading it the wrong way silently keeps or
            drops the observations near the horizon. Such a mask has to carry
            its units. A larger bare number is read as degrees, since as
            radians it would be above 90 deg and be refused anyway.

        Examples
        --------
        .. code-block:: python

            Measurement.resolve_elevation_mask(10.0)                      # 10.0
            Measurement.resolve_elevation_mask(ArrayWUnits(0.2, rad))     # 11.459...
            Measurement.resolve_elevation_mask(ArrayWUnits(0.5, deg))     # 0.5
            Measurement.resolve_elevation_mask(0.5)       # ElevationMaskError
        """
        if isinstance(elevation_mask, ArrayWUnits):
            try:
                degrees = float(elevation_mask.convert_to(Units.get_units("deg")).values)
            except (TypeError, ValueError) as err:
                raise Measurement.ElevationMaskError(
                    f"Elevation mask {elevation_mask} is not a single angle: {err}"
                ) from err
        else:
            try:
                degrees = float(elevation_mask)
            except (TypeError, ValueError) as err:
                raise Measurement.ElevationMaskError(
                    f"Elevation mask {elevation_mask!r} is neither a number of "
                    "degrees nor an ArrayWUnits angle."
                ) from err
            if 0.0 < degrees <= np.pi / 2 and not degrees.is_integer():
                raise Measurement.ElevationMaskError(
                    f"A bare elevation mask of {degrees:g} is read as degrees, but "
                    f"it could be {np.degrees(degrees):.4g} deg given in radians. "
                    f"Pass it with its units: ArrayWUnits({degrees:g}, deg) or "
                    f"ArrayWUnits({degrees:g}, rad)."
                )
        if not 0.0 <= degrees <= 90.0:
            raise Measurement.ElevationMaskError(
                f"Elevation mask of {degrees:g} deg is outside 0 to 90. A bare "
                "number is read as degrees, so a value in radians has to be "
                "passed as an ArrayWUnits to be converted."
            )
        return degrees

    def _resolve_dsn_band(self, band: str):
        """
        Look up a DSN frequency band's parameters in :attr:`dsn_bands`.

        Parameters
        ----------
        band : str
            Band letter, case-insensitive. ``'X'`` and ``'S'`` by default.

        Returns
        -------
        Measurement.DSNBand
            The band's turnaround ratio, exciter scalars, range unit and a
            representative uplink frequency.

        Raises
        ------
        ValueError
            If the band is not in :attr:`dsn_bands`. The message lists the ones
            that are, since the alternative is a silently wrong turnaround
            ratio, which changes the observable without changing its shape.
        """
        key = str(band).strip().upper()
        bands = self._dsn_bands
        if key not in bands:
            raise ValueError(
                f"Unknown DSN frequency band '{band}'. "
                f"Available: {sorted(bands)}."
            )
        return bands[key]

    @staticmethod
    def transponder_delay_seconds(delay_RU: float, reference_frequency: float,
                                  range_unit: float) -> float:
        """
        Convert a spacecraft transponder delay from range units to seconds.

        The delay is divided by ``range_unit * reference_frequency``, the rate
        of the clock the delay is counted against.

        Parameters
        ----------
        delay_RU : float
            Transponder delay as the mission specifies it [RU].
        reference_frequency : float
            The band's range-unit reference frequency [1/s], i.e.
            :attr:`Measurement.DSNBand.ru_reference_frequency`.
        range_unit : float
            The range unit, from the pass's exciter scalars.

        Returns
        -------
        float
            The delay [s].

        Notes
        -----
        Pass the band's reference frequency, **not** its uplink frequency. The
        two differ by 18% in X band, and the delay enters the round-trip light
        time directly rather than acting as a scale factor, so substituting the
        uplink makes the delay 2.4 us longer -- 1.4 km of two-way range. Tried
        on real tracking data, that drove the sequential-ranging residuals from
        55-206 RU to 2330-2694 RU, so the reference frequency is what the data
        supports. Why a downlink-scale reference is the right one is an open
        question; see the attribute's own note.
        """
        return delay_RU / (range_unit * reference_frequency)

    def _default_aux_grids(
        self,
        tstart: EpochArray,
        tend: EpochArray,
        n_obs: int,
        ramp_margin: float = None,
        n_ramp: int = None,
        dt_obs: float = None,
    ) -> tuple[dict, dict]:
        """
        Build the two time grids a ``default_*_aux`` dictionary needs.

        A default aux dict describes two different things sampled on two
        different grids: one value per simulated observation, and one entry per
        ramp-table segment. Everything sized off one grid has to agree:
        :meth:`~scarabaeus.RampTableManager.integrate` indexes ``ramp_freq`` with an index
        derived from ``ramp_sec``, so a ``ramp_freq`` shorter than ``ramp_sec``
        raises IndexError and a longer one leaves its extra entries silently
        dead. Both grids come from here, so they agree by construction.

        Parameters
        ----------
        tstart, tend : EpochArray
            Span to cover. Only the first epoch of each is used.
        n_obs : int
            Number of simulated observations.
        ramp_margin : float, optional
            Seconds the ramp grid extends beyond the span at each end, by
            default :attr:`ramp_margin`. The ramp is queried at the *uplink transmit* time
            ``t1 = t3 - RTLT``, which precedes the first observation's time tag
            by one round-trip light time, so a ramp table that merely covered
            the span would fail on the first observation --
            :class:`~scarabaeus.RampTableManager.RampCoverageError`, which is a gap seen
            in real tracking data. The default clears a round trip to
            anywhere inside the asteroid belt; raise it for a more distant
            target.
        n_ramp : int, optional
            Number of ramp-table entries, by default :attr:`ramp_entries`,
            spread evenly over the widened span.
        dt_obs : float, optional
            Fixed observation spacing [s]. By default the span is divided into
            *n_obs* equal steps, which suits a data type with no standard
            sample rate. Pass a value when the spacing is fixed by the data
            type -- two-way Doppler counts over a standard 60 s interval, and
            that interval must equal the spacing or the simulated counts would
            overlap or leave gaps. *n_obs* is then reduced, with a warning, to
            whatever whole number of steps fits inside the span.

        Returns
        -------
        obs : dict
            ``et``, ``year``, ``doy``, ``sec``, the sample spacing ``dt`` [s]
            and the effective count ``n``, which is *n_obs* unless *dt_obs*
            forced it down.
        ramp : dict
            ``et`` (n_ramp,), ``day``, ``sec`` and the segment length ``dt`` [s].
        """
        t_0 = float(np.atleast_1d(tstart.et)[0])
        t_f = float(np.atleast_1d(tend.et)[0])
        if not t_f > t_0:
            raise ValueError(
                f"tend ({t_f}) must be later than tstart ({t_0}) to build a "
                "default aux dictionary."
            )
        if int(n_obs) < 1:
            raise ValueError(f"n_obs must be at least 1, got {n_obs}.")
        if ramp_margin is None:
            ramp_margin = self._ramp_margin
        if n_ramp is None:
            n_ramp = self._ramp_entries
        if int(n_ramp) < 2:
            raise ValueError(
                f"n_ramp must be at least 2 to define a ramp interval, got "
                f"{n_ramp}."
            )

        # Observation grid: built from a step rather than with np.arange over the
        # span, so the count cannot come out one short or one long on a rounding
        # boundary.
        if dt_obs is None:
            # No standard sample rate: divide the span into exactly n_obs steps.
            dt_obs = (t_f - t_0) / int(n_obs)
            n_eff = int(n_obs)
        else:
            # The spacing is fixed by the data type, so the count gives way
            # instead. Keeping n_obs and stretching the step would change the
            # count interval, which for Doppler is a real quantity (Tc) and not
            # ours to adjust.
            dt_obs = float(dt_obs)
            if not dt_obs > 0.0:
                raise ValueError(f"dt_obs must be positive, got {dt_obs}.")
            n_eff = min(int(n_obs), int(np.floor((t_f - t_0) / dt_obs)))
            if n_eff < 1:
                raise ValueError(
                    f"The span {t_f - t_0:g} s is shorter than one {dt_obs:g} s "
                    "sample, so no observation fits in it."
                )
            if n_eff != int(n_obs):
                warnings.warn(
                    f"Reduced the observation count from {int(n_obs)} to "
                    f"{n_eff}: at the fixed {dt_obs:g} s spacing that is as "
                    f"many as fit in the {t_f - t_0:g} s span. The spacing is "
                    "set by the data type and is not stretched to keep the "
                    "requested count.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        obs_et = t_0 + np.arange(n_eff, dtype=float) * dt_obs
        obs_year, obs_doy, obs_sec = SpiceManager.et2YDS(obs_et)

        # Ramp grid: n_ramp points spanning the widened interval inclusively, so
        # ramp_sec[0] is genuinely before the first uplink transmit time.
        ramp_0 = t_0 - float(ramp_margin)
        ramp_f = t_f + float(ramp_margin)
        dt_ramp = (ramp_f - ramp_0) / (int(n_ramp) - 1)
        ramp_et = ramp_0 + np.arange(int(n_ramp), dtype=float) * dt_ramp
        ramp_year, ramp_day, ramp_sec = SpiceManager.et2YDS(ramp_et)

        obs = {
            "et": obs_et,
            "year": list(obs_year),
            "doy": list(obs_doy),
            "sec": list(obs_sec),
            "dt": dt_obs,
            "n": n_eff,
        }
        ramp = {
            "et": ramp_et,
            "year": list(ramp_year),
            "day": list(ramp_day),
            "sec": list(ramp_sec),
            "dt": dt_ramp,
        }
        return obs, ramp

    def _meas_bias_quantity(self, units) -> ArrayWUnits | None:
        """
        Return the constructor ``meas_bias`` as an ArrayWUnits.

        A number or sequence of numbers is taken in ``units``, the units of the
        observable; an ArrayWFrame or ArrayWUnits keeps its own units.

        Parameters
        ----------
        units : Units
            Units of the observable the bias is added to.

        Returns
        -------
        ArrayWUnits or None
            The bias, or ``None`` when no ``meas_bias`` was given.
        """
        bias = getattr(self, "_meas_bias", None)
        if bias is None:
            return None
        if isinstance(bias, ArrayWFrame):
            return bias.quantity
        if isinstance(bias, ArrayWUnits):
            return bias
        return ArrayWUnits(np.asarray(bias, dtype=float), units)

    def _extract_bias(self, state_key: str, units) -> ArrayWUnits | None:
        """
        Return the measurement bias of this model's instrument.

        Parameters
        ----------
        state_key : str
            Key of the bias in the allowed state dictionary, e.g.
            ``"range_bias"``.
        units : Units
            Units of the observable, in which a numeric ``meas_bias`` is taken.

        Returns
        -------
        ArrayWUnits or None
            The value of the first reference-state entry matching ``state_key``
            whose body is the instrument, which may hold one value per pass;
            otherwise the constructor ``meas_bias`` (see
            :meth:`_meas_bias_quantity`), or ``None`` when there is none.
        """
        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        if self._reference_state_vector is not None and state_key in state_allowed:
            pattern = state_allowed[state_key][2]
            instrument_sid = getattr(self.instrument, "spice_id", None)
            for state_entry in self._reference_state_vector.state:
                component = state_entry[0]
                body = state_entry[4]
                value = state_entry[5]  # ArrayWFrame (even for scalar bias)

                if pattern.match(component) is None:
                    continue
                if getattr(body, "spice_id", None) == instrument_sid:
                    return value.quantity

        return self._meas_bias_quantity(units)

    def set_pass_epochs(self, pass_epochs: list[list] | None):
        """
        Enable per-pass biases.

        Parameters
        ----------
        pass_epochs : list[list[float]] | None
            Observation epochs grouped by pass, e.g. ``[[t0, t1], [t10, t11]]``.
            Pass ``None`` to revert to global-bias mode.
        """
        if pass_epochs is None:
            self._epoch_to_pass_index = None
        else:
            self._epoch_to_pass_index = {
                float(t): i for i, group in enumerate(pass_epochs) for t in group
            }

    def _bias_at_epoch(self, bias: ArrayWUnits | None, epoch) -> ArrayWUnits | None:
        """
        Return the part of a bias that applies at one epoch.

        A bias of one element is global: it applies to every pass. A longer one
        holds one value per pass, and the epoch's pass is looked up in the
        mapping :meth:`set_pass_epochs` builds.

        Parameters
        ----------
        bias : ArrayWUnits or None
            The whole bias, from :meth:`_extract_bias`.
        epoch : EpochArray
            The single epoch being evaluated.

        Returns
        -------
        ArrayWUnits or None
            The bias of the epoch's pass, or ``None`` when there is no bias or
            the epoch belongs to no pass.

        Raises
        ------
        MissingPassIndexError
            If the bias holds one value per pass and no epoch-to-pass mapping
            is set.
        """
        if bias is None or bias.values.size == 1:
            return bias
        if self._epoch_to_pass_index is None:
            raise Measurement.MissingPassIndexError(
                f"The bias has {bias.values.size} elements, so it holds "
                "one bias per pass, but no epoch-to-pass mapping is set and "
                "the bias of each epoch is undefined. Call set_pass_epochs() "
                "(or pass pass_epochs=) first."
            )
        pass_index = self._epoch_to_pass_index.get(float(epoch.times.values))
        values = bias.values.ravel()
        if pass_index is None or pass_index >= len(values):
            return None
        return ArrayWUnits(values[pass_index], bias.units)

    def _bias_partials(self, component: str, bias_dim: int, comp_body, epoch) -> list:
        """
        Return the partials of the observable by a bias state component.

        Parameters
        ----------
        component : str
            Name of the state component.
        bias_dim : int
            Its dimension: 1 for a global bias, the number of passes for one
            bias per pass.
        comp_body : Body or None
            Body the component belongs to.
        epoch : EpochArray
            The single epoch being evaluated.

        Returns
        -------
        list of float
            ``bias_dim`` partials, all 0 for another instrument's bias. For
            this instrument's: 1 for a global bias, or 1 in the slot of the
            epoch's pass and 0 in the others.

        Raises
        ------
        MissingPassIndexError
            If the component holds one bias per pass and no epoch-to-pass
            mapping is set.
        """
        partials = [0.0] * bias_dim
        if comp_body is None or self.instrument.spice_id != comp_body.spice_id:
            return partials
        if bias_dim == 1:
            return [1.0]
        if self._epoch_to_pass_index is None:
            raise Measurement.MissingPassIndexError(
                f"State component '{component}' has dimension "
                f"{bias_dim}, so it holds one bias per pass, but "
                "no epoch-to-pass mapping is set and the partial "
                "for this epoch is undefined. Call "
                "set_pass_epochs() (or pass pass_epochs= to "
                "observed_measurements/computed_measurements) "
                "before computing partials."
            )
        pass_index = self._epoch_to_pass_index.get(float(epoch.times.values))
        if pass_index is not None and pass_index < bias_dim:
            partials[pass_index] = 1.0
        return partials

    @classmethod
    def _electronic_delay_kind(cls, name: str):
        """Electronic-delay key a component name matches.

        Parameters
        ----------
        name : str
            State component name.

        Returns
        -------
        str or None
            Key of :attr:`_electronic_delay_link_ends` whose pattern matches
            ``name``, or None when none does.
        """
        allowed = StateDefinition._generate_allowed_state_dictionary()
        return next((key for key in cls._electronic_delay_link_ends
                     if allowed[key][2].match(name)), None)

    def _electronic_delay_owner(self, kind: str, target):
        """Body that owns an electronic delay of this kind.

        Parameters
        ----------
        kind : str
            Key of :attr:`_electronic_delay_link_ends`.
        target : Spacecraft or Body
            Spacecraft the data tracks.

        Returns
        -------
        Body or GroundStation
            *target* for the transponder delay at t2, this model's instrument
            for the station delays at t1 and t3.
        """
        return target if self._electronic_delay_link_ends[kind] == "t2" else self.instrument

    def _electronic_delay(self, kind: str, target):
        """
        One estimated electronic delay, in seconds.

        Parameters
        ----------
        kind : str
            Key of :attr:`_electronic_delay_link_ends`.
        target : Spacecraft or Body
            Spacecraft the data tracks.

        Returns
        -------
        tuple or None
            ``(name, delay, scale)``: the state entry of this kind owned by
            :meth:`_electronic_delay_owner`, its value in seconds plus the
            offset a finite difference applies, and the seconds in one of the
            units it is declared in, which takes a partial by seconds to the
            declared units. ``None`` when the state has no such entry.

        Notes
        -----
        No delay is split by pass. The station delays at t1 and t3 apply to
        every pass of that station, and the transponder delay at t2 to every
        station and pass tracking *target*. Like the delays of the tracking
        data they lengthen the round-trip light time and leave the epochs alone
        (Moyer 2000, eq. 11-7).
        """
        pattern = StateDefinition._generate_allowed_state_dictionary()[kind][2]
        owner_id = self._spice_id(self._electronic_delay_owner(kind, target))
        for entry in self._state_entries():
            if pattern.match(entry[0]) is None or entry[4] is None:
                continue
            if self._spice_id(entry[4]) != owner_id:
                continue
            value = entry[5] if len(entry) > 5 else None
            quantity = getattr(value, "quantity", value)
            if quantity is None:
                delay, scale = 0.0, 1.0
            else:
                delay = float(np.asarray(quantity.convert_to(sec).values).reshape(-1)[0])
                scale = float(np.asarray(ArrayWUnits(1.0, quantity.units)
                                         .convert_to(sec).values).reshape(-1)[0])
            offset = getattr(self, "_electronic_delay_offset", None)
            if offset is not None and offset[0] == entry[0]:
                delay += offset[1]
            return entry[0], delay, scale
        return None

    def _estimated_electronic_delays(self, target) -> dict:
        """
        Estimated electronic delays at each link end, in seconds.

        Parameters
        ----------
        target : Spacecraft or Body
            Spacecraft the data tracks.

        Returns
        -------
        dict
            ``{"t1", "t2", "t3"} -> float``: the station's uplink delay, the
            spacecraft's transponder delay and the station's downlink delay,
            each 0 when it is not estimated.
        """
        delays = {"t1": 0.0, "t2": 0.0, "t3": 0.0}
        for kind, link_end in self._electronic_delay_link_ends.items():
            delay = self._electronic_delay(kind, target)
            if delay is not None:
                delays[link_end] = delay[1]
        return delays

    def _electronic_delay_partial(self, component: str, comp_body, target) -> float:
        """
        Partial of the round-trip light time by an electronic-delay component.

        Parameters
        ----------
        component : str
            State component name, one :meth:`_electronic_delay_kind` matches.
        comp_body : Body or None
            Body the component belongs to.
        target : Spacecraft or Body
            Spacecraft the data tracks.

        Returns
        -------
        float
            Seconds of light time per declared unit of the component: its
            scale for the delay this model reads, 0 for another station's or
            another spacecraft's.
        """
        kind = self._electronic_delay_kind(component)
        if comp_body is None:
            return 0.0
        delay = self._electronic_delay(kind, target)
        if delay is None or delay[0] != component:
            return 0.0
        owner = self._electronic_delay_owner(kind, target)
        if self._spice_id(comp_body) != self._spice_id(owner):
            return 0.0
        return delay[2]

    def _finite_difference_electronic_delay(self, component, target, step, evaluate):
        """
        Central-difference an electronic delay through the full forward model.

        Parameters
        ----------
        component : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``.
        target : Spacecraft or Body
            Spacecraft the data tracks.
        step : ArrayWUnits or float
            Step of the delay, the entry of :meth:`_fd_step_sizes` under its
            kind; a float is taken in seconds.
        evaluate : callable
            No-argument evaluator returning the noiseless observable rows.

        Returns
        -------
        numpy.ndarray or None
            ``(rows, 1)`` block in the declared units of the entry, or None
            when the delay is not the one this model reads (another station's
            or another spacecraft's). A model whose forward model does not read
            the delay gets a zero column.

        Notes
        -----
        The offset is always cleared, including when an evaluation raises.
        """
        name = component[0]
        owner = component[4] if len(component) > 4 else None
        if self._electronic_delay_partial(name, owner, target) == 0.0:
            return None
        scale = self._electronic_delay(self._electronic_delay_kind(name), target)[2]
        # A bare number is taken in seconds, as _station_eps takes canonical units.
        if hasattr(step, "convert_to"):
            step = step.convert_to(sec).values
        step = float(np.asarray(step).reshape(-1)[0])
        outputs = []
        try:
            for sign in (1, -1):
                self._electronic_delay_offset = (name, sign * step)
                outputs.append(np.asarray(evaluate(), dtype=float).reshape(-1))
        finally:
            self._electronic_delay_offset = None
        return (self._fd_difference(*outputs) / (2 * step) * scale).reshape(-1, 1)

    def _scalar_state_value(self, name: str, error: type):
        """
        Value of a one-number model parameter the state estimates.

        Parameters
        ----------
        name : str
            Key of the parameter in the allowed-state dictionary, such as
            ``"solar_corona_scale"`` or ``"gamma_rel"``.
        error : type
            Exception class to raise, nested in this class.

        Returns
        -------
        float or None
            The entry's value, or None when the state does not declare it or
            declares it without a value.

        Raises
        ------
        error
            If the entry is not one finite number, or the state declares it
            more than once.
        """
        pattern = StateDefinition._generate_allowed_state_dictionary()[name][2]
        entries = [entry for entry in self._state_entries() if pattern.match(entry[0])]
        if len(entries) > 1:
            raise error(
                f"{name} is declared {len(entries)} times; "
                "it is one parameter shared by every link"
            )
        if not entries:
            return None
        entry = entries[0]
        value = entry[5] if len(entry) > 5 else None
        quantity = getattr(value, "quantity", value)
        if quantity is None:
            return None
        values = np.asarray(getattr(quantity, "values", quantity), dtype=float).reshape(-1)
        if entry[1] != 1 or values.size != 1 or not np.isfinite(values[0]):
            raise error(f"{name} must be one finite number, got {values}")
        return float(values[0])

    def _solar_corona_scale(self) -> float:
        """
        Estimated scale on the solar-corona delay.

        Returns
        -------
        float
            The ``solar_corona_scale`` value plus the offset a finite
            difference applies; 1 when the state does not estimate it or
            declares it without a value.

        Raises
        ------
        SolarCoronaScaleError
            If the entry is not one finite number, or the state declares it
            more than once.
        """
        scale = self._scalar_state_value("solar_corona_scale", Measurement.SolarCoronaScaleError)
        scale = 1.0 if scale is None else scale
        offset = getattr(self, "_solar_corona_scale_offset", None)
        return scale + offset if offset is not None else scale

    def _gamma_rel(self) -> float:
        """
        PPN :math:`\\gamma` of the light-time solution.

        Returns
        -------
        float
            The ``gamma_rel`` value plus the offset a finite difference
            applies; :attr:`Constants.gamma_GR` when the state does not
            estimate it or declares it without a value.

        Raises
        ------
        GammaRelError
            If the entry is not one finite number, or the state declares it
            more than once.

        Notes
        -----
        The value reaches every place the barycentric and geocentric solvers
        use :math:`\\gamma`: the Sun and body Shapiro delays (Moyer 2000,
        eq. 8-55), the station's move into SSB space-time (eq. 4-10) and the
        geocentric solution's inverse of it. At the default the light time is
        bit-identical to one solved without the argument.
        """
        gamma = self._scalar_state_value("gamma_rel", Measurement.GammaRelError)
        gamma = float(np.squeeze(constants.gamma_GR.values)) if gamma is None else gamma
        offset = getattr(self, "_gamma_rel_offset", None)
        return gamma + offset if offset is not None else gamma

    @staticmethod
    def _apply_solar_corona_scale(rtlt: float, rtlt_parts, scale: float):
        """
        Scale the solar-corona delay inside a round-trip light time.

        Parameters
        ----------
        rtlt : float
            Round-trip light time [s] from the light-time solver, holding the
            nominal corona delay.
        rtlt_parts : array_like
            Its 17 parts, with the down- and up-leg corona delays at 12 and 14.
        scale : float
            Corona scale, from :meth:`_solar_corona_scale`.

        Returns
        -------
        rtlt : float
            The light time with the corona delay scaled.
        corona : float
            The nominal two-leg corona delay [s], which is the partial of the
            light time by the scale.

        Notes
        -----
        The corona is the last term the solvers add, from the light time they
        have already converged, so scaling it moves the light time and leaves
        the epochs t1 and t2 alone, as the media delays do. At a scale of 1 the
        light time comes back unchanged.
        """
        parts = np.asarray(rtlt_parts).ravel()
        corona = float(parts[12]) + float(parts[14])
        return rtlt + (scale - 1.0) * corona, corona

    def _finite_difference_offset(self, attribute: str, step, evaluate):
        """
        Central-difference a one-number model parameter through the full forward model.

        Parameters
        ----------
        attribute : str
            Offset attribute the parameter's getter adds, such as
            ``"_solar_corona_scale_offset"`` (:meth:`_solar_corona_scale`) or
            ``"_gamma_rel_offset"`` (:meth:`_gamma_rel`).
        step : float
            Step of the parameter, dimensionless.
        evaluate : callable
            No-argument evaluator returning the noiseless observable rows.

        Returns
        -------
        numpy.ndarray
            ``(rows, 1)`` block of the observable per unit of the parameter.

        Notes
        -----
        The offset is always cleared, including when an evaluation raises.
        """
        step = float(np.asarray(getattr(step, "values", step)).reshape(-1)[0])
        outputs = []
        try:
            for sign in (1, -1):
                setattr(self, attribute, sign * step)
                outputs.append(np.asarray(evaluate(), dtype=float).reshape(-1))
        finally:
            setattr(self, attribute, None)
        return (self._fd_difference(*outputs) / (2 * step)).reshape(-1, 1)

    @staticmethod
    def _resolve_epoch_array(epoch_array, epoch_start=None, epoch_end=None, tstep=1.0):
        """
        Return the epochs to evaluate, from either form the callers advertise.

        ``computed_measurements`` and ``write_observed_measurements`` both
        document ``epoch_start`` / ``epoch_end`` / ``tstep`` as an alternative to
        ``epoch_array`` ("Required if ``epoch_array`` is not provided"). This
        builds that grid when ``epoch_array`` is not given.

        Parameters
        ----------
        epoch_array : EpochArray or None
            Explicit epochs. Wins if given.
        epoch_start, epoch_end : Epoch or float, optional
            Interval bounds, used when *epoch_array* is None.
        tstep : float, optional
            Step in seconds across that interval. Defaults to 1.0.

        Returns
        -------
        EpochArray

        Raises
        ------
        ValueError
            If neither an ``epoch_array`` nor a complete start/end pair is given.
        """
        if epoch_array is not None:
            return epoch_array
        if epoch_start is None or epoch_end is None:
            raise ValueError(
                "No epochs to evaluate: pass epoch_array, or both epoch_start "
                "and epoch_end (with tstep). Neither was provided."
            )
        _, sec_u = Units.get_units(["km", "sec"])
        return EpochArray.interval(
            epoch_start, epoch_end, dt=ArrayWUnits(float(tstep), sec_u), sys="TDB"
        )

    def _fd_step_sizes(self) -> dict:
        """
        Finite-difference perturbations for the spacecraft state components.

        Defaults to 10 km on each position component, 0.01 km/s on each
        velocity component, 1 microsecond on each electronic delay and 1 on
        the solar-corona scale, overridden by any entry of ``self.eps_dict``. The same dictionary may
        also carry station steps keyed by parameter kind (see
        :attr:`station_eps_defaults`), so supplying only those leaves the
        spacecraft steps at their defaults.

        Returns
        -------
        dict
            ``{"dx","dy","dz","dvx","dvy","dvz"} -> ArrayWUnits`` in km and
            km/s, the keys of :attr:`_electronic_delay_link_ends` -> seconds,
            ``"solar_corona_scale"`` and ``"gamma_rel"`` -> float, plus any
            other supplied entries.
        """
        km_u, sec_u = Units.get_units(["km", "sec"])
        steps = {
            "dx": ArrayWUnits(1e1, km_u),
            "dy": ArrayWUnits(1e1, km_u),
            "dz": ArrayWUnits(1e1, km_u),
            "dvx": ArrayWUnits(1e-2, km_u / sec_u),
            "dvy": ArrayWUnits(1e-2, km_u / sec_u),
            "dvz": ArrayWUnits(1e-2, km_u / sec_u),
        }
        # The delays enter the light time linearly, so the step only has to
        # stay clear of the forward model's rounding.
        steps.update({kind: ArrayWUnits(1e-6, sec_u)
                      for kind in self._electronic_delay_link_ends})
        # The observable is linear in the corona scale, so a whole unit keeps
        # the difference far above the light-time noise.
        steps["solar_corona_scale"] = 1.0
        # Gamma scales the Shapiro delays and the station's SSB space-time
        # transform, both close to linear in it.
        steps["gamma_rel"] = 1.0
        steps.update(getattr(self, "eps_dict", None) or {})
        return steps

    def _cache_matches(self, epoch_array: EpochArray) -> bool:
        """Check whether the light-time cache was built for ``epoch_array``.

        Parameters
        ----------
        epoch_array : EpochArray
            Measurement epochs.

        Returns
        -------
        bool
            ``True`` when the cache exists and its epochs match ``epoch_array`` to
            within 1 ms.
        """
        if not hasattr(self, "_partials_cache_key"):
            return False
        key = np.asarray([e.times.values for e in epoch_array], dtype=float).reshape(-1)
        cached = np.asarray(self._partials_cache_key, dtype=float).reshape(-1)
        return key.shape == cached.shape and np.allclose(key, cached, rtol=0, atol=1e-3)

    def _ensure_partials_cache(
        self, target: Spacecraft, epoch_array: EpochArray, frame: Frame
    ) -> None:
        """
        Make sure the light-time cache is present and current for ``epoch_array``.

        The cache holds the quantities of the forward light-time solution that the
        analytical partials reuse, e.g. the transmit frequencies, the light times
        and the GR and solar-corona delays.

        Parameters
        ----------
        target : Spacecraft
            Spacecraft the data tracks.
        epoch_array : EpochArray
            Measurement epochs.
        frame : Frame
            Inertial frame of the light-time solution.

        Notes
        -----
        On a cache miss this runs the noise-free forward model for these epochs
        rather than fail or use stale values. In the normal OD flow the forward model
        has already run, so this does nothing.
        """
        if self._cache_matches(epoch_array):
            return
        self.computed_measurements(
            target=target, epoch_array=epoch_array, frame=frame, noisy=False
        )

    def _compute_partials_with_fd(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
        bias_key: str | None = None,
    ) -> list:
        """
        Compute the partials by batched forward finite differences.

        Parameters
        ----------
        target : Spacecraft
            Spacecraft the data tracks.
        epoch_array : EpochArray
            Measurement epochs.
        frame : Frame, optional
            Inertial frame the state partials are expressed in. Defaults to
            ``J2000``.
        bias_key : str, optional
            Key of this model's measurement bias in the allowed state dictionary,
            e.g. ``"range_bias_RU"``.

        Returns
        -------
        list of np.ndarray
            One partial row per epoch, with a column for each state component.

        Notes
        -----
        Steps come from :meth:`_fd_step_sizes`. Position and velocity columns are
        differenced through :meth:`_batched_component_partials`; ``eta_srp`` and
        ``dv_man`` columns are zero, the ``bias_key`` columns come from
        :meth:`_bias_partials` (1 for this station, in the slot of the epoch's
        pass for one bias per pass, and 0 for others), station-geometry and
        zenith-delay columns are central differences through the forward model
        (:meth:`_finite_difference_station`), as are the station's uplink and
        downlink delays and the tracked spacecraft's transponder delay
        (:meth:`_finite_difference_electronic_delay`) and the solar-corona
        scale and PPN gamma (:meth:`_finite_difference_offset`), and any other
        component is zero.
        Without a state definition the row holds position and velocity only.
        """
        eps_dict = self._fd_step_sizes()

        _, _, _, meas_nom_out = self.computed_measurements(
            target=target,
            epoch_array=epoch_array,
            frame=frame,
            noisy=False,
            eps_override=None,
        )
        meas_nom = meas_nom_out.quantity.values

        N = epoch_array.size
        position = ["dx", "dy", "dz"]
        velocity = ["dvx", "dvy", "dvz"]
        H_blocks = []

        if self._state_definition is None:
            for components in (position, velocity):
                H_blocks.append(
                    self._batched_component_partials(
                        components, eps_dict, target, epoch_array, frame, meas_nom
                    )
                )
            return [np.hstack(H_blocks)[i, :] for i in range(N)]

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        for comp_def in self._state_definition:
            comp_name = comp_def[0]
            comp_dim = comp_def[1]
            comp_body = comp_def[4] if len(comp_def) > 4 else None

            if state_allowed["position"][2].match(comp_name):
                H_blocks.append(
                    self._batched_component_partials(
                        position, eps_dict, target, epoch_array, frame, meas_nom
                    )
                )

            elif state_allowed["velocity"][2].match(comp_name):
                H_blocks.append(
                    self._batched_component_partials(
                        velocity, eps_dict, target, epoch_array, frame, meas_nom
                    )
                )

            elif state_allowed["eta_srp"][2].match(comp_name):
                H_blocks.append(np.zeros((N, 1)))

            elif state_allowed["dv_man"][2].match(comp_name):
                H_blocks.append(np.zeros((N, 3)))

            elif bias_key is not None and state_allowed[bias_key][2].match(comp_name):
                H_blocks.append(
                    np.array(
                        [
                            self._bias_partials(comp_name, comp_dim, comp_body, epoch)
                            for epoch in epoch_array
                        ]
                    )
                )

            elif (self._is_station_parameter(comp_name)
                  or self._is_media_parameter(comp_name)):
                # eps_override only moves the spacecraft, so the station is
                # perturbed in the state instead, through the full forward model.
                block = self._finite_difference_station(
                    comp_def,
                    lambda: self.computed_measurements(
                        target=target, epoch_array=epoch_array, frame=frame,
                        noisy=False)[-1].quantity.values)
                H_blocks.append(np.zeros((N, comp_dim)) if block is None else block)

            elif self._electronic_delay_kind(comp_name) is not None:
                block = self._finite_difference_electronic_delay(
                    comp_def, target, eps_dict[self._electronic_delay_kind(comp_name)],
                    lambda: self.computed_measurements(
                        target=target, epoch_array=epoch_array, frame=frame,
                        noisy=False)[-1].quantity.values)
                H_blocks.append(np.zeros((N, comp_dim)) if block is None else block)

            elif state_allowed["solar_corona_scale"][2].match(comp_name):
                H_blocks.append(self._finite_difference_offset(
                    "_solar_corona_scale_offset", eps_dict["solar_corona_scale"],
                    lambda: self.computed_measurements(
                        target=target, epoch_array=epoch_array, frame=frame,
                        noisy=False)[-1].quantity.values))

            elif state_allowed["gamma_rel"][2].match(comp_name):
                H_blocks.append(self._finite_difference_offset(
                    "_gamma_rel_offset", eps_dict["gamma_rel"],
                    lambda: self.computed_measurements(
                        target=target, epoch_array=epoch_array, frame=frame,
                        noisy=False)[-1].quantity.values))

            else:
                H_blocks.append(np.zeros((N, comp_dim)))

        return [np.hstack(H_blocks)[i, :] for i in range(N)]

    def _batched_component_partials(
        self,
        components: list[str],
        eps_dict: dict,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame,
        meas_nom: np.ndarray,
    ) -> np.ndarray:
        """
        Forward-difference partials of the observable for a set of state
        components, one perturbed forward-model run per component.

        Parameters
        ----------
        components : list of str
            Perturbation keys, e.g. ``["dx", "dy", "dz"]``.
        eps_dict : dict
            Step for each key, as returned by :meth:`_fd_step_sizes`.
        target : Spacecraft
            Spacecraft the data tracks.
        epoch_array : EpochArray
            Measurement epochs.
        frame : Frame
            Inertial frame of the light-time solution.
        meas_nom : np.ndarray
            Nominal noise-free measurements (N,).

        Returns
        -------
        np.ndarray
            Partials of shape (N, len(components)), each column the
            :meth:`_fd_difference` of the perturbed and nominal values divided
            by the step.
        """
        N = epoch_array.size
        H = np.zeros((N, len(components)))

        for j, comp in enumerate(components):
            _, _, _, meas_pert_out = self.computed_measurements(
                target=target,
                epoch_array=epoch_array,
                frame=frame,
                noisy=False,
                eps_override={comp: eps_dict[comp]},
            )
            meas_pert = meas_pert_out.quantity.values
            H[:, j] = self._fd_difference(meas_pert, meas_nom) / eps_dict[comp].values

        return H

    def _fd_difference(self, meas_pert: np.ndarray, meas_nom: np.ndarray) -> np.ndarray:
        """
        Difference a perturbed and a nominal observable for finite differences.

        Parameters
        ----------
        meas_pert : np.ndarray
            Perturbed noise-free measurements (N,).
        meas_nom : np.ndarray
            Nominal noise-free measurements (N,).

        Returns
        -------
        np.ndarray
            ``meas_pert - meas_nom``; a model whose observable wraps overrides it.
        """
        return meas_pert - meas_nom

    def computed_measurements(self, *args, **kwargs):
        """
        Evaluate the forward model. Implemented by every concrete model.

        There is no shared implementation, but there *is* a shared contract --
        both :meth:`write_observed_measurements` and
        :meth:`generate_measurement_dataset` unpack four values from this call,
        so a model that returns anything else fails at the call site with no
        explanation. Subclasses must return::

            t1, t2, t3, meas

        ``meas`` is the computed observable as an ``ArrayWFrame``. The three
        epochs describe *where in time the observable lives*, which for a
        two-way radiometric measurement is not one instant:

        ``t1``
            Transmission from the ground station. Used as the regression
            abscissa when detrending residuals for outlier rejection
            (:meth:`~scarabaeus.MeasurementEditing._huber_mad_evaluate`).
        ``t2``
            Reflection at the spacecraft. This is the one the estimator cares
            about: it is the epoch the filter associates the measurement with,
            and what measurements from different models are matched on
            (:meth:`~scarabaeus.FilterDataManager._combine_indices_by_t2`, and the LKF).
        ``t3``
            Reception at the ground station, i.e. the epoch the observation is
            time-tagged with in the data file. Currently carried through to the
            dataset for traceability rather than read by the estimator.

        For a single-epoch observable -- the optical models, where the
        observation is not spread across a light-time path in this way -- all
        three collapse to the observation epoch. That is a correct statement
        about the observable, not padding to satisfy the signature.

        Raises
        ------
        NotImplementedError
            If a subclass has not implemented it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement computed_measurements(). "
            "It must return (t1, t2, t3, meas); see Measurement.computed_measurements "
            "for what the three epochs mean."
        )


    # "meas_ideal" was the original key for simulated measurements. It is a
    # misnomer -- the models simulate, and simulated is not the same as ideal:
    # write_observed_measurements(noisy=True) adds AWGN, so the contents are
    # deliberately non-ideal. The key is now written as "meas_sim".
    #
    # It is also a *persisted* key, so files written before the change still say
    # "meas_ideal". The alias below keeps those readable. Remove it, and the
    # fallback in _resolve_meas_key, once the old files are gone.
    _MEAS_KEY_ALIASES = {"meas_sim": "meas_ideal", "meas_ideal": "meas_sim"}

    @staticmethod
    def _resolve_meas_key(trk_data, meas_name: str, file_name: str = None) -> str:
        """
        Return the measurement key actually present in *trk_data*.

        Prefers the requested name and falls back to its legacy/modern alias, so
        that new readers open old files and old scripts open new files. Warns on
        a fallback, because the mismatch is worth knowing about even though it is
        handled.

        Parameters
        ----------
        trk_data : dict
            A loaded tracking-data file.
        meas_name : str
            The measurement block the caller asked for.
        file_name : str, optional
            Named in the error, to say which file was looked in.

        Returns
        -------
        str
            The requested name, or its alias when that is the one present.

        Raises
        ------
        KeyError
            If neither the requested name nor its alias is in the file. The
            message lists the blocks that are there, and says so explicitly
            when the file turns out to be a simulated one, because a script
            that reads real data and then reloads its own simulated output
            needs a different name for each and reusing the real one is the
            easy mistake to make.
        """
        if meas_name in trk_data:
            return meas_name
        alias = Measurement._MEAS_KEY_ALIASES.get(meas_name)
        if alias is not None and alias in trk_data:
            warnings.warn(
                f"Measurement block '{meas_name}' not found in this file; using "
                f"'{alias}'. 'meas_ideal' was renamed to 'meas_sim' because the "
                "models simulate rather than idealise (noisy=True adds AWGN). "
                "Files written before the rename still use the old key; this "
                "fallback will be removed once they are regenerated.",
                DeprecationWarning,
                stacklevel=3,
            )
            return alias

        where = f" in '{file_name}'" if file_name else ""
        available = sorted(
            key for key, value in trk_data.items() if isinstance(value, dict)
        )
        simulated = [key for key in Measurement._MEAS_KEY_ALIASES if key in trk_data]
        if simulated:
            hint = (
                f" This is a simulated file: its measurements are under "
                f"'{simulated[0]}', which is what this parameter defaults to, so "
                f"the name only has to be given when reading real tracking data."
            )
        else:
            hint = (
                " Real tracking data names its blocks by measurement type, e.g."
                " '2W_sranging' or '2W_doppler'; a file written by"
                " write_observed_measurements uses 'meas_sim'."
            )
        raise KeyError(
            f"Measurement block '{meas_name}' not found{where}. "
            f"Available blocks: {available}.{hint}"
        )

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_sim",
        units: Units = spc_ad_units,
        frame: Frame = J2000,
        **read_kwargs,
    ) -> ObsData:
        """
        Reads measurements from a .json file.

        Parameters
        ----------
        file_name : str
            The filename of the .json file containig the measurement information.

        meas_name : str, optional
            The name of the measurement data to access from the dictionary.
            Defaults to ``'meas_sim'``. Files written before the rename use
            ``'meas_ideal'``; either name resolves to whichever block the file
            actually contains, with a DeprecationWarning on the fallback.

        units : Units, optional
            Units to be used to write the output AWU. Defaults to ``unitless``.

        frame : Frame, optional
            Frame to be used to write the output AWF. Defaults to a J2000
            Frame object.

        **read_kwargs
            Options for the model's own :meth:`_read_aux`, forwarded unchanged.
            Models with no auxiliary data to read ignore them.

            This is where ``sc_transponder_delay_RU`` goes for the DSN
            radiometric models, which is the only such option today. It is not
            a named parameter here because this signature is shared by every
            measurement model, optical ones included.

        Returns
        -------
        ObsData
            A :class:`~scarabaeus.measurements.ObsData.ObsData` named tuple with
            five fields, addressable by name or index:

            * ``[0]`` ``meas_time_et`` : EpochArray
                Observation epochs. Built from the file's UTC station tag via
                :meth:`~scarabaeus.SpiceManager.YDS2et`; see the comment at the conversion
                below for why this carries SPICE's time model deliberately.
            * ``[1]`` ``meas_sec`` : numpy.ndarray
                The observation times as seconds of day, as given in the file.
            * ``[2]`` ``meas_obs`` : ArrayWFrame
                The observed quantities, with units and frame attached.
            * ``[3]`` ``meas_outliers`` : numpy.ndarray
                Per-observation outlier flags, 1 = reject. Seeded from the
                file's own ``outlier_flag`` and additionally set here for
                records carrying the ``-1`` no-data sentinel and for the rows
                the model knows to be invalid (:meth:`_flag_invalid_rows`).
                Downstream this is the authoritative flag array: it is what the
                filter, :meth:`~scarabaeus.MeasurementSpec.outlier_flags` and the
                editors gate on, and :meth:`generate_measurement_dataset` writes
                into it when the forward model cannot compute an observation.
            * ``[4]`` ``aux_dict`` : dict
                Ancillary per-observation data read by the model's ``_read_aux``
                (ramp tables, transponder ratios, electronic delays, ...). None
                for models that need none.

        Notes
        -----
        The writing of the json assumes or requires units and frames.

        Every row in the file is returned, including rows flagged bad. Nothing
        is dropped here, so the observations, the ``aux_dict`` and any later
        outlier rejection all stay on one consistent set of indices.
        """
        ## Read the measurement file (JSON or pickle, auto-detected by extension)
        trk_data = load(os.path.join(file_name))

        # Accept either the current "meas_sim" key or the legacy "meas_ideal".
        meas_name = Measurement._resolve_meas_key(trk_data, meas_name, file_name)

        # Read every row in the file. Points already flagged as bad in the file
        # are kept, and inherit their flag through ObsData.meas_outliers, rather
        # than being dropped here. That keeps the observations, the aux_dict
        # (which _read_aux reads at full length) and any later outlier rejection
        # on one consistent set of indices.
        idx_meas = np.arange(len(trk_data[meas_name]["outlier_flag"]))

        # Convert the YDS format into ET format.
        #
        # The file carries station time (UTC), and YDS2et resolves it through
        # spice.str2et. The ET that comes out is a *carrier* for that tag, not a
        # claim about TDB: its TDB-TT term is SPICE's single-sinusoid DELTET
        # model, which is good to only ~30 us.
        #
        # That is deliberate and must stay. _compute_precise_RTLT inverts it with
        # spice.unitim(tt_et, "ET", "TAI") as its first step, and because both
        # directions use the same DELTET model the round trip recovers the true
        # TAI exactly -- verified bit-exact, 0.0000 us. Only then does the RTLT
        # convert TAI to TDB with _compute_et_from_tai, which is ~5.5x more accurate
        # than SPICE and is where the physics actually needs the precision.
        #
        # So do NOT "improve" this by emitting Moyer ET here. The unitim step
        # downstream expects SPICE ET; feeding it Moyer ET makes that step
        # recover a TAI 28.5 us off. The 30 us never reaches the physics as things
        # stand, because it cancels in the round trip between here and _compute_precise_RTLT.
        meas_time_et = []
        for idx in idx_meas:
            meas_time_et.append(
                SpiceManager.YDS2et(
                    [trk_data[meas_name]["year"][idx]],
                    [trk_data[meas_name]["doy"][idx]],
                    [trk_data[meas_name]["sec"][idx]],
                )[0]
            )

        # Prepare output for list saving: time(et), obs, time(sec)
        meas_obs = np.array(trk_data[meas_name]["meas"])[idx_meas]
        meas_sec = np.array(trk_data[meas_name]["sec"])[idx_meas]
        meas_outliers = np.array(trk_data[meas_name]["outlier_flag"])[idx_meas]

        # TRK-2-34 writes -1 in the measurement field when the record carries no
        # measurement -- typically at the head of a pass, before the station has
        # acquired the uplink. The files do not mark these in outlier_flag, so
        # unless they are caught here they are differenced against a computed
        # value and enter the fit as data.
        #
        # They are not harmless, and not reliably caught downstream:
        #   * Where the ramp table also fails to cover the epoch (same root
        #     cause: no uplink yet), the computed value used to come out 0, so
        #     the residual was -1 RU -- a -2 sigma point against sigma = 0.5,
        #     comfortably inside any sensible rejection gate. Ten of those in one
        #     DSS-24 pass were a coherent, same-signed pull on 42% of that pass's
        #     ranging, and being identical they deflated the MAD enough to mask
        #     three genuine outliers.
        #   * Where the ramp table does cover the epoch, the computed value is
        #     real and the residual is enormous, so rejection removes the point
        #     -- but by accident, and the huge residual distorts the robust scale
        #     for the rest of the pass on its way out.
        #
        # The same sentinel convention is already honoured for the auxiliary
        # fields (_generate_electronic_delays_dictionary treats -1 and -99 as
        # invalid); this extends that to the measurement itself.
        no_data = np.asarray(meas_obs, dtype=float) == -1.0
        if np.any(no_data):
            n_new = int(np.sum(no_data & (np.asarray(meas_outliers) == 0)))
            meas_outliers = np.asarray(meas_outliers).copy()
            meas_outliers[no_data] = 1
            if n_new:
                print(
                    f"  {n_new} observation(s) in '{meas_name}' carry the -1 "
                    f"no-data sentinel and were flagged as outliers "
                    f"(indices {np.flatnonzero(no_data).tolist()})"
                )

        # Transform the output in EpochArray and AWF
        meas_time_et = EpochArray(
            ArrayWUnits(np.array(meas_time_et), spc_time_units).values, sys="TDB"
        )
        if units is None:
            meas_obs = ArrayWFrame(ArrayWUnits(meas_obs, None), frame)
        else:
            meas_obs = ArrayWFrame(meas_obs, units, frame)

        # Construct the aux_dict as necessary for this measurement type
        aux_dict = self._read_aux(trk_data, meas_name, **read_kwargs)

        # The read owns the aux data: it is the only thing that produces it, so
        # it is also what installs it on the model.
        #
        # The aux data had two owners. ObsData.aux_dict is what the read returns,
        # and computed_measurements_dict is what computed_measurements actually
        # reads (every aux_dict = self.computed_measurements_dict in DopplerReal
        # and SequentialRangingReal). Nothing connected them except one line in
        # MeasurementSpec.build, so a spec assembled any other way --
        # from_models, from_dict, from_dicts, or the models driven directly --
        # reached computed_measurements with the attribute still None and failed
        # with "TypeError: 'NoneType' object is not subscriptable", pointing at
        # neither the cause nor the fix. Every caller had to know to copy it
        # across by hand.
        #
        # Only this method ever produces an aux_dict: _read_aux is a no-op on the
        # base and only DopplerReal and SequentialRangingReal implement it, and
        # none of the models that override observed_measurements build one. So
        # assigning here covers every path by which aux data can exist.
        #
        # Guarded on None so a model that was handed a dict explicitly -- a
        # default_*_aux for simulation, say -- does not have it cleared by
        # reading a file whose type carries no aux data.
        if aux_dict is not None:
            self._computed_measurements_dict = aux_dict

        # Flag rows this model can tell are not measurements (placeholders)
        meas_outliers = self._flag_invalid_rows(
            np.array(trk_data[meas_name]["meas"], dtype=float)[idx_meas],
            meas_outliers,
            aux_dict,
        )

        return ObsData(meas_time_et, meas_sec, meas_obs, meas_outliers, aux_dict)

    def _read_aux(self, trk_data, meas_name, **read_kwargs):
        """
        Model-specific auxiliary data from an already-loaded file.

        Nothing by default: a model with no auxiliary data to read ignores any
        read options it is handed, so a DSN-only option passed to a model that has
        no use for it is not an error.

        Parameters
        ----------
        trk_data : dict
            The loaded tracking file.
        meas_name : str
            Key of the measurement block in ``trk_data``.
        **read_kwargs
            Model-specific read options, forwarded from
            :meth:`observed_measurements`. Ignored here.

        Returns
        -------
        dict or None
            None: the base model has no auxiliary data.
        """
        return None

    def _flag_invalid_rows(self, meas_values, meas_outliers, aux_dict):
        """Outlier flags with the rows this model knows are invalid set to 1.

        Called by :meth:`observed_measurements` with the raw values of the file.
        The base model knows of no invalid rows.

        Parameters
        ----------
        meas_values : numpy.ndarray
            The observed values as read, one per row.
        meas_outliers : numpy.ndarray
            The outlier flags so far, 1 = reject.
        aux_dict : dict or None
            The model's auxiliary data, as returned by :meth:`_read_aux`.

        Returns
        -------
        numpy.ndarray
            ``meas_outliers``, unchanged.
        """
        return meas_outliers
    
    def _use_leg_state_definition(self, epoch: EpochArray) -> None:
        """
        Switch to the state definition of the sequence leg holding an epoch.

        Parameters
        ----------
        epoch : EpochArray
            The single epoch being evaluated.

        Notes
        -----
        The last epoch of a leg other than the final one takes the definition
        of the leg after the node that follows it. An epoch that no leg holds
        keeps the current definition, and without a sequence definition this
        does nothing.
        """
        if self._sequence_definition is None:
            return
        # Find the corresponding epoch array and update the state definition
        for idx_epochs_vec, epoch_array_leg in enumerate(
            self._sequence_definition.epochs_vec
        ):
            epoch_time_value = epoch.times.values
            t_vals = epoch_array_leg.times.values
            # Nodes have scalar epoch arrays — skip them
            if (
                not isinstance(t_vals, (list, np.ndarray))
                or np.asarray(t_vals).ndim == 0
            ):
                continue
            t_start = t_vals[0]
            t_end = t_vals[-1]
            if epoch_time_value == t_end and idx_epochs_vec != (
                len(self._sequence_definition.epochs_vec) - 1
            ):
                # Special case for the last time in the array
                self._state_definition = self._sequence_definition.models[
                    idx_epochs_vec + 2
                ].state_definition
                return
            elif t_start <= epoch_time_value <= t_end:
                # Update state definition for the found epoch
                self._state_definition = self._sequence_definition.models[
                    idx_epochs_vec
                ].state_definition
                return  # Exit once the correct epoch array is found

    def compute_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Stacks together measurement _partials for an epoch array at different epochs.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch_array : EpochArray
            The epochs.

        frame : Frame, optional
            The reference frame. Defaults to a J2000 Frame object.

        Returns
        -------
        _partials : list
            A list with all the _partials evaluated at different epochs in the ``epoch_array``.
        """
        # Generate _partials
        _partials = []
        print("")
        print("Generating _partials computed measurements...")
        update, close = Measurement._progress_monitor(
            num_total=epoch_array.size, desc="Partials"
        )
        if self._sequence_definition is not None:
            for i, epoch in enumerate(epoch_array):
                self._use_leg_state_definition(epoch)

                # Compute measurement _partials after updating the state definition
                _partials.append(self._partials(target, epoch, frame))
                update(i)

            close()
        else:
            for i, epoch in enumerate(epoch_array):
                _partials.append(self._partials(target, epoch, frame))
                update(i)
            close()
        return _partials

    def residuals(
        self, observed_meas: ArrayWFrame, computed_meas: ArrayWFrame
    ) -> ArrayWFrame:
        """
        Generates the measurement model's residuals given observed and computed ArrayWFrames.

        Parameters
        ----------
        observed_meas : ArrayWFrame
            The observed measurements values (O).

        computed_meas : ArrayWFrame
            The computed measurements values (C).

        Returns
        -------
        residuals : ArrayWFrame
            AWF with the residual O-C.
        """
        # Check if the observed and computed measurements share the same reference frame
        if observed_meas.frame.name != computed_meas.frame.name:
            raise ValueError(
                "Observed and Computed measurements don't share the same reference frames."
            )
        if (
            observed_meas.quantity.units.dimensions
            != computed_meas.quantity.units.dimensions
        ):
            bad_unts_err = (
                "Observed and Computed measurements do not share units. "
                f"Observed units: {observed_meas.quantity.units.name}. "
                f"Computed units: {computed_meas.quantity.units.name}."
            )
            raise ValueError(bad_unts_err)

        # Compute residuals
        observed_val = observed_meas.quantity.values
        computed_val = computed_meas.quantity.values
        common_units = observed_meas.quantity.units
        common_frame = observed_meas.frame
        residuals = []
        for i, (o, c) in enumerate(zip(observed_val, computed_val, strict=True)):
            residuals.append(o - c)
        residuals_AWF = ArrayWFrame(np.array(residuals), common_units, common_frame)

        return residuals_AWF

    def generate_measurement_dataset(
        self,
        dataset_name: str | None = None,
        target: Body = None,
        observed_meas: tuple | list | None = None,
        epochs: EpochArray | None = None,
        frame: Frame = J2000,
        noisy: bool = False,
    ) -> list[MeasurementDataSet]:
        """
        Generates a MeasurementDataSet object that can be used by filters downstream.

        Parameters
        ----------
        dataset_name : str, optional
            The name of the MeasurementDataSet, which becomes the residual key
            in ``solution.json``. Defaults to
            :attr:`default_dataset_name` — ``"<instrument>_<measurement_type>"``.

        target : Spacecraft
            The target spacecraft.

        observed_meas : tuple or list, optional
            Observed measurements as ``(epoch_array, meas_sec, meas_obs, outlier_flag)``.
            The computed values are evaluated at ``epoch_array``. Give this or ``epochs``,
            not both. Defaults to ``None``.

        epochs : EpochArray, optional
            Epochs to simulate the measurements at, with no outliers flagged. Give this or
            ``observed_meas``, not both. Defaults to ``None``.

        frame : Frame, optional
            The reference frame. Defaults to a J2000 Frame object.

        noisy : bool, optional
            Indicates if noise is added to the measurements or not. Defaults to ``False``.

        Returns
        -------
        mds_list : list[MeasurementDataSet]
            A list of MeasurementDataSet objects representing the measurements with their key properties to be used by a filter.

        Raises
        ------
        ValueError
            If neither or both of ``observed_meas`` and ``epochs`` are given, if
            ``observed_meas`` carries fewer than four entries (the fourth is the
            outlier-flag array, without which the observations cannot be kept on
            one index set with their flags), or if the model has more than two
            components.

        Notes
        -----
        This is the single bridge from a measurement model to the estimator:
        :meth:`~scarabaeus.FilterOD.fit` and :meth:`~scarabaeus.MeasurementSpec.compute` both come
        through here, and :class:`MeasurementDataSet` is populated nowhere else.
        It pairs each observation with its computed value, residual, partial and
        sigma on one consistent set of indices, which is what lets several
        models with different epochs and different observables be combined into
        one filter update.

        That is also why it, rather than the caller, unpacks ``t1, t2, t3`` from
        :meth:`computed_measurements`. The estimator needs the epochs alongside
        the observable -- ``t2`` to match measurements across models, ``t1`` to
        detrend residuals during outlier rejection -- so they have to survive the
        trip into the dataset. See :meth:`computed_measurements` for what each
        epoch means and why a single-epoch observable collapses all three.

        The MeasurementDataSet output is generated in 6 steps:

        #. Computed measurements
        #. Partials
        #. Residuals
        #. Sigmas
        #. Outlier flag
        #. Pack everything in a list
        #. Pack the list in a MeasurementDataSet object
        """

        # ----------------------------
        # Select epochs source
        # ----------------------------
        if observed_meas is None and epochs is None:
            raise ValueError("Provide exactly one of: observed_meas OR epochs.")

        if observed_meas is not None and epochs is not None:
            raise ValueError("Provide only one of: observed_meas OR epochs (not both).")

        # Fall back to the instrument-derived name rather than carrying a None
        # through: it used to reach the datasets verbatim, giving a set_name of
        # None for a scalar model and "None_ra"/"None_dec" for a split one.
        if not dataset_name:
            dataset_name = self.default_dataset_name

        # observed_meas convention: (epoch_array, meas_sec, meas_obs, meas_outliers)
        if observed_meas is not None:
            if len(observed_meas) < 4:
                raise ValueError(
                    f"Expected observed_meas with 4 entries (t_et, sec, obs, outlier_flag), got {len(observed_meas)}."
                )
            epoch_array = observed_meas[0]
            outlier_flag = np.array(observed_meas[3])
        else:
            epoch_array = epochs
            outlier_flag = np.zeros(epoch_array.size)  # no outliers by definition

        # ----------------------------
        # 1) Computed measurements
        # ----------------------------
        print()
        print("=" * 80)
        print(
            f'Generating computed measurements for the dataset "{dataset_name}"'.center(
                80
            )
        )
        print("=" * 80)

        # Cleared before the run so the list below describes only this call.
        self._uncomputable_indices = []

        t1, t2, t3, _meas = self.computed_measurements(
            target=target,
            epoch_array=epoch_array,
            frame=frame,
            noisy=noisy,
        )

        # Observations the forward model could not compute get flagged here rather
        # than left for outlier rejection to notice. The model records their
        # indices while it runs and emits NaN for them, so the flag is set by the
        # code that identified the problem instead of being inferred downstream
        # from an implausible residual. This is the authoritative flag array --
        # the model's own aux copy is not the one the filter reads.
        uncomputable = getattr(self, "_uncomputable_indices", None) or []
        if len(uncomputable) > 0:
            idx = np.asarray(sorted(set(int(i) for i in uncomputable)), dtype=int)
            already = int(np.sum(np.asarray(outlier_flag).ravel()[idx]))
            outlier_flag = np.asarray(outlier_flag).ravel()
            outlier_flag[idx] = 1
            print(
                f"  {len(idx)} observation(s) could not be computed and were "
                f"flagged as outliers ({already} already flagged in the source "
                f"file): index {idx.tolist()}"
            )
            # Keep the caller's observation record in step, so a spec that is
            # re-computed or written back carries the same flags.
            if observed_meas is not None:
                try:
                    src = np.asarray(observed_meas[3]).ravel()
                    src[idx] = 1
                except Exception:
                    # A read-only or non-array flag container: the dataset still
                    # carries the flag, which is what the filter gates on.
                    pass

        # number of components per epoch
        if len(_meas.quantity.shape) == 2:
            n_components = _meas.quantity.shape[1]
        else:
            n_components = 1

        # ----------------------------
        # 2) Partials
        # ----------------------------
        print("")
        print("=" * 80)
        print(f'Generating _partials for the dataset "{dataset_name}"'.center(80))
        print("=" * 80)

        _measurements_partials = self.compute_partials(
            target=target, epoch_array=epoch_array
        )
        partials_list = [np.asarray(p) for p in _measurements_partials]
        # Each model returns its partials already in its observable units (Hz
        # for DopplerReal, RU for SequentialRangingReal); scaling here would apply
        # it twice.

        # ----------------------------
        # 4) Sigmas
        # ----------------------------
        sigma = self._require_sigma(dataset_name, units=_meas.quantity.units)
        _sigmas = np.array([sigma for _ in range(epoch_array.size)])

        # ----------------------------
        # 3) Observed + residuals
        # ----------------------------
        if observed_meas is not None:
            obs_awf = observed_meas[2]
            _residuals = self.residuals(observed_meas=obs_awf, computed_meas=_meas)

            obs_vals = obs_awf.quantity.values
            res_vals = _residuals.quantity.values.copy()
        else:
            # epochs-only mode: fabricate observed/residuals arrays with correct shape
            if n_components == 1:
                obs_vals = np.zeros(epoch_array.size)
                res_vals = np.zeros(epoch_array.size)
            else:
                obs_vals = np.zeros((epoch_array.size, n_components))
                res_vals = np.zeros((epoch_array.size, n_components))

        _outlier_flag = np.array(outlier_flag)

        # A measurement whose computed value is undefined (NaN, e.g. a DSN
        # round trip with no ramp record at the transmit time) cannot be
        # processed: flag it
        undefined = ~np.isfinite(np.asarray(_meas.quantity.values, dtype=float))
        if undefined.ndim > 1:
            undefined = undefined.any(axis=tuple(range(1, undefined.ndim)))
        _outlier_flag[undefined] = 1

        # ----------------------------
        # 6) Names
        # ----------------------------
        names = [
            "t1",
            "t2",
            "t3",
            "computed",
            "observed",
            "residuals",
            "partials",
            "sigma",
            "outlier_flag",
        ]

        # ----------------------------
        # 7) Outlier gating
        # In epochs-only mode threshold/bias should stay None -> no gating
        # ----------------------------
        threshold = None
        bias = None

        # if you ever enable it, guard it:
        # if observed_meas is None:
        #     threshold = None
        #     bias = None

        if threshold is None or bias is None:
            residuals_w_outliers = res_vals
            partials_w_outliers = partials_list
        else:
            gate = threshold + bias
            idx_outliers = np.where(np.abs(res_vals) > gate)[0]
            res_vals[idx_outliers] = 0.0
            residuals_w_outliers = res_vals
            partials_w_outliers = [
                np.zeros_like(p) if i in idx_outliers else p
                for i, p in enumerate(partials_list)
            ]

        # ----------------------------
        # Pack output
        # ----------------------------
        dataset_list_output = []

        if n_components == 1:
            # robust extraction of time arrays
            try:
                t1_vals = t1[:].times.values
            except (TypeError, AttributeError):
                t1_vals = t1[:]
            try:
                t2_vals = t2[:].times.values
            except (TypeError, AttributeError):
                t2_vals = t2[:]
            try:
                t3_vals = t3[:].times.values
            except (TypeError, AttributeError):
                t3_vals = t3[:]

            values = [
                t1_vals,
                t2_vals,
                t3_vals,
                _meas.quantity[:].values,
                obs_vals,
                residuals_w_outliers,
                partials_w_outliers,
                _sigmas[:],
                _outlier_flag[:],
            ]

            _data_set = MeasurementDataSet(
                set_name=dataset_name,
                measurements=values,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=self._measurement_type,
                units=_meas.quantity.units,
                frame=frame,
            )
            return _data_set

        elif n_components == 2:
            # For 2-components keep your split datasets, but use obs_vals/res_vals that may be zeros
            values_1 = [
                t1[:].times.values,
                t2[:].times.values,
                t3[:].times.values,
                _meas.quantity[:, 0].values,
                obs_vals[:, 0],
                (
                    residuals_w_outliers[:, 0]
                    if residuals_w_outliers.ndim == 2
                    else residuals_w_outliers
                ),
                [element[0] for element in partials_w_outliers],
                _sigmas[:, 0] if _sigmas.ndim == 2 else _sigmas,
                _outlier_flag[:],
            ]
            values_2 = [
                t1[:].times.values,
                t2[:].times.values,
                t3[:].times.values,
                _meas.quantity[:, 1].values,
                obs_vals[:, 1],
                (
                    residuals_w_outliers[:, 1]
                    if residuals_w_outliers.ndim == 2
                    else residuals_w_outliers
                ),
                [element[1] for element in partials_w_outliers],
                _sigmas[:, 1] if _sigmas.ndim == 2 else _sigmas,
                _outlier_flag[:],
            ]

            # One scalar dataset per component, named from the caller's
            # ``dataset_name`` exactly as the single-component branch above is.
            # The old hard-coded "Opnav_u"/"Opnav_v" ignored that argument, so
            # two optical instruments in the same run produced colliding
            # dataset names -- and labelled every model's components as pixel
            # coordinates, including the ones that measure angles.
            label_1, label_2 = self.component_labels

            _data_set_1 = MeasurementDataSet(
                set_name=f"{dataset_name}_{label_1}",
                measurements=values_1,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=f"{self._measurement_type} {label_1}",
                units=_meas.quantity.units,
                frame=frame,
            )
            _data_set_2 = MeasurementDataSet(
                set_name=f"{dataset_name}_{label_2}",
                measurements=values_2,
                names=names,
                instrument=self.instrument,
                target=target,
                epochs_t3=epoch_array,
                measurement_type=f"{self._measurement_type} {label_2}",
                units=_meas.quantity.units,
                frame=frame,
            )
            return [_data_set_1, _data_set_2]

        else:
            raise ValueError(
                "The number of components (>2) is not supported in current SCB version."
            )

    def _build_perturbation_vector(self, eps_override: dict) -> ArrayWUnits:
        """
        Constructs a 6-element perturbation vector [dx, dy, dz, dvx, dvy, dvz]
        as an ArrayWUnits from the given eps_override dictionary.

        Parameters
        ----------
        eps_override : dict
            Dictionary with keys like 'dx', 'dvy', etc., each mapped to an ArrayWUnits.

        Returns
        -------
        delta : ArrayWUnits
            1x6 perturbation vector with units.
        """
        component_labels = ["dx", "dy", "dz", "dvx", "dvy", "dvz"]
        values = np.zeros(6)
        units = [None] * 6

        if eps_override is not None:
            for i, label in enumerate(component_labels):
                if label in eps_override:
                    eps = eps_override[label]
                    values[i] = eps.values
                    units[i] = eps.units

        # Fill missing units with defaults to ensure consistency
        km, sec = Units.get_units(["km", "sec"])
        default_units = [km, km, km, km / sec, km / sec, km / sec]
        units = [u if u is not None else default_units[i] for i, u in enumerate(units)]

        return ArrayWUnits(values.reshape(1, 6), units)

    @staticmethod
    def _station_correction_j2000(station_correction, epoch) -> np.ndarray:
        """Station correction rotated into J2000, or zeros when nothing corrects it.

        Parameters
        ----------
        station_correction : callable or None
            Function of epoch returning the ITRF93 correction in km and km/s.
        epoch : float
            TDB seconds past J2000.

        Returns
        -------
        numpy.ndarray
            Six components in km and km/s.

        Notes
        -----
        The light-time chain reads the station from several places at several
        epochs. Each one has to see the same displaced station, or a two-way
        observable ends up with one leg moved and one leg not, which halves
        every station partial without producing any other visible symptom.
        """
        if station_correction is None:
            return np.zeros(6)
        return np.asarray(spice.sxform("ITRF93", "J2000", float(epoch))) @ np.asarray(
            station_correction(float(epoch)), dtype=float).reshape(6)

    @staticmethod
    def _compute_CN_lt(
        receiver_pos: np.ndarray,
        transmitter_id: str,
        t_rcv: np.ndarray,
        delta: np.ndarray,
        way: str = "up",
        t_rcv_l: float = 0.0,
        gs_delta_awf: ArrayWFrame | None = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Compute converged Newtonian one-way light time between a receiver and a transmiter at reception epoch t_rcv

        Parameters
        ----------
        receiver_pos : np.ndarray
            SSB position of the receiving body (ground station at t3 or spacecraft at t2),
            already in the SSB space-time frame and already carrying any perturbation.

        transmitter_id : str
            Name of the emitting body (ground station at t1 or spacecraft at t2).

        t_rcv : np.ndarray
            Epoch(s) at which the signal is received [ET seconds past J2000].

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s]. Added to the spacecraft
            transmitter on the "down" leg only; on the "up" leg the spacecraft is the receiver,
            whose position already carries it.

        way: str
            Signal path, can be "up" (ground station transmits, taken in the SSB space-time
            frame of Moyer 2000 eq. 4-10) or "down" (spacecraft transmits)

        t_rcv_l : float, optional
            Low part of the reception epoch [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        gs_delta_awf : ArrayWFrame | None, optional
            Station location offset of the ground-station transmitter, "up" leg only.
            See :meth:`_compute_GS_in_SSB_space_time_RLT`.

        station_correction : callable or None
            Optional function of epoch returning the transmitting station's
            ITRF93 correction in km and km/s. Only meaningful on the up leg,
            where the station is the transmitter; on the down leg the station
            is the receiver and its corrected position arrives in
            ``receiver_pos`` already. Without this the up leg would keep the
            nominal station while the down leg moved, halving every station
            partial of a two-way observable.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        light_time : np.ndarray
            One-way geometric light time [s].

        Raises
        ------
        Measurement.LightTimeConvergenceError
            If the light-time root search does not converge.

        Notes
        -----
        All computations are internally defaulted to be executed in J2000.
        """
        # Define constants to use in this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]

        # The receiver is kept fixed (the np.ndarray of its position is passed), while the position of the transmitter is changed (back in time) to find the transmittions time and interval between transmission and reception.

        def _correction(epoch):
            """Transmitter correction in J2000, zero unless a station moves.

            Parameters
            ----------
            epoch : float
                TDB seconds past J2000.

            Returns
            -------
            numpy.ndarray
                Six components in km and km/s.
            """
            return Measurement._station_correction_j2000(station_correction, epoch)

        # Initial guess
        transmitter_pos0 = spice.spkezr(transmitter_id, t_rcv, "J2000", "NONE", "SSB")[
            0
        ] + _correction(t_rcv)
        range0 = np.linalg.norm(transmitter_pos0[:3] - receiver_pos)
        tof_0 = range0 / c

        # Define the residual function
        def _tof_residual(tof):
            t_trn_h, t_trn_l = EpochArray._add_duration(t_rcv, t_rcv_l, -tof)
            if way == "up":
                # Ground station in the SSB space-time frame, as on the down leg
                transmitter_pos_it = (
                    SpiceManager.get_state_precise(
                        "Earth", t_trn_h, "J2000", "SSB", "NONE", t_trn_l
                    ).values[:3]
                    + Measurement._compute_GS_in_SSB_space_time_RLT(
                        transmitter_id, t_trn_h, t_trn_l, gs_delta_awf,
                        station_correction=station_correction,
                        gamma=gamma,
                    )[:3]
                )
            else:
                transmitter_pos_it = (
                    SpiceManager.get_state_precise(
                        transmitter_id, t_trn_h, "J2000", "SSB", "NONE", t_trn_l
                    ).values[:3]
                    + delta[:3]
                )
            range_ = np.linalg.norm(transmitter_pos_it - receiver_pos)
            return (range_ / c) - tof

        # Solve using a root finder
        tof_final = root_scalar(
            _tof_residual, method="brentq", bracket=[0, 2 * tof_0], xtol=1e-15
        )
        if not tof_final.converged:
            raise Measurement.LightTimeConvergenceError(
                f"Light-time root search did not converge ({way} leg, "
                f"transmitter {transmitter_id}, t_rcv {t_rcv!r})."
            )

        return tof_final.root

    @staticmethod
    def _compute_RLT_SUN_lt(
        r1_r2: float, r2_r3: float, r12_r23: float, gamma: float = None
    ) -> float:
        """
        Compute Relativistic lightime delay due to the Sun. This effects takes into account the following General Relativity (GR) effects:
        1) Curvature of the lightpath due to gravity
        2) Reduction in the coordinate of the speed of light
        The value computed in this method is equal to the second term of [Moyer 2000], eq.8-55

        Parameters
        ----------
        r1_r2 : float
            Position vector. For way == 'dwn', r1_r2 is the norm of the position vector from the Sun to the spacecraft at reflection time t2.
            For way == 'up', r1_r2 is the norm of the position vector from the Sun to the ground-station at reception time t3.

        r2_r3 : float
            Position vector. For way == 'dwn', r2_r3 is the norm of the position vector from the Sun to the ground-station at reception time t3.
            For way == 'up', r2_r3 is the norm of the position vector from the Sun to the spacecraft at reflection time t2.

        r12_r23 : float
            Norm of the position vector difference between r2 and r1 or r3 and r2. See Moyer 2000, eq.8-57 and eq.8-58

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        dt_GR : float
            One-Way GR lightime delay [s]
        """
        # Define constants to use in this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_S = constants.SUN.GM.values  # GM of the Sun
        if gamma is None:
            gamma = constants.gamma_GR.values  # Gamma value for GR

        # Compute different terms in Moyer 2000, eq.8-55
        sun_term = (1 + gamma) * mu_S / c**2
        numerator = r1_r2 + r2_r3 + r12_r23 + sun_term
        denominator = r1_r2 + r2_r3 - r12_r23 + sun_term

        # Generate GR lightime delay
        dt_GR = ((1 + gamma) * mu_S / c**3) * np.log(numerator / denominator)

        return dt_GR

    @staticmethod
    def _get_bodies_for_RLT(
        tx_h: float,
        tx_l: float,
        ssb_to_gs_tx_RLTframe: np.ndarray,
        bodies: tuple = None,
    ) -> list:
        """
        Generate the quantities of interest of the partecipating bodies for the GR delays

        Parameters
        ----------
        tx_h : float
            Integer part of the time tx at which to compute the geometric vectors

        tx_l : float
            Fractional part of the time tx at which to compute the geometric vectors

        ssb_to_gs_tx_RLTframe : np.ndarray
            State vector (pos,vel) from SSB to GS at tx in the SSB space-time relativistic frame of reference

        bodies : tuple, optional
            ``(spice_id, GM)`` pairs of the bodies to include. Defaults to
            :attr:`Constants.RLT_BODIES`.

        Returns
        -------
        body_to_gs_tx : list[np.ndarray]
            State vector (pos,vel) from each body to the GS at tx in J2000, in
            the order of ``bodies``
        """
        if bodies is None:
            bodies = constants.RLT_BODIES

        return [
            ssb_to_gs_tx_RLTframe
            - SpiceManager.get_state_precise(
                str(body_id), tx_h, "J2000", "SSB", "None", tx_l
            ).values
            for body_id, _ in bodies
        ]

    @staticmethod
    def _get_bodies_to_sc_for_RLT(
        sc_spice_id: str,
        t2_h: float,
        t2_l: float,
        delta: np.ndarray,
        bodies: tuple = None,
    ) -> list:
        """
        State of the spacecraft relative to each body of the GR delay, at the
        reflection time t2.

        Parameters
        ----------
        sc_spice_id : str
            Spice ID of the spacecraft

        t2_h, t2_l : float
            High and low parts of t2, see
            :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        delta : np.ndarray
            Perturbation added to each state [km, km/s], as in
            :meth:`_compute_RLT_BODIES_lt`.

        bodies : tuple, optional
            ``(spice_id, GM)`` pairs of the bodies to include. Defaults to
            :attr:`Constants.RLT_BODIES`.

        Returns
        -------
        body_to_sc_t2 : list[np.ndarray]
            State vector (pos,vel) from each body to the spacecraft at t2 in
            J2000, plus ``delta``, in the order of ``bodies``
        """
        if bodies is None:
            bodies = constants.RLT_BODIES

        return [
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", str(body_id), "None", t2_l
            ).values
            + delta
            for body_id, _ in bodies
        ]

    @staticmethod
    def _compute_RLT_BODIES_lt(
        t2_t1: float,
        t3_t2: float,
        body_id,
        ssb_to_gs_t_31_RLT: np.ndarray,
        way: str = None,
        delta: np.ndarray = None,
        t2_t1_l: float = 0.0,
        t3_t2_l: float = 0.0,
        bodies: tuple = None,
        body_to_gs: list = None,
        body_to_sc: list = None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Compute Relativistic lightime delay due to different bodies.
        The value computed in this method is equal to the third term of [Moyer 2000], eq.8-55.
        The second term corresponds to the one computed with "_compute_RLT_SUN_lt".
        The first term is the Newtonian component, that can be computed with "_compute_CN_lt".

        Parameters
        ----------
        t2_t1 : float
            Reflection time t2 (for way == 'dwn') or transmittion time t1 (for way == 'up')

        t3_t2 : float
            Reception time t3 (for way == 'dwn') or reflection time t2 (for way == 'up')

        body_id : str
            Spice ID of the spacecraft (the signal end point at t2)

        ssb_to_gs_t_31_RLT : np.ndarray
            State vector (pos,vel) from SSB to GS at t3 (for way == 'dwn') or t1 (for way == 'up') in a SSB space-time relativistic frame of reference

        way : str
            Signal path, can be "up" for 1-2 or "dwn" for 2-3

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s]. This is used to compute the GR delay due to bodies other than the Sun.

        t2_t1_l : float, optional
            Low part of ``t2_t1`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        t3_t2_l : float, optional
            Low part of ``t3_t2`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        bodies : tuple, optional
            ``(spice_id, GM)`` pairs of the bodies to include, GM an
            ArrayWUnits in km^3/s^2. Defaults to :attr:`Constants.RLT_BODIES`;
            an empty tuple turns the term off.

        body_to_gs : list[np.ndarray], optional
            The station-end states, as :meth:`_get_bodies_for_RLT` returns them for
            the station time. Computed here when None. The light-time solution
            passes them because they do not change while the other end iterates.

        body_to_sc : list[np.ndarray], optional
            The spacecraft-end states plus ``delta``, as
            :meth:`_get_bodies_to_sc_for_RLT` returns them for t2. Computed here
            when None.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        dt_GR_bodies : float
            GR lightime delay due to bodies other than the Sun, as per Moyer 2000, eq.8-55

        Raises
        ------
        RLTBodyOnPathError
            If the leg ends at, or runs through, the centre of one of ``bodies``.

        """
        if bodies is None:
            bodies = constants.RLT_BODIES
        if way not in ("dwn", "up") or len(bodies) == 0:
            return 0

        # Define constants
        if gamma is None:
            gamma = constants.gamma_GR.values  # Gamma for GR
        c = constants.c.values  # Speed of light in vacuum [km/sec]

        # The spacecraft end is always at t2, the station end at t3 (dwn) or t1 (up)
        if way == "dwn":
            t2_h, t2_l = t2_t1, t2_t1_l
            tgs_h, tgs_l = t3_t2, t3_t2_l
        else:
            tgs_h, tgs_l = t2_t1, t2_t1_l
            t2_h, t2_l = t3_t2, t3_t2_l
        if delta is None:
            delta = np.zeros(6)

        # Body-to-gs vectors at the station time, body-to-spacecraft vectors at t2
        if body_to_gs is None:
            body_to_gs = Measurement._get_bodies_for_RLT(
                tgs_h, tgs_l, ssb_to_gs_t_31_RLT, bodies=bodies
            )
        if body_to_sc is None:
            body_to_sc = Measurement._get_bodies_to_sc_for_RLT(
                body_id, t2_h, t2_l, delta, bodies=bodies
            )

        dt_GR_bodies = 0
        for (body_spice_id, mu_b), r2_vec, r1_vec in zip(bodies, body_to_gs, body_to_sc):
            r1 = np.linalg.norm(r1_vec[:3])
            r2 = np.linalg.norm(r2_vec[:3])
            r12 = np.linalg.norm(r2_vec[:3] - r1_vec[:3])
            if r1 + r2 - r12 <= 0.0:
                raise Measurement.RLTBodyOnPathError(
                    f"Body {body_spice_id} lies on the {way} leg to {body_id} "
                    f"(r1 = {r1:.6g} km); drop it from rlt_bodies for this target."
                )
            dt_GR_bodies += ((1 + gamma) * mu_b.values / c**3) * np.log(
                (r1 + r2 + r12) / (r1 + r2 - r12)
            )
        return dt_GR_bodies

    @staticmethod
    def _compute_RLT_lt(
        components: str = "RLT",
        r1_r2_vec: np.ndarray = None,
        r2_r3_vec: np.ndarray = None,
        t2_t1: float = None,
        t3_t2: float = None,
        sc_spice_id: str = None,
        ssb_to_gs_t_31_RLT: np.ndarray = None,
        way: str = None,
        delta: np.ndarray = None,
        t2_t1_l: float = 0.0,
        t3_t2_l: float = 0.0,
        bodies: tuple = None,
        body_to_gs: list = None,
        body_to_sc: list = None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Compute Relativistic lightime delay due to the Sun and to different bodies.
        The value computed in this method is equal to the second and third terms of [Moyer 2000], eq.8-55.
        The second term corresponds to the one computed with "_compute_RLT_SUN_lt".
        The third term corresponds to the one computed with "_compute_RLT_BODIES_lt".

        Parameters
        ----------
        components : str
            input to control whether 'SUN',

        r1_r2_vec : np.ndarray
            Position vector. For way == 'dwn', r1_r2_vec is the position vector from the body to the spacecraft at reflection time t2.
            For way == 'up', r1_r2_vec is the position vector from the body to the ground-station at reception time t3.

        r2_r3_vec : np.ndarray
            Position vector. For way == 'dwn', r2_r3_vec is the position vector from the body to the ground-station at reception time t3.
            For way == 'up', r2_r3_vec is the position vector from the body to the spacecraft at reflection time t2.

        t2_t1: float
            Reflection time t2 (for way == 'dwn') or transmission time t1 (for way == 'up')

        t3_t2: float
            Reception time t3 (for way == 'dwn') or reflection time t2 (for way == 'up')

        sc_spice_id: str
            Spice ID of the spacecraft

        ssb_to_gs_t_31_RLT : np.ndarray
            State vector (pos,vel) from SSB to GS at t3 (for way == 'dwn') or t1 (for way == 'up') in a SSB space-time relativistic frame of reference

        way : str
            Signal path, can be "up" for 1-2 or "dwn" for 2-3

        delta : np.ndarray
            Perturbation vector for finite differencing [km, km/s]. This is used to compute the GR delay due to bodies other than the Sun.

        t2_t1_l : float, optional
            Low part of ``t2_t1`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        t3_t2_l : float, optional
            Low part of ``t3_t2`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        bodies : tuple, optional
            ``(spice_id, GM)`` pairs of the bodies other than the Sun, see
            :meth:`_compute_RLT_BODIES_lt`. Defaults to :attr:`Constants.RLT_BODIES`.

        body_to_gs, body_to_sc : list[np.ndarray], optional
            Precomputed body states, passed to :meth:`_compute_RLT_BODIES_lt`.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        dt_GR_bodies : float
            GR lightime delay due to bodies and the Sun, as per Moyer 2000, eq.8-55

        r32_r21_vec: np.ndarray
            r32 vector (for way == 'dwn') or r21 vector (for way == 'up'). Used to determine convergence in the Moyer's 32-steps algorithm for precise RTLT.

        Notes
        -----
        for way == 'dwn':
            r2_r1_vec = r2_vec = sun_to_sc_t2
            r3_r2_vec = r3_vec = sun_to_gs_t3
            r23_r12_vec = r32_vec = gs_to_sc (t3-t2), not defined in time

        for way == 'up':
            r2_r1_vec = r1_vec = sun_to_gs_t1
            r3_r2_vec = r2_vec = sun_to_sc_t2
            r23_r12_vec = r12_vec = sc_to_gs (t2-t1), not defined in time

        """
        # Sun-Relativistic (RLT) component of equation 8.55 (Part 2 of 3)
        if components == "RLT-SUN" or components == "RLT":
            r23_r12_vec = r2_r3_vec - r1_r2_vec
            r32_r21 = np.linalg.norm(r23_r12_vec[:3])
            r2_r1 = np.linalg.norm(r1_r2_vec[:3])
            r3_r2 = np.linalg.norm(r2_r3_vec[:3])
            # Compute RLT effect (downlink)
            dt_GR_SUN = Measurement._compute_RLT_SUN_lt(r2_r1, r3_r2, r32_r21, gamma=gamma)
            # Step (15 - part 1)
            if r32_r21 == 0:
                r32_r21 = np.finfo(float).eps
        else:
            dt_GR_SUN = 0
            r23_r12_vec = np.zeros(3)
            r32_r21 = np.finfo(float).eps

        # Body-Relativistic (RLT) component of equation 8.55 (Part 3 of 3)
        if components == "RLT-BODIES" or components == "RLT":
            dt_GR_BODIES = Measurement._compute_RLT_BODIES_lt(
                t2_t1,
                t3_t2,
                sc_spice_id,
                ssb_to_gs_t_31_RLT,
                way,
                delta,
                t2_t1_l=t2_t1_l,
                t3_t2_l=t3_t2_l,
                bodies=bodies,
                body_to_gs=body_to_gs,
                body_to_sc=body_to_sc,
                gamma=gamma,
            )
        else:
            dt_GR_BODIES = 0

        # Stack relativistic components toghether
        if components == "RLT-SUN":
            dt_GR = dt_GR_SUN
        elif components == "RLT-BODIES":
            dt_GR = dt_GR_BODIES
        elif components == "RLT":
            dt_GR = dt_GR_SUN + dt_GR_BODIES
        else:
            dt_GR = 0

        return dt_GR, r23_r12_vec

    @staticmethod
    def _approx_et_minus_tai(t) -> float:
        """
        Simplified ET-TAI series (Moyer 2000, eq. 2-27), valid after 2000.

        Parameters
        ----------
        t : float
            Epoch [s past J2000]

        Returns
        -------
        dt : float
            ET - TAI [s]
        """
        M = 6.239996 + 1.99096871e-7 * t  # Mean anomaly [rad]
        E = M + 0.01671 * np.sin(M)  # Eccentric anomaly [rad]
        return constants.TT_minus_TAI.values + 1.657e-3 * np.sin(E)

    @staticmethod
    def _et_minus_tai_series(
        X_B_S, X_B_C, X_E_B, X_E_C, X_A_E, X_J_S, X_Sa_S, X_S_C
    ) -> float:
        """
        Vector ET-TAI series (Moyer 2000, eq. 2-26).

        ``X_i_j`` is the J2000 state of point i relative to point j [km, km/s]:
        C (SSB), S (Sun), B (Earth-Moon barycentre), E (Earth), J (Jupiter),
        Sa (Saturn), A (station atomic clock, position only is read).

        Returns
        -------
        dt : float
            ET - TAI [s]
        """
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_S = constants.SUN.GM.values
        mu_J = constants.JUPITER.GM.values
        mu_Sa = constants.SATURN.GM.values

        f1 = constants.TT_minus_TAI.values
        f2 = (2 / (c**2)) * np.dot(X_B_S[3:6], X_B_S[0:3])
        f3 = (1 / (c**2)) * np.dot(X_B_C[3:6], X_E_B[0:3])
        f4 = (1 / (c**2)) * np.dot(X_E_C[3:6], X_A_E[0:3])
        f5 = (1 / (c**2) * (mu_J / (mu_S + mu_J))) * np.dot(X_J_S[3:6], X_J_S[0:3])
        f6 = (1 / (c**2) * (mu_Sa / (mu_S + mu_Sa))) * np.dot(
            X_Sa_S[3:6], X_Sa_S[0:3]
        )
        f7 = (1 / (c**2)) * np.dot(X_S_C[3:6], X_B_S[0:3])
        return f1 + f2 + f3 + f4 + f5 + f6 + f7

    @staticmethod
    def _compute_delta_et_tai(
        t,
        gs_spice_id,
        method: int = 1,
        gs_delta_awf: ArrayWFrame | None = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Method to compute the reltivistic time difference between Ephemeris Time (ET) and International Atomic Time (TAI) at a ground station on Earth.

        Parameters
        ----------
        t : float
            ephemeris time when to compute the correction term

        gs_spice_id : str
            spice id string of the ground station

        method : int
            Method to be used amongst 1 (simplified), 2 (Vector expression, with GS in J2000), 3 (Vector expression, with GS in J2000 SSB relativisti space-time reference)

        gs_delta_awf : ArrayWFrame | None, optional
            Station location offset, see :meth:`_compute_GS_in_SSB_space_time_RLT`. Used by methods 2 and 3.

        station_correction : callable or None, optional
            Function of epoch returning the station's ITRF93 correction in km and
            km/s, see :meth:`_station_correction_j2000`. Used by methods 2 and 3.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        dt : float
            time difference in seconds between ET and TAI at a given epoch

        """
        if method == 1:
            dt = Measurement._approx_et_minus_tai(t)
        elif method == 2 or method == 3:  # Vector expression relativistic correction
            # X_i_j is state of point i relative to point j
            # C (SSB), S (SUN), B (EARTH-MOON Barycenter), E (EARTH)
            # M (MOON), J (JUPITER), Sa (SATURN), A (Atomic clock on Earth which reads TAI)
            X_B_S = spice.spkezr("3", t, "J2000", "None", "Sun")[
                0
            ]  # Earth-Moon Barycenter relative to Sun
            X_B_C = spice.spkezr("3", t, "J2000", "None", "SSB")[
                0
            ]  # Earth-Moon Barycenter relative to SSB
            X_E_B = spice.spkezr("399", t, "J2000", "None", "3")[
                0
            ]  # Earth relative to Earth-Moon Barycenter
            X_E_C = spice.spkezr("399", t, "J2000", "None", "SSB")[
                0
            ]  # Earth relative to SSB
            if method == 2:
                X_A_E = SpiceManager.get_state(gs_spice_id, t, "J2000", "399").values  # Earth to GS atomic clock
                if station_correction is not None:
                    X_A_E = X_A_E + Measurement._station_correction_j2000(
                        station_correction, t
                    )
                if gs_delta_awf is not None:
                    X_A_E = X_A_E[:3] + Measurement._compute_gs_delta_J2000(
                        gs_delta_awf, t
                    )
            elif method == 3:
                X_A_E = Measurement._compute_GS_in_SSB_space_time_RLT(
                    gs_spice_id, t, 0.0, gs_delta_awf,
                    station_correction=station_correction,
                    gamma=gamma,
                )  # Earth to GS atomic clock, in SSB Space-Time relativistic frame
            X_J_S = spice.spkezr("599", t, "J2000", "None", "Sun")[
                0
            ]  # Jupiter relative to Sun
            X_Sa_S = spice.spkezr("699", t, "J2000", "None", "Sun")[
                0
            ]  # Saturn relative to Sun
            X_S_C = spice.spkezr("10", t, "J2000", "None", "SSB")[
                0
            ]  # Sun relative to SSB

            dt = Measurement._et_minus_tai_series(
                X_B_S, X_B_C, X_E_B, X_E_C, X_A_E, X_J_S, X_Sa_S, X_S_C
            )

        else:
            raise ValueError(
                "Invalid method selected. Choose 1 (Simplified), 2 (Vector, J2000), 3 (Vector, J2000 SSB GR-ST))."
            )

        return dt

    @staticmethod
    def _corona_uplink_frequency(
        solar_corona_dict: dict, tt_et: float, tt_et_l: float, rtlt: float
    ) -> float:
        """
        Uplink frequency of the solar-corona delay: the ramp transmit frequency at t1.

        The corona delay goes as :math:`1/f^2`, at :math:`f_T(t_1)` on the up-leg and
        :math:`M_2 f_T(t_1)` on the down-leg (Moyer 2000, Sec. 13.2.8). The tracking
        file's ``ul_freq`` is the frequency at the time tag t3, which is off by the
        ramp rate times one round trip.

        Parameters
        ----------
        solar_corona_dict : dict
            As in :meth:`_compute_precise_RTLT`.
        tt_et, tt_et_l : float
            Reception epoch t3 [ET s], as a (high, low) pair.
        rtlt : float
            Round-trip light time without the corona delay [s]. The corona delay
            itself moves t1 by microseconds, which changes the frequency by ~1e-3 Hz.

        Returns
        -------
        float
            :math:`f_T(t_1)` [Hz], or ``ul_freq`` when the dict has no ``"ramp"`` or
            the ramp does not cover t1.

        Raises
        ------
        RampTableManager.RampTableYearError
            If the ramp table or t1 crosses a new year and the ramp has no years.
        """
        ramp = solar_corona_dict.get("ramp")
        if ramp is None:
            return solar_corona_dict["ul_freq"]
        # t1 in seconds of the t3 day, as the callers form it for the ramp integration
        years, doys, secs = SpiceManager.et2YDS([tt_et])
        t1_sec, t1_sec_l = EpochArray._add_duration(secs[0], tt_et_l, -rtlt)
        try:
            return scb.RampTableManager.frequency_at(
                ramp, t1_sec, doys[0], years[0], t1_sec_l
            )
        except scb.RampTableManager.RampCoverageError:
            # The callers' ramp integration meets the same gap at t1 and flags the
            # observation as an outlier, so the time-tag frequency only has to
            # keep the light-time solution going
            return solar_corona_dict["ul_freq"]

    @staticmethod
    def _compute_precise_RTLT(
        sc_spice_id: str,
        gs_spice_id: str,
        tt_et: float,
        electronic_delays_dict: dict,
        solar_corona_dict: dict,
        delta_partials: ArrayWUnits,
        gs_delta_awf: ArrayWFrame | None = None,
        tt_et_l: float = 0.0,
        rlt_bodies: tuple = None,
        light_time_settings: "dict | None" = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Method to compute the precise Round-Trip Light-Time (RTLT) between a gs-based DSN antenna and a spacecraft.
        This method computes the RTLT for a 2-way path between:

        - A DSN antenna transmitting a signal at t1
        - A spacecraft reflecting a signal at t2
        - A DSN antenna receiving a signal at t3

           implement the necessary time delays between transmittion from a gs at (t1) and reception at the same gs at (t3).

        Parameters
        ----------
        sc_spice_id : str
            spice id string of the spacecraft

        gs_spice_id : str
            spice id string of the ground station

        tt_et : float
            time-tag in ephemeris time of the reception time at the gs (t3)

        electronic_delays_dict : dict
            dictionary of electronic delays

        solar_corona_dict : dict
            Solar-corona parameters: the turnaround ratio ``"M2"``, the uplink frequency
            at the time tag ``"ul_freq"`` [Hz] (zero turns the corona delay off) and,
            optionally, the ``"ramp"`` table as :meth:`_ramp_table` builds it. With a
            ramp, the delay is computed at the transmit frequency at t1, see
            :meth:`_corona_uplink_frequency`; without, at ``ul_freq``. An optional
            ``"coefficient"`` [us GHz^2 m^2] replaces the default of
            :meth:`MediaCorrections._compute_SC_lt`.

        delta_partials : ArrayWUnits
            Offset added to the spacecraft state at t2, ``values[0]`` being the 6-vector
            [km, km/s]. Used to generate numeric partials; zero for a nominal computation.

        gs_delta_awf : ArrayWFrame | None
            Ground-station location offset as stored in the state (``gs_delta_location_ECEF_[tag]``,
            with its own frame and units). If not None, it is added to the geocentric station position
            at t1 and t3 before the SSB space-time transform (Moyer 2000, eq. 4-10), so it enters the
            light times and the ET-TAI terms.

        tt_et_l : float, optional
            Low part of ``tt_et`` [s]. Epochs t1, t2, t3 are carried internally as (high, low)
            pairs, see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`, and ephemerides are evaluated with
            :meth:`~scarabaeus.SpiceManager.get_state_precise`.

        rlt_bodies : tuple, optional
            ``(spice_id, GM)`` pairs of the bodies other than the Sun in the relativistic
            light-time term (Moyer 2000, eq. 8-55). Defaults to :attr:`Constants.RLT_BODIES`;
            an empty tuple leaves only the Sun.

        light_time_settings : dict, optional
            Convergence tolerance and iteration caps of the solution, any of the keys
            of :attr:`LIGHT_TIME_SETTINGS`; missing keys take the default.
            See :meth:`_resolve_light_time_settings`.

        station_correction : callable, optional
            Function of epoch returning the station's six-component ITRF93
            correction in km and km/s, see :meth:`_station_state_correction`.
            It carries the estimated station location and the Earth-orientation
            and body-orientation deltas, and is applied wherever the chain reads
            the station: t1 and t3, both legs and the ET-TAI terms. Defaults to
            None, no correction. A caller passing it should leave
            ``gs_delta_awf`` None, or the location offset is applied twice.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        rtlt : float
            Sum of the rtlt delays components as a single delay value in second between t1 and t3

        rtlt_parts : np.array(17,1)
            rtlt decomposed in its components, expressed in seconds

        t2 : float
            Reflection epoch at the spacecraft [ET s], from the light-time solution
            (t3 minus the down-leg Newtonian and relativistic terms).

        t1 : float
            Transmission epoch at the ground station [ET s], from the light-time solution
            (t2 minus the up-leg Newtonian and relativistic terms).

        Notes
        -----
        The returned t2 and t1 are the nearest floats to the internal (high, low) pairs. As in
        Moyer 2000 eq. 11-7, the solar corona, antenna and electronic delays enter ``rtlt`` only,
        not the epochs.

        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """
        # Define constants to be used within this method
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        # Define debug flag for CN method
        CN_flag = False  # Internal flag used to debug and use Converged or Un-Converged computation for Newtonian lightime
        settings = Measurement._resolve_light_time_settings(light_time_settings)
        tol, max_it = settings["tolerance"], settings["max_iterations"]

        ### Moyer 32-steps algorithm ###
        # For reference, see Moyer 2000, Sec. 8.3.6

        ### down-leg portion (PART 1 of 2): from t3 (reception at the ground-station) to t2 (transmission from spacecraft) ###

        # Step (1) - Transform t3(ST) to t3(TAI)
        # We go from ET to TAI because we enter this function not in sec but in time, se we need to convert back to ST
        # ET-TAI with the same model as spice.unitim(tt_et, "ET", "TAI"), kept as a separate small
        # number so that the TAI epoch is not rounded.
        #
        # Using the same DELTET model here as observed_measurements used on the way in (via
        # spice.str2et) is deliberate: the two cancel, so the TAI recovered is exact to the
        # leap-second table and SPICE's ~30 us TDB-TT error never reaches the light-time
        # solution. The accurate TAI->TDB conversion is step (2) below. If tt_et ever stops
        # being SPICE-model ET, this inverse stops being exact -- change both ends together.
        tt_tai_h, tt_tai_l = EpochArray._add_duration(
            tt_et,
            tt_et_l,
            -Measurement._compute_delta_et_tai(tt_et, gs_spice_id, 1)
            - electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["trn_tt_dly"]
            - electronic_delays_dict["array_delay"],
        )  # As per [1], Notes 2,3, and 4
        # tt_tai_i = (spice.unitim(tt_et, "ET", "TAI") - electronic_delays_dict["rcv_tt_dly"]) # NOTE: For debug purposes, 2.0
        # tt_tai_i = spice.unitim(tt_et, 'ET', 'TAI') # NOTE: For debug purposes, 1.0

        # Step (2) - Transform t3(TAI) to t3(ET)
        _, dt_et_tai_t3, r_stn_BC = Measurement._compute_et_from_tai(
            tt_tai_h, gs_spice_id, tt_tai_l, gs_delta_awf, settings["et_tai_iterations"],
            station_correction=station_correction,
            gamma=gamma,
        )
        t3_h, t3_l = EpochArray._add_duration(tt_tai_h, tt_tai_l, dt_et_tai_t3)

        # Step (3) - Skipped, done with Spice + RLT-corrected GS location

        # Step (4) - Compute quantities of interest for B's and S
        """
        Barycentric Frame Only: for each body B and the sun S (the ones considered in eq. 8-55 for RLT delays), 
        calculate vector and scalar distances from the body to the received at t3.
        The quantities "body_to_gs_t3" are computed below, once, with "_get_bodies_for_RLT".
        """
        ssb_to_earth_t3 = SpiceManager.get_state_precise(
            "Earth", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_sun_t3 = SpiceManager.get_state_precise(
            "10", t3_h, "J2000", "SSB", "None", t3_l
        ).values
        ssb_to_gs_t3_RLTframe = (
            np.concatenate((r_stn_BC, np.zeros(3))) + ssb_to_earth_t3
        )
        sun_to_gs_t3 = ssb_to_gs_t3_RLTframe - ssb_to_sun_t3
        # The bodies' end of the down leg is at the fixed t3
        body_to_gs_t3 = Measurement._get_bodies_for_RLT(
            t3_h, t3_l, ssb_to_gs_t3_RLTframe, bodies=rlt_bodies
        )
        # tt_et = tt_et # Method 2, DEBUG
        # r_stn_BC = SpiceManager.get_state(gs_spice_id,tt_et,'J2000','Earth','None') # Method 2, DEBUG
        # r_stn_BC = r_stn_BC[0:3].values # Method 2, DEBUG

        # Step (5) - Skipped, computation performed in Barycentric Frame

        # Step (6) - setup IC for iteration loop on down-leg lightime
        lt_23 = 0  # prediction of the down-leg lightime for the first iteration
        t2_h, t2_l = EpochArray._add_duration(t3_h, t3_l, -lt_23)  # first estimate of t2

        # Steps (7, 8, 9, 10) - Skipped for free spacecraft, with 2W assumption (sranging or doppler), or because steps are substituted by Spice interpolation
        delta_t2 = np.inf  # at least one pass, whatever the tolerance
        max_it_counter = 0
        while (
            abs(delta_t2) > tol and max_it_counter < max_it
        ):  # step(16) - Stay within the while loop while the absolute value of delta_t2 is above the tolerance
            max_it_counter = max_it_counter + 1
            # Step (11) - get the spacecraft sstate in SSB at t2
            ssb_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    sc_spice_id, t2_h, "J2000", "SSB", "None", t2_l
                ).values
                + delta_partials.values[0]
            )
            # Step (12) - Compute quantities of interest for B's and S. Same as step (4): quantities of interest fo the bodies are computed inside the "_compute_RLT_BODIES_lt" method
            sun_to_sc_t2 = (
                SpiceManager.get_state_precise(
                    sc_spice_id, t2_h, "J2000", "10", "None", t2_l
                ).values
                + delta_partials.values[0]
            )
            # Step (13) - Skipped, computation performed in Barycentric Frame and not geocentric

            # Step (14) - Generate r_23_vec, r_23_vec_dot and r_23 and compute RLT_23. This computation is split in two part and not performed originally with the 3 terms from Moyer 2000 eq. 8-55.
            # Converged newtonian (CN) component of Moyer 2000 eq. 8-55s (1st term)
            if CN_flag:
                # Note: this term is different than the one in Moyer 2000 eq. 8-55, becuase it's already internally converged.
                dt_CN_dwn = Measurement._compute_CN_lt(
                    ssb_to_gs_t3_RLTframe[:3],
                    sc_spice_id,
                    t3_h,
                    delta_partials.values[0],
                    "down",
                    t3_l,
                    gamma=gamma,
                )
            else:
                dt_CN_dwn = (
                    np.linalg.norm(ssb_to_sc_t2[:3] - ssb_to_gs_t3_RLTframe[:3]) / c
                )
                # Sun and Body Relativistic (RLT) components of Moyer 2000 eq. 8-55 (2nd and 3rd terms)
            dt_GR_dwn, r23_vec = Measurement._compute_RLT_lt(
                "RLT",
                sun_to_sc_t2,
                sun_to_gs_t3,
                t2_h,
                t3_h,
                sc_spice_id,
                ssb_to_gs_t3_RLTframe,
                "dwn",
                delta_partials.values[0],
                t2_t1_l=t2_l,
                t3_t2_l=t3_l,
                bodies=rlt_bodies,
                body_to_gs=body_to_gs_t3,
                gamma=gamma,
            )
            r23 = np.linalg.norm(r23_vec[:3])
            # Get the velocity of the spacecraft at reflection time t2 in SSB space-time frame
            r_dot_2_C_t2 = ssb_to_sc_t2[3:6]
            # Compute p_dot from Moyer 2000 eq. 8-60
            p_dot_23 = np.dot(r23_vec[:3] / r23, r_dot_2_C_t2)
            # Step (15) - calculate the linear differential correction delta_t2, Moyer 2000 eq. 8-72
            # t3 - t2 from the pairs: the high-part difference is exact (Sterbenz)
            delta_t2 = ((t3_h - t2_h) + (t3_l - t2_l) - dt_CN_dwn - dt_GR_dwn) / (
                1 - p_dot_23 / c
            )  # with i = 2, j = 3. Note: dt_CN_dwn is converged with respecto the un-converged term in 8-72
            t2_h, t2_l = EpochArray._add_duration(t2_h, t2_l, delta_t2)  # update t2

        # Step (17) - Map everything calculated at the last estimate of the converged value for t2(ET). From Moyer, this is substituted by spice calls
        ssb_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "SSB", "None", t2_l
            ).values
            + delta_partials.values[0]
        )
        sun_to_sc_t2 = (
            SpiceManager.get_state_precise(
                sc_spice_id, t2_h, "J2000", "10", "None", t2_l
            ).values
            + delta_partials.values[0]
        )
        # The bodies' end of the up leg is at the converged t2: compute it once
        body_to_sc_t2 = Measurement._get_bodies_to_sc_for_RLT(
            sc_spice_id, t2_h, t2_l, delta_partials.values[0], bodies=rlt_bodies
        )

        # Step (18) - Using the mapping quantities from Step (17), repeat steps 11 to 14
        if CN_flag:
            dt_CN_dwn = Measurement._compute_CN_lt(
                ssb_to_gs_t3_RLTframe[:3],
                sc_spice_id,
                t3_h,
                delta_partials.values[0],
                "down",
                t3_l,
                gamma=gamma,
            )
        else:
            dt_CN_dwn = (
                np.linalg.norm(ssb_to_sc_t2[:3] - ssb_to_gs_t3_RLTframe[:3]) / c
            )
        dt_GR_dwn, r23_vec = Measurement._compute_RLT_lt(
            "RLT",
            sun_to_sc_t2,
            sun_to_gs_t3,
            t2_h,
            t3_h,
            sc_spice_id,
            ssb_to_gs_t3_RLTframe,
            "dwn",
            delta_partials.values[0],
            t2_t1_l=t2_l,
            t3_t2_l=t3_l,
            bodies=rlt_bodies,
            body_to_gs=body_to_gs_t3,
            body_to_sc=body_to_sc_t2,
            gamma=gamma,
        )

        # Step (19) - Skipped since 2W (sranging or doppler) assumption
        ### up-leg portion (PART 2 of 2): from t2 (reception from spacecraft) to t1 (transmission from ground station) ###

        # Step (20) - Get the first estimate for t1 using the converged delta time from t3 to t2
        r2_vec = ssb_to_sc_t2
        r3_vec = ssb_to_gs_t3_RLTframe
        r23_vec = r3_vec - r2_vec
        r23 = np.linalg.norm(r23_vec[:3])
        r_dot_E = np.dot(
            r23_vec[:3] / r23, ssb_to_earth_t3[3:6]
        )  # Moyer 2000, eq. 8-79
        lt_12 = (dt_CN_dwn + dt_GR_dwn) * (1 - 2 * r_dot_E / c)
        t1_h, t1_l = EpochArray._add_duration(t2_h, t2_l, -lt_12)  # first estimate of t1

        # Step (21) - Skipped, done with Spice + RLT-corrected GS location
        # Step (22) - Skipped, the transmitter is not an Earth satellite

        delta_t1 = np.inf  # at least one pass, whatever the tolerance
        max_it_counter = 0
        while (
            abs(delta_t1) > tol and max_it_counter < max_it
        ):  # step(29) - Stay within the while loop while the absolute value of delta_t1 is above the tolerance
            max_it_counter = max_it_counter + 1
            ssb_to_earth_t1 = SpiceManager.get_state_precise(
                "Earth", t1_h, "J2000", "SSB", "None", t1_l
            ).values
            # Steps (23, 24) - Compute ground-station state in SSB space-time relativistic frame at t1
            earth_to_gs_t1_RLTframe = Measurement._compute_GS_in_SSB_space_time_RLT(
                gs_spice_id, t1_h, t1_l, gs_delta_awf, ssb_to_earth=ssb_to_earth_t1,
                station_correction=station_correction,
                gamma=gamma,
            )
            # Step (25) - Compute quantities of interest for B's and S. Same as step (4): quantities of interest fo the bodies are computed inside the "_compute_RLT_BODIES_lt" method
            ssb_to_sun_t1 = SpiceManager.get_state_precise(
                "10", t1_h, "J2000", "SSB", "None", t1_l
            ).values
            ssb_to_gs_t1_RLTframe = ssb_to_earth_t1 + earth_to_gs_t1_RLTframe
            sun_to_gs_t1 = ssb_to_gs_t1_RLTframe - ssb_to_sun_t1
            # Step (26) - Skipped, coputation in Barycentric already perforned at step 25
            # Step (27) - Generate r_12_vec, r_12_vec_dot and r_12 and compute RLT_12. This computation is split in two part and not performed originally with the 3 terms from Moyer 2000 eq. 8-55.
            # Converged newtonian (CN) component of Moyer 2000 eq. 8-55s (1st term)
            if CN_flag:
                # Note: this term is different than the one in Moyer 2000 eq. 8-55, becuase it's already internally converged.
                dt_CN_up = Measurement._compute_CN_lt(
                    ssb_to_sc_t2[:3],
                    gs_spice_id,
                    t2_h,
                    delta_partials.values[0],
                    "up",
                    t2_l,
                    gs_delta_awf,
                    station_correction=station_correction,
                    gamma=gamma,
                )
            else:
                dt_CN_up = (
                    np.linalg.norm(ssb_to_sc_t2[:3] - ssb_to_gs_t1_RLTframe[:3]) / c
                )
            # Sun and Body Relativistic (RLT) components of Moyer 2000 eq. 8-55 (2nd and 3rd terms)
            dt_GR_up, r12_vec = Measurement._compute_RLT_lt(
                "RLT",
                sun_to_gs_t1,
                sun_to_sc_t2,
                t1_h,
                t2_h,
                sc_spice_id,
                ssb_to_gs_t1_RLTframe,
                "up",
                delta_partials.values[0],
                t2_t1_l=t1_l,
                t3_t2_l=t2_l,
                bodies=rlt_bodies,
                body_to_sc=body_to_sc_t2,
                gamma=gamma,
            )
            r12 = np.linalg.norm(r12_vec[:3])
            # Get the velocity of the spacecraft at reflection time t2 in SSB space-time frame
            r_dot_1_C_t1 = ssb_to_gs_t1_RLTframe[3:6]
            # Compute p_dot from Moyer 2000 eq. 8-60
            p_dot_12 = np.dot(r12_vec[:3] / r12, r_dot_1_C_t1)
            # Step (28) - calculate the linear differential correction delta_t1, Moyer 2000 eq. 8-72
            delta_t1 = ((t2_h - t1_h) + (t2_l - t1_l) - dt_CN_up - dt_GR_up) / (
                1 - p_dot_12 / c
            )
            t1_h, t1_l = EpochArray._add_duration(t1_h, t1_l, delta_t1)  # update t1

        # Step (30) - Map everything calculated at the last estimate of the converged value for t1(ET). From Moyer, this is substituted by spice calls
        ssb_to_earth_t1 = SpiceManager.get_state_precise(
            "Earth", t1_h, "J2000", "SSB", "None", t1_l
        ).values
        ssb_to_sun_t1 = SpiceManager.get_state_precise(
            "10", t1_h, "J2000", "SSB", "None", t1_l
        ).values

        # Step (31) - Using the mapping quantities from Step (30), repeat steps 24 to 27
        earth_to_gs_t1_RLTframe = Measurement._compute_GS_in_SSB_space_time_RLT(
            gs_spice_id, t1_h, t1_l, gs_delta_awf, ssb_to_earth=ssb_to_earth_t1,
            station_correction=station_correction,
            gamma=gamma,
        )
        ssb_to_gs_t1_RLTframe = ssb_to_earth_t1 + earth_to_gs_t1_RLTframe
        sun_to_gs_t1 = ssb_to_gs_t1_RLTframe - ssb_to_sun_t1
        if CN_flag:
            dt_CN_up = Measurement._compute_CN_lt(
                ssb_to_sc_t2[:3],
                gs_spice_id,
                t2_h,
                delta_partials.values[0],
                "up",
                t2_l,
                gs_delta_awf,
                station_correction=station_correction,
                gamma=gamma,
            )
        else:
            dt_CN_up = (
                np.linalg.norm(ssb_to_sc_t2[:3] - ssb_to_gs_t1_RLTframe[:3]) / c
            )
        dt_GR_up, r12_vec = Measurement._compute_RLT_lt(
            "RLT",
            sun_to_gs_t1,
            sun_to_sc_t2,
            t1_h,
            t2_h,
            sc_spice_id,
            ssb_to_gs_t1_RLTframe,
            "up",
            delta_partials.values[0],
            t2_t1_l=t1_l,
            t3_t2_l=t2_l,
            bodies=rlt_bodies,
            body_to_sc=body_to_sc_t2,
            gamma=gamma,
        )

        # Step (32) - convert t1(ET) to t1(TAI) via Moyer eq. 2-23
        dt_et_tai_t1 = Measurement._compute_delta_et_tai(
            t1_h, gs_spice_id, 3, gs_delta_awf, station_correction=station_correction,
            gamma=gamma,
        )
        dt_et_tai_t3 = Measurement._compute_delta_et_tai(
            t3_h, gs_spice_id, 3, gs_delta_awf, station_correction=station_correction,
            gamma=gamma,
        )

        t1_tai = (
            t1_h - dt_et_tai_t1
        )  # unused, but can be exported outside for debugging purposes

        # STEP (33) - Extra, compute extra effects on rtlt delay
        # Antenna effects
        dt_A_t3 = 0
        dt_A_t1 = 0

        # Transmitter and receiver delays
        dt_tau_D = (
            electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["dl_zheight_corr"]
            + electronic_delays_dict["scft_transpd_delay_s"]
        )  # Downlink delays
        dt_tau_U = (
            electronic_delays_dict["trn_tt_dly"]
            + electronic_delays_dict["ul_zheight_corr"]
        )  # Uplink delays

        ### OUTPUT handling ####
        rtlt_parts = np.zeros((17, 1))
        # Compute RTLT toghether
        # From Moyer 2000, pg. 11-11, eq.11-7
        rtlt_parts[0] = dt_CN_dwn  # Converged Newtonian, down-leg
        rtlt_parts[1] = dt_GR_dwn  # Relativistic, down-leg
        rtlt_parts[2] = dt_CN_up  # Converged Newtonian, up-leg
        rtlt_parts[3] = dt_GR_up  # Relativistic, up-leg
        rtlt_parts[4] = -dt_et_tai_t3  # ET-TAI at t3
        rtlt_parts[5] = dt_et_tai_t1  # ET-TAI at t1
        rtlt_parts[6] = 0  # UTC-TAI at t3, currently not modeled
        rtlt_parts[7] = 0  # UTC-TAI at t1, currently not modeled
        rtlt_parts[8] = 0  # UTC-ST at t3, currently not modeled
        rtlt_parts[9] = 0  # UTC-ST at t1, currently not modeled
        # Station-location term. Station location and Earth orientation are not
        # added here: gs_delta_awf and station_correction move the station at t1
        # and t3, so they reach the light time through the geometry and adding
        # them again would count them twice.
        rtlt_parts[10] = 0  # Rc, solve-for station location / EOP
        rtlt_parts[11] = dt_A_t3  # A at t3
        rtlt_parts[13] = dt_A_t1  # A at t1
        rtlt_parts[15] = dt_tau_D  # Downlink delay
        rtlt_parts[16] = dt_tau_U  # Uplink delay

        # Solar corona effects, last: they are at the transmit frequency at t1,
        # which is read from the ramp at t3 minus the rest of the round trip
        if solar_corona_dict["ul_freq"] != 0 and solar_corona_dict["M2"] != 0:
            f_T = Measurement._corona_uplink_frequency(
                solar_corona_dict, tt_et, tt_et_l, np.sum(rtlt_parts)
            )
            coefficient = solar_corona_dict.get("coefficient")
            rtlt_parts[12] = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2, sun_to_gs_t3, f_T * solar_corona_dict["M2"],
                coefficient=coefficient,
            )  # Solar Corona, down-leg
            rtlt_parts[14] = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2, sun_to_gs_t1, f_T, coefficient=coefficient
            )  # Solar Corona, up-leg

        return np.sum(rtlt_parts), rtlt_parts, t2_h, t1_h

    @staticmethod
    def _compute_geocentric_RTLT(
        sc_spice_id: str,
        gs_spice_id: str,
        tt_et: float,
        electronic_delays_dict: dict,
        solar_corona_dict: dict,
        delta_partials: ArrayWUnits,
        gs_delta_awf: ArrayWFrame | None = None,
        tt_et_l: float = 0.0,
        rlt_bodies: tuple = None,
        light_time_settings: "dict | None" = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Round-trip light time in the local geocentric space-time frame of reference
        (Moyer 2000, Sec. 8.3.1), for a spacecraft near the Earth.

        Same inputs, outputs and ``rtlt_parts`` layout as :meth:`_compute_precise_RTLT`,
        which it replaces when a real model is built with ``light_time_frame="geocentric"``.
        The differences from the barycentric solution are:

        * positions are geocentric (J2000 axes, Earth origin);
        * the coordinate time is TT, so ET - TAI is the constant
          :attr:`~scarabaeus.utils.Constants.Constants.TT_minus_TAI` at both stations and
          ``rtlt_parts[4]`` and ``[5]`` cancel;
        * only the Earth contributes to the relativistic delay (Moyer eq. 8-67):

          .. math::

              RLT_{ij} = \\frac{(1 + \\gamma)\\,\\mu_E}{c^3}
                  \\ln \\frac{r_i + r_j + r_{ij}}{r_i + r_j - r_{ij}}

        * the station position is not transformed by Moyer eq. 4-10. The spacecraft
          ephemeris is barycentric, so its Earth-relative position is moved into the
          geocentric frame with the inverse of eq. 4-10 instead.

        Each leg is solved with the differential corrector of Moyer eq. 8-72, on
        (high, low) epoch pairs as in the barycentric solution.

        Parameters
        ----------
        sc_spice_id, gs_spice_id, tt_et, electronic_delays_dict, solar_corona_dict, delta_partials, gs_delta_awf, tt_et_l, station_correction
            As in :meth:`_compute_precise_RTLT`.

        rlt_bodies : tuple, optional
            Accepted so both solvers share one signature, and ignored: in the
            geocentric frame only the Earth enters the relativistic delay
            (Moyer 2000, eq. 8-67).

        light_time_settings : dict, optional
            As in :meth:`_compute_precise_RTLT`.

        station_correction : callable, optional
            Function of epoch returning this station's six-component ITRF93
            correction in km and km/s, added to the station state. Defaults
            to ``None``, no correction.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        rtlt : float
            Sum of ``rtlt_parts`` [s].

        rtlt_parts : np.array(17,1)
            As in :meth:`_compute_precise_RTLT`.

        t2, t1 : float
            Reflection and transmission epochs [TDB s], so that callers can use them
            exactly as the barycentric ones.

        Notes
        -----
        The ephemerides are barycentric (TDB), so each TT event is read at
        :math:`TDB = TT + (TDB - TT)_{\\oplus} + \\mathbf{v}_E \\cdot \\mathbf{r} / c^2`,
        with :math:`(TDB - TT)_{\\oplus}` the value at the geocentre and
        :math:`\\mathbf{r}` the geocentric position of the participant. The geocentric term comes from the Moyer
        eq. 2-26 series at t3 with the station term removed. Its change over one round
        trip is below 1e-9 s and is not followed.

        The Sun's relativistic delay of the barycentric solution, about 2e-8 of the
        range, is absorbed here by the potential scale of the inverse eq. 4-10
        transform. What is left out is the tidal part of the external potential and
        the delay of every body other than the Earth. Against the barycentric solution
        on the same trajectory from DSS-14, the two-way range differs by less than
        0.1 mm in low Earth orbit, 0.2 mm at GEO and 3 mm for a low lunar orbiter.
        Without the spacecraft transform the difference would be about 3e-8 of the
        range plus 0.16 m at the station: 0.1 m in low Earth orbit, 1 m at GEO and
        10 m at lunar distance.
        """
        c = constants.c.values  # Speed of light in vacuum [km/sec]
        mu_E = constants.EARTH.GM.values  # [km^3/sec^2]
        if gamma is None:
            gamma = constants.gamma_GR.values
        tt_tai = constants.TT_minus_TAI.values  # [s]
        mu_sun = constants.SUN.GM.values
        Ltilde_GR = constants.L_tilde_GR.values
        d_sc = delta_partials.values[0]
        settings = Measurement._resolve_light_time_settings(light_time_settings)
        tol, max_it = settings["tolerance"], settings["max_iterations"]

        def _rlt_earth(r_a, r_b, r_ab):
            # Earth-only relativistic light-time delay, Moyer 2000 eq. 8-67
            return (1 + gamma) * mu_E / c**3 * np.log(
                (r_a + r_b + r_ab) / (r_a + r_b - r_ab)
            )

        # Step (1) - t3(ST) to t3(TAI), as in the barycentric solution
        tai3_h, tai3_l = EpochArray._add_duration(
            tt_et,
            tt_et_l,
            -Measurement._compute_delta_et_tai(tt_et, gs_spice_id, 1)
            - electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["trn_tt_dly"]
            - electronic_delays_dict["array_delay"],
        )
        # Step (2) - t3(TAI) to t3(TT)
        t3_h, t3_l = EpochArray._add_duration(tai3_h, tai3_l, tt_tai)

        # TDB - TT at the geocentre: the station's ET - TAI (Moyer eq. 2-26) less
        # its station term v_E · r_A / c² and less TT - TAI
        _, dt_et_tai_stn, r_A_E = Measurement._compute_et_from_tai(
            tai3_h, gs_spice_id, tai3_l, gs_delta_awf, settings["et_tai_iterations"],
            station_correction=station_correction,
            gamma=gamma,
        )
        t3_tdb_h, t3_tdb_l = EpochArray._add_duration(tai3_h, tai3_l, dt_et_tai_stn)
        v_E = SpiceManager.get_state_precise(
            "Earth", t3_tdb_h, "J2000", "SSB", "None", t3_tdb_l
        ).values[3:6]
        dt_tdb_tt = dt_et_tai_stn - np.dot(v_E, r_A_E) / c**2 - tt_tai

        def _tdb(t_h, t_l, r_geo):
            return EpochArray._add_duration(
                t_h, t_l, dt_tdb_tt + np.dot(v_E, r_geo[:3]) / c**2
            )

        def _gs_state(t_h, t_l, r_geo):
            tdb_h, tdb_l = _tdb(t_h, t_l, r_geo)
            state = SpiceManager.get_state_precise(
                gs_spice_id, tdb_h, "J2000", "Earth", "None", tdb_l
            ).values.copy()
            state[:3] += Measurement._compute_gs_delta_J2000(gs_delta_awf, tdb_h)
            state += Measurement._station_correction_j2000(station_correction, tdb_h)
            return state

        # The spacecraft ephemeris is barycentric, read relative to the Earth. Move
        # its position into the geocentric frame with the inverse of Moyer 2000
        # eq. 4-10, the transform the barycentric solution applies to the station.
        sun_to_earth_t3 = SpiceManager.get_state_precise(
            "Earth", t3_tdb_h, "J2000", "Sun", "None", t3_tdb_l
        ).values
        bary_scale = (
            1 + Ltilde_GR + gamma * mu_sun / np.linalg.norm(sun_to_earth_t3[:3]) / c**2
        )

        def _sc_state(t_h, t_l, r_geo):
            tdb_h, tdb_l = _tdb(t_h, t_l, r_geo)
            state = (
                SpiceManager.get_state_precise(
                    sc_spice_id, tdb_h, "J2000", "Earth", "None", tdb_l
                ).values
                + d_sc
            )
            state[:3] = bary_scale * state[:3] + np.dot(v_E, state[:3]) / (2 * c**2) * v_E
            return state

        earth_to_gs_t3 = _gs_state(t3_h, t3_l, r_A_E)
        r3 = np.linalg.norm(earth_to_gs_t3[:3])

        # Down leg, t3 -> t2 (Moyer 2000 eq. 8-67 and 8-72)
        t2_h, t2_l = t3_h, t3_l
        earth_to_sc_t2 = _sc_state(t2_h, t2_l, np.zeros(3))
        delta_t2 = np.inf  # at least one pass, whatever the tolerance
        max_it_counter = 0
        while abs(delta_t2) > tol and max_it_counter < max_it:
            max_it_counter = max_it_counter + 1
            earth_to_sc_t2 = _sc_state(t2_h, t2_l, earth_to_sc_t2)
            r23_vec = earth_to_gs_t3[:3] - earth_to_sc_t2[:3]
            r23 = np.linalg.norm(r23_vec)
            dt_CN_dwn = r23 / c
            dt_GR_dwn = _rlt_earth(np.linalg.norm(earth_to_sc_t2[:3]), r3, r23)
            p_dot_23 = np.dot(r23_vec / r23, earth_to_sc_t2[3:6])  # Moyer 2000 eq. 8-60
            delta_t2 = ((t3_h - t2_h) + (t3_l - t2_l) - dt_CN_dwn - dt_GR_dwn) / (
                1 - p_dot_23 / c
            )
            t2_h, t2_l = EpochArray._add_duration(t2_h, t2_l, delta_t2)

        # Map to the converged t2
        earth_to_sc_t2 = _sc_state(t2_h, t2_l, earth_to_sc_t2)
        r2 = np.linalg.norm(earth_to_sc_t2[:3])
        r23_vec = earth_to_gs_t3[:3] - earth_to_sc_t2[:3]
        r23 = np.linalg.norm(r23_vec)
        dt_CN_dwn = r23 / c
        dt_GR_dwn = _rlt_earth(r2, r3, r23)

        # Up leg, t2 -> t1. First guess as Moyer 2000 eq. 8-79, with the
        # receiver's geocentric velocity in place of the Earth's barycentric one
        r_dot_3 = np.dot(r23_vec / r23, earth_to_gs_t3[3:6])
        lt_12 = (dt_CN_dwn + dt_GR_dwn) * (1 - 2 * r_dot_3 / c)
        t1_h, t1_l = EpochArray._add_duration(t2_h, t2_l, -lt_12)

        earth_to_gs_t1 = earth_to_gs_t3
        delta_t1 = np.inf  # at least one pass, whatever the tolerance
        max_it_counter = 0
        while abs(delta_t1) > tol and max_it_counter < max_it:
            max_it_counter = max_it_counter + 1
            earth_to_gs_t1 = _gs_state(t1_h, t1_l, earth_to_gs_t1)
            r12_vec = earth_to_sc_t2[:3] - earth_to_gs_t1[:3]
            r12 = np.linalg.norm(r12_vec)
            dt_CN_up = r12 / c
            dt_GR_up = _rlt_earth(np.linalg.norm(earth_to_gs_t1[:3]), r2, r12)
            p_dot_12 = np.dot(r12_vec / r12, earth_to_gs_t1[3:6])  # Moyer 2000 eq. 8-60
            delta_t1 = ((t2_h - t1_h) + (t2_l - t1_l) - dt_CN_up - dt_GR_up) / (
                1 - p_dot_12 / c
            )
            t1_h, t1_l = EpochArray._add_duration(t1_h, t1_l, delta_t1)

        # Map to the converged t1
        earth_to_gs_t1 = _gs_state(t1_h, t1_l, earth_to_gs_t1)
        r12 = np.linalg.norm(earth_to_sc_t2[:3] - earth_to_gs_t1[:3])
        dt_CN_up = r12 / c
        dt_GR_up = _rlt_earth(np.linalg.norm(earth_to_gs_t1[:3]), r2, r12)

        # Epochs returned in TDB, like the barycentric ones
        t2_tdb_h, t2_tdb_l = _tdb(t2_h, t2_l, earth_to_sc_t2)
        t1_tdb_h, t1_tdb_l = _tdb(t1_h, t1_l, earth_to_gs_t1)

        # Transmitter and receiver delays, as in the barycentric solution
        dt_tau_D = (
            electronic_delays_dict["rcv_tt_dly"]
            + electronic_delays_dict["dl_zheight_corr"]
            + electronic_delays_dict["scft_transpd_delay_s"]
        )
        dt_tau_U = (
            electronic_delays_dict["trn_tt_dly"]
            + electronic_delays_dict["ul_zheight_corr"]
        )

        rtlt_parts = np.zeros((17, 1))
        rtlt_parts[0] = dt_CN_dwn  # Converged Newtonian, down-leg
        rtlt_parts[1] = dt_GR_dwn  # Earth relativistic, down-leg
        rtlt_parts[2] = dt_CN_up  # Converged Newtonian, up-leg
        rtlt_parts[3] = dt_GR_up  # Earth relativistic, up-leg
        rtlt_parts[4] = -tt_tai  # ET-TAI at t3
        rtlt_parts[5] = tt_tai  # ET-TAI at t1
        # [6]-[9] UTC-TAI and UTC-ST, not modeled; [11], [13] antenna, not modeled.
        # gs_delta_awf and station_correction are in the geometry, so [10] stays 0.
        rtlt_parts[15] = dt_tau_D  # Downlink delay
        rtlt_parts[16] = dt_tau_U  # Uplink delay

        # Solar corona, with the same model and frequency as the barycentric solution
        if solar_corona_dict["ul_freq"] != 0 and solar_corona_dict["M2"] != 0:
            f_T = Measurement._corona_uplink_frequency(
                solar_corona_dict, tt_et, tt_et_l, np.sum(rtlt_parts)
            )
            sun_to_earth_t3 = SpiceManager.get_state_precise(
                "Earth", t3_tdb_h, "J2000", "10", "None", t3_tdb_l
            ).values
            sun_to_earth_t1 = SpiceManager.get_state_precise(
                "Earth", t1_tdb_h, "J2000", "10", "None", t1_tdb_l
            ).values
            sun_to_earth_t2 = SpiceManager.get_state_precise(
                "Earth", t2_tdb_h, "J2000", "10", "None", t2_tdb_l
            ).values
            sun_to_sc_t2 = sun_to_earth_t2 + earth_to_sc_t2
            coefficient = solar_corona_dict.get("coefficient")
            rtlt_parts[12] = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2,
                sun_to_earth_t3 + earth_to_gs_t3,
                f_T * solar_corona_dict["M2"],
                coefficient=coefficient,
            )  # Solar Corona, down-leg
            rtlt_parts[14] = scb.MediaCorrections._compute_SC_lt(
                sun_to_sc_t2,
                sun_to_earth_t1 + earth_to_gs_t1,
                f_T,
                coefficient=coefficient,
            )  # Solar Corona, up-leg

        return np.sum(rtlt_parts), rtlt_parts, t2_tdb_h, t1_tdb_h

    @staticmethod
    def _compute_et_from_tai(
        t: float,
        gs_spice_id: str,
        t_l: float = 0.0,
        gs_delta_awf: ArrayWFrame | None = None,
        iterations: int = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Method to convert a ground-station TAI reading to Ephemeris Time (ET), with the ET-TAI series of Moyer 2000 eq. 2-26.
        The series is evaluated at ET = TAI + (approximate ET-TAI), then again at the ET each evaluation returns.

        Parameters
        ----------
        t : float
            TAI epoch read by the ground-station clock

        gs_spice_id : str
            spice id string of the ground station

        t_l : float, optional
            Low part of ``t`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        gs_delta_awf : ArrayWFrame | None, optional
            Station location offset, see :meth:`_compute_GS_in_SSB_space_time_RLT`.

        iterations : int, optional
            Number of evaluations of the series. Defaults to
            ``LIGHT_TIME_SETTINGS["et_tai_iterations"]``.

        station_correction : callable, optional
            Function of epoch returning the station's ITRF93 correction in km and
            km/s, see :meth:`_compute_GS_in_SSB_space_time_RLT`.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        t_et : float
            ephemeris time (high part)

        dt: float
            et-tai correction

        X_A_E: np.ndarray
            Geocentric ground-station position in the SSB space-time frame (Moyer 2000, eq. 4-10)

        """
        # Evaluate the series at ET = TAI + (approximate ET-TAI), then again at
        # the ET each evaluation returns.
        if iterations is None:
            iterations = Measurement.LIGHT_TIME_SETTINGS["et_tai_iterations"]
        dt = Measurement._approx_et_minus_tai(t)
        for _ in range(iterations):
            et_h, et_l = EpochArray._add_duration(t, t_l, dt)
            states = [
                SpiceManager.get_state_precise(
                    target, et_h, "J2000", observer, "None", et_l
                ).values
                for target, observer in (
                    ("3", "Sun"),  # X_B_S
                    ("3", "SSB"),  # X_B_C
                    ("Earth", "3"),  # X_E_B
                    ("Earth", "SSB"),  # X_E_C
                    ("599", "Sun"),  # X_J_S
                    ("699", "Sun"),  # X_Sa_S
                    ("Sun", "SSB"),  # X_S_C
                )
            ]
            # Moyer 2000, eq. 4-10
            X_A_E = Measurement._compute_GS_in_SSB_space_time_RLT(
                gs_spice_id, et_h, et_l, gs_delta_awf, ssb_to_earth=states[3],
                station_correction=station_correction,
                gamma=gamma,
            )[:3]
            dt = Measurement._et_minus_tai_series(*states[:4], X_A_E, *states[4:])
        et_h, et_l = EpochArray._add_duration(t, t_l, dt)

        return et_h, dt, X_A_E

    @staticmethod
    def _compute_gs_delta_J2000(
        gs_delta_awf: ArrayWFrame | None, t: float
    ) -> np.ndarray:
        """
        Rotate a ground-station location offset into J2000.

        Parameters
        ----------
        gs_delta_awf : ArrayWFrame | None
            Station location offset as stored in the state (``gs_delta_location_ECEF_[tag]``),
            with its own frame and units.

        t : float
            Epoch of the rotation [ET seconds past J2000].

        Returns
        -------
        delta_J2000 : np.ndarray
            Offset in J2000 [km], shape (3,). Zeros if ``gs_delta_awf`` is None.
        """
        if gs_delta_awf is None:
            return np.zeros(3)
        A = SpiceManager.get_xfrm(
            frame_from=gs_delta_awf.frame.name,
            frame_to="J2000",
            epoch=t,
        )
        delta_km = gs_delta_awf.quantity.convert_to(km).values.reshape(3)
        return (A @ delta_km).reshape(3)

    @staticmethod
    def _compute_GS_in_SSB_space_time_RLT(
        gs_spice_id: str,
        t: float,
        t_l: float = 0.0,
        gs_delta_awf: ArrayWFrame | None = None,
        ssb_to_earth: np.ndarray = None,
        station_correction=None,
        gamma: float = None,
    ) -> np.ndarray:
        """
        Method to compute GS state vector in a Solar System Barycentric space-time relativistic frame of reference.

        Parameters
        ----------
        gs_spice_id : str
            spice id string of the ground station

        t : float
            time in ephemeris time

        t_l : float, optional
            Low part of ``t`` [s], see :meth:`~scarabaeus.timeAndFrame.EpochArray._add_duration`.

        gs_delta_awf : ArrayWFrame | None, optional
            Station location offset (ECEF, as stored in the state), added to the geocentric
            station position before the eq. 4-10 transform. See :meth:`_compute_gs_delta_J2000`.

        ssb_to_earth : np.ndarray, optional
            State of the Earth relative to the SSB at ``t`` in J2000 [km, km/s], for a
            caller that already has it. Read from SPICE when None.
        station_correction : callable, optional
            Function of ``t`` returning this station's six-component ITRF93
            correction in km and km/s, from
            :meth:`_station_state_correction`. This is the single point where
            an estimated station location or Earth-orientation delta enters the
            light-time chain, so the correction is applied exactly once no
            matter how many times the solver iterates. Defaults to ``None``,
            no correction.

        gamma : float, optional
            PPN :math:`\\gamma` of the light-time solution. Defaults to
            :attr:`Constants.gamma_GR`; a model estimating ``gamma_rel`` passes
            its value, see :meth:`_gamma_rel`.

        Returns
        -------
        rBC_state : np.ndarray
            State vector (velocity padded to zeros) ot the ground station

        """
        # Define constants
        mu_sun = constants.SUN.GM.values
        Ltilde_GR = constants.L_tilde_GR.values
        gamma_GR = constants.gamma_GR.values if gamma is None else gamma
        c = constants.c.values

        # Compute sun-earth and ssb-earth states
        sun_to_earth_t = SpiceManager.get_state_precise(
            "Earth", t, "J2000", "Sun", "None", t_l
        ).values
        if ssb_to_earth is None:
            ssb_to_earth = SpiceManager.get_state_precise(
                "Earth", t, "J2000", "SSB", "None", t_l
            ).values
        # Compute UE and VE
        UE = mu_sun / np.linalg.norm(sun_to_earth_t[:3])
        VE = ssb_to_earth[3:6]
        # GS state in geocentric J2000 coordinates (SPICE applies the ITRF93 -> J2000 rotation)
        rGC = SpiceManager.get_state_precise(
            gs_spice_id, t, "J2000", "Earth", "None", t_l
        ).values
        rGC = rGC + Measurement._station_correction_j2000(station_correction, t)
        if gs_delta_awf is not None:
            rGC = rGC[:3] + Measurement._compute_gs_delta_J2000(gs_delta_awf, t)
        # Apply relativistic correction to transform GS position in SSB space-time relativistic frame (Moyer 2000, eq. 4-10)
        rBC = (1 - Ltilde_GR - (gamma_GR * UE / c**2)) * rGC[:3] - (
            1 / (2 * c**2)
        ) * np.dot(VE, rGC[:3]) * VE
        # Padd rBC for easier math
        rBC_state = np.concatenate([rBC, np.zeros(3)])
        return rBC_state

    @staticmethod
    def _rtlt_failure_message(model, index, total, epoch_et, target, cause):
        """
        Describe an RTLT failure precisely enough to act on.

        The caller needs to know which measurement failed, not just that one
        did: the index identifies it within the model's own measurement arrays,
        which is the same index space the outlier flags use, so the point can be
        flagged and the run repeated. The epoch, station and target are what the
        underlying SPICE error has to be debugged against.

        Parameters
        ----------
        model : Measurement
            The measurement model that was computing.
        index : int
            Zero-based index of the measurement within this computation.
        total : int
            Number of measurements being computed.
        epoch_et : float
            Ephemeris time of the measurement.
        target : Body
            Target whose light time was being solved.
        cause : Exception
            The original failure.

        Returns
        -------
        str
        """
        # Never let the diagnostics hide the failure they describe.
        try:
            utc = SpiceManager.et2utc(epoch_et)
        except Exception:
            utc = "unavailable"
        station = getattr(model.instrument, "name", "unknown")
        station_id = getattr(model.instrument, "spice_id", "unknown")
        target_name = getattr(target, "name", "unknown")
        target_id = getattr(target, "spice_id", "unknown")

        return (
            f"{type(model).__name__}: round-trip light time failed for "
            f"measurement index {index} of {total} in dataset '{model.name}'.\n"
            f"  epoch   : {epoch_et} ET ({utc} UTC)\n"
            f"  station : {station} (spice_id {station_id})\n"
            f"  target  : {target_name} (spice_id {target_id})\n"
            f"  cause   : {type(cause).__name__}: {cause}\n"
            f"The index above addresses this model's measurement arrays, the "
            f"same indexing the outlier flags use, so this point can be flagged "
            f"and the computation rerun if the failure is expected."
        )

    @staticmethod
    def _generate_electronic_delays_dictionary(aux_dict, idx, enabled=True):
        """
        Generate a dictionary of electronic delays as per [1]

        Parameters
        ----------
        aux_dict : dict
            Dictionary of auxiliary parameters for the computed measurements.
        idx : int
            Measurement index; per-epoch entries of ``aux_dict`` are read at
            this row.
        enabled : bool, optional
            If False, every delay and calibration is returned as zero (the
            uplink frequency is kept). Defaults to True.

        Returns
        -------
        electronic_delays_dict : dict
            Electronic delays and calibrations of measurement ``idx``, with
            keys ``ul_freq`` [Hz], ``rcv_tt_dly``, ``trn_tt_dly``,
            ``ul_zheight_corr``, ``dl_zheight_corr`` [s], ``ul_stn_cal``,
            ``dl_stn_cal`` [RU], ``scft_transpd_delay`` [RU],
            ``scft_transpd_delay_s`` [s] and ``array_delay`` [s]. Invalid
            sentinels in [1] (``-1``, ``-99``) are mapped to a zero delay.

        Notes
        -----
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """

        # Conver the invalid delays to 0 [s] delays

        # Receive and transmit time-tag delays
        #
        # On the differing lengths within aux_dict: every key indexed here by
        # measurement index must run at the measurement count, which
        # _per_epoch_aux checks before the loop. In a TRK-2-34 file each SFDU
        # record has its own cadence and time tags (aux_SFDU_2 every second,
        # aux_SFDU_7 at the ranging cadence, ...), so _read_aux aligns them to
        # the measurement epochs by time tag (align_aux_to_epochs) rather than
        # by row. The ramp table entries (ramp_sec, ramp_day, ramp_freq,
        # ramp_rate, ramp_type) keep their own cadence and are consumed by
        # RampTableManager by time rather than by index.
        rcv_tt_dly_i = aux_dict["rcv_time_tag_delay"][idx]
        trn_tt_dly_i = aux_dict["transmit_time_tag_delay"][idx]  # [s]
        if rcv_tt_dly_i == -1:
            rcv_tt_dly_i = 0
        if trn_tt_dly_i == -1:
            trn_tt_dly_i = 0
        # Uplink and Downlink zheight corrections
        ul_zheight_corr_i = aux_dict["ul_zheight_corr"][idx]  # [s]
        dl_zheight_corr_i = aux_dict["dl_zheight_corr"][idx]  # [s]
        if ul_zheight_corr_i == -99.0:  # Invalid value for ul_zheight_corr (see [1])
            ul_zheight_corr_i = 0
        if dl_zheight_corr_i == -99.0:  # Invalid value for dl_zheight_corr (see [1])
            dl_zheight_corr_i = 0
        # Array delay: optional, per epoch or a single value for the pass.
        # Absent means no arraying equipment, i.e. a zero delay.
        array_delay = aux_dict.get("array_delay", 0.0)
        if isinstance(array_delay, (list, tuple, np.ndarray)):
            array_delay_i = array_delay[idx]  # [s]
        else:
            array_delay_i = array_delay  # [s]
        if array_delay_i in (-99, -99.0, -1, -1.0):
            array_delay_i = 0

        # Compact the electronic delays in a dictionary
        electronic_delays_dict = {
            "ul_freq": aux_dict["ul_freq"][
                idx
            ],  # uplink frequency at time tag [Hz], see [1] 3-70
            "rcv_tt_dly": rcv_tt_dly_i,  # transmitter timetag delay [s], see [1] 3-11
            "trn_tt_dly": trn_tt_dly_i,  # receiver timetag delay [s], see [1] 3-10
            "ul_zheight_corr": ul_zheight_corr_i,  # uplink zheight correction [s], see [1] 3-10
            "dl_zheight_corr": dl_zheight_corr_i,  # downlink zheight correction [s], see [1] 3-11
            "ul_stn_cal": aux_dict["ul_stn_cal"][
                idx
            ],  # uplink GS calibration [RU], see [1] 3-68
            "dl_stn_cal": aux_dict["dl_stn_cal"][
                idx
            ],  # downlink GS calibration [RU], see [1] 3-68
            "scft_transpd_delay": aux_dict[
                "scft_transpd_delay"
            ],  # spacecraft transponder delay correction [RU], see [1] 3-12
            "scft_transpd_delay_s": aux_dict[
                "scft_transpd_delay_s"
            ],  # spacecraft transponder delay correction [s]
            "array_delay": array_delay_i,  # time delay added to the path delay due to arraying equipment [s], see [1] 3-11
        }
        if not enabled:
            # Bypass: every delay and calibration is zero, the uplink
            # frequency (a signal property, not a delay) is kept.
            for key in electronic_delays_dict:
                if key != "ul_freq":
                    electronic_delays_dict[key] = 0.0
        return electronic_delays_dict

    def _aux_per_epoch(self, n_epochs: int) -> dict:
        """Validate ``computed_measurements_dict`` for ``n_epochs`` epochs.

        Parameters
        ----------
        n_epochs : int
            Number of measurement epochs.

        Returns
        -------
        dict
            Validated copy from :meth:`_per_epoch_aux` holding the required keys
            of :attr:`AUX_KEYS` and the optional keys present. The transponder
            delays ``scft_transpd_delay`` and ``scft_transpd_delay_s`` may be
            scalars.

        Raises
        ------
        Measurement.AuxDataError
            If the dictionary is None, a required or ramp-table key is missing, or a
            per-epoch entry has neither one value nor ``n_epochs`` values.
        """
        aux = self.computed_measurements_dict or {}
        return self._per_epoch_aux(
            self.computed_measurements_dict,
            n_epochs,
            keys=(
                *self.AUX_KEYS.required,
                *(k for k in self.AUX_KEYS.optional if k in aux),
            ),
            scalar_keys=("scft_transpd_delay", "scft_transpd_delay_s"),
        )

    @staticmethod
    def _per_epoch_aux(aux_dict, n_epochs, keys, scalar_keys=()):
        """Validate an aux_dict and return a copy with per-epoch arrays.

        Parameters
        ----------
        aux_dict : dict
            Auxiliary dictionary of a real measurement model.
        n_epochs : int
            Number of measurement epochs.
        keys : iterable of str
            Per-epoch keys. A single value is broadcast to every epoch; any
            other length must equal ``n_epochs``.
        scalar_keys : iterable of str, optional
            Keys that must be present but are used as they are.

        Returns
        -------
        dict
            Shallow copy of ``aux_dict`` whose ``keys`` are float arrays of
            length ``n_epochs``; the caller's dictionary is not modified.

        Raises
        ------
        Measurement.AuxDataError
            If ``aux_dict`` is None, a key is missing, a per-epoch entry has
            the wrong length, or the ramp table entries are empty or of
            different lengths.
        """
        if aux_dict is None:
            raise Measurement.AuxDataError(
                "computed_measurements_dict is None: load observations with "
                "observed_measurements (which reads the aux data) or build "
                "one with default_*_aux for synthetic measurements."
            )
        required = (*keys, *scalar_keys, *Measurement.RAMP_AUX_KEYS)
        missing = [key for key in required if key not in aux_dict]
        if missing:
            raise Measurement.AuxDataError(
                f"computed_measurements_dict is missing {missing}."
            )
        out = dict(aux_dict)
        for key in keys:
            values = np.atleast_1d(np.asarray(aux_dict[key], dtype=float))
            if values.size == 1:
                values = np.full(n_epochs, values[0])
            elif values.size != n_epochs:
                raise Measurement.AuxDataError(
                    f"computed_measurements_dict['{key}'] has {values.size} "
                    f"entries for {n_epochs} measurement epochs; align records "
                    "with their own cadence to the measurement epochs by time "
                    "tag (Measurement.align_aux_to_epochs)."
                )
            out[key] = values
        ramp_keys = Measurement.RAMP_AUX_KEYS + (
            (Measurement.RAMP_YEAR_KEY,)
            if aux_dict.get(Measurement.RAMP_YEAR_KEY) is not None else ()
        )
        ramp_lengths = {key: np.atleast_1d(aux_dict[key]).size for key in ramp_keys}
        if len(set(ramp_lengths.values())) != 1 or min(ramp_lengths.values()) < 1:
            raise Measurement.AuxDataError(
                f"ramp table entries must be non-empty and of equal length, got {ramp_lengths}."
            )
        return out

    @staticmethod
    def align_aux_to_epochs(source: dict, key: str, target: dict, invalid=None) -> np.ndarray:
        """Sample a TRK-2-34 record at the epochs of another record by time tag.

        Each SFDU record of a tracking file has its own cadence (e.g. the
        uplink z-height correction every second, the ranging aux at the
        ranging cadence), so its rows do not correspond to the measurement
        rows. The value in force at each target epoch is the latest source
        entry at or before it (zero-order hold); target epochs earlier than
        the whole source take its first entry.

        Parameters
        ----------
        source : dict
            Record holding ``year``, ``doy``, ``sec`` and ``key``.
        key : str
            Field of ``source`` to sample.
        target : dict
            Record holding the ``year``, ``doy``, ``sec`` time tags to sample
            at, typically the measurement record (e.g. ``trk["2W_doppler"]``).
        invalid : float, optional
            Source entries equal to this value are skipped (e.g. ``0`` for
            the uplink frequency of rows with no uplink).

        Returns
        -------
        np.ndarray
            ``source[key]`` at each target epoch, of length ``len(target["sec"])``.

        Raises
        ------
        Measurement.AuxDataError
            If ``source`` holds no valid entry of ``key``.
        """
        year0 = min(np.min(source["year"]), np.min(target["year"]))

        def time_tag(record):
            # Monotone across midnight and new year: a 366-day year only
            # spaces the years apart, which a zero-order hold allows.
            return (
                (np.asarray(record["year"], dtype=float) - year0)
                * 366.0 * constants.day_to_sec
                + np.asarray(record["doy"], dtype=float) * constants.day_to_sec
                + np.asarray(record["sec"], dtype=float)
            )

        t_source = time_tag(source)
        values = np.asarray(source[key])
        if invalid is not None:
            valid = values != invalid
            t_source, values = t_source[valid], values[valid]
        if values.size == 0:
            raise Measurement.AuxDataError(
                f"record holds no valid '{key}' entry to align."
            )
        order = np.argsort(t_source, kind="stable")
        t_source, values = t_source[order], values[order]
        idx = np.searchsorted(t_source, time_tag(target), side="right") - 1
        return values[np.clip(idx, 0, values.size - 1)]

    @staticmethod
    def _aux_field(trk_data: dict, meas_name: str, candidates, invalid=None, default=None):
        """One aux quantity at the rows of the measurement record of a tracking file.

        Parameters
        ----------
        trk_data : dict
            Tracking file contents.
        meas_name : str
            Measurement record (e.g. ``"2W_doppler"``); its rows set the output length.
        candidates : iterable of (str or None, str)
            ``(record, field)`` pairs tried in order; ``None`` names the
            measurement record itself, whose fields are row-aligned and taken
            as they are. A field of another record is sampled at the
            measurement epochs by time tag (:meth:`align_aux_to_epochs`),
            since each SFDU record has its own cadence.
        invalid : float, optional
            Sentinel skipped: rows holding it take the value in force from
            the valid rows, in the measurement record as in another record.
        default : float, optional
            Value broadcast to every row when no candidate is present, or
            when the first candidate present holds only ``invalid`` entries.

        Returns
        -------
        np.ndarray
            One value per row of ``trk_data[meas_name]``.

        Raises
        ------
        Measurement.AuxDataError
            If no candidate is present, or the first one present holds only
            ``invalid`` entries, and no default is given.
        """
        record = trk_data[meas_name]
        for record_name, field in candidates:
            if record_name is None and field in record:
                source = record
                values = np.asarray(record[field])
                if invalid is None or not np.any(values == invalid):
                    return values
            elif record_name is not None and field in trk_data.get(record_name, {}):
                source = trk_data[record_name]
            else:
                continue
            try:
                return Measurement.align_aux_to_epochs(
                    source, field, record, invalid=invalid
                )
            except Measurement.AuxDataError:
                if default is None:
                    raise
                break
        if default is not None:
            return np.full(len(record["sec"]), default)
        raise Measurement.AuxDataError(
            f"tracking file has none of {list(candidates)} for '{meas_name}' "
            "(None is the measurement record itself)."
        )

    def _read_dsn_aux(self, trk_data: dict, meas_name: str,
                      sc_transponder_delay_RU: float = 0.0, band: str = "X") -> dict:
        """Aux quantities shared by the real DSN models, from a tracking file.

        Reads a TRK-2-34 file whose fields sit either in the measurement
        record (the layout of a file written by
        :meth:`generate_measurements`, and of the uplink fields of some
        converted files) or in the auxiliary SFDU records, which are aligned
        to the measurement epochs by time tag (:meth:`_aux_field`).

        Parameters
        ----------
        trk_data : dict
            Tracking file contents.
        meas_name : str
            Measurement record (e.g. ``"2W_doppler"``).
        sc_transponder_delay_RU : float, optional
            Spacecraft transponder delay [RU]. Zero takes the delay of the
            measurement record (``scft_transpd_delay``) when it has one.
        band : str, optional
            DSN band whose range-unit reference frequency converts the
            transponder delay to seconds (see :meth:`transponder_delay_seconds`),
            and whose range unit stands in when the file has no exciter
            scalars. Defaults to ``"X"``.

        Returns
        -------
        dict
            The uplink frequency, station calibrations, zenith height
            corrections, time-tag delays and ``array_delay``, one entry per
            measurement; ``exc_scalar_num`` and ``exc_scalar_den``
            when the file has them; the scalars ``RU`` (range unit, from the
            exciter scalars or else the band's), ``scft_transpd_delay`` [RU]
            and ``scft_transpd_delay_s`` [s]; and the ramp table entries
            (:attr:`RAMP_AUX_KEYS`, and
            :attr:`RAMP_YEAR_KEY` when the file
            gives years) from SFDU 9.

        Raises
        ------
        Measurement.AuxDataError
            If a required field is in no record, the ramp table is missing,
            or the transponder delay varies within the pass.
        ValueError
            If ``band`` is not a supported DSN band.
        """
        def field(candidates, invalid=None, default=None):
            return Measurement._aux_field(
                trk_data, meas_name, candidates, invalid=invalid, default=default
            )

        aux = {
            # Zero marks a row with no uplink reported
            "ul_freq": field(
                [
                    (None, "ul_freq"),
                    ("aux_SFDU_7", "ul_freq"),
                    ("2W_sranging", "ul_freq"),
                    ("aux_SFDU_9", "ramp_freq"),
                ],
                invalid=0.0,
            ),
            "ul_stn_cal": field(
                [(None, "ul_stn_cal"), ("aux_SFDU_7", "ul_stn_cal"), ("2W_sranging", "ul_stn_cal")],
                invalid=0.0,
                default=0.0,
            ),
            "dl_stn_cal": field(
                [(None, "dl_stn_cal"), ("aux_SFDU_7", "dl_stn_cal"), ("2W_sranging", "dl_stn_cal")],
                invalid=0.0,
                default=0.0,
            ),
            "ul_zheight_corr": field(
                [(None, "ul_zheight_corr"), ("aux_SFDU_2", "ul_zheight_corr")]
            ),
            "dl_zheight_corr": field(
                [(None, "dl_zheight_corr"), ("aux_SFDU_3", "dl_zheight_corr")]
            ),
            "transmit_time_tag_delay": field(
                [(None, "transmit_time_tag_delay"), ("aux_SFDU_2", "transmit_time_tag_delay")]
            ),
            "rcv_time_tag_delay": field(
                [(None, "rcv_time_tag_delay"), ("aux_SFDU_3", "rcv_time_tag_delay")]
            ),
            "array_delay": field([(None, "array_delay")], default=0.0),
        }
        try:
            for key in ("exc_scalar_num", "exc_scalar_den"):
                aux[key] = field(
                    [(None, key), ("aux_SFDU_7", key), ("2W_sranging", key)]
                )
        except Measurement.AuxDataError:
            aux.pop("exc_scalar_num", None)

        # Range unit of the pass, from its exciter scalars (16 * num / den)
        band_spec = self._resolve_dsn_band(band)
        if "exc_scalar_num" in aux:
            aux["RU"] = 16.0 * float(aux["exc_scalar_num"][0]) / float(aux["exc_scalar_den"][0])
        else:
            aux["RU"] = band_spec.range_unit

        # Spacecraft transponder delay: a mission constant
        delay_RU = float(sc_transponder_delay_RU)
        record_delay = np.asarray(trk_data[meas_name].get("scft_transpd_delay", []), dtype=float)
        if delay_RU == 0.0 and record_delay.size:
            if np.unique(record_delay).size > 1:
                raise Measurement.AuxDataError(
                    f"'{meas_name}' scft_transpd_delay varies within the pass; "
                    "pass the transponder delay as sc_transponder_delay_RU."
                )
            delay_RU = float(record_delay[0])
        aux["scft_transpd_delay"] = delay_RU  # [RU]
        aux["scft_transpd_delay_s"] = Measurement.transponder_delay_seconds(
            delay_RU, band_spec.ru_reference_frequency.values, aux["RU"]
        )  # [s]

        # Transmitter frequency ramp table, at its own cadence
        if "aux_SFDU_9" not in trk_data:
            raise Measurement.AuxDataError(
                "tracking file has no ramp table (aux_SFDU_9)."
            )
        ramp = trk_data["aux_SFDU_9"]
        aux["ramp_sec"] = ramp["sec"]
        aux["ramp_day"] = ramp["doy"]
        aux["ramp_freq"] = ramp["ramp_freq"]
        aux["ramp_rate"] = ramp["ramp_rate"]
        aux["ramp_type"] = ramp["ramp_type"]
        if "year" in ramp:
            aux[Measurement.RAMP_YEAR_KEY] = ramp["year"]
        return aux

    @staticmethod
    def _progress_monitor(num_total: int, desc: str = "Processing", unit: str = "obs"):
        """
        Returns a tqdm-based progress updater function.

        Parameters
        ----------
        num_total : int
            Total number of items to process.
        desc : str
            Description for the tqdm bar.
        unit : str
            Unit label (e.g., 'obs', 'steps').

        Returns
        -------
        update_fn : Callable[[int], None]
            A function to call with the current index.
        close_fn : Callable[[], None]
            A function to finalize and close the bar.
        """
        pbar = tqdm(
            total=num_total,
            desc=desc,
            unit=unit,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]",
        )

        def update(current_index: int):
            pbar.n = current_index + 1
            pbar.refresh()

        def close():
            pbar.n = num_total
            pbar.refresh()
            pbar.close()

        return update, close
