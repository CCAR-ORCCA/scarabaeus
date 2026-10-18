# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    Measurement,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Spacecraft,
    Instrument,
    SpiceManager,
    RampTableManager,
    MediaCorrections,
    Noise,
    StateDefinition,
)
from typing import Optional

import scarabaeus.utils.NumpyWrapper as np
import copy as copy
import warnings

# ------------------#
#  Generate Units  #
# ------------------#
km, unitless = Units.get_units(["km", "unitless"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class SequentialRangingReal(Measurement):
    """ Models the real sequential ranging measurement model.

    Generates computed sequential ranging observables between a DSN ground
    station and a spacecraft using the precise round-trip light-time (RTLT)
    formulation, including relativistic corrections and optional media
    corrections.  Observables are expressed in Range Units (RU).

    **Count observable**

    The sequential ranging observable integrates the piecewise-linear uplink
    ramp frequency :math:`f_T(t)` over the round-trip window
    :math:`[t_1,\\,t_3]`, converts to Range Units, and reduces modulo the
    ranging-code period :math:`M`:

    .. math::

        \\rho_{RU} = \\Bigl(\\mathrm{RU}\\cdot
                    \\int_{t_1}^{t_3} f_T(t)\\,dt\\Bigr)\\bmod M
                    + b_{RU}

    where :math:`\\mathrm{RU}` is the Range Unit conversion factor,
    :math:`M` is the modulo length of the ranging code in RU, and
    :math:`b_{RU}` is an optional station range bias.

    **Media corrections**

    When a :class:`~scarabaeus.MediaCorrections` object is attached, the
    total one-way excess path delay (Moyer eq. 10-27) is

    .. math::

        \\Delta\\tau_{total} = \\frac{1}{c}\\bigl(
            p_{tropo,t_3} + p_{tropo,t_1}
          + p_{iono,t_3}  + p_{iono,t_1}
        \\bigr)

    and the media-corrected observable becomes (Moyer eq. 13-127):

    .. math::

        \\rho_{RU}^{corr} = \\rho_{RU}
          + \\mathrm{RU}\\cdot f_{T,1}\\cdot\\Delta\\tau_{total}

    Note that the ionospheric contribution enters with a **positive** sign
    (opposite to Doppler).

    Parameters
    ----------
    name : str
        Name of the measurement model.
    instrument : Instrument
        Ground station or antenna instrument (e.g. GroundStation).
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    meas_bias : float or ArrayWUnits, optional
        Range bias of the ground station [RU], used when the state vector
        holds no ``range_bias_RU`` of this station. One value applies to every
        pass; more hold one per pass, grouped by
        :meth:`~scarabaeus.Measurement.set_pass_epochs`. The same holds for a
        ``range_bias_RU`` state component. Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : MissionSequence, optional
        Mission sequence whose per-leg state definitions replace
        *state_definition*. Defaults to ``None``.
    epoch_sec : EpochArray, optional
        Epoch expressed in seconds of day. Defaults to ``None``.
    computed_measurements_dict : dict, optional
        Auxiliary tracking-data dictionary required by
        :meth:`computed_measurements` (only optional when the model is used
        for observed measurements alone). Holds the TRK-2-34 [1]_ auxiliary
        quantities of each tracking pass, typically extracted from the DSN
        TRK-2-34 archive SFDU records — see the OSIRIS-REx real-measurement
        tutorial (``advanced_RealMSR_OSIRIS_REx_OD``) for a worked example of
        building it. Unless noted otherwise, each key maps to a list/array
        with one entry per observation epoch:

        * ``'RU'`` : seconds-to-Range-Unit conversion factor
          (pre-frequency-multiplication), derived from the SFDU 7 exciter
          scalars as ``16 * exc_scalar_num / exc_scalar_den``.
        * ``'M'`` : sequential-ranging modulo [RU] (SFDU 7); computed range
          observables are reported modulo this value.
        * ``'M2_num'``, ``'M2_den'`` : numerator and denominator of the
          spacecraft transponder turn-around ratio ``M2`` (SFDU 16).
        * ``'ul_freq'`` : uplink frequency at the time tag [Hz]
          (SFDU 7, TRK-2-34 3-70).
        * ``'rcv_time_tag_delay'``, ``'transmit_time_tag_delay'`` : receiver
          and transmitter time-tag delays [s] (TRK-2-34 3-10/3-11); ``-1``
          marks an invalid entry and is treated as 0.
        * ``'ul_zheight_corr'``, ``'dl_zheight_corr'`` : uplink and downlink
          z-height corrections [s] (SFDUs 2 and 3); ``-99.0`` marks an
          invalid entry and is treated as 0.
        * ``'ul_stn_cal'``, ``'dl_stn_cal'`` : uplink and downlink
          ground-station calibrations [RU] (SFDU 7, TRK-2-34 3-68).
        * ``'scft_transpd_delay'`` : spacecraft transponder delay [RU],
          scalar (mission constant).
        * ``'scft_transpd_delay_s'`` : the same transponder delay expressed
          in seconds, scalar.
        * ``'array_delay'`` : arraying-equipment path delay [s], optional;
          ``-1`` / ``-99`` mark invalid entries and are treated as 0.
        * ``'exc_scalar_num'``, ``'exc_scalar_den'`` : exciter scalars
          (SFDU 7), optional; written with generated measurements so that
          ``'RU'`` can be read back.
        * ``'rng_type'``, ``'version_num'`` : ranging type and format version
          (SFDU 7), optional; carried into generated measurements.
        * ``'ramp_sec'``, ``'ramp_day'``, ``'ramp_freq'``, ``'ramp_rate'``,
          ``'ramp_type'`` : transmitter frequency ramp table (SFDU 9) —
          seconds of day, day of year, ramp reference frequency [Hz], ramp
          rate [Hz/s], and ramp type of each ramp record (one entry per ramp
          record, not per observation).
        * ``'ramp_year'`` : year of each ramp record (SFDU 9 ``year``),
          optional; needed when the ramp table or a round trip crosses a new
          year.

        Defaults to ``None``.
    partials_method : {"analytical", "fd"}, optional
        Which H-tilde implementation to use. ``"analytical"`` (the default) uses
        the Moyer partials, which differentiate the same light-time solution the
        forward model produced and reuse its cached endpoints, for roughly 380x
        less cost than ``"fd"`` on this model. ``"fd"`` finite-differences the
        real forward model -- one full forward run per state component, slow,
        but it differentiates exactly the observable being fitted. Defaults to
        ``None``, i.e. ``"analytical"``.
    eps_dict : dict, optional
        Perturbation magnitudes for finite-difference partials. Spacecraft
        entries are keyed ``dx`` to ``dvz``; station, Earth-orientation,
        zenith-delay and electronic-delay entries are keyed by StateDefinition
        parameter kind, e.g. ``gs_delta_location_CYL``, ``eop_delta_ut1`` or
        ``transponder_delay_bias``, and are read only by
        ``partials_method="fd"`` -- the analytical partials differentiate those
        parameters in closed form. Omitted keys fall back to the in-code
        defaults; see :attr:`~scarabaeus.Measurement.station_eps_defaults`.
        Defaults to ``None``.
    media_corrections : MediaCorrections, optional
        Media-correction object (troposphere / ionosphere). Defaults to ``None``.
    light_time_frame : {"barycentric", "geocentric"}, optional
        Space-time frame of the round-trip light-time solution. ``"barycentric"``
        (Moyer 2000, Sec. 8.3.6) is valid for any spacecraft. ``"geocentric"``
        (Moyer 2000, Sec. 8.3.1) keeps only the Earth's relativistic delay and is
        meant for spacecraft near the Earth; see
        :meth:`~scarabaeus.Measurement._compute_geocentric_RTLT` for what it
        neglects. Defaults to ``"barycentric"``.
    rlt_bodies : tuple, optional
        ``(spice_id, GM)`` pairs of the bodies other than the Sun in the
        relativistic light-time term. Defaults to ``None``, i.e.
        :attr:`Constants.RLT_BODIES`; an empty tuple leaves only the Sun. The
        default takes Mars, Jupiter and Saturn at their barycentres, which is
        not accurate enough for a Jupiter or Saturn orbiter: pass the planet
        centres instead (see :attr:`Constants.RLT_BODIES`).
    solar_corona : bool, optional
        Apply the solar-corona delay to the light time. Defaults to ``True``.
    electronic_delays : bool, optional
        Apply the time-tag, z-height, spacecraft transponder and arraying
        delays of ``computed_measurements_dict``. Defaults to ``True``.
    light_time_settings : dict, optional
        Convergence settings of the light-time solution, any of the keys of
        :attr:`Measurement.LIGHT_TIME_SETTINGS`: ``"tolerance"`` [s],
        ``"max_iterations"`` per leg and ``"et_tai_iterations"``. Missing keys
        take the default. Raise the tolerance for a spacecraft beyond about
        8 AU, where 1e-12 s is below the float spacing of the light time and a
        leg runs to ``max_iterations``. Defaults to ``None``, i.e. all defaults.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.
    scarabaeus.DopplerReal : Companion real Doppler model.
    scarabaeus.MediaCorrections : Tropospheric and ionospheric path-delay model.

    References
    ----------
    .. [1] "TRK-2-34, DSN Tracking System Data Archival Format",
           DSN No. 820-013, TRK-2-34, Rev. N.
    """

    AUX_KEYS = Measurement.TrackingAuxKeys(
        required=(
            "RU", "M", "M2_num", "M2_den", "ul_freq",
            "rcv_time_tag_delay", "transmit_time_tag_delay",
            "ul_zheight_corr", "dl_zheight_corr", "ul_stn_cal", "dl_stn_cal",
        ),
        optional=(
            "array_delay", "exc_scalar_num", "exc_scalar_den",
            "rng_type", "version_num",
        ),
        record_fields={
            "rng_modulo": "M",
            "scft_transpd_turn_num": "M2_num",
            "scft_transpd_turn_den": "M2_den",
        },
        outlier_flag="outlier_flag_sranging",
    )
    """Auxiliary-data layout of this model."""

    # -------------#
    # Constructor #
    # -------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        sigma: ArrayWUnits = None,
        meas_bias=None,
        state_definition=None,
        sequence_definition=None,
        epoch_sec=None,
        computed_measurements_dict=None,
        partials_method: str = None,
        eps_dict: dict = None,
        media_corrections: Optional[MediaCorrections] = None,
        light_time_frame: str = "barycentric",
        rlt_bodies: tuple = None,
        solar_corona: bool = True,
        electronic_delays: bool = True,
        light_time_settings: dict = None,
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument
        self._sigma = sigma
        self._meas_bias = meas_bias
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._epoch_sec = epoch_sec
        self._partials_method = Measurement._resolve_partials_method(
            partials_method, supported=type(self).PARTIALS_METHODS
        )
        self.compute_partials = self._compute_partials
        self.eps_dict = eps_dict
        self.computed_measurements = self.computed_measurements
        self._computed_measurements_dict = computed_measurements_dict
        self._measurement_type = "SequentialRangingReal"
        self._station_override = None

        # Light-time corrections
        self._rlt_bodies = rlt_bodies
        self._solar_corona = solar_corona
        self._electronic_delays = electronic_delays

        # Media correction
        if media_corrections is not None:
            self.media_corrections = media_corrections

        # Space-time frame of the light-time solution
        self._light_time_frame = Measurement._resolve_light_time_frame(light_time_frame)
        self._light_time_settings = Measurement._resolve_light_time_settings(
            light_time_settings
        )

    # ---------#
    # Methods #
    # ---------#
    def computed_measurements(
        self,
        target: Spacecraft,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        eps_override: dict = None,
    ) -> ArrayWFrame:
        """
        Generate the computed two-way DSN sequential ranging measurement.

        Parameters
        ----------
        target : Spacecraft
            target spacecraft for the two-way measurement model

        epoch_array : EpochArray
            An array of epochs (times) at which the measurements should be computed. If provided, it overrides epoch_start, epoch_end, and tstep.

        epoch_start : Epoch
            The starting epoch for the range measurement computations. Required if `epoch_array` is not provided.

        epoch_end : Epoch
            The ending epoch for the range measurement computations. Required if `epoch_array` is not provided.

        tstep : float
            The time step, in seconds, between consecutive range measurements. If `epoch_array` is not provided. Defaults to 1 second.

        frame : Frame
            The reference frame in which the measurement is generated.

        noisy: bool
            Whether to add AWGN at the model's sigma to the computed sequential ranging
            measurement. For simulating observations; leave False when computing
            predicted measurements for a filter. Defaults to False.

        eps_override: dict
            Dictionary of delta's to compute the numeric _partials

        Returns
        -------
        computed_meas : ArrayWFrame
            An ArrayWFrame object with the computed measurement values expressed in RU (adimensional)

        Raises
        ------
        Measurement.AuxDataError
            If ``computed_measurements_dict`` is missing, lacks a key, or holds
            a per-epoch entry whose length differs from the epochs.
        Measurement.MissingSigmaError
            If ``noisy`` is True and the model has no ``sigma``.
        Measurement.StationKernelRequiredError
            If the station is defined by body-fixed coordinates instead of an
            ephemeris.
        RuntimeError
            If a light-time solution fails (e.g. does not converge); it names
            the epoch and the target, and chains the original error.
        RampTableManager.RampTableYearError
            If the ramp table or a round trip crosses a new year and
            ``computed_measurements_dict`` has no ``'ramp_year'``.
        MediaCorrections.IncompleteTroposphereError
            If ``media_corrections`` has only one of the two troposphere files.
        Measurement.MissingPassIndexError
            If the bias holds one value per pass (more than one element) and no
            epoch-to-pass mapping is set.

        Notes
        -----
        This model has been developed following the details in Moyer 2000.
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        ``noisy=False`` gives the predicted observable, against which real
        observations are differenced. ``noisy=True`` adds white Gaussian
        noise of standard deviation ``sigma`` to it, which simulates
        observations.

        A round trip that starts before the first ramp record has no defined
        transmit frequency. Its computed value is NaN, with a warning, and
        :meth:`Measurement.generate_measurement_dataset` flags it as an
        outlier.

        Estimated electronic delays lengthen the round-trip light time on top of
        the tracking data's own: ``uplink_delay_bias_[tag]`` of this station at
        t1, ``transponder_delay_bias`` of *target* at t2 and
        ``downlink_delay_bias_[tag]`` of this station at t3. Each holds one
        value for every pass and applies even with ``electronic_delays=False``.

        """
        self._require_station_kernel()

        # 0. Parameters check
        # Accept either an explicit epoch_array or the documented
        # epoch_start/epoch_end/tstep interval form.
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )
        if epoch_array.system != "TDB":
            epoch_array = epoch_array.to(sys="TDB")
        if noisy and self.sigma is None:
            raise self.MissingSigmaError(
                "SequentialRangingReal: noisy=True needs the measurement sigma."
            )
        # 1. Initialization
        delta = self._build_perturbation_vector(
            eps_override
        )  # Delta dict for _partials
        media_corrections = self.media_corrections  # MediaCorrection object

        # Pull current bias + gs delta from state (if provided). The bias holds
        # one value for every pass or one per pass (see set_pass_epochs).
        bias = self._extract_bias("range_bias_RU", unitless)
        bias_RU = np.array(
            [
                0.0 if (b := self._bias_at_epoch(bias, epoch)) is None
                else float(np.squeeze(b.values))
                for epoch in epoch_array
            ]
        )  # [RU], one per epoch
        # Station location, Earth orientation and tides move the station inside
        # the light-time solution through this correction; the estimated zenith
        # delays are read once here and handed to the media model.
        station_correction = self._itrf_station_correction()
        zt, zi = self._zenith_delays()
        # Estimated electronic delays [s]: the station's at t1 and t3 and the
        # spacecraft's transponder delay at t2, one value for every pass.
        estimated_delays = self._estimated_electronic_delays(target)
        # Estimated scale on the solar-corona delay, 1 when not estimated.
        corona_scale = self._solar_corona_scale()

        # 2. Extract aux_dictionary quantities, one entry per epoch
        tt_et = np.atleast_1d(epoch_array.times.values)
        aux_dict = self._aux_per_epoch(len(tt_et))
        RU = aux_dict["RU"]  # RU conversion factor (pre-frequency multiplication factor)
        M = aux_dict["M"]  # Modulo M [RU]
        M2_num = aux_dict["M2_num"]  # Spacecraft turn-around ratio (NUMERATOR)
        M2_den = aux_dict["M2_den"]  # Spacecraft turn-around ratio (DENOMINATOR)
        M2 = M2_num / M2_den  # Spacecraft turn-around ratio

        # 3. Compute receive time t3 (et, and y,d,s), and group the ramp table,
        # read by the corona delay in the RTLT and by the integration below
        t3_year_all, t3_doy_all, t3_sec_all = SpiceManager.et2YDS(tt_et)
        ramp_data = Measurement._ramp_table(aux_dict)

        # 4. Compute precise RTLT
        rtlt = []
        t1 = []
        t2 = []
        t3 = epoch_array
        # Endpoint quantities kept for the analytical (Moyer) partials, which
        # differentiate this same light-time solution rather than re-solving it.
        # Recording them is pure observation: nothing below reads these lists, so
        # the computed measurement is unaffected whichever partials method runs.
        _gr_cache = []   # GR (down, up) delays per epoch [s]
        _sc_cache = []   # solar corona (down+up) per epoch [s]
        print("")
        print("Generating computed measurements...")
        update, close = Measurement._progress_monitor(
            num_total=len(tt_et), desc="2W Sequential Ranging"
        )
        # Loop through different t3; every aux entry is read at row i of the
        # measurement epochs (_aux_per_epoch checked the lengths)
        for i in range(len(tt_et)):
            tt_et_i = tt_et[i]
            # Pack a dictionary for the electronic delays of measurement i
            electronic_delays_dict = Measurement._generate_electronic_delays_dictionary(
                aux_dict, i, enabled=self._electronic_delays
            )
            # The estimated delays add to the tracking data's own, whether or
            # not those are applied: t1 to the uplink z-height correction, t2
            # to the transponder delay and t3 to the downlink z-height
            # correction. Those three enter only the uplink and downlink delays
            # of the light time; the time-tag delays would also move t3.
            electronic_delays_dict["ul_zheight_corr"] = (
                electronic_delays_dict["ul_zheight_corr"] + estimated_delays["t1"])
            electronic_delays_dict["scft_transpd_delay_s"] = (
                electronic_delays_dict["scft_transpd_delay_s"] + estimated_delays["t2"])
            electronic_delays_dict["dl_zheight_corr"] = (
                electronic_delays_dict["dl_zheight_corr"] + estimated_delays["t3"])
            # Pack a dictionary for Solar Corona delays (a zero uplink
            # frequency turns the corona delay off; the ramp gives the
            # frequency at t1)
            solar_corona_dict = {
                "M2": M2[i],
                "ul_freq": aux_dict["ul_freq"][i] if self._solar_corona else 0.0,
                "ramp": ramp_data,
            }
            # Generate the rtlt's

            try:
                rtlt_i, parts_i, t2_i, t1_i = self._rtlt_solver()(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=tt_et_i,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta,
                    station_correction=station_correction,
                    rlt_bodies=self._rlt_bodies,
                    light_time_settings=getattr(self, "_light_time_settings", None),
                )
                #
            except Exception as e:
                # Previously this substituted rtlt = 0 and carried on, which
                # produced a computed measurement of zero that is
                # indistinguishable downstream from a real one: it reaches the
                # residuals, the filter and the solution as though valid.
                raise RuntimeError(
                    Measurement._rtlt_failure_message(
                        model=self,
                        index=i,
                        total=len(tt_et),
                        epoch_et=tt_et_i,
                        target=target,
                        cause=e,
                    )
                ) from e
            #
            rtlt_i, sc_i = Measurement._apply_solar_corona_scale(rtlt_i, parts_i, corona_scale)
            rtlt.append(rtlt_i)
            t2.append(t2_i)
            t1.append(t1_i)
            # Endpoint detail for the analytical partials. The parts vector packs
            # the per-leg delay breakdown: [1] and [3] are the down- and up-leg GR
            # (Shapiro) delays, [12] and [14] the down- and up-leg solar corona,
            # cached unscaled as the partial of the light time by its scale.
            parts_i = np.asarray(parts_i).ravel()
            _gr_cache.append((float(parts_i[1]), float(parts_i[3])))
            _sc_cache.append(sc_i)
            update(i)
            #
        close()

        rtlt = np.array(rtlt)
        t1 = np.array(t1)
        t2 = np.array(t2)

        # 5. Define begin and end intervals for integration with ramp tables,
        # t1 = t3 - RTLT in seconds of day as a (high, low) pair (a float
        # second of day resolves only ~7e-12 s, f_T*7e-12 ~ 0.05 cycles)
        t3_sec_integration = np.array(t3_sec_all, dtype=float)
        t1_sec_integration, t1_sec_l = EpochArray._add_duration(
            t3_sec_integration, 0.0, -rtlt
        )

        # Adjust for doy rollover events. The year is not rolled: the ramp
        # integration counts days from 1 January of the t3 year, so day 0 is
        # 31 December of the year before.
        t3_doy_integration = np.array(t3_doy_all, dtype=float)
        day_shift = np.floor(t1_sec_integration / constants.day_to_sec)
        t1_sec_integration, t1_sec_l = EpochArray._add_duration(
            t1_sec_integration, t1_sec_l, -day_shift * constants.day_to_sec
        )
        t1_doy_integration = t3_doy_integration + day_shift

        # 7. Generate the computed measurement by integration with the ramp table
        sranging_meas_list = []
        uncovered = []  # round trips with no ramp record at the transmit time
        first_gap = None  # the first such gap's explanation, reported once
        used_tropo_flag = np.zeros(len(rtlt))
        used_iono_flag = np.zeros(len(rtlt))
        _f_t1_cache = []   # transmit frequency f_T(t1) per epoch [Hz]
        for i in range(len(rtlt)):
            # 7.1 Perform the integration between t1 and t3 to generate f_int
            try:
                f_int, aux_int = RampTableManager.integrate(
                    ramp_data,
                    t1_sec_integration[i],
                    t3_sec_integration[i],
                    t1_doy_integration[i],
                    t3_doy_integration[i],
                    t3_year_all[i],
                    t3_year_all[i],
                    t1_q_sec_l=t1_sec_l[i],
                )
            except RampTableManager.RampCoverageError as e:
                # No uplink frequency at t1: the observable is undefined. Record
                # it so generate_measurement_dataset can flag it as an outlier --
                # the code that detected the problem names the point, rather than
                # leaving outlier rejection to infer it from a wild residual --
                # and emit NaN so that if the flag were ever missed the result is
                # conspicuous instead of a plausible-looking number.
                if not hasattr(self, "_uncomputable_indices"):
                    self._uncomputable_indices = []
                self._uncomputable_indices.append(i)
                # A coverage gap can span many consecutive observations; the
                # first explanation is reported once, with the count, below.
                if not uncovered:
                    first_gap = str(e)
                uncovered.append(i)
                f_int, aux_int = float("nan"), {"f_iq1": float("nan"),
                                               "f_iq2": float("nan"),
                                               "n_segments": 0}
            # Endpoint transmit frequency for the analytical partials. Taken here
            # rather than in the media branch below so it is recorded whether or
            # not media corrections are applied; ul_freq is the unramped fallback.
            _f_t1_cache.append(float(aux_int.get("f_iq1", aux_dict["ul_freq"][i])))
            #
            sranging_meas = (RU[i] * f_int) % M[i] + bias_RU[i]

            # 7.2 Add the correction term as per [1], Note 39 # NOTE: electronic delays works better following Moyer instead of [1], including them in the precise_RTLT instead
            # sranging_calib_corr_i = SequentialRangingReal._compute_sranging_correction(electronic_delays_dict, RU[i], aux_dict['ul_freq'][i]) # Method (2), compute the correction term
            sranging_calib_corr_i = 0  # Method (1), do NOT compute the correction term, delays are already accounted in the precise RTLT
            sranging_meas_corr = sranging_meas - sranging_calib_corr_i

            # 7.3 Add media correction effects
            if media_corrections is None:
                sranging_meas_list.append(sranging_meas_corr)
            else:
                if (
                    aux_int["n_segments"] >= 1
                ):  # Only compute media correction if the integration is successful
                    #
                    # Generate frequencies of interest to apply media effects
                    f_t1 = aux_int["f_iq1"]  # Moyer 13.2.8, [Hz]
                    f_t3 = f_t1 * M2[i]  # Moyer 13.2.8, [Hz]
                    # Compute tropo + iono delays
                    p_tropo_t3, p_iono_t3 = media_corrections.computed_rtlt_delays(
                        tt_et[i], str(target.spice_id), f_t3,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    p_tropo_t1, p_iono_t1 = media_corrections.computed_rtlt_delays(
                        tt_et[i] - rtlt[i], str(target.spice_id), f_t1,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    # NOTE: Sign of iono is negative for doppler, positive for sranging
                    p_total_s = MediaCorrections.path_delay_to_light_time(
                        p_tropo_t3 + p_tropo_t1 + p_iono_t3 + p_iono_t1
                    )  # Moyer 10-27, [s]
                    # Add delay into measurement
                    p_total_RU = p_total_s * RU[i] * f_t1  # Moyer 13-127
                    sranging_meas_media = (
                        sranging_meas + p_total_RU - sranging_calib_corr_i
                    )
                    sranging_meas_list.append(sranging_meas_media)
                    # Keep track of the media correction used
                    # .values because the media delays are unit-bearing now;
                    # ArrayWUnits refuses comparison against a bare number.
                    if (p_tropo_t3.values != 0.0) or (p_tropo_t1.values != 0.0):
                        used_tropo_flag[i] = 1
                    if (p_iono_t3.values != 0.0) or (p_iono_t1.values != 0.0):
                        used_iono_flag[i] = 1
                else:
                    sranging_meas_list.append(sranging_meas_corr)

        # 7.4 Print which media corrections are being applied (tropo/iono) - for debugging purposes
        if media_corrections is not None:
            n_tropo = int(sum(used_tropo_flag))
            n_iono = int(sum(used_iono_flag))
            parts = []
            if n_tropo > 0:
                parts.append(f"tropo ({n_tropo}/{len(used_tropo_flag)} measurements)")
            if n_iono > 0:
                parts.append(f"iono ({n_iono}/{len(used_iono_flag)} measurements)")
            print(f"Applying media corrections: {', '.join(parts)}")

        if uncovered:
            warnings.warn(
                f"SequentialRangingReal '{self.name}': {len(uncovered)} of "
                f"{len(rtlt)} round trips have no ramp record at their transmit "
                f"time (epoch indices {uncovered}); their computed range is NaN "
                f"and they are flagged as outliers. First: {first_gap}"
            )

        meas_awu = ArrayWUnits(sranging_meas_list, km / km)

        # Noise, for simulating observations. Predicted measurements for a
        # filter are computed with noisy=False and are left untouched.
        if noisy:
            meas_awu = meas_awu + Noise().generate_AWGN_with_units(
                mu=0.0,
                sigma=self.sigma.values,
                units=self.sigma.units,
                count=len(sranging_meas_list),
            )

        # Publish the endpoint cache for the analytical (Moyer) partials. Keyed by
        # the reception epochs so the partials can confirm the cache is current for
        # the epoch_array they are handed (see _cache_matches).
        self._last_gr = np.array(_gr_cache).reshape(-1, 2)  # GR (dn, up) per epoch [s]
        self._last_sc = np.array(_sc_cache)      # solar corona (dn+up) per epoch [s]
        self._last_rtlt = np.array(rtlt)         # round-trip light time per epoch [s]
        self._last_tt_et = np.array(tt_et)       # reception epoch t3 [ET s]
        self._last_f_t1 = np.array(_f_t1_cache)  # transmit frequency f_T(t1) [Hz]
        # Fingerprint of the reception epochs this cache was built for
        self._partials_cache_key = np.array(tt_et)

        return t1, t2, t3, ArrayWFrame(meas_awu, J2000)

    def _compute_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list[np.ndarray]:
        """
        Compute the sequential-ranging partials with the method chosen by
        ``partials_method``.

        Parameters
        ----------
        target : Spacecraft
            Spacecraft the ranging data tracks.
        epoch_array : EpochArray
            Measurement epochs.
        frame : Frame, optional
            Inertial frame the state partials are expressed in. Defaults to
            ``J2000``.

        Returns
        -------
        list of np.ndarray
            One partial row per epoch, with a column for each state component.

        Raises
        ------
        Measurement.SequencePartialsError
            If a sequence definition is set and ``partials_method`` is
            ``"fd"``, which builds every column from one state definition.
        ValueError
            If ``partials_method`` is not one of :attr:`PARTIALS_METHODS`. The
            constructor already rejects such a value, so this only fires when the
            attribute is set directly.

        Notes
        -----
        ``"analytical"`` (the default) calls :meth:`_compute_analytical_partials`
        and ``"fd"`` calls :meth:`_compute_partials_with_fd`.
        """

        method = self._partials_method
        if method == "fd" and self._sequence_definition is not None:
            raise Measurement.SequencePartialsError(
                f"{type(self).__name__} cannot finite-difference the partials "
                "across a mission sequence; use partials_method='analytical'."
            )
        if method == "fd":
            return self._compute_partials_with_fd(
                target, epoch_array, frame, bias_key="range_bias_RU"
            )
        if method == "analytical":
            return self._compute_analytical_partials(target, epoch_array, frame)
        # Unreachable via the constructor, which validates against
        # PARTIALS_METHODS. Guards against _partials_method being set
        # directly.
        raise ValueError(
            f"{type(self).__name__} has no implementation for "
            f"partials_method={method!r}. Supported: "
            f"{', '.join(type(self).PARTIALS_METHODS)}."
        )

    # ---------------------------------------------------------------------#
    # region   Analytical (Moyer) partials                                #
    # ---------------------------------------------------------------------#
    # Ported from giovanni_real_msr_experiments. These differentiate the same
    # light-time solution the forward model produces, reusing its cached
    # endpoints rather than re-solving, which is what makes them ~two orders of
    # magnitude cheaper than the finite-difference path.

    def _compute_h_tilde_pos_moyer(
        self,
        n_hat_down: np.ndarray,
        n_hat_up: np.ndarray,
        v_sc: np.ndarray,
        v_gs: np.ndarray,
        c: float,
    ) -> list:
        """
        Exact two-way range position partial at the spacecraft reflection time
        :math:`t_2`.

        Parameters
        ----------
        n_hat_down, n_hat_up : np.ndarray
            Unit vectors (3,) from the station to the spacecraft on the downleg and
            on the upleg.
        v_sc : np.ndarray
            Spacecraft velocity at :math:`t_2` in the light-time frame [km/s].
        v_gs : np.ndarray
            Station velocity at :math:`t_1` in the light-time frame [km/s].
        c : float
            Speed of light [km/s].

        Returns
        -------
        list of float
            :math:`(c/2)\\, \\partial\\rho / \\partial\\mathbf{r}_{SC}(t_2)` [-]. The
            caller's scale factor :math:`2\\, RU\\, f_T(t_1) / c` completes the
            conversion to RU/km.

        Notes
        -----
        The range observable is :math:`F(t_1)\\, \\rho` with :math:`\\rho` the round-trip
        light time (Moyer eq. 13-128), so

        .. math::

            \\frac{\\partial\\rho_{RU}}{\\partial\\mathbf{r}(t_2)} = F(t_1)\\,
                \\frac{\\partial\\rho}{\\partial\\mathbf{r}_{SC}(t_2)}

        :math:`\\rho` is geometric in the reflection-time position, so this is the only
        non-zero state partial at :math:`t_2`. The gradient of
        :meth:`Measurement.rtlt_grad_pos_exact` includes the :math:`v/c` corrections
        of Moyer eqs. 12-12 and 12-17, with the station velocity in the upleg
        denominator, :math:`\\dot{p}_{12} = \\hat{\\mathbf{r}}_{12} \\cdot \\mathbf{v}_{GS}`.

        References
        ----------
        .. [1] Moyer, T. D. (2000). *Formulation for Observed and Computed Values of
           Deep Space Network Data Types for Navigation*. JPL Monograph 2,
           eqs. 12-12, 12-17, 12-23 and 13-128.
        """
        grad = Measurement.rtlt_grad_pos_exact(n_hat_down, n_hat_up, v_sc, v_gs, c)
        return list(grad * c / 2.0)

    def _compute_h_tilde_vel_moyer(self) -> list:
        """
        Velocity partial of the range at the reflection time :math:`t_2`, which is
        zero.

        Returns
        -------
        list of float
            ``[0.0, 0.0, 0.0]``.

        Notes
        -----
        A round-trip light time depends only on the spacecraft position at
        :math:`t_2`, not on its velocity (Moyer eq. 12-23). The velocity sensitivity at
        the filter epoch comes from the STM mapping the filter applies downstream,

        .. math::

            H_{k,0} = \\left[ \\frac{\\partial\\rho}{\\partial\\mathbf{r}}, \\mathbf{0} \\right]
                \\Phi(t_0, t_2)

        so a non-zero velocity term here would count that mapping twice.
        """
        return [0.0, 0.0, 0.0]

    def _compute_analytical_partials(
        self, target: Spacecraft, epoch_array: EpochArray, frame: Frame = J2000
    ) -> np.ndarray:
        """
        Assemble the analytical H-tilde rows for this ranging model.

        Groups together the different components of the measurement partials in
        the global H-tilde, one row per epoch, ordered by state component. The
        light-time geometry is evaluated on both legs, so the partials follow the
        Moyer formulation rather than an instantaneous approximation.

        Parameters
        ----------
        target : Spacecraft
            Target spacecraft the observable is computed to.
        epoch_array : EpochArray
            Epochs at which the partials are evaluated. These are the receive
            times ``t3``; the spacecraft transponding time ``t2`` and the station
            uplink transmit time ``t1`` are recovered from the light-time
            solution.
        frame : Frame, optional
            Reference frame in which the partials are expressed. Defaults to
            ``J2000``.

        Returns
        -------
        list
            One H-tilde row per epoch, holding the measurement partials by
            state component.

        Raises
        ------
        Measurement.MissingPassIndexError
            If the state holds a stacked per-pass ``range_bias_RU`` for this
            station (dimension > 1) and no epoch-to-pass mapping is set.

        Notes
        -----
        The light-time cache is populated on demand through
        :meth:`_ensure_partials_cache`, which runs the forward model when the
        cached RTLT, delays and transmit frequencies do not match
        ``epoch_array``.

        Each row is converted to Range Units with the per-epoch factor
        :math:`2\\,RU\\,f_{T_1} / c` (Moyer 13-128), since the range observable
        is :math:`\\rho_{RU} = RU \\int f_T\\,dt`.

        With a sequence definition each row takes the state definition of the
        leg holding its epoch (see :meth:`_use_leg_state_definition`), so rows
        of different legs can differ in width.
        """

        # Ensure the light-time cache (RTLT, GR/SC delays, transmit freqs) is current
        # for these epochs — transparently runs the forward model on a cache miss.
        self._ensure_partials_cache(target, epoch_array, frame)

        partials_list = []
        c_val = constants.c.values  # km/s
        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        # The station the forward model ranged from: the kernel station plus
        # every estimated and attached correction (ITRF93).
        station_correction = self._itrf_station_correction()

        for i, epoch in enumerate(epoch_array):
            # In a sequence each row takes the state definition of its leg
            self._use_leg_state_definition(epoch)

            # ------------------------------------------------------------------ #
            # Downlink leg: LT-corrected (SC at transmit time t2, GS at t3)     #
            # ------------------------------------------------------------------ #
            rel_down = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.instrument.name,
                aberration_correction="LT",
            )

            # Move the station where the forward model put it
            if station_correction is not None:
                t3_et = float(epoch.times.values)
                delta_in_frame = SpiceManager.get_xfrm(
                    frame_from="ITRF93", frame_to=frame.name, epoch=t3_et
                ) @ station_correction(t3_et)[:3]
                rel_down = ArrayWUnits(
                    np.hstack([
                        rel_down[0:3].values - delta_in_frame,
                        rel_down[3:6].values,
                    ]),
                    rel_down.units,
                )

            rho_down = float(rel_down[0:3].norm().values)
            n_hat_down = rel_down[0:3].values / rho_down

            # ------------------------------------------------------------------ #
            # Uplink leg: GS as seen from SC at t2, LT-corrected.               #
            # spkezr(GS, t2, 'LT', SC) → r_gs(t1) − r_sc(t2).                  #
            # Negate to get GS→SC direction (same convention as downlink).       #
            # ------------------------------------------------------------------ #
            t2 = epoch.times.values - rho_down / c_val
            rel_up_rev = SpiceManager.get_state(
                trgt_bdy=self.instrument.name,
                epoch_time=t2,
                reference_frame=frame.name,
                obsvr_bdy=target.name,
                aberration_correction="LT",
            )
            rho_up = float(rel_up_rev[0:3].norm().values)
            n_hat_up = -rel_up_rev[0:3].values / rho_up  # negate: SC→GS → GS→SC

            # Transmitter velocities of Moyer 8-60, in the light-time frame: the
            # SC at t2 on the downleg, the station at t1 on the upleg. The
            # LT-corrected state above carries the SC velocity relative to the
            # station, which is off by the station's own ~30 km/s and left the
            # position row ~1e-4 relative short of the forward model.
            v_sc_t2 = self._light_time_velocity(target.name, t2, frame.name)
            v_gs_t1 = self._light_time_velocity(
                self.instrument.name, t2 - rho_up / c_val, frame.name
            )

            # Stack h_tilde based on StateVector components
            h_tilde = []
            target_sid = getattr(target, "spice_id", None)

            if self._state_definition is None:
                h_tilde.extend(self._compute_h_tilde_pos_moyer(n_hat_down, n_hat_up, v_sc_t2, v_gs_t1, c_val))
                h_tilde.extend(self._compute_h_tilde_vel_moyer())
            else:
                for component_def in self._state_definition:
                    component = component_def[0]
                    comp_body = component_def[4] if len(component_def) > 4 else None
                    comp_body_sid = getattr(comp_body, "spice_id", None)

                    if state_allowed["position"][2].match(component):
                        if comp_body_sid == target_sid:
                            h_tilde.extend(self._compute_h_tilde_pos_moyer(
                                n_hat_down, n_hat_up, v_sc_t2, v_gs_t1, c_val
                            ))
                        else:
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["velocity"][2].match(component):
                        # Zero at t2 (ρ is geometric in position); the filter's
                        # Φ(t0,t2) produces the epoch-velocity sensitivity.
                        h_tilde.extend(self._compute_h_tilde_vel_moyer())

                    elif state_allowed["eta_srp"][2].match(component):
                        # No direct sensitivity: SRP moves the light time only through
                        # the trajectory, which the filter maps through the STM.
                        h_tilde.extend([0.0])

                    elif self._is_station_parameter(component):
                        # Location (Cartesian or cylindrical), orientation, EOP
                        # and Love/Shida numbers move the station inside the
                        # light-time solution, at t1 and t3, so each end carries
                        # its own Jacobian, Moyer 12-12/12-17 weight and ET - TAI
                        # change. rtlt_parts[10] stays zero: adding it as well
                        # would double-count the geometric term.
                        t3 = float(self._last_tt_et[i])
                        t1 = t3 - float(self._last_rtlt[i])
                        row = self._rtlt_station_row(
                            component_def, t3, t1, n_hat_down, n_hat_up,
                            v_sc_t2, v_gs_t1, frame.name, c_val)
                        # ρ_km = c·ρ/2 ⇒ pre-scale ×c/2; trailing scale_fac=2·RU·f_T1/c → RU
                        h_tilde.extend(list(row * c_val / 2.0))

                    elif "range_bias_RU" in state_allowed and state_allowed["range_bias_RU"][
                        2
                    ].match(component):
                        # ρ_RU = ρ_RU_geom + bias_RU  ⇒  ∂ρ_RU/∂bias = 1 for THIS station,
                        # in the slot of the epoch's pass for one bias per pass.
                        # Stored pre-scaled (×1/scale_fac_i) so the trailing per-epoch
                        # scale_fac_i = 2·RU·f_T1/c yields exactly 1.
                        bias_partials = self._bias_partials(
                            component, component_def[1], comp_body, epoch
                        )
                        if any(bias_partials):
                            aux_dict = self.computed_measurements_dict
                            idx = min(i, len(aux_dict["RU"]) - 1)
                            RU_i = float(aux_dict["RU"][idx])
                            f_t1_i = float(self._last_f_t1[i])
                            scale_fac_i = 2.0 * RU_i * f_t1_i / c_val
                            bias_partials = [p / scale_fac_i for p in bias_partials]
                        h_tilde.extend(bias_partials)

                    elif state_allowed["dv_man"][2].match(component):
                        # "dv_man" regex already matches dv_man, dv_man1, dv_man2, ...
                        # No direct sensitivity: the maneuver moves the light time only
                        # through the trajectory, which the filter maps through the STM.
                        h_tilde.extend([0.0] * 3)
                    # ---------------------------------------------------- #
                    # Moyer correction partials (Ch. 10 + Eq. 13-127)       #
                    # ---------------------------------------------------- #
                    elif state_allowed["gamma_rel"][2].match(component):
                        # ∂ρ_RU/∂γ = F[t1]·∂ρ/∂γ, with each leg's RLT/(1+γ) weighted
                        # by its Moyer 12-12/12-17 factor, since the Shapiro delay sits
                        # inside the light-time solution and moves t2 and t1, plus the
                        # γ in the eq. 4-10 space-time transform of the station
                        # (barycentric) or spacecraft (geocentric). [Moyer 8-55, 13-128]
                        # Pre-scale (c/2)·∂ρ/∂γ matches the geometric pos/vel convention
                        # so the trailing scale_fac (= 2·RU·f_T1/c) recovers RU.
                        gamma = float(constants.gamma_GR.values)  # what the light time used
                        w_dn, w_up = Measurement.rtlt_leg_weights(
                            n_hat_down, n_hat_up, v_sc_t2, v_gs_t1, c_val
                        )
                        gr_dn, gr_up = self._last_gr[i]
                        drho_dgamma = (w_dn * gr_dn + w_up * gr_up) / (1.0 + gamma)  # [s]
                        t3 = float(self._last_tt_et[i])
                        t1 = t3 - float(self._last_rtlt[i])
                        g_sc = Measurement.rtlt_grad_pos_exact(
                            n_hat_down, n_hat_up, v_sc_t2, v_gs_t1, c_val
                        )
                        g3, g1 = Measurement.rtlt_grad_station_exact(
                            n_hat_down, n_hat_up, v_sc_t2, v_gs_t1, c_val
                        )
                        drho_dgamma += (
                            g_sc @ self._gamma_position_shift(target.name, t2, frame.name)
                            + g3 @ self._gamma_position_shift(self.instrument.name, t3, frame.name)
                            + g1 @ self._gamma_position_shift(self.instrument.name, t1, frame.name)
                        )
                        h_tilde.extend([drho_dgamma * c_val / 2.0])

                    elif state_allowed["solar_corona_scale"][2].match(component):
                        # ∂ρ_RU/∂κ_SC = F[t1]·∆SC,  SC is part of ρ (Moyer 10-76, 13-128).
                        drho_dkappa = float(self._last_sc[i])  # [s]
                        h_tilde.extend([drho_dkappa * c_val / 2.0])

                    elif self._is_media_parameter(component):
                        # ∂ρ_RU/∂ρ_z = RU·f_T1/c·(m(E_t3, f_T3) + m(E_t1, f_T1)): the
                        # zenith delay is mapped on each leg at its own elevation
                        # and, for the ionosphere, its own frequency (Moyer 10-11,
                        # 10-27, 13-127). Positive for ranging, as in the forward model.
                        parameter = next((p for p in self._media_parameters()
                                          if p["name"] == component), None)
                        if self._media_corrections is None or parameter is None:
                            h_tilde.extend([0.0] * component_def[1])
                        else:
                            aux_dict = self.computed_measurements_dict
                            idx = min(i, len(aux_dict["RU"]) - 1)
                            RU_i = float(aux_dict["RU"][idx])
                            f_t1_i = float(self._last_f_t1[i])
                            M2_i = float(aux_dict["M2_num"][idx]) / float(aux_dict["M2_den"][idx])
                            f_t3_i = f_t1_i * M2_i
                            t3 = float(self._last_tt_et[i])
                            t1 = t3 - float(self._last_rtlt[i])
                            column = 0 if parameter["kind"] == "tropo_zenith_delay" else 1
                            a3 = self._media_corrections.zenith_delay_partials(
                                t3, str(target.spice_id), f_t3_i)[column]
                            a1 = self._media_corrections.zenith_delay_partials(
                                t1, str(target.spice_id), f_t1_i)[column]
                            c_m_per_s = float(c_val) * constants.km_to_m
                            # Already in RU: divide by scale_fac_i like the bias row.
                            scale_fac_i = 2.0 * RU_i * f_t1_i / c_val
                            h_tilde.extend(list(RU_i * f_t1_i / c_m_per_s * (a3 + a1)
                                                * parameter["scale"] / scale_fac_i))

                    elif self._electronic_delay_kind(component) is not None:
                        # ∂ρ/∂τ = 1 for the station's delay at t1 or t3 and the
                        # spacecraft's at t2 (Moyer 11-7); pre-scaled by c/2 like
                        # the light-time rows, so scale_fac gives RU·f_T1.
                        h_tilde.extend([c_val / 2.0 * self._electronic_delay_partial(
                            component, comp_body, target)])

                    else:
                        h_tilde.extend([0.0 for _ in range(component_def[1])])

            partials_list.append(h_tilde)
            #
        #
        # Per-epoch scale factor F[t1] = RU·f_T1 mapping the (c/2)-pre-scaled
        # light-time partials to Range Units (Moyer 13-128): the range observable
        # is ρ_RU = RU·∫f_T dt, so ∂ρ_RU/∂x = RU·f_T1·∂ρ/∂x and, since each
        # h_tilde entry is (c/2)·∂ρ/∂x, the trailing factor is scale_fac = 2·RU·f_T1/c.
        # (The previous constant 1000/RU assumed f_T1 = 500·c/RU², a ~4% approximation.)
        aux_dict = self.computed_measurements_dict
        RU_arr = aux_dict["RU"]
        out = []
        for ii in range(len(partials_list)):
            RU_i = float(RU_arr[min(ii, len(RU_arr) - 1)])
            f_t1_i = float(self._last_f_t1[ii])
            scale_fac_i = 2.0 * RU_i * f_t1_i / c_val
            out.append(scale_fac_i * np.array(partials_list[ii]))
        return out

    # endregion Analytical (Moyer) partials #
    # ---------------------------------------------------------------------#

    def _fd_difference(self, meas_pert: np.ndarray, meas_nom: np.ndarray) -> np.ndarray:
        """
        Difference a perturbed and a nominal range for finite differences.

        Parameters
        ----------
        meas_pert : np.ndarray
            Perturbed noise-free measurements (N,) [RU].
        meas_nom : np.ndarray
            Nominal noise-free measurements (N,) [RU].

        Returns
        -------
        np.ndarray
            ``meas_pert - meas_nom`` wrapped into :math:`[-M/2, M/2)`.

        Notes
        -----
        The observable is a phase modulo the range ambiguity :math:`M`, so a
        perturbed value that wraps past :math:`M` would otherwise difference to
        about :math:`-M`. See :meth:`_wrap_to_modulo`.
        """
        return self._wrap_to_modulo(meas_pert - meas_nom)

    def residuals(
        self, observed_meas: ArrayWFrame, computed_meas: ArrayWFrame
    ) -> ArrayWFrame:
        """
        Generates the sequential ranging residuals, wrapped to the range ambiguity.

        Parameters
        ----------
        observed_meas : ArrayWFrame
            The observed measurements values (O) [RU].

        computed_meas : ArrayWFrame
            The computed measurements values (C) [RU].

        Returns
        -------
        residuals : ArrayWFrame
            AWF with the residual O-C wrapped into :math:`[-M/2, M/2)`.

        Notes
        -----
        Both O and C are reported modulo :math:`M`, and C carries the bias and
        media terms added after the modulo, so a plain O-C lands anywhere in
        :math:`(-M, M)`: a point whose O and C sit on opposite sides of a wrap
        jumps by a whole modulus, which is frequent with a small :math:`M`. The
        residual is the difference closest to zero, see :meth:`_wrap_to_modulo`.
        """
        residuals_AWF = super().residuals(observed_meas, computed_meas)
        wrapped = self._wrap_to_modulo(residuals_AWF.quantity.values)
        return ArrayWFrame(wrapped, residuals_AWF.quantity.units, residuals_AWF.frame)

    def _wrap_to_modulo(self, delta: np.ndarray) -> np.ndarray:
        """
        Wrap a difference of two ranges into :math:`[-M/2, M/2)`.

        Parameters
        ----------
        delta : np.ndarray
            Differences of two sequential ranging observables (N,) [RU].

        Returns
        -------
        np.ndarray
            ``delta`` less the whole number of moduli :math:`M` closest to it,
            with :math:`M` taken per epoch from the aux data.

        Notes
        -----
        The whole cycles are removed from the difference itself: shifting by
        :math:`M/2` before the modulo would round the difference to the spacing
        of :math:`M/2`, 8192 RU for :math:`M = 10^{20}`.
        """
        delta = np.asarray(delta, dtype=float)
        M = self._aux_per_epoch(delta.size)["M"]
        return delta - M * np.floor(delta / M + 0.5)

    @staticmethod
    def _compute_sranging_correction(electronic_delays_dict, RU_i, ul_freq_i):
        """
        Generate the sequential ranging correction term as per [1], Note 39

        Parameters
        ----------
        electronic_delays_dict : dict
            Dictionary of electronic delays

        RU_i : float
            seconds-to-RU conversion factor (pre-frequency multiplier component)

        ul_freq_i : float
            frequency to use in the seconds-to-RU conversion factor

        Returns
        -------
        sranging_calib_corr : float
            Value of the 2-way sequential ranging correction term in [1], Note 39.

        Notes
        -----
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """
        # Compute calibration correction terms by seconds and RU components
        calib_corr_1 = (
            electronic_delays_dict["ul_stn_cal"] + electronic_delays_dict["dl_stn_cal"]
        )  # [RU]
        calib_corr_2 = (
            electronic_delays_dict["dl_zheight_corr"]
            + electronic_delays_dict["ul_zheight_corr"]
            - electronic_delays_dict["trn_tt_dly"]
            - electronic_delays_dict["rcv_tt_dly"]
            - electronic_delays_dict["array_delay"]
        )  # [s]
        sranging_calib_corr = (
            calib_corr_1
            + calib_corr_2 * (RU_i * ul_freq_i)
            + electronic_delays_dict["scft_transpd_delay"]
        )  # [RU]
        return sranging_calib_corr
    
    @staticmethod
    def default_seqranging_aux(
        tstart: EpochArray,
        tend: EpochArray,
        N: int = 1,
        ramp_margin: float = None,
        n_ramp: int = None,
        band: str = "X",
    ) -> dict:
        """
        Default ``computed_measurements_dict`` for synthetic sequential ranging.

        Parameters
        ----------
        tstart, tend : EpochArray
            Span the simulated observations cover.
        N : int, optional
            Number of observations. Every per-observation entry is this long,
            including year/doy/sec. Defaults to ``1``.
        ramp_margin : float, optional
            Seconds the ramp grid extends beyond the span at each end. Defaults to
            :attr:`Constants.RAMP_MARGIN_DEFAULT`. The ramp grid runs wider than the
            observation span because the ramp is queried at the uplink transmit
            time, one round-trip light time before the first time tag.
        n_ramp : int, optional
            Number of ramp-table entries. Defaults to
            :attr:`Constants.RAMP_ENTRIES_DEFAULT`.
        band : str, optional
            DSN band letter, ``"X"`` or ``"S"`` (case-insensitive), resolved by
            :meth:`Measurement._resolve_dsn_band`. It sets the uplink frequency,
            turnaround ratio, exciter scalar and range unit. Defaults to ``"X"``.

        Returns
        -------
        SeqRanging_aux_data : dict
            Aux values needed for computing measurements. Every per-observation
            array is ``N`` long and every ramp array is ``n_ramp`` long; see
            :meth:`Measurement._default_aux_grids`.
        """
        obs, ramp = SequentialRangingReal._default_aux_grids(
            tstart, tend, N, ramp_margin=ramp_margin, n_ramp=n_ramp
        )
        (year, day, sec) = (obs["year"], obs["doy"], obs["sec"])
        ramp_day, ramp_sec, ramp_year = ramp["day"], ramp["sec"], ramp["year"]
        n_ramp = len(ramp_sec)
        # A constant-rate ramp referenced to the first entry, so the
        # piecewise-linear interpolant RampTableManager.integrate builds is
        # continuous across the segment boundaries.
        band_spec = Measurement._resolve_dsn_band(band)
        ramp_f0 = band_spec.uplink_frequency.values
        ramp_rate = constants.RAMP_RATE_DEFAULT.values
        ramp_freq = (
            ramp_f0 + ramp_rate * (ramp["et"] - ramp["et"][0])
        ).tolist()
        rcv_time_tag_delay = float(constants.RCV_TIME_TAG_DELAY_DEFAULT.values)

        SeqRanging_aux_data = {
            #
            # Sranging auxiliary data (from 2W_sranging, SFDU_7)
            #
            "RU": [band_spec.range_unit for _ in range(N)],
            "M": [constants.RANGE_MODULO_DEFAULT for _ in range(N)],
            "M2_num": [band_spec.turnaround_num for _ in range(N)],
            "M2_den": [band_spec.turnaround_den for _ in range(N)],
            "rng_type": [constants.RANGING_TYPE_DEFAULT for _ in range(N)],
            "outlier_flag_sranging": [0 for _ in range(N)],
            "year_SFDU7": year,
            "doy_SFDU7": day,
            "sec_SFDU7": sec,
            "version_num" : [constants.RANGING_VERSION_DEFAULT for _ in range(N)],
            #
            # Delays auxiliary data (from 2W_sranging, SFDU_7)
            #
            "ul_freq": [band_spec.uplink_frequency.values for _ in range(N)],     
            "ul_zheight_corr": [0.0 for _ in range(N)],
            "dl_zheight_corr": [0.0 for _ in range(N)],
            "scft_transpd_delay": constants.TRANSPONDER_DELAY_DEFAULT, # [RU]
            "scft_transpd_delay_s": Measurement.transponder_delay_seconds(
                constants.TRANSPONDER_DELAY_DEFAULT,
                band_spec.ru_reference_frequency.values,
                band_spec.range_unit),
            "transmit_time_tag_delay": [0.0 for _ in range(N)],
            "rcv_time_tag_delay": [rcv_time_tag_delay for _ in range(N)],
            "array_delay": [0.0 for _ in range(N)],
            "ul_stn_cal": [constants.UL_STATION_CAL_DEFAULT for _ in range(N)],
            "dl_stn_cal": [constants.DL_STATION_CAL_DEFAULT for _ in range(N)],
            "exc_scalar_num": [band_spec.exciter_scalar_num for _ in range(N)],
            "exc_scalar_den": [band_spec.exciter_scalar_den for _ in range(N)],
            #
            # Ramp table auxiliary data (from SFDU_9)
            #
            "ramp_sec": ramp_sec,
            "ramp_day": ramp_day,
            "ramp_freq": ramp_freq,
            "ramp_rate": [ramp_rate for _ in range(n_ramp)],
            "ramp_type": [constants.RAMP_TYPE_DEFAULT for _ in range(n_ramp)],
            # Year of each ramp record, so the table may cross a new year
            "ramp_year": ramp_year,
        }

        Measurement.validate_aux(SeqRanging_aux_data, label="default_seqranging_aux")
        return SeqRanging_aux_data

    def _write_aux_data(self, dict_DB, dict_meas):
        """
        Add this model's auxiliary data to a file being written.

        Called by :meth:`~scarabaeus.Measurement.write_observed_measurements`
        once it has built the measurement block, to fill in the fields that are
        specific to sequential ranging. Everything written comes from the
        model's ``computed_measurements_dict``, which was filled either by
        ``_read_aux`` from a real pass or by
        :meth:`~scarabaeus.SequentialRangingReal.default_seqranging_aux` for a
        simulated one.

        Both arguments are modified in place.

        Parameters
        ----------
        dict_DB : dict
            The top-level file dictionary. Gains an ``"aux_SFDU_9"`` entry
            holding the uplink ramp table, which is one row per ramp segment
            rather than one per observation and so cannot live in the
            measurement block, with the year of each record when the aux data
            has ``ramp_year``.
        dict_meas : dict
            The measurement block, one value per observation. Gains the ranging
            configuration (modulo, transponder turnaround ratio, type, version),
            the station delays and calibrations, the uplink frequency, the
            exciter scalars and the spacecraft transponder delay.

        Raises
        ------
        TypeError
            If the model has no auxiliary data, via
            :meth:`~scarabaeus.Measurement.validate_aux`. Raised here rather
            than surfacing as a less obvious error from the first key read.
        ValueError
            If the auxiliary data is ragged, via the same check.

        Notes
        -----
        Every column added here has to be as long as the set of epochs being
        simulated, or
        :meth:`~scarabaeus.Measurement.write_observed_measurements` refuses to
        write the file. The auxiliary dict describes whichever observations it
        was built for, which need not be the ones being written.

        Two fields are handled specially, for reasons given at the code below:
        the time columns are left as the caller wrote them, and the outlier
        flags are taken from the auxiliary copy only when it describes these
        same observations.
        """
        # Reported here rather than as 'NoneType is not subscriptable' from
        # the first key read below.
        Measurement.validate_aux(
            self._computed_measurements_dict, label=self.name
        )

        dict_meas["rng_modulo"] = self._computed_measurements_dict["M"]
        dict_meas["scft_transpd_turn_num"] = self._computed_measurements_dict["M2_num"]
        dict_meas["scft_transpd_turn_den"] = self._computed_measurements_dict["M2_den"]
        dict_meas["rng_type"] = self._computed_measurements_dict["rng_type"]
        dict_meas["version_num"] = self._computed_measurements_dict["version_num"]

        # year / doy / sec are deliberately NOT copied from the aux dict.
        #
        # Sequential ranging has no separate SFDU-7 block in the file: the
        # SFDU-7 fields *are* the measurement block, which is why _read_aux
        # reads them as trk_data[meas_name]["year"]. The keys do belong here --
        # but write_observed_measurements has already written them, converted
        # from the epochs actually simulated. The year_SFDU7/doy_SFDU7/sec_SFDU7
        # copies describe whatever pass _read_aux was given, or
        # default_seqranging_aux's own interval grid, neither of which is the
        # grid being simulated. Overwriting with them silently retagged the
        # measurements with the wrong times (observed_measurements rebuilds its
        # epochs from these very columns), and could leave the block ragged.
        # DopplerReal avoids this only because it has a real aux_SFDU_16 block
        # to put its copies in.
        #
        # outlier_flag follows the aux copy when it describes these same
        # observations: MeasurementSpec.flag keeps that copy in step with
        # ObsData.meas_outliers (see AUX_KEYS), so a spec that was
        # edited and then written back out saves the post-edit flags. When the
        # lengths disagree the aux copy is about different observations, so the
        # fresh zeros that write_observed_measurements already wrote stand.
        aux_flags = self._computed_measurements_dict.get(
            self.AUX_KEYS.outlier_flag
        )
        if aux_flags is not None and len(aux_flags) == len(dict_meas["outlier_flag"]):
            dict_meas["outlier_flag"] = aux_flags
        
        dict_meas["ul_freq"] = self._computed_measurements_dict["ul_freq"]
        dict_meas["ul_zheight_corr"] = self._computed_measurements_dict["ul_zheight_corr"]
        dict_meas["dl_zheight_corr"] = self._computed_measurements_dict["dl_zheight_corr"]
        dict_meas["transmit_time_tag_delay"] = self._computed_measurements_dict["transmit_time_tag_delay"]
        dict_meas["rcv_time_tag_delay"] = self._computed_measurements_dict["rcv_time_tag_delay"]
        dict_meas["array_delay"] = self._computed_measurements_dict["array_delay"]
        dict_meas["ul_stn_cal"] = self._computed_measurements_dict["ul_stn_cal"]
        dict_meas["dl_stn_cal"] = self._computed_measurements_dict["dl_stn_cal"]
        dict_meas["exc_scalar_num"] = self._computed_measurements_dict["exc_scalar_num"]
        dict_meas["exc_scalar_den"] = self._computed_measurements_dict["exc_scalar_den"]
        # The transponder delay is a mission constant; a TRK-2-34 measurement
        # record carries it on every row, and _read_aux reads it back from there.
        if "scft_transpd_delay" in self._computed_measurements_dict:
            dict_meas["scft_transpd_delay"] = [
                float(np.squeeze(self._computed_measurements_dict["scft_transpd_delay"]))
            ] * len(dict_meas["sec"])

        aux_SFDU9 = {
            "sec" : self._computed_measurements_dict["ramp_sec"],
            "doy" : self._computed_measurements_dict["ramp_day"],
            "ramp_freq" : self._computed_measurements_dict["ramp_freq"],
            "ramp_rate" : self._computed_measurements_dict["ramp_rate"],
            "ramp_type" :  self._computed_measurements_dict["ramp_type"]
        }
        if self._computed_measurements_dict.get(self.RAMP_YEAR_KEY) is not None:
            aux_SFDU9["year"] = self._computed_measurements_dict[self.RAMP_YEAR_KEY]

        dict_DB["aux_SFDU_9"] = aux_SFDU9

    def _flag_invalid_rows(self, meas_values, meas_outliers, aux_dict):
        """Outlier flags with the rows whose range is not a phase in :math:`[0, M)`
        set to 1.

        DSN files open a pass with placeholder rows (range -1, logged while the
        ranging components are acquired) that carry no outlier flag.

        Parameters
        ----------
        meas_values : np.ndarray
            The observed ranges as read, one per row [RU].
        meas_outliers : np.ndarray
            The outlier flags so far, 1 = reject.
        aux_dict : dict
            The model's auxiliary data; its ``"M"`` entry is the range ambiguity of
            each row.

        Returns
        -------
        np.ndarray
            A copy of ``meas_outliers`` with the invalid rows set to 1.
        """
        M = np.asarray(aux_dict["M"], dtype=float)
        valid = (meas_values >= 0.0) & (meas_values < M)
        flags = np.array(meas_outliers)
        flags[~valid] = 1
        return flags

    def _read_aux(self, trk_data, meas_name, sc_transponder_delay_RU=0.0,
                  band: str = "X"):
        """
        Collect the auxiliary data a sequential-ranging pass needs.

        Called by :meth:`~scarabaeus.Measurement.observed_measurements` on an
        already-loaded tracking file. The returned dict is both handed back as
        ``ObsData.aux_dict`` and installed on the model as its
        ``computed_measurements_dict``, which is what the forward model reads,
        so the two cannot fall out of step.

        The observable cannot be reconstructed from the observations alone: the
        range unit, the transponder turnaround ratio, the station delays and
        the uplink ramp table are all needed to form a computed range, and all
        of them live in the file alongside the measurements.

        Parameters
        ----------
        trk_data : dict
            A loaded tracking-data file. Each quantity is read from the block
            named by *meas_name* when it is there, and otherwise from the
            auxiliary SFDU record that carries it (see
            :meth:`~scarabaeus.Measurement._read_dsn_aux`): ``"aux_SFDU_7"``
            for the ranging modulo and type, ``"aux_SFDU_16"`` for the
            transponder turnaround ratio, ``"aux_SFDU_2"`` and
            ``"aux_SFDU_3"`` for the zenith height corrections and time-tag
            delays, and ``"aux_SFDU_9"`` for the uplink ramp table.
        meas_name : str
            Key of the sequential-ranging measurement block, normally
            ``"2W_sranging"``.
        sc_transponder_delay_RU : float, optional
            The spacecraft transponder delay in range units. Defaults to
            ``0.0``, which takes the delay the measurement block carries
            (``scft_transpd_delay``), or no delay when it has none.
        band : str, optional
            DSN band whose reference frequency converts the transponder delay
            to seconds, and whose range unit stands in when the file has no
            exciter scalars. Defaults to ``"X"``.

        Returns
        -------
        dict
            The auxiliary data, in three groups: the ranging configuration and
            per-observation time tags, the station and spacecraft delays, and
            the uplink ramp table from ``"aux_SFDU_9"``. The ramp entries are
            one per segment; everything else is one per observation.

        Raises
        ------
        Measurement.AuxDataError
            If a required field is in no block, the ramp table is missing, or
            the transponder delay varies within the pass (see
            :meth:`~scarabaeus.Measurement._read_dsn_aux`).
        ValueError
            If *band* is not one of the supported letters, or if the
            per-observation fields and the ramp table are not each internally
            consistent in length, via
            :meth:`~scarabaeus.Measurement.validate_aux`.

        Notes
        -----
        Each SFDU record of a TRK-2-34 file has its own cadence and time tags,
        so its rows do not correspond to the ranging rows. Every value that is
        not in the ranging block itself is therefore sampled at the ranging
        epochs by time tag, taking the entry in force at each (see
        :meth:`~scarabaeus.Measurement.align_aux_to_epochs`).

        The range unit ``"RU"`` is formed per observation from the exciter
        scalars recorded in the file (``16 * num / den``), so a pass in any
        band converts correctly; only a file without exciter scalars falls
        back to the range unit of *band*. See
        :meth:`~scarabaeus.Measurement.transponder_delay_seconds` for the
        transponder delay conversion.
        """
        aux = Measurement._read_dsn_aux(
            trk_data, meas_name, sc_transponder_delay_RU, band
        )
        record = trk_data[meas_name]
        n_meas = len(record["sec"])
        if "exc_scalar_num" in aux:
            aux["RU"] = 16.0 * np.asarray(aux["exc_scalar_num"], dtype=float) / np.asarray(
                aux["exc_scalar_den"], dtype=float
            )
        else:
            aux["RU"] = np.full(n_meas, float(aux["RU"]))
        aux["M"] = Measurement._aux_field(
            trk_data, meas_name, [(None, "rng_modulo"), ("aux_SFDU_7", "rng_modulo")]
        )
        for record_field, key in self.AUX_KEYS.record_fields.items():
            if key == "M":
                continue
            aux[key] = Measurement._aux_field(
                trk_data, meas_name, [(None, record_field), ("aux_SFDU_16", key)]
            )
        for key in ("rng_type", "version_num"):
            try:
                aux[key] = Measurement._aux_field(
                    trk_data, meas_name, [(None, key), ("aux_SFDU_7", key)]
                )
            except Measurement.AuxDataError:
                pass
        aux[self.AUX_KEYS.outlier_flag] = np.asarray(
            record.get("outlier_flag", np.zeros(n_meas, dtype=int))
        )
        aux["year_SFDU7"] = record["year"]
        aux["doy_SFDU7"] = record["doy"]
        aux["sec_SFDU7"] = record["sec"]
        Measurement.validate_aux(aux, label="_read_aux")
        return aux
