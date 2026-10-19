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
    Instrument,
    Spacecraft,
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
km, sec = Units.get_units(["km", "sec"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class DopplerReal(Measurement):
    """Models the real Doppler measurement model.

    Generates computed Doppler observables between a DSN ground station and a
    spacecraft using the precise round-trip light-time (RTLT) formulation,
    including relativistic corrections and optional media corrections.
    Observables are expressed in Hz.

    **Count-interval observable**

    The measurement epoch :math:`t_3` is the signal reception time.  The
    count interval :math:`T_c` defines a window
    :math:`[t_{3,s},\\,t_{3,e}] = [t_3 - T_c/2,\\; t_3 + T_c/2]`.
    The precise RTLT back-propagates each endpoint to the uplink times
    :math:`t_{1,s}` and :math:`t_{1,e}`.

    The Doppler observable (Moyer 2000, Sec. 13.2) is

    .. math::

        f_D = -\\frac{M_2}{T_c}
              \\int_{t_{1,s}}^{t_{1,e}} f_T(t)\\, dt + b_{f_D}

    where :math:`f_T(t)` is the piecewise-linear uplink ramp frequency
    integrated via the ramp table, :math:`M_2 = M_{2,num}/M_{2,den}` is
    the spacecraft transponder turn-around ratio, and :math:`b_{f_D}` is
    an optional station Doppler bias (Hz).

    **Media corrections**

    When a :class:`~scarabaeus.MediaCorrections` object is attached, the
    signed excess path delays at the count-interval endpoints (Moyer
    eq. 10-28, 10-29) are

    .. math::

        \\Delta\\tau_{s/e} = \\frac{1}{c}\\bigl(
            p_{tropo,t_3} - p_{iono,t_3}
          + p_{tropo,t_1} - p_{iono,t_1}
        \\bigr)_{s/e}

    and the media-corrected observable becomes (Moyer eq. 13-58):

    .. math::

        f_D^{corr} = f_D
          + \\frac{M_2}{T_c}
            \\bigl(f_{T,e}\\,\\Delta\\tau_e - f_{T,s}\\,\\Delta\\tau_s\\bigr)

    Note that the ionospheric contribution enters with a **negative** sign
    for Doppler (opposite to sequential ranging).

    Parameters
    ----------
    name : str
        Name of the measurement model.
    instrument : Instrument
        Ground station or antenna instrument (e.g. GroundStation).
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    meas_bias : float or ArrayWUnits, optional
        Doppler bias of the ground station [Hz], used when the state vector
        holds no ``rangerate_bias_Hz`` of this station. One value applies to
        every pass; more hold one per pass, grouped by
        :meth:`~scarabaeus.Measurement.set_pass_epochs`. The same holds for a
        ``rangerate_bias_Hz`` state component. Defaults to ``None``.
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
        quantities of each tracking pass. :meth:`observed_measurements` fills
        it from the tracking file (each SFDU record aligned to the
        measurement epochs by time tag), and the model's
        :meth:`default_doppler_aux` builds one for synthetic measurements. Unless noted otherwise, each
        key maps to a list/array with one entry per observation epoch (a
        single value is applied to every epoch):

        * ``'Tc'`` : Doppler count interval [s] (SFDU 16).
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
        * ``'exc_scalar_num'``, ``'exc_scalar_den'`` : exciter scalar
          numerator and denominator (SFDU 7), optional. Read only to convert a
          nonzero ``'scft_transpd_delay'`` to seconds.
        * ``'ramp_sec'``, ``'ramp_day'``, ``'ramp_freq'``, ``'ramp_rate'``,
          ``'ramp_type'`` : transmitter frequency ramp table (SFDU 9) —
          seconds of day, day of year, ramp reference frequency [Hz], ramp
          rate [Hz/s], and ramp type of each ramp record (one entry per ramp
          record, not per observation).
        * ``'ramp_year'`` : year of each ramp record (SFDU 9 ``year``),
          optional; needed when the ramp table or a round trip crosses a new
          year.

        Defaults to ``None``.
    partials_method : {"analytical", "fd", "fd_endpoint"}, optional
        Which H-tilde implementation to use. ``"analytical"`` (the default) uses
        the Moyer partials, which differentiate the same light-time solution the
        forward model produced and reuse its cached endpoints, for roughly 50x
        less cost than ``"fd"`` on this model. ``"fd"`` finite-differences the
        real forward model -- one full forward run per state component, slow,
        but it differentiates exactly the observable being fitted.
        ``"fd_endpoint"`` is the finite-difference reference with the state
        perturbation mapped to both ends of the count interval, see
        :meth:`_compute_partials_endpoint_fd`. Defaults to ``None``, i.e.
        ``"analytical"``.
    eps_dict : dict, optional
        Perturbation magnitudes for finite-difference partials. Spacecraft
        entries are keyed ``dx`` to ``dvz``; station, Earth-orientation and
        zenith-delay entries are keyed by StateDefinition parameter kind, e.g.
        ``gs_delta_location_CYL`` or ``eop_delta_ut1``, and are read only by
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
    scarabaeus.SequentialRangingReal : Companion real ranging model.
    scarabaeus.MediaCorrections : Tropospheric and ionospheric path-delay model.

    References
    ----------
    .. [1] "TRK-2-34, DSN Tracking System Data Archival Format",
           DSN No. 820-013, TRK-2-34, Rev. N.
    """

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
        # Reflection-time offsets (t2s - t2_mid, t2e - t2_mid) per epoch [s], set
        # only while _compute_partials_endpoint_fd is running.
        self._endpoint_offsets = None
        self.compute_partials = self._compute_partials
        self.eps_dict = eps_dict
        self.computed_measurements = self.computed_measurements
        self._computed_measurements_dict = computed_measurements_dict
        self._measurement_type = "DopplerReal"
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

        self.count_interval = type(self).COUNT_INTERVAL

    # ------------#
    # Properties #
    # ------------#
    @property
    def count_interval(self) -> ArrayWUnits:
        """
        Doppler count interval :math:`T_c` [s] of the data
        :meth:`default_doppler_aux` simulates, and the bin width of
        :meth:`compress_doppler_to_60s`. Starts as :attr:`COUNT_INTERVAL`.

        A real pass keeps the count time of its tracking file.

        Set an ArrayWUnits time, or a bare, positive number of seconds.
        """
        return ArrayWUnits(self._count_interval, sec)

    @count_interval.setter
    def count_interval(self, value: ArrayWUnits | float) -> None:
        self._count_interval = Measurement._radiometric_default(
            "count_interval", value, sec, minimum=0.0, strict=True
        )

    # ---------#
    # Methods  #
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
        Generate the computed two-way DSN doppler measurement.

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
            Whether to add AWGN at the model's sigma to the computed Doppler
            measurement. For simulating observations; leave False when computing
            predicted measurements for a filter. Defaults to False.

        eps_override: dict
            Dictionary of delta's to compute the numeric _partials

        Returns
        -------
        computed_meas : ArrayWFrame
            An ArrayWFrame object with the computed measurement values expressed in Hz

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

        A count interval that starts before the first ramp record has no
        defined transmit frequency. Its computed value is NaN, with a warning,
        and :meth:`Measurement.generate_measurement_dataset` flags it as an
        outlier.

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
                "DopplerReal: noisy=True needs the measurement sigma."
            )

        # 1. Initialization
        delta = self._build_perturbation_vector(
            eps_override
        )  # Delta dict for _partials
        media_corrections = self.media_corrections  # MediaCorrection object

        # Pull current bias + gs delta from state (if provided). The bias holds
        # one value for every pass or one per pass (see set_pass_epochs).
        bias = self._extract_bias("rangerate_bias_Hz", 1 / sec)
        if bias is not None:
            bias = ArrayWUnits(
                np.atleast_1d(ArrayWUnits.get_value_in_target_units(bias, 1 / sec)),
                1 / sec,
            )
        bias_Hz = np.array(
            [
                0.0 if (b := self._bias_at_epoch(bias, epoch)) is None
                else float(np.squeeze(b.values))
                for epoch in epoch_array
            ]
        )  # [Hz], one per epoch
        # Station location, Earth orientation and tides move the station inside
        # the light-time solution through this correction; the estimated zenith
        # delays are read once here and handed to the media model.
        station_correction = self._itrf_station_correction()
        zt, zi = self._zenith_delays()
        # Estimated scale on the solar-corona delay, 1 when not estimated.
        corona_scale = self._solar_corona_scale()
        # PPN gamma of the light-time solution, Constants.gamma_GR when not estimated.
        gamma = self._gamma_rel()

        # 2. Extract aux_dictionary quantities of interest, one entry per epoch
        t3_et_all = np.atleast_1d(epoch_array.times.values)
        aux_dict = self._aux_per_epoch(len(t3_et_all))
        M2_num = aux_dict["M2_num"]  # Spacecraft turn-around ratio (NUMERATOR)
        M2_den = aux_dict["M2_den"]  # Spacecraft turn-around ratio (DENOMINATOR)
        M2 = M2_num / M2_den  # Spacecraft turn-around ratio
        Tc = aux_dict["Tc"]  # Count interval [s]

        t3s_et_all = t3_et_all - Tc / 2
        t3e_et_all = t3_et_all + Tc / 2

        # Time tags of the count interval for the ramp integration, built from
        # one conversion of t3. Converting t3s and t3e separately rounds each
        # to the ET resolution (~1.2e-7 s at 2018 epochs) and TDB->UTC rounds
        # them differently, so their difference could be off by a count of
        # f_T*1.2e-7 ~ 860 cycles, i.e. ~17 Hz on 60 s X-band Doppler.
        t3_year_all, t3_doy_all, t3_sec_all = SpiceManager.et2YDS(t3_et_all)
        t3_sec_all = np.asarray(t3_sec_all, dtype=float)

        # 3. Group data for the ramp table, read by the corona delay in the
        # RTLT and by the integration below
        ramp_data = Measurement._ramp_table(aux_dict)

        # 4. Compute precise RTLT
        rtlt_s = []
        rtlt_e = []
        t1 = []
        t2 = []
        t3 = epoch_array
        # Endpoint quantities kept for the analytical (Moyer) partials, which
        # differentiate this same light-time solution rather than re-solving it.
        # Recording them is pure observation: nothing below reads these lists, so
        # the computed measurement is unaffected whichever partials method runs.
        _gr_s_cache = []   # GR (down, up) delays at start of count interval [s]
        _gr_e_cache = []   # GR (down, up) delays at end of count interval [s]
        _sc_s_cache = []   # solar corona (down+up) at start [s]
        _sc_e_cache = []   # solar corona (down+up) at end [s]
        _t2s_cache = []    # reflection time at start of count interval [ET s]
        _t2e_cache = []    # reflection time at end of count interval [ET s]
        print("")
        print("Generating 2W Doppler computed measurements...")
        update, close = Measurement._progress_monitor(len(t3_et_all), desc="2W Doppler")
        # Loop through different t3
        for i in range(len(t3_et_all)):
            # Pack a dictionary for electronic delays
            # Index by i, not 0. Using 0 applied the first record's electronic
            # delays to every measurement in the pass, which is wrong whenever
            # any of them vary. ul_zheight_corr in particular carries the
            # invalid sentinel -99.0 on some records, mapped to a zero
            # correction, so a single bad first record zeroed the uplink
            # zenith-height correction for the whole pass. SequentialRangingReal
            # has always indexed this by i.
            electronic_delays_dict = Measurement._generate_electronic_delays_dictionary(
                aux_dict, i, enabled=self._electronic_delays
            )
            # Endpoint-mapped perturbation, when requested. The default rule below
            # puts the whole velocity displacement dv*Tc on the second endpoint and
            # leaves the first unperturbed. That is not the derivative the filter
            # wants: the filter's state is referenced to the reflection midpoint
            # t2_mid, so a velocity deviation there reaches the endpoints as
            # dv*(t2x - t2_mid). Those offsets are +/-(t2e - t2s)/2, not +/-Tc/2:
            # the reflection times are spaced by Tc(1 - rho_dot/c), and using Tc/2
            # biased the velocity column by rho_dot/c, ~5e-5 on a cruise pass.
            # This branch applies the mapping. See _compute_partials_endpoint_fd.
            endpoint_dt = self._endpoint_offsets
            delta_s = delta
            if endpoint_dt is not None and eps_override is not None:
                dt_s = ArrayWUnits(float(endpoint_dt[0][i]), sec)
                eps_s = dict(eps_override)
                for a, v in (("dx", "dvx"), ("dy", "dvy"), ("dz", "dvz")):
                    if v in eps_s:
                        eps_s[a] = eps_s[v] * dt_s
                delta_s = self._build_perturbation_vector(eps_s)
            # Get tt at start and end of interval count, as (high, low) pairs built from t3 so
            # that the count interval does not round the epoch. The time-tag delays are applied
            # by the RTLT driver when it forms t3(TAI) -- applying them here as well subjected
            # every Doppler observation to the correction twice (fixed in 19455f9a).
            t3s_h, t3s_l = EpochArray._add_duration(t3_et_all[i], 0.0, -Tc[i] / 2)
            t3e_h, t3e_l = EpochArray._add_duration(t3_et_all[i], 0.0, Tc[i] / 2)

            # Pack a dictionary for Solar Corona delays (a zero uplink
            # frequency turns the corona delay off; the ramp gives the
            # frequency at t1)
            solar_corona_dict = {
                "M2": M2[i],
                "ul_freq": aux_dict["ul_freq"][i] if self._solar_corona else 0.0,
                "ramp": ramp_data,
                "coefficient": self._solar_corona_coefficient,
            }

            # Generate the rtlt's
            try:
                rtlt_s_i, parts_s_i, t2_s_i, t1_s_i = self._rtlt_solver()(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=t3s_h,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta_s,
                    station_correction=station_correction,
                    tt_et_l=t3s_l,
                    rlt_bodies=self._rlt_bodies,
                    light_time_settings=getattr(self, "_light_time_settings", None),
                    gamma=gamma,
                )

                # Update perturbation so second time perturbation is correlated
                # with initial one. Full Tc in the default (asymmetric) rule,
                # t2e - t2_mid under the endpoint-mapped rule -- see delta_s above.
                dt_e = ArrayWUnits(
                    float(endpoint_dt[1][i]) if endpoint_dt is not None
                    else aux_dict["Tc"][i],
                    sec,
                )
                eps_e = dict(eps_override) if (
                    endpoint_dt is not None and eps_override is not None
                ) else eps_override
                if eps_e is not None:
                    if "dvx" in eps_e.keys():
                        eps_e["dx"] = eps_e["dvx"] * dt_e
                    if "dvy" in eps_e.keys():
                        eps_e["dy"] = eps_e["dvy"] * dt_e
                    if "dvz" in eps_e.keys():
                        eps_e["dz"] = eps_e["dvz"] * dt_e
                delta2 = self._build_perturbation_vector(
                    eps_e
                )  # Delta dict for _partials

                rtlt_e_i, parts_e_i, t2_e_i, t1_e_i = self._rtlt_solver()(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=t3e_h,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta2,
                    station_correction=station_correction,
                    tt_et_l=t3e_l,
                    rlt_bodies=self._rlt_bodies,
                    light_time_settings=getattr(self, "_light_time_settings", None),
                    gamma=gamma,
                )
            #
            except Exception as e:
                # Previously this substituted rtlt = 0 for both ends of the
                # count interval and carried on, which produced a computed
                # Doppler measurement of zero that is indistinguishable
                # downstream from a real one: it reaches the residuals, the
                # filter and the solution as though valid.
                raise RuntimeError(
                    Measurement._rtlt_failure_message(
                        model=self,
                        index=i,
                        total=len(t3_et_all),
                        epoch_et=t3_et_all[i],
                        target=target,
                        cause=e,
                    )
                ) from e
            rtlt_s_i, sc_s_i = Measurement._apply_solar_corona_scale(
                rtlt_s_i, parts_s_i, corona_scale)
            rtlt_e_i, sc_e_i = Measurement._apply_solar_corona_scale(
                rtlt_e_i, parts_e_i, corona_scale)
            rtlt_s.append(rtlt_s_i)
            rtlt_e.append(rtlt_e_i)
            t1.append((t1_e_i + t1_s_i) / 2.0)
            t2.append((t2_e_i + t2_s_i) / 2.0)
            # Endpoint detail for the analytical partials. The parts vector packs
            # the per-leg delay breakdown: [1] and [3] are the down- and up-leg GR
            # (Shapiro) delays, [12] and [14] the down- and up-leg solar corona,
            # cached unscaled as the partial of the light time by its scale.
            parts_s_i = np.asarray(parts_s_i).ravel()
            parts_e_i = np.asarray(parts_e_i).ravel()
            _gr_s_cache.append((float(parts_s_i[1]), float(parts_s_i[3])))
            _gr_e_cache.append((float(parts_e_i[1]), float(parts_e_i[3])))
            _sc_s_cache.append(sc_s_i)
            _sc_e_cache.append(sc_e_i)
            _t2s_cache.append(float(t2_s_i))
            _t2e_cache.append(float(t2_e_i))
            update(i)
        close()

        rtlt_s = np.array(rtlt_s)
        rtlt_e = np.array(rtlt_e)
        t1 = np.array(t1)
        t2 = np.array(t2)

        # 5. Define begin and end intervals for integration with ramp tables,
        # t1 = t3 -/+ Tc/2 - RTLT in seconds of day, as (high, low) pairs. A
        # float second of day resolves only ~7e-12 s at 5e4 s, which rounds
        # the count by f_T*7e-12 ~ 0.05 cycles, steps of ~0.5-1 mHz on 60 s
        # X-band Doppler.
        t1s_sec_integration, t1s_sec_l = EpochArray._add_duration(t3_sec_all, 0.0, -Tc / 2)
        t1s_sec_integration, t1s_sec_l = EpochArray._add_duration(
            t1s_sec_integration, t1s_sec_l, -rtlt_s
        )
        t1e_sec_integration, t1e_sec_l = EpochArray._add_duration(t3_sec_all, 0.0, Tc / 2)
        t1e_sec_integration, t1e_sec_l = EpochArray._add_duration(
            t1e_sec_integration, t1e_sec_l, -rtlt_e
        )

        # Adjust for doy rollover events
        # Each end of the count interval carries its own day, brought back
        # within one day: t1 precedes t3 by the light time, and t3e can
        # pass midnight. The year is not rolled: the ramp integration counts
        # days from 1 January of the t3 year, so day 0 or day 366 of a common
        # year is the neighbouring year.
        day_shift = np.floor(t1s_sec_integration / constants.day_to_sec)
        t1s_sec_integration, t1s_sec_l = EpochArray._add_duration(
            t1s_sec_integration, t1s_sec_l, -day_shift * constants.day_to_sec
        )
        t1s_doy_integration = np.asarray(t3_doy_all, dtype=float) + day_shift
        day_shift = np.floor(t1e_sec_integration / constants.day_to_sec)
        t1e_sec_integration, t1e_sec_l = EpochArray._add_duration(
            t1e_sec_integration, t1e_sec_l, -day_shift * constants.day_to_sec
        )
        t1e_doy_integration = np.asarray(t3_doy_all, dtype=float) + day_shift

        # 7. Generate the computed measurement by integration with the ramp table
        doppler_meas_list = []
        uncovered = []  # count intervals with no ramp record at the transmit time
        first_gap = None  # the first such gap's explanation, reported once
        used_tropo_flag = np.zeros(len(rtlt_s))
        used_iono_flag = np.zeros(len(rtlt_s))
        _f_t1s_cache = []   # transmit frequency at start of count interval [Hz]
        _f_t1e_cache = []   # transmit frequency at end of count interval [Hz]
        for i in range(len(rtlt_s)):
            # 7.1 Perform the integration between t1 and t3 to generate f_int
            try:
                f_int, aux_int = RampTableManager.integrate(
                    ramp_data,
                    t1s_sec_integration[i],
                    t1e_sec_integration[i],
                    t1s_doy_integration[i],
                    t1e_doy_integration[i],
                    t3_year_all[i],
                    t3_year_all[i],
                    t1_q_sec_l=t1s_sec_l[i],
                    t2_q_sec_l=t1e_sec_l[i],
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
            # Endpoint transmit frequencies for the analytical partials. Taken here
            # rather than in the media branch below so they are recorded whether or
            # not media corrections are applied; ul_freq is the unramped fallback.
            _f_t1s_cache.append(float(aux_int.get("f_iq1", aux_dict["ul_freq"][i])))
            _f_t1e_cache.append(float(aux_int.get("f_iq2", aux_dict["ul_freq"][i])))
            doppler_meas = -(M2[i] / Tc[i]) * f_int + bias_Hz[i]
            # 7.2 Add media correction effects
            if media_corrections is None:
                doppler_meas_list.append(doppler_meas)
            else:
                if (
                    aux_int["n_segments"] >= 1
                ):  # Only compute media correction if the integration is successful
                    # Generate frequencies of interest
                    f_t1_s = aux_int["f_iq1"]  # Moyer 13.2.8, [Hz]
                    f_t3_s = f_t1_s * M2[i]  # Moyer 13.2.8, [Hz]
                    f_t1_e = aux_int["f_iq2"]  # Moyer 13.2.8, [Hz]
                    f_t3_e = f_t1_e * M2[i]  # Moyer 13.2.8, [Hz]
                    # Compute tropo + iono delays
                    p_tropo_t3s, p_iono_t3s = media_corrections.computed_rtlt_delays(
                        t3s_et_all[i], str(target.spice_id), f_t3_s,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    p_tropo_t1s, p_iono_t1s = media_corrections.computed_rtlt_delays(
                        t3s_et_all[i] - rtlt_s[i],
                        str(target.spice_id),
                        f_t1_s,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    p_tropo_t3e, p_iono_t3e = media_corrections.computed_rtlt_delays(
                        t3e_et_all[i], str(target.spice_id), f_t3_e,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    p_tropo_t1e, p_iono_t1e = media_corrections.computed_rtlt_delays(
                        t3e_et_all[i] - rtlt_e[i],
                        str(target.spice_id),
                        f_t1_e,
                        tropo_zenith_delay=zt, iono_zenith_delay=zi,
                    )  # [m]
                    # NOTE: Sign of iono is negative for doppler, positive for sranging
                    p_total_s = MediaCorrections.path_delay_to_light_time(
                        p_tropo_t3s - p_iono_t3s + p_tropo_t1s - p_iono_t1s
                    )  # Moyer 10-29, [s]
                    p_total_e = MediaCorrections.path_delay_to_light_time(
                        p_tropo_t3e - p_iono_t3e + p_tropo_t1e - p_iono_t1e
                    )  # Moyer 10-28, [s]
                    # Add delay into measurement
                    p_total_Hz = (M2[i] / Tc[i]) * (
                        f_t1_e * p_total_e - f_t1_s * p_total_s
                    )  # Moyer 13-58
                    doppler_meas_media = doppler_meas + p_total_Hz
                    doppler_meas_list.append(doppler_meas_media)
                    # Keep track of the media correction used. .values because
                    # the media delays are unit-bearing now; ArrayWUnits refuses
                    # comparison against a bare number.
                    if (
                        (p_tropo_t3s.values != 0.0)
                        or (p_tropo_t1s.values != 0.0)
                        or (p_tropo_t3e.values != 0.0)
                        or (p_tropo_t1e.values != 0.0)
                    ):
                        used_tropo_flag[i] = 1
                    if (
                        (p_iono_t3s.values != 0.0)
                        or (p_iono_t1s.values != 0.0)
                        or (p_iono_t3e.values != 0.0)
                        or (p_iono_t1e.values != 0.0)
                    ):
                        used_iono_flag[i] = 1
                else:
                    doppler_meas_list.append(doppler_meas)

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
                f"DopplerReal '{self.name}': {len(uncovered)} of {len(rtlt_s)} "
                "count intervals have no ramp record at their transmit time "
                f"(epoch indices {uncovered}); their computed Doppler is NaN and "
                f"they are flagged as outliers. First: {first_gap}"
            )

        meas_awu = ArrayWUnits(doppler_meas_list, 1 / sec)

        # Noise, for simulating observations (the sigma was checked on entry).
        # Predicted measurements for a filter are computed with noisy=False and
        # are left untouched.
        if noisy:
            sigma_Hz = float(
                np.squeeze(ArrayWUnits.get_value_in_target_units(self.sigma, 1 / sec))
            )
            meas_awu = meas_awu + Noise().generate_AWGN_with_units(
                mu=0.0, sigma=sigma_Hz, units=1 / sec, count=len(doppler_meas_list)
            )

        # Publish the endpoint cache for the analytical (Moyer 13-59) partials.
        # Keyed by the reception epochs so the partials can confirm the cache is
        # current for the epoch_array they are handed (see _cache_matches).
        self._last_f_t1s = np.array(_f_t1s_cache)   # f_T at start of count interval [Hz]
        self._last_f_t1e = np.array(_f_t1e_cache)   # f_T at end of count interval [Hz]
        self._last_t3s_et = np.array(t3s_et_all)    # reception epoch at start [ET s]
        self._last_t3e_et = np.array(t3e_et_all)    # reception epoch at end [ET s]
        self._last_rtlt_s = np.array(rtlt_s)        # RTLT at start [s]
        self._last_rtlt_e = np.array(rtlt_e)        # RTLT at end [s]
        self._last_gr_s = np.array(_gr_s_cache).reshape(-1, 2)  # GR (dn, up) at start [s]
        self._last_gr_e = np.array(_gr_e_cache).reshape(-1, 2)  # GR (dn, up) at end [s]
        self._last_sc_s = np.array(_sc_s_cache)     # solar corona (dn+up) at start [s]
        self._last_sc_e = np.array(_sc_e_cache)     # solar corona (dn+up) at end [s]
        self._last_t2s = np.array(_t2s_cache)       # reflection time at start [ET s]
        self._last_t2e = np.array(_t2e_cache)       # reflection time at end [ET s]
        self._last_t2_mid = np.array(t2)            # midpoint reflection time = filter t2
        # Fingerprint of the reception epochs this cache was built for
        self._partials_cache_key = np.array(t3_et_all)

        return t1, t2, t3, ArrayWFrame(meas_awu, J2000)

    def _compute_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list[np.ndarray]:
        """
        Compute the Doppler partials with the method chosen by ``partials_method``.

        Parameters
        ----------
        target : Spacecraft
            Spacecraft the Doppler data tracks.
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
        ``"analytical"`` (the default) calls :meth:`_compute_analytical_partials`,
        ``"fd"`` calls :meth:`_compute_partials_with_fd` and ``"fd_endpoint"`` calls
        :meth:`_compute_partials_endpoint_fd`.
        """

        method = self._partials_method
        if method == "fd" and self._sequence_definition is not None:
            raise Measurement.SequencePartialsError(
                f"{type(self).__name__} cannot finite-difference the partials "
                "across a mission sequence; use partials_method='analytical'."
            )
        if method == "fd":
            return self._compute_partials_with_fd(
                target, epoch_array, frame, bias_key="rangerate_bias_Hz"
            )
        if method == "fd_endpoint":
            return self._compute_partials_endpoint_fd(target, epoch_array, frame)
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

    def _h_tilde_zenith_delay_doppler(
        self,
        a_t3e: float,
        a_t1e: float,
        a_t3s: float,
        a_t1s: float,
        f_T_e: float,
        f_T_s: float,
        M2: float,
        Tc: float,
        sign: float,
    ) -> float:
        """
        Partial of the two-way Doppler observable with respect to a zenith media
        delay.

        Parameters
        ----------
        a_t3e, a_t1e : float
            Partials [m/m] of the one-way slant delay by the zenith delay on the
            downleg (reception :math:`t_3`) and the upleg (transmission
            :math:`t_1`) at the end of the count interval, from
            :meth:`MediaCorrections.zenith_delay_partials`.
        a_t3s, a_t1s : float
            The same partials at the start of the count interval.
        f_T_e, f_T_s : float
            Transmitter frequency [Hz] at the transmission time of the end and of
            the start of the count interval.
        M2 : float
            Spacecraft transponder turnaround ratio.
        Tc : float
            Count interval [s].
        sign : float
            ``+1`` for the troposphere, ``-1`` for the ionosphere.

        Returns
        -------
        float
            :math:`\\partial F / \\partial \\rho_z` [Hz/m].

        Notes
        -----
        The zenith delay maps onto each leg at its own elevation and enters the
        Doppler observable through the count-interval difference of Moyer
        eq. 13-59:

        .. math::

            \\frac{\\partial F}{\\partial \\rho_z} = \\pm \\frac{M_2}{T_c\\,c} \\left[
                f_{T,e} \\left( a(t_{3e}) + a(t_{1e}) \\right)
                - f_{T,s} \\left( a(t_{3s}) + a(t_{1s}) \\right) \\right]

        with :math:`c` in m/s. The sign is negative for the ionosphere, which
        advances the phase and delays the group (Moyer eqs. 10-28 and 10-29).
        For unramped data :math:`f_{T,e} = f_{T,s}`, so the partial follows the
        rate of change of the mapping over the count interval and is typically
        very small; use ranging data (:class:`SequentialRangingReal`) to
        estimate zenith delays.

        References
        ----------
        .. [1] Moyer, T. D. (2000). *Formulation for Observed and Computed Values of
           Deep Space Network Data Types for Navigation*. JPL Monograph 2,
           eqs. 10-28, 10-29 and 13-59.
        """
        return float(
            sign
            * (M2 / Tc)
            * (f_T_e * (a_t3e + a_t1e) - f_T_s * (a_t3s + a_t1s))
            / (float(constants.c.values) * constants.km_to_m)
        )

    def _endpoint_light_path(self, target_name: str, t3x: float, frame_name: str, c: float):
        """
        Light-path geometry at one count-interval endpoint, received at ``t3x``.

        Parameters
        ----------
        target_name : str
            SPICE name of the spacecraft.
        t3x : float
            Reception epoch at the station [ET s].
        frame_name : str
            Inertial frame the vectors are expressed in.
        c : float
            Speed of light [km/s].

        Returns
        -------
        n_dn, n_up : np.ndarray
            Unit vectors (3,) from the station to the spacecraft on the downleg and
            on the upleg.
        v_sc, v_gs : np.ndarray
            Transmitter velocities (3,) of Moyer eq. 8-60 in the light-time frame
            [km/s]: the spacecraft at :math:`t_2` and the station at :math:`t_1`, not
            the station-relative velocity.
        t2x, t1x : float
            Reflection and transmission epochs [ET s].
        """
        rel_dn = SpiceManager.get_state(target_name, t3x, frame_name, self.instrument.name, "LT")
        rho_dn = float(rel_dn[0:3].norm().values)
        n_dn = rel_dn[0:3].values / rho_dn
        t2x = t3x - rho_dn / c
        rel_up = SpiceManager.get_state(self.instrument.name, t2x, frame_name, target_name, "LT")
        rho_up = float(rel_up[0:3].norm().values)
        n_up = -rel_up[0:3].values / rho_up
        v_sc = self._light_time_velocity(target_name, t2x, frame_name)
        t1x = t2x - rho_up / c
        v_gs = self._light_time_velocity(self.instrument.name, t1x, frame_name)
        return n_dn, n_up, v_sc, v_gs, t2x, t1x

    def _doppler_state_row_moyer(
        self,
        target_name: str,
        frame_name: str,
        epoch_idx: int,
        M2: float,
        c: float,
    ) -> np.ndarray:
        """
        Pre-scaled state partial row of the two-way Doppler observable, referenced
        to the spacecraft reflection time :math:`t_{2,mid}`.

        :math:`t_{2,mid}` is the time the OD filter associates with the measurement
        and maps from with :math:`\\Phi(t_0, t_{2,mid})`.

        Parameters
        ----------
        target_name : str
            SPICE name of the spacecraft.
        frame_name : str
            Inertial frame the partials are expressed in.
        epoch_idx : int
            Index of the measurement in the light-time cache filled by
            :meth:`computed_measurements`.
        M2 : float
            Spacecraft transponder turnaround ratio.
        c : float
            Speed of light [km/s].

        Returns
        -------
        np.ndarray
            Row (6,) holding :math:`\\partial F / \\partial \\mathbf{r}` and
            :math:`\\partial F / \\partial \\mathbf{v}` at :math:`t_{2,mid}`,
            pre-scaled by :math:`c / f_{UL}`. The caller's scale factor
            :math:`f_{UL} / c` recovers Hz/km and Hz/(km/s).

        Notes
        -----
        Moyer eq. 13-59 with the count-interval endpoints mapped to
        :math:`t_{2,mid}` through the real state transition matrix (Moyer ch. 12):

        .. math::

            \\frac{\\partial F}{\\partial \\mathbf{X}(t_{2,mid})} =
                \\frac{M_2}{T_c} \\left[
                f_{T,e}\\, \\mathbf{g}^e\\, \\Phi(t_2^e, t_{2,mid})
                - f_{T,s}\\, \\mathbf{g}^s\\, \\Phi(t_2^s, t_{2,mid}) \\right]

        where :math:`\\mathbf{g}^x = [\\partial\\rho^x / \\partial\\mathbf{r}_{SC}(t_2^x),
        \\mathbf{0}]` is the geometric light-time gradient of
        :meth:`Measurement.rtlt_grad_pos_exact` (Moyer eqs. 12-12 and 12-17). It has
        no velocity part because :math:`\\rho` is geometric in the reflection-time
        position, so the velocity sensitivity comes entirely from the
        :math:`\\Phi_{rv}` block and carries the true dynamics. Without a reference
        trajectory the STM falls back to free flight,
        :math:`\\Phi_{rr} = I` and :math:`\\Phi_{rv} = (t_2^x - t_{2,mid})\\, I`.

        References
        ----------
        .. [1] Moyer, T. D. (2000). *Formulation for Observed and Computed Values of
           Deep Space Network Data Types for Navigation*. JPL Monograph 2,
           eqs. 12-12, 12-17 and 13-59.
        """
        aux_dict = self.computed_measurements_dict
        Tc_i = float(aux_dict["Tc"][epoch_idx])
        ul_freq = float(aux_dict["ul_freq"][epoch_idx])

        f_Te = float(self._last_f_t1e[epoch_idx])
        f_Ts = float(self._last_f_t1s[epoch_idx])
        t3e = float(self._last_t3e_et[epoch_idx])
        t3s = float(self._last_t3s_et[epoch_idx])
        t2e = float(self._last_t2e[epoch_idx])
        t2s = float(self._last_t2s[epoch_idx])
        t2_mid = float(self._last_t2_mid[epoch_idx])

        def _grad_pos(t3x):
            """Geometric light-time position gradient :math:`\\partial\\rho / \\partial\\mathbf{r}_{SC}(t_2)` [s/km] for reception at ``t3x``."""
            n_dn, n_up, v_sc, v_gs, *_ = self._endpoint_light_path(target_name, t3x, frame_name, c)
            return Measurement.rtlt_grad_pos_exact(n_dn, n_up, v_sc, v_gs, c)  # (3,) [s/km]

        def _rho_row6(t3x, t2x):
            """Row (6,) :math:`\\partial\\rho / \\partial\\mathbf{X}(t_{2,mid})`: the geometric gradient mapped with the real STM :math:`\\Phi(t_2^x, t_{2,mid})`."""
            if t3x == 0.0:
                return np.zeros(6)
            g = _grad_pos(t3x)                                  # (3,) ∂ρ/∂r_SC(t2x)
            Phi = self._short_arc_stm(t2x, t2_mid)              # (6,6) real STM, or None
            if Phi is None:
                # Free-flight fallback: Φ_rr = I, Φ_rv = (t2x − t2_mid)·I
                row = np.zeros(6)
                row[0:3] = g
                row[3:6] = g * (t2x - t2_mid)
                return row
            # ρ depends on r_SC(t2x) only → use the position rows (0:3) of Φ
            return g @ Phi[0:3, 0:6]                            # (6,)

        row_e = _rho_row6(t3e, t2e)
        row_s = _rho_row6(t3s, t2s)

        # Moyer 13-59, pre-scale (×c/ul_freq so the trailing scale_fac=ul_freq/c gives Hz)
        return (M2 / Tc_i) * (f_Te * row_e - f_Ts * row_s) * (c / ul_freq)

    def _station_row_doppler(
        self,
        component_def,
        target_name: str,
        frame_name: str,
        epoch_idx: int,
        M2: float,
        c: float,
    ) -> list:
        """
        Pre-scaled partial of the two-way Doppler observable with respect to one
        station parameter.

        Parameters
        ----------
        component_def : sequence
            State entry ``(name, dimension, flag, dynamics, owner, ...)``: a
            location (Cartesian or cylindrical), orientation, EOP or Love/Shida
            component.
        target_name : str
            SPICE name of the spacecraft.
        frame_name : str
            Inertial frame of the light-time solution.
        epoch_idx : int
            Index of the measurement in the light-time cache.
        M2 : float
            Spacecraft transponder turnaround ratio.
        c : float
            Speed of light [km/s].

        Returns
        -------
        list of float
            :math:`\\partial F / \\partial p`, one entry per component, pre-scaled
            by :math:`c / f_{UL}` like :meth:`_doppler_state_row_moyer`; zeros
            when the component does not move this station.

        Notes
        -----
        At each count-interval endpoint :math:`x` the round-trip light time moves
        with the station at reception and at transmission,

        .. math::

            \\frac{\\partial\\rho^x}{\\partial p} =
                \\mathbf{g}_3^x J(t_3^x) + \\mathbf{g}_1^x J(t_1^x)

        where :math:`J` is the station Jacobian of
        :meth:`Measurement._station_state_partials` at the leg epoch (Earth
        rotation makes :math:`J(t_3) \\neq J(t_1)`), and :math:`\\mathbf{g}_3`,
        :math:`\\mathbf{g}_1` are the exact per-leg station gradients of
        :meth:`Measurement.rtlt_grad_station_exact` plus the station's ET - TAI
        term from :meth:`Measurement._station_clock_gradient` (Moyer eq. 2-23),
        assembled by :meth:`Measurement._rtlt_station_row`. A static offset has
        a Doppler signature only because :math:`J` rotates over the count
        interval; the endpoints are combined through Moyer eq. 13-59:

        .. math::

            \\frac{\\partial F}{\\partial p} = \\frac{M_2}{T_c} \\left[
                f_{T,e} \\frac{\\partial\\rho^e}{\\partial p}
                - f_{T,s} \\frac{\\partial\\rho^s}{\\partial p} \\right]

        Checked against a central difference of the forward model, with the
        parameter perturbed in the reference state.

        References
        ----------
        .. [1] Moyer, T. D. (2000). *Formulation for Observed and Computed Values of
           Deep Space Network Data Types for Navigation*. JPL Monograph 2,
           eqs. 2-23, 12-12, 12-17 and 13-59.
        """
        aux_dict = self.computed_measurements_dict
        Tc_i = float(aux_dict["Tc"][epoch_idx])
        ul_freq = float(aux_dict["ul_freq"][epoch_idx])
        f_Te = float(self._last_f_t1e[epoch_idx])
        f_Ts = float(self._last_f_t1s[epoch_idx])
        rows = []
        for t3x, rtlt_x in ((self._last_t3e_et[epoch_idx], self._last_rtlt_e[epoch_idx]),
                            (self._last_t3s_et[epoch_idx], self._last_rtlt_s[epoch_idx])):
            t3x = float(t3x)
            n_dn, n_up, v_sc, v_gs, *_ = self._endpoint_light_path(target_name, t3x, frame_name, c)
            rows.append(self._rtlt_station_row(
                component_def, t3x, t3x - float(rtlt_x), n_dn, n_up, v_sc, v_gs, frame_name, c))
        row_e, row_s = rows
        return list((M2 / Tc_i) * (f_Te * row_e - f_Ts * row_s) * (c / ul_freq))

    def _compute_analytical_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> np.ndarray:
        """
        Assemble the analytical H-tilde rows for this Doppler model.

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
            times ``t3``; the spacecraft transmit time ``t2`` and the station
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
            If the state holds a stacked per-pass ``rangerate_bias_Hz`` for
            this station (dimension > 1) and no epoch-to-pass mapping is set.

        Notes
        -----
        The endpoint cache is populated on demand through
        :meth:`_ensure_partials_cache`, which runs the forward model when the
        cached ramp frequencies, reflection times and delays do not match
        ``epoch_array``.

        The per-epoch transponder turnaround ratio :math:`M_2` is taken from the
        ``M2_num`` / ``M2_den`` entries of ``computed_measurements_dict``.

        With a sequence definition each row takes the state definition of the
        leg holding its epoch (see :meth:`_use_leg_state_definition`), so rows
        of different legs can differ in width.
        """

        # Ensure the endpoint cache is present and current for these epochs
        # (transparently runs the forward model on a cache miss).
        self._ensure_partials_cache(target, epoch_array, frame)

        partials_list = []

        c_val = constants.c.values  # km/s
        aux_dict = self.computed_measurements_dict
        M2_num_arr = np.array(aux_dict["M2_num"])
        M2_den_arr = np.array(aux_dict["M2_den"])

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        for i, epoch in enumerate(epoch_array):
            # In a sequence each row takes the state definition of its leg
            self._use_leg_state_definition(epoch)

            # Every row below evaluates its own endpoint geometry (t3e/t3s and
            # their reflection times) from the forward-model cache.

            # Per-epoch transponder ratio M₂
            idx = min(i, len(M2_num_arr) - 1)
            M2_i = float(M2_num_arr[idx]) / float(M2_den_arr[idx])

            # The position and velocity blocks are the two halves of one Moyer
            # row, computed on first use within the epoch
            state_row = None

            # Stack h_tilde based on StateVector components
            h_tilde = []
            target_sid = getattr(target, "spice_id", None)

            if self._state_definition is None:
                state_row = self._doppler_state_row_moyer(
                    target.name, frame.name, idx, M2_i, c_val
                )
                h_tilde.extend(list(state_row))
            else:
                for component_def in self._state_definition:
                    component = component_def[0]
                    comp_body = component_def[4] if len(component_def) > 4 else None
                    comp_body_sid = getattr(comp_body, "spice_id", None)

                    if state_allowed["position"][2].match(component):
                        if comp_body_sid == target_sid:
                            if state_row is None:
                                state_row = self._doppler_state_row_moyer(
                                    target.name, frame.name, idx, M2_i, c_val
                                )
                            h_tilde.extend(list(state_row[0:3]))
                        else:
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["velocity"][2].match(component):
                        # Velocity sensitivity from the Phi_rv block of the real
                        # STM, not the free-flight dt*I, whenever a reference
                        # trajectory is present
                        if comp_body_sid == target_sid:
                            if state_row is None:
                                state_row = self._doppler_state_row_moyer(
                                    target.name, frame.name, idx, M2_i, c_val
                                )
                            h_tilde.extend(list(state_row[3:6]))
                        else:
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["eta_srp"][2].match(component):
                        # No direct sensitivity: SRP moves the light time only through
                        # the trajectory, which the filter maps through the STM.
                        h_tilde.extend([0.0])

                    elif self._is_station_parameter(component):
                        # Location (Cartesian or cylindrical), orientation, EOP
                        # and Love/Shida numbers move the station inside the
                        # light-time solution at t1 and t3 of both endpoints.
                        # rtlt_parts[10] stays zero: adding it as well would
                        # double-count the geometric term.
                        h_tilde.extend(self._station_row_doppler(
                            component_def, target.name, frame.name, i, M2_i, c_val
                        ))

                    elif "rangerate_bias_Hz" in state_allowed and state_allowed[
                        "rangerate_bias_Hz"
                    ][2].match(component):
                        # F = F_geom + bias_Hz  ⇒  ∂F/∂bias = 1 for THIS station, else 0,
                        # in the slot of the epoch's pass for one bias per pass.
                        # Stored pre-scaled (×c/ul_freq) so the trailing scale_fac=ul_freq/c
                        # yields exactly 1.
                        bias_partials = self._bias_partials(
                            component, component_def[1], comp_body, epoch
                        )
                        if any(bias_partials):
                            ul_freq_i = float(aux_dict["ul_freq"][idx])
                            bias_partials = [p * c_val / ul_freq_i for p in bias_partials]
                        h_tilde.extend(bias_partials)

                    elif state_allowed["dv_man"][2].match(component):
                        # No direct sensitivity: the maneuver moves the light time only
                        # through the trajectory, which the filter maps through the STM.
                        h_tilde.extend([0.0] * 3)
                    # ---------------------------------------------------- #
                    # Moyer correction partials (Eqs. 10-28/29 + 13-59)     #
                    # ---------------------------------------------------- #
                    elif state_allowed["gamma_rel"][2].match(component):
                        # ∂F/∂γ = (M₂/Tc)[f_Te·∂ρe/∂γ − f_Ts·∂ρs/∂γ]   [Moyer 8-55, 13-59]
                        # with each leg's RLT/(1+γ) weighted by its Moyer 12-12/12-17
                        # factor, since the Shapiro delay sits inside the light-time
                        # solution, plus the γ in the eq. 4-10 space-time transform of
                        # the station (barycentric) or spacecraft (geocentric). The
                        # station term follows Earth rotation, and without it this
                        # row was 2.4% off. γ is a global PPN parameter.
                        gamma = self._gamma_rel()  # what the light time used
                        Tc_i = float(aux_dict["Tc"][idx])
                        ul_freq_i = float(aux_dict["ul_freq"][idx])
                        f_Te = float(self._last_f_t1e[i])
                        f_Ts = float(self._last_f_t1s[i])
                        drho = []
                        for t3x, gr in ((self._last_t3e_et[i], self._last_gr_e[i]),
                                        (self._last_t3s_et[i], self._last_gr_s[i])):
                            t3x = float(t3x)
                            n_dn, n_up, v_sc, v_gs, t2x, t1x = self._endpoint_light_path(
                                target.name, t3x, frame.name, c_val
                            )
                            w_dn, w_up = Measurement.rtlt_leg_weights(n_dn, n_up, v_sc, v_gs, c_val)
                            g_sc = Measurement.rtlt_grad_pos_exact(n_dn, n_up, v_sc, v_gs, c_val)
                            g3, g1 = Measurement.rtlt_grad_station_exact(n_dn, n_up, v_sc, v_gs, c_val)
                            gs_name = self.instrument.name
                            drho.append(
                                (w_dn * gr[0] + w_up * gr[1]) / (1.0 + gamma)
                                + g_sc @ self._gamma_position_shift(target.name, t2x, frame.name)
                                + g3 @ self._gamma_position_shift(gs_name, t3x, frame.name)
                                + g1 @ self._gamma_position_shift(gs_name, t1x, frame.name)
                            )
                        drho_e, drho_s = drho
                        dF_dgamma = (M2_i / Tc_i) * (f_Te * drho_e - f_Ts * drho_s)
                        # Pre-scale: ×(c/ul_freq) so final ×scale_fac(=ul_freq/c) recovers Hz
                        h_tilde.extend([dF_dgamma * c_val / ul_freq_i])

                    elif state_allowed["solar_corona_scale"][2].match(component):
                        # ∂F/∂κ_SC = (M₂/Tc)[f_Te·∆SC_e − f_Ts·∆SC_s]   [Moyer 10-76, 13-59]
                        # SC enters ρ directly (it is part of the light-time solution,
                        # not an after-the-fact media correction). κ scales the model (κ=1).
                        Tc_i = float(aux_dict["Tc"][idx])
                        ul_freq_i = float(aux_dict["ul_freq"][idx])
                        f_Te = float(self._last_f_t1e[i])
                        f_Ts = float(self._last_f_t1s[i])
                        dF_dkappa = (M2_i / Tc_i) * (
                            f_Te * float(self._last_sc_e[i]) - f_Ts * float(self._last_sc_s[i])
                        )
                        h_tilde.extend([dF_dkappa * c_val / ul_freq_i])

                    elif self._is_media_parameter(component):
                        # ∂F/∂ρ_z, Moyer 10-28/29 + 13-59: the zenith delay mapped on
                        # every leg at its own elevation and, for the ionosphere, its
                        # own frequency. Negative for the ionosphere, as in the
                        # forward model.
                        parameter = next((p for p in self._media_parameters()
                                          if p["name"] == component), None)
                        if self._media_corrections is None or parameter is None:
                            h_tilde.extend([0.0] * component_def[1])
                        else:
                            mc = self._media_corrections
                            sc_id = str(target.spice_id)
                            Tc_i = float(aux_dict["Tc"][idx])
                            ul_freq_i = float(aux_dict["ul_freq"][idx])
                            t3e = float(self._last_t3e_et[i]); t3s = float(self._last_t3s_et[i])
                            t1e = t3e - float(self._last_rtlt_e[i])
                            t1s = t3s - float(self._last_rtlt_s[i])
                            f_t1e = float(self._last_f_t1e[i]); f_t1s = float(self._last_f_t1s[i])
                            column = 0 if parameter["kind"] == "tropo_zenith_delay" else 1
                            a_t3e, a_t1e, a_t3s, a_t1s = (
                                mc.zenith_delay_partials(t, sc_id, f)[column]
                                for t, f in ((t3e, f_t1e * M2_i), (t1e, f_t1e),
                                             (t3s, f_t1s * M2_i), (t1s, f_t1s)))
                            h_val = self._h_tilde_zenith_delay_doppler(
                                a_t3e, a_t1e, a_t3s, a_t1s, f_t1e, f_t1s, M2_i, Tc_i,
                                1.0 if column == 0 else -1.0)
                            h_tilde.extend(list(h_val * parameter["scale"] * c_val / ul_freq_i))

                    else:
                        h_tilde.extend([0.0 for _ in range(component_def[1])])

            partials_list.append(h_tilde)
            #
        #
        # Every entry above is pre-scaled by c/ul_freq of its own epoch, so the
        # trailing factor has to use that same epoch's ul_freq. A single
        # ul_freq[0] left every row whose uplink differs from the first epoch's
        # off by ul_freq[0]/ul_freq[i].
        aux_dict = self.computed_measurements_dict
        ul_freq_arr = aux_dict["ul_freq"]
        c = constants.c.values
        out = []
        for ii in range(len(partials_list)):
            ul_freq_i = float(ul_freq_arr[min(ii, len(ul_freq_arr) - 1)])
            out.append((ul_freq_i / c) * np.array(partials_list[ii]))
        return out

    # endregion Analytical (Moyer) partials #
    # ---------------------------------------------------------------------#

    # This model differences a count interval, so the endpoint-mapped FD
    # reference is meaningful here (and only here).
    PARTIALS_METHODS = ("fd", "fd_endpoint", "analytical")

    COUNT_INTERVAL = ArrayWUnits(60.0, sec)
    """Default Doppler count interval, :math:`T_c`, the starting
    :attr:`count_interval` of a model.

    The count time of simulated Doppler and the bin width
    :meth:`compress_doppler_to_60s` compresses real Doppler to.
    """

    AUX_KEYS = Measurement.TrackingAuxKeys(
        required=(
            "Tc", "M2_num", "M2_den", "ul_freq",
            "rcv_time_tag_delay", "transmit_time_tag_delay",
            "ul_zheight_corr", "dl_zheight_corr", "ul_stn_cal", "dl_stn_cal",
        ),
        optional=("array_delay", "exc_scalar_num", "exc_scalar_den"),
        record_fields={
            "cnt_time": "Tc",
            "scft_transpd_turn_num": "M2_num",
            "scft_transpd_turn_den": "M2_den",
        },
        outlier_flag="outlier_flag_doppler",
    )
    """Auxiliary-data layout of this model."""

    def _compute_partials_endpoint_fd(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Endpoint-mapped central-difference partials, through the full forward model.

        The plain finite-difference path (:meth:`_compute_partials_with_fd`) is not
        a valid reference for two-way Doppler. The observable is a difference
        between the two ends of a count interval, so where a state perturbation
        lands is part of the derivative's definition, not an implementation
        detail. That path perturbs once and puts the entire velocity displacement
        ``dv*Tc`` on the second endpoint, leaving the first unperturbed, because
        its ``delta`` is built once before the epoch loop.

        The filter's state is referenced to the reflection midpoint ``t2_mid``, so
        a velocity deviation there reaches the endpoints as ``dv*(t2x - t2_mid)``,
        with ``t2x`` the reflection times of an unperturbed solution:
        ``-(t2e - t2s)/2`` at the start and ``+(t2e - t2s)/2`` at the end. That is
        ``Tc(1 - rho_dot/c)/2``, not ``Tc/2``; the reflection times are closer
        together than the reception times by the downleg range rate. This method
        applies that mapping and central-differences the result, which makes it
        comparable to :meth:`_compute_analytical_partials` -- the analytical row
        maps the same way, through ``Phi(t2x, t2_mid)``.

        Everything else runs unchanged, so this differences the real observable:
        Shapiro, solar corona, media, ramped frequencies and station electronics
        all participate. That is the difference from the reduced reference used
        during the original verification, which re-solved a geometric light time
        of its own and dropped the Shapiro term.

        Parameters
        ----------
        target : Spacecraft
            The observed spacecraft.
        epoch_array : EpochArray
            Epochs to evaluate.
        frame : Frame, optional
            Reference frame.

        Returns
        -------
        list of numpy.ndarray
            One 1x6 Jacobian row per epoch, ``[d/dr, d/dv]``.
        """
        eps_dict = self._fd_step_sizes()

        comps = ["dx", "dy", "dz", "dvx", "dvy", "dvz"]
        N = epoch_array.size
        H = np.zeros((N, 6))

        # A fresh dict per evaluation: the forward model inserts a position entry
        # derived from the velocity step, so a reused dict would leak one
        # endpoint's displacement into the next call.
        def observable(comp, step):
            _, _, _, out = self.computed_measurements(
                target=target,
                epoch_array=epoch_array,
                frame=frame,
                noisy=False,
                eps_override={comp: step},
            )
            return np.asarray(out.quantity.values, dtype=float).ravel()

        # An unperturbed solution first, for the reflection times the velocity
        # step is mapped through. The perturbed calls overwrite the cache, so the
        # offsets are copied out before they run.
        self.computed_measurements(
            target=target, epoch_array=epoch_array, frame=frame, noisy=False
        )
        was = self._endpoint_offsets
        self._endpoint_offsets = (
            np.array(self._last_t2s - self._last_t2_mid),
            np.array(self._last_t2e - self._last_t2_mid),
        )
        try:
            for j, comp in enumerate(comps):
                eps = eps_dict[comp]
                plus = observable(comp, eps)
                minus = observable(comp, ArrayWUnits(-eps.values, eps.units))
                H[:, j] = (plus - minus) / (2.0 * float(eps.values))
        finally:
            self._endpoint_offsets = was

        return [H[i, :] for i in range(N)]

    def compress_doppler_to_60s(
        self,
        t_sec,
        y,
        *,
        t0=None,
        agg="mean",  # "mean","median","sum","min","max"
        sample_dt=None,  # will auto-detect if None
        min_coverage=0.8,
    ):  # require ≥80% samples in a bin
        """
        Compress Doppler samples to 60-s bins, unless already at ~60-s spacing.

        The bin width is :attr:`count_interval`, 60 s by default; the names
        below say 60 s for that default.

        Parameters
        ----------
        t_sec : array-like
            Sample mid-times [s].
        y : array-like
            Observable per sample (e.g., Hz). NaNs are ignored.
        t0 : float or None
            Bin alignment epoch [s]. If None, uses floor(min(t)/60)*60.
        agg : str
            Aggregation method for 1-s input: "mean","median","sum","min","max".
        sample_dt : float or None
            Nominal spacing of input samples. If None, it is estimated from median diff.
        min_coverage : float
            Required fraction of expected samples (60/sample_dt). If coverage is lower,
            output value is set to NaN.

        Returns
        -------
        t60_mid : np.ndarray
            60-s bin mid-times [s].
        y60 : np.ndarray
            Aggregated values per bin.
        n_used : np.ndarray
            Number of samples per bin.
        """
        t = np.asarray(t_sec, dtype=float)
        v = np.asarray(y, dtype=float)

        # Drop NaNs
        good = np.isfinite(t) & np.isfinite(v)
        t, v = t[good], v[good]
        if t.size == 0:
            return np.array([]), np.array([]), np.array([])

        # Auto-detect sample spacing if not provided
        if sample_dt is None and t.size > 1:
            sample_dt = float(np.median(np.diff(np.sort(t))))
        elif sample_dt is None:
            sample_dt = 1.0

        # === Check if already at ~60-s cadence ===
        dt = self._count_interval
        if np.isclose(sample_dt, dt, rtol=0.05, atol=1.0):
            print("⚠️ Input Doppler already at ~60-s cadence, no compression applied.")
            return t, v, np.ones_like(t, dtype=int)

        # Bin alignment
        if t0 is None:
            t0 = np.floor(t.min() / dt) * dt

        # Bin index per sample
        bin_idx = np.floor((t - t0) / dt).astype(int)
        uniq = np.unique(bin_idx)

        # Expected samples per bin (for coverage check)
        exp_per_bin = max(1, int(round(dt / float(sample_dt))))
        need = int(np.ceil(min_coverage * exp_per_bin))

        # Choose aggregator
        agg = agg.lower()
        if agg == "mean":
            aggregator = lambda arr: np.mean(arr) if arr.size else np.nan
        elif agg == "median":
            aggregator = lambda arr: np.median(arr) if arr.size else np.nan
        elif agg == "sum":
            aggregator = lambda arr: np.sum(arr) if arr.size else np.nan
        elif agg == "min":
            aggregator = lambda arr: np.min(arr) if arr.size else np.nan
        elif agg == "max":
            aggregator = lambda arr: np.max(arr) if arr.size else np.nan
        else:
            raise ValueError("agg must be one of: mean, median, sum, min, max")

        t60_mid, y60, n_used = [], [], []
        for b in uniq:
            sel = bin_idx == b
            vb = v[sel]
            nb = vb.size

            bin_start = t0 + dt * b
            bin_end = bin_start + dt
            t_mid = 0.5 * (bin_start + bin_end)

            if nb >= need:
                yb = aggregator(vb)
            else:
                yb = np.nan  # insufficient coverage

            t60_mid.append(t_mid)
            y60.append(yb)
            n_used.append(nb)

        return np.array(t60_mid), np.array(y60), np.array(n_used)
    
    def default_doppler_aux(
        self,
        tstart: EpochArray,
        tend: EpochArray,
        N: int = None,
        ramp_margin: float = None,
        n_ramp: int = None,
        band: str = "X",
    ) -> dict:
        """
        Default ``computed_measurements_dict`` for synthetic two-way Doppler.

        Every per-observation array is ``N`` long, including year/doy/sec, and every
        ramp array is ``n_ramp`` long. See :meth:`Measurement._default_aux_grids` for
        ``ramp_margin`` and ``n_ramp`` and for why the two groups are sized from one
        place.

        The simulated values are this model's radiometric defaults:
        :attr:`count_interval`, :attr:`dsn_bands`, :attr:`ramp_rate`,
        :attr:`ramp_type`, :attr:`transponder_delay`, :attr:`rcv_time_tag_delay`,
        :attr:`ul_station_cal` and :attr:`dl_station_cal`. Set them on the model
        before calling to change the simulated data.

        Parameters
        ----------
        tstart, tend : EpochArray
            Span the simulated data covers. Only the first epoch of each is used.
        N : int, optional
            Number of data points the aux data describes. By default the span is
            filled with as many whole count intervals of :attr:`count_interval`
            as fit, and a value that does not fit the span is reduced to that
            number.
        ramp_margin : float, optional
            Seconds the ramp grid extends beyond the span at each end. Defaults to
            :attr:`ramp_margin`.
        n_ramp : int, optional
            Number of ramp-table entries. Defaults to :attr:`ramp_entries`.
        band : str, optional
            A key of :attr:`dsn_bands`, ``"X"`` or ``"S"`` by default
            (case-insensitive), resolved by :meth:`Measurement._resolve_dsn_band`. It sets the uplink frequency,
            turnaround ratio, exciter scalar and range unit. Defaults to ``"X"``.

        Returns
        -------
        Doppler_aux_data : dict
            Auxiliary values needed for computing measurements: the
            per-observation Doppler and delay entries, each ``N`` long, and the
            ramp-table entries, each ``n_ramp`` long.
        """
        
        # Tc, the Doppler count interval, is the model's count_interval (the DSN
        # standard 60 s by default) and stays fixed. It also has to equal the
        # sample spacing, or the simulated counts would overlap or leave gaps --
        # so the grid is built on a Tc step and N gives way when it does not fit
        # the span, rather than Tc being stretched to accommodate it.
        count_interval = self._count_interval
        t_0 = float(np.atleast_1d(tstart.et)[0])
        t_f = float(np.atleast_1d(tend.et)[0])
        if N is None:
            N = max(int(np.floor((t_f - t_0) / count_interval)), 1)

        obs, ramp = self._default_aux_grids(
            tstart,
            tend,
            N,
            ramp_margin=ramp_margin,
            n_ramp=n_ramp,
            dt_obs=count_interval,
        )
        # year/doy/sec describe the same N observations as every other
        # per-observation array, so every column of the dict has one length.
        N = obs["n"]  # reduced if N count intervals did not fit the span
        (year, day, sec) = (obs["year"], obs["doy"], obs["sec"])
        ramp_day, ramp_sec, ramp_year = ramp["day"], ramp["sec"], ramp["year"]
        n_ramp = len(ramp_sec)
        # Constant-rate ramp referenced to the first entry, so the
        # piecewise-linear interpolant integrate() builds is continuous.
        band_spec = self._resolve_dsn_band(band)
        ramp_f0 = band_spec.uplink_frequency.values
        ramp_rate = self._ramp_rate
        ramp_freq = (ramp_f0 + ramp_rate * (ramp["et"] - ramp["et"][0])).tolist()
        rcv_time_tag_delay = self._rcv_time_tag_delay

        Doppler_aux_data = {
            #
            # Doppler auxiliary data (from 2W_doppler, SFDU_16)
            #
            "Tc": [count_interval for _ in range(N)],
            "M2_num": [band_spec.turnaround_num for _ in range(N)],
            "M2_den": [band_spec.turnaround_den for _ in range(N)],
            "outlier_flag_doppler": [0 for _ in range(N)],
            "year_SFDU16": year,
            "doy_SFDU16": day,
            "sec_SFDU16": sec,
            #
            # Delays auxiliary data (from 2W_doppler, SFDU_16)
            #
            "ul_zheight_corr": [0.0 for _ in range(N)],
            "dl_zheight_corr": [0.0 for _ in range(N)],
            "scft_transpd_delay": self._transponder_delay, # [RU]
            "scft_transpd_delay_s": Measurement.transponder_delay_seconds(
                self._transponder_delay,
                band_spec.ru_reference_frequency.values,
                band_spec.range_unit),
            "transmit_time_tag_delay": [0.0 for _ in range(N)],
            "rcv_time_tag_delay": [rcv_time_tag_delay for _ in range(N)],
            "array_delay": [0.0 for _ in range(N)],
            #
            "ul_freq": [band_spec.uplink_frequency.values for _ in range(N)],       
            "ul_stn_cal": [self._ul_station_cal for _ in range(N)],
            "dl_stn_cal": [self._dl_station_cal for _ in range(N)],
            "exc_scalar_num": [band_spec.exciter_scalar_num for _ in range(N)],
            "exc_scalar_den": [band_spec.exciter_scalar_den for _ in range(N)],
            "RU": [band_spec.range_unit for _ in range(N)],
            #
            # Ramp table auxiliary data (from SFDU_9)
            #
            "ramp_sec": ramp_sec,
            "ramp_day": ramp_day,
            "ramp_freq": ramp_freq,
            "ramp_rate": [ramp_rate for _ in range(n_ramp)],
            "ramp_type": [self._ramp_type for _ in range(n_ramp)],
            # Year of each ramp record, so the table may cross a new year
            "ramp_year": ramp_year,
        }

        Measurement.validate_aux(Doppler_aux_data, label="default_doppler_aux")
        return Doppler_aux_data

    def _write_aux_data(self, dict_DB, dict_meas):
        """
        Add this model's auxiliary data to a file being written.

        Called by :meth:`~scarabaeus.Measurement.write_observed_measurements`
        once it has built the measurement block, to fill in the fields that are
        specific to two-way Doppler. Everything written comes from the model's
        ``computed_measurements_dict``, filled either by ``_read_aux`` from a
        real pass or by
        :meth:`~scarabaeus.DopplerReal.default_doppler_aux` for a simulated
        one.

        Both arguments are modified in place.

        Parameters
        ----------
        dict_DB : dict
            The top-level file dictionary. Gains three blocks, none of which is
            one row per Doppler observation:

            * ``"aux_SFDU_16"`` -- the count interval, the transponder
              turnaround ratio, the outlier flags and the observation time
              tags.
            * ``"aux_SFDU_9"`` -- the uplink ramp table, one row per ramp
              segment, with the year of each record when the aux data has
              ``ramp_year``.
            * ``"2W_sranging"`` -- the uplink frequency, the station
              calibrations and the exciter scalars, with the observation time
              tags. See the note at the code below for why a Doppler file
              carries a block under this name.
        dict_meas : dict
            The measurement block, one value per observation. Gains the zenith
            height corrections, the time-tag delays, the array delay and the
            spacecraft transponder delay.

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
        Unlike sequential ranging, Doppler has its own ``"aux_SFDU_16"`` block,
        so the observation time tags are written there rather than being left
        to the caller. Sequential ranging has nowhere to put them but the
        measurement block, which the caller has already filled, which is why
        the two models differ on this point.

        Each per-observation value describes one row of the measurement block,
        so the ``"aux_SFDU_16"`` and ``"2W_sranging"`` blocks carry that
        block's time tags: :meth:`_read_aux` samples every auxiliary record at
        the measurement epochs by time tag, as it does for a real TRK-2-34
        file whose records each have their own cadence.

        The five fields in the ``"2W_sranging"`` block are normalised through
        ``numpy.asarray`` before being written, because the two producers of an
        auxiliary dict disagree on their type: ``_read_aux`` builds them as
        arrays and
        :meth:`~scarabaeus.DopplerReal.default_doppler_aux` as plain lists.
        """
        # Reported here rather than as 'NoneType is not subscriptable' from
        # the first key read below.
        Measurement.validate_aux(
            self._computed_measurements_dict, label=self.name
        )

        dict_meas["ul_zheight_corr"] = self._computed_measurements_dict["ul_zheight_corr"]
        dict_meas["dl_zheight_corr"] = self._computed_measurements_dict["dl_zheight_corr"]
        dict_meas["transmit_time_tag_delay"] = self._computed_measurements_dict["transmit_time_tag_delay"]
        dict_meas["rcv_time_tag_delay"] = self._computed_measurements_dict["rcv_time_tag_delay"]
        dict_meas["array_delay"] = self._computed_measurements_dict["array_delay"]
        # The transponder delay is a mission constant; a TRK-2-34 measurement
        # record carries it on every row, and _read_aux reads it back from there.
        if "scft_transpd_delay" in self._computed_measurements_dict:
            dict_meas["scft_transpd_delay"] = [
                float(np.squeeze(self._computed_measurements_dict["scft_transpd_delay"]))
            ] * len(dict_meas["sec"])

        # np.asarray(...).tolist(), not .tolist() on the stored value: _read_aux
        # builds these five as ndarrays (``[...] + np.max(...)`` broadcasts),
        # but default_doppler_aux builds them as plain lists, which have no
        # .tolist() -- so writing simulated Doppler from the default aux dict
        # died with AttributeError before reaching the file.
        twowaysrange = {
            key: np.asarray(self._computed_measurements_dict[key]).tolist()
            for key in (
                "ul_freq",
                "ul_stn_cal",
                "dl_stn_cal",
                "exc_scalar_num",
                "exc_scalar_den",
            )
        }
        # Time tags of the rows these values describe, so _read_aux can align
        # the block to the measurement epochs as it does any auxiliary record
        for key in ("year", "doy", "sec"):
            twowaysrange[key] = np.asarray(dict_meas[key]).tolist()

        # Yes, a Doppler file gets a block named "2W_sranging", and that is on
        # purpose: it is what DopplerReal._read_aux reads these five fields back
        # out of. A real TRK-2-34 pass file carries both data types, and the
        # uplink frequency and the station calibrations live in the ranging
        # records, so _read_aux learned to look there (see the "Method (1)"
        # comment beside it) and the writer has to mirror that for a simulated
        # file to be reloadable. Dropping the block makes the Doppler read fail
        # with KeyError: '2W_sranging'.
        #
        # It holds only those five uplink/calibration fields -- no ranging
        # observations -- so it is misnamed rather than wrong. Pointing
        # SequentialRangingReal at it raises KeyError: 'outlier_flag', i.e. the
        # mistake is loud, not silent. Renaming it would mean a key alias on
        # both sides (as _resolve_meas_key does for meas_sim/meas_ideal), since
        # real files will always say "2W_sranging"; deliberately not done here.
        dict_DB["2W_sranging"] = twowaysrange

        aux_SFDU16 = {
            "Tc" : self._computed_measurements_dict["Tc"],
            "M2_num" : self._computed_measurements_dict["M2_num"],
            "M2_den" : self._computed_measurements_dict["M2_den"],
            "outlier_flag" : self._computed_measurements_dict[
                self.AUX_KEYS.outlier_flag
            ],
            # The rows of the measurement block: the per-observation values
            # above describe those, whatever epochs the aux dict was built for
            "year" : np.asarray(dict_meas["year"]).tolist(),
            "doy" : np.asarray(dict_meas["doy"]).tolist(),
            "sec" : np.asarray(dict_meas["sec"]).tolist(),
        }

        dict_DB["aux_SFDU_16"] = aux_SFDU16

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

    def _read_aux(self, trk_data, meas_name, sc_transponder_delay_RU=0.0,
                  band: str = "X"):
        """
        Collect the auxiliary data a two-way Doppler pass needs.

        Called by :meth:`~scarabaeus.Measurement.observed_measurements` on an
        already-loaded tracking file. The returned dict is both handed back as
        ``ObsData.aux_dict`` and installed on the model as its
        ``computed_measurements_dict``, which is what the forward model reads,
        so the two cannot fall out of step.

        The observable cannot be reconstructed from the observations alone: the
        count interval, the transponder turnaround ratio, the station delays
        and the uplink ramp table are all needed to form a computed Doppler
        shift.

        Parameters
        ----------
        trk_data : dict
            A loaded tracking-data file. Each quantity is read from the block
            named by *meas_name* when it is there, and otherwise from the
            auxiliary SFDU record that carries it (see
            :meth:`~scarabaeus.Measurement._read_dsn_aux`): ``"aux_SFDU_16"``
            for the count interval, turnaround ratio and outlier flags,
            ``"aux_SFDU_7"`` or ``"2W_sranging"`` for the uplink frequency,
            station calibrations and exciter scalars, ``"aux_SFDU_2"`` and
            ``"aux_SFDU_3"`` for the zenith height corrections and time-tag
            delays, and ``"aux_SFDU_9"`` for the uplink ramp table.
        meas_name : str
            Key of the Doppler measurement block, normally ``"2W_doppler"``.
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
            The auxiliary data: one entry per Doppler observation for the
            per-observation values, one per segment for the ramp table.

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
        Each SFDU record of a TRK-2-34 file has its own cadence and time tags
        (``"aux_SFDU_2"`` every second, the ranging records at the ranging
        cadence), so its rows do not correspond to the Doppler rows. Every
        value is therefore sampled at the Doppler epochs by time tag, taking
        the entry in force at each (see
        :meth:`~scarabaeus.Measurement.align_aux_to_epochs`), rather than
        taken row by row or as one representative value for the pass.

        The range unit returned under ``"RU"`` is derived from the exciter
        scalars recorded in the file, so it follows the pass's own band rather
        than being assumed.
        """
        aux = self._read_dsn_aux(
            trk_data, meas_name, sc_transponder_delay_RU, band
        )
        for record_field, key in self.AUX_KEYS.record_fields.items():
            aux[key] = Measurement._aux_field(
                trk_data, meas_name, [(None, record_field), ("aux_SFDU_16", key)]
            )
        aux[self.AUX_KEYS.outlier_flag] = Measurement._aux_field(
            trk_data,
            meas_name,
            [("aux_SFDU_16", "outlier_flag"), (None, "outlier_flag")],
            default=0,
        )
        # The values above are aligned to the Doppler epochs, so they carry
        # the Doppler block's time tags
        record = trk_data[meas_name]
        aux["year_SFDU16"] = record["year"]
        aux["doy_SFDU16"] = record["doy"]
        aux["sec_SFDU16"] = record["sec"]
        Measurement.validate_aux(aux, label="_read_aux")
        return aux
