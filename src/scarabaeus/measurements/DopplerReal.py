# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from fileinput import close
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
    StateArray,
    MediaCorrections,
    StateDefinition,
)
from typing import Optional

import scarabaeus.utils.NumpyWrapper as np
import copy as copy

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
    meas_bias : numpy.ndarray, optional
        Range-rate bias of the ground station. Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : list, optional
        Sequence definition list. Defaults to ``None``.
    epoch_sec : EpochArray, optional
        Epoch expressed in seconds of day. Defaults to ``None``.
    computed_measurements_dict : dict, optional
        Auxiliary dictionary for computed-measurement generation.
        Defaults to ``None``.
    partials_finite_diff : bool, optional
        When ``True``, _partials are computed via finite differences.
        Defaults to ``False``.
    eps_dict : dict, optional
        Perturbation magnitudes for finite-difference _partials.
        Defaults to ``None``.
    media_corrections : MediaCorrections, optional
        Media-correction object (troposphere / ionosphere). Defaults to ``None``.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.
    scarabaeus.SequentialRangingReal : Companion real ranging model.
    scarabaeus.MediaCorrections : Tropospheric and ionospheric path-delay model.
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
        partials_finite_diff: bool = False,
        eps_dict: dict = None,
        media_corrections: Optional[MediaCorrections] = None,
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
        self._partials_finite_diff = partials_finite_diff
        self.compute_partials = self._compute_partials
        self.eps_dict = eps_dict
        self.computed_measurements = self.computed_measurements
        self._computed_measurements_dict = computed_measurements_dict
        self._measurement_type = "DopplerReal"
        self._reference_state_vector: StateArray | None = None

        # Media correction
        self._media_corrections: Optional[MediaCorrections] = None
        if media_corrections is not None:
            self.media_corrections = media_corrections

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def sigma(self) -> ArrayWUnits:
        """
        The standard deviation of the measurement model.
        """
        return self._sigma

    @property
    def epoch_sec(self) -> EpochArray:
        """Observation epochs expressed as seconds of day."""
        return self._epoch_sec

    @property
    def computed_measurements_dict(self) -> dict:
        """
        Auxiliary dictionary needed for the generation of the computed measurements
        """
        return self._computed_measurements_dict

    @property
    def media_corrections(self) -> Optional[MediaCorrections]:
        """Tropospheric and ionospheric path-delay model, or ``None`` if not set."""
        return self._media_corrections

    @media_corrections.setter
    def media_corrections(self, value: MediaCorrections) -> None:
        if not isinstance(value, MediaCorrections):
            raise TypeError("media_corrections must be a MediaCorrections object")
        self._media_corrections = value

    # endregion => Properties #
    # -------------------------#

    # ---------#
    # Methods  #
    # ---------#
    def _extract_real_range_rate_bias(self) -> ArrayWUnits | None:
        """
        Return range rate bias in Hz for THIS station, as stored in the state, if present.
        """
        if self._reference_state_vector is None:
            return self._meas_bias if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Prefer meas_bias_real_* if available in the allowed dictionary
        if "rangerate_bias_Hz" not in state_allowed:
            return None

        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame (even for scalar bias)

            if state_allowed["rangerate_bias_Hz"][2].match(component) is None:
                continue

            body_sid = getattr(body, "spice_id", None)
            if body_sid == self.instrument.spice_id:
                return value.quantity

        return None

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
            Whether to add noise to the computed range measurement. Defaults to False.

        eps_override: dict
            Dictionary of delta's to compute the numeric _partials

        Returns
        -------
        computed_meas : ArrayWFrame
            An ArrayWFrame object with the computed measurement values expressed in Hz

        Raises
        ------
        TypeError('The noise model can't be used on a computed model of a real measurement.')
            | Raised when noise is enabled on a computed model of a real measurement type

        Notes
        -----
        This model has been developed following the details in Moyer 2000.
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """

        # 0. Parameters check
        if epoch_array.system != "TDB":
            epoch_array.convert_to("TDB")
        if noisy is True:
            TypeError(
                "DopplerReal.py: The noise model can't be used on computed of a real model."
            )

        # 1. Initialization
        delta = self._build_perturbation_vector(
            eps_override
        )  # Delta dict for _partials
        media_corrections = self.media_corrections  # MediaCorrection object

        # Pull current bias + gs delta from state (if provided)
        bias_awu = self._extract_real_range_rate_bias()  # ArrayWUnits or None
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        # 2. Extract aux_dictionary quantities of interest
        aux_dict = self.computed_measurements_dict
        RU = aux_dict[
            "RU"
        ]  # RU conversion factor (pre-frequency multiplication factor)
        M2_num = np.array(
            aux_dict["M2_num"]
        )  # Spacecraft turn-around ration (NUMERATOR)
        M2_den = np.array(
            aux_dict["M2_den"]
        )  # Spacecraft turn-around ration (DENOMINATOR)
        M2 = M2_num / M2_den  # Spacecraft turn-around ration
        Tc = aux_dict["Tc"]
        #

        t3s_et_all = epoch_array.times.values - np.array(Tc) / 2
        t3e_et_all = epoch_array.times.values + np.array(Tc) / 2
        t3_et_all = epoch_array.times.values

        t3s_year_all, t3s_doy_all, t3s_sec_all = SpiceManager.et2YDS(t3s_et_all)
        t3e_year_all, t3e_doy_all, t3e_sec_all = SpiceManager.et2YDS(t3e_et_all)
        _, _, t3_sec_all = SpiceManager.et2YDS(t3_et_all)

        # 4. Compute precise RTLT
        rtlt_s = []
        rtlt_e = []
        t1 = []
        t2 = []
        t3 = epoch_array
        print("")
        print("Generating 2W Doppler computed measurements...")
        update, close = Measurement._progress_monitor(len(t3_et_all), desc="2W Doppler")
        # Loop through different t3
        for i in range(len(t3_et_all)):
            # Pack a dictionary for electronic delays
            electronic_delays_dict = Measurement._generate_electronic_delays_dictionary(
                aux_dict, 0
            )
            # Get tt at start and end of interval count
            t3s_i = (
                t3s_et_all[i]
                - electronic_delays_dict["rcv_tt_dly"]
                + electronic_delays_dict["trn_tt_dly"]
            )
            t3e_i = (
                t3e_et_all[i]
                - electronic_delays_dict["rcv_tt_dly"]
                + electronic_delays_dict["trn_tt_dly"]
            )

            # Pack a dictionary for Solar Corona delays
            solar_corona_dict = {
                "M2": M2_num[i] / M2_den[i],
                "ul_freq": aux_dict["ul_freq"][i],
            }

            # Generate the rtlt's

            try:
                rtlt_s_i, _, t2_s_i, t1_s_i = Measurement._compute_precise_RTLT_light(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=t3s_i,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta,
                )

                # Update perturbation so second time perturbation is correlated with initial one
                if eps_override is not None:
                    if "dvx" in eps_override.keys():
                        eps_override["dx"] = eps_override["dvx"] * ArrayWUnits(
                            aux_dict["Tc"][i], sec
                        )
                    if "dvy" in eps_override.keys():
                        eps_override["dy"] = eps_override["dvy"] * ArrayWUnits(
                            aux_dict["Tc"][i], sec
                        )
                    if "dvz" in eps_override.keys():
                        eps_override["dz"] = eps_override["dvz"] * ArrayWUnits(
                            aux_dict["Tc"][i], sec
                        )
                delta2 = self._build_perturbation_vector(
                    eps_override
                )  # Delta dict for _partials

                rtlt_e_i, _, t2_e_i, t1_e_i = Measurement._compute_precise_RTLT_light(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=t3e_i,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta2,
                )
            #
            except Exception as e:
                print(f"⚠️ Warning: RTL computation failed at iteration {i}: {e}")
                rtlt_s_i = 0
                rtlt_e_i = 0
                t2_e_i = t3e_i
                t2_s_i = t3e_i
                t1_e_i = t3e_i
                t1_s_i = t3e_i
            rtlt_s.append(rtlt_s_i)
            rtlt_e.append(rtlt_e_i)
            t1.append((t1_e_i + t1_s_i) / 2.0)
            t2.append((t2_e_i + t2_s_i) / 2.0)
            update(i)
        close()

        rtlt_s = np.array(rtlt_s)
        rtlt_e = np.array(rtlt_e)
        t1 = np.array(t1)
        t2 = np.array(t2)

        # 5. Define begin and end intervals for integration with ramp tables
        t3s_sec_integration = np.array(t3s_sec_all)
        t3e_sec_integration = np.array(t3e_sec_all)
        t1s_sec_integration = t3s_sec_integration - rtlt_s
        t1e_sec_integration = t3e_sec_integration - rtlt_e

        # Adjust for doy rollover events
        t1e_doy_integration = np.array(t3e_doy_all)
        t1s_doy_integration = t1e_doy_integration
        for ii in range(0, len(t1s_sec_integration)):
            if t1s_sec_integration[ii] < 0:
                t1s_doy_integration[ii] = t1s_doy_integration[ii] - 1
                t1s_sec_integration[ii] = t1s_sec_integration[ii] + 86400
            if t1e_sec_integration[ii] < 0:
                t1e_doy_integration[ii] = t1e_doy_integration[ii] - 1
                t1e_sec_integration[ii] = t1e_sec_integration[ii] + 86400

        # 6. Group data for the ramp table
        ramp_data = [
            aux_dict["ramp_sec"],
            aux_dict["ramp_day"],
            aux_dict["ramp_freq"],
            aux_dict["ramp_rate"],
            aux_dict["ramp_type"],
        ]

        # 7. Generate the computed measurement by integration with the ramp table
        doppler_meas_list = []
        used_tropo_flag = np.zeros(len(rtlt_s))
        used_iono_flag = np.zeros(len(rtlt_s))
        for i in range(len(rtlt_s)):
            # 7.1 Perform the integration between t1 and t3 to generate f_int
            f_int, aux_int = RampTableManager.integrate(
                ramp_data,
                t1s_sec_integration[i],
                t1e_sec_integration[i],
                t1s_doy_integration[i],
                t1e_doy_integration[i],
            )
            if bias_awu is not None:
                doppler_meas = -(M2[i] / Tc[i]) * f_int + bias_awu
            else:
                doppler_meas = -(M2[i] / Tc[i]) * f_int
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
                        t3s_et_all[i], str(target.spice_id), f_t3_s, gs_delta_awf
                    )  # [m]
                    p_tropo_t1s, p_iono_t1s = media_corrections.computed_rtlt_delays(
                        t3s_et_all[i] - rtlt_s[i],
                        str(target.spice_id),
                        f_t1_s,
                        gs_delta_awf,
                    )  # [m]
                    p_tropo_t3e, p_iono_t3e = media_corrections.computed_rtlt_delays(
                        t3e_et_all[i], str(target.spice_id), f_t3_e, gs_delta_awf
                    )  # [m]
                    p_tropo_t1e, p_iono_t1e = media_corrections.computed_rtlt_delays(
                        t3e_et_all[i] - rtlt_e[i],
                        str(target.spice_id),
                        f_t1_e,
                        gs_delta_awf,
                    )  # [m]
                    # NOTE: Sign of iono is negative for doppler, positive for sranging
                    p_total_s = (1 / (constants.c.values)) * (
                        (p_tropo_t3s - p_iono_t3s + p_tropo_t1s - p_iono_t1s) * 1e-3
                    )  # Moyer 10-29, [s]
                    p_total_e = (1 / (constants.c.values)) * (
                        (p_tropo_t3e - p_iono_t3e + p_tropo_t1e - p_iono_t1e) * 1e-3
                    )  # Moyer 10-28, [s]
                    # Add delay into measurement
                    p_total_Hz = (M2[i] / Tc[i]) * (
                        f_t1_e * p_total_e - f_t1_s * p_total_s
                    )  # Moyer 13-58
                    doppler_meas_media = doppler_meas + p_total_Hz
                    doppler_meas_list.append(doppler_meas_media)
                    # Keep track of the media correction used
                    if (
                        (p_tropo_t3s != 0.0)
                        or (p_tropo_t1s != 0.0)
                        or (p_tropo_t3e != 0.0)
                        or (p_tropo_t1e != 0.0)
                    ):
                        used_tropo_flag[i] = 1
                    if (
                        (p_iono_t3s != 0.0)
                        or (p_iono_t1s != 0.0)
                        or (p_iono_t3e != 0.0)
                        or (p_iono_t1e != 0.0)
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

        return t1, t2, t3, ArrayWFrame(ArrayWUnits(doppler_meas_list, 1 / sec), J2000)

    def _compute_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> list[np.ndarray]:
        """
        Dispatches computation of Doppler measurement _partials depending on chosen method.

        Raises
        ------
        TypeError if a non-finite differencing method is selected but not implemented.
        """

        if self._sequence_definition is not None:
            raise NotImplementedError(
                "Sequence-based state definitions are not implemented."
            )

        if self._partials_finite_diff:
            return self._compute_partials_with_fd(target, epoch_array, frame)
        else:
            print(
                "Warning: computing idealized partial derivatives of measurement function"
            )
            return self._compute_ideal_partials(target, epoch_array, frame)

    #
    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the position components.

        Parameters:
            relative_state (ArrayWUnits) : relative state between self and target

        Returns:
            np.array: The (3,) vector the partial derivatives of the measurement model by the position components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        """
        range_val = relative_state[0:3].norm()
        range_rate = (relative_state[0:3] @ relative_state[3:6]) / range_val
        h_tilde = [
            (
                (relative_state[3] / range_val)
                - (relative_state[0] * range_rate) / (range_val * range_val)
            ).values,
            (
                (relative_state[4] / range_val)
                - (relative_state[1] * range_rate) / (range_val * range_val)
            ).values,
            (
                (relative_state[5] / range_val)
                - (relative_state[2] * range_rate) / (range_val * range_val)
            ).values,
        ]
        return np.array(h_tilde)

    # velocity
    def _compute_h_tilde_vel(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the velocity components.

        Parameters:
            relative_state (ArrayWUnits) : relative state between self and target

        Returns:
            np.array: The (3,) vector the partial derivatives of the measurement model by the velocity components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        """
        range_val = relative_state[0:3].norm()
        h_tilde = [
            (relative_state[0] / range_val).values,
            (relative_state[1] / range_val).values,
            (relative_state[2] / range_val).values,
        ]
        return np.array(h_tilde)

    #

    def _compute_ideal_partials(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> np.ndarray:
        """
        This method groups toghether the different components of measurement
        _partials in the global H-tilde. It returns the H-tilde array for the modelled measurement.

        Parameters:
            target (Body, Spacecraft) : target spacecraft as scb Spacecraft/Body object
            epoch (Epoch) : epochs as scb Epoch object
            frame (Frame) : reference frame as scb Frame object

        Returns:
            list: The H-tilde array with all measurements _partials from this model by component
        """

        partials_list = []

        for i, epoch in enumerate(epoch_array):
            #
            # Get relative state
            relative_state = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.instrument.name,
            )

            # Apply GS delta location from *state* (if present) by perturbing 3D relative pos
            gs_delta_awf = (
                self._extract_gs_delta_location()
            )  # returns ArrayWFrame or None
            if gs_delta_awf is not None:
                A = SpiceManager.get_xfrm(
                    frame_from=gs_delta_awf.frame.name,
                    frame_to=frame.name,
                    epoch=epoch.times.values,
                )
                delta_in_frame_vals = (
                    A
                    @ gs_delta_awf.quantity.values.reshape(
                        3,
                    )
                ).reshape(
                    3,
                )
                delta_in_frame = ArrayWUnits(
                    delta_in_frame_vals, gs_delta_awf.quantity.units
                )

                # Update relative position in-place (relative_state is ArrayWUnits)
                # Convention: r_rel = r_trg - r_obs, so observer correction enters with minus sign
                relative_state = ArrayWUnits(
                    np.hstack(
                        [
                            relative_state[0:3].values - delta_in_frame.values,
                            relative_state[3:6].values,
                        ]
                    ),
                    relative_state.units,
                )

            # Generate the dictionary of allowed StateVector component from the StateArray class
            state_allowed = StateDefinition._generate_allowed_state_dictionary()

            # Stack h_tilde based on StateVector components
            h_tilde = []
            target_sid = getattr(target, "spice_id", None)

            if (
                self._state_definition is None
            ):  # Assumes that in the nominal case only position and velocity are StateVector components
                h_tilde.extend(self._compute_h_tilde_pos(relative_state))
                h_tilde.extend(self._compute_h_tilde_vel(relative_state))
            else:  # When extra parameters other than position and velocity are part of the StateVector
                for component_def in self._state_definition:
                    component = component_def[0]
                    comp_body = component_def[4] if len(component_def) > 4 else None
                    comp_body_sid = getattr(comp_body, "spice_id", None)

                    if state_allowed["position"][2].match(component):
                        if comp_body_sid == target_sid:
                            # Position of the target spacecraft
                            h_tilde.extend(self._compute_h_tilde_pos(relative_state))
                        else:
                            # Position of another body → no sensitivity
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["velocity"][2].match(component):
                        if comp_body_sid == target_sid:
                            # Velocity of the target spacecraft
                            h_tilde.extend(self._compute_h_tilde_vel(relative_state))
                        else:
                            # Velocity of another body → no sensitivity
                            h_tilde.extend([0.0] * 3)
                    elif state_allowed["eta_srp"][2].match(component):
                        # eta_srp
                        h_tilde.extend(self._compute_h_tilde_eta_srp())
                    elif state_allowed["gs_delta_location_ECEF"][2].match(component):
                        # NOTE: idealized formula does not account for full RTLT sensitivity;
                        # scaling by ul_freq/c at end of this method is also approximate.
                        # Use partials_finite_diff=True or validate before enabling.
                        raise NotImplementedError(
                            "DopplerReal: gs_delta_location_ECEF partial not verified or finished."
                        )
                        # --- unverified implementation (kept for reference) ---
                        # if self.instrument.spice_id == comp_body.spice_id:
                        #     A_ei = SpiceManager.get_xfrm(
                        #         frame_from="IAU_EARTH", frame_to="J2000", epoch=epoch.times.values,
                        #     )
                        #     h_tilde_frame_from = self._compute_h_tilde_gs_delta_location(relative_state)
                        #     h_tilde_frame_to = h_tilde_frame_from @ A_ei
                        #     h_tilde.extend(h_tilde_frame_to)
                        # else:
                        #     h_tilde.extend([0.0, 0.0, 0.0])
                    elif "rangerate_bias_Hz" in state_allowed and state_allowed[
                        "rangerate_bias_Hz"
                    ][2].match(component):
                        # NOTE: ul_freq/c scaling applied at end of this method does not
                        # correctly propagate through the ramp-table integration for a
                        # frequency bias. Use partials_finite_diff=True or validate before enabling.
                        raise NotImplementedError(
                            "DopplerReal: rangerate_bias_Hz partial not verified or finished."
                        )
                        # --- unverified implementation (kept for reference) ---
                        # if self.instrument.spice_id == comp_body.spice_id:
                        #     h_tilde.extend(self._compute_h_tilde_range_bias())
                        # else:
                        #     h_tilde.extend([0.0])
                    elif state_allowed["dv_man"][2].match(component):
                        # dv maneuver
                        h_tilde.extend(self._compute_h_tilde_dv_man(relative_state))
                    else:  # If the component is not above, we add zeros by default.
                        h_tilde.extend([0.0 for _ in range(component_def[1])])
                    #
                #
            #
            h_tilde[0:3] = np.zeros(3)
            partials_list.append(h_tilde)
            #
        #
        aux_dict = self.computed_measurements_dict
        ul_freq = aux_dict["ul_freq"][0]
        c = constants.c.values
        scale_fac = ul_freq / c
        #
        return [
            scale_fac * np.array(partials_list[ii]) for ii in range(len(partials_list))
        ]
        #

    #

    def _compute_partials_with_fd(
        self,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame = J2000,
    ) -> np.ndarray:
        """
        Compute Doppler _partials w.r.t. state vector using batched finite differencing.

        Returns a list of (1xn) Jacobians per epoch.
        """

        # 1. Compute adaptive FD steps from relative state (if not provided)
        if self.eps_dict is None:
            eps_dict = {
                "dx": ArrayWUnits(1e1, km),
                "dy": ArrayWUnits(1e1, km),
                "dz": ArrayWUnits(1e1, km),
                "dvx": ArrayWUnits(1e-2, km / sec),
                "dvy": ArrayWUnits(1e-2, km / sec),
                "dvz": ArrayWUnits(1e-2, km / sec),
            }

        # 2. Nominal measurement
        _, _, _, meas_nom_out = self.computed_measurements(
            target=target,
            epoch_array=epoch_array,
            frame=frame,
            noisy=False,
            eps_override=None,
        )
        meas_nom = meas_nom_out.quantity.values

        N = epoch_array.size
        H_blocks = []

        # 3. If no state_definition, default to pos + vel
        if self._state_definition is None:
            H_blocks.append(
                self._batched_component_partials(
                    ["dx", "dy", "dz"], eps_dict, target, epoch_array, frame, meas_nom
                )
            )
            H_blocks.append(
                self._batched_component_partials(
                    ["dvx", "dvy", "dvz"],
                    eps_dict,
                    target,
                    epoch_array,
                    frame,
                    meas_nom,
                )
            )
            return [np.hstack(H_blocks)[i, :] for i in range(N)]

        # 4. State-definition driven parsing
        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        for comp_def in self._state_definition:
            comp_name = comp_def[0]
            comp_dim = comp_def[1]
            comp_body = comp_def[4] if len(comp_def) > 4 else None

            if state_allowed["position"][2].match(comp_name):
                H_blocks.append(
                    self._batched_component_partials(
                        ["dx", "dy", "dz"],
                        eps_dict,
                        target,
                        epoch_array,
                        frame,
                        meas_nom,
                    )
                )

            elif state_allowed["velocity"][2].match(comp_name):
                H_blocks.append(
                    self._batched_component_partials(
                        ["dvx", "dvy", "dvz"],
                        eps_dict,
                        target,
                        epoch_array,
                        frame,
                        meas_nom,
                    )
                )

            elif state_allowed["eta_srp"][2].match(comp_name):
                H_blocks.append(np.zeros((N, 1)))

            elif any(
                state_allowed[k][2].match(comp_name)
                for k in ["dv_man", "dv_man1", "dv_man2"]
            ):
                H_blocks.append(np.zeros((N, 3)))

            elif state_allowed["rangerate_bias_Hz"][2].match(comp_name):
                if self.instrument.spice_id == comp_body.spice_id:
                    H_blocks.append(np.ones((N, 1)))
                else:
                    H_blocks.append(np.zeros((N, 1)))

            elif state_allowed["gs_delta_location"][2].match(comp_name):
                # NOTE: implementation below contains a variable-name bug — the regex
                # match result shadows the 'frame' parameter, breaking the
                # _batched_component_partials call. Fix before enabling.
                raise NotImplementedError(
                    "DopplerReal: gs_delta_location partial (FD path) not verified or finished."
                )
                # --- unverified implementation (kept for reference; contains frame-shadow bug) ---
                # _, _frame_str, _ = state_allowed["gs_delta_location"][2].match(comp_name).groups()
                # if self.instrument.spice_id == comp_body.spice_id and _frame_str == "ECEF":
                #     h_tilde_to = np.zeros((N, 3))
                #     h_tilde_from = -self._batched_component_partials(
                #         ["dx", "dy", "dz"], eps_dict, target, epoch_array, frame, meas_nom,
                #     )  # shape (N,3)
                #     for i, et in enumerate(epoch_array):
                #         A_ei = SpiceManager.get_xfrm(
                #             frame_from="IAU_EARTH", frame_to="J2000", epoch=et.times.values,
                #         )
                #         h_tilde_to[i] = h_tilde_from[i] @ A_ei
                #     H_blocks.append(h_tilde_to)
                # else:
                #     H_blocks.append(np.zeros((N, 3)))

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
        Compute batched finite-difference _partials for selected components.

        Returns
        -------
        np.ndarray : shape (N, len(components))
        """
        N = epoch_array.size
        H = np.zeros((N, len(components)))

        # desc_str = f"FD Partials: [{', '.join(components)}]"
        for j, comp in enumerate(components):
            _, _, _, meas_pert_out = self.computed_measurements(
                target=target,
                epoch_array=epoch_array,
                frame=frame,
                noisy=False,
                eps_override={comp: eps_dict[comp]},
            )
            meas_pert = meas_pert_out.quantity.values
            H[:, j] = (meas_pert - meas_nom) / eps_dict[comp].values
        #
        return H

    '''
    def _batched_component_partials_cd(
        self,
        components: list[str],
        eps_dict: dict,
        target: Spacecraft,
        epoch_array: EpochArray,
        frame: Frame,
        meas_nom: np.ndarray,
    ) -> np.ndarray:
        """
        Compute batched central finite-difference _partials for selected components.

        Returns
        -------
        np.ndarray : shape (N, len(components))
        """
        N = epoch_array.size
        H = np.zeros((N, len(components)))

        for j, comp in enumerate(components):
            eps = eps_dict[comp]

            # Forward perturbation
            _, _, _, meas_plus_out= self.computed_measurements(
                target=target,
                epoch_array=epoch_array,
                frame=frame,
                noisy=False,
                eps_override={comp: eps},
            )
            meas_plus = meas_plus_out.quantity.values

            # Backward perturbation
            _, _, _, meas_minus_out = self.computed_measurements(
                target=target,
                epoch_array=epoch_array,
                frame=frame,
                noisy=False,
                eps_override={comp: -eps},
            )
            meas_minus = meas_minus_out.quantity.values

            # Central difference
            H[:, j] = (meas_plus - meas_minus) / (2 * eps.values)

        return H
    '''

    @staticmethod
    def compress_doppler_to_60s(
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
        if np.isclose(sample_dt, 60.0, rtol=0.05, atol=1.0):
            print("⚠️ Input Doppler already at ~60-s cadence, no compression applied.")
            return t, v, np.ones_like(t, dtype=int)

        # Bin alignment
        if t0 is None:
            t0 = np.floor(t.min() / 60.0) * 60.0

        # Bin index per sample
        bin_idx = np.floor((t - t0) / 60.0).astype(int)
        uniq = np.unique(bin_idx)

        # Expected samples per bin (for coverage check)
        exp_per_bin = max(1, int(round(60.0 / float(sample_dt))))
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

            bin_start = t0 + 60.0 * b
            bin_end = bin_start + 60.0
            t_mid = 0.5 * (bin_start + bin_end)

            if nb >= need:
                yb = aggregator(vb)
            else:
                yb = np.nan  # insufficient coverage

            t60_mid.append(t_mid)
            y60.append(yb)
            n_used.append(nb)

        return np.array(t60_mid), np.array(y60), np.array(n_used)
