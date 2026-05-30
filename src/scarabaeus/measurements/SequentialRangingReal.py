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
    meas_bias : numpy.ndarray, optional
        Range bias of the ground station. Defaults to ``None``.
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
    scarabaeus.DopplerReal : Companion real Doppler model.
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
        self._measurement_type = "SequentialRangingReal"
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
        """
        EpochArray in seconds of day
        """
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
    # Methods #
    # ---------#
    def _extract_real_range_bias(self) -> ArrayWUnits | None:
        """
        Return range bias in Range Units (RU) for THIS station, as stored in the state, if present.
        """
        if self._reference_state_vector is None:
            return self._meas_bias if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Prefer meas_bias_real_* if available in the allowed dictionary
        if "range_bias_RU" not in state_allowed:
            return None

        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame (even for scalar bias)

            if state_allowed["range_bias_RU"][2].match(component) is None:
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
            Whether to add noise to the computed range measurement. Defaults to False.

        eps_override: dict
            Dictionary of delta's to compute the numeric _partials

        Returns
        -------
        computed_meas : ArrayWFrame
            An ArrayWFrame object with the computed measurement values expressed in RU (adimensional)

        Raises
        ------
        TypeError('The noise model can't be used on a computed model of a real measurement.')
            | Raised when noise is enabled on a computed model of a real measurement type

        Notes
        -----
        This model has been developed following the details in Moyer 2000.
        [1]: "TRK-2-34, DSN Tracking System Data Archival Format", DSN No. 820-013, TRK-2-34, Rev. N

        """
        # 0. Initialization
        if epoch_array.system != "TDB":
            epoch_array.convert_to("TDB")
        if noisy is True:
            TypeError(
                "SequentialRangingReal.py: The noise model can't be used on computed of a real model."
            )

        # 1. Initialization
        delta = self._build_perturbation_vector(
            eps_override
        )  # Delta dict for _partials
        media_corrections = self.media_corrections  # MediaCorrection object

        # Pull current bias + gs delta from state (if provided)
        bias_awu = self._extract_real_range_bias()  # ArrayWUnits or None
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        # 2. Extract aux_dictionary quantities
        aux_dict = self.computed_measurements_dict
        RU = aux_dict[
            "RU"
        ]  # RU conversion factor (pre-frequency multiplication factor)
        M = aux_dict["M"]  # Modulo M [RU]
        M2_num = np.array(
            aux_dict["M2_num"]
        )  # Spacecraft turn-around ration (NUMERATOR)
        M2_den = np.array(
            aux_dict["M2_den"]
        )  # Spacecraft turn-around ration (DENOMINATOR)
        M2 = M2_num / M2_den  # Spacecraft turn-around ration
        # 3. Compute receive time t3 (et, and y,d,s)
        tt_et = epoch_array.times.values
        _, t3_doy_all, t3_sec_all = SpiceManager.et2YDS(tt_et)

        # 4. Compute precise RTLT
        rtlt = []
        t1 = []
        t2 = []
        t3 = epoch_array
        print("")
        print("Generating computed measurements...")
        update, close = Measurement._progress_monitor(
            num_total=len(tt_et), desc="2W Sequential Ranging"
        )
        # Loop through different t3
        for i in range(len(tt_et)):
            tt_et_i = tt_et[i]
            # Pack a dictionary for electronic delays
            electronic_delays_dict = Measurement._generate_electronic_delays_dictionary(
                aux_dict, i
            )
            # Pack a dictionary for Solar Corona delays
            solar_corona_dict = {
                "M2": M2_num[i] / M2_den[i],
                "ul_freq": aux_dict["ul_freq"][i],
            }
            # Generate the rtlt's

            try:
                rtlt_i, _, t2_i, t1_i = Measurement._compute_precise_RTLT(
                    sc_spice_id=str(target.spice_id),
                    gs_spice_id=str(self.instrument.spice_id),
                    tt_et=tt_et_i,
                    electronic_delays_dict=electronic_delays_dict,
                    solar_corona_dict=solar_corona_dict,
                    delta_partials=delta,
                    gs_delta_awf=gs_delta_awf,
                )
                #
            except Exception as e:
                print(f"⚠️ Warning: RTL test computation failed at iteration {i}: {e}")
                # Default fallback values to avoid breaking the loop
                rtlt_i = 0
                t2_i = tt_et_i
                t1_i = tt_et_i
            #
            rtlt.append(rtlt_i)
            t2.append(t2_i)
            t1.append(t1_i)
            update(i)
            #
        close()

        rtlt = np.array(rtlt)
        t1 = np.array(t1)
        t2 = np.array(t2)

        # 5. Define begin and end intervals for integration with ramp tables
        t3_sec_integration = np.array(t3_sec_all)
        t1_sec_integration = t3_sec_integration - rtlt

        # Adjust for doy rollover events
        t3_doy_integration = np.array(t3_doy_all)
        t1_doy_integration = np.copy(t3_doy_integration)
        for ii in range(0, len(t1_sec_integration)):
            #
            if t1_sec_integration[ii] < 0:
                t1_doy_integration[ii] = t1_doy_integration[ii] - 1.0
                t1_sec_integration[ii] = t1_sec_integration[ii] + 86400.0

        # 6. Group data for the ramp table
        ramp_data = [
            aux_dict["ramp_sec"],
            aux_dict["ramp_day"],
            aux_dict["ramp_freq"],
            aux_dict["ramp_rate"],
            aux_dict["ramp_type"],
        ]

        # 7. Generate the computed measurement by integration with the ramp table
        sranging_meas_list = []
        used_tropo_flag = np.zeros(len(rtlt))
        used_iono_flag = np.zeros(len(rtlt))
        for i in range(len(rtlt)):
            # 7.1 Perform the integration between t1 and t3 to generate f_int
            f_int, aux_int = RampTableManager.integrate(
                ramp_data,
                t1_sec_integration[i],
                t3_sec_integration[i],
                t1_doy_integration[i],
                t3_doy_integration[i],
            )
            #
            if bias_awu is not None:
                sranging_meas = (RU[i] * f_int) % M[i] + bias_awu
            else:
                sranging_meas = (RU[i] * f_int) % M[i]

            # 7.2 Add the correction term as per [1], Note 39 # NOTE: electronic delays works better following Moyer instead of [1], including them in the precise_RTLT instead
            # sranging_calib_corr_i = SequentialRangingReal._compute_sranging_correction(electronic_delays_dict, RU[i], aux_dict['ul_freq']) # Method (2), compute the correction term
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
                        tt_et[i], str(target.spice_id), f_t3, gs_delta_awf
                    )  # [m]
                    p_tropo_t1, p_iono_t1 = media_corrections.computed_rtlt_delays(
                        tt_et[i] - rtlt[i], str(target.spice_id), f_t1, gs_delta_awf
                    )  # [m]
                    # NOTE: Sign of iono is negative for doppler, positive for sranging
                    p_total_s = (1 / (constants.c.values)) * (
                        (p_tropo_t3 + p_tropo_t1 + p_iono_t3 + p_iono_t1) * 1e-3
                    )  # Moyer 10-27, [s]
                    # Add delay into measurement
                    p_total_RU = p_total_s * RU[i] * f_t1  # Moyer 13-127
                    sranging_meas_media = (
                        sranging_meas + p_total_RU - sranging_calib_corr_i
                    )
                    sranging_meas_list.append(sranging_meas_media)
                    # Keep track of the media correction used
                    if (p_tropo_t3 != 0.0) or (p_tropo_t1 != 0.0):
                        used_tropo_flag[i] = 1
                    if (p_iono_t3 != 0.0) or (p_iono_t1 != 0.0):
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

        return t1, t2, t3, ArrayWFrame(ArrayWUnits(sranging_meas_list, km / km), J2000)

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
            #
            return self._compute_partials_with_fd(target, epoch_array, frame)
        else:
            print(
                "Warning: computing idealized partial derivatives of measurement function"
            )
            return self._compute_ideal_partials(target, epoch_array, frame)
        #

    #
    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
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
        h_tilde = [
            (relative_state[0] / range_val).values,
            (relative_state[1] / range_val).values,
            (relative_state[2] / range_val).values,
        ]
        return np.array(h_tilde)

    #
    def _compute_h_tilde_vel(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
        with respect to the velocity components.

        Returns:
            np.array: The (3,) vector the partial derivatives of the measurement model by the velocity components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        """
        h_tilde = [
            ArrayWUnits(0, None).values,
            ArrayWUnits(0, None).values,
            ArrayWUnits(0, None).values,
        ]
        return np.array(h_tilde)

    #
    def _compute_ideal_partials(
        self, target: Spacecraft, epoch_array: EpochArray, frame: Frame = J2000
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

            # Apply GS delta location from *state* (if present) by perturbing 3D relative pos ---
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

            state_allowed = StateDefinition._generate_allowed_state_dictionary()

            # Stack h_tilde based on StateVector components
            h_tilde = []
            target_sid = getattr(target, "spice_id", None)

            if self._state_definition is None:
                # nominal case: pos+vel only
                h_tilde.extend(self._compute_h_tilde_pos(relative_state))
                h_tilde.extend(self._compute_h_tilde_vel(relative_state))

            else:
                for component_def in self._state_definition:
                    component = component_def[0]
                    comp_body = component_def[4] if len(component_def) > 4 else None
                    comp_body_sid = getattr(comp_body, "spice_id", None)

                    if state_allowed["position"][2].match(component):
                        if comp_body_sid == target_sid:
                            # Position of the target spacecraft
                            h_tilde.extend(self._compute_h_tilde_pos(relative_state))
                        else:
                            # Position of another body -> no sensitivity
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["velocity"][2].match(component):
                        if comp_body_sid == target_sid:
                            # Velocity of the target spacecraft
                            h_tilde.extend(self._compute_h_tilde_vel(relative_state))
                        else:
                            # Velocity of another body -> no sensitivity
                            h_tilde.extend([0.0] * 3)

                    elif state_allowed["eta_srp"][2].match(component):
                        # eta_srp
                        h_tilde.extend(self._compute_h_tilde_eta_srp())

                    elif state_allowed["gs_delta_location_ECEF"][2].match(component):
                        # NOTE: idealized formula does not account for full RTLT sensitivity;
                        # 1/RU scaling at end of this method is also approximate.
                        # Use partials_finite_diff=True or validate before enabling.
                        raise NotImplementedError(
                            "SequentialRangingReal: gs_delta_location_ECEF partial not verified or finished."
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

                    elif "range_bias_RU" in state_allowed and state_allowed["range_bias_RU"][
                        2
                    ].match(component):
                        # NOTE: 1/RU scaling at end of this method does not correctly
                        # propagate through the ramp-table integration for a range bias.
                        # Use partials_finite_diff=True or validate before enabling.
                        raise NotImplementedError(
                            "SequentialRangingReal: range_bias_RU partial not verified or finished."
                        )
                        # --- unverified implementation (kept for reference) ---
                        # if comp_body is not None and self.instrument.spice_id == comp_body.spice_id:
                        #     h_tilde.extend(self._compute_h_tilde_range_bias())  # dρ/db = 1
                        # else:
                        #     h_tilde.extend([0.0])

                    elif any(
                        state_allowed[dv_key][2].match(component)
                        for dv_key in ["dv_man", "dv_man1", "dv_man2"]
                    ):
                        # dv maneuver (dv_man, dv_man1, dv_man2)
                        h_tilde.extend(self._compute_h_tilde_dv_man())
                    else:  # If the component is not above, we add zeros by default.
                        h_tilde.extend([0.0 for _ in range(component_def[1])])
                    #
                #
            #
            partials_list.append(h_tilde)
            #
        #
        aux_dict = self.computed_measurements_dict
        RU = aux_dict["RU"][0]
        scale_fac = 1.0 / RU * 1000
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
            # default position & velocity perturbations in [km] and [km/s]
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
                state_allowed[key][2].match(comp_name)
                for key in state_allowed
                if any(tag in key for tag in ["dv_man"])
            ):
                H_blocks.append(np.zeros((N, 3)))

            elif state_allowed["range_bias_RU"][2].match(comp_name):
                if self.instrument.spice_id == comp_body.spice_id:
                    H_blocks.append(np.ones((N, 1)))
                else:
                    H_blocks.append(np.zeros((N, 1)))

            elif state_allowed["gs_delta_location"][2].match(comp_name):
                # NOTE: implementation below contains a variable-name bug — the regex
                # match result shadows the 'frame' parameter, breaking the
                # _batched_component_partials call. Fix before enabling.
                raise NotImplementedError(
                    "SequentialRangingReal: gs_delta_location partial (FD path) not verified or finished."
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

        return H

    #
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
