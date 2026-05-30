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
    StateArray,
    Instrument,
    GroundStation,
    Spacecraft,
    SpiceManager,
    Noise,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
km, sec = Units.get_units(["km", "sec"])
spc_pos_units = km
spc_time_units = sec

J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition   #
# --------------------#
class DopplerIdeal(Measurement):
    """Generates ideal Doppler observables between an observer and a target.

    This ideal model assumes no light-time, no media effects, no antenna offset,
    and no spacecraft spin.

    The two-way Doppler observable is modelled as a frequency shift proportional
    to the line-of-sight range rate:

    .. math::

        f_D = -\\frac{f_0}{c}\\,\\dot{\\rho}

    where :math:`f_0` is the transmit frequency, :math:`c` is the speed of
    light, and :math:`\\dot{\\rho}` is the two-way range rate.

    **Three-way Doppler**

    When ``ground_station_2`` is provided the observable becomes

    .. math::

        f_D = -\\frac{f_0\\,TR}{c}\\,(\\dot{\\rho}_1 + \\dot{\\rho}_2)

    where :math:`\\dot{\\rho}_1` and :math:`\\dot{\\rho}_2` are the range
    rates to the transmitting and receiving ground stations respectively.
    Position and velocity partials are implemented for both stations.
    Partials w.r.t. ``gs_delta_location`` or ``rangerate_bias_Hz`` in the
    three-way case raise :exc:`NotImplementedError` until validated.

    Parameters
    ----------
    name : str
        The name of the measurement model.
    instrument : Instrument
        An "antenna" instrument is used for radiometric measurement models.
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : list, optional
        Sequence definition list. Defaults to ``None``.
    antenna_name : str, optional
        Name of the antenna on which the measurement model works. Used to
        read antenna characteristics. Defaults to ``None``.
    doppler_transmit_frequency : ArrayWUnits, optional
        Transmit frequency for Doppler. Defaults to ``None``.
    ground_station_2 : GroundStation, optional
        Receiving ground station for three-way Doppler. Defaults to
        ``None`` (two-way mode).

    Raises
    ------
    NotImplementedError
        If ``gs_delta_location`` or ``rangerate_bias_Hz`` appear in the
        state definition (two-way or three-way).
    RuntimeError
        If the units extracted from the measurements are not consistent.
    RuntimeError
        If neither an EpochArray nor start/end Epochs are provided.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        sigma: ArrayWUnits = None,
        state_definition: list = None,
        sequence_definition=None,
        meas_bias=None,
        antenna_name: str = None,
        doppler_transmit_frequency: ArrayWUnits = None,
        ground_station_2: GroundStation = None,
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument
        self._sigma = sigma
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._antenna_name = antenna_name
        self._meas_bias = meas_bias
        self._doppler_transmit_frequency = doppler_transmit_frequency
        self._ground_station_2 = ground_station_2
        self._measurement_type = "DopplerIdeal"
        self._reference_state_vector: StateArray | None = None

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def sigma(self) -> ArrayWUnits:
        """
        The standard deviation of the measurement model.
        """
        return self._sigma

    @property
    def state_definition(self) -> dict:
        """
        The state vector definition.
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> dict:
        """
        The sequence definition.
        """
        return self._sequence_definition

    @property
    def antenna_name(self) -> str:
        """
        The name of the antenna.
        """
        return self._antenna_name

    @property
    def doppler_transmit_frequency(self):
        """
        The Doppler transmission frequency
        """
        return self._doppler_transmit_frequency

    @property
    def ground_station_2(self) -> GroundStation:
        """
        The second ground station.
        """
        return self._ground_station_2

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _extract_doppler_bias(self) -> ArrayWUnits | None:
        """
        Return range bias for THIS station, as stored in the state, if present.
        """
        if self._reference_state_vector is None:
            return self._meas_bias.quantity if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Prefer meas_bias_ideal_* if available in the allowed dictionary
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
                return value

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
    ) -> ArrayWFrame:
        """
        Computs the doppler measurement between target.name and self.instrument.name (2-way doppler). If a
        receiver station is specified as an instrument (i.e. a ground station), 3-way doppler is computed instead.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch_array : EpochArray, optional
            An array of epochs (times) at which the range rate measurements should be computed. If
            provided, overrides ``epoch_start``, ``epoch_end``, and ``tstep``.

        epoch_start : EpochArray, optional
            The starting epoch for the range ratemeasurement computations. Required if ``epoch_array``
            is not provided.

        epoch_end : EpochArray, optional
            The ending epoch for the range ratemeasurement computations. Required if ``epoch_array``
            is not provided.

        tstep : float
            The time step, in seconds, between consecutive range ratemeasurements if ``epoch_array``
            is not provided. Defaults to ``1`` second.

        frame : Frame
            The reference frame in which the range rate computation is performed.

        noisy : bool
            Indicates if noise is added to the computed range rate measurement. Defaults to ``False``.

        Returns
        -------
        t1, t2, t3 : list[EpochArray]
            Epoch arrays (transmit / bounce / receive); identical for the
            ideal model (no light-time correction).
        meas_AWF : ArrayWFrame
            Computed Doppler measurements (Hz) packed as an
            :class:`~scarabaeus.ArrayWFrame`.
        """
        ## Input check
        if epoch_array is None:
            if epoch_start is None or epoch_end is None:
                raise RuntimeError(
                    "Please provide an EpochArray or provide start and end Epochs"
                )
            else:
                _units = epoch_start.times.units
                epoch_array = [
                    EpochArray(ArrayWUnits(_t, _units), timeFrame=epoch_start.system)
                    for _t in np.arange(
                        epoch_start.times.values, epoch_end.times.values + tstep, tstep
                    )
                ]

        ## Unpack doppler aux data
        antenna_name = self.antenna_name
        transmit_frequency = self.doppler_transmit_frequency
        receive_station = self.ground_station_2

        # Pull current bias + gs delta from state (if provided)
        bias_awu = self._extract_doppler_bias()  # ArrayWUnits or None
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        ## Get Doppler constants
        # Get turn-around-ratio from target
        instList = target.instrument_list
        for inst in instList:
            if inst.name == antenna_name:
                TR = inst.turn_ratio
        # Compute doppler constant
        DopplerConstant = -transmit_frequency * TR / constants.c

        ## Append measurements in a list at multiple epochs
        meas_list = []
        # Get relative states
        for epoch in epoch_array:
            relative_state = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.instrument.name,
            )

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

            # Get range measurement as AWU
            range1 = relative_state[0:3].norm()
            # Get range rate measurement as AWU
            range_rate1 = (relative_state[0:3] @ relative_state[3:6]) / range1

            # Differentiate between two-way and three-way doppler
            if receive_station is None:  # Two-Way doppler
                doppler_meas = 2 * DopplerConstant * range_rate1

            else:  # Three-way doppler
                # NOTE: gs_delta_location for receive_station (GS2) is not applied here.
                # If GS2 location is estimated, computed_measurements will not reflect it.
                relative_state2 = SpiceManager.get_state(
                    trgt_bdy=target.name,
                    epoch_time=epoch.times.values,
                    reference_frame=frame.name,
                    obsvr_bdy=receive_station.name,
                )
                range2 = relative_state2[0:3].norm()
                range_rate2 = (relative_state2[0:3] @ relative_state2[3:6]) / range2
                doppler_meas = DopplerConstant * (range_rate1 + range_rate2)

            # Apply bias
            if bias_awu is not None:
                doppler_meas = doppler_meas + bias_awu

            # Add noise to ideal doppler
            if noisy:
                doppler_meas += Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma.values,
                    units=self.sigma.units,
                )
            meas_list.append(doppler_meas)  # list of AWU measurements

        ## Pack output measurement as an AWF
        # Extract values and units
        values_array = np.array([item.values for item in meas_list])
        units_list = [item.units for item in meas_list]
        units_names = [item.units.name for item in meas_list]
        if len(np.unique(units_names)) != 1:
            raise RuntimeError("multiple units are extracted from the measurements")

        # Put the measurement toghether as an AWF
        meas_AWF = ArrayWFrame(values_array, units_list[0], frame)

        # For compatibility with real measurement models
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    ## Partial math by StateVector components

    # position
    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the position components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            The relative state between self and the target.

        Returns
        -------
        _partials  :numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the position components.

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        range1 = relative_state[0:3].norm()
        range_rate1 = (relative_state[0:3] @ relative_state[3:6]) / range1
        h_tilde = [
            (
                (relative_state[3] / range1)
                - (relative_state[0] * range_rate1) / (range1 * range1)
            ).values,
            (
                (relative_state[4] / range1)
                - (relative_state[1] * range_rate1) / (range1 * range1)
            ).values,
            (
                (relative_state[5] / range1)
                - (relative_state[2] * range_rate1) / (range1 * range1)
            ).values,
        ]
        return np.array(h_tilde)

    # velocity
    def _compute_h_tilde_vel(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the velocity components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            The relative state between self and the target.

        Returns
        -------
        _partials : numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the velocity components.

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        range1 = relative_state[0:3].norm()
        h_tilde = [
            (relative_state[0] / range1).values,
            (relative_state[1] / range1).values,
            (relative_state[2] / range1).values,
        ]
        return np.array(h_tilde)

    # eta_srp
    def _compute_h_tilde_eta_srp(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the srp scaling factor (eta_srp).

        Returns
        -------
        partial : numpy.ndarray
            The scalar partial derivative of the measurement model by the position components.

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        h_tilde = [(ArrayWUnits(0, None)).values]
        return np.array(h_tilde)

    # gs_delta_location
    def _compute_h_tilde_gs_delta_location(
        self, relative_state: ArrayWUnits
    ) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the ground station position components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            The relative state between self and the target.

        Returns
        -------
        _partials : numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the position components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        range1 = relative_state[0:3].norm()
        range_rate1 = (relative_state[0:3] @ relative_state[3:6]) / range1
        h_tilde = [
            (
                -(relative_state[3] / range1)
                + (relative_state[0] * range_rate1) / (range1 * range1)
            ).values,
            (
                -(relative_state[4] / range1)
                + (relative_state[1] * range_rate1) / (range1 * range1)
            ).values,
            (
                -(relative_state[5] / range1)
                + (relative_state[2] * range_rate1) / (range1 * range1)
            ).values,
        ]
        return np.array(h_tilde)

    # range_bias
    def _compute_h_tilde_range_bias(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the range bias.

        Returns
        -------
        partial : numpy.ndarray
            The scalar partial derivative of the measurement model by the position components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        h_tilde = [(ArrayWUnits(0, None)).values]
        return np.array(h_tilde)

    # dv_man
    def _compute_h_tilde_dv_man(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal doppler measurement model
        with respect to the maneuver DV components.

        Returns
        -------
            np.array: The (3,) vector the partial derivatives of the measurement model by the maneuver DV components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """
        # NOTE: In theory, this partial derivative isn't zero during thrusting, but we assume no spacecraft measurements are taken then due to power
        #       constraints, so adding an `if` conditional here isn't necessary; this can be improved later if needed.
        h_tilde = [0, 0, 0]
        return np.array(h_tilde)

    # Compute H_tilde with partial values
    def _partials(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: Frame,
    ) -> list:
        """
        Groups toghether the different components of measurement
        _partials in the global H-tilde. Returns the H-tilde array for the modelled measurement.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch : EpochArray
            The epochs.

        frame : Frame
            The reference frame.

        Returns
        -------
        h_tilde : list
            The H-tilde array with all measurements _partials from this model by component.
        """
        ## Unpack doppler aux data
        antenna_name = self.antenna_name
        transmit_frequency = self.doppler_transmit_frequency
        receive_station = self.ground_station_2

        # Pull current bias + gs delta from state (if provided)
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        # Generate the relative state between instrument and target
        relative_state = SpiceManager.get_state(
            trgt_bdy=target.name,
            epoch_time=epoch.times.values,
            reference_frame=frame.name,
            obsvr_bdy=self.instrument.name,
        )

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

        # Get turn-around-ratio from target
        instList = target.instrument_list
        for inst in instList:
            if inst.name == antenna_name:
                TR = inst.turn_ratio

        # Generate Doppler constant
        DopplerConstant = -transmit_frequency * TR / constants.c

        # Generate the dictionary of allowed StateVector component from the StateArray class
        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Stack h_tilde based on StateVector components
        h_tilde1 = []
        h_tilde2 = []
        # _bias_overrides = []  # list of (slot_idx, unscaled_value) applied post-DopplerConstant scaling
        target_sid = getattr(target, "spice_id", None)

        # Generate doppler _partials (PART-2)
        if self._state_definition is None:
            # Nominal case: only position and velocity for the target spacecraft
            h_tilde1.extend(self._compute_h_tilde_pos(relative_state))
            h_tilde1.extend(self._compute_h_tilde_vel(relative_state))
        else:
            # When multiple parameters/bodies exist in the state vector
            for component_def in self._state_definition:
                component = component_def[0]
                comp_body = component_def[4] if len(component_def) > 4 else None
                comp_body_sid = getattr(comp_body, "spice_id", None)

                if state_allowed["position"][2].match(component):
                    if comp_body_sid == target_sid:
                        # Position of the target spacecraft
                        h_tilde1.extend(self._compute_h_tilde_pos(relative_state))
                    else:
                        # Position of another body -> zero sensitivity
                        h_tilde1.extend([0.0] * 3)

                elif state_allowed["velocity"][2].match(component):
                    if comp_body_sid == target_sid:
                        # Velocity of the target spacecraft
                        h_tilde1.extend(self._compute_h_tilde_vel(relative_state))
                    else:
                        # Velocity of another body -> zero sensitivity
                        h_tilde1.extend([0.0] * 3)

                elif state_allowed["eta_srp"][2].match(component):
                    # eta_srp
                    h_tilde1.extend(self._compute_h_tilde_eta_srp())
                elif state_allowed["gs_delta_location"][2].match(component):
                    raise NotImplementedError(
                        "DopplerIdeal: partials w.r.t. gs_delta_location have not "
                        "been verified. The analytic formula is implemented below "
                        "but must be validated before use in a filter."
                    )
                    # --- unverified implementation (kept for reference) ---
                    # drhodot/dr_GS1 = −drhodot/dr_sc  (chain rule: dr_rel/dr_GS = −I).
                    # _compute_h_tilde_gs_delta_location already applies the sign flip.
                    # Chain rule for IAU_EARTH→J2000: h_from @ A_ei gives df_D/d(dr_EARTH).
                    # if self.instrument.spice_id == comp_body.spice_id:
                    #     A_ei = SpiceManager.get_xfrm(
                    #         frame_from="IAU_EARTH", frame_to="J2000", epoch=epoch.times.values,
                    #     )
                    #     h_tilde1.extend(
                    #         self._compute_h_tilde_gs_delta_location(relative_state) @ A_ei
                    #     )
                    # else:
                    #     h_tilde1.extend([0.0, 0.0, 0.0])
                elif state_allowed["rangerate_bias_Hz"][2].match(component):
                    raise NotImplementedError(
                        "DopplerIdeal: partials w.r.t. rangerate_bias_Hz have not "
                        "been verified. The implementation below is correct but the "
                        "bias override mechanism has not been validated end-to-end."
                    )
                    # --- unverified implementation (kept for reference) ---
                    # df_D/dbias = 1.  The bias is added *after* the DopplerConstant
                    # multiplication (computed_measurements line: doppler_meas += bias),
                    # so the partial must NOT go through the 2K scaling applied to h_tilde1.
                    # Strategy: store a 0 placeholder in h_tilde1 (so scaling gives 0),
                    # and record the (slot, true_value) pair in _bias_overrides to be
                    # applied directly to the final h_tilde after scaling.
                    # if self.instrument.spice_id == comp_body.spice_id:
                    #     _bias_overrides.append((len(h_tilde1), 1.0))
                    # h_tilde1.extend([0.0])  # placeholder — overwritten post-scaling
                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    # dv maneuver (dv_man, dv_man1, dv_man2)
                    h_tilde1.extend(self._compute_h_tilde_dv_man())
                else:  # If the component is not above, we add zeros by default.
                    h_tilde1.extend([0.0 for _ in range(component_def[1])])

        if receive_station is None:  # Two-way doppler
            h_tilde = 2 * DopplerConstant.values * np.array(h_tilde1)
            # for idx, val in _bias_overrides:  # restore unscaled bias partial (= 1)
            #     h_tilde[idx] = val
        else:  # Three-way doppler
            # Build h_tilde2 from GS2 (receive_station) relative state
            relative_state2 = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=receive_station.name,
            )

            if self._state_definition is None:
                h_tilde2.extend(self._compute_h_tilde_pos(relative_state2))
                h_tilde2.extend(self._compute_h_tilde_vel(relative_state2))
            else:
                for component_def in self._state_definition:
                    component = component_def[0]
                    comp_body = component_def[4] if len(component_def) > 4 else None
                    comp_body_sid = getattr(comp_body, "spice_id", None)

                    if state_allowed["position"][2].match(component):
                        if comp_body_sid == target_sid:
                            h_tilde2.extend(self._compute_h_tilde_pos(relative_state2))
                        else:
                            h_tilde2.extend([0.0] * 3)
                    elif state_allowed["velocity"][2].match(component):
                        if comp_body_sid == target_sid:
                            h_tilde2.extend(self._compute_h_tilde_vel(relative_state2))
                        else:
                            h_tilde2.extend([0.0] * 3)
                    elif state_allowed["eta_srp"][2].match(component):
                        h_tilde2.extend(self._compute_h_tilde_eta_srp())
                    elif state_allowed["gs_delta_location"][2].match(component):
                        raise NotImplementedError(
                            "DopplerIdeal (3-way): partials w.r.t. gs_delta_location "
                            "have not been verified for GS2. The analytic formula is "
                            "implemented below but must be validated before use in a filter."
                        )
                        # --- unverified implementation (kept for reference) ---
                        # rhodot2 does not depend on r_GS1, so h_tilde1 contributes 0 for
                        # this slot.  h_tilde2 carries the full GS2 sensitivity:
                        #   drhodot2/dr_GS2 = −drhodot2/dr_sc  (same sign-flip as GS1 case)
                        # if receive_station.spice_id == comp_body.spice_id:
                        #     A_ei = SpiceManager.get_xfrm(
                        #         frame_from="IAU_EARTH", frame_to="J2000", epoch=epoch.times.values,
                        #     )
                        #     h_tilde2.extend(
                        #         self._compute_h_tilde_gs_delta_location(relative_state2) @ A_ei
                        #     )
                        # else:
                        #     h_tilde2.extend([0.0, 0.0, 0.0])
                    elif state_allowed["rangerate_bias_Hz"][2].match(component):
                        raise NotImplementedError(
                            "DopplerIdeal (3-way): partials w.r.t. rangerate_bias_Hz "
                            "have not been verified for GS2. The implementation below is "
                            "correct but the bias override mechanism has not been validated."
                        )
                        # --- unverified implementation (kept for reference) ---
                        # df_D/dbias_GS2 = 1.  Same placeholder/override strategy as GS1:
                        # put 0 in h_tilde2; record (slot, 1.0) in _bias_overrides so
                        # the post-scaling fixup restores the correct unscaled value.
                        # if receive_station.spice_id == comp_body.spice_id:
                        #     _bias_overrides.append((len(h_tilde2), 1.0))
                        # h_tilde2.extend([0.0])  # placeholder — overwritten post-scaling
                    elif any(
                        state_allowed[key][2].match(component)
                        for key in state_allowed
                        if any(tag in key for tag in ["dv_man"])
                    ):
                        h_tilde2.extend(self._compute_h_tilde_dv_man())
                    else:
                        h_tilde2.extend([0.0 for _ in range(component_def[1])])

            h_tilde = DopplerConstant.values * np.array(
                np.array(h_tilde1) + np.array(h_tilde2)
            )
            # for idx, val in _bias_overrides:  # restore unscaled bias partials (= 1)
            #     h_tilde[idx] = val
        return h_tilde
