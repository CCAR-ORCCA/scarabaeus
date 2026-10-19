# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    Measurement,
    MissionSequence,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Instrument,
    GroundStation,
    Spacecraft,
    Noise,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
km, sec, Hz = Units.get_units(["km", "sec", "Hz"])
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
    Station-location and EOP partials differentiate the complete observable
    through the common Measurement geometry. The ``rangerate_bias_Hz``
    partial is 1 for this station's bias, in the slot of the epoch's pass
    when the bias holds one value per pass.

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
    sequence_definition : MissionSequence, optional
        Mission sequence whose per-leg state definitions replace
        *state_definition*. Defaults to ``None``.
    meas_bias : float | ArrayWUnits | ArrayWFrame, optional
        Doppler bias; a float is taken in Hz. A ``rangerate_bias_Hz`` entry
        for this station in the reference state overrides it; it holds one
        value for every pass or one per pass (see
        :meth:`~scarabaeus.Measurement.set_pass_epochs`) and must be given in
        the units of the computed Doppler. Defaults to ``None``.
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
    MissingPassIndexError
        If the bias holds one value per pass and no epoch-to-pass mapping is
        set.
    RuntimeError
        If the units extracted from the measurements are not consistent.
    ValueError
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

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def state_definition(self) -> list:
        """
        The state vector definition.
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> MissionSequence:
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
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        ## Unpack doppler aux data
        antenna_name = self.antenna_name
        transmit_frequency = self.doppler_transmit_frequency
        receive_station = self.ground_station_2

        # One bias for every pass, or one per pass.
        _bias_full = self._extract_bias("rangerate_bias_Hz", Hz)

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
            bias_awu = self._bias_at_epoch(_bias_full, epoch)
            relative_state = self._relative_state(target, epoch, frame)

            range1 = relative_state[0:3].norm()
            # Get range rate measurement as AWU
            range_rate1 = (relative_state[0:3] @ relative_state[3:6]) / range1

            # Differentiate between two-way and three-way doppler
            if receive_station is None:  # Two-Way doppler
                doppler_meas = 2 * DopplerConstant * range_rate1

            else:  # Three-way doppler
                relative_state2 = self._relative_state(
                    target, epoch, frame, receive_station)
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
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        h_tilde = [(ArrayWUnits(0, None)).values]
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
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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

        # Generate the relative state between instrument and target
        relative_state = self._relative_state(target, epoch, frame)


        instList = target.instrument_list
        for inst in instList:
            if inst.name == antenna_name:
                TR = inst.turn_ratio

        # Generate Doppler constant
        DopplerConstant = -transmit_frequency * TR / constants.c

        # Generate the dictionary of allowed StateVector component from the StateArray class
        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Express each block in the delivered frequency units before assembly.
        # Keep each leg's own contribution: a station parameter chains through
        # the geometry of the station it moves, and only that one, while a
        # body-owned Earth-orientation parameter moves both legs at once.
        factor = float(DopplerConstant.values)
        leg_scale = 2.0 if receive_station is None else 1.0
        H_pos = leg_scale * factor * self._compute_h_tilde_pos(relative_state)
        H_vel = leg_scale * factor * self._compute_h_tilde_vel(relative_state)
        legs = [(self.instrument, H_pos, H_vel)]
        if receive_station is not None:
            relative_state2 = self._relative_state(target, epoch, frame, receive_station)
            H_pos_2 = factor * self._compute_h_tilde_pos(relative_state2)
            H_vel_2 = factor * self._compute_h_tilde_vel(relative_state2)
            legs.append((receive_station, H_pos_2, H_vel_2))
            H_pos = H_pos + H_pos_2
            H_vel = H_vel + H_vel_2

        if self._state_definition is None:
            return np.r_[H_pos, H_vel]

        h_tilde = []
        for component_def in self._state_definition:
            name, dim, _, _, owner = component_def[:5]
            owner_sid = getattr(owner, "spice_id", None)
            if state_allowed["position"][2].match(name):
                block = H_pos if owner_sid == target.spice_id else np.zeros(dim)
            elif state_allowed["velocity"][2].match(name):
                block = H_vel if owner_sid == target.spice_id else np.zeros(dim)
            elif state_allowed["rangerate_bias_Hz"][2].match(name):
                block = np.asarray(self._bias_partials(name, dim, owner, epoch))
            elif self._is_station_parameter(name):
                # Each leg contributes through the station it moves; a
                # body-owned orientation parameter moves both, so they add.
                block = np.zeros(dim)
                for station, leg_pos, leg_vel in legs:
                    block = block + self._compute_h_tilde_station(
                        component_def, epoch.times.values, frame,
                        leg_pos, leg_vel, station)[0]
            else:  # If the component is not above, we add zeros by default.
                block = np.zeros(dim)
            h_tilde.extend(block)
        return np.asarray(h_tilde)
