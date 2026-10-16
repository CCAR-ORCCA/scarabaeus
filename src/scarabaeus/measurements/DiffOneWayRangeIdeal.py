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
    Spacecraft,
    Instrument,
    GroundStation,
    Noise,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
km, sec = Units.get_units(["km", "sec"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class DiffOneWayRangeIdeal(Measurement):
    """Models the ideal differential one-way ranging measurement model.

    Generates differenced ranging observables between two observers and a
    target.  Observables are expressed in seconds.

    The differential one-way range (DOR) observable is the difference in
    signal travel time from the target to two ground stations:

    .. math::

        \\Delta\\tau = \\frac{\\rho_1 - \\rho_2}{c}

    where :math:`\\rho_1` and :math:`\\rho_2` are the ranges to ground
    stations 1 and 2 respectively, and :math:`c` is the speed of light.

    Parameters
    ----------
    name : str
        Name of the measurement model.
    instrument : Instrument
        Instrument object associated with the measurement model.
    sigma : ArrayWUnits, optional
        Measurement standard deviation in seconds. Defaults to ``None``.
    meas_bias : float | ArrayWUnits | ArrayWFrame, optional
        Measurement bias, added to the first station's range; a float is
        taken in km. A ``dor_bias`` entry for this station in the reference
        state overrides it; it holds one value for every pass or one per
        pass (see :meth:`~scarabaeus.Measurement.set_pass_epochs`) and must
        be given in km.
        Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : MissionSequence, optional
        Mission sequence whose per-leg state definitions replace
        *state_definition*. Defaults to ``None``.
    ground_station_2 : GroundStation, optional
        Second ground station needed for the differential measurement model.
        Defaults to ``None``.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.
    """

    # -------------#
    # Constructor #
    # -------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        sigma: ArrayWUnits = None,
        meas_bias: float = None,
        state_definition: list = None,
        sequence_definition=None,
        ground_station_2: GroundStation = None,
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument
        self._sigma = sigma  # For now the sigma is the same for all measurement
        self._meas_bias = (
            meas_bias  # For now the range_bias is the same for all measurement
        )
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._station = ground_station_2
        self._measurement_type = "DiffOneWayRangeIdeal"

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def station(self) -> ArrayWUnits:
        """
        The station of the measurement model.
        """
        return self._station

    @property
    def meas_bias(self):
        """Range bias added to the first station's range [km]."""
        return self._meas_bias

    @property
    def state_definition(self) -> list:
        """
        The state vector definition
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> MissionSequence:
        """
        The sequence definition.
        """
        return self._sequence_definition

    # endregion => Properties #
    # -------------------------#

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
    ) -> ArrayWFrame:
        """
        Compute the differential one-way range between the two stations.

        At each epoch the geometric ranges from the first station
        (:attr:`instrument`) and the second station (:attr:`station`) to the
        target are differenced and divided by the speed of light,
        :math:`(\\rho_2 - \\rho_1) / c`. A ``gs_delta_location`` in the reference
        state shifts the first station, and the bias of
        :meth:`_extract_bias` is added to the first station's range.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.
        epoch_array : EpochArray, optional
            Epochs at which the measurements are computed. If provided, it
            overrides ``epoch_start``, ``epoch_end`` and ``tstep``. Defaults to
            ``None``.
        epoch_start : EpochArray, optional
            First epoch of the computation. Required if ``epoch_array`` is not
            provided. Defaults to ``None``.
        epoch_end : EpochArray, optional
            Last epoch of the computation. Required if ``epoch_array`` is not
            provided. Defaults to ``None``.
        tstep : float, optional
            Time step between consecutive measurements, in seconds, when
            ``epoch_array`` is not provided. Defaults to ``1``.
        frame : Frame, optional
            Reference frame in which the relative states are computed. Defaults to
            ``J2000``.
        noisy : bool, optional
            Whether to add white Gaussian noise with standard deviation
            :attr:`sigma`. Defaults to ``False``.

        Returns
        -------
        t1 : EpochArray
            The computation epochs, returned for compatibility with the real
            measurement models.
        t2 : EpochArray
            Same as ``t1``.
        t3 : EpochArray
            Same as ``t1``.
        meas_AWF : ArrayWFrame
            The DOR measurements, in seconds.

        Raises
        ------
        ValueError
            If neither ``epoch_array`` nor ``epoch_start`` and ``epoch_end`` are
            provided.
        RuntimeError
            If the measurements do not all share one unit.
        MissingPassIndexError
            If the bias holds one value per pass (more than one element)
            and no epoch-to-pass mapping is set.

        Notes
        -----
        The station delta location and the bias are applied to the first station
        only.
        """
        ## Input check
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        # Get AWU for speed-of-light constant
        c = constants.c

        # One bias for every pass, or one per pass.
        _bias_full = self._extract_bias("dor_bias", km)

        ## Append measurements in a list at multiple epochs
        meas_list = []
        for epoch in epoch_array:
            bias_awu = self._bias_at_epoch(_bias_full, epoch)

            # Generate the relative state between first GS and target
            relative_state1 = self._relative_state(target, epoch, frame)
            # Generate the relative state between second GS and target
            relative_state2 = self._relative_state(target, epoch, frame, self.station)

            range1 = relative_state1[0:3].norm()
            range2 = relative_state2[0:3].norm()
            # Add range bias if requested
            if bias_awu is not None:
                range1 = range1 + bias_awu

            # Generate ideal DOR measurement
            DOR_meas = (range2 - range1) / c

            # Add noise to ideal DOR
            if noisy:
                DOR_meas += Noise().generate_AWGN_with_units(
                    mu=0.0, sigma=self.sigma.values, units=self.sigma.units
                )
            meas_list.append(DOR_meas)  # list of AWU measurements

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
    def _compute_h_tilde_pos(
        self, relative_state_1: ArrayWUnits, relative_state_2: ArrayWUnits
    ) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal DOR measurement model
        with respect to the position components.

        Parameters
        ----------
        relative_state_1 : ArrayWUnits
            The relative state between ground station #1 and the target.

        relative_state_2 : ArrayWYnts
            The relative state between ground station #2 and the target.

        Returns
        -------
        _partials : numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the position components

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        range1 = relative_state_1[0:3].norm()
        range2 = relative_state_2[0:3].norm()
        # Get AWU for speed-of-light constant
        c = constants.c
        h_tilde = [
            (
                ((relative_state_2[0] / range2) - (relative_state_1[0] / range1)) / c
            ).values,
            (
                ((relative_state_2[1] / range2) - (relative_state_1[1] / range1)) / c
            ).values,
            (
                ((relative_state_2[2] / range2) - (relative_state_1[2] / range1)) / c
            ).values,
        ]
        return np.array(h_tilde)

    def _leg_position_partials(
        self, relative_state_1: ArrayWUnits, relative_state_2: ArrayWUnits
    ) -> tuple:
        """
        Split the position partial into each station's own contribution.

        :meth:`_compute_h_tilde_pos` returns the difference the observable
        actually measures. A station parameter, though, moves one station only,
        so its partial has to chain through that leg alone; a body-owned
        Earth-orientation parameter moves both and the two blocks add.

        Parameters
        ----------
        relative_state_1, relative_state_2 : ArrayWUnits
            Relative states from station #1 and station #2 to the target.

        Returns
        -------
        tuple of numpy.ndarray
            The two ``(3,)`` contributions to the DOR partial with respect to
            target position, in seconds per km, summing to the full partial.
        """
        c = constants.c
        range1 = relative_state_1[0:3].norm()
        range2 = relative_state_2[0:3].norm()
        return (
            np.array([(-(relative_state_1[i] / range1) / c).values for i in range(3)]),
            np.array([((relative_state_2[i] / range2) / c).values for i in range(3)]),
        )

    def _compute_h_tilde_vel(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the DOR measurement model
        with respect to the velocity components.

        Returns
        -------
        _partials : numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the velocity components

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        h_tilde = [
            ArrayWUnits(0, None).values,
            ArrayWUnits(0, None).values,
            ArrayWUnits(0, None).values,
        ]
        return np.array(h_tilde)

    def _compute_h_tilde_eta_srp(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal DOR measurement model
        with respect to the srp scaling factor (eta_srp).

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

    def _compute_h_tilde_gs1_delta_location(
        self, relative_state_1: ArrayWUnits
    ) -> np.ndarray:
        """Partial of :math:`\\Delta\\tau` w.r.t. ground-station 1 position (J2000).

        The DOR observable is :math:`\\Delta\\tau = (\\rho_2 - \\rho_1)/c`.
        Since :math:`\\rho_1 = |\\mathbf{r}_{sc} - \\mathbf{r}_{GS1}|`,

        .. math::

            \\frac{\\partial\\Delta\\tau}{\\partial\\mathbf{r}_{GS1}}
            = +\\frac{\\hat{\\boldsymbol{\\rho}}_1}{c}

        Parameters
        ----------
        relative_state_1 : ArrayWUnits
            Relative state :math:`\\mathbf{r}_{sc} - \\mathbf{r}_{GS1}` in J2000.

        Returns
        -------
        h_tilde : numpy.ndarray
            Shape ``(3,)`` partial derivatives in J2000 frame [s/km].

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, eq. 4.2.6.
        """
        range1 = relative_state_1[0:3].norm()
        c = constants.c
        h_tilde = [
            ((relative_state_1[0] / range1) / c).values,
            ((relative_state_1[1] / range1) / c).values,
            ((relative_state_1[2] / range1) / c).values,
        ]
        return np.array(h_tilde)

    def _compute_h_tilde_gs2_delta_location(
        self, relative_state_2: ArrayWUnits
    ) -> np.ndarray:
        """Partial of :math:`\\Delta\\tau` w.r.t. ground-station 2 position (J2000).

        Since :math:`\\rho_2 = |\\mathbf{r}_{sc} - \\mathbf{r}_{GS2}|`,

        .. math::

            \\frac{\\partial\\Delta\\tau}{\\partial\\mathbf{r}_{GS2}}
            = -\\frac{\\hat{\\boldsymbol{\\rho}}_2}{c}

        Parameters
        ----------
        relative_state_2 : ArrayWUnits
            Relative state :math:`\\mathbf{r}_{sc} - \\mathbf{r}_{GS2}` in J2000.

        Returns
        -------
        h_tilde : numpy.ndarray
            Shape ``(3,)`` partial derivatives in J2000 frame [s/km].

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, eq. 4.2.6.
        """
        range2 = relative_state_2[0:3].norm()
        c = constants.c
        h_tilde = [
            (-(relative_state_2[0] / range2) / c).values,
            (-(relative_state_2[1] / range2) / c).values,
            (-(relative_state_2[2] / range2) / c).values,
        ]
        return np.array(h_tilde)

    def _compute_h_tilde_dor_bias(self) -> np.ndarray:
        """Partial of :math:`\\Delta\\tau` w.r.t. the DOR range bias.

        The bias is added to :math:`\\rho_1` before dividing by :math:`c`:

        .. math::

            \\Delta\\tau = \\frac{\\rho_2 - (\\rho_1 + b)}{c}
            \\implies
            \\frac{\\partial\\Delta\\tau}{\\partial b} = -\\frac{1}{c}

        where :math:`b` is the range bias in km and :math:`c` is the speed
        of light in km/s.

        Returns
        -------
        h_tilde : numpy.ndarray
            Shape ``(1,)`` scalar partial [s/km].

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, eq. 4.2.6.
        """
        c = constants.c
        h_tilde = [(-ArrayWUnits(1.0, None) / c).values]
        return np.array(h_tilde)

    def _compute_h_tilde_dv_man(self) -> np.ndarray:
        """
        Generates the portion in the h_tilde matrix relative to the partial of the ideal DOR measurement model
        with respect to the maneuver DV components.

        Returns
        -------
        _partials : numpy.ndarray
            The 3x1 partial derivatives of the measurement model by the maneuver DV components

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        h_tilde = [0, 0, 0]
        return np.array(h_tilde)

    def _partials(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: Frame,
    ) -> list:
        """
        Groups toghether the different components of measurement
        _partials in the global H-tilde. It returns the H-tilde array for the modelled measurement.

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

        Raises
        ------
        MissingPassIndexError
            If the state holds a stacked per-pass bias for this station
            (dimension > 1) and no epoch-to-pass mapping is set.
        """
        # Generate the relative state between first GS and target
        relative_state_1 = self._relative_state(target, epoch, frame)

        relative_state_2 = self._relative_state(target, epoch, frame, self.station)

        # Generate the dictionary of allowed StateVector component from the StateArray class
        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Stack h_tilde based on StateVector components
        h_tilde = []
        target_sid = getattr(target, "spice_id", None)
        if (
            self._state_definition is None
        ):  # Assumes that in the nominal case only position and velocity are StateVector components
            h_tilde.extend(
                self._compute_h_tilde_pos(relative_state_1, relative_state_2)
            )
            h_tilde.extend(self._compute_h_tilde_vel())
        else:
            for component_def in self._state_definition:
                component = component_def[0]
                comp_body = component_def[4] if len(component_def) > 4 else None
                comp_body_sid = getattr(comp_body, "spice_id", None)

                if state_allowed["position"][2].match(component):
                    if comp_body_sid == target_sid:
                        # Position of the target spacecraft
                        h_tilde.extend(
                            self._compute_h_tilde_pos(
                                relative_state_1, relative_state_2
                            )
                        )
                    else:
                        # Position of some other body -> no sensitivity
                        h_tilde.extend([0.0] * 3)
                elif state_allowed["velocity"][2].match(component):
                    # velocity
                    if comp_body_sid == target_sid:
                        # Velocity of the target spacecraft
                        h_tilde.extend(self._compute_h_tilde_vel())
                    else:
                        # Velocity of some other body → no sensitivity
                        h_tilde.extend([0.0] * 3)

                elif state_allowed["eta_srp"][2].match(component):
                    # eta_srp
                    h_tilde.extend(self._compute_h_tilde_eta_srp())

                elif state_allowed["dor_bias"][2].match(component):
                    # The bias is a range on the first leg, so each slot
                    # carries -1/c rather than 1.
                    per_km = float(np.ravel(self._compute_h_tilde_dor_bias())[0])
                    h_tilde.extend(
                        per_km * np.asarray(
                            self._bias_partials(
                                component, component_def[1], comp_body, epoch
                            )
                        )
                    )
                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    # dv maneuver (dv_man, dv_man1, dv_man2)
                    h_tilde.extend(self._compute_h_tilde_dv_man())
                elif self._is_station_parameter(component):
                    leg_1, leg_2 = self._leg_position_partials(
                        relative_state_1, relative_state_2)
                    h_tilde.extend(
                        self._compute_h_tilde_station(
                            component_def, epoch.times.values, frame,
                            leg_1, station=self.instrument)[0]
                        + self._compute_h_tilde_station(
                            component_def, epoch.times.values, frame,
                            leg_2, station=self.station)[0])

                else:  # If the component is not above, we add zeros by default.
                    h_tilde.extend([0.0 for _ in range(component_def[1])])
        return h_tilde
