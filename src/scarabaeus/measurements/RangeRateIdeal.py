# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Measurement,
    MissionSequence,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    Spacecraft,
    Instrument,
    Noise,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np

# ------------------#
#  Generate Units  #
# ------------------#
km, sec = Units.get_units(["km", "sec"])

# Generate common units and frames
spc_pos_units = km
spc_time_units = sec
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class RangeRateIdeal(Measurement):
    """Models the ideal range-rate measurement model.

    Generates range-rate observables between an observer and a target,
    expressed in km/s.

    The ideal observable is the line-of-sight projection of the relative
    velocity (no light-time delay, no media effects):

    .. math::

        \\dot{\\rho} = \\frac{(\\mathbf{r}_{sc} - \\mathbf{r}_{gs})
                              \\cdot (\\mathbf{v}_{sc} - \\mathbf{v}_{gs})}
                             {\\|\\mathbf{r}_{sc} - \\mathbf{r}_{gs}\\|}

    Parameters
    ----------
    name : str
        The name of the measurement model.
    instrument : Instrument
        Instrument object from Scarabaeus. An 'antenna' instrument is used
        for radiometric measurement models.
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    meas_bias : float | ArrayWUnits | ArrayWFrame, optional
        Ground station range-rate bias; a float is taken in km/sec. A
        ``rangerate_bias`` entry for this station in the reference state
        overrides it.
        Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : MissionSequence, optional
        Mission sequence whose per-leg state definitions replace
        *state_definition*. Defaults to ``None``.

    Raises
    ------
    RuntimeError
        If the units extracted from the measurements are not consistent.
    ValueError
        If neither an EpochArray nor start/end Epochs are provided.

    See Also
    --------
    scarabaeus.Measurement : Abstract base class for all measurement models.

    Examples
    --------
    .. code-block:: python

        # import libraries
        import scarabaeus as scb

        # Generate an antenna object and link it to an existing orbiter
        Orbiter_spice_id = -64
        antenna_sc = scb.Antenna("Antenna_for_radiometric", spice_id = Orbiter_spice_id)
        Orbiter.add_instrument([antenna_sc])
        Doppler_transmit_frequency = scb.ArrayWUnits(
            8.8 * 10**9, sec**-1
        )

        # Generate a Centroid measurement model
        rangerate_sigma = scb.ArrayWUnits(1e-5, km / sec)

        # Generate a ground-station object
        GS1 = scb.GroundStation("DSS-14")

        # Generate an ideal range measurement model
        RangeRate_GS1 = scb.RangeRateIdeal(
            name="GS1 Ideal RangeRate Model", instrument=GS1, sigma=rangerate_sigma
        )

        # Write observed measurements in .json format
        RangeRate_GS1.write_observed_measurements(
            target=Orbiter,
            epoch_array=epoch_array_rangerate,
            noisy=True,
            file_name="ideal_range_rate",
        )

        # Read observed measurement in .json format
        obs_quantities_rangerate = RangeRate_GS1.observed_measurements(
            file_name="data/dwn_data/ideal_range_rate.json", meas_name="meas_sim", units=km / sec
        )

        # Generate computed measurements on a trajectory
        computed_rangerate_GS1 = RangeRate_GS1.computed_measurements(
            target=Orbiter_perturbed,
            epoch_array=epoch_rangerate_GS1_et,
            frame=J2000,
        )

        # Compute the _partials
        rangerate_GS1_partials = RangeRate_GS1.compute_partials(
            target=Orbiter_perturbed, epoch_array=epoch_rangerate_GS1_et, frame=J2000
        )

        # Compute the residuals
        residuals_rangerate_GS1 = RangeRate_GS1.residuals(observed_rangerate_GS1, computed_rangerate_GS1)

        # Generate a measurement dataset
        rangerate_dataset_GS1 = RangeRate_GS1.generate_measurement_dataset(
            "GS1 RangeRate",
            target=Orbiter_perturbed,
            observed_meas=obs_quantities_rangerate,
        )
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        sigma: ArrayWUnits = None,
        meas_bias: float = None,
        state_definition: list = None,
        sequence_definition: MissionSequence = None,
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument
        self._sigma = sigma
        self._meas_bias = meas_bias
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "RangeRateIdeal"

    # ----------------#
    # region Methods #
    # ----------------#

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_sim",
        units=None,
        frame: Frame = J2000,
        pass_epochs: list[list] | None = None,
    ):
        """
        Load range-rate measurements from a .json file.

        Parameters
        ----------
        file_name : str
            Path to the ``.json`` file holding the observed measurements.
        meas_name : str, optional
            Name of the measurement entry to read from the file. Defaults to
            ``"meas_sim"``.
        units : Units, optional
            Units to assign to the loaded measurements. Defaults to ``None``.
        frame : Frame, optional
            Reference frame to assign to the loaded measurements. Defaults to
            ``J2000``.
        pass_epochs : list[list[float]], optional
            Observation epochs grouped by tracking pass, e.g.
            ``[[t0, t1, t2], [t10, t11]]``. Must be provided when the state
            vector contains a stacked ``rangerate_bias_<suffix>`` entry
            (dim > 1). Defaults to ``None``.

        Returns
        -------
        observed_meas : ObsData
            The observed measurement row produced by
            :meth:`~scarabaeus.Measurement.observed_measurements`.
        """
        result = super().observed_measurements(file_name, meas_name, units, frame)
        if pass_epochs is not None:
            self.set_pass_epochs(pass_epochs)
        return result

    def computed_measurements(
        self,
        target: Spacecraft,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        pass_epochs: list[list] | None = None,
    ) -> ArrayWFrame:
        """
        Compute the range rate measurement between the observer and a target spacecraft.

        Calculates the range rate measurement from the current spacecraft to
        a specified target spacecraft, taking into account optional
        parameters like specific epochs, time steps, reference frame and
        noise settings.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft for which the range rate measurement is to
            be computed.
        epoch_array : EpochArray, optional
            An array of epochs (times) at which the range rate measurements
            should be computed. If provided, it overrides ``epoch_start``,
            ``epoch_end`` and ``tstep``. Defaults to ``None``.
        epoch_start : EpochArray, optional
            The starting epoch for the range rate measurement computations.
            Required if ``epoch_array`` is not provided. Defaults to ``None``.
        epoch_end : EpochArray, optional
            The ending epoch for the range rate measurement computations.
            Required if ``epoch_array`` is not provided. Defaults to ``None``.
        tstep : float, optional
            The time step, in seconds, between consecutive range rate
            measurements if ``epoch_array`` is not provided. Defaults to
            1 second.
        frame : Frame, optional
            The reference frame in which the range rate computation is
            performed. Defaults to ``J2000``.
        noisy : bool, optional
            Whether to add noise to the computed range rate measurement.
            Defaults to ``False``.
        pass_epochs : list[list[float]], optional
            Observation epochs grouped by tracking pass, e.g.
            ``[[t0, t1, t2], [t10, t11]]``. When given, per-pass range-rate
            biases are enabled for this computation. Defaults to ``None``.

        Returns
        -------
        t1 : list of EpochArray
            The computation epochs, returned for compatibility with the real
            measurement models.
        t2 : list of EpochArray
            The computation epochs, returned for compatibility with the real
            measurement models.
        t3 : list of EpochArray
            The computation epochs, returned for compatibility with the real
            measurement models.
        meas_AWF : ArrayWFrame
            The computed range rate measurement as an array with frames.

        Raises
        ------
        ValueError
            If neither an :class:`~scarabaeus.EpochArray` nor start and end
            Epochs are provided.
        RuntimeError
            If multiple units are extracted from the measurements.
        MissingPassIndexError
            If the bias holds one value per pass (more than one element)
            and no epoch-to-pass mapping is set.
        """
        if pass_epochs is not None:
            self.set_pass_epochs(pass_epochs)

        ## Input check
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        # One bias for every pass, or one per pass.
        _bias_full = self._extract_bias("rangerate_bias", km / sec)

        ## Append measurements in a list at multiple epochs
        meas_list = []
        # Get relative states
        for epoch in epoch_array:
            bias_awu = self._bias_at_epoch(_bias_full, epoch)

            relative_state = self._relative_state(target, epoch, frame)


            range_meas = relative_state[0:3].norm()
            # Get range rate measurement as AWU
            range_rate_meas = (relative_state[0:3] @ relative_state[3:6]) / range_meas

            # Apply bias
            if bias_awu is not None:
                range_rate_meas = range_rate_meas + bias_awu

            # Add noise if requested
            if noisy:
                range_rate_meas += Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma.values,
                    units=self.sigma.units,
                )
            meas_list.append(range_rate_meas)  # list of AWU measurements

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
    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> list:
        """
        Generate the ``h_tilde`` block for the position components.

        Builds the portion of the ``h_tilde`` matrix holding the partial of
        the ideal range-rate measurement model with respect to the position
        components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            Relative state between self and target.

        Returns
        -------
        h_tilde : numpy.ndarray
            The (3,) vector of partial derivatives of the measurement model
            by the position components.

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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
        Generate the ``h_tilde`` block for the velocity components.

        Builds the portion of the ``h_tilde`` matrix holding the partial of
        the ideal range-rate measurement model with respect to the velocity
        components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            Relative state between self and target.

        Returns
        -------
        h_tilde : numpy.ndarray
            The (3,) vector of partial derivatives of the measurement model
            by the velocity components.

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        range_val = relative_state[0:3].norm()
        h_tilde = [
            (relative_state[0] / range_val).values,
            (relative_state[1] / range_val).values,
            (relative_state[2] / range_val).values,
        ]
        return np.array(h_tilde)

    # eta_srp
    def _compute_h_tilde_eta_srp(self) -> list:
        """
        Generate the ``h_tilde`` block for the SRP scaling factor.

        Builds the portion of the ``h_tilde`` matrix holding the partial of
        the ideal range-rate measurement model with respect to the SRP
        scaling factor (``eta_srp``).

        Returns
        -------
        h_tilde : numpy.ndarray
            The (1,) scalar partial derivative of the measurement model by
            the position components.

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """
        h_tilde = [(ArrayWUnits(0, None)).values]
        return np.array(h_tilde)

    # dv_man
    def _compute_h_tilde_dv_man(self, relative_state: ArrayWUnits) -> list:
        """
        Generate the ``h_tilde`` block for the maneuver DV components.

        Builds the portion of the ``h_tilde`` matrix holding the partial of
        the ideal range-rate measurement model with respect to the maneuver
        DV components.

        Parameters
        ----------
        relative_state : ArrayWUnits
            Relative state between self and target.

        Returns
        -------
        h_tilde : numpy.ndarray
            The (3,) vector of partial derivatives of the measurement model
            by the maneuver DV components.

        Notes
        -----
        In theory, this partial derivative isn't zero during thrusting, but
        we assume no spacecraft measurements are taken then due to power
        constraints, so adding an ``if`` conditional here isn't necessary;
        this can be improved later if needed. The non-zero form would be

        .. code-block:: python

            relative_pos = relative_state[0:3]
            range_val = relative_state[0:3].norm()
            h_tilde = [(relative_pos[i] / range_val).values for i in range(3)]

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
        """

        h_tilde = [0, 0, 0]
        return np.array(h_tilde)

    ## Compute H_tilde with partial values
    def _partials(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Group the measurement partials into the global H-tilde.

        Groups together the different components of the measurement partials
        in the global H-tilde and returns the H-tilde array for the modelled
        measurement.

        Parameters
        ----------
        target : Spacecraft or Body
            Target spacecraft as a :class:`~scarabaeus.Spacecraft` or Body
            object.
        epoch : EpochArray
            Epochs at which the partials are evaluated.
        frame : Frame, optional
            Reference frame in which the partials are computed. Defaults to
            ``J2000``.

        Returns
        -------
        h_tilde : list
            The H-tilde array with all measurement partials from this model
            by component.

        Raises
        ------
        MissingPassIndexError
            If the state holds a stacked per-pass bias for this station
            (dimension > 1) and no epoch-to-pass mapping is set.
        """

        # Get relative state, about the same geometry the forward model uses.
        relative_state = self._relative_state(target, epoch, frame)


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


                elif "rangerate_bias" in state_allowed and state_allowed[
                    "rangerate_bias"
                ][2].match(component):
                    h_tilde.extend(
                        self._bias_partials(
                            component, component_def[1], comp_body, epoch
                        )
                    )
                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    h_tilde.extend(self._compute_h_tilde_dv_man(relative_state))
                elif self._is_station_parameter(component):
                    # Range rate sees the station through both its position and
                    # its velocity, so the chain rule needs both partials.
                    h_tilde.extend(self._compute_h_tilde_station(
                        component_def, epoch.times.values, frame,
                        self._compute_h_tilde_pos(relative_state),
                        self._compute_h_tilde_vel(relative_state))[0])

                else:  # If the component is not above, we add zeros by default.
                    h_tilde.extend([0.0 for _ in range(component_def[1])])
        return h_tilde
