# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Measurement,
    Units,
    Frame,
    ArrayWUnits,
    ArrayWFrame,
    EpochArray,
    StateArray,
    Spacecraft,
    SpiceManager,
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
    meas_bias : float, optional
        Ground station range-rate bias. Defaults to ``None``.
    state_definition : list, optional
        StateVector definition list. Defaults to ``None``.
    sequence_definition : list, optional
        Sequence definition list. Defaults to ``None``.

    Raises
    ------
    RuntimeError
        If the units extracted from the measurements are not consistent.
    RuntimeError
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
        Orbiter.addInstrument([antenna_sc])
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
            file_name="data/dwn_data/ideal_range_rate.json", meas_name="meas_ideal", units=km / sec
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
        sequence_definition: list = None,
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
        self._reference_state_vector: StateArray | None = None
        self._epoch_to_pass_index: dict | None = None

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def sigma(self) -> ArrayWUnits:
        """Measurement noise standard deviation."""
        return self._sigma

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    @staticmethod
    def _build_pass_index(obs_epochs, gap_threshold_sec: float = 300.0) -> dict:
        """
        Build {epoch_value: pass_index} mapping.
        A new pass begins whenever consecutive epoch gap > gap_threshold_sec.
        """
        epochs = sorted(float(t) for t in obs_epochs)
        mapping: dict = {}
        pass_idx = 0
        for i, t in enumerate(epochs):
            if i > 0 and (t - epochs[i - 1]) > gap_threshold_sec:
                pass_idx += 1
            mapping[t] = pass_idx
        return mapping

    def set_pass_epochs(self, pass_epochs: list[list] | None):
        """
        Enable per-pass range-rate biases.

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

    def set_pass_index(self, mapping: dict | None):
        """Directly assign a pre-built ``{epoch_value: pass_index}`` mapping (or None to disable)."""
        self._epoch_to_pass_index = mapping

    def _extract_rangerate_bias(self) -> ArrayWUnits | None:
        """
        Return range-rate bias for THIS station (global mode — first matching state entry).
        """
        if self._reference_state_vector is None:
            return self._meas_bias.quantity if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        if "rangerate_bias" not in state_allowed:
            return None

        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame (even for scalar bias)

            if state_allowed["rangerate_bias"][2].match(component) is None:
                continue

            body_sid = getattr(body, "spice_id", None)
            if body_sid == self.instrument.spice_id:
                return value.quantity

        return None

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_ideal",
        units=None,
        frame: Frame = J2000,
        pass_epochs: list[list] | None = None,
    ):
        """
        Load range-rate measurements from a .json file.

        pass_epochs : list[list[float]], optional
            Observation epochs grouped by tracking pass, e.g.
            ``[[t0, t1, t2], [t10, t11]]``.  Must be provided when the state
            vector contains a stacked ``rangerate_bias_<suffix>`` entry (dim > 1).
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

        This function calculates the range rate measurement (distance) from the current spacecraft
        to a specified target spacecraft, taking into account optional parameters like
        specific epochs, time steps, reference frame, and noise settings.

        :param target: The target spacecraft for which the range rate measurement is to be computed.
        :type target: Spacecraft

        :param epoch_array: An array of epochs (times) at which the range rate measurements should be computed.
                            If provided, it overrides epoch_start, epoch_end, and tstep.
        :type epoch_array: EpochArray, optional

        :param epoch_start: The starting epoch for the range ratemeasurement computations.
                            Required if `epoch_array` is not provided.
        :type epoch_start: Epoch, optional

        :param epoch_end: The ending epoch for the range ratemeasurement computations.
                          Required if `epoch_array` is not provided.
        :type epoch_end: Epoch, optional

        :param tstep: The time step, in seconds, between consecutive range ratemeasurements
                      if `epoch_array` is not provided. Defaults to 1 second.
        :type tstep: float, optional

        :param frame: The reference frame in which the range rate computation is performed.
        :type frame: Frame, optional

        :param noisy: Whether to add noise to the computed range rate measurement. Defaults to False.
        :type noisy: bool, optional

        :param antenna_name: name of the antenna to be used for Doppler
        :type antenna_name: str

        :param transmit_frequency: transmit frequency, used to compute the doppler count
        :type transmit_frequency: ArrayWUnits

        :param receive_station: receiving station object for three-way doppler, defaults to None
        :type receive_station: GroundStation

        :raises ValueError: An EpochArray is not provided

        :returns: The computed range rate measurement as an array with frames.
        :rtype: ArrayWFrame
        """
        if pass_epochs is not None:
            self.set_pass_epochs(pass_epochs)

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

        # Extract full bias vector once (dim=1 global, dim=n_passes stacked per-pass).
        _bias_full = self._extract_rangerate_bias()
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        ## Append measurements in a list at multiple epochs
        meas_list = []
        # Get relative states
        for epoch in epoch_array:
            if self._epoch_to_pass_index is not None and _bias_full is not None:
                _pidx = self._epoch_to_pass_index.get(float(epoch.times.values))
                _bvals = _bias_full.values.ravel()
                bias_awu = (
                    ArrayWUnits(_bvals[_pidx], _bias_full.units)
                    if _pidx is not None and _pidx < len(_bvals)
                    else None
                )
            else:
                bias_awu = _bias_full

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

    # eta_srp
    def _compute_h_tilde_eta_srp(self) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the srp scaling factor (eta_srp).

        Parameters:

        Returns:
            np.array: The (1,) scalar partial derivative of the measurement model by the position components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        """
        h_tilde = [(ArrayWUnits(0, None)).values]
        return np.array(h_tilde)

    # gs_delta_location
    def _compute_h_tilde_gs_delta_location(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the ground station position components.

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
                -(relative_state[3] / range_val)
                + (relative_state[0] * range_rate) / (range_val * range_val)
            ).values,
            (
                -(relative_state[4] / range_val)
                + (relative_state[1] * range_rate) / (range_val * range_val)
            ).values,
            (
                -(relative_state[5] / range_val)
                + (relative_state[2] * range_rate) / (range_val * range_val)
            ).values,
        ]
        return np.array(h_tilde)

    # range_bias
    def _compute_h_tilde_range_bias(self) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the range bias.

        Returns:
            np.array: The (1,) vector the partial derivatives of the measurement model by the position components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        """
        h_tilde = [(ArrayWUnits(1, None)).values]
        return np.array(h_tilde)

    # dv_man
    def _compute_h_tilde_dv_man(self, relative_state: ArrayWUnits) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal rangerate measurement model
        with respect to the maneuver DV components.

        Parameters:
            relative_state (ArrayWUnits) : relative state between self and target

        Returns:
            np.array: The (3,) vector the partial derivatives of the measurement model by the maneuver DV components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

        Notes
        -----
        In theory, this partial derivative isn't zero during thrusting, but we assume no spacecraft measurements are taken then due to power
        constraints, so adding an `if` conditional here isn't necessary; this can be improved later if needed.
        relative_pos = relative_state[0:3]
        range_val = relative_state[0:3].norm()
        h_tilde = [(relative_pos[i] / range_val).values for i in range(3)]
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
        This method groups toghether the different components of measurement
        _partials in the global H-tilde. It returns the H-tilde array for the modelled measurement.

        Parameters:
            target (Body, Spacecraft) : target spacecraft as scb Spacecraft/Body object
            epoch (Epoch) : epochs as scb Epoch object
            frame (Frame) : reference frame as scb Frame object

        Returns:
            list: The H-tilde array with all measurements _partials from this model by component
        """

        # Get relative state
        relative_state = SpiceManager.get_state(
            trgt_bdy=target.name,
            epoch_time=epoch.times.values,
            reference_frame=frame.name,
            obsvr_bdy=self.instrument.name,
        )

        # Apply GS delta location from *state* (if present) by perturbing 3D relative pos
        gs_delta_awf = self._extract_gs_delta_location()  # returns ArrayWFrame or None
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
                    # gs_delta_location
                    if self.instrument.spice_id == comp_body.spice_id:
                        A_ei = SpiceManager.get_xfrm(
                            frame_from="IAU_EARTH",
                            frame_to="J2000",
                            epoch=epoch.times.values,
                        )
                        h_tilde_frame_from = self._compute_h_tilde_gs_delta_location(
                            relative_state
                        )
                        h_tilde_frame_to = h_tilde_frame_from @ A_ei
                        h_tilde.extend(h_tilde_frame_to)
                    else:
                        # not matching gs (Independent from measurement model - KEEP HERE)
                        h_tilde.extend([0.0, 0.0, 0.0])
                elif "rangerate_bias" in state_allowed and state_allowed[
                    "rangerate_bias"
                ][2].match(component):
                    _bias_dim = component_def[1]
                    if (
                        comp_body is not None
                        and self.instrument.spice_id == comp_body.spice_id
                    ):
                        if self._epoch_to_pass_index is not None:
                            # Stacked per-pass: indicator vector with 1.0 at active pass index.
                            _cur = self._epoch_to_pass_index.get(
                                float(epoch.times.values)
                            )
                            indicator = [0.0] * _bias_dim
                            if _cur is not None and _cur < _bias_dim:
                                indicator[_cur] = 1.0
                            h_tilde.extend(indicator)
                        else:
                            h_tilde.extend(
                                self._compute_h_tilde_range_bias()
                            )  # drhodot/db = 1
                    else:
                        # not matching gs
                        h_tilde.extend([0.0] * _bias_dim)
                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    h_tilde.extend(self._compute_h_tilde_dv_man(relative_state))
                else:  # If the component is not above, we add zeros by default.
                    h_tilde.extend([0.0 for _ in range(component_def[1])])
        return h_tilde
