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
    Noise,
    Instrument,
    StateDefinition,
)
import scarabaeus.utils.NumpyWrapper as np
import numpy.typing as npt

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
class RangeIdeal(Measurement):
    """ Models the ideal range measurement model.

    Generates ranging observables between an observer and a target.
    The observable is expressed in km and equals the Euclidean distance
    between spacecraft and ground station (plus noise, if added).

    Assuming no light-time delay and no media effects, the range is

    .. math::

        \\rho = \\|\\mathbf{r}_{sc}(t) - \\mathbf{r}_{gs}(t)\\|_2

    where :math:`\\mathbf{r}_{sc}` and :math:`\\mathbf{r}_{gs}` are
    the inertial position vectors of the spacecraft and ground station
    respectively.

    Parameters
    ----------
    name : str
        Name of the measurement model.
    instrument : Instrument
        Instrument object from scarabaeus. An 'antenna' instrument is used
        for radiometric measurement models.
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    meas_bias : float, optional
        Ground station range bias. Defaults to ``None``.
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
        antenna_sc = Antenna("Antenna_for_radiometric", spice_id = Orbiter_spice_id)
        Orbiter.addInstrument([antenna_sc])
        Doppler_transmit_frequency = ArrayWUnits(
            8.8 * 10**9, sec**-1
        )

        # Generate a Centroid measurement model
        range_sigma = ArrayWUnits(1e-3, km)

        # Generate a ground-station object
        GS1 = GroundStation("DSS-14")

        # Generate an ideal range measurement model
        Range_GS1 = RangeIdeal(
            name="GS1 Ideal Range Model", instrument=GS1, sigma=range_sigma
        )

        # Write observed measurements in .json format
        Range_GS1.write_observed_measurements(
            target=Orbiter,
            epoch_array=epoch_array_range,
            noisy=True,
            file_name="ideal_range",
        )

        # Read observed measurement in .json format
        obs_quantities_range = Range_GS1.observed_measurements(
            file_name="data/dwn_data/ideal_range.json", meas_name="meas_ideal", units=km
        )

        # Generate computed measurements on a trajectory
        computed_range_GS1 = Range_GS1.computed_measurements(
            target=Orbiter_perturbed,
            epoch_array=epoch_range_GS1_et,
            frame=J2000,
        )

        # Compute the _partials
        range_GS1_partials = Range_GS1.compute_partials(
            target=Orbiter_perturbed, epoch_array=epoch_range_GS1_et, frame=J2000
        )

        # Compute the residuals
        residuals_range_GS1 = Range_GS1.residuals(observed_range_GS1, computed_range_GS1)

        # Generate a measurement dataset
        range_dataset_GS1 = Range_GS1.generate_measurement_dataset(
            "GS1 Range",
            target=Orbiter_perturbed,
            observed_meas=obs_quantities_range,
        )
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
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument
        self._sigma = sigma
        self._meas_bias = meas_bias
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "RangeIdeal"
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
        Enable per-pass range biases.

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

    def _extract_range_bias(self) -> ArrayWUnits | None:
        """
        Return range bias for THIS station (global mode — first matching state entry).
        """
        if self._reference_state_vector is None:
            return self._meas_bias.quantity if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        if "range_bias" not in state_allowed:
            return None

        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame (even for scalar bias)

            if state_allowed["range_bias"][2].match(component) is None:
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
        Load range measurements from a .json file.

        pass_epochs : list[list[float]], optional
            Observation epochs grouped by tracking pass, e.g.
            ``[[t0, t1, t2], [t10, t11]]``.  Must be provided when the state
            vector contains a stacked ``range_bias_<suffix>`` entry (dim > 1).
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
        Compute the range measurement between an observer and a target.

        This function calculates the range measurement (distance) from the current spacecraft
        to a specified target spacecraft, taking into account optional parameters like
        specific epochs, time steps, reference frame, and noise settings.

        Parameters:
            target (Spacecraft) : The target spacecraft for which the range measurement is to be computed.
            epoch_array (EpochArray) : An array of epochs (times) at which the range measurements should be computed. If provided, it overrides epoch_start, epoch_end, and tstep.
            epoch_start (Epoch) : The starting epoch for the range measurement computations. Required if `epoch_array` is not provided.
            epoch_end (Epoch) : The ending epoch for the range measurement computations. Required if `epoch_array` is not provided.
            tstep (float) : The time step, in seconds, between consecutive range measurements. If `epoch_array` is not provided. Defaults to 1 second.
            frame (Frame) : The reference frame in which the range computation is performed.
            noisy (bool) : Whether to add noise to the computed range measurement. Defaults to False.
            antenna_name (str) : name of the antenna to be used for Doppler
            ransmit_frequency (ArrayWUnits) : transmit frequency, used to compute the doppler count
            receive_station (GroundStation) : receiving station object for three-way doppler, defaults to None

        Returns:
            ArrayWFrame: The computed range measurement as an array with frames.
        """
        if pass_epochs is not None:
            self.set_pass_epochs(pass_epochs)

        ## Input check
        if epoch_array is None:
            if epoch_start is None or epoch_end is None:
                raise RuntimeError(
                    "Please provide an EpochArray or provide start and end Epochs"
                )
            _units = epoch_start.times.units
            epoch_array = [
                EpochArray(ArrayWUnits(_t, _units), timeFrame=epoch_start.system)
                for _t in np.arange(
                    epoch_start.times.values, epoch_end.times.values + tstep, tstep
                )
            ]

        # Extract full bias vector once (dim=1 global, dim=n_passes stacked per-pass).
        _bias_full = self._extract_range_bias()
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        meas_list = []
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

            # Apply GS delta location if present: modify 3D relative position then take norm
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

            # Now compute range from (possibly corrected) relative position
            range_meas = relative_state[0:3].norm()

            # Apply bias
            if bias_awu is not None:
                range_meas = range_meas + bias_awu

            # Noise
            if noisy:
                range_meas += Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma.values,
                    units=self.sigma.units,
                )

            meas_list.append(range_meas)

        values_array = np.array([item.values for item in meas_list])
        units_list = [item.units for item in meas_list]
        units_names = [item.units.name for item in meas_list]
        if len(np.unique(units_names)) != 1:
            raise RuntimeError("multiple units are extracted from the measurements")

        meas_AWF = ArrayWFrame(values_array, units_list[0], frame)

        # Compatibility with real measurement models
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    ## Partial math by StateVector components

    # position
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

    # velocity
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

    # eta_srp
    def _compute_h_tilde_eta_srp(self) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
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
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
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
        h_tilde = [
            (-relative_state[0] / range_val).values,
            (-relative_state[1] / range_val).values,
            (-relative_state[2] / range_val).values,
        ]
        return np.array(h_tilde)

    # range_bias
    def _compute_h_tilde_range_bias(self) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
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
    def _compute_h_tilde_dv_man(self) -> list:
        """
        This method generate the portion in the h_tilde matrix relative to the partial of the ideal range measurement model
        with respect to the maneuver DV components.

        Returns:
            np.array: The (3,) vector the partial derivatives of the measurement model by the position maneuver DV components

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).

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

        # Apply GS delta location from *state* (if present) by perturbing 3D relative pos ---
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
                    if self.instrument.spice_id == comp_body.spice_id:
                        # matching gs
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

                # bias is meas_bias_ideal_* in your allowed dict
                elif "range_bias" in state_allowed and state_allowed["range_bias"][
                    2
                ].match(component):
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
                            )  # dρ/db = 1
                    else:
                        # not matching gs
                        h_tilde.extend([0.0] * _bias_dim)

                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    h_tilde.extend(self._compute_h_tilde_dv_man())
                else:  # If the component is not above, we add zeros by default.
                    h_tilde.extend([0.0 for _ in range(component_def[1])])

        return h_tilde
