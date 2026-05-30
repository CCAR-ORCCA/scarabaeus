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
    Spacecraft,
    Instrument,
    GroundStation,
    Noise,
    SpiceManager,
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
    meas_bias : float, optional
        Measurement bias. Defaults to ``None``.
    state_definition : StateVector, optional
        State vector definition list. Defaults to ``None``.
    sequence_definition : list, optional
        Sequence definition list. Defaults to ``None``.
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
        state_definition: StateArray = None,
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
        self._reference_state_vector: StateArray | None = None

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
    def sigma(self) -> ArrayWUnits:
        """
        The standard deviation of the measurement model.
        """
        return self._sigma

    @property
    def meas_bias(self):
        """Constant range bias applied to all computed observables [s]."""
        return self._meas_bias

    @property
    def state_definition(self) -> dict:
        """
        The state vector definition
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> dict:
        """
        The sequence definition.
        """
        return self._sequence_definition

    # endregion => Properties #
    # -------------------------#

    # ---------#
    # Methods #
    # ---------#

    def _extract_dor_bias(self) -> ArrayWUnits | None:
        """
        Return range bias for THIS station, as stored in the state, if present.
        """
        if self._reference_state_vector is None:
            return self._meas_bias.quantity if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()

        # Prefer meas_bias_ideal_* if available in the allowed dictionary
        if "dor_bias" not in state_allowed:
            return None

        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]  # ArrayWFrame (even for scalar bias)

            if state_allowed["dor_bias"][2].match(component) is None:
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
    ) -> ArrayWFrame:
        """
        Computes the doppler measurement between ``target.name`` and ``self.instrument.name`` (2-way doppler).

        If a receiver_station is specified as an instrument (i.e. a ground station) 3-way doppler is computed instead.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft.

        epoch_array : EpochArray, optional
            An array of epochs (times) at which the range rate measurements should be computed.
            If provided, overrides ``epoch_start``, ``epoch_end``, and ``tstep``.

        epoch_start : EpochArray, optional
            The starting epoch for the range ratemeasurement computations.
            Required if ``epoch_array`` is not provided.

        epoch_end : EpochArray, optional
            The ending epoch for the range ratemeasurement computations.
            Required if ``epoch_array`` is not provided.

        tstep :float
            The time step, in seconds, between consecutive range ratemeasurements
            if ``epoch_array`` is not provided. Defaults to ``1`` second.

        frame : Frame, optional
            The reference frame in which the range rate computation is performed.
            Defaults to a J2000 Frame object.

        noisy : bool
            Indicates if noise is added to the computed range rate measurement. Defaults to ``False``.
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

        # Get AWU for speed-of-light constant
        c = constants.c

        # Pull current bias + gs delta from state (if provided)
        bias_awu = self._extract_dor_bias()  # ArrayWUnits or None
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        ## Append measurements in a list at multiple epochs
        meas_list = []
        for epoch in epoch_array:
            # NOTE: Missing handling of gs delta location and bias for the second station (if present).
            # This is currently only implemented for the first station, but it should be extended to both stations if needed.

            # Generate the relative state between first GS and target
            relative_state1 = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.instrument.name,
            )
            # Generate the relative state between second GS and target
            relative_state2 = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.station.name,
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
                relative_state1 = ArrayWUnits(
                    np.hstack(
                        [
                            relative_state1[0:3].values - delta_in_frame.values,
                            relative_state1[3:6].values,
                        ]
                    ),
                    relative_state1.units,
                )

            # Get range measurements as AWU
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
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
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
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
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
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
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
        Tapley, B. D.; Schutz, B. E.; Born, G. H. (2004).
        *Statistical Orbit Determination*.
        Elsevier Academic Press. ISBN 978-0-12-683630-1. Eq. 4.2.6.
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
        Tapley, B. D.; Schutz, B. E.; Born, G. H. (2004).
        *Statistical Orbit Determination*.
        Elsevier Academic Press. ISBN 978-0-12-683630-1. Eq. 4.2.6.
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
        Tapley, B. D.; Schutz, B. E.; Born, G. H. (2004).
        *Statistical Orbit Determination*.
        Elsevier Academic Press. ISBN 978-0-12-683630-1. Eq. 4.2.6.
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
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination. Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
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
        """
        # Pull current bias + gs delta from state (if provided)
        gs_delta_awf = self._extract_gs_delta_location()  # ArrayWFrame or None

        # Generate the relative state between first GS and target
        relative_state_1 = SpiceManager.get_state(
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
            relative_state_1 = ArrayWUnits(
                np.hstack(
                    [
                        relative_state_1[0:3].values - delta_in_frame.values,
                        relative_state_1[3:6].values,
                    ]
                ),
                relative_state_1.units,
            )

        # Generate the relative state between second GS and target
        relative_state_2 = SpiceManager.get_state(
            trgt_bdy=target.name,
            epoch_time=epoch.times.values,
            reference_frame=frame.name,
            obsvr_bdy=self.station.name,
        )

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
                elif state_allowed["gs_delta_location"][2].match(component):
                    raise NotImplementedError(
                        "DiffOneWayRangeIdeal: partials w.r.t. gs_delta_location have not "
                        "been verified. The analytic formulas are implemented in "
                        "_compute_h_tilde_gs1_delta_location / _compute_h_tilde_gs2_delta_location "
                        "but must be validated before use in a filter."
                    )
                    # --- unverified implementation (kept for reference) ---
                    # A_ei = SpiceManager.get_xfrm(
                    #     frame_from="IAU_EARTH", frame_to="J2000", epoch=epoch.times.values,
                    # )
                    # if self.instrument.spice_id == comp_body.spice_id:
                    #     # GS1: ddtau/dr_GS1 = +rhohat1/c
                    #     h_from = self._compute_h_tilde_gs1_delta_location(relative_state_1)
                    #     h_tilde.extend(h_from @ A_ei)
                    # elif self._station is not None and self._station.spice_id == comp_body.spice_id:
                    #     # GS2: ddtau/dr_GS2 = −rhohat2/c
                    #     h_from = self._compute_h_tilde_gs2_delta_location(relative_state_2)
                    #     h_tilde.extend(h_from @ A_ei)
                    # else:
                    #     h_tilde.extend([0.0, 0.0, 0.0])
                elif "dor_bias" in state_allowed and state_allowed["dor_bias"][2].match(
                    component
                ):
                    raise NotImplementedError(
                        "DiffOneWayRangeIdeal: partials w.r.t. dor_bias have not been "
                        "verified. The analytic formula is implemented in "
                        "_compute_h_tilde_dor_bias but must be validated before use in a filter."
                    )
                    # --- unverified implementation (kept for reference) ---
                    # if self.instrument.spice_id == comp_body.spice_id:
                    #     h_tilde.extend(self._compute_h_tilde_dor_bias())
                    # else:
                    #     h_tilde.extend([0.0])
                elif any(
                    state_allowed[key][2].match(component)
                    for key in state_allowed
                    if any(tag in key for tag in ["dv_man"])
                ):
                    # dv maneuver (dv_man, dv_man1, dv_man2)
                    h_tilde.extend(self._compute_h_tilde_dv_man())
                else:  # If the component is not above, we add zeros by default.
                    h_tilde.extend([0.0 for _ in range(component_def[1])])
        return h_tilde
