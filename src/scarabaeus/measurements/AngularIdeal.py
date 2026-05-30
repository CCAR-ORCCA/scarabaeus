# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np

from scarabaeus import (
    ArrayWFrame,
    ArrayWUnits,
    EpochArray,
    Frame,
    Measurement,
    Noise,
    SpiceManager,
    StateArray,
    Units,
    StateDefinition,
)
from scarabaeus.body.Body import Body

# ------------------#
#  Generate Units  #
# ------------------#
km, sec, rad = Units.get_units(["km", "sec", "rad"])

# Generate common units and frames
spc_pos_units = km
spc_time_units = sec
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class AngularIdeal(Measurement):
    """ Models the ideal angular measurement model.

    Generates angular observables (right ascension and declination) between
    an observer and a target, expressed in radians.

    Given the line-of-sight unit vector
    :math:`\\hat{\\boldsymbol{\\ell}} = (\\ell_x, \\ell_y, \\ell_z)`, the
    observables are

    .. math::

        \\alpha = \\mathrm{atan2}(\\ell_y,\\,\\ell_x),
        \\qquad
        \\delta = \\arcsin(\\ell_z)

    No light-time or atmospheric refraction effects are modelled.

    Parameters
    ----------
    name : str
        Name of the measurement model.
    observer : Body
        Body object from scarabaeus. This is the "observer" in the angular measurement.
    sigma : ArrayWUnits, optional
        Measurement standard deviation. Defaults to ``None``.
    meas_bias : float, optional
        Angular bias. Defaults to ``None``.
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

        import numpy as np

        import scarabaeus as scb

        ...

        # Initialize the body object for the orbiter
        Orbiter_mass = scb.ArrayWUnits(1200.0, kg)
        Orbiter_area = scb.ArrayWUnits(1.2e-05, km**2)
        Orbiter_cr_srp = scb.ArrayWUnits(0, None)
        Orbiter = scb.Spacecraft(
            "Orbiter",
            -64,
            Orbiter_mass,
            Orbiter_area,
            Orbiter_cr_srp,
        )

        frame = ECLIPJ2000

        # (1) Initialize Beacon Bodies
        earth_beacon = scb.Body("Earth", spice_id=399)

        # (2) Initialize Measurement Model
        celnav_beacon_ideal = scb.AngularIdeal(
            "CelNav Beacon Model",
            Orbiter,
        )


        # define measurement epochs
        start_epoch = 1000000000.0
        stop_epoch = start_epoch + 1 * 86400
        measurement_epochs = scb.EpochArray(np.linspace(start_epoch, stop_epoch, 100))

        # take measurements of Earth beacon from orbiter
        meas = celnav_beacon_ideal.computed_measurements(
            earth_beacon, epoch_array=measurement_epochs, frame=frame
        )
        _partials = celnav_beacon_ideal.compute_partials(earth_beacon, measurement_epochs, frame)
        celnav_beacon_ideal.write_observed_measurements(
            target=earth_beacon,
            epoch_array=measurement_epochs,
            frame=frame,
            noisy=False,
            file_name="earth_beacon",
        )

    """

    # -------------#
    # Constructor #
    # -------------#
    def __init__(
        self,
        name: str,
        observer: Body,
        sigma: ArrayWUnits = None,
        meas_bias=None,
        state_definition=None,
        sequence_definition=None,
    ):
        super().__init__(name, observer)

        # Initialize attributes
        self._sigma = sigma
        self._meas_bias = meas_bias
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "AngularIdeal"
        self._reference_state_vector: StateArray | None = None
        self._epoch_to_image_index: dict | None = None

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

    def set_image_epochs(self, image_epochs: list | None):
        """
        Enable per-image angular biases.

        Parameters
        ----------
        image_epochs : list[float] | None
            Ordered list of observation epoch values, one per image.
            Image index equals the position in the list.
            Pass ``None`` to revert to global-bias mode.
        """
        if image_epochs is None:
            self._epoch_to_image_index = None
        else:
            self._epoch_to_image_index = {
                float(t): i for i, t in enumerate(image_epochs)
            }

    def set_image_index(self, mapping: dict | None):
        """Directly assign a pre-built ``{epoch_value: image_index}`` mapping (or None to disable)."""
        self._epoch_to_image_index = mapping

    def _extract_angular_bias(self) -> ArrayWUnits | None:
        """
        Return angular bias [b_ra, b_dec] for this observer (global mode — first match).
        Falls back to ``meas_bias`` when no reference state vector is set.
        """
        if self._reference_state_vector is None:
            return self._meas_bias.quantity if self._meas_bias is not None else None

        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        if "angular_bias" not in state_allowed:
            return None

        obs_sid = getattr(self.instrument, "spice_id", None)
        for state_entry in self._reference_state_vector.state:
            component = state_entry[0]
            body = state_entry[4]
            value = state_entry[5]

            if state_allowed["angular_bias"][2].match(component) is None:
                continue
            if getattr(body, "spice_id", None) == obs_sid:
                return value.quantity

        return None

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_ideal",
        units=None,
        frame: Frame = J2000,
        image_epochs: list | None = None,
    ):
        """
        Load angular measurements from a .json file.

        image_epochs : list[float], optional
            Ordered list of observation epoch values, one per image.
            Must be provided when the state vector contains a stacked
            ``angular_bias_<suffix>`` entry (dim > 2).
        """
        result = super().observed_measurements(file_name, meas_name, units, frame)
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)
        return result

    def computed_measurements(
        self,
        target: Body,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        image_epochs: list | None = None,
    ) -> ArrayWFrame:
        """
        Compute the angular measurement between an observer and a target.

        This function calculates the angular measurement (right ascension/declination) from the observer
        to a specified target body, taking into account optional parameters like
        specific epochs, time steps, reference frame, and noise settings.

        Parameters:
            target (Body) : The target spacecraft for which the angular measurement is to be computed.
            epoch_array (EpochArray) : An array of epochs (times) at which the angular measurements should be computed. If provided, it overrides epoch_start, epoch_end, and tstep.
            epoch_start (Epoch) : The starting epoch for the angular measurement computations. Required if `epoch_array` is not provided.
            epoch_end (Epoch) : The ending epoch for the angular measurement computations. Required if `epoch_array` is not provided.
            tstep (float) : The time step, in seconds, between consecutive angular measurements. If `epoch_array` is not provided. Defaults to 1 second.
            frame (Frame) : The reference frame in which the angular computation is performed.
            noisy (bool) : Whether to add noise to the computed angular measurement. Defaults to False.
            image_epochs (list[float], optional) : Ordered list of observation epoch values (one per image).
                Must be provided when the state vector contains a stacked ``angular_bias_<suffix>`` entry.

        Returns:
            ArrayWFrame: The computed angular measurement as an array with frames.
        """
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)

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

        # Extract full bias vector once (dim=2 global, dim=2*n_images stacked per-image).
        _bias_full = self._extract_angular_bias()

        ## Append measurements in a list at multiple epochs
        meas_list = []
        for epoch in epoch_array:
            # Determine active bias for this epoch
            if self._epoch_to_image_index is not None and _bias_full is not None:
                _iidx = self._epoch_to_image_index.get(float(epoch.times.values))
                _bvals = _bias_full.values.ravel()
                if _iidx is not None and 2 * _iidx + 1 < len(_bvals):
                    bias_awu = ArrayWUnits(
                        _bvals[2 * _iidx : 2 * (_iidx + 1)], _bias_full.units
                    )
                else:
                    bias_awu = None
            else:
                bias_awu = _bias_full

            relative_state = SpiceManager.get_state(
                trgt_bdy=target.name,
                epoch_time=epoch.times.values,
                reference_frame=frame.name,
                obsvr_bdy=self.instrument.name,
            )
            # Get right ascension and declination measurement as AWU
            ra_meas = ArrayWUnits(
                np.arctan2(relative_state[1].values, relative_state[0].values), rad
            )
            dec_meas = ArrayWUnits(
                np.arctan(
                    relative_state[2].values
                    / np.sqrt(
                        relative_state[0].values ** 2 + relative_state[1].values ** 2
                    ),
                ),
                rad,
            )

            # Apply bias [b_ra, b_dec]
            if bias_awu is not None:
                b = bias_awu.values.ravel()
                ra_meas = ra_meas + ArrayWUnits(b[0], bias_awu.units)
                dec_meas = dec_meas + ArrayWUnits(b[1], bias_awu.units)

            # Add noise to ideal angular measurement
            if noisy:
                ra_meas += Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma[0].values,
                    units=self.sigma[0].units,
                )
                dec_meas += Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma[1].values,
                    units=self.sigma[1].units,
                )

            meas = ra_meas.append(dec_meas)
            meas_list.append(meas)

        ## Pack output measurement as an AWF
        values_array = np.array([item.values for item in meas_list])
        units_list = [item.units for item in meas_list]
        units_names = [item.units.name for item in meas_list]
        if len(np.unique(units_names)) != 1:
            raise RuntimeError("multiple units are extracted from the measurements")

        meas_AWF = ArrayWFrame(values_array, units_list[0], frame)

        # For compatibility with real measurement models
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    ## Partial math by StateVector components

    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Generate the portion of the H-tilde matrix corresponding to the partial
        derivatives of the ideal angular measurement model with respect to the
        position components (RA, DEC w.r.t. x, y, z).

        Parameters
        ----------
        relative_state : ArrayWUnits
            Relative state between observer and target (position-only),
            expected shape (3,) with components [x, y, z].

        Returns
        -------
        np.ndarray
            2x3 array containing the partial derivatives:
            [ d(RA)/dx  d(RA)/dy  d(RA)/dz
              d(DEC)/dx d(DEC)/dy d(DEC)/dz ]

        References
        ----------
        Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). Statistical Orbit Determination.
        Elsevier Academic Press. ISBN 978-0-12-683630-1 (p. 161, eq. 4.2.6).
        """

        x = float(relative_state[0].values)
        y = float(relative_state[1].values)
        z = float(relative_state[2].values)

        rho2 = x * x + y * y
        r2 = rho2 + z * z

        eps = np.finfo(float).eps
        rho2_safe = rho2 if rho2 > eps else eps
        rho_safe = np.sqrt(rho2_safe)
        r2_safe = r2 if r2 > eps else eps

        h_tilde_ra = [
            y / rho2_safe,
            -x / rho2_safe,
            0.0,
        ]
        h_tilde_dec = [
            (x * z) / (r2_safe * rho_safe),
            (y * z) / (r2_safe * rho_safe),
            -rho_safe / r2_safe,
        ]

        return np.array([h_tilde_ra, h_tilde_dec], dtype=float)

    def _compute_h_tilde_vel(self) -> np.ndarray:
        """
        Partial of (RA, DEC) w.r.t. velocity — zero for instantaneous model.

        Returns
        -------
        np.ndarray
            2×3 zero matrix.
        """
        return np.zeros((2, 3), dtype=float)

    def _compute_h_tilde_pos_secondary(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Partial of (RA, DEC) w.r.t. the *observer* position (negative of target partial).
        """
        return -self._compute_h_tilde_pos(relative_state)

    def _compute_h_tilde_vel_secondary(self) -> np.ndarray:
        """
        Partial of (RA, DEC) w.r.t. observer velocity — always zero for the ideal model.

        Returns
        -------
        np.ndarray
            2×3 zero matrix.
        """
        return np.zeros((2, 3), dtype=float)

    def _compute_h_tilde_angular_bias(self) -> np.ndarray:
        """
        Partial of (RA, DEC) w.r.t. [b_ra, b_dec] — identity (global mode).

        Returns
        -------
        np.ndarray
            2×2 identity: [[1, 0], [0, 1]].
        """
        return np.eye(2, dtype=float)

    ## Compute H_tilde with partial values
    def _partials(
        self,
        target: Body,
        epoch: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Groups together the different components of measurement _partials in the
        global H-tilde.  Returns a 2-row list ``[row_ra, row_dec]``.

        Sign convention (same as CentroidingIdeal):
            relative_state = r_target − r_observer
            target   position   ->  +H   (dmeas/dr_target   = +H_pos)
            observer position   ->  −H   (dmeas/dr_observer  = −H_pos)

        Parameters
        ----------
        target : Body
        epoch : EpochArray
        frame : Frame

        Returns
        -------
        list
            ``[row_ra, row_dec]`` — each row is a list of floats of length
            equal to the total state dimension.
        """

        relative_state = SpiceManager.get_state(
            trgt_bdy=target.name,
            epoch_time=epoch.times.values,
            reference_frame=frame.name,
            obsvr_bdy=self.instrument.name,
        )

        H_pos = self._compute_h_tilde_pos(relative_state)  # (2, 3)
        H_vel = self._compute_h_tilde_vel()  # (2, 3) zeros

        state_allowed = StateDefinition._generate_allowed_state_dictionary()
        target_sid = getattr(target, "spice_id", None)
        observer_sid = getattr(self.instrument, "spice_id", None)

        # Legacy: no state_definition  -> assume target pos + vel only
        if self._state_definition is None:
            row_ra = list(H_pos[0]) + list(H_vel[0])
            row_dec = list(H_pos[1]) + list(H_vel[1])
            return [row_ra, row_dec]

        row_ra = []
        row_dec = []

        for comp_def in self._state_definition:
            comp_name = comp_def[0]
            comp_dim = comp_def[1]
            comp_body = comp_def[4] if len(comp_def) > 4 else None
            comp_body_sid = getattr(comp_body, "spice_id", None)

            # Position
            # target  -> +H,  observer  -> −H
            if state_allowed["position"][2].match(comp_name):
                if comp_body_sid == target_sid:
                    row_ra.extend(H_pos[0].tolist())
                    row_dec.extend(H_pos[1].tolist())
                elif comp_body_sid == observer_sid:
                    row_ra.extend((-H_pos[0]).tolist())
                    row_dec.extend((-H_pos[1]).tolist())
                else:
                    row_ra.extend([0.0] * comp_dim)
                    row_dec.extend([0.0] * comp_dim)

            #  Velocity
            elif state_allowed["velocity"][2].match(comp_name):
                row_ra.extend([0.0] * comp_dim)
                row_dec.extend([0.0] * comp_dim)

            #  Angular bias [b_ra, b_dec]
            # Global (dim=2): [[1,0],[0,1]]
            # Stacked per-image (dim=2*N): indicator at slot [2*img, 2*img+1]
            elif "angular_bias" in state_allowed and state_allowed["angular_bias"][
                2
            ].match(comp_name):
                if comp_body_sid == observer_sid:
                    if self._epoch_to_image_index is not None:
                        _cur = self._epoch_to_image_index.get(float(epoch.times.values))
                        ind_ra = [0.0] * comp_dim
                        ind_dec = [0.0] * comp_dim
                        if _cur is not None and 2 * _cur + 1 < comp_dim:
                            ind_ra[2 * _cur] = 1.0
                            ind_dec[2 * _cur + 1] = 1.0
                        row_ra.extend(ind_ra)
                        row_dec.extend(ind_dec)
                    else:
                        row_ra.extend([1.0, 0.0])  # global (dim=2)
                        row_dec.extend([0.0, 1.0])
                else:
                    row_ra.extend([0.0] * comp_dim)
                    row_dec.extend([0.0] * comp_dim)

            #  Unknown / unhandled components: zeros
            else:
                row_ra.extend([0.0] * comp_dim)
                row_dec.extend([0.0] * comp_dim)

        return [row_ra, row_dec]
