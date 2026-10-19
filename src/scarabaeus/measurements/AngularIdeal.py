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
    Units,
    StateDefinition,
)
from scarabaeus.body.Body import Body
from scarabaeus.spacecraft.Instrument import Instrument

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
    meas_bias : array_like | ArrayWUnits | ArrayWFrame, optional
        Angular bias ``(b_ra, b_dec)``; plain numbers are taken in rad. An
        ``angular_bias`` entry for this observer in the reference state
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

    # ------------------#
    # region   Errors  #
    # ------------------#
    # Custom exceptions: named on the model and raised on the
    # line that detects the failure, so tracebacks point at the real cause
    # rather than a generic Exception/RuntimeError.
    class MissingImageIndexError(Exception):
        """A stacked per-image angular bias has no epoch-to-image mapping.

        Raised by the forward model and the partials when the bias, from
        ``meas_bias`` or an ``angular_bias`` state entry, has dimension
        ``2 * n_images`` but :meth:`set_image_epochs` was never called, so
        there is no way to know which image slot an epoch belongs to.
        """

    # endregion Errors #
    # ------------------#

    # -------------#
    # Constructor #
    # -------------#
    def __init__(
        self,
        name: str,
        observer: Body | Instrument,
        sigma: ArrayWUnits = None,
        meas_bias=None,
        state_definition=None,
        sequence_definition=None,
        observed_target: Body = None,
    ):
        super().__init__(name, observer)

        # Initialize attributes
        self._sigma = sigma
        self._meas_bias = meas_bias
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "AngularIdeal"
        self._component_labels = ("ra", "dec")
        self._epoch_to_image_index: dict | None = None
        # Optical opNav: the observer is this model's ``instrument`` (the
        # spacecraft) and the body it images is ``observed_target`` (e.g. an
        # asteroid / beacon).  Filters build datasets with
        # ``target = propagator.primary_body`` (the estimated spacecraft, which
        # for an optical model is the *observer*), so when ``observed_target`` is
        # set we substitute it in :meth:`generate_measurement_dataset` -- without
        # it, observer == target and the line of sight collapses to NaN.
        self._observed_target = observed_target

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def _observer_designator(self) -> str:
        """
        SPICE designator for the observer.

        The observer may be given as the spacecraft itself or as one of its
        cameras — this model is :class:`~scarabaeus.CentroidingIdeal` with the
        attitude taken out of the loop, so the same instrument should be
        acceptable. A ``Camera``'s ``name`` is the instrument's and means
        nothing to SPICE, but its ``spice_id`` is the body it rides on, so the
        id is preferred wherever there is one. A ``Spacecraft`` observer
        resolves identically either way.
        """
        return self._spice_designator(self.instrument)

    @property
    def observed_target(self) -> Body:
        """Beacon body imaged by this optical model (``None`` if not configured)."""
        return self._observed_target

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def generate_measurement_dataset(
        self,
        dataset_name: str | None = None,
        target: Body = None,
        observed_meas=None,
        epochs: EpochArray | None = None,
        frame: Frame = J2000,
        noisy: bool = False,
    ):
        """
        Build the filter dataset for this optical measurement.

        Parameters
        ----------
        dataset_name : str, optional
            Name of the dataset, which becomes its residual key in
            ``solution.json``. Defaults to :attr:`default_dataset_name`.
        target : Body, optional
            Body the filter passes as the target, normally the estimated
            spacecraft. It is replaced by the configured ``observed_target`` when
            one is set. Defaults to ``None``.
        observed_meas : tuple or list, optional
            Observed measurements as ``(epoch_array, meas_sec, meas_obs,
            outlier_flag)``. Give this or ``epochs``, not both. Defaults to
            ``None``.
        epochs : EpochArray, optional
            Epochs to simulate the measurements at. Give this or
            ``observed_meas``, not both. Defaults to ``None``.
        frame : Frame, optional
            Reference frame of the measurements. Defaults to ``J2000``.
        noisy : bool, optional
            Whether noise is added to simulated measurements. Defaults to
            ``False``.

        Returns
        -------
        list of MeasurementDataSet
            The datasets built by :meth:`Measurement.generate_measurement_dataset`
            for the substituted target.

        Notes
        -----
        ``AngularIdeal`` is an optical model: the *observer* is this model's
        instrument (the spacecraft) and the *observed* body is a fixed beacon.
        Filters call this with ``target = propagator.primary_body`` (the estimated
        spacecraft), which is the right target for radiometric models, but for an
        optical model the estimated spacecraft is the observer. When an
        ``observed_target`` was configured it is therefore substituted for the
        passed ``target``\\; otherwise observer and target coincide and the line of
        sight degenerates to :math:`\\arctan(0 / \\sqrt{0})`, a NaN.

        The partials still map to the estimated spacecraft (the observer): with a
        ``state_definition`` supplied, :meth:`_partials` returns :math:`-H` for the
        observer position, so the optical track informs the spacecraft estimate.
        """
        if self._observed_target is not None:
            target = self._observed_target
        return super().generate_measurement_dataset(
            dataset_name,
            target,
            observed_meas=observed_meas,
            epochs=epochs,
            frame=frame,
            noisy=noisy,
        )

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
        """Set the epoch-to-image mapping directly.

        Parameters
        ----------
        mapping : dict or None
            ``{epoch_value: image_index}`` mapping used to pick the per-image slot
            of a stacked ``angular_bias``, or ``None`` to clear it. Use
            :meth:`set_image_epochs` to build it from an ordered list of epochs.
        """
        self._epoch_to_image_index = mapping

    def observed_measurements(
        self,
        file_name,
        meas_name: str = "meas_sim",
        units=None,
        frame: Frame = J2000,
        image_epochs: list | None = None,
    ):
        """
        Load angular measurements from a .json file.

        Parameters
        ----------
        file_name : str
            Name of the .json file holding the measurements.
        meas_name : str, optional
            Key of the measurement block to read. Defaults to ``'meas_sim'``.
        units : Units, optional
            Units to interpret the stored values in. Defaults to ``None``.
        frame : Frame, optional
            Reference frame the loaded measurements are expressed in. Defaults
            to ``J2000``.
        image_epochs : list, optional
            Ordered list of observation epoch values, one per image. Must be
            provided when the state vector contains a stacked
            ``angular_bias_<suffix>`` entry (dimension > 2). Defaults to ``None``.

        Returns
        -------
        ObsData
            The observation tuple returned by
            :meth:`Measurement.observed_measurements`.
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

        Calculates the angular measurement (right ascension / declination) from
        the observer to a specified target body, taking into account optional
        parameters like specific epochs, time steps, reference frame and noise
        settings.

        Parameters
        ----------
        target : Body
            The target body the angular measurement is computed to.
        epoch_array : EpochArray, optional
            Epochs at which the angular measurements are computed. Overrides
            *epoch_start*, *epoch_end* and *tstep* when given. Defaults to
            ``None``.
        epoch_start : EpochArray, optional
            Starting epoch for the angular measurement computations. Required
            if *epoch_array* is not provided. Defaults to ``None``.
        epoch_end : EpochArray, optional
            Ending epoch for the angular measurement computations. Required if
            *epoch_array* is not provided. Defaults to ``None``.
        tstep : float, optional
            Time step, in seconds, between consecutive angular measurements
            when *epoch_array* is not provided. Defaults to ``1`` second.
        frame : Frame, optional
            Reference frame in which the angular computation is performed.
            Defaults to ``J2000``.
        noisy : bool, optional
            Whether to add noise to the computed angular measurement. Defaults
            to ``False``.
        image_epochs : list, optional
            Ordered list of observation epoch values, one per image. Must be
            provided when the state vector contains a stacked
            ``angular_bias_<suffix>`` entry. Defaults to ``None``.

        Returns
        -------
        tuple
            ``(t1, t2, t3, meas)``, per the contract in
            :meth:`Measurement.computed_measurements`.

            This observable is **single-epoch** -- unlike a two-way radiometric
            measurement it is not spread across a light-time path with distinct
            transmit, reflect and receive times -- so ``t1``, ``t2`` and ``t3``
            are all the observation epoch. They are returned, rather than a bare
            measurement, because the estimator reads ``t1`` when detrending
            residuals and matches measurements on ``t2``; collapsing them here
            is a correct statement about this observable, not padding.

            ``meas`` is the computed angular measurement as an
            :class:`~scarabaeus.ArrayWFrame`.

        Raises
        ------
        ValueError
            If neither *epoch_array* nor both *epoch_start* and *epoch_end* are
            provided.
        RuntimeError
            If the per-epoch measurements do not share one unit.
        MissingImageIndexError
            If the bias holds one ``(b_ra, b_dec)`` pair per image (more
            than two elements) and no epoch-to-image mapping is set.
        """
        if image_epochs is not None:
            self.set_image_epochs(image_epochs)

        ## Input check
        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        # Extract full bias vector once (dim=2 global, dim=2*n_images stacked per-image).
        _bias_full = self._extract_bias("angular_bias", rad)
        if (
            _bias_full is not None
            and _bias_full.values.size > 2
            and self._epoch_to_image_index is None
        ):
            raise AngularIdeal.MissingImageIndexError(
                f"The angular bias has {_bias_full.values.size} elements, so it "
                "holds one (b_ra, b_dec) bias per image, but no epoch-to-image "
                "mapping is set and the bias of each epoch is undefined. Call "
                "set_image_epochs() (or pass image_epochs=) first."
            )

        ## Append measurements in a list at multiple epochs
        meas_list = []
        for epoch in epoch_array:
            # Determine active bias for this epoch
            # A single (b_ra, b_dec) pair is global: it applies to every image.
            if (
                self._epoch_to_image_index is not None
                and _bias_full is not None
                and _bias_full.values.size > 2
            ):
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

            relative_state = self._relative_state(target, epoch, frame)
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

        ## Pack output measurement as an AWF.
        # Ravel each epoch's [RA, Dec] to a flat (n_components,) row so the stack
        # is exactly (n_epochs, n_components). Without this, when SpiceManager
        # returns length-1-array components the per-epoch values come out (2, 1)
        # and the stack becomes 3-D (n_epochs, 2, 1); generate_measurement_dataset
        # then splits it into datasets whose `computed` stays 2-D while `observed`
        # is scalar, which later blows up the combined block.
        values_array = np.array([np.ravel(item.values) for item in meas_list])
        units_list = [item.units for item in meas_list]
        units_names = [item.units.name for item in meas_list]
        if len(np.unique(units_names)) != 1:
            raise RuntimeError("multiple units are extracted from the measurements")

        meas_AWF = ArrayWFrame(values_array, units_list[0], frame)

        # Single-epoch observable: there is no light-time path with distinct
        # transmit / reflect / receive times to distinguish, so all three
        # collapse to the observation epoch. Returned because the estimator
        # detrends on t1 and matches measurements on t2 -- see
        # Measurement.computed_measurements for the contract.
        t1 = epoch_array
        t2 = epoch_array
        t3 = epoch_array

        return t1, t2, t3, meas_AWF

    ## Partial math by StateVector components

    def _compute_h_tilde_pos(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Partials of the ideal right ascension and declination with respect to
        position.

        The derivatives are taken with respect to the relative position
        :math:`[x, y, z] = \\mathbf{r}_{target} - \\mathbf{r}_{observer}`, so they
        are also the partials with respect to the target position\\; the observer
        position enters with the opposite sign (see
        :meth:`_compute_h_tilde_pos_secondary`). With
        :math:`\\alpha = \\operatorname{atan2}(y, x)`,
        :math:`\\delta = \\arctan(z / \\rho)`, :math:`\\rho^2 = x^2 + y^2` and
        :math:`r^2 = \\rho^2 + z^2`, eq. 4.2.6 of [1]_ gives

        .. math::

            \\frac{\\partial \\alpha}{\\partial [x, y, z]} =
                \\left[ -\\frac{y}{\\rho^2},\\quad \\frac{x}{\\rho^2},\\quad 0 \\right]

            \\frac{\\partial \\delta}{\\partial [x, y, z]} =
                \\left[ -\\frac{x z}{r^2 \\rho},\\quad -\\frac{y z}{r^2 \\rho},\\quad
                \\frac{\\rho}{r^2} \\right]

        Parameters
        ----------
        relative_state : ArrayWUnits
            Position of the target relative to the observer, shape (3,), with
            components :math:`[x, y, z]`.

        Returns
        -------
        np.ndarray
            Shape (2, 3): row 0 is :math:`\\partial \\alpha / \\partial [x, y, z]`
            and row 1 is :math:`\\partial \\delta / \\partial [x, y, z]`.

        References
        ----------
        .. [1] Tapley, B. D., Schutz, B. E., & Born, G. H. (2004). *Statistical
           Orbit Determination*. Elsevier Academic Press. ISBN
           978-0-12-683630-1, p. 161, eq. 4.2.6.
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

        # d(RA)/dr and d(DEC)/dr with respect to the TARGET position, with
        # r = r_target - r_observer. RA = atan2(y, x) gives dRA/dx = -y/rho^2,
        # and DEC = atan(z/rho) gives dDEC/dz = +rho/r^2. The observer partial
        # is the negation, supplied by _compute_h_tilde_pos_secondary.
        h_tilde_ra = [
            -y / rho2_safe,
            x / rho2_safe,
            0.0,
        ]
        h_tilde_dec = [
            -(x * z) / (r2_safe * rho_safe),
            -(y * z) / (r2_safe * rho_safe),
            rho_safe / r2_safe,
        ]

        return np.array([h_tilde_ra, h_tilde_dec], dtype=float)

    def _compute_h_tilde_vel(self) -> np.ndarray:
        """
        Partials of right ascension and declination with respect to the target
        velocity.

        Returns
        -------
        np.ndarray
            Zeros of shape (2, 3); the ideal angles are instantaneous and depend on
            position only.
        """
        return np.zeros((2, 3), dtype=float)

    def _compute_h_tilde_pos_secondary(self, relative_state: ArrayWUnits) -> np.ndarray:
        """
        Partials of right ascension and declination with respect to the observer
        position.

        Parameters
        ----------
        relative_state : ArrayWUnits
            Position of the target relative to the observer, shape (3,).

        Returns
        -------
        np.ndarray
            Shape (2, 3), the negative of :meth:`_compute_h_tilde_pos`, because the
            observer position enters the relative position with a minus sign.
        """
        return -self._compute_h_tilde_pos(relative_state)

    def _compute_h_tilde_vel_secondary(self) -> np.ndarray:
        """
        Partials of right ascension and declination with respect to the observer
        velocity.

        Returns
        -------
        np.ndarray
            Zeros of shape (2, 3)\\; the ideal angles depend on position only.
        """
        return np.zeros((2, 3), dtype=float)

    def _compute_h_tilde_angular_bias(self) -> np.ndarray:
        """
        Partials of right ascension and declination with respect to a global
        ``[b_ra, b_dec]`` bias.

        Returns
        -------
        np.ndarray
            The (2, 2) identity, since each bias adds directly to its angle.
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
        Assemble the right-ascension and declination rows of the global H-tilde.

        Parameters
        ----------
        target : Body
            The body the line of sight points to.
        epoch : EpochArray
            Single epoch at which the partials are evaluated.
        frame : Frame, optional
            Reference frame in which the partials are expressed. Defaults to
            ``J2000``.

        Returns
        -------
        list
            ``[row_ra, row_dec]``, each a list of floats as long as the state.
            Without a ``state_definition`` each row holds the target position and
            velocity partials only.

        Raises
        ------
        MissingImageIndexError
            If the state holds a stacked per-image ``angular_bias``
            (dimension > 2) and no epoch-to-image mapping is set.

        Notes
        -----
        Sign convention, the same as :class:`~scarabaeus.CentroidingIdeal`: the
        relative position is :math:`\\mathbf{r}_{target} - \\mathbf{r}_{observer}`,
        so the target position enters with :math:`+H_{pos}` and the observer
        position with :math:`-H_{pos}`. Velocity columns are zero. An
        ``angular_bias`` of the observer gets the (2, 2) identity when it is
        global, or ones in the two columns of the current image when it is stacked
        per image. Every other component gets zeros.
        """

        relative_state = self._relative_state(target, epoch, frame)

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
                    if self._epoch_to_image_index is not None and comp_dim > 2:
                        _cur = self._epoch_to_image_index.get(float(epoch.times.values))
                        ind_ra = [0.0] * comp_dim
                        ind_dec = [0.0] * comp_dim
                        if _cur is not None and 2 * _cur + 1 < comp_dim:
                            ind_ra[2 * _cur] = 1.0
                            ind_dec[2 * _cur + 1] = 1.0
                        row_ra.extend(ind_ra)
                        row_dec.extend(ind_dec)
                    elif comp_dim > 2:
                        raise AngularIdeal.MissingImageIndexError(
                            f"State component '{comp_name}' has dimension {comp_dim}, so "
                            "it holds one (b_ra, b_dec) bias per image, but no "
                            "epoch-to-image mapping is set and the partial for this "
                            "epoch is undefined. Call set_image_epochs() (or pass "
                            "image_epochs= to observed_measurements/"
                            "computed_measurements) before computing partials."
                        )
                    else:
                        row_ra.extend([1.0, 0.0])  # global (dim=2)
                        row_dec.extend([0.0, 1.0])
                else:
                    row_ra.extend([0.0] * comp_dim)
                    row_dec.extend([0.0] * comp_dim)

            #  Station geometry, when this model observes from a surface
            #  station: RA/DEC is a topocentric direction, so moving the
            #  station moves it.
            elif self._is_station_parameter(comp_name):
                block = self._compute_h_tilde_station(
                    comp_def, epoch.times.values, frame, H_pos, H_vel)
                row_ra.extend(block[0].tolist())
                row_dec.extend(block[1].tolist())

            #  Unknown / unhandled components: zeros
            else:
                row_ra.extend([0.0] * comp_dim)
                row_dec.extend([0.0] * comp_dim)

        return [row_ra, row_dec]
