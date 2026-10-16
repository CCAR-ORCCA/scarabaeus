# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    ArrayWFrame,
    ArrayWUnits,
    CustomParameter,
    EpochArray,
    Frame,
    Instrument,
    Measurement,
    Noise,
    SpiceManager,
    StateDefinition,
)
from scarabaeus.utils.CustomCallable import CustomCallable

import numpy as np

J2000 = Frame("J2000")

# Names a user callable may declare. Rejected at construction otherwise, so a
# typo surfaces immediately instead of during a partials sweep.
_CONTEXT_NAMES = (
    "position",
    "velocity",
    "relative_state",
    "epoch",
    "params",
    "target",
    "instrument",
    "frame",
    "model",
)


class CustomMeasurement(Measurement):
    """ User-supplied measurement model, defined entirely from a script.

    ``CustomMeasurement`` is the observation-side counterpart of
    :class:`~scarabaeus.CustomDynamicModel`: it turns a plain Python function
    into a first-class Scarabaeus measurement, so a new observable — and any
    biases or calibration terms it carries — can be added and estimated without
    changing the library source.

    The model computes

    .. math::

        y = h(\\mathbf{r}_\\text{rel},\\, \\dot{\\mathbf{r}}_\\text{rel},\\,
              \\mathbf{p},\\, t)

    where :math:`\\mathbf{r}_\\text{rel}` is the target position relative to the
    observing instrument and :math:`\\mathbf{p}` are the declared
    :class:`~scarabaeus.CustomParameter` objects.  It supplies the corresponding
    row(s) of the measurement Jacobian

    .. math::

        \\tilde{H} = \\left[
            \\frac{\\partial h}{\\partial \\mathbf r},\\;
            \\frac{\\partial h}{\\partial \\dot{\\mathbf r}},\\;
            \\frac{\\partial h}{\\partial \\mathbf p}
        \\right]

    with a zero block for every state component the model does not depend on.
    Any Jacobian not given analytically is evaluated by central finite
    differences of the user function, so a working model needs only *observable*.

    **Sign convention.** The relative state is
    ``r_target - r_instrument``. The Jacobian is therefore ``+dh/dr`` for the
    target's position column and ``-dh/dr`` for the observer's, matching the
    built-in models.

    **Calling convention.** Every user callable is invoked with keyword
    arguments; declare only the ones you need, in any order:

    ==================  ======================================================
    ``position``        ``(3,)`` target position relative to the instrument, km
    ``velocity``        ``(3,)`` relative velocity, km/s
    ``relative_state``  ``(6,)`` the two stacked
    ``epoch``           measurement epoch, TDB seconds past J2000
    ``params``          ``dict`` of this model's parameters at their current
                        estimated values (or their defaults)
    ``target``          the observed body
    ``instrument``      the observing instrument
    ``frame``           reference :class:`~scarabaeus.Frame`
    ``model``           this ``CustomMeasurement`` instance
    ==================  ======================================================

    A callable declaring ``**kwargs`` receives all of them.

    Parameters
    ----------
    name : str
        Name of the measurement model.
    instrument : Instrument
        Observing instrument (ground station, camera, ...).
    observable : callable
        Returns the computed measurement: a scalar, or a ``(dimension,)`` array
        for a multi-component observable. Required.
    sigma : ArrayWUnits
        Measurement noise standard deviation, also used to generate noise.
    units : Units, optional
        Units of the observable. Defaults to the units of *sigma*.
    dimension : int, optional
        Number of components the observable returns. Defaults to 1.
    parameters : list, optional
        Solve-for parameters this measurement depends on, as
        :class:`~scarabaeus.CustomParameter` objects or plain names. Their
        ``partial`` callable, when supplied, is
        :math:`\\partial h / \\partial p` with shape ``(dimension, p_dim)``.
    partial_by_position : callable, optional
        :math:`\\partial h / \\partial \\mathbf r` as ``(dimension, 3)``.
        Defaults to a central finite difference of *observable*.
    partial_by_velocity : callable, optional
        :math:`\\partial h / \\partial \\dot{\\mathbf r}` as ``(dimension, 3)``.
        Defaults to a central finite difference of *observable*.
    depends_on_position, depends_on_velocity : bool, optional
        Whether the observable depends on relative position / velocity.
        Inferred from the signature of *observable* when ``None``.
    position_step : float, optional
        Absolute finite-difference step on position, km. Defaults to
        ``1e-6 * max(|r_i|, 1)`` per component.
    velocity_step : float, optional
        Absolute finite-difference step on velocity, km/s. Defaults to
        ``1e-9 * max(|v_i|, 1)`` per component.
    state_definition : list, optional
        State layout used to build the Jacobian columns. Normally supplied by
        :meth:`~scarabaeus.Measurement.update_reference_state`.
    sequence_definition : MissionSequence, optional
        Sequence whose per-leg state definitions are used instead.

    See Also
    --------
    scarabaeus.CustomParameter : Declaration of an estimable custom parameter.
    scarabaeus.CustomDynamicModel : The dynamics-side counterpart.
    scarabaeus.RangeIdeal : Built-in model this one mirrors structurally.

    Notes
    -----
    Positions and velocities are handed over in km and km/s. The observable is
    returned in *units* (or the units of *sigma*).

    Examples
    --------
    Range with an estimated station bias, written from a script:

    .. code-block:: python

        import numpy as np
        import scarabaeus as scb

        km = scb.Units.get_units("km")
        bias = scb.CustomParameter("station_bias", units=km, default=0.0)

        def biased_range(position, params):
            return np.linalg.norm(position) + params["station_bias"][0]

        model = scb.CustomMeasurement(
            "GS1 Custom Range",
            instrument = gs,
            observable = biased_range,
            sigma      = scb.ArrayWUnits(1e-3, km),
            parameters = [bias],
        )

    The bias is estimated by adding it to the state definition:

    .. code-block:: python

        state_def = (scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .custom_param("station_bias", gs, scb.ArrayWFrame(0.0, km, None))
        )
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ShapeError(ValueError):
        """Raised when a user callable returns an array of unexpected shape."""

    class DuplicateParameterError(ValueError):
        """Raised when the same parameter name is declared twice by one model."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        observable,
        sigma: ArrayWUnits = None,
        units=None,
        dimension: int = 1,
        parameters: list | None = None,
        partial_by_position=None,
        partial_by_velocity=None,
        depends_on_position: bool | None = None,
        depends_on_velocity: bool | None = None,
        position_step: float | None = None,
        velocity_step: float | None = None,
        state_definition=None,
        sequence_definition=None,
    ):
        super().__init__(name, instrument)

        self._name = name
        self._instrument = instrument
        self._sigma = sigma
        self._state_definition = state_definition
        self._sequence_definition = sequence_definition
        self._measurement_type = "CustomMeasurement"

        self._dimension = int(dimension)
        self._position_step = position_step
        self._velocity_step = velocity_step

        if units is not None:
            self._units = units
        elif sigma is not None:
            self._units = sigma.units
        else:
            raise self.MissingSigmaError(
                f"Custom measurement '{name}' needs either 'units' or a 'sigma' "
                "carrying them, so the computed measurement can be returned "
                "with units."
            )

        # Declaration order fixes the column order of d(h)/d(parameters).
        self._parameters: dict[str, CustomParameter] = {}
        for entry in parameters or []:
            parameter = (
                entry
                if isinstance(entry, CustomParameter)
                else CustomParameter(str(entry))
            )
            if parameter.name in self._parameters:
                raise self.DuplicateParameterError(
                    f"Custom measurement '{name}' declares parameter "
                    f"'{parameter.name}' more than once."
                )
            self._parameters[parameter.name] = parameter

        # Bind the user callables once; a bad argument name fails here.
        self._observable = CustomCallable(
            observable, _CONTEXT_NAMES, "observable", name
        )
        self._partial_by_position = (
            None
            if partial_by_position is None
            else CustomCallable(
                partial_by_position, _CONTEXT_NAMES, "partial_by_position", name
            )
        )
        self._partial_by_velocity = (
            None
            if partial_by_velocity is None
            else CustomCallable(
                partial_by_velocity, _CONTEXT_NAMES, "partial_by_velocity", name
            )
        )
        self._parameter_partials = {
            key: CustomCallable(
                parameter.partial, _CONTEXT_NAMES, f"partial of '{key}'", name
            )
            for key, parameter in self._parameters.items()
            if parameter.partial is not None
        }

        # Structural dependence: decides whether a Jacobian block exists at all,
        # so it must not depend on the epoch or on any value.
        self._depends_on_position = (
            self._observable.reads("position") or self._observable.reads("relative_state")
            if depends_on_position is None
            else bool(depends_on_position)
        )
        self._depends_on_velocity = (
            self._observable.reads("velocity") or self._observable.reads("relative_state")
            if depends_on_velocity is None
            else bool(depends_on_velocity)
        )
        if partial_by_position is not None:
            self._depends_on_position = True
        if partial_by_velocity is not None:
            self._depends_on_velocity = True

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def units(self):
        """Units of the computed measurement."""
        return self._units

    @property
    def dimension(self) -> int:
        """Number of components the observable returns."""
        return self._dimension

    @property
    def parameters(self) -> dict:
        """Declared solve-for parameters, keyed by name."""
        return self._parameters

    @property
    def depends_on_position(self) -> bool:
        """``True`` when the observable is a function of relative position."""
        return self._depends_on_position

    @property
    def depends_on_velocity(self) -> bool:
        """``True`` when the observable is a function of relative velocity."""
        return self._depends_on_velocity

    # endregion Properties #
    # ----------------------#

    # ------------------------#
    # region Context / calls #
    # ------------------------#
    def _relative_state(self, target, epoch, frame) -> np.ndarray:
        """
        Return the target state relative to the instrument.

        Parameters
        ----------
        target : Body
            Observed body.
        epoch : EpochArray
            Measurement epoch.
        frame : Frame
            Reference frame of the returned state.

        Returns
        -------
        numpy.ndarray
            ``(6,)`` relative state ``r_target - r_instrument`` in km and km/s.

        Notes
        -----
        Delegates to :meth:`~scarabaeus.Measurement._relative_state`, so an
        instrument that is a ground station picks up its estimated location and
        orientation corrections, its solid tide, and table coordinates with no
        station SPK. Only the return type differs: a bare array, which is what
        the user-supplied observable and its partials are written against.
        """
        return np.asarray(Measurement._relative_state(
            self, target, epoch, frame).values, dtype=float).ravel()

    def _resolve_parameters(self, target) -> dict:
        """
        Read this model's parameters out of the reference state vector.

        A measurement parameter is normally owned by the instrument (a station
        bias) but may be owned by the target; entries owned by either are
        accepted, the instrument taking precedence. Parameters absent from the
        state fall back to their declared default.

        Parameters
        ----------
        target : Body
            Observed body, used to match target-owned parameters.

        Returns
        -------
        dict
            Mapping of parameter name to its current value.
        """
        values = {key: parameter.default for key, parameter in self._parameters.items()}

        if self._reference_state_vector is None:
            return values

        instrument_sid = getattr(self.instrument, "spice_id", None)
        target_sid = getattr(target, "spice_id", None)
        matched_instrument = set()

        for entry in self._reference_state_vector.state:
            if len(entry) < 6:
                continue

            component, _, _, _, owner, raw_value = entry
            if component not in self._parameters:
                continue

            owner_sid = getattr(owner, "spice_id", None)
            if owner_sid not in (instrument_sid, target_sid):
                continue
            if component in matched_instrument:
                continue

            raw = getattr(raw_value, "quantity", raw_value)
            values[component] = np.atleast_1d(
                np.asarray(getattr(raw, "values", raw), dtype=float)
            ).ravel()

            if owner_sid == instrument_sid:
                matched_instrument.add(component)

        return values

    def _build_context(self, relative_state, epoch, target, frame) -> dict:
        """
        Assemble the keyword-argument context handed to the user callables.

        Parameters
        ----------
        relative_state : numpy.ndarray
            ``(6,)`` relative state.
        epoch : EpochArray
            Measurement epoch.
        target : Body
            Observed body.
        frame : Frame
            Reference frame.

        Returns
        -------
        dict
            Mapping of every context name to its current value.
        """
        return {
            "position": relative_state[0:3],
            "velocity": relative_state[3:6],
            "relative_state": relative_state,
            "epoch": float(np.atleast_1d(epoch.times.values)[0]),
            "params": self._resolve_parameters(target),
            "target": target,
            "instrument": self.instrument,
            "frame": frame,
            "model": self,
        }

    def _as_vector(self, values, role: str) -> np.ndarray:
        """
        Coerce a user return value to a 1-D array of length :attr:`dimension`.

        Parameters
        ----------
        values : array_like or ArrayWUnits
            Raw user output.
        role : str
            Description used in error messages.

        Returns
        -------
        numpy.ndarray
            Flattened copy of *values*.

        Raises
        ------
        CustomMeasurement.ShapeError
            If the value does not hold exactly :attr:`dimension` elements.
        """
        array = np.atleast_1d(
            np.asarray(getattr(values, "values", values), dtype=float)
        ).ravel()

        if array.size != self._dimension:
            raise self.ShapeError(
                f"The {role} of custom measurement '{self._name}' returned "
                f"{array.size} element(s); expected {self._dimension}."
            )

        return array

    def _as_matrix(self, values, columns: int, role: str) -> np.ndarray:
        """
        Coerce a user return value to a ``(dimension, columns)`` array.

        Parameters
        ----------
        values : array_like or ArrayWUnits
            Raw user output.
        columns : int
            Expected number of columns.
        role : str
            Description used in error messages.

        Returns
        -------
        numpy.ndarray
            Reshaped copy of *values*.

        Raises
        ------
        CustomMeasurement.ShapeError
            If the value cannot be reshaped.
        """
        array = np.asarray(getattr(values, "values", values), dtype=float)

        if array.size != self._dimension * columns:
            raise self.ShapeError(
                f"The {role} of custom measurement '{self._name}' returned shape "
                f"{array.shape}; expected ({self._dimension}, {columns})."
            )

        return array.reshape(self._dimension, columns)

    def _evaluator(self, context: dict, key: str):
        """
        Build an evaluator of the observable over one perturbed context slot.

        Parameters
        ----------
        context : dict
            Evaluation context; temporarily mutated by the returned function.
        key : str
            ``"position"``, ``"velocity"``, or a declared parameter name.

        Returns
        -------
        callable
            Maps a perturbed value to the ``(dimension,)`` observable.
        """
        if key in ("position", "velocity"):
            offset = 0 if key == "position" else 3

            def evaluate(perturbed):
                original = context[key]
                original_state = context["relative_state"]
                combined = original_state.copy()
                combined[offset : offset + 3] = perturbed
                context[key] = perturbed
                context["relative_state"] = combined
                try:
                    return self._as_vector(self._observable(context), "observable")
                finally:
                    context[key] = original
                    context["relative_state"] = original_state

            return evaluate

        params = context["params"]

        def evaluate(perturbed):
            original = params[key]
            params[key] = perturbed
            try:
                return self._as_vector(self._observable(context), "observable")
            finally:
                params[key] = original

        return evaluate

    # endregion Context / calls #
    # ---------------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def computed_measurements(
        self,
        target,
        epoch_array: EpochArray = None,
        epoch_start: EpochArray = None,
        epoch_end: EpochArray = None,
        tstep: float = 1,
        frame: Frame = J2000,
        noisy: bool = False,
        pass_epochs: list[list] | None = None,
    ):
        """
        Evaluate the user observable over an epoch array.

        Parameters
        ----------
        target : Body
            The observed body.
        epoch_array : EpochArray, optional
            Epochs at which to evaluate. Overrides the start/end/step form.
        epoch_start, epoch_end : EpochArray, optional
            Bounds used when *epoch_array* is not given.
        tstep : float, optional
            Step in seconds for the start/end form. Defaults to 1.
        frame : Frame, optional
            Reference frame of the relative state. Defaults to J2000.
        noisy : bool, optional
            Add Gaussian noise of standard deviation :attr:`sigma`.
        pass_epochs : list of list, optional
            Pass grouping, forwarded to :meth:`set_pass_epochs`.

        Returns
        -------
        tuple
            ``(t1, t2, t3, measurements)`` — the three time tags (identical for
            an ideal model) and the computed measurements as an
            :class:`~scarabaeus.ArrayWFrame`.
        """
        if pass_epochs is not None:
            self.set_pass_epochs(pass_epochs)

        epoch_array = Measurement._resolve_epoch_array(
            epoch_array, epoch_start, epoch_end, tstep
        )

        values = []
        for epoch in epoch_array:
            relative_state = self._relative_state(target, epoch, frame)
            context = self._build_context(relative_state, epoch, target, frame)
            measurement = self._as_vector(self._observable(context), "observable")

            if noisy:
                measurement = measurement + Noise().generate_AWGN_with_units(
                    mu=0.0,
                    sigma=self.sigma.values,
                    units=self.sigma.units,
                ).values

            values.append(measurement)

        stacked = np.array(values)
        if self._dimension == 1:
            stacked = stacked.reshape(-1)

        measurements = ArrayWFrame(stacked, self._units, frame)

        # Ideal model: the three time tags coincide.
        return epoch_array, epoch_array, epoch_array, measurements

    def _partials(
        self,
        target,
        epoch: EpochArray,
        frame: Frame = J2000,
    ) -> list:
        """
        Build the measurement Jacobian row(s) for one epoch.

        One column block is emitted per state component, in the order of the
        state definition: the target's position/velocity get ``+dh/dr`` and
        ``+dh/dv``, the instrument's get the negated blocks, this model's own
        parameters get ``dh/dp``, and everything else gets zeros.

        Parameters
        ----------
        target : Body
            The observed body.
        epoch : EpochArray
            Epoch at which the Jacobian is evaluated.
        frame : Frame, optional
            Reference frame of the relative state. Defaults to J2000.

        Returns
        -------
        list
            A flat list for a scalar observable, or one row per component for a
            multi-component observable.
        """
        relative_state = self._relative_state(target, epoch, frame)
        context = self._build_context(relative_state, epoch, target, frame)

        partial_position = self._compute_partial_by_position(context)
        partial_velocity = self._compute_partial_by_velocity(context)

        target_sid = getattr(target, "spice_id", None)
        instrument_sid = getattr(self.instrument, "spice_id", None)

        rows = [[] for _ in range(self._dimension)]

        def emit(block):
            """Append one column block (dimension x n) to every row."""
            for index in range(self._dimension):
                rows[index].extend(np.asarray(block)[index].tolist())

        if self._state_definition is None:
            emit(partial_position)
            emit(partial_velocity)
        else:
            state_allowed = StateDefinition._generate_allowed_state_dictionary()

            for component_def in self._state_definition:
                component = component_def[0]
                component_dim = component_def[1]
                owner = component_def[4] if len(component_def) > 4 else None
                owner_sid = getattr(owner, "spice_id", None)

                if state_allowed["position"][2].match(component):
                    if owner_sid == target_sid:
                        emit(partial_position)
                    elif owner_sid == instrument_sid:
                        emit(-partial_position)
                    else:
                        emit(np.zeros((self._dimension, component_dim)))

                elif state_allowed["velocity"][2].match(component):
                    if owner_sid == target_sid:
                        emit(partial_velocity)
                    elif owner_sid == instrument_sid:
                        emit(-partial_velocity)
                    else:
                        emit(np.zeros((self._dimension, component_dim)))

                elif component in self._parameters and owner_sid in (
                    target_sid,
                    instrument_sid,
                ):
                    emit(self._compute_partial_by_parameter(component, context))

                elif self._is_station_parameter(component):
                    # A station parameter reaches the observable through the
                    # instrument's geometry, so it chains the same position and
                    # velocity partials the user's model already supplies.
                    emit(self._compute_h_tilde_station(
                        component_def, epoch.times.values, frame,
                        partial_position, partial_velocity))

                else:
                    emit(np.zeros((self._dimension, component_dim)))

        return rows[0] if self._dimension == 1 else rows

    def _compute_partial_by_position(self, context: dict) -> np.ndarray:
        """
        Jacobian of the observable with respect to relative position.

        Parameters
        ----------
        context : dict
            Evaluation context.

        Returns
        -------
        numpy.ndarray
            ``(dimension, 3)`` block; zeros when there is no dependence.
        """
        if not self._depends_on_position:
            return np.zeros((self._dimension, 3))

        if self._partial_by_position is not None:
            return self._as_matrix(
                self._partial_by_position(context), 3, "partial_by_position"
            )

        values = context["position"]
        steps = (
            CustomCallable.default_step(values, 1e-6)
            if self._position_step is None
            else np.full(3, float(self._position_step))
        )

        return CustomCallable.finite_difference(
            self._evaluator(context, "position"), values, steps, self._dimension
        )

    def _compute_partial_by_velocity(self, context: dict) -> np.ndarray:
        """
        Jacobian of the observable with respect to relative velocity.

        Parameters
        ----------
        context : dict
            Evaluation context.

        Returns
        -------
        numpy.ndarray
            ``(dimension, 3)`` block; zeros when there is no dependence.
        """
        if not self._depends_on_velocity:
            return np.zeros((self._dimension, 3))

        if self._partial_by_velocity is not None:
            return self._as_matrix(
                self._partial_by_velocity(context), 3, "partial_by_velocity"
            )

        values = context["velocity"]
        steps = (
            CustomCallable.default_step(values, 1e-9)
            if self._velocity_step is None
            else np.full(3, float(self._velocity_step))
        )

        return CustomCallable.finite_difference(
            self._evaluator(context, "velocity"), values, steps, self._dimension
        )

    def _compute_partial_by_parameter(
        self, parameter_name: str, context: dict
    ) -> np.ndarray:
        """
        Jacobian of the observable with respect to one declared parameter.

        Parameters
        ----------
        parameter_name : str
            Name of a parameter declared by this model.
        context : dict
            Evaluation context.

        Returns
        -------
        numpy.ndarray
            ``(dimension, parameter.dimension)`` block.
        """
        parameter = self._parameters[parameter_name]

        analytic = self._parameter_partials.get(parameter_name)
        if analytic is not None:
            return self._as_matrix(
                analytic(context),
                parameter.dimension,
                f"partial of '{parameter_name}'",
            )

        values = context["params"][parameter_name]

        return CustomCallable.finite_difference(
            self._evaluator(context, parameter_name),
            values,
            parameter.finite_difference_step(values),
            self._dimension,
        )

    def __repr__(self) -> str:
        return (
            f"CustomMeasurement(name={self._name!r}, "
            f"dimension={self._dimension}, parameters={list(self._parameters)})"
        )

    # endregion Methods #
    # -------------------#
