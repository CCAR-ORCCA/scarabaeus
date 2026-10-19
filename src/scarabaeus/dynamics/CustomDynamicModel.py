# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    CelestialBody,
    CustomParameter,
    DynamicModel,
    Frame,
    StateArray,
)

from scarabaeus.utils.CustomCallable import CustomCallable

import numpy as np

# Names a user callable may declare in its signature. Anything else is rejected
# at construction time so typos surface immediately instead of at the first
# right-hand-side evaluation, deep inside the integrator.
_CONTEXT_NAMES = (
    "position",
    "velocity",
    "epoch",
    "params",
    "state",
    "body",
    "spacecraft",
    "origin",
    "frame",
    "mass",
    "model",
)


class CustomDynamicModel(DynamicModel):
    """ User-supplied force model, defined entirely from a script.

    ``CustomDynamicModel`` turns a plain Python function into a first-class
    Scarabaeus force model: its acceleration is summed into the equations of
    motion alongside the built-in models, and the parameters it declares are
    propagated, differentiated, and estimated exactly like ``eta_srp`` or
    ``a_fogm``.  Adding a force therefore requires no change to the library
    source — build the model in the script and hand it to
    :class:`~scarabaeus.ForceModelTranslation` through ``custom_models``.

    The acceleration enters the equation of motion as one more term of

    .. math::

        \\ddot{\\mathbf r} =
            \\sum_i \\mathbf a_i(\\mathbf r,\\,\\dot{\\mathbf r},\\,\\mathbf p,\\,t)

    and the model supplies the blocks of
    :math:`\\partial \\mathbf f / \\partial \\mathbf p` needed by the variational
    equations: :math:`\\partial \\mathbf a / \\partial \\mathbf r`,
    :math:`\\partial \\mathbf a / \\partial \\dot{\\mathbf r}` and, for each
    declared :class:`~scarabaeus.CustomParameter`,
    :math:`\\partial \\mathbf a / \\partial p`.  Any Jacobian the user does not
    provide analytically is evaluated by central finite differences of the user
    functions, so a model is usable from its acceleration alone.

    **Calling convention.** Every user callable is invoked with keyword
    arguments; declare only the ones you need, in any order. The available names
    are:

    ==============  =========================================================
    ``position``    ``(3,)`` position of *body* w.r.t. the propagation origin, km
    ``velocity``    ``(3,)`` inertial velocity of *body*, km/s
    ``epoch``       current epoch, TDB seconds past J2000
    ``params``      ``dict`` mapping this model's parameter names to their
                    current values (estimated value when in the state, else the
                    parameter default)
    ``state``       full propagated state, keyed by ``(name, spice_id)``
    ``body``        the body the acceleration acts on
    ``spacecraft``  ``primary_body`` of the owning force model
    ``origin``      central body of the propagation
    ``frame``       base :class:`~scarabaeus.Frame` of the propagation
    ``mass``        mass of ``spacecraft`` at ``epoch``, kg
    ``model``       this ``CustomDynamicModel`` instance
    ==============  =========================================================

    A callable declaring ``**kwargs`` receives all of them.

    Parameters
    ----------
    name : str
        Identifier of the model. Appears in
        :attr:`~scarabaeus.ForceModelTranslation.dynamics_mode_list` and must be
        unique among the active force models.
    acceleration : callable
        Returns the ``(3,)`` acceleration contribution, in km/s^2. Required.
    parameters : list, optional
        Solve-for parameters consumed by this model, given as
        :class:`~scarabaeus.CustomParameter` objects or as plain names (which
        are registered as scalar, dimensionless parameters).
    partial_by_position : callable, optional
        Returns :math:`\\partial \\mathbf a / \\partial \\mathbf r` as ``(3, 3)``.
        Defaults to a central finite difference of *acceleration*.
    partial_by_velocity : callable, optional
        Returns :math:`\\partial \\mathbf a / \\partial \\dot{\\mathbf r}` as
        ``(3, 3)``. Defaults to a central finite difference of *acceleration*.
    depends_on_position : bool, optional
        Whether the acceleration depends on position. Inferred from the
        signature of *acceleration* when ``None`` (the default); set it
        explicitly when the dependence is hidden, e.g. read out of ``state``.
    depends_on_velocity : bool, optional
        Same, for the velocity dependence. When it resolves to ``False`` the
        (structurally zero) :math:`\\partial \\mathbf a / \\partial \\dot{\\mathbf r}`
        block is skipped entirely.
    position_step : float, optional
        Absolute finite-difference step on position, in km. Defaults to
        ``1e-6 * max(|r_i|, 1)`` per component.
    velocity_step : float, optional
        Absolute finite-difference step on velocity, in km/s. Defaults to
        ``1e-9 * max(|v_i|, 1)`` per component.

    See Also
    --------
    scarabaeus.CustomParameter : Declaration of an estimable custom parameter.
    scarabaeus.ForceModelTranslation : Aggregator that hosts custom models.
    scarabaeus.DynamicModel : Abstract base class of all force model components.

    Notes
    -----
    The model is bound to a concrete state and frame when the owning
    :class:`~scarabaeus.ForceModelTranslation` is initialized by the
    :class:`~scarabaeus.Propagator`. Build one instance per force model: sharing
    an instance between force models built on different spacecraft (e.g. a truth
    run and a filter reference) raises
    :class:`~scarabaeus.CustomDynamicModel.RebindError`.

    Accelerations must be returned in km/s^2 and positions/velocities are handed
    over in km and km/s — the internal units of the propagator.

    Examples
    --------
    An exponential-atmosphere drag force with an estimated drag coefficient:

    .. code-block:: python

        import numpy as np
        import scarabaeus as scb

        cd = scb.CustomParameter("drag_cd", default=2.2)

        def drag(position, velocity, params, spacecraft):
            h = np.linalg.norm(position) - 6378.0
            rho = 3.6e-4 * np.exp(-h / 88.667)          # kg/km**3
            area_over_mass = 1e-5                        # km**2/kg
            speed = np.linalg.norm(velocity)

            return -0.5 * params["drag_cd"][0] * area_over_mass * rho * speed * velocity

        drag_model = scb.CustomDynamicModel("Drag", drag, parameters=[cd])

        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            custom_models=[drag_model],
        )

    The parameter is estimated by adding it to the state definition:

    .. code-block:: python

        state_def = (scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .custom_param("drag_cd", sc, scb.ArrayWFrame(2.2, None, None))
        )
    """

    # Raised by the shared CustomCallable binder; aliased here so callers can
    # catch it off the model they were using.
    SignatureError = CustomCallable.SignatureError

    class ShapeError(ValueError):
        """Raised when a user callable returns an array of unexpected shape."""

    class DuplicateParameterError(ValueError):
        """Raised when the same parameter name is declared twice by one model."""

    class NotBoundError(RuntimeError):
        """Raised when the model is used before the force model initializes it."""

    class RebindError(RuntimeError):
        """Raised when one model instance is shared between two different spacecraft."""

    # -------------#
    # Constructor  #
    # -------------#
    def __init__(
        self,
        name: str,
        acceleration,
        parameters: list | None = None,
        partial_by_position=None,
        partial_by_velocity=None,
        depends_on_position: bool | None = None,
        depends_on_velocity: bool | None = None,
        position_step: float | None = None,
        velocity_step: float | None = None,
    ):
        self._name = str(name)

        # Bound by _bind() when the owning ForceModelTranslation initializes.
        self._base_frame: Frame | None = None
        self._primary_body = None
        self._origin = None
        self._t0 = None
        self._tf = None

        self._position_step = position_step
        self._velocity_step = velocity_step

        # Normalize the parameter declarations, keeping insertion order: it sets
        # the column order of the d(acceleration)/d(parameters) blocks.
        self._parameters: dict[str, CustomParameter] = {}
        for entry in parameters or []:
            parameter = (
                entry
                if isinstance(entry, CustomParameter)
                else CustomParameter(str(entry))
            )
            if parameter.name in self._parameters:
                raise self.DuplicateParameterError(
                    f"Custom model '{self._name}' declares parameter "
                    f"'{parameter.name}' more than once."
                )
            self._parameters[parameter.name] = parameter

        # Bind every user callable to the context vocabulary once. The binding
        # records which context entries each callable declares, so calls pass
        # exactly those and nothing else.
        self._acceleration = self._bind_callable(acceleration, "acceleration")
        self._partial_by_position = self._bind_callable(
            partial_by_position, "partial_by_position"
        )
        self._partial_by_velocity = self._bind_callable(
            partial_by_velocity, "partial_by_velocity"
        )

        # Per-parameter callables, keyed by (parameter name, role). Bound here so
        # a bad signature is reported at construction, and stored so the
        # evaluation paths can call them without re-inspecting.
        self._parameter_callables: dict = {}
        for parameter in self._parameters.values():
            for role, function in (
                ("partial", parameter.partial),
                ("dynamics", parameter.dynamics),
                ("partial_dynamics_by_param", parameter.partial_dynamics_by_param),
                ("partial_dynamics_by_position", parameter.partial_dynamics_by_position),
                ("partial_dynamics_by_velocity", parameter.partial_dynamics_by_velocity),
            ):
                bound = self._bind_callable(
                    function, f"{role} of parameter '{parameter.name}'"
                )
                if bound is not None:
                    self._parameter_callables[(parameter.name, role)] = bound

        # Structural dependence flags: they decide whether a Jacobian block is
        # produced at all, so they must not depend on the state or the epoch.
        self._depends_on_position = (
            self._infers_dependence("position")
            if depends_on_position is None
            else bool(depends_on_position)
        )
        self._depends_on_velocity = (
            self._infers_dependence("velocity")
            if depends_on_velocity is None
            else bool(depends_on_velocity)
        )
        if partial_by_position is not None:
            self._depends_on_position = True
        if partial_by_velocity is not None:
            self._depends_on_velocity = True

        self._dynamic_parameters = {
            key: parameter
            for key, parameter in self._parameters.items()
            if parameter.is_dynamic
        }

        # The spacecraft mass is the only context entry that costs anything to
        # evaluate (it may interpolate a mass profile), so it is filled in only
        # when some callable actually asked for it.
        self._needs_mass = any(
            bound is not None and "mass" in bound.names
            for bound in (
                self._acceleration,
                self._partial_by_position,
                self._partial_by_velocity,
                *self._parameter_callables.values(),
            )
        )

        # Zero blocks handed back when a dependence is structurally absent.
        self._zero_3x3 = np.zeros((3, 3))

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """Identifier of the model within the force model aggregator."""
        return self._name

    @property
    def parameters(self) -> dict:
        """Declared solve-for parameters, keyed by name."""
        return self._parameters

    @property
    def dynamic_parameters(self) -> dict:
        """Declared parameters that carry their own equation of motion."""
        return self._dynamic_parameters

    @property
    def depends_on_position(self) -> bool:
        """``True`` when the acceleration is a function of position."""
        return self._depends_on_position

    @property
    def depends_on_velocity(self) -> bool:
        """``True`` when the acceleration is a function of velocity."""
        return self._depends_on_velocity

    @property
    def primary_body(self):
        """Spacecraft the owning force model acts on, or ``None`` before binding."""
        return self._primary_body

    @property
    def base_frame(self) -> Frame | None:
        """Reference frame of the propagation, or ``None`` before binding."""
        return self._base_frame

    @property
    def origin(self):
        """Central body of the propagation, or ``None`` before binding."""
        return self._origin

    # endregion Properties #
    # ----------------------#

    # ------------------------#
    # region Initialization  #
    # ------------------------#
    def _bind(
        self,
        origin,
        base_frame: Frame,
        primary_body,
        t0=None,
        tf=None,
    ) -> None:
        """
        Attach the model to a concrete state, frame and spacecraft.

        Called by :meth:`~scarabaeus.ForceModelTranslation._initialize_force_models`;
        scripts never call it directly.

        Parameters
        ----------
        origin : Body
            Central body of the propagation, exposed to user callables.
        base_frame : Frame
            Reference frame of all dynamics computations.
        primary_body : Spacecraft
            Spacecraft the owning force model acts on.
        t0, tf : EpochArray or float, optional
            Propagation bounds, exposed for models that need them.

        Raises
        ------
        CustomDynamicModel.RebindError
            If this instance is already bound to a different spacecraft. Binding
            again would silently repoint the earlier force model at the new
            spacecraft, so build one model instance per force model instead.
        """
        previous = getattr(self._primary_body, "spice_id", None)
        current = getattr(primary_body, "spice_id", None)
        if self._primary_body is not None and previous != current:
            raise self.RebindError(
                f"Custom model '{self._name}' is already bound to spacecraft "
                f"{previous} and cannot be re-bound to {current}. Build a "
                "separate CustomDynamicModel instance for each ForceModel."
            )

        self._base_frame = base_frame
        self._primary_body = primary_body
        self._origin = origin
        self._t0 = t0
        self._tf = tf

    def _bind_callable(self, function, role: str):
        """
        Bind a user callable to this model's context vocabulary.

        Parameters
        ----------
        function : callable or None
            User-supplied function. ``None`` is passed through.
        role : str
            Human-readable description used in error messages.

        Returns
        -------
        CustomCallable or None
            The bound callable, or ``None`` when *function* was ``None``.
        """
        if function is None:
            return None

        return CustomCallable(function, _CONTEXT_NAMES, role, self._name)

    def _infers_dependence(self, context_name: str) -> bool:
        """
        Report whether the acceleration callable reads *context_name*.

        Parameters
        ----------
        context_name : str
            One of the context vocabulary names, e.g. ``"velocity"``.

        Returns
        -------
        bool
            ``True`` when the acceleration callable declares the name or
            ``state`` (from which it could read it).
        """
        return self._acceleration.reads(context_name)

    # endregion Initialization #
    # --------------------------#

    # ------------------------#
    # region Context / calls #
    # ------------------------#
    def _build_context(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> dict:
        """
        Assemble the keyword-argument context handed to the user callables.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` position of *body* relative to the propagation origin, km.
        current_state : dict
            Propagated state keyed by ``(name, spice_id)``.
        epoch : float
            Current epoch, TDB seconds.
        body : CelestialBody
            Body the acceleration acts on.

        Returns
        -------
        dict
            Mapping of every context name to its current value.
        """
        spice_id = getattr(body, "spice_id", None)

        velocity = current_state.get(("velocity", spice_id))
        if velocity is None:
            velocity = np.zeros(3)
        else:
            velocity = np.asarray(
                getattr(velocity, "values", velocity), dtype=float
            ).ravel()

        params = {
            key: parameter.resolve(current_state, spice_id)
            for key, parameter in self._parameters.items()
        }

        return {
            "position": np.asarray(position, dtype=float).ravel(),
            "velocity": velocity,
            "epoch": epoch,
            "params": params,
            "state": current_state,
            "body": body,
            "spacecraft": self._primary_body,
            "origin": self._origin,
            "frame": self._base_frame,
            "mass": self._resolve_mass(epoch) if self._needs_mass else None,
            "model": self,
        }

    def _resolve_mass(self, epoch: float) -> float | None:
        """
        Return the spacecraft mass at *epoch* in kilograms, or ``None``.

        Parameters
        ----------
        epoch : float
            Current epoch, TDB seconds.

        Returns
        -------
        float or None
            Mass in kg, or ``None`` when the spacecraft carries no usable mass
            profile.
        """
        if self._primary_body is None:
            return None

        # Body.mass() raises when no mass profile was configured; a custom model
        # may legitimately run against a massless body, so treat that as "unknown".
        try:
            mass = self._primary_body.mass(epoch)
        except Exception:
            return None

        if mass is None:
            return None

        values = np.atleast_1d(np.asarray(getattr(mass, "values", mass), dtype=float))

        return float(values[0])

    def _as_vector(self, values, size: int, role: str) -> np.ndarray:
        """
        Coerce a user return value to a 1-D array of length *size*.

        Parameters
        ----------
        values : array_like or ArrayWUnits
            Raw user output.
        size : int
            Expected number of elements.
        role : str
            Description used in error messages.

        Returns
        -------
        numpy.ndarray
            Flattened copy of *values*.

        Raises
        ------
        CustomDynamicModel.ShapeError
            If the value does not hold exactly *size* elements.
        """
        array = np.asarray(
            getattr(values, "values", values), dtype=float
        ).ravel()

        if array.size != size:
            raise self.ShapeError(
                f"The {role} of custom model '{self._name}' returned "
                f"{array.size} element(s); expected {size}."
            )

        return array

    def _as_matrix(self, values, shape: tuple, role: str) -> np.ndarray:
        """
        Coerce a user return value to a 2-D array of the requested *shape*.

        Parameters
        ----------
        values : array_like or ArrayWUnits
            Raw user output.
        shape : tuple of int
            Expected ``(rows, columns)``.
        role : str
            Description used in error messages.

        Returns
        -------
        numpy.ndarray
            Reshaped copy of *values*.

        Raises
        ------
        CustomDynamicModel.ShapeError
            If the value cannot be reshaped to *shape*.
        """
        array = np.asarray(getattr(values, "values", values), dtype=float)

        if array.size != shape[0] * shape[1]:
            raise self.ShapeError(
                f"The {role} of custom model '{self._name}' returned shape "
                f"{array.shape}; expected {shape}."
            )

        return array.reshape(shape)

    # endregion Context / calls #
    # ---------------------------#

    # ------------------------------#
    # region Finite differences    #
    # ------------------------------#
    def _evaluator(self, function, context: dict, key: str, size: int, role: str):
        """
        Build a perturbation-aware evaluator of *function* over one context slot.

        The perturbed value is written into the context (and, for position and
        velocity, into the shared state dictionary as well) so that the user
        function sees a consistent picture regardless of where it reads the
        quantity from. The original entries are restored afterwards.

        Parameters
        ----------
        function : callable
            Bound user callable to evaluate.
        context : dict
            Evaluation context; temporarily mutated by the returned evaluator.
        key : str
            Context slot to perturb — ``"position"``, ``"velocity"``, or a
            parameter name inside ``context["params"]``.
        size : int
            Expected length of the callable's output.
        role : str
            Description used in error messages.

        Returns
        -------
        callable
            Function mapping a perturbed value to the ``(size,)`` output.
        """
        state = context["state"]
        spice_id = getattr(context["body"], "spice_id", None)

        if key in ("position", "velocity"):
            state_key = (key, spice_id)

            def evaluate(perturbed):
                original_context = context[key]
                original_state = state.get(state_key)
                context[key] = perturbed
                if original_state is not None:
                    state[state_key] = perturbed
                try:
                    return self._as_vector(function(context), size, role)
                finally:
                    context[key] = original_context
                    if original_state is not None:
                        state[state_key] = original_state

            return evaluate

        params = context["params"]
        state_key = (key, spice_id)

        def evaluate(perturbed):
            original_param = params[key]
            original_state = state.get(state_key)
            params[key] = perturbed
            if original_state is not None:
                state[state_key] = perturbed
            try:
                return self._as_vector(function(context), size, role)
            finally:
                params[key] = original_param
                if original_state is not None:
                    state[state_key] = original_state

        return evaluate

    # endregion Finite differences #
    # ------------------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def compute_acceleration(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Evaluate the user-supplied acceleration contribution.

        Parameters
        ----------
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the acceleration acts on.

        Returns
        -------
        numpy.ndarray
            3x1 acceleration contribution, in :math:`\\frac{km}{s^2}`.
        """
        context = self._build_context(position, current_state, epoch, body)

        return self._as_vector(
            self._acceleration(context), 3, "acceleration"
        )

    def _compute_partial_by_position(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Jacobian of the custom acceleration with respect to position.

        Uses the user-supplied ``partial_by_position`` when available, otherwise
        a central finite difference of the acceleration. Returns a zero block
        when the acceleration does not depend on position.

        Parameters
        ----------
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the acceleration acts on.

        Returns
        -------
        numpy.ndarray
            The 3x3 Jacobian, in units of :math:`\\frac{1}{s^2}`.
        """
        if not self._depends_on_position:
            return self._zero_3x3

        context = self._build_context(position, current_state, epoch, body)

        if self._partial_by_position is not None:
            return self._as_matrix(
                self._partial_by_position(context),
                (3, 3),
                "partial_by_position",
            )

        values = context["position"]
        steps = (
            CustomCallable.default_step(values, 1e-6)
            if self._position_step is None
            else np.full(3, float(self._position_step))
        )

        return CustomCallable.finite_difference(
            self._evaluator(self._acceleration, context, "position", 3, "acceleration"),
            values,
            steps,
            3,
        )

    def _compute_partial_by_velocity(
        self,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Jacobian of the custom acceleration with respect to velocity.

        Uses the user-supplied ``partial_by_velocity`` when available, otherwise
        a central finite difference of the acceleration. Returns a zero block
        when the acceleration does not depend on velocity.

        Parameters
        ----------
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the acceleration acts on.

        Returns
        -------
        numpy.ndarray
            The 3x3 Jacobian, in units of :math:`\\frac{1}{s}`.
        """
        if not self._depends_on_velocity:
            return self._zero_3x3

        context = self._build_context(position, current_state, epoch, body)

        if self._partial_by_velocity is not None:
            return self._as_matrix(
                self._partial_by_velocity(context),
                (3, 3),
                "partial_by_velocity",
            )

        values = context["velocity"]
        steps = (
            CustomCallable.default_step(values, 1e-9)
            if self._velocity_step is None
            else np.full(3, float(self._velocity_step))
        )

        return CustomCallable.finite_difference(
            self._evaluator(self._acceleration, context, "velocity", 3, "acceleration"),
            values,
            steps,
            3,
        )

    def _compute_partial_by_parameter(
        self,
        parameter_name: str,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Jacobian of the custom acceleration with respect to one custom parameter.

        Parameters
        ----------
        parameter_name : str
            Name of a parameter declared by this model.
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the acceleration acts on.

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(3, parameter.dimension)``, in units of
            :math:`\\frac{km}{s^2}` divided by the parameter units.
        """
        parameter = self._parameters[parameter_name]
        context = self._build_context(position, current_state, epoch, body)

        if parameter.partial is not None:
            return self._as_matrix(
                self._parameter_callables[(parameter.name, "partial")](context),
                (3, parameter.dimension),
                f"partial of parameter '{parameter_name}'",
            )

        values = context["params"][parameter_name]

        return CustomCallable.finite_difference(
            self._evaluator(
                self._acceleration, context, parameter_name, 3, "acceleration"
            ),
            values,
            parameter.finite_difference_step(values),
            3,
        )

    def _compute_parameter_dynamics(
        self,
        parameter_name: str,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Evaluate :math:`\\dot p` for one dynamic custom parameter.

        Parameters
        ----------
        parameter_name : str
            Name of a dynamic parameter declared by this model.
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the parameter belongs to.

        Returns
        -------
        numpy.ndarray
            Time derivative of the parameter, shape ``(dimension,)``.
        """
        parameter = self._parameters[parameter_name]
        context = self._build_context(position, current_state, epoch, body)

        return self._as_vector(
            self._parameter_callables[(parameter.name, "dynamics")](context),
            parameter.dimension,
            f"dynamics of parameter '{parameter_name}'",
        )

    def _compute_parameter_dynamics_partial(
        self,
        parameter_name: str,
        with_respect_to: str,
        position: np.ndarray,
        current_state: dict,
        epoch: float,
        body: CelestialBody,
    ) -> np.ndarray:
        """
        Jacobian of a dynamic parameter's rate with respect to another quantity.

        Parameters
        ----------
        parameter_name : str
            Name of a dynamic parameter declared by this model.
        with_respect_to : str
            ``"position"``, ``"velocity"``, or the name of a parameter declared
            by this model.
        position : numpy.ndarray
            3x1 position of the body relative to the central body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the parameter belongs to.

        Returns
        -------
        numpy.ndarray
            Jacobian of shape ``(dimension, n)``, with ``n = 3`` for position and
            velocity and ``n = dimension`` of the differentiated parameter
            otherwise.
        """
        parameter = self._parameters[parameter_name]
        rows = parameter.dimension
        role = f"dynamics of parameter '{parameter_name}'"

        analytic_role = (
            "partial_dynamics_by_param"
            if with_respect_to == parameter_name
            else f"partial_dynamics_by_{with_respect_to}"
        )
        analytic = self._parameter_callables.get((parameter_name, analytic_role))

        context = self._build_context(position, current_state, epoch, body)

        if with_respect_to == "position":
            columns = 3
            values = context["position"]
            steps = (
                CustomCallable.default_step(values, 1e-6)
                if self._position_step is None
                else np.full(3, float(self._position_step))
            )
        elif with_respect_to == "velocity":
            columns = 3
            values = context["velocity"]
            steps = (
                CustomCallable.default_step(values, 1e-9)
                if self._velocity_step is None
                else np.full(3, float(self._velocity_step))
            )
        else:
            differentiated = self._parameters[with_respect_to]
            columns = differentiated.dimension
            values = context["params"][with_respect_to]
            steps = differentiated.finite_difference_step(values)

        if analytic is not None:
            return self._as_matrix(
                analytic(context),
                (rows, columns),
                f"partial_dynamics_by_{with_respect_to} of parameter "
                f"'{parameter_name}'",
            )

        return CustomCallable.finite_difference(
            self._evaluator(
                self._parameter_callables[(parameter_name, "dynamics")],
                context,
                with_respect_to,
                rows,
                role,
            ),
            values,
            steps,
            rows,
        )

    def _check_partials(self, position, current_state, epoch, body, write_line) -> None:
        """
        Compare every user-supplied Jacobian against a finite difference.

        Invoked by :meth:`~scarabaeus.ForceModelTranslation._check_all_partials`
        when the ``check_force_model_partials`` environment variable is set.
        Blocks that are already finite differences are skipped — comparing them
        to themselves proves nothing.

        Parameters
        ----------
        position : ArrayWUnits
            Position of the body, in kilometers.
        current_state : dict
            Dictionary containing the current state parameters.
        epoch : float
            Current simulation epoch in seconds.
        body : CelestialBody
            Body the acceleration acts on.
        write_line : callable
            Diagnostic writer ``(model, dependent, independent, analytical, algo)``.
        """
        position_values = np.asarray(
            getattr(position, "values", position), dtype=float
        ).ravel()

        context = self._build_context(position_values, current_state, epoch, body)
        axes = ("x", "y", "z")

        if self._partial_by_position is not None:
            analytical = self._as_matrix(
                self._partial_by_position(context),
                (3, 3),
                "partial_by_position",
            )
            numerical = CustomCallable.finite_difference(
                self._evaluator(
                    self._acceleration, context, "position", 3, "acceleration"
                ),
                context["position"],
                CustomCallable.default_step(context["position"], 1e-6),
                3,
            )
            for column, independent in enumerate(axes):
                for row, dependent in enumerate(axes):
                    write_line(
                        self._name,
                        dependent,
                        f"position_{independent}",
                        analytical[row][column],
                        numerical[row][column],
                    )

        if self._partial_by_velocity is not None:
            analytical = self._as_matrix(
                self._partial_by_velocity(context),
                (3, 3),
                "partial_by_velocity",
            )
            numerical = CustomCallable.finite_difference(
                self._evaluator(
                    self._acceleration, context, "velocity", 3, "acceleration"
                ),
                context["velocity"],
                CustomCallable.default_step(context["velocity"], 1e-9),
                3,
            )
            for column, independent in enumerate(axes):
                for row, dependent in enumerate(axes):
                    write_line(
                        self._name,
                        dependent,
                        f"velocity_{independent}",
                        analytical[row][column],
                        numerical[row][column],
                    )

        for key, parameter in self._parameters.items():
            if parameter.partial is None:
                continue

            analytical = self._as_matrix(
                self._parameter_callables[(parameter.name, "partial")](context),
                (3, parameter.dimension),
                f"partial of parameter '{key}'",
            )
            values = context["params"][key]
            numerical = CustomCallable.finite_difference(
                self._evaluator(self._acceleration, context, key, 3, "acceleration"),
                values,
                parameter.finite_difference_step(values),
                3,
            )
            for column in range(parameter.dimension):
                for row, dependent in enumerate(axes):
                    write_line(
                        self._name,
                        dependent,
                        f"{key}_{column}",
                        analytical[row][column],
                        numerical[row][column],
                    )

    def __repr__(self) -> str:
        return (
            f"CustomDynamicModel(name={self._name!r}, "
            f"parameters={list(self._parameters)})"
        )

    # endregion Methods #
    # -------------------#
