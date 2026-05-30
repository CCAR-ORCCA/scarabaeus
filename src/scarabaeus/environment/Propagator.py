# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    Units,
    Frame,
    EpochArray,
    Body,
    ForceModel,
    Spacecraft,
    ArrayWUnits,
    StateArray,
    SpiceManager,
    ArrayWFrame,
    StateDefinition,
    Maneuver,
    GroundStation,
    IAS15,
    ForceModelRotation,
    ForceModelTranslation,
    Maneuver,
)
from typing import Optional, Union, Sequence, Literal, Tuple
import copy
import scarabaeus.utils.NumpyWrapper as np
import numpy
import os
from scipy.integrate import solve_ivp

from tqdm import tqdm

# ----------------------------#
#  Generate Units and Frames  #
# ----------------------------#

kg, km, sec, unitless = Units.get_units(["kg", "km", "sec", "unitless"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# -----------------------#
#    Class Definition   #
# -----------------------#
class Propagator:
    """ Scarabaeus interface to numerical propagation.

    This class integrates the equations of motion for a given state
    :math:`\\mathbf{x}` over a specified timespan:

    .. math::

        \\dot{\\mathbf{x}} = \\mathbf{f}(\\mathbf{x}, t),
        \\qquad
        \\mathbf{x} =
        \\begin{bmatrix}
            \\mathbf{r} \\\\ \\dot{\\mathbf{r}} \\\\ \\mathbf{p}
        \\end{bmatrix}

    where :math:`\\mathbf{r}` is position, :math:`\\dot{\\mathbf{r}}` is
    velocity, and :math:`\\mathbf{p}` are dynamic/static parameters.
    When ``propagate_STM=True`` the State Transition Matrix
    :math:`\\boldsymbol{\\Phi}` is propagated alongside the state via the
    variational equation:

    .. math::

        \\dot{\\boldsymbol{\\Phi}}(t, t_0) =
            \\mathbf{A}(t)\\,\\boldsymbol{\\Phi}(t, t_0),
        \\qquad
        \\boldsymbol{\\Phi}(t_0, t_0) = \\mathbf{I}

    where :math:`\\mathbf{A}(t) = \\partial\\mathbf{f}/\\partial\\mathbf{x}`
    is the Jacobian of the dynamics.

    Parameters
    ----------
    primary_body : Spacecraft
        Nominal "primary" object associated with the propagated state. Used for
        convenience and compatibility; individual force models may internally
        act on different objects or subsets of the state.
    state_vector : StateArray
        Initial state (position/velocity + parameters) at ``epoch = state_vector.epoch``.
    tspan : EpochArray or tuple of two EpochArrays
        Times at which the solution is requested. If given an EpochArray, solution
        will be computed at each time step. If given a tuple of two EpochArrays,
        solution will computed at time steps determined by the selected integrator.
    force_models : ForceModel or list[ForceModel]
        One or more pre-configured force model instances. Each will be
        initialized with the given ``state_vector`` and ``base_frame``.
        This argument is required in the current interface.
    base_frame : Frame, optional
        Frame in which dynamics are evaluated. Defaults to J2000 if ``None``.
    atol, rtol : float, optional
        Absolute/relative tolerances for the numerical integrator.
    integrator : str, optional
        Integration scheme identifier (e.g. ``"DOP853"``).
    propagate_STM : bool, optional
        If True, propagate the state transition matrix.
    STM0 : np.ndarray, optional
        Initial STM. If None, identity is used.
    look_event : str, optional
        Placeholder for event handling (WIP).
    convert_to_original_frame : bool, optional
        If True, map results back to the original state frame after propagation.
    mass_is_variable : bool, optional
        If True, treat mass as a dynamic state (used with thrust models).
    display_progress : bool, optional
        If True, displays integration progress bar in the terminal. Defaults to ``True``.
    copy_force_models : bool, optional
        If True, creates deep copies of the provided force models to avoid side effects.

    Notes
    -----
    **Typical usage sequence:**

    1. Construct a :class:`~scarabaeus.StateArray` (via :class:`~scarabaeus.StateDefinition`).
    2. Construct one or more :class:`~scarabaeus.ForceModel` instances
       (e.g. :class:`~scarabaeus.ForceModelTranslation`) configured for the relevant objects.
    3. Pass the model(s) to :class:`~scarabaeus.Propagator`, which will bind them to the
       provided state and frame via ``initialize_force_models``.

    - ``force_models`` may be a single instance or a list, enabling multi-object
      or multi-model configurations.
    - Each provided force model must implement:
        - ``initialize_force_models(state_vector, base_frame, t0)``
        - ``dynamics_mode_list`` (property)
        - the appropriate acceleration/partial APIs used by the propagator.

    See Also
    --------
    scarabaeus.StateArray : State vector consumed and produced by the propagator.
    scarabaeus.ForceModelTranslation : Translational force model aggregator.
    scarabaeus.ScenarioSetup : Multi-leg propagation driver that wraps Propagator.
    scarabaeus.MissionSequence : Ordered container of legs passed to ScenarioSetup.
    """

    def __init__(
        self,
        primary_body: Spacecraft,
        state_vector: StateArray,
        tspan: EpochArray | Tuple[EpochArray, EpochArray],
        force_models: Union[ForceModel, Sequence[ForceModel]],
        base_frame: Optional[Frame] = None,
        atol: float = 2.220446049250313e-14,
        rtol: float = 2.220446049250313e-14,
        integrator: Literal["DOP853", "IAS15"] = "DOP853",
        propagate_STM: bool = True,
        STM0=None,
        look_event: Optional[str] = None,
        convert_to_original_frame: bool = False,
        mass_is_variable: bool = False,
        display_progress: bool = True,
        copy_force_models: bool = True,
        backward: bool = False,
    ):
        # store base objects / copies
        self._primary_body = primary_body
        self._prop_origin = state_vector.origin

        self._full_state_vector_original_frames = copy.deepcopy(state_vector)
        self._full_state_vector = copy.deepcopy(state_vector)

        self._propagated_state_array_original_frame = []
        self._propagated_state_array = []

        self._t0 = state_vector.epoch
        self._tspan_original_frame = copy.deepcopy(tspan)

        self._look_event = look_event
        self._convert_to_original_frame = convert_to_original_frame
        self._mass_is_variable = mass_is_variable
        self._display_progress = display_progress
        self._backward = backward

        # Base frame
        self.base_frame = base_frame if base_frame is not None else J2000

        # Normalize tspan to TDB for consistency with ephemerides / forces
        # self._tspan = (
        #     copy.deepcopy(tspan).convert_to("TDB")
        #     if tspan.system != "TDB"
        #     else copy.deepcopy(tspan)
        # )
        self._tspan = copy.deepcopy(tspan)
        match self._tspan:
            case EpochArray():
                # given time vector to solve along
                if self._tspan.system != "TDB":
                    # convert to TDB if not already
                    self._tspan.to(sys="TDB")
            case tuple():
                # given (t0, tf) tuple
                for t in self._tspan:
                    # convert both t0 and tf to TDB is not already
                    if t.system != "TDB":
                        t.to(sys="TDB")
            case _:
                # not the correct type -> raise error
                raise ValueError(
                    "Time span must be either an EpochArray or tuple of two EpochArrays. ",
                    f"Received {type(self._tspan)}",
                )

        # Numerics
        self._atol = atol
        self._rtol = rtol
        self._integrator = integrator
        self._propagate_STM = propagate_STM

        # Force models: normalize to list and initialize all
        if force_models is None:
            raise ValueError(
                "Propagator requires at least one ForceModel instance. "
                "Got force_model=None."
            )

        if isinstance(force_models, ForceModel):
            self._force_models: list[ForceModel] = (
                [copy.deepcopy(force_models)] if copy_force_models else [force_models]
            )
        else:
            # Accept any non-empty iterable / sequence of ForceModel
            fm_list = list(force_models)
            if not fm_list:
                raise ValueError("force_models list is empty.")
            if not all(isinstance(fm, ForceModel) for fm in fm_list):
                raise TypeError(
                    "All entries in force_models must be instances of ForceModel."
                )
            self._force_models = (
                [copy.deepcopy(fm) for fm in fm_list] if copy_force_models else fm_list
            )

        # Bind each ForceModel to the working state and frame
        for fm in self._force_models:
            fm.initialize_force_models(
                state_vector=self._full_state_vector,
                base_frame=self.base_frame,
            )

        # Combined dynamics mode list (union over all models)
        self._dynamics_mode_list = sorted(
            {
                mode
                for fm in self._force_models
                for mode in getattr(fm, "dynamics_mode_list", [])
            }
        )

        # frame conversion & validation hooks (your existing implementations)
        self._convert_to_base_frame()
        self._validate_configuration()

        # build reduced state & STM
        self._y_full0 = self.full_state_vector.values0.values.copy().reshape(-1)
        self._create_reduced_state_vector()

        n_state = state_vector.size
        if STM0 is None:
            self._STM0 = np.identity(n_state).reshape(n_state**2)
        else:
            self._STM0 = STM0.reshape(n_state**2)

        match self.tspan:
            case EpochArray():
                # given EpochArray -> preallocate solution size for requested times
                init_size = self.tspan.size
            case tuple():
                # given (t0, tf) tuple -> don't know what the solution size will be
                init_size = 1

        self._ys = np.zeros(
            (init_size, self._reduced_state_vector.size + self._STM0.size)
        )
        self._As = np.zeros((0, n_state, n_state))
        self._As_list = []
        self._times = np.zeros((init_size, 1))

        # initial condition (state + STM)
        self._ys[0] = np.concatenate(
            (self._reduced_state_vector.values0.values, self._STM0)
        )

        # units / frames bookkeeping
        self._state_units, self._state_frames = (
            self._accumulate_and_save_units_and_frames()
        )
        self._STM_units = self._accumulate_and_save_stm_units()
        self._STM_frame = self._accumulate_and_save_frame_ratios()

        # optional mass history
        self._mass_profile = None

        # precompute ODE cache (must be last — needs all state/index structures ready)
        self._precompute_ode_cache()

    def _create_reduced_state_vector(self):
        # Separate dynamic and static parameters
        self._dynamic_indices = []
        self._static_indices = []
        self._static_values = []

        for i, param in enumerate(self.full_state_vector.state_definition):
            if param[3] == "static":
                self._static_indices.append(i)
                self._static_values.append(
                    self.full_state_vector.state[i][-1].quantity.values
                )
            else:
                self._dynamic_indices.append(i)

        # Extract only dynamic parameters for the reduced state
        reduced_state = [self.full_state_vector.state[i] for i in self._dynamic_indices]

        # Create a new StateArray without modifying the existing one
        self._reduced_state_vector = StateArray(
            epoch=self.full_state_vector.epoch,
            origin=self.full_state_vector.origin,
            state=StateDefinition.from_components(reduced_state),
        )

    def _validate_configuration(self):
        """
        Validate the propagator configuration for consistency.

        Checks
        ------
        - If 'eta_srp' appears in the state, at least one force model must
          enable an SRP contribution ('CannonballSRP' or 'nPlateSRP').
        - If any force model enables 'ThreeBodyGravity', its point-mass list
          must be non-empty and must not include the propagation origin body.
        - The initial epoch of `tspan` must match the epoch of the state vector.
        - If mass is modeled as dynamic, `primary_body` must be a Spacecraft.

        Raises
        ------
        ValueError
            On inconsistent SRP, third-body, or time-span configuration.
        TypeError
            If dynamic-mass modeling is requested for a non-spacecraft object.
        """

        # SRP: eta_srp present -> require SRP model
        try:
            components = self._full_state_vector.state_definition
        except AttributeError:
            components = getattr(self._full_state_vector, "state", [])

        has_eta_srp = any(entry[0] == "eta_srp" for entry in components)

        has_srp_model = any(
            any(
                mode in getattr(fm, "dynamics_mode_list", [])
                for mode in ("CannonballSRP", "nPlateSRP")
            )
            for fm in self._force_models
        )

        if has_eta_srp and not has_srp_model:
            raise ValueError(
                "Invalid configuration: 'eta_srp' is present in the state vector, "
                "but no SRP model is enabled (expected 'CannonballSRP' or 'nPlateSRP')."
            )

        # Time span: tspan[0] must match state epoch
        # Use EpochArray semantics if available
        t0 = self._t0.times

        match self._tspan:
            case EpochArray():
                # NOTE: why are we testing for hasattr? what does use EpochArray semantics mean?
                tspan0 = (
                    self._tspan.times[0]
                    if hasattr(self._tspan, "__getitem__")
                    else None
                )
            case tuple():
                tspan0 = self._tspan[0].times

        # if tspan0 is not None:
        # If EpochArray supports direct comparison, this is enough.
        # if tspan0 != t0:
        # Optionally guard NaN-like epochs if your EpochArray allows them.
        # try:
        # if not np.isnan(float(t0)):
        # raise ValueError(
        #    "Invalid configuration: the initial time of `tspan` "
        #    "must be equal to the epoch of the initial state."
        # )
        # except AttributeError:
        # Fallback if `.times` is not present; treat as strict mismatch.
        # raise ValueError(
        #    "Invalid configuration: the initial time of `tspan` "
        #    "must be equal to the epoch of the initial state."
        # )

        # Third-body: require point masses & exclude origin
        origin_name = getattr(self._prop_origin, "name", None)
        origin_name_low = origin_name.lower() if origin_name else None

        for fm in self._force_models:
            modes = getattr(fm, "dynamics_mode_list", [])
            if "ThreeBodyGravity" in modes:
                third_bodies = getattr(fm, "third_bodies", []) or []

                if not third_bodies:
                    raise ValueError(
                        "Invalid configuration: 'ThreeBodyGravity' is enabled in a ForceModel "
                        "but its 'third_bodies' list is empty."
                    )

                if origin_name_low is not None:
                    pm_names_low = []
                    for pm in third_bodies:
                        if hasattr(pm, "name"):
                            pm_names_low.append(pm.name.lower())
                        else:
                            pm_names_low.append(str(pm).lower())

                    if origin_name_low in pm_names_low:
                        raise ValueError(
                            f"Invalid third-body configuration: the origin body '{origin_name}' "
                            f"cannot be included in the 'third_bodies' list."
                        )

        # Dynamic mass: must be a Spacecraft
        if self._mass_is_variable and not isinstance(self._primary_body, Spacecraft):
            raise TypeError(
                "Invalid configuration: dynamic mass modeling requires "
                "'primary_body' to be a Spacecraft, "
                f"but got type '{type(self._primary_body).__name__}'."
            )

    def _convert_to_base_frame(self):
        """
        Converts all components of the state vector to the base frame before propagation.
        Stores the original frames for restoring later.
        """
        self._original_frames = {}  # Dictionary to store original frames
        converted_state = []  # New state vector after frame conversion

        for i, param in enumerate(self._full_state_vector.state):
            name, size, est_type, dyn_type, body, awf = param  # Unpack state component

            # Store the original frame
            self._original_frames[i] = awf.frame

            # Convert only if the frame is different from base_frame
            if name == "position" or name == "velocity":
                if awf.frame:
                    if awf.frame.name != self._base_frame.name:
                        awf.convert_to(self._base_frame, self._t0)

            # Append the converted state component to the new list
            converted_state.append((name, size, est_type, dyn_type, body, awf))

        # Reinitialize the state vector with the converted components
        self._full_state_vector = StateArray(
            epoch=self._full_state_vector.epoch,
            origin=self._full_state_vector.origin,
            state=StateDefinition.from_components(converted_state),
        )

    def _accumulate_and_save_units_and_frames(self):
        """
        Accumulate the units and frames of each state vector component.

        Returns
        -------
        tuple
            A 2-tuple ``(units, frames)`` where each is a list with one entry
            per scalar element of the state vector.
        """
        # Initialize lists to store units and frames
        units = []
        frames = []

        # Iterate through the state vector components
        for param in self.full_state_vector.state:
            # Extract the name, size, frame, and unit of the component
            _, size, _, _, _, value_template = param

            # Repeat the unit and frame for each component in the parameter
            for _ in range(size):
                units.append(
                    value_template.quantity.units
                )  # Store unit for each component
                frames.append(value_template.frame)  # Store frame for each component

        return units, frames

    def _accumulate_and_save_stm_units(self):
        """
        Build the units matrix for the State Transition Matrix (STM).

        For each time-derivative entry in ``state_dot_definition`` and for each
        state component in ``state_definition``, this method queries the
        appropriate :class:`ForceModel` for the corresponding partial derivative.
        The units of these partials are stored in a matrix with shape
        (n_state, n_state), aligned with the STM layout.

        Any STM entry without an associated physical partial is set to unitless.

        Returns
        -------
        np.ndarray
            Matrix of STM entry units with shape
            (full_state_vector.size, full_state_vector.size).
        """
        n_full = self._full_state_vector.size
        STM_units = np.full((n_full, n_full), None, dtype=object)

        # Precompute indices for full state and its derivatives
        _, _, indices_map, indices_dot_map, _ = self._precompute_indices()

        # Initial state / epoch context
        Y0 = self._ys[0]

        match self._tspan:
            case EpochArray():
                # get t0 for EpochArray configuration
                epoch0 = self._tspan.times.values[0]
            case tuple():
                # get t0 for tuple of t0, tf EpochArrays configuration
                epoch0 = self._tspan[0].times.values

        full_state = self._full_state_vector._make_direct_representation(Y0)

        # Helper: position unit for a given body from template state
        def _get_position_unit(target_body):
            for name, _, _, _, param_body, template in self._full_state_vector.state:
                if name == "position" and getattr(
                    param_body, "spice_id", None
                ) == getattr(target_body, "spice_id", None):
                    return template.quantity.units
            return None

        unit_cache = {}

        # ============================================================
        # Loop over derivatives (rows)
        # ============================================================
        for param_dot in self._full_state_vector.state_dot_definition:
            name_dot, _, _, dynamics_type, body = param_dot

            # Static params (gm, cnm, pole_*, etc.) have zero derivative;
            # their STM rows are left unset and filled with unitless below.
            if dynamics_type != "dynamic":
                continue

            # Resolve position indices: for non-spacecraft, fall back to primary
            if isinstance(body, Spacecraft):
                pos_start, pos_end = indices_map.get(
                    ("position", body.spice_id), (None, None)
                )
            else:
                pos_start, pos_end = indices_map.get(
                    ("position", self._primary_body.spice_id), (None, None)
                )

            start_index, end_index = indices_dot_map.get(
                (name_dot, body.spice_id), (None, None)
            )

            if None in (pos_start, pos_end, start_index, end_index):
                raise ValueError(
                    f"Indices not found for derivative '{name_dot}' and body {body}."
                )

            # Get / cache units for this body's position
            if body.spice_id not in unit_cache:
                unit_cache[body.spice_id] = _get_position_unit(
                    body
                ) or _get_position_unit(self._primary_body)
                if unit_cache[body.spice_id] is None:
                    raise ValueError(
                        f"Could not find 'position' units for body {body} "
                        f"or fallback primary_body {self._primary_body}."
                    )

            pos_unit = unit_cache[body.spice_id]
            position = ArrayWUnits(Y0[pos_start:pos_end], pos_unit)

            # Pick the correct force model for this body
            fm = self._get_force_model_for_body(body)

            # Build the 'dot' part name, as in ode()
            param_dot_text = ".".join([str(x) for x in param_dot[:-1]])

            # ========================================================
            # Loop over state components (columns)
            # ========================================================
            for param in self._full_state_vector.state_definition:
                param_text = ".".join([str(x) for x in param[:-1]])
                partial_name = "_by_".join([param_dot_text, param_text])

                partial = (
                    fm._collect_partials(
                        name=partial_name,
                        position=position,
                        current_state=full_state,
                        epoch=epoch0,
                        body=body,
                    )
                    if fm is not None
                    else None
                )

                if partial is not None:
                    # param[-1] is the owner/body of this parameter
                    param_body = param[-1]
                    param_start, param_end = indices_map.get(
                        (param[0], param_body.spice_id), (None, None)
                    )

                    if None in (param_start, param_end):
                        raise ValueError(
                            f"Indices not found for parameter '{param[0]}' "
                            f"and body {param_body} in STM units accumulation."
                        )

                    # Extract units from partial; assume ArrayWUnits-like
                    partial_units = getattr(partial, "units", None)
                    if partial_units is None:
                        # If units are missing, fall back to unitless for safety
                        partial_units = unitless

                    # Fill the block [start_index:end_index, param_start:param_end]
                    # We multiply by 'sec' here to align with your original convention.
                    for i in range(start_index, end_index):
                        for j in range(param_start, param_end):
                            STM_units[i, j] = partial_units * sec

        # Fill all unset entries with unitless
        for i in range(n_full):
            for j in range(n_full):
                if STM_units[i, j] is None:
                    STM_units[i, j] = unitless

        return STM_units

    def _accumulate_and_save_frame_ratios(self):
        """
        Compute and store frame-ratio entries for the STM frames matrix.

        Returns
        -------
        np.ndarray
            STM frames matrix of shape ``(n, n)`` where each element is a
            frame-ratio object relating component *i* to component *j*.
        """
        # Initialize STM frames matrix with None
        STM_frames = numpy.full(
            (self.full_state_vector.size, self.full_state_vector.size), None
        )

        # Precompute indices for state components
        _, _, indices_map, _, _ = self._precompute_indices()

        # Loop through the state vector to retrieve frames for each state component
        for i, (name_i, size_i, _, _, body_i, value_template_i) in enumerate(
            self.full_state_vector.state
        ):
            frame_i = (
                value_template_i.frame
            )  # Get the frame for component i (position or velocity)

            # Track the start and end indices for component i
            start_index_i, end_index_i = indices_map.get(
                (name_i, body_i.spice_id), (None, None)
            )

            # Handle missing indices for component i
            if start_index_i is None or end_index_i is None:
                raise ValueError(
                    f"Indices not found for component {name_i} and body {body_i}."
                )

            for j, (name_j, size_j, _, _, body_j, value_template_j) in enumerate(
                self.full_state_vector.state
            ):
                frame_j = value_template_j.frame  # Get the frame for component j

                # Track the start and end indices for component j
                start_index_j, end_index_j = indices_map.get(
                    (name_j, body_j.spice_id), (None, None)
                )

                # Handle missing indices for component j
                if start_index_j is None or end_index_j is None:
                    raise ValueError(
                        f"Indices not found for component {name_j} and body {body_j}."
                    )

                # Check if either frame is None and build the frame ratio accordingly
                if frame_i is None and frame_j is None:
                    frame_ratio = "None/None"
                elif frame_i is None:
                    frame_ratio = f"None/{frame_j.name}"
                elif frame_j is None:
                    frame_ratio = f"{frame_i.name}/None"
                else:
                    # If both frames are valid, calculate the ratio
                    frame_ratio = f"{frame_i.name}/{frame_j.name}"

                # Set the frame ratio for STM matrix at position (i, j)
                # For example, frame_i/frame_j for the state components i and j
                for k in range(size_i):  # Iterate over the size of the component i
                    for l in range(size_j):  # Iterate over the size of the component j
                        # Map frame_i/frame_j to the correct positions in STM matrix
                        STM_frames[start_index_i + k, start_index_j + l] = frame_ratio

        return STM_frames

    # ----------------------#
    # region    Properties #
    # ----------------------#
    # --- Core Attributes ---
    @property
    def primary_body(self) -> Spacecraft:
        """
        The primary celestial body for the propagator.

        :base: Propagator
        :type: scb.Body
        """
        return self._primary_body

    @property
    def full_state_vector(self) -> StateArray:
        """
        The state vector associated with the propagation.

        :base: Propagator
        :type: scb.StateArray
        """
        return self._full_state_vector

    @property
    def full_state_vector_original_frames(self) -> StateArray:
        """
        The state vector not associated with the propagation, but with
        the original frames.

        :base: Propagator
        :type: scb.StateArray
        """
        return self._full_state_vector_original_frames

    @property
    def propagated_state_array_original_frame(self) -> StateArray | list:
        """
        The state vector not associated propagated, but with
        the original frames.

        :base: Propagator
        :type: scb.StateArray
        """
        return self._propagated_state_array_original_frame

    @property
    def prop_origin(self) -> Body:
        """
        The origin body in the reference frame.

        :base: Propagator
        :type: scb.Body
        """
        return self._prop_origin

    @property
    def base_frame(self) -> Frame:
        """
        The base reference frame for propagation.

        :base: Propagator
        :type: str
        """
        return self._base_frame

    @base_frame.setter
    def base_frame(self, input_val):
        if input_val is None:
            input_val = J2000

        self._base_frame = input_val

    @property
    def t0(self) -> EpochArray:
        """
        The initial epoch for propagation.

        :base: Propagator
        :type: scb.EpochArray
        """
        return self._t0

    @property
    def tspan_original_frame(self) -> EpochArray:
        """
        The time span over which propagation occurs in the original frame.

        :base: Propagator
        :type: scb.EpochArray
        """
        return self._tspan_original_frame

    @property
    def tspan(self) -> EpochArray | tuple:
        """
        The time span over which propagation occurs.

        :base: Propagator
        :type: EpochArray | Tuple[EpochArray, EpochArray]
        """
        return self._tspan

    @property
    def backward(self) -> bool:
        """
        Whether the propagation runs backward in time.

        :base: Propagator
        :type: bool
        """
        return self._backward

    @property
    def look_event(self) -> str | None:
        """
        The event type for the propagator.

        :base: Propagator
        :type: str
        """
        return self._look_event

    # --- Propagation Settings ---
    @property
    def atol(self) -> float:
        """
        Absolute tolerance for numerical integration.

        :base: Propagator
        :type: float
        """
        return self._atol

    @property
    def rtol(self) -> float:
        """
        Relative tolerance for numerical integration.

        :base: Propagator
        :type: float
        """
        return self._rtol

    @property
    def integrator(self) -> str:
        """
        The numerical integrator used for propagation.

        :base: Propagator
        :type: str
        """
        return self._integrator

    @property
    def propagate_STM(self) -> bool:
        """
        Indicates whether the State Transition Matrix (STM) is propagated.

        :base: Propagator
        :type: bool
        """
        return self._propagate_STM

    @property
    def integrationYs(self) -> numpy.ndarray | None:
        """Raw numerical integration output (states + flattened STM rows)."""
        return self._integrationYs

    @property
    def integrationEts(self) -> numpy.ndarray | None:
        """Epoch array (TDB seconds) corresponding to each integrator output step."""
        return self._integrationEts

    @property
    def STM0(self) -> numpy.ndarray:
        """
        The initial State Transition Matrix (STM).

        :base: Propagator
        :type: np.ndarray
        """
        return self._STM0

    @property
    def STM_units(self) -> numpy.ndarray | None:
        """The units of the State Transition Matrix (STM).

        :base: Propagator
        :type: np.ndarray
        """
        return self._STM_units

    @property
    def STM_frame(self) -> numpy.ndarray | None:
        """The frame ratios of the State Transition Matrix (STM).

        :base: Propagator
        :type: np.ndarray
        """
        return self._STM_frame

    @property
    def state_units(self) -> numpy.ndarray | None:
        """The units of the State .

        :base: Propagator
        :type: np.ndarray
        """
        return self._state_units

    @property
    def state_frames(self) -> numpy.ndarray | None:
        """The frames of the State.

        :base: Propagator
        :type: np.ndarray
        """
        return self._state_frames

    # --- Output Data ---
    @property
    def ys(self) -> numpy.ndarray:
        """
        Numerical solution of the propagated state.

        :base: Propagator
        :type: np.ndarray
        """
        return self._ys

    @property
    def As(self) -> numpy.ndarray:
        """
        Dynamic Matrices of the propagated state.

        :base: Propagator
        :type: np.ndarray
        """
        return self._As

    @property
    def times(self) -> numpy.ndarray:
        """
        Time array corresponding to the propagated solution.

        :base: Propagator
        :type: np.ndarray
        """
        return self._times

    @property
    def state(self) -> numpy.ndarray | None:
        """
        State array corresponding to the propagated solution.

        :base: Propagator
        :type: np.ndarray
        """
        return self._state

    @property
    def STM(self) -> ArrayWFrame | None:
        """
        State Transition Matrix array corresponding to the propagated solution.

        :base: Propagator
        :type: np.ndarray
        """
        return self._STM

    @property
    def propagated_state_array(self) -> StateArray:
        """
        State array corresponding to the propagated solution.

        :base: Propagator
        :type: np.ndarray
        """
        return self._propagated_state_array

    @property
    def mass_profile(self) -> object | None:
        """
        Mass profile of the spacecraft.

        :base: Propagator
        :type: SCBPolynomial
        """
        return self._mass_profile

    # --- Configuration ---
    @property
    def dynamics_mode_list(self) -> list[str]:
        """
        List of active dynamics modes during propagation.

        :base: Propagator
        :type: list[str]
        """
        return self._dynamics_mode_list

    @property
    def force_models(self) -> list[ForceModel]:
        """
        Force model used for propagation.

        :base: Propagator
        :type: scb.ForceModel
        """
        return self._force_models

    # ----------------------#
    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _convert_to_original_frames(self, propagated_state):
        """
        Converts the propagated state back to the original frames.

        Parameters
        ----------
        propagated_state : StateArray
            The state array after propagation in the base frame.

        Returns
        -------
        StateArray
            A new state array with components converted back to their original frames.
        """
        restored_state = []
        for i, param in enumerate(propagated_state.state):
            name, size, est_type, dyn_type, body, awf = param

            # Get the original frame
            original_frame = self._original_frames.get(i, self._base_frame)

            # Convert back only if needed
            if name == "position" or name == "velocity":
                if awf.frame != original_frame:
                    awf.convert_to(original_frame, self._t0)

            restored_state.append((name, size, est_type, dyn_type, body, awf))

        # Reinitialize a new state vector with restored components
        return StateArray(
            epoch=propagated_state.epoch,
            origin=self._prop_origin,
            state=StateDefinition.from_components(restored_state),
        )

    def _precompute_indices(self):
        """
        Precompute and return indices for position, reduced state, and full state vectors. In addition,
        precompute which rows of the State Transition Matrix (STM) correspond to dynamic derivatives.

        :returns: A tuple of dictionaries containing precomputed indices.
        """
        reduced_indices_map = {
            (param[0], param[-1].spice_id): self._reduced_state_vector.find_indices(
                param[0], param[-1], flag_dot=False
            )
            for param in self._reduced_state_vector.state_definition
        }

        reduced_indices_dot_map = {
            (param[0], param[-1].spice_id): self._reduced_state_vector.find_indices(
                param[0], param[-1], flag_dot=True
            )
            for param in self._reduced_state_vector.state_dot_definition
        }

        indices_map = {
            (param[0], param[-1].spice_id): self.full_state_vector.find_indices(
                param[0], param[-1], flag_dot=False
            )
            for param in self.full_state_vector.state_definition
        }

        indices_dot_map = {
            (param[0], param[-1].spice_id): self.full_state_vector.find_indices(
                param[0], param[-1], flag_dot=True
            )
            for param in self.full_state_vector.state_dot_definition
        }

        # Precompute which STM rows correspond to dynamic derivatives
        stm_dyn_rows = []
        for param_dot in self.full_state_vector.state_dot_definition:
            name, _, _, dynamics_type, body = param_dot
            if dynamics_type != "dynamic":
                continue
            row_start, row_end = indices_dot_map.get(
                (name, body.spice_id), (None, None)
            )
            if None in (row_start, row_end):
                raise ValueError(
                    f"STM row indices not found for dynamic derivative '{name}' and body {body}."
                )
            stm_dyn_rows.extend(range(row_start, row_end))

        stm_dynamic_rows = np.array(stm_dyn_rows, dtype=int)

        return (
            reduced_indices_map,
            reduced_indices_dot_map,
            indices_map,
            indices_dot_map,
            stm_dynamic_rows,
        )

    def _precompute_ode_cache(self) -> None:
        """
        Build per-integration caches used inside ode() on every step.

        Called once from __init__ and reinitialize() so that ode() itself
        only does work that changes with the state (force evaluations, matmuls).
        """
        (
            reduced_indices_map,
            reduced_indices_dot_map,
            indices_map,
            indices_dot_map,
            _,
        ) = self._precompute_indices()

        self._n_full = self._full_state_vector.size
        self._n_red = self._reduced_state_vector.size
        primary_sid = getattr(self._primary_body, "spice_id", None)

        # Position unit per body spice_id (avoids linear scan every ODE call)
        self._pos_unit_cache = {}
        for name, _, _, _, body, template in self._full_state_vector.state:
            if name == "position":
                self._pos_unit_cache[getattr(body, "spice_id", None)] = (
                    template.quantity.units
                )
        fallback_unit = self._pos_unit_cache.get(primary_sid)

        # Force model per body spice_id (avoids linear scan every ODE call)
        self._fm_cache = {}
        all_dots = list(self._full_state_vector.state_dot_definition) + list(
            self._reduced_state_vector.state_dot_definition
        )
        for p in all_dots:
            body = p[-1]
            sid = getattr(body, "spice_id", None)
            if sid not in self._fm_cache:
                try:
                    self._fm_cache[sid] = self._get_force_model_for_body(body)
                except (ValueError, TypeError):
                    self._fm_cache[sid] = None

        # Red → full index mapping as a flat list (avoids dict.get per ODE call)
        self._red_to_full_slices = []
        for key, (i0r, i1r) in reduced_indices_map.items():
            i0f, i1f = indices_map.get(key, (None, None))
            if None not in (i0f, i1f):
                self._red_to_full_slices.append((i0r, i1r, i0f, i1f))

        # Dynamics plan — one entry per reduced state_dot that is "dynamic".
        # Tuple layout:
        #   velocity    : ("velocity",    s, e, vel_s, vel_e)
        #   acceleration: ("acceleration",s, e, pos_s, pos_e, pos_unit, fm, body)
        #   other param : ("param",       s, e, pos_s, pos_e, pos_unit, fm, body, name)
        self._dyn_plan = []
        for param_dot in self._reduced_state_vector.state_dot_definition:
            name, _, _, dynamics_type, body = param_dot
            if dynamics_type != "dynamic":
                continue
            sid = getattr(body, "spice_id", None)
            pos_s, pos_e = reduced_indices_map.get(("position", sid), (None, None))
            s, e = reduced_indices_dot_map.get((name, sid), (None, None))
            pos_unit = self._pos_unit_cache.get(sid) or fallback_unit
            fm = self._fm_cache.get(sid)
            if name == "velocity":
                vel_s, vel_e = reduced_indices_map.get(("velocity", sid), (None, None))
                self._dyn_plan.append(("velocity", s, e, vel_s, vel_e))
            elif name == "acceleration":
                self._dyn_plan.append(
                    ("acceleration", s, e, pos_s, pos_e, pos_unit, fm, body)
                )
            else:
                self._dyn_plan.append(
                    ("param", s, e, pos_s, pos_e, pos_unit, fm, body, name)
                )

        # STM plan — one row-block entry per dynamic state_dot of the full state.
        # col_entries: list of (partial_name_str, col_s, col_e) — strings prebuilt here.
        self._stm_plan = []
        for param_dot in self._full_state_vector.state_dot_definition:
            dot_name, _, _, dynamics_type, dot_body = param_dot
            if dynamics_type != "dynamic":
                continue
            dot_sid = getattr(dot_body, "spice_id", None)
            row_s, row_e = indices_dot_map.get((dot_name, dot_sid), (None, None))
            pos_s, pos_e = reduced_indices_map.get(("position", dot_sid), (None, None))
            if None in (pos_s, pos_e):
                pos_s, pos_e = reduced_indices_map.get(
                    ("position", primary_sid), (None, None)
                )
            pos_unit = self._pos_unit_cache.get(dot_sid) or fallback_unit
            fm = self._fm_cache.get(dot_sid)
            col_entries = []
            for param in self._full_state_vector.state_definition:
                var_name = param[0]
                var_body = param[-1]
                var_sid = getattr(var_body, "spice_id", None)
                col_s, col_e = indices_map.get((var_name, var_sid), (None, None))
                if None not in (col_s, col_e):
                    pname = f"{dot_name}@{dot_sid}_by_{var_name}@{var_sid}"
                    col_entries.append((pname, col_s, col_e))
            self._stm_plan.append(
                (row_s, row_e, pos_s, pos_e, pos_unit, fm, dot_body, col_entries)
            )

        # Cache env-var flag so ode() doesn't hit os.environ every step
        self._check_partials = os.environ.get("check_force_model_partials") == "TRUE"

    def _get_force_model_for_body(self, body) -> ForceModel:
        """
        Select the ForceModel instance responsible for the given body.

        Priority:
        1. A model whose `primary_body` or `main_spacecraft` has the same spice_id.

        Raises
        ------
        ValueError
            If no suitable ForceModel is found for the given body,
            except when `body` is a GroundStation (force model not applicable).
        """
        if body is None:
            raise ValueError("Cannot select ForceModel for body=None.")

        # Skip GroundStations silently — no force model is expected
        if isinstance(body, GroundStation):
            return None

        body_spice_id = getattr(body, "spice_id", None)

        for fm in self._force_models:
            primary = getattr(fm, "primary_body", None) or getattr(
                fm, "main_spacecraft", None
            )
            if primary is not None:
                if getattr(primary, "spice_id", None) == body_spice_id:
                    return fm

        raise ValueError(
            f"No ForceModel configured for body {getattr(body, 'name', body)} "
            f"(spice_id={body_spice_id}). "
            "Make sure you created a ForceModel for this object and added it to Propagator(force_models=...)."
        )

    def ode(
        self,
        epoch: float,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the time derivative of the state and STM for numerical integration.

        Y layout: [y_reduced, vec(Phi_full)].
        All static loop structure is precomputed in _precompute_ode_cache().
        """
        Y = np.asarray(Y)
        Y_dot = np.zeros_like(Y)

        n_full = self._n_full
        n_red = self._n_red

        y_state = Y[:n_red]
        stm_flat = Y[n_red:]

        # Expand reduced → full state using precomputed slice list
        y_full = self._y_full0.copy()
        for i0r, i1r, i0f, i1f in self._red_to_full_slices:
            y_full[i0f:i1f] = y_state[i0r:i1r]
        full_state = self.full_state_vector._make_direct_representation(y_full)

        # ============================================================
        # 1) Dynamics of reduced state  (plan built once in __init__)
        # ============================================================
        position = None  # kept for the optional partials check below
        body = None
        for entry in self._dyn_plan:
            kind = entry[0]
            s, e = entry[1], entry[2]
            if kind == "velocity":
                Y_dot[s:e] = y_state[entry[3] : entry[4]]
            else:
                pos_s, pos_e, pos_unit, fm, body = (
                    entry[3],
                    entry[4],
                    entry[5],
                    entry[6],
                    entry[7],
                )
                position = ArrayWUnits(y_state[pos_s:pos_e], pos_unit)
                if kind == "acceleration":
                    Y_dot[s:e] = fm._compute_total_acceleration(
                        position=position,
                        current_state=full_state,
                        epoch=epoch,
                        body=body,
                    )
                else:  # "param"
                    dyn = fm._collect_parameter_dynamics(
                        name=entry[8],
                        position=position,
                        current_state=full_state,
                        epoch=epoch,
                        body=body,
                    )
                    Y_dot[s:e] = dyn.values if hasattr(dyn, "values") else dyn

        # ============================================================
        # 2) STM propagation  (plan built once in __init__)
        # ============================================================
        if self._propagate_STM:
            STM_current = stm_flat.reshape((n_full, n_full))
            STM_dot = np.zeros((n_full, n_full))

            for (
                row_s,
                row_e,
                pos_s,
                pos_e,
                pos_unit,
                fm,
                dot_body,
                col_entries,
            ) in self._stm_plan:
                if fm is None:
                    continue
                position = ArrayWUnits(y_state[pos_s:pos_e], pos_unit)
                for pname, col_s, col_e in col_entries:
                    partial = fm._collect_partials(
                        name=pname,
                        position=position,
                        current_state=full_state,
                        epoch=epoch,
                        body=dot_body,
                    )
                    if partial is None:
                        continue
                    block = np.atleast_2d(
                        np.asarray(
                            partial.values if hasattr(partial, "values") else partial,
                            dtype=float,
                        )
                    )
                    if block.shape != (row_e - row_s, col_e - col_s):
                        raise ValueError(
                            f"Partial '{pname}' returned shape {block.shape}, "
                            f"expected ({row_e - row_s}, {col_e - col_s})."
                        )
                    STM_dot[row_s:row_e, col_s:col_e] = block

            # dPhi/dt = A * Phi
            Y_dot[n_red:] = (STM_dot @ STM_current).ravel()

            # accumulate A-matrix into a list (converted to array at end of propagate)
            self._As_list.append(STM_dot)

        # Optional partial-derivative checking (flag cached; no env lookup per step)
        if self._check_partials:
            for fm in self._force_models:
                fm._check_all_partials(position, full_state, epoch, body)

        return Y_dot

    def propagate(self, display_progress: bool | None = None) -> None:
        """
        Propagates the state of the system using numerical integration.

        This method integrates the ordinary differential equations (ODEs) that describe the system's dynamics
        using the specified integration method. The integration is performed over the time span defined by
        the initial and final times of the system's time vector.

        Returns
        -------
            None
        """
        # notify beginning propagation if propagator setting is true
        if display_progress is not None:
            self._display_progress = display_progress
        if self._display_progress:
            print("")
            print("=" * 80)
            print("Starting propagation...".center(80))
            print("=" * 80)

        # Reset A-matrix accumulation list for this run
        self._As_list = []

        # Handle dynamic mass initialization
        self._initialize_mass_profile()

        fun_ode = lambda t, y: self.ode(epoch=t, Y=y)

        # condition time span for selected integrator
        match self.tspan:
            case EpochArray():
                dt = (self.tspan.times[1] - self.tspan.times[0]).values
                tspan_to_integrator = (
                    self.tspan.times.values
                )  # convert to array of floats
            case tuple():
                dt = (self.tspan[1].times - self.tspan[0].times).values
                # convert to tuple of floats
                tspan_to_integrator = (
                    self.tspan[0].times.values,
                    self.tspan[1].times.values,
                )

        # Integrate the ODE
        match self.integrator:
            case "IAS15":
                # input handling
                if self.full_state_vector.size != self._reduced_state_vector.size:
                    raise ValueError(
                        "IAS15 integration is currently only supported "
                        "for full state vector propagation."
                    )

                correct_identity = np.identity(self.full_state_vector.size).reshape(
                    self.full_state_vector.size**2
                )
                if not np.array_equal(
                    self._STM0,
                    correct_identity,
                ):
                    raise ValueError(
                        "IAS15 integration is not currently supported "
                        "unless the initial STM is an identity matrix."
                    )

                # initialize and integrate
                ias15 = IAS15(dt, 0, fun_ode, self.full_state_vector.size)
                ts, ys = ias15.integrate(tspan_to_integrator, self.ys[0][0:6])

                # Store the solution
                dynamic_dims = [
                    self.full_state_vector.state[i][1] for i in self._dynamic_indices
                ]
                total_dynamic_dim = sum(dynamic_dims)

                # Extract propagated dynamic state and STM
                propagated_dynamic_state = np.array(ys)[:, :total_dynamic_dim]
                propagated_stm = np.array(ys)[:, total_dynamic_dim:]

                # notify end of propagation if propagator setting is true
                if self._display_progress:
                    print(
                        " =================== IAS15 integration complete. =================="
                    )

            case "DOP853":
                # condition tspan for scipy integrator
                match self.tspan:
                    case EpochArray():
                        t_start = self.tspan.times[0].values
                        t_end = self.tspan.times[-1].values

                        t_eval = self.tspan.times.values
                    case tuple():
                        raise ValueError(
                            "Non-selective time steps currently "
                            "unsupported for DOP853 integration. "
                            "Provide an EpochArray instead of tuple."
                        )
                        # t_start = self.tspan[0].times.values
                        # t_end   = self.tspan[1].times.values

                        # t_eval = None

                total_time = abs(t_end - t_start)

                with tqdm(
                    total=total_time,
                    desc="Integrating",
                    unit="s",
                    ncols=100,
                    disable=not self._display_progress,
                    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} {unit} [{elapsed}<{remaining}]",
                ) as pbar:

                    def progress_monitor(t, y):
                        if self._display_progress:
                            delta_t = abs(float(t - t_start))
                            pbar.n = delta_t
                            pbar.set_postfix({"∆t": f"{delta_t:.2f} s"}, refresh=False)
                            pbar.refresh()
                        return 0

                    # Get batch boundaries if piecewise GM is active
                    batch_breakpoints = []
                    for fm in self._force_models:
                        if (
                            hasattr(fm, "pfogm_batch_length")
                            and fm.pfogm_batch_length is not None
                            and hasattr(fm, "pfogm_n_batches")
                            and hasattr(fm, "t0")
                        ):
                            for k in range(1, fm.pfogm_n_batches):
                                bp = fm.t0.times.values + k * fm.pfogm_batch_length
                                if t_start < bp < t_end:
                                    batch_breakpoints.append(bp)
                            break

                    # Build arc boundaries: [t_start, breakpoint_1, ..., t_end]
                    arc_boundaries = sorted(
                        set([t_start] + batch_breakpoints + [t_end]),
                        reverse=self._backward,
                    )

                    # Integrate arc by arc
                    y_current = self.ys[0]
                    all_t = []
                    all_y = []

                    # Small time offset to avoid integrating exactly onto an internal discontinuity
                    eps_time = max(
                        1e-12,
                        10 * np.finfo(float).eps * max(1.0, abs(t_start), abs(t_end)),
                    )

                    for i in range(len(arc_boundaries) - 1):
                        t_arc_start = arc_boundaries[i]
                        t_arc_end = arc_boundaries[i + 1]
                        is_last_arc = i == len(arc_boundaries) - 2

                        # Avoid duplicate evaluation at internal boundaries
                        if self._backward:
                            if not is_last_arc:
                                t_arc_eval = t_eval[
                                    (t_eval <= t_arc_start) & (t_eval > t_arc_end)
                                ]
                                t_span_end = t_arc_end + eps_time
                            else:
                                t_arc_eval = t_eval[
                                    (t_eval <= t_arc_start) & (t_eval >= t_arc_end)
                                ]
                                t_span_end = t_arc_end
                        else:
                            if not is_last_arc:
                                t_arc_eval = t_eval[
                                    (t_eval >= t_arc_start) & (t_eval < t_arc_end)
                                ]
                                t_span_end = t_arc_end - eps_time
                            else:
                                t_arc_eval = t_eval[
                                    (t_eval >= t_arc_start) & (t_eval <= t_arc_end)
                                ]
                                t_span_end = t_arc_end

                        # Skip degenerate arcs
                        if (self._backward and t_span_end >= t_arc_start) or (
                            not self._backward and t_span_end <= t_arc_start
                        ):
                            continue

                        # If there are no requested output times in this arc, still advance the state
                        if len(t_arc_eval) == 0:
                            self._ode_sol = solve_ivp(
                                fun=fun_ode,
                                t_span=(t_arc_start, t_span_end),
                                y0=y_current,
                                method=self.integrator,
                                rtol=self.rtol,
                                atol=self.atol,
                                dense_output=True,
                                events=progress_monitor,
                            )

                            if not self._ode_sol.success:
                                raise RuntimeError(
                                    f"ODE integration failed at arc [{t_arc_start}, {t_arc_end}]: "
                                    f"{self._ode_sol.message}"
                                )

                            # Restart exactly at the next boundary for the next arc
                            restart_time = (
                                t_arc_end if not is_last_arc else self._ode_sol.t[-1]
                            )
                            y_current = self._ode_sol.sol(restart_time)
                            continue

                        self._ode_sol = solve_ivp(
                            fun=fun_ode,
                            t_span=(t_arc_start, t_span_end),
                            y0=y_current,
                            method=self.integrator,
                            t_eval=t_arc_eval,
                            rtol=self.rtol,
                            atol=self.atol,
                            dense_output=True,
                            events=progress_monitor,
                        )

                        if not self._ode_sol.success:
                            raise RuntimeError(
                                f"ODE integration failed at arc [{t_arc_start}, {t_arc_end}]: "
                                f"{self._ode_sol.message}"
                            )

                        # Restart exactly at the next boundary for the next arc
                        restart_time = (
                            t_arc_end if not is_last_arc else self._ode_sol.t[-1]
                        )
                        y_current = self._ode_sol.sol(restart_time)

                        # Concatenate outputs without duplicating the first sample of each later arc
                        if all_t:
                            all_t.append(self._ode_sol.t)
                            all_y.append(self._ode_sol.y)
                        else:
                            all_t.append(self._ode_sol.t)
                            all_y.append(self._ode_sol.y)

                    # Stitch arcs back together so the rest of propagate() works unchanged
                    self._ode_sol.t = numpy.concatenate(all_t)
                    self._ode_sol.y = numpy.concatenate(all_y, axis=1)

                    if self._display_progress:
                        pbar.n = total_time
                        pbar.set_postfix({"∆t": f"{total_time:.2f} s"})
                        pbar.refresh()

                # Post-check
                if not self._ode_sol.success:
                    raise RuntimeError(
                        f"ODE integration failed: {self._ode_sol.message}"
                    )

                if self._display_progress:
                    print(
                        ""
                    )  # stops the progress monitor from overwriting whatever is printed next
                    print(
                        " =================== DOP853 integration complete. =================="
                    )

                # Check for event and handle it
                if self._ode_sol.status == 1 and self._ode_sol.t_events:
                    t_event = self._ode_sol.t_events[0][0]
                    y_event = self._ode_sol.y[:, self._ode_sol.t == t_event].flatten()
                    sol_event = solve_ivp(
                        fun=self.ode2,
                        t_span=(t_event, self.tspan.time[-1]),
                        y0=y_event,
                        method=self.integrator,
                        t_eval=self.tspan.time[self.tspan.time > t_event],
                        rtol=self.rtol,
                        atol=self.atol,
                        dense_output=False,
                    )
                    if sol_event:
                        # Combine the initial and resumed solutions
                        self._ode_sol.t = np.concatenate(
                            (self._ode_sol.t, sol_event.t[1:])
                        )
                        self._ode_sol.y = np.concatenate(
                            (self._ode_sol.y, sol_event.y[:, 1:]), axis=1
                        )

                # Flip to ascending time order for backward propagation so that
                # Trajectory.get_STM (which assumes ascending epochs) works correctly.
                if self._backward:
                    self._ode_sol.t = self._ode_sol.t[::-1]
                    self._ode_sol.y = self._ode_sol.y[:, ::-1]
                    self._As_list = self._As_list[::-1]

                # Extract propagated dynamic state and STM
                dynamic_dims = [
                    self.full_state_vector.state[i][1] for i in self._dynamic_indices
                ]
                total_dynamic_dim = sum(dynamic_dims)

                # Extract propagated dynamic state and STM
                propagated_dynamic_state = self._ode_sol.y.T[:, :total_dynamic_dim]
                propagated_stm = self._ode_sol.y.T[:, total_dynamic_dim:]

            case _:
                raise ValueError(
                    f"Selected integrator {self.integrator} is not recognized. "
                    "Expected IAS15 or DOP853."
                )

        # Convert A-matrix list to array once (avoids O(N²) copy inside ode())
        self._As = (
            numpy.stack(self._As_list)
            if self._As_list
            else numpy.zeros((0, self._n_full, self._n_full))
        )

        # Allocate the full state array
        full_state = np.zeros(
            (propagated_dynamic_state.shape[0], self.full_state_vector.size)
        )

        # Initialize pointers
        start_idx = 0  # Tracks the position in the propagated dynamic state
        cumulative_offset = 0  # Tracks the position in the full state vector

        # Iterate through all parameters in their natural order
        for i in range(len(self.full_state_vector.state)):
            if i in self._dynamic_indices:
                # Determine the dimension for this dynamic parameter
                dim = self.full_state_vector.state[i][1]

                # Assign dynamic state values
                full_state = np.set(
                    full_state,
                    (slice(None), slice(cumulative_offset, cumulative_offset + dim)),
                    (propagated_dynamic_state[:, start_idx : start_idx + dim]),
                )

                # Update the propagated dynamic state pointer
                start_idx += dim
            elif i in self._static_indices:
                # Determine the dimension for this static parameter
                dim = self.full_state_vector.state[i][1]

                # Assign static values
                full_state = np.set(
                    full_state,
                    (slice(None), slice(cumulative_offset, cumulative_offset + dim)),
                    (self._static_values[self._static_indices.index(i)]),
                )

            # Update the cumulative offset to account for this parameter
            cumulative_offset += dim

        # Update the class attributes with the full propagated state
        self._ys = np.concatenate((full_state, propagated_stm), axis=1)
        try:
            self._times = EpochArray(
                self._ode_sol.t, "TDB"
            )  # Try accessing self._ode_sol_t
        except AttributeError:
            self._times = EpochArray(
                ts, "TDB"
            )  # If AttributeError occurs, fallback to ts
        self._state = ArrayWFrame(
            ArrayWUnits(full_state, self._state_units), self.base_frame
        )
        if self.propagate_STM:
            self._STM = ArrayWFrame(
                ArrayWUnits(propagated_stm, self._STM_units), self.base_frame
            )

        # Create the propagated state array
        state_components = []
        col_offset = 0  # Tracks column index in full_state

        for (
            name,
            size,
            est_type,
            dyn_type,
            body,
            value_template,
        ) in self.full_state_vector.state:
            data = (
                full_state[:, col_offset : col_offset + size][0]
                if dyn_type == "static"
                else full_state[:, col_offset : col_offset + size]
            )

            component = (
                name,
                size,
                est_type,
                dyn_type,
                body,
                ArrayWFrame(
                    ArrayWUnits(data, value_template.quantity.units),
                    value_template.frame,
                ),
            )
            state_components.append(component)
            col_offset += size

        propagated_state_array = StateArray(
            epoch=self._times,
            origin=self._prop_origin,
            state=StateDefinition.from_components(state_components),
        )

        # Store the propagated state array as an attribute
        self._propagated_state_array = propagated_state_array

        # Convert the propagated state back to the original frames
        if self._convert_to_original_frame:
            # Preallocate list
            propagated_states_original = [None] * self.tspan.size

            # Convert full_state to a NumPy array for efficient indexing
            full_state_array = np.array(full_state)
            state_definitions = self.full_state_vector.state  # Extract once
            original_frames = self._original_frames  # Avoid repeated attribute lookups

            # Create epoch array for frame transformations
            epochs_array = self.tspan_original_frame

            # Extract state components for all time steps in one pass
            for i, epoch in enumerate(self.tspan):
                # print(i)
                current_state_values = full_state_array[i]  # Extract state at time `i`
                state_components = []

                start_index = 0
                for idx, (
                    name,
                    size,
                    est_type,
                    dyn_type,
                    body,
                    value_template,
                ) in enumerate(state_definitions):
                    component_values = current_state_values[
                        start_index : start_index + size
                    ]
                    start_index += size

                    original_frame = original_frames[idx]

                    # Wrap with ArrayWFrame
                    component_AWF = ArrayWFrame(
                        component_values,
                        value_template.quantity.units,
                        original_frame if original_frame else self.base_frame,
                    )

                    # Convert component back to original frame if necessary
                    if original_frame:
                        if self.base_frame.name != original_frame.name:
                            component_AWF.convert_to(original_frame, epochs_array[i])

                    state_components.append(
                        (name, size, est_type, dyn_type, body, component_AWF)
                    )

                # Create final StateArray with converted components
                propagated_states_original[i] = StateArray(
                    epoch=epoch, origin=self.origin, state=state_components
                )

            # Store in class
            self._propagated_state_array_original_frame = propagated_states_original

            if propagated_stm is True:
                raise NotImplementedError(
                    "STM conversion to original frames not yet implemented."
                )

        # notify completion of propagation if propagator setting is true
        if self._display_progress:
            print("Propagation complete.")

    def _get_event_function(self):
        """
        Returns the appropriate event function based on the specified event_type.

        Returns:
            function: Event function to be passed to the integrator.
        """
        if self._look_event == "EARTH_SOI":

            def soi_event(t, Y):
                """
                Detects entry into the Earth's Sphere of Influence (SOI).
                """
                # Retrieve position indices dynamically from the state vector
                position_indices = self.full_state_vector.find_indices(
                    "position", self.origin, flag_dot=False
                )
                if position_indices is None:
                    raise ValueError(
                        "Position indices not found in the state vector for SOI calculation."
                    )

                # Extract position using the indices
                pos_start, pos_end = position_indices
                position = Y[
                    pos_start:pos_end
                ]  # Position of the object with respect to the origin

                # Compute the distance from the origin to the object
                distance_from_origin = np.linalg.norm(position)

                # Calculate the distance from Earth:
                # Earth distance = distance_from_origin + distance of the origin from Earth
                if self.origin["spice_name"] == "EARTH":
                    origin_to_earth = 0  # Distance is zero if origin is Earth
                else:
                    # Get the state vector of the origin with respect to Earth
                    state, _ = SpiceManager.get_state(
                        target=self.origin["spice_name"],
                        et=t,  # Current epoch in ephemeris time
                        ref=self.ref_frame,  # Reference frame
                        abcorr="NONE",  # No light-time corrections
                        obs="EARTH",  # Observer is Earth
                    )
                origin_to_earth = np.linalg.norm(
                    state[:3]
                )  # Use only the position components
                total_distance_to_earth = distance_from_origin + origin_to_earth

                # Get SOI parameters for Earth
                sun_mass = constants.SUN.mass.values
                earth_mass = constants.EARTH.mass.values
                # earth_sun_distance = Constants.SUN["average_distances"]["Earth"].values  # Average Earth-Sun distance
                # NOTE: this constant doesn't exist? setting to 1 AU for now
                AU = Units.get_units("AU")
                earth_sun_distance = ArrayWUnits(1, AU).convert_to(km)

                # Calculate Sphere of Influence (SOI) radius
                soi_radius = earth_sun_distance * (earth_mass / sun_mass) ** (2 / 5)

                # Event condition: difference to SOI radius
                return total_distance_to_earth - soi_radius

            soi_event.terminal = True
            soi_event.direction = (
                -1
            )  # Detect when entering the SOI (decreasing distance)
            return soi_event

        elif self._look_event is None:
            return None
        else:
            raise ValueError(f"Unsupported event type: {self._look_event}")

    def reinitialize(
        self,
        primary_body: Body | None = None,
        state_vector: StateArray | None = None,
        tspan: EpochArray | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        integrator: str | None = None,
        propagate_STM: bool | None = None,
        STM0: np.ndarray | None = None,
        look_event: str | None = None,
        base_frame: Frame | None = None,
        convert_to_original_frame: bool | None = None,
        maneuver: Maneuver = None,
        force_models: Optional[Union[ForceModel, Sequence[ForceModel]]] = None,
        backward: bool | None = None,
    ) -> "Propagator":
        """
        Reconfigure the propagator with updated inputs and rebuild its internal state.

        Any argument left as ``None`` is left unchanged.

        Parameters
        ----------
        primary_body : Body, optional
            Updated primary propagated object.
        state_vector : StateArray, optional
            Updated initial state. If provided, internal copies and t0 are updated.
        tspan : EpochArray, optional
            Updated propagation timespan. Converted to TDB internally if needed.
        atol, rtol : float, optional
            Updated absolute / relative tolerances.
        integrator : str, optional
            Updated integrator identifier.
        propagate_STM : bool, optional
            Whether to propagate the STM.
        STM0 : np.ndarray, optional
            Updated initial STM (flattened or matrix); reshaped internally.
        look_event : str, optional
            Updated event specification (WIP hook).
        base_frame : Frame, optional
            Updated base frame for propagation.
        convert_to_original_frame : bool, optional
            Updated flag to map results back to original frame after propagation.
        maneuver : Maneuver, optional
            Updated maneuver model; propagated into compatible force models.
        force_model : ForceModel or sequence of ForceModel, optional
            Replace the current force model set with the provided one(s).
            Each will be re-initialized against the (possibly updated) state.

        Returns
        -------
        Propagator
            The updated propagator instance (for chaining).
        """

        # Update primary spacecraft
        if primary_body is not None:
            self._primary_body = primary_body

        # Update state vector
        if state_vector is not None:
            self._full_state_vector_original_frames = copy.deepcopy(state_vector)
            self._full_state_vector = copy.deepcopy(state_vector)
            self._prop_origin = state_vector.origin
            self._t0 = state_vector.epoch

        # Update tspan
        if tspan is not None:
            self._tspan_original_frame = copy.deepcopy(tspan)
            self._tspan = (
                copy.deepcopy(tspan).to(sys="TDB")
                if tspan.system != "TDB"
                else copy.deepcopy(tspan)
            )

        # If t0 not set yet (e.g. only tspan passed), try to infer from tspan
        if getattr(self, "_t0", None) is None and hasattr(self, "_tspan"):
            try:
                self._t0 = self._tspan[0]
            except Exception:
                pass

        # Update backward flag
        if backward is not None:
            self._backward = backward

        # Update maneuver (propagate into force models later)
        if maneuver is not None:
            self._maneuver = maneuver

        # Update numerics / options
        if atol is not None:
            self._atol = atol
        if rtol is not None:
            self._rtol = rtol
        if integrator is not None:
            self._integrator = integrator
        if propagate_STM is not None:
            self._propagate_STM = propagate_STM
        if look_event is not None:
            self._look_event = look_event
        if base_frame is not None:
            self.base_frame = base_frame
        if convert_to_original_frame is not None:
            self._convert_to_original_frame = convert_to_original_frame

        # Update STM0
        if STM0 is not None:
            STM0 = np.array(STM0)
            n_state = self._full_state_vector.size
            self._STM0 = STM0.reshape(n_state**2)

        # Replace force models if requested
        if force_models is not None:
            if isinstance(force_models, ForceModel):
                self._force_models = [force_models]
            else:
                fm_list = list(force_models)
                if not fm_list:
                    raise ValueError(
                        "force_model list provided to reinitialize is empty."
                    )
                if not all(isinstance(fm, ForceModel) for fm in fm_list):
                    raise TypeError(
                        "All entries in force_model must be instances of ForceModel."
                    )
                self._force_models = fm_list

        # Ensure we have force models
        if not getattr(self, "_force_models", None):
            raise RuntimeError(
                "No ForceModel instances available in Propagator. "
                "Provide at least one via constructor or reinitialize()."
            )

        # Push maneuver into relevant force models (if any)
        if getattr(self, "_maneuver", None) is not None:
            for fm in self._force_models:
                if hasattr(fm, "maneuver"):
                    fm.maneuver = self._maneuver

        # Re-bind all force models to the (possibly updated) state & frame
        for fm in self._force_models:
            fm.initialize_force_models(
                state_vector=self._full_state_vector,
                base_frame=self.base_frame,
            )

        # Internal frame work & consistency checks
        self._convert_to_base_frame()
        self._validate_configuration()

        # Rebuild reduced state & integration arrays
        self._create_reduced_state_vector()

        n_state = self._full_state_vector.size
        if not hasattr(self, "_STM0"):
            # If STM0 not set yet, default to identity
            self._STM0 = np.identity(n_state).reshape(n_state**2)

        # Resize integration storage
        self._ys = np.zeros(
            (self.tspan.size, self._reduced_state_vector.size + self._STM0.size)
        )
        self._As = np.zeros((0, n_state, n_state))
        self._As_list = []
        self._times = np.zeros((self.tspan.size, 1))

        # Initial condition: reduced state at t0 + STM0
        self._ys[0] = np.concatenate(
            (self._reduced_state_vector.values0.values, self._STM0)
        )

        # Refresh units / frames metadata
        self._state_units, self._state_frames = (
            self._accumulate_and_save_units_and_frames()
        )
        self._STM_units = self._accumulate_and_save_stm_units()
        self._STM_frame = self._accumulate_and_save_frame_ratios()

        self._precompute_ode_cache()

        return self

    def to_dict(self) -> dict:
        """
        Serialize the Propagator configuration to a dictionary.

        Notes
        -----
        - The state and tspan are stored in their original frames as provided
          at construction time.
        - All attached force models are serialized via their own ``to_dict``
          method if available; otherwise only their class name is stored.
        """
        force_models_data = []
        for fm in getattr(self, "_force_models", []):
            if hasattr(fm, "to_dict"):
                force_models_data.append(
                    {
                        "class": fm.__class__.__name__,
                        "config": fm.to_dict(),
                    }
                )
            else:
                force_models_data.append(
                    {
                        "class": fm.__class__.__name__,
                        "config": None,
                    }
                )

        return {
            "primary_body": (
                self._primary_body.name
                if getattr(self, "_primary_body", None) is not None
                else None
            ),
            "base_frame": self.base_frame.name if hasattr(self, "base_frame") else None,
            "convert_to_original_frame": self._convert_to_original_frame,
            "state_vector": self._full_state_vector_original_frames.to_dict(),
            "tspan": self._tspan_original_frame.to_dict(),
            "atol": self._atol,
            "rtol": self._rtol,
            "integrator": self._integrator,
            "propagate_STM": self._propagate_STM,
            "STM0": self._STM0.tolist() if hasattr(self, "_STM0") else None,
            "look_event": self._look_event,
            "mass_is_variable": self._mass_is_variable,
            "force_models": force_models_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Propagator":
        """
        Reconstruct a Propagator instance from a dictionary produced by ``to_dict``.

        This expects that each serialized force model entry provides enough
        information (via `"class"` and `"config"`) to reconstruct the
        corresponding :class:`ForceModel` subclass.

        Parameters
        ----------
        data : dict
            Serialized propagator configuration.

        Returns
        -------
        Propagator
            A new Propagator instance with reconstructed state, time span,
            and force models.
        """
        # --- Core objects ---
        primary_name = data.get("primary_body")
        primary_body = Spacecraft(primary_name) if primary_name is not None else None

        base_frame_name = data.get("base_frame")
        base_frame = Frame(base_frame_name) if base_frame_name is not None else J2000

        state_vector = StateArray.from_dict(data["state_vector"])
        tspan = EpochArray.from_dict(data["tspan"])

        # --- Rebuild force models ---
        force_models_cfg = data.get("force_models", [])
        force_models = []

        for fm_info in force_models_cfg:
            cls_name = fm_info.get("class")
            cfg = fm_info.get("config", None)

            # You can expand this mapping as you add more subclasses.
            if cls_name == "ForceModelTranslation":
                if hasattr(ForceModelTranslation, "from_dict") and cfg is not None:
                    fm = ForceModelTranslation.from_dict(cfg)
                else:
                    # Minimal fallback: require primary body and basic flags in cfg
                    raise ValueError(
                        "ForceModelTranslation config missing or from_dict not implemented."
                    )
            elif cls_name == "ForceModelRotation":
                fm = ForceModelRotation()
            else:
                raise ValueError(
                    f"Unknown or unsupported ForceModel subclass '{cls_name}' "
                    "in Propagator.from_dict."
                )

            force_models.append(fm)

        if not force_models:
            raise ValueError(
                "No force models found in serialized data. "
                "At least one ForceModel is required."
            )

        # --- STM0 reconstruction ---
        stm0_arr = data.get("STM0")
        STM0 = np.array(stm0_arr) if stm0_arr is not None else None

        # --- Instantiate Propagator ---
        prop = cls(
            primary_body=primary_body,
            state_vector=state_vector,
            tspan=tspan,
            force_model=force_models,
            base_frame=base_frame,
            atol=data.get("atol", 2.220446049250313e-14),
            rtol=data.get("rtol", 2.220446049250313e-14),
            integrator=data.get("integrator", "DOP853"),
            propagate_STM=data.get("propagate_STM", True),
            STM0=STM0,
            look_event=data.get("look_event"),
            convert_to_original_frame=data.get("convert_to_original_frame", False),
            mass_is_variable=data.get("mass_is_variable", False),
        )

        return prop

    def _sync_fm_spacecraft_mass_profile(self) -> None:
        """Push the primary body's current mass profile into FiniteBurn sub-models."""
        updated_profile = self._primary_body.mass_profile
        for fm in self._force_models:
            inner_models = getattr(fm, "force_models", {})
            fb = (
                inner_models.get("FiniteBurn")
                if isinstance(inner_models, dict)
                else None
            )
            if fb is not None and hasattr(fb, "_spacecraft"):
                fb._spacecraft._mass_profile = updated_profile

    def _initialize_mass_profile(self):
        """
        Initialize or update the mass profile of the primary spacecraft.

        If the associated force model defines a maneuver (finite burn with
        prescribed mass flow), integrate m(t) over the propagation timespan.
        Otherwise, assume constant mass over [t0, tf] and fit a trivial profile.
        """

        # Short-hands for times
        match self._tspan:
            case EpochArray():
                # EpochArray configuration
                t0 = self._tspan.times.values[0]
                tf = self._tspan.times.values[-1]
            case tuple():
                # tuple of t0, tf EpochArrays configuration
                t0 = self._tspan[0].times.values
                tf = self._tspan[1].times.values

        # Select the relevant force model for the primary spacecraft
        # Prefer a model explicitly tied to this spacecraft/body if available.
        primary_fm = None
        for fm in self._force_models:
            if getattr(fm, "primary_body", None) is self._primary_body:
                primary_fm = fm
                break

        # Fallback: use the first model if none explicitly matches
        if primary_fm is None:
            primary_fm = self._force_models[0]

        maneuver = getattr(primary_fm, "maneuver", None)

        # Case 1: maneuver present -> integrate mass depletion over t
        if maneuver is not None:
            # # Very temporary guard, as in your original code
            # if len(self._tspan.times.values) < 100:
            #     raise ValueError(
            #         "Insufficient temporal sampling for a smooth 6th order "
            #         "polynomial-based mass flow fit."
            #     )

            # Initial total mass at t0
            m0 = self._primary_body.mass(t0)

            def mass_dynamics(t, m):
                # maneuver.mass_flow assumed as polynomial coefficients
                poly = np.poly1d(maneuver.mass_flow.values[::-1])
                m_dot = poly(t)
                return -m_dot

            sol = solve_ivp(
                mass_dynamics,
                t_span=(t0, tf),
                y0=[m0.values],
                t_eval=self._tspan.times.values,
                method=self._integrator,
                rtol=self._rtol,
                atol=self._atol,
                dense_output=False,
            )

            if not sol.success:
                raise RuntimeError(
                    f"Mass integration failed during maneuver-driven profile: {sol.message}"
                )

            # Fit/update spacecraft mass profile from integrated samples
            self._primary_body.fit_mass_profile_from_data(
                t_array=sol.t,
                m_array=sol.y[0],
                domain=(t0, tf),
                units=m0.units,
            )
            self._mass_profile = self._primary_body.mass_profile
            self._sync_fm_spacecraft_mass_profile()

        # Case 2: no maneuver -> constant mass profile over [t0, tf]
        else:
            t_array_span = np.linspace(t0, tf, 60)

            # Get mass directly from stored profile to avoid range check
            mass_profile = self._primary_body.mass_profile
            if isinstance(mass_profile, ArrayWUnits):
                # constant mass stored directly
                m_val = float(np.asarray(mass_profile.values).ravel()[0])
                m_units = mass_profile.units
            else:
                # polynomial — read value at any valid stored time
                t_valid = float(self._primary_body._mass_profiles_tspan[-1][0])
                poly = self._primary_body._mass_profiles[-1]
                m_val = float(poly(t_valid))
                m_units = self._primary_body._mass_units

            m_array = np.full_like(t_array_span, m_val, dtype=float)

            self._primary_body.fit_mass_profile_from_data(
                t_array=t_array_span,
                m_array=m_array,
                domain=(t0, tf),
                units=m_units,
            )
            self._mass_profile = self._primary_body.mass_profile
