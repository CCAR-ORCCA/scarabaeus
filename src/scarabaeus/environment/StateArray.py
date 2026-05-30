# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    ArrayWUnits,
    ImpulsiveBurn,
    ArrayWFrame,
    Body,
    EpochArray,
    Spacecraft,
    GroundStation,
    StateDefinition,
)

import scarabaeus.utils.NumpyWrapper as np
import numpy
import re
import copy


class StateArray:
    """ Container for an estimation state vector at a given epoch and origin.

    The full estimation state is a stacked vector of kinematic components
    and scalar/vector parameters for one or more bodies:

    .. math::

        \\mathbf{x} =
        \\begin{bmatrix}
            \\mathbf{r}_1 \\\\ \\dot{\\mathbf{r}}_1 \\\\
            \\vdots \\\\
            \\mathbf{r}_N \\\\ \\dot{\\mathbf{r}}_N \\\\
            p_1 \\ \\cdots \\ p_M
        \\end{bmatrix}

    where :math:`\\mathbf{r}_i`, :math:`\\dot{\\mathbf{r}}_i` are the
    position and velocity of body :math:`i`, and :math:`p_j` are
    scalar parameters (static or dynamic, estimated or considered).

    This class holds a *definition* and corresponding *values* for a mixed state:
    spacecraft/body states (e.g., position/velocity) and scalar/vector parameters
    (e.g., SRP scale, gravity terms). The state definition is provided via a
    :class:`~scarabaeus.StateDefinition` builder, which yields the canonical tuple encoding
    expected internally.

    Parameters
    ----------
    epoch : EpochArray
        Epoch for which the state vector is defined (time tag of this state).
    origin : Body
        Origin body with respect to which kinematic components (e.g., position/velocity) are expressed.
    state : StateDefinition
        Fluent state builder. When iterated, it produces a sequence of 6-tuples:

        ``(name: str, dim: int, estimation: str, dynamics: str, owner, value)``

        where:
          - ``name``: component name. Reserved: ``"position"``, ``"velocity"``.
            Arbitrary parameter names (e.g., ``"eta_srp"``, ``"Cr"``, ``"mu"``) are allowed.
          - ``dim``: component dimension (e.g., 3 for position/velocity; 1 for scalars).
          - ``estimation``: ``"estimated"`` or ``"considered"``.
          - ``dynamics``: ``"dynamic"`` or ``"static"``.
          - ``owner``: the entity the component belongs to (e.g., :class:`~scarabaeus.Body`,
            :class:`~scarabaeus.Spacecraft`, :class:`~scarabaeus.GroundStation`, or ``None`` for global params).
          - ``value``: initial value, typically an :class:`~scarabaeus.ArrayWUnits`
            (vector of size ``dim`` for vector components; scalar for ``dim == 1``).

    See Also
    --------
    scarabaeus.StateDefinition : Fluent builder for constructing valid state components.
    scarabaeus.FilterOD : Filter that consumes StateArray in estimation workflows.
    scarabaeus.SolutionOD : OD result container that references StateArray.

    Notes
    -----
    - **Units**: supply ``value`` with units via :class:`~scarabaeus.ArrayWUnits` (e.g., km, km/s, rad).
    - **Dimensions**: reserved kinematic components use ``dim = 3``; scalar parameters default to ``dim = 1``.
    - **Allowed tokens** (case-sensitive):
        ``estimation ∈ {"estimated","considered"}``,
        ``dynamics ∈ {"dynamic","static"}``.
    - :class:`~scarabaeus.StateDefinition` performs light validation (e.g., ``len(value) == dim`` when vector-like).

    Examples
    --------
    Build a state vector using the fluent builder:

    .. code-block:: python

        import scarabaeus as scb
        from scarabaeus import StateDefinition

        # Owners
        Orbiter = scb.Spacecraft("SCB-1")
        Vesta   = scb.Body("Vesta")

        # Initial values with units
        pos_0 = scb.ArrayWUnits([+10.0, -2.0, +1.0], "km")
        vel_0 = scb.ArrayWUnits([ -0.002, 0.001, 0.0005], "km/s")
        eta_0 = scb.ArrayWUnits(1.0, "1")  # unitless

        state_def = (
            StateDefinition()
            .position(Orbiter, pos_0)                   # 3D, estimated+dynamic by default
            .velocity(Orbiter, vel_0)                   # 3D
            .param("eta_srp", Orbiter, eta_0, static=True)  # scalar, static
        )

        state_vec = scb.StateArray(epoch=epoch_array[0], origin=Vesta, state=state_def)

    Advanced: add considered/static parameters and station terms:

    .. code-block:: python

        DSN = scb.GroundStation("DSS-14")
        range_bias = scb.ArrayWUnits(0.0, "m/s")  # range-rate bias, considered

        state_def = (
            StateDefinition()
            .position(Orbiter, pos_0)
            .velocity(Orbiter, vel_0)
            .param("Cr", Orbiter, scb.ArrayWUnits(1.8, "1"), static=True)
            .param("mu", Vesta, scb.ArrayWUnits(17.8e-3, "km^3/s^2"), estimated=False, static=True)
            .param("range_bias_RU", DSN, range_bias, estimated=False, static=True)
        )

        state_vec = scb.StateArray(epoch=epoch_array[0], origin=Vesta, state=state_def)

    Raises
    ------
    ValueError
        If the builder emits an unsupported token for ``estimation``/``dynamics``,
        or if a vector ``value`` length is inconsistent with ``dim`` (when validated).

    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        epoch: EpochArray,
        origin: Body,
        state: StateDefinition,
    ):
        # Initialize properties
        self._epoch = epoch
        self._origin = origin
        self._state = state.components
        self._state_definition = self._extract_state_definition()
        self._values0 = self._extract_values0()
        self._values_array = self._extract_values_array()
        self._state_dot_definition = self._generate_state_dot_definition()
        self._size = sum([x[1] for x in self.state])
        self._index_rep = self._make_index_rep()

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def epoch(self) -> EpochArray:
        """
        Epoch for which the state vector is defined.
        """
        return self._epoch

    @property
    def origin(self) -> Body:
        """
        Origin for which the state vector is defined.
        """
        return self._origin

    @property
    def state(self) -> list:
        """
        Full state vector definition as defined and input to the constructor.
        """
        return self._state

    @property
    def values0(self) -> ArrayWUnits:
        """Numerical values of the state vector at t0, as an ``ArrayWUnits``."""
        return self._values0

    @property
    def values_array(self) -> dict:
        """Dict mapping ``(name, owner)`` keys to initial ``ArrayWUnits`` values."""
        return self._values_array

    @property
    def state_definition(self) -> list:
        """
        Parameter definition representation of the state vector. Does not include values.
        """
        return self._state_definition

    @property
    def state_dot_definition(self) -> list:
        """
        Parameter definition representation of the time derivative of the state
        vector. Does not include values.
        """
        return self._state_dot_definition

    @property
    def size(self) -> int:
        """
        Size (length) of the state vector.
        """
        return self._size

    @property
    def index_rep(self) -> dict:
        """
        Representation of the state that links the parameters to the location of
        that parameter's value(s) in the state vector.
        """
        return self._index_rep

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _extract_state_definition(self) -> list:
        """
        Generates a standardized definition of the state.
        Uses only the parameter definitions used in constructing the state Vector.
        Does not include the parameter values.
        This representation is used in the propagator when computing accelerations, general dynamics, and partials (the jocobian).

        Returns
        -------
        list
            State vector consisting only of the parameters and their definitions
        """
        # Initialize
        state_definition = []

        # Extract parameter definitions
        for parameter in self.state:
            state_definition.append(parameter[0:-1])

        return state_definition

    def _extract_values_array(self) -> dict:
        """
        Extract a body-aware representation of the state vector.

        For each state entry of the form:
            (name, dim, est, dyn, owner, ArrayWFrame)

        this returns a dictionary:

            (name, spice_id) -> (values, units)

        where:
        - `spice_id` is taken from `owner.spice_id` when available,
        - if `owner` has no `spice_id` (e.g. global parameter), `spice_id` is None.

        For dynamic parameters:
        - If values are given as a time history (shape (N, dim) or (N,)),
        N is checked to be consistent across parameters and with epoch.size.
        - If values are just a single vector of size `dim`, they are treated as
        initial values only (no time history check).
        """

        extracted_data: dict = {}
        dynamic_sizes = set()

        for entry in self.state:
            if len(entry) < 6:
                raise ValueError(
                    f"Invalid state entry format: {entry!r}. "
                    "Expected (name, dim, est, dyn, owner, ArrayWFrame)."
                )

            name = entry[0]
            dim = entry[1]
            dyn_flag = entry[3]  # 'dynamic' or 'static'
            owner = entry[4]
            awf = entry[5]

            if not isinstance(awf, ArrayWFrame):
                raise TypeError(
                    f"State values must be ArrayWFrame, got {type(awf)} "
                    f"for parameter '{name}'."
                )

            awu = awf.quantity
            vals = awu.values
            units = awu.units

            # Key: (name, spice_id)
            spice_id = getattr(owner, "spice_id", None)
            key = (name, spice_id)

            if key in extracted_data:
                raise ValueError(
                    f"Duplicate state component for key {key}: "
                    "check your StateDefinition for repeated entries."
                )

            # --- Static parameters: must match declared dimension exactly ---
            if dyn_flag == "static":
                if np.atleast_1d(vals).size != dim:
                    raise ValueError(
                        f"Static parameter '{name}' must have size {dim}, "
                        f"got {np.atleast_1d(vals).size}."
                    )

            # --- Dynamic parameters: distinguish initial vector vs time history ---
            elif dyn_flag == "dynamic":
                vals_arr = np.asarray(vals)

                # Case 1: looks like a single vector (e.g. (3,) or (dim,))
                # -> treat as initial condition only, no epoch-size check.
                if vals_arr.ndim == 1 and vals_arr.size == dim:
                    pass

                # Case 2: looks like time history (N, dim) or (N,) with N != dim
                else:
                    # Interpret first axis as time samples
                    if vals_arr.ndim == 1:
                        N = vals_arr.size
                    else:
                        N = vals_arr.shape[0]

                    dynamic_sizes.add(N)

            extracted_data[key] = (vals, units)

        # --- Consistency check for true time-varying dynamics ---
        if dynamic_sizes:
            if len(dynamic_sizes) != 1:
                raise ValueError(
                    "Dynamic parameters have inconsistent time lengths: "
                    f"{sorted(dynamic_sizes)}. They must all share the same N."
                )

            N = dynamic_sizes.pop()
            epoch_size = self.epoch.size

            if epoch_size != N:
                raise ValueError(
                    f"Mismatch between epoch size ({epoch_size}) and dynamic parameter "
                    f"time length ({N}). They must be equal for time-varying states."
                )

        return extracted_data

    def _extract_values0(self) -> ArrayWUnits:
        """
        Generates a state vector representation consisting of only the parameter values.
        This vector is what is used when propagating, for example.

        Returns
        -------
        ArrayWUnits
            Vector of values that define the state.
        """

        # Initialize
        extracted_vals = np.array([])
        extracted_unts = numpy.array([])

        # Extract values and units
        for parameter in self.state:
            # Extracting the last element of the parameter tuple
            param_awf = parameter[-1]

            # Check if the parameter is of type ArrayWFrame
            if not isinstance(param_awf, ArrayWFrame):
                raise TypeError(
                    "The state vector values should be of type ArrayWFrame."
                )

            # Converting ArrayWFrame to ArrayWUnits
            param_awu = param_awf.quantity

            # Extract values and units
            extracted_vals = np.hstack(
                (extracted_vals, np.atleast_2d(param_awu.values)[0])
            )
            extracted_unts = numpy.hstack((extracted_unts, param_awu.units))

        return ArrayWUnits(extracted_vals, extracted_unts)

    def _generate_state_dot_definition(self) -> list:
        """
        Generates a standardized definition of the first time derivative of the state.
        Uses only the parameter definitions used in constructing the state Vector.
        Does not include the parameter values.
        This representation is used in the propagator when computing accelerations, general dynamics, and partials (the jocobian).

        Returns
        -------
        list
            Vector consisting of the definitions of the time derivative of the parameters in the state vector
        """
        # Initialize
        state_dot_definition = []

        # Generate state_dot definition
        for parameter in self.state_definition:
            new_parameter = None
            if parameter[0] == "position":
                new_parameter = ("velocity",)
            elif parameter[0] == "velocity":
                new_parameter = ("acceleration",)
            else:
                new_parameter = ("{0}_dot".format(parameter[0]),)
            state_dot_definition.append(new_parameter + parameter[1:])

        return state_dot_definition

    def _make_index_rep(self) -> dict:
        """
        Build an index map from state definition entries to slices in the flat state vector.

        For each state entry of the form:
            (name, dim, est, dyn, owner, value)

        this returns a dictionary mapping:

            (name, spice_id) -> slice(i_start, i_end)

        where:
        - `spice_id` is taken from `owner.spice_id` when available.
        - If `owner` has no `spice_id` (e.g. global parameter), the key is just `name`.

        This makes multi-body states unambiguous, e.g.:

            ('position', -1000) -> Orbiter position slice
            ('position', 399)   -> Earth position slice

        instead of overwriting everything under "position".
        """

        index_rep: dict = {}
        idx = 0

        for entry in self.state:
            if len(entry) < 5:
                raise ValueError(
                    f"Invalid state entry format: {entry!r}. "
                    "Expected at least (name, dim, est, dyn, owner, ...)."
                )

            name = entry[0]
            dim = entry[1]
            owner = entry[4]

            spice_id = getattr(owner, "spice_id", None)

            # Key by (name, spice_id) when available, else by name only
            key = (name, spice_id) if spice_id is not None else name

            if key in index_rep:
                raise ValueError(
                    f"Duplicate state component for key {key!r} in StateArray. "
                    "Use distinct owners or ensure each (name, spice_id) pair "
                    "appears only once."
                )

            index_rep[key] = slice(idx, idx + dim)
            idx += dim

        return index_rep

    def _make_direct_representation(self, y=None) -> dict:
        """
        Build a direct numeric-ID-based representation of the current state.

        Each entry is keyed as (name, spice_id), where `spice_id` is taken from the
        owner object of that state component. Only numeric identifiers are used,
        no object or string aliases.

        Parameters
        ----------
        y : np.ndarray, optional
            Flat state vector at the current step. If None, uses ``self.values``.

        Returns
        -------
        dict
            Mapping from (name, spice_id) → np.ndarray or scalar.
        """
        if y is None:
            y = self.values

        direct_rep: dict = {}
        idx = 0

        for entry in self.state:
            if len(entry) < 5:
                raise ValueError(
                    f"Invalid state entry format: {entry!r}. "
                    "Expected at least (name, dim, est, dyn, owner, ...)."
                )

            name = entry[0]
            dim = entry[1]
            owner = entry[4]

            spice_id = getattr(owner, "spice_id", None)
            if spice_id is None:
                raise ValueError(
                    f"Owner {owner} has no 'spice_id'. "
                    "All propagated bodies must define one."
                )

            start, end = idx, idx + dim
            if end > len(y):
                raise ValueError(
                    f"Inconsistent StateArray layout: {name} needs [{start}:{end}] "
                    f"but y has length {len(y)}."
                )

            values = y[start:end] if dim > 1 else y[start]
            direct_rep[(name, spice_id)] = values
            idx = end

        return direct_rep

    def reinitialize(self, **kwargs):
        """
        Reinitializes the StateArray object with new attribute values in place.

        Parameters
        ----------
        **kwargs :
            Keyword arguments representing the new attribute values.
                - epoch (EpochArray, optional): New epoch array
                - origin (Body, optional): New origin body
                - state (StateDefinition or list, optional): New state definition

        Modifies:
            Updates the existing instance (`self`) instead of creating a new one.

        Notes
        -----
        If 'state' is provided as a raw list of tuples, it will be automatically
        converted to a StateDefinition object for compatibility.
        """
        self._epoch = kwargs.get("epoch", self._epoch)
        self._origin = kwargs.get("origin", self._origin)

        # Handle state input - accept both StateDefinition and raw list
        new_state = kwargs.get("state", self._state)
        if new_state is not self._state:
            # Convert to StateDefinition if it's a raw list
            if isinstance(new_state, list):
                new_state = StateDefinition.from_components(new_state)
            elif not isinstance(new_state, StateDefinition):
                raise TypeError(
                    "state must be either a StateDefinition instance or a list of component tuples"
                )
            self._state = new_state.components

        # Recompute dependent attributes
        self._state_definition = self._extract_state_definition()
        self._values0 = self._extract_values0()
        self._values_array = self._extract_values_array()
        self._state_dot_definition = self._generate_state_dot_definition()
        self._size = sum([x[1] for x in self._state])
        self._index_rep = self._make_index_rep()

    def find_indices(self, parameter_name: str, body: Body, flag_dot=False) -> tuple:
        """
        Finds the indices of a parameter (e.g., position, velocity) in the state vector for a specific body.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to search for (e.g., "position", "velocity").
        body : Body
            The associated body of the parameter.
        flag_dot : bool
            Flag to indicate whether to search for the time derivative of the parameter

        Returns
        -------
        tuple
            A tuple (start_index, end_index) indicating the slice of the state vector corresponding to the parameter.
            Returns None if the parameter is not found.
        """
        current_index = 0
        state_definition = (
            self.state_dot_definition if flag_dot else self.state_definition
        )
        for param in state_definition:
            name, size, *_, associated_body = param
            if name == parameter_name and associated_body == body:
                return current_index, current_index + size
            current_index += size

        # Parameter not found
        return None

    def add_to_state_vector(self, new_components):
        """
        Adds additional components to the state vector and properly re-initializes the StateArray.

        Parameters
        ----------
        new_components : list or StateDefinition
            New state components to be added.
            Can be either:
            - A StateDefinition object
            - A list of tuples in format: (parameter_name, size, estimated/considered, dynamic/static, body, value)

        Returns
        -------
        StateArray
            A new re-initialized StateArray instance with the updated state.

        Raises
        ------
        TypeError
            If new_components is not a list or StateDefinition.
        ValueError
            If duplicate parameters are detected.
        """
        # Convert to list of tuples if StateDefinition
        if isinstance(new_components, StateDefinition):
            new_components = new_components.components
        elif not isinstance(new_components, list):
            raise TypeError(
                "new_components must be a StateDefinition or list of state components."
            )

        # Build a StateDefinition from existing state
        state_def = StateDefinition.from_components(self._state)

        # Track existing parameters to prevent duplicates
        existing_params = {(comp[0], id(comp[4])) for comp in self._state}

        # Add new components
        for component in new_components:
            param_name, size, est_type, dyn_type, body, value = component

            # Check for duplicates
            param_key = (param_name, id(body))
            if param_key in existing_params:
                raise ValueError(
                    f"Parameter '{param_name}' for body {body} already exists in the state vector."
                )

            # Add to state definition using fluent interface
            if param_name == "position":
                state_def.position(body, value, estimation=est_type, dynamics=dyn_type)
            elif param_name == "velocity":
                state_def.velocity(body, value, estimation=est_type, dynamics=dyn_type)
            else:
                state_def.param(
                    param_name,
                    body,
                    value,
                    dimension=size,
                    estimation=est_type,
                    dynamics=dyn_type,
                )

            existing_params.add(param_key)

        # Return new StateArray with updated state
        return StateArray(epoch=self.epoch, origin=self.origin, state=state_def)

    def add_state_at_epoch(self, epoch, new_values):
        """
        Adds a new state at a specific epoch, keeping epochs in order and
        reinitializing the StateArray.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch at which to add the state.
        new_values : dict
            Dictionary where keys match self._values_array keys, i.e.
            (param_name, body_id), and values are the new state values
            for that epoch (ArrayWFrame, with frame possibly None).

            Example keys:
                {
                    ('position', -1000): ArrayWFrame(..., frame=J2000),
                    ('velocity', -1000): ArrayWFrame(..., frame=J2000),
                    ('eta_srp', -1000): ArrayWFrame(ArrayWUnits(...), frame=None),
                    ('range_bias_1', 399014): ArrayWFrame(ArrayWUnits(...), frame=None),
                    }

        Raises
        ------
        ValueError
            If the new state is inconsistent with existing parameters.
        TypeError
            If a provided value has the wrong type.
        """
        # Ensure epoch is an EpochArray object
        if not isinstance(epoch, EpochArray):
            epoch = EpochArray(np.array([epoch]), sys=self.epoch.system)

        # Ensure frames match
        if epoch.system != self.epoch.system:
            raise ValueError(
                f"Epoch time frame mismatch: expected {self.epoch.system}, got {epoch.system}."
            )

        # Find insertion index to keep epochs sorted
        insert_idx = np.searchsorted(
            np.atleast_1d(self.epoch.times.values), epoch.times.values
        )

        # Deep copy existing epochs and insert the new one
        updated_epochs = np.insert(
            copy.deepcopy(self.epoch.times.values), insert_idx, epoch.times.values
        )

        # Deep copy existing state dictionary
        # Keys are now (param_name, body_id)
        updated_values_array = copy.deepcopy(self._values_array)

        for key, (wrapped_vals, units) in self._values_array.items():
            # key = (param_name, body_id)
            param_name, body_id = key

            # Find the corresponding entry in self.state to get the frame/body
            # self.state entries: (name, size, est_type, dyn_type, body, values)
            matching_param = next(
                (
                    param
                    for param in self.state
                    if param[0] == param_name
                    and getattr(param[4], "spice_id", None) == body_id
                ),
                None,
            )
            if matching_param is None:
                raise ValueError(
                    f"No matching state entry found for parameter {param_name} with body_id {body_id}."
                )

            _, size, est_type, dyn_type, body, state_value = matching_param

            # Unwrap existing values and frame
            if isinstance(wrapped_vals, ArrayWFrame):
                param_frame = wrapped_vals.frame
                existing_vals = wrapped_vals.quantity.values
            elif isinstance(wrapped_vals, ArrayWUnits):
                # Legacy path: scalar/unit-only stored as ArrayWUnits
                # Treat as frame=None; we will re-wrap as ArrayWFrame(..., None)
                param_frame = None
                existing_vals = wrapped_vals.values
            else:
                # Fallback: raw array or scalar
                param_frame = getattr(state_value, "frame", None)
                existing_vals = np.array(wrapped_vals)

            # Decide whether we have an update for this parameter/body
            if key in new_values:
                param_data = new_values[key]

                # Normalize to values + frame
                if isinstance(param_data, ArrayWFrame):
                    new_frame = param_data.frame
                    param_data_values = param_data.quantity.values

                    # Frame consistency check if we have an existing frame
                    if param_frame is not None and new_frame is not None:
                        if new_frame.name != param_frame.name:
                            raise ValueError(
                                f"Frame mismatch for parameter '{param_name}': "
                                f"expected {param_frame}, got {new_frame}."
                            )

                    # Use whatever frame we have (existing or new)
                    final_frame = param_frame if param_frame is not None else new_frame

                elif isinstance(param_data, ArrayWUnits):
                    # Allow ArrayWUnits as input, but scalar components should be
                    # stored as ArrayWFrame(..., frame=None)
                    param_data_values = param_data.values
                    final_frame = param_frame  # may be None
                else:
                    raise TypeError(
                        f"Expected ArrayWFrame or ArrayWUnits for '{param_name}', "
                        f"got {type(param_data)} instead."
                    )

                # Dynamic vs static: if existing_vals is 2D (or more), treat as time-varying
                if np.atleast_1d(existing_vals).ndim > 1:
                    # For dynamic parameters, enforce shape consistency except time dimension
                    if param_data_values.shape[1:] != existing_vals.shape[1:]:
                        raise ValueError(
                            f"New values for '{param_name}' do not match existing shape "
                            f"{existing_vals.shape[1:]}."
                        )

                    # Insert new time slice along axis 0
                    updated_vals = np.insert(
                        existing_vals, insert_idx, param_data_values, axis=0
                    )
                else:
                    # Static parameter: overwrite
                    updated_vals = param_data_values

            else:
                # No update for this parameter; keep existing values as-is
                updated_vals = existing_vals
                final_frame = param_frame

            # Always re-wrap as ArrayWFrame, even for scalar/unit-only parameters
            # For scalars, final_frame is typically None.
            updated_values_array[key] = (
                ArrayWFrame(ArrayWUnits(updated_vals, units), final_frame),
                units,
            )

        # Rebuild state and reinitialize
        new_state = self._rebuild_state(updated_values_array)
        self.reinitialize(
            epoch=EpochArray(updated_epochs, sys=self.epoch.system),
            origin=self.origin,
            state=new_state,
        )

    def remove_state_at_epoch(self, epoch):
        """
        Removes a state at a specific epoch, ensuring consistency across parameters
        and reinitializing the StateArray.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch at which to remove the state.

        Raises
        ------
        ValueError
            If the epoch is not found in the state vector.
        """
        # Ensure epoch is an EpochArray object
        if not isinstance(epoch, EpochArray):
            epoch = EpochArray(np.array([epoch]), sys=self.epoch.system)

        # Ensure frames match
        if epoch.system != self.epoch.system:
            raise ValueError(
                f"Epoch time frame mismatch: expected {self.epoch.system}, got {epoch.system}."
            )

        # Find index of the epoch to remove
        idx_to_remove = np.where(
            np.atleast_1d(self.epoch.times.values) == np.atleast_1d(epoch.times.values)
        )[0]

        if len(idx_to_remove) == 0:
            raise ValueError(f"Epoch {epoch} not found in the state vector.")

        idx_to_remove = idx_to_remove[0]

        # Deep copy existing epochs and remove the target one
        updated_epochs = np.delete(
            copy.deepcopy(self.epoch.times.values), idx_to_remove
        )

        # Deep copy state dictionary (keys are (param_name, body_id))
        updated_values_array = copy.deepcopy(self._values_array)

        for key, (wrapped_vals, units) in self._values_array.items():
            param_name, body_id = key

            # Find the corresponding state entry to get the template/frame
            matching_param = next(
                (
                    param
                    for param in self.state
                    if param[0] == param_name
                    and getattr(param[4], "spice_id", None) == body_id
                ),
                None,
            )
            if matching_param is None:
                raise ValueError(
                    f"No matching state entry found for parameter {param_name} with body_id {body_id}."
                )

            _, _, _, _, _, value_template = matching_param

            # Unwrap existing values and frame
            if isinstance(wrapped_vals, ArrayWFrame):
                param_frame = wrapped_vals.frame
                existing_vals = wrapped_vals.quantity.values
            elif isinstance(wrapped_vals, ArrayWUnits):
                # Legacy path: scalar/unit-only; we will re-wrap as ArrayWFrame(..., None)
                param_frame = getattr(value_template, "frame", None)
                existing_vals = wrapped_vals.values
            else:
                param_frame = getattr(value_template, "frame", None)
                existing_vals = np.array(wrapped_vals)

            # Dynamic vs static parameter
            if np.atleast_1d(existing_vals).ndim > 1:
                # Dynamic: remove the appropriate time index
                updated_vals = np.delete(existing_vals, idx_to_remove, axis=0)
            else:
                # Static: leave unchanged
                updated_vals = existing_vals

            # Always re-wrap as ArrayWFrame (scalar components get frame=None)
            updated_values_array[key] = (
                ArrayWFrame(ArrayWUnits(updated_vals, units), param_frame),
                units,
            )

        # Rebuild state and reinitialize
        new_state = self._rebuild_state(updated_values_array)
        self.reinitialize(
            epoch=EpochArray(updated_epochs, sys=self.epoch.system),
            origin=self.origin,
            state=new_state,
        )

    def update_state_at_epoch(self, epoch, updated_values):
        """
        Updates the state at a given epoch while maintaining structure and
        reinitializing the StateArray.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch at which to update the state.
        updated_values : dict
            Dictionary where keys match self._values_array keys, i.e.
            (param_name, body_id), and values are ArrayWFrame / ArrayWUnits
            with the *new* values for that epoch.

            Example::

                {
                    ('position', -1000): ArrayWFrame(...),
                    ('velocity', -1000): ArrayWFrame(...),
                    ('eta_srp', -1000): ArrayWFrame(ArrayWUnits(...), None),
                }

        Raises
        ------
        ValueError
            If the epoch is not found or values are inconsistent.
        TypeError
            If a provided value has the wrong type.
        """
        # Ensure epoch is an EpochArray object
        if not isinstance(epoch, EpochArray):
            epoch = EpochArray(np.array([epoch]), sys=self.epoch.system)

        # Ensure frames match
        if epoch.system != self.epoch.system:
            raise ValueError(
                f"Epoch time frame mismatch: expected {self.epoch.system}, got {epoch.system}."
            )

        # Find the index of the epoch to update
        idx_to_update = np.where(
            np.atleast_1d(self.epoch.times.values) == np.atleast_1d(epoch.times.values)
        )[0]

        if len(idx_to_update) == 0:
            raise ValueError(f"Epoch {epoch} not found in the state vector.")

        idx_to_update = idx_to_update[0]

        # Deep copy state dictionary
        updated_values_array = copy.deepcopy(self._values_array)

        for key, (wrapped_vals, units) in self._values_array.items():
            param_name, body_id = key

            # Find the corresponding state entry to get the template/frame
            matching_param = next(
                (
                    param
                    for param in self.state
                    if param[0] == param_name
                    and getattr(param[4], "spice_id", None) == body_id
                ),
                None,
            )
            if matching_param is None:
                raise ValueError(
                    f"No matching state entry found for parameter {param_name} with body_id {body_id}."
                )

            _, _, est_type, dyn_type, body, value_template = matching_param

            # Unwrap existing values and frame
            if isinstance(wrapped_vals, ArrayWFrame):
                param_frame = wrapped_vals.frame
                existing_vals = wrapped_vals.quantity.values
            elif isinstance(wrapped_vals, ArrayWUnits):
                param_frame = getattr(value_template, "frame", None)
                existing_vals = wrapped_vals.values
            else:
                param_frame = getattr(value_template, "frame", None)
                existing_vals = np.array(wrapped_vals)

            # Default to existing values/frame
            updated_vals = existing_vals
            final_frame = param_frame

            # If this parameter/body is updated at this epoch
            if key in updated_values:
                param_data = updated_values[key]

                if isinstance(param_data, ArrayWFrame):
                    new_frame = param_data.frame
                    param_data_values = param_data.quantity.values

                    # Frame consistency (only if both non-None)
                    if param_frame is not None and new_frame is not None:
                        if new_frame.name != param_frame.name:
                            raise ValueError(
                                f"Frame mismatch for parameter '{param_name}': "
                                f"expected {param_frame}, got {new_frame}."
                            )

                    final_frame = param_frame or new_frame

                elif isinstance(param_data, ArrayWUnits):
                    # Allow ArrayWUnits for scalar/unit-only, will store as ArrayWFrame(..., None)
                    param_data_values = param_data.values
                    final_frame = param_frame  # may be None
                else:
                    raise TypeError(
                        f"Expected ArrayWFrame or ArrayWUnits for '{param_name}', "
                        f"got {type(param_data)} instead."
                    )

                # Dynamic vs static
                if np.atleast_1d(existing_vals).ndim > 1:
                    # For dynamic parameters:
                    #   existing_vals shape: (n_epochs, *per_epoch_shape)
                    #   per-epoch shape:
                    per_epoch_shape = existing_vals.shape[1:]
                    pv = np.array(param_data_values)

                    # Allow (per_epoch_shape) or (1, *per_epoch_shape)
                    if pv.shape == per_epoch_shape:
                        slice_vals = pv
                    elif pv.shape == (1,) + per_epoch_shape:
                        slice_vals = pv[0]
                    else:
                        raise ValueError(
                            f"New values for '{param_name}' at epoch have incompatible shape "
                            f"{pv.shape}; expected {per_epoch_shape} or {(1,) + per_epoch_shape}."
                        )

                    updated_vals = existing_vals.copy()
                    updated_vals[idx_to_update] = slice_vals
                else:
                    # Static parameter: overwrite entire value
                    updated_vals = param_data_values

            # Always store as ArrayWFrame (scalars get frame=None)
            updated_values_array[key] = (
                ArrayWFrame(ArrayWUnits(updated_vals, units), final_frame),
                units,
            )

        # Rebuild state and reinitialize
        new_state = self._rebuild_state(updated_values_array)
        self.reinitialize(epoch=self.epoch, origin=self.origin, state=new_state)

    def get_state_at_epoch(self, epoch):
        """
        Retrieves the state vector for a given epoch.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch for which to retrieve the state.

        Returns
        -------
        StateArray
            A new StateArray instance with only the specified epoch's data.

        Raises
        ------
        ValueError
            If the epoch is not found in the state vector.
        """
        # Ensure epoch is an EpochArray object
        if not isinstance(epoch, EpochArray):
            epoch = EpochArray(np.array([epoch]), sys=self.epoch.system)

        # Ensure frames match
        if epoch.system != self.epoch.system:
            raise ValueError(
                f"Epoch time frame mismatch: expected {self.epoch.system}, got {epoch.system}."
            )

        # Find the index of the requested epoch
        idx_to_get = np.where(
            np.atleast_1d(self.epoch.times.values) == np.atleast_1d(epoch.times.values)
        )[0]

        if len(idx_to_get) == 0:
            raise ValueError(f"Epoch {epoch} not found in the state vector.")

        idx_to_get = idx_to_get[0]

        # Build new state definition
        state_def = StateDefinition()

        for param in self.state:
            param_name, size, est_type, dyn_type, body, value_template = param
            body_id = getattr(body, "spice_id", None)
            key = (param_name, body_id)

            if key not in self._values_array:
                raise ValueError(
                    f"Values array missing entry for parameter {param_name} with body_id {body_id}."
                )

            wrapped_vals, units = self._values_array[key]

            # Unwrap existing values and frame
            if isinstance(wrapped_vals, ArrayWFrame):
                param_frame = wrapped_vals.frame
                existing_vals = wrapped_vals.quantity.values
            elif isinstance(wrapped_vals, ArrayWUnits):
                param_frame = getattr(value_template, "frame", None)
                existing_vals = wrapped_vals.values
            else:
                param_frame = getattr(value_template, "frame", None)
                existing_vals = np.array(wrapped_vals)

            # Dynamic vs static
            if np.atleast_1d(existing_vals).ndim > 1:
                param_values = existing_vals[idx_to_get]
            else:
                param_values = existing_vals

            # Always wrap as ArrayWFrame, scalar components use frame=None
            param_AWF = ArrayWFrame(ArrayWUnits(param_values, units), param_frame)

            # Add to state definition using fluent interface
            if param_name == "position":
                state_def.position(
                    body, param_AWF, estimation=est_type, dynamics=dyn_type
                )
            elif param_name == "velocity":
                state_def.velocity(
                    body, param_AWF, estimation=est_type, dynamics=dyn_type
                )
            else:
                state_def.param(
                    param_name,
                    body,
                    param_AWF,
                    dimension=size,
                    estimation=est_type,
                    dynamics=dyn_type,
                )

        # Return a new StateArray with only the requested epoch
        return StateArray(epoch=epoch, origin=self.origin, state=state_def)

    def _rebuild_state(self, updated_values_array):
        """
        Reconstructs the state list from updated values array.

        Parameters
        ----------
        updated_values_array : dict
            Updated dictionary of parameter values.

        Returns
        -------
        list
            Rebuilt state list with updated ArrayWFrame objects.
        """
        new_state = []
        for param in self.state:
            param_name, size, est_type, dyn_type, body, _ = param

            # Retrieve updated values
            values, _ = updated_values_array[(param_name, body.spice_id)]

            new_state.append((param_name, size, est_type, dyn_type, body, values))

        return new_state
