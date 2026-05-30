# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Units,
    ArrayWUnits,
    EpochArray,
    MissionSequence,
    StateArray,
    Propagator,
    ArrayWFrame,
    Body,
    StateDefinition,
    ImpulsiveBurn,
    SpiceManager,
)

import scarabaeus.utils.NumpyWrapper as np
import warnings

# ------------------#
#  Generate Units  #
# ------------------#
km, sec = Units.get_units(["km", "sec"])


# --------------------#
#  Class Definition  #
# --------------------#
class ScenarioSetup:
    """Executes a :class:`~scarabaeus.MissionSequence` leg by leg and assembles the full
    propagated trajectory.

    :class:`~scarabaeus.ScenarioSetup` drives the numerical propagation of a
    :class:`~scarabaeus.MissionSequence`: it iterates over every leg and node in order,
    propagates each leg with its own :class:`~scarabaeus.Propagator` (which may carry a
    different force model, reference frame, or origin than adjacent legs),
    applies any impulsive burn at node boundaries, and concatenates the
    per-leg outputs into a unified state and STM history.

    For a sequence of :math:`L` legs each leg :math:`\\ell` produces a local
    STM :math:`\\boldsymbol{\\Phi}_\\ell(t_\\ell^f, t_\\ell^0)`.
    When ``map_STM_to_t0=True`` the STM is accumulated across nodes:

    .. math::

        \\boldsymbol{\\Phi}(t_L, t_0)
        = \\boldsymbol{\\Phi}_L \\cdots \\boldsymbol{\\Phi}_2 \\boldsymbol{\\Phi}_1

    so that it always represents sensitivity with respect to the global
    initial state :math:`\\mathbf{x}(t_0)`.  When ``map_STM_to_t0=False``
    (default) the STM is reset to the identity at each node.

    Parameters
    ----------
    mission_sequence : MissionSequence
        The mission sequence containing legs, burns, and associated state vectors.

    map_STM_to_t0 : bool, optional
        Whether to map the STM (State Transition Matrix) to the initial state.
        Defaults to ``False``.

    See Also
    --------
    scarabaeus.Propagator : the propagator that ScenarioSetup interacts with.

    Notes
    -----
    The ``ScenarioSetup`` class is designed to be compatible with the ``Propagator`` class.
    Any updates to propagation functionality should be reflected here.

    Enforces mission constraints such as maximum numbers of legs and impulsive burns.
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        mission_sequence: MissionSequence,
        map_STM_to_t0: bool = False,
    ):
        self._mission_sequence = mission_sequence
        self._len_sequence = len(self.mission_sequence.names)
        self._map_STM_to_t0 = map_STM_to_t0
        self._propagator = self.mission_sequence.props[0]
        self._origin = self.mission_sequence.origins[0]
        self._atol = 2.220446049250313e-11
        self._rtol = 2.220446049250313e-11
        self._state_size_old = self.mission_sequence.states_n[0]
        self._state_vector_old = self.mission_sequence.models[0]
        self._delta_v = None
        self._primary_body = self.mission_sequence.primary_body

        # Create the full scenario state definition and index mapping based on the mission sequence
        self._scenario_full_state_def = self._build_scenario_state_def()
        self._scenario_index_map = self._build_global_offsets(
            self._scenario_full_state_def
        )
        self._leg_to_global_maps = {}
        self._global_state_value = self._init_global_state_value()
        self._global_state_meta = self._build_global_state_meta()
        self._x_ref0 = (
            np.asarray(self._global_state_value, dtype=float).reshape(-1).copy()
        )
        self._x_ic = self._x_ref0.copy()

        # Create dictionaries to store the segment data
        self._segment_positions = {segment: None for segment in mission_sequence.names}
        self._segment_velocities = {segment: None for segment in mission_sequence.names}
        self._segment_parameters = {segment: None for segment in mission_sequence.names}
        self._segment_STMs = {segment: None for segment in mission_sequence.names}
        self._segment_epochsTDB = {segment: None for segment in mission_sequence.names}

        # Store data
        self._total_components = {}
        self._total_parameters = []
        self._STM = []
        self._full_propagated_state_array = None

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def mission_sequence(self) -> MissionSequence:
        """
        The mission sequence associated with this scenario.
        """
        return self._mission_sequence

    @property
    def len_sequence(self) -> int:
        """
        The total number of elements (legs + burns) in the mission sequence.
        """
        return self._len_sequence

    @property
    def map_STM_to_t0(self) -> bool:
        """
        Indicates whether the STM should be mapped to the initial time.
        ``True`` if the STM is mapped to the initial state, otherwise ``False``.
        """
        return self._map_STM_to_t0

    @property
    def propagator(self) -> Propagator:
        """
        Internal propagator for the current segment. Not intended for external use —
        its value changes at each segment during ``propagate()``.
        """
        return self._propagator

    @propagator.setter
    def propagator(self, input_val):

        self._propagator = input_val

    @property
    def origin(self) -> Body:
        """
        The origin body for the scenario.
        """
        return self._origin

    @property
    def primary_body(self) -> Body:
        """
        The primary body for the scenario.
        """
        return self._primary_body

    @origin.setter
    def origin(self, input_val):
        """
        Sets the origin body for the scenario.
        """
        self._origin = input_val

    @property
    def atol(self) -> float:
        """
        The absolute tolerance for numerical integration.
        """
        return self._atol

    @property
    def rtol(self) -> float:
        """
        The relative tolerance for numerical integration.
        """
        return self._rtol

    @property
    def delta_v(self) -> ArrayWUnits | None:
        """
        The last computed impulsive delta-v, if applicable.
        """
        return self._delta_v

    @property
    def propagated_state_array(self) -> StateArray:
        """
        State array covering the full propagated mission (all legs combined).

        After calling ``propagate()``, this returns a :class:`StateArray` whose
        epoch spans from the start of the first leg to the end of the last leg,
        with the full time-history of position, velocity, and any other dynamic
        state components accumulated across all legs.

        If ``propagate()`` has not been called yet, falls back to the last-leg
        only result with a warning.
        """
        if self._full_propagated_state_array is not None:
            return self._full_propagated_state_array
        warnings.warn(
            "propagated_state_array only contains the last propagated leg. "
            "To write a full multi-leg SPK use Trajectory(leg_array=scenario) instead.",
            UserWarning,
            stacklevel=2,
        )
        return self._propagated_state_array

    @property
    def segment_positions(self) -> dict:
        """
        Maps segment names to position arrays.
        """
        return self._segment_positions

    @property
    def segment_velocities(self) -> dict:
        """
        Maps segment names to velocity arrays.
        """
        return self._segment_velocities

    @property
    def segment_parameters(self) -> dict:
        """
        Maps segment names to parameter arrays.
        """
        return self._segment_parameters

    @property
    def segment_STMs(self) -> dict:
        """
        Maps segment names to STM arrays.
        """
        return self._segment_STMs

    @property
    def segment_epochsTDB(self) -> dict:
        """
        Maps segment names to epoch arrays.
        """
        return self._segment_epochsTDB

    @property
    def total_parameters(self) -> list[ArrayWUnits]:
        """
        The total set of propagated parameters.
        """
        return self._total_parameters

    @property
    def STM(self) -> list[np.ndarray]:
        """
        The total concatenated STM history.
        """
        return self._STM

    @property
    def total_position(self) -> ArrayWFrame | None:
        """
        Position history of the primary body over the full mission (all legs combined).
        """
        awf, _ = self._pick_primary("position")
        return awf

    @property
    def total_velocity(self) -> ArrayWFrame | None:
        """
        Velocity history of the primary body over the full mission (all legs combined).
        """
        awf, _ = self._pick_primary("velocity")
        return awf

    @property
    def total_epochsTDB(self) -> EpochArray | None:
        """
        Epoch array covering the full propagated mission in TDB seconds.
        """
        _, e = self._pick_primary("position")
        return e

    @property
    def total_components(self) -> dict:
        """
        Get the total components dictionary.
        """
        return self._total_components

    @property
    def full_state_vector(self) -> StateArray:
        """
        Full scenario state vector, spanning the union of all legs.
        """
        return StateArray(
            epoch=self.mission_sequence.epochs_vec[0][0],
            origin=self._origin,
            state=StateDefinition.from_components(self._scenario_full_state_def),
        )

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#

    def _build_global_state_meta(self) -> dict:
        """
        Build persistent metadata for the scenario-global state vector.

        Returns
        -------
        dict
        key=(name,size,sid,frame_name,occ) -> meta dict with:
            - name, size, sid, frame_name, occ
            - gs, ge (global slice)
            - units (from template awf)
            - body (object ref), frame (object ref)
            - segments: list of segment names where this key appears (StateArray legs/burns)
            - seg_types: list of corresponding mission_sequence.types entries
        """
        # 1) template map: key -> (body, frame, units) from scenario_full_state_def
        templ = {}
        occ_counter = {}
        for name, size, est_type, dyn_type, body, awf in self._scenario_full_state_def:
            base = (
                name,
                int(size),
                getattr(body, "spice_id", None),
                getattr(getattr(awf, "frame", None), "name", None),
            )
            occ = occ_counter.get(base, 0)
            occ_counter[base] = occ + 1
            key = (*base, int(occ))

            units = getattr(getattr(awf, "quantity", None), "units", None)
            templ[key] = {
                "body": body,
                "frame": getattr(awf, "frame", None),
                "units": units,
            }

        # 2) membership: which segments contain each key
        membership = {}
        for i, model in enumerate(self.mission_sequence.models):
            if not hasattr(model, "state"):
                continue
            seg_name = self.mission_sequence.names[i]
            seg_type = self.mission_sequence.types[i]
            leg_offsets = self._build_global_offsets(
                model.state
            )  # produces same key schema

            for key in leg_offsets.keys():
                entry = membership.setdefault(key, {"segments": [], "seg_types": []})
                entry["segments"].append(seg_name)
                entry["seg_types"].append(seg_type)

        # 3) assemble final meta dict in scenario order
        meta = {}
        for key, (gs, ge) in self._scenario_index_map.items():
            name, size, sid, frame_name, occ = key
            t = templ.get(key, {})
            m = membership.get(key, {"segments": [], "seg_types": []})

            meta[key] = {
                "name": name,
                "size": int(size),
                "sid": sid,
                "frame_name": frame_name,
                "occ": int(occ),
                "gs": int(gs),
                "ge": int(ge),
                "body": t.get("body", None),
                "frame": t.get("frame", None),
                "units": t.get("units", None),
                "segments": list(m["segments"]),
                "seg_types": list(m["seg_types"]),
            }

        return meta

    def _init_global_state_value(self):
        """
        Initialize the global state vector from the scenario definition and mission sequence models.

        This method constructs a global state vector by extracting initial state values from
        all models in the mission sequence. It handles the mapping between local model state
        definitions and a unified global state space.

        The process:
        1. Calculates the total size N of the global state vector from the scenario definition
        2. Iterates through each model in the mission sequence that has a 'state' attribute
        3. Builds a mapping of state keys (name, size, body SPICE ID, frame name, occurrence count)
            to their corresponding state objects (awf - attitude/widget/frame objects)
        4. For each mapped state, retrieves its initial values from the model and places them
            in the appropriate position within the global state vector
        5. Tracks which states have been filled to avoid duplicates

        Returns
        -------
        np.ndarray
            A 1D numpy array of dtype float containing the initial global state values,
            with size equal to the sum of all state sizes in the scenario definition.

        Raises
        ------
        KeyError
            If a state key from a model is not found in the scenario index map.
        """
        N = sum(int(t[1]) for t in self._scenario_full_state_def)
        xg = np.zeros((N,), dtype=float)

        # Fill from first time we see each key in models
        filled = set()
        for model in self.mission_sequence.models:
            if not hasattr(model, "state"):
                continue
            # offsets within this model
            local_off = self._build_global_offsets(model.state)
            for key, (ls, le) in local_off.items():
                if key in filled:
                    continue
                gs, ge = self._scenario_index_map[key]

                # grab the model's initial value template
                # model.state tuples are (..., awf)
                # need the matching awf: easiest is rebuild a dict from model.state using the same keying
                # build once per model:
            # build per-model dict
            occ_counter = {}
            model_map = {}
            for name, size, est_type, dyn_type, body, awf in model.state:
                base = (
                    name,
                    int(size),
                    getattr(body, "spice_id", None),
                    getattr(getattr(awf, "frame", None), "name", None),
                )
                occ = occ_counter.get(base, 0)
                occ_counter[base] = occ + 1
                k = (*base, int(occ))
                model_map[k] = awf

            for k, awf in model_map.items():
                if k in filled:
                    continue
                gs, ge = self._scenario_index_map[k]
                v = np.asarray(awf.quantity.values)
                v = v.reshape(-1)  # 1D
                xg[gs:ge] = v[: (ge - gs)]
                filled.add(k)

        return xg

    def _extract_leg_state_from_global(self, leg_state_def, seg_name):
        """
        Extract leg state values from the global state vector.

        This method retrieves state values for a specific leg segment from the global
        state vector and restructures them into the original leg state definition format.

        Parameters
        ----------
        leg_state_def : list
            List of tuples defining leg state structure.
            Each tuple contains: (name, size, est_type, dyn_type, body, ArrayWFrame)
            - name (str): State variable name
            - size (int or str): Number of elements for this state variable
            - est_type: Estimation type
            - dyn_type: Dynamics type
            - body: Body reference
            - ArrayWFrame: Template array with frame and units information
        seg_name : str
            Segment/leg name key to look up in leg-to-global mappings.

        Returns
        -------
        list
            List of tuples matching leg_state_def structure where values are
            extracted from the global state vector and wrapped with proper units
            and frame information. Each tuple contains:
            (name, size, est_type, dyn_type, body, ArrayWFrame)

        Raises
        ------
        KeyError
            If seg_name is not found in _leg_to_global_maps.
        IndexError
            If block indices exceed bounds of _global_state_value.
        """
        # returns list of tuples like leg_state_def but values taken from global vector
        blocks = self._leg_to_global_maps[seg_name]
        leg_values = np.zeros((sum(int(t[1]) for t in leg_state_def),), dtype=float)

        for ls, le, gs, ge in blocks:
            leg_values[ls:le] = self._global_state_value[gs:ge]

        # now wrap back into (name,size,est_type,dyn_type,body,ArrayWFrame)
        out = []
        cursor = 0
        for name, size, est_type, dyn_type, body, templ_awf in leg_state_def:
            vals = leg_values[cursor : cursor + int(size)]
            out.append(
                (
                    name,
                    size,
                    est_type,
                    dyn_type,
                    body,
                    ArrayWFrame(
                        ArrayWUnits(vals, templ_awf.quantity.units), templ_awf.frame
                    ),
                )
            )
            cursor += int(size)
        return out

    def _update_global_from_leg_terminal(
        self, leg_state_def, seg_name, terminal_state_list
    ):
        """
        Update the global state vector with terminal values from a leg segment.

        This method extracts the final state values from a leg segment and maps them
        to the corresponding positions in the global state vector using pre-computed
        index mappings.

        Parameters
        ----------
        leg_state_def :
            Definition of the leg state structure (not used in method body).
        seg_name : str
            The name of the segment/leg to update from.
        terminal_state_list : list
            A list of tuples containing terminal state data,
            where each tuple is (name, size, ..., awf) where awf is an object with
            a .quantity.values attribute containing the final state values.

        Returns
        -------
        None
            Modifies self._global_state_value in place.

        Notes
        -----
        - Extracts the last sample (-1 index) from multi-epoch values in awf.quantity.values
        - Handles both 1D and multi-dimensional state arrays
        - Uses pre-computed block mappings in self._leg_to_global_maps[seg_name]
          to transfer data from leg indices [ls:le] to global indices [gs:ge]
        """
        blocks = self._leg_to_global_maps[seg_name]

        leg_terminal = []
        for name, size, *_, awf in terminal_state_list:
            v = np.asarray(awf.quantity.values, dtype=float)
            # sa.state awf has shape (Nt, dim) — always take last time step
            if v.ndim >= 2:
                v = v[-1, :]  # shape (dim,)
            v = np.atleast_1d(v).reshape(-1)
            leg_terminal.append(v)
        leg_terminal = np.concatenate(leg_terminal)

        for ls, le, gs, ge in blocks:
            self._global_state_value[gs:ge] = leg_terminal[ls:le]

    def _build_global_offsets(self, state_def):
        """
        Build a dictionary mapping state variable identifiers to their memory offsets.

        This method creates a mapping of state variables to their corresponding positions
        in a flattened state array. Each variable is assigned a tuple representing its
        start and end indices (cursor positions) in the global state buffer.

        Parameters
        ----------
        state_def : list
            A list of tuples, each containing:
            - name (str): The name of the state variable
            - size (int): The size of the state variable
            - est_type: The estimation type
            - dyn_type: The dynamics type
            - body: An object with optional 'spice_id' attribute
            - awf: An object with optional 'frame' attribute containing 'name'

        Returns
        -------
        dict
            A dictionary where keys are tuples of (name, size, spice_id, frame_name, occurrence_index)
            and values are tuples of (start_offset, end_offset) representing the memory location
            of each state variable in the global state array.

        Examples
        --------
        If three variables are added with sizes [2, 3, 2], the returned offsets would be:
        {(var1_key, 0): (0, 2), (var2_key, 0): (2, 5), (var3_key, 0): (5, 7)}
        """
        offsets = {}
        cursor = 0
        occ_counter = {}
        for name, size, est_type, dyn_type, body, awf in state_def:
            base = (
                name,
                int(size),
                getattr(body, "spice_id", None),
                getattr(getattr(awf, "frame", None), "name", None),
            )
            occ = occ_counter.get(base, 0)
            occ_counter[base] = occ + 1
            key = (*base, int(occ))
            offsets[key] = (cursor, cursor + int(size))
            cursor += int(size)
        return offsets

    def _build_scenario_state_def(self):
        """
        Build a unified state definition from all models in the mission sequence.
        This method aggregates state definitions across all models in the mission sequence,
        ensuring that each state component appears only once while preserving a deterministic
        order based on first appearance.
        The method handles duplicate components by comparing their base attributes (name, size,
        body spice_id, and frame name) and occurrence count. Components are deduplicated to
        prevent redundant state entries while maintaining the order in which they are first
        encountered across models.
        Returns
        -------
        StateDefinition
            A unified StateDefinition object constructed from the deduplicated union of all
            state components across all models in the mission sequence.
        Notes
        -----
        - Only models with a "state" attribute are considered.
        - Deduplication is based on component name, size, body spice_id, frame name, and
          occurrence count within each state list.
        - The deterministic order is preserved by iterating through models in sequence and
          adding unseen components as they appear.
        """

        # Gather all components across all segments that have a "state"
        all_defs = []
        for model in self.mission_sequence.models:
            if hasattr(model, "state"):
                all_defs.append(list(model.state))  # list of tuples

        # Union preserving deterministic order:
        # Start from the first model's order, then append unseen components as they appear.
        union = []
        seen = set()

        for state_list in all_defs:
            occ_counter = {}
            for name, size, est_type, dyn_type, body, awf in state_list:
                base = (
                    name,
                    int(size),
                    getattr(body, "spice_id", None),
                    getattr(getattr(awf, "frame", None), "name", None),
                )
                occ = occ_counter.get(base, 0)
                occ_counter[base] = occ + 1
                key = (*base, int(occ))
                if key in seen:
                    continue
                seen.add(key)
                union.append((name, size, est_type, dyn_type, body, awf))

        return StateDefinition.from_components(union)

    def _sid(self, body):
        return getattr(body, "spice_id", None)

    def _frame_name(self, awf):
        return getattr(getattr(awf, "frame", None), "name", None)

    def _comp_key(self, name, size, body, awf, occ=0):
        return (name, int(size), self._sid(body), self._frame_name(awf), int(occ))

    def _pick_primary(self, name: str):
        """
        Return (awf, epochs) for the 'primary' component history.

        Strategy:
        1) Match by primary_body.spice_id (if available) and occ==0
        2) Match by body object identity (meta['body'] == primary_body) and occ==0
        3) Fallback: first component with occ==0 regardless of body
        4) Fallback: first component with matching name
        """
        if not self._total_components:
            raise RuntimeError(
                "No propagated totals found. Call scenario.propagate() before accessing total_* properties."
            )

        pb = self.primary_body
        pb_sid = getattr(pb, "spice_id", None)

        # 1) strict match by spice_id if defined
        if pb_sid is not None:
            for key, item in self._total_components.items():
                kname, _, ksid, _, kocc = key
                if kname == name and ksid == pb_sid and kocc == 0:
                    return item["awf"], item["epochs"]

        # 2) match by body object identity (uses what you already store in meta)
        for key, item in self._total_components.items():
            meta = item.get("meta", {})
            if (
                meta.get("name") == name
                and meta.get("occ") == 0
                and meta.get("body") is pb
            ):
                return item["awf"], item["epochs"]

        # 3) first occurrence (occ==0) regardless of body
        for key, item in self._total_components.items():
            kname, _, _, _, kocc = key
            if kname == name and kocc == 0:
                return item["awf"], item["epochs"]

        # 4) any match
        for key, item in self._total_components.items():
            if key[0] == name:
                return item["awf"], item["epochs"]

        return None, None

    def _append_total_from_state(self, state_list, epochs: EpochArray, drop_last: bool):
        """
        Accumulate total histories per component key:
        key = (name, size, spice_id, frame_name, occurrence)

        state_list: list of (name,size,est_type,dyn_type,body,awf) from StateArray.state
        epochs: EpochArray for that segment
        drop_last: drop endpoint to avoid duplication at nodes
        """
        # Slice epochs
        e = epochs[:-1] if drop_last else epochs
        e_vals = e.times.values
        e_units = e.times.units
        e_frame = e.system

        # occurrence counting *within this state_list*
        occ_counter = {}
        for name, size, _, _, body, awf in state_list:
            base = (name, int(size), self._sid(body), self._frame_name(awf))
            occ = occ_counter.get(base, 0)
            occ_counter[base] = occ + 1

            key = (name, int(size), self._sid(body), self._frame_name(awf), int(occ))

            # Slice / expand data
            q = awf.quantity
            raw_vals = np.asarray(q.values)

            # Number of time samples retained for this segment
            n_keep = len(e_vals)

            if raw_vals.ndim == 0:
                # scalar parameter -> repeat over segment
                vals = np.full((n_keep, 1), float(raw_vals))
            elif raw_vals.ndim == 1:
                if raw_vals.size == int(size):
                    # component vector template (e.g. static 3-vector or scalar packed as len=size)
                    vals = np.tile(raw_vals.reshape(1, -1), (n_keep, 1))
                else:
                    # already a 1D time history for scalar quantity
                    vals = raw_vals[:-1] if drop_last else raw_vals
                    vals = np.asarray(vals).reshape(-1, 1)
            else:
                # assume first dimension is time
                vals = raw_vals[:-1] if drop_last else raw_vals

            new_awf = ArrayWFrame(ArrayWUnits(vals, q.units), awf.frame)
            new_epochs = EpochArray(ArrayWUnits(e_vals, e_units).values, sys=e_frame)

            if key not in self._total_components:
                self._total_components[key] = {
                    "awf": new_awf,
                    "epochs": new_epochs,
                    "meta": {
                        "name": name,
                        "size": int(size),
                        "body": body,
                        "frame": awf.frame,
                        "occ": int(occ),
                    },
                }
            else:
                prev_awf = self._total_components[key]["awf"]
                prev_ep = self._total_components[key]["epochs"]

                stacked_vals = np.vstack(
                    (prev_awf.quantity.values, new_awf.quantity.values)
                )
                stacked_time = np.hstack(
                    (prev_ep.times.values, new_epochs.times.values)
                )

                self._total_components[key]["awf"] = ArrayWFrame(
                    ArrayWUnits(stacked_vals, prev_awf.quantity.units), prev_awf.frame
                )
                self._total_components[key]["epochs"] = EpochArray(
                    ArrayWUnits(stacked_time, prev_ep.times.units).values,
                    sys=prev_ep.system,
                )

    def _build_full_propagated_state_array(self) -> StateArray:
        """
        Build a StateArray covering the full propagated mission from _total_components.

        Uses the accumulated per-component time histories to construct a single
        StateArray whose epoch spans from the start of the first propagated leg to
        the end of the last.  Static parameters are represented by their first
        time-step value; dynamic parameters retain the full time history.

        Returns the last-leg StateArray as a fallback if _total_components is empty.
        """
        if not self._total_components:
            return self._propagated_state_array

        _, full_epochs = self._pick_primary("position")
        if full_epochs is None:
            return self._propagated_state_array

        # Build lookup: key -> (name, size, est_type, dyn_type, body, template_awf)
        scen_def_map = {}
        occ_counter = {}
        for name, size, est_type, dyn_type, body, awf in self._scenario_full_state_def:
            base = (name, int(size), self._sid(body), self._frame_name(awf))
            occ = occ_counter.get(base, 0)
            occ_counter[base] = occ + 1
            key = (*base, int(occ))
            scen_def_map[key] = (name, size, est_type, dyn_type, body, awf)

        # Build state component list with full time-history values
        components = []
        for key, item in self._total_components.items():
            if key not in scen_def_map:
                continue
            name, size, est_type, dyn_type, body, templ_awf = scen_def_map[key]
            full_awf = item["awf"]  # ArrayWFrame with shape (N_total, dim)

            if dyn_type == "static":
                # Use only the representative single value for static parameters
                vals = np.asarray(full_awf.quantity.values)
                if vals.ndim > 1:
                    vals = vals[0]
                use_awf = ArrayWFrame(
                    ArrayWUnits(vals, full_awf.quantity.units), full_awf.frame
                )
            else:
                use_awf = full_awf

            components.append((name, size, est_type, dyn_type, body, use_awf))

        if not components:
            return self._propagated_state_array

        sd = StateDefinition.from_components(components)
        return StateArray(
            epoch=full_epochs,
            origin=self._origin,
            state=sd,
        )

    def propagate(self, verbose: bool = True) -> None:
        """
        Executes the propagation of the mission sequence through all defined legs and impulsive burns.

        This version uses a SCENARIO-GLOBAL state vector (union of all components across all legs)
        as the single source of truth for initial conditions and continuity across segments.

        Key behaviors:
        - For each Leg/Finite Burn:
            * build leg->global mapping
            * extract leg IC from global vector
            * apply any pending impulsive burn to BOTH (global and leg IC) consistently
            * propagate
            * update global vector from terminal propagated state
        - For each Impulsive Burn:
            * snapshot segment outputs
            * store pending dv to be applied at next leg
            * (global velocity is updated at the next leg, using that leg’s velocity key)

        Parameters
        ----------
        verbose : bool, optional
            When ``True`` (default), print segment progress and initial-condition
            summaries to stdout during propagation.  Set to ``False`` to silence
            all output.
        """

        # ----------------------------
        # Local helpers
        # ----------------------------
        def _get_component_awf(state_list, comp_name: str):
            for name, *_, awf in state_list:
                if name == comp_name:
                    return awf
            raise StopIteration(
                f"Component '{comp_name}' not found in propagated StateArray.state"
            )

        def _first_velocity_key(leg_offsets: dict):
            # leg_offsets keys are (name, size, sid, frame_name, occ)
            vel_keys = [k for k in leg_offsets.keys() if k[0] == "velocity"]
            if len(vel_keys) == 0:
                return None
            # Deterministic choice: smallest occ then smallest sid then frame_name
            vel_keys.sort(
                key=lambda k: (k[4], k[2] if k[2] is not None else -1, str(k[3]))
            )
            return vel_keys[0]

        # ----------------------------
        # Pending DV bookkeeping
        # ----------------------------
        # We apply dv to global at the NEXT leg, once we know the leg's velocity key mapping.
        if not hasattr(self, "_pending_dv_vals"):
            self._pending_dv_vals = None
            self._pending_dv_units = None
            self._pending_dv_frame_name = None

        # ---------------------------------
        # Initialize "previous" leg context (for burn snapshots)
        # ---------------------------------
        last_r = None
        last_v = None
        last_par = None
        last_stms = None
        last_epochs = None
        last_leg_offsets = (
            None  # store offsets from last propagated leg (useful for dv routing)
        )

        # Loop through mission sequence
        for sequence in range(self.len_sequence):

            seg_type = self.mission_sequence.types[sequence]
            seg_name = self.mission_sequence.names[sequence]
            if verbose:
                print(
                    f"\n--- Propagating Segment {sequence + 1}/{self.len_sequence}: '{seg_name}' [{seg_type}] ---"
                )

            # ============================================================
            # 1) Legs / Finite burns (propagated arcs)
            # ============================================================
            if seg_type in ("Leg", "Finite Burn"):

                epochs = self.mission_sequence.epochs_vec[sequence]
                self.propagator = self.mission_sequence.props[sequence]
                atol = self.propagator.atol
                rtol = self.propagator.rtol
                leg_model = self.mission_sequence.models[sequence]

                # -------------------------
                # Build / refresh leg->global mapping for this segment
                # -------------------------
                leg_state = leg_model.state  # list of tuples
                leg_offsets = self._build_global_offsets(leg_state)

                map_blocks = []
                for leg_key, (ls, le) in leg_offsets.items():
                    if leg_key not in self._scenario_index_map:
                        raise KeyError(
                            f"Leg component key {leg_key} not found in scenario index map. "
                            "Scenario union state_def is missing a component present in this leg."
                        )
                    gs, ge = self._scenario_index_map[leg_key]
                    map_blocks.append((ls, le, gs, ge))
                self._leg_to_global_maps[seg_name] = map_blocks

                # -------------------------
                # If there is a pending impulsive burn, apply it to the GLOBAL state now
                # (so scenario-global state is correct at the burn node)
                # -------------------------
                if self._pending_dv_vals is not None:
                    vel_key = _first_velocity_key(leg_offsets)
                    if vel_key is None:
                        raise RuntimeError(
                            "Pending delta-v exists but this leg has no 'velocity' component."
                        )
                    gs, ge = self._scenario_index_map[vel_key]
                    if (ge - gs) != 3:
                        raise RuntimeError(
                            f"Velocity block is not 3D (size={ge-gs}); cannot apply delta-v safely."
                        )
                    # Units/frame checks are already enforced at burn time; we just apply numeric dv here
                    self._global_state_value[gs:ge] = (
                        self._global_state_value[gs:ge] + self._pending_dv_vals
                    )

                    # clear pending dv
                    self._pending_dv_vals = None
                    self._pending_dv_units = None
                    self._pending_dv_frame_name = None

                # -------------------------
                # Extract leg IC from scenario-global vector
                # -------------------------
                leg_state_list = self._extract_leg_state_from_global(
                    leg_state, seg_name
                )

                # -------------------------
                # Convert position/velocity if origin changed between legs.
                # Find the last real propagated leg's origin to compare against.
                # -------------------------
                leg_origin = self.mission_sequence.props[sequence].prop_origin
                prev_origin = None
                for k in range(sequence - 1, -1, -1):
                    if self.mission_sequence.types[k] in ("Leg", "Finite Burn"):
                        prev_origin = self.mission_sequence.props[k].prop_origin
                        break
                if prev_origin is None:
                    prev_origin = leg_origin  # first leg, no conversion needed

                origins_differ = (
                    leg_origin is not None
                    and prev_origin is not None
                    and getattr(leg_origin, "name", None)
                    != getattr(prev_origin, "name", None)
                )

                if origins_differ:
                    t_boundary = float(epochs[0].times.values)
                    frame_name = getattr(leg_state_list[0][5].frame, "name", "J2000")
                    # get_state(new_origin w.r.t. prev_origin) then subtract
                    # works in both directions:
                    #   Sun->Mars: subtract pos(Mars w.r.t. Sun)
                    #   Mars->Sun: subtract pos(Sun w.r.t. Mars) = add pos(Mars w.r.t. Sun)
                    offset_state = SpiceManager.get_state(
                        leg_origin.name,
                        t_boundary,
                        frame_name,
                        prev_origin.name,
                    )
                    offset_pos = np.asarray(offset_state.values[0:3])
                    offset_vel = np.asarray(offset_state.values[3:6])

                    converted = []
                    for name, size, est_type, dyn_type, body, awf in leg_state_list:
                        if name == "position":
                            vals = np.asarray(awf.quantity.values) - offset_pos
                            new_awf = ArrayWFrame(
                                ArrayWUnits(vals, awf.quantity.units), awf.frame
                            )
                            converted.append(
                                (name, size, est_type, dyn_type, body, new_awf)
                            )
                        elif name == "velocity":
                            vals = np.asarray(awf.quantity.values) - offset_vel
                            new_awf = ArrayWFrame(
                                ArrayWUnits(vals, awf.quantity.units), awf.frame
                            )
                            converted.append(
                                (name, size, est_type, dyn_type, body, new_awf)
                            )
                        else:
                            converted.append(
                                (name, size, est_type, dyn_type, body, awf)
                            )
                    leg_state_list = converted

                # Create fresh StateArray for this leg
                self._state_vector_old = StateArray(
                    epoch=epochs[0],
                    origin=self.mission_sequence.props[sequence].prop_origin,
                    state=StateDefinition.from_components(leg_state_list),
                )

                # -------------------------
                # STM sized/mapped to this leg definition
                # -------------------------
                new_STM = self._update_stm(leg_state)

                # -------------------------
                # Apply delta_v model to the LEG IC as well (if you still use this hook)
                # (Global has already been updated above; this keeps the leg IC consistent
                #  with whatever reinitialize_with_dv does internally.)
                # -------------------------
                if self.delta_v is not None:
                    self._delta_v = None

                # -------------------------
                # Initialize and propagate
                # -------------------------
                if verbose:
                    print(
                        f"\n[IC] segment='{seg_name}' type='{seg_type}' epoch={epochs[0]}"
                    )
                    for (
                        name,
                        size,
                        est_type,
                        dyn_type,
                        body,
                        awf,
                    ) in self._state_vector_old.state:
                        sid = getattr(body, "spice_id", None)
                        frame = getattr(getattr(awf, "frame", None), "name", None)
                        vals = np.asarray(awf.quantity.values).reshape(-1)
                        head = vals[: min(vals.size, 6)]
                        print(
                            f"  {name:<12s} n={int(size):2d} sid={sid} frame={frame} vals[:6]={head}"
                        )

                self.propagator.reinitialize(
                    state_vector=self._state_vector_old,
                    tspan=epochs,
                    STM0=new_STM,
                    atol=atol,
                    rtol=rtol,
                )

                self.propagator.propagate()
                sa = self.propagator.propagated_state_array
                self._propagated_state_array = sa

                # -------------------------
                # Update scenario-global vector from terminal propagated state
                # -------------------------
                self._update_global_from_leg_terminal(leg_state, seg_name, sa.state)

                # useful printout to know what the global state vector looks like at each node,
                # and to verify the leg->global mapping is correct
                print(f"\n[GLOBAL STATE after '{seg_name}']")
                for key, (gs, ge) in self._scenario_index_map.items():
                    print(
                        f"  key={key} -> global[{gs}:{ge}] = {self._global_state_value[gs:ge]}"
                    )

                # -------------------------
                # Update legacy tracking variable
                # -------------------------
                self._state_size_old = sum(int(t[1]) for t in leg_state)
                self._et_old = epochs[-1]

                # Update state_vector_old to terminal state (needed by _update_stm and _merge_state_list_with_old)
                terminal_state_list = self._extract_leg_state_from_global(
                    leg_state, seg_name
                )
                self._state_vector_old = StateArray(
                    epoch=epochs[-1],
                    origin=self.mission_sequence.props[sequence].prop_origin,
                    state=StateDefinition.from_components(terminal_state_list),
                )

                # -------------------------
                # Extract core components
                # -------------------------
                r = _get_component_awf(sa.state, "position")
                v = _get_component_awf(sa.state, "velocity")

                # "parameters" (optional legacy output): keep as list of components, not flattened
                # to avoid object stacking issues.
                other_components = [
                    (name, awf)
                    for name, *_, awf in sa.state
                    if name not in ("position", "velocity")
                ]
                par = other_components if len(other_components) > 0 else None

                stms = self.propagator.STM if self.propagator.propagate_STM else None

                # Update STM_old from last propagated STM (needed by _update_stm at next node)
                if self.propagator.propagate_STM and stms is not None:
                    stm_last = np.asarray(stms.quantity.values)[-1, :].reshape(
                        self._state_size_old, self._state_size_old
                    )
                    self._STM_old = ArrayWFrame(
                        ArrayWUnits(stm_last, stms.quantity.units),
                        stms.frame,
                    )
                else:
                    self._STM_old = None

                # -------------------------
                # Store per-segment outputs
                # -------------------------
                self.segment_positions[seg_name] = r
                self.segment_velocities[seg_name] = v
                self.segment_parameters[seg_name] = par
                self.segment_STMs[seg_name] = stms
                self.segment_epochsTDB[seg_name] = epochs.times.values

                # -------------------------
                # Store concatenated totals
                # -------------------------
                drop_last = sequence != self.len_sequence - 1
                self._append_total_from_state(sa.state, epochs, drop_last=drop_last)

                # Legacy histories
                self._STM.append(stms)
                self._total_parameters.append(par if par is not None else None)

                # Update last-context for subsequent impulsive burn snapshot
                last_r, last_v, last_par, last_stms, last_epochs = (
                    r,
                    v,
                    par,
                    stms,
                    epochs,
                )
                last_leg_offsets = leg_offsets

            # ============================================================
            # 2) Impulsive Burn event (no propagation)
            # ============================================================
            if seg_type == "Impulsive Burn":

                if last_r is None or last_v is None or last_epochs is None:
                    raise RuntimeError(
                        f"Encountered Impulsive Burn '{seg_name}' before any propagated leg. "
                        "Burn events must follow a Leg/Finite Burn."
                    )

                model = self.mission_sequence.models[sequence]
                if verbose:
                    print(f"\n{model}")

                # Store delta_v model to be applied at the next leg initialization
                self._delta_v = model

                # Snapshot pre-burn r/v
                r_last_vals = np.asarray(last_r.quantity.values)[-1, :].reshape(
                    3,
                )
                v_last_vals = np.asarray(last_v.quantity.values)[-1, :].reshape(
                    3,
                )

                r_last_awf = ArrayWFrame(
                    ArrayWUnits(r_last_vals, last_r.quantity.units), last_r.frame
                )
                v_last_awf = ArrayWFrame(
                    ArrayWUnits(v_last_vals, last_v.quantity.units), last_v.frame
                )

                self._segment_positions[seg_name] = r_last_awf
                self._segment_velocities[seg_name] = v_last_awf  # pre-burn snapshot

                # Extract delta-v as 3-vector
                dv = model.delta_v

                if hasattr(dv, "quantity"):
                    dv_vals = np.asarray(dv.quantity.values).reshape(
                        3,
                    )
                    dv_units = dv.quantity.units
                    dv_frame = getattr(dv, "frame", None)
                else:
                    dv_vals = np.asarray(dv.values).reshape(
                        3,
                    )
                    dv_units = dv.units
                    dv_frame = None

                # Frame + units checks
                if dv_frame is not None:
                    if getattr(dv_frame, "name", None) != getattr(
                        last_v.frame, "name", None
                    ):
                        raise TypeError(
                            "delta_v frame must match velocity frame. "
                            f"Got dv.frame={getattr(dv_frame, 'name', None)} and v.frame={getattr(last_v.frame, 'name', None)}."
                        )

                if getattr(model, "base_frame", None) is not None:
                    if getattr(model.base_frame, "name", None) != getattr(
                        last_v.frame, "name", None
                    ):
                        raise TypeError(
                            "The base frame of delta_v and the velocity must be the same. "
                            f"Got dv base_frame={getattr(model.base_frame, 'name', None)} "
                            f"and v.frame={getattr(last_v.frame, 'name', None)}."
                        )

                if dv_units != last_v.quantity.units:
                    raise TypeError(
                        "delta_v units must match velocity units. "
                        f"Got dv.units={dv_units} and v.units={last_v.quantity.units}."
                    )

                # Post-burn velocity snapshot (for logs)
                v_post_vals = v_last_vals + dv_vals
                v_post_awf = ArrayWFrame(
                    ArrayWUnits(v_post_vals, last_v.quantity.units), last_v.frame
                )
                self._segment_velocities[seg_name] = v_post_awf

                # Store pending dv numerically; apply to global at next leg
                self._pending_dv_vals = dv_vals
                self._pending_dv_units = dv_units
                self._pending_dv_frame_name = (
                    getattr(dv_frame, "name", None)
                    if dv_frame is not None
                    else getattr(last_v.frame, "name", None)
                )

                # Parameters at burn time (legacy): keep as already stored from last leg
                self._segment_parameters[seg_name] = (
                    last_par if last_par is not None else None
                )

                # STM snapshot at burn time
                if last_stms is not None and hasattr(last_stms, "quantity"):
                    st_last = np.asarray(last_stms.quantity.values)[-1, ...]
                    self._segment_STMs[seg_name] = ArrayWFrame(
                        ArrayWUnits(st_last, last_stms.quantity.units),
                        last_stms.frame,
                    )
                else:
                    self._segment_STMs[seg_name] = None

                # Epoch at burn time: keep consistent type (array)
                self._segment_epochsTDB[seg_name] = np.atleast_1d(
                    last_epochs.times.values[-1]
                )

        # Build and cache the full-mission StateArray from all accumulated legs
        self._full_propagated_state_array = self._build_full_propagated_state_array()

    def _merge_state_list_with_old(self, new_list: list) -> list:
        """
        Merge a new state list with the previous state vector, inheriting values from old parameters.
        This method matches parameters in the new list with those from the previous state vector
        using a composite key of (parameter name, spice_id, size). For matched parameters, the
        previous quantity values are inherited; for new parameters, their initial condition values
        are used. All parameters are wrapped in new ArrayWFrame objects.
        Parameters
        ----------
        new_list : list
            A list of tuples, each containing:
            - cname (str): Parameter name
            - csize: Parameter size
            - cest: Parameter estimate
            - cdyn: Parameter dynamics
            - cbody: Body object (may have spice_id attribute)
            - cawf (ArrayWFrame): Parameter with quantity and frame information

        Returns
        -------
        list
            A list of updated tuples with the same structure as new_list, where each
            tuple contains (cname, csize, cest, cdyn, cbody, new_awf) with new_awf
            containing inherited or initial quantity values.
        """

        def _sid(body):
            return getattr(body, "spice_id", None)

        # Build lookup map from previous state
        prev_map = {}
        for pname, psize, _, _, pbody, pawf in self._state_vector_old.state:
            key = (pname, _sid(pbody), int(psize))
            prev_map[key] = pawf

        updated = []

        for cname, csize, cest, cdyn, cbody, cawf in new_list:

            key = (cname, _sid(cbody), int(csize))

            if key in prev_map:
                # Inherit previous value
                prev_awf = prev_map[key]
                values = prev_awf.quantity.values
            else:
                # Truly new parameter → use its own initial condition
                values = cawf.quantity.values

            new_awf = ArrayWFrame(
                ArrayWUnits(values, cawf.quantity.units),
                cawf.frame,
            )

            updated.append((cname, csize, cest, cdyn, cbody, new_awf))

        return updated

    def _update_stm(self, new_list: list):
        """
        Updates the State Transition Matrix (STM) to match a new state vector layout.

        This method constructs a new STM matrix based on the provided `new_list` layout,
        copying matching blocks and cross-partials from the previous STM if available.
        It also handles injection of maneuver delta-v blocks if specified.

        Parameters
        ----------
        new_list : list
            The new state vector layout, where each element is a tuple
            containing (name, size, ..., body, ...). The layout determines the block
            structure of the STM.

        Returns
        -------
        np.ndarray
            The updated STM matrix, sized according to the new state vector layout.

        Raises
        ------
        ValueError
            If the previous STM matrix (`STM_old`) is not square.

        Notes
        -----
        - If no previous STM exists or mapping is disabled, returns an identity matrix,
          with optional delta-v maneuver injection.
        - Matching blocks are copied by (name, spice_id) and occurrence index.
        - Cross-partials between matching blocks are also copied.
        - Delta-v maneuver blocks ("dv_man", "dv_man1", "dv_man2") are injected into
          corresponding velocity blocks if `self.delta_v` is set.
        """

        def _sid(body):
            return getattr(body, "spice_id", None)

        def _build_block_offsets(state_list):
            """
            Returns
            -------
            mapping : dict
                (name, sid, occurrence_index) -> (start, end, size)
            mapping_simple : dict
                (name, sid) -> list of (start,end,size)
            """
            mapping = {}
            mapping_simple = {}
            cursor = 0
            counts = {}
            for name, size, _, _, body, _ in state_list:
                sid = _sid(body)
                k_simple = (name, sid)
                counts[k_simple] = counts.get(k_simple, 0) + 1
                occ = counts[k_simple] - 1
                start = cursor
                end = cursor + int(size)
                mapping[(name, sid, occ)] = (start, end, int(size))
                mapping_simple.setdefault((name, sid), []).append(
                    (start, end, int(size))
                )
                cursor = end
            return mapping, mapping_simple, cursor

        # New STM size
        new_state_size = sum(int(t[1]) for t in new_list)
        new_STM = np.eye(new_state_size)

        # If we don't have an old STM (first leg), just return identity (+ dv_man tweak below)
        if getattr(self, "_STM_old", None) is None or not self.map_STM_to_t0:
            # Still apply dv_man MTM structure if requested
            if self.delta_v is not None:
                _, new_simple, _ = _build_block_offsets(new_list)
                # Find velocity block (prefer same owner as dv_man if ambiguous)
                vel_candidates = [k for k in new_simple.keys() if k[0] == "velocity"]
                for dv_key in ("dv_man", "dv_man1", "dv_man2"):
                    dv_candidates = [k for k in new_simple.keys() if k[0] == dv_key]
                    if not dv_candidates or not vel_candidates:
                        continue
                    # Pair by same sid when possible
                    for dvk in dv_candidates:
                        sid = dvk[1]
                        velk = (
                            ("velocity", sid)
                            if ("velocity", sid) in new_simple
                            else vel_candidates[0]
                        )
                        v0, v1, vdim = new_simple[velk][0]
                        d0, d1, ddim = new_simple[dvk][0]
                        k = vdim if vdim < ddim else ddim
                        new_STM[v0:v1, d0:d1] = np.eye(k)
            return new_STM

        # Build old layout maps using the *existing* state_vector_old.state (previous definition)
        old_list = list(self._state_vector_old.state)

        old_blocks, old_simple, old_size = _build_block_offsets(old_list)
        new_blocks, new_simple, _ = _build_block_offsets(new_list)

        # Old STM numeric matrix (support ArrayWFrame/ArrayWUnits or ndarray)
        try:
            STM_old_mat = np.asarray(self._STM_old.quantity.values, dtype=float)
        except Exception:
            STM_old_mat = np.asarray(self._STM_old, dtype=float)

        # Safety: ensure square
        if STM_old_mat.ndim != 2 or STM_old_mat.shape[0] != STM_old_mat.shape[1]:
            raise ValueError(f"STM_old must be square, got shape {STM_old_mat.shape}.")

        # Copy matching blocks from old STM into new STM
        # We match by (name,sid) and then by occurrence index if multiple.
        for (name, sid), new_list_blocks in new_simple.items():
            old_list_blocks = old_simple.get((name, sid), None)
            if old_list_blocks is None:
                continue
            nmatch = min(len(new_list_blocks), len(old_list_blocks))
            for k in range(nmatch):
                n0, n1, nd = new_list_blocks[k]
                o0, o1, od = old_list_blocks[k]
                dd = min(nd, od)

                # Diagonal block
                new_STM[n0 : n0 + dd, n0 : n0 + dd] = STM_old_mat[
                    o0 : o0 + dd, o0 : o0 + dd
                ]

                # Cross blocks: for every other parameter that also matches, copy cross-partials
                for (name2, sid2), new_list_blocks2 in new_simple.items():
                    old_list_blocks2 = old_simple.get((name2, sid2), None)
                    if old_list_blocks2 is None:
                        continue
                    nmatch2 = min(len(new_list_blocks2), len(old_list_blocks2))
                    for k2 in range(nmatch2):
                        m0, m1, md = new_list_blocks2[k2]
                        p0, p1, pd = old_list_blocks2[k2]
                        dd_r = min(nd, od)
                        dd_c = min(md, pd)
                        new_STM[n0 : n0 + dd_r, m0 : m0 + dd_c] = STM_old_mat[
                            o0 : o0 + dd_r, p0 : p0 + dd_c
                        ]

        # dv_man -> MTM injection
        if self.delta_v is not None:
            vel_candidates = [k for k in new_simple.keys() if k[0] == "velocity"]
            for dv_key in ("dv_man", "dv_man1", "dv_man2"):
                dv_candidates = [k for k in new_simple.keys() if k[0] == dv_key]
                if not dv_candidates or not vel_candidates:
                    continue

                for dvk in dv_candidates:
                    sid = dvk[1]
                    velk = (
                        ("velocity", sid)
                        if ("velocity", sid) in new_simple
                        else vel_candidates[0]
                    )
                    v0, v1, vdim = new_simple[velk][0]
                    d0, d1, ddim = new_simple[dvk][0]
                    new_STM[v0:v1, d0:d1] = np.eye(min(vdim, ddim))

        return new_STM

    def _reinitialize_scenario_state_def(
        self,
        new_state_def: StateDefinition,
        *,
        refill_from_models: bool = True,
        update_models: bool = True,
        update_states_n: bool = False,
    ) -> None:
        """
        Reinitialize the scenario state definition and propagate changes throughout the mission sequence.
        This method performs a comprehensive state schema migration that:
        1. Snapshots the current global state vector
        2. Installs the new state definition and associated index mapping
        3. Projects values from the old global state to the new one, preserving overlapping keys
        4. Optionally refills new state variables from the first model that provides them
        5. Updates all StateArray models with new values from the global state vector
        6. Reinitializes ImpulsiveBurn models from their corresponding DV parameter blocks
        7. Resets cached state tracking variables (STM, state vector, state size)
        Parameters
        ----------
        new_state_def : StateDefinition
            The new state definition schema to install. Contains all state variable specifications.
        refill_from_models : bool, optional
            If True, fill new state variables (not present in the old state) from initial model
            state values. The first model in the mission sequence that contains a matching state
            variable will seed the value. Default is True.
        update_models : bool, optional
            If True, propagate the updated global state values back into all StateArray model
            instances in the mission sequence. Default is True.
        update_states_n : bool, optional
            If True, update the states_n tracking array with the new state dimensions for each
            rebuilt model. Default is False.
        Returns
        -------
        None
        Notes
        -----
        - State key matching is based on (name, size, spice_id, frame_name, occurrence_index)
        - Only overlapping state variables are copied from old to new; missing variables either
          remain zero-initialized or are filled from models if refill_from_models is True.
        - ImpulsiveBurn models are matched to DV parameters by their burn occurrence order.
        - The method assumes DV parameters follow the naming convention "dv_man", "dv_man1", etc.
        - Cached fields (_state_vector_old, _state_size_old, _STM_old, _delta_v) are reset.
        """

        def _flatten_vec(x) -> np.ndarray:
            """Return a contiguous 1D float vector from any array-like."""
            if x is None:
                return None
            a = np.asarray(x, dtype=float)
            return a.reshape(-1)

        def _awf_template_to_1d(awf) -> np.ndarray:
            """
            Extract a 1D template vector from an ArrayWFrame quantity.
            If quantity.values is (Nt,dim), use the first row as the template.
            """
            v = np.asarray(awf.quantity.values, dtype=float)
            if v.ndim == 0:
                return v.reshape(1)
            if v.ndim == 1:
                return v.reshape(-1)
            # v.ndim >= 2: treat as time-history -> take first sample as template
            return v[0, :].reshape(-1)

        def _model_key_map(state_list):
            """key=(name,size,sid,frame,occ) -> (name,size,est_type,dyn_type,body,awf)"""
            occ_counter = {}
            m = {}
            for name, size, est_type, dyn_type, body, awf in state_list:
                base = (
                    name,
                    int(size),
                    getattr(body, "spice_id", None),
                    getattr(getattr(awf, "frame", None), "name", None),
                )
                occ = occ_counter.get(base, 0)
                occ_counter[base] = occ + 1
                key = (*base, int(occ))
                m[key] = (name, size, est_type, dyn_type, body, awf)
            return m

        # -------------------------
        # 1) Snapshot old global state/map (flattened!)
        # -------------------------
        old_def = getattr(self, "_scenario_full_state_def", None)
        old_map = getattr(self, "_scenario_index_map", None)
        old_xg = _flatten_vec(getattr(self, "_global_state_value", None))

        # -------------------------
        # 2) Install new schema + new index map
        # -------------------------
        self._scenario_full_state_def = new_state_def
        self._scenario_index_map = self._build_global_offsets(
            self._scenario_full_state_def
        )

        N_new = sum(int(t[1]) for t in self._scenario_full_state_def)
        xg_new = np.zeros((N_new,), dtype=float)

        # -------------------------
        # 3) Project old global -> new global where keys overlap (safe sizes)
        # -------------------------
        if old_def is not None and old_map is not None and old_xg is not None:
            for key_old, (os, oe) in old_map.items():
                if key_old not in self._scenario_index_map:
                    continue
                ns, ne = self._scenario_index_map[key_old]
                kcopy = min(int(oe - os), int(ne - ns))
                if kcopy <= 0:
                    continue
                xg_new[ns : ns + kcopy] = old_xg[os : os + kcopy]

        # -------------------------
        # 4) Optionally fill NEW keys (not present in old_map) from first model appearance
        # -------------------------
        if refill_from_models:
            old_keys = set(old_map.keys()) if old_map is not None else set()

            for model in self.mission_sequence.models:
                # Only models with a "state" attribute can seed values
                if not hasattr(model, "state"):
                    continue

                # StateArray.state is iterable; other models might have .state too (rare)
                try:
                    m_map = _model_key_map(model.state)
                except Exception:
                    continue

                for k_new, (ns, ne) in self._scenario_index_map.items():
                    if k_new in old_keys:
                        continue
                    if k_new not in m_map:
                        continue

                    *_, awf = m_map[k_new]
                    v = _awf_template_to_1d(awf)
                    xg_new[ns:ne] = v[: (ne - ns)]
                    old_keys.add(k_new)  # first model wins

        # Commit global numeric IC
        self._global_state_value = xg_new

        # -------------------------
        # 5) Push global values back into each StateArray model (values-only)
        #    NOTE: ImpulsiveBurn etc. are handled elsewhere (DV reinit pass).
        # -------------------------
        if update_models:
            for i, model in enumerate(self.mission_sequence.models):

                # Only re-seed StateArray segments here
                if not isinstance(model, StateArray):
                    continue

                # Build rebuilt component list with updated awf values
                occ_counter = {}
                rebuilt = []

                for name, size, est_type, dyn_type, body, awf in model.state:
                    base = (
                        name,
                        int(size),
                        getattr(body, "spice_id", None),
                        getattr(getattr(awf, "frame", None), "name", None),
                    )
                    occ = occ_counter.get(base, 0)
                    occ_counter[base] = occ + 1
                    key = (*base, int(occ))

                    if key in self._scenario_index_map:
                        gs, ge = self._scenario_index_map[key]
                        vals = np.asarray(
                            self._global_state_value[gs:ge], dtype=float
                        ).reshape(-1)

                        new_awf = ArrayWFrame(
                            ArrayWUnits(vals, awf.quantity.units),
                            awf.frame,
                        )
                        rebuilt.append((name, size, est_type, dyn_type, body, new_awf))
                    else:
                        rebuilt.append((name, size, est_type, dyn_type, body, awf))

                new_sd = StateDefinition.from_components(rebuilt)
                self.mission_sequence.models[i] = StateArray(
                    epoch=model.epoch,
                    origin=model.origin,
                    state=new_sd,
                )

                if update_states_n:
                    try:
                        self.mission_sequence.states_n[i] = sum(
                            int(t[1]) for t in rebuilt
                        )
                    except Exception:
                        pass

        # ---------------------------------------------------------
        # 6) Reinitialize ImpulsiveBurn models from global DV blocks
        # ---------------------------------------------------------
        for i, model in enumerate(self.mission_sequence.models):

            # Only care about ImpulsiveBurn segments
            if model.__class__.__name__ != "ImpulsiveBurn":
                continue

            # Convention: DV parameters must exist in global state as
            # "dv_man", "dv_man1", "dv_man2" etc.
            # We match by occurrence index in mission sequence.

            # Build matching DV key candidates
            dv_keys = [
                key
                for key in self._scenario_index_map.keys()
                if key[0].startswith("dv_man")
            ]

            if not dv_keys:
                continue

            # Choose DV key based on order of burn occurrence
            # (burn index among ImpulsiveBurn segments)
            burn_count = (
                sum(
                    1
                    for j in range(i + 1)
                    if self.mission_sequence.types[j] == "Impulsive Burn"
                )
                - 1
            )

            if burn_count >= len(dv_keys):
                continue

            dv_key = sorted(dv_keys)[burn_count]

            gs, ge = self._scenario_index_map[dv_key]
            dv_vals = np.asarray(self._global_state_value[gs:ge], dtype=float).reshape(
                3,
            )

            # Units & frame taken from existing burn model
            base_frame = model.base_frame
            units = model.delta_v.units

            dv_awunits = ArrayWUnits(dv_vals, units)
            dv_awf = ArrayWFrame(dv_awunits, base_frame)
            new_burn = ImpulsiveBurn(delta_v=dv_awf)

            self.mission_sequence.models[i] = new_burn

    def _deviation_history_to_global_dx(
        self,
        deviation_history: list,
        *,
        as_column: bool = True,
        strict: bool = True,
    ) -> np.ndarray:
        """
        Convert a deviation history list to a global state deviation vector.

        This method maps per-segment (Leg/Finite Burn) deviation vectors from a
        deviation_history list to their corresponding positions in the global scenario
        state vector.

        Parameters
        ----------
        deviation_history : list
            A list of deviation vectors, one per active segment (Leg or Finite Burn).
            Each entry should be array-like and will be converted to a 1D numpy array.
            Length must equal the number of active segments in the mission sequence.

        as_column : bool, optional
            If True (default), return the global deviation vector as a column vector
            with shape (N_global, 1). If False, return as a 1D array with shape (N_global,).

        strict : bool, optional
            If True (default), enforce strict size matching. Raises ValueError if any
            deviation entry does not match its expected local dimension or is too short.
            If False, truncates or pads vectors as needed without raising errors.

        Returns
        -------
        np.ndarray
            Global state deviation vector of shape (N_global, 1) if as_column=True,
            or (N_global,) if as_column=False, where N_global is the total dimension
            of the global scenario state.

        Raises
        ------
        ValueError
            If length of deviation_history does not match the number of active segments.
            If strict=True and any deviation entry has mismatched dimensions.

        KeyError
            If a leg component key is not found in the scenario-global index map,
            indicating the scenario union state_def is incomplete.

        Notes
        -----
        - Only "Leg" and "Finite Burn" segments are considered active.
        - The scenario index map is built on-demand if not already present.
        - Leg-to-global mappings are cached in _leg_to_global_maps for reuse.
        """
        # Active segments (only these must be covered by deviation_history)
        active_idx = [
            i
            for i, t in enumerate(self.mission_sequence.types)
            if t in ("Leg", "Finite Burn")
        ]
        if len(deviation_history) != len(active_idx):
            raise ValueError(
                "deviation_history must have one entry per 'Leg'/'Finite Burn' segment.\n"
                f"Expected {len(active_idx)} entries, got {len(deviation_history)}."
            )

        # Ensure global schema/map exist
        if not hasattr(self, "_scenario_index_map") or self._scenario_index_map is None:
            self._scenario_full_state_def = self._build_scenario_state_def()
            self._scenario_index_map = self._build_global_offsets(
                self._scenario_full_state_def
            )

        N_global = (
            int(self._global_state_value.size)
            if hasattr(self, "_global_state_value")
            and self._global_state_value is not None
            else sum(int(t[1]) for t in self._scenario_full_state_def)
        )

        dxg = np.zeros((N_global,), dtype=float)
        # First-write-wins: shared state components (e.g. pos/vel present in multiple legs)
        # must only receive the correction from the FIRST leg that claims them.
        # Later legs' per-leg corrections for shared slots are already encoded (via the STM)
        # in the first leg's correction, so overwriting would double-apply the STM mapping.
        written_mask = np.zeros(N_global, dtype=bool)

        # Build/update leg->global mapping on the fly (does not require propagate())
        hist_cursor = 0
        for seq_i in active_idx:
            seg_name = self.mission_sequence.names[seq_i]
            model = self.mission_sequence.models[seq_i]
            leg_state = model.state  # local definition list of tuples

            # build local offsets and blocks to global
            leg_offsets = self._build_global_offsets(leg_state)

            blocks = []
            local_dim = 0
            for leg_key, (ls, le) in leg_offsets.items():
                local_dim = max(local_dim, int(le))
                if leg_key not in self._scenario_index_map:
                    raise KeyError(
                        f"Leg component key {leg_key} not found in scenario-global map. "
                        "Your scenario union state_def is missing something present in this leg."
                    )
                gs, ge = self._scenario_index_map[leg_key]
                blocks.append((int(ls), int(le), int(gs), int(ge)))

            # store mapping for later use (optional)
            self._leg_to_global_maps[seg_name] = blocks

            # local deviation vector
            v = np.asarray(deviation_history[hist_cursor], dtype=float)
            v = v.reshape(-1)  # 1D
            if strict and v.size != local_dim:
                raise ValueError(
                    f"Deviation entry {hist_cursor} (segment '{seg_name}') has size {v.size}, "
                    f"but local_dim inferred from model.state is {local_dim}."
                )

            # place into global (first-write-wins for shared slots)
            for ls, le, gs, ge in blocks:
                n = min(le - ls, ge - gs)
                if n <= 0:
                    continue

                # if v is shorter and strict=False, truncate safely
                if v.size < le:
                    if strict:
                        raise ValueError(
                            f"Deviation entry {hist_cursor} (segment '{seg_name}') shorter than required. "
                            f"Need at least {le} elems, got {v.size}."
                        )
                    n = max(0, v.size - ls)
                    if n <= 0:
                        continue

                # Skip slots already claimed by an earlier leg.
                # Shared components (pos/vel) appear in multiple legs: their back-mapped
                # correction is already consistent in the first leg's entry; applying the
                # later leg's value (which is forward-propagated via the STM) would
                # over-correct by a factor of Phi, causing divergence.
                if np.any(written_mask[gs : gs + n]):
                    continue

                dxg[gs : gs + n] = v[ls : ls + n]
                written_mask[gs : gs + n] = True

            hist_cursor += 1

        if as_column:
            return dxg.reshape(-1, 1)
        return dxg

    def _reinitialize_from_deviation_global(
        self,
        deviation_global: list,
        *,
        commit: bool = True,  # <- accumulate when True
        relative_to_ref: bool = True,  # True: dx applied to _x_ref0, False: dx applied to _x_ic
    ) -> None:

        dx = np.asarray(deviation_global, dtype=float).reshape(-1)

        # lazily initialize if needed (backward compatible)
        if not hasattr(self, "_x_ref0") or self._x_ref0 is None:
            if (
                not hasattr(self, "_global_state_value")
                or self._global_state_value is None
            ):
                self._global_state_value = self._init_global_state_value()
            self._x_ref0 = (
                np.asarray(self._global_state_value, dtype=float).reshape(-1).copy()
            )
            self._x_ic = self._x_ref0.copy()

        if dx.size != self._x_ref0.size:
            raise ValueError(f"dx has size {dx.size}, expected {self._x_ref0.size}.")

        base = self._x_ref0 if relative_to_ref else self._x_ic
        new_ic = base + dx

        # set current IC + global state
        self._x_ic = new_ic.copy()
        self._global_state_value = self._x_ic.copy()

        # IMPORTANT: if you want accumulation across calls, update the reference
        if commit:
            self._x_ref0 = self._x_ic.copy()

        # push into models
        self._reinitialize_scenario_state_def(
            new_state_def=self._scenario_full_state_def,
            refill_from_models=False,
            update_models=True,
            update_states_n=True,
        )

        # clear caches (same as your current code)
        self._segment_positions = {s: None for s in self.mission_sequence.names}
        self._segment_velocities = {s: None for s in self.mission_sequence.names}
        self._segment_parameters = {s: None for s in self.mission_sequence.names}
        self._segment_STMs = {s: None for s in self.mission_sequence.names}
        self._segment_epochsTDB = {s: None for s in self.mission_sequence.names}
        self._total_components = {}
        self._total_parameters = []
        self._STM = []
        self._delta_v = None
        self._full_propagated_state_array = None
        if hasattr(self, "_pending_dv_vals"):
            self._pending_dv_vals = None
            self._pending_dv_units = None
            self._pending_dv_frame_name = None
