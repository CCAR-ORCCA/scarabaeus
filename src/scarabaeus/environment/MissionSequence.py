# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    ArrayWUnits,
    EpochArray,
    ImpulsiveBurn,
    StateArray,
    StateDefinition,
    Propagator,
    Body,
    Units,
)

import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Utils import Utils

# generate units
sec = Units.get_units("sec")


# ----------------------------------#
#  Sentinel for node segment slots  #
# ----------------------------------#
class _NodeSentinel:
    """ Placeholder stored in node slots of ``periods``, ``models``, ``states_n``,
    ``dts``, and ``num_steps``.  Replaces the raw ``None`` that used to sit
    there so downstream code can tell a node apart from a genuinely missing
    value::

        isinstance(ms.models[i], _NodeSentinel)  # True  → node boundary
        if ms.models[i]:                          # False → node (sentinel is falsy)

    Attributes
    ----------
    name : str
        Human-readable label of the node.
    epoch : float
        TDB epoch (seconds past J2000) at which the node sits.
    """

    __slots__ = ("name", "epoch")

    def __init__(self, name: str, epoch: float):
        self.name = name
        self.epoch = epoch

    def __repr__(self):
        """Return a concise string showing the node label and epoch."""
        return f"<Node '{self.name}' @ {self.epoch:.3f} TDB>"

    def __bool__(self):
        """Return ``False`` so existing ``if model:`` guards treat nodes as falsy."""
        # Falsy like None was, so existing ``if model:`` guards keep working.
        return False


# --------------------#
#  Class Definition  #
# --------------------#
class MissionSequence:
    """ Ordered container of consecutive trajectory legs joined at shared nodes.

    A :class:`~scarabaeus.MissionSequence` partitions a trajectory into *legs* (numerical
    propagation arcs) separated by *nodes* (instantaneous boundary points).
    Legs are continuous propagation intervals; each leg can use an independent
    :class:`~scarabaeus.Propagator`, :class:`~scarabaeus.ForceModel`, reference frame, origin, and
    :class:`~scarabaeus.StateArray` definition — enabling changes of dynamical model,
    reference body, or estimated parameter set between segments.

    Each leg may model **finite burns** (continuous thrust carried by a
    :class:`~scarabaeus.FiniteBurn` force model inside the leg's propagator).  At a node
    between two legs an **impulsive burn** can be applied:

    .. math::

        \\mathbf{x}^+_k = \\mathbf{x}^-_k +
        \\begin{bmatrix} \\mathbf{0} \\\\ \\Delta\\mathbf{v}_k \\\\ \\mathbf{0} \\end{bmatrix}

    where :math:`\\Delta\\mathbf{v}_k` is the delta-v vector of node :math:`k`.
    Nodes with no burn are pure epoch-boundary markers that enforce continuity
    and allow the propagator or state definition to change.

    Parameters
    ----------
    name : str
        Human-readable identifier for the mission sequence.
    grid_bound_tol : float, optional
        Tolerance, in TDB seconds, on the bounds of a grid chosen by the
        integrator (see :attr:`grid_bound_tol`). Defaults to ``1e-6``.

    Attributes
    ----------
    grid_bound_tol : float
        How far, in TDB seconds, an integrator-chosen grid may sit from the
        bounds its segment declared. Settable at any time.
    deferred_grids : list of bool
        Which segments take their grid from the integrator.

    See Also
    --------
    scarabaeus.Propagator : Numerical propagator used by each leg.
    scarabaeus.ScenarioSetup : Executes a MissionSequence leg by leg.
    scarabaeus.StateArray : State vector definition for each leg.

    Notes
    -----
    * The first leg must define the complete initial state.
    * Each leg can carry a different :class:`Propagator` (and thus a different
      :class:`ForceModel`, frame, or origin).
    * Estimated impulsive burns must be followed by a ``"dv_man"`` parameter
      in the next leg's :class:`StateArray`.
    * Epoch continuity across nodes is enforced automatically.

    Examples
    --------
    .. code-block:: python

        import scarabaeus as scb

        mission = scb.MissionSequence(name="EMA123")

        # Minimal — everything derived from the propagator:
        mission.addLeg("Outbound Leg", prop=prop1)
        mission.addBurn("Midcourse Correction", dV=dV, prop=prop1)
        mission.addLeg("Final Approach", prop=prop2)

        # Full explicit — backwards-compatible with old call style:
        mission.addLeg(
            name="Outbound Leg",
            period=ArrayWUnits(86400, "s"),
            state_vector=state_vec,
            prop=prop1,
            dt=ArrayWUnits(60, "s"),
        )
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class TimeSpanError(ValueError):
        """Raised when a segment's time span does not define an output grid."""

    class GridToleranceError(ValueError):
        """Raised when the tolerance on an integrator-chosen grid is not a
        non-negative number of seconds."""

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(self, name: str, grid_bound_tol: float = 1e-6):
        self._name = name
        self._primary_body = None
        self.grid_bound_tol = grid_bound_tol

        self._types = []
        self._names = []
        self._periods = []
        self._models = []
        self._props = []
        self._states_n = []
        self._origins = []
        self._dts = []
        self._num_steps = []
        self._last_idx = []
        self._epochs_vec = []
        self._deferred_grids = []
        # (carrying model, declared model, carried keys) per segment that
        # carries static parameters; see carry_static_parameters().
        self._carry_records = []

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def name(self) -> str:
        """The name of the mission sequence."""
        return self._name

    @property
    def primary_body(self) -> Body | None:
        """Primary propagated object for the whole sequence (must be consistent)."""
        return self._primary_body

    @property
    def types(self) -> list[str]:
        """List of segment types, such as ``"Leg"`` or ``"Impulsive Burn"``."""
        return self._types

    @property
    def names(self) -> list[str]:
        """
        List of names corresponding to each sequence segment.
        """
        return self._names

    @property
    def periods(self) -> list[ArrayWUnits] | list[None]:
        """
        Duration of each segment, in seconds, for legs and finite burns;
        ``None`` for impulsive burns and a :class:`_NodeSentinel` for the
        boundary nodes inserted between consecutive propagated segments.
        """
        return self._periods

    @property
    def models(self) -> list[StateArray] | list[ImpulsiveBurn]:
        """
        List of state models, representing either propagation states or impulsive maneuvers.
        """
        return self._models

    @property
    def props(self) -> list[Propagator] | None:
        """
        List of propagator models for each leg in the sequence.
        """
        return self._props

    @property
    def states_n(self) -> list[int] | None:
        """
        List of the number of states in each leg, or None for impulsive burns.
        """
        return self._states_n

    @property
    def origins(self) -> list[int] | None:
        """
        List of origins for each leg in the sequence.
        """
        return self._origins

    @property
    def dts(self) -> list[ArrayWUnits] | None:
        """
        List of time step lengths used for integration in each segment.
        """
        return self._dts

    @property
    def num_steps(self) -> list[int] | None:
        """
        Number of integration steps per leg, or None for impulsive burns.
        """
        return self._num_steps

    @property
    def last_idx(self) -> list[int]:
        """
        Indices marking the last step of each segment within the overall timeline.
        """
        return self._last_idx

    @property
    def epochs_vec(self) -> list[EpochArray]:
        """
        List of `EpochArray` objects defining the timestamps associated with each mission phase.
        """
        return self._epochs_vec

    @epochs_vec.setter
    def epochs_vec(self, input_val):
        self._epochs_vec = input_val

    @property
    def deferred_grids(self) -> list[bool]:
        """
        Per-segment flag marking the legs whose output grid is chosen by the
        integrator rather than declared up front (a ``(t0, tf)`` tuple
        ``tspan``).  Until :meth:`propagate` runs, those segments hold only
        their two bounds in :attr:`epochs_vec`; the propagated grid replaces
        them through :meth:`adopt_propagated_grid`.
        """
        return self._deferred_grids

    @property
    def grid_bound_tol(self) -> float:
        """
        Tolerance, in TDB seconds, on the bounds of an integrator-chosen grid.

        :meth:`adopt_propagated_grid` rejects a propagated grid whose first or
        last epoch sits further than this from the bounds the segment declared,
        because that gap or overlap would break epoch continuity with the
        neighbouring segments.  The default of ``1e-6`` is about a decimal
        digit above the float64 resolution of an ephemeris epoch, so it
        absorbs rounding without hiding a grid that genuinely stops short.

        Raises
        ------
        GridToleranceError
            If set to something other than a non-negative real number.
        """
        return self._grid_bound_tol

    @grid_bound_tol.setter
    def grid_bound_tol(self, value: float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.GridToleranceError(
                f"grid_bound_tol must be a number of TDB seconds, got "
                f"{type(value).__name__}."
            )
        if not np.isfinite(value) or value < 0.0:
            raise self.GridToleranceError(
                f"grid_bound_tol must be finite and non-negative, got {value}."
            )
        self._grid_bound_tol = float(value)

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def _enforce_primary_body(self, prop: Propagator) -> None:
        if prop is None:
            return

        pb = getattr(prop, "primary_body", None)
        if pb is None:
            raise ValueError(
                "Propagator has no primary_body; cannot enforce sequence consistency."
            )

        if self._primary_body is None:
            self._primary_body = pb
            return

        sid_seq = getattr(self._primary_body, "spice_id", None)
        sid_new = getattr(pb, "spice_id", None)

        if sid_seq is None or sid_new is None:
            # fallback stricter object identity if spice_id missing
            if pb is not self._primary_body:
                raise ValueError(
                    "MissionSequence primary_body mismatch (missing spice_id on one of the bodies)."
                )
        else:
            if sid_new != sid_seq:
                raise ValueError(
                    f"MissionSequence primary_body mismatch: "
                    f"sequence spice_id={sid_seq}, new segment spice_id={sid_new}."
                )

    def _resolve_state_vector(
        self,
        state_vector: StateArray | None,
        prop: Propagator | None,
        seg_name: str,
    ) -> StateArray:
        """
        Return a StateArray for this segment.

        Priority:
        1. Explicit ``state_vector`` argument.
        2. ``prop.state_vector`` (the propagator's own state definition).

        Raises if neither is available.
        """
        if state_vector is not None:
            return state_vector

        sv = getattr(prop, "state_vector", None)
        if sv is None:
            sv = getattr(prop, "full_state_vector", None)
        if sv is not None:
            return sv

        raise ValueError(
            f"Segment '{seg_name}': no `state_vector` provided and the propagator "
            "has no `state_vector` / `full_state_vector` attribute. "
            "Supply `state_vector` explicitly."
        )

    def _resolve_epochs(
        self,
        prop: Propagator | None,
        period: ArrayWUnits | None,
        dt: ArrayWUnits | None,
        epochs: EpochArray | None,
        state_vector: StateArray,
        seg_name: str,
    ) -> tuple:
        """
        Resolve the epoch array, period, and num_steps for a segment.

        Priority:
        1. Explicit ``epochs`` — use as-is, derive period from span.
        2. ``period`` + ``dt`` — build a uniform grid.
        3. ``prop.tspan`` — use the propagator's own time span.

        A ``(t0, tf)`` tuple ``tspan`` declares no grid: the segment holds its
        two bounds and is flagged as deferred, so that :class:`ScenarioSetup`
        integrates between the bounds and adopts the grid the integrator picks
        (see :meth:`adopt_propagated_grid`).  Both DOP853 and IAS15 place their
        own output times, so either can carry such a segment.

        Returns
        -------
        epoch_vec : EpochArray
            Epochs of the segment.
        period : ArrayWUnits
            Time span of the segment, in seconds.
        num_steps : int
            Number of epochs in ``epoch_vec``.
        deferred : bool
            ``True`` when ``epoch_vec`` holds bounds awaiting the integrator's
            own grid.
        """
        if epochs is not None:
            if dt is not None:
                raise ValueError(
                    f"Segment '{seg_name}': provide either `epochs` or `dt`, not both."
                )
            t_vals = epochs.et  # raw ET seconds, rep-independent
            if period is None:
                period = ArrayWUnits(float(t_vals[-1]) - float(t_vals[0]), sec)
            else:
                period = self._to_seconds(period, "period", seg_name)
            return epochs, period, len(t_vals), False

        if period is not None and dt is not None:
            # The grid is built on TDB seconds, so a period or step given in
            # another time unit (days, hours) is converted first.
            period = self._to_seconds(period, "period", seg_name)
            dt = self._to_seconds(dt, "dt", seg_name)
            if not self.epochs_vec:
                start_epoch = float(state_vector.epoch.et)
            else:
                start_epoch = float(self.epochs_vec[-1][-1].et)
            t_arr = np.append(
                np.arange(start_epoch, start_epoch + period.values, dt.values),
                start_epoch + period.values,
            )
            epoch_vec = EpochArray(t_arr, sys = "TDB", rep = 'AWU')
            num_steps = np.arange(0, period.values, dt.values).size
            return epoch_vec, period, num_steps, False

        # fall back to prop.tspan
        tspan = getattr(prop, "tspan", None) if prop is not None else None
        if tspan is None:
            raise ValueError(
                f"Segment '{seg_name}': no `epochs`, no `period`+`dt` provided, "
                "and the propagator has no `tspan`. Supply at least one of these."
            )
        # A (t0, tf) tuple carries no inner grid: keep the bounds and let the
        # integrator fill in between them at propagation time.
        if isinstance(tspan, tuple):
            bounds = EpochArray(
                np.array([float(tspan[0].et), float(tspan[1].et)]),
                sys="TDB",
                rep="AWU",
            )
            derived_period = ArrayWUnits(
                float(bounds.et[-1]) - float(bounds.et[0]), sec
            )
            return bounds, derived_period, len(bounds), True

        t_vals = tspan.et  # raw ET seconds, rep-independent
        derived_period = ArrayWUnits(float(t_vals[-1]) - float(t_vals[0]), sec)
        return tspan, derived_period, len(t_vals), False

    def _to_seconds(
        self, duration: ArrayWUnits, label: str, seg_name: str
    ) -> ArrayWUnits:
        """
        Return a segment's ``period`` or ``dt`` converted to seconds.

        Raises
        ------
        TimeSpanError
            If ``duration`` is not a time.
        """
        if duration.units.dimensions != sec.dimensions:
            raise self.TimeSpanError(
                f"Segment '{seg_name}': `{label}` must be a time, got {duration} "
                f"with dimensions {duration.units.dimensions}."
            )
        return duration.convert_to(sec)

    def adopt_propagated_grid(self, index: int, epochs: EpochArray) -> None:
        """
        Replace a deferred segment's bounds with the grid the integrator used.

        Called by :class:`~scarabaeus.ScenarioSetup` right after a segment whose
        ``tspan`` was a ``(t0, tf)`` tuple has been propagated, so that the rest
        of the framework — SPK writing, measurement mapping, process noise,
        filter leg boundaries — sees the epochs the states actually sit on.
        ``num_steps`` and ``last_idx`` are refreshed to match.

        Parameters
        ----------
        index : int
            Index of the segment in the sequence.
        epochs : EpochArray
            Epochs of the propagated states, in ascending order.

        Raises
        ------
        TimeSpanError
            If the propagated grid does not start and end at the segment's
            declared bounds, to within :attr:`grid_bound_tol`, which would
            break epoch continuity with the neighbouring segments.
        """
        bounds = self._epochs_vec[index]
        for label, declared, obtained in (
            ("start", float(bounds[0].et), float(epochs[0].et)),
            ("end", float(bounds[-1].et), float(epochs[-1].et)),
        ):
            if abs(declared - obtained) > self._grid_bound_tol:
                raise self.TimeSpanError(
                    f"Segment '{self._names[index]}': the propagated grid does not "
                    f"reach its declared {label} epoch ({obtained} vs {declared} "
                    f"TDB, off by {abs(declared - obtained):.3e} s against a "
                    f"grid_bound_tol of {self._grid_bound_tol:.3e} s), which would "
                    "break continuity with the adjacent segments."
                )

        self._epochs_vec[index] = epochs
        self._num_steps[index] = len(epochs)
        self._periods[index] = ArrayWUnits(
            float(epochs[-1].et) - float(epochs[0].et), sec
        )
        self._rebuild_last_idx()

    def _rebuild_last_idx(self) -> None:
        """
        Recompute the cumulative ``last_idx`` chain from the current
        ``num_steps``, replaying the rule the ``add*`` methods apply: each
        propagated segment advances the count by its number of steps, and each
        node or impulsive burn overlaps the previous segment's endpoint.
        """
        last_idx = []
        for i, seg_type in enumerate(self._types):
            if seg_type in ("Leg", "Finite Burn"):
                steps = self._num_steps[i]
                last_idx.append(last_idx[-1] + steps if last_idx else steps)
            else:
                if last_idx:
                    last_idx[-1] -= 1
                last_idx.append(last_idx[-1] if last_idx else 0)
        self._last_idx = last_idx

    def addLeg(
        self,
        name,
        period: ArrayWUnits = None,
        state_vector: StateArray = None,
        prop: Propagator = None,
        dt: ArrayWUnits = None,
        epochs: EpochArray = None,
    ) -> None:
        """
        Adds a propagation leg to the mission sequence.

        All of ``period``, ``state_vector``, ``prop``, ``dt``, and ``epochs``
        are optional.  The minimum required call is just a ``name`` — as long as
        the propagator carries a ``state_vector`` and a ``tspan``:

        .. code-block:: python

            MS.addLeg("coast", prop=prop_leg)

        Resolution order:

        * ``state_vector``: explicit arg, then ``prop.state_vector``
        * epoch grid: explicit ``epochs``, then a uniform ``period`` + ``dt``
          grid, then ``prop.tspan``
        * ``period``: derived from whichever epoch source is used.

        When ``prop.tspan`` is a ``(t0, tf)`` tuple the leg declares no grid of
        its own: it holds the two bounds, and :meth:`propagate` adopts the
        epochs the integrator's own steps landed on (see
        :meth:`adopt_propagated_grid`).  Both integrators do this — DOP853
        through ``t_eval=None``, IAS15 through its adaptive steps.  How fine
        the resulting grid is, is the error control's call — on slow dynamics
        it can be the two bounds alone.  A default degree-3 type-13 SPK segment
        still writes from two nodes, but one cubic then spans the whole leg;
        give the leg ``epochs`` or ``period`` + ``dt`` when the output density
        matters.

        If the immediately preceding segment is a ``"Finite Burn"``, a boundary
        node is inserted automatically before this leg.

        Parameters
        ----------
        name : str
            The name of the leg.
        period : ArrayWUnits, optional
            Duration of the leg, in any time unit (stored in seconds).
            Derived from ``epochs`` or ``prop.tspan`` when not provided.
        state_vector : StateArray, optional
            State schema for this leg.  Derived from ``prop.state_vector``
            when not provided.
        prop : Propagator, optional
            Propagator for this leg.  Its ``tspan`` and ``state_vector`` are
            used as fallbacks when the explicit arguments are absent.
        dt : ArrayWUnits, optional
            Output sampling interval, in any time unit (used with ``period``
            for a uniform grid).
        epochs : EpochArray, optional
            Explicit epoch grid.  Takes priority over all other epoch sources.

        Returns
        -------
        None
        """
        # resolve state_vector first (needed by _validate_leg and epoch resolution)
        state_vector = self._resolve_state_vector(state_vector, prop, name)

        self._validate_leg(name, period, dt, state_vector, prop, epochs)
        self._enforce_primary_body(prop)

        epoch_vec, period, num_steps, deferred = self._resolve_epochs(
            prop, period, dt, epochs, state_vector, name
        )

        # Auto-insert a boundary node when a leg follows a finite burn or another leg
        if self.types and self.types[-1] in ("Finite Burn", "Leg"):
            self.addNode(f"{name} Start Node", prop)

        # ---------------------------------#
        # add leg to the mission sequence #
        # ---------------------------------#
        # append to the lists

        self.types.append("Leg")
        self.names.append(name)
        self.periods.append(period)
        self.models.append(state_vector)
        self.states_n.append(sum(t[1] for t in state_vector.state))
        self.props.append(prop)
        self.origins.append(prop.prop_origin if prop is not None else None)
        self.dts.append(dt)
        self.last_idx.append(
            self.last_idx[-1] + num_steps if self.last_idx else num_steps
        )
        self.num_steps.append(num_steps)
        self.epochs_vec.append(epoch_vec)
        self.deferred_grids.append(deferred)

        # Check consistency of the sequence
        if len(self.types) > 1:
            self._check_sequence_consistency()

    def addFiniteBurn(
        self,
        name,
        period: ArrayWUnits = None,
        state_vector: StateArray = None,
        prop: Propagator = None,
        dt: ArrayWUnits = None,
        epochs: EpochArray = None,
        initial_node_name: str | None = None,
        add_terminal_node: bool = False,
        terminal_node_name: str | None = None,
    ) -> None:
        """
        Add a finite burn maneuver to the mission sequence.

        All of ``period``, ``state_vector``, ``prop``, ``dt``, and ``epochs``
        are optional.  The minimum required call is just a ``name`` — as long as
        the propagator carries a ``state_vector`` and a ``tspan``:

        .. code-block:: python

            MS.addFiniteBurn("burn 1", prop=prop_fb)

        A ``(t0, tf)`` tuple ``tspan`` defers the grid to the integrator, as in
        :meth:`addLeg`.

        A boundary node is automatically inserted before the burn unless this
        is the very first segment in the sequence.

        Parameters
        ----------
        name : str
            Name of this finite burn segment.
        period : ArrayWUnits, optional
            Duration of the burn arc, in any time unit (stored in seconds).
            Derived automatically when not given.
        state_vector : StateArray, optional
            State schema (position/velocity + burn parameters).  Derived from
            ``prop.state_vector`` when not provided.
        prop : Propagator, optional
            Propagator for this burn arc.
        dt : ArrayWUnits, optional
            Output sampling interval, in any time unit (uniform grid only).
        epochs : EpochArray, optional
            Explicit epoch grid.  Takes priority over everything else.
        initial_node_name : str, optional
            Name for the auto-inserted start node.  Auto-generated if None.
        add_terminal_node : bool, optional
            Insert a node after the burn arc.  Default False.
        terminal_node_name : str, optional
            Name for the terminal node.  Auto-generated if None.

        Returns
        -------
        None
        """
        # resolve state_vector first
        state_vector = self._resolve_state_vector(state_vector, prop, name)

        self._enforce_primary_body(prop)

        epoch_vec, period, num_steps, deferred = self._resolve_epochs(
            prop, period, dt, epochs, state_vector, name
        )

        # Insert a boundary node unless this is the very first segment
        if self.types and self.types[-1] != "Node":
            node_name = initial_node_name or f"{name} Start Node"
            self.addNode(node_name, prop)

        # Append to the lists
        self.types.append("Finite Burn")
        self.names.append(name)
        self.periods.append(period)
        self.models.append(state_vector)
        self.states_n.append(sum(t[1] for t in state_vector.state))
        self.props.append(prop)
        self.origins.append(prop.prop_origin if prop is not None else None)
        self.dts.append(dt)
        self.last_idx.append(
            self.last_idx[-1] + num_steps if self.last_idx else num_steps
        )
        self.num_steps.append(num_steps)
        self.epochs_vec.append(epoch_vec)
        self.deferred_grids.append(deferred)

        # Check consistency of the sequence
        if len(self.types) > 1:
            self._check_sequence_consistency()

        # Optionally add a node at the end of the finite burn segment to mark
        # the Finite Burn -> next segment boundary explicitly.
        if add_terminal_node:
            if self.types[-1] != "Node":
                node_name = terminal_node_name or f"{name} End Node"
                self.addNode(node_name, prop)

    def addBurn(
        self,
        name: str,
        dV: ImpulsiveBurn,
        prop: Propagator,
    ) -> None:
        """
        Adds an impulsive burn to the mission sequence.

        Inserts an impulsive burn event, modifying the spacecraft's
        velocity instantaneously. The burn is added between two propagation legs,
        and a corresponding velocity update (``dv_man``) is introduced in the next
        leg.

        Parameters
        ----------
        name : str
            The name of the impulsive burn.

        dV : ImpulsiveBurn
            The impulsive burn model defining the velocity change.

        prop : Propagator
            The propagator responsible for simulating the spacecraft dynamics.

        Returns
        -------
        None
        """
        # Checks
        self._validate_impulsive_burn(name, dV, prop)
        self._enforce_primary_body(prop)

        # Decrease the size of the last leg (node is overlapped)
        self.last_idx[-1] -= 1

        # Append to the lists
        self.types.append("Impulsive Burn")
        self.names.append(name)
        self.periods.append(None)
        self.models.append(dV)
        self.states_n.append(None)
        self.props.append(prop)
        self.origins.append(prop.prop_origin if prop is not None else None)
        self.dts.append(None)
        self.num_steps.append(None)
        self.last_idx.append(self.last_idx[-1])

        # Create epochs vector
        epoch_vec = EpochArray(
            np.array([float(self.epochs_vec[-1][-1].et)]),
            sys="TDB",
        )
        self.epochs_vec.append(epoch_vec)
        self.deferred_grids.append(False)

    def addNode(
        self,
        name: str,
        prop: Propagator,
    ) -> None:
        """
        Adds a zero-duration node to the mission sequence.

        Inserts an event marker at the current sequence boundary without applying
        any state discontinuity. This is useful for identifying mission events
        such as the end of a finite-burn arc, the beginning of an observation arc,
        or any other boundary that later processing should recognize as a node.

        Unlike :meth:`addBurn`, this method does not modify the spacecraft state
        and does not introduce an impulsive maneuver parameter in the subsequent
        leg. The node simply records the boundary epoch as an explicit sequence
        element.

        A :class:`_NodeSentinel` is stored in the slots that have no meaningful
        value for a node (``periods``, ``models``, ``states_n``, ``dts``,
        ``num_steps``).  It is falsy like ``None`` was, but carries the node
        ``name`` and ``epoch`` so downstream code can inspect it instead of
        getting an opaque ``None``::

            isinstance(ms.models[i], _NodeSentinel)  # True  → node
            if ms.models[i]:                          # False → node

        Parameters
        ----------
        name : str
            The name of the node.
        prop : Propagator
            The propagator associated with this boundary.

        Returns
        -------
        None
        """
        # Checks
        self._enforce_primary_body(prop)

        if not self.names:
            raise ValueError("A node cannot be added before the first leg is defined.")

        if name in self.names:
            raise ValueError(f"A sequence element named '{name}' already exists.")

        # Decrease the size of the last leg (node is overlapped with leg endpoint)
        self.last_idx[-1] -= 1

        boundary_epoch = float(self.epochs_vec[-1][-1].et)
        sentinel = _NodeSentinel(name, boundary_epoch)

        # Append to the lists — sentinel in slots where None used to live
        self.types.append("Node")
        self.names.append(name)
        self.periods.append(sentinel)  # was None
        self.models.append(sentinel)  # was None
        self.states_n.append(sentinel)  # was None
        self.props.append(prop)
        self.origins.append(prop.prop_origin if prop is not None else None)
        self.dts.append(sentinel)  # was None
        self.num_steps.append(sentinel)  # was None
        self.last_idx.append(self.last_idx[-1])

        # Create epochs vector
        epoch_vec = EpochArray(np.array([boundary_epoch]), sys = 'TDB')
        self.epochs_vec.append(epoch_vec)
        self.deferred_grids.append(False)

    def _validate_leg(
        self,
        name: str,
        period: ArrayWUnits | None,
        dt: ArrayWUnits | None,
        state_vector: StateArray,
        prop: Propagator | None,
        epochs_vec: EpochArray = None,
    ) -> None:
        """
        Validate a propagation leg before adding it to the mission sequence.

        NaN-free semantics:

        - For the first leg: epoch + full initial state must be defined.
        - For subsequent legs: epoch/pos/vel values are allowed but will be
          ignored by ScenarioSetup, which inherits values structurally from
          the previous leg.
        """
        if not isinstance(name, str):
            raise TypeError("Leg name must be a string.")

        if name in self.names:
            raise ValueError(f"Leg name '{name}' already exists in sequence.")

        if period is not None and not isinstance(period, ArrayWUnits):
            raise TypeError("period must be an ArrayWUnits instance.")

        if dt is not None and not isinstance(dt, ArrayWUnits):
            raise TypeError("dt must be an ArrayWUnits instance if provided.")

        if epochs_vec is not None and not isinstance(epochs_vec, EpochArray):
            raise TypeError("epochs_vec must be an EpochArray if provided.")

        if not isinstance(state_vector, StateArray):
            raise TypeError("state_vector must be a StateArray.")

        if prop is not None and not isinstance(prop, Propagator):
            raise TypeError("prop must be a Propagator.")

        # --------------------
        # structural checks
        # --------------------
        if (
            state_vector.state[0][0] != "position"
            or state_vector.state[1][0] != "velocity"
        ):
            raise ValueError(
                "For MissionSequence, state_vector must have first two components "
                "as ('position','velocity')."
            )

        # Prop/state compatibility — only when prop is provided
        if prop is not None and prop.full_state_vector.size != state_vector.size:
            raise ValueError(
                "Inconsistency: leg state_vector and propagator.full_state_vector "
                "must have the same size."
            )

        # dv_man rules (keep your current constraints)
        if not self.types:
            # first leg: dv_man not allowed
            for j, curr_tuple in enumerate(state_vector.state):
                if curr_tuple[0] == "dv_man":
                    raise ValueError("dv_man must not be present in the first leg.")
            if state_vector.epoch is None:
                raise ValueError("First leg must provide a valid epoch.")
        else:
            # subsequent legs: if dv_man present, must follow an impulsive burn
            for j, curr_tuple in enumerate(state_vector.state):
                if curr_tuple[0] in ("dv_man", "dv_man1", "dv_man2"):
                    if self.types[-1] != "Impulsive Burn":
                        raise ValueError(
                            f"{curr_tuple[0]} present, but previous segment is not an Impulsive Burn."
                        )
                    # dv_man must be last due to MTM construction (your current implementation)
                    if curr_tuple[2] == "estimated" and j != (
                        len(state_vector.state) - 1
                    ):
                        raise ValueError(
                            f"Estimated {curr_tuple[0]} must be the last element in the state vector definition."
                        )

    def _validate_impulsive_burn(
        self,
        name: str,
        dV: ImpulsiveBurn,
        prop: Propagator,
    ) -> None:
        """
        Validates the parameters of an impulsive burn before adding it.

        Ensures that the burn name is unique, follows a valid mission sequence
        structure, and conforms to expected data types.

        Parameters
        ----------
        name : str
            The name of the impulsive burn.

        dV : ImpulsiveBurn
            The impulsive burn model.

        prop : Propagator
            The propagator responsible for simulating the spacecraft dynamics.

        Returns
        -------
        None
        """
        if not isinstance(name, str):
            raise TypeError(
                "The name of the impulsive burn should be a string. Please provide a string."
            )

        if name in self.names:
            raise ValueError(
                f"The name {name} is already defined as a sequence name. Please choose a different name."
            )

        if not isinstance(dV, ImpulsiveBurn):
            raise TypeError("The dV model should be a scb.impulsiveBurnModel object.")

        if not self.types:
            raise ValueError(
                "A dV model cannot be the first segment in a sequence. You can assume it as an initial condition velocity."
            )

        if self.types[-1] == "Impulsive Burn":
            raise ValueError(
                "An impulsive burn cannot be followed by another impulsive burn."
            )

    def _check_sequence_consistency(self) -> None:
        """
        Checks mission sequence consistency (NaN-free version).

        Enforces:
        - epoch continuity between consecutive epochs_vec entries
        - relative ordering constraint: you can add/remove components but not reorder shared ones
        - dv_man exclusivity constraints
        """

        # Collect only legs (StateArray models), skipping nodes and impulsive burns
        list_state_vectors = [
            obj.state for obj in self.models if isinstance(obj, StateArray)
        ]

        # 1) epoch continuity between consecutive epochs_vec (raw ET floats)
        for i in range(1, len(self.epochs_vec)):
            last_epoch_previous_leg = float(self.epochs_vec[i - 1][-1].et)
            first_epoch_new_leg = float(self.epochs_vec[i][0].et)
            if last_epoch_previous_leg != first_epoch_new_leg:
                raise Exception(
                    f"Epoch continuity error: first epoch of segment {i} ({first_epoch_new_leg}) "
                    f"does not match last epoch of segment {i-1} ({last_epoch_previous_leg})."
                )

        # Helper: identity used for "same parameter across legs"
        def _sid(body):
            return getattr(body, "spice_id", None)

        def _key(tup):
            # (name, body_id, size) is a robust identity
            return (tup[0], _sid(tup[4]), int(tup[1]))

        # 2) relative order constraint + dv_man exclusivity
        for i in range(1, len(list_state_vectors)):
            prev_state_vec = list_state_vectors[i - 1]
            curr_state_vec = list_state_vectors[i]

            prev_keys = [_key(t) for t in prev_state_vec]
            curr_keys = [_key(t) for t in curr_state_vec]

            # dv_man family exclusivity within the current leg
            curr_names = [t[0] for t in curr_state_vec]
            dv_keys_present = [
                k for k in ("dv_man", "dv_man1", "dv_man2") if k in curr_names
            ]
            if len(dv_keys_present) > 1:
                raise Exception(
                    f"Inconsistency: {dv_keys_present} cannot be present together in leg {i}."
                )

            # Map previous key -> index for relative-order checking
            prev_index = {k: idx for idx, k in enumerate(prev_keys)}
            last_seen = -1
            for k in curr_keys:
                if k in prev_index:
                    idx_prev = prev_index[k]
                    if idx_prev < last_seen:
                        raise Exception(
                            f"Inconsistency: state element {k} in leg {i} is out of relative order "
                            f"compared to leg {i-1}."
                        )
                    last_seen = idx_prev

        # Optional: keep your "implementation limits" checks below (legs/burn counts, etc.)
        leg_count = sum(1 for leg in self.types if leg == "Leg")
        impulsive_burn_count = sum(1 for burn in self.types if burn == "Impulsive Burn")

    def carry_static_parameters(self) -> list[frozenset]:
        """
        Carry static parameters through the segments that do not declare them.

        A static parameter declared on two segments with others in between
        (a thrust group resumed after a coast, a bias observed on alternate
        arcs) is one parameter of the scenario. Each segment in between gets
        the parameter added to its state, so its value and covariance flow
        through that segment instead of being dropped at the first boundary
        and re-introduced with a fresh a priori at the next one. A carried
        component is dormant there: the propagator holds it constant and hides
        it from the force models (``Propagator.reinitialize``).

        Carried estimated components go before the segment's first considered
        one, as :class:`StateDefinition` requires; carried considered
        components go last. ``dv_man`` blocks are never carried: they belong
        to the impulsive burn in front of the segment declaring them.

        The carry is recomputed from the states the segments were declared
        with, so calling this again is harmless and accounts for segments
        added in the meantime.

        Returns
        -------
        list of frozenset
            One entry per segment: the ``(name, spice_id)`` of its dormant
            components; empty for segments carrying nothing, nodes, and
            impulsive burns.
        """
        positions = [
            i
            for i, model in enumerate(self._models)
            if self._types[i] in ("Leg", "Finite Burn")
            and isinstance(model, StateArray)
        ]

        # States as declared, i.e. without anything a previous call carried.
        records = {}
        declared = {}
        for i in positions:
            record = next(
                (r for r in self._carry_records if r[0] is self._models[i]), None
            )
            records[i] = record
            declared[i] = record[1] if record is not None else self._models[i]

        first_entry = {}
        seen_in = {}
        for i in positions:
            for key, _start, _size, entry in Utils.enumerate_state_blocks(
                declared[i].state
            ):
                first_entry.setdefault(key, entry)
                seen_in.setdefault(key, []).append(i)

        dormant = [frozenset()] * len(self._models)
        carry_records = []
        for i in positions:
            model = declared[i]
            present = {
                key for key, *_ in Utils.enumerate_state_blocks(model.state)
            }
            missing = [
                (key, entry)
                for key, entry in first_entry.items()
                if entry[3] == "static"
                and not entry[0].startswith("dv_man")
                and key not in present
                and seen_in[key][0] < i < seen_in[key][-1]
            ]
            if not missing:
                self._models[i] = model
                self._states_n[i] = sum(int(t[1]) for t in model.state)
                continue

            carried_keys = tuple(key for key, _entry in missing)
            record = records[i]
            if record is not None and record[2] == carried_keys:
                carrying = record[0]
            else:
                components = list(model.state)
                n_estimated = next(
                    (k for k, t in enumerate(components) if t[2] == "considered"),
                    len(components),
                )
                components[n_estimated:n_estimated] = [
                    entry for _key, entry in missing if entry[2] != "considered"
                ]
                components.extend(
                    entry for _key, entry in missing if entry[2] == "considered"
                )
                carrying = StateArray(
                    epoch=model.epoch,
                    origin=model.origin,
                    state=StateDefinition.from_components(components),
                )

            self._models[i] = carrying
            self._states_n[i] = sum(int(t[1]) for t in carrying.state)
            carry_records.append((carrying, model, carried_keys))
            dormant[i] = frozenset(
                (entry[0], getattr(entry[4], "spice_id", None))
                for _key, entry in missing
            )

        self._carry_records = carry_records
        return dormant

    def list_forces(self) -> dict:
        """
        The forces acting on each integrated object, segment by segment.

        Returns
        -------
        dict
            Maps the name of every propagated segment — legs and finite
            burns, in sequence order — to that segment's
            :meth:`~scarabaeus.Propagator.list_forces`: a dict from the name
            of each integrated object to the labels of the forces it feels
            there.
            Impulsive burns and nodes are instantaneous, so they have no
            entry.

        See Also
        --------
        scarabaeus.Propagator.list_forces : The listing of one segment.
        scarabaeus.ForceModelTranslation.list_forces : The labels of one force model.

        Examples
        --------
        >>> seq.list_forces()
        {'Cruise': {'Orbiter': ['ThreeBodyGravity(EARTH, JUPITER)', 'PointMassGravity(SUN)']},
         'Approach': {'Orbiter': ['ThreeBodyGravity(EARTH, JUPITER)', 'CannonballSRP', 'PointMassGravity(SUN)']}}
        """
        return {
            name: prop.list_forces()
            for kind, name, prop in zip(self.types, self.names, self.props)
            if kind in ("Leg", "Finite Burn") and prop is not None
        }

    def propagate(self, map_STM_to_t0: bool = False, verbose: bool = True) -> "ScenarioSetup":
        """
        Propagate the mission sequence and return the resulting scenario.

        Creates a :class:`ScenarioSetup` from this sequence, runs propagation
        through all legs and burns, and returns the populated scenario object.
        This is the recommended entry point for propagation — it avoids the
        need to instantiate :class:`ScenarioSetup` directly.

        Parameters
        ----------
        map_STM_to_t0 : bool, optional
            If ``True``, the State Transition Matrix is mapped continuously
            from the initial epoch ``t0`` across all legs, so that the STM
            at any epoch represents sensitivity to the global initial state.
            If ``False`` (default), the STM is reset to identity at each leg
            boundary, representing only within-leg sensitivity.

        verbose : bool, optional
            When ``True`` (default), print segment progress and initial-condition
            summaries during propagation. Set to ``False`` to silence all output.

        Returns
        -------
        scenario : ScenarioSetup
            The propagated scenario, with ``total_position``, ``total_velocity``,
            ``total_epochsTDB``, ``segment_positions``, ``segment_velocities``,
            and ``STM`` populated and ready for use.

        Examples
        --------
        .. code-block:: python

            scenario = MS.propagate()
            traj = scb.Trajectory("traj.bsp", leg_array=scenario)

            scenario = MS.propagate(map_STM_to_t0=True)
            scenario = MS.propagate(verbose=False)
        """
        # Local import to avoid circular dependency with ScenarioSetup
        from scarabaeus.environment.ScenarioSetup import ScenarioSetup

        scenario = ScenarioSetup(self, map_STM_to_t0=map_STM_to_t0)
        scenario.propagate(verbose=verbose)
        return scenario

    def addInitialCoast(
        self,
        name: str = "Initial coast",
        period: ArrayWUnits = None,
        state_vector: StateArray = None,
        prop: Propagator = None,
        dt: ArrayWUnits = None,
        epochs: EpochArray = None,
    ) -> None:
        """
        Prepend an initial coasting leg to the mission sequence.

        Can be called after finite burns have already been added.
        The leg is inserted at position 0, before all existing segments.

        Parameters
        ----------
        name : str, optional
            Name of the initial coast leg. Defaults to ``"Initial coast"``.
        prop : Propagator, optional
            Propagator for this leg.
        """
        # Save existing sequence
        saved = dict(
            types=self._types[:],
            names=self._names[:],
            periods=self._periods[:],
            models=self._models[:],
            props=self._props[:],
            states_n=self._states_n[:],
            origins=self._origins[:],
            dts=self._dts[:],
            num_steps=self._num_steps[:],
            last_idx=self._last_idx[:],
            epochs_vec=self._epochs_vec[:],
            deferred_grids=self._deferred_grids[:],
            primary_body=self._primary_body,
        )

        # Reset sequence
        self._types = []
        self._names = []
        self._periods = []
        self._models = []
        self._props = []
        self._states_n = []
        self._origins = []
        self._dts = []
        self._num_steps = []
        self._last_idx = []
        self._epochs_vec = []
        self._deferred_grids = []
        self._primary_body = None

        # Add initial coast first
        self.addLeg(
            name=name,
            period=period,
            state_vector=state_vector,
            prop=prop,
            dt=dt,
            epochs=epochs,
        )

        # Re-add all saved segments
        for i in range(len(saved["types"])):
            self._types.append(saved["types"][i])
            self._names.append(saved["names"][i])
            self._periods.append(saved["periods"][i])
            self._models.append(saved["models"][i])
            self._props.append(saved["props"][i])
            self._states_n.append(saved["states_n"][i])
            self._origins.append(saved["origins"][i])
            self._dts.append(saved["dts"][i])
            self._num_steps.append(saved["num_steps"][i])
            self._last_idx.append(saved["last_idx"][i])
            self._epochs_vec.append(saved["epochs_vec"][i])
            self._deferred_grids.append(saved["deferred_grids"][i])

    def addFinalCoast(
        self,
        name: str = "Final coast",
        period: ArrayWUnits = None,
        state_vector: StateArray = None,
        prop: Propagator = None,
        dt: ArrayWUnits = None,
        epochs: EpochArray = None,
    ) -> None:
        """
        Append a final coasting leg to the mission sequence.

        Can be called after all finite burns have been added.

        Parameters
        ----------
        name : str, optional
            Name of the final coast leg. Defaults to ``"Final coast"``.
        prop : Propagator, optional
            Propagator for this leg.
        """
        self.addLeg(
            name=name,
            period=period,
            state_vector=state_vector,
            prop=prop,
            dt=dt,
            epochs=epochs,
        )
