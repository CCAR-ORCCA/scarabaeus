# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Frame,
    ArrayWUnits,
    Body,
    EpochArray,
    MissionSequence,
    SpiceManager,
    CelestialBody,
    Spacecraft,
    ArrayWFrame,
)
import os
import hashlib
import scarabaeus.utils.NumpyWrapper as np
import numpy as _np
import copy
from typing import List
import warnings
import bisect
from pathlib import Path


# -----------------------#
#    Class Definition   #
# -----------------------#
class Trajectory:
    """Represents a spacecraft or celestial object's trajectory in a defined reference frame and origin.

    This class encapsulates trajectory data, including position, velocity, estimated parameters,
    and state transition matrices (STMs). It supports both single-phase trajectories and multi-leg
    mission sequences.

    The trajectory stores the discrete time history
    :math:`\\{(t_i,\\,\\mathbf{r}_i,\\,\\dot{\\mathbf{r}}_i)\\}_{i=0}^{N}` and
    writes it into a SPICE SPK kernel so that the state at any epoch
    :math:`t \\in [t_0, t_N]` can be retrieved via Hermite interpolation, which
    constrains the position curve with the sampled velocities:

    .. math::

        \\mathbf{r}(t) \\approx \\sum_{k=0}^{d} c_k\\,t^k, \\qquad
        \\mathbf{r}(t_i) = \\mathbf{r}_i, \\quad
        \\dot{\\mathbf{r}}(t_i) = \\dot{\\mathbf{r}}_i

    where :math:`d` is ``degree_poly`` (default 3). The degree is always odd: each
    node supplies both a position and a velocity, so a window of :math:`(d+1)/2`
    nodes fixes all :math:`d+1` coefficients.

    The trajectory can be initialized using:

    * Explicit position, velocity, and epoch inputs.
    * A ``StateArray`` directly from a propagator.

    Parameters
    ----------
    name : str
        The name of the trajectory.

    ref_frame : str, optional
        The reference frame in which the trajectory is defined (e.g., ``"J2000"``).
        Defaults to ``None``.

    origin : str, optional
        The origin of the trajectory (e.g., ``"SUN"``). Defaults to ``None``.

    body : CelestialBody, optional
        The celestial body or spacecraft associated with this trajectory. Defaults to
        ``None``.

    degree_poly : int, optional
        The degree of the Hermite polynomial used for SPICE interpolation. Must be
        odd; the interpolation window spans ``(degree_poly + 1) // 2`` states.
        Defaults to ``3``, the two-node cubic, which is the only degree whose
        interpolant is continuous across every node.

    parameters : list of ArrayWUnits, optional
        Additional estimated parameters related to the trajectory (e.g., range biases,
        solar radiation pressure coefficients). Defaults to ``None``.

    offset : ArrayWUnits, optional
        A fixed position offset for coordinate transformations. Defaults to ``None``.

    STMs : list, optional
        The state transition matrices (STMs) associated with the trajectory. Defaults to ``None``.

    propagator_settings_dict : dict, optional
        A dictionary of propagator settings for numerical integration. Defaults to ``None``.

    state_array : StateArray, optional
        If provided, initializes the trajectory from a propagated `StateArray`, extracting
        position, velocity, epoch, reference frame, and parameters. Defaults to ``None``.

    per_body_spk : bool, optional
        When the trajectory holds several bodies, write one SPK per body rather
        than a single file with a segment per body. See :attr:`per_body_spk`.
        Defaults to ``False``.
    max_interp_error : float or None, optional
        Relative tolerance applied when an STM lookup falls between two stored
        epochs and has to interpolate. See :attr:`max_interp_error`, which also
        sets it after construction. ``None`` skips the check. Defaults to ``1e-3``.

    Raises
    ------
    ValueError
        * If both ``state_definition`` and ``sequence_definition`` are provided simultaneously.
        * If both standard trajectory inputs and a ``StateArray`` are provided.
        * If ``StateArray`` is missing position or velocity components.

    Exception
        If the trajectory contains fewer than two data points.

    See Also
    --------
    scarabaeus.MissionSequence : Multi-leg trajectory container.
    scarabaeus.Propagator : Numerical propagator that produces trajectory data.
    scarabaeus.StateArray : State vector passed to the propagator.

    Notes
    -----
    * If ``state_definition`` is provided, the trajectory follows a single continuous model.
    * If ``sequence_definition`` is used, the trajectory consists of multiple segments (e.g., before and after a maneuver).
    * The trajectory is stored in a SPICE kernel format for interpolation and retrieval.

    **Saving to disk — SPK and JSON sidecars**

    Calling :meth:`write_to_spk` writes two kinds of files that always live next to each other:

    1. **SPK binary** (``<name>.bsp``) — the position/velocity time-series in SPICE
       type-13 (unequally spaced Hermite) format.  State at any epoch inside the
       covered arc is retrieved via SPICE polynomial interpolation, with the stored
       velocities constraining the position curve so that the velocity SPICE returns
       is the exact derivative of the position it returns.

    2. **JSON sidecars** — plain-text files written alongside the ``.bsp`` with the
       same base name and the suffixes listed below.  They carry metadata that cannot
       be embedded in a standard SPK segment:

       * ``<name>_parameters.json`` — time-history of estimated parameters (e.g. range
         biases, SRP coefficients, empirical accelerations), plus the full state
         definition that describes what each column represents.
       * ``<name>_STMs.json`` — state transition matrices and their timestamps
         (written only when STMs are attached via :meth:`add_STMs`).
       * ``<name>_settings.json`` — serialised ``propagator_settings_dict`` (force
         model, integrator tolerances, etc.) so the trajectory can be reproduced.
       * ``<name>_mass_profile.json`` — polynomial coefficients of the spacecraft mass
         as a function of time (written only when a ``mass_profile`` is provided).

    Each sidecar is **optional**: the file is created only when the corresponding data
    is not ``None``.  All four sidecars share the same path prefix as the ``.bsp`` so
    that loading the kernel and its associated metadata is unambiguous.

    Examples
    --------
    **Standard Initialization:**

    .. code-block:: python

        import scarabaeus as scb

        # Define epochs
        epoch_start = scb.SpiceManager.jd2et(2461809.72995654)
        epoch_end = scb.SpiceManager.jd2et(2461809.72995654 + 1)
        epochs = scb.EpochArray(np.linspace(epoch_start, epoch_end, 100),sys="TDB")

        # Define position and velocity
        position = scb.ArrayWUnits(np.random.rand(100, 3), "km")
        velocity = scb.ArrayWUnits(np.random.rand(100, 3), "km/s")

        # Initialize trajectory
        traj = scb.Trajectory(
            name="Test_Trajectory",
            position=position,
            velocity=velocity,
            epoch=epochs,
            ref_frame="J2000",
            origin="SUN",
            body=scb.CelestialBody.from_constants("EARTH")
        )

        # Retrieve state at a given epoch
        state = traj.get_state(epochs[0])

        >>> print("Position:", state["position"])
        'Position: [0.123, 0.456, 0.789] [km]'

    **Alternative Initialization from a Propagator:**

    .. code-block:: python

        # Initialize trajectory using propagated StateArray
        traj = scb.Trajectory("Orbiter_trajectory", state_array=propagated_state_array)
    """

    def __init__(
        self,
        ref_frame=None,
        origin=None,
        body=None,
        degree_poly: int = 3,
        parameters=None,
        offset=None,
        STMs=None,
        STMs_timestamp=None,
        propagator_settings_dict=None,
        state_array=None,
        leg_array=None,
        mass_profile=None,
        use_saved_stms: bool = False,
        max_interp_error: float | None = 1e-3,
        per_body_spk: bool = False,
    ):
        # ------------------------------------------------------------------
        # Safety checks / store basics
        # ------------------------------------------------------------------
        if state_array is not None and leg_array is not None:
            raise ValueError("Provide either `state_array` or `leg_array`.")

        self._degree_poly = degree_poly
        self._offset = offset
        self._mass_profile = mass_profile
        self._propagator_settings_dict = propagator_settings_dict
        # When True, STMs are sourced from the saved ``_stms.json`` sidecar
        # (via :meth:`get_STMs`) instead of the in-memory ``_STMs`` list —
        # mirroring how state/parameters are read back from the SPK + JSON.
        self._use_saved_stms = use_saved_stms
        self._spk_path = None  # intialize as empty
        # Relative tolerance applied when a STM lookup lands between two stored
        # epochs and has to interpolate; see the ``max_interp_error`` property.
        self._max_interp_error = max_interp_error
        # Whether a multi-body trajectory writes one SPK per body; see the
        # ``per_body_spk`` property.
        self._per_body_spk = bool(per_body_spk)

        # ==================================================================
        # 1) Initialization from StateArray (supports multi-body)
        # ==================================================================
        if state_array is not None:
            # note how initialized
            self._from_state_array = True
            self._state_array = state_array

            # save the rest
            self._state_definition = state_array.state
            self._sequence_definition = None

            # Epoch & origin from StateArray
            self._epoch = state_array.epoch
            self._origin = state_array.origin.name

            # Build body_map: spice_id -> {body, pos:(vals,units), vel:(vals,units)}
            body_map = {}
            for entry in state_array.state:
                pname, _, _, _, owner, _ = entry
                if pname not in ("position", "velocity"):
                    continue
                if not isinstance(owner, (CelestialBody, Spacecraft)):
                    continue

                sid = owner.spice_id
                body_map.setdefault(sid, {"body": owner, "pos": None, "vel": None})

                key = (pname, sid)
                if key not in state_array.values_array:
                    continue

                vals, unts = state_array.values_array[key]
                if pname == "position":
                    body_map[sid]["pos"] = (vals, unts)
                else:
                    body_map[sid]["vel"] = (vals, unts)

            # Keep only complete (pos,vel) pairs
            body_map = {
                sid: info
                for sid, info in body_map.items()
                if info["pos"] is not None and info["vel"] is not None
            }

            if not body_map:
                raise ValueError(
                    "StateArray must contain at least one "
                    "(position, velocity) pair for a body."
                )

            self._body_map = body_map

            # Reference frame
            self._ref_frame = state_array.state[0][5].frame

            # Store STM info (caller-provided)
            self._STMs = STMs
            self._STMs_timestamp = STMs_timestamp

            # Primary body anchors this Trajectory object
            primary_sid, primary_info = next(iter(body_map.items()))
            primary_body = primary_info["body"]
            (p_pos_vals, p_pos_u), (p_vel_vals, p_vel_u) = (
                primary_info["pos"],
                primary_info["vel"],
            )

            self._body = primary_body
            self._position = ArrayWUnits(p_pos_vals, p_pos_u)
            self._velocity = ArrayWUnits(p_vel_vals, p_vel_u)

            # Keep legacy `_parameters` field as primary body's params
            (
                self._parameters,
                self._parameter_units,
            ) = self._extract_body_parameters_from_state(state_array, self._body)

        # ==================================================================
        # 2) Initialization from leg_array (multi-leg)
        # ==================================================================
        else:
            if leg_array is None:
                # Bare trajectory over a body already in the SPICE pool: no
                # state or sequence of its own, so get_state reads SPICE only
                self._from_state_array = False
                self._state_array = None
                self._state_definition = None
                self._sequence_definition = None
                self._epoch = None
                self._ref_frame = ref_frame
                self._origin = origin.name if hasattr(origin, "name") else origin
                self._body = body
                self._parameters = parameters
                self._parameter_units = None
                self._STMs = STMs
                self._STMs_timestamp = STMs_timestamp
                return

            # note how initialized
            self._from_state_array = False
            self._state_array = state_array

            # save everything else
            seq = leg_array
            self._seq = seq
            self._sequence_definition = seq.mission_sequence
            self._state_definition = None

            # Global time span from sequence
            self._epoch = seq.total_epochsTDB
            self._mass_profile = mass_profile
            self._offset = offset
            self._propagator_settings_dict = propagator_settings_dict

            # Store parameters / STMs. A sequence whose legs estimate nothing
            # still yields one None per leg — a truthy list — so collapse the
            # all-empty case to None and keep `parameters` honest.
            leg_parameters = [
                p.quantity if isinstance(p, ArrayWFrame) else p
                for p in (seq.total_parameters or [])
            ]
            self._parameters = (
                leg_parameters
                if SpiceManager._has_parameter_payload(leg_parameters)
                else None
            )
            # Sequence legs carry their own per-leg units already.
            self._parameter_units = None
            # Same for the STMs: a sequence propagated with propagate_STM=False
            # yields one None per leg, again a truthy list.
            seq_stms = getattr(seq, "STM", None)
            self._STMs = (
                [stm.quantity.values for stm in seq_stms]
                if seq_stms and all(stm is not None for stm in seq_stms)
                else STMs
            )
            self._STMs_timestamp = STMs_timestamp

            # Collect NON-impulsive legs with their SEQUENCE INDEX
            legs = []
            for i in range(seq.len_sequence):
                ev_type = seq.mission_sequence.types[i]
                if ev_type == "Impulsive Burn" or ev_type == "Node":
                    continue

                key = seq.mission_sequence.names[i]
                pos_awf = seq.segment_positions[key]
                vel_awf = seq.segment_velocities[key]

                body_i = seq.mission_sequence.props[i].primary_body
                origin_i = seq.mission_sequence.origins[i]
                frame_i = pos_awf.frame

                epoch_leg = EpochArray(
                    seq.segment_epochsTDB[key],
                    seq.total_epochsTDB.system,
                )

                legs.append(
                    dict(
                        seq_idx=i,
                        key=key,
                        pos_awf=pos_awf,
                        vel_awf=vel_awf,
                        body=body_i,
                        origin=origin_i,
                        frame=frame_i,
                        epoch_leg=epoch_leg,
                    )
                )

            if not legs:
                raise ValueError("leg_array contains no non-impulsive legs to write.")

            # Expose kinematics
            self._legs = legs
            body_id_blocks, origin_id_blocks, frame_blocks = [], [], []
            for k, leg in enumerate(legs):
                ep = leg["epoch_leg"]
                # drop duplicate boundary sample
                if k > 0:
                    t_vals = ep.times.values[1:]
                else:
                    t_vals = ep.times.values
                n = len(t_vals)
                # Concatenate labels (per-sample)
                body_id_blocks.append(np.full(n, leg["body"], dtype=object))
                origin_id_blocks.append(np.full(n, leg["origin"], dtype=object))
                frame_blocks.append(np.full(n, leg["frame"], dtype=object))

            # Keep old scalars for backwards compatibility (anchor)
            self._body = np.concatenate(body_id_blocks)
            self._origin = np.concatenate(origin_id_blocks)
            self._ref_frame = np.concatenate(frame_blocks)
            self._epoch = seq.total_epochsTDB
            self._position = (
                seq.total_position.quantity
            )  # or .quantity.values etc depending on type
            self._velocity = seq.total_velocity.quantity

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def per_body_spk(self) -> bool:
        """
        Whether a multi-body trajectory is written as one SPK per body.

        ``False`` (the default) writes a single ``.bsp`` carrying one segment
        per body — compact, and what SPICE is designed for. ``True`` writes
        ``mission_ORBITER.bsp``, ``mission_EARTH.bsp`` and so on, each with its
        own sidecars, for a downstream tool that expects one body per kernel.

        Either way the bodies' parameter sidecars are kept apart: they are
        named per body whenever there is more than one, since every body writes
        ``_parameters.json`` and a shared name would leave only the last.

        Ignored for a single-body trajectory, which always writes one file.
        """
        return self._per_body_spk

    @per_body_spk.setter
    def per_body_spk(self, value: bool) -> None:
        """Choose one SPK per body, or one file with a segment per body."""
        self._per_body_spk = bool(value)

    @property
    def max_interp_error(self) -> float | None:
        """
        Relative tolerance on linear interpolation of the STM.

        Every ``get_STM`` / ``get_STM_sequence`` lookup that falls between two
        stored epochs interpolates, and rejects the result when the estimated
        **relative** error exceeds this value. The estimate is curvature-based
        and scales as :math:`dt^2`, so halving the STM grid step cuts it by
        four.

        The default, ``1e-3``, admits a grid step of roughly a minute on a low
        Earth orbit. Raise it to accept a looser fit on a coarser grid, or set
        it to ``None`` to skip the check entirely.

        Notes
        -----
        This replaces an absolute threshold on the first difference
        ``alpha (1 - alpha) ||Phi_1 - Phi_0||``, which measured
        :math:`\\|\\Phi\\|` rather than the interpolation error and so failed
        on a long arc at any grid step. Values carried over from that scheme
        are not comparable.
        """
        return self._max_interp_error

    @max_interp_error.setter
    def max_interp_error(self, value):
        """Set the relative STM interpolation tolerance for this trajectory."""
        if value is not None:
            value = float(value)
            if value <= 0.0:
                raise ValueError(
                    "max_interp_error must be a positive relative tolerance, "
                    f"got {value!r}. Use None to disable the check."
                )
        self._max_interp_error = value

    @property
    def position(self) -> ArrayWUnits:
        """
        The position history of the trajectory in the specified reference frame.
        """
        return self._position

    @property
    def velocity(self) -> ArrayWUnits:
        """
        The velocity history of the trajectory in the specified reference frame.
        """
        return self._velocity

    @property
    def epoch(self) -> EpochArray:
        """
        The epochs associated with the trajectory.
        """
        return self._epoch

    @property
    def start_epoch(self) -> EpochArray:
        """
        The start epoch of the trajectory.
        """
        return self._epoch[0]

    @property
    def end_epoch(self) -> EpochArray:
        """
        The end epoch of the trajectory.
        """
        return self._epoch[-1]

    @property
    def ref_frame(self) -> Frame:
        """
        The reference frame of the trajectory.
        """
        return self._ref_frame

    @property
    def origin(self) -> str:
        """
        The origin of the trajectory.
        """
        return self._origin

    @property
    def body(self) -> Body:
        """
        The celestial body or spacecraft associated with the trajectory.
        """
        return self._body

    @property
    def poly_deg(self) -> int:
        """
        The degree of the polynomial used for interpolation.
        """
        return self._degree_poly

    @property
    def parameters(self) -> list:
        """
        Additional estimated parameters associated with the trajectory. These parameters
        include mission-specific estimated values, such as:

        * Ground station range biases (e.g., clock offsets, tropospheric delays).
        * Additional environmental parameters (e.g., solar radiation pressure coefficients).
        * Model calibration parameters (e.g., empirical accelerations).

        If no additional parameters exist, returns ``None``.
        """
        return self._parameters

    @property
    def parameter_units(self) -> list | None:
        """
        Per-parameter units backing :attr:`parameters`, in column order.

        A list of ``(name, dim, units)``, one entry per estimated parameter of
        the primary body. Parameters of one body need not share units — a
        spacecraft solving for a delta-v (km/s) and a maneuver-epoch
        correction (sec) is ordinary — and when they differ, :attr:`parameters`
        is deliberately unitless because no single label describes the stacked
        block. This is then the only faithful record, and it is what gets
        written to the parameters sidecar.

        ``None`` for sequence trajectories, whose legs carry their own units,
        and for trajectories with no estimated parameters.
        """
        return getattr(self, "_parameter_units", None)

    @property
    def state_definition(self) -> dict:
        """
        The definition of the trajectory's state vector. This describes the structure
        of the estimated state, specifying:

        * Which variables are included in the state (e.g., position, velocity, parameters).
        * Whether they are dynamic or static.
        * Their associated reference objects (e.g., spacecraft, ground stations).
        """
        return self._state_definition

    @property
    def sequence_definition(self) -> MissionSequence:
        """
        The mission sequence definition governing the trajectory's structure. This property is used when
        the trajectory consists of multiple segments (legs), where different state definitions or
        models are required for each segment.
        It is typically used for:

        * Handling maneuvers (e.g., impulsive burns).
        * Multi-phase missions with different dynamical regimes.
        * Modeling spacecraft transitions between different reference frames.
        """
        return self._sequence_definition

    @property
    def offset(self) -> ArrayWUnits:
        """
        A fixed position offset for coordinate transformations.
        """
        return self._offset

    @property
    def STMs(self) -> list | None:
        """
        The state transition matrices (STMs) associated with the trajectory.
        """
        return self._STMs

    @property
    def use_saved_stms(self) -> bool:
        """
        When ``True``, :meth:`get_STMs` reads STMs from the saved ``_stms.json``
        sidecar instead of the in-memory ``STMs`` list.
        """
        return self._use_saved_stms

    @use_saved_stms.setter
    def use_saved_stms(self, value: bool) -> None:
        self._use_saved_stms = bool(value)

    @property
    def propagator_settings_dict(self) -> dict | None:
        """
        The dictionary of propagator settings for numerical integration.
        """
        return self._propagator_settings_dict

    @property
    def STMs_timestamp(self) -> List | None:
        """
        The epochs at which each STM was generated. Must correspond
        one-to-one with the `STMs` list.
        """
        return self._STMs_timestamp

    @property
    def spk_path(self) -> str | None:
        """
        The filepath for the Trajectory's associated SPK file.
        Returns ``None`` if no file exists.
        """
        return self._spk_path

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods  #
    # ----------------#
    @staticmethod
    def _normalize_stms(stms):
        """
        Normalize attached STMs into a list of ``(n, n)`` ``numpy`` arrays so the
        STM JSON sidecar can be written (``SpiceManager._prepare_stm_data``).

        Accepts: an ``(N, n, n)`` array, an ``(N, n*n)`` flattened time series,
        a single ``(n, n)`` matrix, a list of ``ArrayWFrame`` / ``ndarray`` /
        flattened vectors, or one ``ArrayWFrame`` / ``ArrayWUnits`` holding
        the whole series (``Propagator.STM``). Returns ``None`` if no STMs are
        present.
        """
        if stms is None:
            return None

        # A whole-series container (Propagator.STM) has no __iter__, so Python
        # would iterate it through __getitem__ until its (n, n) unit grid runs
        # out, keeping only the first n epochs. Read its values instead.
        if hasattr(stms, "quantity"):
            stms = stms.quantity
        if hasattr(stms, "values") and hasattr(stms, "units"):
            stms = stms.values

        # Try a homogeneous numeric array first.
        try:
            arr = _np.asarray(stms, dtype=float)
        except Exception:
            arr = None

        if arr is not None and arr.dtype != object:
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
            if arr.ndim == 2:
                rows, cols = arr.shape
                n = int(round(cols**0.5))
                # (N, n*n) flattened series vs a single (n, n) matrix
                if rows != cols and n * n == cols:
                    return [arr[i].reshape(n, n) for i in range(rows)]
                return [arr]
            if arr.ndim == 1:
                n = int(round(arr.size**0.5))
                return [arr.reshape(n, n)]

        # Heterogeneous: iterate element-wise (e.g. list of ArrayWFrame).
        out = []
        for M in stms:
            v = _np.asarray(
                M.quantity.values if hasattr(M, "quantity") else M, dtype=float
            )
            if v.ndim == 1:
                n = int(round(v.size**0.5))
                v = v.reshape(n, n)
            out.append(v)
        return out or None

    def _stm_units_dict(self):
        """
        Best-effort extraction of the STM units as a serializable dict
        (``Units.to_dict()``), or ``None`` when the attached STMs carry no unit
        information.

        Prefers units captured by :meth:`add_STMs`, then falls back to
        inspecting an ``ArrayWFrame`` STM container supplied at construction.
        The STM frame is written separately (as the trajectory ``reference_frame``)
        by :meth:`SpiceManager._prepare_stm_data`.

        Returns
        -------
        dict or None
            The STM units serialized via ``Units.to_dict()``, or ``None`` when no
            unit information is available.
        """
        units = getattr(self, "_STM_units", None)
        if units is None:
            stms = self._STMs
            if hasattr(stms, "quantity"):
                units = getattr(stms.quantity, "units", None)
            elif isinstance(stms, list) and stms and hasattr(stms[0], "quantity"):
                units = getattr(stms[0].quantity, "units", None)
        if isinstance(units, list):
            units = units[0] if units else None
        return units.to_dict() if hasattr(units, "to_dict") else None

    def write_to_spk(
        self,
        spk_path: str,
        stm_format: str = "json",
        sidecar_format: str = "json",
        append_if_exists: bool = True,
        write_stms: bool = True,
        per_body_spk: bool = None,
    ) -> None:
        """
        Write the trajectory to disk as a SPICE SPK file plus JSON sidecars.

        The ``.bsp`` file contains the position/velocity time-series (SPICE type-13
        Hermite segments). See Notes section for information on additional data
        that cannot be embedded in an SPK.

        Parameters
        ----------
        spk_path : str
            Destination file path.  Must end with ``.bsp``.
        stm_format : {"json", "pickle"}, optional
            On-disk format for the STM sidecar. ``"json"`` (default) writes a
            human-readable ``_stms.json``; ``"pickle"`` writes a compact binary
            ``_stms.pkl`` (lighter on disk for large STM histories). Either is
            read back transparently by :meth:`get_STMs`.
        sidecar_format : {"json", "pickle"}, optional
            On-disk format for the parameters / settings / mass-profile sidecars.
            ``"json"`` (default) writes ``*_parameters.json`` etc.; ``"pickle"``
            writes ``*_parameters.pkl`` etc. (same content, binary). Read back
            transparently by the ``SpiceManager`` getters.
        append_if_exists : bool, optional
            If the given spk_path already exists:
                - ``True``: append the trajectory segment to that file.
                - ``False``: delete the existing file and write a new one in its place.

            Defaults to ``True``.
        write_stms : bool, optional
            Whether to write the STM sidecar at all. It is by far the largest
            artifact a trajectory produces — one matrix per output epoch,
            growing with the square of the state size — so a run that only
            wants the state history can skip it entirely. Defaults to ``True``.
            The STMs held in memory are unaffected; only the file is skipped,
            so anything reading them back later (``use_saved_stms``) needs it.
        per_body_spk : bool, optional
            When the trajectory holds several bodies, write **one SPK per body**
            — ``mission_EARTH.bsp``, ``mission_ORBITER.bsp`` — instead of a
            single file carrying one segment per body. Each file then owns its
            sidecars, which is what a downstream tool expecting one body per
            kernel wants. Defaults to :attr:`per_body_spk`, i.e. ``False``:
            one file with a segment per body. Ignored for a single body.

        Raises
        ------
        ValueError
            If ``spk_path`` does not end with ``.bsp``.
        UserWarning
            If this ``Trajectory`` already has an associated SPK at a different path.

        Notes
        -----
        The written SPK is furnished. SPICE resolves each epoch from the
        last-loaded kernel that covers it, so within its time span this file
        shadows any SPK loaded earlier for the same body (a reconstructed
        mission ephemeris, for instance). Unload it, or write it to a path you
        can unload, if those earlier kernels must stay in effect.

        Additional data that cannot be embedded in an SPK
        segment is written as plain-text JSON files with the same base path:

        * ``<spk_path base>_parameters.json`` — estimated parameters and state
          definition (written when :attr:`parameters` is not ``None``).
        * ``<spk_path base>_settings.json``   — propagator settings dictionary
          (written when :attr:`propagator_settings_dict` is not ``None``).
        * ``<spk_path base>_mass_profile.json`` — spacecraft mass polynomial
          (written when a ``mass_profile`` was passed at construction time).
        * ``<spk_path base>_stms.json``        — state transition matrices
          (written when STMs are attached and ``write_stms`` is ``True``).

        Each sidecar is created only when the corresponding data is present.
        All files share the same directory and path prefix as the ``.bsp``.

        Examples
        --------
        .. code-block:: python

            traj.write_to_spk("/output/mission.bsp")
            # Produces: mission.bsp, mission_parameters.json,
            #           mission_settings.json  (if propagator settings exist)
        """
        # ensure path ends with .bsp
        if not spk_path.endswith(".bsp"):
            raise ValueError("SPK path must end with .bsp extension.")

        # save path to trajectory
        if append_if_exists:
            if self._spk_path is None:
                # no associated SPK -> save to trajectory
                self._spk_path = spk_path
            else:
                # previously existing SPK path -> warn that it's being lost
                raise UserWarning(
                    f"An associated SPK file for this Trajectory already exists at:\n"
                    f"{self._spk_path}\nThe file will not be overwritten, but it "
                    f"will no longer be associated with this Trajectory. Now using:\n"
                    f"{spk_path}"
                )
        else:  # append_if_exists = False
            self._spk_path = spk_path
            # delete existing file on disk if it exists
            existing = Path(spk_path)
            if existing.exists():
                existing.unlink()

        if per_body_spk is None:
            per_body_spk = self.per_body_spk

        # write to SPK depending on construction
        if self._from_state_array:
            # constructed from a state array
            is_single_body = len(self._body_map) == 1

            for sid, info in self._body_map.items():
                b = info["body"]
                (bpos_vals, _), (bvel_vals, _) = info["pos"], info["vel"]

                # One body per file, or one file with a segment per body. Either
                # way each body needs its own sidecar stem: they all write
                # ``_parameters.json``, so a shared stem loses every body but
                # the last.
                if is_single_body:
                    body_spk_path = spk_path
                    body_sidecar_name = None
                elif per_body_spk:
                    body_spk_path = self._body_tagged_path(spk_path, b)
                    body_sidecar_name = None
                else:
                    body_spk_path = spk_path
                    body_sidecar_name = self._body_tagged_path(spk_path, b)

                spk_segid = self._seg_id(body_spk_path)
                state_spice = self._state_spice_from_pv(bpos_vals, bvel_vals)

                # Per-body parameters extracted from the StateArray, with the
                # per-parameter units that describe each stacked column.
                params_json, params_units = self._extract_body_parameters_from_state(
                    self._state_array, b
                )

                # Metadata only once (primary)
                # TDB epochs once
                epochs_TDB_list = self._epochs_tdb_list(self._epoch)

                primary_sid, primary_info = next(iter(self._body_map.items()))

                is_primary = sid == primary_sid
                stms_json = (
                    self._normalize_stms(self._STMs)
                    if (is_primary and write_stms)
                    else None
                )
                prop_settings = self._propagator_settings_dict if is_primary else None
                mass_prof = self._mass_profile if is_primary else None

                SpiceManager.write_spk_segment_type13(
                    SPK_name=body_spk_path,
                    SPK_SEG_ID=spk_segid,
                    body=b,
                    origin_ID=SpiceManager.name_to_ID(self._origin),
                    ref_frame=self._ref_frame,
                    epochs_TDB=epochs_TDB_list,
                    degree_poly=self._degree_poly,
                    state_SPICE=state_spice,
                    parameters_JSON=params_json,
                    parameters_units=params_units,
                    state_definition=self._state_definition,
                    sequence_definition=self._sequence_definition,
                    leg_data=None,
                    STMs_JSON=stms_json,
                    STM_units=self._stm_units_dict() if is_primary else None,
                    propagator_settings=prop_settings,
                    mass_profile=mass_prof,
                    stm_format=stm_format,
                    sidecar_format=sidecar_format,
                    sidecar_name=body_sidecar_name,
                )

        else:
            # constructed from a scenario
            # Identify bodies across legs
            body_ids = {leg["body"].spice_id for leg in self._legs}
            is_single_body = len(body_ids) == 1

            # write files depending on how many bodies
            match is_single_body:
                case True:
                    # Single-body sequence: one SPK
                    spk_segid = self._seg_id(spk_path)

                    self._write_segments_only(spk_path, spk_segid, self._legs)

                    # Write JSON once (all metadata)
                    self._write_metadata_only_json(
                        spk_name=spk_path,
                        body_legs=self._legs,
                        params_json=self._parameters,
                        stms_json=self._STMs if write_stms else None,
                        settings_json=self._propagator_settings_dict,
                        mass_prof=self._mass_profile,
                        seq=self._seq,
                        stm_format=stm_format,
                        sidecar_format=sidecar_format,
                    )
                case False:
                    # Multi-body sequence: one SPK per body
                    legs_by_body = {}
                    for leg in self._legs:
                        sid = leg["body"].spice_id
                        legs_by_body.setdefault(sid, []).append(leg)

                    for sid, body_legs in legs_by_body.items():
                        spk_segid = self._seg_id(spk_path)

                        self._write_segments_only(spk_path, spk_segid, body_legs)

                        # Metadata only for "primary body" SPK by convention,
                        # OR if you want per-body metadata, set params/stms appropriately here.
                        is_primary = sid == self._legs[0]["body"].spice_id

                        params_json = self._parameters if is_primary else None
                        stms_json = self._STMs if (is_primary and write_stms) else None
                        settings_json = (
                            self._propagator_settings_dict if is_primary else None
                        )
                        mass_prof = self._mass_profile if is_primary else None

                        if any(
                            v is not None
                            for v in (params_json, stms_json, settings_json, mass_prof)
                        ):
                            self._write_metadata_only_json(
                                spk_name=spk_path,
                                body_legs=body_legs,
                                params_json=params_json,
                                stms_json=stms_json,
                                settings_json=settings_json,
                                mass_prof=mass_prof,
                                seq=self._seq,
                                stm_format=stm_format,
                                sidecar_format=sidecar_format,
                            )

    def _write_segments_only(self, spk_name, spk_segid, body_legs):
        SpiceManager.unload_kernel_from_pool(spk_name)
        # Append all segments
        for leg in body_legs:
            epochs_tdb = self._ensure_tdb_list(leg["epoch_leg"])
            state_spice = self._state_spice_from_awf(leg["pos_awf"], leg["vel_awf"])

            # No JSON here (parameters/STM/settings/mass profile deferred)
            SpiceManager.write_spk_segment_type13(
                SPK_name=spk_name,
                SPK_SEG_ID=spk_segid,
                body=leg["body"],
                origin_ID=leg["origin"].spice_id,
                ref_frame=leg["frame"],
                epochs_TDB=epochs_tdb,
                degree_poly=self._degree_poly,
                state_SPICE=state_spice,
                parameters_JSON=None,
                # STMs_JSON=None,
                state_definition=self._state_definition,
                sequence_definition=self._sequence_definition,
                leg_data=None,
                propagator_settings=None,
                mass_profile=None,
            )

    def _write_metadata_only_json(
        self,
        spk_name: str,
        body_legs: list,
        params_json: dict,
        stms_json: dict,
        settings_json: dict,
        mass_prof: dict,
        seq,
        stm_format: str = "json",
        sidecar_format: str = "json",
    ):
        """
        Write SPK metadata to a JSON file for the given spacecraft legs.

        This method constructs leg-specific data and writes comprehensive metadata about
        the trajectory, including epochs, reference frames, propagation settings, and
        mass profiles, to a JSON file via the SpiceManager.

        Parameters
        ----------
        spk_name : str
            The name of the SPK (Spacecraft and Planet Kernel) file.
        body_legs : list
            A list of dictionaries containing leg information, where each
            dictionary includes 'body', 'origin', and 'frame' keys. If empty, the method
            returns early without writing metadata.
        params_json : dict
            JSON-formatted parameters for the trajectory.
        stms_json : dict
            JSON-formatted State Transition Matrices (STMs).
        settings_json : dict
            JSON-formatted propagator settings.
        mass_prof : dict
            Mass profile data for the spacecraft.

        Returns
        -------
        None
            Returns early without writing if body_legs is empty.

        Raises
        ------
        Potentially from SpiceManager.write_spk_metadata_json depending on
        implementation details.

        Notes
        -----
        - Uses the first leg's frame, origin, and body as top-level metadata.
        - Leg-specific frame, origin, and body information is stored separately in leg_data.
        - Epochs are derived from the global sequence span (already in TDB format).
        """
        if not body_legs:
            return

        leg_data = self._build_leg_data_map(body_legs)

        # For JSON, we just need a time reference. Use the global sequence span (already TDB).
        epochs_tdb_global = seq.total_epochsTDB.times.values.tolist()

        # Use the first leg's frame/origin/body for top-level metadata.
        # (Leg-specific frame/origin/body live in leg_data["by_seq_idx"].)
        first = body_legs[0]

        SpiceManager.write_spk_metadata_json(
            SPK_name=spk_name,
            body=first["body"],
            origin_ID=first["origin"].spice_id,
            ref_frame=first["frame"],
            epochs_TDB=epochs_tdb_global,
            degree_poly=self._degree_poly,
            parameters_JSON=params_json,
            STMs_JSON=self._normalize_stms(stms_json),
            STM_units=self._stm_units_dict(),
            state_definition=self._state_definition,
            sequence_definition=self._sequence_definition,
            leg_data=leg_data,
            propagator_settings=settings_json,
            mass_profile=mass_prof,
            stm_format=stm_format,
            sidecar_format=sidecar_format,
        )

    def _ensure_tdb_list(self, epoch_array):
        if epoch_array.system != "TDB":
            epoch_array = copy.deepcopy(epoch_array).to(sys="TDB")
        return epoch_array.times.values.tolist()

    def _seg_id(self, spk_name: str) -> str:
        return f"SEGID_{hashlib.sha1(spk_name.encode()).hexdigest()[:10]}"

    def _epochs_tdb_list(self, epoch_array):
        """Return epochs as a python list in TDB."""
        if epoch_array.system != "TDB":
            epoch_array = copy.deepcopy(epoch_array).to(sys="TDB")
        return epoch_array.times.values.tolist()

    def _state_spice_from_pv(self, pos_vals, vel_vals) -> list:
        """pos_vals, vel_vals assumed (N,3) arrays -> list of N 6-vectors."""
        return np.concatenate((pos_vals, vel_vals), axis=1).tolist()

    def _state_spice_from_awf(self, pos_awf, vel_awf) -> list:
        pos_awu = pos_awf.quantity
        vel_awu = vel_awf.quantity
        return np.concatenate((pos_awu.values, vel_awu.values), axis=1).tolist()

    # Build leg_data as mapping keyed by sequence index
    def _build_leg_data_map(self, legs_subset):
        """
        Return leg_data in a robust form:
        leg_data["by_seq_idx"][seq_idx] = {body_ID, origin_ID, reference_frame}
        """
        by_seq = {}
        for leg in legs_subset:
            by_seq[leg["seq_idx"]] = dict(
                body_ID=leg["body"].spice_id,
                origin_ID=leg["origin"].spice_id,
                reference_frame=leg["frame"].name,
            )
        return {"by_seq_idx": by_seq}

    def _extract_body_parameters_from_state(self, state_array, target_body):
        """
        Collect the parameters from `state_array` that `target_body`'s sidecar
        holds: its own and, when it is the first propagated body, those whose
        owner is not propagated (a ground station's range bias, say). The
        selection is :meth:`SpiceManager._parameter_entries`, the one the
        sidecar labels are written from, so columns and names cannot drift.

        Parameters of a single body need not share units. A spacecraft solving
        for both a delta-v (km/s) and a maneuver-epoch correction (sec) is
        ordinary, as is FiniteBurn's dimensionless ``k_thrust`` alongside the
        angular ``dRA``/``dDEC``. The values are still returned stacked, so
        every existing consumer keeps its shape, and the per-parameter units
        are reported separately for the sidecar to record.

        Returns
        -------
        tuple
            ``(stacked, blocks)`` where ``stacked`` is an
            :class:`ArrayWUnits` of shape ``(n_epochs, total_dim)`` and
            ``blocks`` is a list of ``(name, dim, units)`` describing each
            parameter in column order. ``stacked`` carries the shared units
            when every parameter agrees, and is left dimensionless when they
            do not — mislabelling the whole block with whichever unit came
            first would be worse. In that case ``blocks`` is the only faithful
            record of units, so it is what gets serialized.

            ``(None, None)`` when the body has no parameters.
        """
        if state_array is None or target_body is None:
            return None, None

        epoch_size = state_array.epoch.size
        if epoch_size <= 0:
            return None, None

        values_map = state_array.values_array
        param_blocks = []
        param_meta = []
        target_sid = getattr(target_body, "spice_id", None)

        for name, size, _, dyn_type, owner in SpiceManager._parameter_entries(
            state_array.state_definition, target_sid
        ):
            key = (name, getattr(owner, "spice_id", None))
            if key not in values_map:
                continue

            vals, units = values_map[key]
            vals = np.atleast_1d(vals)

            if dyn_type == "static":
                if vals.size != size:
                    raise ValueError(
                        f"Static parameter '{name}' for body {owner} must have size {size}, got {vals.size}."
                    )
                vals = np.tile(vals, (epoch_size, 1))
            elif dyn_type == "dynamic":
                if vals.shape[0] != epoch_size:
                    raise ValueError(
                        f"Dynamic parameter '{name}' for body {owner} has length {vals.shape[0]}, "
                        f"but epoch size is {epoch_size}."
                    )
                if vals.ndim == 1:
                    vals = vals.reshape(epoch_size, 1)
            else:
                raise ValueError(
                    f"Unknown dynamics type '{dyn_type}' for parameter '{name}'."
                )

            param_blocks.append(vals)
            param_meta.append((name, int(vals.shape[1]), units))

        if not param_blocks:
            return None, None

        stacked = np.hstack(param_blocks)

        # Shared units let the stacked array stay fully self-describing, which
        # is the common case and what the legacy sidecar fields encode. When
        # they differ there is no single honest label for the whole block, so
        # the stacked array is left unitless and `param_meta` carries the truth.
        first_units = param_meta[0][2]
        uniform = all(u == first_units for _, _, u in param_meta)

        return ArrayWUnits(stacked, first_units if uniform else None), param_meta

    def get_STMs(
        self,
        epoch_input: EpochArray,
        origin_input=None,
        ref_frame_input=None,
        folder_path_override=None,
    ):
        """
        Retrieve the State Transition Matrix at a given epoch.

        Mirrors :meth:`get_state`'s disk-vs-memory behaviour for STMs. When
        :attr:`use_saved_stms` is ``True`` and an associated SPK exists, the STM
        is read (and interpolated if needed) from the saved ``_stms.json``
        sidecar via :meth:`SpiceManager.get_STMs`. Otherwise the in-memory STM
        nearest the requested epoch is returned.

        Parameters
        ----------
        epoch_input : EpochArray
            Scalar epoch at which to retrieve the STM.
        origin_input, ref_frame_input : optional
            Override origin / reference frame; default to the trajectory's own.
        folder_path_override : str or os.PathLike or iterable, optional
            Where to look for the STM sidecar: an SPK, a file, a folder, or an
            iterable of those — see :meth:`SpiceManager.sidecar_files`.
            Defaults to this trajectory's own SPK.

        Returns
        -------
        numpy.ndarray
            The ``(n, n)`` State Transition Matrix at the requested epoch.
        """
        origin = origin_input if origin_input else self.origin
        ref_frame = ref_frame_input if ref_frame_input else self.ref_frame
        body = self.body

        if epoch_input.system != "TDB":
            epoch_input = epoch_input.to(sys="TDB")
        epoch_et = epoch_input.times.values
        epoch_et = float(np.atleast_1d(epoch_et)[0])

        # Disk source: read from the saved _stms.json sidecar.
        if self._use_saved_stms:
            if folder_path_override is None and self._spk_path is not None:
                folder_path_override = os.path.abspath(self._spk_path)
            return SpiceManager.get_STMs(
                body,
                epoch_et,
                ref_frame,
                origin,
                folder_path_override=folder_path_override,
            )

        # Memory source: nearest stored STM to the requested epoch.
        if self._STMs is None:
            raise ValueError("No STMs are attached to this Trajectory.")
        stms = self._normalize_stms(self._STMs)
        timestamps = self._STMs_timestamp
        if timestamps is None:
            # No timestamps: fall back to the first STM.
            return stms[0]
        ts = np.asarray([float(np.atleast_1d(t)[0]) for t in timestamps], dtype=float)
        idx = int(np.argmin(np.abs(ts - epoch_et)))
        return stms[idx]

    def get_state(
        self,
        epoch_input: EpochArray,
        origin_input=None,
        ref_frame_input=None,
        idx_leg_input=None,
        folder_path_override=None,
    ):
        """
        Retrieves the trajectory state at a specified epoch.

        Queries SPICE to obtain position, velocity, and any associated parameters
        for a given epoch, considering optional reference frame and origin transformations.

        Parameters
        ----------
        epoch_input : EpochArray
            The epoch at which to compute the state.

        origin_input : str, optional
            Custom origin for the state computation. Defaults to the trajectory's origin.

        ref_frame_input : str, optional
            Custom reference frame for the state computation. Defaults to the trajectory's
            reference frame.

        idx_leg_input : int, optional
            The index of the trajectory leg to retrieve if the trajectory is sequence-based.
            Defaults to ``None``.

        folder_path_override : str or os.PathLike or iterable, optional
            Where to look for the parameters and propagator-settings sidecars:
            an SPK, a file, a folder, or an iterable of those — see
            :meth:`SpiceManager.sidecar_files`. Defaults to ``None``, which
            uses this trajectory's own SPK.

        Returns
        -------
        state : dict
            The queried state, containing:

            * ``"position"`` : ArrayWUnits = The position vector. Expressed in units of kilometers.
            * ``"velocity"`` : ArrayWUnits = The velocity vector. Expressed in units of :math:`\\frac{km}{s}`
            * ``"epoch"`` : EpochArray = The requested epoch.
            * ``"parameters"`` = Any estimated parameters, if available.
            * ``"state_definition"`` : dict = The state definition used.
            * ``"propagator_settings"`` : dict =  The settings used for propagation, if available.
        """
        # Determine the origin and reference frame to use
        origin = origin_input if origin_input else self.origin
        ref_frame = ref_frame_input if ref_frame_input else self.ref_frame
        body = self.body

        # Resolve the sidecars: explicit override → this trajectory's own SPK,
        # which stands for the sidecars written beside it and then for its
        # folder → global fallback
        if folder_path_override is None and self._spk_path is not None:
            folder_path_override = os.path.abspath(self._spk_path)

        # Check if the epoch is in TDB format
        if epoch_input.system != "TDB":
            epoch_input = epoch_input.to(sys="TDB")

        # Get state from SPICE (raw ET seconds, rep-independent). get_state
        # only ever queries a single epoch, so always reduce to a scalar —
        # SpiceManager.get_state's _validate_epoch_input rejects a list.
        et_vals = np.asarray(epoch_input.et, dtype=float)
        epoch_to_spcmngr = float(np.atleast_1d(et_vals)[0])

        # Sequence-built trajectory queried without a leg index: the
        # backwards-compatibility anchors (body/origin/frame) are per-sample
        # arrays, not scalars, so resolve the leg covering the requested epoch
        # and use its metadata instead.
        if (
            idx_leg_input is None
            and getattr(self, "_legs", None) is not None
            and isinstance(body, np.ndarray)
        ):
            t_req = float(np.atleast_1d(et_vals)[0])
            leg_bounds = [
                (
                    float(np.asarray(leg["epoch_leg"].et, dtype=float)[0]),
                    float(np.asarray(leg["epoch_leg"].et, dtype=float)[-1]),
                )
                for leg in self._legs
            ]
            for i, (t0_leg, tf_leg) in enumerate(leg_bounds):
                if t0_leg - 1e-6 <= t_req <= tf_leg + 1e-6:
                    idx_leg_input = i
                    break
            else:
                # epoch outside every leg span -> use the leg with the
                # closest boundary
                idx_leg_input = int(
                    np.argmin(
                        [
                            min(abs(t0_leg - t_req), abs(tf_leg - t_req))
                            for t0_leg, tf_leg in leg_bounds
                        ]
                    )
                )

        # If sequence-based and leg index provided, pick per-leg metadata
        if (
            idx_leg_input is not None
            and hasattr(self, "_legs")
            and self._legs is not None
        ):
            leg = self._legs[idx_leg_input]
            body = leg.get("body", body)
            origin = leg.get("origin", origin)
            ref_frame = leg.get("frame", ref_frame)

        state = SpiceManager.get_state(body.name, epoch_to_spcmngr, ref_frame, origin)

        # Get parameters if available
        parameters_output = None
        if self.parameters is not None:
            if idx_leg_input is None:
                # Case: No specific leg input, retrieve general parameters
                parameters_output = SpiceManager.get_parameters(
                    body.name,
                    epoch_to_spcmngr,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )
            elif (
                self.parameters[idx_leg_input] is not None
                and isinstance(self.parameters[idx_leg_input][0][1], ArrayWFrame)
                and not np.isnan(
                    self.parameters[idx_leg_input][0][1].quantity.values
                ).any()
            ):
                # Case: Specific leg input with valid parameters
                parameters_output = SpiceManager.get_parameters(
                    body.name,
                    epoch_to_spcmngr,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )

        # Get settings if available
        settings_output = None
        if self._propagator_settings_dict is not None:
            if idx_leg_input is None:
                # Case: No specific leg input, retrieve general settings
                settings_output = SpiceManager.get_propagator_settings(
                    body.name,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )
            elif (
                idx_leg_input in self._propagator_settings_dict
                and self._propagator_settings_dict[idx_leg_input] is not None
            ):
                # Case: Specific leg input with valid settings
                settings_output = SpiceManager.get_propagator_settings(
                    body.name,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )

        # Extract position and velocity
        position_output, velocity_output = state[0:3], state[3:6]

        # Extract state definition
        if idx_leg_input is None:
            state_def_output = self.state_definition
        else:
            idx_leg = 0 if idx_leg_input == 0 else 2 * idx_leg_input
            state_def_output = self.sequence_definition.models[idx_leg].state_definition

        # Return state as a dictionary
        return {
            "position": position_output,
            "velocity": velocity_output,
            "epoch": epoch_input,
            "ref_frame": ref_frame,
            "origin": origin,
            "body": body.name,
            "parameters": parameters_output,
            "state_definition": state_def_output,
            "propagator_settings": settings_output,
        }

    def get_STM_JSON(
        self,
        epoch_input,
        origin_input=None,
        ref_frame_input=None,
        idx_leg_input=None,
        folder_path_override=None,
    ) -> dict:
        """
        Retrieves the state transition matrix (STM) at a specified epoch from JSON file.

        Queries SpiceManager to obtain the STM corresponding to a given epoch, considering
        optional reference frame and origin transformations.

        Parameters
        ----------
        epoch_input : EpochArray
            The epoch at which to retrieve the STM.

        origin_input : str, optional
            Custom origin for the STM retrieval. Defaults to the trajectory's origin.

        ref_frame_input : str, optional
            Custom reference frame for the STM retrieval. Default is the trajectory's reference frame.

        idx_leg_input : int, optional
            The index of the trajectory leg to retrieve if the trajectory is sequence-based.
            Defaults to ``None``.

        folder_path_override : str or os.PathLike or iterable, optional
            Where to look for the STM and propagator-settings sidecars: an SPK,
            a file, a folder, or an iterable of those — see
            :meth:`SpiceManager.sidecar_files`. Defaults to ``None``, which
            uses this trajectory's own SPK.

        Returns
        -------
        stm : dict
            The queried STM, containing:

            * ``"STM"`` : numpy.ndarray = The state transition matrix.
            * ``"epoch"`` : EpochArray = The requested epoch.
            * ``"state_definition"`` : dict = The state definition used.
            * ``"propagator_settings"`` : dict = The settings used for propagation, if available.
        """
        # Determine the origin and reference frame to use
        origin = origin_input if origin_input else self.origin
        ref_frame = ref_frame_input if ref_frame_input else self.ref_frame

        # Resolve the sidecars: explicit override → this trajectory's own SPK,
        # which stands for the sidecars written beside it and then for its
        # folder → global fallback
        if folder_path_override is None and self._spk_path is not None:
            folder_path_override = os.path.abspath(self._spk_path)

        # Get state transition matrices if available
        stm_output = None
        if self.STMs is not None:
            # SpiceManager.get_STMs takes a scalar TDB epoch, not a list.
            epoch_et = float(np.atleast_1d(epoch_input.times.values)[0])
            if idx_leg_input is None:
                # Case: No specific leg input, retrieve general STM
                stm_output = SpiceManager.get_STMs(
                    self.body.name, epoch_et, ref_frame, origin,
                    folder_path_override=folder_path_override,
                )
            elif idx_leg_input in self.STMs and self.STMs[idx_leg_input] is not None:
                # Case: Specific leg input with valid STM
                stm_output = SpiceManager.get_STMs(
                    self.body.name, epoch_et, ref_frame, origin,
                    folder_path_override=folder_path_override,
                )

        # Get settings if available
        settings_output = None
        if self._propagator_settings_dict is not None:
            if idx_leg_input is None:
                # Case: No specific leg input, retrieve general settings
                settings_output = SpiceManager.get_propagator_settings(
                    self.body.name,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )
            elif (
                idx_leg_input in self._propagator_settings_dict
                and self._propagator_settings_dict[idx_leg_input] is not None
            ):
                # Case: Specific leg input with valid settings
                settings_output = SpiceManager.get_propagator_settings(
                    self.body.name,
                    ref_frame,
                    origin,
                    folder_path_override=folder_path_override,
                )

        # Extract state definition
        if idx_leg_input is None:
            state_def_output = self.state_definition
        else:
            idx_leg = 0 if idx_leg_input == 0 else 2 * idx_leg_input
            state_def_output = self.sequence_definition.models[idx_leg].state_definition

        # Return state as a dictionary
        return {
            "STM": stm_output,
            "epoch": epoch_input,
            "ref_frame": ref_frame,
            "origin": origin,
            "body": self.body.name,
            "state_definition": state_def_output,
            "propagator_settings": settings_output,
        }

    @staticmethod
    def _body_tagged_path(path: str, body) -> str:
        """
        ``<base>_<body>.bsp`` — a path or sidecar stem unique to one body.

        Used both to split a multi-body trajectory into one SPK per body and,
        when they do share a file, to keep their parameters sidecars apart:
        every body writes ``_parameters.json``, so a shared stem means the last
        body silently overwrites the others. Readers match a sidecar on the
        body id it records, not on the file name, so the tag is free.

        Parameters
        ----------
        path : str
            The base path, ending in ``.bsp``.
        body : Body
            The body to tag with.

        Returns
        -------
        str
            The tagged path.
        """
        base, ext = os.path.splitext(path)
        tag = str(getattr(body, "name", body) or "body")
        tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)
        return f"{base}_{tag}{ext}"

    @staticmethod
    def _stm_interp_rel_error(times, flats, i, alpha):
        """
        Relative error of linearly interpolating the STM inside bracket ``i``.

        The error of linear interpolation is set by **curvature**, not slope:

        .. math::

            e(t) = -\\frac{\\alpha(1-\\alpha)}{2}\\,h^{2}\\,\\Phi''(\\xi),
            \\qquad t = t_{i-1} + \\alpha h,

        so the estimate is built from the second divided difference over the
        bracket and one neighbouring node, which is
        :math:`\\Phi[t_a,t_b,t_c] = \\Phi''(\\xi)/2`, giving

        .. math::

            \\|e\\| \\le \\alpha(1-\\alpha)\\,h^{2}\\,
                      \\|\\Phi[t_a,t_b,t_c]\\|.

        The result is reported **relative** to the matrix being interpolated. An
        STM mixes units across its blocks and grows with the arc, so the absolute
        Frobenius norm of a first difference reports the size of the matrix rather
        than the quality of the interpolation.

        Parameters
        ----------
        times : array_like
            Sorted STM epochs.
        flats : sequence
            Flattened STMs, one per epoch.
        i : int
            Index of the right-hand bracket node, so the bracket is
            ``[times[i-1], times[i]]``.
        alpha : float
            Interpolation weight in ``[0, 1]``, zero at ``times[i-1]``.

        Returns
        -------
        float or None
            Estimated relative error, or ``None`` when the grid holds no third
            node to take a curvature from — with two points, linear interpolation
            is all the data supports and there is nothing to check it against.
        """
        n = len(times)

        # Third node for the curvature: prefer the one before the bracket, so a
        # lookup near the end of the grid is still covered.
        if i - 2 >= 0:
            a, b, c = i - 2, i - 1, i
        elif i + 1 < n:
            a, b, c = i - 1, i, i + 1
        else:
            return None

        ta, tb, tc = float(times[a]), float(times[b]), float(times[c])
        if tb == ta or tc == tb or tc == ta:
            return None

        fa = _np.asarray(flats[a], dtype=float).ravel()
        fb = _np.asarray(flats[b], dtype=float).ravel()
        fc = _np.asarray(flats[c], dtype=float).ravel()

        # Newton second divided difference: Phi[ta, tb, tc] = Phi''(xi) / 2.
        curvature = ((fc - fb) / (tc - tb) - (fb - fa) / (tb - ta)) / (tc - ta)

        h = float(times[i]) - float(times[i - 1])
        absolute = alpha * (1.0 - alpha) * h * h * float(_np.linalg.norm(curvature))

        f0 = _np.asarray(flats[i - 1], dtype=float).ravel()
        f1 = _np.asarray(flats[i], dtype=float).ravel()
        scale = float(_np.linalg.norm((1.0 - alpha) * f0 + alpha * f1))
        if scale <= 0.0:
            return absolute
        return absolute / scale

    def get_STM_sequence(
        self,
        epoch: float,
        idx: int = None,
        idx_leg: int = None,
        tol_time: float = 1e-9,
    ) -> np.ndarray:
        """
        Retrieve/interpolate the STM at epoch for BOTH:

        - non-sequence: ``self._STMs`` is (N, n^2) and ``self._STMs_timestamp`` is (N,)
        - sequence: ``self._STMs`` is a list of arrays, one per leg (or per segment block)

        Behavior (sequence):

        - if idx_leg is given: interpolate within that leg
        - else: auto-detect leg from ``self._legs[*]["epoch_leg"]`` if available,
          otherwise fall back to using global timestamps if provided.

        Notes
        -----
        - Works with flat STMs stored as length n^2 vectors.
        - Uses linear interpolation in coefficient space.
        - Handles exact hits within tol_time.
        - Interpolated results are checked against
          :attr:`max_interp_error`, this trajectory's relative tolerance.
        """
        max_interp_error = self.max_interp_error

        def _reshape_stm(flat_stm):
            arr = _np.asarray(flat_stm, dtype=float).ravel()
            dim = int(_np.sqrt(arr.size))
            if dim * dim != arr.size:
                raise ValueError(f"STM size {arr.size} is not a perfect square")
            return arr.reshape(dim, dim)

        def _interp_flat(times, flats, t):
            times = _np.asarray(times, dtype=float).ravel()
            if times.size < 1:
                raise ValueError("Empty STM time grid.")
            # enforce monotonic nondecreasing
            if _np.any(_np.diff(times) < 0):
                raise ValueError(
                    "STM timestamps are not sorted (must be nondecreasing)."
                )

            i = bisect.bisect_left(times.tolist(), float(t))

            # clamp / boundary handling
            if i <= 0:
                if abs(times[0] - t) <= tol_time:
                    return _np.asarray(flats[0], dtype=float).ravel()
                raise ValueError(
                    f"Epoch {t} before first STM timestamp {times[0]} (no extrapolation)."
                )
            if i >= times.size:
                if abs(times[-1] - t) <= tol_time:
                    return _np.asarray(flats[-1], dtype=float).ravel()
                raise ValueError(
                    f"Epoch {t} beyond last STM timestamp {times[-1]} (no extrapolation)."
                )

            # exact hit within tol
            if abs(times[i] - t) <= tol_time:
                return _np.asarray(flats[i], dtype=float).ravel()
            if abs(times[i - 1] - t) <= tol_time:
                return _np.asarray(flats[i - 1], dtype=float).ravel()

            t0, t1 = times[i - 1], times[i]
            f0 = _np.asarray(flats[i - 1], dtype=float).ravel()
            f1 = _np.asarray(flats[i], dtype=float).ravel()

            dt = t1 - t0
            if dt == 0.0:
                raise ValueError(
                    f"Duplicate STM timestamps at {t0}; cannot interpolate."
                )
            a = (t - t0) / dt
            f = (1.0 - a) * f0 + a * f1

            if max_interp_error is not None:
                err_est = self._stm_interp_rel_error(times, flats, i, a)
                if err_est is not None and err_est > max_interp_error:
                    raise ValueError(
                        f"Estimated relative interpolation error {err_est:.3e} "
                        f"exceeds threshold {max_interp_error:.3e} at epoch {t}. "
                        f"Brackets: [{t0}, {t1}] (dt={dt:.6e}), alpha={a:.6f}. "
                        "The error scales as dt^2: halving the STM grid step "
                        "cuts it by four. Set Trajectory.max_interp_error to "
                        "accept a looser fit, or None to skip the check."
                    )
            return f

        # -------------------------
        # Case A: non-sequence (single block)
        # -------------------------
        if not isinstance(self._STMs, list):
            if self._STMs_timestamp is None:
                if idx is None:
                    raise ValueError(
                        "No STM timestamps available and idx not provided."
                    )
                return _reshape_stm(self._STMs[idx])
            flat = _interp_flat(self._STMs_timestamp, self._STMs, epoch)
            return _reshape_stm(flat)

        # -------------------------
        # Case B: sequence (list of blocks)
        # -------------------------
        stms_blocks = self._STMs

        # Helper: get per-leg time grid
        def _leg_time_grid(j):
            # Prefer explicit per-leg timestamps if you store them as list-of-arrays
            if isinstance(self._STMs_timestamp, list):
                return _np.asarray(self._STMs_timestamp[j], dtype=float).ravel()
            # Else: derive from epoch_leg if available
            if (
                hasattr(self, "_legs")
                and self._legs is not None
                and j < len(self._legs)
            ):
                ep = self._legs[j].get("epoch_leg", None)
                if ep is not None:
                    return _np.asarray(ep.times.values, dtype=float).ravel()
            # Else: no per-leg grid; cannot safely interpolate
            return None

        # Helper: auto-detect leg from epoch using epoch_leg spans
        def _detect_leg_from_epoch(t):
            if hasattr(self, "_legs") and self._legs is not None:
                for j, leg in enumerate(self._legs):
                    ep = leg.get("epoch_leg", None)
                    if ep is None:
                        continue
                    tt = _np.asarray(ep.times.values, dtype=float).ravel()
                    if tt.size == 0:
                        continue
                    if (t >= tt[0] - tol_time) and (t <= tt[-1] + tol_time):
                        return j
            return None

        # pick leg
        if idx_leg is None:
            idx_leg = _detect_leg_from_epoch(epoch)

        # If we still don't know the leg:
        # - if you DO have a global timestamp vector for the whole sequence, use it (rare in your current impl)
        if idx_leg is None:
            if self._STMs_timestamp is not None and not isinstance(
                self._STMs_timestamp, list
            ):
                # flatten blocks into one list
                flats = []
                for blk in stms_blocks:
                    flats.extend(list(blk))
                flat = _interp_flat(self._STMs_timestamp, flats, epoch)
                return _reshape_stm(flat)
            raise ValueError(
                "Sequence STM interpolation needs idx_leg OR leg epoch spans (self._legs[*]['epoch_leg']) "
                "OR per-leg timestamps (self._STMs_timestamp as list)."
            )

        if idx_leg < 0 or idx_leg >= len(stms_blocks):
            raise IndexError(
                f"idx_leg={idx_leg} out of range for {len(stms_blocks)} STM blocks."
            )

        # If caller wants direct indexing within leg:
        if idx is not None and (self._STMs_timestamp is None):
            return _reshape_stm(stms_blocks[idx_leg][idx])

        times_leg = _leg_time_grid(idx_leg)
        if times_leg is None:
            # as a last resort, if idx was provided
            if idx is not None:
                return _reshape_stm(stms_blocks[idx_leg][idx])
            raise ValueError(
                f"No timestamps/epoch grid for leg {idx_leg}. Provide timestamps per leg or pass idx."
            )

        flat = _interp_flat(times_leg, stms_blocks[idx_leg], epoch)
        return _reshape_stm(flat)

    def get_STM(
        self,
        epoch: float,
        idx: int = None,
    ) -> np.ndarray:
        """
        Retrieve the State Transition Matrix (STM) at a given epoch or by index.

        Uses binary search for efficient lookup and linear interpolation between
        stored STMs. An interpolated result is checked against
        :attr:`max_interp_error`, this trajectory's relative tolerance.

        Parameters
        ----------
        epoch : float
            The time at which to retrieve/interpolate the STM
        idx : int, optional
            Direct index into the STM array (used when timestamps unavailable)

        Returns
        -------
        np.ndarray
            The STM reshaped as a square matrix

        Raises
        ------
        ValueError
            If neither timestamps nor index provided, epoch not found, or
            interpolation error exceeds threshold
        """

        max_interp_error = self.max_interp_error

        # Helper function: Reshape flat STM array into square matrix
        def reshape_stm(flat_stm):
            """Convert 1D STM array to 2D square matrix."""
            arr = np.asarray(flat_stm)
            dim = int(np.sqrt(arr.size))

            # Validate that array size is a perfect square
            if dim * dim != arr.size:
                raise ValueError(f"STM size {arr.size} is not a perfect square")

            return arr.reshape(dim, dim)

        # Case 1: No timestamps available - use direct indexing
        if self._STMs_timestamp is None:
            if idx is None:
                raise ValueError("Neither STM timestamps nor idx provided.")
            return reshape_stm(self._STMs[idx])

        # Case 2: Timestamps available - find/interpolate STM at requested epoch
        # Convert timestamps to numpy array for efficient searching
        times = np.asarray(self._STMs_timestamp).ravel()

        # Find insertion point: times[i-1] < epoch <= times[i]
        i = bisect.bisect_left(times, epoch)

        # Handle edge cases
        if i == 0:
            # Epoch is before first timestamp - check if it's an exact match
            if times[0] == epoch:
                return reshape_stm(self._STMs[0])
            else:
                raise ValueError(
                    f"Epoch {epoch} is before the earliest STM timestamp {times[0]}. "
                    "Cannot extrapolate backwards."
                )
        if i >= len(times):
            raise ValueError(
                f"Epoch {epoch} is beyond the latest STM timestamp {times[-1]}. "
                "Cannot extrapolate forwards."
            )

        # Check for exact match at the found position
        if times[i] == epoch:
            return reshape_stm(self._STMs[i])

        # Linear interpolation between two surrounding STMs
        # Get bracketing times and STMs
        t0, t1 = times[i - 1], times[i]
        STM0, STM1 = self._STMs[i - 1], self._STMs[i]
        # Calculate interpolation weight (0 at t0, 1 at t1)
        dt = t1 - t0
        alpha = (epoch - t0) / dt

        # Linearly interpolate between the two STMs
        interp_STM_flat = (1.0 - alpha) * STM0 + alpha * STM1

        # Interpolation error estimation and validation
        if max_interp_error is not None:
            err_est = self._stm_interp_rel_error(times, self._STMs, i, alpha)

            # Check if error exceeds threshold
            if err_est is not None and err_est > max_interp_error:
                raise ValueError(
                    f"Estimated relative interpolation error {err_est:.3e} "
                    f"exceeds threshold {max_interp_error:.3e} at epoch {epoch}.\n"
                    f"Bracketing times: [{t0}, {t1}] (dt={dt:.6e})\n"
                    f"Interpolation weight alpha={alpha:.6f}\n"
                    "The error scales as dt^2, so halving the STM grid step "
                    "cuts it by four. Set Trajectory.max_interp_error to accept "
                    "a looser fit, or None to skip the check."
                )

        # Reshape and return the interpolated STM
        interp_STM = reshape_stm(interp_STM_flat)

        return interp_STM

    def get_offset_position(
        self, epoch_input, origin_input=None, ref_frame_input=None
    ) -> ArrayWUnits:
        """
        Computes the trajectory position with an applied constant offset.

        If a reference frame transformation is required, the offset is rotated accordingly.
        The method also accounts for origin shifts when necessary.

        Parameters
        ----------
        epoch_input : EpochArray
            The epoch at which to compute the offset position.

        origin_input : str, optional
            Custom origin for position computation. Defaults to the trajectory's origin.

        ref_frame_input : str, optional
            Custom reference frame for position computation. Defaults to the trajectory's reference frame.

        Returns
        -------
        offset_pos : ArrayWUnits
            The computed position including the constant offset. Expressed in
            units of kilometers.
        """
        # Use default reference frame and origin if not specified
        origin = origin_input or self.origin
        ref_frame = ref_frame_input or self.ref_frame

        # Return the constant offset directly if no transformation is needed
        if not ref_frame_input and not origin_input:
            return self._offset

        # Apply rotation if the reference frame is different
        if ref_frame_input:
            # get_xfrm/get_pos take a scalar TDB epoch, not an EpochArray.
            epoch_et = float(np.atleast_1d(epoch_input.to(sys="TDB").times.values)[0])
            # get_xfrm needs plain frame-name strings, not Frame objects.
            ref_frame_from = self.ref_frame.name if isinstance(self.ref_frame, Frame) else self.ref_frame
            ref_frame_to = ref_frame_input.name if isinstance(ref_frame_input, Frame) else ref_frame_input

            dcm = SpiceManager.get_xfrm(
                ref_frame_from, ref_frame_to, epoch_et
            )
            offset_rotated = self._offset.rotate_dcm(ArrayWUnits(dcm, None))

            # Return rotated offset if no origin transformation is required
            if not origin_input:
                return offset_rotated

            # Compute the position of the origin relative to the query origin
            pos_origin = SpiceManager.get_pos(
                self.origin,
                epoch_et,
                ref_frame_input,
                origin,
                aberration_correction="None",
            )
            return pos_origin + offset_rotated

    def set_offset(self, offset: ArrayWUnits) -> None:
        """
        Sets the constant position offset for the trajectory.

        Parameters
        ----------
        offset : ArrayWUnits
            The constant offset in position. Expressed in units of kilometers.

        Returns
        -------
        None
        """
        if offset.units.dimensions.name != "Length":
            raise Exception(
                "The physical type of the offset ArrayWUnits must be 'Length'."
            )
        self._offset = offset

    def add_STMs(self, STMs: list, timestamps: List = None):
        """
        Associates state transition matrices (STMs) with the trajectory.

        Parameters
        ----------
        STMs : list or ArrayWFrame
            A list of state transition matrices or a single ArrayWFrame object.
        timestamps : EpochArray, optional
            Epoch of each STM. Defaults to ``None``: a single ArrayWFrame is
            aligned 1:1 with :attr:`epoch`, with a warning.
        """
        if isinstance(STMs, list):
            self._STMs = [stm.quantity.values for stm in STMs]
            self._STM_units = [stm.quantity.units for stm in STMs]
            self._STM_frames = [stm.frame for stm in STMs]
        else:
            self._STMs = STMs.quantity.values
            self._STM_units = STMs.quantity.units
            self._STM_frames = STMs.frame

        if isinstance(self._STMs, list):
            n_stms = sum([arr.shape[0] for arr in self._STMs])
        else:
            n_stms = len(self._STMs)

        # Timestamps handling
        if timestamps is not None:
            n_times = len(timestamps.times.values)
            if n_times != n_stms:
                raise ValueError(
                    f"Number of STM timestamps ({n_times}) does not match number of STMs ({n_stms})."
                )
            self._STMs_timestamp = timestamps.times.values

        elif not isinstance(self._STMs, list):  # NOTE: means no sequence!
            # fallback to trajectory epochs
            n_epochs = len(self._epoch.times.values)
            if n_epochs != n_stms:
                raise ValueError(
                    f"No STM timestamps provided and trajectory has {n_epochs} epochs "
                    f"but {n_stms} STMs were given: cannot align."
                )
            warnings.warn(
                "No STM timestamps provided: falling back to trajectory epochs. "
                "Ensure STMs are aligned 1:1 with `self.epoch`.",
                UserWarning,
            )
            self._STMs_timestamp = self._epoch.times.values
