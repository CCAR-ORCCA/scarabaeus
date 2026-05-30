# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from os import name
import re
import numpy as np
from typing import Union, Optional, Dict, List
from scarabaeus import ArrayWUnits, Body, Spacecraft, GroundStation


# -----------------------#
#    Class Definition   #
# -----------------------#
class StateDefinition:
    """ Fluent builder for constructing state vector definitions compatible with StateArray.

    This class provides a clean, type-safe interface for building state vectors by replacing
    raw tuple lists with method chaining. It validates components against allowed patterns
    and enforces structural constraints (e.g., velocity must follow position).

    Parameters
    ----------
    None

    See Also
    --------
    scarabaeus.StateArray : The class that consumes StateDefinition objects.

    Notes
    -----
    **Ordering rules**

    * Dynamic components must appear **before** static components.
    * Estimated components must appear **before** considered components.
    * Each ``.velocity()`` call must immediately follow its paired ``.position()`` call.
    * Each of ``dv_man`` / ``dv_man1`` / ``dv_man2`` may appear at most once per state.
    * ``dRA[N]`` and ``dDEC[N]`` finite-burn angle pairs must both be present for the same
      owner and numeric suffix.
    * ``"considered"`` estimation mode is not yet fully implemented.

    **Estimation and dynamics modes**

    Every component accepts two keyword arguments:

    * ``estimation`` — ``"estimated"`` *(default)*: corrected by the filter;
      ``"considered"``: contributes to covariance only, not corrected.
    * ``dynamics`` — ``"dynamic"`` *(default)*: propagated by the force model;
      ``"static"``: held constant across the arc, only corrected by the filter.

    **Complete table of allowed state components**

    All parameter names are case-sensitive. ``[N]`` denotes an optional non-negative integer
    suffix (e.g. ``dRA``, ``dRA1``, ``dRA2``). ``[tag]`` denotes a free-form identifier
    string such as a ground-station or camera name.

    .. list-table::
       :header-rows: 1
       :widths: 32 18 5 20 12 45

       * - Name pattern
         - Category
         - Dim
         - Physical type
         - Unit
         - Description
       * - ``position``
         - Kinematics
         - 3
         - Length
         - km
         - Cartesian position. Added via ``.position()``.
       * - ``velocity``
         - Kinematics
         - 3
         - Velocity
         - km/s
         - Cartesian velocity. Added via ``.velocity()``.
       * - ``eta_srp``
         - SRP
         - 1
         - Dimensionless
         - --
         - Cannonball SRP scale factor. Nominal value 1.0.
       * - ``k_thrust[N]``
         - Finite burn
         - 1
         - Dimensionless
         - --
         - Thrust-magnitude scale factor for burn arc ``N``.
       * - ``dRA[N]``
         - Finite burn
         - 1
         - Angle
         - rad
         - Right-ascension pointing correction for burn ``N``.
       * - ``dDEC[N]``
         - Finite burn
         - 1
         - Angle
         - rad
         - Declination pointing correction for burn ``N``.
       * - ``dv_man[N]``
         - Impulsive maneuver
         - 3
         - Velocity
         - km/s
         - Impulsive delta-v vector. Suffixes: none, ``1``, ``2``.
       * - ``gs_delta_location_ECEF_[tag]``
         - Ground station
         - 3
         - Length
         - km
         - ECEF position correction for station ``[tag]``.
       * - ``range_bias_[tag]``
         - Radiometric bias
         - dyn
         - Length
         - km / m
         - Additive range bias for station ``[tag]``.
       * - ``rangerate_bias_[tag]``
         - Radiometric bias
         - dyn
         - Velocity
         - km/s
         - Additive range-rate bias for station ``[tag]``.
       * - ``range_bias_RU_[tag]``
         - Radiometric bias
         - 1
         - Dimensionless
         - RU
         - Range bias in range units for station ``[tag]``.
       * - ``rangerate_bias_Hz_[tag]``
         - Radiometric bias
         - 1
         - per Time
         - Hz
         - Doppler bias in Hz for station ``[tag]``.
       * - ``angular_bias_[tag]``
         - Optical navigation
         - dyn
         - Dimensionless
         - pix / rad
         - Additive RA/DEC angular bias for camera ``[tag]``.
       * - ``centroid_bias_[tag]``
         - Optical navigation
         - dyn
         - Dimensionless
         - pixels
         - Pixel centroid bias for camera ``[tag]``.
       * - ``cam_focal_length_[tag]``
         - Optical navigation
         - 2
         - Dimensionless
         - pix/rad
         - Focal length (fx, fy) for camera ``[tag]``.
       * - ``cam_pp_offset_[tag]``
         - Optical navigation
         - 2
         - Dimensionless
         - pixels
         - Principal-point offset (cx, cy) for camera ``[tag]``.
       * - ``gm``
         - Gravity
         - 1
         - Gravitational parameter
         - km^3/s^2
         - GM of any body in the force model.
       * - ``cnm_N_M``
         - Gravity (harmonics)
         - 1
         - Dimensionless
         - --
         - Cosine spherical harmonic coefficient C_nm, degree N order M.
       * - ``snm_N_M``
         - Gravity (harmonics)
         - 1
         - Dimensionless
         - --
         - Sine spherical harmonic coefficient S_nm, degree N order M.
       * - ``pole_ra``
         - Body orientation
         - 1
         - Angle
         - rad
         - Right ascension of the body's north pole.
       * - ``pole_dec``
         - Body orientation
         - 1
         - Angle
         - rad
         - Declination of the body's north pole.
       * - ``pole_pm``
         - Body orientation
         - 1
         - Dimensionless
         - deg/day
         - Prime-meridian rotation rate (IAU convention).
       * - ``A2_yark``
         - Small-body
         - 1
         - Length / Time^2
         - AU/day^2
         - Yarkovsky transverse acceleration coefficient A2.
       * - ``a_fogm``
         - Empirical accel.
         - 3
         - Acceleration
         - km/s^2
         - FOGM empirical acceleration vector.
       * - ``beta_fogm``
         - Empirical accel.
         - 3
         - per Time
         - 1/s
         - FOGM inverse correlation time beta = 1/tau per axis.
       * - ``a_pfogm``
         - Empirical accel.
         - 3*n
         - Acceleration
         - km/s^2
         - Piecewise FOGM accel.; dim = 3 x n_batches.
       * - ``beta_pfogm``
         - Empirical accel.
         - 3*n
         - per Time
         - 1/s
         - Piecewise FOGM correlation time; same sizing as ``a_pfogm``.

    *dyn* in the Dim column means the dimension is inferred from the supplied initial value.
    For ``a_pfogm`` / ``beta_pfogm`` it must be a multiple of 3.

    Examples
    --------
    Basic state vector with position, velocity, and a static SRP parameter:

    .. code-block:: python

        import scarabaeus as scb

        state_def = (StateDefinition()
            .position(Orbiter, pos_0)
            .velocity(Orbiter, vel_0)
            .param("eta_srp", Orbiter, par1_0, dynamics="static")
        )

        state_vector_0 = scb.StateArray(
            epoch=epoch_array[0],
            origin=origin,
            state=state_def
        )

    Multi-spacecraft configuration:

    .. code-block:: python

        state_def = (StateDefinition()
            .position(Orbiter1, pos_1)
            .velocity(Orbiter1, vel_1)
            .param("eta_srp", Orbiter1, srp_1, dynamics="static")
            .position(Orbiter2, pos_2)
            .velocity(Orbiter2, vel_2)
            .param("eta_srp", Orbiter2, srp_2, dynamics="static")
        )

    State with spherical harmonics and a range bias:

    .. code-block:: python

        state_def = (StateDefinition()
            .position(Orbiter, pos_0)
            .velocity(Orbiter, vel_0)
            .param("cnm_2_0", Vesta, c20_0, dynamics="static")
            .param("cnm_2_2", Vesta, c22_0, dynamics="static")
            .param("snm_2_1", Vesta, s21_0, dynamics="static")
            .param("range_bias_DSS14", DSS14, rb_0, dynamics="static")
        )

    Accessing components directly:

    .. code-block:: python

        state_def = StateDefinition().position(Orbiter, pos_0).velocity(Orbiter, vel_0)
        components = state_def.components  # list of (name, dim, est, dyn, owner, value)
        print(f"State has {len(state_def)} components")
    """

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(self):
        self._components = []
        self._position_count = 0
        self._velocity_count = 0
        self._dv_maneuvers = []
        self._finite_burn_pairs = (
            {}
        )  # key: (owner_id, suffix) -> {"dRA": bool, "dDEC": bool}
        self._last_component_was_position = False
        self._component_registry = {}  # Track (name, owner) pairs to prevent duplicates

        # Ordering / phase flags
        self._seen_static = False  # once True, no more dynamic allowed
        self._seen_considered = False  # once True, no more estimated allowed

    # endregion Constructor #
    # -----------------------#

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def components(self) -> List[tuple]:
        """
        The internal list of state components as tuples.

        Returns
        -------
        list of tuple
            List of component tuples in format:
            (name, dimension, estimation, dynamics, owner, value)
        """
        return self._components

    # endregion Properties #
    # ----------------------#

    # ----------------------#
    # region    Methods     #
    # ----------------------#
    def position(self, owner, value, estimation="estimated", dynamics="dynamic"):
        """
        Add a position component to the state vector.

        Parameters
        ----------
        owner : Body, Spacecraft, or GroundStation
            The body/spacecraft associated with this position.
        value : ArrayWUnits
            Position vector with units (must be 3D with Length dimensions).
        estimation : str, optional
            Estimation mode: "estimated" or "considered". Defaults to "estimated".
            Note: "considered" is not yet implemented.
        dynamics : str, optional
            Dynamics mode: "dynamic" or "static". Defaults to "dynamic".

        Returns
        -------
        StateDefinition
            Self for method chaining.

        Raises
        ------
        ValueError
            If component validation fails (wrong dimension, physical type, etc.)
            or if duplicate component for the same owner exists.
        NotImplementedError
            If estimation="considered" is specified.
        """
        est, dyn = self._validate_modes(estimation, dynamics)
        self._enforce_global_order(est, dyn)
        self._validate_component("position", 3, value)
        self._check_duplicate_component("position", owner)

        self._components.append(("position", 3, est, dyn, owner, value))
        self._register_component("position", owner)
        self._position_count += 1
        self._last_component_was_position = True
        return self

    def velocity(self, owner, value, estimation="estimated", dynamics="dynamic"):
        """
        Add a velocity component to the state vector.

        The velocity component must immediately follow a position component.

        Parameters
        ----------
        owner : Body, Spacecraft, or GroundStation
            The body/spacecraft associated with this velocity.
        value : ArrayWUnits
            Velocity vector with units (must be 3D with Velocity dimensions).
        estimation : str, optional
            Estimation mode: "estimated" or "considered". Defaults to "estimated".
            Note: "considered" is not yet implemented.
        dynamics : str, optional
            Dynamics mode: "dynamic" or "static". Defaults to "dynamic".

        Returns
        -------
        StateDefinition
            Self for method chaining.

        Raises
        ------
        ValueError
            If velocity is not added immediately after position, if validation fails,
            or if duplicate component for the same owner exists.
        NotImplementedError
            If estimation="considered" is specified.
        """
        # Check that position was added immediately before
        if not self._last_component_was_position:
            raise ValueError(
                "The 'velocity' component must appear immediately after a 'position' component."
            )

        est, dyn = self._validate_modes(estimation, dynamics)
        self._enforce_global_order(est, dyn)
        self._validate_component("velocity", 3, value)
        self._check_duplicate_component("velocity", owner)

        self._components.append(("velocity", 3, est, dyn, owner, value))
        self._register_component("velocity", owner)
        self._velocity_count += 1
        self._last_component_was_position = False
        return self

    def param(
        self,
        name,
        owner,
        value,
        dimension=None,
        estimation="estimated",
        dynamics="dynamic",
    ):
        """
        Add a generic parameter to the state vector.

        This method can be used for any allowed state component beyond position and velocity,
        such as solar radiation pressure coefficients, measurement biases, or delta-v maneuvers.

        Parameters
        ----------
        name : str
            Parameter name (e.g., "eta_srp", "dv_man", "range_bias").
            Must match an allowed pattern in the state dictionary.
        owner : Body, Spacecraft, GroundStation, or None
            The body/spacecraft associated with this parameter, or None for global parameters.
        value : ArrayWUnits
            Parameter value with units.
        dimension : int, optional
            Parameter dimension/size. If None, inferred from value length. Defaults to None.
        estimation : str, optional
            Estimation mode: "estimated" or "considered". Defaults to "estimated".
            Note: "considered" is not yet implemented.
        dynamics : str, optional
            Dynamics mode: "dynamic" or "static". Defaults to "dynamic".

        Returns
        -------
        StateDefinition
            Self for method chaining.

        Raises
        ------
        ValueError
            If parameter name is not allowed, dimension/physical type mismatch,
            multiple delta-v maneuvers are specified, or duplicate component exists.
        NotImplementedError
            If estimation="considered" is specified.
        """
        dim = dimension if dimension is not None else value.shape[0]
        est, dyn = self._validate_modes(estimation, dynamics)
        self._enforce_global_order(est, dyn)

        # Track delta-v maneuvers: allow dv_man + dv_man1 + dv_man2 to coexist,
        # but prevent duplicates of the same name.
        if name in ("dv_man", "dv_man1", "dv_man2"):
            if name in self._dv_maneuvers:
                raise ValueError(
                    f"Duplicate maneuver component '{name}' detected. "
                    f"Already present: {self._dv_maneuvers}. "
                    f"Each of dv_man/dv_man1/dv_man2 may appear at most once."
                )
            self._dv_maneuvers.append(name)

        # Track finite-burn angular bias pairs (dRA/dDEC with suffix),
        # but do not validate completeness yet because components are
        # added one at a time in from_components()/extend_from_components().
        m = re.match(r"^(dRA|dDEC)(\d*)$", name)
        if m:
            base = m.group(1)  # dRA or dDEC
            suffix = m.group(2)  # "", "1", "2", ...
            owner_id = id(owner) if owner is not None else None
            key = (owner_id, suffix)

            if key not in self._finite_burn_pairs:
                self._finite_burn_pairs[key] = {"dRA": False, "dDEC": False}

            # Prevent duplicate of same component for same owner+suffix
            if self._finite_burn_pairs[key][base]:
                raise ValueError(
                    f"Duplicate component '{name}' detected for the same owner. "
                    f"dRA/dDEC with suffix '{suffix}' must appear at most once."
                )

            self._finite_burn_pairs[key][base] = True

        # Validate against allowed state dictionary
        self._validate_component(name, dim, value)
        self._check_duplicate_component(name, owner)

        self._components.append((name, dim, est, dyn, owner, value))
        self._register_component(name, owner)
        self._last_component_was_position = False

        return self

    @classmethod
    def empty_pv(
        cls,
        owner,
        frame=None,
        pos_units=None,
        vel_units=None,
        estimation: str = "estimated",
        dynamics: str = "dynamic",
    ) -> "StateDefinition":
        """
        Create a StateDefinition with NaN position and velocity for a given owner.

        Useful for non-initial legs where ICs are inherited from the previous leg.

        Parameters
        ----------
        owner : Spacecraft
            The body associated with the state.
        frame : Frame, optional
            Reference frame. If None, position/velocity are stored with no frame.
        pos_units : Units, optional
            Position units. Defaults to km.
        vel_units : Units, optional
            Velocity units. Defaults to km/sec.
        estimation : str, optional
            Defaults to ``"estimated"``.
        dynamics : str, optional
            Defaults to ``"dynamic"``.

        Returns
        -------
        StateDefinition
            A new StateDefinition with NaN position and velocity.

        Examples
        --------
        .. code-block:: python

            # Coast leg — just pos/vel
            leg_state = StateDefinition.nan_pv(Orbiter, frame=J2000)

            # Finite burn leg — pos/vel + burn params
            leg_state = (StateDefinition.nan_pv(Orbiter, frame=J2000)
                .param("k_thrust1", Orbiter, ArrayWFrame(0.9, None, None), dynamics="static")
                .param("dRA1",      Orbiter, ArrayWFrame(0.4, rad, J2000), dynamics="static")
                .param("dDEC1",     Orbiter, ArrayWFrame(0.3, rad, J2000), dynamics="static")
            )
        """
        from scarabaeus import ArrayWUnits, ArrayWFrame

        if pos_units is None:
            from scarabaeus import Units

            pos_units = Units.get_units(["km"])
        if vel_units is None:
            from scarabaeus import Units

            km, sec = Units.get_units(["km", "sec"])
            vel_units = km / sec

        pos_val = ArrayWFrame(ArrayWUnits(np.full(3, np.nan), pos_units), frame)
        vel_val = ArrayWFrame(ArrayWUnits(np.full(3, np.nan), vel_units), frame)

        return (
            cls()
            .position(owner, pos_val, estimation=estimation, dynamics=dynamics)
            .velocity(owner, vel_val, estimation=estimation, dynamics=dynamics)
        )

    # endregion Methods #
    # -------------------#

    # ----------------------#
    # region    Validation #
    # ----------------------#
    @classmethod
    def _generate_allowed_state_dictionary(cls) -> Dict[str, List]:
        """
        Generate a dictionary of components allowed in a StateArray.

        Returns
        -------
        dict
            Dictionary mapping component patterns to their specifications.
            Format: ``{name: [physicalType, size, regex_pattern]}``
        """
        state_allowed = {
            "position": ["Length", 3, re.compile(r"^(position)$")],
            "velocity": ["Velocity", 3, re.compile(r"^(velocity)$")],
            # SRP
            "eta_srp": ["Dimensionless", 1, re.compile(r"^(eta_srp)$")],
            # Finite burn estimable parameters
            # allows: k_thrust, k_thrust1, k_thrust2, ...
            "k_thrust": ["Dimensionless", 1, re.compile(r"^(k_thrust\d*)$")],
            # allows: dRA, dRA1, dRA2, ...
            "dRA": ["Angle", 1, re.compile(r"^(dRA\d*)$")],
            # allows: dDEC, dDEC1, dDEC2, ...
            "dDEC": ["Angle", 1, re.compile(r"^(dDEC\d*)$")],
            # Ground station / tracking
            "gs_delta_location_ECEF": [
                "Length",
                3,
                re.compile(r"^(gs_delta_location_ECEF)_(\w+)$"),
            ],
            "range_bias": [
                "Length",
                "dynamic",
                re.compile(r"^(range_bias)_(\w+)$"),
            ],
            "rangerate_bias": [
                "Velocity",
                "dynamic",
                re.compile(r"^(rangerate_bias)_(\w+)$"),
            ],
            "range_bias_RU": [
                "Dimensionless",
                1,
                re.compile(r"^(range_bias_RU)_(\w+)$"),
            ],
            "rangerate_bias_Hz": [
                "per Time",
                1,
                re.compile(r"^(rangerate_bias_Hz)_(\w+)$"),
            ],
            # Optical navigation — angular (RA/DEC) bias and pixel/sample biases
            "angular_bias": [
                "Dimensionless",
                "dynamic",
                re.compile(r"^(angular_bias)_(\w+)$"),
            ],
            "centroid_bias": [
                "Dimensionless",
                "dynamic",
                re.compile(r"^(centroid_bias)_(\w+)$"),
            ],
            "cam_focal_length": [
                "Dimensionless",
                2,
                re.compile(r"^(cam_focal_length)_(\w+)$"),
            ],
            "cam_pp_offset": [
                "Dimensionless",
                2,
                re.compile(r"^(cam_pp_offset)_(\w+)$"),
            ],
            # Maneuvers
            "dv_man": ["Velocity", 3, re.compile(r"^(dv_man\d*)$")],
            # Gravitational parameter (GM) of any body
            "gm": ["Gravitational Parameter", 1, re.compile(r"^(gm)$")],
            # Spherical harmonics coefficients — cnm_n_m and snm_n_m (dimensionless)
            "cnm": ["Dimensionless", 1, re.compile(r"^(cnm_\d+_\d+)$")],
            "snm": ["Dimensionless", 1, re.compile(r"^(snm_\d+_\d+)$")],
            # Pole orientation: RA and DEC in radians, PM rate in deg/day (dimensionless)
            "pole_ra":  ["Angle", 1, re.compile(r"^(pole_ra)$")],
            "pole_dec": ["Angle", 1, re.compile(r"^(pole_dec)$")],
            "pole_pm":  ["Dimensionless", 1, re.compile(r"^(pole_pm)$")],
            # Small-body perturbations
            "A2_yark": ["Length / Time^2", 1, re.compile(r"^(A2_yark)$")],
            # First-order Gauss-Markov empirical acceleration
            "a_fogm": ["Acceleration", 3, re.compile(r"^(a_fogm)$")],
            "beta_fogm": ["per Time", 3, re.compile(r"^(beta_fogm)$")],
            # Piecewise First-order Gauss-Markov empirical acceleration
            "a_pfogm": ["Acceleration", "dynamic", re.compile(r"^(a_pfogm)$")],
            "beta_pfogm": ["per Time", "dynamic", re.compile(r"^(beta_pfogm)$")],
        }

        return state_allowed

    def _validate_component(self, name: str, dimension: int, value: ArrayWUnits):
        """
        Validate that a component matches the allowed state dictionary.

        Checks component name pattern, dimension, and physical type against
        the allowed state dictionary.

        Parameters
        ----------
        name : str
            Component name
        dimension : int
            Component dimension/size
        value : ArrayWUnits
            Component value with units

        Raises
        ------
        ValueError
            If component name is not allowed, dimension doesn't match,
            or physical type is incorrect.
        """
        allowed = self._generate_allowed_state_dictionary()

        # Check if name matches any allowed pattern
        matched = False
        matched_key = None

        for key, (phys_type, expected_dim, pattern) in allowed.items():
            if pattern.match(name):
                matched = True
                matched_key = key

                # Check dimension
                if expected_dim == "dynamic":
                    if dimension % 3 != 0:
                        if name in ("a_pfogm", "beta_pfogm"):
                            raise ValueError(
                                f"Component '{name}' is a piecewise Gauss-Markov state "
                                f"and must have dimension 3 * n_batches, "
                                f"got {dimension}. Example: n_batches=2 -> dimension=6."
                            )
                else:
                    if dimension != expected_dim:
                        raise ValueError(
                            f"Component '{name}' has incorrect dimension. "
                            f"Expected {expected_dim}, got {dimension}"
                        )

                # Check physical type (dimensions)
                try:
                    actual_dims = value.quantity.units.dimensions.name
                    if actual_dims != phys_type:
                        raise ValueError(
                            f"Component '{name}' has incorrect physical type. "
                            f"Expected '{phys_type}', got '{actual_dims}'"
                        )
                except AttributeError:
                    # If we can't get dimensions, skip this check
                    pass

                break

        if not matched:
            allowed_names = list(allowed.keys())
            raise ValueError(
                f"Component '{name}' is not allowed in StateArray. "
                f"Allowed components: {allowed_names}"
            )

    def _check_duplicate_component(self, name: str, owner):
        """
        Check if a component with the same name and owner already exists.

        Parameters
        ----------
        name : str
            Component name
        owner : Body, Spacecraft, GroundStation, or None
            Component owner

        Raises
        ------
        ValueError
            If a duplicate component is detected for the same owner.
        """
        # Create a hashable key for the component
        owner_id = id(owner) if owner is not None else None
        component_key = (name, owner_id)

        if component_key in self._component_registry:
            owner_str = f"owner {owner}" if owner is not None else "no owner (global)"
            raise ValueError(
                f"Duplicate component '{name}' detected for {owner_str}. "
                f"Each component can only be defined once per owner."
            )

    def _register_component(self, name: str, owner):
        """
        Register a component to track duplicates.

        Parameters
        ----------
        name : str
            Component name
        owner : Body, Spacecraft, GroundStation, or None
            Component owner
        """
        owner_id = id(owner) if owner is not None else None
        component_key = (name, owner_id)
        self._component_registry[component_key] = True

    @staticmethod
    def _validate_modes(estimation, dynamics):
        """
        Validate estimation and dynamics modes.

        Parameters
        ----------
        estimation : str
            Must be "estimated" or "considered"
        dynamics : str
            Must be "static" or "dynamic"

        Returns
        -------
        tuple of str
            Validated (estimation, dynamics) pair

        Raises
        ------
        ValueError
            If invalid mode values are provided
        NotImplementedError
            If estimation="considered" is specified
        """
        est = estimation.lower().strip()
        dyn = dynamics.lower().strip()

        if est not in ("estimated", "considered"):
            raise ValueError(
                f"Estimation must be 'estimated' or 'considered', got: {est}"
            )

        if dyn not in ("static", "dynamic"):
            raise ValueError(f"Dynamics must be 'static' or 'dynamic', got: {dyn}")

        return est, dyn

    def _enforce_global_order(self, est: str, dyn: str):
        """
        Enforce ordering constraints:
        - dynamic components must come before static components
        - estimated components must come before considered components
        - position/velocity ordering is handled elsewhere
        """
        # If we have started adding static stuff, you cannot add dynamic anymore
        if dyn == "dynamic" and self._seen_static:
            raise ValueError(
                "Ordering rule violated: dynamic components must appear before static components."
            )

        # If we have started adding considered stuff, you cannot add estimated anymore
        if est == "estimated" and self._seen_considered:
            raise ValueError(
                "Ordering rule violated: estimated components must appear before considered components."
            )

        # Update phase flags
        if dyn == "static":
            self._seen_static = True
        if est == "considered":
            self._seen_considered = True

    # endregion Validation #
    # ----------------------#

    # ----------------------#
    # region List Interface #
    # ----------------------#
    def __iter__(self):
        """
        Iterate over state components as tuples.

        Yields
        ------
        tuple
            Component tuple (name, dimension, estimation, dynamics, owner, value)
        """
        return iter(self._components)

    def __getitem__(self, index):
        """
        Get a state component by index.

        Parameters
        ----------
        index : int or slice
            Component index or slice

        Returns
        -------
        tuple or list of tuple
            Component tuple(s) (name, dimension, estimation, dynamics, owner, value)
        """
        return self._components[index]

    def __len__(self):
        """
        Get the number of components in the state vector.

        Returns
        -------
        int
            Number of components
        """
        return len(self._components)

    def __repr__(self):
        """
        String representation of the StateDefinition.

        Returns
        -------
        str
            Representation showing number of components
        """
        return f"StateDefinition({len(self._components)} components)"

    # endregion List Interface #
    # --------------------------#

    # ----------------------------#
    # region  From-List Builders  #
    # ----------------------------#
    @classmethod
    def from_components(
        cls,
        components: List[tuple],
        *,
        clear: bool = True,
    ) -> "StateDefinition":
        """
        Construct a StateDefinition from a pre-existing `(name, dim, est, dyn, owner, value)` list.

        Parameters
        ----------
        components : list of tuple
            Each tuple must be: (name:str, dim:int, estimation:str, dynamics:str, owner, value)
        clear : bool, optional
            Ignored for classmethod (kept for API symmetry). Always starts from an empty definition.

        Returns
        -------
        StateDefinition
            Newly built object populated from the provided list.

        Raises
        ------
        ValueError / NotImplementedError
            Propagates validation errors from position/velocity/param methods.
        """
        sd = cls()
        sd.extend_from_components(components, clear=True)
        return sd

    def extend_from_components(
        self,
        components: List[tuple],
        *,
        clear: bool = False,
    ) -> "StateDefinition":
        """
        Append components from a `(name, dim, est, dyn, owner, value)` list into this builder.

        Parameters
        ----------
        components : list of tuple
            Each tuple must be: (name:str, dim:int, estimation:str, dynamics:str, owner, value)
        clear : bool, optional
            If True, reset the builder before appending.

        Returns
        -------
        StateDefinition
            Self, for chaining.

        Notes
        -----
        - This method *routes* each tuple through the canonical public APIs
          (.position, .velocity, .param), so all validations and ordering rules apply.
        - Dimensional consistency and allowed-name checks are enforced by _validate_component().
        """
        if clear:
            self._components.clear()
            self._position_count = 0
            self._velocity_count = 0
            self._dv_maneuvers.clear()
            self._last_component_was_position = False
            self._component_registry.clear()
            self._finite_burn_pairs.clear()

        for idx, tup in enumerate(components):
            try:
                name, dim, est, dyn, owner, value = self._normalize_tuple(tup)
            except Exception as e:
                raise ValueError(
                    f"Invalid component at index {idx}: {tup!r}. Error: {e}"
                ) from e

            if name == "position":
                self.position(owner, value, estimation=est, dynamics=dyn)
            elif name == "velocity":
                self.velocity(owner, value, estimation=est, dynamics=dyn)
            else:
                # Generic parameter; enforce provided dimension
                self.param(
                    name, owner, value, dimension=dim, estimation=est, dynamics=dyn
                )
        # Enforce dRA/dDEC pairing consistency only after all components
        # have been appended.
        for (owner_id_chk, suffix_chk), flags in self._finite_burn_pairs.items():
            if flags["dRA"] != flags["dDEC"]:
                missing = "dDEC" if flags["dRA"] else "dRA"
                name_missing = f"{missing}{suffix_chk}"
                raise ValueError(
                    f"Incomplete finite-burn angular pair for owner id={owner_id_chk}: "
                    f"missing '{name_missing}'. dRA and dDEC must coexist."
                )

        # Enforce a_fogm ordering: it must appear immediately after the velocity
        # of the same owner/body.
        if name == "a_fogm":
            if len(self._components) < 3:
                raise ValueError(
                    "'a_fogm' must appear after 'position' and 'velocity' of the same owner."
                )

            prev = self._components[-2]
            prev2 = self._components[-3]

            if not (
                prev[0] == "velocity"
                and prev2[0] == "position"
                and prev[4] is owner
                and prev2[4] is owner
            ):
                raise ValueError(
                    "'a_fogm' must appear immediately after 'position' and 'velocity' "
                    "of the same owner."
                )

        return self

    @staticmethod
    def _normalize_tuple(t: tuple):
        """
        Validate and normalize a raw component tuple to:
        (name:str, dim:int, est:str, dyn:str, owner, value)
        """
        if not isinstance(t, tuple) or len(t) != 6:
            raise TypeError(
                "Component must be a 6-tuple: (name, dim, est, dyn, owner, value)"
            )
        name, dim, est, dyn, owner, value = t

        # Basic type checks; detailed checks are done downstream in the public APIs.
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        if not isinstance(dim, int):
            raise TypeError("dim must be an int")
        if not isinstance(est, str) or not isinstance(dyn, str):
            raise TypeError("est and dyn must be str")
        # owner can be None / Body / Spacecraft / GroundStation; value is ArrayWUnits-like

        # Normalize casing/whitespace of mode strings here; public APIs will validate tokens
        est = est.strip().lower()
        dyn = dyn.strip().lower()
        return name, dim, est, dyn, owner, value

    def to_components(self) -> list[tuple]:
        """
        Return the state definition as a list of legacy tuples:
        (name, dim, estimation, dynamics, owner, value)
        """
        return list(self._components)

    @classmethod
    def from_legacy(cls, components: list[tuple]) -> "StateDefinition":
        """
        Build a StateDefinition from a legacy tuple list
        (name, dim, est, dyn, owner, value).
        """
        return cls.from_components(components)

    @classmethod
    def as_state_definition(cls, obj) -> "StateDefinition":
        """
        Accept either:
          - an existing StateDefinition, or
          - a legacy list of tuples (name, dim, est, dyn, owner, value),
        and return a StateDefinition instance.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, list):
            return cls.from_components(obj)
        raise TypeError(
            "as_state_definition expects a StateDefinition or a list of legacy tuples."
        )

    # endregion From-List Builders #
