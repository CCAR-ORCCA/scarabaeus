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
    **Custom components**

    Beyond the built-in component names, user scripts may register their own
    solve-for parameters with :meth:`register_custom_parameter` (or implicitly
    through :meth:`custom_param`) so that a
    :class:`~scarabaeus.CustomDynamicModel` can estimate them without any
    change to the library source.

    **Ordering rules**

    * Dynamic components must appear **before** static components.
    * Estimated components must appear **before** considered components.
    * Each ``.velocity()`` call must immediately follow its paired ``.position()`` call.
    * Each of ``dv_man`` / ``dv_man1`` / ``dv_man2`` may appear at most once per state.
    * ``dRA[N]`` and ``dDEC[N]`` finite-burn angle pairs must both be present for the same
      owner and numeric suffix.

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
       * - ``eta_thermal``
         - Thermal radiation
         - 1
         - Dimensionless
         - --
         - Thermal re-radiation scale factor. Nominal value 1.0.
       * - ``eta_antenna``
         - Antenna radiation
         - 1
         - Dimensionless
         - --
         - Antenna/LIDAR radiation pressure scale factor. Nominal value 1.0.
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
         - Cartesian correction in the station's declared body-fixed frame.
       * - ``gs_delta_location_CYL_[tag]``
         - Ground station
         - 3
         - Length
         - km
         - Local cylindrical delta (radius, east arc length, Z).
       * - ``eop_delta_pole_x``, ``eop_delta_pole_y``
         - Earth or Earth station
         - 1 each
         - Angle
         - rad
         - Intended minus nominal terrestrial xp/yp; static measurement parameters.
       * - ``eop_delta_ut1``
         - Earth or Earth station
         - 1
         - Time
         - sec
         - Intended minus nominal UT1; positive advances spin.
       * - ``eop_delta_celestial_pole_x``, ``eop_delta_celestial_pole_y``
         - Earth or Earth station
         - 1 each
         - Angle
         - rad
         - Intended minus nominal celestial pole offsets dX/dY (nutation).
       * - ``body_orientation_delta``
         - Central body or station
         - 3
         - Angle
         - rad
         - Active body-fixed X/Y/Z station rotation; usable on any planet.
       * - ``love_h2``, ``shida_l2``
         - Central body or station
         - 1 each
         - unitless
         - —
         - Degree-2 displacement Love and Shida numbers. Scale how far the
           solid tide moves a station, so they need an attached
           :class:`~scarabaeus.SolidTide`; without one the partial is zero.
       * - ``love_k_[n]_[m]``
         - Central body
         - 1 each
         - unitless
         - —
         - Degree-``n`` order-``m`` *potential* Love number. Scales how far the
           solid tide moves the gravity field, a different quantity from
           ``love_h2``. Reaches the observable only through the trajectory, so
           it needs a :class:`~scarabaeus.SolidTide` on the harmonics model.
       * - ``tropo_zenith_delay_[tag]``
         - Ground station
         - 1
         - Length
         - km
         - Estimated zenith tropospheric delay, mapped to the slant direction
           by the Chao wet mapping. Additive on the modelled troposphere.
       * - ``iono_zenith_delay_[tag]``
         - Ground station
         - 1
         - Length
         - km
         - Estimated zenith ionospheric delay **at 2295 MHz**, scaled by
           (2295 MHz / f)^2 and mapped by the thin-shell obliquity.
       * - ``dt_dv``
         - Spacecraft
         - 1
         - Time
         - sec
         - Time offset of an impulsive maneuver.
       * - ``dv_smooth``
         - Spacecraft
         - 3
         - Velocity
         - km/sec
         - Smoothed impulsive burn components.
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
         - dyn
         - Dimensionless
         - RU
         - Range bias in range units for station ``[tag]``; one element
           for every pass or one per pass.
       * - ``rangerate_bias_Hz_[tag]``
         - Radiometric bias
         - dyn
         - per Time
         - Hz
         - Doppler bias in Hz for station ``[tag]``; one element for every
           pass or one per pass.
       * - ``dor_bias_[tag]``
         - Radiometric bias
         - dyn
         - Length
         - km
         - Range bias on the first station ``[tag]`` of a differential
           one-way range; one element for every pass or one per pass.
       * - ``uplink_delay_bias_[tag]``
         - Ground station
         - 1
         - Time
         - sec
         - Electronic delay correction at t1 (transmit) for station ``[tag]``,
           common to its passes; enters the round-trip light time of range
           measurements.
       * - ``transponder_delay_bias``
         - Spacecraft
         - 1
         - Time
         - sec
         - Transponder delay correction at t2, common to every station and
           pass; enters the round-trip light time of range measurements.
       * - ``downlink_delay_bias_[tag]``
         - Ground station
         - 1
         - Time
         - sec
         - Electronic delay correction at t3 (receive) for station ``[tag]``,
           common to its passes; enters the round-trip light time of range
           measurements.
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
       * - ``pole_w0``
         - Body orientation
         - 1
         - Angle
         - rad
         - Prime-meridian angle at J2000 (IAU convention).
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
       * - ``tropo_z_dry``
         - Media
         - 1
         - Length
         - m
         - Additive zenith dry tropospheric delay correction (Moyer 10-7).
       * - ``tropo_z_wet``
         - Media
         - 1
         - Length
         - m
         - Additive zenith wet tropospheric delay correction (Moyer 10-7).
       * - ``iono_delay_bias``
         - Media
         - 1
         - Dimensionless
         - --
         - Multiplicative scale on the ionospheric delay (Moyer 10-11).
       * - ``gamma_rel``
         - Relativity
         - 1
         - Dimensionless
         - --
         - PPN parameter gamma in the light-time delay (Moyer 12-24).
       * - ``solar_corona_scale``
         - Media
         - 1
         - Dimensionless
         - --
         - Multiplicative scale on the solar-corona delay (Moyer 10-76).
       * - ``a_fourier``
         - Empirical accel.
         - 3*(2N+1)
         - Acceleration
         - km/s^2
         - Fourier series coefficients; dim = 3 x (2 x degree + 1).
       * - ``a_ppoly``
         - Empirical accel.
         - 3*(N_b+p)
         - Acceleration
         - km/s^2
         - Piecewise polynomial coefficients; order p = 0 or 1 over N_b batches.

    *dyn* in the Dim column means the dimension is inferred from the supplied initial value.
    For ``a_pfogm`` / ``beta_pfogm`` / ``a_fourier`` / ``a_ppoly`` it must be a
    multiple of 3.

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

    # Derived from ValueError: these replace the generic raises this class used
    # before, and callers (including the test suite) catch ValueError.
    class UnknownComponentError(ValueError):
        """Raised when a component name matches no allowed (built-in or custom) pattern."""

    class ReservedComponentNameError(ValueError):
        """Raised when a custom parameter name collides with a built-in component."""

    class CustomParameterConflictError(ValueError):
        """Raised when a custom parameter is re-registered with a different definition."""

    # Registry of user-defined state components, shared by every StateDefinition.
    # Layout matches _generate_allowed_state_dictionary entries:
    #     name -> [physical_type, size, compiled_pattern]
    # ``physical_type`` may be None, which disables the unit-dimension check.
    _custom_state_components: Dict[str, List] = {}

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

    def custom_param(
        self,
        name,
        owner,
        value,
        dimension=None,
        estimation="estimated",
        dynamics="static",
        physical_type=None,
    ):
        """
        Register and add a user-defined parameter in a single call.

        Convenience wrapper around :meth:`register_custom_parameter` followed by
        :meth:`param`, so a script can introduce a brand-new solve-for parameter
        — one consumed by a :class:`~scarabaeus.CustomDynamicModel` — without
        touching the library source.

        Parameters
        ----------
        name : str
            Parameter name (e.g. ``"drag_cd"``). Must not collide with a
            built-in component name.
        owner : Body, Spacecraft, GroundStation, or None
            The body this parameter belongs to.
        value : ArrayWUnits or ArrayWFrame
            Initial value with units.
        dimension : int, optional
            Parameter dimension. Inferred from ``value`` when ``None``.
        estimation : str, optional
            ``"estimated"`` (default) or ``"considered"``.
        dynamics : str, optional
            ``"static"`` (default, constant parameter) or ``"dynamic"`` (the
            parameter has its own ODE, supplied through
            :class:`~scarabaeus.CustomParameter`).
        physical_type : str, optional
            Dimension name enforced on ``value`` (e.g. ``"Acceleration"``).
            ``None`` (default) skips the check.

        Returns
        -------
        StateDefinition
            Self, for method chaining.

        Examples
        --------
        .. code-block:: python

            state_def = (scb.StateDefinition()
                .position(sc, pos_0)
                .velocity(sc, vel_0)
                .custom_param("drag_cd", sc, scb.ArrayWFrame(2.2, None, None))
            )
        """
        dim = dimension if dimension is not None else value.shape[0]
        self.register_custom_parameter(name, dimension=dim, physical_type=physical_type)

        return self.param(
            name,
            owner,
            value,
            dimension=dimension,
            estimation=estimation,
            dynamics=dynamics,
        )

    @classmethod
    def register_custom_parameter(
        cls, name: str, dimension=1, physical_type: str | None = None
    ) -> None:
        """
        Declare a user-defined state component so it passes state validation.

        Registration is global and idempotent: re-registering the same name with
        the same dimension and physical type is a no-op, which lets a script
        build several state definitions (truth, reference, filter) from the same
        parameter declaration.

        Parameters
        ----------
        name : str
            Component name. Must be a plain identifier (letters, digits and
            underscores) and must not match any built-in component pattern.
        dimension : int or str, optional
            Fixed component size, or ``"dynamic"`` to accept any size.
            Defaults to 1.
        physical_type : str, optional
            Required unit-dimension name (e.g. ``"Acceleration"``,
            ``"Dimensionless"``). ``None`` (default) disables the check.

        Raises
        ------
        StateDefinition.ReservedComponentNameError
            If ``name`` is not a valid identifier or collides with a built-in
            component pattern.
        StateDefinition.CustomParameterConflictError
            If ``name`` was already registered with a different dimension or
            physical type.

        See Also
        --------
        scarabaeus.CustomParameter : Declares a parameter and registers it here.
        scarabaeus.CustomDynamicModel : Consumes the registered parameter.
        """
        if not re.fullmatch(r"[A-Za-z_]\w*", str(name)):
            raise cls.ReservedComponentNameError(
                f"Custom parameter name '{name}' is not a valid identifier. "
                "Use letters, digits and underscores, starting with a letter "
                "or underscore."
            )

        # Built-in names are matched case-insensitively downstream (the STM
        # partial keys are lower-cased), so guard against a case-only collision.
        builtins = cls._generate_builtin_state_dictionary()
        for key, (_, _, pattern) in builtins.items():
            if pattern.match(name) or pattern.match(str(name).lower()):
                raise cls.ReservedComponentNameError(
                    f"Custom parameter name '{name}' collides with the built-in "
                    f"state component '{key}'. Choose a different name."
                )

        lowered = str(name).lower()
        for existing in cls._custom_state_components:
            if existing != name and existing.lower() == lowered:
                raise cls.CustomParameterConflictError(
                    f"Custom parameter '{name}' differs from the already "
                    f"registered '{existing}' only by case. Partial-derivative "
                    "keys are case-insensitive, so the two cannot coexist."
                )

        entry = [physical_type, dimension, re.compile(rf"^({re.escape(name)})$")]

        previous = cls._custom_state_components.get(name)
        if previous is not None and (
            previous[0] != entry[0] or previous[1] != entry[1]
        ):
            raise cls.CustomParameterConflictError(
                f"Custom parameter '{name}' is already registered with "
                f"physical_type={previous[0]!r} and dimension={previous[1]!r}; "
                f"cannot re-register with physical_type={entry[0]!r} and "
                f"dimension={entry[1]!r}."
            )

        cls._custom_state_components[name] = entry

    @classmethod
    def unregister_custom_parameter(cls, name: str) -> None:
        """
        Remove a previously registered custom state component.

        Parameters
        ----------
        name : str
            Name passed to :meth:`register_custom_parameter`. Unknown names are
            ignored.
        """
        cls._custom_state_components.pop(name, None)

    @classmethod
    def custom_parameters(cls) -> Dict[str, List]:
        """
        Return a copy of the registered custom state components.

        Returns
        -------
        dict
            Mapping ``name -> [physical_type, size, pattern]``.
        """
        return dict(cls._custom_state_components)

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

        Built-in components come first, followed by any component registered
        through :meth:`register_custom_parameter`.

        Returns
        -------
        dict
            Dictionary mapping component patterns to their specifications.
            Format: ``{name: [physicalType, size, regex_pattern]}``
        """
        state_allowed = cls._generate_builtin_state_dictionary()
        state_allowed.update(cls._custom_state_components)

        return state_allowed

    @classmethod
    def _generate_builtin_state_dictionary(cls) -> Dict[str, List]:
        """
        Generate the library-defined portion of the allowed-component dictionary.

        Kept separate from :meth:`_generate_allowed_state_dictionary` so that
        :meth:`register_custom_parameter` can detect collisions against built-in
        names only.

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
            # Spacecraft emission (thermal re-radiation, antenna/LIDAR recoil)
            "eta_thermal": ["Dimensionless", 1, re.compile(r"^(eta_thermal)$")],
            "eta_antenna": ["Dimensionless", 1, re.compile(r"^(eta_antenna)$")],
            # Finite burn estimable parameters
            # allows: k_thrust, k_thrust1, k_thrust2, ...
            "k_thrust": ["Dimensionless", 1, re.compile(r"^(k_thrust\d*)$")],
            # allows: dRA, dRA1, dRA2, ...
            "dRA": ["Angle", 1, re.compile(r"^(dRA\d*)$")],
            # allows: dDEC, dDEC1, dDEC2, ...
            "dDEC": ["Angle", 1, re.compile(r"^(dDEC\d*)$")],
            # Smoothed impulsive maneuver estimable parameters
            # allows: dv_smooth, dv_smooth1, dv_smooth2, ...
            "dv_smooth": ["Velocity", 3, re.compile(r"^(dv_smooth\d*)$")],
            # Correction to the nominal maneuver epoch, NOT the epoch itself.
            # allows: dt_dv, dt_dv1, dt_dv2, ...
            "dt_dv": ["Time", 1, re.compile(r"^(dt_dv\d*)$")],
            # Ground station / tracking
            "gs_delta_location_ECEF": [
                "Length",
                3,
                re.compile(r"^(gs_delta_location_ECEF)_(\w+)$"),
            ],
            # Cylindrical station location, the parametrisation MONTE reports:
            # (radius, east, Z), where east is the along-longitude arc r*dlon so
            # all three carry length units. Convert a longitude sigma with
            # east = radius * longitude_in_radians. The cylindrical basis is the
            # Earth-fixed basis turned about Z by the station longitude, so the
            # two forms differ by a rotation. A diagonal cylindrical covariance
            # generally becomes correlated when rotated into Cartesian axes.
            "gs_delta_location_CYL": [
                "Length",
                3,
                re.compile(r"^(gs_delta_location_CYL)_(\w+)$"),
            ],
            # Earth-specific deltas against the loaded nominal orientation.
            # Owner is Earth (global to registered Earth stations), or a station.
            "eop_delta_pole_x": ["Angle", 1, re.compile(r"^eop_delta_pole_x$")],
            "eop_delta_pole_y": ["Angle", 1, re.compile(r"^eop_delta_pole_y$")],
            "eop_delta_ut1": ["Time", 1, re.compile(r"^eop_delta_ut1$")],
            "eop_delta_celestial_pole_x": ["Angle", 1,
                                           re.compile(r"^eop_delta_celestial_pole_x$")],
            "eop_delta_celestial_pole_y": ["Angle", 1,
                                           re.compile(r"^eop_delta_celestial_pole_y$")],
            # Generic active body-fixed X/Y/Z station rotation, in radians.
            "body_orientation_delta": ["Angle", 3, re.compile(r"^body_orientation_delta$")],
            # Degree-2 tidal response of the deformed body. Global to the body,
            # so no station tag: every station on it shares one solid tide.
            "love_h2": [None, 1, re.compile(r"^love_h2$")],
            "shida_l2": [None, 1, re.compile(r"^shida_l2$")],
            # Potential Love numbers, one per (degree, order) like cnm/snm.
            # Unlike h2 and l2, which a station feels directly, these reach
            # the observable only by bending the trajectory, so their
            # sensitivity comes through the STM.
            "love_k": [None, 1, re.compile(r"^love_k_(\d+)_(\d+)$")],
            # Estimated zenith path delays, per station. Additive on top of
            # whatever MediaCorrections models, and mapped to the slant
            # direction by the same elevation mapping the model uses.
            "tropo_zenith_delay": ["Length", 1, re.compile(r"^tropo_zenith_delay_(\w+)$")],
            "iono_zenith_delay": ["Length", 1, re.compile(r"^iono_zenith_delay_(\w+)$")],
            # The lookaheads keep range_bias_RU_* and rangerate_bias_Hz_* from
            # matching here first and being validated as a length / velocity.
            "range_bias": [
                "Length",
                "dynamic",
                re.compile(r"^(range_bias)_(?!RU_)(\w+)$"),
            ],
            "rangerate_bias": [
                "Velocity",
                "dynamic",
                re.compile(r"^(rangerate_bias)_(?!Hz_)(\w+)$"),
            ],
            "range_bias_RU": [
                "Dimensionless",
                "dynamic",
                re.compile(r"^(range_bias_RU)_(\w+)$"),
            ],
            "rangerate_bias_Hz": [
                "per Time",
                "dynamic",
                re.compile(r"^(rangerate_bias_Hz)_(\w+)$"),
            ],
            # Range bias on the first station of a differential one-way range
            "dor_bias": [
                "Length",
                "dynamic",
                re.compile(r"^(dor_bias)_(\w+)$"),
            ],
            # Electronic delay corrections, none split by pass. The uplink (t1)
            # and downlink (t3) delays are owned by the station; the
            # transponder delay (t2) by the spacecraft, common to every station
            "uplink_delay_bias": [
                "Time",
                1,
                re.compile(r"^(uplink_delay_bias)_(\w+)$"),
            ],
            "transponder_delay_bias": [
                "Time",
                1,
                re.compile(r"^(transponder_delay_bias)$"),
            ],
            "downlink_delay_bias": [
                "Time",
                1,
                re.compile(r"^(downlink_delay_bias)_(\w+)$"),
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
            # Pole orientation: RA, DEC and the J2000 prime meridian in radians,
            # PM rate in deg/day (dimensionless)
            "pole_ra":  ["Angle", 1, re.compile(r"^(pole_ra)$")],
            "pole_dec": ["Angle", 1, re.compile(r"^(pole_dec)$")],
            "pole_pm":  ["Dimensionless", 1, re.compile(r"^(pole_pm)$")],
            "pole_w0":  ["Angle", 1, re.compile(r"^(pole_w0)$")],
            # Small-body perturbations
            "A2_yark": ["Length / Time^2", 1, re.compile(r"^(A2_yark)$")],
            # First-order Gauss-Markov empirical acceleration
            "a_fogm": ["Acceleration", 3, re.compile(r"^(a_fogm)$")],
            "beta_fogm": ["per Time", 3, re.compile(r"^(beta_fogm)$")],
            # Piecewise First-order Gauss-Markov empirical acceleration
            "a_pfogm": ["Acceleration", "dynamic", re.compile(r"^(a_pfogm)$")],
            "beta_pfogm": ["per Time", "dynamic", re.compile(r"^(beta_pfogm)$")],
            # Media correction parameters (Moyer Ch. 10, 12, 13)
            # Additive corrections to the zenith tropospheric delays [m].
            # Partial of ρ w.r.t. these uses the Chao dry/wet mapping function (Moyer 10-7).
            "tropo_z_dry": ["Length", 1, re.compile(r"^(tropo_z_dry)$")],
            "tropo_z_wet": ["Length", 1, re.compile(r"^(tropo_z_wet)$")],
            # Multiplicative scale on the ionospheric correction (dimensionless).
            # Partial of ρ w.r.t. this uses the iono model value (Moyer 10-11).
            "iono_delay_bias": ["Dimensionless", 1, re.compile(r"^(iono_delay_bias)$")],
            # PPN relativity parameter γ (Moyer 12-24: ∂ρ/∂γ = (RLT_dn + RLT_up)/(1+γ)).
            # Uses the GR light-time components exposed from compute_precise_RTLT.
            "gamma_rel": ["Dimensionless", 1, re.compile(r"^(gamma_rel)$")],
            # Multiplicative scale on the solar-corona model (dimensionless, nominal 1).
            # Partial uses the per-leg SC delays from compute_precise_RTLT (Moyer 10-76).
            "solar_corona_scale": ["Dimensionless", 1, re.compile(r"^(solar_corona_scale)$")],
            # Fourier empirical acceleration coefficients
            "a_fourier": ["Acceleration", "dynamic", re.compile(r"^(a_fourier)$")],
            "a_ppoly": ["Acceleration", "dynamic", re.compile(r"^(a_ppoly)$")],
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
        StateDefinition.UnknownComponentError
            If the component name matches no allowed (built-in or custom) pattern.
        ValueError
            If the dimension doesn't match or the physical type is incorrect.
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
                        if name == "a_fourier":
                            raise ValueError(
                                f"Component '{name}' holds one Fourier series per "
                                f"axis and must have dimension 3 * (2 * degree + 1), "
                                f"got {dimension}. Example: degree=1 -> dimension=9, "
                                "degree=0 (constant) -> dimension=3."
                            )
                        if name == "a_ppoly":
                            raise ValueError(
                                f"Component '{name}' holds piecewise polynomial "
                                f"coefficients and must have dimension 3 * n_batches "
                                f"(order 0) or 3 * (n_batches + 1) (order 1), got "
                                f"{dimension}. Example: n_batches=4 -> dimension=12 "
                                "(order 0) or 15 (order 1)."
                            )
                else:
                    if dimension != expected_dim:
                        raise ValueError(
                            f"Component '{name}' has incorrect dimension. "
                            f"Expected {expected_dim}, got {dimension}"
                        )

                # Check physical type (dimensions). Custom components may declare
                # ``None`` to opt out of the check entirely.
                if phys_type is not None:
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
            raise self.UnknownComponentError(
                f"Component '{name}' is not allowed in StateArray. "
                f"Allowed components: {allowed_names}. "
                "Register a user-defined parameter first with "
                "StateDefinition.register_custom_parameter(), scarabaeus."
                "CustomParameter(), or add it with .custom_param()."
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
        ValueError
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
        Return a StateDefinition from either an existing StateDefinition or a
        legacy list of tuples ``(name, dim, est, dyn, owner, value)``.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, list):
            return cls.from_components(obj)
        raise TypeError(
            "as_state_definition expects a StateDefinition or a list of legacy tuples."
        )

    # endregion From-List Builders #
