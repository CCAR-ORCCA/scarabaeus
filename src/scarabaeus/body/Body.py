# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, EpochArray, ArrayWFrame, Frame

import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.SCBPolynomial import SCBPolynomial

# define units
km, sec = Units.get_units(["km", "sec"])

# class definition
class Body:
    """ Defines the simplest form of ephemeris object in Scarabaeus.

    From NAIF [1]_:

        "An ephemeris object is any object that may have ephemeris or trajectory data such as a planet, natural satellite, tracking station, spacecraft, barycenter (the center of mass of a group of bodies), asteroid, or comet. Each body in the solar system is associated with an integer code for use with SPICE."

    Body objects serve to hold information relevant to some ephemeris object that exists within the kernel pool.

    Parameters
    ----------
    name : str
        The unique name of the body.

    spice_id : int or str, optional
        The NAIF ID code of the object. May be given as an integer ID or as a string. Defaults to ``None``.

    mass : ArrayWUnits or SCBPolynomial, optional
        The mass of the body, supports static and time variant definitions depending 
        on the received input type. Defaults to ``None``. Accepted types are:
        
        * :class:`~scarabaeus.ArrayWUnits` - static mass definition.
        * :class:`~scarabaeus.SCBPolynomial` - time variant mass definition, must 
          be a function of time.

    mass_poly_deg : int, optional
        The degree of the polynomial used to fit the mass profile. Defaults to ``3``.

    See Also
    --------
    scarabaeus.SpiceManager : Provides many useful methods related to SPICE objects.

    Notes
    -----
    Every Body instance corresponds to a NAIF ephemeris object identified by an integer
    NAIF ID code.  When a ``name`` is supplied without a ``spice_id``, Scarabaeus queries the
    loaded kernel pool via :meth:`~scarabaeus.SpiceManager.name_to_ID`. If found, the ID is 
    assigned automatically.  For new bodies not yet registered in any kernel, a ``spice_id`` 
    must be supplied explicitly and the body is added to the pool via :meth:`~scarabaeus.SpiceManager.def_new_body`.

    References
    ----------
    .. [1] https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html#Introduction
    """

    # region Constructor
    def __init__(
        self,
        name: str,
        spice_id: int | str = None,
        mass: ArrayWUnits | SCBPolynomial = None,
        mass_poly_deg: int = 1,
    ):
        # define properties
        self._name = name
        self.spice_id = spice_id
        self._mass_profile = mass
        self._mass_units = mass.units if mass is not None else None
        self._mass_poly_deg = mass_poly_deg

        self._mass_profiles = []
        self._mass_profiles_tspan = []

    # region Properties 
    @property
    def name(self) -> str:
        """
        The name of the Body object.
        """
        return self._name

    @property
    def spice_id(self) -> int | str:
        """ The SPICE ID of the Body object. """
        return self._spice_id

    @spice_id.setter
    def spice_id(self, input_val):
        """ Set the SPICE ID, resolving it against the loaded kernel pool.

        Four cases are handled:

        1. No ID given and name not in pool → ``ValueError`` (ID must be supplied).
        2. No ID given but name found in pool → ID looked up automatically.
        3. ID given, name not in pool → ID used directly and body registered via
           ``SpiceManager.def_new_body``.
        4. ID given and matches pool entry → pool value used.

        Parameters
        ----------
        input_val : int, str, or None
            The NAIF ID to assign, or ``None`` to resolve from the kernel pool.

        Raises
        ------
        ValueError
            If no ID is provided and the body name is not found in the pool,
            or if the provided ID conflicts with the pool entry for this name.
        """
        # Check if a SPICE ID corresponding to this name exists in the loaded
        # kernels. If not, the user should provide an ID as an input.
        from scarabaeus import SpiceManager

        spice_id_pool = SpiceManager.name_to_ID(self.name)

        if (input_val == None) and (spice_id_pool == None):
            # no SPICE ID given and not found in SPICE pool based off given name
            err_msg = f"Body {self.name} not found in the kernel pool. A SPICE ID must be provided as an input for new bodies."
            raise ValueError(err_msg)

        elif (input_val == None) and (spice_id_pool != None):
            # no SPICE ID given but a name was given and it exists in the SPICE pool -> get ID from the pool and assign
            self._spice_id = spice_id_pool

        elif (input_val != None) and (spice_id_pool == None):
            # SPICE ID given, not found in SPICE pool based off given name -> assign using given SPICE ID
            self._spice_id = input_val

            # and add to the SPICE pool
            SpiceManager.def_new_body(self.name, input_val)

        elif input_val == spice_id_pool:
            # SPICE ID given, matches ID found in SPICE pool -> assign using the pool
            self._spice_id = spice_id_pool

        else:
            # SPICE ID given, but doesn't match the ID defined in the SPICE pool based off the given name
            err_msg = (
                f"Input SPICE ID {input_val} does not match the SPICE ID {spice_id_pool} "
                f"found in the kernel pool for the body {self.name}."
            )
            raise ValueError(err_msg)

    def mass(self, t: EpochArray | float | np.ndarray = None) -> ArrayWUnits:
        """
        Returns the mass of the body. If the mass is time-varying, evaluates it at time t.
        If the mass is constant, returns the same value regardless of t.

        Parameters
        ----------
        t : EpochArray, float, or np.ndarray, optional
            Time(s) in ephemeris seconds (TDB) at which to evaluate the mass. Or None for constant mass.

        Returns
        -------
        ArrayWUnits
            Mass at time t or constant mass.
        """
        if isinstance(self._mass_profile, ArrayWUnits):
            return self._mass_profile

        elif isinstance(self._mass_profile, SCBPolynomial):
            # If mass is time-varying, should evaluate at t
            if t is None:
                raise ValueError(
                    "Must provide 't' (EpochArray) when mass is time-varying."
                )

            # If EpochArray, ensure TDB then extract raw epoch(s)
            if isinstance(t, EpochArray):
                if t.system != "TDB":
                    raise ValueError("EpochArray must usesys='TDB'.")
                t_vals = t.times.values
            # floats or numpy arrays are passed through
            elif isinstance(t, (float, np.ndarray)):
                t_vals = t

            else:
                raise TypeError(
                    f"t must be float, np.ndarray, or EpochArray; got {type(t)}"
                )

            # Fast path: the current profile always covers the active integration span
            # (set by _initialize_mass_profile just before propagate()). Avoid O(n)
            # linear search through the accumulated profile list.
            try:
                return ArrayWUnits(self._mass_profile(t_vals), self._mass_units)
            except ValueError:
                return self.get_mass(t_vals)

        else:
            raise TypeError(
                f"Unsupported internal mass type: {type(self._mass_profile.values)}"
            )

    @property
    def mass_profile(self) -> ArrayWUnits | SCBPolynomial:
        """
        Stored constant or polynomial mass profile.
        """
        return self._mass_profile

    # endregion Properties

    # region Methods 
    def get_state(self, epoch_0, reference_frame: str = "J2000", origin: str = "EARTH") -> ArrayWFrame:
        """
        Retrieves the state of the body relative to a given origin in a specified reference frame.

        Parameters
        ----------
        epoch_0 : EpochArray
            The epoch times for which the state is to be computed.

        reference_frame : str
            The reference frame in which the state is desired.
            Defaults to '`J2000`'.

        origin : str
            The origin body relative to which the state is computed.
            Defaults to ``'EARTH'``.

        Returns
        -------
        state_vector : ArrayWFrame
            The state vector of the body relative to the origin.
        """
        if not isinstance(epoch_0, EpochArray):
            raise TypeError("epoch_0 must be an instance of EpochArray.")

        if not isinstance(reference_frame, str) or not isinstance(origin, str):
            raise TypeError("reference_frame and origin must be strings.")

        # Convert self.spice_id to string if it is an integer
        target = str(self.spice_id) if isinstance(self.spice_id, int) else self.spice_id

        try:
            # Retrieve state from SpiceManager
            from scarabaeus import SpiceManager

            state_vector = SpiceManager.get_state(
                trgt_bdy=target,
                epoch_time=epoch_0.times.values,
                reference_frame=reference_frame,
                obsvr_bdy=origin,
            )
        except Exception as e:
            raise RuntimeError(f"Error retrieving state for target {target}: {e}")

        return ArrayWFrame(state_vector, Frame(reference_frame))

    def set_mass_profile(self, poly: SCBPolynomial, units: Units) -> None:
        """
        Sets the mass profile of the body using a polynomial function and associated units.

        Parameters
        ----------
        poly : SCBPolynomial
            A polynomial object representing the mass profile.
        units : Units
            The units associated with the mass profile.

        Returns
        -------
        None
        """

        self._mass_profile = poly
        self._mass_units = units
        self._mass_profiles.append(poly)

    def get_mass(self, t: EpochArray) -> ArrayWUnits:
        """
        Evaluates the time-varying mass profile at specified epochs.

        Parameters
        ----------
        t : EpochArray
            Time(s) in ephemeris seconds (TDB).

        Returns
        -------
        mass : ArrayWUnits
            Evaluated mass at each input epoch.
        """
        # if not isinstance(self._mass_profile.values, SCBPolynomial):
        #    raise TypeError(
        #        "get_mass() can only be used with polynomial mass profiles."
        #    )

        # NOTE: code when the force models will have (if ever) epoch in EpochArray format
        # Check if the time frame is TDB
        # t_eval = (
        #    copy.deepcopy(t).convert_to("TDB")
        #    if t.system != "TDB"
        #    else copy.deepcopy(t)
        # )

        # t_sec = t_eval.times.values  # Ephemeris time in seconds
        for t_span, poly in zip(self._mass_profiles_tspan, self._mass_profiles):
            t_lo = min(t_span[0], t_span[-1])
            t_hi = max(t_span[0], t_span[-1])
            if t_lo <= t <= t_hi:
                m_vals = poly(t)
                return ArrayWUnits(m_vals, self._mass_units)

        if not (t_lo <= t <= t_hi):
            raise ValueError(
                f"Provided time t={t} is outside the valid range [{t_lo}, {t_hi}]. "
            )

    def fit_mass_profile_from_data(
        self,
        t_array: np.ndarray,
        m_array: np.ndarray,
        domain: tuple,
        units: Units,
    ) -> None:
        """
        Fits a polynomial to a mass time history and sets it as the new mass profile.

        Parameters
        ----------
        t_array : numpy.ndarray
            Time values in seconds (ephemeris time).

        m_array : numpy.ndarray
            Corresponding mass values.

        deg : int
            Degree of the polynomial.

        domain : tuple
            Domain [t0, tf] for the polynomial conversion.

        units : ArrayWUnits
            Units to assign to the mass profile. If None, uses existing mass profile units.

        Returns
        -------
        None
        """
        poly = SCBPolynomial.fit(t_array, m_array, deg=self._mass_poly_deg).convert(
            domain=domain
        )
        self.set_mass_profile(poly, units=units)
        self._mass_profiles_tspan.append(t_array)
