# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import constants, PCInfo, Body, Units, ArrayWUnits, Frame, GroundStation
import sys, inspect

if sys.version_info >= (3, 11):
    from typing import Self, Tuple

# define units
kg, km, sec, mu = Units.get_units(["kg", "km", "sec", "mu"])

# class definition
class CelestialBody(Body):
    """ Defines an astronomical body such as a planet, moon, asteroid, or comet.

        Parameters
        ----------
        name : str
            The written name of the CelestialBody.

        mass : ArrayWUnits, optional
            The mass of the CelestialBody, in any mass units. Recommend expressed in
            units of kilograms. Derived from ``grav_param`` when omitted. Defaults to
            ``None``.

        mean_radius : ArrayWUnits, optional
            The mean radius of the CelestialBody, in kilometers. Defaults to ``None``.

        grav_param : ArrayWUnits, optional
            The gravitational parameter of the CelestialBody, in units of
            :math:`\\frac{km^3}{s^2}`. Derived from ``mass`` when omitted. Defaults
            to ``None``.

        base_frame : Frame or str, optional
            The body-centered, body-fixed reference frame attached to the CelestialBody,
            as a Frame or a frame name. Defaults to ``None``.

        spice_id : int or str, optional
            The NAIF ID code of the object. May be given as an integer ID or as a string.
            Defaults to ``None``.

        barycenter_id : int, optional
            The NAID ID code of the object's barycenter_id. Defaults to ``None``.
        
        barycenter_GM : ArrayWUnits, optional
            The gravitational parameter of the body's barycenter if it has one. Defaults 
            to ``None``.

        gs_list : list, optional
            A list containing all of the ground stations on the surface of the CelestialBody.
            Defaults to ``[]``.

        See Also
        --------
        scarabaeus.constants : Scarabaeus' constants library, including planetary constants.
        scarabaeus.Frame : Scarabaeus' representation of reference frames.

        Notes
        -----
        The gravitational parameter :math:`\\mu` is defined as

        .. math::

            \\mu = GM

        where :math:`G = 6.674\\times10^{-20}\\ \\mathrm{km}^3\\,\\mathrm{kg}^{-1}\\,\\mathrm{s}^{-2}`
        is the universal gravitational constant and :math:`M` is the body mass.
        The circular orbital velocity at distance :math:`r` and surface escape
        velocity are

        .. math::

            v_{circ} = \\sqrt{\\frac{\\mu}{r}},
            \\qquad
            v_{esc} = \\sqrt{\\frac{2\\mu}{R_{mean}}}

        Use :attr:`grav_param` (:math:`\\mu = GM`) rather than :attr:`mass` for
        orbital computations to avoid rounding errors from :math:`G`.

        Only one of ``mass`` and ``grav_param`` is needed: the missing one is
        derived through :math:`\\mu = GM` with :attr:`Constants.G
        <scarabaeus.utils.Constants.Constants.G>`. When both are given, both are
        kept as supplied. When neither is given, both stay ``None``, which is
        fine for a body used only as an origin or an ephemeris target but not
        for one that exerts gravity.

        Examples
        --------
        Create a CelestialBody object with numerical values that are half that of Jupiter.

        >>> import scarabaeus as scb
        >>> jov_constants = scb.constants.JUPITER
        >>> half_jupiter = scb.CelestialBody(
        ...     name        = "HALF JUPITER",
        ...     mass        = jov_constants.mass / 2,
        ...     mean_radius = jov_constants.mean_radius / 2,
        ...     grav_param  = jov_constants.GM / 2,
        ...     base_frame  = jov_constants.ref_frame,
        ...     spice_id    = 50000,
        ...     gs_list     = []
        ... )

        Create a body from its gravitational parameter alone; the mass is derived.

        >>> half_jupiter = scb.CelestialBody(
        ...     name       = "HALF JUPITER",
        ...     grav_param = jov_constants.GM / 2,
        ...     spice_id   = 50000,
        ... )
    """

    # region Constructor
    def __init__(
        self,
        name: str,
        mass: ArrayWUnits = None,
        mean_radius: ArrayWUnits = None,
        grav_param: ArrayWUnits = None,
        base_frame: Frame | str = None,
        spice_id: int | str = None,
        barycenter_id: int | str = None,
        barycenter_GM: ArrayWUnits = None,
        gs_list: list = [],
    ):
        # mu = G*M ties mass and GM together, so derive whichever one was left
        # out. The setter validates grav_param before it is divided by G.
        self.grav_param = grav_param  # input validation in setter
        if mass is None and grav_param is not None:
            mass = grav_param / constants.G
        elif grav_param is None and mass is not None:
            self.grav_param = self._grav_param_from_mass(mass)

        # parent class initialization
        super().__init__(name, spice_id, mass)

        # define class properties
        # Stored exactly as supplied — a Frame or a frame-name string. Building a
        # Frame here would call SPICE, and CelestialBody (notably via
        # from_constants) must stay constructible before any kernel is loaded.
        # The accessors below present a single, predictable type instead.
        self._base_frame = base_frame
        self._base_frame_obj = base_frame if isinstance(base_frame, Frame) else None
        self.mean_radius = mean_radius  # input validation in setter
        self.gs_list = gs_list
        self._barycenter_id = barycenter_id
        self._barycenter_GM = barycenter_GM

    # region Errors
    class UnresolvedBaseFrameError(ValueError):
        """The body's base frame could not be resolved into a :class:`Frame`."""

    class InvalidMassError(ValueError):
        """The mass is not a constant ArrayWUnits mass, so no GM can be derived from it."""

    # region Properties
    @property
    def base_frame(self) -> Frame:
        """ The reference frame attached to the CelestialBody.

        Always a :class:`~scarabaeus.Frame`. A name supplied as a string is
        resolved on first access and cached, so callers never have to test the
        type. Use :attr:`base_frame_name` when only the name is needed — that
        accessor never touches SPICE and never raises.

        Raises
        ------
        CelestialBody.UnresolvedBaseFrameError
            If the stored name is not a frame SPICE recognizes, which usually
            means the required kernels have not been loaded yet.
        """
        if self._base_frame_obj is None and self._base_frame is not None:
            try:
                self._base_frame_obj = Frame(str(self._base_frame))
            except Exception as error:
                raise self.UnresolvedBaseFrameError(
                    f"Base frame '{self._base_frame}' of body '{self.name}' could "
                    "not be resolved. Load the kernels that define it, or pass a "
                    "Frame object when constructing the CelestialBody."
                ) from error

        return self._base_frame_obj

    @property
    def base_frame_name(self) -> str | None:
        """ Name of the reference frame attached to the CelestialBody.

        The plain frame name, whether the frame was supplied as a
        :class:`~scarabaeus.Frame` or as a string. Resolves without SPICE, so it
        is safe before kernels are loaded and is what should be handed to
        SPICE-facing routines.
        """
        frame = self._base_frame

        if frame is None:
            return None

        return frame.name if isinstance(frame, Frame) else str(frame)

    @property
    def mean_radius(self) -> ArrayWUnits | None:
        """ The mean radius of the CelestialBody, if defined. """
        return self._mean_radius

    @mean_radius.setter
    def mean_radius(self, input_val):
        """ Set the mean radius, enforcing ``ArrayWUnits`` type and km units.

            Parameters
            ----------
            input_val : ArrayWUnits or None
                Mean radius in kilometres, or ``None`` if undefined.
        """
        ## input validation 
        # no radius given -> leave undefined
        if input_val is None:
            self._mean_radius = None
            return

        # ensure given value is an AWU
        if not isinstance(input_val, ArrayWUnits):
            not_awu_err = (
                "Given argument [mean_radius] should be of "
                f"type ArrayWUnits. Received: {type(input_val)}."
            )
            raise TypeError(not_awu_err)

        # ensure given value is in km
        if input_val.units != km:
            not_km_err = (
                "Mean radius units should be given in km. "
                f"Received: {input_val.units}."
            )
            raise ValueError(not_km_err)

        # valid input
        self._mean_radius = input_val

    @property
    def grav_param(self) -> ArrayWUnits | None:
        """ The gravitational parameter of the CelestialBody given,
            by convention, in units of :math:`\\frac{km^3}{s^2}`, if defined.
        """
        return self._grav_param

    @grav_param.setter
    def grav_param(self, input_val):
        """ Set the gravitational parameter, enforcing ``ArrayWUnits`` type and km^3/s^2 units.

            Parameters
            ----------
            input_val : ArrayWUnits or None
                Gravitational parameter in km^3/s^2 (``mu`` unit), or ``None`` if
                undefined.
        """
        ## input validation
        # no GM given -> leave undefined
        if input_val is None:
            self._grav_param = None
            return

        # ensure given value is an AWU
        if not isinstance(input_val, ArrayWUnits):
            not_awu_err = (
                "Given argument [grav_param] should be of "
                f"type ArrayWUnits. Received: {type(input_val)}"
            )
            raise TypeError(not_awu_err)

        # ensure given value is in correct units
        if input_val.units != mu:
            not_km_err = (
                "Gravitational parameter units should be given in "
                f"km^3*s^-2. Received: {input_val.units}."
            )
            raise ValueError(not_km_err)

        # valid input
        self._grav_param = input_val

    @property
    def barycenter_id(self) -> int:
        """ The SPICE ID of the object's barycenter. """
        return self._barycenter_id
    
    @property
    def barycenter_GM(self) -> ArrayWUnits | None:
        """ The gravitational parameter of the body's system barycenter, if defined. """
        return self._barycenter_GM

    @property
    def gs_list(self) -> list[GroundStation]:
        """ All GroundStation objects attached to the surface of the CelestialBody. """
        return self._gs_list

    @gs_list.setter
    def gs_list(self, input_val):
        """ Set the ground-station list, validating each element is a ``GroundStation``.

            Parameters
            ----------
            input_val : list of GroundStation or None
                Collection of ground stations to attach to this body.
        """
        ## input validation
        # must be of type Groundstation
        if input_val is not None:
            for gs in input_val:
                if not isinstance(gs, GroundStation):
                    raise TypeError("Inputs to 'gs_list' must be of type GroundStation.")

        # valid input
        self._gs_list = input_val

    # endregion Properties

    # region Methods
    def __repr__(self) -> str:
        """
        Overloading the printed representation of a CelestialBody object.
        """
        return self.name

    def disp_properties(self) -> None:
        """
        Displays all properties of the CelestialBody object in a readable format.
        """
        # format properties
        header = f'{self.name.center(len(self.name) + 55)}\n{"="*(len(self.name) + 55)}'
        # Stored profile, not mass(): a body may have no mass defined.
        mass = f"mass                     : {self.mass_profile}"
        mean_rad = f"mean_radius              : {self.mean_radius}"
        mu = f"gravitational parameter  : {self.grav_param}"
        # Name accessor: displaying a body must not depend on loaded kernels.
        frame = f"reference frame          : {self.base_frame_name}"
        spice_id = f"SPICE ID                 : {self.spice_id}"
        gs = f"attached ground stations : {self.gs_list}"

        # display them
        print(f"{header}\n{mass}\n{mean_rad}\n{mu}\n{frame}\n{spice_id}\n{gs}")

    def _grav_param_from_mass(self, mass: ArrayWUnits) -> ArrayWUnits:
        """ Derive the gravitational parameter from the mass, :math:`\\mu = GM`.

            Parameters
            ----------
            mass : ArrayWUnits
                Constant mass of the body, in any mass units.

            Returns
            -------
            grav_param : ArrayWUnits
                Gravitational parameter in :math:`\\frac{km^3}{s^2}`.

            Raises
            ------
            CelestialBody.InvalidMassError
                If ``mass`` is not an ``ArrayWUnits`` in mass units, e.g. a
                time-varying :class:`~scarabaeus.utils.SCBPolynomial.SCBPolynomial`
                profile.
        """
        if not isinstance(mass, ArrayWUnits) or mass.units.dimensions != kg.dimensions:
            raise self.InvalidMassError(
                "Cannot derive [grav_param] from [mass]: expected an ArrayWUnits in "
                f"mass units, received {mass!r}. Pass [grav_param] as well."
            )

        return constants.G * mass.convert_to(kg)

    @classmethod
    def from_constants(cls, cb_names: str | list[str]) -> Self | tuple[Self, ...]:
        """ Create a CelestialBody from a given name using its
            respective values as defined in :class:`~scarabaeus.utils.constants`.

            Parameters
            ----------
            cb_names : str or list of str
                The name or names of the desired body to create a
                CelestialBody for.

            Returns
            -------
            cb_out : CelestialBody
                A CelestialBody object initialized with all relevant values as
                defined by the given constant value.

            Raises
            ------
            not_def_err
                Raised when the given ``cb_name`` does not match an object in
                Scarabaeus' constants library.

            See Also
            --------
            scarabaeus.Constants : Scarabaeus' constants library.

            Notes
            -----
            This method performs the same functionality as manually accessing each
            of the pertinent constants values and assigning them by normal
            construction. It is a faster alternative when initalizing
            CelestialBody objects by values from constants.

            Examples
            --------
            Create Jupiter with constants values.

            >>> import scarabaeus as scb
            >>> jupiter = scb.CelestialBody.from_constants('JUPITER')

            Create the Mars system using constants values in one line.

            >>> mars, phobos, deimos = scb.CelestialBody.from_constants(['MARS', 'PHOBOS', 'DEIMOS'])

        """
        #------------------#
        # input validation #
        #------------------#
        # must be a string
        # if not isinstance(cb_name, str):
        #     not_str_err = (
        #         "Argument [ccb_name] must be an object of "
        #         f"type string. Received: {type(cb_name)}"
        #     )
        #     raise TypeError(not_str_err)
        if not isinstance(cb_names, list):
            cb_names = [cb_names]

        to_return = [None] * len(cb_names)

        for i, cb_name in enumerate(cb_names):
            # must be defined in Constants
            if not hasattr(constants, cb_name.upper()):
                not_def_err = (f"Requested constant {cb_name.upper()} not defined for CelestialBody "
                                "instantiation.")
                raise ValueError(not_def_err)

            # must be a valid constant for building
            req_const: PCInfo = getattr(constants, cb_name.upper())
            try:
                req_name = req_const.name
                req_mass = req_const.mass
                req_frme = req_const.ref_name
                req_rad = req_const.mean_radius
                req_mu = req_const.GM
                req_spice = req_const.body_center_id
                req_bary_id = req_const.barycenter_id
                req_bary_mu  =req_const.GM_barycenter

            except:
                # constant doesn't have enough information to build with -> raise error
                not_def_err = (f'Requested constant {cb_name} not defined for CelestialBody '
                            'instantiation.')
                raise ValueError(not_def_err)

            else:
                # --------------------------------------#
                # valid request -> build CB and return #
                # --------------------------------------#
                to_return[i] = cls(req_name, req_mass, req_rad, req_mu, req_frme, req_spice, req_bary_id, req_bary_mu)
        
        if len(to_return) == 1:
            # one body -> give as just object itself
            return to_return[0]
        else:
            # multiple bodies -> give as list
            return tuple(to_return)
        
    @classmethod
    def planets(cls) -> list[Self]:
        """ Create each of the eight planets as defined in constants.

            Convenience wrapper for :func:`from_constants`.

            Parameters
            ----------
            None

            Returns
            -------
            planets : list of CelestialBody
                List containing the eight planets with parameters defined by constants:

                * Mercury
                * Venus
                * Earth
                * Mars
                * Jupiter
                * Saturn
                * Uranus
                * Neptune
        """
        return cls.from_constants(['MERCURY', 'VENUS', 'EARTH', 'MARS',
                                   'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE'])

    def barycenter(self) -> Self:
        """ Stripped version of the """
        if not self._barycenter_id or not self._barycenter_GM:
            raise ValueError(f'Barycenter undefined for CelestialBody: {self}. Cannot create barycenter equivalent.')
        
        return CelestialBody(name = f'{self._name} Barycenter',
                             mass = self.mass(None),
                             mean_radius = ArrayWUnits(0, km),
                             grav_param = self._barycenter_GM,
                             base_frame = self._base_frame,
                             spice_id = None,
                             barycenter_id = self._barycenter_id)
