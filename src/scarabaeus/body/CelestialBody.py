# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import constants, PCInfo, Body, Units, ArrayWUnits, Frame, GroundStation
import sys, inspect

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# define units
kg, km, sec, mu = Units.get_units(["kg", "km", "sec", "mu"])

# class definition
class CelestialBody(Body):
    """ Defines an astronomical body such as a planet, moon, asteroid, or comet.

        Parameters
        ----------
        name : str
            The written name of the CelestialBody.

        mass : ArrayWUnits
            The mass of the CelestialBody. Recommend expressed in units of kilograms.

        mean_radius : ArrayWUnits
            The mean radius of the CelestialBody. Recommend expressed in units of kilometers.

        grav_param : ArrayWUnits
            The gravitational parameter of the CelestialBody. Recommend expressed in units
            of :math:`\\frac{km^3}{s^2}`.

        base_frame : Frame
            The body-centered, body-fixed reference frame attached to the CelestialBody.

        spice_id : int or str, optional
            The NAIF ID code of the object. May be given as an integer ID or as a string.
            Defaults to ``None``.

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
    """

    # region Constructor
    def __init__(
        self,
        name: str,
        mass: ArrayWUnits,
        mean_radius: ArrayWUnits,
        grav_param: ArrayWUnits,
        base_frame: Frame,
        spice_id: int | str = None,
        barycenter_id: int | str = None,
        gs_list: list = [],
    ):
        # parent class initialization
        super().__init__(name, spice_id, mass)

        # define class properties
        self._base_frame = base_frame
        self.mean_radius = mean_radius  # input validation in setter
        self.grav_param = grav_param  # input validation in setter
        self.gs_list = gs_list
        self.barycenter_id = barycenter_id

    # region Properties
    @property
    def base_frame(self) -> Frame:
        """ The reference frame attached to the CelestialBody. """
        return self._base_frame

    @property
    def mean_radius(self) -> ArrayWUnits:
        """ The mean radius of the CelestialBody. """
        return self._mean_radius

    @mean_radius.setter
    def mean_radius(self, input_val):
        """ Set the mean radius, enforcing ``ArrayWUnits`` type and km units.

            Parameters
            ----------
            input_val : ArrayWUnits
                Mean radius in kilometres.
        """
        ## input validation 
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
    def grav_param(self) -> ArrayWUnits:
        """ The gravitational parameter of the CelestialBody given,
            by convention, in units of :math:`\\frac{km^3}{s^2}`.
        """
        return self._grav_param

    @grav_param.setter
    def grav_param(self, input_val):
        """ Set the gravitational parameter, enforcing ``ArrayWUnits`` type and km³/s² units.

            Parameters
            ----------
            input_val : ArrayWUnits
                Gravitational parameter in km³/s² (``mu`` unit).
        """
        ## input validation
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
        mass = f"mass                     : {self.mass}"
        mean_rad = f"mean_radius              : {self.mean_radius}"
        mu = f"gravitational parameter  : {self.grav_param}"
        frame = f"reference frame          : {self.base_frame}"
        spice_id = f"SPICE ID                 : {self.spice_id}"
        gs = f"attached ground stations : {self.gs_list}"

        # display them
        print(f"{header}\n{mass}\n{mean_rad}\n{mu}\n{frame}\n{spice_id}\n{gs}")

    @classmethod
    def from_constants(cls, cb_name: str) -> Self:
        """
        Creates a CelestialBody object from a given name using its
        respective values as defined in :class:`~scarabaeus.utils.constants`.

        Parameters
        ----------
        cb_name : str
            The name of the desired celestial body to create a
            Scarabaeus CelestialBody object for.

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
        """
        #------------------#
        # input validation #
        #------------------#
        # must be a string
        if not isinstance(cb_name, str):
            not_str_err = (
                "Argument [ccb_name] must be an object of "
                f"type string. Received: {type(cb_name)}"
            )
            raise TypeError(not_str_err)

        # must be defined in Constants
        if not hasattr(constants, cb_name.upper()):
            not_def_err = (f"Requested constant {cb_name} not defined for CelestialBody "
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

        except:
            # constant doesn't have enough information to build with -> raise error
            not_def_err = (f'Requested constant {cb_name} not defined for CelestialBody '
                           'instantiation.')
            raise ValueError(not_def_err)

        else:
            # --------------------------------------#
            # valid request -> build CB and return #
            # --------------------------------------#
            return cls(req_name, req_mass, req_rad, req_mu, req_frme, req_spice, req_bary_id)