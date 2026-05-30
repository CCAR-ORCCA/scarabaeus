# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits

from dataclasses import dataclass

# units
kg, km, sec = Units.get_units(['kg', 'km', 'sec'])

# planetary constants data class
@dataclass
class PCInfo:
    """ Planetary Constants dataclass. """
    name: str
    """ Display name. """
    spice_name: str
    """ SPICE name. """
    ref_name: str
    """ Reference frame of the body. """
    body_center_id: int
    """ SPICE integer ID code for the body center. """
    barycenter_id: int
    """ SPICE integer ID code for the body barycenter. 
        Note that only major planets and the solar system 
        have an associated barycenter ID.
    """
    mass: ArrayWUnits
    """ Mass of the body. """
    GM: ArrayWUnits
    """ Gravitational parameter of the body. """
    mean_radius: ArrayWUnits
    """ Mean radius of the body. """
    equatorial_radius: ArrayWUnits
    """ Equatorial radius of the body. Undefined for 
        irregularly shaped bodies.
    """
    GM_barycenter: ArrayWUnits = None
    """ Gravitaitonal parameter of the body's system barycenter. 
        Note that, generally, only major planets and the solar system have 
        an associated barycenter.
    """
    primary_body: str = None
    """ The primary body if it has one. """

#----------------------#
# Define all constants #
#----------------------#
@dataclass
class Constants:
    """ Constant values defined within Scarabaeus.

        Notes
        -----
        Contains physical constants and planetary constants.
    """
    # region Physical
    G = ArrayWUnits(6.67430e-20, km**3 / (kg * sec**2))
    """ Universal gravitational constant.
    
        .. [1] NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?bg
    """

    c = ArrayWUnits(299792.458, km / sec)
    """ Speed of light in a vacuum.
    
        .. [1] NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?c
    """

    Stefan_Boltzmann = ArrayWUnits(5.670374419e-8, kg / sec**3)
    """ Stefan-Boltzmann constant.

        Note that the units of the Stefan-Boltzmann constant are W/(m^2*K^4). 
        :class:`~scarabaeus.units.ArrayWUnits` does not support units of Kelvin 
        (K), so the constant is returned in kg/s^3, or W/m^2, with the 1/K^4 omitted.

        .. [1] NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?sigma
    """

    srp_constant1 = ArrayWUnits((5.67e-8 * (6.96e5) ** 2 * (5778) ** 4) / c.values,
                                kg * km / sec**2)
    """ SRP Constant #1. """

    srp_constant2 = ArrayWUnits(101.9794375700e12, (km * sec * kg) / sec**3)
    """ SRP Constant #2. """

    gamma_GR = ArrayWUnits(1, None)
    """ Gamma constant for General Relativity.

        .. [1] `Moyer, T.D. (2000). Formulation for Observed and Computed 
            Values of Deep Space Network Data Types for Navigation. JPL Publication 
            00-7, Monograph 2, Deep Space Communications and Navigation Series, 
            pp. 4-13, Eq. 4-17.`_
    """

    L_tilde_GR = ArrayWUnits(1.480827e-8, None)
    """ L tilde constant for General Relativity.

        .. [1] `Moyer, T.D. (2000). Formulation for Observed and Computed 
            Values of Deep Space Network Data Types for Navigation. JPL Publication 
            00-7, Monograph 2, Deep Space Communications and Navigation Series, 
            pp. 4-13, Eq. 4-17.`_
    """

    # endregion Physical

    # region Major Planets
    SUN = PCInfo(name              = 'Sun',
                 spice_name        = 'SUN',
                 ref_name          = 'IAU_SUN',
                 body_center_id    = 10,
                 barycenter_id     = 0,    # SSB
                 mass              = ArrayWUnits(1.9884098713264208e30, kg),
                 GM                = ArrayWUnits(0.132712440041939e12, km**3 / sec**2),
                 mean_radius       = ArrayWUnits(696000.0, km),
                 equatorial_radius = None)
    """ Planetary constants and related information for the Sun, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`. 
        The mean and equatorial radii of the Sun are taken to be equivalent, 
        and so only ``mean_radius`` is given.
    """
    
    MERCURY = PCInfo(name              = 'Mercury',
                     spice_name        = 'MERCURY',
                     ref_name          = 'IAU_MERCURY',
                     body_center_id    = 199,
                     barycenter_id     = 1,
                     mass              = ArrayWUnits(3.300987369461969e23, kg),
                     GM                = ArrayWUnits(2.20317800000000e04, km**3 / sec**2),
                     GM_barycenter     = ArrayWUnits(0.2203178e5, km**3 / sec**2),
                     mean_radius       = ArrayWUnits(2439.7, km),
                     equatorial_radius = ArrayWUnits(2440.53, km))
    """ Planetary constants and related information for Mercury, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    VENUS = PCInfo(name              = 'Venus',
                   spice_name        = 'VENUS',
                   ref_name          = 'IAU_VENUS',
                   body_center_id    = 299,
                   barycenter_id     = 2,
                   mass              = ArrayWUnits(4.867305814842005e24, kg),
                   GM                = ArrayWUnits(3.2485859200000000e05, km**3 / sec**2),
                   GM_barycenter     = ArrayWUnits(0.324858592e6, km**3 / sec**2),
                   mean_radius       = ArrayWUnits(6051.8, km),
                   equatorial_radius = ArrayWUnits(6051.8, km))
    """ Planetary constants and related information for Venus, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    EARTH = PCInfo(name              = 'Earth',
                   spice_name        = 'EARTH',
                   ref_name          = 'ITRF93',
                   body_center_id    = 399,
                   barycenter_id     = 3,
                   mass              = ArrayWUnits(5.972168398724898e24, kg),
                   GM                = ArrayWUnits(398600.435436096, km**3 / sec**2),
                   GM_barycenter     = ArrayWUnits(403503.23550226, km**3 / sec**2),
                   mean_radius       = ArrayWUnits(6371.0084, km),
                   equatorial_radius = ArrayWUnits(6378.1366, km))
    """ Planetary constants and related information for Earth, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    MARS = PCInfo(name              = 'Mars',
                  spice_name        = 'MARS',
                  ref_name          = 'IAU_MARS',
                  body_center_id    = 499,
                  barycenter_id     = 4,
                  mass              = ArrayWUnits(6.416908682663213e23, kg),
                  GM                = ArrayWUnits(42828.37362069909, km**3 / sec**2),
                  GM_barycenter     = ArrayWUnits(4.28283752140000e04, km**3 / sec**2),
                  mean_radius       = ArrayWUnits(3389.50, km),
                  equatorial_radius = ArrayWUnits(3396.19, km))
    """ Planetary constants and related information for Mars, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    JUPITER = PCInfo(name              = 'Jupiter',
                     spice_name        = 'JUPITER',
                     ref_name          = 'IAU_JUPITER',
                     body_center_id    = 599,
                     barycenter_id     = 5,
                     mass              = ArrayWUnits(1.898124671078627e27, kg),
                     GM                = ArrayWUnits(126686534.92180079, km**3 / sec**2),
                     GM_barycenter     = ArrayWUnits(1.26712764800000e08, km**3 / sec**2),
                     mean_radius       = ArrayWUnits(69911, km),
                     equatorial_radius = ArrayWUnits(71492.0, km))
    """ Planetary constants and related information for Jupiter, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    SATURN = PCInfo(name              = 'Saturn',
                    spice_name        = 'SATURN',
                    ref_name          = 'IAU_SATURN',
                    body_center_id    = 699,
                    barycenter_id     = 6,
                    mass              = ArrayWUnits(5.683173890692993e26, kg),
                    GM                = ArrayWUnits(37931207.49865224, km**3 / sec**2),
                    GM_barycenter     = ArrayWUnits(3.79405852000000e07, km**3 / sec**2),
                    mean_radius       = ArrayWUnits(58232.0, km),
                    equatorial_radius = ArrayWUnits(60268.0, km))
    """ Planetary constants and related information for Saturn, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    URANUS = PCInfo(name              = 'Uranus',
                    spice_name        = 'URANUS',
                    ref_name          = 'IAU_URANUS',
                    body_center_id    = 799,
                    barycenter_id     = 7,
                    mass              = ArrayWUnits(8.680987253013813e25, kg),
                    GM                = ArrayWUnits(5793951.322279009, km**3 / sec**2),
                    GM_barycenter     = ArrayWUnits(05.79454860000001e06, km**3 / sec**2),
                    mean_radius       = ArrayWUnits(25362, km),
                    equatorial_radius = ArrayWUnits(25559.0, km))
    """ Planetary constants and related information for Uranus, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    NEPTUNE = PCInfo(name              = 'Neptune',
                     spice_name        = 'NEPTUNE',
                     ref_name          = 'IAU_NEPTUNE',
                     body_center_id    = 899,
                     barycenter_id     = 8,
                     mass              = ArrayWUnits(1.0240923396370664e26, kg),
                     GM                = ArrayWUnits(6835099.502439672, km**3 / sec**2),
                     GM_barycenter     = ArrayWUnits(6.83652710058002e06, km**3 / sec**2),
                     mean_radius       = ArrayWUnits(24622, km),
                     equatorial_radius = ArrayWUnits(24764.0, km))
    """ Planetary constants and related information for Neptune, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """
    
    # endregion Major Planets

    # region Minor Planets/Moons
    PLUTO = PCInfo(name              = 'Pluto',
                   spice_name        = 'PLUTO',
                   ref_name          = 'IAU_PLUTO',
                   body_center_id    = 999,
                   barycenter_id     = 9,
                   mass              = ArrayWUnits(1.3029288730816338e22, kg),
                   GM                = ArrayWUnits(869.6138177608749, km**3 / sec**2),
                   GM_barycenter     = ArrayWUnits(0.977000000000001e3, km**3 / sec**2),
                   mean_radius       = ArrayWUnits(1195.0, km),
                   equatorial_radius = ArrayWUnits(1195.0, km))
    """ Planetary constants and related information for Pluto, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    MOON = PCInfo(name              = 'Moon',
                  spice_name        = 'MOON',
                  ref_name          = 'IAU_MOON',
                  body_center_id    = 301,
                  barycenter_id     = None,
                  mass              = ArrayWUnits(7.345789170645305e22, kg),
                  GM                = ArrayWUnits(4.90280006616380e03, km**3 / sec**2),
                  mean_radius       = ArrayWUnits(1737.4, km),
                  equatorial_radius = ArrayWUnits(1738.14, km),
                  primary_body      = 'Earth')
    """ Planetary constants and related information for the Moon, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    PHOBOS = PCInfo(name              = 'Phobos',
                    spice_name        = 'PHOBOS',
                    ref_name          = 'IAU_PHOBOS',
                    body_center_id    = 401,
                    barycenter_id     = None,
                    mass              = ArrayWUnits(1.0619160161956238e16, kg),
                    GM                = ArrayWUnits(0.0007087546066894452, km**3 / sec**2),
                    mean_radius       = ArrayWUnits(11.4, km),
                    equatorial_radius = None,
                    primary_body      = 'Mars')
    """ Planetary constants and related information for Phobos, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """
    
    DEIMOS = PCInfo(name              = 'Deimos',
                    spice_name        = 'DEIMOS',
                    ref_name          = 'IAU_DEIMOS',
                    body_center_id    = 402,
                    barycenter_id     = None,
                    mass              = ArrayWUnits(1440685861906164.2, kg),
                    GM                = ArrayWUnits(9.615569648120313e-05, km**3 / sec**2),
                    mean_radius       = ArrayWUnits(6.0, km),
                    equatorial_radius = None,
                    primary_body      = 'Mars')
    """ Planetary constants and related information for Deimos, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.
    """

    JUSTITIA = PCInfo(name              = 'Justitia',
                      spice_name        = 'JUSTITIA',
                      ref_name          = None,
                      body_center_id    = 20000269,
                      barycenter_id     = None,
                      mass              = ArrayWUnits(7 * 1e-3, km**3 / sec**2) / G,
                      GM                = ArrayWUnits(7 * 1e-3, km**3 / sec**2),
                      mean_radius       = ArrayWUnits(50.728, km),
                      equatorial_radius = None,
                      primary_body      = 'SUN')
    """ Planetary constants and related information for Justitia, given as 
        :class:`~scarabaeus.utils.Constants.PCInfo`.

        .. [1] JPL, https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=20000269
    """
    
    DANIEL_SCHEERES = PCInfo(name              = 'Scheeres',
                             spice_name        = 'SCHEERES',
                             ref_name          = None,
                             body_center_id    = 20008887,
                             barycenter_id     = None,
                             mass              = ArrayWUnits(6.24e14, kg),
                             GM                = ArrayWUnits(4.16e-5, km**3 / sec**2),
                             mean_radius       = ArrayWUnits(3.907, km),
                             equatorial_radius = None,
                             primary_body      = 'SUN')
    """ Planetary constants and related information for asteroid (8887) Scheeres,
        given as :class:`~scarabaeus.utils.Constants.PCInfo`.

        Named in honor of Prof. Daniel J. Scheeres (ORCCA, CU Boulder), pioneer
        of small-body gravitational dynamics and orbital mechanics.
        Mean radius from NEOWISE thermal survey (diameter 7.814 ± 0.071 km,
        albedo 0.187 ± 0.014). GM estimated from assumed S-type bulk density
        of 2500 kg/m³; no spacecraft measurement exists.

        .. [1] JPL Small-Body Database, https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=8887
    """

    JAY_MCMAHON = PCInfo(name              = 'McMahon',
                         spice_name        = 'MCMAHON',
                         ref_name          = None,
                         body_center_id    = 20046829,
                         barycenter_id     = None,
                         mass              = ArrayWUnits(4.29e13, kg),
                         GM                = ArrayWUnits(2.86e-6, km**3 / sec**2),
                         mean_radius       = ArrayWUnits(1.6, km),
                         equatorial_radius = None,
                         primary_body      = 'SUN')
    """ Planetary constants and related information for asteroid (46829) McMahon,
        given as :class:`~scarabaeus.utils.Constants.PCInfo`.

        Named in honor of Prof. Jay W. McMahon (ORCCA, CU Boulder), co-developer
        of Scarabaeus. A main-belt binary system (1998 OS14); the radius is
        estimated from absolute magnitude H = 15.10 assuming S-type albedo ~0.15.
        GM estimated from assumed bulk density of 2500 kg/m³; no spacecraft
        measurement exists.

        .. [1] JPL Small-Body Database, https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=46829
    """

    MATTIA_PUGLIATTI = PCInfo(name              = 'Mattia Pugliatti',
                              spice_name        = 'MATTIA_PUGLIATTI',
                              ref_name          = None,
                              body_center_id    = 2042042,
                              barycenter_id     = None,
                              mass              = ArrayWUnits(0.042, km**3 / sec**2) / G,
                              GM                = ArrayWUnits(0.042, km**3 / sec**2),
                              mean_radius       = ArrayWUnits(4.2, km),
                              equatorial_radius = None,
                              primary_body      = 'SUN')
    """ Planetary constants and related information for asteroid Mattia Pugliatti,
        given as :class:`~scarabaeus.utils.Constants.PCInfo`.

        A fictitious S-type asteroid in the main belt, named in honor of
        Dr. Mattia Pugliatti (ORCCA, CU Boulder), co-developer of Scarabaeus.
        All physical parameters are suspiciously close to multiples of 42.
    """

    # endregion Minor Planets/Moons