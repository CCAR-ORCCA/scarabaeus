# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits

from dataclasses import dataclass

# units
kg, km, sec, m, rad, deg, day, minute, hr = Units.get_units(
    ['kg', 'km', 'sec', 'm', 'rad', 'deg', 'day', 'min', 'hr']
)

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

# solid tide constants data class
@dataclass(frozen=True)
class SolidTideInfo:
    """ Solid-tide constants of a body, used by
        :class:`~scarabaeus.body.SolidTide.SolidTide`.

        The first four fields are the nominal tidal response any body needs.
        The rest are the numbers of the IERS Conventions (2010) station
        displacement routine DEHANTTIDEINEL, Chapter 7, and exist for Earth
        only. That routine's reference radius is Earth's equatorial radius,
        ``Constants.EARTH.equatorial_radius``.
    """
    love_h2: float
    """ Degree-2 radial Love number, :math:`h^{(0)}`. """
    shida_l2: float
    """ Degree-2 transverse Shida number, :math:`l^{(0)}`. """
    love_k: dict
    """ Potential Love numbers :math:`k_{nm}` keyed by ``(degree, order)``.
        They drive the change of the external field, not the surface motion.
    """
    tide_bodies: tuple
    """ Tide-raising bodies, by SPICE name. """
    love_h3: float = None
    """ Degree-3 radial Love number. """
    shida_l3: float = None
    """ Degree-3 transverse Shida number. """
    love_h2_latitude: float = None
    """ :math:`h^{(2)}` in :math:`h_2 = h^{(0)} + h^{(2)}(3\\sin^2\\phi - 1)/2`,
        the latitude dependence of the degree-2 Love number.
    """
    shida_l2_latitude: float = None
    """ :math:`l^{(2)}`, the same latitude dependence of the Shida number. """
    out_of_phase_diurnal: tuple = None
    """ :math:`(h^I, l^I)` of the diurnal band, mantle anelasticity. """
    out_of_phase_semidiurnal: tuple = None
    """ :math:`(h^I, l^I)` of the semidiurnal band. """
    shida_l1: tuple = None
    """ :math:`l^{(1)}` of the diurnal and semidiurnal bands, in that order. """
    mass_ratio_sun: float = None
    """ Sun-to-body mass ratio the routine scales by; its own value, not a
        ratio of this module's GMs.
    """
    mass_ratio_moon: float = None
    """ Moon-to-body mass ratio the routine scales by. """
    step2_diurnal: tuple = None
    """ Frequency-dependent corrections, diurnal band, Table 7.3a. Each row:
        multipliers of the arguments (s, h, p, N', ps), then radial in and out
        of phase, transverse in and out of phase, in mm.
    """
    step2_long_period: tuple = None
    """ Frequency-dependent corrections, long-period band, Table 7.3b, in the
        same column layout as :attr:`step2_diurnal`.
    """
    step2_arguments: dict = None
    """ Polynomials of the step-2 arguments in degrees, coefficients in
        ascending powers of TT Julian centuries from J2000: the mean
        longitudes of the Moon (``s``, before precession), Sun (``h``), lunar
        perigee (``p``), the negative lunar node (``N'``), solar perigee
        (``ps``), general precession in longitude (``pr``), and Greenwich mean
        sidereal time at 0h UT (``GMST``).
    """
    permanent_tide_radial: tuple = None
    """ :math:`(a, b)` of :math:`\\Delta r = (a + b P_2) P_2`, m, IERS
        eq. 7.14a, with :math:`P_2 = (3\\sin^2\\phi - 1)/2`.
    """
    permanent_tide_north: tuple = None
    """ :math:`(a, b)` of :math:`\\Delta n = (a + b P_2) \\sin 2\\phi`, m, IERS
        eq. 7.14b.
    """

# pole tide constants data class
@dataclass(frozen=True)
class PoleTideInfo:
    """ Pole-tide constants of a body, used by
        :class:`~scarabaeus.body.PoleTide.PoleTide`.

        The mean rotation pole is fixed by the degree-2 order-1 harmonics,
        :math:`\\bar X = -\\bar C_{21}/(\\sqrt3\\,\\bar J_2)` and
        :math:`\\bar Y = \\bar S_{21}/(\\sqrt3\\,\\bar J_2)`, as in Moyer.
    """
    love_h2: float
    """ Degree-2 radial Love number of the pole-tide response. """
    shida_l2: float
    """ Degree-2 transverse Shida number of the pole-tide response. """
    c21: float
    """ Normalized :math:`\\bar C_{21}` defining the mean pole. """
    s21: float
    """ Normalized :math:`\\bar S_{21}` defining the mean pole. """
    j2: float
    """ Normalized second zonal harmonic :math:`\\bar J_2 = -\\bar C_{20}`. """

# ocean loading constants data class
@dataclass(frozen=True)
class OceanLoadingInfo:
    """ Tidal constituents of the ocean-loading displacement, used by
        :class:`~scarabaeus.body.OceanLoading.OceanLoading`.

        The constituents are those of a Scherneck (Onsala) BLQ table, in its
        column order. The astronomical argument of constituent :math:`i` is
        :math:`\\sum_j n^i_j\\beta_j + \\chi_i` over the Doodson variables
        :math:`(\\tau, s, h, p, N', p_s)`.
    """
    tides: tuple
    """ Constituent names, in BLQ column order. """
    doodson: tuple
    """ Doodson multipliers :math:`n^i_j` of :math:`(\\tau, s, h, p, N', p_s)`,
        one row per constituent.
    """
    schwiderski_phase: tuple
    """ Schwiderski phase :math:`\\chi_i` of each constituent, degrees. """
    fundamental_arguments: dict
    """ Delaunay arguments ``l``, ``l'``, ``F``, ``D``, ``Omega`` in arcseconds,
        coefficients in ascending powers of TDB Julian centuries from J2000.
    """
    nodal_factor: tuple
    """ Nodal amplitude factor :math:`f = \\sum_k a_k \\cos kN`, one row
        :math:`(a_0, a_1, a_2, a_3)` per constituent, where :math:`N` is the
        longitude of the Moon's ascending node.
    """
    nodal_angle: tuple
    """ Nodal phase :math:`u = \\sum_k b_k \\sin kN` in degrees, one row
        :math:`(b_1, b_2, b_3)` per constituent.
    """
    gmst_minus_era: tuple
    """ GMST minus the Earth rotation angle, arcseconds, coefficients in
        ascending powers of TT Julian centuries from J2000.
    """

# DSN frequency-band dataclass
@dataclass(frozen=True)
class DSNBand:
    """ The band-dependent numbers a two-way radiometric observable needs.

        One of these per DSN frequency band. Collected here so the turnaround
        ratio and range-unit scale are defined once per band rather than as
        literals in each measurement model.

        Attributes
        ----------
        name : str
            Band letter, ``'X'`` or ``'S'``.
        uplink_multiplier : int
            Multiple of the DSN reference frequency the uplink carrier sits at:
            749 for X band, 221 for S band.
        downlink_multiplier : int
            Same for the downlink: 880 for X band, 240 for S band.
        turnaround_num, turnaround_den : int
            The transponder turnaround ratio :math:`M_2 = f_{down}/f_{up}` as a
            fraction, 880/749 for X-up/X-down and 240/221 for S-up/S-down. Kept
            as two integers because that is how the tracking files carry it and
            because the ratio is exact.
        exciter_scalar_num, exciter_scalar_den : float
            The exciter scalars a tracking file carries, from which the range
            unit follows as ``((den / num) / 16) ** -1``. That works out to
            ``uplink_multiplier_S / (2 * uplink_multiplier)``, i.e. 221/1498 for
            X band and 1/2 for S band, so that the range unit is the same
            physical quantity in either band: the product of the range unit and
            the uplink frequency is band-invariant at 1.055181 GHz.
        ru_reference_frequency : ArrayWUnits
            Reference frequency used to convert a spacecraft transponder delay
            from range units to seconds, as ``delay_RU / (range_unit *
            ru_reference_frequency)`` [1/s].

            This is NOT the uplink frequency, and substituting the uplink
            frequency is measurably wrong: doing so makes the delay 18% longer,
            1.4 km of two-way range, and drives the sequential-ranging residuals
            on real tracking data from 55-206 RU up to 2330-2694 RU. The X-band
            value is the one the models used as a literal and is what fits the
            data; why a downlink-scale reference is the right one here is an
            open question, and the S-band entry simply scales it by the
            downlink multipliers, 240/880, so it is unverified against data.
        uplink_frequency : ArrayWUnits
            Representative uplink carrier frequency [1/s]. The two defaults are
            one band apart in the same channel, 7.1523135969977 GHz for X band
            and 2.110362223 GHz for S band, their ratio being exactly
            749/221. A real pass carries its own, so this only seeds simulated
            auxiliary data.
    """
    name: str
    uplink_multiplier: int
    downlink_multiplier: int
    turnaround_num: int
    turnaround_den: int
    exciter_scalar_num: float
    exciter_scalar_den: float
    uplink_frequency: ArrayWUnits
    ru_reference_frequency: ArrayWUnits

    @property
    def range_unit(self) -> float:
        """ The range unit implied by the exciter scalars, ``RU``.

            One range unit of observed count corresponds to a round-trip phase
            of this many cycles of the uplink carrier.
        """
        return ((self.exciter_scalar_den / self.exciter_scalar_num) / 16) ** -1


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

    zero_Celsius = ArrayWUnits(273.15, None)
    """ Temperature of 0 degrees Celsius, in kelvin.

        :class:`~scarabaeus.units.ArrayWUnits` does not support units of Kelvin
        (K), so the value is stored unitless. A Celsius temperature :math:`t`
        is :math:`t + 273.15` K.

        .. [1] BIPM, The International System of Units (SI), 9th ed., 2019, Sec. 2.3.1.
    """

    srp_constant1 = ArrayWUnits((5.67e-8 * (6.96e5) ** 2 * (5778) ** 4) / c.values,
                                kg * km / sec**2)
    """ SRP Constant #1. """

    srp_constant2 = ArrayWUnits(101.9794375700e12, (km * sec * kg) / sec**3)
    """ SRP Constant #2. """

    earth_rotation_rate = ArrayWUnits(7.292115e-5, sec**-1)
    """ Earth's nominal mean angular velocity.

        The IERS conventional value, used where a uniform rotation rate is
        adequate — converting a UT1 offset into an angle, for instance. Actual
        Earth rotation varies at the 1e-8 level and a kernel's own orientation
        model carries that; this is not a substitute for it.

        .. [1] IERS Conventions (2010), IERS Technical Note No. 36, Table 1.1.
    """

    gamma_GR = ArrayWUnits(1, None)
    """ Gamma constant for General Relativity.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed 
            Values of Deep Space Network Data Types for Navigation. JPL Publication 
            00-7, Monograph 2, Deep Space Communications and Navigation Series, 
            pp. 4-13, Eq. 4-17.
    """

    L_tilde_GR = ArrayWUnits(1.480827e-8, None)
    """ L tilde constant for General Relativity.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed
            Values of Deep Space Network Data Types for Navigation. JPL Publication
            00-7, Monograph 2, Deep Space Communications and Navigation Series,
            pp. 4-13, Eq. 4-17.
    """

    TT_minus_TAI = ArrayWUnits(32.184, sec)
    """ Terrestrial Time minus International Atomic Time.

        Exact by definition. It is ET - TAI in the local geocentric space-time
        frame, where the coordinate time is TT.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed
            Values of Deep Space Network Data Types for Navigation. JPL Publication
            00-7, Monograph 2, Deep Space Communications and Navigation Series,
            Sec. 2.3.
    """

    omega_Earth = ArrayWUnits(7.2921150e-5, rad / sec)
    """ Nominal mean angular velocity of the Earth.

        Used by the EOP partials to turn a UT1 correction into a station
        displacement along the direction of the Earth's rotation.

        .. [1] Petit, G., Luzum, B. (eds.) (2010). IERS Conventions (2010),
            IERS Technical Note 36, Table 1.1.
    """

    # endregion Physical

    # region Solid tide
    EARTH_SOLID_TIDE = SolidTideInfo(
        love_h2                  = 0.6078,
        shida_l2                 = 0.0847,
        love_k                   = {(2, 0): 0.29525, (2, 1): 0.29470, (2, 2): 0.29801,
                                    (3, 0): 0.09300, (3, 1): 0.09300, (3, 2): 0.09300,
                                    (3, 3): 0.09400},
        tide_bodies              = ('MOON', 'SUN'),
        love_h3                  = 0.292,
        shida_l3                 = 0.015,
        love_h2_latitude         = -0.0006,
        shida_l2_latitude        = 0.0002,
        out_of_phase_diurnal     = (-0.0025, -0.0007),
        out_of_phase_semidiurnal = (-0.0022, -0.0007),
        shida_l1                 = (0.0012, 0.0024),
        mass_ratio_sun           = 332946.0482,
        mass_ratio_moon          = 0.0123000371,
        step2_diurnal            = (
            (-3, 0, 2, 0, 0, -0.01, 0, 0, 0), (-3, 2, 0, 0, 0, -0.01, 0, 0, 0),
            (-2, 0, 1, -1, 0, -0.02, 0, 0, 0), (-2, 0, 1, 0, 0, -0.08, 0, -0.01, 0.01),
            (-2, 2, -1, 0, 0, -0.02, 0, 0, 0), (-1, 0, 0, -1, 0, -0.10, 0, 0, 0),
            (-1, 0, 0, 0, 0, -0.51, 0, -0.02, 0.03), (-1, 2, 0, 0, 0, 0.01, 0, 0, 0),
            (0, -2, 1, 0, 0, 0.01, 0, 0, 0), (0, 0, -1, 0, 0, 0.02, 0, 0, 0),
            (0, 0, 1, 0, 0, 0.06, 0, 0, 0), (0, 0, 1, 1, 0, 0.01, 0, 0, 0),
            (0, 2, -1, 0, 0, 0.01, 0, 0, 0), (1, -3, 0, 0, 1, -0.06, 0, 0, 0),
            (1, -2, 0, -1, 0, 0.01, 0, 0, 0), (1, -2, 0, 0, 0, -1.23, -0.07, 0.06, 0.01),
            (1, -1, 0, 0, -1, 0.02, 0, 0, 0), (1, -1, 0, 0, 1, 0.04, 0, 0, 0),
            (1, 0, 0, -1, 0, -0.22, 0.01, 0.01, 0), (1, 0, 0, 0, 0, 12.00, -0.80, -0.67, -0.03),
            (1, 0, 0, 1, 0, 1.73, -0.12, -0.10, 0), (1, 0, 0, 2, 0, -0.04, 0, 0, 0),
            (1, 1, 0, 0, -1, -0.50, -0.01, 0.03, 0), (1, 1, 0, 0, 1, 0.01, 0, 0, 0),
            (0, 1, 0, 1, -1, -0.01, 0, 0, 0), (1, 2, -2, 0, 0, -0.01, 0, 0, 0),
            (1, 2, 0, 0, 0, -0.11, 0.01, 0.01, 0), (2, -2, 1, 0, 0, -0.01, 0, 0, 0),
            (2, 0, -1, 0, 0, -0.02, 0, 0, 0), (3, 0, 0, 0, 0, 0, 0, 0, 0),
            (3, 0, 0, 1, 0, 0, 0, 0, 0)),
        step2_long_period        = (
            (0, 0, 0, 1, 0, 0.47, 0.16, 0.23, 0.07),
            (0, 2, 0, 0, 0, -0.20, -0.11, -0.12, -0.05),
            (1, 0, -1, 0, 0, -0.11, -0.09, -0.08, -0.04),
            (2, 0, 0, 0, 0, -0.13, -0.15, -0.11, -0.07),
            (2, 0, 0, 1, 0, -0.05, -0.06, -0.05, -0.03)),
        step2_arguments          = {
            's':    (218.31664563, 481267.88194, -0.0014663889, 0.00000185139),
            'h':    (280.46645, 36000.7697489, 0.00030322222, 0.000000020, -0.00000000654),
            'p':    (83.35324312, 4069.01363525, -0.01032172222, -0.0000124991, 0.00000005263),
            "N'":   (234.95544499, 1934.13626197, -0.00207561111, -0.00000213944, 0.00000001650),
            'ps':   (282.93734098, 1.71945766667, 0.00045688889, -0.00000001778, -0.00000000334),
            'pr':   (0.0, 1.396971278, 0.000308889, 0.000000021, 0.000000007),
            'GMST': (280.4606184, 36000.7700536, 0.00038793, -0.0000000258)},
        permanent_tide_radial    = (-0.1206, 0.0001),
        permanent_tide_north     = (-0.0252, -0.0001))
    """ Earth's solid-tide constants, given as
        :class:`~scarabaeus.utils.Constants.SolidTideInfo`: the nominal Love
        and Shida numbers of Chapter 7, the potential Love numbers of Table
        6.3, and every number of the DEHANTTIDEINEL routine.

        .. [1] IERS Conventions (2010), IERS Technical Note No. 36, Chapters 6
            and 7, and the routine DEHANTTIDEINEL with ST1IDIU, ST1ISEM, ST1L1,
            STEP2DIU and STEP2LON (iers-conventions.obspm.fr).
    """

    EARTH_POLE_TIDE = PoleTideInfo(
        love_h2  = 0.6090,
        shida_l2 = 0.0852,
        c21      = -0.17e-9,
        s21      = 1.19e-9,
        j2       = 4.8417e-4)
    """ Earth's pole-tide constants, given as
        :class:`~scarabaeus.utils.Constants.PoleTideInfo`: the Love numbers
        and mean-pole harmonics of the JPL navigation formulation.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed
            Values of Deep Space Network Data Types for Navigation. JPL
            Publication 00-7, Section 5.
    """

    EARTH_OCEAN_LOADING = OceanLoadingInfo(
        tides                 = ('M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1',
                                 'Mf', 'Mm', 'Ssa'),
        doodson               = ((2, 0, 0, 0, 0, 0), (2, 2, -2, 0, 0, 0),
                                 (2, -1, 0, 1, 0, 0), (2, 2, 0, 0, 0, 0),
                                 (1, 1, 0, 0, 0, 0), (1, -1, 0, 0, 0, 0),
                                 (1, 1, -2, 0, 0, 0), (1, -2, 0, 1, 0, 0),
                                 (0, 2, 0, 0, 0, 0), (0, 1, 0, -1, 0, 0),
                                 (0, 0, 2, 0, 0, 0)),
        schwiderski_phase     = (0.0, 0.0, 0.0, 0.0, 90.0, -90.0, -90.0, -90.0,
                                 0.0, 0.0, 0.0),
        fundamental_arguments = {
            'l':     (485866.733, 1717915922.633, 31.310, 0.064),
            "l'":    (1287099.804, 129596581.244, -0.577, -0.012),
            'F':     (335778.877, 1739527263.137, -13.257, 0.011),
            'D':     (1072261.307, 1602961601.328, -6.891, 0.019),
            'Omega': (450160.280, -6962890.539, 7.455, 0.008)},
        # M2 and N2, K1, O1 and Q1, Mf, Mm and K2 follow the lunar node;
        # S2, P1 and Ssa are solar and carry none.
        nodal_factor          = ((1.0004, -0.0373, 0.0002, 0.0),
                                 (1.0, 0.0, 0.0, 0.0),
                                 (1.0004, -0.0373, 0.0002, 0.0),
                                 (1.0241, 0.2863, 0.0083, -0.0015),
                                 (1.0060, 0.1150, -0.0088, 0.0006),
                                 (1.0089, 0.1871, -0.0147, 0.0014),
                                 (1.0, 0.0, 0.0, 0.0),
                                 (1.0089, 0.1871, -0.0147, 0.0014),
                                 (1.0429, 0.4135, -0.0040, 0.0),
                                 (1.0000, -0.1300, 0.0013, 0.0),
                                 (1.0, 0.0, 0.0, 0.0)),
        nodal_angle           = ((-2.14, 0.0, 0.0),
                                 (0.0, 0.0, 0.0),
                                 (-2.14, 0.0, 0.0),
                                 (-17.74, 0.68, -0.04),
                                 (-8.86, 0.68, -0.07),
                                 (10.80, -1.34, 0.19),
                                 (0.0, 0.0, 0.0),
                                 (10.80, -1.34, 0.19),
                                 (-23.74, 2.68, -0.38),
                                 (0.0, 0.0, 0.0),
                                 (0.0, 0.0, 0.0)),
        gmst_minus_era        = (0.014506, 4612.156534, 1.3915817, -0.00000044,
                                 -0.000029956, -0.0000000368))
    """ The eleven ocean-loading constituents, given as
        :class:`~scarabaeus.utils.Constants.OceanLoadingInfo`: Doodson
        numbers, Schwiderski phases, the IAU 1980 fundamental arguments, the
        nodal corrections of the lunar constituents, and the GMST that turns
        an Earth rotation angle into the Doodson variable :math:`\\tau`.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed
            Values of Deep Space Network Data Types for Navigation. JPL
            Publication 00-7, Section 5.
        .. [2] McCarthy, D.D. (1996). IERS Conventions (1996), IERS Technical
            Note 21, Chapter 7, routine ARG.
        .. [3] Pugh, D.T. (1987). Tides, Surges and Mean Sea-Level. Wiley,
            Table 4.3, after Schureman, P. (1958), Manual of Harmonic Analysis
            and Prediction of Tides, US Coast and Geodetic Survey SP 98.
        .. [4] Petit, G. and Luzum, B. (2010). IERS Conventions (2010), IERS
            Technical Note 36, eq. 5.32.
    """
    # endregion Solid tide
    # region Radiometric
    DSN_BANDS = {
        "X": DSNBand(name                 = 'X',
                     uplink_multiplier    = 749,
                     downlink_multiplier  = 880,
                     turnaround_num       = 880,
                     turnaround_den       = 749,
                     exciter_scalar_num   = 221.0,
                     exciter_scalar_den   = 23968.0,
                     uplink_frequency     = ArrayWUnits(7.1523135969977e09, 1 / sec),
                     ru_reference_frequency = ArrayWUnits(8445.767679e06, 1 / sec)),
        "S": DSNBand(name                 = 'S',
                     uplink_multiplier    = 221,
                     downlink_multiplier  = 240,
                     turnaround_num       = 240,
                     turnaround_den       = 221,
                     exciter_scalar_num   = 221.0,
                     exciter_scalar_den   = 7072.0,
                     uplink_frequency     = ArrayWUnits(2.110362223e09, 1 / sec),
                     ru_reference_frequency = ArrayWUnits(8445.767679e06 * 240 / 880, 1 / sec)),
    }
    """ DSN frequency-band parameters, keyed by band letter, each a
        :class:`~scarabaeus.utils.Constants.DSNBand`.

        Supported bands are ``'X'`` and ``'S'``. ``'X'`` holds the values the
        measurement models used as literals before they were collected here, so
        it is the default everywhere and reproduces the previous numbers exactly.

        .. [1] Moyer, T. D., Formulation for Observed and Computed Values of Deep
            Space Network Data Types for Navigation, JPL Publication 00-7,
            Monograph 2, Deep Space Communications and Navigation Series,
            Chapter 13.
        .. [2] DSN Telecommunications Link Design Handbook, 810-005, Module 201,
            Frequency and Channel Assignments.
    """

    RANGE_MODULO_DEFAULT = 1073741824
    """ Default sequential-ranging code modulo, :math:`2^{30}`.

        The observable wraps at this count, so a range change larger than the
        modulo is ambiguous. Carried per observation in the auxiliary data
        because a pass may use a shorter code.
    """

    RAMP_RATE_DEFAULT = ArrayWUnits(0.2826, 1 / sec**2)
    """ Default uplink frequency ramp rate used when simulating a ramp table.

        A representative DSN ramp slope, not a physical constant: real passes
        carry their measured rate in the ramp table.
    """

    RAMP_MARGIN_DEFAULT = ArrayWUnits(3600.0, sec)
    """ Default span a simulated ramp table extends before the first and after
        the last observation.

        The uplink transmit time precedes the first tag by the round-trip light
        time, so a table that starts at the first tag raises
        :class:`~scarabaeus.measurements.RampTableManager.RampTableManager.RampCoverageError`.
        One hour clears a spacecraft out to the asteroid belt.
    """

    RAMP_ENTRIES_DEFAULT = 12
    """ Default number of entries of a simulated ramp table. """

    RAMP_TYPE_DEFAULT = 1
    """ Default ramp type of a simulated ramp table, the SFDU 9 ``ramp_type``
        code of a ramped segment.
    """

    DOPPLER_COUNT_INTERVAL_DEFAULT = ArrayWUnits(60.0, sec)
    """ Default Doppler count interval, :math:`T_c`.

        The count time of simulated Doppler and the bin width
        ``compress_doppler_to_60s`` compresses real Doppler to.
    """

    TRANSPONDER_DELAY_DEFAULT = 1.0e-5
    """ Default spacecraft transponder delay [RU] of simulated auxiliary data.

        Representative, not physical: real passes carry the calibrated delay.
    """

    RCV_TIME_TAG_DELAY_DEFAULT = ArrayWUnits(8.0e-4, sec)
    """ Default receive time-tag delay of simulated auxiliary data.

        Representative, not physical: real passes carry the station value.
    """

    UL_STATION_CAL_DEFAULT = 5000.0
    """ Default uplink station calibration [RU] of simulated auxiliary data.

        Representative, not physical: real passes carry the station value.
    """

    DL_STATION_CAL_DEFAULT = 850000.0
    """ Default downlink station calibration [RU] of simulated auxiliary data.

        Representative, not physical: real passes carry the station value.
    """

    RANGING_TYPE_DEFAULT = 0
    """ Default SFDU 7 ``rng_type`` code of simulated sequential ranging. """

    RANGING_VERSION_DEFAULT = 2
    """ Default SFDU 7 ``version_num`` of simulated sequential ranging. """

    SOLAR_CORONA_DELAY_COEFF = 1.3446e-19
    """ Coefficient of the solar-corona group delay, in us GHz^2 m^2:
        delay [us] = ``SOLAR_CORONA_DELAY_COEFF`` * STEC [electrons/m^2] / f^2 [GHz^2].

        .. [1] Ho, C. M. (2010). 106, Rev. B, Solar Corona and Solar Wind
            Effects. DSN Telecommunications Link Design Handbook, Eq. (1).
    """
    # endregion Radiometric

    # region Unit conversions
    km_to_m = float(ArrayWUnits(1.0, km).convert_to(m).values)
    """ Metres per kilometre.

        Positions and the speed of light are kept in km and km/s, while the
        media models and the solar corona model work in metres. Multiply by
        this to go from km to m, divide by it to go from m to km.
    """

    min_to_sec = float(ArrayWUnits(1.0, minute).convert_to(sec).values)
    """ Seconds per minute. """

    hour_to_sec = float(ArrayWUnits(1.0, hr).convert_to(sec).values)
    """ Seconds per hour. """

    day_to_sec = float(ArrayWUnits(1.0, day).convert_to(sec).values)
    """ Seconds per day. """

    deg_per_day_to_rad_per_sec = float(ArrayWUnits(1.0, deg / day).convert_to(rad / sec).values)
    """ Radians per second per degree per day.

        Prime-meridian rates (``pole_pm``) are carried in deg/day, as the IAU
        rotation models give them, while the dynamics run in rad/s.
    """

    day_per_julian_century = 36525.0
    """ Days per Julian century.

        The IAU rotation models give the pole right ascension and declination
        as polynomials in Julian centuries past J2000, and the prime meridian
        in days past J2000.
    """

    # endregion Unit conversions

    # region Measurement processing
    SCHEDULE_EDGE_TOLERANCE = ArrayWUnits(0.01, sec)
    """ Slack when deciding whether a scheduled grid point reaches the end of
        the schedule, a UTC midnight or an hour-of-day bound.

        A grid at a fixed TDB cadence drifts against the UTC calendar by the
        TDB-UTC periodic terms (< 2 ms), on top of the ET float rounding.
    """

    PASS_GAP_DEFAULT = ArrayWUnits(21600.0, sec)
    """ Default gap that splits measurements into passes for the Huber/MAD
        outlier filter of
        :class:`~scarabaeus.orbitDetermination.MeasurementEditing.MeasurementEditing`.

        Longer than any gap inside one tracking pass: over a 6.2 h pass the
        ranging has a median gap of 523 s and a largest one of 3300 s, the
        Doppler 60 s and 2498 s. A 300 s gap would cut that pass into 38
        pieces.
    """

    PASS_MIN_POINTS_DEFAULT = 5
    """ Fewest points a pass segment may have before the Huber/MAD filter
        merges it into its neighbour rather than detrending it on its own: a
        fit to a near-empty segment absorbs the outliers it should flag.
    """

    ISOLATION_FOREST_MIN_POINTS = 8
    """ Fewest active, finite points the isolation-forest outlier filter runs
        on. Below this it flags only the non-finite points.
    """


    # endregion Measurement processing

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
        Mean radius from NEOWISE thermal survey (diameter 7.814 +/- 0.071 km,
        albedo 0.187 +/- 0.014). GM estimated from assumed S-type bulk density
        of 2500 kg/m^3; no spacecraft measurement exists.

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
        GM estimated from assumed bulk density of 2500 kg/m^3; no spacecraft
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

    # region Light-time
    RLT_BODIES = (
        (str(EARTH.body_center_id), EARTH.GM),
        (str(MOON.body_center_id), MOON.GM),
        (str(MARS.barycenter_id), MARS.GM_barycenter),
        (str(JUPITER.barycenter_id), JUPITER.GM_barycenter),
        (str(SATURN.barycenter_id), SATURN.GM_barycenter),
    )
    """ Default bodies of the relativistic light-time delay, as
        ``(spice_id, GM)`` pairs, besides the Sun.

        Earth and the Moon are taken at their centres, as the station sits next
        to them. Mars, Jupiter and Saturn are taken at their system barycentres
        with the system GM, so that only the planetary ephemeris is needed, not
        the satellite kernels. For a Mars orbiter (DSS-34 tracking, 2020-21 and
        2025) this moves the round-trip light time by at most 2.1 ps, 0.63 mm of
        round-trip path, mostly Jupiter's GM difference, and the 60 s X-band
        Doppler by less than 1e-6 mHz. Close to Jupiter or Saturn the barycentre
        is a poor stand-in for the planet (an orbiter at 1.1 planet radii sees
        up to 3 cm and 0.25 mHz), so pass the planet centres there.

        .. [1] Moyer, T.D. (2000). Formulation for Observed and Computed
            Values of Deep Space Network Data Types for Navigation. JPL Publication
            00-7, Monograph 2, Deep Space Communications and Navigation Series,
            Eq. 8-55.
    """

    # endregion Light-time