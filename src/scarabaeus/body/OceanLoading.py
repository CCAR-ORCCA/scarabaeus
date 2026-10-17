# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from dataclasses import dataclass

import spiceypy as spice

from scarabaeus import ArrayWUnits, Units
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Constants import Constants as constants

km, sec = Units.get_units(["km", "sec"])


@dataclass(frozen=True)
class OceanLoading:
    r"""Ocean-loading displacement of a station, from BLQ coefficients.

    Ocean loading is the crust's response to the weight of the ocean tide.
    The tide moves a large mass of water over the crust twice a day, and the
    crust sags under it. A coastal station moves by up to a few centimetres,
    and an inland one by a few millimetres. The response depends on the
    station and the ocean model, so it is given per station as the amplitudes
    and phases of a BLQ table, such as those from the Onsala loading service
    (http://holt.oso.chalmers.se/loading/).

    For each constituent :math:`i` with astronomical argument
    :math:`\theta_i + \chi_i` (see :class:`~scarabaeus.utils.Constants.OceanLoadingInfo`)
    and nodal corrections :math:`f_i, u_i`, the displacement in m is (Moyer,
    with the nodal corrections of IERS Conventions (2010), eq. 7.16):

    .. math::

        \Delta z &= \sum_i f_i A^r_i \cos(\theta_i + \chi_i + u_i - \phi^r_i) \\
        \Delta n &= -\sum_i f_i A^S_i \cos(\theta_i + \chi_i + u_i - \phi^S_i) \\
        \Delta e &= -\sum_i f_i A^W_i \cos(\theta_i + \chi_i + u_i - \phi^W_i)

    Here :math:`A` and :math:`\phi` are the BLQ amplitudes and Greenwich phase
    lags of the radial, south and west components.

    Parameters
    ----------
    amplitudes : array_like, shape (3, 11)
        Amplitudes, m. The rows are radial, west and south, and the columns
        are the constituents of :attr:`Constants.EARTH_OCEAN_LOADING`, in the
        order of a BLQ table.
    phases : array_like, shape (3, 11)
        Greenwich phase lags, degrees, in the same layout.
    rate_step : float, optional
        Half-interval in seconds for the central difference that supplies the
        body-fixed rate, default 60.

    Notes
    -----
    The nodal corrections :math:`f` and :math:`u` follow the 18.6-year
    regression of the lunar node (:meth:`nodal_corrections`). Moyer leaves
    them out; they scale the lunar terms by up to 4% for M2 and 40% for Mf.

    The Doodson variable :math:`\tau` needs UT1. When the station's
    Earth-fixed frame is given it is read from that frame's rotation, so the
    phase follows the same Earth rotation that places the station (see
    :meth:`arguments`); without a frame UTC stands in for UT1, which moves
    the phase by at most a few millidegrees.

    References
    ----------
    Moyer, T. D., *Formulation for Observed and Computed Values of Deep Space
    Network Data Types for Navigation*, JPL Publication 00-7, 2000.

    IERS Conventions (2010), IERS Technical Note 36, Section 7.1.2,
    eq. 7.16.
    """

    #: Row units of :meth:`state_displacement`.
    STATE_UNITS = (km,) * 3 + (km / sec,) * 3

    amplitudes: tuple
    phases: tuple
    rate_step: float = 60.0

    class CoefficientShapeError(ValueError):
        """Raised by the constructor when the amplitudes or phases are not
        3 x (number of constituents) finite values."""

    class StationNotFoundError(KeyError):
        """Raised by :meth:`from_blq` when the station is not in the file,
        or its block has fewer than six rows of coefficients."""

    class UndefinedVerticalError(ValueError):
        """Raised when the position is at the body centre or on the rotation
        axis. The east direction is undefined there."""

    def __post_init__(self):
        """Validate the coefficients and store them as nested tuples.

        Raises
        ------
        CoefficientShapeError
            If :attr:`amplitudes` or :attr:`phases` is not 3 x (number of
            constituents) finite values.
        ValueError
            If :attr:`rate_step` is not positive and finite.
        """
        count = len(constants.EARTH_OCEAN_LOADING.tides)
        for name in ("amplitudes", "phases"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, count) or not np.all(np.isfinite(value)):
                raise self.CoefficientShapeError(
                    f"{name} must be 3 x {count} finite values "
                    f"(radial, west, south by {constants.EARTH_OCEAN_LOADING.tides}), "
                    f"got shape {value.shape}")
            object.__setattr__(self, name, tuple(tuple(row) for row in value.tolist()))
        if not np.isfinite(self.rate_step) or self.rate_step <= 0:
            raise ValueError("rate_step must be positive and finite")

    @classmethod
    def from_blq(cls, path, station: str, rate_step: float = 60.0) -> "OceanLoading":
        """Read one station's coefficients from a BLQ file.

        Parameters
        ----------
        path : str or path-like
            BLQ file. Lines starting with ``$$`` are comments. Each station
            is a name line followed by six rows of numbers: the radial, west
            and south amplitudes in m, then their phases in degrees.
        station : str
            Station name as written in the file. Case and repeated spaces are
            ignored.
        rate_step : float, optional
            Half-interval in seconds for the central difference that supplies
            the body-fixed rate, default 60.

        Returns
        -------
        OceanLoading
            Model holding that station's amplitudes and phases.

        Raises
        ------
        StationNotFoundError
            If the station is not in the file, or its name is not followed by
            six rows of numbers.
        """
        wanted = " ".join(str(station).split()).upper()
        with open(path) as handle:
            lines = [line.strip() for line in handle]
        lines = [line for line in lines if line and not line.startswith("$$")]
        for index, line in enumerate(lines):
            if " ".join(line.split()).upper() != wanted:
                continue
            rows = lines[index + 1:index + 7]
            try:
                values = [[float(item) for item in row.split()] for row in rows]
            except ValueError:
                values = []
            if len(values) != 6:
                raise cls.StationNotFoundError(
                    f"{station!r} in {path} is not followed by six rows of coefficients")
            return cls(amplitudes=values[:3], phases=values[3:], rate_step=rate_step)
        raise cls.StationNotFoundError(f"{station!r} is not in {path}")

    @staticmethod
    def _delaunay(epoch: float) -> dict:
        """IAU 1980 Delaunay arguments, degrees.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.

        Returns
        -------
        dict
            ``l``, ``l'``, ``F``, ``D`` and ``Omega``, degrees.
        """
        centuries = float(epoch) / (36525.0 * 86400.0)
        delaunay = {}
        for name, coefficients in constants.EARTH_OCEAN_LOADING.fundamental_arguments.items():
            value = 0.0
            for coefficient in reversed(coefficients):
                value = value * centuries + coefficient
            delaunay[name] = value / 3600.0
        return delaunay

    @staticmethod
    def earth_rotation_angle(epoch: float, frame: str) -> float:
        r"""Earth rotation angle of an Earth-fixed frame, radians.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str
            Earth-fixed frame, e.g. ``"ITRF93"``.

        Returns
        -------
        float
            Angle from the celestial to the terrestrial intermediate origin,
            in :math:`(-\pi, \pi]`.

        Notes
        -----
        With :math:`M` the J2000-to-``frame`` rotation, the third row of
        :math:`M` is the pole :math:`(X, Y, Z)` in J2000. The CIO-based
        matrix :math:`Q = R_3(-E) R_2(-d) R_3(E) R_3(s)` (IERS 2010 eq. 5.6)
        with :math:`E = \mathrm{atan2}(Y, X)`, :math:`d = \arccos Z`
        (eq. 5.7) and :math:`s \approx -XY/2`, the leading term of eq. 5.9,
        leaves :math:`M Q = R_3(\theta)` up to the polar motion, whose effect
        on :math:`\theta` is second order. Through the IERS definition
        :math:`\theta = 2\pi(0.7790572732640 + 1.00273781191135448\,D_U)`
        (eq. 5.14), with :math:`D_U` the UT1 Julian date minus 2451545.0,
        this angle is UT1: for ITRF93 it recovers the tabulated UT1 - UTC
        to tens of microseconds.

        References
        ----------
        IERS Conventions (2010), IERS Technical Note 36, Section 5.4.4,
        eqs. 5.6, 5.7 and 5.9, and Section 5.5.3, eq. 5.14.
        """
        matrix = spice.pxform("J2000", frame, float(epoch))
        x, y, z = matrix[2]
        node, tilt = np.arctan2(y, x), np.arccos(np.clip(z, -1.0, 1.0))
        cio = (spice.rotate(-node, 3) @ spice.rotate(-tilt, 2)
               @ spice.rotate(node, 3) @ spice.rotate(-x * y / 2.0, 3))
        rotation = matrix @ cio
        return float(np.arctan2(rotation[0, 1], rotation[0, 0]))

    @staticmethod
    def arguments(epoch: float, frame: str = None) -> np.ndarray:
        r"""Astronomical arguments :math:`\theta_i + \chi_i` of the
        constituents, degrees.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str, optional
            Earth-fixed frame whose rotation supplies UT1 for :math:`\tau`,
            through :meth:`earth_rotation_angle` and GMST from IERS 2010
            eq. 5.32. When None, the default, UTC stands in for UT1.

        Returns
        -------
        numpy.ndarray
            One argument per constituent of :attr:`Constants.EARTH_OCEAN_LOADING`.

        References
        ----------
        IERS Conventions (2010), IERS Technical Note 36, Section 5.5.7,
        eq. 5.32.
        """
        info = constants.EARTH_OCEAN_LOADING
        epoch = float(epoch)
        delaunay = OceanLoading._delaunay(epoch)
        s = delaunay["F"] + delaunay["Omega"]
        h = s - delaunay["D"]
        p = s - delaunay["l"]
        node = -delaunay["Omega"]
        solar_perigee = h - delaunay["l'"]
        tt_centuries = spice.unitim(epoch, "TDB", "TDT") / (36525.0 * 86400.0)
        if frame is None:
            # tau = GMST + 180 deg - s, as in the IERS routine STEP2DIU: the
            # hour angle supplies the daily turn and the polynomial the rest.
            julian_date_utc = float(spice.et2utc(epoch, "J", 14).split()[1])
            hours = ((julian_date_utc - 0.5) % 1.0) * 24.0
            gmst = 0.0
            for coefficient in reversed(constants.EARTH_SOLID_TIDE.step2_arguments["GMST"]):
                gmst = gmst * tt_centuries + coefficient
            tau = hours * 15.0 + gmst - s
        else:
            offset = 0.0
            for coefficient in reversed(info.gmst_minus_era):
                offset = offset * tt_centuries + coefficient
            gmst = np.rad2deg(OceanLoading.earth_rotation_angle(epoch, frame)) + offset / 3600.0
            tau = gmst + 180.0 - s
        beta = np.array([tau, s, h, p, node, solar_perigee])
        return (np.asarray(info.doodson, dtype=float) @ beta
                + np.asarray(info.schwiderski_phase, dtype=float))

    @staticmethod
    def nodal_corrections(epoch: float) -> tuple:
        r"""Nodal factors :math:`f_i` and angles :math:`u_i` of the constituents.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.

        Returns
        -------
        tuple of numpy.ndarray
            ``(f, u)``: the amplitude factors, and the phase corrections in
            degrees, one per constituent of
            :attr:`Constants.EARTH_OCEAN_LOADING`.

        Notes
        -----
        :math:`f = \sum_k a_k \cos kN` and :math:`u = \sum_k b_k \sin kN`,
        with :math:`N` the longitude of the Moon's ascending node, the
        Delaunay :math:`\Omega`.

        References
        ----------
        Pugh, D. T., *Tides, Surges and Mean Sea-Level*, Wiley, 1987,
        Table 4.3, after Schureman, P., *Manual of Harmonic Analysis and
        Prediction of Tides*, US Coast and Geodetic Survey Special
        Publication 98, 1958.
        """
        info = constants.EARTH_OCEAN_LOADING
        node = np.deg2rad(OceanLoading._delaunay(epoch)["Omega"])
        multiples = np.arange(4)
        factor = np.asarray(info.nodal_factor, dtype=float) @ np.cos(multiples * node)
        angle = np.asarray(info.nodal_angle, dtype=float) @ np.sin(multiples[1:] * node)
        return factor, angle

    def displacement(self, position_body_fixed, epoch: float,
                     frame: str = None) -> ArrayWUnits:
        """Ocean-loading displacement of the station.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position, Earth-fixed. It only sets the local
            vertical, east and north directions. Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Earth-fixed frame of the position. Its rotation supplies UT1 for
            the arguments (:meth:`arguments`); the displacement is returned in
            the axes of the position. When None, the default, UTC stands in
            for UT1.

        Returns
        -------
        ArrayWUnits
            Displacement in km, shape (3,).

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.
        """
        return ArrayWUnits.from_trusted(self._displacement(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame), km)

    def _displacement(self, position_body_fixed, epoch: float,
                      frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`displacement`: position in km, displacement
        in km, shape (3,)."""
        epoch = float(getattr(epoch, "et", epoch))
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        radius = np.linalg.norm(position)
        horizontal = np.hypot(position[0], position[1])
        if radius <= 0 or horizontal <= 0:
            raise self.UndefinedVerticalError(
                "Ocean-loading directions are undefined at the centre or on the axis")
        factor, angle = self.nodal_corrections(epoch)
        argument = np.deg2rad(self.arguments(epoch, frame) + angle)
        amplitude = np.asarray(self.amplitudes) * factor
        phase = np.deg2rad(np.asarray(self.phases))
        radial, west, south = np.sum(amplitude * np.cos(argument - phase), axis=1) / 1000.0

        up = position / radius
        east_axis = np.array([-position[1], position[0], 0.0]) / horizontal
        north_axis = np.cross(up, east_axis)
        return radial * up - south * north_axis - west * east_axis

    def state_displacement(self, position_body_fixed, epoch: float,
                           frame: str = None) -> ArrayWUnits:
        """Ocean-loading displacement and its body-fixed rate.

        The rate is a central difference over :attr:`rate_step`. This is the
        method a :class:`~scarabaeus.GroundStation` calls through its
        :attr:`~scarabaeus.GroundStation.surface_displacement`.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position, Earth-fixed. It only sets the local
            vertical, east and north directions. Bare numbers are km.
        epoch : float or EpochArray
            TDB seconds past J2000.
        frame : str or Frame, optional
            Earth-fixed frame of the position, passed to
            :meth:`displacement`. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            Displacement and rate, km and km/s, shape (6,).

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.
        """
        return ArrayWUnits.from_row_units(self._state_displacement(
            ArrayWUnits.values_in(position_body_fixed, km), epoch, frame),
            self.STATE_UNITS)

    def _state_displacement(self, position_body_fixed, epoch: float,
                            frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement`: position in km,
        displacement and rate in km and km/s, shape (6,)."""
        epoch = float(getattr(epoch, "et", epoch))
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        delta = self._displacement(position, epoch, frame)
        ahead = self._displacement(position, epoch + self.rate_step, frame)
        behind = self._displacement(position, epoch - self.rate_step, frame)
        return np.concatenate([delta, (ahead - behind) / (2.0 * self.rate_step)])
