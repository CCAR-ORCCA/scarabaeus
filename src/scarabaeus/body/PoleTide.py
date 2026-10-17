# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from dataclasses import dataclass

from scarabaeus import ArrayWUnits, Units
import scarabaeus.utils.NumpyWrapper as np
from scarabaeus.utils.Constants import Constants as constants

km, sec, rad = Units.get_units(["km", "sec", "rad"])


@dataclass(frozen=True)
class PoleTide:
    r"""Pole-tide displacement of a surface point.

    The pole tide is the solid body's response to the wander of its rotation
    axis. Polar motion moves the rotation axis relative to the crust, which
    changes the centrifugal potential felt at the surface. The body deforms
    in response, like it does under the solid tide, and a station moves by up
    to about 2.5 cm radially and 7 mm horizontally at Earth.

    With :math:`X, Y` the pole coordinates, :math:`\bar X, \bar Y` the mean
    pole, :math:`\phi, \lambda` the geocentric latitude and longitude of the
    station at radius :math:`r`, and :math:`\omega^2 r^2/g` with
    :math:`g = GM/r^2`, the displacement is (Moyer, with IERS Conventions
    (2010), eq. 7.26):

    .. math::

        \Delta r &= -\frac{h_2}{2}\frac{\omega^2 r^2}{g}\sin 2\phi
            \left[(X - \bar X)\cos\lambda - (Y - \bar Y)\sin\lambda\right] \\
        \Delta n &= -l_2\frac{\omega^2 r^2}{g}\cos 2\phi
            \left[(X - \bar X)\cos\lambda - (Y - \bar Y)\sin\lambda\right] \\
        \Delta e &= l_2\frac{\omega^2 r^2}{g}\sin\phi
            \left[(X - \bar X)\sin\lambda + (Y - \bar Y)\cos\lambda\right]

    Parameters
    ----------
    pole_x, pole_y : ArrayWUnits or float
        Pole coordinates :math:`X, Y`; bare numbers are radians, and they are
        stored as radians. They are the same numbers as
        :attr:`~scarabaeus.timeAndFrame.BodyOrientation.BodyOrientation.pole_x`
        and ``pole_y``.
    central_body : str, optional
        Deformed body, default EARTH. Its GM and the rotation rate set the
        scale. It must be listed in :attr:`NOMINAL_RESPONSE`.
    love_h2, shida_l2 : float, optional
        Love and Shida numbers of the response. Default to
        :attr:`Constants.EARTH_POLE_TIDE`.

    Notes
    -----
    The pole is held constant, so the displacement does not vary in time and
    its body-fixed rate is zero. At Earth the pole moves a few milliarcseconds
    a day, a rate of about 1e-12 km/s, which is far below Doppler noise.

    The displacement is linear in :math:`h_2` and :math:`l_2` and, separately,
    in the pole offset, so :meth:`state_displacement_partials` and
    :meth:`state_displacement_pole_partials` are exact. The second is what lets
    an estimated ``eop_delta_pole_x``/``eop_delta_pole_y`` move the pole tide
    along with the station rotation.

    The east component takes the sign of IERS eq. 7.26, which is also the
    sign the centrifugal potential gives. Moyer prints the opposite sign.
    Using that sign would move the station by at most about 1 mm.

    References
    ----------
    Moyer, T. D., *Formulation for Observed and Computed Values of Deep Space
    Network Data Types for Navigation*, JPL Publication 00-7, 2000.

    IERS Conventions (2010), IERS Technical Note 36, Section 7.1.4.
    """

    #: Central bodies with a nominal response, each a
    #: :class:`~scarabaeus.utils.Constants.PoleTideInfo` from Constants.
    NOMINAL_RESPONSE = {"EARTH": constants.EARTH_POLE_TIDE}

    #: Row units of :meth:`state_displacement` and of its Love/Shida partials.
    STATE_UNITS = (km,) * 3 + (km / sec,) * 3
    #: Row units of :meth:`state_displacement_pole_partials`.
    POLE_PARTIAL_UNITS = (km / rad,) * 3 + (km / sec / rad,) * 3

    pole_x: float
    pole_y: float
    central_body: str = "EARTH"
    love_h2: float = None
    shida_l2: float = None

    class UnknownPoleTideResponseError(ValueError):
        """Raised by the constructor when no pole-tide response is known for
        the central body. The mean pole and the rotation rate are the body's
        own, so any body other than Earth needs a
        :class:`~scarabaeus.utils.Constants.PoleTideInfo` in Constants."""

    class UndefinedVerticalError(ValueError):
        """Raised when the position is at the body centre or on the rotation
        axis. The east direction is undefined there."""

    def __post_init__(self):
        """Fill unset Love and Shida numbers from the nominal response and
        store every coefficient as a float.

        Raises
        ------
        UnknownPoleTideResponseError
            If :attr:`central_body` is not in :attr:`NOMINAL_RESPONSE`.
        ValueError
            If a pole coordinate, Love or Shida number is not finite, or, from
            :meth:`~scarabaeus.ArrayWUnits.values_in`, a pole coordinate is
            not an angle.
        """
        nominal = self.NOMINAL_RESPONSE.get(str(self.central_body).upper())
        if nominal is None:
            raise self.UnknownPoleTideResponseError(
                f"No pole-tide response for central body {self.central_body!r}. "
                f"Known bodies: {sorted(self.NOMINAL_RESPONSE)}")
        for name in ("love_h2", "shida_l2"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, getattr(nominal, name))
        for name in ("pole_x", "pole_y"):
            object.__setattr__(self, name, ArrayWUnits.values_in(getattr(self, name), rad).item())
        for name in ("pole_x", "pole_y", "love_h2", "shida_l2"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def mean_pole(self) -> tuple:
        r""":math:`(\bar X, \bar Y)` in radians, from the degree-2, order-1
        harmonics of :attr:`NOMINAL_RESPONSE`."""
        info = self.NOMINAL_RESPONSE[str(self.central_body).upper()]
        scale = np.sqrt(3.0) * info.j2
        return -info.c21 / scale, info.s21 / scale

    def displacement(self, position_body_fixed, epoch: float = None,
                     frame: str = None) -> ArrayWUnits:
        """Pole-tide displacement of a body-fixed point.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in axes whose z is the reference pole that
            :attr:`pole_x` and :attr:`pole_y` are measured from (ITRF at
            Earth). Bare numbers are km.
        epoch : float, optional
            TDB seconds past J2000. Not used, because the pole is constant; it
            is accepted so the call matches the other surface-displacement
            models. Defaults to ``None``.
        frame : str, optional
            Not used, for the same reason. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            Displacement in km, shape (3,), in the axes of the position.

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.
        """
        return ArrayWUnits.from_trusted(
            self._displacement(ArrayWUnits.values_in(position_body_fixed, km)), km)

    def _displacement(self, position_body_fixed, epoch: float = None,
                      frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`displacement`: position in km, displacement
        in km, shape (3,)."""
        mean_x, mean_y = self.mean_pole
        return self._response(position_body_fixed, self.love_h2, self.shida_l2,
                              self.pole_x - mean_x, self.pole_y - mean_y)

    def _response(self, position_body_fixed, love_h2, shida_l2, delta_x, delta_y):
        """Displacement for given Love/Shida numbers and pole offset, km.

        Parameters
        ----------
        position_body_fixed : array_like, shape (3,)
            Undisplaced position, km; see :meth:`displacement`.
        love_h2, shida_l2 : float
            Love and Shida numbers.
        delta_x, delta_y : float
            Pole minus mean pole, radians.

        Returns
        -------
        numpy.ndarray
            Displacement in km, shape (3,). Linear in ``(love_h2, shida_l2)``
            and, separately, in ``(delta_x, delta_y)``, so a unit input gives
            the exact partial.

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.
        """
        position = np.asarray(position_body_fixed, dtype=float).reshape(3)
        radius = np.linalg.norm(position)
        horizontal = np.hypot(position[0], position[1])
        if radius <= 0 or horizontal <= 0:
            raise self.UndefinedVerticalError(
                "Pole-tide displacement is undefined at the centre or on the axis")
        body = getattr(constants, str(self.central_body).upper())
        # omega^2 r^2 / g with g = GM / r^2
        scale = float(constants.earth_rotation_rate.values)**2 * radius**4 / float(body.GM.values)

        sin_lat, cos_lat = position[2] / radius, horizontal / radius
        sin_lon, cos_lon = position[1] / horizontal, position[0] / horizontal
        meridian = delta_x * cos_lon - delta_y * sin_lon

        radial = -0.5 * love_h2 * scale * 2.0 * sin_lat * cos_lat * meridian
        north = -shida_l2 * scale * (cos_lat**2 - sin_lat**2) * meridian
        east = shida_l2 * scale * sin_lat * (delta_x * sin_lon + delta_y * cos_lon)

        up = position / radius
        east_axis = np.array([-sin_lon, cos_lon, 0.0])
        north_axis = np.cross(up, east_axis)
        return radial * up + north * north_axis + east * east_axis

    def state_displacement(self, position_body_fixed, epoch: float = None,
                           frame: str = None) -> ArrayWUnits:
        """Pole-tide displacement and its body-fixed rate.

        The rate is zero because the pole is constant; see Notes of the class.
        This is the method a :class:`~scarabaeus.GroundStation` calls through
        its :attr:`~scarabaeus.GroundStation.surface_displacement`.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position in axes whose z is the reference pole that
            :attr:`pole_x` and :attr:`pole_y` are measured from (ITRF at
            Earth). Bare numbers are km.
        epoch : float, optional
            TDB seconds past J2000. Not used, because the pole is constant.
            Defaults to ``None``.
        frame : str, optional
            Not used. Defaults to ``None``.

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
            ArrayWUnits.values_in(position_body_fixed, km)), self.STATE_UNITS)

    def _state_displacement(self, position_body_fixed, epoch: float = None,
                            frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement`: position in km,
        displacement and rate in km and km/s, shape (6,)."""
        return np.concatenate([self._displacement(position_body_fixed), np.zeros(3)])

    def state_displacement_partials(self, position_body_fixed, epoch: float = None,
                                    frame: str = None) -> ArrayWUnits:
        """Love/Shida sensitivity of the displacement and of its body-fixed rate.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position; see :meth:`displacement`. Bare numbers are
            km.
        epoch : float, optional
            Not used, because the pole is constant. Defaults to ``None``.
        frame : str, optional
            Not used. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            ``(6, 2)`` Jacobian in km and km/s per unit Love or Shida number,
            columns :math:`h_2` and :math:`l_2`, the same columns as
            :meth:`~scarabaeus.SolidTide.state_displacement_partials`. The rate
            rows are zero.

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.

        Notes
        -----
        :math:`h_2` scales only the radial term and :math:`l_2` only the two
        horizontal ones, so each column is the displacement at a unit number
        and zero for the other: exact, not a linearization.
        """
        return ArrayWUnits.from_row_units(self._state_displacement_partials(
            ArrayWUnits.values_in(position_body_fixed, km)), self.STATE_UNITS)

    def _state_displacement_partials(self, position_body_fixed, epoch: float = None,
                                     frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement_partials`: position in
        km, ``(6, 2)`` Jacobian in km and km/s."""
        mean_x, mean_y = self.mean_pole
        dx, dy = self.pole_x - mean_x, self.pole_y - mean_y
        columns = np.column_stack([self._response(position_body_fixed, 1.0, 0.0, dx, dy),
                                   self._response(position_body_fixed, 0.0, 1.0, dx, dy)])
        return np.vstack([columns, np.zeros((3, 2))])

    def state_displacement_pole_partials(self, position_body_fixed, epoch: float = None,
                                         frame: str = None) -> ArrayWUnits:
        """Pole-coordinate sensitivity of the displacement and of its rate.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like, shape (3,)
            Undisplaced position; see :meth:`displacement`. Bare numbers are
            km.
        epoch : float, optional
            Not used, because the pole is constant. Defaults to ``None``.
        frame : str, optional
            Not used. Defaults to ``None``.

        Returns
        -------
        ArrayWUnits
            ``(6, 2)`` Jacobian in km/rad and km/s/rad, columns :attr:`pole_x`
            and :attr:`pole_y`. The rate rows are zero.

        Raises
        ------
        UndefinedVerticalError
            If the position is at the body centre or on the rotation axis.

        Notes
        -----
        The displacement is linear in the offset from the mean pole, so each
        column is the displacement for a unit offset in that coordinate: exact.
        A :class:`~scarabaeus.GroundStation` sums these through
        :meth:`~scarabaeus.GroundStation.pole_tide_partials` so an estimated
        ``eop_delta_pole_x``/``eop_delta_pole_y`` moves this tide too.
        """
        return ArrayWUnits.from_row_units(self._state_displacement_pole_partials(
            ArrayWUnits.values_in(position_body_fixed, km)), self.POLE_PARTIAL_UNITS)

    def _state_displacement_pole_partials(self, position_body_fixed, epoch: float = None,
                                          frame: str = None) -> np.ndarray:
        """Numeric core of :meth:`state_displacement_pole_partials`: position
        in km, ``(6, 2)`` Jacobian in km/rad and km/s/rad."""
        columns = np.column_stack([
            self._response(position_body_fixed, self.love_h2, self.shida_l2, 1.0, 0.0),
            self._response(position_body_fixed, self.love_h2, self.shida_l2, 0.0, 1.0)])
        return np.vstack([columns, np.zeros((3, 2))])
