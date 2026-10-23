# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Generic body orientation and Earth EOP conventions in one public class."""

from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import erfa
import numpy as np
from numpy.polynomial import chebyshev
import spiceypy as spice

from scarabaeus import ArrayWUnits, Units
from scarabaeus.timeAndFrame.SpiceManager import SpiceManager
from scarabaeus.utils.Constants import Constants as constants

km, sec, rad, deg, day = Units.get_units(["km", "sec", "rad", "deg", "day"])

@dataclass(frozen=True)
class BodyOrientation:
    r"""A body-frame correction or an absolute planetary orientation model.

    Use the constructor for a small active rotation, :meth:`earth_correction`
    for Earth-convention deltas, :meth:`earth_eop` for absolute Earth inputs,
    and :meth:`from_rotation` for another body's absolute orientation law.
    An instance has exactly one of these modes. ``dataclasses.replace`` makes
    an independent instance with revised inputs, e.g. for an EOP comparison.

    Parameters
    ----------
    rotation_vector : array_like, shape (3,), optional
        Active right-hand body-fixed X/Y/Z correction in radians. Surface
        displacement is :math:`\delta r = \varepsilon \times r` to first order.
    body_fixed_frame : str, optional
        SPICE axes of the correction. Required for a correction instance.
    rotation : callable, optional
        Absolute inertial-to-body rotation callback; see :meth:`from_rotation`.
    frame_class_id : int, optional
        Binary PCK class ID; Earth ITRF93 uses 3000. See :meth:`from_rotation`.
    base_frame, interpolation_frame : str, optional
        Inertial callback axes and PCK interpolation axes. Defaults to J2000
        and ECLIPJ2000 respectively.
    max_interval : float, optional
        Maximum Chebyshev record length, seconds; default 21600.
    description : str, optional
        Provenance for a user-supplied rotation law.
    pole_x, pole_y, tai_minus_ut1, model : optional
        Absolute Earth EOPs and convention; see :meth:`earth_eop`.
    ut1_time_scale : str, optional
        Which time scale UT1 is formed from; see :meth:`earth_eop`.
    celestial_pole_dx, celestial_pole_dy : float, optional
        Modern CPOs in radians; see :meth:`earth_eop`.
    nutation_delta_psi, nutation_delta_epsilon : float, optional
        Legacy nutation corrections in radians; see :meth:`earth_eop`.

    Notes
    -----
    Corrections affect station measurement geometry, including velocity, but
    do not replace loaded kernels or alter gravity. An absolute model can
    write a PCK shared by all SPICE frame consumers. A leap-second kernel is
    needed for absolute Earth models; generic callbacks define their own time
    dependence using TDB seconds past J2000. UT1 is an Earth convention.

    See :ref:`orientation-station-estimation` for EOP signs, PCK precedence,
    planetary frames, cylindrical coordinates and complete estimation
    examples.
    """

    class NonFiniteOffsetError(ValueError):
        """A celestial pole offset given to :meth:`celestial_pole_correction`
        was not finite."""

    class MissingPoleCoefficientsError(ValueError):
        """:meth:`iau_angles` needed the secular PCK coefficients of a body
        but was given none."""

    # Unannotated, so the dataclass keeps them as class attributes, not fields.
    IAU_ROTATION_ITEMS = ("POLE_RA", "POLE_DEC", "PM")
    """Kernel-pool names of the IAU secular rotation constants, in the row
    order of :meth:`iau_pole_coefficients`."""

    IAU_POLE_PARAMETERS = ("pole_ra", "pole_dec", "pole_pm", "pole_w0")
    """State keys of an estimated IAU body orientation.

    Pole right ascension and declination (rad), prime-meridian rate (deg/day)
    and prime-meridian angle at J2000 (rad). Each replaces its term of the
    body's secular PCK model, as :meth:`iau_angles` describes.
    """

    IAU_POLE_UNITS = dict(zip(IAU_POLE_PARAMETERS, (rad, rad, deg / day, rad)))
    """Units of each of :attr:`IAU_POLE_PARAMETERS`."""

    STATE_UNITS = (km,) * 3 + (km / sec,) * 3
    """Row units of :meth:`state_displacement`."""

    rotation_vector: np.ndarray | None = None
    body_fixed_frame: str | None = None
    rotation: Callable | None = None
    frame_class_id: int = 3000
    base_frame: str = "J2000"
    interpolation_frame: str = "ECLIPJ2000"
    max_interval: float = 21600.
    description: str = "User-supplied absolute body orientation"
    pole_x: float | None = None
    pole_y: float | None = None
    tai_minus_ut1: float | None = None
    model: str = "EOP1"
    celestial_pole_dx: float = 0.
    celestial_pole_dy: float = 0.
    nutation_delta_psi: float = 0.
    nutation_delta_epsilon: float = 0.
    ut1_time_scale: str = "TAI"

    def __post_init__(self):
        """Check that exactly one mode is selected and normalise its inputs.

        Raises
        ------
        ValueError
            If not exactly one of ``rotation_vector``, ``rotation`` or the
            absolute Earth EOPs is supplied. For a correction: if
            ``rotation_vector`` is not finite or ``body_fixed_frame`` is
            missing. For an absolute model: if ``frame_class_id`` is not an
            integer, ``max_interval`` is not positive and finite, or
            ``base_frame`` or ``interpolation_frame`` is not inertial. For
            Earth EOPs: if one of ``pole_x``, ``pole_y`` and ``tai_minus_ut1``
            is missing, an EOP or CPO is not finite, ``base_frame`` is not
            J2000, ``model`` is unknown, celestial pole offsets are given with
            IAU1976/1980, nutation corrections are given with IAU2006/2000A,
            or ``ut1_time_scale`` is not ``"TAI"`` or ``"TDB"``.
        TypeError
            If ``rotation`` is not callable.
        """
        earth = any(x is not None for x in (self.pole_x, self.pole_y, self.tai_minus_ut1))
        modes = int(self.rotation_vector is not None) + int(self.rotation is not None) + int(earth)
        if modes != 1:
            raise ValueError("Provide exactly one of rotation_vector, rotation, or absolute Earth EOPs")
        if self.rotation_vector is not None:
            vector = np.asarray(self.rotation_vector, dtype=float).reshape(3).copy()
            if not np.all(np.isfinite(vector)):
                raise ValueError("rotation_vector must be finite")
            if not isinstance(self.body_fixed_frame, str) or not self.body_fixed_frame.strip():
                raise ValueError("body_fixed_frame is required for a correction")
            vector.setflags(write=False)
            object.__setattr__(self, "rotation_vector", vector)
            return
        if isinstance(self.frame_class_id, bool) or int(self.frame_class_id) != self.frame_class_id:
            raise ValueError("frame_class_id must be an integer")
        object.__setattr__(self, "frame_class_id", int(self.frame_class_id))
        if not np.isfinite(self.max_interval) or self.max_interval <= 0:
            raise ValueError("max_interval must be positive and finite")
        for frame in (self.base_frame, self.interpolation_frame):
            if spice.frinfo(spice.namfrm(frame))[1] != 1:
                raise ValueError(f"{frame} must be inertial")
        if self.rotation is not None:
            if not callable(self.rotation):
                raise TypeError("rotation must be callable")
            return
        if any(x is None for x in (self.pole_x, self.pole_y, self.tai_minus_ut1)):
            raise ValueError("pole_x, pole_y and tai_minus_ut1 are all required")
        if self.base_frame != "J2000":
            raise ValueError("Earth EOP rotation matrices use J2000 input axes")
        aliases = {"EOP1": "IAU1976/1980", "ITRF93": "IAU1976/1980",
                   "EOP2": "IAU2006/2000A", "IERS2010": "IAU2006/2000A"}
        object.__setattr__(self, "model", aliases.get(self.model, self.model))
        if self.model not in ("IAU1976/1980", "IAU2006/2000A"):
            raise ValueError("model must be EOP1 (ITRF93) or EOP2 (IERS2010), "
                             "also accepted as IAU1976/1980 or IAU2006/2000A")
        for name in ("pole_x", "pole_y", "tai_minus_ut1",
                     "celestial_pole_dx", "celestial_pole_dy",
                     "nutation_delta_psi", "nutation_delta_epsilon"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.model == "IAU1976/1980" and (
                self.celestial_pole_dx != 0 or self.celestial_pole_dy != 0):
            raise ValueError("celestial dX/dY require IAU2006/2000A; they are not legacy dPsi/dEpsilon")
        if self.model == "IAU2006/2000A" and (
                self.nutation_delta_psi != 0 or self.nutation_delta_epsilon != 0):
            raise ValueError("legacy dPsi/dEpsilon require IAU1976/1980; they are not modern dX/dY")
        if self.ut1_time_scale not in ("TAI", "TDB"):
            raise ValueError('ut1_time_scale must be "TAI" or "TDB"')

    @classmethod
    def earth_correction(cls, delta_pole_x=0., delta_pole_y=0., delta_ut1=0.):
        """Construct a small Earth correction relative to the nominal kernel.

        Parameters
        ----------
        delta_pole_x, delta_pole_y : float, optional
            Intended minus nominal terrestrial xp/yp, radians. The active
            body-fixed rotation is (-delta_yp, -delta_xp, omega*delta_ut1).
            Defaults to ``0.``.
        delta_ut1 : float, optional
            Intended minus nominal UT1, seconds. Positive advances Earth spin.
            Defaults to ``0.``.

        Returns
        -------
        BodyOrientation
            First-order station correction in ITRF93. Absolute EOP numbers
            must not be passed here; use :meth:`earth_eop` for those.

        Raises
        ------
        ValueError
            If any of the three deltas is not finite.
        """
        return cls(np.array([-delta_pole_y, -delta_pole_x,
                             float(constants.earth_rotation_rate.values)*delta_ut1]), "ITRF93")

    @staticmethod
    def earth_correction_partials():
        """Return the rotation-vector column of each Earth EOP delta.

        Returns
        -------
        dict
            Keyword accepted by :meth:`earth_correction` mapped to the constant
            :math:`\\partial\\varepsilon/\\partial p`, shape (3,), in radians
            per radian for the pole terms and radians per second for UT1.

        Notes
        -----
        :meth:`earth_correction` is linear in its three inputs, so these
        columns are exact rather than a linearization, and a station partial
        chains them straight into
        :meth:`~scarabaeus.GroundStation.rotation_partials`. Keying by the
        constructor's own argument names keeps the two from drifting apart.
        """
        return {"delta_pole_x": np.array([0., -1., 0.]),
                "delta_pole_y": np.array([-1., 0., 0.]),
                "delta_ut1": np.array([0., 0.,
                                       float(constants.earth_rotation_rate.values)])}

    @staticmethod
    def celestial_pole_correction(delta_x=0., delta_y=0.):
        r"""Return the J2000 rotation vector of a celestial-pole-offset correction.

        Parameters
        ----------
        delta_x, delta_y : float, optional
            Intended minus nominal celestial pole offsets :math:`\delta X`,
            :math:`\delta Y`, radians. Defaults to ``0.``.

        Returns
        -------
        numpy.ndarray
            Rotation vector :math:`(-\delta Y, \delta X, 0)`, rad, shape (3,),
            in J2000 axes and held fixed there. Unlike :meth:`earth_correction`
            it does not turn with the Earth; apply it with
            :meth:`~scarabaeus.GroundStation.inertial_rotation_partials`.

        Raises
        ------
        NonFiniteOffsetError
            If either offset is not finite. It is also a ValueError.

        Notes
        -----
        To first order the CIP matrix :math:`Q(X, Y)` of IERS Conventions
        (2010) eq. 5.10 is :math:`I + [\omega\times]` with this
        :math:`\omega`. Frame bias and :math:`Q \ne I` change the column by
        about 3e-3 of itself, far below what a measurement can see.

        The same rotation corrects the IAU 1980 nutation of a ``model="EOP1"``
        kernel, with :math:`\delta X = \delta\psi\sin\varepsilon_A` and
        :math:`\delta Y = \delta\varepsilon` (IERS Conventions (2010)
        eq. 5.26 without its precession cross terms, which are about 6e-3 of
        the offsets).
        """
        values = np.array([float(delta_x), float(delta_y)])
        if not np.all(np.isfinite(values)):
            raise BodyOrientation.NonFiniteOffsetError(
                f"Celestial pole offsets must be finite, got dX={delta_x}, dY={delta_y}")
        return np.array([-values[1], values[0], 0.])

    @staticmethod
    def celestial_pole_correction_partials():
        """Return the rotation-vector column of each celestial pole offset.

        Returns
        -------
        dict
            Keyword accepted by :meth:`celestial_pole_correction` mapped to the
            constant :math:`\\partial\\omega/\\partial p`, shape (3,), in
            radians per radian, J2000 axes. Exact, since the correction is
            linear in both offsets.
        """
        return {"delta_x": np.array([0., 1., 0.]),
                "delta_y": np.array([-1., 0., 0.])}

    @staticmethod
    def iau_pole_coefficients(body):
        r"""Return a body's secular IAU rotation constants from the kernel pool.

        Parameters
        ----------
        body : str or int
            SPICE body name or ID, e.g. ``"MARS"`` or ``499``.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3). Rows are ``POLE_RA``, ``POLE_DEC`` (deg, deg per
            Julian century, deg per century squared) and ``PM`` (deg, deg/day,
            deg/day squared). A term missing from the kernel is zero.

        Notes
        -----
        A text PCK such as ``pck00010.tpc`` must be loaded. Only the secular
        terms are read; the nutation-precession series of the full model are
        left to SPICE.
        """
        rows = np.zeros((3, 3))
        for row, item in zip(rows, BodyOrientation.IAU_ROTATION_ITEMS):
            values = np.asarray(spice.bodvrd(str(body), item, 3)[1], dtype=float)
            row[:values.size] = values
        return rows

    @staticmethod
    def iau_angles(epoch, frame=None, coefficients=None, *, pole_ra=None,
                   pole_dec=None, pole_pm=None, pole_w0=None):
        r"""Return the IAU pole and prime-meridian angles of a body and their rates.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str, optional
            The body's SPICE body-fixed frame, e.g. ``"IAU_MARS"``. Used when
            no angle is replaced. Defaults to ``None``.
        coefficients : array_like, shape (3, 3), optional
            Secular constants as returned by :meth:`iau_pole_coefficients`;
            shorter rows are padded with zeros.
            Needed when an angle is replaced or ``frame`` is ``None``.
            Defaults to ``None``.
        pole_ra, pole_dec : float, optional
            Estimated pole right ascension and declination, rad. Each replaces
            its secular polynomial and is constant in time. Defaults to
            ``None``.
        pole_pm : float, optional
            Estimated prime-meridian rate, deg/day. Gives
            :math:`W = W_0 + \mathrm{PM}\,d` without the quadratic term.
            Defaults to ``None``.
        pole_w0 : float, optional
            Estimated prime-meridian angle at J2000, rad. Replaces
            :math:`W_0`. Defaults to ``None``.

        Returns
        -------
        angles : numpy.ndarray
            :math:`(\alpha, \delta, W)`, rad, shape (3,).
        rates : numpy.ndarray
            :math:`(\dot\alpha, \dot\delta, \dot W)`, rad/s, shape (3,).

        Raises
        ------
        MissingPoleCoefficientsError
            If the secular path is needed and ``coefficients`` is ``None``.
            It is also a ValueError.

        Notes
        -----
        With ``frame`` given and nothing replaced, the angles are SPICE's full
        model, read back from ``sxform("J2000", frame)`` as the 3-1-3 Euler
        angles :math:`(\alpha + 90^\circ, 90^\circ - \delta, W)`, so the
        nutation-precession terms and the time derivatives are included and
        :math:`W` is reduced to :math:`[-\pi, \pi]`.

        Otherwise the secular model of the kernel is used,
        :math:`\alpha = \alpha_0 + \alpha_1 T + \alpha_2 T^2`, likewise
        :math:`\delta`, and :math:`W = W_0 + W_1 d + W_2 d^2`, with :math:`d`
        days and :math:`T` Julian centuries past J2000, and each estimated
        value replaces its term as described above. This is the estimated-pole
        model of :class:`~scarabaeus.SphericalHarmonicsGravity`, so gravity and
        the optical measurement models see one body orientation.
        """
        epoch = float(epoch)
        if frame is not None and all(value is None for value in
                                     (pole_ra, pole_dec, pole_pm, pole_w0)):
            euler, _ = spice.xf2eul(spice.sxform("J2000", str(frame), epoch), 3, 1, 3)
            return (np.array([euler[2] - np.pi / 2, np.pi / 2 - euler[1], euler[0]]),
                    np.array([euler[5], -euler[4], euler[3]]))
        if coefficients is None:
            raise BodyOrientation.MissingPoleCoefficientsError(
                "The secular IAU coefficients are needed when a pole angle is "
                "replaced or no body-fixed frame is given; pass "
                "BodyOrientation.iau_pole_coefficients(body)")
        table = np.zeros((3, 3))
        for row, given in zip(table, coefficients):
            given = np.asarray(given, dtype=float).ravel()[:3]
            row[:given.size] = given
        days = epoch / constants.day_to_sec
        centuries = days / constants.day_per_julian_century
        century_seconds = constants.day_per_julian_century * constants.day_to_sec
        angles = np.deg2rad(table @ np.array([1., centuries, centuries**2]))
        rates = np.deg2rad(table[:, 1] + 2. * table[:, 2] * centuries) / century_seconds
        w0 = np.deg2rad(table[2, 0]) if pole_w0 is None else float(pole_w0)
        if pole_pm is None:
            angles[2] = w0 + np.deg2rad(table[2, 1] * days + table[2, 2] * days**2)
            rates[2] = np.deg2rad(table[2, 1] + 2. * table[2, 2] * days) / constants.day_to_sec
        else:
            angles[2] = w0 + np.deg2rad(float(pole_pm) * days)
            rates[2] = float(pole_pm) * constants.deg_per_day_to_rad_per_sec
        for index, value in enumerate((pole_ra, pole_dec)):
            if value is not None:
                angles[index], rates[index] = float(value), 0.
        return angles, rates

    @staticmethod
    def iau_matrix(angles):
        r"""Return the body-fixed to J2000 rotation of IAU pole angles.

        Parameters
        ----------
        angles : array_like, shape (3,)
            :math:`(\alpha, \delta, W)`, rad, e.g. from :meth:`iau_angles`.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3), :math:`R_z(\alpha + \pi/2)\,R_x(\pi/2 - \delta)\,
            R_z(W)` with active rotations, so :math:`r_{J2000} = R\,r_{body}`.
            It equals SPICE's ``pxform(frame, "J2000")`` for the angles of
            :meth:`iau_angles`.
        """
        def active(axis, angle):
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:
                return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        alpha, delta, meridian = (float(value) for value in angles)
        return (active(2, alpha + np.pi / 2) @ active(0, np.pi / 2 - delta)
                @ active(2, meridian))

    @classmethod
    def earth_eop(cls, pole_x, pole_y, tai_minus_ut1, model="EOP1", *,
                  celestial_pole_dx=0., celestial_pole_dy=0.,
                  nutation_delta_psi=0., nutation_delta_epsilon=0.,
                  ut1_time_scale="TAI"):
        """Construct absolute, constant Earth EOPs independently of a loaded PCK.

        Parameters
        ----------
        pole_x, pole_y : float
            Absolute terrestrial polar-motion coordinates in radians.
        tai_minus_ut1 : float
            Absolute TAI-UT1 in seconds: UT1 = TAI - tai_minus_ut1.
        model : str, optional
            EOP1: ITRF93-style legacy IAU1976/1980 orientation,
            equinox/GMST82 + (dpsi+dPsi)cos(eps), s'=0. EOP2: IERS2010 orientation,
            IAU2006/2000A CIO/ERA with s06 and sp00. The descriptive names
            ITRF93, IERS2010, IAU1976/1980 and IAU2006/2000A are also accepted.
            The three EOP numbers do not determine this convention.
            Defaults to ``"EOP1"``.
        celestial_pole_dx, celestial_pole_dy : float, optional
            Modern observed celestial pole offsets dX/dY, radians. Supported
            only for IAU2006/2000A; distinct from terrestrial xp/yp.
            Defaults to ``0.``.
        nutation_delta_psi, nutation_delta_epsilon : float, optional
            IAU1980 nutation longitude/obliquity corrections, radians. Supported
            only for IAU1976/1980. dPsi also enters the equation of equinoxes.
            Defaults to ``0.``.
        ut1_time_scale : str, optional
            Which time scale UT1 is formed from. Defaults to ``"TAI"``, which
            is correct: UT1 = TAI - tai_minus_ut1, with TAI reached from the
            ephemeris epoch through the full TDB->TT->TAI chain. ``"TDB"``
            instead treats the ephemeris epoch as if it were TT, so
            UT1 = TDB - 32.184 - tai_minus_ut1 and the TDB-TT periodic term
            never reaches the spin angle.

            Choose ``"TDB"`` only to reproduce another tool that omits that
            conversion. It is not a refinement and it is not small: TDB-TT is
            periodic with an amplitude of 1.66 ms, which is 0.5 m of station
            motion at the equator, well above the noise of a metre-class
            ranging system. It varies annually, so the error it causes differs
            between arcs at different times of year.

        Returns
        -------
        BodyOrientation
            Absolute model supporting rotation_components, rotation_matrix and
            write_pck. The PCK uses ITRF93 class ID 3000.

        Raises
        ------
        ValueError
            If ``model`` or ``ut1_time_scale`` is not recognised, an input is
            not finite, celestial pole offsets are given with IAU1976/1980, or
            nutation corrections are given with IAU2006/2000A.

        Notes
        -----
        EOPs/CPOs stay constant over coverage; model precession/nutation and
        spin vary with time. ERFA supplies SOFA-derived routines. Precession
        and nutation use TT; spin uses UT1. CPOs default to zero and are never
        inferred from observations. Load an LSK for the TDB/TT/TAI conversion.
        The legacy celestial axes are mean J2000; modern axes are GCRS/ICRS.
        SPICE calls both J2000 without converting between the conventions.
        EOP1/EOP2 select the rotation convention, not new station coordinates.
        Both export PCK class ID 3000 so existing ITRF93-referenced station
        SPKs use the selected orientation. EOP2 implements the IERS2010
        rotation chain with the supplied constant EOPs/CPOs; no time-varying
        tidal EOP series is synthesized.

        References
        ----------
        https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf
        https://github.com/liberfa/erfa
        """
        return cls(pole_x=pole_x, pole_y=pole_y, tai_minus_ut1=tai_minus_ut1,
                   model=model, celestial_pole_dx=celestial_pole_dx,
                   celestial_pole_dy=celestial_pole_dy,
                   nutation_delta_psi=nutation_delta_psi,
                   nutation_delta_epsilon=nutation_delta_epsilon,
                   ut1_time_scale=ut1_time_scale)

    @classmethod
    def from_rotation(cls, rotation, frame_class_id, *, base_frame="J2000",
                      interpolation_frame="ECLIPJ2000", max_interval=21600.,
                      description="User-supplied absolute body orientation"):
        """Construct an absolute rotation law for any planet or other body.

        Parameters
        ----------
        rotation : callable
            Vectorized function of TDB seconds, returning base_frame-to-body
            matrices with shape ``epoch.shape + (3,3)``.
        frame_class_id : int
            PCK frame class ID (not the body or frame ID). Load an FK mapping
            this ID to the intended class-2 body-fixed frame.
        base_frame : str, optional
            Callback inertial axes. Defaults to ``"J2000"``.
        interpolation_frame : str, optional
            PCK inertial base. Select axes avoiding the body's 3-1-3 Euler
            pole singularity. Defaults to ``"ECLIPJ2000"``.
        max_interval : float, optional
            Maximum Chebyshev record length, seconds. Use a shorter interval
            for fast rotation. Interpolation is checked. Defaults to
            ``21600.``.
        description : str, optional
            Provenance recorded in the PCK comments. Defaults to
            ``"User-supplied absolute body orientation"``.

        Returns
        -------
        BodyOrientation
            Absolute model supporting rotation_matrix and write_pck. The
            callback supplies the physics; no Earth rotation law is assumed.

        Raises
        ------
        ValueError
            If ``frame_class_id`` is not an integer, ``max_interval`` is not
            positive and finite, or ``base_frame`` or ``interpolation_frame``
            is not inertial.
        TypeError
            If ``rotation`` is not callable.
        """
        return cls(rotation=rotation, frame_class_id=frame_class_id,
                   base_frame=base_frame, interpolation_frame=interpolation_frame,
                   max_interval=max_interval, description=description)

    def displacement(
        self,
        position_body_fixed,
        epoch: float = None,
        frame_from: str = None,
        frame_to: str = None,
    ) -> ArrayWUnits:
        """
        Displacement of a body-fixed point caused by the correction.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like
            Position in the body-fixed frame, shape (3,). Bare numbers are km.
        epoch : float, optional
            TDB seconds past J2000. Required only when ``frame_to`` is given
            and differs from ``frame_from``. Defaults to ``None``.
        frame_from : str, optional
            The body-fixed frame ``position_body_fixed`` is expressed in.
            Defaults to ``None``, meaning :attr:`body_fixed_frame`.
        frame_to : str, optional
            Rotate the displacement into this frame before returning it.
            Defaults to ``None``, keeping it in ``frame_from``.

        Returns
        -------
        ArrayWUnits
            Displacement, km, shape (3,).

        Raises
        ------
        ValueError
            If this instance is not a correction model, or ``frame_to``
            differs from ``frame_from`` and ``epoch`` is not given.
        RuntimeError
            If SPICE cannot resolve the rotation from ``frame_from`` to
            ``frame_to``.
        """
        if self.rotation_vector is None:
            raise ValueError("displacement requires a correction model")
        frame_from = frame_from or self.body_fixed_frame
        position = ArrayWUnits.values_in(position_body_fixed, km).reshape(3)
        delta = np.cross(self.rotation_vector, position)

        if frame_to is None or frame_to == frame_from:
            return ArrayWUnits.from_trusted(delta, km)

        if epoch is None:
            raise ValueError("epoch is required when frame_to is given")
        try:
            rotation = SpiceManager.get_xfrm(frame_from, frame_to, epoch)
        except Exception as error:
            raise RuntimeError(
                f"cannot rotate the body-orientation correction from "
                f"{frame_from!r} to {frame_to!r} at epoch {epoch}: {error}"
            ) from error
        return ArrayWUnits.from_trusted(np.asarray(rotation) @ delta, km)

    def state_displacement(
        self,
        position_body_fixed,
        epoch: float,
        frame_from: str = None,
        frame_to: str = "J2000",
        velocity_body_fixed=None,
    ) -> ArrayWUnits:
        """
        Position *and* velocity displacement of a body-fixed point.

        A station is static in the body-fixed frame, so the correction moves it
        by :math:`\\varepsilon \\times r` there and leaves its body-fixed
        velocity at zero. Its inertial velocity still changes, by
        :math:`\\Omega \\times \\delta r` — the moved point sits on a different
        radius from the spin axis. That term is small but not negligible for
        Doppler: 23 m of displacement is 1.7 mm/s, against measurement noise
        often quoted at 1 mm/s.

        The state transformation carries both terms, so the body-fixed
        six-vector ``[delta_r, delta_v]`` (``delta_v`` is zero for a static
        station) is rotated with ``sxform`` rather than the position alone.

        Parameters
        ----------
        position_body_fixed : ArrayWUnits, ArrayWFrame or array_like
            Position in the body-fixed frame, shape (3,). Bare numbers are km.
        epoch : float
            TDB seconds past J2000.
        frame_from : str, optional
            The body-fixed frame ``position_body_fixed`` is expressed in.
            Defaults to ``None``, meaning :attr:`body_fixed_frame`.
        frame_to : str, optional
            Frame to express the displacement in. Defaults to ``"J2000"``.
        velocity_body_fixed : ArrayWUnits, ArrayWFrame or array_like, optional
            Nominal body-fixed velocity, shape (3,); bare numbers are km/s. Its
            cross product with the correction supplies the body-fixed velocity
            delta. Defaults to ``None``, a zero velocity.

        Returns
        -------
        ArrayWUnits
            ``[dx, dy, dz, dvx, dvy, dvz]`` in km and km/s, shape (6,).

        Raises
        ------
        ValueError
            If this instance is not a correction model.
        RuntimeError
            If SPICE cannot resolve the state transformation from
            ``frame_from`` to ``frame_to``.
        """
        import spiceypy as spice

        if self.rotation_vector is None:
            raise ValueError("displacement requires a correction model")
        frame_from = frame_from or self.body_fixed_frame
        position = ArrayWUnits.values_in(position_body_fixed, km).reshape(3)
        velocity = (np.zeros(3) if velocity_body_fixed is None
                    else ArrayWUnits.values_in(velocity_body_fixed, km / sec).reshape(3))
        delta_bf = np.concatenate([np.cross(self.rotation_vector, position),
                                   np.cross(self.rotation_vector, velocity)])
        if frame_to == frame_from:
            return ArrayWUnits.from_row_units(delta_bf, self.STATE_UNITS)
        try:
            transform = spice.sxform(frame_from, frame_to, float(epoch))
        except Exception as error:
            raise RuntimeError(
                f"cannot rotate the body-orientation state correction from "
                f"{frame_from!r} to {frame_to!r} at epoch {epoch}: {error}"
            ) from error
        return ArrayWUnits.from_row_units(np.asarray(transform) @ delta_bf,
                                          self.STATE_UNITS)


    @property
    def eop_option(self):
        """Selected Earth convention: EOP1 (ITRF93), EOP2 (IERS2010), or None.

        The option names identify the legacy and IERS rotation conventions.
        They do not perform a change of terrestrial station-coordinate datum.
        """
        if self.pole_x is None:
            return None
        return "EOP1" if self.model == "IAU1976/1980" else "EOP2"

    @property
    def convention(self):
        """The complete convention for report labels and PCK provenance."""
        if self.pole_x is None:
            return self.description
        if self.model == "IAU1976/1980":
            return ("EOP1 (ITRF93); IAU 1976/1980; GMST82 + (dPsi80+dPsi)cos(eps); s'=0; "
                    f"dPsi={self.nutation_delta_psi:.9g}, "
                    f"dEps={self.nutation_delta_epsilon:.9g} rad")
        return ("EOP2 (IERS2010); IAU 2006/2000A; CIO/ERA; s06, s'=sp00; "
                f"dX={self.celestial_pole_dx:.9g}, dY={self.celestial_pole_dy:.9g} rad")

    @staticmethod
    def _jd_pair(seconds):
        """Split seconds past J2000 into the two-part Julian date ERFA takes.

        Parameters
        ----------
        seconds : float or np.ndarray
            Seconds past J2000 in any uniform time scale; the Julian date is
            in that same scale.

        Returns
        -------
        jd_day : float or np.ndarray
            ``2451545.0`` plus the whole days elapsed, so it falls at noon.
        jd_fraction : float or np.ndarray
            Remaining fraction of a day, in [0, 1).
        """
        days = np.floor(seconds / 86400.0)
        return 2451545.0 + days, (seconds - days * 86400.0) / 86400.0

    def rotation_components(self, epoch):
        """Return (W, R, Q), with celestial -> terrestrial = W @ R @ Q.

        Each component has shape (3, 3), or epoch.shape + (3, 3) for arrays.
        For IAU1976/1980, Q maps mean-J2000 to the true equator/equinox of
        date and R uses GAST. For IAU2006/2000A, Q maps GCRS/ICRS to CIRS
        and R uses ERA. SPICE names the inertial base J2000 in both cases;
        it does not itself convert the legacy model's mean-J2000 axes to ICRS.
        The loaded Earth PCK is never consulted or decomposed. Its three
        rotation angles do not uniquely supply xp, yp, UT1 and two CPOs.

        Parameters
        ----------
        epoch : float or array_like
            TDB seconds past J2000.

        Returns
        -------
        polar_motion : np.ndarray
            W, the polar-motion matrix built from ``pole_x``, ``pole_y`` and
            s' (zero for IAU1976/1980).
        rotation : np.ndarray
            R, the spin about the pole of date by GAST (IAU1976/1980) or ERA
            (IAU2006/2000A).
        celestial : np.ndarray
            Q, precession-nutation from the celestial axes to the true
            equator and equinox of date (IAU1976/1980) or to CIRS
            (IAU2006/2000A).

        Raises
        ------
        ValueError
            If this instance has no absolute Earth EOPs, ``epoch`` is not
            finite, or the corrected celestial pole has X^2 + Y^2 >= 1.
        SpiceyError
            If no leap-second kernel is loaded.
        """
        if self.pole_x is None:
            raise ValueError("rotation_components requires earth_eop inputs")
        epochs = np.asarray(epoch, dtype=float)
        if not np.all(np.isfinite(epochs)):
            raise ValueError("epoch must be finite")
        tt = np.array([spice.unitim(float(t), "TDB", "TDT")
                       for t in epochs.ravel()]).reshape(epochs.shape)
        tt1, tt2 = self._jd_pair(tt)
        if self.ut1_time_scale == "TAI":
            atomic = np.array([spice.unitim(float(t), "TDB", "TAI")
                               for t in epochs.ravel()]).reshape(epochs.shape)
        else:
            # The ephemeris epoch used as if it were TT, so TAI = epoch - 32.184
            # and the TDB-TT periodic term never enters the spin angle.
            atomic = epochs - 32.184
        ut11, ut12 = self._jd_pair(atomic - self.tai_minus_ut1)
        if self.model == "IAU1976/1980":
            eps = erfa.obl80(tt1, tt2)
            dpsi, deps = erfa.nut80(tt1, tt2)
            celestial = erfa.numat(eps, dpsi + self.nutation_delta_psi,
                                   deps + self.nutation_delta_epsilon) @ erfa.pmat76(tt1, tt2)
            # JPL/NAIF (and MONTE) equation of the equinoxes: (dpsi+dPsi)cos(eps),
            # without the EqEq94 complementary terms (-2.6 mas in 2029).
            angle = erfa.gmst82(ut11, ut12) + (dpsi + self.nutation_delta_psi) * np.cos(eps)
            sp = np.zeros_like(epochs)
        else:
            x, y, _ = erfa.xys06a(tt1, tt2)
            x, y = x + self.celestial_pole_dx, y + self.celestial_pole_dy
            if np.any(x*x + y*y >= 1):
                raise ValueError("corrected celestial pole must satisfy X^2 + Y^2 < 1")
            celestial = erfa.c2ixys(x, y, erfa.s06(tt1, tt2, x, y))
            angle = erfa.era00(ut11, ut12)
            sp = erfa.sp00(tt1, tt2)
        polar_motion = erfa.pom00(self.pole_x, self.pole_y, sp)
        rotation = erfa.rz(angle, np.broadcast_to(np.eye(3), celestial.shape).copy())
        return polar_motion, rotation, celestial

    def rotation_matrix(self, epoch):
        """Return celestial -> terrestrial rotation at TDB seconds past J2000.

        See :meth:`rotation_components` for the celestial axes of each Earth
        convention.

        Parameters
        ----------
        epoch : float or array_like
            TDB seconds past J2000.

        Returns
        -------
        np.ndarray
            Rotation from the inertial base frame to the body-fixed frame,
            shape (3, 3) for a scalar epoch or ``epoch.shape + (3, 3)``.

        Raises
        ------
        ValueError
            If this instance is a correction model or ``epoch`` is not
            finite, or as raised by :meth:`rotation_components`.
        """
        if self.rotation_vector is not None:
            raise ValueError("rotation_matrix requires an absolute orientation model")
        if self.rotation is not None:
            epochs = np.asarray(epoch, dtype=float)
            if not np.all(np.isfinite(epochs)):
                raise ValueError("epoch must be finite")
            return np.asarray(self.rotation(epochs), dtype=float).reshape(epochs.shape + (3, 3))
        polar_motion, rotation, celestial = self.rotation_components(epoch)
        return polar_motion @ rotation @ celestial

    def _pck_comments(self):
        """Provenance lines for the PCK comment area.

        Returns
        -------
        list of str
            The ``description`` alone for a user-supplied rotation, or the
            convention, EOP values, UT1 time scale and ERFA version for an
            Earth EOP model.
        """
        if self.pole_x is None:
            return [self.description]
        return [
            "SCB absolute constant Earth EOP inputs; no observation fitting.",
            self.convention + ".",
            f"PoleX [rad]: {self.pole_x:.17g}",
            f"PoleY [rad]: {self.pole_y:.17g}",
            f"TAI-UT1 [s]: {self.tai_minus_ut1:.17g}",
            f"UT1 formed from: {self.ut1_time_scale}"
            + ("" if self.ut1_time_scale == "TAI" else
               " (ephemeris epoch used as TT; TDB-TT omitted)"),
            f"ERFA {erfa.__version__}; SOFA-derived routines.",
        ]

    def write_pck(self, path, start, stop):
        """Write a new type-2 PCK for this model's frame class ID.

        start/stop are TDB seconds past J2000. The caller must explicitly
        load the returned file after competing PCKs to select it within
        this interval. Existing files are not overwritten or unloaded.
        Outside this interval SPICE may use another loaded PCK.

        Degree-12 Chebyshev records span at most :attr:`max_interval` (six
        hours by default). Euler angles use :attr:`interpolation_frame`
        (ECLIPJ2000 by default) as their base to avoid the near-J2000-pole
        singularity. Their interpolation is checked against direct
        :meth:`rotation_matrix` values at independent points, including
        every record boundary, before writing.

        Parameters
        ----------
        path : str or pathlib.Path
            Output PCK file. It must not exist yet.
        start, stop : float
            Coverage interval, TDB seconds past J2000, with ``stop > start``.

        Returns
        -------
        pathlib.Path
            The written PCK file.

        Raises
        ------
        ValueError
            If this instance is a correction model; ``start`` or ``stop`` is
            not finite or ``stop <= start``; :meth:`rotation_matrix` does not
            return proper orthogonal matrices; or the checked interpolation
            error exceeds 5e-11.
        FileExistsError
            If ``path`` already exists.
        """
        if self.rotation_vector is not None:
            raise ValueError("write_pck requires an absolute orientation model")
        start, stop = float(start), float(stop)
        if not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError("start and stop must be finite, with stop > start")
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        count = max(1, int(np.ceil((stop - start) / self.max_interval)))
        interval = (stop - start) / count
        degree = 12
        nodes = np.polynomial.chebyshev.chebpts1(degree + 1)
        checks = np.linspace(-1.0, 1.0, 2 * degree + 3)
        ecliptic_to_j2000 = spice.pxform(self.interpolation_frame, self.base_frame, 0.0)
        coefficients = []
        max_error = 0.0
        for i in range(count):
            middle = start + (i + 0.5) * interval
            epochs = middle + nodes * interval / 2.0
            matrices = self.rotation_matrix(epochs) @ ecliptic_to_j2000
            if (not np.all(np.isfinite(matrices))
                    or not np.allclose(matrices @ matrices.swapaxes(-1, -2), np.eye(3), atol=1e-12, rtol=0)
                    or not np.allclose(np.linalg.det(matrices), 1, atol=1e-12, rtol=0)):
                raise ValueError("rotation_matrix must return proper orthogonal matrices")
            # m2eul returns angle3, angle2, angle1. Binary PCK type 2 stores
            # angle1, angle2, angle3 of the 3-1-3 sequence (no RA/DEC offsets).
            angles = np.array([spice.m2eul(m, 3, 1, 3)[::-1] for m in matrices])
            angles = np.unwrap(angles, axis=0)
            coeff = chebyshev.chebfit(nodes, angles, degree).T
            coefficients.append(coeff.ravel())
            fitted = chebyshev.chebval(checks, coeff.T).T
            actual = self.rotation_matrix(middle + checks * interval / 2.0)
            for values, expected in zip(fitted, actual):
                matrix = spice.eul2m(*values[::-1], 3, 1, 3)
                max_error = max(max_error, float(np.max(np.abs(
                    matrix - expected @ ecliptic_to_j2000))))
        # TDB/TT conversion at modern epochs is limited by the resolution of
        # a double holding ~1e9 seconds. Allow that sub-millimetre floor.
        if max_error > 5e-11:
            raise ValueError(f"Body PCK interpolation error {max_error:.3g} exceeds 5e-11")

        comments = self._pck_comments() + [
            f"Coverage [TDB seconds]: {start:.17g} to {stop:.17g}",
            f"Frame class ID {self.frame_class_id}; inertial base {self.interpolation_frame}.",
            f"Degree {degree}; {count} records; max checked matrix error {max_error:.3g}.",
        ]
        identifier = "SCB body orientation"
        handle = spice.pckopn(str(path), identifier, 2048)
        try:
            spice.dafac(handle, comments)
            spice.pckw02(handle, self.frame_class_id, self.interpolation_frame, start, stop,
                         identifier, interval, count,
                         degree, np.concatenate(coefficients), start)
        finally:
            spice.pckcls(handle)
        return path
