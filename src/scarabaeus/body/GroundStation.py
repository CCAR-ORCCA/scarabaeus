# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (Body, Spacecraft, EpochArray, SpiceManager, ArrayWUnits, ArrayWFrame,
                        Dimensions, Frame, Units)
from scarabaeus.utils.Utils import Utils
import scarabaeus.utils.NumpyWrapper as np
import spiceypy as spice

km, sec, rad, deg, day, unitless = Units.get_units(["km", "sec", "rad", "deg", "day", "unitless"])


# class definition
class GroundStation(Body):
    """Represents a ground station for spacecraft communication modeling.

    A ground station is a :class:`~scarabaeus.Body` with no mass or SRP
    properties.  Its primary role is to supply the observer position for
    radiometric observables (range, range-rate, Doppler).

    Parameters
    ----------
    name : str
        The name of the ground station.

    spice_id : int or float, optional
        The SPICE ID of the ground station, defaults to ``None``.

    central_body : str or Body, optional
        Body supporting the station, default EARTH. The supplied coordinates
        or station kernels express its position relative to this center.
    body_fixed_frame : Frame or str, optional
        Body-fixed SPICE frame. For another planet supply its frame, e.g.
        IAU_MARS, together with ``central_body="MARS"``. Defaults to
        ``'ITRF93'``.
    body_fixed_position : ArrayWUnits or array_like, optional
        Fixed Cartesian station coordinates relative to ``central_body`` in
        ``body_fixed_frame``. Bare values are km. When supplied, no station SPK
        is needed; otherwise the station position comes from loaded kernels.
        Use :meth:`from_cylindrical` for a longitude/radius/Z table.
    body_fixed_velocity : ArrayWUnits or array_like, optional
        Constant body-fixed velocity of explicit coordinates, plate motion.
        Bare values are km/s. Requires ``position_epoch``.
    position_epoch : EpochArray or float, optional
        Single epoch at which ``body_fixed_position`` holds; a float is TDB
        seconds past J2000. Defaults to ``None``.
    surface_displacement : object or list of objects, optional
        Time-varying station displacement, e.g. :class:`~scarabaeus.SolidTide`,
        :class:`~scarabaeus.PoleTide`, :class:`~scarabaeus.OceanLoading`, or a
        list of them, summed. See :attr:`surface_displacement`.
    barycentric_transformation : bool, optional
        Apply the geocentric-to-barycentric transformation to space-fixed
        states, default false. See :attr:`barycentric_transformation`.
    body_orientation : BodyOrientation, optional
        Small orientation correction to the loaded body-fixed frame, from
        :meth:`~scarabaeus.BodyOrientation.earth_correction`. Absolute EOPs
        are loaded as a PCK instead. See :attr:`body_orientation`.

    See Also
    --------
    scarabaeus.CelestialBody : Planetary body on whose surface a GroundStation is placed.
    """

    STATE_UNITS = (km,) * 3 + (km / sec,) * 3
    """Row units of a station state or state delta, and of its Jacobian with
    respect to a unitless parameter such as a Love number."""

    LOCATION_PARTIAL_UNITS = (unitless,) * 3 + (unitless / sec,) * 3
    """Row units of :meth:`location_partials`, per km of location delta."""

    ROTATION_PARTIAL_UNITS = (km / rad,) * 3 + (km / sec / rad,) * 3
    """Row units of a state Jacobian per radian of rotation or pole angle."""

    POLE_PM_PARTIAL_UNITS = (km / (deg / day),) * 3 + (
        Units(Dimensions([0, 1, 0, -1]), [0., km.scales[1] * day.scales[2], 0., deg.scales[3]]),) * 3
    """Row units of the ``pole_pm`` column of :meth:`iau_pole_partials`, per
    deg/day. The velocity rows are (km/s)/(deg/day): time cancels there, and
    :class:`~scarabaeus.Units` keeps no scale for a cancelled dimension, so
    the day-to-second factor is carried by the length scale."""

    # region Constructor
    def __init__(self, name: str, spice_id: int | str = None, *,
                 central_body="EARTH", body_fixed_frame="ITRF93", body_fixed_position=None,
                 body_fixed_velocity=None, position_epoch=None,
                 surface_displacement=None, barycentric_transformation=False,
                 body_orientation=None):
        # Parent class initializations
        super().__init__(name, spice_id)

        # Optional EOP correction applied on top of the loaded kernels
        self.body_orientation = body_orientation
        self.body_fixed_frame = body_fixed_frame
        self.central_body = central_body
        self.body_fixed_position = body_fixed_position
        self.body_fixed_velocity = body_fixed_velocity
        self.position_epoch = position_epoch
        self.surface_displacement = surface_displacement
        self.barycentric_transformation = barycentric_transformation

    class MissingPositionEpochError(Exception):
        """A station was given a :attr:`body_fixed_velocity` but no
        :attr:`position_epoch` to move it from."""

    class VelocityWithoutPositionError(Exception):
        """A :attr:`body_fixed_velocity` was set on a station whose position
        comes from a station kernel, which carries its own motion."""

    class InvalidEpochError(ValueError):
        """A :attr:`position_epoch` was not finite."""

    @classmethod
    def from_cylindrical(cls, name, spice_id=None, *, longitude,
                         cylindrical_radius, z, central_body="EARTH",
                         body_fixed_frame="ITRF93", **kwargs):
        r"""Define a fixed station from absolute cylindrical table coordinates.

        Parameters
        ----------
        name : str
            Station name.
        spice_id : int or str, optional
            NAIF identifier; required for a name unknown to SPICE.
            Defaults to ``None``, resolved from ``name``.
        longitude : ArrayWUnits or float
            East-positive longitude. Bare values are radians; unit-bearing
            values may use degrees. This is an angle, not an east arc length.
        cylindrical_radius, z : ArrayWUnits or float
            Distance from the spin axis and signed height along that axis.
            Bare values are km. These are absolute coordinates, not estimates
            or corrections to a station kernel.
        central_body : str or Body, optional
            Supporting body, default EARTH.
        body_fixed_frame : str, optional
            Frame of the supplied table, default ITRF93.
        **kwargs
            Passed to the constructor, e.g. ``body_fixed_velocity``,
            ``position_epoch``, ``surface_displacement``,
            ``barycentric_transformation`` or ``body_orientation``.

        Returns
        -------
        GroundStation
            Station with Cartesian coordinates
            :math:`(\rho\cos\lambda,\rho\sin\lambda,Z)` and zero velocity in
            the supplied frame. Frame rotation supplies its inertial velocity.

        Raises
        ------
        ValueError
            For non-scalar/non-finite inputs or a negative cylindrical radius.
        """
        values = []
        for value, unit in ((longitude, "rad"), (cylindrical_radius, "km"), (z, "km")):
            if hasattr(value, "convert_to"):
                value = value.convert_to(Units.get_units([unit])).values
            value = np.asarray(value, dtype=float)
            if value.size != 1 or not np.all(np.isfinite(value)):
                raise ValueError("Cylindrical station coordinates must be finite scalars")
            values.append(float(value.reshape(-1)[0]))
        angle, radius, height = values
        if radius < 0:
            raise ValueError("Cylindrical radius must be nonnegative")
        return cls(name, spice_id, central_body=central_body,
                   body_fixed_frame=body_fixed_frame,
                   body_fixed_position=[radius*np.cos(angle), radius*np.sin(angle), height],
                   **kwargs)

    # region Properties
    @property
    def body_fixed_position(self):
        """Explicit Cartesian coordinates in km, or None for a kernel station.

        Returns a copy. Assign three finite lengths (ArrayWUnits or km values)
        to select a fixed table position; assign None to use the station SPK.
        No station kernel is written, loaded, or modified by this setting.
        """
        if self._body_fixed_position is None:
            return None
        return ArrayWUnits(self._body_fixed_position.copy(), km)

    @body_fixed_position.setter
    def body_fixed_position(self, value):
        if value is None:
            self._body_fixed_position = None
            return
        if hasattr(value, "convert_to"):
            value = value.convert_to(km).values
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("body_fixed_position must contain three finite Cartesian lengths")
        self._body_fixed_position = value.copy()

    @property
    def body_fixed_velocity(self):
        """Constant body-fixed velocity of explicit coordinates in km/s, or None.

        Plate motion. Published station coordinates hold at an epoch,
        :attr:`position_epoch`, and move from there at a published rate: the
        station is at ``body_fixed_position + body_fixed_velocity * (t -
        position_epoch)`` at every epoch ``t``, with this velocity as its
        body-fixed rate. Returns a copy. Assign three finite speeds
        (ArrayWUnits or km/s values), or None for a station that does not move.
        """
        if self._body_fixed_velocity is None:
            return None
        return ArrayWUnits(self._body_fixed_velocity.copy(), km / sec)

    @body_fixed_velocity.setter
    def body_fixed_velocity(self, value):
        if value is None:
            self._body_fixed_velocity = None
            return
        if hasattr(value, "convert_to"):
            value = value.convert_to(km / sec).values
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("body_fixed_velocity must contain three finite Cartesian speeds")
        self._body_fixed_velocity = value.copy()

    @property
    def position_epoch(self):
        """TDB seconds past J2000 at which :attr:`body_fixed_position` holds, or None.

        Needed once :attr:`body_fixed_velocity` is set. There is no default:
        a coordinate table names its epoch, and a wrong one moves the station
        by its velocity times the error. Assign a single
        :class:`~scarabaeus.EpochArray`, in any time system, or TDB seconds;
        a non-finite epoch raises :class:`InvalidEpochError`.
        """
        return self._position_epoch

    @position_epoch.setter
    def position_epoch(self, value):
        if value is None:
            self._position_epoch = None
            return
        value = float(getattr(value, "et", value))
        if not np.isfinite(value):
            raise self.InvalidEpochError("position_epoch must be finite")
        self._position_epoch = value

    @property
    def earth_orientation(self):
        """
        Earth orientation correction applied to this station, or ``None``.

        A body-fixed frame from a binary PCK already carries that kernel's polar
        motion and UT1. Setting an
        :class:`~scarabaeus.BodyOrientation` here applies the difference
        between the kernels' values and the ones an analysis pins, so the
        station's inertial position matches the intended Earth orientation.

        See Also
        --------
        scarabaeus.BodyOrientation.earth_correction : Construct an Earth EOP correction.
        """
        return self._earth_orientation

    @earth_orientation.setter
    def earth_orientation(self, value):
        self.body_orientation = value

    @property
    def body_orientation(self):
        """Optional :class:`~scarabaeus.BodyOrientation` station correction.

        ``earth_orientation`` remains an alias. This is a correction to the
        loaded nominal frame, not an absolute EOP model or a force-model change.
        """
        return self._earth_orientation

    @body_orientation.setter
    def body_orientation(self, value):
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        if value is not None and not isinstance(value, BodyOrientation):
            raise TypeError("body_orientation must be BodyOrientation or None")
        if value is not None and value.rotation_vector is None:
            raise ValueError("Attach a correction model; load an absolute model as a PCK instead")
        self._earth_orientation = value

    @property
    def surface_displacement(self):
        """Optional time-varying displacement of this station in its body-fixed frame.

        Set a :class:`~scarabaeus.SolidTide` to move the station with the
        solid tide. Unlike :attr:`body_orientation`, this is not a correction to
        a nominal that might instead have been right: the station really is
        somewhere else at this epoch, so the displacement enters
        :meth:`nominal_state` and every estimated parameter is linearised about
        the displaced point.

        Any object exposing ``state_displacement(position, epoch, frame)`` and
        returning six components, as an :class:`~scarabaeus.ArrayWUnits` or
        bare km and km/s, works, such as :class:`~scarabaeus.PoleTide` and
        :class:`~scarabaeus.OceanLoading`. The position it receives is bare km.
        A list or tuple of such models is summed, each evaluated at the
        undisplaced position. The property returns whatever was set.
        """
        return self._surface_displacement

    @property
    def tide_reference(self):
        """The attached tide whose Love and Shida numbers an estimate refers to.

        Returns
        -------
        object or None
            The first :attr:`surface_displacement` model with
            ``state_displacement_partials``, or None when none is attached.

        Notes
        -----
        A :class:`~scarabaeus.SolidTide` and a :class:`~scarabaeus.PoleTide`
        carry different nominal :math:`h_2` and :math:`l_2`. One estimated
        number is read as this model's, and every attached tide shifts by the
        estimate minus this model's number (:meth:`tide_correction`). Leaving
        the estimate at this model's number therefore leaves every tide where
        it was.
        """
        return next((model for model in self._surface_displacements
                     if hasattr(model, "state_displacement_partials")), None)

    @surface_displacement.setter
    def surface_displacement(self, value):
        models = value if isinstance(value, (list, tuple)) else (value,)
        for model in models:
            if model is not None and not hasattr(model, "state_displacement"):
                raise TypeError("surface_displacement needs a state_displacement("
                                "position, epoch, frame) method")
        self._surface_displacement = value
        self._surface_displacements = tuple(model for model in models if model is not None)

    @property
    def barycentric_transformation(self) -> bool:
        r"""Whether space-fixed states carry the geocentric-to-barycentric transformation.

        Station coordinates are geocentric and in terrestrial units; a
        spacecraft trajectory and the planetary ephemerides are barycentric, in
        TDB units. Combining them consistently takes the Lorentz transformation
        of the station's geocentric space-fixed position into the solar-system
        barycentric frame (Moyer, *Formulation for Observed and Computed Values
        of Deep Space Network Data Types for Navigation*, 2000, eq. 4-10):

        .. math::

            \mathbf r_B = \left(1 - \tilde L - \frac{\gamma U}{c^2}\right)\mathbf r
                - \frac{(\mathbf V \cdot \mathbf r)\,\mathbf V}{2c^2}

        with :math:`U` the Sun's potential at the central body, :math:`\mathbf V`
        the central body's barycentric velocity, and :math:`\tilde L`,
        :math:`\gamma` from :data:`~scarabaeus.constants`. The scale term is
        2.5e-8, about 16 cm at the Earth's surface, and the velocity term up to
        3 cm on a daily cycle. JPL's navigation software reports station states
        this way.

        When true, :meth:`nominal_state` applies it to output in any axes but
        the station's body-fixed frame, whose output is unchanged; being a
        scale and a dot product, it is the same in any axes. Only the position
        is transformed. The velocity term would be 0.01 mm/s, and the
        light-time observables' own station transformation
        (``Measurement._compute_GS_in_SSB_space_time_RLT``) is likewise applied
        to position alone. Defaults to false.

        Raises
        ------
        TypeError
            If set to anything other than a bool.
        """
        return self._barycentric_transformation

    @barycentric_transformation.setter
    def barycentric_transformation(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("barycentric_transformation must be True or False")
        self._barycentric_transformation = value

    def _to_barycentric(self, position, epoch: float, frame: str) -> np.ndarray:
        """Carry a geocentric space-fixed position into the barycentric frame.

        See :attr:`barycentric_transformation` for the transformation and
        its size.

        Parameters
        ----------
        position : array_like, shape (3,)
            Station position relative to :attr:`central_body`, km, in
            ``frame`` axes.
        epoch : float
            TDB seconds past J2000.
        frame : str
            SPICE frame name of the axes ``position`` is expressed in.

        Returns
        -------
        numpy.ndarray
            Transformed position in km, shape (3,), still in ``frame`` axes.
        """
        from scarabaeus import constants

        centre = str(getattr(self.central_body, "spice_id", self.central_body))
        c = constants.c.values
        potential = constants.SUN.GM.values / np.linalg.norm(
            spice.spkpos(centre, epoch, "J2000", "NONE", "SUN")[0])
        velocity = np.asarray(spice.spkezr(centre, epoch, "J2000", "NONE", "SSB")[0][3:])
        if frame != "J2000":
            # The transformation is built from a scale and a dot product, so it
            # holds in any axes once the velocity is expressed in them.
            velocity = spice.pxform("J2000", frame, epoch) @ velocity
        return ((1 - constants.L_tilde_GR.values
                 - constants.gamma_GR.values * potential / c**2) * position
                - np.dot(velocity, position) * velocity / (2 * c**2))

    @property
    def body_fixed_frame(self) -> str:
        """Body-fixed SPICE frame name of this station, e.g. ITRF93 or IAU_MARS.

        Station coordinates, surface displacements and estimated location and
        orientation corrections are all expressed in these axes. Assign a
        :class:`~scarabaeus.Frame` or a frame name; the name is stored.
        """
        return self._body_fixed_frame

    @body_fixed_frame.setter
    def body_fixed_frame(self, value):
        self._body_fixed_frame = str(getattr(value, "name", value))

    def _undisplaced_state(self, epoch: float) -> np.ndarray:
        """Body-fixed state of this station before any surface displacement.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.

        Returns
        -------
        numpy.ndarray
            Position and velocity relative to :attr:`central_body` in
            :attr:`body_fixed_frame`, km and km/s, shape (6,): the explicit
            coordinates moved by any :attr:`body_fixed_velocity`, or the
            station kernel state.

        Raises
        ------
        MissingPositionEpochError
            If :attr:`body_fixed_velocity` is set but :attr:`position_epoch`
            is not.
        VelocityWithoutPositionError
            If :attr:`body_fixed_velocity` is set on a station whose position
            comes from a station kernel.
        """
        if self._body_fixed_position is not None:
            if self._body_fixed_velocity is None:
                return np.r_[self._body_fixed_position, np.zeros(3)]
            if self._position_epoch is None:
                raise self.MissingPositionEpochError(
                    f"{self.name} has a body_fixed_velocity but no position_epoch "
                    "for its body_fixed_position")
            elapsed = float(epoch) - self._position_epoch
            return np.r_[self._body_fixed_position + self._body_fixed_velocity * elapsed,
                         self._body_fixed_velocity]
        if self._body_fixed_velocity is not None:
            raise self.VelocityWithoutPositionError(
                f"{self.name} takes its position from a station kernel; "
                "body_fixed_velocity applies to body_fixed_position only")
        center = getattr(self.central_body, "spice_id", self.central_body)
        return np.asarray(spice.spkezr(str(self.spice_id), float(epoch),
                          self.body_fixed_frame, "NONE", str(center))[0])

    @staticmethod
    def _framed(values, units, frame) -> ArrayWFrame:
        """Attach units and a frame to a numeric state or position.

        Parameters
        ----------
        values : numpy.ndarray
            Numbers in ``units``.
        units : Units or sequence of Units
            One unit for every element, or one per row, e.g.
            :attr:`STATE_UNITS`.
        frame : Frame or str
            Axes of ``values``. A :class:`~scarabaeus.Frame` is reused as is.

        Returns
        -------
        ArrayWFrame
            ``values`` with its units, in ``frame``.
        """
        quantity = (ArrayWUnits.from_trusted(values, units) if isinstance(units, Units)
                    else ArrayWUnits.from_row_units(values, units))
        return ArrayWFrame(quantity, frame if isinstance(frame, Frame) else Frame(frame))

    @staticmethod
    def _model_values(model, method: str, row_units, *arguments) -> np.ndarray:
        """Bare-number result of one :attr:`surface_displacement` model method.

        Parameters
        ----------
        model : object
            An attached surface-displacement model.
        method : str
            Name of its public method, e.g. ``"state_displacement"``.
        row_units : sequence of Units
            Units of each row of the result, the ones it is returned in.
        *arguments
            Bare-number arguments of the method: position in km, epoch in TDB
            seconds past J2000 and a frame name.

        Returns
        -------
        numpy.ndarray
            The result in ``row_units``. A model with a numeric core
            ``_<method>``, as :class:`~scarabaeus.SolidTide`,
            :class:`~scarabaeus.PoleTide` and :class:`~scarabaeus.OceanLoading`
            have, is read through it, skipping the units its public method
            attaches only for them to be stripped here. Any other model may
            return ArrayWUnits or bare numbers, taken to be in ``row_units``.
        """
        core = getattr(model, "_" + method, None)
        if core is not None:
            return core(*arguments)
        return ArrayWUnits.values_in(getattr(model, method)(*arguments), row_units)

    def nominal_state(self, epoch, frame=None) -> ArrayWFrame:
        """Return this station's uncorrected state relative to its central body.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output SPICE frame. Defaults to :attr:`body_fixed_frame`.

        Returns
        -------
        ArrayWFrame
            Six components in km and km/s in ``frame``, from the explicit coordinates
            (moved by any :attr:`body_fixed_velocity`) or loaded station SPK,
            plus any attached
            :attr:`surface_displacement`. Orientation still comes from SPICE
            frames. This deliberately bypasses *estimated* corrections so they
            are applied exactly once by :meth:`state_correction`; a tidal
            displacement is not estimated, it is where the station actually is,
            so estimated corrections are linearised about the displaced point.
            Any ``frame`` but the body-fixed one also carries the
            :attr:`barycentric_transformation` when that is enabled.

        Raises
        ------
        MissingPositionEpochError
            If :attr:`body_fixed_velocity` is set but :attr:`position_epoch`
            is not.
        VelocityWithoutPositionError
            If :attr:`body_fixed_velocity` is set on a kernel station.
        """
        return self._framed(self._nominal_state(epoch, frame), self.STATE_UNITS,
                            self.body_fixed_frame if frame is None else frame)

    def _nominal_state(self, epoch, frame=None) -> np.ndarray:
        """Numeric core of :meth:`nominal_state`, in km and km/s."""
        frame = getattr(frame, "name", frame) or self.body_fixed_frame
        epoch = float(getattr(epoch, "et", epoch))
        # Build the state body-fixed first so a surface displacement, which is
        # defined in those axes, is added before any frame transformation.
        state = self._undisplaced_state(epoch)
        position = state[:3]
        for model in self._surface_displacements:
            state = state + self._model_values(model, "state_displacement", self.STATE_UNITS,
                                               position, epoch, self.body_fixed_frame)
        if frame == self.body_fixed_frame:
            return state
        state = spice.sxform(self.body_fixed_frame, frame, epoch) @ state
        if self._barycentric_transformation:
            state = np.r_[self._to_barycentric(state[:3], epoch, frame), state[3:]]
        return state

    def cylindrical_basis(self, epoch, frame=None) -> tuple:
        r"""Return the local cylindrical basis and its time derivative.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch within the required kernel coverage; a float is TDB
            seconds past J2000.
        frame : Frame or str, optional
            Body-fixed axes. Defaults to :attr:`body_fixed_frame`.

        Returns
        -------
        basis : numpy.ndarray
            (3,3) unitless matrix mapping (radial, east, Z) lengths into
            Cartesian body-fixed coordinates, like any direction cosine matrix.
        derivative : ArrayWUnits
            Its (3,3) rate, per second.

        Raises
        ------
        ValueError
            At the spin axis, where nominal longitude is undefined.

        Notes
        -----
        The basis is evaluated at the actual epoch using the station's own
        central body. East is an arc length, :math:`\rho\,\Delta\lambda`,
        not a longitude angle. Body-fixed station motion gives
        :math:`\dot\lambda=(x\dot y-y\dot x)/(x^2+y^2)`; its contribution
        to the inertial velocity must be retained when estimating Doppler.
        """
        basis, derivative = self._cylindrical_basis(epoch, frame)
        return basis, ArrayWUnits.from_trusted(derivative, unitless / sec)

    def _cylindrical_basis(self, epoch, frame=None) -> tuple:
        """Numeric core of :meth:`cylindrical_basis`, the rate per second."""
        x, y, _, vx, vy, _ = self._nominal_state(epoch, frame)
        rho2 = x*x + y*y
        if rho2 <= np.finfo(float).tiny:
            raise ValueError("Cylindrical station longitude is undefined on the spin axis")
        longitude = np.arctan2(y, x)
        cosine, sine = np.cos(longitude), np.sin(longitude)
        basis = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        longitude_rate = (x*vy - y*vx) / rho2
        derivative = longitude_rate * np.array([[-sine, -cosine, 0.],
                                                [cosine, -sine, 0.], [0., 0., 0.]])
        return basis, derivative

    def state_correction(self, epoch, frame="J2000", *,
                         location=None, location_frame=None,
                         cylindrical: bool = False, rotation_vector=None) -> ArrayWFrame:
        """Transform station location and orientation corrections into a state delta.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        location : ArrayWFrame, ArrayWUnits or array_like, shape (3,), optional
            Estimated body-fixed coordinate delta. Bare numbers are km.
            Defaults to ``None``, meaning zero.
        location_frame : Frame or str, optional
            Axes of ``location``. Defaults to the frame of an ArrayWFrame
            ``location``, else :attr:`body_fixed_frame`.
        cylindrical : bool, optional
            If true, interpret ``location`` as (delta radius, east arc length,
            delta Z). Otherwise it is Cartesian. Defaults to ``False``.
        rotation_vector : ArrayWUnits or array_like, shape (3,), optional
            Estimated active body-fixed X/Y/Z rotation, added to the attached
            :attr:`body_orientation`, if present. Bare numbers are radians.
            Defaults to ``None``, meaning zero.

        Returns
        -------
        ArrayWFrame
            Position/velocity displacement in km and km/s in ``frame``,
            shape (6,).

        Raises
        ------
        ValueError
            For non-finite corrections, undefined cylindrical longitude, or an
            attached orientation expressed in a different body-fixed frame.

        Notes
        -----
        Location is applied before the first-order rigid rotation, retaining
        the location/orientation cross term. The full ``sxform`` carries both
        the rotated displacement and its velocity. This method uses nominal
        kernels directly and does not change their frame definitions.
        """
        return self._framed(self._state_correction(
            epoch, frame, location=ArrayWUnits.values_in(location, km),
            location_frame=location_frame or getattr(location, "frame", None),
            cylindrical=cylindrical,
            rotation_vector=ArrayWUnits.values_in(rotation_vector, rad)),
            self.STATE_UNITS, frame)

    def _state_correction(self, epoch, frame="J2000", *,
                          location=None, location_frame=None,
                          cylindrical: bool = False, rotation_vector=None) -> np.ndarray:
        """Numeric core of :meth:`state_correction`: km and radians in, km and km/s out."""
        epoch = float(getattr(epoch, "et", epoch))
        fixed_frame = self.body_fixed_frame
        frame = getattr(frame, "name", frame)
        nominal = self._nominal_state(epoch)
        delta = np.zeros(6)
        rotation = self._rotation_vector(rotation_vector)
        if location is not None:
            values = np.asarray(location, dtype=float).reshape(3)
            location_frame = getattr(location_frame, "name", location_frame) or fixed_frame
            if cylindrical:
                basis, derivative = self._cylindrical_basis(epoch, location_frame)
                location_state = np.r_[basis @ values, derivative @ values]
            else:
                location_state = np.r_[values, np.zeros(3)]
            delta += spice.sxform(location_frame, fixed_frame, epoch) @ location_state
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(rotation)):
            raise ValueError("Station corrections must be finite")
        moved = nominal + delta
        delta += np.r_[np.cross(rotation, moved[:3]), np.cross(rotation, moved[3:])]
        return spice.sxform(fixed_frame, frame, epoch) @ delta

    def _centre_offset(self, epoch: float, frame: str, origin) -> np.ndarray:
        """Position of this station's central body relative to ``origin``.

        Zero in the ordinary case where the two are the same body, so a
        station-relative geometry does not pay for a redundant SPK query.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        frame : str
            SPICE frame name of the returned vector.
        origin : str or int or Body
            Observing body, as a SPICE name or ID or an object with a ``name``.

        Returns
        -------
        numpy.ndarray
            Offset in km, shape (3,).
        """
        centre = getattr(self.central_body, "spice_id", self.central_body)
        if spice.bods2c(str(centre)) == spice.bods2c(str(getattr(origin, "name", origin))):
            return np.zeros(3)
        return np.asarray(spice.spkezr(str(centre), float(epoch), frame, "NONE",
                                       str(getattr(origin, "name", origin)))[0])[:3]

    def _rotation_vector(self, extra=None) -> np.ndarray:
        """Total body-fixed correction rotation applied to this station.

        Parameters
        ----------
        extra : array_like, shape (3,), optional
            Additional active X/Y/Z rotation in radians, typically the current
            estimate. Defaults to ``None``, meaning zero.

        Returns
        -------
        numpy.ndarray
            ``extra`` plus the rotation vector of any attached
            :attr:`body_orientation`, in radians, shape (3,).

        Raises
        ------
        ValueError
            If the attached :attr:`body_orientation` uses a body-fixed frame
            other than :attr:`body_fixed_frame`.
        """
        rotation = np.zeros(3) if extra is None else np.asarray(extra, dtype=float).reshape(3).copy()
        if self.body_orientation is not None:
            if self.body_orientation.body_fixed_frame != self.body_fixed_frame:
                raise ValueError("Station and body_orientation must use the same body-fixed frame")
            rotation = rotation + self.body_orientation.rotation_vector
        return rotation

    def location_partials(self, epoch, frame="J2000", *,
                          cylindrical: bool = False, location_frame=None,
                          rotation_vector=None) -> ArrayWUnits:
        """Sensitivity of this station's state to its body-fixed location delta.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        cylindrical : bool, optional
            Differentiate with respect to (radius, east arc length, Z) rather
            than Cartesian components. Defaults to ``False``.
        location_frame : Frame or str, optional
            Axes the delta is declared in. Defaults to
            :attr:`body_fixed_frame`.
        rotation_vector : ArrayWUnits or array_like, shape (3,), optional
            Estimated body-fixed rotation, added to any attached
            :attr:`body_orientation` for the location/orientation cross term.
            Bare numbers are radians. Defaults to ``None``, meaning zero.

        Returns
        -------
        ArrayWUnits
            ``(6, 3)`` Jacobian in km and km/s per km
            (:attr:`LOCATION_PARTIAL_UNITS`), in ``frame`` axes, matching the
            differentiated :meth:`state_correction` exactly rather than
            approximating it.

        Raises
        ------
        ValueError
            For undefined cylindrical longitude, or an attached orientation
            expressed in a different body-fixed frame.

        Notes
        -----
        The cylindrical basis rotates with the body, so its time derivative
        contributes a velocity row: a station's east coordinate is not static
        in inertial space. Omitting it biases any Doppler partial.
        """
        return ArrayWUnits.from_row_units(self._location_partials(
            epoch, frame, cylindrical=cylindrical, location_frame=location_frame,
            rotation_vector=ArrayWUnits.values_in(rotation_vector, rad)),
            self.LOCATION_PARTIAL_UNITS)

    def _location_partials(self, epoch, frame="J2000", *,
                           cylindrical: bool = False, location_frame=None,
                           rotation_vector=None) -> np.ndarray:
        """Numeric core of :meth:`location_partials`, rotation in radians."""
        epoch = float(getattr(epoch, "et", epoch))
        fixed_frame = self.body_fixed_frame
        frame = getattr(frame, "name", frame)
        location_frame = getattr(location_frame, "name", location_frame) or fixed_frame
        if cylindrical:
            basis, derivative = self._cylindrical_basis(epoch, location_frame)
        else:
            basis, derivative = np.eye(3), np.zeros((3, 3))
        # A location delta is three numbers with no velocity of its own: the
        # velocity rows come from the basis turning with the body, not from a
        # separate rate being estimated.
        block = np.vstack([basis, derivative])
        to_fixed = np.asarray(spice.sxform(location_frame, fixed_frame, epoch))
        rotation = Utils.skew(self._rotation_vector(rotation_vector))
        cross_term = np.block([[rotation, np.zeros((3, 3))], [np.zeros((3, 3)), rotation]])
        jacobian = (np.eye(6) + cross_term) @ to_fixed @ block
        return np.asarray(spice.sxform(fixed_frame, frame, epoch)) @ jacobian

    def tide_partials(self, epoch, frame="J2000") -> ArrayWUnits:
        """Sensitivity of this station's state to the Love and Shida numbers.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.

        Returns
        -------
        ArrayWUnits
            ``(6, 2)`` Jacobian in km and km/s (:attr:`STATE_UNITS`) per unit
            :math:`h_2` or :math:`l_2`, in ``frame`` axes, or zeros when no
            :attr:`surface_displacement` is attached.

        Notes
        -----
        The tidal response is what these numbers scale, so a station with no
        tide model attached has no sensitivity to them -- estimating them then
        gives a zero column, which a filter will not like. The Jacobian is
        taken about the undisplaced position, which is where the tide model
        itself evaluates. It sums every attached model, since
        :meth:`tide_correction` shifts all of them by the same amount.
        """
        return ArrayWUnits.from_row_units(self._tide_partials(epoch, frame), self.STATE_UNITS)

    def _tide_partials(self, epoch, frame="J2000") -> np.ndarray:
        """Numeric core of :meth:`tide_partials`, in km and km/s."""
        epoch = float(getattr(epoch, "et", epoch))
        models = [model for model in self._surface_displacements
                  if hasattr(model, "state_displacement_partials")]
        if not models:
            return np.zeros((6, 2))
        fixed_frame = self.body_fixed_frame
        frame = getattr(frame, "name", frame)
        undisplaced = self._undisplaced_state(epoch)
        jacobian = sum(self._model_values(model, "state_displacement_partials", self.STATE_UNITS,
                                          undisplaced[:3], epoch, fixed_frame)
                       for model in models)
        return np.asarray(spice.sxform(fixed_frame, frame, epoch)) @ jacobian

    def displacement_state(self, epoch, frame=None) -> ArrayWFrame:
        """Return the total attached surface displacement of this station.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to :attr:`body_fixed_frame`.

        Returns
        -------
        ArrayWFrame
            Sum of every attached :attr:`surface_displacement` and its rate, in
            km and km/s in ``frame``, shape (6,); zeros when none is attached.

        Notes
        -----
        This is the term :meth:`nominal_state` adds to the coordinates or the
        station SPK. Models that read the station from the kernel directly
        rather than through :meth:`nominal_state` add it themselves, so every
        model places the station at the same displaced point.
        """
        return self._framed(self._displacement_state(epoch, frame), self.STATE_UNITS,
                            self.body_fixed_frame if frame is None else frame)

    def _displacement_state(self, epoch, frame=None) -> np.ndarray:
        """Numeric core of :meth:`displacement_state`, in km and km/s."""
        frame = getattr(frame, "name", frame) or self.body_fixed_frame
        epoch = float(getattr(epoch, "et", epoch))
        if not self._surface_displacements:
            return np.zeros(6)
        position = self._undisplaced_state(epoch)[:3]
        state = sum(self._model_values(model, "state_displacement", self.STATE_UNITS,
                                       position, epoch, self.body_fixed_frame)
                    for model in self._surface_displacements)
        if frame == self.body_fixed_frame:
            return state
        return np.asarray(spice.sxform(self.body_fixed_frame, frame, epoch)) @ state

    def tide_correction(self, epoch, frame="J2000", *,
                        love_h2=None, shida_l2=None) -> ArrayWFrame:
        r"""Move the attached tide from its own Love numbers to estimated ones.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        love_h2, shida_l2 : float, optional
            Estimated :math:`h_2` and :math:`l_2`, read as the numbers of
            :attr:`tide_reference`. Defaults to ``None``, meaning each attached
            model keeps its own number.

        Returns
        -------
        ArrayWFrame
            State delta in km and km/s in ``frame``, shape (6,); zeros when no
            tide model with Love-number partials is attached.

        Notes
        -----
        :meth:`nominal_state` already carries each tide model at its own
        :math:`h_2` and :math:`l_2`, so only the difference is added here. Every
        attached model shifts by the same difference, the estimate minus the
        :attr:`tide_reference` number: a solid tide and a pole tide keep their
        different nominal numbers and move together, so :meth:`tide_partials`
        is their sum. The degree-2 displacement is linear in both numbers,
        which makes that sum times the difference exact for the closed-form
        models and first order for the IERS 2010 routine.
        """
        return self._framed(self._tide_correction(
            epoch, frame, love_h2=love_h2, shida_l2=shida_l2), self.STATE_UNITS, frame)

    def _tide_correction(self, epoch, frame="J2000", *,
                         love_h2=None, shida_l2=None) -> np.ndarray:
        """Numeric core of :meth:`tide_correction`, in km and km/s."""
        epoch = float(getattr(epoch, "et", epoch))
        reference = self.tide_reference
        if reference is None or (love_h2 is None and shida_l2 is None):
            return np.zeros(6)
        change = np.array([
            0.0 if love_h2 is None else float(love_h2) - float(reference.love_h2),
            0.0 if shida_l2 is None else float(shida_l2) - float(reference.shida_l2)])
        return self._tide_partials(epoch, frame) @ change

    def pole_tide_partials(self, epoch, frame="J2000") -> ArrayWUnits:
        """Sensitivity of this station's state to the pole the pole tide uses.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.

        Returns
        -------
        ArrayWUnits
            ``(6, 2)`` Jacobian in km and km/s per radian of pole x and y
            (:attr:`ROTATION_PARTIAL_UNITS`), in ``frame`` axes, or zeros when
            no model with ``state_displacement_pole_partials``, such as a
            :class:`~scarabaeus.PoleTide`, is attached.

        Notes
        -----
        An estimated ``eop_delta_pole_x``/``eop_delta_pole_y`` turns the
        station through :meth:`rotation_partials` and also moves the pole the
        pole tide responds to. The tide is linear in the pole, so this Jacobian
        times the pole delta is the exact change, taken about the undisplaced
        position like :meth:`tide_partials`.
        """
        return ArrayWUnits.from_row_units(self._pole_tide_partials(epoch, frame),
                                          self.ROTATION_PARTIAL_UNITS)

    def _pole_tide_partials(self, epoch, frame="J2000") -> np.ndarray:
        """Numeric core of :meth:`pole_tide_partials`, in km and km/s per radian."""
        epoch = float(getattr(epoch, "et", epoch))
        models = [model for model in self._surface_displacements
                  if hasattr(model, "state_displacement_pole_partials")]
        if not models:
            return np.zeros((6, 2))
        fixed_frame = self.body_fixed_frame
        frame = getattr(frame, "name", frame)
        undisplaced = self._undisplaced_state(epoch)
        jacobian = sum(self._model_values(model, "state_displacement_pole_partials",
                                          self.ROTATION_PARTIAL_UNITS,
                                          undisplaced[:3], epoch, fixed_frame)
                       for model in models)
        return np.asarray(spice.sxform(fixed_frame, frame, epoch)) @ jacobian

    def rotation_partials(self, epoch, frame="J2000", *,
                          location=None, location_frame=None,
                          cylindrical: bool = False) -> ArrayWUnits:
        r"""Sensitivity of this station's state to a body-fixed rotation vector.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        location : ArrayWFrame, ArrayWUnits or array_like, shape (3,), optional
            Current estimated location delta, so the derivative is taken about
            the same displaced point :meth:`state_correction` rotates. Bare
            numbers are km. Defaults to ``None``, meaning zero.
        location_frame : Frame or str, optional
            Axes of ``location``. Defaults to the frame of an ArrayWFrame
            ``location``, else :attr:`body_fixed_frame`.
        cylindrical : bool, optional
            Interpret ``location`` as (radius, east, Z). Defaults to
            ``False``.

        Returns
        -------
        ArrayWUnits
            ``(6, 3)`` Jacobian in km and km/s per radian
            (:attr:`ROTATION_PARTIAL_UNITS`), in ``frame`` axes.

        Raises
        ------
        ValueError
            For undefined cylindrical longitude, or an attached orientation
            expressed in a different body-fixed frame.

        Notes
        -----
        A small active rotation moves the station by
        :math:`\varepsilon \times r`, so the derivative is :math:`-[r\times]`
        — evaluated on both the position and the body-fixed velocity, since a
        rotated station also sits on a different radius from the spin axis.
        This is the block every Earth-orientation parameter chains through.
        """
        return ArrayWUnits.from_row_units(self._rotation_partials(
            epoch, frame, location=ArrayWUnits.values_in(location, km),
            location_frame=location_frame or getattr(location, "frame", None),
            cylindrical=cylindrical),
            self.ROTATION_PARTIAL_UNITS)

    def _rotation_partials(self, epoch, frame="J2000", *,
                           location=None, location_frame=None,
                           cylindrical: bool = False) -> np.ndarray:
        """Numeric core of :meth:`rotation_partials`: location in km, km and km/s per radian out."""
        epoch = float(getattr(epoch, "et", epoch))
        fixed_frame = self.body_fixed_frame
        frame = getattr(frame, "name", frame)
        moved = self._nominal_state(epoch, fixed_frame)
        if location is not None:
            moved = moved + self._location_partials(
                epoch, fixed_frame, cylindrical=cylindrical,
                location_frame=location_frame) @ np.asarray(location, dtype=float).reshape(3)
        jacobian = np.vstack([-Utils.skew(moved[:3]), -Utils.skew(moved[3:])])
        return np.asarray(spice.sxform(fixed_frame, frame, epoch)) @ jacobian

    def inertial_rotation_partials(self, epoch, frame="J2000") -> ArrayWUnits:
        r"""Sensitivity of this station's state to a J2000 rotation vector.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.

        Returns
        -------
        ArrayWUnits
            ``(6, 3)`` Jacobian in km and km/s per radian
            (:attr:`ROTATION_PARTIAL_UNITS`), in ``frame`` axes. The
            correction is linear in the rotation, so the Jacobian times the
            rotation vector is also the state correction itself.

        Notes
        -----
        A small rotation :math:`\omega` held fixed in inertial space moves the
        station by :math:`\omega \times r_I` and its inertial velocity by
        :math:`\omega \times v_I`, so the derivative is :math:`-[r_I\times]`
        and :math:`-[v_I\times]`. This differs from :meth:`rotation_partials`,
        whose rotation is held fixed in the body and so turns with it. A
        celestial-pole offset is of this inertial kind.

        The rotation acts on :meth:`nominal_state`, not on the estimated
        location or body-fixed corrections; those cross terms are products of
        two small corrections.
        """
        return ArrayWUnits.from_row_units(self._inertial_rotation_partials(epoch, frame),
                                          self.ROTATION_PARTIAL_UNITS)

    def _inertial_rotation_partials(self, epoch, frame="J2000") -> np.ndarray:
        """Numeric core of :meth:`inertial_rotation_partials`, in km and km/s per radian."""
        epoch = float(getattr(epoch, "et", epoch))
        frame = getattr(frame, "name", frame)
        inertial = np.asarray(spice.sxform(self.body_fixed_frame, "J2000", epoch)) @ \
            self._nominal_state(epoch, self.body_fixed_frame)
        jacobian = np.vstack([-Utils.skew(inertial[:3]), -Utils.skew(inertial[3:])])
        return np.asarray(spice.sxform("J2000", frame, epoch)) @ jacobian

    @staticmethod
    def _iau_axes(angles) -> np.ndarray:
        r"""J2000 axes the IAU pole angles turn the body about.

        Parameters
        ----------
        angles : array_like, shape (3,)
            :math:`(\alpha, \delta, W)`, rad.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3). Columns are :math:`\hat z`, :math:`-\hat n` and
            :math:`\hat p`, with :math:`\hat n = (-\sin\alpha, \cos\alpha, 0)`
            the node and :math:`\hat p` the pole, so that
            :math:`\partial R / \partial a_k = [e_k \times] R` for the
            :meth:`~scarabaeus.BodyOrientation.iau_matrix` :math:`R`, and the
            body's angular velocity is this matrix times the angle rates.
        """
        alpha, delta = float(angles[0]), float(angles[1])
        return np.array([
            [0., np.sin(alpha), np.cos(delta) * np.cos(alpha)],
            [0., -np.cos(alpha), np.cos(delta) * np.sin(alpha)],
            [1., 0., np.sin(delta)]])

    def _iau_pole_motion(self, epoch: float, pole: dict) -> tuple:
        r"""This station's J2000 state under an estimated IAU body orientation.

        Parameters
        ----------
        epoch : float
            TDB seconds past J2000.
        pole : dict
            Estimated ``pole_ra``, ``pole_dec``, ``pole_w0`` (rad) and
            ``pole_pm`` (deg/day) of the central body, the arguments of
            :meth:`~scarabaeus.BodyOrientation.iau_angles`. A missing one keeps
            its secular term.

        Returns
        -------
        moved : numpy.ndarray
            Station state under the estimated orientation, J2000, shape (6,).
        nominal : numpy.ndarray
            :meth:`nominal_state` in J2000, shape (6,).
        angles, rates : numpy.ndarray
            Estimated :math:`(\alpha, \delta, W)`, rad, and their rates,
            rad/s, each shape (3,).
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        centre = getattr(self.central_body, "spice_id", self.central_body)
        coefficients = BodyOrientation.iau_pole_coefficients(centre)
        angles, rates = BodyOrientation.iau_angles(epoch, None, coefficients, **pole)
        secular, secular_rates = BodyOrientation.iau_angles(epoch, None, coefficients)
        nominal = np.asarray(spice.sxform(self.body_fixed_frame, "J2000", epoch)) @ \
            self._nominal_state(epoch, self.body_fixed_frame)
        # P carries the secular orientation onto the estimated one, so it is
        # the identity whenever the estimate reproduces the secular model.
        mapping = BodyOrientation.iau_matrix(angles) @ BodyOrientation.iau_matrix(secular).T
        mapping_rate = (Utils.skew(self._iau_axes(angles) @ rates) @ mapping
                        - mapping @ Utils.skew(self._iau_axes(secular) @ secular_rates))
        moved = np.concatenate([mapping @ nominal[:3],
                                mapping_rate @ nominal[:3] + mapping @ nominal[3:]])
        return moved, nominal, angles, rates

    @staticmethod
    def _pole_values(pole: dict) -> dict:
        """Strip estimated IAU pole values to the floats the numeric cores take.

        Parameters
        ----------
        pole : dict
            ``pole_ra``, ``pole_dec``, ``pole_w0`` and ``pole_pm`` as
            :class:`~scarabaeus.ArrayWUnits` or bare numbers, which are taken to
            be in :attr:`BodyOrientation.IAU_POLE_UNITS` already.

        Returns
        -------
        dict
            The same keys as floats in radians and deg/day. A ``None`` or an
            unknown name is passed through for
            :meth:`~scarabaeus.BodyOrientation.iau_angles` to handle.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        units = BodyOrientation.IAU_POLE_UNITS
        return {name: (value if value is None or name not in units
                       else ArrayWUnits.values_in(value, units[name]).item())
                for name, value in pole.items()}

    def iau_pole_correction(self, epoch, frame="J2000", **pole) -> ArrayWFrame:
        r"""State correction from an estimated IAU pole and prime meridian.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        **pole : ArrayWUnits or float
            Estimated ``pole_ra``, ``pole_dec``, ``pole_w0`` and ``pole_pm`` of
            the central body, as
            :meth:`~scarabaeus.BodyOrientation.iau_angles` takes them. Bare
            numbers are radians and, for ``pole_pm``, deg/day.

        Returns
        -------
        ArrayWFrame
            Six components in km and km/s in ``frame``, to add to
            :meth:`nominal_state`.

        Raises
        ------
        MissingPoleCoefficientsError
            From :meth:`~scarabaeus.BodyOrientation.iau_angles`, raised by
            :meth:`~scarabaeus.BodyOrientation.iau_pole_coefficients` reading
            no secular constants for the central body.

        Notes
        -----
        The estimate is applied as a perturbation of the station's own frame:
        with :math:`R_e` and :math:`R_s` the body-fixed to J2000 rotations of
        :meth:`~scarabaeus.BodyOrientation.iau_angles` with and without the
        estimated values, the station moves to :math:`r' = P r` with
        :math:`P = R_e R_s^T`, and

        .. math::
            v' = \dot P r + P v, \qquad
            \dot P = [\omega_e \times] P - P [\omega_s \times],

        where :math:`r, v` are :meth:`nominal_state` in J2000 and
        :math:`\omega` the body's angular velocity. :math:`P` is exact, not
        linearized, and is the identity whenever the estimate equals the
        secular model; for a body whose PCK frame has no nutation terms
        (Mars in ``pck00010``) the station then turns with exactly the
        orientation that :class:`~scarabaeus.SphericalHarmonicsGravity` and the
        optical models apply for the same state. An estimated ``pole_ra`` or
        ``pole_dec`` is constant and ``pole_pm`` drops the quadratic term, as
        there, so an estimate held at the PCK constant term drifts from the
        secular model by the dropped rate over the years since J2000.

        The perturbation acts on :meth:`nominal_state`, not on the estimated
        location or body-fixed corrections; those cross terms are products of
        two corrections.
        """
        return self._framed(self._iau_pole_correction(epoch, frame, **self._pole_values(pole)),
                            self.STATE_UNITS, frame)

    def _iau_pole_correction(self, epoch, frame="J2000", **pole) -> np.ndarray:
        """Numeric core of :meth:`iau_pole_correction`: radians and deg/day in, km and km/s out."""
        epoch = float(getattr(epoch, "et", epoch))
        frame = getattr(frame, "name", frame)
        moved, nominal, _, _ = self._iau_pole_motion(epoch, pole)
        return np.asarray(spice.sxform("J2000", frame, epoch)) @ (moved - nominal)

    def iau_pole_partials(self, epoch, frame="J2000", **pole) -> ArrayWUnits:
        r"""Sensitivity of this station's state to the estimated IAU pole.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Output axes. Defaults to ``'J2000'``.
        **pole : ArrayWUnits or float
            Values to linearize about, as in :meth:`iau_pole_correction`; a
            missing one is linearized about its secular term.

        Returns
        -------
        ArrayWUnits
            ``(6, 4)`` Jacobian of :meth:`iau_pole_correction` in ``frame``
            axes, columns ``pole_ra``, ``pole_dec``, ``pole_pm``, ``pole_w0``
            (the order of :attr:`BodyOrientation.IAU_POLE_PARAMETERS`), in km
            and km/s per radian (:attr:`ROTATION_PARTIAL_UNITS`) and, for
            ``pole_pm``, per deg/day (:attr:`POLE_PM_PARTIAL_UNITS`).

        Raises
        ------
        MissingPoleCoefficientsError
            As in :meth:`iau_pole_correction`.

        Notes
        -----
        Each angle turns the estimated orientation about one J2000 axis
        :math:`e` of :meth:`_iau_axes`, so with :math:`g` the derivative of the
        angle with respect to the parameter (1, or :math:`d\,\pi/180` for
        ``pole_pm`` with :math:`d` days past J2000),

        .. math::
            \frac{\partial r'}{\partial q} = g\, e \times r', \qquad
            \frac{\partial v'}{\partial q} = \dot g\, e \times r'
            + g\,(\dot e \times r' + e \times v'),

        where :math:`\dot e` follows the estimated orientation:
        :math:`\dot{\hat z} = 0`,
        :math:`-\dot{\hat n} = \dot\alpha_e (\cos\alpha_e, \sin\alpha_e, 0)`
        and :math:`\dot{\hat p} = \omega_e \times \hat p`.
        """
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        result = ArrayWUnits.from_row_units(
            self._iau_pole_partials(epoch, frame, **self._pole_values(pole)),
            self.ROTATION_PARTIAL_UNITS)
        column = BodyOrientation.IAU_POLE_PARAMETERS.index("pole_pm")
        for row, unit in enumerate(self.POLE_PM_PARTIAL_UNITS):
            result.units[row, column] = unit
        return result

    def _iau_pole_partials(self, epoch, frame="J2000", **pole) -> np.ndarray:
        """Numeric core of :meth:`iau_pole_partials`, per radian and per deg/day."""
        from scarabaeus import constants
        from scarabaeus.timeAndFrame.BodyOrientation import BodyOrientation
        epoch = float(getattr(epoch, "et", epoch))
        frame = getattr(frame, "name", frame)
        moved, _, angles, rates = self._iau_pole_motion(epoch, pole)
        position, velocity = moved[:3], moved[3:]
        axes = self._iau_axes(angles)
        alpha = float(angles[0])
        axis_rates = np.column_stack([
            np.zeros(3),
            rates[0] * np.array([np.cos(alpha), np.sin(alpha), 0.]),
            np.cross(axes @ rates, axes[:, 2])])
        days = epoch / constants.day_to_sec
        # (axis, angle per parameter, its rate) in IAU_POLE_PARAMETERS order.
        columns = {
            "pole_ra": (0, 1., 0.), "pole_dec": (1, 1., 0.),
            "pole_pm": (2, np.deg2rad(days), np.deg2rad(1.) / constants.day_to_sec),
            "pole_w0": (2, 1., 0.)}
        jacobian = np.zeros((6, len(BodyOrientation.IAU_POLE_PARAMETERS)))
        for index, name in enumerate(BodyOrientation.IAU_POLE_PARAMETERS):
            axis, gain, gain_rate = columns[name]
            turned = np.cross(axes[:, axis], position)
            jacobian[:3, index] = gain * turned
            jacobian[3:, index] = gain_rate * turned + gain * (
                np.cross(axis_rates[:, axis], position)
                + np.cross(axes[:, axis], velocity))
        return np.asarray(spice.sxform("J2000", frame, epoch)) @ jacobian

    # region Methods
    def eop_displacement(self, epoch, frame="J2000", as_state: bool = False):
        """
        This station's displacement from its Earth orientation correction.

        Parameters
        ----------
        epoch : EpochArray or float
            Single epoch; a float is TDB seconds past J2000.
        frame : Frame or str, optional
            Frame to express the displacement in. Defaults to ``'J2000'``.
        as_state : bool, optional
            Return position and velocity rather than position alone. The
            velocity term matters for Doppler — see
            :meth:`~scarabaeus.BodyOrientation.state_displacement`. Defaults
            to ``False``.

        Returns
        -------
        ArrayWFrame or None
            Displacement in km (and km/s when ``as_state``) in ``frame``,
            shape (3,) or (6,), or ``None`` when no correction is attached.

        Notes
        -----
        The station's body-fixed position comes from its explicit coordinates
        or the loaded station kernels, relative to :attr:`central_body`,
        so this is a first-order rigid rotation of that nominal position — see
        :class:`~scarabaeus.BodyOrientation`.
        """
        if self._earth_orientation is None:
            return None

        epoch = float(getattr(epoch, "et", epoch))
        frame = frame if isinstance(frame, Frame) else Frame(frame)
        nominal = self._nominal_state(epoch, self.body_fixed_frame)
        if as_state:
            delta = self.body_orientation.state_displacement(
                nominal[:3], epoch, frame_from=self.body_fixed_frame,
                frame_to=frame.name, velocity_body_fixed=nominal[3:])
        else:
            delta = self.body_orientation.displacement(
                nominal[:3], epoch, frame_from=self.body_fixed_frame, frame_to=frame.name)
        return ArrayWFrame(delta, frame)

    def station_visibility(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        elevation_mask: float = 10.0,
        return_angles: bool = False,
        frame: str = "J2000",
        origin: str = "EARTH",
    ):
        """
        Determines if a spacecraft is visible to the ground station given an elevation
        mask for a single epoch or a sequence of epochs.

        Positions are retrieved via SPICE using the SPICE names/IDs of this
        ground station and the target spacecraft, then the elevation (and
        optionally azimuth) are computed in the requested reference frame.

        The station elevation angle :math:`\\varepsilon` above the local horizon
        is computed from

        .. math::

            \\varepsilon = 90^\\circ - \\theta_{nadir},
            \\qquad
            \\cos\\theta_{nadir}
            = \\frac{(\\mathbf{r}_{sc} - \\mathbf{r}_{gs})\\cdot\\mathbf{r}_{gs}}
                {\\|\\mathbf{r}_{sc} - \\mathbf{r}_{gs}\\|\\,\\|\\mathbf{r}_{gs}\\|}

        The station is visible when :math:`\\varepsilon \\geq \\varepsilon_{mask}`
        (default 10°).

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft. Its SPICE ID is used to query its state
            via SPICE.

        epoch : EpochArray
            The epoch or sequence of epochs at which to evaluate visibility.

        elevation_mask : float, optional
            Minimum elevation angle in degrees above the horizon for the
            spacecraft to be considered visible.  Defaults to ``10.0``.

        return_angles : bool, optional
            If ``True``, also returns the elevation and azimuth angles.
            Defaults to ``False``.

        frame : str, optional
            Reference frame for position queries.  Defaults to ``'J2000'``.

        origin : str, optional
            Observer body for SPICE state queries.  Defaults to ``'EARTH'``.

        Returns
        -------
        visible : bool or np.ndarray of bool
            ``True`` if the spacecraft is above the elevation mask as seen from
            this ground station.  Returns a scalar for a single epoch or an
            array for multiple epochs.

        elevation : float or np.ndarray of float
            Elevation angle(s) in degrees above the horizon.  Only returned if
            ``return_angles = True``.

        azimuth : float or np.ndarray of float
            Azimuth angle(s) in degrees measured clockwise from North.  Only
            returned if ``return_angles = True``.
        """

        epoch_vals = np.atleast_1d(epoch.times.values)
        scalar_input = epoch_vals.size == 1
        et_arr = epoch_vals.astype(float)

        # Vectorised SPICE queries — one list-comprehension per body avoids
        # Python-loop overhead from SpiceManager wrappers. The station goes
        # through nominal_state so a table-defined station, which has no SPK of
        # its own, is visible on the same footing as a kernel one.
        obs_pos = np.array(
            [self._nominal_state(t, frame)[:3] + self._centre_offset(t, frame, origin)
             for t in et_arr]
        )
        tgt_pos = np.array(
            [
                spice.spkezr(str(target.spice_id), t, frame, "None", origin)[0][:3]
                for t in et_arr
            ]
        )

        delta = tgt_pos - obs_pos  # (N, 3)
        obs_norm = np.linalg.norm(obs_pos, axis=1)  # (N,)
        delta_norm = np.linalg.norm(delta, axis=1)  # (N,)

        cos_theta = np.sum(delta * obs_pos, axis=1) / (obs_norm * delta_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        elevation = np.degrees(np.pi / 2.0 - np.arccos(cos_theta))  # (N,)

        visible_out = elevation >= elevation_mask  # bool array (N,)

        if scalar_input:
            visible_out = bool(visible_out[0])

        if not return_angles:
            return visible_out

        # Azimuth (only when requested)
        obs_unit = obs_pos / obs_norm[:, None]  # (N, 3)
        delta_unit = delta / delta_norm[:, None]  # (N, 3)
        delta_horiz = (
            delta_unit - np.sum(delta_unit * obs_unit, axis=1)[:, None] * obs_unit
        )

        z_axis = np.array([0.0, 0.0, 1.0])
        north = z_axis - np.sum(z_axis * obs_unit, axis=1)[:, None] * obs_unit
        north_norm = np.linalg.norm(north, axis=1)
        degenerate = north_norm < 1e-10
        north[~degenerate] /= north_norm[~degenerate, None]
        north[degenerate] = np.array([1.0, 0.0, 0.0])

        east = np.cross(obs_unit, north)  # (N, 3)
        azimuth = (
            np.degrees(
                np.arctan2(
                    np.sum(delta_horiz * east, axis=1),
                    np.sum(delta_horiz * north, axis=1),
                )
            )
            % 360.0
        )

        if scalar_input:
            return bool(visible_out), float(elevation[0]), float(azimuth[0])
        return visible_out, elevation, azimuth

    def elevation(
        self,
        target: Spacecraft,
        epoch: EpochArray,
        frame: str = "J2000",
        origin: str = "EARTH",
    ) -> float | np.ndarray:
        """
        Computes the elevation angle of a spacecraft as seen from the ground station.

        Parameters
        ----------
        target : Spacecraft
            The target spacecraft object.  Its SPICE ID is used to query its state
            via SPICE.

        epoch : EpochArray
            The epoch or sequence of epochs at which to evaluate elevation.

        frame : str, optional
            Reference frame for position queries.  Defaults to ``'J2000'``.

        origin : str, optional
            Observer body for SPICE state queries.  Defaults to ``'EARTH'``.

        Returns
        -------
        elevation : float or numpy.ndarray
            Elevation angle(s) in degrees above the horizon.  Returns a scalar
            for a single epoch or an array for multiple epochs.
        """
        # condition inputs
        epoch_vals = np.atleast_1d(epoch.times.values)
        scalar_input = epoch_vals.size == 1

        ## compute elevation for each epoch
        elevation_list = []
        for t in epoch_vals:
            # get states of observer and target
            observer_state = SpiceManager.get_state(
                trgt_bdy=self._name,
                epoch_time=t,
                reference_frame=frame,
                obsvr_bdy=origin,
            )
            target_state = SpiceManager.get_state(
                trgt_bdy=str(target.spice_id),
                epoch_time=t,
                reference_frame=frame,
                obsvr_bdy=origin,
            )

            observer_pos = np.asarray(observer_state.values[0:3])
            target_pos = np.asarray(target_state.values[0:3])

            # compute elevation angle at this epoch
            delta = target_pos - observer_pos

            cos_theta = np.dot(delta, observer_pos) / (
                np.linalg.norm(observer_pos) * np.linalg.norm(delta)
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            elevation_list.append(np.degrees(np.pi / 2.0 - theta))

        # return epoch or epochs
        return elevation_list[0] if scalar_input else np.array(elevation_list)
