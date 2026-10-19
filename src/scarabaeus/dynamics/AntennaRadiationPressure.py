# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    CelestialBody,
    DynamicModel,
    Frame,
    Units,
    ArrayWUnits,
    EpochArray,
    SpiceManager,
    Spacecraft,
    DynamicalUtils,
)
from scarabaeus import constants
from scarabaeus.utils.EphemerisCache import build_ephemeris_cache
from scarabaeus.utils.Serialization import awu_to_dict, awu_from_dict
import scarabaeus.utils.NumpyWrapper as np

_SUN_SPICE_ID = 10

# ------------------#
#  Generate Units  #
# ------------------#
kg, m, km, sec, rad = Units.get_units(["kg", "m", "km", "sec", "rad"])
W = kg * m**2 * sec**-3
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class AntennaRadiationPressure(DynamicModel):
    """ Recoil acceleration from power the spacecraft transmits: radio
        antennas and active instruments such as a LIDAR.

        An emitter of radiated power :math:`P` whose beam is a cone of
        half-power beamwidth :math:`\\alpha_0` around the boresight
        :math:`\\hat{\\mathbf{a}}_z` pushes the spacecraft with

        .. math::

            \\mathbf{a}_k = -\\frac{P_k}{m\\,c}\\,
                \\cos^2\\!\\left(\\frac{\\alpha_{0,k}}{4}\\right)\\hat{\\mathbf{a}}_{z,k}

        where :math:`m` is the spacecraft mass and :math:`c` the speed of
        light. The factor :math:`\\cos^2(\\alpha_0/4)` is the boresight share of
        the momentum of a beam spread uniformly over the cone; it is close to
        one for a high-gain antenna and about 0.91 for a 70 degree low-gain
        antenna. The total acceleration is

        .. math::

            \\mathbf{a}_\\text{ant} = \\eta_\\text{ant}
                \\sum_{k\\,\\text{on}} \\mathbf{a}_k

        with :math:`\\eta_\\text{ant}` an estimable scale factor (default 1),
        read from the state under ``("eta_antenna", spice_id)``.

        **Emitters.** ``emitters`` is a list of dicts, each holding

        * ``"name"``: label;
        * ``"power"``: :class:`~scarabaeus.ArrayWUnits` radiated power;
        * ``"hpbw"``: :class:`~scarabaeus.ArrayWUnits` half-power beamwidth,
          within [0, 360) degrees;
        * ``"boresight"``: beam direction in the spacecraft frame;
        * ``"on_intervals"`` (optional): list of ``(start, end)`` pairs, in TDB
          seconds past J2000 or as :class:`~scarabaeus.EpochArray`, when the
          emitter transmits (``start <= t < end``). Omitted or ``None`` means
          always on; an empty list means never on.

        Quantities with units may also be given in their serialized form
        (``{"values", "units"}``), which is what :meth:`emitters_to_dict`
        writes.

        **Attitude.** Boresights are fixed in the spacecraft frame and reach
        the inertial frame through the spacecraft-fixed to inertial rotation
        :math:`C` of :class:`~scarabaeus.DynamicalUtils`, for the ``"CK"``,
        ``"inertialFixedAttitude"`` and ``"nadirPointingSun"`` values of
        ``Spacecraft.attitudeMode``. Only in the last one, where the spacecraft
        :math:`-X` axis points at the Sun and the Y axis follows the
        heliocentric velocity, does the acceleration depend on the state,
        through the rotation of the boresights, and the model reports
        ``depends_on_velocity``.

        Parameters
        ----------
        origin : CelestialBody
            Propagation origin. Used for the Sun-relative geometry of the
            ``nadirPointingSun`` attitude.

        spacecraft : Spacecraft
            The emitting spacecraft. Its mass and attitude are read at
            evaluation time.

        base_frame : Frame
            Inertial frame of the propagated state.

        emitters : list of dict
            The emitters, as described above.

        t0, tf : float, optional
            Propagation span in TDB seconds past J2000. When both are given and
            the frame is J2000, the Sun position is read from a Chebyshev
            ephemeris cache instead of SPICE.

        Notes
        -----
        Switching an emitter on or off is a step in the acceleration; the
        partials take the emitters on at the evaluation epoch. The
        scale-factor partial is the acceleration with
        :math:`\\eta_\\text{ant} = 1`.

        Raises
        ------
        InvalidEmitterError
            If an emitter is malformed.
        DynamicalUtils.InvalidAttitudeModeError
            If the spacecraft attitude mode is not one of the three above.
        DynamicalUtils.InvalidFixedAttitudeError
            If an ``inertialFixedAttitude`` spacecraft has no 3x3 attitude.
        DynamicalUtils.MissingVelocityError
            If a ``nadirPointingSun`` spacecraft has no velocity in the
            state.

        See Also
        --------
        scarabaeus.ThermalRadiation : Thermal re-radiation recoil.
        scarabaeus.nPlateSRP : Solar radiation pressure on a plate model.
        scarabaeus.DynamicalUtils : Spacecraft attitude modes.

        References
        ----------
        J. L. Geeraert, J. M. Leonard, P. W. Kenneally, P. G. Antreasian,
        M. C. Moreau, and D. S. Lauretta, "OSIRIS-REx Navigation Small Force
        Models," in *AAS/AIAA Astrodynamics Specialist Conference, 2019*,
        Advances in the Astronautical Sciences, Vol. 171, Univelt, 2020,
        pp. 931-946.
    """

    PARAMETER_NAME = "eta_antenna"
    """ State key of the estimable scale factor. """

    # Speed of light as a raw float, m/s.
    _C_MPS = float(constants.c.convert_to(m / sec).values)

    # ------------------#
    # region   Errors  #
    # ------------------#
    class InvalidEmitterError(Exception):
        """An emitter is malformed.

        An emitter needs a name, a non-negative power quantity, a half-power
        beamwidth angle within [0, 360) degrees, a nonzero spacecraft-fixed
        boresight, and ``on_intervals`` that are ``None`` or a list of
        ``(start, end)`` pairs with ``start < end``.
        """

    # endregion Errors #
    # ------------------#

    # --------------------#
    # region Constructor #
    # --------------------#
    def __init__(
        self,
        origin: CelestialBody,
        spacecraft: Spacecraft,
        base_frame: Frame,
        emitters: list,
        t0: float = None,
        tf: float = None,
    ):
        # Origin and reference frame
        self._origin = origin
        self._origin_name = self._origin.name
        self._frame = base_frame
        self._spacecraft = spacecraft
        self._origin_spice_name = getattr(constants, self._origin_name.upper()).spice_name

        # Sun-position source, as in nPlateSRP: zero for a heliocentric
        # origin, a Chebyshev cache over the propagation span when the frame
        # is J2000, SPICE otherwise.
        self._origin_spice_id = getattr(self._origin, "spice_id", None)
        self._origin_is_sun = self._origin_spice_id == _SUN_SPICE_ID
        self._sun_eph = None
        frame_name = getattr(base_frame, "name", str(base_frame))
        if (
            t0 is not None
            and tf is not None
            and self._origin_spice_id is not None
            and not self._origin_is_sun
            and frame_name.upper() == "J2000"
        ):
            try:
                self._sun_eph = build_ephemeris_cache(
                    [self._origin_spice_id, _SUN_SPICE_ID],
                    t0 - 86400.0,
                    tf + 86400.0,
                )
            except Exception:
                self._sun_eph = None  # no/partial SPK coverage -> SPICE fallback

        self._emitters = self.emitters_from_dict(emitters)

        # Recoil force of each emitter along its boresight, P cos^2(hpbw/4) / c,
        # in kg km/s^2.
        self._forces = np.array(
            [
                float(e["power"].convert_to(W).values)
                * np.cos(float(e["hpbw"].convert_to(rad).values) / 4.0) ** 2
                / self._C_MPS
                * 1e-3
                for e in self._emitters
            ]
        )
        self._boresights_b = np.array([e["boresight"] for e in self._emitters], dtype=float)

    # endregion Constructor #
    # --------------------#

    # -------------------------#
    # region ====> Properties #
    # -------------------------#
    @property
    def spacecraft(self) -> Spacecraft:
        """ The emitting spacecraft. """
        return self._spacecraft

    @property
    def origin(self) -> CelestialBody:
        """ The propagation origin. """
        return self._origin

    @property
    def origin_name(self) -> str:
        """ The name of the propagation origin. """
        return self._origin_name

    @property
    def frame(self) -> Frame:
        """ The inertial frame of the propagated state. """
        return self._frame

    @property
    def param_name(self) -> str:
        """ State key of the estimable scale factor. """
        return self.PARAMETER_NAME

    @property
    def depends_on_velocity(self) -> bool:
        """ Whether the acceleration depends on the spacecraft velocity.

            True only for the ``nadirPointingSun`` attitude, where the
            spacecraft Y axis follows the heliocentric velocity.
        """
        return DynamicalUtils.depends_on_velocity(self._spacecraft)

    @property
    def emitters(self) -> list:
        """ The validated emitters. """
        return self._emitters

    @property
    def emitter_names(self) -> list:
        """ Emitter names, in the order of :meth:`active_emitters`. """
        return [e["name"] for e in self._emitters]

    # endregion => Properties  #
    # -------------------------#

    # ---------------------------#
    # region Emitter definition #
    # ---------------------------#
    @classmethod
    def emitters_from_dict(cls, emitters) -> list:
        """
        Validate emitters and return them with live quantities.

        Parameters
        ----------
        emitters : list of dict
            The emitters, live or as written by :meth:`emitters_to_dict`; see
            the class docstring.

        Returns
        -------
        list of dict
            Each emitter with ``power`` and ``hpbw`` as
            :class:`~scarabaeus.ArrayWUnits`, a unit ``boresight``, and
            ``on_intervals`` as ``None`` or a list of ``(start, end)`` floats in
            TDB seconds past J2000.

        Raises
        ------
        InvalidEmitterError
            If ``emitters`` is not a non-empty list or an emitter is malformed.
        """
        err = cls.InvalidEmitterError
        if not isinstance(emitters, (list, tuple)) or not emitters:
            raise err(
                "AntennaRadiationPressure: emitters must be a non-empty list of dicts. "
                f"Received {type(emitters).__name__}."
            )
        return [cls._validate_emitter(k, e) for k, e in enumerate(emitters)]

    @classmethod
    def emitters_to_dict(cls, emitters) -> list:
        """
        Serialize emitters to plain JSON types.

        Parameters
        ----------
        emitters : list of dict
            The emitters in any form :meth:`emitters_from_dict` accepts.

        Returns
        -------
        list of dict
            The emitters with quantities as ``{"values", "units"}`` dicts;
            :meth:`emitters_from_dict` reads them back.
        """
        return [
            {
                "name": e["name"],
                "power": awu_to_dict(e["power"]),
                "hpbw": awu_to_dict(e["hpbw"]),
                "boresight": np.asarray(e["boresight"]).tolist(),
                "on_intervals": (
                    None if e["on_intervals"] is None
                    else [[start, end] for start, end in e["on_intervals"]]
                ),
            }
            for e in cls.emitters_from_dict(emitters)
        ]

    def to_dict(self) -> list:
        """
        Serialize this model's emitters; see :meth:`emitters_to_dict`.

        Returns
        -------
        list of dict
            The serialized emitters.
        """
        return self.emitters_to_dict(self._emitters)

    @staticmethod
    def _epoch_seconds(value) -> float:
        """
        An epoch as TDB seconds past J2000.

        Parameters
        ----------
        value : float or EpochArray
            The epoch; a one-element :class:`~scarabaeus.EpochArray` or a number.

        Returns
        -------
        float
            TDB seconds past J2000.
        """
        if isinstance(value, EpochArray):
            value = np.asarray(value.et, dtype=float).reshape(-1)
            if value.size != 1:
                raise ValueError("one epoch expected")
            return float(value[0])
        return float(value)

    @classmethod
    def _validate_emitter(cls, k: int, emitter) -> dict:
        """
        Validate one emitter.

        Parameters
        ----------
        k : int
            Emitter index, for error messages.
        emitter : dict
            The emitter; see the class docstring.

        Returns
        -------
        dict
            The emitter with live quantities, a unit boresight and float
            ``on_intervals``.

        Raises
        ------
        InvalidEmitterError
            If the emitter is malformed.
        """
        err = cls.InvalidEmitterError
        if not isinstance(emitter, dict):
            raise err(f"AntennaRadiationPressure: emitter {k} must be a dict.")
        missing = [key for key in ("name", "power", "hpbw", "boresight") if key not in emitter]
        if missing:
            raise err(f"AntennaRadiationPressure: emitter {k} is missing {missing}.")
        name = str(emitter["name"])

        quantities = {}
        for key, unit, what in (("power", W, "a power"), ("hpbw", rad, "an angle")):
            value = emitter[key]
            if isinstance(value, dict) and "values" in value and "units" in value:
                value = awu_from_dict(value)
            if not isinstance(value, ArrayWUnits):
                raise err(
                    f"AntennaRadiationPressure: emitter '{name}' {key} must be an "
                    f"ArrayWUnits (or its serialized form) so its units are known. "
                    f"Received {type(value).__name__}."
                )
            try:
                raw = np.asarray(value.convert_to(unit).values, dtype=float)
            except ValueError:
                raise err(f"AntennaRadiationPressure: emitter '{name}' {key} must be "
                          f"{what}, not {value.units}.")
            if raw.size != 1 or not np.isfinite(raw).all():
                raise err(f"AntennaRadiationPressure: emitter '{name}' {key} must be "
                          "one finite value.")
            quantities[key] = (value, float(raw.reshape(-1)[0]))

        if quantities["power"][1] < 0.0:
            raise err(f"AntennaRadiationPressure: emitter '{name}' power must be "
                      "non-negative.")
        if not 0.0 <= quantities["hpbw"][1] < 2.0 * np.pi:
            raise err(f"AntennaRadiationPressure: emitter '{name}' hpbw must lie in "
                      "[0, 360) degrees.")

        try:
            boresight = np.asarray(emitter["boresight"], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            boresight = None
        if boresight is None or boresight.shape != (3,) or not np.isfinite(boresight).all():
            raise err(f"AntennaRadiationPressure: emitter '{name}' boresight must be a "
                      "finite 3-vector.")
        norm = np.linalg.norm(boresight)
        if norm == 0.0:
            raise err(f"AntennaRadiationPressure: emitter '{name}' boresight must be "
                      "nonzero.")

        intervals = emitter.get("on_intervals")
        if intervals is not None:
            try:
                intervals = [
                    (cls._epoch_seconds(start), cls._epoch_seconds(end))
                    for start, end in intervals
                ]
            except (TypeError, ValueError):
                raise err(f"AntennaRadiationPressure: emitter '{name}' on_intervals "
                          "must be a list of (start, end) epochs.")
            if any(not start < end for start, end in intervals):
                raise err(f"AntennaRadiationPressure: emitter '{name}' on_intervals "
                          "must have start < end.")

        return {
            "name": name,
            "power": quantities["power"][0],
            "hpbw": quantities["hpbw"][0],
            "boresight": boresight / norm,
            "on_intervals": intervals,
        }

    # endregion Emitter definition #
    # ---------------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    @staticmethod
    def _raw(value):
        """
        Strip autograd boxes from ``value``.

        Parameters
        ----------
        value : float or numpy.ndarray or ArrayBox
            Possibly traced value.

        Returns
        -------
        float or numpy.ndarray
            The underlying plain value.
        """
        while isinstance(value, np.autograd_types):
            value = value._value
        return value

    def _get_scale(self, current_state: dict, body: Spacecraft):
        """
        Scale factor ``eta_antenna`` for ``body``, 1 when not in the state.

        Parameters
        ----------
        current_state : dict
            The propagated state.
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        float or ArrayBox
            The scale factor, or the traced box while autograd is tracing it.
        """
        key = (self.PARAMETER_NAME, getattr(body, "spice_id", None))
        if key in current_state:
            return self._scalar_from_state(current_state[key])
        return 1.0

    def _mass_value(self, epoch: float) -> float:
        """
        Spacecraft mass at ``epoch`` in kg, memoized per epoch.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        float
            Spacecraft mass in kg.
        """
        if self.__dict__.get("_mass_epoch") != epoch:
            self._mass_cache = self._spacecraft.mass(epoch).values
            self._mass_epoch = epoch
        return self._mass_cache

    def _sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Sun->origin position (km, raw numpy), memoized per epoch.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin position in km.
        """
        if self.__dict__.get("_s2o_epoch") != epoch:
            self._s2o_cache = self._lookup_sun_to_origin(epoch)
            self._s2o_epoch = epoch
        return self._s2o_cache

    def _lookup_sun_to_origin(self, epoch: float) -> np.ndarray:
        """
        Uncached Sun->origin lookup: identically zero for a heliocentric
        origin; Chebyshev ephemeris cache when available (falling back to
        SPICE for epochs outside the cached span); SPICE otherwise.
        """
        if self._origin_is_sun:
            return np.zeros(3)
        if self._sun_eph is not None:
            try:
                return np.asarray(
                    self._sun_eph.get_relative_position(
                        self._origin_spice_id, _SUN_SPICE_ID, epoch
                    ),
                    dtype=float,
                )
            except Exception:
                pass  # epoch outside the cached span -> SPICE fallback
        return SpiceManager.get_pos_values(
            self._origin_spice_name, epoch, self._frame, "SUN"
        )

    def _sun_to_origin_velocity(self, epoch: float) -> np.ndarray:
        """
        Sun->origin velocity (km/s, raw numpy), memoized per epoch. Only the
        ``nadirPointingSun`` attitude needs it.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Sun->origin velocity in km/s.
        """
        if self.__dict__.get("_s2o_vel_epoch") != epoch:
            if self._origin_is_sun:
                self._s2o_vel_cache = np.zeros(3)
            else:
                self._s2o_vel_cache = DynamicalUtils.sun_to_origin_velocity(
                    self._origin_spice_name, self._frame, epoch
                )
            self._s2o_vel_epoch = epoch
        return self._s2o_vel_cache

    def _attitude_dcm_ck(self, body: Spacecraft, epoch: float) -> np.ndarray:
        """
        Spacecraft-fixed to inertial DCM from the C-kernel, memoized per epoch.

        Parameters
        ----------
        body : Spacecraft
            Spacecraft whose C-kernel attitude is queried.
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` spacecraft-fixed to inertial DCM.
        """
        if self.__dict__.get("_att_dcm_epoch") != epoch:
            self._att_dcm_cache = DynamicalUtils.ck_dcm(body, epoch, self._frame)
            self._att_dcm_epoch = epoch
        return self._att_dcm_cache

    def _attitude(
        self, position, current_state: dict, epoch: float, body: Spacecraft,
        derivatives: bool = False,
    ) -> tuple:
        """
        Spacecraft-fixed to inertial rotation and, for the nadir attitude,
        its derivatives, from :class:`~scarabaeus.DynamicalUtils`.

        The derivatives are returned per axis, ``D[j]`` being the Jacobian of
        axis :math:`j` (see
        :meth:`~scarabaeus.DynamicalUtils.nadirPointingSun`).

        Parameters
        ----------
        position : numpy.ndarray or ArrayBox
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.
        derivatives : bool, optional
            Also return the nadir derivatives. Defaults to ``False``.

        Returns
        -------
        tuple
            ``(C, D_r, D_v)``: ``C`` is the ``(3, 3)`` spacecraft-fixed to
            inertial DCM; ``D_r`` and ``D_v`` are ``(3, 3, 3)`` arrays, or
            ``None`` when the attitude does not depend on the state or
            ``derivatives`` is false.

        Raises
        ------
        DynamicalUtils.InvalidAttitudeModeError
            If the attitude mode is not supported.
        DynamicalUtils.InvalidFixedAttitudeError
            If ``fixedInertialAttitude`` is not a 3x3 array.
        DynamicalUtils.MissingVelocityError
            If the nadir attitude finds no velocity in the state.
        """
        attitude_mode = DynamicalUtils.attitude_mode(body)

        if attitude_mode == "CK":
            return self._attitude_dcm_ck(body, epoch), None, None

        if attitude_mode == "inertialFixedAttitude":
            return DynamicalUtils.fixed_dcm(body), None, None

        # Nadir: -X at the Sun, Y along the heliocentric velocity.
        sun2body = self._sun_to_origin(epoch) + position
        velocity = DynamicalUtils.velocity_from_state(current_state, body)
        sun2body_vel = self._sun_to_origin_velocity(epoch) + velocity
        return DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, derivatives)

    def active_emitters(self, epoch: float) -> np.ndarray:
        """
        Which emitters transmit at ``epoch``.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            Boolean flag per emitter, in the order of :attr:`emitter_names`.
        """
        t = float(self._raw(epoch))
        return np.array(
            [
                e["on_intervals"] is None
                or any(start <= t < end for start, end in e["on_intervals"])
                for e in self._emitters
            ],
            dtype=bool,
        )

    def _body_force(self, epoch: float) -> np.ndarray:
        """
        Summed recoil force of the emitters on at ``epoch``, in the spacecraft
        frame, before the minus sign: :math:`\\sum_{k\\,\\text{on}} f_k\\,\\hat{\\mathbf{a}}_{z,k}`.

        Parameters
        ----------
        epoch : float
            TDB epoch (seconds past J2000).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` force in kg km/s^2.
        """
        on = self.active_emitters(epoch).astype(float)
        return np.dot(on * self._forces, self._boresights_b)

    def _unscaled_acceleration(
        self, position, current_state: dict, epoch: float, body: Spacecraft
    ):
        """
        Acceleration with the scale factor set to 1.

        Parameters
        ----------
        position : numpy.ndarray or ArrayBox
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        numpy.ndarray or ArrayBox
            ``(3,)`` acceleration in km/s^2.
        """
        force_b = self._body_force(epoch)
        if not force_b.any():
            return np.zeros(3)
        dcm, _, _ = self._attitude(position, current_state, epoch, body)
        return -np.dot(dcm, force_b) / self._mass_value(epoch)

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Antenna and instrument recoil acceleration.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state, holding ``("eta_antenna", spice_id)`` when the
            scale factor is estimated.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` acceleration in km/s^2.
        """
        eta = self._get_scale(current_state, body)
        return eta * self._unscaled_acceleration(position, current_state, epoch, body)

    def _state_partial(
        self, position, current_state: dict, epoch: float, body: Spacecraft, which: int
    ) -> np.ndarray:
        """
        Position (``which=1``) or velocity (``which=2``) partial,
        :math:`-\\frac{\\eta}{m}\\sum_j F_{b,j}\\,D[j]`.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.
        which : int
            1 for the position partial, 2 for the velocity partial.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` partial.
        """
        if not self.depends_on_velocity:
            return np.zeros((3, 3))
        force_b = self._body_force(epoch)
        if not force_b.any():
            return np.zeros((3, 3))
        derivs = self._attitude(position, current_state, epoch, body, derivatives=True)
        eta = self._get_scale(current_state, body)
        jac = DynamicalUtils.body_vector_jacobian(force_b, derivs[which])
        return -eta * jac / self._mass_value(epoch)

    def _compute_partial_by_position(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to the spacecraft position.

        Nonzero only for the ``nadirPointingSun`` attitude, where the
        boresights turn with the Sun direction.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` partial in 1/s^2.
        """
        return self._state_partial(position, current_state, epoch, body, 1)

    def _compute_partial_by_velocity(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to the spacecraft velocity.

        Nonzero only for the ``nadirPointingSun`` attitude, whose Y axis
        follows the heliocentric velocity.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` partial in 1/s.
        """
        return self._state_partial(position, current_state, epoch, body, 2)

    def _compute_partial_by_eta(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to ``eta_antenna``: the
        acceleration with the scale factor set to 1.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state.
        epoch : float
            TDB epoch (seconds past J2000).
        body : Spacecraft
            The emitting spacecraft.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` partial in km/s^2.
        """
        return self._unscaled_acceleration(position, current_state, epoch, body)

    # endregion Methods #
    # ----------------#
