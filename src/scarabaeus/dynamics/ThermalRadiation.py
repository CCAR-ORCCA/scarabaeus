# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    CelestialBody,
    DynamicModel,
    Frame,
    Units,
    ArrayWUnits,
    SpiceManager,
    Spacecraft,
    DynamicalUtils,
)
from scarabaeus import constants
from scarabaeus.utils.EphemerisCache import build_ephemeris_cache
from scarabaeus.utils.Serialization import awu_to_dict, awu_from_dict, load_json
from scipy.interpolate import CubicSpline
from pathlib import Path
import os
import scarabaeus.utils.NumpyWrapper as np

_SUN_SPICE_ID = 10

# ------------------#
#  Generate Units  #
# ------------------#
kg, m, km, sec, rad, AU = Units.get_units(["kg", "m", "km", "sec", "rad", "AU"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class ThermalRadiation(DynamicModel):
    """ Recoil acceleration from the thermal radiation the spacecraft emits.

        Each surface :math:`i` of area :math:`A_i`, emissivity
        :math:`\\varepsilon_i` and outward normal :math:`\\hat{\\mathbf{n}}_i`
        radiates as a Lambertian emitter at temperature :math:`T_i`. The recoil
        of that emission is

        .. math::

            \\mathbf{a}_i = -\\frac{2}{3}\\,
                \\frac{\\sigma A_i \\varepsilon_i}{m\\,c}\\,T_i^4\\,\\hat{\\mathbf{n}}_i

        where :math:`\\sigma` is the Stefan-Boltzmann constant, :math:`m` the
        spacecraft mass and :math:`c` the speed of light. The factor 2/3 is the
        momentum a Lambertian surface carries away along its normal. The total
        acceleration is

        .. math::

            \\mathbf{a}_\\text{therm} = \\eta_\\text{therm} \\sum_i \\mathbf{a}_i

        with :math:`\\eta_\\text{therm}` an estimable scale factor (default 1),
        read from the state under ``("eta_thermal", spice_id)``.

        **Temperatures.** A surface temperature is read from a table of
        temperature against off-Sun angle :math:`\\theta`, given at a few
        Sun-spacecraft distances :math:`d`:

        .. math::

            T_i = T_{\\text{table}(i)}\\!\\left(\\theta_i,\\, d\\right),\\qquad
            \\cos\\theta_i = \\hat{\\boldsymbol{\\rho}}_i\\cdot\\hat{\\mathbf{u}}_\\odot

        where :math:`\\hat{\\mathbf{u}}_\\odot` is the spacecraft-to-Sun unit
        vector and :math:`\\hat{\\boldsymbol{\\rho}}_i` the table's reference
        direction: the table's ``angle_reference`` when it has one (for example
        the spacecraft axis the thermal analysis measured the off-Sun angle
        from), otherwise the surface's own normal. Each distance row is a cubic
        spline in :math:`\\theta`, with zero slope imposed at 0 and 180 degrees
        when the table reaches them, since a temperature is symmetric about
        those angles. Rows are joined by linear interpolation in :math:`d`.
        Outside the tabulated angles and distances the temperature is held at
        the nearest edge.

        A surface can carry a second, ``shadow_temperature`` table, used when
        :math:`\\hat{\\mathbf{n}}_i\\cdot\\hat{\\mathbf{u}}_\\odot \\le 0`. This is
        how a high-gain antenna built from many plates picks an illuminated or
        a shadowed temperature for each plate.

        **Attitude.** Normals and angle references are fixed in the spacecraft
        frame and reach the inertial frame through the spacecraft-fixed to
        inertial rotation :math:`C` of :class:`~scarabaeus.DynamicalUtils`,
        for the ``"CK"``, ``"inertialFixedAttitude"`` and
        ``"nadirPointingSun"`` values of ``Spacecraft.attitudeMode``. In
        the last one the spacecraft :math:`-X` axis points at the Sun and the
        Y axis follows the heliocentric velocity, so :math:`C` depends on the
        state, the partials carry the rotation terms and the model reports
        ``depends_on_velocity``.

        **Thermal model.** ``thermal_model`` is a dict, or the path of a JSON
        file holding one, with two entries:

        ``"temperature_tables"``
            ``{name: table}``, each table holding

            * ``"angles"``: :class:`~scarabaeus.ArrayWUnits` of off-Sun angles,
              strictly increasing within [0, 180] degrees, at least two;
            * ``"distances"``: :class:`~scarabaeus.ArrayWUnits` of Sun distances,
              strictly increasing, at least one;
            * ``"temperatures"``: array of shape ``(n_distances, n_angles)``
              (a single distance may be given as a 1-D array);
            * ``"temperature_units"``: ``"K"`` or ``"degC"``;
            * ``"angle_reference"`` (optional): spacecraft-fixed direction the
              off-Sun angle is measured from.

        ``"surfaces"``
            list of dicts, each holding

            * ``"name"``: label;
            * ``"area"``: :class:`~scarabaeus.ArrayWUnits` area;
            * ``"emissivity"``: value in [0, 1];
            * ``"normal"``: outward normal in the spacecraft frame;
            * ``"temperature"``: name of its temperature table;
            * ``"shadow_temperature"`` (optional): name of the table used when
              the surface faces away from the Sun.

        Quantities with units may also be given in their serialized form
        (``{"values", "units"}``, see :func:`~scarabaeus.utils.Serialization.awu_to_dict`),
        which is what :meth:`model_to_dict` writes.

        Parameters
        ----------
        origin : CelestialBody
            Propagation origin. Used for the Sun-relative geometry.

        spacecraft : Spacecraft
            The emitting spacecraft. Its mass and attitude are read at
            evaluation time.

        base_frame : Frame
            Inertial frame of the propagated state.

        thermal_model : dict or str or pathlib.Path
            Surfaces and temperature tables, as described above.

        t0, tf : float, optional
            Propagation span in TDB seconds past J2000. When both are given and
            the frame is J2000, the Sun position is read from a Chebyshev
            ephemeris cache instead of SPICE.

        Notes
        -----
        The position partial carries the temperature change with the Sun
        distance and the off-Sun angle, and, for the Sun-pointed attitude, the
        rotation of the surface normals. The scale-factor partial is the
        acceleration with :math:`\\eta_\\text{therm} = 1`.

        Switching a surface between its illuminated and shadowed tables is a
        step in the acceleration; the partials take the table in use.

        Raises
        ------
        InvalidThermalModelError
            If ``thermal_model`` is not a dict or a JSON file of one, or lacks
            its two entries.
        InvalidTemperatureTableError
            If a temperature table is malformed.
        InvalidSurfaceError
            If a surface is malformed.
        UnknownTemperatureTableError
            If a surface names a table that is not defined.
        DynamicalUtils.InvalidAttitudeModeError
            If the spacecraft attitude mode is not one of the three above.
        DynamicalUtils.InvalidFixedAttitudeError
            If an ``inertialFixedAttitude`` spacecraft has no 3x3 attitude.
        DynamicalUtils.MissingVelocityError
            If a ``nadirPointingSun`` spacecraft has no velocity in the
            state.

        See Also
        --------
        scarabaeus.AntennaRadiationPressure : Antenna and LIDAR recoil.
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

    PARAMETER_NAME = "eta_thermal"
    """ State key of the estimable scale factor. """

    # Conversions used in the evaluation, as raw floats.
    _AU_KM = float(ArrayWUnits(1.0, AU).convert_to(km).values)
    _C_MPS = float(constants.c.convert_to(m / sec).values)

    # Cosines are kept this far inside [-1, 1] before arccos. At exactly +-1 the
    # arccos derivative is infinite, which autograd turns into NaN when a
    # partial is being checked; the offset moves the angle by about 1.4e-6 rad.
    _COS_GUARD = 1e-12

    # Recoil of a Lambertian emitter along its normal, as a fraction of P/c.
    _LAMBERT_FACTOR = 2.0 / 3.0

    # Accepted temperature units and the offset that brings each to kelvin.
    _TEMPERATURE_OFFSETS = {"K": 0.0, "degC": float(constants.zero_Celsius.values)}

    # ------------------#
    # region   Errors  #
    # ------------------#
    class InvalidThermalModelError(Exception):
        """The thermal model is not a dict, or lacks its entries.

        ``thermal_model`` must be a dict, or the path of a JSON file holding
        one, with ``"temperature_tables"`` and ``"surfaces"`` entries.
        """

    class InvalidTemperatureTableError(Exception):
        """A temperature table is malformed.

        Angles must be an angle quantity strictly increasing within [0, 180]
        degrees, distances a length quantity strictly increasing and positive,
        temperatures of shape ``(n_distances, n_angles)`` and above absolute
        zero, and the temperature units ``"K"`` or ``"degC"``.
        """

    class InvalidSurfaceError(Exception):
        """A surface is malformed.

        A surface needs a name, a positive area quantity, an emissivity in
        [0, 1], a nonzero spacecraft-fixed normal and a temperature table.
        """

    class UnknownTemperatureTableError(Exception):
        """A surface names a temperature table that is not defined."""

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
        thermal_model,
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

        self._thermal_model = self.model_from_dict(thermal_model)
        tables = self._thermal_model["temperature_tables"]
        surfaces = self._thermal_model["surfaces"]

        # Per-surface constants: k_i = (2/3) sigma eps_i A_i / c, in kg km/s^2/K^4
        sigma = float(constants.Stefan_Boltzmann.convert_to(kg * sec**-3).values)
        self._surface_names = [s["name"] for s in surfaces]
        self._normals_b = np.array([s["normal"] for s in surfaces], dtype=float)
        areas_m2 = np.array(
            [float(s["area"].convert_to(m**2).values) for s in surfaces]
        )
        emissivities = np.array([s["emissivity"] for s in surfaces], dtype=float)
        self._k = self._LAMBERT_FACTOR * sigma * emissivities * areas_m2 / self._C_MPS * 1e-3

        # One group per table in use: the fitted spline plus the surfaces that
        # read it, each with the role it plays (0 always, 1 when lit, 2 when
        # shadowed) and the direction its off-Sun angle is measured from.
        n_s = len(surfaces)
        members = {name: [] for name in tables}
        for i, s in enumerate(surfaces):
            shadow = s.get("shadow_temperature")
            members[s["temperature"]].append((i, 0 if shadow is None else 1))
            if shadow is not None:
                members[shadow].append((i, 2))

        self._groups = []
        for name, table in tables.items():
            if not members[name]:
                continue
            idx = np.array([i for i, _ in members[name]], dtype=int)
            ref = table.get("angle_reference")
            ref_b = (
                np.tile(ref, (len(idx), 1)) if ref is not None else self._normals_b[idx]
            )
            selector = np.zeros((len(idx), n_s))
            selector[np.arange(len(idx)), idx] = 1.0
            self._groups.append(
                {
                    "name": name,
                    "surfaces": idx,
                    "roles": np.array([r for _, r in members[name]], dtype=int),
                    "ref_b": ref_b,
                    "selector": selector,
                    **self._fit_table(table),
                }
            )

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
    def thermal_model(self) -> dict:
        """ The validated thermal model (surfaces and temperature tables). """
        return self._thermal_model

    @property
    def surface_names(self) -> list:
        """ Surface names, in the order of :meth:`surface_temperatures`. """
        return list(self._surface_names)

    # endregion => Properties  #
    # -------------------------#

    # ----------------------------#
    # region Model definition I/O #
    # ----------------------------#
    @classmethod
    def model_from_dict(cls, thermal_model) -> dict:
        """
        Validate a thermal model and return it with live quantities.

        Accepts the live form (quantities as :class:`~scarabaeus.ArrayWUnits`),
        the serialized form written by :meth:`model_to_dict`, or the path of a
        JSON file holding either.

        Parameters
        ----------
        thermal_model : dict or str or pathlib.Path
            The thermal model; see the class docstring for its layout.

        Returns
        -------
        dict
            ``{"temperature_tables": {...}, "surfaces": [...]}`` with angles,
            distances and areas as :class:`~scarabaeus.ArrayWUnits`,
            temperatures as ``(n_distances, n_angles)`` arrays in the declared
            units, and normals and angle references as unit vectors.

        Raises
        ------
        InvalidThermalModelError
            If the model is not a dict or lacks its entries.
        InvalidTemperatureTableError
            If a table is malformed.
        InvalidSurfaceError
            If a surface is malformed.
        UnknownTemperatureTableError
            If a surface names an undefined table.
        """
        if isinstance(thermal_model, (str, os.PathLike)):
            thermal_model = load_json(Path(thermal_model))
        if not isinstance(thermal_model, dict):
            raise cls.InvalidThermalModelError(
                "ThermalRadiation: thermal_model must be a dict or the path of a JSON "
                f"file holding one. Received {type(thermal_model).__name__}."
            )
        missing = [k for k in ("temperature_tables", "surfaces") if k not in thermal_model]
        if missing:
            raise cls.InvalidThermalModelError(
                f"ThermalRadiation: thermal_model is missing {missing}."
            )
        tables_in = thermal_model["temperature_tables"]
        surfaces_in = thermal_model["surfaces"]
        if not isinstance(tables_in, dict) or not tables_in:
            raise cls.InvalidThermalModelError(
                "ThermalRadiation: 'temperature_tables' must be a non-empty dict "
                "of {name: table}."
            )
        if not isinstance(surfaces_in, (list, tuple)) or not surfaces_in:
            raise cls.InvalidThermalModelError(
                "ThermalRadiation: 'surfaces' must be a non-empty list."
            )

        tables = {
            str(name): cls._validate_table(str(name), table)
            for name, table in tables_in.items()
        }
        surfaces = [cls._validate_surface(k, s, tables) for k, s in enumerate(surfaces_in)]
        return {"temperature_tables": tables, "surfaces": surfaces}

    @classmethod
    def model_to_dict(cls, thermal_model) -> dict:
        """
        Serialize a thermal model to plain JSON types.

        Parameters
        ----------
        thermal_model : dict or str or pathlib.Path
            The thermal model in any form :meth:`model_from_dict` accepts.

        Returns
        -------
        dict
            The model with quantities as ``{"values", "units"}`` dicts and
            arrays as lists; :meth:`model_from_dict` reads it back.
        """
        model = cls.model_from_dict(thermal_model)
        tables = {}
        for name, t in model["temperature_tables"].items():
            ref = t["angle_reference"]
            tables[name] = {
                "angles": awu_to_dict(t["angles"]),
                "distances": awu_to_dict(t["distances"]),
                "temperatures": np.asarray(t["temperatures"]).tolist(),
                "temperature_units": t["temperature_units"],
                "angle_reference": None if ref is None else np.asarray(ref).tolist(),
            }
        surfaces = [
            {
                "name": s["name"],
                "area": awu_to_dict(s["area"]),
                "emissivity": s["emissivity"],
                "normal": np.asarray(s["normal"]).tolist(),
                "temperature": s["temperature"],
                "shadow_temperature": s["shadow_temperature"],
            }
            for s in model["surfaces"]
        ]
        return {"temperature_tables": tables, "surfaces": surfaces}

    def to_dict(self) -> dict:
        """
        Serialize this model's thermal model; see :meth:`model_to_dict`.

        Returns
        -------
        dict
            The serialized thermal model.
        """
        return self.model_to_dict(self._thermal_model)

    @staticmethod
    def _as_quantity(value, error, what: str) -> ArrayWUnits:
        """
        Read a quantity given live or serialized.

        Parameters
        ----------
        value : ArrayWUnits or dict
            The quantity, or its ``{"values", "units"}`` form.
        error : type
            Exception class to raise.
        what : str
            Description used in the error message.

        Returns
        -------
        ArrayWUnits
            The quantity.

        Raises
        ------
        error
            If ``value`` is neither form.
        """
        if isinstance(value, ArrayWUnits):
            return value
        if isinstance(value, dict) and "values" in value and "units" in value:
            return awu_from_dict(value)
        raise error(
            f"ThermalRadiation: {what} must be an ArrayWUnits (or its serialized "
            f"form) so its units are known. Received {type(value).__name__}."
        )

    @staticmethod
    def _unit_vector(value, error, what: str) -> np.ndarray:
        """
        Normalize a nonzero 3-vector.

        Parameters
        ----------
        value : array_like
            The vector.
        error : type
            Exception class to raise.
        what : str
            Description used in the error message.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` unit vector.

        Raises
        ------
        error
            If ``value`` is not a finite nonzero 3-vector.
        """
        try:
            vec = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            vec = None
        if vec is None or vec.shape != (3,) or not np.all(np.isfinite(vec)):
            raise error(f"ThermalRadiation: {what} must be a finite 3-vector.")
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise error(f"ThermalRadiation: {what} must be nonzero.")
        return vec / norm

    @classmethod
    def _validate_table(cls, name: str, table) -> dict:
        """
        Validate one temperature table.

        Parameters
        ----------
        name : str
            Table name, for error messages.
        table : dict
            The table; see the class docstring.

        Returns
        -------
        dict
            The table with live quantities and a ``(n_distances, n_angles)``
            temperature array.

        Raises
        ------
        InvalidTemperatureTableError
            If the table is malformed.
        """
        err = cls.InvalidTemperatureTableError
        if not isinstance(table, dict):
            raise err(f"ThermalRadiation: table '{name}' must be a dict.")
        missing = [
            k for k in ("angles", "distances", "temperatures", "temperature_units")
            if k not in table
        ]
        if missing:
            raise err(f"ThermalRadiation: table '{name}' is missing {missing}.")

        angles = cls._as_quantity(table["angles"], err, f"table '{name}' angles")
        distances = cls._as_quantity(table["distances"], err, f"table '{name}' distances")
        try:
            ang = np.atleast_1d(np.asarray(angles.convert_to(rad).values, dtype=float))
        except ValueError:
            raise err(f"ThermalRadiation: table '{name}' angles must be an angle, "
                      f"not {angles.units}.")
        try:
            dist = np.atleast_1d(np.asarray(distances.convert_to(AU).values, dtype=float))
        except ValueError:
            raise err(f"ThermalRadiation: table '{name}' distances must be a length, "
                      f"not {distances.units}.")

        if ang.ndim != 1 or ang.size < 2 or np.any(np.diff(ang) <= 0.0):
            raise err(f"ThermalRadiation: table '{name}' needs at least two strictly "
                      "increasing angles.")
        if ang[0] < -1e-12 or ang[-1] > np.pi + 1e-12:
            raise err(f"ThermalRadiation: table '{name}' angles must lie within "
                      "[0, 180] degrees.")
        if dist.ndim != 1 or dist.size < 1 or np.any(np.diff(dist) <= 0.0) or dist[0] <= 0.0:
            raise err(f"ThermalRadiation: table '{name}' needs at least one positive, "
                      "strictly increasing distance.")

        units = table["temperature_units"]
        if units not in cls._TEMPERATURE_OFFSETS:
            raise err(f"ThermalRadiation: table '{name}' temperature_units must be one "
                      f"of {tuple(cls._TEMPERATURE_OFFSETS)}. Received {units!r}.")
        temps = np.asarray(table["temperatures"], dtype=float)
        if temps.ndim == 1 and dist.size == 1:
            temps = temps.reshape(1, -1)
        if temps.shape != (dist.size, ang.size):
            raise err(f"ThermalRadiation: table '{name}' temperatures have shape "
                      f"{temps.shape}; expected (n_distances, n_angles) = "
                      f"{(dist.size, ang.size)}.")
        if not np.all(np.isfinite(temps)) or np.any(
            temps + cls._TEMPERATURE_OFFSETS[units] <= 0.0
        ):
            raise err(f"ThermalRadiation: table '{name}' temperatures must be finite "
                      "and above absolute zero.")

        ref = table.get("angle_reference")
        if ref is not None:
            ref = cls._unit_vector(ref, err, f"table '{name}' angle_reference")

        return {
            "angles": angles,
            "distances": distances,
            "temperatures": temps,
            "temperature_units": units,
            "angle_reference": ref,
        }

    @classmethod
    def _validate_surface(cls, k: int, surface, tables: dict) -> dict:
        """
        Validate one surface.

        Parameters
        ----------
        k : int
            Surface index, for error messages.
        surface : dict
            The surface; see the class docstring.
        tables : dict
            The validated temperature tables.

        Returns
        -------
        dict
            The surface with a live area and a unit normal.

        Raises
        ------
        InvalidSurfaceError
            If the surface is malformed.
        UnknownTemperatureTableError
            If it names an undefined table.
        """
        err = cls.InvalidSurfaceError
        if not isinstance(surface, dict):
            raise err(f"ThermalRadiation: surface {k} must be a dict.")
        missing = [
            key for key in ("name", "area", "emissivity", "normal", "temperature")
            if key not in surface
        ]
        if missing:
            raise err(f"ThermalRadiation: surface {k} is missing {missing}.")
        name = str(surface["name"])

        area = cls._as_quantity(surface["area"], err, f"surface '{name}' area")
        try:
            area_m2 = np.asarray(area.convert_to(m**2).values, dtype=float)
        except ValueError:
            raise err(f"ThermalRadiation: surface '{name}' area must be an area, "
                      f"not {area.units}.")
        if area_m2.size != 1 or not float(area_m2) > 0.0:
            raise err(f"ThermalRadiation: surface '{name}' area must be one positive value.")

        try:
            emissivity = float(surface["emissivity"])
        except (TypeError, ValueError):
            emissivity = float("nan")
        if not 0.0 <= emissivity <= 1.0:
            raise err(f"ThermalRadiation: surface '{name}' emissivity must lie in [0, 1].")

        normal = cls._unit_vector(surface["normal"], err, f"surface '{name}' normal")

        lit_table = surface["temperature"]
        shadow_table = surface.get("shadow_temperature")
        for table_name in (lit_table, shadow_table):
            if table_name is not None and table_name not in tables:
                raise cls.UnknownTemperatureTableError(
                    f"ThermalRadiation: surface '{name}' names temperature table "
                    f"'{table_name}', which is not defined. Defined: {sorted(tables)}."
                )

        return {
            "name": name,
            "area": area,
            "emissivity": emissivity,
            "normal": normal,
            "temperature": lit_table,
            "shadow_temperature": shadow_table,
        }

    @classmethod
    def _fit_table(cls, table: dict) -> dict:
        """
        Fit the angle splines of one temperature table.

        Each distance row is a cubic spline in the off-Sun angle (radians),
        with zero slope at 0 and :math:`\\pi` when the table reaches them and
        not-a-knot ends otherwise. The spline coefficients are kept so the
        evaluation can be written in the NumPy backend and traced.

        Parameters
        ----------
        table : dict
            A validated table.

        Returns
        -------
        dict
            ``angles`` (rad), ``distances`` (AU) and ``coeffs`` of shape
            ``(4, n_angles - 1, n_distances)``, highest power first, in kelvin.
        """
        ang = np.atleast_1d(np.asarray(table["angles"].convert_to(rad).values, dtype=float))
        dist = np.atleast_1d(np.asarray(table["distances"].convert_to(AU).values, dtype=float))
        temps_k = table["temperatures"] + cls._TEMPERATURE_OFFSETS[table["temperature_units"]]
        zero_slope = (1, np.zeros(dist.size))
        bc_type = (
            zero_slope if abs(ang[0]) < 1e-12 else "not-a-knot",
            zero_slope if abs(ang[-1] - np.pi) < 1e-12 else "not-a-knot",
        )
        spline = CubicSpline(ang, temps_k.T, axis=0, bc_type=bc_type)
        return {"angles": ang, "distances": dist, "coeffs": np.asarray(spline.c)}

    # endregion Model definition I/O #
    # ----------------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    @staticmethod
    def _raw(value):
        """
        Strip autograd boxes from ``value``.

        Table lookups and switches (spline interval, lit or shadowed) are
        chosen from the plain value; the traced value still flows through the
        arithmetic, so the derivative inside the chosen branch survives.

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
        Scale factor ``eta_thermal`` for ``body``, 1 when not in the state.

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
        sun2body=None, derivatives: bool = False,
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
        sun2body : numpy.ndarray or ArrayBox, optional
            Sun->spacecraft position in km, when the caller already has it.
        derivatives : bool, optional
            Also return the nadir derivatives. Defaults to ``False``.

        Returns
        -------
        tuple
            ``(C, D_r, D_v)``: ``C`` is the ``(3, 3)`` spacecraft-fixed to
            inertial DCM; ``D_r`` and ``D_v`` are ``(3, 3, 3)`` arrays with
            ``D[j]`` the Jacobian of axis ``j``, or ``None`` when the attitude
            does not depend on the state or ``derivatives`` is false.

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
        if sun2body is None:
            sun2body = self._sun_to_origin(epoch) + position
        velocity = DynamicalUtils.velocity_from_state(current_state, body)
        sun2body_vel = self._sun_to_origin_velocity(epoch) + velocity
        return DynamicalUtils.nadirPointingSun(sun2body, sun2body_vel, derivatives)

    def _table_eval(self, group: dict, theta, dist) -> tuple:
        """
        Temperatures of one table at the given angles and distance.

        Parameters
        ----------
        group : dict
            A fitted table group.
        theta : numpy.ndarray or ArrayBox
            Off-Sun angles in rad, one per member surface.
        dist : float or ArrayBox
            Sun-spacecraft distance in AU.

        Returns
        -------
        tuple
            ``(T, dT_dtheta, dT_dd)``: temperatures in K, and their derivatives
            with respect to the angle (K/rad) and the distance (K/AU) as plain
            arrays, zero where the table is held at its edge.
        """
        ang, dists, c = group["angles"], group["distances"], group["coeffs"]

        # Angle: clamp to the table, pick the interval from the plain value.
        th_raw = self._raw(theta)
        inside_ang = (th_raw > ang[0]) & (th_raw < ang[-1])
        th = np.clip(theta, ang[0], ang[-1])
        i = np.clip(np.searchsorted(ang, self._raw(th), side="right") - 1, 0, ang.size - 2)
        dx = (th - ang[i])[:, None]
        ci = c[:, i, :]  # (4, n_members, n_distances)
        rows = ((ci[0] * dx + ci[1]) * dx + ci[2]) * dx + ci[3]
        dx_raw = self._raw(dx)
        drows = (3.0 * ci[0] * dx_raw + 2.0 * ci[1]) * dx_raw + ci[2]

        if dists.size == 1:
            dT_dth = np.where(inside_ang, drows[:, 0], 0.0)
            return rows[:, 0], dT_dth, np.zeros(th_raw.shape)

        # Distance: linear between the bracketing rows, held at the edges.
        d_raw = float(self._raw(dist))
        j = int(np.clip(np.searchsorted(dists, d_raw, side="right") - 1, 0, dists.size - 2))
        span = dists[j + 1] - dists[j]
        w = np.clip((dist - dists[j]) / span, 0.0, 1.0)
        T = (1.0 - w) * rows[:, j] + w * rows[:, j + 1]

        w_raw = float(self._raw(w))
        dT_dth = (1.0 - w_raw) * drows[:, j] + w_raw * drows[:, j + 1]
        dT_dth = np.where(inside_ang, dT_dth, 0.0)
        inside_d = dists[0] < d_raw < dists[-1]
        rows_raw = self._raw(rows)
        dT_dd = (rows_raw[:, j + 1] - rows_raw[:, j]) / span if inside_d else np.zeros(th_raw.shape)
        return T, dT_dth, dT_dd

    def _evaluate(
        self, position, current_state: dict, epoch: float, body: Spacecraft,
        partials: bool = False,
    ) -> dict:
        """
        Surface temperatures and the unscaled acceleration, with the pieces
        the partials need.

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
        partials : bool, optional
            Also compute the temperature derivatives. Defaults to ``False``.

        Returns
        -------
        dict
            ``T`` (K per surface), ``acc`` (unscaled acceleration, km/s^2),
            and with ``partials`` the geometry and ``dT_dr``/``dT_dv``
            (K/km and K/(km/s) per surface).
        """
        sun2body = self._sun_to_origin(epoch) + position
        dcm, D_r, D_v = self._attitude(
            position, current_state, epoch, body, sun2body=sun2body, derivatives=partials
        )
        s_norm = np.sqrt(np.dot(sun2body, sun2body))
        x_hat = sun2body / s_norm
        u_sun = -x_hat  # spacecraft -> Sun
        dist_au = s_norm / self._AU_KM

        normals = np.dot(self._normals_b, np.transpose(dcm))  # (n_s, 3) inertial
        lit = (np.dot(self._raw(normals), self._raw(u_sun)) > 0.0).astype(float)

        n_s = self._normals_b.shape[0]
        T = np.zeros(n_s)
        if partials:
            x_r, u_r, s_r = self._raw(x_hat), self._raw(u_sun), float(self._raw(s_norm))
            dT_dr = np.zeros((n_s, 3))
            dT_dv = np.zeros((n_s, 3))

        for g in self._groups:
            roles, idx = g["roles"], g["surfaces"]
            weight = np.where(roles == 1, lit[idx], np.where(roles == 2, 1.0 - lit[idx], 1.0))

            ref = np.dot(g["ref_b"], np.transpose(dcm))  # (n_g, 3) inertial
            cos_t = np.clip(np.dot(ref, u_sun), -1.0 + self._COS_GUARD, 1.0 - self._COS_GUARD)
            theta = np.arccos(cos_t)
            T_g, dT_dth, dT_dd = self._table_eval(g, theta, dist_au)
            # Scatter into the surface array without assigning into it.
            T = T + np.dot(weight * T_g, g["selector"])

            if not partials:
                continue
            # dT/dcos(theta) = -T'(theta)/sin(theta); the guard keeps sin > 0.
            cos_r = self._raw(cos_t)
            dT_dcos = -dT_dth / np.sqrt(1.0 - cos_r**2)
            ref_r = self._raw(ref)
            # d cos/dr = (d rho/dr)^T u - (I - x x^T) rho / |s|
            dcos_dr = -(ref_r - np.outer(np.dot(ref_r, x_r), x_r)) / s_r
            dcos_dv = np.zeros_like(dcos_dr)
            if D_r is not None:
                # d cos/dq += u^T d rho/dq, with rho turning with the attitude
                dcos_dr = dcos_dr + u_r @ DynamicalUtils.body_vector_jacobian(g["ref_b"], D_r)
                dcos_dv = u_r @ DynamicalUtils.body_vector_jacobian(g["ref_b"], D_v)
            dT_dr_g = dT_dcos[:, None] * dcos_dr + np.outer(dT_dd, x_r) / self._AU_KM
            dT_dr = dT_dr + np.dot(g["selector"].T, weight[:, None] * dT_dr_g)
            dT_dv = dT_dv + np.dot(g["selector"].T, weight[:, None] * dT_dcos[:, None] * dcos_dv)

        mass = self._mass_value(epoch)
        acc = -np.dot(self._k * T**4, normals) / mass

        out = {"T": T, "acc": acc}
        if partials:
            out.update(
                {"normals": self._raw(normals), "mass": mass, "D_r": D_r, "D_v": D_v,
                 "dT_dr": dT_dr, "dT_dv": dT_dv}
            )
        return out

    def _jacobian(self, ev: dict, dT: np.ndarray, D) -> np.ndarray:
        """
        Unscaled acceleration Jacobian from temperature and attitude
        derivatives.

        .. math::

            \\frac{\\partial\\mathbf{a}}{\\partial\\mathbf{x}} = -\\frac{1}{m}\\sum_i k_i
                \\left[4T_i^3\\,\\hat{\\mathbf{n}}_i
                \\left(\\frac{\\partial T_i}{\\partial\\mathbf{x}}\\right)^T
                + T_i^4\\,\\frac{\\partial\\hat{\\mathbf{n}}_i}{\\partial\\mathbf{x}}\\right]

        Parameters
        ----------
        ev : dict
            Output of :meth:`_evaluate` with ``partials``.
        dT : numpy.ndarray
            ``(n_s, 3)`` temperature derivatives.
        D : numpy.ndarray or None
            ``(3, 3, 3)`` attitude-axis derivatives, or ``None``.

        Returns
        -------
        numpy.ndarray
            ``(3, 3)`` Jacobian.
        """
        T = ev["T"]
        J = np.einsum("i,ia,ib->ab", 4.0 * self._k * T**3, ev["normals"], dT)
        if D is not None:
            J = J + DynamicalUtils.body_vector_jacobian(
                np.dot(self._k * T**4, self._normals_b), D
            )
        return -J / ev["mass"]

    def surface_temperatures(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Temperature of every surface, in the order of :attr:`surface_names`.

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
            Surface temperatures in K.
        """
        return self._evaluate(position, current_state, epoch, body)["T"]

    def compute_acceleration(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Thermal re-radiation acceleration.

        Parameters
        ----------
        position : numpy.ndarray
            ``(3,)`` spacecraft position relative to the origin, km.
        current_state : dict
            The propagated state, holding ``("eta_thermal", spice_id)`` when the
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
        return eta * self._evaluate(position, current_state, epoch, body)["acc"]

    def _compute_partial_by_position(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to the spacecraft position.

        The temperature derivative is

        .. math::

            \\frac{\\partial T_i}{\\partial\\mathbf{r}} =
                \\frac{\\partial T_i}{\\partial d}\\,\\frac{\\hat{\\mathbf{s}}}{\\text{AU}}
                + \\frac{\\partial T_i}{\\partial\\cos\\theta_i}
                \\left[\\left(\\frac{\\partial\\hat{\\boldsymbol{\\rho}}_i}{\\partial\\mathbf{r}}\\right)^T
                \\hat{\\mathbf{u}}_\\odot
                - \\frac{(I - \\hat{\\mathbf{s}}\\hat{\\mathbf{s}}^T)\\,\\hat{\\boldsymbol{\\rho}}_i}
                        {\\lVert\\mathbf{s}\\rVert}\\right],\\qquad
            \\frac{\\partial T_i}{\\partial\\cos\\theta_i} = -\\frac{T_i'(\\theta_i)}{\\sin\\theta_i}

        with :math:`\\mathbf{s}` the Sun-spacecraft vector; the reference and
        normal derivatives are nonzero only for the Sun-pointed attitude.

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
        eta = self._get_scale(current_state, body)
        ev = self._evaluate(position, current_state, epoch, body, partials=True)
        return eta * self._jacobian(ev, ev["dT_dr"], ev["D_r"])

    def _compute_partial_by_velocity(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to the spacecraft velocity.

        Nonzero only for the Sun-pointed attitude, whose Y axis follows the
        heliocentric velocity and so turns the normals and the angle
        references.

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
        if not self.depends_on_velocity:
            return np.zeros((3, 3))
        eta = self._get_scale(current_state, body)
        ev = self._evaluate(position, current_state, epoch, body, partials=True)
        return eta * self._jacobian(ev, ev["dT_dv"], ev["D_v"])

    def _compute_partial_by_eta(
        self, position: np.ndarray, current_state: dict, epoch: float, body: Spacecraft
    ) -> np.ndarray:
        """
        Partial of the acceleration with respect to ``eta_thermal``: the
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
        return self._evaluate(position, current_state, epoch, body)["acc"]

    # endregion Methods #
    # ----------------#
