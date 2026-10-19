# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, EpochArray, SpiceManager

import copy
import json, os

import numpy as np
import spiceypy as spice

# ------------------#
#  Generate Units  #
# ------------------#
km = Units.get_units("km")


# --------------------#
#  Class Definition  #
# --------------------#
class nPlateModel:
    """ Defines a flat plate model used to represent the spacecraft structure for calculating solar
    radiation pressure acting on it.

    The SRP force on a single plate of area :math:`A` with reflectivity
    coefficient :math:`C_r`, illuminated at angle of incidence
    :math:`\\theta` from the sun, is

    .. math::

        \\mathbf{F}_{SRP} = \\frac{P_{sun}\\,A\\cos\\theta}{c}
                           \\bigl[(1 - C_r)\\,\\hat{\\ell}
                                 + 2C_r\\cos\\theta\\,\\hat{n}\\bigr]

    where :math:`P_{sun}` is the solar radiation pressure at 1 AU,
    :math:`c` is the speed of light, :math:`\\hat{\\ell}` is the unit
    sun-direction vector, and :math:`\\hat{n}` is the outward plate normal.
    The total SRP force on the spacecraft is the vector sum over all
    illuminated plates. A double-sided plate lit from behind is illuminated
    on its back face, so it acts as a plate with the opposite normal and the
    same coefficients; a single-sided plate lit from behind contributes
    nothing.

    Parameters
    ----------
    plates_config_path : str or os.PathLike, optional
        Path to a JSON file containing all necessary information to configure the N-plate model. See
        Notes section for more information on configuration file formatting.
    plates_data : dict, optional
        The same configuration already in memory, with the keys the file would
        have held. Use this when the plate layout is assembled in a script,
        pulled from a database, or varied across a study, rather than living in
        a file on disk. Supply this or ``plates_config_path``, not both.

        The dictionary is copied, so the model does not alias the caller's
        lists and later edits to it cannot silently change the spacecraft.

    Raises
    ------
    ConfigurationError
        If neither or both sources are supplied, the configuration's columns
        disagree on how many plates there are, or a ``frames`` or
        ``double sided`` entry is invalid.

    See Also
    --------
    scarabaeus.Spacecraft : Assign an nPlateModel to a craft.
    scarabaeus.nPlateSRP : Calculate solar radiation pressure.

    Notes
    -----
    The N-plate model requires a JSON file with the following properties:

    * ``names``: an array of strings defining the name of each plate in the configuration.
    * ``areas``: an array of numbers defining the area, in square kilometers, of each plate in the
      configuration.
    * ``ref coeffs``: an array of numbers defining the reflectivity coefficient of each plate in the
      configuration.
    * ``abs coeffs``: an array of numbers defining the absorption coefficient of each plate in the
      configuration.
    * ``dref coeffs``: an array of numbers defining the diffuse reflection coefficient of each plate
      in the configuration.
    * ``sref coeffs``: an array of numbers defining the specular reflection coefficient of each plate
      in the configuration.
    * ``normal vectors``: an array of numerical 3-element arrays defining the components of the normal
      vector (u vector) for each plate in the configuration. For CK frames, the array defines the
      normal vector to be rotated by the quaternion derived from its respective C-kernel.
    * ``types``: an array of strings where

      * ``"Fixed"`` defines a plate whose normal is constant in the frame given by ``frames``,
        by default the spacecraft body frame. That frame is oriented by the Spacecraft
        ``attitudeMode``; it is not an inertial frame.
      * ``"CK"`` defines a time-variant plate with associated reference frame defined by a
        SPICE C-matrix kernel.

    * ``ids``: an array of integers and/or nulls, where

      * a null defines a Fixed plate — it has no NAIF integer code.
      * an integer defines the NAIF ID of a CK plate.

    * ``frames`` (optional): an array of strings, integers and/or nulls giving the frame a Fixed
      plate's normal is constant in, where

      * a null is the spacecraft body frame (the default for every plate).
      * a string or integer is a SPICE frame name or frame ID; the normal is rotated out of
        that frame at each epoch, whatever the Spacecraft ``attitudeMode``. Use this when a
        CK (through its frame kernel) or any other SPICE frame defines the bus orientation.
        CK plates must leave it null.

    * ``double sided`` (optional): an array of booleans, true for a plate that is also
      illuminated from behind (e.g. a solar array). Defaults to false for every plate, the
      right choice for bus faces, which would otherwise double their area seen from the Sun.

    .. note:: The ``abs``, ``dref``, and ``sref`` must sum to unity for each plate.

    .. note:: The two faces of a double-sided plate share its coefficients. A back face with
       different optical properties is a second, single-sided plate with the opposite normal.

    .. dropdown:: Following this format, an example plate configuration file looks like:
        :open:

        .. code-block:: JSON

            {
                "names"          : ["plusX"  , "minusX"  , "SA1"    ],
                "areas"          : [10       , 10        , 10       ],
                "ref coeffs"     : [1.5      , 1.5       , 1.1      ],
                "abs coeffs"     : [0.33     , 0.33      , 0.33     ],
                "dref coeffs"    : [0.33     , 0.33      , 0.33     ],
                "sref coeffs"    : [0.34     , 0.34      , 0.34     ],
                "normal vectors" : [[1, 0, 0], [-1, 0, 0], [1, 0, 0]],
                "types"          : ["Fixed"  , "Fixed"   , "CK"     ],
                "ids"            : [null     , null      , -64001   ],
                "frames"         : [null     , null      , null     ],
                "double sided"   : [false    , false     , true     ]
            }
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class ConfigurationError(ValueError):
        """Raised when a plate configuration cannot define an N-plate model."""

    class PlateOrientationError(Exception):
        """Raised when SPICE cannot orient a CK plate or a plate defined in a named frame."""

    # endregion Errors #
    # ------------------#

    # -------------#
    # Constructor #
    # -------------#
    def __init__(self, plates_config_path=None, plates_data: dict = None):
        # A configuration is either on disk or already in hand; taking both
        # would leave which one won up to the reader.
        if (plates_config_path is None) == (plates_data is None):
            raise self.ConfigurationError(
                "specify exactly one of 'plates_config_path' (a JSON file) or "
                "'plates_data' (the same configuration as a dict); received "
                f"plates_config_path={plates_config_path!r}, "
                f"plates_data={type(plates_data).__name__ if plates_data is not None else None}."
            )

        if plates_config_path is not None:
            source = os.fspath(plates_config_path)
            with open(source, "r") as file:
                config = json.load(file)
        else:
            if not isinstance(plates_data, dict):
                raise self.ConfigurationError(
                    "'plates_data' must be a dict holding the same keys the "
                    f"configuration file would; received {type(plates_data).__name__}."
                )
            source = "the supplied plates_data"
            # Copied so the model does not alias the caller's lists: a
            # configuration edited afterwards must not change this spacecraft.
            config = copy.deepcopy(plates_data)

        # validate data in config
        if "names" not in config:
            raise self.ConfigurationError(
                f'Configuration from {source} has no "names" property, so the '
                "number of plates it describes is undefined. Expected keys: "
                '"names", "areas", "ref coeffs", "abs coeffs", "dref coeffs", '
                '"sref coeffs", "normal vectors", "types", "ids", and optionally '
                '"frames", "double sided".'
            )

        for key in config:
            if len(config[key]) != len(config["names"]):
                bad_config_err = (
                    f"Mismatched column count in {source}. "
                    f'Property "names" contains data for {len(config["names"])} '
                    f'plates while property "{key}" contains data for {len(config[key])}.'
                )
                raise self.ConfigurationError(bad_config_err)

        # assign properties
        self.names = config["names"]
        self.areas = config["areas"]
        self.ref_coeffs = config["ref coeffs"]
        self.abs_ref_coeffs = config["abs coeffs"]
        self.diff_ref_coeffs = config["dref coeffs"]
        self.spec_ref_coeffs = config["sref coeffs"]
        self.normals = config["normal vectors"]
        self.plate_types = config["types"]
        self.plate_ids = config["ids"]
        # optional columns: older configurations predate them
        self.frames = config.get("frames", [None] * len(self.names))
        self.double_sided = config.get("double sided", [False] * len(self.names))

    # ----------------------#
    # region    Properties #
    # ----------------------#
    @property
    def names(self) -> list[str]:
        """
        The names of each panel defined in the N-plate model.
        """
        return self._names

    @names.setter
    def names(self, input_val):
        # all names must be unique
        if len(set(input_val)) != len(input_val):
            non_unique_err = (
                "All plate names must be unique. Received plate " f"names: {input_val}."
            )
            raise ValueError(non_unique_err)

        # valid input -> assign property
        self._names = input_val

    @property
    def areas(self) -> list[ArrayWUnits]:
        """
        The surface are of each plate in the N-plate model. Expressed in units
        of :math:`km^2`.
        """
        return self._areas

    @areas.setter
    def areas(self, input_val):
        as_awus = []
        # look at each element of the list to...
        for element in input_val:
            # ...validate that values are numerical
            if not isinstance(element, (int, float)):
                not_num_err = (
                    'All elements of configuration property "areas" must be '
                    f"of type float or int. Received {type(element)}."
                )
                raise TypeError(not_num_err)

            # ...convert to AWU and save to list
            as_awus.append(ArrayWUnits(element, km**2))

        # valid input -> assign property
        self._areas = as_awus

    @property
    def ref_coeffs(self) -> list[float]:
        """
        The reflectivity coefficients of each plate in the N-plate model.
        """
        return self._ref_coeffs

    @ref_coeffs.setter
    def ref_coeffs(self, input_val):
        # must be numerical
        self._validate_numerical(input_val, "ref coeffs")

        # valid input -> assign property
        self._ref_coeffs = input_val

    @property
    def abs_ref_coeffs(self) -> list[float]:
        """
        The absorption coefficients of each plate in the N-plate model.
        """
        return self._abs_ref_coeffs

    @abs_ref_coeffs.setter
    def abs_ref_coeffs(self, input_val):
        # must be numerical
        self._validate_numerical(input_val, "abs coeffs")

        # valid input -> assign property
        self._abs_ref_coeffs = input_val

    @property
    def diff_ref_coeffs(self) -> list[float]:
        """
        The diffuse reflection coefficients of each plate in the N-plate model.
        """
        return self._diff_ref_coeffs

    @diff_ref_coeffs.setter
    def diff_ref_coeffs(self, input_val):
        # must be numerical
        self._validate_numerical(input_val, "dref coeffs")

        # valid input -> assign property
        self._diff_ref_coeffs = input_val

    @property
    def spec_ref_coeffs(self) -> list[float]:
        """
        The specular reflection coefficients of each plate in the N-plate model.
        """
        return self._spec_ref_coeffs

    @spec_ref_coeffs.setter
    def spec_ref_coeffs(self, input_val):
        # must be numerical
        self._validate_numerical(input_val, "sref coeffs")

        # valid input -> assign property
        self._spec_ref_coeffs = input_val

    @property
    def normals(self) -> list[np.ndarray]:
        """
        The normal vectors of each plate in the N-plate model.
        """
        return self._normals

    @normals.setter
    def normals(self, input_val):
        for normal in input_val:
            if len(normal) != 3:
                bad_norm_err = (
                    "Normal vectors must be a 1x3 vector. " f"Received {normal}."
                )
                raise ValueError(bad_norm_err)
        self._normals = input_val

    @property
    def plate_types(self) -> list[str]:
        """
        The type of each plate in the N-plate model. Either:

        * ``'Fixed'``, representing a static plate.
        * ``'CK'``, representing a time-variant plate, with reference frame
          defined by a SPICE C-matrix kernel.
        """
        return self._plate_types

    @plate_types.setter
    def plate_types(self, input_val):
        # elements must be either 'Fixed' or 'CK'
        if not (
            all(isinstance(item, str) and item in ("Fixed", "CK") for item in input_val)
        ):
            err_msg = (
                "'plates_type' should be a string list containing either 'Fixed' or 'CK'. "
                + "'Fixed' is for plates that have a normal unit vector that is fixed to "
                + "the body frame. 'CK' if its not fixed"
            )
            raise ValueError(err_msg)

        # valid input -> assign property
        self._plate_types = input_val

    @property
    def plate_ids(self) -> list[int | None]:
        """
        The NAIF integer code for each plate in the N-plate model. Fixed type plates have no
        associated ID, therefore all plate ID's of the respective type are ``None``. All
        CK type plate ID's are given as ``int``.
        """
        return self._ck_ids

    @plate_ids.setter
    def plate_ids(self, input_val):
        for element in input_val:
            if not isinstance(element, int):
                if element is not None:
                    not_int_err = (
                        'All elements of configuration property "spice ids" must be '
                        f"of type int or None. Received {type(element)}."
                    )
                    raise TypeError(not_int_err)

        # valid input -> assign property
        self._ck_ids = input_val

    @property
    def frames(self) -> list[str | int | None]:
        """
        The frame each plate's normal is constant in: ``None`` for the
        spacecraft body frame (oriented by the Spacecraft ``attitudeMode``),
        otherwise a SPICE frame name or frame ID. Always ``None`` for CK
        plates, whose orientation comes from their C-kernel.
        """
        return self._frames

    @frames.setter
    def frames(self, input_val):
        for name, plate_type, frame in zip(self.names, self.plate_types, input_val):
            # bool is an int subclass, but True/False is never a frame ID
            if frame is not None and (
                isinstance(frame, bool) or not isinstance(frame, (str, int, np.integer))
            ):
                raise self.ConfigurationError(
                    f'Plate "{name}" has frame {frame!r}; a "frames" entry must be a '
                    "SPICE frame name (str), a SPICE frame ID (int), or null for the "
                    "spacecraft body frame."
                )
            if plate_type == "CK" and frame is not None:
                raise self.ConfigurationError(
                    f'CK plate "{name}" has frame {frame!r}; a CK plate is oriented by '
                    'its C-kernel, so its "frames" entry must be null. Use a "Fixed" '
                    "plate to fix a normal in a named frame."
                )

        # valid input -> assign property
        self._frames = list(input_val)

    @property
    def double_sided(self) -> list[bool]:
        """
        Whether each plate is double-sided: lit from behind, it acts as a
        plate with the opposite normal. Single-sided plates lit from behind
        contribute nothing.
        """
        return self._double_sided

    @double_sided.setter
    def double_sided(self, input_val):
        for name, flag in zip(self.names, input_val):
            if not isinstance(flag, (bool, np.bool_)):
                raise self.ConfigurationError(
                    f'Plate "{name}" has "double sided" entry {flag!r}; it must be '
                    "true or false."
                )

        # valid input -> assign property
        self._double_sided = [bool(flag) for flag in input_val]

    @property
    def body_frame_plates(self) -> list[bool]:
        """
        Whether each plate's normal is given in the spacecraft body frame
        (a Fixed plate with no frame). :meth:`get_all_normals` returns these
        normals as configured and the others in an inertial frame.
        """
        return [
            plate_type == "Fixed" and frame is None
            for plate_type, frame in zip(self.plate_types, self.frames)
        ]

    # endregion Properties #
    # ----------------------#

    # ----------------#
    # region Methods #
    # ----------------#
    def __repr__(self):
        """
        Overloading the printable representation operator for an nPlateModel object.
        """
        return f"N-Plate Model containing {len(self.names)} plates"

    def to_dict(self) -> dict:
        """
        Serialize the plate configuration to a JSON-friendly dict.

        The dict holds exactly the keys a configuration file would, so it can be
        fed straight back to the constructor as ``plates_data`` (which is what
        :meth:`from_dict` does). Areas are written as plain floats in
        :math:`km^2`, the units the constructor expects.

        Returns
        -------
        dict
            Serialized plate configuration.
        """
        return {
            "names": list(self.names),
            "areas": [float(np.asarray(a.values).ravel()[0]) for a in self.areas],
            "ref coeffs": list(self.ref_coeffs),
            "abs coeffs": list(self.abs_ref_coeffs),
            "dref coeffs": list(self.diff_ref_coeffs),
            "sref coeffs": list(self.spec_ref_coeffs),
            "normal vectors": [np.asarray(n).ravel().tolist() for n in self.normals],
            "types": list(self.plate_types),
            "ids": list(self.plate_ids),
            "frames": [int(f) if isinstance(f, np.integer) else f for f in self.frames],
            "double sided": list(self.double_sided),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "nPlateModel":
        """
        Reconstruct an nPlateModel from a dict produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialized plate configuration.

        Returns
        -------
        nPlateModel
            The reconstructed model.
        """
        return cls(plates_data=data)

    def _validate_numerical(self, list2check: list, prop2check: str):
        """
        Validate that every element of a list is a numeric scalar.

        Parameters
        ----------
        list2check : list
            The list of values to validate.
        prop2check : str
            The name of the configuration property being checked, used in the
            error message.

        Raises
        ------
        TypeError
            If any element is not an ``int`` or ``float``.
        """
        for element in list2check:
            if not isinstance(element, (int, float)):
                not_num_err = (
                    f'All elements of configuration property "{prop2check}" must be '
                    f"of type float or int. Received {type(element)}."
                )
                raise TypeError(not_num_err)

    @staticmethod
    def _epoch_to_et(epoch):
        """
        Raw ephemeris time (TDB seconds past J2000) of an epoch.

        Accepts an EpochArray or a bare numeric ET, which avoids the wrapper
        overhead on every ODE evaluation.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch in ephemeris time (TDB).

        Returns
        -------
        float or np.ndarray
            The ET value(s).

        Raises
        ------
        ValueError
            If ``epoch`` is an EpochArray not in TDB.
        TypeError
            If ``epoch`` is neither an EpochArray nor numeric.
        """
        if isinstance(epoch, EpochArray):
            if epoch.system != "TDB":
                raise ValueError(
                    "Argument [epoch] must be given in ephemeris time (TDB). "
                    f"Received {epoch.system}."
                )
            return epoch.times.values
        if isinstance(epoch, (int, float, np.floating, np.ndarray)):
            return epoch
        raise TypeError(
            f"Argument [epoch] must be an EpochArray or numeric ET value. "
            f"Received {type(epoch)}."
        )

    def _get_ck_normals(
        self, epoch, sc_id: int, include_IDs: bool = False, inertial_frame: str = "J2000"
    ):
        """
        Queries the normal vectors of each C-Kernel-defined panel in the N-plate model at
        a given epoch, expressed in an inertial frame.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch at which to query the C-kernel in ephemeris time (TDB).
            Accepts either an EpochArray (TDB) or a raw numeric ET value (seconds past J2000).

        sc_id : int
            NAIF spacecraft ID code for the craft C-kernel plates are attached to. Required to
            convert to encoded SCLK ticks.

        include_IDs : bool, optional
            Flag that signifies whether or not to include the SPICE ID(s) of queried C-kernel
            plates. Defaults to ``False``.

        inertial_frame : str, optional
            The frame the normals are returned in. Defaults to ``"J2000"``.

        Returns
        -------
        normals : list
            If ``include_IDs == False``, a list of normals for each CK type panel
            in the model.

        ids_and_normals : list
            If ``include_IDs == True``, a list where each element is a list with the
            SPICE ID and normal for each CK type panel in the model.

        Raises
        ------
        PlateOrientationError
            If SPICE cannot convert ``epoch`` to spacecraft clock ticks for
            ``sc_id`` (no SCLK kernel), or cannot orient a CK plate at
            ``epoch`` (no CK for its ID, or no coverage at that epoch).
        """
        et_value = self._epoch_to_et(epoch)

        # -----------------------------#
        # valid inputs -> get normals #
        # -----------------------------#
        # convert from ET to SCLK
        try:
            epoch = SpiceManager.sce2c(sc_id, et_value)
        except spice.utils.exceptions.SpiceyError as err:
            spice.reset()
            reason = getattr(err, "short", None) or err
            raise self.PlateOrientationError(
                f"Unable to orient the CK plates at ET {et_value}: no spacecraft "
                f"clock for spacecraft {sc_id} ({reason}). Load its SCLK kernel."
            ) from err

        # get normals for each CK plate
        ck_id_inds = [i for i, e in enumerate(self.plate_types) if e == "CK"]

        if include_IDs:
            ids_and_normals = []
        else:
            normals = []

        for i in ck_id_inds:
            # get the SPICE ID of the plate
            plate_id = self.plate_ids[i]

            # query its attitude from the C-kernel: the C-matrix rotates
            # inertial vectors into the plate frame, so its transpose takes the
            # plate-frame normal out to the inertial frame
            try:
                cmat = spice.ckgp(plate_id, epoch, 0, inertial_frame)[0]
            except spice.utils.exceptions.SpiceyError as err:
                # spiceypy raises NotFoundError without clearing the SPICE
                # error status, which would fail the next, unrelated, call
                spice.reset()
                reason = getattr(err, "short", None) or err
                raise self.PlateOrientationError(
                    f'Unable to orient CK plate "{self.names[i]}" (CK ID {plate_id}) '
                    f"relative to {inertial_frame} at ET {et_value}: {reason}. "
                    f"Load a CK for ID {plate_id} that covers this epoch."
                ) from err
            base_normal = np.asarray(self.normals[i], dtype=float)
            rot_normal = np.asarray(cmat).T @ base_normal

            if include_IDs:
                # bundle ID and normal as one entry
                ids_and_normals.append([plate_id, rot_normal])
            else:
                # only save normal
                normals.append(rot_normal)

        if include_IDs:
            # return ID-normal pairs for each CK plate
            return ids_and_normals
        else:
            # return only normals
            return normals

    def _get_frame_normals(self, epoch, inertial_frame: str = "J2000") -> list:
        """
        Queries the normal vectors of each Fixed panel defined in a named frame
        at a given epoch, expressed in an inertial frame.

        Parameters
        ----------
        epoch : EpochArray or float
            The epoch in ephemeris time (TDB).
        inertial_frame : str, optional
            The frame the normals are returned in. Defaults to ``"J2000"``.

        Returns
        -------
        list
            The normal of each Fixed plate with a frame, in plate order.

        Raises
        ------
        PlateOrientationError
            If a frame ID is unknown to SPICE, or SPICE cannot rotate a frame
            into ``inertial_frame`` at ``epoch`` (e.g. no CK coverage).
        """
        et = float(np.squeeze(self._epoch_to_et(epoch)))

        normals = []
        for name, plate_type, frame, normal in zip(
            self.names, self.plate_types, self.frames, self.normals
        ):
            if plate_type != "Fixed" or frame is None:
                continue
            frame_name = frame
            if not isinstance(frame, str):
                # resolved per call, not cached: the frame kernel may be
                # loaded after the model is built
                frame_name = spice.frmnam(int(frame))
                if not frame_name:
                    raise self.PlateOrientationError(
                        f'Plate "{name}" is fixed in frame ID {frame}, which no loaded '
                        "kernel defines. Load the frame kernel (FK) that defines it."
                    )
            try:
                dcm = SpiceManager.get_xfrm(frame_name, inertial_frame, et)
            except spice.utils.exceptions.SpiceyError as err:
                raise self.PlateOrientationError(
                    f'Unable to orient plate "{name}" (fixed in frame {frame_name}) '
                    f"relative to {inertial_frame} at ET {et}: {err.short or err}"
                ) from err
            normals.append(np.asarray(dcm) @ np.asarray(normal, dtype=float))

        return normals

    def get_all_normals(self, epoch, sc_id: int, inertial_frame: str = "J2000"):
        """
        Queries the normal vectors of each panel in the N-plate model at a given epoch.

        Normals of plates in the spacecraft body frame (``body_frame_plates``)
        are returned as configured; the others, CK plates and Fixed plates in a
        named frame, are rotated into ``inertial_frame``. A caller that knows
        the attitude maps the latter into the body frame with its transpose.

        Parameters
        ----------
        epoch : EpochArray
            The epoch at which to query the C-kernel in ephemeris time (TDB) for all
            CK type panels in the model.

        sc_id : int
            NAIF spacecraft ID code for the craft C-kernel plates are attached to. Required to
            convert to encoded SCLK ticks.

        inertial_frame : str, optional
            The frame CK and named-frame normals are returned in. Defaults to
            ``"J2000"``.

        Returns
        -------
        normals : list
            A list of normals for each panel in the model.

        Raises
        ------
        PlateOrientationError
            If a CK plate or a Fixed plate's named frame cannot be oriented at
            ``epoch``.
        """
        if all(self.body_frame_plates):
            # return all normals
            return self.normals

        # check for C-kernel plates
        ck_normals = []
        if "CK" in self.plate_types:
            # query normals for all C-kernel plates at the given time
            ck_normals = self._get_ck_normals(
                epoch, sc_id, inertial_frame=inertial_frame
            )
        frame_normals = self._get_frame_normals(epoch, inertial_frame)

        normals = []
        ck_ind = 0  # counter to grab from queried CK normals
        frame_ind = 0  # counter to grab from queried named-frame normals
        for i, normal in enumerate(self.normals):
            # constant in the body frame -> pull from config file
            if self.body_frame_plates[i]:
                normals.append(normal)
            # pull from queried normals
            elif self.plate_types[i] == "CK":
                normals.append(ck_normals[ck_ind])
                ck_ind += 1
            else:
                normals.append(frame_normals[frame_ind])
                frame_ind += 1

        # return constant and non-constant normals
        return normals
