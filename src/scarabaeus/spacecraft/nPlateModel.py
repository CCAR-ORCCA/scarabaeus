# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, EpochArray, SpiceManager

import json, warnings

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
    illuminated plates.

    Parameters
    ----------
    plates_config_path : str
        Path to a JSON file containing all necessary information to configure the N-plate model. See
        Notes section for more information on configuration file formatting.

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
        * ``"Fixed"`` defines a static plate.
        * ``"CK"`` defines a time-variant plate with associated reference frame defined by a
          SPICE C-matrix kernel.
    * ``ids``: an array of integers and/or nulls, where
        * a null defines a Fixed plate — it has no NAIF integer code.
        * an integer defines the NAIF ID of a CK plate.

    .. note:: The ``abs``, ``dref``, and ``sref`` must sum to unity for each plate.

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
                "ids"            : [null     , null      , -64001   ]
            }

    References
    ----------
    """

    # -------------#
    # Constructor #
    # -------------#
    def __init__(self, plates_config_path: str):
        # read in plate config file
        with open(plates_config_path, "r") as file:
            config = json.load(file)

        # validate data in config
        for key in config:
            if len(config[key]) != len(config["names"]):
                bad_config_err = (
                    f"Mismatched column count in file {plates_config_path}. "
                    f'Property "names" contains data for {len(config["names"])} '
                    f'plates while property "{key}" contains data for {len(config[key])}.'
                )
                raise ValueError(bad_config_err)

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

    def _get_ck_normals(self, epoch, sc_id: int, include_IDs: bool = False):
        """
        Queries the normal vectors of each C-Kernel-defined panel in the N-plate model at
        a given epoch.

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

        Returns
        -------
        normals : list
            If ``include_IDs == False``, a list of normals for each CK type panel
            in the model.

        ids_and_normals : list
            If ``include_IDs == True``, a list where each element is a list with the
            SPICE ID and normal for each CK type panel in the model.
        """
        # ------------------#
        # input validation #
        # ------------------#
        # Extract raw ET value — accept EpochArray or bare numeric to avoid wrapper overhead
        # on every ODE evaluation.  CK availability is validated implicitly: spice.ckgp raises
        # if no appropriate kernel is loaded, and get_all_normals catches that exception.
        if isinstance(epoch, EpochArray):
            if epoch.system != "TDB":
                raise ValueError(
                    "Argument [epoch] must be given in ephemeris time (TDB). "
                    f"Received {epoch.system}."
                )
            et_value = epoch.times.values
        elif isinstance(epoch, (int, float, np.floating, np.ndarray)):
            et_value = epoch
        else:
            raise TypeError(
                f"Argument [epoch] must be an EpochArray or numeric ET value. "
                f"Received {type(epoch)}."
            )

        # -----------------------------#
        # valid inputs -> get normals #
        # -----------------------------#
        # convert from ET to SCLK
        epoch = SpiceManager.sce2c(sc_id, et_value)

        # get normals for each CK plate
        ck_id_inds = [i for i, e in enumerate(self.plate_types) if e == "CK"]

        if include_IDs:
            ids_and_normals = []
        else:
            normals = []

        for i in ck_id_inds:
            # get the SPICE ID of the plate
            plate_id = self.plate_ids[i]

            # query its attitude from the C-kernel and convert to quaternion
            dcm = spice.ckgp(plate_id, epoch, 0, "J2000")[0]
            base_normal = self.normals[i]
            rot_normal = np.array(dcm) @ base_normal

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

    def get_all_normals(self, epoch, sc_id: int):
        """
        Queries the normal vectors of each panel in the N-plate model at a given epoch.

        Parameters
        ----------
        epoch : EpochArray
            The epoch at which to query the C-kernel in ephemeris time (TDB) for all
            CK type panels in the model.

        sc_id : int
            NAIF spacecraft ID code for the craft C-kernel plates are attached to. Required to
            convert to encoded SCLK ticks.

        Returns
        -------
        normals : list
            A list of normals for each panel in the model.
        """
        # check for C-kernel plates
        if "CK" in self.plate_types:
            # query normals for all C-kernel plates at the given time
            try:
                ck_normals = self._get_ck_normals(epoch, sc_id)
            except:
                cant_find_warn = (
                    "Unable to retrieve orientation for one or more CK plates "
                    f"at epoch {epoch}. Normals set to [1, 0, 0]."
                )
                warnings.warn(cant_find_warn)
                ck_normals = [1, 0, 0] * np.ones((self.plate_types.count("CK"), 1))

            normals = []
            ck_ind = 0  # counter to grab from queried CK normals
            for i, normal in enumerate(self.normals):
                # constant for fixed plate -> pull from config file
                if self.plate_types[i] == "Fixed":
                    normals.append(normal)
                # pull from queried normals
                else:
                    normals.append(ck_normals[ck_ind])
                    ck_ind += 1

            # return constant and non-constant normals
            return normals

        else:
            # return all normals
            return self.normals
