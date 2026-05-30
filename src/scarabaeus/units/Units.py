# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Dimensions

import numpy as np
import numpy.typing as npt
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import re, math, random


# class definition
class Units:
    """Defines  measurements as defined by the International System of
    Units [[1]_]. From this system, the following four base units are adopted:

    1. Gram, as a measurement of mass
    2. Meter, as a measurement of length
    3. Second, as a measurement of time
    4. Radian, as a measurement of angle

    These units form the base of all measurement operations in Scarabaeus.

    Parameters
    ----------
    dimensions : Dimensions
        The physical dimensions of the unit.

    scales : numpy.ndarray | list
        A 1x4 vector describing the scale relationships for each dimension. For example,
        a kilogram would contain the scales ``[1e3, 0, 0, 0]``.

    See Also
    --------
    scarabaeus.Dimensions : the building blocks of Units.
    scarabaeus.ArrayWUnits : interface between numbers and units in Scarabaeus.

    Notes
    -----
    Lunar distance defined by JPL [[2]_].
    Astronomical unit defined by the IAU [[3]_].

    References
    ----------
    .. [1] NIST,
      https://www.nist.gov/pml/owm/metric-si/si-units

    .. [2] JPL CNEOS,
      https://cneos.jpl.nasa.gov/glossary/LD.html

    .. [3] XXVIII General Assembly of IAU, Beijing, 31 Aug 2012
    """

    # class constants
    # metric prefixes and their respective scaling value
    _prefixes = {
        "T": 1e12,  # tera
        "G": 1e9,  # giga
        "M": 1e6,  # mega
        "k": 1e3,  # kilo
        "h": 1e2,  # hecto
        "da": 1e1,  # deca
        "d": 1e-1,  # deci
        "c": 1e-2,  # centi
        "m": 1e-3,  # milli
        "micro": 1e-6,  # micro
        "n": 1e-9,  # nano
        "pico": 1e-12,  # pico
    }

    # full definitions of each base unit
    # note that units have an extra flag "prfx" to denote units that should or
    # should not be mutated by metric prefixes
    _base_units = {
        "g": {
            "dimensions": np.array([1, 0, 0, 0]),
            "scales": np.array([1, 0, 0, 0]),
            "prfx": True,
        },
        "m": {
            "dimensions": np.array([0, 1, 0, 0]),
            "scales": np.array([0, 1, 0, 0]),
            "prfx": True,
        },
        "sec": {
            "dimensions": np.array([0, 0, 1, 0]),
            "scales": np.array([0, 0, 1, 0]),
            "prfx": True,
        },
        "rad": {
            "dimensions": np.array([0, 0, 0, 1]),
            "scales": np.array([0, 0, 0, 1]),
            "prfx": True,
        },
        "unitless": {
            "dimensions": np.array([0, 0, 0, 0]),
            "scales": np.array([0, 0, 0, 0]),
            "prfx": True,
        },
    }

    # named units with special scale relations (ie 1 day = 86400 sec)
    _unit_dictionary = {
        # length units
        "lsec": {
            "dimensions": np.array([0, 1, 0, 0]),
            "scales": np.array([0, 299792458, 0, 0]),
            "prfx": False,
        },
        "lmin": {
            "dimensions": np.array([0, 1, 0, 0]),
            "scales": np.array([0, 17987547480.0, 0, 0]),
            "prfx": False,
        },
        "LD": {
            "dimensions": np.array([0, 1, 0, 0]),
            "scales": np.array([0, 384400000, 0, 0]),
            "prfx": False,
        },
        "AU": {
            "dimensions": np.array([0, 1, 0, 0]),
            "scales": np.array([0, 149597870700.0, 0, 0]),
            "prfx": False,
        },
        # time units
        "min": {
            "dimensions": np.array([0, 0, 1, 0]),
            "scales": np.array([0, 0, 60, 0]),
            "prfx": False,
        },
        "hr": {
            "dimensions": np.array([0, 0, 1, 0]),
            "scales": np.array([0, 0, 3600, 0]),
            "prfx": False,
        },
        "day": {
            "dimensions": np.array([0, 0, 1, 0]),
            "scales": np.array([0, 0, 86400, 0]),
            "prfx": False,
        },
        "Hz": {
            "dimensions": np.array([0, 0, -1, 0]),
            "scales": np.array([0, 0, 1, 0]),
            "prfx": True,
        },
        # angle units
        "deg": {
            "dimensions": np.array([0, 0, 0, 1]),
            "scales": np.array([0, 0, 0, np.pi / 180.0]),
            "prfx": False,
        },
        "arcsec": {
            "dimensions": np.array([0, 0, 0, 1]),
            "scales": np.array([0, 0, 0, np.pi / (180.0 * 3600.0)]),
            "prfx": True,
        },
        "arcmin": {
            "dimensions": np.array([0, 0, 0, 1]),
            "scales": np.array([0, 0, 0, np.pi / (60.0 * 180.0)]),
            "prfx": False,
        },
        # compound units
        "N": {
            "dimensions": np.array([1, 1, -2, 0]),
            "scales": np.array([1e3, 1, 1, 0]),
            "prfx": True,
        },
        "mu": {
            "dimensions": np.array([0, 3, -2, 0]),
            "scales": np.array([0, 1e9, 1, 0]),
            "prfx": True,
        },
        # dimensionless units
        "pix": {
            "dimensions": np.array([0, 0, 0, 0]),
            "scales": np.array([0, 0, 0, 0]),
            "prfx": False,
        },
    }

    # region Constructor
    def __init__(
        self, dimensions: Dimensions = None, scales: npt.NDArray | list = None
    ):
        # unitless construction
        if (dimensions.powers is None and scales is None) or (
            np.all(dimensions.powers == 0)
        ):
            self._name = "unitless"
            self.dimensions = Dimensions([0, 0, 0, 0])  # input validation in setter
            self._scales = np.array([0.0, 0.0, 0.0, 0.0])
            self._unitless = True

        # non-unitless construction
        elif (all(dimensions.powers) != None) and (all(scales) != None):
            # don't assign name, it will be created if needed (many times the name isn't use, so don't want extra overhead generating it now)
            self._name = None

            # assign powers and scales from input
            self.dimensions = dimensions  # input validation in setter
            self.scales = (
                scales  # input validation and/or conditioning in property setter
            )

            # necessarily non-unitless -> set flag accordingly
            self._unitless = False

    # region Properties
    @property
    def name(self) -> str:
        """The written name of the unit."""
        # if a name has been generated, return in
        if self._name is not None:
            return self._name

        # otherwise, name hasn't been generated yet -> create one and return
        else:
            # generate and set
            self._name = self._name_casting(self.dimensions.powers, self.scales)

            # now return
            return self._name

    @property
    def dimensions(self) -> Dimensions:
        """Dimensions object containing information about the dimensionality of
        the unit.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, input_val):
        ## input validation
        # must be a Dimensions object
        if not isinstance(input_val, Dimensions):
            not_dim_err = (
                "Argument [dimensions] must be an object of type Dimensions "
                f"Received: {type(input_val)}."
            )
            raise TypeError(not_dim_err)

        # valid input
        self._dimensions = input_val

    @property
    def scales(self) -> np.ndarray:
        """1x4 array defining the unit's scaled relation to each base dimension."""
        return self._scales

    @scales.setter
    def scales(self, input_val):
        ## input validation
        # must be a numpy array or list
        if not (isinstance(input_val, np.ndarray) or isinstance(input_val, list)):
            not_np_or_list_err = (
                "Argument [scales] must be an object of type np.ndarry or "
                f"list. Received: {type(input_val)}."
            )
            raise TypeError(not_np_or_list_err)

        # convert list input to numpy array
        if isinstance(input_val, list):
            input_val = np.array(input_val)

        # elements must all be numerical
        if not np.issubdtype(input_val.dtype, np.number):
            raise TypeError(
                "The elements in the input scales must be numerical values."
            )

        # valid input
        self._scales = input_val

    @property
    def unitless(self) -> bool:
        """
        Boolean flag indicating if the Unit object is unitless or not.

        ``True`` if the Unit object is unitless., otherwise ``False``.
        """
        return self._unitless

    # endregion Properties

    # region Operators
    def __repr__(self) -> str:
        return self.name

    def __mul__(self, other) -> Self:
        ## Unit object * Unit object
        if isinstance(other, self.__class__):
            ## multiplied by another unit -> perform logic on powers and scales
            # add powers
            new_pwr = self.dimensions.powers + other.dimensions.powers
            new_dim = Dimensions(new_pwr)

            # multiply scales
            new_scl = np.zeros(4)
            for i, (self_scl, other_scl, self_pow, other_pow) in enumerate(
                zip(
                    self.scales,
                    other.scales,
                    self.dimensions.powers,
                    other.dimensions.powers,
                )
            ):
                if new_pwr[i] != 0:
                    if (self_scl != 0) and (other_scl != 0):
                        ## both are nonzero -> keep their product
                        # get scale power with respect to base units and use same sign as power
                        self_scl_wrt_base = math.copysign(
                            math.log10(self_scl), self_pow
                        )
                        other_scl_wrt_base = math.copysign(
                            math.log10(other_scl), other_pow
                        )

                        # raise back to a power
                        if new_pwr[i] > 0:
                            new_scl[i] = 10.0 ** (
                                self_scl_wrt_base + other_scl_wrt_base
                            )
                        else:
                            new_scl[i] = 10.0 ** abs(
                                self_scl_wrt_base + other_scl_wrt_base
                            )

                    else:
                        # one is zero -> keep nonzero scale
                        if self_scl != 0:
                            new_scl[i] = self_scl
                        else:
                            new_scl[i] = other_scl

            return Units(new_dim, new_scl)

        ## Unit object * float or int
        elif isinstance(other, float) or isinstance(other, int):
            # multiplied by a single value -> perform logic on scales only
            if other != 0:
                ## nonzero product -> normal multiplication
                # scales multiplied by vectorized other
                new_scl = self.scales * (other * np.ones(4, dtype=float))

                # return new unit
                return Units(self.dimensions, new_scl)
            else:
                # multiplying by 0 -> return 0
                return 0.0

        ## Unit object * np.array or list
        elif isinstance(other, np.ndarray) or isinstance(other, list):
            # force numpy array to simplify calculation
            if isinstance(other, list):
                other = np.array(other)

            # initialize the list/numpy array to return
            to_return = []

            # loop through each element and perform multiplication there
            # if other.ndim > 1:
            #     for i in range(other.ndim + 1):
            #         for element in other[i]:
            #             to_return.append(self * element)

            #     # reshape back to original form
            #     to_return = np.reshape(to_return, other.shape)
            # else:
            #     for element in other:
            #         to_return.append(self * element)
            for element in other.flatten():
                to_return.append(self * element)

            to_return = np.reshape(to_return, other.shape)

            # return depending on type being multiplied by
            if isinstance(other, np.ndarray):
                # multiplied by array -> return an array
                return to_return
            else:
                # multiplied by list -> return a list
                return to_return.tolist()

        ## Unit object * invalid object
        else:
            # can't multiply by this -> raise error
            cant_mul_err = (
                "Unit objects do not support multiplication by objects of "
                f"type {type(other)}"
            )
            raise TypeError(cant_mul_err)

    def __rmul__(self, other) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other) -> Self:
        ## Unit object / Unit object
        if isinstance(other, self.__class__):
            # subtract powers
            new_pwr = self.dimensions.powers - other.dimensions.powers
            new_dim = Dimensions(new_pwr)

            # boolean mask delineating zero and non-zero elements in the dimensional powers array
            dim_mask = np.zeros(4, dtype=float)
            dim_mask[np.nonzero(new_dim.powers)] = 1

            # divide scales per element
            new_scl = []
            for i, scale in enumerate(self.scales):
                if dim_mask[i] == 1:
                    # scaling relevant to dimensions -> divide
                    if (scale != 0) and (other.scales[i] != 0):
                        # both have scaling for this dimension -> divide
                        new_scl.append(scale / other.scales[i])

                    elif (scale != 0) and (other.scales[i] == 0):
                        # self has scaling for this dimensions but other does not -> keep self's scaling
                        new_scl.append(scale)

                    elif (scale == 0) and (other.scales[i] != 0):
                        # self does not have scaling for this dimension but other does -> keep other's scaling
                        new_scl.append(other.scales[i])

                    else:
                        # both don't have scales -> set = 0
                        new_scl.append(0)
                else:
                    # scaling irrelevant to dimensions -> set = 0
                    new_scl.append(0)

            # return new unit
            return Units(new_dim, np.array(new_scl))

        ## Unit object / float or int
        elif isinstance(other, float) or isinstance(other, int):
            # raise divisor to a negative power and multiply
            return self * (other**-1)

        ## Unit object / invalid object
        else:
            # can't divide by this -> raise error
            cant_div_err = (
                "Unit objects do not support division by objects of "
                f"type {type(other)}"
            )
            raise TypeError(cant_div_err)

    def __rtruediv__(self, other) -> Self:
        ## float or int / Unit object
        if isinstance(other, float) or isinstance(other, int):
            # flip powers
            new_pwr = []
            for power in self.dimensions.powers:
                new_pwr.append(-power if power != 0 else 0)

            new_dim = Dimensions(np.array(new_pwr))

            # multiply scales by other
            new_scl = (other * np.ones(4, dtype=float)) * self.scales

            # return new unit
            return Units(new_dim, new_scl)

        ## np.array or list of Unit objects / Unit object
        elif isinstance(other, np.ndarray) or isinstance(other, list):
            to_return = []
            for element in other:
                # first ensure element is a valid object for division
                if not (
                    isinstance(element, Units)
                    or isinstance(element, float)
                    or isinstance(element, int)
                ):
                    bad_type_err = (
                        "All elements of division array-like must be of "
                        "type Unit, float, or int."
                    )
                    raise TypeError(bad_type_err)

                # valid divisor -> divide and append
                to_return.append(element / self)

            # return depending on type being divided by
            if isinstance(other, list):
                # divided by list -> return a list
                return to_return
            else:
                # divided by ndarry -> return an ndarray
                return np.array(to_return)

        ## invalid object / Unit object
        else:
            # can't divide by this -> raise error
            cant_rdiv_err = (
                "Unit objects do not support reverse order division by objects of "
                f"type {type(other)}"
            )
            raise TypeError(cant_rdiv_err)

    def __pow__(self, other) -> Self:
        if isinstance(other, (int, float)):
            new_pwr = self.dimensions.powers * other
            new_dim = Dimensions(new_pwr)

            new_scl = np.power(
                self.scales.astype("float"), np.where(self.scales != 0, abs(other), 1)
            )

            return Units(new_dim, new_scl)

        else:
            raise TypeError(
                "Exponentiation of Unit objects is only supported for int or float. "
                f"Received exponentiation request with object of type: {type(other)}"
            )

    def sqrt(self) -> Self:
        """The square root of the Unit object (equivalent to ``unit ** 0.5``)."""
        return self**0.5

    def __eq__(self, other) -> bool:
        # input validation
        if not isinstance(other, self.__class__):
            bad_type_err = (
                "Cannot verify equality of Unit object with non-Unit object. "
                f"Received object of type: {type(other)}"
            )
            raise TypeError(bad_type_err)

        # valid input -> return equality
        if (self.dimensions == other.dimensions) and (
            np.array_equal(self.scales, other.scales)
        ):
            # dimensions and scales equal -> return True
            return True
        else:
            # dimensions and/or scales not equal -> return false
            return False

    def __ne__(self, other) -> bool:
        # opposite of equality
        return not self.__eq__(other)

    # endregion Operators

    # region Methods
    @classmethod
    def get_units(cls, unit_str: str | list[str]) -> Self | tuple[Self]:
        # NOTE: the table in the docstring notes section only works with that last line having an extra space for some reason
        """Returns a Unit object or objects described by the given string formatted as a
        metric prefix (if necessary) and metric base unit.

        If a ``list[str]`` is given, a Unit object for each element in the list will
        be returned as a tuple of length n where n equals the length of the given
        list.

        Additionally, there are a few single ``str`` inputs that return different sets
        of Unit objects. For example, ``'base'`` returns a tuple containing each of the
        four base units. See the Notes section for more information on unique single ``str``
        inputs.

        See tables in the Notes section for metric base units and prefixes respectively.

        Parameters
        ----------
        unit_str : str
            The name of the requested unit.

        Returns
        -------
        req_units : Units
            A single Units or n-length tuple of Units objects where n equals the number of
            requested units. If only one unit is requested, a single Units object is returned.

        See Also
        --------
        units-in-scb : For an in-depth explanation of how Units are used throughout Scarabaeus.

        Notes
        -----
        **Base Units**

        +--------+-----------+-----------+
        | Name   | Dimension | Symbol    |
        +========+===========+===========+
        | Gram   | Mass      | ``'g'``   |
        +--------+-----------+-----------+
        | Meter  | Length    | ``'m'``   |
        +--------+-----------+-----------+
        | Second | Time      | ``'sec'`` |
        +--------+-----------+-----------+
        | Radian | Angle     | ``'rad'`` |
        +--------+-----------+-----------+

        **Metric Prefixes**

        +-------+----------+-------+--+-------+-------------+--------+
        | Name  | Prefix   | Power |  | Name  | Prefix      | Power  |
        +=======+==========+=======+==+=======+=============+========+
        | Tera  | ``'T'``  | 10e12 |  | Deci  | ``'d'``     | 10e-1  |
        +-------+----------+-------+--+-------+-------------+--------+
        | Giga  | ``'G'``  | 10e9  |  | Centi | ``'c'``     | 10e-2  |
        +-------+----------+-------+--+-------+-------------+--------+
        | Mega  | ``'M'``  | 10e6  |  | Milli | ``'m'``     | 10e-3  |
        +-------+----------+-------+--+-------+-------------+--------+
        | Kilo  | ``'k'``  | 10e3  |  | Micro | ``'micro'`` | 10e-6  |
        +-------+----------+-------+--+-------+-------------+--------+
        | Hecto | ``'h'``  | 10e2  |  | Nano  | ``'n'``     | 10e-9  |
        +-------+----------+-------+--+-------+-------------+--------+
        | Deka  | ``'da'`` | 10e1  |  | Pico  | ``'pico'``  | 10e-12 |
        +-------+----------+-------+--+-------+-------------+--------+

        In addition to direct requests, ``get_units()`` also includes a few "set strings" that return mutiple
        relevant units with a single call. The sets and their commands are below:

        +----------------------+---------------------+-------------------------------------------------+
        | Single ``str`` Input | Return Tuple Length | Return Tuple Contents                           |
        +======================+=====================+=================================================+
        | ``'base'``           | ``4``               | (gram, meter, second, radian)                   |
        +----------------------+---------------------+-------------------------------------------------+
        | ``'common'``         | ``7``               | | (kilogram, kilometer, second, radian          |
        |                      |                     | | degree, Newton, :math:`{\\mu}`)                |
        +----------------------+---------------------+-------------------------------------------------+

        Examples
        --------
        Create a single unit using ``get_units()``:

        >>> import scarabaeus as scb
        >>> kg = scb.Units.get_units('kg')

        Create multiple singular units using ``get_units()``:

        >>> km, sec, mN, rad = scb.Units.get_units(['km', 'sec', 'mN', 'rad'])

        Create multiple units with a using ``get_units()`` set command:

        >>> g, m, sec, rad = scb.Units.get_units('base')
        """
        match unit_str:
            # requested unique string commands
            case "base":
                return cls.get_units(["g", "m", "sec", "rad"])

            case "common":
                return cls.get_units(["kg", "km", "sec", "rad", "deg", "N", "mu"])

            case "time set":
                return cls.get_units(
                    ["sec", "microsec", "msec", "min", "hr", "day", "Hz"]
                )

            case "length set":
                return cls.get_units(["km", "lsec", "lmin", "LD", "AU"])

            case "angular set":
                return cls.get_units(["rad", "deg", "arcsec", "arcmin"])

            # not a unique str command -> get unit(s) accordingly
            case _:
                # turn singular inputs into a list to loop through
                if isinstance(unit_str, str):
                    unit_str = [unit_str]

                # create units for each given string
                to_return = []
                for unit in unit_str:
                    ## input validation
                    if unit == "":
                        raise ValueError("Please provide a valid unit.")

                    # before assigning, check for compound units
                    operator_syms = ["/", "*", "^"]
                    compound = False
                    for symbol in operator_syms:
                        if unit.find(symbol) != -1:
                            # definition contains an operator -> compound unit, set flag
                            compound = True

                    # perform logic depending on unit dimension
                    if not compound:
                        # requested non compound, append and break
                        init_vals = cls._get_unit_name(cls, unit)
                        to_return.append(cls(init_vals[0], init_vals[1]))
                    else:
                        # NOTE: this is not fully implemented currently
                        raise Exception(
                            "Units does not currently support compound unit production. Please "
                            "only request single units with optional prefixes like [mm], [kg], or [day]."
                        )
                        # # requested compound unit
                        # # ensure all open parentheses have a matching closed parentheses
                        # all_parens = [m for m in re.findall(r'[(\($\))]', unit_str)]
                        # if len(all_parens) % 2 == 0:
                        #     # even number of parentheses, now check that each open has a closed
                        #     paren_str = ''.join(all_parens)
                        #     if paren_str.count('(') != paren_str.count('('):
                        #         # found missing open or closed parentheses -> raise error
                        #         raise ValueError('Unit definition missing opening or closing brackets.')
                        # else:
                        #     # odd number of parentheses, necessarily missing an open or close -> raise error
                        #     raise ValueError('Unit definition missing opening or closing brackets.')

                        # # order of operations: start with ()
                        # # isolate each set of ()
                        # paren_terms = []          # ordered from innmerost to outermost parentheses
                        # search_parens = unit_str
                        # for i in range(len(all_parens)):
                        #     # find innermost parentheses
                        #     try:
                        #         inner_loc = [m.span() for m in re.finditer(r'(\([^\(]*?\))', search_parens)][0]
                        #     except:
                        #         # no more inside parens, stop looping
                        #         break
                        #     else:
                        #         # inside paren, append
                        #         paren_terms.append((search_parens[inner_loc[0] : inner_loc[1]]).replace('()', ''))
                        #         # remove them from the unit_str
                        #         search_parens = search_parens[:inner_loc[0]] + search_parens[inner_loc[1]:]

                        # # evaluate each set of terms in parens
                        # parens_eval = []
                        # for term in paren_terms:
                        #     # first check for exponents
                        #     raised_units = re.split(r'[^]', term)

                        # return cls(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))

                # return n Unit objects where n = the number of requested units
                if random.random() < 0.00001:
                    print(random.choice(_vandv_log))

                if len(to_return) > 1:
                    # multiple units requested -> return them all as a tuple
                    return tuple(to_return)
                else:
                    # one unit requested -> return as a single object
                    return to_return[0]

    def _get_unit_name(self, unit_str: str) -> tuple[Dimensions, np.array]:
        """Internal method to get power and scale arrays for a one-dimensional unit string."""
        # consolidate possible singular units
        single_units = self._base_units
        single_units.update(self._unit_dictionary)

        # check if base unit or scaled
        try:
            self._base_units[unit_str]
        except:
            # requested scaled unit
            potential_units = self._base_units
            potential_units.update(self._unit_dictionary)
            # isolate base unit first
            for key in potential_units:
                base = re.findall(key + r"\Z", unit_str)
                if base != []:
                    base = "".join(base)  # regex returns as a list, want as a string
                    break

            # verify that the requested singular unit exists
            if base not in potential_units.keys():
                # requested unit not in library
                bad_base_err = (
                    f"The requested unit [{base}] does not exist. Please verify that "
                    "you are using the correct metric unit."
                )
                raise ValueError(bad_base_err)

            # now isolate prefix
            prefix = unit_str[0 : (len(unit_str) - len(base))]

            # verify that requested singular unit is allowed to use prefixes
            if "prfx" in potential_units[base].keys():
                if potential_units[base]["prfx"] == False:
                    # requested unit incompatible with prefixes
                    no_prfxes_err = f"The requested unit [{base}] is incompatible with the given prefix."
                    raise ValueError(no_prfxes_err)

            # and check against the prefixes library
            try:
                self._prefixes[prefix]
            except:
                # not in the prefixes library -> raise error
                bad_pref_err = (
                    f"The requested metric prefix [{prefix}] does not exist. Please "
                    "verify that you are using the correct metric prefix."
                )
                raise ValueError(bad_pref_err)
            else:
                # prefix exists in library -> multiply base scale by prefix
                base_scales = np.array(potential_units[base]["scales"])
                scales_to_return = self._prefixes[prefix] * base_scales

                # return unit
                return (
                    Dimensions(np.array(self._base_units[base]["dimensions"])),
                    scales_to_return,
                )
        else:
            # requested non-scaled base unit
            # return it here
            return (
                Dimensions(np.array(self._base_units[unit_str]["dimensions"])),
                np.array(self._base_units[unit_str]["scales"]),
            )

    def _name_casting(self, in_power: npt.NDArray, in_scale: npt.NDArray) -> str:
        """Internal method for fetching named units or creating compound
        names for unnamed units.
        """
        # first check if in named units
        for unit_name, p in self._unit_dictionary.items():
            if all(p["dimensions"] == in_power) and all(p["scales"] == in_scale):
                return unit_name

        # not in named units dictionary -> try to created compound name
        try:
            numerator = []
            denominator = []
            for i, (power, scale) in enumerate(zip(in_power, in_scale)):
                if power != 0:
                    ## dimension exists -> generate name for this unit
                    # find prefix if not base unit already
                    prfx = ""
                    # undo scaling due to powers
                    prfx_scale = 10 ** ((math.log10(scale)) / abs(power))
                    if prfx_scale != 1:
                        # not a base unit -> find the base unit
                        prfx = list(self._prefixes.keys())[
                            list(self._prefixes.values()).index(prfx_scale)
                        ]

                    # find the base unit
                    base_unit = ["g", "m", "sec", "rad"][
                        i
                    ]  # picking from this list is faster and easier than using the class dictionary

                    # denote exponent if it exists (supports fractional powers)
                    power_abs = abs(power)
                    if power_abs != 1:
                        power_disp = (
                            int(power_abs)
                            if float(power_abs).is_integer()
                            else power_abs
                        )
                        raised_to = f"^{power_disp}"
                    else:
                        raised_to = ""

                    # build dimension's name and save to corresponding half of the fraction
                    if power > 0:
                        numerator.append(f"{prfx}{base_unit}{raised_to}")
                    else:
                        denominator.append(f"{prfx}{base_unit}{raised_to}")

            # combine each component of the unit's name into a single string and return
            if numerator and not denominator:
                # only numerator
                compound_name = "*".join(numerator)

            elif not numerator and denominator:
                # only denominator
                for i, unit in enumerate(denominator):
                    # add negative sign to exponent
                    as_list = list(unit)
                    as_list.insert(unit.index("^") + 1, "-")
                    denominator[i] = "".join(as_list)

                compound_name = "*".join(denominator)

            else:
                ## numerator and denominator
                # join everything into a single string for numerator and denominator
                num_str, den_str = "*".join(numerator), "*".join(denominator)

                # add parentheses for clarity to numerator and/or denominator if more than one unit
                if len(numerator) > 1:
                    num_str = f"({num_str})"

                if len(denominator) > 1:
                    den_str = f"({den_str})"

                # join as fraction
                compound_name = f"{num_str}/{den_str}"

            return compound_name

        except:
            # couldn' find unit name -> give basic info
            info = []
            # collect power and scale relative to base unit
            for power, scale, base in zip(in_power, in_scale, ["g", "m", "sec", "rad"]):
                if power != 0:
                    info.append(f"({scale:.2E} {base}^{power})")

            # consolidate and return
            return f'[{" ".join(info)}]'

    @classmethod
    def disp_named_units(cls) -> None:
        """Displays all named units, their scalings with respect to each dimensional
        power, dimension, and whether or not they may be mutated with metric prefixes.

        Includes base units.
        """
        # display header
        head_str = f'|{" " * 41}List of Named Units{" " * 42}|'
        print(f'|{"=" * (len(head_str) - 2)}|')
        print(head_str)
        print(f'|{"=" * (len(head_str) - 2)}|')

        ## loop through all named units and format their information accordingly
        # consolidate named units
        named_units = cls._base_units
        named_units.update(cls._unit_dictionary)

        names = [*named_units.keys()]

        # loop
        for i in range(0, len(names), 2):
            # get first column information
            first_name = f'Name      : "{names[i]}"'
            first_dims = (
                f'Dimensions: {Dimensions(named_units[names[i]]["dimensions"]).name}'
            )
            first_scls = f'Scaling   : {named_units[names[i]]["scales"]}'
            first_prfx = f'Prefixable: {named_units[names[i]]["prfx"]}'

            # second column information
            try:
                secnd_name = names[i + 1]
            except:
                # can't index i+1, set to nothing
                secnd_name = ""
                secnd_dims = ""
                secnd_scls = ""
                secnd_prfx = ""
            else:
                # can index i+1, set to values of i+1
                secnd_name = f'Name      : "{names[i + 1]}"'
                secnd_dims = f'Dimensions: {Dimensions(named_units[names[i + 1]]["dimensions"]).name}'
                secnd_scls = f'Scaling   : {named_units[names[i + 1]]["scales"]}'
                secnd_prfx = f'Prefixable: {named_units[names[i + 1]]["prfx"]}'

            # display column or columns accordingly
            print(
                f'| {first_name}{" " * (int(len(head_str) / 2) - len(first_name))}| '
                f'{secnd_name}{" " * (int(len(head_str) / 2) - len(secnd_name) - 5)}|'
            )
            print(
                f'| {first_dims}{" " * (int(len(head_str) / 2) - len(str(first_dims)))}| '
                f'{secnd_dims}{" " * (int(len(head_str) / 2) - len(str(secnd_dims)) - 5)}|'
            )
            print(
                f'| {first_scls}{" " * (int(len(head_str) / 2) - len(str(first_scls)))}| '
                f'{secnd_scls}{" " * (int(len(head_str) / 2) - len(str(secnd_scls)) - 5)}|'
            )
            print(
                f'| {first_prfx}{" " * (int(len(head_str) / 2) - len(str(first_prfx)))}| '
                f'{secnd_prfx}{" " * (int(len(head_str) / 2) - len(str(secnd_prfx)) - 5)}|'
            )
            if i < len(names) - 2:
                print(f'|{"-" * (len(head_str) - 2)}|')

        # close table
        print(f'|{"=" * (len(head_str) - 2)}|')


_vandv_log = [
    """
    ════════════════════════════════════════════════════════
          \\ | /
       '-.;;;.-'
      -==;;;;;==-      s c a r a b a e u s
       .-';;;'-.
         /|||\\

    The Zen of Orbit Determination
    (or: things the scarab understood before the estimator did)

    Measurement arc post-fit residuals should be small.
    Prediction residuals should make you comfortable.
    If they do not, start worrying.

    Orbit determination is fundamentally about prediction.
    The measurements simply exist to humble you.

    Radiometric tracking can measure a spacecraft from millions of kilometers away.
    Yet somehow we still lose an afternoon to a sign convention.

    Never run OD on an empty stomach.

    In the presence of process noise, use stochastics.
    In the absence of process noise, use stochastics anyway.
    The universe has never once propagated a perfect two body trajectory.

    Covariances should be neither optimistic nor catastrophic.
    Goldilocks would have made an excellent navigation engineer.
    
    State vectors should also remain reasonably small.
    The information matrix is not a storage unit.

    Consider parameters exist because we are humble.
    Or because we do not trust the force model.
    Same thing.

    The smoother always knows what you should have done yesterday.

    Multi-Arc OD is just admitting reality one discontinuity at a time.

    The scarab rolls the Sun across the sky every day.
    No one told it estimation theory was difficult.
    Neither should you.

    Congratulations, you hit the 0.001% event.
    Your filter calls it an outlier.
    We call it Tuesday.

    ════════════════════════════════════════════════════════
""",
    """
    ════════════════════════════════════════════════════════
      the scarab rolled the sun across the sky
      and your code still compiled.
      coincidence?

      (yes. but still impressive.)

      also: the scarabaeus beetle navigates at night
      using the Milky Way as a compass.
      you navigate using stack traces and print statements.
      you are basically the same.
    ════════════════════════════════════════════════════════
""",
]
