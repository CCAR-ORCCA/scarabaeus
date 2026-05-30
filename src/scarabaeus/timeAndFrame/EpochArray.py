# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Dimensions, Units, ArrayWUnits
import spiceypy as spice

import scarabaeus.utils.NumpyWrapper as np
import numpy
from numpy.typing import NDArray, ArrayLike

from typing import Literal, TypeAlias, get_args
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import re

# define units
sec, hr, day, unitless = Units.get_units(["sec", "hr", "day", "unitless"])

# class definition
class EpochArray:
    """Time interface within Scarabaeus.

    Performs conversions between time systems and representations.

    Parameters
    ----------
    values : str or float or array_like
        The time or times contained within the EpochArray. Valid input types are:

        * :obj:`str` - single epoch input. Numerical strings are cast as floats
          and assumed to be in seconds past J2000 (TDB). Non-numerical strings
          are parsed according to their format/representation (ISO, calendar,
          DOY, JD).
        * ``float``- single epoch input. Assumed to be in seconds past J2000 (TDB).
        * :obj:`list` or :class:`~numpy.ndarray` - Multi-epoch input. Element
          types follow the same conventions as their single epoch counterparts, see
          above bullets.

    system : TimeSystem, optional
        The uniform time system in which EpochArray is defined. Defaults to ``'TDB'``.
        Supported systems are:

        * ``'TDB'`` or ``'ET'`` - Barycentric Dynamical Time (TDB), also known as Ephemeris
          Time (ET).
        * ``'TT'`` or ``'TDT'`` - Terrestrial Time (TT), formerly referred to as Terrestrial
          Dynamical Time (TDT). Both refer to the same time system.
        * ``'TAI'`` - International Atomic Time (TAI).
        * ``'SCLK'`` - Spacecraft Clock Time. An SCLK kernel must be loaded and a spacecraft
          SPICE ID must be provided.
        * ``'GPS'`` - Global Position System (GPS) Time.

    rep : {'NUM', 'AWU', 'CAL', 'DOY', 'ISOC', 'ISOD', 'JDTDB', 'JED', 'JDTDT', 'JDUTC'}, optional
        The representation in which the EpochArray displays and returns :attr:`times`. The type
        of the returned time value or values depends on the selected representation, noted
        as "r_type" in the following bullets along with an example, "ex_ret", of the return.
        Defaults to ``'NUM'``. Supported representations, their resepective return types, and
        example returns are:

        * ``'NUM'`` - numerical representation in the EpochArray's time system. Only valid
          for the uniform time systems TDB, TAI, TT, GPS, JDTDB, JDTDT, and their synonym
          systems if they have one, e.g. TDB and ET. r_type = ``float``,
          ex_ret = ``833544069.1849113``.
        * ``'AWU'`` - numerical representation with associated units, r_type =
          :class:`~scarabaeus.units.ArrayWUnits` with units of seconds, ex_ret =
          ``833544069.1849113 sec``.
        * ``'CAL'`` - Calendar date. An LSK must be loaded for this representation,
          r_type = ``str``, ex_ret = ``'2026 JUN 12:00:00.000'``.
        * ``'DOY'`` - Day of year. An LSK must be loaded for this representation,
          r_type = ``str``, ex_ret = ``'2026-152 // 12:00:00.000'``.
        * ``'ISOC'`` - International Organization for Standardization (ISO) 8601,
          calendar, r_type = ``str``, ex_ret = ``'2026-06-01T12:00:00.000'``.
        * ``'ISOD'`` - International Organization for Standardization (ISO) 8601,
          day of year, r_type = ``str``, ex_ret = ``'2026-152T12:00:00.000'``.
        * ``'JDTDB'`` or ``'JED'`` - Julian Date relative to TDB (JDTB), also known
          as Julian Ephemeris Date (JED).
        * ``'JDTDT'`` - Julian Date relative to TDT (JDTDT).
        * ``'JDUTC'`` - DESC, r_type = ``str``, ex_ret = ``'JD '``.

    prec : int, optional
        Digits of precision in fractional seconds or days. Defaults to ``16``.

    disp_len : int, optional
        Number of digits/characters to display. Defaults to ``25``.

    sc_id : int, optional
        SPICE ID of an associated onboard clock, required for SCLK definitions.

    Raises
    ------
    MissingLSKError
        Raised if a time or times are defined in a non-TDB system and
        there is no recognized leapsecond kernel (LSK) furnished in
        the kernel pool. See the `SPICE LSK time system reading`_ for more.

    .. _SPICE LSK time system reading: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html#The%20Leapseconds%20Kernel%20LSK

    SCLKError
        Raised if:

        * a time or times are given in an SCLK representation when
          there is no recognized spacecraft clock time kernel (SCLK) furnished
          in the kernel pool. See the `SPICE SCLK time system reading`_ for more.
        * an epoch is requested with a time representation other than ``'NUM'`` or
          ``'AWU'`` as there is no conversion between spacecraft clock ticks and
          any of the other representations.

    .. _SPICE SCLK time system reading: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html#Spacecraft%20Clock%20SCLK

    SCIDError
        Raised if:

        * a time or times are given in SCLK but a spacecraft ID was not given.
        * a time or times are given in SCLK but the given spacecraft ID could not
          be located in the kernel pool.

        See the `SPICE SCLK conversions reading`_ for more.

    .. _SPICE SCLK conversions reading: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/sclk.html#Converting%20between%20SCLK%20strings%20and%20ET%20or%20UTC

    BadTimeError
        Raised if:

        * a time is given in an unrecognized format.
        * a time is given in a recognized format but the time itself is
            unrecognizable, e.g, ``'2026 FEB 50'``.

    References
    ----------
    .. [1] NASA/NAIF. (2021, December 23). SPICE Time Subsystem.
            Navigation and Ancillary Information Facility.
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html
    .. [2] NASA/NAIF. (2021, September 4). SCLK Required Reading.
            Navigation and Ancillary Information Facility.
            https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/sclk.html#SCLK%20kernels
    """

    # supported systems and representations
    TimeSystem: TypeAlias = Literal[
        "TDB", "ET", "UTC", "TT", "TDT", "TAI", "SCLK", "GPS"
    ]
    TimeRepresentation: TypeAlias = Literal[
        "NUM", "AWU", "CAL", "DOY", "ISOC", "ISOD", "JDTDB", "JED", "JDTDT", "JDUTC"
    ]

    # region Constructor
    def __init__(
        self,
        times: ArrayLike,
        sys: TimeSystem = "TDB",
        rep: TimeRepresentation = "AWU",
        prec: int = 16,
        disp_len: int = 25,
        sc_id: int = None,
    ):
        # need SC id for SCLK system validation
        self._sc_id = sc_id

        # times, system, and representation info
        self._validate_and_set_system(sys)
        self._validate_and_set_rep(rep)
        self._validate_and_set_to_et(times)

        # precision and display
        self._prec = prec
        self._disp_len = disp_len

        # shape info
        self._shape = self._times.shape
        self._size = self._times.size
        self._ndim = self._times.ndim

    # region Properties 
    @property
    def system(self) -> str:
        """The time system in which the EpochArray is defined."""
        return self._system

    def _validate_and_set_system(self, input_val) -> None:
        """Internal method to do input handling and set self._system."""
        # must be valid system
        if input_val not in get_args(self.TimeSystem):
            raise ValueError(f"Received unknown time system {input_val}")

        ## special cases
        # can't set system to SCLK without a spacecraft ID
        if input_val == "SCLK" and not self.sc_id:
            err_str = (
                "Cannot define the requested epoch in SCLK without an "
                "associated spacecraft ID."
            )
            raise type("SCIDError", (Exception,), {})(err_str)

        # good -> set attribute
        self._system = input_val

    @property
    def times(self) -> numpy.ndarray[float] | numpy.ndarray[str] | ArrayWUnits:
        """
        The numerical values of each epoch contained within the EpochArray,
        defined in the EpochArray's time :attr:`system` and formatted based
        on its :attr:`representation`.
        """
        # error string for repeated error during match
        sclk_err_str = (
            "Cannot convert from SCLK ticks to requested "
            f"representation {self._representation}."
        )

        # return based on format
        match self._representation:
            case "NUM":
                # no numerical representation for UTC
                if self._system == "UTC":
                    err_str = f"Cannot represent {self._system} epoch " "in JDTDB."
                    raise type("BadRepError", (Exception,), {})(err_str)
                # convert from internal TDB to defined system
                return self._times_in_sys(self._system)
            case "AWU":
                # no numerical representation for UTC
                if self._system == "UTC":
                    err_str = f"Cannot represent {self._system} epoch " "in JDTDB."
                    raise type("BadRepError", (Exception,), {})(err_str)
                ## convert from internal TDB to defined system and cast to AWU
                if self._system in {"TDB", "ET", "TT", "TDT", "TAI", "GPS"}:
                    # defined in seconds -> return as seconds
                    return ArrayWUnits(self._times_in_sys(self._system), sec)
                else:
                    # SCLK is defined in ticks -> return as unitless
                    return ArrayWUnits(self._times_in_sys(self._system), unitless)
            case "CAL":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                # otherwise convert to calendar and return
                return self._times_in_rep("C")
            case "DOY":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                # otherwise convert to DOY and return
                return self._times_in_rep("D")
            case "ISOC":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                # otherwise convert to ISO calendar and return
                return self._times_in_rep("ISOC")
            case "ISOD":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                # otherwise convert to ISO DOY and return
                return self._times_in_rep("ISOD")
            case "JDTDB":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                if self._system not in {"TDB", "ET"}:
                    # can only convert to JDTDB if in TDB system
                    err_str = (
                        f"Cannot represent {self._representation} epoch " "in JDTDB."
                    )
                    raise type("BadRepError", (Exception,), {})(err_str)

                # otherwise convert to JDTDB and return
                return self._times_in_rep("JDTDB")
            case "JED":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                if self._system not in {"TDB", "ET"}:
                    # can only convert to JED if in TDB system
                    err_str = f"Cannot represent {self._system} epoch " "in JED."
                    raise type("BadRepError", (Exception,), {})(err_str)

                # otherwise convert to JED (same as JDTDB) and return
                return self._times_in_rep("JED")
            case "JDTDT":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                if self._system not in {"TT", "TDT"}:
                    # can only convert to JDTDT if in TTTB system
                    err_str = f"Cannot represent {self._system} epoch " "in JDTDT."
                    raise type("BadRepError", (Exception,), {})(err_str)

                # otherwise convert to JDTDT and return
                return self._times_in_rep("JDTDT")
            case "JDUTC":
                if self._system == "SCLK":
                    # can't represent ticks in this format
                    raise type("SCLKError", (Exception,), {})(sclk_err_str)

                # otherwise convert to julian date UTC and return
                return self._times_in_rep("J")  # 'J' for et2utc = JDUTC

    def _validate_and_set_to_et(self, times):
        # if isinstance(times, (list, np.ndarray, tuple, str, float, int)):
        #     ## valid input -> check and condition
        #     if isinstance(times, (str, float, int)):
        #         # scalar input -> check and make np array for consistency
        #         self._times = np.array(self._condition_and_validate_time(times))
        #     else:
        #         # vector or matrix input -> force numpy and flatten
        #         times = np.asarray(times)
        #         to_check  = times.flatten()

        #         # check each element
        #         cond = np.array([self._condition_and_validate_time(t, i) for
        #                          i, t in enumerate(to_check)])

        #         # save with same input shape
        #         self._times = cond.reshape(times.shape)
        # else:
        #     # invalid input -> raise error
        #     raise TypeError('Received unsupported time input of type: '
        #                     f'{type(times)}.')

        scalar = isinstance(times, (str, float, int))
        items = [times] if scalar else numpy.asarray(times).flatten().tolist()

        et_vals = []
        for t in items:
            match t:
                case str():
                    match self._system:
                        case "TDB" | "ET":
                            # append TDB so spice knows its TDB
                            et_vals.append(spice.str2et(f"{t} TDB"))
                        case "UTC":
                            if self.representation == "NUM":
                                raise ValueError(
                                    "No numerical representation of UTC exists. "
                                    "Provide a different representation for UTC times."
                                )
                            # str2et already assumes UTC
                            et_vals.append(spice.str2et(t))
                        case _:
                            # can only parse TDB strings
                            raise TypeError(
                                f"String input not supported for {self._system}."
                            )
                case float() | int():
                    match self._system:
                        case "TDB" | "ET":
                            et_vals.append(float(t))
                        case "SCLK":
                            et_vals.append(spice.sct2e(self._sc_id, float(t)))
                        case _:
                            et_vals.append(spice.unitim(float(t), self._system, "TDB"))
                case _:
                    raise TypeError(f"Unsupported time input type: {type(t)}.")

        result = numpy.array(et_vals)

        if scalar:
            self._times = result[0]
        else:
            self._times = result.reshape(numpy.asarray(times).shape)

    def _times_in_rep(self, rep_str):
        """Internal method to convert times to the correct format."""
        et = self._times

        # scalar: 0-dimensional or single element
        is_scalar = et.ndim == 0 or et.size == 1

        if rep_str in {"JDTDB", "JED", "JDTDT"}:
            # unitim handles scalar only — loop for arrays
            if is_scalar:
                return spice.unitim(float(et), "TDB", rep_str)
            else:
                converted = numpy.zeros(et.shape).flatten()
                for i, t in enumerate(et.flatten()):
                    converted[i] = spice.unitim(float(t), "TDB", rep_str)
                return converted.reshape(et.shape)
        else:
            # et2utc handles lists
            if is_scalar:
                return spice.et2utc([float(et)], rep_str, self._prec, self._disp_len)[0]
            else:
                flat = et.flatten().tolist()
                result = spice.et2utc(flat, rep_str, self._prec, self._disp_len)
                return numpy.array(result).reshape(et.shape)

    def _times_in_sys(self, system):
        """Internal method to convert fom internal TDB to the actual time system."""
        if system in {"TDB", "ET", "UTC"}:
            # system already matches internal TDB -> bypass conversion
            # NOTE: UTC doesn't match, but it goes to et2utc the same as TDB or ET
            return self._times

        # system is different than internal TDB -> convert and return
        if system != "SCLK":
            # everything but SCLK can be converted with unitim
            if not self.shape:
                # given scalar input
                return spice.unitim(self._times.tolist(), "TDB", system)
            else:
                # given non scalar input -> need to loop since unitim only takes scalars
                converted = np.zeros(self.shape).flatten()
                for i, t in enumerate(self._times.flatten()):
                    converted[i] = spice.unitim(t, "TDB", system)

                return converted.reshape(self.shape)

        else:
            # use sce2c for SCLK
            if not self.shape:
                # given scalar input
                return spice.sce2c(self.sc_id, self._times.tolist())
            else:
                # given non scalar input -> need to loop since unitim only takes scalars
                converted = np.zeros(self.shape).flatten()
                for i, t in enumerate(self._times.flatten()):
                    converted[i] = spice.unitim(t, "TDB", system)

                return converted.reshape(self.shape)

    def _condition_and_validate_time(self, time, ind=None):
        """
        Internal method for validating and conditioning a time input.
        Ensures valid formatting of input and converts to TDB if
        necessary for internal tracking.
        """
        # convert to internal format of ET
        match time:
            case str():
                ## validate string and convert to et if good
                err = None  # save to raise SCB wrapped error if str2et fails
                try:
                    match self._system:
                        case "TDB" | "ET":
                            # append TDB/ET to string so str2et knows its TDB not UTC
                            et = spice.str2et(f"{time} TDB")
                        case "TT" | "TDT" | "TAI":
                            # these work with str2et directly
                            et = spice.str2et(time)
                        case "GPS":
                            # # parse as TDB and convert from GPS to TDB
                            # et = spice.unitim(spice.str2et(f'{time} TDB'),
                            #                   'GPS', 'TDB')
                            in_gps = spice.unitim(spice.str2et(time), "ET", "GPS")
                            et = spice.unitim(in_gps, "GPS", "TDB")
                        case "SCLK":
                            raise TypeError(
                                "Cannot parse time value string for "
                                "system SCLK. Please provide a float "
                                "or int."
                            )
                except Exception as e:
                    match str(e):
                        case s if "SPICE(NOLEAPSECONDS)" in s:
                            ## requested time needs a leap second kernel
                            err_str = (
                                "Requested time or times cannot be defined. "
                                "The necessary leapsecond kernel (LSK) could not "
                                "be located in the kernel pool. -- SPICE(NOLEAPSECONDS)"
                            )
                            err = type("MissingLSKError", (Exception,), {})(err_str)

                        case s if "SPICE(UNPARSEDTIME)" in s:
                            ## requested time is unrecognized
                            # get line from SPICE error that says why it failed
                            spice_line = [
                                line.strip() for line in s.split("\n") if line.strip()
                            ][3]

                            # get first character of bad string and bad substring
                            strs = re.search(r'<(.*?)>.*?"(.*?)"', spice_line)
                            if strs:
                                first, substr = strs.group(1), strs.group(2)
                                err_str = (
                                    "Requested time or times cannot be parsed. "
                                    f"Substring beginning at <{first}> is not recognized in: "
                                    f'"{substr}" -- SPICE(UNPARSEDTIME)'
                                )
                                err = type("BadTimeError", (Exception,), {})(err_str)
                            else:
                                # different UNPARSEDTIME error -> just raise it directly
                                raise e

                        case s if "SPICE(BADTIMESTRING)" in s:
                            ## requested time doesn't make sense
                            # get line from SPICE error that says why it failed
                            spice_line = [
                                line.strip() for line in s.split("\n") if line.strip()
                            ][3]

                            # get requested month
                            mnth = re.search(r"month of (\w+)", spice_line).group(1)

                            ## get requested day and the month's day bounds
                            # isolate all numbers in the line
                            p = r"[-+]?\d*\.\d+([eEdD][-+]?\d+)?"

                            # convert from scientific notation to integers
                            nums = [
                                int(float(m.group(0).replace("D", "E")))
                                for m in re.finditer(p, spice_line)
                            ]

                            # requested day is first, min is second, max is third
                            err_str = (
                                f"Provided date {nums[0]} {mnth} does not exist within "
                                f"the specified month's interval of [{nums[1]} {mnth}, "
                                f"{nums[2]} {mnth}]. -- SPICE(BADTIMESTRING)"
                            )
                            err = type("BadTimeError", (Exception,), {})(err_str)

                # raise if failed, return if passed
                if err:
                    raise err
                else:
                    return et

            case float():
                match self._system:
                    case "TDB" | "ET":
                        # already TDB -> good to return
                        return time
                    case "SCLK":
                        # convert from SCLK to TDB and return
                        return spice.sct2e(self._sc_id, time)
                    case _:
                        # convert from anything else to TDB
                        return spice.unitim(time, self._system, "TDB")

            case _:
                if ind:
                    # given index -> element of time vector, note index as well
                    raise TypeError(
                        "Received unsupported time input of type: "
                        f"{type(time)} at index {ind}."
                    )
                else:
                    # no index -> scalar time, don't note location
                    raise TypeError(
                        "Received unsupported time input of type: " f"{type(time)}."
                    )

    @property
    def representation(self) -> str:
        """
        The time representation in which the EpochArray's times are
        formatted.

        The representation determines the format in which the EpochArray is
        displayed as well as the type of object returned by its :attr:`times`.
        """
        return self._representation

    def _validate_and_set_rep(self, rep) -> None:
        """Set the time representation of the EpochArray."""
        ## input handling
        match rep:
            case str():
                # must be a valid representation
                if rep not in set(get_args(self.TimeRepresentation)):
                    err_str = f"Unrecognized time representation {rep}."
                    raise type("BadRepError", (Exception,), {})(err_str)
            case _:
                # must be a string
                raise TypeError(
                    "Received unsupported representation input " f"of type {type(rep)}."
                )

        # make sure LSK is loaded for representations that need it
        if rep in {"CAL", "DOY", "ISOC", "ISOD", "JDTDB", "JED", "JDTDT", "JDUTC"}:
            try:
                spice.et2utc(0.0, "C", 30)  # dummy et to check for LSK
            except Exception as e:
                if "SPICE(MISSINGTIMEINFO)" in str(e):
                    # no loaded LSK
                    err_str = (
                        f"Cannot set requested time representation {rep}. "
                        "The necessary leapsecond kernel (LSK) could not "
                        "be located in the kernel pool. -- SPICE(MISSINGTIMEINFO)"
                    )
                    raise type("MissingLSKError", (Exception,), {})(err_str)
                else:
                    raise e

        # all good -> save to self
        self._representation = rep

    @property
    def prec(self) -> int:
        """
        DESC
        """
        return self._prec

    @property
    def disp_len(self) -> int:
        """
        DESC
        """
        return self._disp_len

    @property
    def sc_id(self) -> int:
        """
        DESC
        """
        return self._sc_id

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the EpochArray
        """
        return self._shape

    @property
    def size(self) -> int:
        """
        The size of the EpochArray.
        """
        return self._size

    # endregion Properties

    # region Operators
    def __repr__(self) -> str:
        # use times getter instead of internal times so format matches
        return f"{self.times} ({self._system})"

    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, key) -> Self:
        """
        Get a sub-array in the EpochArray object.
        """
        # can't index 0 dimensional array
        if self._ndim == 0:
            raise IndexError("too many indices for 0 dimensional EpochArray")

        # otherwise return index
        return EpochArray._from_et(
            self._times[key],
            self._system,
            self._representation,
            self._prec,
            self._disp_len,
            self._sc_id,
        )

    def __setitem__(self, key) -> None:
        """
        Set a sub-array in the EpochArray object.
        """
        raise Exception(
            "__setitem__ currently not supported for EpochArray. Index .times directly for now."
        )

    @classmethod
    def _from_et(cls, tdb, system, rep, prec, disp_len, sc_id) -> Self:
        """Construct directly from internal TDB to skip normal checks."""
        obj = cls.__new__(cls)
        obj._sc_id = sc_id
        obj._system = system
        obj._times = numpy.asarray(tdb)
        obj._representation = rep
        obj._prec = prec
        obj._disp_len = disp_len
        obj._shape = obj._times.shape
        obj._size = obj._times.size
        obj._ndim = obj._times.ndim
        return obj

    def __add__(self, other: ArrayWUnits) -> Self:
        """
        Overloading the addition operator of an EpochArray object.

        One valid case:
        - EpochArray + AWU w/ dim time = EpochArray

        This is specified in the Notes section of the main docstring.
        """
        ## input validation
        match other:
            case ArrayWUnits():
                ## adding AWU -> more validation
                # can only add a time valued awu
                if other.units.dimensions.name != "Time":
                    raise ValueError(
                        "Cannot add an ArrayWUnits with "
                        "units of dimension "
                        f"{other.units.dimensions.name}"
                    )

                ## add in interal TDB
                # force seconds
                other = other.convert_to(sec)

                # add and return as new EpochArray
                new_tdb = self._times + other.values
                # return EpochArray(new_tdb, 'TDB',
                #                     self._representation, self._prec,
                #                     self._disp_len, self._sc_id)
                return self._from_et(
                    new_tdb,
                    self._system,
                    self._representation,
                    self._prec,
                    self._disp_len,
                    self._sc_id,
                )
            case _:
                # can only add an awu
                raise TypeError(
                    "unsupported operand type(s) for +:"
                    f"'{type(self)}' and '{type(other)}'"
                )

    def __sub__(self, other: ArrayWUnits | Self) -> ArrayWUnits | Self:
        """
        Overloading the subtraction operator of an EpochArray object.

        Two valid cases:
        - EpochArray - AWU w/ dim time = EpochArray
        - EpochArray - EpochArray      = AWU w/ dim time

        This is specified in the Notes section of the main docstring.
        """
        ## input validation
        match other:
            case EpochArray():
                ## subtracting self - EpochArray -> more validation
                # can only subtract values in the same time system
                tdb_same, tt_same = {"TDB", "ET"}, {"TT", "TDT"}  # synonym systems
                if not (
                    (self._system in tdb_same and other.system in tdb_same)
                    or (self._system in tt_same and other.system in tt_same)
                    or (self._system == other.system)
                ):
                    raise ValueError(
                        "Cannot subtract epochs defined in "
                        "different time systems: "
                        f"{self.system} - {other.system} "
                        "is not valid."
                    )

                # subtract in internal TDB values and return as AWU
                new_vals = self._times - other._times
                return ArrayWUnits(new_vals, sec)

            case ArrayWUnits():
                ## subtracting self - AWU -> more validation
                # can only subtract a time valued awu
                if other.units.dimensions.name != "Time":
                    raise ValueError(
                        "Cannot subtract an ArrayWUnits with "
                        "units of dimension "
                        f"{other.units.dimensions.name}"
                    )

                ## subtract in interal TDB
                # force seconds
                other = other.convert_to(sec)

                # subtract and return as new EpochArray
                new_tdb = self._times - other.values
                # return EpochArray(new_tdb, 'TDB',
                #                     self._representation, self._prec,
                #                     self._disp_len, self._sc_id)
                return self._from_et(
                    new_tdb,
                    self._system,
                    self._representation,
                    self._prec,
                    self._disp_len,
                    self._sc_id,
                )
            case _:
                # can only subtract another epocharray or an awu
                raise TypeError(
                    "unsupported operand type(s) for -:"
                    f"'{type(self)}' and '{type(other)}'"
                )

    def __mul__(self, other):
        """
        Overloading the mutiplication operator of an EpochArray object.
        """
        # can't do this math -> raise error
        no_math_err = "Multiplication is not defined for " "EpochArray objects."
        raise ArithmeticError(no_math_err)

    def __rmul__(self, other):
        """
        Overloading the reverse order multiplication operator of an EpochArray object.
        """
        # can't do this math -> raise error
        no_math_err = (
            "Reverse order multiplication is not defined for " "EpochArray objects."
        )
        raise ArithmeticError(no_math_err)

    def __truediv__(self, other):
        """
        Overloading the division operator of an EpochArray object.
        """
        # can't do this math -> raise error
        no_math_err = "True division is not defined for " "EpochArray objects."
        raise ArithmeticError(no_math_err)

    def __rtruediv__(self, other):
        """
        Overloading the reverse order division operator of an EpochArray object.
        """
        # can't do this math -> raise error
        no_math_err = (
            "Reverse order true division is not defined for " "EpochArray objects."
        )
        raise ArithmeticError(no_math_err)

    def __pow__(self, other):
        """
        Overloading the power operator of an EpochArray object.
        """
        # can't do this math -> raise error
        no_math_err = "Exponentiation is not defined for " "EpochArray objects."
        raise ArithmeticError(no_math_err)

    def __eq__(self, other):
        """
        Overloading the equality operator of an EpochArray object.
        """
        # ------------------#
        # input validation #
        # ------------------#
        # equating to None -> not equal
        if other == None:
            return False

        # must be another EpochArray
        if not isinstance(other, self.__class__):
            not_ea_err = (
                "Equality is not defined for EpochArray objects "
                "between non-EpochArray objects. Received: "
                f"{type(other)}."
            )
            raise TypeError(not_ea_err)

        # -------------------#
        # perform operation #
        # -------------------#
        if not np.array_equal(self.times, other.times):
            # one or more times don't match -> return false
            return False

        elif self.system != other.system:
            # time frames not the same      -> return false
            return False

        else:
            # times and frames equal        -> return true
            return True

    def __ne__(self, other):
        """
        Overloading the not-equals operator of an EpochArray object.
        """
        # opposite of equality
        return not self.__eq__(other)

    # endregion Operators

    # region Methods
    def to(self, sys: TimeSystem = None, rep: TimeRepresentation = None) -> Self:
        """Convert to the given time system and representation.

        Parameters
        ----------
        sys : TimeSystem, optional
            {sys_desc}
        rep : TimeRepresentation, optional
            DESC
        """
        # ## input handling
        # # can't convert if not given anything
        # if not sys and not rep:
        #     raise ValueError('Received neither a new time system nor a '
        #                      'new representation to convert to. Please '
        #                      'provide atleast one of the two.')

        # ## conversion
        # # convert between systems
        # if sys:
        #     ## given new system -> convert
        #     # ensure float is in TDB
        #     # if sys and sys not in {'TDB', 'ET'}:
        #     #     print(f'NOT TDB, is {self._system}')
        #     times = self._times_in_sys(sys)
        # else:
        #     # not given new system -> keep current
        #     times = self._times
        #     sys = self._system

        # # return new version
        # if not rep: rep = self._representation # keep current representation if not given new one
        # return EpochArray(self._times, sys, rep, self._prec,
        #                     self._disp_len, self._sc_id)
        if not sys and not rep:
            raise ValueError(
                "Received neither a new time system nor a "
                "new representation to convert to. Please "
                "provide atleast one of the two."
            )

        if not sys:
            sys = self._system
        if not rep:
            rep = self._representation

        return EpochArray._from_et(
            self._times, sys, rep, self._prec, self._disp_len, self._sc_id
        )

    def duration(self, units: Units | str = "sec"):
        """Find the length of time spanned by the EpochArray.

        Singular (scalar) epochs and non-chronological arrays are
        undefined.

        Parameters
        ----------
        units : :class:`~scarabaeus.units.Units` of time or str, optional
            The time units to return the duration in. Defaults to seconds.

        Returns
        -------
        dur : :class:`~scarabaeus.units.ArrayWUnits`
            The duration of time spanned by the EpochArray in the requested
            units, which are seconds by default.

        Raises
        ------
        InvalidDurError
            Raise if:

            * the EpochArray contains a single epoch (scalar).
            * the EpochArray does not contain chronological epochs, i.e.
              it doesn't contain strictly increasing times in TDB.
        """
        ## make sure duration can be calculated for times values
        if self.size == 1:
            # can't find duration of scalar epoch
            err_str = "Cannot define duration for scalar EpochArray."
            raise type("InvalidDurError", (Exception,), {})(err_str)

        # if chronological, all times in TDB will be increasing
        if not np.all(np.diff(self._times) > 0):
            # epoch array is not chronological
            err_str = "Cannot define duration for non-chronological EpochArray."
            raise type("InvalidDurError", (Exception,), {})(err_str)

        ## handle units input
        # parse
        match units:
            case Units():
                # already a unit -> keep going
                pass
            case str():
                # create unit if passed as string
                units = Units.get_units(units)
            case _:
                # raise error if not unit or string
                raise TypeError(f"Received invalid units of type {type(units)}.")

        # make sure units of time
        if units.dimensions != Dimensions([0, 0, 1, 0]):
            raise ValueError(
                "Cannot convert compute duration in units with "
                f"dimensions of {units.dimensions.name}."
            )

        ## return in requested units
        # compute duration with internal ephemeris seconds
        dur_sec = ArrayWUnits(self._times[-1] - self._times[0], sec)

        # convert to units if not seconds
        if units != sec:
            return dur_sec.convert_to(units)
        else:
            return dur_sec

    def enforce_precision(self, precision: int = 13):
        """
        Return a high-precision string representation of this epoch.

        Formats the internal time value as a fixed-length decimal string with
        *precision* digits after the decimal point, padding or truncating as
        needed. Useful when interfacing with SPICE routines that require
        precise numeric strings.

        Parameters
        ----------
        precision : int, optional
        """
        # need to convert to double to count places before and after decimal point
        if isinstance(self.times, str):
            # string input, convert to longdouble
            time_double = np.longdouble(self.times)
        elif isinstance(self.times, np.float128):
            # good data type, assign and keep moving
            time_double = self.times
        else:
            raise ValueError(
                "Time must be string or long double type in order to enforce high precision."
            )

        string_time = str(time_double)
        time_integer = str(round(time_double))  # separate integer from decimal
        len_integer = len(time_integer)  # get the length of the integer time

        string_decimal = string_time[len_integer:]

        if len(string_decimal) > precision:
            string_decimal = str(np.round(np.longdouble(string_decimal)))
            string_decimal = string_decimal[1:]

        while len(string_decimal) < precision:
            string_decimal += "0"

        precise_time = str(time_integer) + string_decimal  # output is string

        return precise_time

    IntervalBound: TypeAlias = Self | float | str

    @staticmethod
    def interval(
        start: IntervalBound,
        end: IntervalBound,
        dt: ArrayWUnits = None,
        n_epochs: int = None,
        sys: TimeSystem = None,
        rep: TimeRepresentation = "AWU",
    ):
        """Create epochs across an interval with given time step and time system.

        Supports two different forms of interval definition:

        * step definition - create evenly stepped epochs along the half open
          interval [start, end), defined by the step size ``dt``.
        * count definition - create evenly spaced epochs along the closed
          interval [start, end], defined by the total number of epochs
          ``n_epochs``.

        Cannot define both a ``dt`` and an ``n_epochs`` as they are mutually
        exclusive and define different interval construction processes.

        Parameters
        ----------
        start : IntervalBound
            The starting epoch, defined as one of the following types:

            * :class:`~scarabaeus.timeAndFrame.EpochArray` - Defined in
              its system and representation.
            * ``float`` - Assumed to be given in TDB unless another
              system is provided.
            * ``str`` - Numerical strings are cast as floats and assumed to be
              in seconds past J2000 (TDB). Non-numerical strings are parsed
              according to their format/representation (ISO, calendar, DOY, JD).

        end : IntervalBound
            The ending epoch, defined as one of the following types:

            * :class:`~scarabaeus.timeAndFrame.EpochArray` - Defined in
              its system and representation.
            * ``float`` - Assumed to be given in TDB unless another
              system is provided.
            * ``str`` - Numerical strings are cast as floats and assumed to be
              in seconds past J2000 (TDB). Non-numerical strings are parsed
              according to their format/representation (ISO, calendar, DOY, JD).

        dt : ArrayWUnits, optional
            The amount of time to space each epoch by along the duration of the
            interval. Mutually exclusive with ``num_epochs``. Defaults to ``None``.

        n_epochs : int, optional
            The number of total epochs to place evenly within the interval. Mutually
            exclusive with ``dt``. Defaults to ``None``.

        sys : TimeSystem, optional
            The time system in which the epochs are defined. If ``start``
            is of type :class:`~scarabaeus.timeAndFrame.EpochArray`, its
            time system will be used unless a different system is provided. If ``start``
            is not an :class:`~scarabaeus.timeAndFrame.EpochArray` but ``end`` is,
            its time system will be used unless a different system is provided. If
            neither ``start`` nor ``end`` are :class:`~scarabaeus.timeAndFrame.EpochArray`
            and the system is not defined, an error will be raised.

            See :class:`~scarabaeus.timeAndFrame.EpochArray` for all valid time
            systems.

        rep : TimeRepresentation, optional
            The time representation in which the epochs are defined. If ``start``
            is of type :class:`~scarabaeus.timeAndFrame.EpochArray`, its
            representation will be used unless a different one is provided. If ``start``
            is not an :class:`~scarabaeus.timeAndFrame.EpochArray` but ``end`` is,
            its representation will be used unless a different one is provided. If
            neither ``start`` nor ``end`` are :class:`~scarabaeus.timeAndFrame.EpochArray`,
            the representation is assumed to be ``'NUM'``.

            See :class:`~scarabaeus.timeAndFrame.EpochArray` for all valid time
            representations.

        Returns
        -------
        time_interval : EpochArray
            The interval of epochs.

        Examples
        --------
        Create interval using ``str`` and ``float`` type interval bounds:

        .. code-block:: python
            import scarabaeus as scb
            hr = scb.Units.get_units('hr')
            epochs = scb.EpochArray.interval(start = '2026 JUN 01 00:00:00.000',
                                             end   = 574538961669.1835, # 1 day later in TDB
                                             dt    = scb.ArrayWUnits(1, hr))

        Create interval using :class:`~scarabaeus.timeAndFrame.EpochArray`
        type interval bounds:

        .. code-block:: python
            import scarabaeus as scb
            hr = scb.Units.get_units('hr')

            t0 = scb.EpochArray('2026 JUN 01 00:00:00.000')
            tf = scb.EpochArray('2026 JUN 02 00:00:00.000')

            epochs = scb.EpochArray.interval(start = t0, end = tf, dt = scb.ArrayWUnits(1, hr))
        """
        ## input handling
        # correct types for inputs (start and stop checks in system checking later)
        if not isinstance(dt, ArrayWUnits | None):
            raise TypeError(f"Received invalid dt value of type {type(dt)}.")

        if not isinstance(n_epochs, int | None):
            raise TypeError(
                f"Received invalid n_epochs value of type {type(n_epochs)}."
            )

        # can't use both dt and n_epochs
        if dt and n_epochs:
            raise ValueError(
                "Cannot define interval in terms of both dt and n_epochs. "
                "Provide only dt or n_epochs."
            )

        # can't use negative or 0 dt
        if dt and dt.values <= 0.0:
            raise ValueError("Cannot define interval with time step <= 0.")

        # can't use negative or 0 n_epochs
        if n_epochs and n_epochs <= 0:
            raise ValueError("Cannot define interval with 0 or less epochs.")

        ## figure out system
        found_sys = False
        # first check if start has a system defined
        match start:
            case EpochArray():
                # start has a system -> use that one
                sys = start.system
                found_sys = True
            case float():
                pass  # <- valid
            case str():
                pass  # <- valid
            case _:
                raise TypeError(f"Received invalid start value of type {type(start)}.")

        # next check end
        match end:
            case EpochArray():
                # end has a system -> use if start didn't have one
                if not found_sys:
                    sys = end.system
                    found_sys = True
            case float():
                pass  # <- valid
            case str():
                pass  # <- valid
            case _:
                raise TypeError(f"Received invalid end value of type {type(start)}.")

        # finally, use the given system if it exists
        if not sys:
            raise ValueError(
                "Neither start nor end contain time system information and "
                "no system was given. Provide a time system or define "
                "one of the interval bounds as an EpochArray."
            )

        ## convert start and end if neccessary and get in TDB
        if not isinstance(start, EpochArray):
            start = EpochArray(start, sys="TDB", rep="NUM")

        if not isinstance(end, EpochArray):
            end = EpochArray(end, sys="TDB", rep="NUM")

        ## create interval depending on construction type
        if dt:
            # step definiton
            interval = np.arange(start.times, end.times, dt.convert_to(sec).values)
        else:
            # count definition
            interval = np.linspace(start.times, end.times, n_epochs)

        ## convert to requested system and representation and return
        return EpochArray(interval, sys=sys, rep=rep)


    @staticmethod
    def _split_fractional_part(t: float):
        """
        Function that split a float value into integer and fractional components.
        If time XXXX.YYYY is given as input, this function returns XXXX and YYYY.
        Used for precise ephemeris retrivial with ad hoc interpolation.

        Parameters
        ----------
        t : float
            Time as a float (e.g. XXXX.YYYY = 12321412.2138121)

        Returns
        -------
        t_i : float
            Integer part of the input time (e.g. XXXX = 12321412)

        t_f : float
            Fractional part of the input time (e.g. YYYY = 2138121)

        Notes
        -----
        See also the method SpiceManager.get_state_precise() which needs integer and fractional time for precise
        ephemeris retrivial
        """
        # Split between integer (t_i) and fractional (t_f) parts
        t_i = np.floor(t)
        t_f = t - t_i
        return t_i, t_f
