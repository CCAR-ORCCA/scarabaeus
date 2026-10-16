# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    Dimensions,
    Units,
    Frame,
    EpochArray,
    ArrayWUnits,
    ArrayWFrame,
    Body,
)
from scarabaeus.utils.SCBPolynomial import SCBPolynomial

import warnings
import os, pathlib
import json
import pickle
import spiceypy as spice
import scarabaeus.utils.NumpyWrapper as np
import numpy as _np
from numpy.typing import NDArray
import math
from functools import lru_cache

# ------------------#
#  Generate Units  #
# ------------------#
from scarabaeus import Units

km, sec, rad = Units.get_units(["km", "sec", "rad"])


# -----------------------#
#    Class Definition   #
# -----------------------#
class SpiceManager:
    """
    Unified interface to NASA SPICE via SpiceyPy.

    Provides static utility methods for loading/unloading SPICE kernels,
    querying body states and positions, converting between time systems,
    resolving body and frame identifiers, and computing geometric quantities
    (light time, ground-station visibility, etc.).

    All methods that return physical quantities wrap their output in
    :class:`~scarabaeus.ArrayWUnits` or :class:`~scarabaeus.ArrayWFrame`
    to carry units and frames explicitly.

    Notes
    -----
    ``SpiceManager`` is a pure-static class; it holds no instance state.
    Configuration is done through class-level attributes:

    * ``kernel_list`` — list of currently loaded kernel paths.
    * ``kernel_folder`` — default folder scanned for scenario kernels.
    * ``poly_interp_deg`` — polynomial degree for state interpolation.

    References
    ----------
    .. [1] Annex et al., (2020). SpiceyPy: a Pythonic Wrapper for the SPICE
       Toolkit. *Journal of Open Source Software*, 5(46), 2050.
       https://doi.org/10.21105/joss.02050

    .. [2] Acton, C.H.; "Ancillary Data Services of NASA's Navigation and
       Ancillary Information Facility;" *Planetary and Space Science*,
       Vol. 44, No. 1, pp. 65–70, 1996.
       DOI 10.1016/0032-0633(95)00107-7

    .. [3] Acton, C.; Bachman, N.; Semenov, B.; Wright, E.; "A look toward
       the future in the handling of space science mission geometry;"
       *Planetary and Space Science* (2017).
       DOI 10.1016/j.pss.2017.02.013
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class TooFewStatesError(ValueError):
        """Raised when a segment has fewer states than its interpolation degree needs."""

    class EvenDegreeError(ValueError):
        """Raised when an even interpolation degree is given for a type-13 segment,
        whose Hermite polynomials are built from a position and a velocity per node
        and so always have odd degree."""

    class SidecarNotFoundError(ValueError):
        """Raised when no sidecar for the requested body, origin, and frame is
        found in any of the locations searched."""

    class STMLayoutError(ValueError):
        """Raised when the STMs given for a sequence's STM sidecar line up
        neither with its legs, one block per leg on that leg's epoch grid, nor
        with the global epoch grid, one matrix per epoch."""

    # endregion Errors #
    # ------------------#

    _spc_pos_units = km
    _spc_time_units = sec
    _spc_angle_units = rad
    _spc_vel_units = km / sec

    kernel_list = []
    kernel_folder = None  # None → computed from os.getcwd() at call time
    poly_interp_deg = 3

    # Room for the coverage window read back from an SPK before appending to it.
    # A window holds two doubles per interval, and a trajectory written leg by
    # leg leaves one interval per leg, so this allows 5000 of them.
    _spk_coverage_cell_size = 10000
    poly_interp_par_deg = 1

    # ----------------------------#
    # region Internal Methods    #
    # ----------------------------#
    def _validate_epoch_input(input_epoch) -> float | int:
        """
        Validate and extract a scalar epoch value from flexible input types.

        Accepts either an :class:`~scarabaeus.EpochArray` containing exactly
        one epoch, or a bare ``float`` / ``int``.  Used internally to reduce
        code repetition across SPICE-wrapper methods.

        Parameters
        ----------
        input_epoch : EpochArray or float or int
            The epoch to validate.  ``EpochArray`` inputs must contain exactly
            one value; their numeric content is extracted and returned.
            ``float`` / ``int`` inputs are returned unchanged.

        Returns
        -------
        epoch : float or int
            The scalar epoch value in TDB seconds past J2000.

        Raises
        ------
        ValueError
            If *input_epoch* is an EpochArray with more than one entry.
        TypeError
            If *input_epoch* is not an EpochArray, float, or int.
        """
        if isinstance(input_epoch, EpochArray):
            # given an epocharray -> make sure singular value
            if input_epoch.size != 1:
                bad_sze_err = (
                    "Argument [epoch_time] must be a single value. "
                    f"Received: {input_epoch}."
                )
                raise ValueError(bad_sze_err)
            else:
                # single value -> extract numerical data in second
                input_epoch = input_epoch.to(sys="TDB", rep="NUM")
                input_epoch = input_epoch.times

        elif not (isinstance(input_epoch, float) or isinstance(input_epoch, int)):
            # anything besides an awu or float is invalid
            # if it's a float don't need to do any conditioning
            bad_type_err = (
                "Argument [epoch_time] must be an object of type "
                f"EpochArray, int, or float. Received: {type(input_epoch)}."
            )
            raise TypeError(bad_type_err)

        return input_epoch

    # endregion Internal Methods #
    # ----------------------------#

    # --------------------------#
    # region Kernel Methods    #
    # --------------------------#
    @classmethod
    def print_kernels(cls):
        """Print the source and type of every currently loaded SPICE kernel.

        Iterates over all kernels tracked by ``spice.ktotal`` and prints
        each entry's file path and kernel type to stdout.

        Returns
        -------
        None
        """
        # Get all kernels
        count = spice.ktotal("all")

        # First print "Kernels Loaded" header
        print(("=" * 80) + "\n" + (" " * 33) + "Kernels Loaded:\n" + ("=" * 80))

        # Now print list of loaded kernels
        for i in range(count):
            print(
                "Source:   "
                + spice.kdata(i, "all")[0]
                + "   ("
                + spice.kdata(i, "all")[1]
                + ")"
            )

    @classmethod
    def ckbrief(cls, ck_file: str, tol: int = 0, disp: bool = False) -> list[dict]:
        """
        Extracts ID and interval information from a given C-kernel.

        Parameters
        ----------
        ck_file : str
            The path to the C-kernel file to be examined.

        disp : bool, optional
            Flag to include a formatted print-out of the brief. Defaults to ``False``.

        Returns
        -------
        brief : list of dict
            A list containing information about each frame in the C-kernel. Each element
            of the list is a dict containing the following keys:

            * ``'ID'`` : the NAIF integer code of the frame.
            * ``'SCLK_INTERVAL'`` : the interval or intervals for which the frame is
              defined in the C-kernel, represented as encoded SCLK ticks by floats.
            * ``'TDB_INTERVAL'`` : the interval or intervals for which the frame is
              defined in the C-kernel, represented as ephemeris time (TDB) EpochArray
              objects.
        """
        ids = spice.ckobj(ck_file)
        brief = []
        for id in ids:
            # get encoded SCLK coverage
            sclk_cov = spice.ckcov(ck_file, id, False, "SEGMENT", tol, "SCLK")
            sclk_interval = []
            for i in range(0, len(sclk_cov), 2):
                sclk_interval.append((sclk_cov[i], sclk_cov[i + 1]))

            # and TDB coverage
            tdb_cov = spice.ckcov(ck_file, id, False, "SEGMENT", tol, "TDB")
            tdb_interval = []
            for i in range(0, len(tdb_cov), 2):
                tdb_interval.append(
                    (EpochArray(tdb_cov[i], "TDB"), EpochArray(tdb_cov[i + 1], "TDB"))
                )

            # collect as dict and save to brief
            brief.append(
                {"ID": id, "SCLK_INTERVAL": sclk_interval, "TDB_INTERVAL": tdb_interval}
            )

        # display if requested
        if disp:
            title = f"Brief {pathlib.Path(ck_file).name}"
            bars = "=" * (len(title) + 15)
            print(f"{bars}\n{title.center(len(bars))}\n{bars}")
            for entry in brief:
                # print the ID
                print(f'ID: {entry["ID"]}')
                # its intervals in SCLK
                for interval in entry["SCLK_INTERVAL"]:
                    print(f"Interval (SCLK): {interval[0]} - {interval[1]}")
                # and in TDB
                for interval in entry["TDB_INTERVAL"]:
                    print(f"Interval (TDB): {interval[0]} - {interval[1]}")
                # spacing
                print()

        # return brief
        return brief

    @classmethod
    def clear_kernels(cls):
        """Unload all SPICE kernels from the kernel pool.

        Calls ``spice.kclear()`` to flush every kernel currently held in
        memory.

        Returns
        -------
        None
        """
        spice.kclear()

    # endregion Kernel Methods #
    # --------------------------#

    # ------------------------------#
    # region Trajectory Methods    #
    # ------------------------------#
    @classmethod
    def get_pos(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        ab_correct: str = "None",
    ):
        """
        Get the position of a target body relative to an observer body at a given epoch.
        This method computes the position vector of a target body as observed from an
        observer body at a specified epoch time, in a given reference frame.

        Parameters
        ----------
        trgt_bdy : str | Body
            The target body whose position is to be computed.
            Can be specified as a string (body name) or a Body object.
        epoch_time : float | EpochArray
            The epoch time(s) at which to compute
            the position. Can be a single float value or an EpochArray for
            multiple time points.
        reference_frame : str | Frame
            The reference frame in which to express
            the position vector. Can be specified as a string (frame name) or
            a Frame object.
        obsvr_bdy : str | Body
            The observer body from which the target position
            is measured. Can be specified as a string (body name) or a Body object.
        ab_correct : str, optional
            Aberration correction specification.
            Defaults to "None". Common values include "NONE", "LT", "LT+S", "CN",
            "CN+S", "XLT", "XLT+S", "XCN", "XCN+S".

        Returns
        -------
        The position vector(s) of the target body relative to the observer body.
        The exact return type depends on the implementation of _get_pos_core.

        Notes
        -----
        This method is a wrapper around _get_pos_core and delegates the actual
        computation to that core method.
        """

        pos = cls._get_pos_core(
            trgt_bdy, epoch_time, reference_frame, obsvr_bdy, ab_correct
        )

        return pos

    @classmethod
    def get_pos_values(
        cls, trgt_bdy, epoch_time, reference_frame, obsvr_bdy, ab_correct="None"
    ):
        """
        Raw-array variant of :meth:`get_pos` for hot inner loops.

        Returns the position as a plain numpy array in kilometers, skipping the
        ``ArrayWUnits`` wrap/unwrap that :meth:`get_pos` performs. Numerically
        identical to ``get_pos(...).values``; use this where the caller only needs
        the values (e.g. force-model acceleration during propagation).

        Parameters
        ----------
        trgt_bdy, obsvr_bdy : str or Body
            Target and observer bodies.
        epoch_time : float
            TDB epoch (seconds past J2000).
        reference_frame : str or Frame
            Output reference frame.
        ab_correct : str, optional
            Aberration correction. Defaults to ``"None"``.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` position vector in km.
        """
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name
        state = spice.spkezr(
            trgt_bdy, epoch_time, reference_frame, ab_correct, obsvr_bdy
        )
        return np.array(state[0])[0:3]

    @classmethod
    def _get_pos_callback(cls, trgt_bdy, reference_frame, obsvr_bdy, ab_correct):
        def spice_callback(epoch_time):
            result = cls._get_pos_core(
                trgt_bdy, float(epoch_time), reference_frame, obsvr_bdy, ab_correct
            )
            return result.values

        return spice_callback

    @classmethod
    def _get_pos_core(
        cls, trgt_bdy, epoch_time, reference_frame, obsvr_bdy, ab_correct
    ):
        """
        Get the position of a target body relative to an origin body at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the position is to be determined. Can be a string or a Body object.

        epoch_time : float or EpochArray
            The epoch time at which the position is to be determined. Can be a float or an EpochArray object.

        reference_frame : str or Frame
            The reference frame in which the position is to be determined. Can be a string or a Frame object.

        obsvr_bdy : str or Body
            The observing body from which the position is to be measured. Can be a string or a Body object.

        ab_correct : str, optional
            Aberration correction to be applied. Defaults to ``None``.

        Returns
        -------
        pos : ArrayWUnits
            The position of the target body relative to the origin body, expressed in units of kilometers.
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        # epoch_time = cls._validate_epoch_input(epoch_time)

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # get position from SPICE -> spkezr returns a tuple of the state [pos_x pos_y pos_z vel_x vel_y vel_z], and
        # the one way light time between observer and target
        state = spice.spkezr(
            trgt_bdy, epoch_time, reference_frame, ab_correct, obsvr_bdy
        )

        # only need the state information
        state = np.array(state[0])

        # only need the position elements of the state (first, second, and third elements)
        pos = state[0:3]

        # convert to an ArrayWUnits and return. Needs to output in km
        return ArrayWUnits(pos, cls._spc_pos_units)

    @classmethod
    def get_vel(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        ab_correct: str = "None",
    ):
        """
        Calculates the velocity of a target body relative to an observer body at a specified epoch time and reference frame.

        Parameters
        ----------
        trgt_bdy : str | Body
            The target body for which the velocity is to be computed. Can be a string identifier or a Body object.
        epoch_time : float | EpochArray
            The epoch time(s) at which to compute the velocity. Can be a single float or an array of epochs.
        reference_frame : str | Frame
            The reference frame in which the velocity is expressed. Can be a string or a Frame object.
        obsvr_bdy : str | Body
            The observer body relative to which the velocity is computed. Can be a string identifier or a Body object.
        ab_correct : str, optional
            Aberration correction flag. Defaults to "None".

        Returns
        -------
        Velocity or array-like
            The computed velocity of the target body relative to the observer in the specified reference frame.
        """

        vel = cls._get_vel_core(
            trgt_bdy, epoch_time, reference_frame, obsvr_bdy, ab_correct
        )

        return vel

    @classmethod
    def _get_vel_callback(cls, trgt_bdy, reference_frame, obsvr_bdy, ab_correct):
        def spice_callback(epoch_time):
            result = cls._get_vel_core(
                trgt_bdy, float(epoch_time), reference_frame, obsvr_bdy, ab_correct
            )
            return result.values

        return spice_callback

    @classmethod
    def _get_vel_core(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        ab_correct: str = "None",
    ):
        """
        Get the velocity of a target body relative to an origin at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the velocity is to be determined. Can be a string or a Body object.

        epoch_time : float or EpochArray
            The epoch time at which the velocity is to be determined. Can be an EpochArray or a float.

        reference_frame : str or Frame
            The reference frame in which the velocity is to be determined. Can be a string or a Frame object.

        obsvr_bdy : str or Body
            The origin relative to which the velocity is to be determined. Can be a string or a Body object.

        ab_correct : str, optional
            Aberration correction to be applied. Defaults to ``None``.

        Returns
        -------
        vel : ArrayWUnits
            The velocity of the target body as an ArrayWUnits object with the appropriate units.
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time = cls._validate_epoch_input(epoch_time)

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # --------------#
        # get velocity #
        # --------------#
        # get state from SPICE
        state = spice.spkezr(
            targ=trgt_bdy,
            et=epoch_time,
            ref=reference_frame,
            abcorr=ab_correct,
            obs=obsvr_bdy,
        )

        # extract velocity components
        vels = np.array(state[0])

        # cast as awu and return. Needs to output in km/sec
        return ArrayWUnits(vels[3:6], cls._spc_vel_units)

    @classmethod
    def get_state(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        aberration_correction: str = "None",
    ):
        """
        Retrieve the state (position and velocity) of a target body relative to an origin body at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the state is to be retrieved. Can be a string name or a Body object.

        epoch_time : float or EpochArray
            The epoch time at which the state is to be retrieved. Can be an EpochArray object or a float representing the time.

        reference_frame : str or Frame
            The reference frame in which the state is to be retrieved. Can be a Frame object or a string name.

        obsvr_bdy : str or Body
            The observing body relative to which the state is to be retrieved. Can be a string name or a Body object.

        aberration_correction : str, optional
            The aberration correction to be applied. Defaults to ``None``.

        Returns
        -------
        state : ArrayWUnits
            An array containing the state (position and velocity) of the target body with associated units.
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time = cls._validate_epoch_input(epoch_time)

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # -----------#
        # get state #
        # -----------#

        # Get state from SPICE
        state = spice.spkezr(
            trgt_bdy, epoch_time, reference_frame, aberration_correction, obsvr_bdy
        )
        state = np.array(state[0])

        return ArrayWUnits.from_trusted(state, cls._state_units())

    @classmethod
    def get_state_precise(
        cls,
        trgt_bdy: str | Body,
        epoch_time_high: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        aberration_correction: str = "None",
        epoch_time_low: float | None = None,
    ):
        """
        Retrieve the state of a target body at a two-part epoch ``epoch_time_high + epoch_time_low``.

        SPICE is evaluated once at the nearest representable epoch, and the position is then moved
        along the velocity by the part of the epoch that a single float cannot hold. The neglected
        second-order term is ``|a| d^2 / 2``, with ``|d|`` at most half an ULP of the epoch (~60 ns
        near J2000 + 26 yr), i.e. below 1e-14 km for any solar-system acceleration.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the state is to be retrieved. Can be a string name or a Body object.

        epoch_time_high : float or EpochArray
            High part of the epoch, in TDB seconds past J2000. Can be an EpochArray object or a float.

        reference_frame : str or Frame
            The reference frame in which the state is to be retrieved. Can be a Frame object or a string name.

        obsvr_bdy : str or Body
            The observing body relative to which the state is to be retrieved. Can be a string name or a Body object.

        aberration_correction : str, optional
            The aberration correction to be applied. Defaults to ``None``.

        epoch_time_low : float, optional
            Low part of the epoch [s], typically from :meth:`EpochArray._add_duration`. ``None``
            is treated as ``0.0``, which reproduces :meth:`get_state`.

        Returns
        -------
        state : ArrayWUnits
            An array containing the state (position and velocity) of the target body with associated units.
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time_high = cls._validate_epoch_input(epoch_time_high)
        if epoch_time_low is None:
            epoch_time_low = 0.0

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # -----------#
        # get state #
        # -----------#

        # Nearest representable epoch (t) and the remainder SPICE cannot see (d)
        t, d = EpochArray._two_sum(epoch_time_high, epoch_time_low)

        state = np.array(
            spice.spkezr(
                trgt_bdy,
                t,
                reference_frame,
                aberration_correction,
                obsvr_bdy,
            )[0]
        )

        # First-order shift of the position by the remainder
        state = np.concatenate((state[:3] + state[3:] * d, state[3:]))

        return ArrayWUnits.from_trusted(state, cls._state_units())

    @classmethod
    def _state_units(cls) -> _np.ndarray:
        """Return the per-element units of a SPICE state, km and km/s.

        :meth:`get_state` and :meth:`get_state_precise` hand this to
        :meth:`ArrayWUnits.from_trusted`: the light-time solutions call them
        hundreds of thousands of times per dataset, and validating a fixed
        list of six units through the constructor cost more than SPICE did. A
        new array each call, since ``ArrayWUnits.__setitem__`` can write into
        it.

        Returns
        -------
        numpy.ndarray
            Object array ``[km, km, km, km/s, km/s, km/s]``.
        """
        return _np.array([cls._spc_pos_units] * 3 + [cls._spc_vel_units] * 3)

    @classmethod
    def get_state_antenna(
        cls,
        trgt_bdy: Body,
        epoch_time: float,
        reference_frame: Frame,
        obsvr_bdy: Body,
        antenna_offset: np.ndarray,
        offset_reference_frame: Frame,
        ab_correct: str = "None",
    ):
        """
        Retrieve the state (position and velocity) of the antenna on a target spacecraft relative to an origin body at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str
            The target body for which the position is to be determined. Can be a string or a Body object.

        epoch_time : float
            The epoch time at which the position is to be determined. Can be a float or an EpochArray object.

        reference_frame : Frame
            The reference frame in which the position is to be determined. Can be a string or a Frame object.

        obsvr_bdy : str
            The observing body from which the position is to be measured. Can be a string or a Body object.

        antenna_offset: np.ndarray
            The antenna-CoM offset vector, expressed in the "spacecraft_body_frame" reference frame.

        offset_reference_frame: Frame
            The spaceraft body-fixed reference frame. The origin of this frame is used to account for the antenna offset vector.

        ab_correct : str, optional
            Aberration correction to be applied. Defaults to ``None``.


        Returns
        -------
        state : np.ndarra()
            The State of the antenna relative to the origin body, expressed in units of km and km/sec
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time = cls._validate_epoch_input(epoch_time)

        # Get the spacecraft state in "reference_frame" (f1)
        sc_state_f1 = spice.spkezr(
            trgt_bdy, epoch_time, reference_frame.name, ab_correct, obsvr_bdy
        )[0]
        sc_pos_f1 = np.array(
            sc_state_f1[:3]
        )  # Position of the spacecraft COM in the "reference_frame"
        sc_vel_f1 = np.array(
            sc_state_f1[3:6]
        )  # Velocity of the spacecraft COM in the "reference_frame"

        # Get the rotation matrix between "offset_reference_frame" (f2) and "reference_frame" (f1) (Will complain if there is no attitude or spacecraft clock kernel)
        DCM_from_f2_to_f1 = Frame.get_DCM(
            offset_reference_frame, reference_frame, EpochArray(epoch_time, "TDB")
        )
        # Convert the "antenna_offset" vector defined in "offset_reference_frame" as a vector defined in "reference_frame"
        delta_antenna_pos_f1 = DCM_from_f2_to_f1 @ antenna_offset

        # Put toghether pos and vel
        antenna_pos_f1 = sc_pos_f1 + delta_antenna_pos_f1
        antenna_vel_f1 = (
            sc_vel_f1  # Approximation of antenna velocity as spacecraft velocity
        )
        antenna_state = np.hstack((antenna_pos_f1, antenna_vel_f1))
        return antenna_state

    @classmethod
    def get_elevation_angle(
        cls,
        target: str,
        et,
        station: str = "DSS-14",
        abcorr: str = "LT+S",
    ) -> float:
        """
        Compute the elevation angle of a target body as seen from a DSN ground station using SPICE.

        This method uses the SPICE toolkit to calculate the elevation angle of a target
        relative to a Deep Space Network (DSN) ground station. The calculation is performed
        in the station's local topocentric frame (e.g., ``DSS-14_TOPO``), where the
        convention is X = East, Y = North, and Z = Up.

        Parameters
        ----------
        target : str
            The name or NAIF ID of the target body. Examples: ``"MRO"`` or ``"-74"``.

        et : float
            Ephemeris time at which to compute the elevation angle.

        station : str
            The DSN ground station name or NAIF ID used as the observer. Must match the
            identifiers defined in the loaded SPICE kernels. Defaults to ``"DSS-14"``.

        abcorr : str, optional
            Aberration correction to apply when computing the geometry. Examples:
            - ``"NONE"``: No corrections.
            - ``"LT"``: Light-time correction only.
            - ``"LT+S"``: Light-time + stellar aberration correction (default).

        Returns
        -------
        elevation_angle : ArrayWUnits
            The elevation angle of the target relative to the local horizon at the
            specified DSN station, expressed in radians (or the units defined by
            ``cls._spc_angle_units``). Positive values indicate the target is above
            the horizon, while negative values indicate it is below.

        Notes
        -----
        - The station's topocentric frame (e.g., ``DSS-14_TOPO``) must be defined in the
          loaded SPICE kernels.
        - The axis convention for topocentric frames is assumed to be:
          X = East, Y = North, Z = Up.
        - To obtain azimuth as well, use ``atan2(East, North)`` in the same frame.
        - Requires SPICE kernels for:

          * DSN station locations and frames.
          * Target body ephemerides.
          * Leapseconds (LSK).
        """
        topo_frame = f"{station}_TOPO"

        # Convention assumed: X=East, Y=North, Z=Up (common for *_TOPO frames)
        pos_topo, _ = spice.spkpos(target, et, topo_frame, abcorr, station)
        e, n, u = pos_topo
        elevation_angle = np.arctan2(u, np.hypot(e, n))
        return ArrayWUnits(elevation_angle, cls._spc_angle_units)

    @classmethod
    def get_sep_angle(
        cls,
        target: str,
        et,
        station: str = "DSS-14",
        abcorr: str = "LT+S",
    ) -> float:
        """
        Compute the Sun–Earth–Probe (SEP) angle of a target body as seen from a DSN ground station.

        The SEP angle is defined as the angle at the observer (DSN station) between
        the vector pointing to the Sun and the vector pointing to the spacecraft (target).
        This is the standard "solar elongation angle" used in mission operations to assess
        geometry for radiometric tracking and solar corona effects.

        Parameters
        ----------
        target : str
            The name or NAIF ID of the target body. Examples: ``"OSIRIS-REX"`` or ``"-64"``.

        et : float
            Ephemeris Time (seconds past J2000 TDB) at which to compute the SEP angle.

        station : str, optional
            DSN ground station name or NAIF ID used as the observer.
            Must match the identifiers defined in the loaded SPICE kernels.
            Default is ``"DSS-14"``.

        abcorr : str, optional
            Aberration correction to apply when computing the geometry. Examples:
            - ``"NONE"``: No corrections.
            - ``"LT"``: Light-time correction only.
            - ``"LT+S"``: Light-time + stellar aberration correction (default).

        Returns
        -------
        sep_angle : ArrayWUnits
            The SEP angle in radians (or converted via ``cls._spc_angle_units``).
            Larger angles correspond to better tracking geometry (target farther
            from the Sun as seen from Earth).

        Notes
        -----
        - Requires SPICE kernels for:
            * DSN station locations (SPK + topocentric frame definitions).
            * Target body ephemerides (SPK).
            * Solar system ephemerides (SPK).
            * Leapseconds (LSK).
        - This angle is distinct from **elevation angle** (target above station
          horizon). SEP measures angular separation from the Sun, not from the horizon.
        - Often used to set operational limits during solar conjunctions.

        Examples
        --------
        .. code-block:: python

            et = scb.SpiceManager.utc2et("2017-09-22T16:52:00")  # OSIRIS-REx EGA
            sep = MyClass.get_sep_angle("OSIRIS-REX", et, station="DSS-14")
            print(sep.to("deg"))
        """
        # Sun-DSS vector
        r_sun_dss, _ = spice.spkpos("SUN", et, "J2000", abcorr, station)
        # Earth-SC vector
        r_sc_dss, _ = spice.spkpos(target, et, "J2000", abcorr, station)
        # SEP angle, angle between the two dss-based vectors
        sep_angle = spice.vsep(r_sun_dss, r_sc_dss)  # in [rad] from Spice
        return ArrayWUnits(sep_angle, cls._spc_angle_units)

    @classmethod
    def get_lighttime(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        aberration_correction: str = "None",
    ):
        """
        Calculate the light time between a target and an origin at a given epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body or its name as a string.

        epoch_time : EpochArray or float
            The epoch time at which to calculate the light time.

        reference_frame : str or Frame
            The reference frame or its name as a string.

        origin : str or Body
            The origin body or its name as a string.

        aberration_correction : str, optional
            The aberration correction to apply. Defaults to ``None``.

        Returns
        -------
        light_time : ArrayWUnits
            The light time as an array with units.
        """

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # Get state from SPICE
        state = spice.spkezr(
            trgt_bdy, epoch_time, reference_frame, aberration_correction, obsvr_bdy
        )
        lighttime = np.array(state[1])

        return ArrayWUnits(lighttime, cls._spc_time_units)

    # Parsed-sidecar cache shared by every get_parameters call:
    #     absolute path -> (mtime, size, parsed contents)
    _sidecar_cache: dict = {}
    # Epoch grids derived from STM sidecars, keyed by the parsed object's id and
    # holding a reference to it so the entry cannot outlive its source.
    _sidecar_epoch_cache: dict = {}

    @classmethod
    def _load_sidecar_cached(cls, path: str):
        """
        Load a parameters sidecar, reusing the parse when the file is unchanged.

        Parameters
        ----------
        path : str
            Absolute path to a ``.json`` or ``.pkl`` sidecar.

        Returns
        -------
        object
            The parsed sidecar contents, or ``None`` when it cannot be read.

        Notes
        -----
        The cache key is the file's ``(mtime, size)``, so rewriting a sidecar
        invalidates it without any explicit bookkeeping. Use
        :meth:`clear_sidecar_cache` to drop everything, for instance after
        deleting files behind the library's back.
        """
        from scarabaeus.utils.Serialization import load as _load_sidecar

        try:
            stat = os.stat(path)
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

        cached = cls._sidecar_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]

        try:
            contents = _load_sidecar(path)
        except Exception:
            contents = None

        cls._sidecar_cache[path] = (key, contents)

        return contents

    @classmethod
    def _sidecar_derived(cls, payload, key, build):
        """
        Memoize something derived from a parsed sidecar.

        Sidecar contents arrive from JSON as Python lists, and the getters are
        called once per epoch — so anything derived from the whole file
        (converting an epoch grid to an array, fitting a parameter's time
        series) costs O(N) on every lookup and turns retrieval into O(N*M).
        For a finely sampled arc that is the difference between seconds and a
        run that looks hung.

        Derived values are cached against the parsed payload itself, so they
        inherit the file cache's ``(mtime, size)`` invalidation: rewriting a
        sidecar produces a new payload and the old entry is simply never
        consulted again.

        Parameters
        ----------
        payload : dict
            Parsed sidecar contents, used as the cache's owner.
        key : hashable
            Identifies the derived value within that sidecar.
        build : callable
            Produces the value on a miss.

        Returns
        -------
        object
            The cached or freshly built value.
        """
        entry = cls._sidecar_epoch_cache.get(id(payload))
        if entry is None or entry[0] is not payload:
            entry = (payload, {})
            cls._sidecar_epoch_cache[id(payload)] = entry

        values = entry[1]
        if key not in values:
            values[key] = build()
        return values[key]

    @classmethod
    def _sidecar_epochs_cached(cls, stm_data, leg_key=None):
        """
        Epoch grid of an STM sidecar as a float array, parsed once.

        Parameters
        ----------
        stm_data : dict
            Parsed STM sidecar.
        leg_key : hashable, optional
            Leg identifier, for sequence sidecars that store one STM grid per
            leg.

        Returns
        -------
        numpy.ndarray
            Ascending epoch grid, TDB seconds.
        """

        def build():
            if leg_key is None:
                raw = stm_data["epochsTDB"]
            else:
                raw = stm_data["legs"][leg_key]["STM_epochsTDB"]
            return np.array(raw, dtype=float)

        return cls._sidecar_derived(stm_data, ("stm_epochs", leg_key), build)

    @classmethod
    def clear_sidecar_cache(cls) -> None:
        """Drop every cached parameters sidecar."""
        cls._sidecar_cache.clear()
        cls._sidecar_epoch_cache.clear()

    # Names this library gives the sidecars it writes next to an SPK. They are
    # a preference, not a requirement: a file named anything else is read when
    # it is pointed at, or when it is the only candidate in a folder.
    _SIDECAR_SUFFIXES = {
        "parameters": ("_parameters.json", "_parameters.pkl"),
        "stms": ("_stms.json", "_stms.pkl"),
        "settings": ("_settings.json", "_settings.pkl"),
        "mass_profile": ("_mass_profile.json", "_mass_profile.pkl"),
    }

    @classmethod
    def sidecar_files(cls, location=None, kind: str = "parameters") -> list[str]:
        """
        List the sidecar files a lookup would read, in the order it reads them.

        Sidecars are found by *where they are*, not by a folder layout this
        library imposes, so ``location`` can be whatever handle the caller
        happens to hold:

        * an **SPK** (``.bsp``) — the sidecars written beside it are read
          first (``<base>_parameters.json`` and friends), then anything else
          in the same folder, which covers a run whose sidecars are shared
          under a different basename;
        * a **file** — read as-is, whatever it is called;
        * a **folder** — the files in it whose names carry this library's
          suffix for *kind* come first, then any other ``.json`` / ``.pkl``,
          so a folder following someone else's naming still resolves;
        * an **iterable** of any of the above — searched in the given order,
          for payloads split across locations;
        * ``None`` — :attr:`kernel_folder` when set, then
          ``<cwd>/data/kernels/scenario``.

        Locations that do not exist are skipped rather than raised on: the
        callers report every location they consulted when a lookup finds
        nothing, which is more useful than failing on the first missing one.

        Parameters
        ----------
        location : str or os.PathLike or iterable, optional
            Where to look, as described above. Defaults to ``None``.
        kind : str, optional
            Which payload is wanted — ``'parameters'``, ``'stms'``,
            ``'settings'``, or ``'mass_profile'``. Only sets which suffixes are
            preferred. Defaults to ``'parameters'``.

        Returns
        -------
        list of str
            Absolute paths of existing candidate files, most specific first,
            without duplicates.

        Examples
        --------
        .. code-block:: python

            SpiceManager.sidecar_files("/runs/case7/traj.bsp")
            SpiceManager.sidecar_files(["/runs/case7", "/shared/sidecars"])
        """
        suffixes = cls._SIDECAR_SUFFIXES.get(kind, cls._SIDECAR_SUFFIXES["parameters"])

        # Normalize whatever was given into an ordered list of locations
        if location is None:
            locations = [] if cls.kernel_folder is None else [cls.kernel_folder]
            locations.append(os.path.join(os.getcwd(), "data", "kernels", "scenario"))
        elif isinstance(location, (str, os.PathLike)):
            locations = [location]
        else:
            locations = list(location)

        def _folder_files(folder: str) -> list[str]:
            """Suffix matches first, then any other sidecar-shaped file."""
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                return []
            preferred = [n for n in names if n.endswith(suffixes)]
            others = [
                n for n in names if n.endswith((".json", ".pkl")) and n not in preferred
            ]
            return [os.path.join(folder, n) for n in preferred + others]

        files = []
        for loc in locations:
            loc = os.path.abspath(os.fspath(loc))

            if os.path.isdir(loc):
                files += _folder_files(loc)
            elif os.path.isfile(loc):
                if loc.endswith(".bsp"):
                    # An SPK stands for its own sidecars, then for its folder
                    base = loc[: -len(".bsp")]
                    files += [
                        base + suffix
                        for suffix in suffixes
                        if os.path.isfile(base + suffix)
                    ]
                    files += _folder_files(os.path.dirname(loc))
                else:
                    files.append(loc)

        # De-duplicate, keeping the order the locations were given in
        seen = set()
        return [f for f in files if not (f in seen or seen.add(f))]

    @classmethod
    def _sidecar_locations_searched(cls, location=None) -> str:
        """Human-readable list of the locations a lookup consulted."""
        if location is None:
            locations = [] if cls.kernel_folder is None else [cls.kernel_folder]
            locations.append(os.path.join(os.getcwd(), "data", "kernels", "scenario"))
        elif isinstance(location, (str, os.PathLike)):
            locations = [location]
        else:
            locations = list(location)

        return ", ".join(str(loc) for loc in locations) or "nowhere"

    # Keys that identify what a sidecar holds, so a candidate resolved by
    # location alone can still be told apart: an STM sidecar carries the same
    # body/origin/frame as the parameters sidecar beside it, and would otherwise
    # satisfy an identity match for either.
    _SIDECAR_MARKERS = {
        "parameters": ("parameters", "legs"),
        "stms": ("STMs",),
        "settings": ("integrator", "force_models"),
        "mass_profile": ("mass_profile", "coefficients"),
    }

    # Markers that some kinds also embed beside their own payload, for context
    # only: a sequence STM sidecar carries the sequence's "legs" next to its
    # "STMs". They identify a kind only when no other kind's payload is present.
    _SIDECAR_CONTEXT_MARKERS = ("legs",)

    @classmethod
    def _sidecar_is_kind(cls, data, kind: str) -> bool:
        """
        Whether a parsed sidecar holds the payload *kind* asks for.

        A file carrying another kind's payload belongs to that kind, unless the
        shared marker is one of ``_SIDECAR_CONTEXT_MARKERS``.

        Parameters
        ----------
        data : object
            Parsed sidecar contents.
        kind : str
            A key of ``_SIDECAR_MARKERS``: ``'parameters'``, ``'stms'``,
            ``'settings'`` or ``'mass_profile'``.

        Returns
        -------
        bool
            Whether *data* is a sidecar of that kind.
        """
        if not isinstance(data, dict):
            return False

        markers = cls._SIDECAR_MARKERS.get(kind, ())
        if not any(marker in data for marker in markers):
            return False

        # A file that carries another kind's payload belongs to that kind,
        # unless the shared marker is context it embeds beside our own.
        other_payloads = [
            marker
            for other, other_markers in cls._SIDECAR_MARKERS.items()
            if other != kind
            for marker in other_markers
            if marker not in cls._SIDECAR_CONTEXT_MARKERS
        ]
        return not any(marker in data for marker in other_payloads)

    @staticmethod
    def _sidecar_has_identity(data) -> bool:
        """Whether a sidecar records which body, origin, and frame it describes."""
        if not isinstance(data, dict):
            return False
        return isinstance(data.get("legs"), dict) or "body_ID" in data

    @staticmethod
    def _sidecar_matches(data, body_id: int, origin_id: int, frame_name: str) -> bool:
        """
        Whether a sidecar was written for this body, origin, and frame.

        Handles both shapes the writers produce: identifiers at the top level,
        or one entry per leg under ``'legs'`` for a mission sequence, where a
        match on any leg makes the file the right one.

        Parameters
        ----------
        data : object
            Parsed sidecar contents.
        body_id : int
            SPICE ID of the target body.
        origin_id : int
            SPICE ID of the origin body.
        frame_name : str
            Name of the reference frame.

        Returns
        -------
        bool
            ``True`` when the sidecar describes the requested combination.
        """
        if not isinstance(data, dict):
            return False

        def _match(entry) -> bool:
            return (
                isinstance(entry, dict)
                and entry.get("body_ID") == body_id
                and entry.get("origin_ID") == origin_id
                and entry.get("reference_frame") == frame_name
            )

        legs = data.get("legs")
        if isinstance(legs, dict):
            return any(_match(leg) for leg in legs.values())

        return _match(data)

    @classmethod
    def get_parameters(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
        per_parameter: bool = False,
        sidecar_path=None,
    ):
        """
        Retrieve and interpolate estimated parameters from a companion JSON file.

        Reads the candidates :meth:`sidecar_files` resolves from
        *sidecar_path* — an SPK, a file, a folder, or several of them — and
        takes the first whose ``body_ID``, ``origin_ID``, and
        ``reference_frame`` match the supplied arguments, at the top level or
        in any leg.  If the file contains a multi-leg sequence, the
        leg whose ``epochsTDB`` interval contains *epoch_time* is selected.
        Within each leg (or in the flat, non-sequence case) the stored parameter
        time series is fitted with a polynomial of degree
        ``cls.poly_interp_par_deg`` and evaluated at *epoch_time*.

        Parameters
        ----------
        trgt_bdy : str or Body
            Target body whose parameters are requested.
        epoch_time : float or EpochArray
            TDB epoch (seconds past J2000) at which to evaluate the parameters.
        reference_frame : str or Frame
            Reference frame recorded in the JSON file.
        obsvr_bdy : str or Body
            Observer (origin) body recorded in the JSON file.
        folder_path_override : str, optional
            Deprecated alias for *sidecar_path*, kept working for callers that
            pass a folder.  Defaults to ``None``.
        per_parameter : bool, optional
            Return a ``{name: ArrayWUnits}`` mapping instead of the stacked
            vector.  Defaults to ``False``.
        sidecar_path : str or os.PathLike or iterable, optional
            Where to look for the sidecar: an SPK, a file, a folder, or an
            iterable of those — see :meth:`sidecar_files`.  When ``None``
            (default), ``cls.kernel_folder`` is used if set; otherwise falls
            back to ``<cwd>/data/kernels/scenario``.

        Returns
        -------
        parameters : ArrayWUnits or dict of str to ArrayWUnits
            By default an :class:`ArrayWUnits` holding the stacked parameter
            vector, always — the return type does not vary with the contents of
            the file.

            Its units are the shared units when every parameter of the body
            agrees, which is the usual case. A body's parameters need not agree,
            though: solving for a delta-v (km/s) alongside a maneuver-epoch
            correction (sec) is ordinary. No single unit then describes the
            stack, so it is reported dimensionless rather than mislabelled with
            whichever unit came first, and ``per_parameter`` is the way to
            recover what each column actually is.

            With ``per_parameter=True``, a ``{name: ArrayWUnits}`` mapping
            instead, each entry carrying its own units.

        Raises
        ------
        SpiceManager.SidecarNotFoundError
            If no matching sidecar is found; the message lists every location
            searched.
        ValueError
            If the sidecar is malformed (e.g. a leg's ``parameters`` entry), or
            if *epoch_time* does not fall within any leg's time span.

        Warns
        -----
        RuntimeWarning
            If more than one matching sidecar is found; the most recently
            modified file is used.
        """

        location = sidecar_path if sidecar_path is not None else folder_path_override

        json_parameters = None
        matched_paths = []

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        trgt_id = spice.bodn2c(trgt_bdy)
        obsvr_id = spice.bodn2c(obsvr_bdy)

        # ------------------------------------------------------------------
        # Iterate over the candidate sidecars, most specific location first
        # ------------------------------------------------------------------
        for path in cls.sidecar_files(location, kind="parameters"):
            # JSON or pickle, auto-detected by extension. Parsed through a
            # process-wide cache: Trajectory.get_state() calls this once per
            # epoch, so without it a scenario folder holding N sidecars is
            # re-read N times per epoch — the dominant cost of plotting a
            # solution. Entries are keyed on the file's modification time, so
            # a rewritten sidecar is re-read automatically.
            json_data = cls._load_sidecar_cached(path)

            if not cls._sidecar_is_kind(json_data, "parameters"):
                continue

            if cls._sidecar_matches(json_data, trgt_id, obsvr_id, reference_frame):
                matched_paths.append(path)
                json_parameters = json_data

        # Print error if needed
        if len(matched_paths) > 1:

            # Pick most recently modified matching file
            latest_idx = np.argmax([os.path.getmtime(p) for p in matched_paths])
            latest_path = matched_paths[latest_idx]

            warnings.warn(
                "Multiple matching JSON files found. "
                f"Using most recently modified file: {os.path.basename(latest_path)}",
                RuntimeWarning,
            )

            json_parameters = cls._load_sidecar_cached(latest_path)

        elif not matched_paths:
            raise cls.SidecarNotFoundError(
                f"No parameters sidecar for body {trgt_id} about origin "
                f"{obsvr_id} in frame '{reference_frame}' was found. Searched: "
                f"{cls._sidecar_locations_searched(location)}. Pass "
                "`sidecar_path` pointing at the SPK, the sidecar file, or the "
                "folder holding it."
            )

        # Determine if the JSON contains sequences
        if "legs" in json_parameters:
            parameter = None
            units = None

            # Find the appropriate leg for the given epoch_time
            for idx, leg in enumerate(json_parameters["legs"].values()):
                if not isinstance(leg, dict):
                    continue
                epochsTDB = cls._sidecar_derived(
                    json_parameters,
                    ("leg_epochs", idx),
                    lambda leg=leg: np.array(leg["epochsTDB"], dtype=float),
                )
                if epochsTDB.size == 0:
                    continue

                if epochsTDB[0] <= epoch_time <= epochsTDB[-1]:
                    leg_par = leg.get("parameters", None)

                    # If no parameters on this leg, skip
                    if leg_par is None:
                        continue

                    # ------------------------------------------------------------------
                    # Expect "definition blob" list:
                    #   [name0, dict0, name1, dict1, ...]
                    # where each dict may contain either:
                    #   - constant:   "values": [p...]                (no "epochsTDB")
                    #   - time series: "epochsTDB": [...], "values": [[...],[...],...]
                    # ------------------------------------------------------------------
                    if not isinstance(leg_par, (list, tuple)) or len(leg_par) < 2:
                        raise ValueError(
                            f"Unexpected 'parameters' format in leg {idx}: expected list/tuple [name, dict, ...]."
                        )

                    vals_out = []
                    # (name, dim, Units) per parameter. A Units object holds a
                    # single 1x4 dimension spec, so several parameters cannot
                    # share one by concatenating their powers.
                    blocks = []

                    for i in range(0, len(leg_par), 2):
                        if i + 1 >= len(leg_par):
                            raise ValueError(
                                f"Malformed parameters list in leg {idx}: odd length {len(leg_par)}."
                            )

                        d = leg_par[i + 1]
                        if not isinstance(d, dict) or "values" not in d:
                            raise ValueError(
                                f"Malformed parameters entry in leg {idx} at pair {i//2}."
                            )

                        powers = d.get("unitsPower")
                        par_units = (
                            Units(
                                Dimensions(np.array(powers, dtype=int).ravel()),
                                np.array(d.get("unitsScale", []), dtype=float).ravel(),
                            )
                            if powers
                            else None
                        )

                        v_raw = d.get("values", None)
                        if v_raw is None:
                            raise ValueError(
                                f"Missing 'values' in parameters dict for leg {idx}."
                            )

                        # A leg parameter is either constant over the leg (one
                        # value per component) or a time series (one row per
                        # epoch). Decide on the array's rank, not on the Python
                        # type: a size-1 parameter such as a solved-for SRP
                        # scale factor is serialised by ``ndarray.tolist()`` as
                        # a bare float, which no isinstance check for
                        # list/tuple recognises. Such an entry used to fall
                        # through to the time-series branch and raise
                        # ``IndexError`` on a 0-d array, which
                        # ``SolutionOD._compute_state_est`` swallowed — leaving
                        # an all-zero state estimate that looked like an answer.
                        v_arr = np.asarray(v_raw, dtype=float)

                        # Case A: constant (scalar, or a P-vector)
                        # Example: "values": [0,0,0]  or  "values": 1.0
                        if v_arr.ndim <= 1:
                            v = np.atleast_1d(v_arr).ravel()
                            vals_out.append(v)
                            blocks.append((leg_par[i], v.size, par_units))
                            continue

                        # Case B: time series
                        # Example: "epochsTDB": [t0..tN], "values": [[p...],[p...],...]
                        p_epochs = d.get("epochsTDB", None)
                        if p_epochs is None:
                            # If values look 2D but no epochs provided, fall back to leg epochsTDB (same length check below)
                            p_epochs = epochsTDB

                        deg = int(cls.poly_interp_par_deg)

                        # The fit depends only on the file's contents, so it is
                        # done once per parameter rather than once per epoch.
                        # A least-squares solve over the whole grid on every
                        # lookup is what makes a finely sampled arc appear to
                        # hang, and it only bites once a parameter is solved for.
                        def _fit(p_epochs=p_epochs, v_raw=v_raw, idx=idx, deg=deg):
                            p_e = np.array(p_epochs, dtype=float)
                            V = np.array(v_raw, dtype=float)

                            # Allow V as (N,P) or (N,) for scalar param
                            if V.ndim == 1:
                                V = V.reshape(-1, 1)

                            if p_e.shape[0] != V.shape[0]:
                                raise ValueError(
                                    f"Parameter time series length mismatch in leg {idx}: "
                                    f"len(epochs)={p_e.shape[0]} vs values rows={V.shape[0]}"
                                )

                            # coeff shape: (deg+1, P)
                            return np.polyfit(p_e, V, deg)

                        coeff = cls._sidecar_derived(
                            json_parameters, ("param_fit", idx, i, deg), _fit
                        )
                        v = np.polyval(coeff, epoch_time).ravel()  # -> (P,)
                        vals_out.append(v)
                        blocks.append((leg_par[i], v.size, par_units))

                    # Concatenate all parameters in order
                    parameter = (
                        np.concatenate(vals_out)
                        if len(vals_out)
                        else np.array([], dtype=float)
                    )
                    if per_parameter:
                        return cls._split_parameters(parameter, blocks)
                    units = cls._stacked_parameter_units(blocks)
                    break

            if parameter is None:
                raise ValueError(
                    "The given epoch_time does not fall within any leg's epochsTDB."
                )

        # Nominal case (no sequence)
        else:
            # The fit depends only on the file, so it is done once per sidecar
            # rather than once per epoch. Refitting here — a least-squares
            # solve over the whole grid, plus two list-to-array conversions —
            # is what makes a finely sampled arc appear to hang, since the
            # caller asks once per measurement.
            def _fit_flat():
                parameters = np.array(json_parameters.get("parameters"))
                epochsTDB = np.array(json_parameters.get("epochsTDB"))
                return np.polyfit(epochsTDB, parameters, cls.poly_interp_par_deg)

            json_poly_coefficients = cls._sidecar_derived(
                json_parameters, ("flat_fit", int(cls.poly_interp_par_deg)), _fit_flat
            )
            parameter = np.polyval(json_poly_coefficients, epoch_time)

            # Units. `parametersUnits` is per-parameter and authoritative; the
            # scalar unitsPower/unitsScale are the legacy single-unit form and
            # only exist when every parameter agrees. A body whose parameters
            # carry different units (delta-v in km/s next to an epoch
            # correction in sec) therefore has no single label, and each
            # parameter is returned separately.
            per_param = json_parameters.get("parametersUnits")
            if per_param:
                blocks = cls._parameter_unit_blocks(per_param)
                units = cls._stacked_parameter_units(blocks)

                if per_parameter:
                    return cls._split_parameters(parameter, blocks)
            else:
                units = Units(
                    Dimensions(np.array(json_parameters.get("unitsPower"))),
                    np.array(json_parameters.get("unitsScale")),
                )
                if per_parameter:
                    return {"parameters": ArrayWUnits(parameter, units)}

        return ArrayWUnits(parameter, units)

    @staticmethod
    def _parameter_unit_blocks(per_param) -> list:
        """Decode a sidecar ``parametersUnits`` list into ``(name, dim, Units)``."""
        blocks = []
        for d in per_param:
            powers = d.get("unitsPower")
            units = (
                Units(
                    Dimensions(np.array(powers, dtype=int)),
                    np.array(d.get("unitsScale", []), dtype=float),
                )
                if powers
                else None
            )
            blocks.append((d["name"], int(d.get("dim", 1)), units))
        return blocks

    @staticmethod
    def _same_units(a, b) -> bool:
        """
        Units equality that also accepts ``None`` (a unitless parameter).

        Parameters
        ----------
        a, b : Units or None
            Units to compare.

        Returns
        -------
        bool
            Whether both are ``None``, or both are equal Units.
        """
        # Units.__eq__ raises TypeError against anything that is not a Units
        if a is None or b is None:
            return a is b
        return a == b

    @classmethod
    def _stacked_parameter_units(cls, blocks):
        """Units of a stacked parameter vector built from ``(name, dim, Units)``.

        The shared units when every parameter agrees. When they differ no
        single unit describes the stack, so it is returned dimensionless
        rather than mislabelled; ``per_parameter`` gives the per-column units.

        Parameters
        ----------
        blocks : list of tuple
            ``(name, dim, Units or None)`` per parameter, as from
            :meth:`_parameter_unit_blocks`.

        Returns
        -------
        Units or None
            The shared units (``None`` when every parameter is unitless, or
            there are none), or dimensionless when they differ.
        """
        all_units = [u for _, _, u in blocks]
        first = all_units[0] if all_units else None
        if all(cls._same_units(u, first) for u in all_units):
            return first
        return Units(Dimensions([0, 0, 0, 0]), [1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def _split_parameters(parameter, blocks) -> dict:
        """Split a stacked parameter vector into ``{name: ArrayWUnits}``."""
        values = np.atleast_1d(parameter)
        out, col = {}, 0
        for name, dim, units in blocks:
            chunk = values[col : col + dim]
            out[name] = ArrayWUnits(chunk[0] if dim == 1 else chunk, units)
            col += dim
        return out

    @classmethod
    def get_STMs(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
        sidecar_path=None,
    ) -> np.ndarray:
        """
        Retrieve or interpolate the State Transition Matrix (STM) at a given epoch.

        Reads the candidates :meth:`sidecar_files` resolves from
        *sidecar_path* — an SPK, a file, a folder, or several of them — and
        takes the first whose ``body_ID``, ``origin_ID``, and
        ``reference_frame`` match the supplied arguments.  If *epoch_time*
        equals a stored epoch the corresponding STM is returned directly.
        Otherwise element-wise linear interpolation is performed between the
        two bracketing epochs.

        A sequence sidecar stores one STM block per leg, each on that leg's
        own grid.  The STM is then taken from the first leg whose span holds
        *epoch_time*, so an epoch on the boundary between two legs reads the
        end of the earlier one.

        Parameters
        ----------
        trgt_bdy : str or Body
            Target body whose STMs are requested.
        epoch_time : float
            TDB epoch (seconds past J2000) at which to retrieve/interpolate
            the STM.
        reference_frame : str or Frame
            Reference frame recorded in the STM JSON file.
        obsvr_bdy : str or Body
            Observer (origin) body recorded in the STM JSON file.
        folder_path_override : str, optional
            Deprecated alias for *sidecar_path*, kept working for callers that
            pass a folder.  Defaults to ``None``.
        sidecar_path : str or os.PathLike or iterable, optional
            Where to look for the sidecar: an SPK, a file, a folder, or an
            iterable of those — see :meth:`sidecar_files`.  Defaults to
            ``None``.

        Returns
        -------
        STM : numpy.ndarray
            Shape ``(n, n)`` State Transition Matrix evaluated at *epoch_time*,
            ``n`` being the size of the state it was propagated with.

        Raises
        ------
        SidecarNotFoundError
            If no matching STM sidecar is found in any location searched.
        ValueError
            If *epoch_time* lies outside the stored time span, or outside every
            leg's span for a sequence sidecar.
        """
        # 1) Normalize inputs
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        trgt_id = spice.bodn2c(trgt_bdy)
        obsvr_id = spice.bodn2c(obsvr_bdy)

        # 2) Find the right sidecar among the candidate locations
        location = sidecar_path if sidecar_path is not None else folder_path_override

        stm_data = None
        for path in cls.sidecar_files(location, kind="stms"):
            jd = cls._load_sidecar_cached(path)
            if not cls._sidecar_is_kind(jd, "stms"):
                continue
            if cls._sidecar_matches(jd, trgt_id, obsvr_id, reference_frame):
                stm_data = jd
                break

        if stm_data is None:
            raise cls.SidecarNotFoundError(
                f"No STM sidecar for body {trgt_id} about origin {obsvr_id} in "
                f"frame '{reference_frame}' was found. Searched: "
                f"{cls._sidecar_locations_searched(location)}. Pass "
                "`sidecar_path` pointing at the SPK, the sidecar file, or the "
                "folder holding it."
            )

        # 3) Extract the appropriate epochs array
        #    (either across a single span or within the correct leg)
        legs = stm_data.get("legs")
        if legs and all("STM_epochsTDB" in leg for leg in legs.values()):
            # sequence case: one STM block per leg, keyed by leg
            for leg_key in legs:
                leg_epochs = cls._sidecar_epochs_cached(stm_data, leg_key)
                if leg_epochs[0] <= epoch_time <= leg_epochs[-1]:
                    epochs = leg_epochs
                    stms = stm_data["STMs"][leg_key]
                    break
            else:
                raise ValueError(f"Epoch {epoch_time} is not in any leg.")
        else:
            # non‐sequence
            epochs = cls._sidecar_epochs_cached(stm_data)
            stms = stm_data["STMs"]
            if not (epochs[0] <= epoch_time <= epochs[-1]):
                raise ValueError("Epoch out of stored range.")

        # 4) Locate the epoch by binary search. A linear scan here would be
        #    O(N) per lookup, and the caller asks once per measurement.
        #    epochs[idx] is the first stored epoch at or after epoch_time, and
        #    the range check above keeps idx inside the grid.
        idx = int(np.searchsorted(epochs, epoch_time))

        # 5) Exact match: return the stored matrix directly. A relative
        #    tolerance would not do here: at ET ~7.6e8 s, np.isclose's
        #    default spans ~7600 s and swallows every interpolation.
        if epochs[idx] == epoch_time:
            return np.array(stms[str(float(epochs[idx]))])

        # 6) Otherwise interpolate element-wise between the bracketing epochs
        t1, t2 = float(epochs[idx - 1]), float(epochs[idx])
        STM1 = np.array(stms[str(t1)])
        STM2 = np.array(stms[str(t2)])
        alpha = (epoch_time - t1) / (t2 - t1)
        return (1 - alpha) * STM1 + alpha * STM2

    @classmethod
    def get_propagator_settings(
        cls,
        trgt_bdy: str | Body,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
        sidecar_path=None,
    ):
        """
        Retrieves the propagator settings for a given target, reference frame, and origin from JSON files.

        Reads the candidates :meth:`sidecar_files` resolves from
        *sidecar_path* — an SPK, a file, a folder, or several of them — and
        takes the first whose ``body_ID``, ``origin_ID``, and
        ``reference_frame`` match the supplied arguments.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body or its name.
        reference_frame : str or Frame
            The reference frame or its name.
        obsvr_bdy : str or Body
            The origin body or its name.
        folder_path_override : str, optional
            Deprecated alias for *sidecar_path*, kept working for callers that
            pass a folder.  Defaults to ``None``.
        sidecar_path : str or os.PathLike or iterable, optional
            Where to look for the sidecar: an SPK, a file, a folder, or an
            iterable of those — see :meth:`sidecar_files`.  Defaults to
            ``None``.

        Returns
        -------
        dict
            The propagator settings data from the corresponding JSON file.

        Raises
        ------
        SidecarNotFoundError
            If no matching settings sidecar is found in any location searched.
        """
        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        trgt_id = spice.bodn2c(trgt_bdy)
        obsvr_id = spice.bodn2c(obsvr_bdy)

        location = sidecar_path if sidecar_path is not None else folder_path_override

        settings_data = None
        for path in cls.sidecar_files(location, kind="settings"):
            json_data = cls._load_sidecar_cached(path)

            if not cls._sidecar_is_kind(json_data, "settings"):
                continue

            # A propagator-settings payload records no body, origin, or frame
            # of its own: it is identified by the SPK it was written beside,
            # which is what the location resolves. Identifiers are still
            # honoured when a writer did record them.
            if cls._sidecar_has_identity(json_data) and not cls._sidecar_matches(
                json_data, trgt_id, obsvr_id, reference_frame
            ):
                continue

            settings_data = json_data
            break

        if not settings_data:
            raise cls.SidecarNotFoundError(
                f"No propagator-settings sidecar for body {trgt_id} about "
                f"origin {obsvr_id} in frame '{reference_frame}' was found. "
                f"Searched: {cls._sidecar_locations_searched(location)}. Pass "
                "`sidecar_path` pointing at the SPK, the sidecar file, or the "
                "folder holding it."
            )

        return settings_data

    @classmethod
    def get_attitude(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        tol: int,
        frame_id: int = None,
    ):
        """
        Get the attitude of a target body at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the attitude is to be determined. Can be an instance of Body or a string representing the body's name.
        epoch_time : float or EpochArray
            The epoch time at which the attitude is to be determined. Can be an instance of EpochArray or a float representing the time.
        reference_frame : str or Frame
            The reference frame in which the attitude is to be determined. Can be a string or an instance of Frame.
        tol : int
            Time tolerance for the C-kernel lookup, in spacecraft clock ticks.
        frame_id : int, optional
            The frame of the target body. If not given, frame ID is assumed to be the body's ID multiplied by 1000. Defaults
            to ``None``.

        Returns
        -------
        cmat : np.ndarray
            (3, 3) rotation matrix from ``reference_frame`` to the body frame,
            as returned by ``spice.ckgpav``.
        av : np.ndarray
            (3,) angular velocity of the body frame relative to
            ``reference_frame``, in rad/s, expressed in ``reference_frame``.
        clkout : float
            Spacecraft clock time, in ticks, of the returned pointing.
        """

        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time = cls._validate_epoch_input(epoch_time)
        # sce2c needs a Python float: an int epoch, a numpy scalar and the
        # (1,) array an EpochArray yields all come here.
        epoch_time = float(_np.asarray(epoch_time, dtype=float).ravel()[0])

        # Ensure target, reference_frame are the types SPICE expects
        # (trgt_bdy must be coerced to its int NAIF id before sce2c, which
        # requires a raw int spacecraft ID).
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.spice_id

        # --------------#
        # get attitude #
        # --------------#
        # Get s/c clock and attitude from SPICE
        t_clicks = spice.sce2c(trgt_bdy, epoch_time)  # Convert in clock ticks

        if not frame_id:
            # didn't get a frame ID -> assumed its ID * 1000
            frame_id = trgt_bdy * 1000

        state = spice.ckgpav(frame_id, t_clicks, tol, reference_frame)
        return (state[0], state[1], state[2])

    @staticmethod
    def get_observer_target_visibility_windows(
        obsvr_bdy: str | Body,
        trgt_bdy: str | Body,
        epoch_start: float | EpochArray,
        epoch_end: float | EpochArray,
        step_size: float = 300.0,
        angle_limit: float = 10.0,
        inframe: str = "obs",
        abcorr: str = "None",
        crdsys: str = "LATITUDINAL",
        coord: str = "LATITUDE",
        relate: str = ">",
        adjust: float = 0.0,
        num_intervals: int = 1000,
    ):
        """
        Determine the visibility windows between an observer and a target within a specified time range.

        Parameters
        ----------
        obsvr_bdy : Union[str, scb.Body]
            The observer, either as a string name or a scb.Body object.
        trgt_bdy : Union[str, scb.Body]
            The target, either as a string name or a scb.Body object.
        epoch_start : Union[float, scb.EpochArray]
            The start time of the observation window.
        epoch_end : Union[float, scb.EpochArray]
            he end time of the observation window.
        step_size : float, optional
            The step size in seconds for the search. Default is 300.0.
        angle_limit : float, optional
            The angular limit for visibility in degrees. Default is 10.0.
        inframe : str, optional
            The reference frame for the observation. Default is "obs".
        abcorr : str, optional
            Aberration correction. Default is "None".
        crdsys : str, optional
            Coordinate system. Default is "LATITUDINAL".
        coord : str, optional
            Coordinate type. Default is "LATITUDE".
        relate : str, optional
            Relational operator for the coordinate comparison. Default is ">".
        adjust : float, optional
            Adjustment value for the search. Default is 0.0.
        num_intervals : int, optional
            Maximum number of intervals for the search. Default is 1000.

        Returns
        -------
        tuple
            A tuple containing the number of visibility windows and a list of visibility windows.
        """

        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        cond_list = []
        for in_epoch, arg_name in zip(
            [epoch_start, epoch_end], ["epoch_start", "epoch_end"]
        ):
            if isinstance(in_epoch, EpochArray):
                # given an epocharray -> make sure singular value
                if in_epoch.size != 1:
                    bad_sze_err = (
                        f"Argument [{arg_name}] must be a single value. "
                        f"Received: {in_epoch}."
                    )
                    raise ValueError(bad_sze_err)
                else:
                    # single value -> extract numerical data
                    cond_list.append(arg_name)

            elif not (isinstance(in_epoch, float) or isinstance(in_epoch, int)):
                # anything besides an awu or float is invalid
                # if it's a float don't need to do any conditioning
                bad_type_err = (
                    f"Argument [{arg_name}] must be an object of type "
                    f"EpochArray, int, or float. Received: {type(in_epoch)}."
                )
                raise TypeError(bad_type_err)

        # condition either of the input epochs if necessary
        for to_cond in cond_list:
            match to_cond:
                case "epoch_start":
                    epoch_start = epoch_start.times.values
                case "epoch_end":
                    epoch_end = epoch_end.times.values

        # -------------#
        # get windows #
        # -------------#
        # Use the default frame centered on the observer unless otherwise specified
        if inframe == "obs":
            obsvr_name = obsvr_bdy.name if isinstance(obsvr_bdy, Body) else obsvr_bdy
            _, inframe = spice.cnmfrm(obsvr_name)

        # Spice window
        cnfine = spice.stypes.SPICEDOUBLE_CELL(2)

        # Set the start and end time of the search window
        spice.wninsd(epoch_start, epoch_end, cnfine)

        # Configure the output window optional parameter and set to 2x the max number of intervals
        result_spice_window = spice.stypes.SPICEDOUBLE_CELL(2 * num_intervals)

        # Ensure reference_frame, origin are a string
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        spice.gfposc(
            target=trgt_bdy,
            inframe=inframe,
            abcorr=abcorr,
            obsrvr=obsvr_bdy,
            crdsys=crdsys,
            coord=coord,
            relate=relate,
            refval=spice.rpd() * angle_limit,
            adjust=adjust,
            step=step_size,
            nintvls=num_intervals,
            cnfine=cnfine,
            result=result_spice_window,
        )

        # Extract the results of the search
        num_windows = spice.wncard(result_spice_window)

        # Instatiate an output
        visibility_windows = []

        # Loop through windows and extract the start and stop times
        if num_windows != 0:
            for i in range(num_windows):
                visibility_windows.append(spice.wnfetd(result_spice_window, i))

        return num_windows, visibility_windows

    # ------------------------------#
    # endregion Trajectory Methods #
    # ------------------------------#

    # -------------------------#
    # region Epoch Methods    #
    # -------------------------#
    @staticmethod
    def str2et(s: str) -> float:
        """Convert a time string to ephemeris time (TDB seconds past J2000).

        Thin wrapper around ``spice.str2et``.  Accepts a wide range of
        formats (ISO calendar, DOY, JD, etc.).

        Parameters
        ----------
        s : str
            Time string recognised by SPICE (e.g. ``"2017-09-22T16:52:00"``).
            Julian-Date strings are treated as JDUTC.

        Returns
        -------
        et : float
            TDB seconds past J2000 corresponding to *s*.
        """
        return spice.str2et(s)

    @staticmethod
    def utc2et(utc: str) -> float:
        """Convert a UTC time string to ephemeris time (TDB seconds past J2000).

        Thin wrapper around ``spice.utc2et``.

        Parameters
        ----------
        utc : str
            UTC time string in a format recognised by SPICE
            (e.g. ``"2017-09-22T16:52:00"``).

        Returns
        -------
        et : float
            Equivalent ephemeris time in TDB seconds past J2000.
        """
        return spice.utc2et(utc)

    @staticmethod
    def utc2tdb(utc: str) -> float:
        """Convert a UTC time string to TDB seconds past J2000.

        Alias for :meth:`utc2et`.

        Parameters
        ----------
        utc : str
            UTC time string in a format recognised by SPICE.

        Returns
        -------
        tdb : float
            TDB seconds past J2000.
        """
        return spice.utc2et(utc)

    @staticmethod
    def sclk2et(scId: int, sclk: str) -> float:
        """Convert a spacecraft clock string to ephemeris time.

        Thin wrapper around ``spice.scs2e``.

        Parameters
        ----------
        scId : int
            NAIF spacecraft integer ID (e.g. ``-64`` for OSIRIS-REx).
        sclk : str
            Spacecraft clock time string as it appears in the SCLK kernel.

        Returns
        -------
        et : float
            TDB seconds past J2000 corresponding to *sclk*.
        """
        return spice.scs2e(scId, sclk)

    @staticmethod
    def sce2c(sc: int, et: float) -> float:
        """Convert ephemeris time to encoded spacecraft clock ticks.

        Thin wrapper around ``spice.sce2c``.

        Parameters
        ----------
        sc : int
            NAIF spacecraft integer ID (e.g. ``-64`` for OSIRIS-REx).
        et : float
            TDB seconds past J2000 to convert.

        Returns
        -------
        sclk_ticks : float
            Encoded SCLK ticks corresponding to *et*.
        """
        return spice.sce2c(sc, et)

    @staticmethod
    def et2utc(et: float, form: str = "ISOC", prec: int = 27, utclen: int = 27) -> str:
        """Convert ephemeris time to a UTC string.

        Parameters
        ----------
        et : float
            TDB seconds past J2000.
        form : str, optional
            Output format code passed to ``spice.et2utc``.  Common values:
            ``"ISOC"`` (ISO calendar, default), ``"ISOD"`` (ISO day-of-year),
            ``"C"`` (calendar), ``"D"`` (day-of-year), ``"J"`` (Julian date).
        prec : int, optional
            Number of digits of precision for fractional seconds.  Default 27.
        utclen : int, optional
            Length of the output string buffer.  Default 27.

        Returns
        -------
        utc : str
            UTC time string in the requested format.
        """
        ans = spice.et2utc([et], form, prec, utclen)
        return ans[0]

    @staticmethod
    def et2sclk(scId: int, et: float, lenout: int = 30) -> str:
        """Convert ephemeris time to a spacecraft clock string.

        Thin wrapper around ``spice.sce2s``.

        Parameters
        ----------
        scId : int
            NAIF spacecraft integer ID.
        et : float
            TDB seconds past J2000 to convert.
        lenout : int, optional
            Maximum length of the output SCLK string.  Default 30.

        Returns
        -------
        sclk : str
            Spacecraft clock string corresponding to *et*.
        """
        return spice.sce2s(scId, et, lenout)

    @staticmethod
    def et2jd(et: float, form: str = "J", prec: int = 27, utclen: int = 27) -> str:
        """Convert ephemeris time to a Julian Date string (JDUTC).

        Parameters
        ----------
        et : float
            TDB seconds past J2000.
        form : str, optional
            Format code; default ``"J"`` produces a Julian Date string.
        prec : int, optional
            Digits of precision for fractional days.  Default 27.
        utclen : int, optional
            Length of the output string buffer.  Default 27.

        Returns
        -------
        jd : str
            Julian Date string.  The returned value is always JDUTC.
        """
        etArr = [et]
        ans = spice.et2utc(etArr, form, prec, utclen)
        return ans[0]

    @staticmethod
    def et2cal(et: float | NDArray):
        """Convert ephemeris time to a calendar UTC string.

        Accepts scalar floats, 0-D arrays, or N-D arrays and preserves the
        input shape in the output.

        Parameters
        ----------
        et : float or numpy.ndarray
            TDB seconds past J2000.  May be scalar or array-valued.

        Returns
        -------
        cal : str or numpy.ndarray of str
            Calendar UTC string(s) in ``"C"`` (civil) format.  A scalar
            *et* returns a single ``str``; a 1-D array returns a 1-D array
            of strings; a higher-dimensional array is reshaped back to the
            original shape.
        """
        # condition scalar inputs
        is_scalar, et_shape = False, None
        match et:
            case float():
                # convert scalar float to list
                et, is_scalar = [et], True
            case np.ndarray():
                if not et.shape:
                    # convert to list if single element
                    et, is_scalar = [et.tolist()], True
                else:
                    # flatten if more than 1 dimensional
                    et, et_shape = et.flatten().tolist(), et.shape

        # return in same form as input
        if is_scalar:
            # given scalar input
            return spice.et2utc(et, "C", 30, 50)[0]
        elif et_shape:
            # given more than 1 dimensional input
            return spice.et2utc(et, "C", 30, 50).reshape(et_shape)
        else:
            # given 1 dimensional input
            return spice.et2utc(et, "C", 30, 50)

    @staticmethod
    def cal2et(cal: str) -> float:
        """Convert a calendar UTC string to ephemeris time.

        Alias for :meth:`str2et`; accepts any time string format recognised
        by SPICE.

        Parameters
        ----------
        cal : str
            Calendar or ISO time string (e.g. ``"2017 SEP 22 16:52:00"``).

        Returns
        -------
        et : float
            TDB seconds past J2000.
        """
        return spice.str2et(cal)

    @staticmethod
    def jd2et(jd: float) -> float:
        """Convert a Julian Date (JDTDB) to ephemeris time.

        Parameters
        ----------
        jd : float
            Julian Date in the TDB time scale.

        Returns
        -------
        et : float
            TDB seconds past J2000.

        Notes
        -----
        A leap-second correction (``spice.deltet``) is added to account for
        the drift between JD and ET.
        """
        var1 = spice.unitim(jd, "JDTDB", "TDB")
        var2 = spice.deltet(var1, "ET")
        return var1 + var2

    @staticmethod
    def et2YDS(et_list):
        """
        Method to convert a list of ephemeris times into (year, DOY, seconds) format.

        Parameters
        ----------
        et_list : float or list of float
            List of ephemeris times

        Returns
        -------
        (years,doys,secs) : Tuple of lists
            List of years, day of years, and seconds associated to the input et_list

        """
        years = []
        doys = []
        secs = []
        # A single epoch is a bare float, which is not iterable on its own.
        for et in np.atleast_1d(et_list):
            # Convert ET to calendar string in DOY format: 'YYYY-DOY::HR:MN:SC'
            utc_str = spice.et2utc(et, "ISOD", 9)  # Precision is
            date_part, time_part = utc_str.split("T")
            year_str, doy_str = date_part.split("-")
            hh, mm, ss = map(float, time_part.split(":"))
            year = int(year_str)
            doy = int(doy_str)
            seconds = int(hh) * 3600 + int(mm) * 60 + ss
            years.append(year)
            doys.append(doy)
            secs.append(seconds)
        return years, doys, secs

    @staticmethod
    def YDS2et(year, doy, sec):
        """
        Method to convert a year, DOY, seconds into ephemeris times.

        Parameters
        ----------
        year : float or list of float
            Year

        doy : float or list of float
            Day of the year

        sec : float or list of float
            Second of the year

        Returns
        -------
        et : list
            List of ephemeris times

        """
        if not isinstance(year, (list, tuple)):
            year = [year]
        if not isinstance(doy, (list, tuple)):
            doy = [doy]
        if not isinstance(sec, (list, tuple)):
            sec = [sec]
        et = []
        for idx in range(len(sec)):
            trk_year = year[idx]
            trk_doy = doy[idx]
            trk_sec = float(sec[idx])
            # Convert seconds to HH:MM:SS
            hours = int(trk_sec // 3600)
            minutes = int((trk_sec % 3600) // 60)
            seconds = trk_sec % 60
            # Format time string: 'YYYY-DOY::HH:MM:SS.sssssssss' (fixed point: a
            # general format turns a rounding-sized second into '1.19e-07',
            # which SPICE cannot parse)
            tt_str = (
                f"{trk_year}-{trk_doy:03d}::{hours:02d}:{minutes:02d}:{seconds:012.9f}"
            )
            # Convert to ephemeris time
            tt_ET = spice.str2et(tt_str)
            et.append(tt_ET)
        return et

    # NOTE: convert_time is disabled and kept for reference (internal issue #315)
    # @staticmethod
    # def convert_time(self,
    #                 from_time : str | int | float,
    #                 from_sys  : Literal['str', 'et', 'sclk', 'utc', 'cal', 'jd'],
    #                 to_sys    : Literal['str', 'et', 'sclk', 'utc', 'cal', 'jd'],
    #                 format    : str = None, precision : int = None, len_out : int = None, sc_id : int = None):
    #     """
    #         Method that acts as a wrapper for all SPICE time subsystem routines.

    #         :param from_time: The given time to convert
    #         :type from_time: str | int | float

    #         :param from_sys: The time system that the given time is in
    #         :type from_sys: str

    #         :param to_sys: The time system to convert the given time to
    #         :type to_sys: str

    #         :param format: From SPICE: format of output time string:
    #                 - 'C'      Calendar format, UTC
    #                 - 'D'      Day-of-Year format, UTC
    #                 - 'J'      Julian Date format, UTC
    #                 - 'ISOC'   ISO Calendar format, UTC.
    #                 - 'ISOD'   ISO Day-of-Year format, UTC.

    #                 Defaults to value depending on given time system to convert to:
    #                 - :code:`'ISOC'`   When converting from ephemeris time to UTC
    #                 - :code:`'J'`      When converting from ephemeris time to Julian Date
    #                 - :code:`'C'`      When converting from ephemeris time to Calendar Date
    #         :type format: str, optional

    #         :param precision: From SPICE: The number of digits of precision to which
    #             fractional seconds (for Calendar and Day-of-Yearformats) or days
    #             (for Julian Date format) are to be computed. If zero or smaller,
    #             no decimal point is appended to the output string. If greater
    #             than 14, it is treated as 14.

    #             Defaults to value depending on given time system to convert to:
    #                 - :code:`27`   When converting from ephemeris time to UTC or Julian Date
    #                 - :code:`30`      When converting from ephemeris time to Calendar Date
    #         :type precision: int, optional

    #         :param len_out: From SPICE: the length of the output string plus 1.

    #             Defaults to value depending on given time system to convert to:
    #                 - :code:`27`     When converting from ephemeris time to UTC, S/C Clock, or Julian Date
    #                 - :code:`50`     When converting from ephemeris time to Calendar Date
    #         :type len_out: int, optional

    #         :param sc_id: From SPICE: NAIF spacecraft clock ID code. Defaults to :code:`None`
    #         :type sc_id: int, optional

    #         .. seealso:: `SPICE time subsystem routines <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/time.html#Time%20Subsystem%20Routines>`__
    #     """
    #     #-----------------------------------------------------------#
    #     # return converted value depending on requested time system #
    #     #-----------------------------------------------------------#
    #     match from_sys:
    #         case 'str':
    #             # converting from string input...
    #             if to_sys == 'et':
    #                 #-----------------------------#
    #                 # ...to ephemeris time output #
    #                 #-----------------------------#
    #                 return spice.str2et(from_time)
    #             else:
    #                 # no other format conversion options, raise error
    #                 raise ValueError(f'Cannot convert from {from_sys} to {to_sys}')

    #         case 'et':
    #             # converting from ephemeris time input...
    #             if to_sys == 'utc':
    #                 #----------------------------------#
    #                 # ...to coordinated universal time #
    #                 #----------------------------------#
    #                 # default values if none given
    #                 if isinstance(format, None):
    #                     format = 'ISOC'
    #                 if isinstance(precision, None):
    #                     precision = 27
    #                 if isinstance(len_out, None):
    #                     len_out = 27

    #                 # call SPICE function
    #                 ans = spice.et2utc([from_time], format, precision, len_out)
    #                 return ans[0]
    #             elif to_sys == 'sclk':
    #                 #-----------------------------#
    #                 # ...to spacecraft clock time #
    #                 #-----------------------------#
    #                 # default values if none given
    #                 if isinstance(len_out, None):
    #                     len_out = 27

    #                 return spice.scs2e(sc_id, from_time, len_out)

    #             elif to_sys == 'jd':
    #                 #-------------------#
    #                 # ...to julian date #
    #                 #-------------------#
    #                 # default values if none given
    #                 if isinstance(format, None):
    #                     format = 'J'
    #                 if isinstance(precision, None):
    #                     precision = 27
    #                 if isinstance(len_out, None):
    #                     len_out = 27

    #                 # call SPICE function
    #                 ans = spice.et2utc([from_time], format, precision, len_out)
    #                 return ans[0]

    #             elif to_sys == 'cal':
    #                 #---------------------#
    #                 # ...to calendar date #
    #                 #---------------------#
    #                 # default values if none given
    #                 if isinstance(format, None):
    #                     format = 'C'
    #                 if isinstance(precision, None):
    #                     precision = 30
    #                 if isinstance(len_out, None):
    #                     len_out = 50

    #                 # call SPICE function
    #                 ans = spice.et2utc([from_time], format, precision, len_out)
    #                 return ans[0]

    #             else:
    #                 # no other format conversion options, raise error
    #                 raise ValueError(f'Cannot convert from {from_sys} to {to_sys}')

    #         case the rest:
    #             rest of the conversions...

    # endregion Epoch Methods #
    # -------------------------#

    # -------------------------#
    # region Frame Methods    #
    # -------------------------#
    @staticmethod
    def frame_name_to_frame_id(frame_name: str) -> int:
        """Return the SPICE integer frame ID for a given frame name.

        Parameters
        ----------
        frame_name : str
            SPICE name of the reference frame (e.g. ``"J2000"``).

        Returns
        -------
        frame_id : int
            SPICE integer frame ID corresponding to *frame_name*.

        Notes
        -----
        The frame ID is distinct from the body NAIF ID used in SPK/PCK
        kernels.
        """
        return spice.namfrm(frame_name)

    @staticmethod
    @lru_cache(maxsize=1024)
    def frame_id_to_frame_name(frame_id: int) -> str:
        """Return the SPICE frame name for a given integer frame ID.

        Results are cached with :func:`functools.lru_cache` for performance.

        Parameters
        ----------
        frame_id : int
            SPICE integer frame ID to resolve.

        Returns
        -------
        frame_name : str
            SPICE name of the reference frame.

        Notes
        -----
        The frame ID is distinct from the body NAIF ID used in SPK/PCK
        kernels.
        """
        return spice.frmnam(frame_id)

    @lru_cache
    @staticmethod
    def list_all_frames(max_id: int = None) -> list:
        """List all SPICE reference frames currently detectable in the pool.

        This includes all built in SPICE frames.

        Parameters
        ----------
        max_id : int, optional
            Keep only frames whose NAIF ID is at most this value. Enumerating
            every loaded frame is slow once a large kernel set is on the pool,
            so a caller that only needs the low-numbered inertial frames can
            cap the search. Defaults to no limit.

        Returns
        -------
        frames_list : list of str
            Names of all reference frames found.
        """
        all_frames = []
        for frame in (spice.bltfrm, spice.kplfrm):
            # check built in (bltfrm) and loaded (kplfrm)
            for frame_class in range(1, 6):  # SPICE frame classes are 1-5
                # check for each frame class
                for frame_id in frame(frame_class):
                    if max_id is not None and frame_id > max_id:
                        continue
                    # get name of the frame and save
                    all_frames.append(SpiceManager.frame_id_to_frame_name(frame_id))

        return all_frames

    @staticmethod
    def list_all_ck_ids(ck_file: str) -> list[int]:
        """Return all NAIF IDs present in a C-kernel file.

        Parameters
        ----------
        ck_file : str
            Absolute or relative path to the CK file.

        Returns
        -------
        ids : list of int
            NAIF integer IDs of every object stored in *ck_file*.
        """
        return spice.ckobj(ck_file)

    @staticmethod
    def list_all_bodies() -> list[int]:
        """List every body found across all loaded SPK kernels.

        Iterates over each SPK kernel currently loaded in the SPICE kernel
        pool, and for every kernel prints its filename followed by the NAIF
        integer ID and resolved name of each body it contains. Bodies whose
        ID cannot be resolved to a name are printed as ``(no name)``.

        Returns
        -------
        all_ids : list of int
            Sorted, de-duplicated NAIF integer IDs of every body found
            across the loaded SPK kernels.
        """
        all_ids = set()
        n_files = spice.ktotal("SPK")
        for i in range(n_files):
            filename, filtyp, srcfil, handle = spice.kdata(i, "SPK")
            print(f"\n--- {filename} ---")
            ids = spice.spkobj(filename)
            for body_id in ids:
                all_ids.add(int(body_id))
                try:
                    name = spice.bodc2n(body_id)
                    print(f"  {body_id:>10}: {name}")
                except:
                    print(f"  {body_id:>10}: (no name)")

        return sorted(all_ids)

    @classmethod
    def get_xfrm(
        cls,
        frame_from: str,
        frame_to: str,
        epoch,
    ):
        """
        Computes the direction cosine matrix (DCM) to transform coordinates from one reference frame to another at a given epoch.

        Parameters
        ----------
        frame_from : str
            The name of the source reference frame.
        frame_to : str
            The name of the target reference frame.
        epoch :
            The epoch at which the transformation is computed. The type depends on the implementation (e.g., float, datetime).

        Returns
        -------
        numpy.ndarray
            The direction cosine matrix (DCM) representing the transformation from `frame_from` to `frame_to` at the specified epoch.
        """

        DCM = cls._get_xfrm_core(
            frame_from,
            frame_to,
            epoch,
        )

        return DCM

    @classmethod
    def _get_xfrm_callback(
        cls,
        frame_from,
        frame_to,
    ):
        def spice_callback(epoch):
            result = cls._get_xfrm_core(
                frame_from,
                frame_to,
                float(epoch),
            )
            return result

        return spice_callback

    @staticmethod
    def _get_xfrm_core(frame_from: str, frame_to: str, epoch) -> np.array:
        """Return the DCM :math:`\\mathbf{R}` that rotates *frame_from* into *frame_to*.

        Wraps ``spice.pxform`` to return the direction cosine matrix such that

        .. math::

            \\mathbf{v}_{\\text{to}} = \\mathbf{R}\\,\\mathbf{v}_{\\text{from}}

        Parameters
        ----------
        frame_from : str
            Name of the source SPICE reference frame.
        frame_to : str
            Name of the target SPICE reference frame.
        epoch : float
            TDB epoch in seconds past J2000 at which the rotation is evaluated.

        Returns
        -------
        DCM : numpy.ndarray
            Shape ``(3, 3)`` direction cosine matrix.

        Raises
        ------
        spiceypy.utils.exceptions.SpiceyError
            If either frame name is not recognised by the loaded kernels.
        """
        DCM = spice.pxform(frame_from, frame_to, epoch)
        return DCM

    @staticmethod
    def _write_pck_frame(
        file_name: str, frame_name: str, frame_class: int, frame_id: int
    ):
        """Write a PCK SPICE frame definition file.

        Generates a text PCK kernel that registers a new body-fixed reference
        frame.  See the NAIF frames documentation for the PCK frame class
        specification.

        Parameters
        ----------
        file_name : str
            Output file path.  The ``.pck`` extension is appended if absent.
        frame_name : str
            SPICE name to assign to the new frame (e.g. ``"EROS_FIXED"``).
        frame_class : int
            SPICE frame class code (e.g. ``2`` for body-fixed PCK frames).
        frame_id : int
            Unique SPICE integer frame ID (e.g. ``2000433``).

        Raises
        ------
        RuntimeError
            If *frame_id* is already registered in the SPICE pool.
        """
        # Ensure the file name has the correct extension
        if not file_name.endswith(".pck"):
            file_name += ".pck"
        # Define the content of the frame definition file
        content = f"""
        \\begindata
            FRAME_{frame_name}       =  {frame_id}
            FRAME_{frame_id}_NAME    = '{frame_name}'
            FRAME_{frame_id}_CLASS   =  {frame_class}
            FRAME_{frame_id}_CLASS_ID =  {frame_id}
            FRAME_{frame_id}_CENTER  =  {frame_id}

            OBJECT_{frame_id}_FRAME  = '{frame_name}'
        \\begintext
        """

        # Check if the frame is already in the SPICE pool
        # (spice.frmnam never actually raises -- it returns "" for an
        # unregistered frame_id -- so no try/except is needed here)
        existing_frame_name = spice.frmnam(frame_id)
        if existing_frame_name:
            raise RuntimeError(
                f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                f"Please choose a different frame ID or unregister the existing frame."
            )

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists. It will be overwritten.")

        # Write the content to the file
        with open(file_name, "w") as file:
            file.write(content.strip())
        print(f"Frame definition file '{file_name}' has been created.")

    @staticmethod
    def _write_ck_frame(
        file_name: str,
        frame_name: str,
        frame_id: int,
        center_id: int,
        sclk_id: int,
        spk_id: int,
    ):
        """Write a CK SPICE frame definition file.

        Generates a text FK kernel that registers a new spacecraft
        attitude-based (CK) reference frame.

        Parameters
        ----------
        file_name : str
            Output file path.  The ``.tf`` extension is appended if absent.
        frame_name : str
            SPICE name to assign to the new frame (e.g. ``"MGS_SPACECRAFT"``).
        frame_id : int
            Unique SPICE integer frame ID (e.g. ``-94000``).
        center_id : int
            NAIF body ID of the frame's center body (e.g. ``-94``).
        sclk_id : int
            Spacecraft clock ID associated with the CK data (e.g. ``-94``).
        spk_id : int
            SPK body ID associated with the frame (e.g. ``-94``).

        Raises
        ------
        RuntimeError
            If *frame_id* is already registered in the SPICE pool.
        """
        # Ensure the file name has the correct extension
        if not file_name.endswith(".tf"):
            file_name += ".tf"

        # Frame definition content
        content = f"""
        \\begindata

            FRAME_{frame_name}       =  {frame_id}
            FRAME_{frame_id}_NAME    = '{frame_name}'
            FRAME_{frame_id}_CLASS   =  3
            FRAME_{frame_id}_CLASS_ID =  {frame_id}
            FRAME_{frame_id}_CENTER  =  {center_id}

            CK_{frame_id}_SCLK       =  {sclk_id}
            CK_{frame_id}_SPK        =  {spk_id}

            OBJECT_{center_id}_FRAME = '{frame_name}'

        \\begintext
        """

        # Check if the frame is already in the SPICE pool
        # (spice.frmnam never actually raises -- it returns "" for an
        # unregistered frame_id -- so no try/except is needed here)
        existing_frame_name = spice.frmnam(frame_id)
        if existing_frame_name:
            raise RuntimeError(
                f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                f"Please choose a different frame ID or unregister the existing frame."
            )

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists. It will be overwritten.")

        # Write to the file
        with open(file_name, "w") as file:
            file.write(content.strip())

        print(f"CK frame definition file '{file_name}' has been created.")

    @staticmethod
    def _write_tk_frame(
        file_name: str,
        frame_name: str,
        frame_id: int,
        center_id: int,
        relative_frame: int,
        matrix=None,
    ):
        """Write a TK (text-kernel) SPICE frame definition file.

        Generates a text FK kernel that registers a new constant-offset (TK)
        reference frame defined by a fixed rotation matrix relative to an
        existing frame.

        Parameters
        ----------
        file_name : str
            Output file path.  The ``.tf`` extension is appended if absent.
        frame_name : str
            SPICE name to assign to the new frame (e.g. ``"MARS_FIXED"``).
        frame_id : int
            Unique SPICE integer frame ID (e.g. ``1400499``).
        center_id : int
            NAIF body ID of the frame's center body (e.g. ``499``).
        relative_frame : str
            Name of the parent reference frame (e.g. ``"IAU_MARS"``).
        matrix : list of list of float, optional
            :math:`3 \\times 3` rotation matrix from *relative_frame* to this
            frame.  Defaults to the identity matrix.
        """
        # Ensure the file name has the correct extension
        if not file_name.endswith(".tf"):
            file_name += ".tf"

        # Use identity matrix as the default
        if matrix is None:
            matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Convert the matrix into a flattened string representation
        matrix_flat = " ".join(map(str, [elem for row in matrix for elem in row]))

        # Frame definition content
        content = f"""
        \\begindata

            FRAME_{frame_name}       =  {frame_id}
            FRAME_{frame_id}_NAME    = '{frame_name}'
            FRAME_{frame_id}_CLASS   =  4
            FRAME_{frame_id}_CLASS_ID =  {frame_id}
            FRAME_{frame_id}_CENTER  =  {center_id}

            OBJECT_{center_id}_FRAME = '{frame_name}'

        \\begintext

        \\begindata

            TKFRAME_{frame_id}_RELATIVE = '{relative_frame}'
            TKFRAME_{frame_id}_SPEC     = 'MATRIX'
            TKFRAME_{frame_id}_MATRIX   = ( {matrix_flat} )

        \\begintext
        """

        # Check if the frame is already in the SPICE pool
        # (spice.frmnam never actually raises -- it returns "" for an
        # unregistered frame_id -- so no try/except is needed here)
        existing_frame_name = spice.frmnam(frame_id)
        if existing_frame_name:
            raise RuntimeError(
                f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                f"Please choose a different frame ID or unregister the existing frame."
            )

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists. It will be overwritten.")

        # Write to the file
        with open(file_name, "w") as file:
            file.write(content.strip())
        print(f"TK frame definition file '{file_name}' has been created.")

    @staticmethod
    def get_frame_w_spice_id(id_number: int):
        """Return the frame code and name associated with a NAIF body ID.

        Parameters
        ----------
        id_number : int
            NAIF integer body ID to look up (e.g. ``499`` for Mars).

        Returns
        -------
        result : tuple of (int, str)
            ``(frame_code, frame_name)`` where *frame_code* is the integer
            SPICE frame ID and *frame_name* is its string identifier.

        Raises
        ------
        ValueError
            If *id_number* has no associated frame in the kernel pool.
        """
        try:
            frame_code, frame_name = spice.cidfrm(id_number)
        except spice.stypes.SpiceyError:
            raise ValueError(f"Frame '{str(id_number)}' is not recognized.")
        return (frame_code, frame_name)

    @staticmethod
    def matrix_times_vector(matrix, vector):
        """Multiply a :math:`3 \\times 3` matrix by a 3-vector via SPICE.

        Computes :math:`\\mathbf{w} = \\mathbf{M}\\,\\mathbf{v}` using
        ``spice.mxv`` (CSPICE ``mxv_c``).

        Parameters
        ----------
        matrix : numpy.ndarray
            Shape ``(3, 3)`` double-precision matrix (e.g. a rotation matrix).
        vector : numpy.ndarray
            Shape ``(3,)`` double-precision vector.

        Returns
        -------
        result : numpy.ndarray
            Shape ``(3,)`` product :math:`\\mathbf{M}\\mathbf{v}`.
        """
        return spice.mxv(matrix, vector)

    # endregion Frame Methods #
    # -------------------------#

    # ------------------------#
    # region Body Methods    #
    # ------------------------#
    @staticmethod
    def get_id_from_string(name_string: str) -> int:
        """Return the NAIF integer ID for a body given its SPICE name.

        Parameters
        ----------
        name_string : str
            SPICE body name (e.g. ``"EARTH"``, ``"MARS"``, ``"-60000"``).

        Returns
        -------
        id_num : int
            NAIF integer ID of the body.
        """
        id_num = spice.bods2c(name_string)
        return id_num

    @staticmethod
    def get_name_from_id(spice_id: int) -> str:
        """Return the SPICE body name for a given NAIF integer ID.

        Parameters
        ----------
        spice_id : int
            NAIF integer body ID (e.g. ``399`` for Earth).

        Returns
        -------
        name : str
            SPICE body name corresponding to *spice_id*.
        """
        return spice.bodc2n(spice_id)

    @staticmethod
    def get_intervals(spkFile: str, objID: int) -> list[str]:
        """Return the coverage interval of a body in an SPK file as JD strings.

        Retrieves the first (and typically only) time window for *objID* from
        the SPK coverage window and converts the endpoints to Julian Date
        strings.

        Parameters
        ----------
        spkFile : str
            Full path to the SPK file.  The file must already be accessible
            on disk (it does not need to be loaded into the kernel pool).
        objID : int
            NAIF integer body ID whose coverage is requested.

        Returns
        -------
        interval : list of str
            ``[start_jd, end_jd]`` where each element is a Julian Date string
            produced by ``spice.et2utc``.
        """
        cover = spice.spkcov(spkFile, objID)
        end_times = spice.wnfetd(cover, 0)
        start = spice.et2utc([end_times[0]], "J", 6, 30)[0]
        end = spice.et2utc([end_times[1]], "J", 6, 30)[0]
        return [start, end]

    # endregion Body Methods #
    # ------------------------#

    # --------------------------#
    # region Static Methods    #
    # --------------------------#
    @staticmethod
    def load_kernel(kernel_filenames):
        """
        Load SPICE kernels into the SPICE system.

        Parameters
        ----------
        kernel_filenames : str or list
            A single kernel filename or a list of kernel filenames to be loaded.

        Returns
        -------
        None
        """
        # Check if the input is a string or a list
        if isinstance(kernel_filenames, str):
            kernel_filenames = [kernel_filenames]

        # Load the kernels
        for kernel_filename in kernel_filenames:
            spice.furnsh(kernel_filename)

    @staticmethod
    def load_kernel_from_mkfile(furnshKernelFilename):
        """
        Load a SPICE kernel from a given MK file.

        Parameters
        ----------
        furnshKernelFilename : str
            The filename of the MK file to load.

        Returns
        -------
        None
        """
        spice.furnsh(furnshKernelFilename)

    @staticmethod
    def unload_kernel_from_pool(kernel_filename: str):
        """
        Unload a kernel from the pool. If the kernel is not in the
        pool then no action is taken.

        Parameters
        ----------
        kernel_filename : str
            Filename of the Spice kernel being unloaded.
        """
        spice.unload(kernel_filename)

    @staticmethod
    def check_kernel_status_in_pool(kernel_filename: str) -> bool:
        """
        Check if a Spice kernel is loaded in the kernel pool.

        Parameters
        ----------
        kernel_filename : str
            Filename of the kernel to be checked.

        Returns
        -------
        bool
            Returns True if the kernel is in the pool and False if not.
        """
        # Get the total number of loaded kernels
        num_kernels = spice.ktotal("ALL")

        # Loop through all the loaded kernels and check if the kernel_filename is in the pool
        # spice.kdata returns either (file, type, source, handle, found) or
        # (file, type, source, handle) depending on spiceypy version/config.
        for i in range(num_kernels):
            file = spice.kdata(i, "ALL")[0]
            if kernel_filename in file:
                return True

        return False

    @staticmethod
    def spk_files_for_body(spice_id, epochs=None) -> list:
        """
        List the loaded SPK files that provide ephemeris for a body.

        Useful for recording which trajectory a computation actually used,
        since a body's ephemeris is resolved from global kernel-pool state
        rather than from anything passed in at the call site.

        Parameters
        ----------
        spice_id : int or str
            NAIF ID of the body.
        epochs : array_like, optional
            ET epochs, in seconds. When given, only the files that resolve the
            body at one of these epochs are listed: for each epoch, the
            last-loaded file whose coverage contains it. Defaults to ``None``.

        Returns
        -------
        list of str
            Paths of the loaded SPKs covering this body, in load order. Without
            ``epochs``, SPK precedence (last-loaded-wins) makes the final entry
            the one that resolves wherever its coverage reaches, but an earlier
            file still resolves epochs outside it.
        """
        try:
            target = int(spice_id)
        except (TypeError, ValueError):
            return []

        files, windows = [], []
        try:
            for i in range(spice.ktotal("SPK")):
                path = spice.kdata(i, "SPK")[0]
                try:
                    if target in spice.spkobj(path):
                        files.append(path)
                        if epochs is not None:
                            cover = spice.spkcov(path, target)
                            windows.append([
                                spice.wnfetd(cover, k)
                                for k in range(spice.wncard(cover))
                            ])
                except Exception:
                    # Unreadable or non-SPK entry; skip rather than fail the
                    # caller, since this is diagnostic information.
                    continue
        except Exception as e:
            print(f"Error listing SPK files for body {spice_id}: {e}")

        if epochs is None:
            return files

        # Walk the files newest first, as SPICE does, giving each epoch to
        # the first one that covers it
        unresolved = _np.unique(_np.asarray(epochs, dtype=float).ravel())
        resolving = set()
        for idx in reversed(range(len(files))):
            if unresolved.size == 0:
                break
            covered = _np.zeros(unresolved.shape, dtype=bool)
            for start, stop in windows[idx]:
                covered |= (unresolved >= start) & (unresolved <= stop)
            if covered.any():
                resolving.add(idx)
                unresolved = unresolved[~covered]
        return [files[idx] for idx in sorted(resolving)]

    @staticmethod
    def increase_kernel_priority(kernel_filename: str):
        """
        This function increases the priority of kernel
        to the top of the pool for kernels of the same type
        by unloading then loading it.

        Note: this is the method recommended by the Spice
        documentation.

        Parameters
        ----------
        kernel_filename : str
            Filename of the
        """
        spice.unload(kernel_filename)
        spice.furnsh(kernel_filename)

    @staticmethod
    def def_new_body(name, spiceId):
        """
        Define a new celestial body in the SPICE kernel.

        Parameters
        ----------
        name : str
            The name of the celestial body.
        spiceId : int
            The SPICE ID of the celestial body.

        Returns
        -------
        None
        """
        spice.boddef(name, spiceId)

    @classmethod
    def write_spk_segment_type13(
        cls,
        SPK_name: str,
        SPK_SEG_ID: str,
        body: Body,
        origin_ID: float,
        ref_frame: Frame,
        epochs_TDB: list,
        degree_poly: int,
        state_SPICE: list,
        parameters_JSON: list = None,
        parameters_units: list = None,
        STMs_JSON: list = None,
        STM_units: dict = None,
        state_definition: list = None,
        sequence_definition: list = None,
        leg_data: dict = None,
        propagator_settings: dict = None,
        mass_profile: SCBPolynomial = None,
        stm_format: str = "json",
        sidecar_format: str = "json",
        sidecar_name: str = None,
    ):
        """
        Write a segment of type 13 to an SPK file.

        Type 13 stores the same ``(epoch, position, velocity)`` records as type 9,
        in the same number of bytes, but interpolates them with Hermite polynomials
        that use the sampled velocities as derivative constraints on the position
        curve. The returned velocity is then the exact derivative of the returned
        position, and at ``degree_poly=3`` the two-node window makes the interpolant
        C1-continuous across every node, removing the jumps a shifting type-9
        Lagrange window produced at each node crossing.

        Parameters
        ----------
        SPK_name : str
            The name of the SPK file.
        SPK_SEG_ID : str
            The segment ID.
        body : object
            The body object.
        origin_ID : int
            The ID of the origin.
        ref_frame : frame
            The reference frame.
        epochs_TDB : list
            The array of epochs in TDB format.
        degree_poly : int
            The degree of the polynomial.
        state_SPICE : list
            The array of states in SPICE format.
        parameters_JSON : list, optional
            The array of parameters in JSON format.
        STMs_JSON : list, optional
            The array of state transition matrices in JSON format.
        state_definition : list, optional
            The state definition.
        sequence_definition : object, optional
            The sequence definition.
        leg_data : dict, optional
            The body, origin, and reference frame data for each leg.
        propagator_settings : dict, optional
            The propagator settings.
        mass_profile : SCBPolynomial, optional
            The mass profile.

        Returns
        -------
        None

        Raises
        ------
        EvenDegreeError
            If ``degree_poly`` is even. Type 13 builds each Hermite polynomial from
            a position and a velocity per node, so its degree is always odd.
        TooFewStatesError
            If the segment holds fewer than ``(degree_poly + 1) // 2`` states, which
            is the window SPK type 13 needs to build its interpolating polynomial.
        """
        # Initialize
        body_ID = body.spice_id
        num_states = len(epochs_TDB)

        # Checked before the file is opened, rather than left to CSPICE, whose
        # SPICE(INVALIDDEGREE) names neither the segment nor the way out.
        if degree_poly % 2 == 0:
            # The odd degrees either side, keeping only the ones SPICE would take:
            # degree 0 has no odd neighbour below it, so don't offer -1.
            options = sorted({max(degree_poly - 1, 1), max(degree_poly + 1, 1)})
            advice = " or ".join(str(degree) for degree in options)
            raise cls.EvenDegreeError(
                f"SPK segment '{SPK_SEG_ID}' (body {body_ID}) asks for degree "
                f"{degree_poly}, but type 13 interpolates with Hermite polynomials "
                "built from a position and a velocity per node, so the degree is "
                f"always odd. Use {advice}."
            )

        # A type-13 window spans (degree + 1) / 2 states, half what the same degree
        # costs under type 9, because every node also supplies a derivative.
        min_states = (degree_poly + 1) // 2

        # Checked before the file is opened, rather than left to CSPICE, whose
        # SPICE(TOOFEWSTATES) names neither the segment nor the way out.
        if num_states < min_states:
            raise cls.TooFewStatesError(
                f"SPK segment '{SPK_SEG_ID}' (body {body_ID}) holds {num_states} "
                f"state(s), but a degree-{degree_poly} Hermite polynomial needs at "
                f"least {min_states}. Propagate the segment on a finer output grid "
                "(more epochs), or lower the Trajectory's `degree_poly`."
            )

        # Handle SPK file opening
        SPK_handle = cls._open_spk_file(SPK_name, body_ID, epochs_TDB)

        # Write SPK segment
        cls._write_spk_segment(
            SPK_handle,
            body_ID,
            origin_ID,
            ref_frame,
            epochs_TDB,
            degree_poly,
            num_states,
            state_SPICE,
            SPK_name,
            SPK_SEG_ID,
        )

        # Sidecar extension follows the requested format (".json" or ".pkl").
        side_ext = (
            ".pkl" if str(sidecar_format).lower() in ("pickle", "pkl") else ".json"
        )

        # Write parameters sidecar only when parameters were actually estimated
        if cls._has_parameter_payload(parameters_JSON):
            data = cls._prepare_parameters_data(
                body_ID,
                origin_ID,
                ref_frame,
                epochs_TDB,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
                parameters_units=parameters_units,
            )
            # Sidecars follow ``sidecar_name`` rather than the SPK, so several
            # bodies sharing one .bsp cannot overwrite each other's parameters.
            # Readers locate a sidecar by body id, not by filename.
            cls._write_parameters_to_json(
                sidecar_name or SPK_name, data, f"_parameters{side_ext}"
            )

        # Write STMs sidecar if provided
        if STMs_JSON is not None:
            stm_data = cls._prepare_stm_data(
                STMs_JSON,
                epochs_TDB,
                body_ID,
                origin_ID,
                ref_frame,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
                STM_units=STM_units,
            )
            cls._write_stm_sidecar(SPK_name, stm_data, stm_format)

        # Write propagator settings sidecar if provided
        if propagator_settings is not None:
            cls._write_parameters_to_json(
                SPK_name, propagator_settings, f"_settings{side_ext}"
            )

        # Write mass profile sidecar if provided
        if mass_profile is not None:
            mass_profile_data = cls._prepare_mass_profile_data(body, mass_profile)
            cls._write_parameters_to_json(
                SPK_name, mass_profile_data, f"_mass_profile{side_ext}"
            )

        # Close SPK file
        spice.spkcls(SPK_handle)

        # Load the SPK file
        spice.furnsh(SPK_name)

    @classmethod
    def _open_spk_file(cls, SPK_name, body_ID, epochs_TDB):
        """
        Open an SPK file for writing or appending.
        IMPORTANT: CSPICE cannot spkopa an SPK that is currently loaded (furnsh).

        Appending is the ordinary way to grow an SPK, so an existing file is not
        itself worth a warning. Re-covering epochs the file already holds for
        this body is: segments do not merge, the later one simply outranks the
        earlier, so the states already on file quietly stop being the ones SPICE
        returns over the overlap.

        Parameters
        ----------
        SPK_name : str
            Path of the SPK to create or append to.
        body_ID : int
            SPICE ID of the body whose segment is about to be written; only this
            body's existing coverage can be shadowed by it.
        epochs_TDB : list
            Epochs of the segment about to be written, ascending. Its first and
            last entries are the coverage being claimed.
        """
        # Ensure not loaded before opening for write/append
        # (spice.unload is a no-op for a file that isn't loaded, so no
        # try/except is needed here)
        spice.unload(SPK_name)

        if not os.path.isfile(SPK_name):
            return spice.spkopn(SPK_name, SPK_name, 0)

        # Read the coverage before spkopa: the file has to be opened read-only
        # to be queried, and it cannot be open twice at once.
        overlaps = cls._overlapping_coverage(
            SPK_name, body_ID, epochs_TDB[0], epochs_TDB[-1]
        )
        if overlaps:
            windows = "; ".join(
                f"{spice.etcal(lo)} to {spice.etcal(hi)} TDB" for lo, hi in overlaps
            )
            print(
                f"Warning: SPK file '{SPK_name}' already covers body {body_ID} over "
                f"{windows}. The segment being appended takes priority there, so the "
                "states already on file will no longer be the ones returned."
            )

        return spice.spkopa(SPK_name)

    @classmethod
    def _overlapping_coverage(cls, SPK_name, body_ID, t0, tf):
        """
        Parts of an existing SPK's coverage that a new segment would write over.

        Returns the intervals of ``SPK_name``'s current coverage for ``body_ID``
        that intersect ``[t0, tf]``, as ``(start, stop)`` pairs in TDB seconds,
        empty when the new segment breaks no ground. Segments that merely touch
        -- one ending exactly where the next begins -- share a single instant
        rather than an interval and are not reported, since chaining legs end to
        end is how a sequence is written, not a collision.

        Parameters
        ----------
        SPK_name : str
            Path of an existing SPK, not currently open for writing.
        body_ID : int
            SPICE ID to look up. A body absent from the file has empty coverage,
            which ``spkcov`` reports without complaint.
        t0, tf : float
            Start and end of the segment about to be written, TDB seconds.
        """
        cover = spice.spkcov(
            SPK_name, int(body_ID), spice.cell_double(cls._spk_coverage_cell_size)
        )

        overlaps = []
        for interval in range(spice.wncard(cover)):
            start, stop = spice.wnfetd(cover, interval)
            lo, hi = max(start, t0), min(stop, tf)
            if hi > lo:
                overlaps.append((lo, hi))

        return overlaps

    @staticmethod
    def _write_spk_segment(
        SPK_handle,
        body_ID,
        origin_ID,
        ref_frame,
        epochs_TDB,
        degree_poly,
        num_states,
        state_SPICE,
        SPK_name,
        SPK_SEG_ID,
    ):
        """
        Write a trajectory segment to an SPK file as a type-13 (unequally spaced
        Hermite) segment. The argument list matches ``spkw09``; only the way the
        records are interpolated on read-back differs.

        Parameters
        ----------
        SPK_handle : int
            Handle to the SPK file.
        body_ID : int
            The ID of the body.
        origin_ID : int
            The ID of the origin.
        ref_frame : str
            The reference frame.
        epochs_TDB : list
            The array of epochs in TDB format.
        degree_poly : int
            The degree of the polynomial.
        num_states : int
            The number of states.
        state_SPICE : list
            The array of states in SPICE format.
        SPK_name : str
            The name of the SPK file.

        Returns
        -------
        None
        """
        spice.spkw13(
            SPK_handle,
            body_ID,
            origin_ID,
            ref_frame.name,
            epochs_TDB[0],
            epochs_TDB[-1],
            SPK_SEG_ID,
            degree_poly,
            num_states,
            state_SPICE,
            epochs_TDB,
        )

    @classmethod
    def write_spk_metadata_json(
        cls,
        SPK_name: str,
        body: Body,
        origin_ID: float,
        ref_frame: Frame,
        epochs_TDB: list,
        degree_poly: int,
        parameters_JSON: list = None,
        parameters_units: list = None,
        STMs_JSON: list = None,
        STM_units: dict = None,
        state_definition: list = None,
        sequence_definition: list = None,
        leg_data: dict = None,
        propagator_settings: dict = None,
        mass_profile: SCBPolynomial = None,
        stm_format: str = "json",
        sidecar_format: str = "json",
        sidecar_name: str = None,
    ):
        """
        Write ONLY the JSON sidecars associated with an SPK (parameters/STMs/settings/mass profile).

        This avoids writing any additional SPK segments (no dummy segments),
        and avoids SPICE descriptor-time errors.

        Notes
        -----
        - epochs_TDB is included so JSON has a time reference; for sequences it can be the
          concatenated full epochs list or the global seq.total_epochsTDB list.
        - For sequence_definition != None, epochs_TDB is not directly used for legs epochs;
          _prepare_sequence_data uses sequence_definition.epochs_vec.
        """
        body_ID = body.spice_id

        # Sidecar extension follows the requested format (".json" or ".pkl").
        side_ext = (
            ".pkl" if str(sidecar_format).lower() in ("pickle", "pkl") else ".json"
        )

        # Parameters sidecar, only when parameters were actually estimated
        if cls._has_parameter_payload(parameters_JSON):
            data = cls._prepare_parameters_data(
                body_ID,
                origin_ID,
                ref_frame,
                epochs_TDB,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
                parameters_units=parameters_units,
            )
            # Sidecars follow ``sidecar_name`` rather than the SPK, so several
            # bodies sharing one .bsp cannot overwrite each other's parameters.
            # Readers locate a sidecar by body id, not by filename.
            cls._write_parameters_to_json(
                sidecar_name or SPK_name, data, f"_parameters{side_ext}"
            )

        # STMs JSON
        if STMs_JSON is not None:
            stm_data = cls._prepare_stm_data(
                STMs_JSON,
                epochs_TDB,
                body_ID,
                origin_ID,
                ref_frame,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
                STM_units=STM_units,
            )
            cls._write_stm_sidecar(SPK_name, stm_data, stm_format)

        # Settings sidecar
        if propagator_settings is not None:
            cls._write_parameters_to_json(
                SPK_name, propagator_settings, f"_settings{side_ext}"
            )

        # Mass profile sidecar
        if mass_profile is not None:
            mass_profile_data = cls._prepare_mass_profile_data(body, mass_profile)
            cls._write_parameters_to_json(
                SPK_name, mass_profile_data, f"_mass_profile{side_ext}"
            )

    @classmethod
    def _has_parameter_payload(cls, parameters_JSON) -> bool:
        """
        Report whether a parameters payload carries any estimated parameter.

        A state vector holding only position and velocity produces an empty
        payload: ``None`` from the flat state-array path, or one ``None`` per
        leg from a sequence. The per-leg list is truthy even when every entry
        is empty, so an ``is not None`` check alone still writes a sidecar
        whose ``parameters`` fields are all null — a file that later matches
        in :meth:`get_parameters` and shadows the run that actually estimated
        something. Callers use this to skip the write instead.

        Parameters
        ----------
        parameters_JSON : ArrayWUnits, ArrayWFrame, list, tuple or None
            Payload as assembled by ``Trajectory``: a single array for the
            flat case, or one entry per leg for a sequence.

        Returns
        -------
        bool
            ``True`` when at least one parameter value is present.
        """
        if parameters_JSON is None:
            return False

        # Sequence payloads nest one entry per leg, and a leg entry may itself
        # be wrapped in a list.
        if isinstance(parameters_JSON, (list, tuple)):
            return any(cls._has_parameter_payload(entry) for entry in parameters_JSON)

        # Unwrap ArrayWFrame -> ArrayWUnits -> ndarray; anything else is used
        # as-is so plain sequences of numbers still measure correctly.
        values = getattr(parameters_JSON, "quantity", parameters_JSON)
        values = getattr(values, "values", values)

        return np.size(values) > 0

    @staticmethod
    def _parameter_entries(state_definition, body_ID=None) -> list:
        """
        The state components that are genuinely estimated parameters of a body.

        Everything that is not a position or a velocity, optionally restricted
        to one owner. The rule this replaces — "skip the first two entries" —
        assumed the state opened with the primary body's position and velocity
        and held no others, so a second *propagated* body had its pos/vel
        listed as parameters of the first.

        A sidecar describes one body, so ``body_ID`` narrows the list to the
        parameters that body's sidecar holds: its own and, for the first
        propagated body in the state, every parameter whose owner is not
        propagated (a ground station's range bias, say). Such an owner has no
        trajectory, hence no sidecar, of its own, and the first propagated body
        is the trajectory a filter reads, so every parameter in the state lands
        in exactly one sidecar. Without ``body_ID`` every parameter is returned.

        Parameters
        ----------
        state_definition : list
            State components as ``(name, dim, estimation, dynamics, owner,
            value)`` tuples.
        body_ID : int, optional
            SPICE id of the body the sidecar is for.

        Returns
        -------
        list
            The subset that body's sidecar holds, in state order.
        """
        entries = []
        propagated = []
        for entry in state_definition or []:
            # An entry with no readable name does not name a parameter. The
            # rule this replaced never looked at the leading components at all,
            # so it tolerated placeholders there; keep that tolerance.
            try:
                name = str(entry[0]).lower()
            except (TypeError, IndexError):
                continue
            if name not in ("position", "velocity"):
                entries.append(entry)
                continue
            sid = getattr(entry[4], "spice_id", None)
            if sid not in propagated:
                propagated.append(sid)

        if body_ID is None:
            return entries

        hosts_unpropagated = bool(propagated) and propagated[0] == body_ID
        return [
            entry
            for entry in entries
            if (sid := getattr(entry[4], "spice_id", None)) == body_ID
            or (hosts_unpropagated and sid not in propagated)
        ]

    @classmethod
    def _prepare_parameters_data(
        cls,
        body_ID,
        origin_ID,
        ref_frame,
        epochs_TDB,
        parameters_JSON,
        state_definition,
        sequence_definition,
        leg_data,
        parameters_units=None,
    ):
        """
        Prepare parameters data for JSON export.

        Parameters
        ----------
        body_ID : int
            The ID of the body.
        origin_ID : int
            The ID of the origin.
        ref_frame : str
            The reference frame.
        epochs_TDB : list
            The array of epochs in TDB format.
        parameters_JSON : dict
            The array of parameters in JSON format.
        state_definition : list
            The state definition.
        sequence_definition : object
            The sequence definition.
        leg_data : dict, optional
            The body, origin, and reference frame data for each leg.

        Returns
        -------
        dict
            Prepared parameters data.
        """

        data = {}

        if sequence_definition is not None:
            data["legs"] = cls._prepare_sequence_data(
                sequence_definition, leg_data, parameters_JSON
            )
        else:
            data = {
                "body_ID": body_ID,
                "origin_ID": origin_ID,
                "reference_frame": ref_frame.name,
            }

            data.update(
                {
                    "parameters_def": [
                        entry[0]
                        for entry in cls._parameter_entries(state_definition, body_ID)
                    ],
                    "parameters_body": [
                        entry[4]._name
                        for entry in cls._parameter_entries(state_definition, body_ID)
                    ],
                    "epochsTDB": epochs_TDB,
                    "parameters": parameters_JSON.values.tolist(),
                }
            )
            data.update(cls._parameter_units_payload(parameters_JSON, parameters_units))

        return data

    @classmethod
    def _parameter_units_payload(cls, parameters_JSON, parameters_units) -> dict:
        """
        Build the units fields of a parameters sidecar.

        Parameters of one body need not share units — a spacecraft solving for
        a delta-v (km/s) and a maneuver-epoch correction (sec) is ordinary — so
        the stacked ``parameters`` block cannot always be described by a single
        unit. ``parametersUnits`` records one entry per parameter and is the
        authoritative field.

        The legacy scalar ``unitsPower`` / ``unitsScale`` are still emitted
        whenever every parameter agrees, so any file that could be written
        before this change is unchanged and older readers keep working. They
        are omitted when the units differ, rather than silently mislabelling
        the whole block with whichever unit happened to come first.

        Parameters
        ----------
        parameters_JSON : ArrayWUnits
            The stacked parameter values. Carries shared units when uniform.
        parameters_units : list of tuple or None
            ``(name, dim, units)`` per parameter, as produced by
            ``Trajectory._extract_body_parameters_from_state``. When ``None``
            the caller had no per-parameter breakdown and the stacked units are
            used.

        Returns
        -------
        dict
            The units fields to merge into the sidecar payload.
        """

        def _u(units):
            if units is None:
                return {"unitsPower": [], "unitsScale": []}
            return {
                "unitsPower": units.dimensions.powers.tolist(),
                "unitsScale": units.scales.tolist(),
            }

        if parameters_units:
            payload = {
                "parametersUnits": [
                    {"name": name, "dim": int(dim), **_u(units)}
                    for name, dim, units in parameters_units
                ]
            }
            all_units = [u for _, _, u in parameters_units]
            if all(SpiceManager._same_units(u, all_units[0]) for u in all_units):
                payload.update(_u(all_units[0]))
            return payload

        return _u(getattr(parameters_JSON, "units", None))

    @staticmethod
    def _prepare_sequence_data(sequence_definition, leg_data, parameters_JSON):
        """
        Prepare sequence data for JSON export.

        Supports robust leg metadata format:
        leg_data = {"by_seq_idx": { seq_idx : {body_ID, origin_ID, reference_frame}, ... }}

        Parameters
        ----------
        sequence_definition : MissionSequence
            The sequence definition; its legs and finite burns are exported.
        leg_data : dict
            The body, origin, and reference frame of each leg, either keyed by
            sequence index under ``"by_seq_idx"`` or as lists indexed by leg
            under ``"body_ID"``, ``"origin_ID"`` and ``"reference_frame"``.
        parameters_JSON : list or None
            One entry per leg: ``None`` or empty for no parameters, the leg's
            ``(name, ArrayWFrame)`` components (written as the
            ``[name0, value0, name1, value1, ...]`` list
            :meth:`get_parameters` reads), or a single ArrayWFrame or value.

        Returns
        -------
        dict
            Per-leg metadata and parameters, keyed ``"leg_<i>"``.

        Raises
        ------
        KeyError
            If ``leg_data["by_seq_idx"]`` has no entry for a leg.
        """
        # Split epochs per event
        split_epochs = [
            np.atleast_1d(epoch.times.values)
            for epoch in sequence_definition.epochs_vec
        ]
        split_epochs = [
            arr[:-1] if len(arr) > 1 and i != len(split_epochs) - 1 else arr
            for i, arr in enumerate(split_epochs)
        ]

        legs = {}
        num_events = 0

        def _is_named_pairs(leg_par):
            return isinstance(leg_par, (list, tuple)) and all(
                isinstance(p, (list, tuple))
                and len(p) == 2
                and isinstance(p[0], str)
                for p in leg_par
            )

        # Helper to get leg meta robustly
        def _get_leg_meta(seq_idx, leg_i):
            # New robust format
            if isinstance(leg_data, dict) and "by_seq_idx" in leg_data:
                meta = leg_data["by_seq_idx"].get(seq_idx, None)
                if meta is None:
                    raise KeyError(
                        f"leg_data missing metadata for sequence index {seq_idx}. "
                        "Ensure leg_data['by_seq_idx'] is built for the legs contained in this SPK."
                    )
                return meta["body_ID"], meta["origin_ID"], meta["reference_frame"]

            return (
                leg_data["body_ID"][leg_i],
                leg_data["origin_ID"][leg_i],
                leg_data["reference_frame"][leg_i],
            )

        for idx, (name, model, typ) in enumerate(
            zip(
                sequence_definition.names,
                sequence_definition.models,
                sequence_definition.types,
            )
        ):
            if typ == "Impulsive Burn" or typ == "Node":
                num_events += 1
                continue

            if typ not in ("Leg", "Finite Burn"):
                # Anything else is an event, not a propagated span, and has no
                # state history to register. A finite burn *is* a propagated
                # span: the rest of the library treats it as an active segment
                # (MissionSequence, ScenarioSetup, and Trajectory's own leg
                # collection all test ``in ("Leg", "Finite Burn")``), and the
                # index arithmetic below already assumes it occupies a leg slot
                # -- ``num_events`` only counts impulsive burns and nodes.
                # Excluding it here dropped every burn span from the sidecar's
                # leg registry, so a solution could not resolve state history
                # across the burns.
                continue

            leg_i = idx - num_events

            body_ID, origin_ID, reference_frame = _get_leg_meta(idx, leg_i)

            # Parameters payload for this leg
            leg_entry = None if parameters_JSON is None else parameters_JSON[leg_i]
            if leg_entry is None or (
                isinstance(leg_entry, (list, tuple)) and len(leg_entry) == 0
            ):
                params_payload = None
                units_power = []
                units_scale = []
                params_frame = []
            elif _is_named_pairs(leg_entry):
                # The leg's (name, ArrayWFrame) components. All of them are
                # written, as the [name0, value0, name1, value1, ...] list
                # get_parameters reads, so every parameter (a maneuver epoch
                # next to its delta-v, a station's range bias) reaches the
                # sidecar; each value carries its own units and frame, so the
                # leg-level unit fields stay empty.
                params_payload = [item for pair in leg_entry for item in pair]
                units_power = []
                units_scale = []
                params_frame = []
            else:
                pj = leg_entry[0] if isinstance(leg_entry, (list, tuple)) else leg_entry
                if isinstance(pj, ArrayWFrame):
                    params_payload = np.atleast_1d(pj.quantity.values).tolist()
                    units_power = pj.quantity.units.dimensions.powers.tolist()
                    units_scale = pj.quantity.units.scales.tolist()
                    params_frame = pj.frame.name
                else:
                    params_payload = pj
                    units_power = []
                    units_scale = []
                    params_frame = []

            sequence_data = {
                "body_ID": body_ID,
                "origin_ID": origin_ID,
                "reference_frame": reference_frame,
                "name": name,
                "parameters_def": [
                    entry[0]
                    for entry in SpiceManager._parameter_entries(
                        model.state_definition
                    )
                ],
                "parameters_body": [
                    entry[4]._name
                    for entry in SpiceManager._parameter_entries(
                        model.state_definition
                    )
                ],
                "epochsTDB": split_epochs[idx],
                "parameters": params_payload,
                "unitsPower": units_power,
                "unitsScale": units_scale,
                "parameters_frame": params_frame,
            }

            legs[f"leg_{leg_i}"] = sequence_data

        return legs

    @classmethod
    def _prepare_stm_data(
        cls,
        STMs,
        epochs_TDB,
        body_ID,
        origin_ID,
        ref_frame,
        parameters_JSON,
        state_definition,
        sequence_definition,
        leg_data,
        STM_units=None,
    ):
        """
        Prepare STM data for JSON export.

        Without a sequence the STMs are keyed by epoch on the global grid
        *epochs_TDB*, one matrix per epoch.

        A sequence propagates each leg (or finite burn) with its own STM,
        restarted at the leg's first epoch and sized to that leg's state, so
        consecutive legs both hold an STM at their shared boundary epoch.  When
        *STMs* is one block per leg, each block's rows lying on that leg's
        grid in *sequence_definition*, the blocks are written per leg:
        ``"STMs"`` maps each leg key to its own epoch-keyed matrices, and
        that leg's entry in ``"legs"`` records the grid as ``STM_epochsTDB``.
        A sequence whose *STMs* is one matrix per global epoch keeps the
        global layout.

        The payload records the STM ``reference_frame`` and ``units`` explicitly so
        the matrices are self-describing on disk.

        Parameters
        ----------
        STMs : list
            STMs, one matrix per epoch of *epochs_TDB*, or, for a sequence,
            one ``(N_k, n*n)`` or ``(N_k, n, n)`` block per leg on that leg's
            grid.
        epochs_TDB : list
            The global epoch grid, in TDB seconds past J2000.
        body_ID : int
            The ID of the body.
        origin_ID : int
            The ID of the origin.
        ref_frame : Frame
            The reference frame.
        parameters_JSON : list or None
            The parameters per leg, for the sequence leg metadata (see
            :meth:`_prepare_sequence_data`).
        state_definition : list or None
            The state definition.
        sequence_definition : MissionSequence or None
            The sequence definition, or ``None`` for a single arc.
        leg_data : dict or None
            The body, origin, and reference frame data for each leg.
        STM_units : dict or None
            Serialized units (``Units.to_dict()``) for the STM entries, or ``None``
            when the STMs carry no unit information.

        Returns
        -------
        dict
            JSON-serializable STM payload.

        Raises
        ------
        STMLayoutError
            If a sequence's *STMs* match neither its legs' grids nor
            *epochs_TDB*.
        """
        # Normalize epochs list
        epochs_list = list(epochs_TDB)

        # Normalize STMs to one 2-D array per epoch.
        # Accept: list of ndarray, ndarray (N, n, n), or list-like.
        stms_list = []
        if STMs is None:
            stms_list = []
        else:
            if isinstance(STMs, _np.ndarray):
                # could be (N,n,n) or (n,n) if single
                if STMs.ndim == 3:
                    stms_list = list(STMs)  # N views, no copy
                elif STMs.ndim == 2:
                    stms_list = [STMs]
                else:
                    stms_list = [_np.atleast_2d(STMs)]
            else:
                # list / iterable, already per-epoch
                stms_list = list(STMs)

        stm_data = {
            "body_ID": int(body_ID),
            "origin_ID": int(origin_ID),
            "reference_frame": ref_frame.name,
            "units": STM_units,
            "epochsTDB": epochs_list,
            "STMs": {str(t): M for t, M in zip(epochs_list, stms_list)},
        }

        # Add parameter definitions for context. The STM spans the whole
        # state, so every parameter column is labelled — not only those of the
        # body this sidecar is written for. The old "skip the first two
        # entries" rule listed a second propagated body's position and
        # velocity as parameters.
        if sequence_definition is None:
            entries = cls._parameter_entries(state_definition)
            stm_data.update(
                {
                    "parameters_def": [entry[0] for entry in entries],
                    "parameters_body": [entry[4]._name for entry in entries],
                }
            )
        else:
            # For sequences, keep leg metadata (useful for readers) but do not
            # pretend it's "parameters_def" only.
            legs = cls._prepare_sequence_data(
                sequence_definition, leg_data, parameters_JSON
            )
            stm_data["legs"] = legs

            # Each leg's full grid, both ends included: the leg's STM block
            # lies on it, unlike the leg's "epochsTDB", which drops the
            # boundary epoch the next leg starts on.
            leg_grids = [
                _np.atleast_1d(epoch.times.values).astype(float)
                for epoch, typ in zip(
                    sequence_definition.epochs_vec, sequence_definition.types
                )
                if typ in ("Leg", "Finite Burn")
            ]

            def _is_leg_block(block, grid):
                """Whether ``block`` holds one STM per epoch of ``grid``.

                Parameters
                ----------
                block : array_like
                    One leg's STMs, either ``(N, n, n)`` or flattened rows
                    ``(N, n*n)``.
                grid : np.ndarray
                    That leg's STM epochs, shape (N,).

                Returns
                -------
                bool
                    True if ``block`` has one square STM per epoch in
                    ``grid``; a lone ``(n, n)`` matrix is rejected.
                """
                block = _np.asarray(block)
                if block.ndim == 3:
                    return block.shape[0] == len(grid)
                # (N, n*n) rows; a lone (n, n) matrix has n columns, not n*n
                if block.ndim == 2:
                    n = math.isqrt(block.shape[1])
                    return block.shape[0] == len(grid) and n * n == block.shape[1]
                return False

            if (
                stms_list
                and len(stms_list) == len(legs) == len(leg_grids)
                and all(map(_is_leg_block, stms_list, leg_grids))
            ):
                stm_data["STMs"] = {}
                for leg_key, block, grid in zip(legs, stms_list, leg_grids):
                    block = _np.asarray(block, dtype=float)
                    n = math.isqrt(block[0].size)
                    stm_data["STMs"][leg_key] = {
                        str(t): M
                        for t, M in zip(grid.tolist(), block.reshape(-1, n, n))
                    }
                    legs[leg_key]["STM_epochsTDB"] = grid.tolist()
            elif stms_list and len(stms_list) != len(epochs_list):
                raise cls.STMLayoutError(
                    f"Got {len(stms_list)} STM entries for a sequence of "
                    f"{len(legs)} leg(s) (grid lengths "
                    f"{[len(g) for g in leg_grids]}) over {len(epochs_list)} "
                    "global epochs. Pass one (N_k, n*n) or (N_k, n, n) block "
                    "per leg on that leg's grid, or one matrix per global epoch."
                )

        return stm_data

    @staticmethod
    def _prepare_mass_profile_data(body, mass_profile: SCBPolynomial):
        """
        Prepare mass profile data (polynomial) for JSON export.

        Parameters
        ----------
        body : Body
            Body object, used to retrieve metadata such as name, ID, and units.

        mass_profile : SCBPolynomial
            SCBPolynomial mass profile (e.g., from body.mass_profile).

        Returns
        -------
        dict
            Mass profile dictionary ready for JSON serialization.
        """
        return {
            "body_ID": body.spice_id,
            "name": body.name,
            "mass_poly": {
                "coeffs": mass_profile.coefficients.tolist(),
                "domain": list(mass_profile.domain),
                "mass_units_power": body._mass_units.dimensions.powers,
                "mass_units_scale": body._mass_units.scales,
            },
        }

    @classmethod
    def _write_stm_sidecar(cls, SPK_name, stm_data, stm_format="json"):
        """
        Write the STM sidecar in the requested format.

        ``stm_format="json"`` (default) produces a human-readable ``_stms.json``;
        ``"pickle"`` produces a compact binary ``_stms.pkl`` of the same payload
        dict, which is lighter on disk for large STM time histories. The reader
        :meth:`get_STMs` accepts either.

        Parameters
        ----------
        SPK_name : str
            Path to the ``.bsp`` file; the sidecar is written alongside it with
            the same base name.
        stm_data : dict
            STM payload from :meth:`_prepare_stm_data`.
        stm_format : {"json", "pickle"}, optional
            On-disk format. Defaults to ``"json"``.

        Raises
        ------
        ValueError
            If ``stm_format`` is not ``"json"`` or ``"pickle"``.
        """
        fmt = (stm_format or "json").lower()
        if fmt in ("pickle", "pkl"):
            folder = os.path.dirname(SPK_name)
            base = os.path.basename(SPK_name)
            out_name = os.path.join(folder, base.replace(".bsp", "") + "_stms.pkl")
            with open(out_name, "wb") as f:
                pickle.dump(stm_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "json":
            # Written compact
            cls._write_parameters_to_json(SPK_name, stm_data, "_stms.json", indent=None)
        else:
            raise ValueError(
                f"Unknown stm_format '{stm_format}'. Use 'json' or 'pickle'."
            )

    @staticmethod
    def _write_parameters_to_json(SPK_name, data, suffix, indent=4):
        def _default(o):
            """
            Fallback serializer for non-JSON-able objects.
            Must return only JSON-serializable types (dict/list/str/float/int/bool/None).
            """
            # numpy scalars
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()

            # numpy arrays
            if isinstance(o, np.ndarray):
                return o.tolist()

            # Handle SCB-like wrappers (duck typing) safely (properties may throw!)
            clsname = o.__class__.__name__

            # ArrayWFrame: usually has .quantity and .frame
            if clsname == "ArrayWFrame":
                out = {}
                try:
                    q = o.quantity
                except Exception:
                    q = None
                try:
                    fr = o.frame
                except Exception:
                    fr = None

                if fr is not None:
                    out["frame"] = getattr(fr, "name", str(fr))
                else:
                    out["frame"] = None

                if q is not None:
                    # values
                    if hasattr(q, "values"):
                        out["values"] = np.asarray(q.values).tolist()
                    else:
                        out["values"] = str(q)

                    # units
                    try:
                        u = q.units
                        out["unitsPower"] = u.dimensions.powers.tolist()
                        out["unitsScale"] = u.scales.tolist()
                    except Exception:
                        out["unitsPower"] = []
                        out["unitsScale"] = []
                else:
                    out["values"] = None
                    out["unitsPower"] = []
                    out["unitsScale"] = []

                return out

            # ArrayWUnits: usually has .values and .units
            if clsname == "ArrayWUnits":
                out = {}
                try:
                    out["values"] = np.asarray(o.values).tolist()
                except Exception:
                    out["values"] = str(o)

                try:
                    u = o.units
                    out["unitsPower"] = u.dimensions.powers.tolist()
                    out["unitsScale"] = u.scales.tolist()
                except Exception:
                    out["unitsPower"] = []
                    out["unitsScale"] = []
                return out

            # EpochArray / Time wrappers etc: try common field names
            for attr in ("times", "values", "value"):
                try:
                    v = getattr(o, attr)
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                except Exception:
                    pass

            # Last resort: stringify
            return str(o)

        folder = os.path.dirname(SPK_name)
        base = os.path.basename(SPK_name)
        out_name = os.path.join(folder, base.replace(".bsp", "") + suffix)

        if suffix.endswith(".pkl"):
            # Store the *same* plain-dict content as the JSON path (normalised via
            # ``_default``) as a binary pickle, so the readers parse it identically.
            plain = json.loads(json.dumps(data, default=_default))
            with open(out_name, "wb") as pkl_file:
                pickle.dump(plain, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(out_name, "w") as json_file:
                json.dump(data, json_file, indent=indent, default=_default)

    @staticmethod
    def name_to_ID(name):
        # NOTE: change to name2id() ?
        """
        Converts a given name to its corresponding ID in the kernel pool.

        Parameters
        ----------
        name : str
            The name to be converted.

        Returns
        -------
        int or None
            The ID corresponding to the given name, or None if the name is not found in the kernel pool.
        """
        try:
            # Check if this name is already in the kernel pool
            spiceId = spice.bodn2c(name)
        except:  # noqa: E722
            spiceId = None
        return spiceId

    @staticmethod
    def id2name(id: int):
        """Convert given SPICE ID to the name of the associated body."""
        return spice.bodc2n(id)

    @staticmethod
    def generate_metakernel(directory: str, output_file: str = "output_metakernel.tm"):
        """
        List all files in a directory and write them into a SPICE metakernel.

        Parameters
        ----------
        directory : str
            Path to the directory containing kernels.
        output_file : str
            Name of the output metakernel file (default 'output_metakernel.tm').
        """
        # Collect all files (skip subdirectories)
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        files.sort()  # optional: alphabetic order

        # Write the metakernel
        with open(output_file, "w") as mk:
            mk.write("KPL/MK\n")
            mk.write("\\begindata\n\n")
            mk.write("KERNELS_TO_LOAD = (\n")

            for i, f in enumerate(files):
                line = f"    '{directory}/{f}'"
                if i < len(files) - 1:
                    line += ","
                line += "\n"
                mk.write(line)

            mk.write(")\n\n")
            mk.write("\\begintext\n")

        print(f"Metakernel written to {output_file}")

    @staticmethod
    def _open_ck_file(ck_path: str, internal_name: str = "SCB_CK", ifname_len: int = 0):
        """
        Open (create/overwrite) a CK file for writing.

        Notes
        -----
        - CSPICE CK writer opens a DAF. Like SPKs, you should NOT have the file loaded
          when opening for write. We'll attempt unload first.
        """
        if not ck_path.endswith(".bc") and not ck_path.endswith(".ck"):
            ck_path += ".bc"

        # Unload if loaded (safe-guard); spice.unload is a no-op for a file
        # that isn't loaded, so no try/except is needed here
        spice.unload(ck_path)

        # Ensure directory exists
        folder = os.path.dirname(os.path.abspath(ck_path))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        # Remove existing file (overwrite behavior)
        if os.path.isfile(ck_path):
            os.remove(ck_path)

        # ckopn: open new CK for write
        # ifname_len is the size of the comment area in bytes; 0 is fine for most use
        handle = spice.ckopn(ck_path, internal_name, ifname_len)
        return ck_path, handle

    @staticmethod
    def _ensure_quat_array(quats: np.ndarray, nrec: int) -> np.ndarray:
        q = np.asarray(quats, dtype=float)

        # Accept (nrec,4) or (4,nrec)
        if q.shape == (4, nrec):
            q = q.T
        if q.shape != (nrec, 4):
            raise ValueError(
                f"quats must be shape (nrec,4) or (4,nrec). Got {q.shape}."
            )
        return q

    @staticmethod
    def _ensure_av_array(avvs: np.ndarray, nrec: int) -> np.ndarray:
        a = np.asarray(avvs, dtype=float)

        # Accept (nrec,3) or (3,nrec)
        if a.shape == (3, nrec):
            a = a.T
        if a.shape != (nrec, 3):
            raise ValueError(f"avvs must be shape (nrec,3) or (3,nrec). Got {a.shape}.")
        return a

    # Unused: only ever referenced from a commented-out branch in
    # write_ck_type3 (Option B below). Kept for reference.
    # @staticmethod
    # def _finite_difference_body_rates(
    #     et_list: np.ndarray, c_ref_to_inst_list: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Estimate body angular velocity (in REF frame) from a sequence of DCMs C_ref_to_inst.
    #
    #     Returns
    #     -------
    #     avvs : (N,3) angular velocity vectors expressed in REF frame
    #
    #     Notes
    #     -----
    #     - CKW03 expects AVVs expressed in the base frame REF. :contentReference[oaicite:1]{index=1}
    #     - This is a pragmatic FD estimate; if your mode can provide analytical rates,
    #       pass them in instead.
    #     """
    #     et = np.asarray(et_list, dtype=float)
    #     C = np.asarray(c_ref_to_inst_list, dtype=float)  # (N,3,3)
    #     N = C.shape[0]
    #     if N < 2:
    #         return np.zeros((N, 3))
    #
    #     # Convert to C_inst_to_ref for convenience: C_ir = C_ri^T
    #     C_ir = np.transpose(C, (0, 2, 1))
    #
    #     w_ref = np.zeros((N, 3))
    #     for k in range(1, N - 1):
    #         dt = et[k + 1] - et[k - 1]
    #         if dt <= 0:
    #             continue
    #
    #         # Central difference on C_ir
    #         dC = (C_ir[k + 1] - C_ir[k - 1]) / dt
    #
    #         # Kinematics: dC_ir = -[w]_x * C_ir  =>  [w]_x = -(dC_ir) * C_ir^T
    #         # with C_ir^T = C_ri
    #         W = -(dC @ C_ir[k].T)
    #
    #         # Extract vector from skew-symmetric matrix:
    #         # [  0  -wz  wy]
    #         # [ wz   0  -wx]
    #         # [-wy  wx   0 ]
    #         wx = 0.5 * (W[2, 1] - W[1, 2])
    #         wy = 0.5 * (W[0, 2] - W[2, 0])
    #         wz = 0.5 * (W[1, 0] - W[0, 1])
    #         w_ref[k, :] = np.array([wx, wy, wz])
    #
    #     # Endpoints: one-sided
    #     w_ref[0, :] = w_ref[1, :]
    #     w_ref[-1, :] = w_ref[-2, :]
    #     return w_ref

    @classmethod
    def write_ck_type3(
        cls,
        ck_path: str,
        inst_id: int,
        ref_frame: str,
        sclk_id: int,
        et_list: list[float] | np.ndarray,
        quats: np.ndarray | None = None,
        avvs: np.ndarray | None = None,
        dcm_cb=None,
        segid: str = "SCB GENERATED CK TYPE 3",
        make_intervals: str = "per_sample",
        load_after: bool = True,
    ):
        """
        Write a CK type 3 segment (discrete pointing) to a new CK file.

        Parameters
        ----------
        ck_path : str
            Output CK file path (.bc or .ck).
        inst_id : int
            NAIF instrument/frame ID for the CK segment (the CK "INST" id).
        ref_frame : str
            Base/reference frame name (usually 'J2000').
        sclk_id : int
            NAIF spacecraft clock ID code used by sce2c().
        et_list : array-like
            ET (TDB seconds past J2000) sample times.
        quats : ndarray, optional
            Quaternions rotating vectors from ref_frame to inst frame.
            Shape (N,4) or (4,N). If None, must provide dcm_cb.
        avvs : ndarray, optional
            Angular velocities expressed in ref_frame. Shape (N,3) or (3,N).
            If None, AVFLAG=0 and zeros are written (or FD-estimated if dcm_cb provided and you want).
        dcm_cb : callable, optional
            Function et -> (3,3) DCM C_ref_to_inst. If provided and quats is None,
            quats are built via spice.m2q(C). spice.m2q uses SPICE quaternion convention. :contentReference[oaicite:2]{index=2}
        segid : str
            Segment ID (<= 40 chars recommended).
        make_intervals : {'per_sample','single'}
            - 'per_sample': NINTS=N-1, STARTS=sclkdp[:-1]
            - 'single'    : NINTS=1, STARTS=[sclkdp[0]]
        load_after : bool
            If True, furnsh() the CK after writing.

        Returns
        -------
        ck_path : str
            Path to the written CK file.

        Notes
        -----
        CKW03 requirements include monotonically increasing SCLKDP and descriptor times
        containing the data range. :contentReference[oaicite:3]{index=3}
        """
        et = np.asarray(et_list, dtype=float).ravel()
        if et.size < 1:
            raise ValueError("et_list must contain at least one epoch.")
        if np.any(np.diff(et) <= 0.0):
            raise ValueError("et_list must be strictly increasing.")

        nrec = int(et.size)

        # Build quats from callback if needed
        C_list = None
        if quats is None:
            if dcm_cb is None:
                raise ValueError("Provide either quats or dcm_cb(et)->DCM.")
            C_list = np.zeros((nrec, 3, 3), dtype=float)
            q_list = np.zeros((nrec, 4), dtype=float)
            for k, t in enumerate(et):
                C = np.asarray(dcm_cb(float(t)), dtype=float)
                if C.shape != (3, 3):
                    raise ValueError(f"dcm_cb must return (3,3). Got {C.shape}.")
                C_list[k, :, :] = C
                q_list[k, :] = np.asarray(spice.m2q(C), dtype=float)
            quats = q_list

        quats = cls._ensure_quat_array(quats, nrec)

        # SCLKDP from ET
        sclkdp = np.array([spice.sce2c(sclk_id, float(t)) for t in et], dtype=float)
        if np.any(np.diff(sclkdp) <= 0.0):
            raise ValueError(
                "Encoded SCLK times must be strictly increasing (check SCLK kernel/ET list)."
            )

        # Interpolation intervals for type 3
        if make_intervals not in ("per_sample", "single"):
            raise ValueError("make_intervals must be 'per_sample' or 'single'.")

        if make_intervals == "single" or nrec == 1:
            nints = 1
            starts = np.array([sclkdp[0]], dtype=float)
        else:
            nints = nrec - 1
            starts = np.array(sclkdp[:-1], dtype=float)

        # Angular velocity vectors
        if avvs is None:
            # Option A: write AVFLAG=0 and zeros
            avflag = 0
            avvs_use = np.zeros((nrec, 3), dtype=float)

            # Option B (commented): if you want FD-estimated rates when dcm_cb used:
            # if C_list is not None:
            #     avflag = 1
            #     avvs_use = cls._finite_difference_body_rates(et, C_list)
        else:
            avflag = 1
            avvs_use = cls._ensure_av_array(avvs, nrec)

        # Descriptor times must span the data
        begtim = float(sclkdp[0])
        endtim = float(sclkdp[-1])

        ck_path, handle = cls._open_ck_file(
            ck_path, internal_name="SCB_CK_TYPE3", ifname_len=0
        )
        try:
            spice.ckw03(
                handle,
                begtim,
                endtim,
                int(inst_id),
                str(ref_frame),
                int(avflag),
                str(segid),
                int(nrec),
                sclkdp,
                quats,
                avvs_use,
                int(nints),
                starts,
            )
        finally:
            spice.ckcls(handle)

        if load_after:
            spice.furnsh(ck_path)

        return ck_path

    # endregion Static Methods #
    # --------------------------#
