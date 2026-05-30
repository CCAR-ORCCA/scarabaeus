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
import spiceypy as spice
import scarabaeus.utils.NumpyWrapper as np
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

    _spc_pos_units = km
    _spc_time_units = sec
    _spc_angle_units = rad
    _spc_vel_units = km / sec

    kernel_list = []
    kernel_folder = None  # None → computed from os.getcwd() at call time
    poly_interp_deg = 3
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

        # Get units
        units = [
            cls._spc_pos_units,
            cls._spc_pos_units,
            cls._spc_pos_units,
            cls._spc_vel_units,
            cls._spc_vel_units,
            cls._spc_vel_units,
        ]  # Needs to output in km and km/sec
        return ArrayWUnits(state, units)

    @classmethod
    def get_state_precise(
        cls,
        trgt_bdy: str | Body,
        epoch_time_high: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        aberration_correction: str = "None",
        epoch_time_low: float | EpochArray = None,
    ):
        """
        Retrieve the state (position and velocity) of a target body relative to an origin body at a specified epoch time.

        Parameters
        ----------
        trgt_bdy : str or Body
            The target body for which the state is to be retrieved. Can be a string name or a Body object.

        epoch_time_high : float or EpochArray
            The epoch time at which the state is to be retrieved. Can be an EpochArray object or a float representing the time. This is the high component

        reference_frame : str or Frame
            The reference frame in which the state is to be retrieved. Can be a Frame object or a string name.

        obsvr_bdy : str or Body
            The observing body relative to which the state is to be retrieved. Can be a string name or a Body object.

        aberration_correction : str, optional
            The aberration correction to be applied. Defaults to ``None``.

        epoch_time_low : float or EpochArray
            The epoch time at which the state is to be retrieved. Can be an EpochArray object or a float representing the time. This is the low component

        Returns
        -------
        state : ArrayWUnits
            An array containing the state (position and velocity) of the target body with associated units.
        """
        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time_high = cls._validate_epoch_input(epoch_time_high)

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

        # Get states from SPICE at epoch_time_high separated by "dt" seconds
        state_start = np.array(
            spice.spkezr(
                trgt_bdy,
                epoch_time_high,
                reference_frame,
                aberration_correction,
                obsvr_bdy,
            )[0]
        )
        state_end = np.array(
            spice.spkezr(
                trgt_bdy,
                epoch_time_high + 1,
                reference_frame,
                aberration_correction,
                obsvr_bdy,
            )[0]
        )

        # Stack the two state vectors into a 2D array of shape (2, 6)
        states = np.vstack((state_start, state_end))  # shape: (2, 6)

        # Perform linear interpolation at dtepoch_time_low_small for each component
        state = (1 - epoch_time_low) * states[0] + epoch_time_low * states[1]

        # Get units
        units = [
            cls._spc_pos_units,
            cls._spc_pos_units,
            cls._spc_pos_units,
            cls._spc_vel_units,
            cls._spc_vel_units,
            cls._spc_vel_units,
        ]  # Needs to output in km and km/sec
        return ArrayWUnits(state, units)

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

    @classmethod
    def get_parameters(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
    ):
        """
        Retrieve and interpolate estimated parameters from a companion JSON file.

        Searches the scenario kernel folder for a ``*_parameters.json`` file
        whose ``body_ID``, ``origin_ID``, and ``reference_frame`` fields match
        the supplied arguments.  If the file contains a multi-leg sequence, the
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
            Absolute path to the folder that contains the JSON files.  When
            ``None`` (default), ``cls.kernel_folder`` is used if set; otherwise
            falls back to ``<cwd>/data/kernels/scenario``.

        Returns
        -------
        parameters : ArrayWUnits
            Interpolated parameter vector with units reconstructed from the
            ``unitsPower`` / ``unitsScale`` fields stored in the JSON.

        Raises
        ------
        ValueError
            If no matching JSON file is found, or if *epoch_time* does not fall
            within any leg's time span.
        RuntimeWarning
            If more than one matching JSON file is found; the most recently
            modified file is used.
        """

        # Define the folder path containing the JSON files
        if not folder_path_override:
            # not given folder path -> assume matches data folder
            folder_path = (
                cls.kernel_folder
                if cls.kernel_folder is not None
                else os.getcwd() + "/data/kernels/scenario"
            )
        else:
            folder_path = folder_path_override
            
        json_parameters = None
        match_count = 0
        matched_filenames = []

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
        # Iterate over the files in the folder
        # MINIMAL CHANGE: fix recognition logic (no double count, ANY leg match)
        # ------------------------------------------------------------------
        for filename in os.listdir(folder_path):
            if not filename.endswith(".json"):
                continue

            with open(os.path.join(folder_path, filename), "r") as file:
                json_data = json.load(file)

            matched_this_file = False

            # Sequence JSON: match if ANY leg matches (IDs live inside each leg)
            if (
                isinstance(json_data, dict)
                and "legs" in json_data
                and isinstance(json_data["legs"], dict)
            ):
                for leg in json_data["legs"].values():
                    if not isinstance(leg, dict):
                        continue
                    if (
                        leg.get("body_ID") == trgt_id
                        and leg.get("origin_ID") == obsvr_id
                        and leg.get("reference_frame") == reference_frame
                    ):
                        matched_this_file = True
                        break

            # Non-sequence JSON fallback: match at top-level
            elif isinstance(json_data, dict):
                if (
                    json_data.get("body_ID") == trgt_id
                    and json_data.get("origin_ID") == obsvr_id
                    and json_data.get("reference_frame") == reference_frame
                ):
                    matched_this_file = True

            if matched_this_file:
                match_count += 1
                matched_filenames.append(filename)
                json_parameters = json_data

        # Print error if needed
        if match_count > 1:

            # Pick most recently modified matching file
            latest_idx = np.argmax(
                [
                    os.path.getmtime(os.path.join(folder_path, f))
                    for f in matched_filenames
                ]
            )

            latest_filename = matched_filenames[latest_idx]

            warnings.warn(
                "Multiple matching JSON files found. "
                f"Using most recently modified file: {latest_filename}",
                RuntimeWarning,
            )

            with open(os.path.join(folder_path, latest_filename), "r") as file:
                json_parameters = json.load(file)

        elif match_count == 0:
            raise ValueError(
                "The JSON file containing the estimated parameters could not be located."
            )

        # Determine if the JSON contains sequences
        if "legs" in json_parameters:
            parameter = None
            units = None

            # Find the appropriate leg for the given epoch_time
            for idx, leg in enumerate(json_parameters["legs"].values()):
                epochsTDB = np.array(leg["epochsTDB"], dtype=float)
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
                    unitsP_out = []
                    unitsS_out = []

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

                        up = np.array(d.get("unitsPower", []), dtype=int).ravel()
                        us = np.array(d.get("unitsScale", []), dtype=float).ravel()

                        v_raw = d.get("values", None)
                        if v_raw is None:
                            raise ValueError(
                                f"Missing 'values' in parameters dict for leg {idx}."
                            )

                        # Case A: constant (1xP or P-vector)
                        # Example: "values": [0,0,0]
                        if isinstance(v_raw, (list, tuple)) and (
                            len(v_raw) == 0 or not isinstance(v_raw[0], (list, tuple))
                        ):
                            v = np.array(v_raw, dtype=float).ravel()
                            vals_out.append(v)
                            unitsP_out.append(up)
                            unitsS_out.append(us)
                            continue

                        # Case B: time series
                        # Example: "epochsTDB": [t0..tN], "values": [[p...],[p...],...]
                        p_epochs = d.get("epochsTDB", None)
                        if p_epochs is None:
                            # If values look 2D but no epochs provided, fall back to leg epochsTDB (same length check below)
                            p_epochs = epochsTDB

                        p_epochs = np.array(p_epochs, dtype=float)
                        V = np.array(v_raw, dtype=float)

                        # Allow V as (N,P) or (N,) for scalar param
                        if V.ndim == 1:
                            V = V.reshape(-1, 1)

                        if p_epochs.shape[0] != V.shape[0]:
                            raise ValueError(
                                f"Parameter time series length mismatch in leg {idx}: "
                                f"len(epochs)={p_epochs.shape[0]} vs values rows={V.shape[0]}"
                            )

                        # Interpolate each component (polyfit degree cls.poly_interp_par_deg)
                        deg = int(cls.poly_interp_par_deg)
                        coeff = np.polyfit(p_epochs, V, deg)  # coeff shape: (deg+1, P)
                        v = np.polyval(coeff, epoch_time).ravel()  # -> (P,)
                        vals_out.append(v)

                        # Units: replicate per-component if only one units spec provided
                        if up.size not in (0, v.size):
                            if up.size == 4:  # common "Dimensions" length
                                up = np.tile(up, v.size)
                            else:
                                up = np.resize(up, v.size)
                        if us.size not in (0, v.size):
                            us = np.resize(us, v.size)

                        unitsP_out.append(up)
                        unitsS_out.append(us)

                    # Concatenate all parameters in order
                    parameter = (
                        np.concatenate(vals_out)
                        if len(vals_out)
                        else np.array([], dtype=float)
                    )

                    leg_unitsPower = (
                        np.concatenate(unitsP_out)
                        if len(unitsP_out)
                        else np.array([], dtype=int)
                    )
                    leg_unitsScale = (
                        np.concatenate(unitsS_out)
                        if len(unitsS_out)
                        else np.array([], dtype=float)
                    )

                    units = Units(
                        Dimensions(leg_unitsPower.tolist()), leg_unitsScale.tolist()
                    )
                    break

            if parameter is None:
                raise ValueError(
                    "The given epoch_time does not fall within any leg's epochsTDB."
                )

        # Nominal case (no sequence)
        else:
            parameters = np.array(json_parameters.get("parameters"))
            epochsTDB = np.array(json_parameters.get("epochsTDB"))
            unitsPower = np.array(json_parameters.get("unitsPower"))
            unitsScale = np.array(json_parameters.get("unitsScale"))

            # Perform interpolation with the data
            json_poly_coefficients = np.polyfit(
                epochsTDB, parameters, cls.poly_interp_par_deg
            )
            parameter = np.polyval(json_poly_coefficients, epoch_time)

            # Get units
            units = Units(Dimensions(unitsPower), unitsScale)

        return ArrayWUnits(parameter, units)

    @classmethod
    def get_STMs(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
    ) -> np.ndarray:
        """
        Retrieve or interpolate the State Transition Matrix (STM) at a given epoch.

        Searches the scenario kernel folder for a ``*_stms.json`` file whose
        ``body_ID``, ``origin_ID``, and ``reference_frame`` fields match the
        supplied arguments.  If *epoch_time* coincides (within floating-point
        tolerance) with a stored epoch the corresponding STM is returned
        directly.  Otherwise element-wise linear interpolation is performed
        between the two bracketing epochs.

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

        Returns
        -------
        STM : numpy.ndarray
            Shape ``(6, 6)`` State Transition Matrix evaluated at *epoch_time*.

        Raises
        ------
        ValueError
            If no matching STM JSON file is found, or if *epoch_time* lies
            outside the stored time span.
        """
        # 1) Normalize inputs
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        # 2) Find the right JSON file
        stm_data = None
        _kf = (
            folder_path_override
            if folder_path_override is not None
            else cls.kernel_folder
            if cls.kernel_folder is not None
            else os.getcwd() + "/data/kernels/scenario"
        )
        for fn in os.listdir(_kf):
            if fn.endswith("_stms.json"):
                with open(os.path.join(_kf, fn), "r") as f:
                    jd = json.load(f)
                if (
                    jd["body_ID"] == spice.bodn2c(trgt_bdy)
                    and jd["origin_ID"] == spice.bodn2c(obsvr_bdy)
                    and jd["reference_frame"] == reference_frame
                ):
                    stm_data = jd
                    break
        if stm_data is None:
            raise ValueError("No matching STM JSON found.")

        # 3) Extract the appropriate epochs array
        #    (either across a single span or within the correct leg)
        if "parameters_def" in stm_data and "legs" in stm_data["parameters_def"]:
            # sequence‐case: find the leg containing your time
            for leg in stm_data["parameters_def"]["legs"].values():
                leg_epochs = np.array(leg["epochsTDB"], dtype=float)
                if leg_epochs[0] <= epoch_time <= leg_epochs[-1]:
                    epochs = leg_epochs
                    break
            else:
                raise ValueError("Epoch not in any leg.")
        else:
            # non‐sequence
            epochs = np.array(stm_data["epochsTDB"], dtype=float)
            if not (epochs[0] <= epoch_time <= epochs[-1]):
                raise ValueError("Epoch out of stored range.")

        # 4) If exact match, return it
        #    Convert to Python float before str‐lookup
        match_idx = np.where(np.isclose(epochs, epoch_time))[0]
        if match_idx.size == 1:
            t0 = float(epochs[match_idx[0]])
            return np.array(stm_data["STMs"][str(t0)])

        # 5) Otherwise find neighbors and do linear element‐wise interpolation
        #    (epochs is sorted ascending)
        idx = np.searchsorted(epochs, epoch_time)
        t1, t2 = float(epochs[idx - 1]), float(epochs[idx])
        STM1 = np.array(stm_data["STMs"][str(t1)])
        STM2 = np.array(stm_data["STMs"][str(t2)])
        alpha = (epoch_time - t1) / (t2 - t1)
        return (1 - alpha) * STM1 + alpha * STM2

    @classmethod
    def get_propagator_settings(
        cls,
        trgt_bdy: str | Body,
        reference_frame: str | Frame,
        obsvr_bdy: str | Body,
        folder_path_override: str = None,
    ):
        """
        Retrieves the propagator settings for a given target, reference frame, and origin from JSON files.
        Parameters
        ----------
        trgt_bdy : str or Body
            The target body or its name.
        reference_frame : str or Frame
            The reference frame or its name.
        obsvr_bdy : str or Body
            The origin body or its name.

        Returns
        -------
        dict
            The propagator settings data from the corresponding JSON file.

        Raises
        ------
        ValueError
            If the settings JSON file could not be located.
        """

        folder_path = (
            folder_path_override
            if folder_path_override is not None
            else cls.kernel_folder
            if cls.kernel_folder is not None
            else os.getcwd() + "/data/kernels/scenario"
        )
        settings_data = None

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(obsvr_bdy, Body):
            obsvr_bdy = obsvr_bdy.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        for filename in os.listdir(folder_path):
            if filename.endswith("_settings.json"):
                with open(os.path.join(folder_path, filename), "r") as file:
                    json_data = json.load(file)
                if (
                    json_data["body_ID"] == spice.bodn2c(trgt_bdy)
                    and json_data["origin_ID"] == spice.bodn2c(obsvr_bdy)
                    and json_data["reference_frame"] == reference_frame
                ):
                    settings_data = json_data
                    break

        if not settings_data:
            raise ValueError("Settings JSON file could not be located.")

        return settings_data

    @classmethod
    def get_attitude(
        cls,
        trgt_bdy: str | Body,
        epoch_time: float | EpochArray,
        reference_frame: str | Frame,
        tol: int,
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
            The tolerance value for the attitude determination.

        Returns
        -------
        tuple
            A tuple containing:
            - ArrayWUnits: The state vector of the target body.
            - ArrayWUnits: The angular velocity of the target body.
            - ArrayWUnits: The angular acceleration of the target body.
        """

        # -----------------------------------#
        # input validation and conditioning #
        # -----------------------------------#
        epoch_time = cls._validate_epoch_input(epoch_time)

        # --------------#
        # get attitude #
        # --------------#
        # Get s/c clock and attitude from SPICE
        t_clicks = spice.sce2c(trgt_bdy, epoch_time)  # Convert in clock ticks

        # Ensure target, reference_frame, origin are a string
        if isinstance(reference_frame, Frame):
            reference_frame = reference_frame.name
        if isinstance(trgt_bdy, Body):
            trgt_bdy = trgt_bdy.name

        trgt_bdy_frm_id = trgt_bdy * 1000
        state = spice.ckgpav(trgt_bdy_frm_id, t_clicks, tol, reference_frame)
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
            _, inframe = spice.cnmfrm(obsvr_bdy.name)

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
        for et in et_list:
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
            # Format time string: 'YYYY-DOY::HH:MM:SS'
            tt_str = (
                f"{trk_year}-{trk_doy:03d}::{hours:02d}:{minutes:02d}:{seconds:06.9}"
            )
            # Convert to ephemeris time
            tt_ET = spice.str2et(tt_str)
            et.append(tt_ET)
        return et

    # NOTE: see issue https://orccagitlab.colorado.edu/javi9068/scarabaeus/-/issues/315#note_2131 for context
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
    @lru_cache(maxsize=None)
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

    @staticmethod
    def list_all_frames(max_id: int = 100000) -> list:
        """List all SPICE reference frames currently detectable in the pool.

        Scans integer frame IDs in :math:`[-\\texttt{max\\_id},\\,+\\texttt{max\\_id})` and
        collects every ID for which SPICE returns a non-empty name.

        Parameters
        ----------
        max_id : int, optional
            Half-range of the frame-ID scan window.  Defaults to ``100000``.

        Returns
        -------
        frames_list : list of str
            Names of all reference frames found.

        Notes
        -----
        Frame IDs are distinct from body NAIF IDs used in SPK/PCK kernels.
        """
        frames_list = []
        for frame_id in range(-max_id, max_id):
            try:
                frame_name = SpiceManager.frame_id_to_frame_name(frame_id)
                if frame_name:
                    frames_list.append(frame_name)
            except Exception:
                pass  # Ignore errors for undefined frame IDs
        return frames_list

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
    def test(file):
        spice.ckgp(
            1835067803272,
        )

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
        try:
            existing_frame_name = spice.frmnam(frame_id)
            if existing_frame_name:
                raise RuntimeError(
                    f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                    f"Please choose a different frame ID or unregister the existing frame."
                )
        except spice.stypes.SpiceyError:
            # This occurs if the frame ID is not found, which is fine
            pass

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists.")
            # Prompt the user for confirmation
            while True:
                # user_input = (
                #    input("Do you want to overwrite it? (Y/N): ").strip().upper()
                # )
                user_input = "Y"  # Force user_input to overwrite
                if user_input == "Y":
                    break
                elif user_input == "N":
                    raise RuntimeError(
                        f"Execution aborted due to existing file: {file_name}"
                    )
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")

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
        try:
            existing_frame_name = spice.frmnam(frame_id)
            if existing_frame_name:
                raise RuntimeError(
                    f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                    f"Please choose a different frame ID or unregister the existing frame."
                )
        except spice.stypes.SpiceyError:
            # This occurs if the frame ID is not found, which is fine
            pass

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists.")
            # Prompt the user for confirmation
            while True:
                # user_input = (
                #    input("Do you want to overwrite it? (Y/N): ").strip().upper()
                # )
                user_input = "Y"  # Force user_input to overwrite
                if user_input == "Y":
                    break
                elif user_input == "N":
                    raise RuntimeError(
                        f"Execution aborted due to existing file: {file_name}"
                    )
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")

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
        try:
            existing_frame_name = spice.frmnam(frame_id)
            if existing_frame_name:
                raise RuntimeError(
                    f"Frame ID {frame_id} is already registered as '{existing_frame_name}' in the SPICE pool. "
                    f"Please choose a different frame ID or unregister the existing frame."
                )
        except spice.stypes.SpiceyError:
            # This occurs if the frame ID is not found, which is fine
            pass

        # Check if the file already exists
        if os.path.isfile(file_name):
            print(f"Warning: File '{file_name}' already exists.")
            # Prompt the user for confirmation
            while True:
                # user_input = (
                #    input("Do you want to overwrite it? (Y/N): ").strip().upper()
                # )
                user_input = "Y"  # Force user_input to overwrite
                if user_input == "Y":
                    break
                elif user_input == "N":
                    raise RuntimeError(
                        f"Execution aborted due to existing file: {file_name}"
                    )
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")

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
        frame_code, frame_name = spice.cidfrm(id_number)
        if frame_name == None:
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
            produced by ``spice.timout``.
        """
        cover = spice.spkcov(spkFile, objID)
        end_times = spice.wnfetd(cover, 0)
        start = spice.timout(end_times[0], "JD")
        end = spice.timout(end_times[1], "JD")
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
        try:
            # Get the total number of loaded kernels
            num_kernels = spice.ktotal("ALL")

            # Loop through all the loaded kernels and check if the kernel_filename is in the pool
            for i in range(num_kernels):
                file, _, _, _, _ = spice.kdata(i, "ALL")
                if kernel_filename in file:
                    return True
        except Exception as e:
            print(f"Error checking kernel: {e}")

        return False

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
    def write_spk_segment_type9(
        cls,
        SPK_name: str,
        SPK_SEG_ID: str,
        body: Body,
        origin_ID: float,
        ref_frame: Frame,
        epochs_TDB: list,
        degree_poly: float,
        state_SPICE: list,
        parameters_JSON: list = None,
        STMs_JSON: list = None,
        state_definition: list = None,
        sequence_definition: list = None,
        leg_data: dict = None,
        propagator_settings: dict = None,
        mass_profile: SCBPolynomial = None,
    ):
        """
        Write a segment of type 9 to an SPK file.

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
        """
        # Handle SPK file opening
        SPK_handle = cls._open_spk_file(SPK_name)

        # Initialize
        body_ID = body.spice_id

        # Write SPK segment
        num_states = len(epochs_TDB)
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

        # Write parameters to JSON if provided
        if parameters_JSON is not None:
            data = cls._prepare_parameters_data(
                body_ID,
                origin_ID,
                ref_frame,
                epochs_TDB,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
            )
            cls._write_parameters_to_json(SPK_name, data, "_parameters.json")

        # Write STMs to JSON if provided
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
            )
            cls._write_parameters_to_json(SPK_name, stm_data, "_STMs.json")

        # Write propagator settings to JSON if provided
        if propagator_settings is not None:
            cls._write_parameters_to_json(
                SPK_name, propagator_settings, "_settings.json"
            )

        # Write mass profile to JSON if provided
        if mass_profile is not None:
            # Write mass profile to JSON if provided
            mass_profile_data = cls._prepare_mass_profile_data(body, mass_profile)
            cls._write_parameters_to_json(
                SPK_name, mass_profile_data, "_mass_profile.json"
            )

        # Close SPK file
        spice.spkcls(SPK_handle)

        # Load the SPK file
        spice.furnsh(SPK_name)

    @staticmethod
    def _open_spk_file(SPK_name):
        """
        Open an SPK file for writing or appending.
        IMPORTANT: CSPICE cannot spkopa an SPK that is currently loaded (furnsh).
        """
        # Ensure not loaded before opening for write/append
        try:
            spice.unload(
                SPK_name
            )  # safe if not loaded? in CSPICE it can error; catch below
        except Exception:
            pass

        if os.path.isfile(SPK_name):
            print(
                f"Warning: SPK file '{SPK_name}' already exists. The trajectory segment will be appended to this file."
            )
            SPK_handle = spice.spkopa(SPK_name)
        else:
            SPK_handle = spice.spkopn(SPK_name, SPK_name, 0)

        return SPK_handle

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
        Write a trajectory segment to an SPK file.

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
        spice.spkw09(
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
        degree_poly: float,
        parameters_JSON: list = None,
        STMs_JSON: list = None,
        state_definition: list = None,
        sequence_definition: list = None,
        leg_data: dict = None,
        propagator_settings: dict = None,
        mass_profile: SCBPolynomial = None,
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

        # Parameters JSON
        if parameters_JSON is not None:
            data = cls._prepare_parameters_data(
                body_ID,
                origin_ID,
                ref_frame,
                epochs_TDB,
                parameters_JSON,
                state_definition,
                sequence_definition,
                leg_data,
            )
            cls._write_parameters_to_json(SPK_name, data, "_parameters.json")

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
            )
            cls._write_parameters_to_json(SPK_name, stm_data, "_STMs.json")

        # Settings JSON
        if propagator_settings is not None:
            cls._write_parameters_to_json(
                SPK_name, propagator_settings, "_settings.json"
            )

        # Mass profile JSON
        if mass_profile is not None:
            mass_profile_data = cls._prepare_mass_profile_data(body, mass_profile)
            cls._write_parameters_to_json(
                SPK_name, mass_profile_data, "_mass_profile.json"
            )

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
                    "parameters_def": np.array(
                        [
                            state_definition[i + 2][0]
                            for i in range(len(state_definition) - 2)
                        ]
                    ).tolist(),
                    "parameters_body": np.array(
                        [
                            state_definition[i + 2][4]._name
                            for i in range(len(state_definition) - 2)
                        ]
                    ).tolist(),
                    "epochsTDB": epochs_TDB,
                    "parameters": parameters_JSON.values.tolist(),
                    "unitsPower": parameters_JSON.units.dimensions.powers.tolist(),
                    "unitsScale": parameters_JSON.units.scales.tolist(),
                }
            )

        return data

    @staticmethod
    def _prepare_sequence_data(sequence_definition, leg_data, parameters_JSON):
        """
        Prepare sequence data for JSON export.

        Supports robust leg metadata format:
        leg_data = {"by_seq_idx": { seq_idx : {body_ID, origin_ID, reference_frame}, ... }}
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

            if typ != "Leg":
                # If you have other event types, skip them consistently
                continue

            leg_i = idx - num_events

            body_ID, origin_ID, reference_frame = _get_leg_meta(idx, leg_i)

            # Parameters payload for this leg
            if parameters_JSON is None:
                params_payload = None
                units_power = []
                units_scale = []
                params_frame = []
            else:
                pj = (
                    parameters_JSON[leg_i][0]
                    if isinstance(parameters_JSON[leg_i], (list, tuple))
                    else parameters_JSON[leg_i]
                )
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
                "parameters_def": np.array(
                    [
                        model.state_definition[i + 2][0]
                        for i in range(len(model.state_definition) - 2)
                    ]
                ).tolist(),
                "parameters_body": np.array(
                    [
                        model.state_definition[i + 2][4]._name
                        for i in range(len(model.state_definition) - 2)
                    ]
                ).tolist(),
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
    ):
        """
        Prepare STM data for JSON export.

        For sequence_definition != None, we still export STMs on a global time grid
        (same schema as non-sequence), because STMs are typically computed on the
        propagated timeline, not per event index.

        Returns
        -------
        dict
            JSON-serializable STM payload.
        """
        import numpy as np

        # Normalize epochs list
        epochs_list = list(epochs_TDB)

        # Normalize STMs to plain lists
        # Accept: list of ndarray, ndarray (N, n, n), or list-like
        stms_list = []
        if STMs is None:
            stms_list = []
        else:
            if isinstance(STMs, np.ndarray):
                # could be (N,n,n) or (n,n) if single
                if STMs.ndim == 3:
                    stms_list = [STMs[k, :, :].tolist() for k in range(STMs.shape[0])]
                elif STMs.ndim == 2:
                    stms_list = [STMs.tolist()]
                else:
                    stms_list = [np.atleast_2d(STMs).tolist()]
            else:
                # list / iterable
                for M in STMs:
                    if isinstance(M, np.ndarray):
                        stms_list.append(M.tolist())
                    else:
                        # maybe already list
                        stms_list.append(M)

        stm_data = {
            "body_ID": int(body_ID),
            "origin_ID": int(origin_ID),
            "reference_frame": ref_frame.name,
            "epochsTDB": epochs_list,
            "STMs": {str(t): M for t, M in zip(epochs_list, stms_list)},
        }

        # Add parameter definitions for context (same as non-sequence case)
        if sequence_definition is None:
            stm_data.update(
                {
                    "parameters_def": np.array(
                        [
                            state_definition[i + 2][0]
                            for i in range(len(state_definition) - 2)
                        ]
                    ).tolist(),
                    "parameters_body": np.array(
                        [
                            state_definition[i + 2][4]._name
                            for i in range(len(state_definition) - 2)
                        ]
                    ).tolist(),
                }
            )
        else:
            # For sequences, keep leg metadata (useful for readers) but do not
            # pretend it's "parameters_def" only.
            stm_data["legs"] = cls._prepare_sequence_data(
                sequence_definition, leg_data, parameters_JSON
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
                "coeffs": mass_profile.coef.tolist(),
                "domain": list(mass_profile.domain),
                "mass_units_power": body._mass_units.dimensions.powers,
                "mass_units_scale": body._mass_units.scales,
            },
        }

    @staticmethod
    def _write_parameters_to_json(SPK_name, data, suffix):
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

        with open(out_name, "w") as json_file:
            json.dump(data, json_file, indent=4, default=_default)

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

        # Unload if loaded (safe-guard)
        try:
            spice.unload(ck_path)
        except Exception:
            pass

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

    @staticmethod
    def _finite_difference_body_rates(
        et_list: np.ndarray, c_ref_to_inst_list: np.ndarray
    ) -> np.ndarray:
        """
        Estimate body angular velocity (in REF frame) from a sequence of DCMs C_ref_to_inst.

        Returns
        -------
        avvs : (N,3) angular velocity vectors expressed in REF frame

        Notes
        -----
        - CKW03 expects AVVs expressed in the base frame REF. :contentReference[oaicite:1]{index=1}
        - This is a pragmatic FD estimate; if your mode can provide analytical rates,
          pass them in instead.
        """
        et = np.asarray(et_list, dtype=float)
        C = np.asarray(c_ref_to_inst_list, dtype=float)  # (N,3,3)
        N = C.shape[0]
        if N < 2:
            return np.zeros((N, 3))

        # Convert to C_inst_to_ref for convenience: C_ir = C_ri^T
        C_ir = np.transpose(C, (0, 2, 1))

        w_ref = np.zeros((N, 3))
        for k in range(1, N - 1):
            dt = et[k + 1] - et[k - 1]
            if dt <= 0:
                continue

            # Central difference on C_ir
            dC = (C_ir[k + 1] - C_ir[k - 1]) / dt

            # Kinematics: dC_ir = -[w]_x * C_ir  =>  [w]_x = -(dC_ir) * C_ir^T
            # with C_ir^T = C_ri
            W = -(dC @ C_ir[k].T)

            # Extract vector from skew-symmetric matrix:
            # [  0  -wz  wy]
            # [ wz   0  -wx]
            # [-wy  wx   0 ]
            wx = 0.5 * (W[2, 1] - W[1, 2])
            wy = 0.5 * (W[0, 2] - W[2, 0])
            wz = 0.5 * (W[1, 0] - W[0, 1])
            w_ref[k, :] = np.array([wx, wy, wz])

        # Endpoints: one-sided
        w_ref[0, :] = w_ref[1, :]
        w_ref[-1, :] = w_ref[-2, :]
        return w_ref

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
