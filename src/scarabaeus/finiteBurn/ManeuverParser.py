# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import os

from scarabaeus import (
    Units,
    Frame,
    EpochArray,
    Body,
    ForceModel,
    Spacecraft,
    ArrayWUnits,
    StateArray,
    SpiceManager,
    ArrayWFrame,
    Maneuver,
)

kg, km, sec, N = Units.get_units(["kg", "km", "sec", "N"])

SpiceManager.clear_kernels()
furnshKernelFilename = os.getcwd() + "/data/kernels/locked/locked_generic.tm"
# SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)


class ManeuverParser:
    """ Parses and manages an ordered list of finite-burn :class:`~scarabaeus.Maneuver` objects.

    Maneuvers can be loaded from a comma-delimited file via
    :meth:`_parse_maneuver_spec`, or built and stored programmatically through
    :meth:`get_burns` / :meth:`set_burns`.  An individual burn can be replaced
    in-place by matching on start time with :meth:`_update_burn_by_start_time`.

    Parameters
    ----------
    None

    Notes
    -----
    The internal burn list is accessible and replaceable via :meth:`get_burns`
    and :meth:`set_burns`.  The file-based parser expects a specific
    comma-delimited column layout; see :meth:`_parse_maneuver_spec` for details.

    See Also
    --------
    scarabaeus.Maneuver : Finite-burn maneuver data container parsed by this class.
    scarabaeus.FiniteBurn : Force model that consumes Maneuver objects during propagation.

    Examples
    --------
    .. code-block:: python

        parser = ManeuverParser()
        with open("maneuvers.csv") as fh:
            burns = parser._parse_maneuver_spec(fh)
    """

    def __init__(self):
        self._burns = []

    def get_burns(self) -> list:
        """
        Return the current list of :class:`Maneuver` objects.

        Returns
        -------
        list of Maneuver
        """
        return self._burns

    def set_burns(self, burns) -> None:
        """
        Replace the internal burn list.

        Parameters
        ----------
        burns : iterable of Maneuver
            New collection of maneuvers.  Converted to a plain list on assignment.
        """
        self._burns = list(burns)

    def _parse_maneuver_spec(self, handle) -> list:
        """
        Parse a comma-delimited maneuver specification file and populate the
        internal burn list.

        Expected column layout (0-indexed, repeating block of 13 per maneuver):

        * col 1  – number of maneuvers in the row
        * col 3  – burn start time (UTC string, converted via SpiceManager)
        * col 4  – thrust direction *ux* (unitless)
        * col 5  – thrust direction *uy* (unitless)
        * col 6  – thrust direction *uz* (unitless)
        * col 7  – thrust magnitude (N)
        * col 9  – mass-flow rate (kg/s)
        * col 13 – burn duration (s)

        The header row is skipped.  Each maneuver in the row is appended to
        :attr:`_burns`.

        Parameters
        ----------
        handle : file-like
            Open file handle positioned at the start of the file.

        Returns
        -------
        list of Maneuver
            The updated internal burn list.
        """
        for line in handle.readlines()[1:]:
            linesplit = line.split(",")
            num_maneuvers = int(linesplit[1])
            idx_add = 0
            for n in range(num_maneuvers):
                ux = ArrayWUnits(float(linesplit[4 + idx_add]), None)
                uy = ArrayWUnits(float(linesplit[5 + idx_add]), None)
                uz = ArrayWUnits(float(linesplit[6 + idx_add]), None)
                start_et = float(SpiceManager.str2et(linesplit[3 + idx_add]))
                start_time = EpochArray(start_et, sys="TDB")
                end_time = EpochArray(start_et + float(linesplit[13 + idx_add]), sys="TDB")
                mass_flow = ArrayWUnits(float(linesplit[9 + idx_add]), kg / sec)
                thrust = ArrayWUnits(float(linesplit[7 + idx_add]), N)

                burn = Maneuver(thrust, mass_flow, start_time, end_time, ux, uy, uz)
                self._burns.append(burn)
                idx_add += 13

        return self._burns

    def _update_burn_by_start_time(self, maneuver: Maneuver, fb_list: list) -> tuple:
        """
        Replace the burn in *fb_list* whose start time matches *maneuver*.

        Matching uses an absolute tolerance of 1 µs (1e-6 s).

        Parameters
        ----------
        maneuver : Maneuver
            Replacement maneuver.  Must be a :class:`Maneuver` instance.
        fb_list : list of Maneuver
            List to search and update in-place.

        Returns
        -------
        fb_list : list of Maneuver
            The updated list.
        idx : int
            Index of the replaced entry.

        Raises
        ------
        ValueError
            If *maneuver* is not a :class:`Maneuver`, or no burn with a
            matching start time is found.
        """
        if not isinstance(maneuver, Maneuver):
            raise ValueError("maneuver data has to be a Maneuver object")

        for idx, man in enumerate(fb_list):
            if abs(man.start_time.times.values - maneuver.start_time.times.values) < 1e-6:
                fb_list[idx] = maneuver
                return fb_list, idx

        raise ValueError(f"No burn found with start_time == {maneuver.start_time.times.values}")
