# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from datetime import date

from scarabaeus import constants, Units, Frame, EpochArray
import scarabaeus.utils.NumpyWrapper as np
import numpy.typing as npt

# ------------------#
#  Generate Units  #
# ------------------#
km, sec = Units.get_units(["km", "sec"])

# Generate common units and frames
spc_pos_units = km
spc_time_units = sec
(J2000, ITRF93, ECLIPJ2000, IAUEARTH) = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class RampTableManager:
    """
    Handles ramp-frequency table integration for real DSN measurement models.

    Provides static utilities to locate the correct ramp-table interval for a
    query epoch and to integrate the piecewise-linear transmit frequency over
    an observation window, supporting multi-segment and day-spanning cases.

    Notes
    -----
    This class is a work-in-progress; the interface may change as additional
    DSN data types are supported.

    References
    ----------
    .. [1] Moyer, T. D. *Formulation for Observed and Computed Values of Deep Space
       Network Data Types for Navigation*. Monograph 2, Deep Space Communications
       and Navigation Series, JPL/Wiley, 2000.
    """

    # ------------------#
    # region   Errors  #
    # ------------------#
    class RampCoverageError(Exception):
        """The ramp table does not cover the requested integration window.

        Raised by :meth:`integrate` when the table is empty or the window
        starts before the first ramp record, where the transmit frequency is
        undefined. Integrating anyway would return a zero count, and a computed
        measurement of exactly 0.0 that downstream code cannot distinguish from
        a real one.
        """

    class RampTableYearError(ValueError):
        """The ramp table or the window crosses a new year, but no years were given.

        Raised by :meth:`integrate`. Days of year alone cannot place day 1
        after day 365 (or 366), so the integration would be a year off.
        """

    # endregion Errors #
    # ------------------#

    # ---------#
    # Methods #
    # ---------#
    @staticmethod
    def _find_index_for_query(ref: npt.NDArray, query: float):
        """
        Find the index of the ramp-table interval that contains *query*.

        Parameters
        ----------
        ref : numpy.ndarray
            Sorted reference time array (ramp-table time column).
        query : float
            Query epoch [s] to locate within *ref*.

        Returns
        -------
        int
            Index ``i`` such that ``ref[i] <= query < ref[i+1]``.
            Returns ``0`` if no matching interval is found.
        """
        # Find the indices where the difference changes sign (i.e., the query is between these indices)
        indices = np.where((ref[:-1] <= query) & (ref[1:] > query))[0]
        # Return the first matching index
        if len(indices) > 0:
            return indices[0]
        else:
            # print(f"[WARNING] RampTableManager: No index found for t = {query}")
            # print(f"Returning 0 as placeholder")
            return 0  # None  # If no such index exists

    @staticmethod
    def _day_number(year, doy) -> np.ndarray:
        """Return the day count since 0001-01-01 of day ``doy`` of ``year``.

        Parameters
        ----------
        year : int or array_like of int
            Calendar year.
        doy : float or array_like of float
            Day of year, counted from 1. It may run past either end of the year:
            day 0 is 31 December of the year before, and day 366 of a common year
            is 1 January of the next.

        Returns
        -------
        numpy.ndarray
            Proleptic Gregorian day number, with the fraction of ``doy`` kept, in
            the shape of ``doy``.
        """
        years, inverse = np.unique(np.asarray(year, dtype=int), return_inverse=True)
        jan1 = np.array([date(int(y), 1, 1).toordinal() for y in years.ravel()])
        return jan1[inverse].reshape(np.shape(doy)) + np.asarray(doy, dtype=float) - 1.0

    @staticmethod
    def integrate(
        ramp_table: dict,
        t1_q_sec: float,
        t2_q_sec: float,
        t1_q_doy: float,
        t2_q_doy: float,
        t1_q_year: int = None,
        t2_q_year: int = None,
        t1_q_sec_l: float = 0.0,
        t2_q_sec_l: float = 0.0,
    ):
        """
        Integrate the piecewise-linear transmit frequency over a query window.

        Parameters
        ----------
        ramp_table : dict
            Ramp-table data with five entries (indexed 0–4): seconds of day,
            day-of-year, reference frequency [Hz], frequency rate [Hz/s], and
            ramp-type flag. An optional sixth entry holds the year of each
            record (None when unknown).
        t1_q_sec : float
            Start of the integration window in seconds of day.
        t2_q_sec : float
            End of the integration window in seconds of day.
        t1_q_doy : float
            Day-of-year corresponding to *t1_q_sec*.
        t2_q_doy : float
            Day-of-year corresponding to *t2_q_sec*. Either day may run past
            the ends of its year (day 0, day 366 of a common year).
        t1_q_year, t2_q_year : int, optional
            Years of *t1_q_doy* and *t2_q_doy*. Used, together with the ramp
            years, to cross a new year.
        t1_q_sec_l, t2_q_sec_l : float, optional
            Low parts of *t1_q_sec* and *t2_q_sec* [s], for window ends built
            as (high, low) pairs with :meth:`EpochArray._add_duration`.

        Returns
        -------
        f_int : float
            Integrated frequency over the query window, in Hz s.
        aux : dict
            Auxiliary debug quantities (segment boundaries, per-segment
            frequencies, day-span flag, etc.).

        Raises
        ------
        RampTableManager.RampCoverageError
            If the ramp table is empty or *t1* precedes its first record.
        RampTableManager.RampTableYearError
            If the ramp table or the window crosses a new year and the ramp or
            query years are not given.

        Notes
        -----
        Ramp records and query epochs are mapped onto one continuous time
        axis counted from the first ramp day, so windows and ramp segments may
        cross midnight, and a new year when the years are given. The last
        ramp record is extrapolated past its epoch.

        The window ends stay (high, low) pairs and every width is a difference
        of nearby epochs, so the integral keeps the window to ~1e-13 s. A float
        second of day resolves only ~7e-12 s at 5e4 s, and f_T times that is
        ~0.05 cycles at X band, ~1 mHz on 60 s Doppler.
        """
        f_int = 0
        aux = {
            'span_day_flag': 0,
            'f_iq1': 0,
            'f_iq2': 0,
            'n_segments': 0
        }

        # Extract ramp table data from SFDU 9
        ramp_sec = np.array(ramp_table[0])
        ramp_day = np.array(ramp_table[1])
        ramp_freq =  np.array(ramp_table[2])
        ramp_rate =  np.array(ramp_table[3])
        ramp_type =  np.array(ramp_table[4])

        if ramp_sec.size == 0:
            raise RampTableManager.RampCoverageError(
                "Ramp table is empty: the transmit frequency is undefined."
            )

        # Day numbers of records and queries: days of year, or days since
        # 0001-01-01 when the years are known, so a new year can be crossed
        ramp_year = ramp_table[5] if len(ramp_table) > 5 else None
        if ramp_year is not None and t1_q_year is not None and t2_q_year is not None:
            ramp_dn = RampTableManager._day_number(ramp_year, ramp_day)
            t1_q_dn = RampTableManager._day_number(t1_q_year, t1_q_doy)
            t2_q_dn = RampTableManager._day_number(t2_q_year, t2_q_doy)
        else:
            ramp_dn = ramp_day.astype(float)
            t1_q_dn, t2_q_dn = float(t1_q_doy), float(t2_q_doy)
            # Days of year that jump by more than half a year are a new year
            days = np.unique(np.r_[ramp_dn, t1_q_dn, t2_q_dn])
            if days.size > 1 and np.max(np.diff(days)) > 183:
                raise RampTableManager.RampTableYearError(
                    f"Ramp records (days {ramp_day.min()}-{ramp_day.max()}) and the "
                    f"integration window (days {t1_q_doy}, {t2_q_doy}) cross a new "
                    "year: give the ramp years (aux 'ramp_year', SFDU 9 'year') "
                    "and the window years."
                )

        # Seconds since the start of the first ramp day, for records and
        # queries alike, so segments and windows can cross midnight
        ramp_day0 = np.min(ramp_dn)
        ramp_sec = ramp_sec + (ramp_dn - ramp_day0) * constants.day_to_sec
        t1_q_sec, t1_q_sec_l = EpochArray._add_duration(
            t1_q_sec, t1_q_sec_l, (t1_q_dn - ramp_day0) * constants.day_to_sec
        )
        t2_q_sec, t2_q_sec_l = EpochArray._add_duration(
            t2_q_sec, t2_q_sec_l, (t2_q_dn - ramp_day0) * constants.day_to_sec
        )
        if t1_q_doy != t2_q_doy:
            aux['span_day_flag'] = 1

        # Integration window
        W = (t2_q_sec - t1_q_sec) + (t2_q_sec_l - t1_q_sec_l)

        # Segment indices: last ramp record at or before each window end
        iq1_t = np.where((t1_q_sec - ramp_sec) + t1_q_sec_l >= 0)[0]
        iq2_t = np.where((t2_q_sec - ramp_sec) + t2_q_sec_l >= 0)[0]

        if iq1_t.size == 0 or iq2_t.size == 0:
            # The query window starts before the ramp table's first entry, so no
            # segment covers it. This is usually a coverage gap of seconds at a
            # pass boundary: the observable is timestamped by its *reception*
            # time t3, but the ramp is queried at the *uplink transmit* time
            # t1 = t3 - RTLT, which can precede the table by up to one round-trip
            # light time.
            raise RampTableManager.RampCoverageError(
                f"Integration window [day {t1_q_doy}, "
                f"{t1_q_sec - (t1_q_dn - ramp_day0) * constants.day_to_sec} s] to "
                f"[day {t2_q_doy}, {t2_q_sec - (t2_q_dn - ramp_day0) * constants.day_to_sec} s] "
                f"starts before the first ramp record at [day {ramp_day[0]}, "
                f"{ramp_table[0][0]} s]. This query is an uplink transmit time, "
                f"which precedes the measurement's reception time by one "
                f"round-trip light time, so an observation near the start of a "
                f"pass can fall outside a table that covers the pass itself. Mark "
                f"that observation as an outlier, or supply a ramp table that "
                f"extends earlier."
            )
        else:
            iq1 = iq1_t[-1]
            iq2 = iq2_t[-1]
            aux['iq1'] = iq1
            aux['iq2'] = iq2
            # Offsets of the window ends into their segments
            dt1 = (t1_q_sec - ramp_sec[iq1]) + t1_q_sec_l
            dt2 = (t2_q_sec - ramp_sec[iq2]) + t2_q_sec_l
            f_iq1 = ramp_freq[iq1] + ramp_rate[iq1] * dt1
            f_iq2 = ramp_freq[iq2] + ramp_rate[iq2] * dt2
            aux['f_iq1'] = f_iq1
            aux['f_iq2'] = f_iq2
            n_segments = iq2 - iq1 + 1
            aux['n_segments'] = n_segments
            W_i = np.zeros(n_segments)
            f_i = np.zeros(n_segments)
            if n_segments > 1:
                W_i[0] = (ramp_sec[iq1 + 1] - t1_q_sec) - t1_q_sec_l
                W_i[-1] = dt2
                for ii in range(1, n_segments - 1):
                    W_i[ii] = ramp_sec[iq1 + ii + 1] - ramp_sec[iq1 + ii]
            else:
                W_i[0] = W
            if n_segments > 1:
                f_i[0] = f_iq1 + 0.5 * ramp_rate[iq1] * W_i[0]
                f_i[-1] = ramp_freq[iq2] + 0.5 * ramp_rate[iq2] * W_i[-1]
                for ii in range(1, n_segments - 1):
                    f_i[ii] = ramp_freq[iq1 + ii] + 0.5 * ramp_rate[iq1 + ii] * W_i[ii]
            else:
                f_i[0] = f_iq1 + 0.5 * ramp_rate[iq1] * W_i[0]
            # Time vector and reference frequencies for aux
            t_i = np.zeros(n_segments + 1)
            f0_i = np.zeros(n_segments + 1)
            if n_segments > 1:
                t_i[0] = t1_q_sec
                t_i[-1] = t2_q_sec
                f0_i[0] = f_iq1
                f0_i[-1] = f_iq2
                for ii in range(1, n_segments):
                    t_i[ii] = ramp_sec[iq1 + ii]
                    f0_i[ii] = ramp_freq[iq1 + ii]
            else:
                t_i[0] = t1_q_sec
                t_i[1] = t2_q_sec
                f0_i[0] = f_iq1
                f0_i[1] = f_iq2
            aux['W'] = W
            aux['t_i'] = t_i
            aux['f0_i'] = f0_i
            f_int = np.sum(f_i * W_i)
        return f_int, aux

    @staticmethod
    def frequency_at(
        ramp_table: dict,
        t_sec: float,
        t_doy: float,
        t_year: int = None,
        t_sec_l: float = 0.0,
    ) -> float:
        """
        Transmit frequency of the ramp table at one epoch.

        Parameters
        ----------
        ramp_table : dict
            Ramp-table data, as :meth:`integrate` reads it.
        t_sec : float
            Epoch in seconds of day. It may run past either end of *t_doy*.
        t_doy : float
            Day-of-year of *t_sec*.
        t_year : int, optional
            Year of *t_doy*, needed to cross a new year.
        t_sec_l : float, optional
            Low part of *t_sec* [s].

        Returns
        -------
        float
            Frequency of the ramp segment holding the epoch, extrapolated
            linearly from its record [Hz].

        Raises
        ------
        RampTableManager.RampCoverageError
            If the ramp table is empty or the epoch precedes its first record.
        RampTableManager.RampTableYearError
            If the ramp table or the epoch crosses a new year and the years
            are not given.

        Notes
        -----
        The frequency at the start of a zero-width :meth:`integrate` window,
        so the segment lookup, day mapping and year handling are the same.
        """
        _, aux = RampTableManager.integrate(
            ramp_table, t_sec, t_sec, t_doy, t_doy, t_year, t_year, t_sec_l, t_sec_l
        )
        return float(aux['f_iq1'])
