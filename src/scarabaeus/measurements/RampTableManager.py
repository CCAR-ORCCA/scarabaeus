# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, Frame
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
    Moyer, T. D. *Formulation for Observed and Computed Values of Deep Space
    Network Data Types for Navigation*. Monograph 2, Deep Space Communications
    and Navigation Series, JPL/Wiley, 2000.
    """

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
    def integrate(
        ramp_table: dict,
        t1_q_sec: float,
        t2_q_sec: float,
        t1_q_doy: float,
        t2_q_doy: float,
    ):
        """
        Integrate the piecewise-linear transmit frequency over a query window.

        Parameters
        ----------
        ramp_table : dict
            Ramp-table data with five entries (indexed 0–4): seconds of day,
            day-of-year, reference frequency [Hz], frequency rate [Hz/s], and
            ramp-type flag.
        t1_q_sec : float
            Start of the integration window in seconds of day.
        t2_q_sec : float
            End of the integration window in seconds of day.
        t1_q_doy : float
            Day-of-year corresponding to *t1_q_sec*.
        t2_q_doy : float
            Day-of-year corresponding to *t2_q_sec*.

        Returns
        -------
        f_int : float
            Integrated frequency [Hz·s] over the query window.
        aux : dict
            Auxiliary debug quantities (segment boundaries, per-segment
            frequencies, day-span flag, etc.).
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

        # Adjust for day wrap
        ramp_day0 = np.min(ramp_day)
        i_ramp_day_2 = np.where(ramp_day > ramp_day0)[0]
        if i_ramp_day_2.size > 0:
            ramp_sec[i_ramp_day_2] += 86400.0

        if t1_q_doy > ramp_day0:
            t1_q_sec += 86400
        if t2_q_doy > ramp_day0:
            t2_q_sec += 86400
        if t1_q_doy != t2_q_doy:
            aux['span_day_flag'] = 1
            t1_q_sec += 86400.0
            t2_q_sec += 86400.0

        # Linear ramp interpolant
        ft = lambda t, t0, f0, f0_dot: f0 + f0_dot * (t - t0)

        # Integration window
        W = t2_q_sec - t1_q_sec

        # Segment indices
        iq1_t = np.where(t1_q_sec > ramp_sec)[0]
        iq2_t = np.where(t2_q_sec > ramp_sec)[0]

        if iq1_t.size == 0 or iq2_t.size == 0:
            f_int = 0
        else:
            iq1 = iq1_t[-1]
            iq2 = iq2_t[-1]
            aux['iq1'] = iq1
            aux['iq2'] = iq2
            f_iq1 = ramp_freq[iq1] + ramp_rate[iq1] * (t1_q_sec - ramp_sec[iq1])
            f_iq2 = ramp_freq[iq2] + ramp_rate[iq2] * (t2_q_sec - ramp_sec[iq2])
            aux['f_iq1'] = f_iq1
            aux['f_iq2'] = f_iq2
            n_segments = iq2 - iq1 + 1
            aux['n_segments'] = n_segments
            W_i = np.zeros(n_segments)
            f_i = np.zeros(n_segments)
            if n_segments > 1:
                W_i[0] = ramp_sec[iq1 + 1] - t1_q_sec
                W_i[-1] = t2_q_sec - ramp_sec[iq2]
                for ii in range(1, n_segments - 1):
                    W_i[ii] = ramp_sec[iq1 + ii + 1] - ramp_sec[iq1 + ii]
            else:
                W_i[0] = t2_q_sec - t1_q_sec
            if n_segments > 1:
                f_i[0] = ramp_freq[iq1] + ramp_rate[iq1] * (t1_q_sec - ramp_sec[iq1]) + 0.5 * ramp_rate[iq1] * W_i[0]
                f_i[-1] = ramp_freq[iq2] + 0.5 * ramp_rate[iq2] * W_i[-1]
                for ii in range(1, n_segments - 1):
                    f_i[ii] = ramp_freq[iq1 + ii] + 0.5 * ramp_rate[iq1 + ii] * W_i[ii]
            else:
                f_i[0] = ft(t1_q_sec, ramp_sec[iq1], ramp_freq[iq1], ramp_rate[iq1]) + 0.5 * ramp_rate[iq1] * W_i[0]
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
