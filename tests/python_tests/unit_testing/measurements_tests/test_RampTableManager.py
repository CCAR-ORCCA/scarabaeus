# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the RampTableManager class  #
# =============================================#
from fractions import Fraction

import numpy as np
import pytest

import src.scarabaeus as scb

F0, RATE = 7.18e9, 0.5      # X-band uplink [Hz], ramp rate [Hz/s]


# -------------------#
# region    Helpers #
# -------------------#
# Ramp records over the afternoon of one day; the precision windows straddle them
DAY = 158
RAMP_SEC = [40000.0, 50030.0, 50045.25, 60000.0]
RAMP_FREQ = [F0, F0 + 1000.0, F0 - 500.0, F0]
TOL_CYCLES = 1e-3           # 20 uHz on 60 s X-band Doppler

# Start of the linear ramp used across midnight and the new year [s of day]
T0 = 80000.0


def _afternoon_ramp():
    n = len(RAMP_SEC)
    return [RAMP_SEC, [DAY] * n, RAMP_FREQ, [RATE] * n, [1] * n]


def _exact_integral(t1, t2):
    """Integral of the afternoon ramp between two exact (Fraction) seconds of day [Hz s]."""
    total = Fraction(0)
    for k, (tk, fk) in enumerate(zip(RAMP_SEC, RAMP_FREQ)):
        t_end = Fraction(RAMP_SEC[k + 1]) if k + 1 < len(RAMP_SEC) else t2
        a, b = max(t1, Fraction(tk)), min(t2, t_end)
        if b > a:
            f_a = Fraction(fk) + Fraction(RATE) * (a - Fraction(tk))
            total += (f_a + Fraction(RATE) * (b - a) / 2) * (b - a)
    return total


def _window(t3_sec, half, rtlt_s, rtlt_e):
    """Window ends t3 -/+ half - RTLT as (high, low) pairs, as DopplerReal builds them."""
    add = scb.EpochArray._add_duration
    s = add(*add(t3_sec, 0.0, -half), -rtlt_s)
    e = add(*add(t3_sec, 0.0, half), -rtlt_e)
    exact_s = Fraction(t3_sec) - Fraction(half) - Fraction(rtlt_s)
    exact_e = Fraction(t3_sec) + Fraction(half) - Fraction(rtlt_e)
    return s, e, exact_s, exact_e


def _linear_ramp(days, secs, years=None):
    """A continuous linear ramp f(t) = F0 + RATE (t - T0), sampled at the records."""
    elapsed = (np.asarray(days) - days[0]) * 86400.0 + np.asarray(secs) - T0
    table = [list(secs), list(days), list(F0 + RATE * elapsed), [RATE] * len(days),
             [1] * len(days)]
    return table + [years] if years is not None else table


def _linear_integral(t1_elapsed, t2_elapsed):
    """Integral of the linear ramp between two times counted from T0 [Hz s]."""
    return (F0 + RATE * 0.5 * (t1_elapsed + t2_elapsed)) * (t2_elapsed - t1_elapsed)

# endregion Helpers #
# -------------------#


# --------------#
# region Tests #
# --------------#
class TestSubUlpWindow:
    """
    The transmit window of a real Doppler count is t3 -/+ Tc/2 - RTLT in seconds of
    day. A float near 5e4 s resolves only ~7e-12 s, and the X-band count over the
    window moves by f_T * 7e-12 ~ 0.05 cycles per ULP, ~1 mHz on 60 s Doppler. With
    the window ends passed as (high, low) pairs the integral must match exact
    rational arithmetic to well below that, and must not move when a window end
    crosses a rounding boundary.
    """

    def test_integral_matches_exact_arithmetic(self):
        rng = np.random.default_rng(5)
        # Single-segment windows and windows across one or two ramp records
        for t3_sec in np.r_[rng.uniform(45000.0, 46000.0, 20), rng.uniform(50835.0, 50890.0, 20)]:
            rtlt_s = 812.3 + rng.uniform(0.0, 1e-3)
            rtlt_e = rtlt_s + rng.uniform(-1e-5, 1e-5)
            (s_h, s_l), (e_h, e_l), exact_s, exact_e = _window(t3_sec, 30.0, rtlt_s, rtlt_e)
            f_int, _ = scb.RampTableManager.integrate(
                _afternoon_ramp(), s_h, e_h, DAY, DAY, t1_q_sec_l=s_l, t2_q_sec_l=e_l
            )
            assert abs(Fraction(f_int) - _exact_integral(exact_s, exact_e)) < TOL_CYCLES

    def test_integral_continuous_across_rounding_boundary(self):
        # Shift the window by steps far below one ULP of the seconds of day: the
        # integral must follow F0 * shift smoothly, not jump by f_T * ULP
        t3_sec, ulp = 45678.9, np.spacing(45678.9)
        steps = np.linspace(0.0, 3.0 * ulp, 31)
        counts = []
        for d in steps:
            (s_h, s_l), (e_h, e_l), _, _ = _window(t3_sec, 30.0, 812.3 - d, 812.3 - d)
            counts.append(scb.RampTableManager.integrate(
                _afternoon_ramp(), s_h, e_h, DAY, DAY, t1_q_sec_l=s_l, t2_q_sec_l=e_l
            )[0])
        # Constant window width: only the ramp slope moves the count
        drift = RATE * 60.0 * steps
        assert np.max(np.abs(np.asarray(counts) - counts[0] - drift)) < TOL_CYCLES


class TestNewYear:
    """
    Days of year alone cannot place 1 January after 31 December, so a ramp table or
    an integration window that crosses a new year needs the years. With them the
    integral matches the same ramp written on consecutive days; without them
    integrate must refuse rather than integrate a year off.
    """

    @pytest.mark.parametrize("year, last_day", [(2018, 365), (2020, 366)])
    def test_window_across_new_year(self, year, last_day):
        ramp = _linear_ramp([last_day, 1], [T0, 100.0], years=[year, year + 1])
        t1_sec, t2_sec = 86390.0, 10.0
        f_int, _ = scb.RampTableManager.integrate(ramp, t1_sec, t2_sec, last_day, 1, year, year + 1)

        t1_elapsed, t2_elapsed = t1_sec - T0, 86400.0 + t2_sec - T0
        assert f_int == pytest.approx(_linear_integral(t1_elapsed, t2_elapsed), rel=1e-14)

        # The same ramp written on consecutive days of one year
        same = _linear_ramp([last_day, last_day + 1], [T0, 100.0])
        f_same, _ = scb.RampTableManager.integrate(same, t1_sec, t2_sec, last_day, last_day + 1)
        assert f_int == pytest.approx(f_same, rel=1e-14)

    def test_window_day_zero_of_next_year(self):
        # Sequential ranging rolls t1 back to day 0 of the t3 year
        ramp = _linear_ramp([365, 1], [T0, 100.0], years=[2018, 2019])
        f_rolled, _ = scb.RampTableManager.integrate(ramp, 86390.0, 10.0, 0, 1, 2019, 2019)
        f_dated, _ = scb.RampTableManager.integrate(ramp, 86390.0, 10.0, 365, 1, 2018, 2019)
        assert f_rolled == pytest.approx(f_dated, rel=1e-14)

    def test_new_year_without_years_raises(self):
        ramp = _linear_ramp([365, 1], [T0, 100.0])
        with pytest.raises(scb.RampTableManager.RampTableYearError):
            scb.RampTableManager.integrate(ramp, 86390.0, 10.0, 365, 1)
        # Ramp years without window years are not enough either
        with pytest.raises(scb.RampTableManager.RampTableYearError):
            scb.RampTableManager.integrate(ramp + [[2018, 2019]], 86390.0, 10.0, 365, 1)

    def test_years_do_not_change_a_pass_within_one_year(self):
        ramp = _linear_ramp([158, 158, 159], [T0, 86000.0, 500.0])
        f_plain, _ = scb.RampTableManager.integrate(ramp, 85000.0, 300.0, 158, 159)
        f_dated, _ = scb.RampTableManager.integrate(
            ramp + [[2018] * 3], 85000.0, 300.0, 158, 159, 2018, 2018
        )
        assert f_dated == f_plain


class TestFrequencyAt:
    """
    The transmit frequency at one epoch, which the solar-corona delay reads at t1.
    It must follow the ramp inside a segment, across midnight and a new year, and
    from a second of day past either end of the day it is given with.
    """

    def test_follows_the_ramp_inside_a_segment(self):
        for t in (40000.0, 45000.0, 50040.0, 70000.0):
            k = np.searchsorted(RAMP_SEC, t, side="right") - 1
            expected = RAMP_FREQ[k] + RATE * (t - RAMP_SEC[k])
            assert scb.RampTableManager.frequency_at(_afternoon_ramp(), t, DAY) == pytest.approx(expected, rel=1e-15)

    def test_across_midnight_and_new_year(self):
        ramp = _linear_ramp([365, 1], [T0, 100.0], years=[2018, 2019])
        expected = F0 + RATE * (86400.0 + 50.0 - T0)
        assert scb.RampTableManager.frequency_at(ramp, 50.0, 1, 2019) == pytest.approx(expected, rel=1e-15)
        # The same epoch as a negative second of the next day
        assert scb.RampTableManager.frequency_at(ramp, 50.0 - 86400.0, 2, 2019) == pytest.approx(expected, rel=1e-15)

    def test_low_part_moves_the_epoch(self):
        f = scb.RampTableManager.frequency_at(_afternoon_ramp(), 45000.0, DAY, t_sec_l=1e-3)
        assert f == pytest.approx(F0 + RATE * (5000.0 + 1e-3), rel=1e-15)

    def test_before_the_first_record_raises(self):
        with pytest.raises(scb.RampTableManager.RampCoverageError):
            scb.RampTableManager.frequency_at(_afternoon_ramp(), 39999.0, DAY)

# endregion Tests #
# --------------#
