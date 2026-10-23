# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==============================================#
#  Functional Tests for the DopplerReal class  #
# ==============================================#
"""
Requires the OSIRIS-REx tracking data under data/; skipped without it.
"""
import os

import numpy as np
import pytest

import src.scarabaeus as scb

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DATA = os.path.join(ROOT, "data")
RADIOMETRIC = os.path.join(DATA, "measurements", "radiometric")

# First OSIRIS-REx pass of 2018 DoY 158 (DSS-65): the uplink ramp starts at
# 68128 s, after the transmit time of the first Doppler rows
OREX_TRACKING = os.path.join(RADIOMETRIC, "orex_beno_2018_158_184515_2018_158_230501_65.json")
OREX_METAKERNEL = os.path.join(DATA, "kernels", "orex_real_data.tm")

needs_orex = pytest.mark.skipif(
    not (os.path.isfile(OREX_TRACKING) and os.path.isfile(OREX_METAKERNEL)),
    reason="OSIRIS-REx tracking data not available",
)


# --------------------#
# region    Fixtures #
# --------------------#

@pytest.fixture(scope="module")
def orex_kernels():
    cwd = os.getcwd()
    os.chdir(ROOT)              # metakernel carries repo-relative paths
    scb.SpiceManager.clear_kernels()
    scb.SpiceManager.load_kernel_from_mkfile(OREX_METAKERNEL)
    yield
    os.chdir(cwd)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestDopplerReal:
    """Ramp coverage of generate_measurement_dataset on an OSIRIS-REx pass (DSS-65)."""

    @needs_orex
    def test_uncovered_round_trips_are_nan_and_flagged(self, orex_kernels):
        """
        A count interval that starts before the first ramp record has no
        defined transmit frequency. It is computed as NaN with a warning and
        flagged as an outlier, rather than integrating a zero frequency or
        raising and losing the whole pass.
        """
        kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])
        gs = scb.GroundStation("DSS-65")
        model = scb.DopplerReal("t", gs, sigma=scb.ArrayWUnits(1e-3, 1 / sec))
        # The built-in NAIF name of -64: a new name would be registered with
        # boddef, which kclear does not undo, and clash with later test modules
        orbiter = scb.Spacecraft(
            "OSIRIS-REX", -64, scb.ArrayWUnits(1200.0, kg),
            scb.ArrayWUnits(1.2e-05, km**2), scb.ArrayWUnits(0.8, None),
        )
        J2000 = scb.Frame.generate_common_frames()[0]

        obs = model.observed_measurements(
            file_name=OREX_TRACKING, meas_name="2W_doppler", units=1 / sec
        )
        with pytest.warns(UserWarning, match="before the first ramp record"):
            ds = model.generate_measurement_dataset(
                "t", target=orbiter, observed_meas=obs, frame=J2000
            )

        computed = np.asarray(ds.data["computed"], dtype=float)
        flags = np.asarray(ds.data["outlier_flag"]).astype(bool)
        undefined = ~np.isfinite(computed)

        # Only the head of the pass is uncovered; the rest is computed and kept
        assert undefined[0] and not undefined.all()
        assert np.flatnonzero(undefined).max() < np.flatnonzero(~undefined).min()
        assert np.array_equal(flags & undefined, undefined), "NaN rows must be flagged"
        assert np.all(np.isfinite(computed[~flags]))

# endregion Tests #
# --------------#
