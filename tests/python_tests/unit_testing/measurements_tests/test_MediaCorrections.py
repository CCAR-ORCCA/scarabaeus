# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
#=============================================#
#  Unit Tests for the MediaCorrections class  #
#=============================================#
import json
import math
import sys

import src.scarabaeus as scb

import numpy as np
import pytest

#--------------------#
# region    Fixtures #
#--------------------#
@pytest.fixture
def instrument():
    """
        Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")

# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(instrument,media_corr_file):
    """
        Verifies that object is constructed correctly.
    """
    name="GS1 Media Corrections"
    tropo_seasonal_file_path = str(media_corr_file)
    media_correction = scb.MediaCorrections(name=name, instrument=instrument, tropo_seasonal_file_path=tropo_seasonal_file_path)

    assert media_correction._name == name
    assert media_correction._instrument == instrument

def _empty_json(path):
    path.write_text(json.dumps({"rows": []}))
    return path

def test_iono_without_frequency_raises(instrument, tmp_path):
    """
        An ionosphere file cannot be scaled without the link frequency; refuse it
        before any SPICE query rather than end in a TypeError deep in the scaling.
    """
    media_correction = scb.MediaCorrections(
        name="GS1 Media Corrections", instrument=instrument,
        iono_file_path=_empty_json(tmp_path / "iono.json"),
    )
    with pytest.raises(scb.MediaCorrections.MissingIonosphereFrequencyError):
        media_correction.computed_rtlt_delays(0.0, "-64")

def test_half_troposphere_raises(instrument, tmp_path):
    """
        The troposphere needs both the seasonal and the non-seasonal file; half a
        model used to be dropped silently.
    """
    media_correction = scb.MediaCorrections(name="GS1 Media Corrections", instrument=instrument)
    media_correction._tropo_seasonal_file_path = _empty_json(tmp_path / "tro.json")
    with pytest.raises(scb.MediaCorrections.IncompleteTroposphereError):
        media_correction.computed_rtlt_delays(0.0, "-64", frequency=7.18e9)

def test_solar_corona_delay_grazing_ray():
    """
        A ray passing 5 solar radii from the Sun, with both ends ~1 AU away, gets
        the Ho (2010) Eq. (1) delay of the infinite-line STEC of the Eq. (2)
        density, in closed form. The finite ends cut ~0.3 %.
    """
    C = scb.constants
    R0 = float(C.SUN.mean_radius.values)  # [km]
    AU = 1.495978707e8  # [km]
    a_min = 5 * R0
    f_GHz = 8.4
    dt = scb.MediaCorrections._compute_SC_lt(
        np.array([1.5 * AU, a_min, 0, 0, 0, 0]), np.array([-AU, a_min, 0, 0, 0, 0]), f_GHz * 1e9
    )

    a, R = a_min * C.km_to_m, R0 * C.km_to_m  # [m]
    def line_integral(n):  # int_-inf^inf (a^2 + l^2)^(-n/2) dl
        return a ** (1 - n) * math.sqrt(math.pi) * math.gamma((n - 1) / 2) / math.gamma(n / 2)
    stec = 2.21e14 * R**6 * line_integral(6) + 1.55e12 * R**2.3 * line_integral(2.3)
    expected_us = C.SOLAR_CORONA_DELAY_COEFF * stec / f_GHz**2

    assert dt == pytest.approx(expected_us * 1e-6, rel=1e-2)

def test_zenith_delay_partials_are_the_slopes_of_the_applied_delays(instrument, monkeypatch):
    """
        The analytical zenith-delay partials the light-time models use are the
        slopes of the slant delays computed_rtlt_delays applies, with or without
        media files: the Chao wet mapping for the troposphere and the thin-shell
        mapping, scaled from S band to the link frequency, for the ionosphere.
    """
    elevation = 0.3
    radian = scb.Units.get_units(["rad"])

    class FixedElevation:
        @staticmethod
        def get_elevation_angle(**kwargs):
            return scb.ArrayWUnits(elevation, radian)

    monkeypatch.setattr(sys.modules[scb.MediaCorrections.__module__], "SpiceManager", FixedElevation)
    media = scb.MediaCorrections(name="GS1 Media Corrections", instrument=instrument)
    tropo_zenith, iono_zenith = 2.0, 0.5
    s_band = media.zenith_delay_partials(0.0, "-64", 2295e6)
    assert s_band == pytest.approx((scb.MediaCorrections.chao_mapping(elevation)[1],
                                    scb.MediaCorrections.ionosphere_mapping(elevation)), rel=1e-15)
    for frequency in (None, 0, 2295e6, 8.4e9):
        d_tropo, d_iono = media.zenith_delay_partials(0.0, "-64", frequency)
        p_tropo, p_iono = media.computed_rtlt_delays(
            0.0, "-64", frequency, tropo_zenith_delay=tropo_zenith, iono_zenith_delay=iono_zenith)
        assert float(p_tropo.values) == pytest.approx(d_tropo * tropo_zenith, rel=1e-15)
        assert float(p_iono.values) == pytest.approx(d_iono * iono_zenith, rel=1e-15)
        scale = 1.0 if not frequency else (2295e6 / frequency) ** 2
        assert d_iono == pytest.approx(scale * s_band[1], rel=1e-15)

# endregion Tests #
#--------------#
