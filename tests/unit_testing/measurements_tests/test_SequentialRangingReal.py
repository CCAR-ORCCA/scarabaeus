# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np
import pytest


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture
def instrument():
    """
    Fixture to create a sample measurement for testing.
    """
    return scb.GroundStation("DSS-14")


@pytest.fixture
def computed_meas_dict(trk_data_file):
    """
    Fixture to create a sample computed measurements dictionary for testing.
    """
    trk_data = scb.Utils.load_json(str(trk_data_file))
    computed_meas_dict = {
        # Doppler auxiliary data
        "Tc": trk_data["aux_SFDU_16"]["Tc"],
        "M2_num": trk_data["aux_SFDU_16"]["M2_num"],
        "M2_den": trk_data["aux_SFDU_16"]["M2_den"],
        "rcv_time_tag_delay": trk_data["2W_doppler"]["rcv_time_tag_delay"],
        "transmit_time_tag_delay": trk_data["2W_doppler"]["transmit_time_tag_delay"],
        "outlier_flag_doppler": trk_data["aux_SFDU_16"]["outlier_flag"],
        "year_SFDU16": trk_data["aux_SFDU_16"]["year"],
        "doy_SFDU16": trk_data["aux_SFDU_16"]["doy"],
        "sec_SFDU16": trk_data["aux_SFDU_16"]["sec"],
        "M": trk_data["aux_SFDU_7"]["rng_modulo"],
        "rng_type": trk_data["aux_SFDU_7"]["rng_type"],
        "outlier_flag_sranging": trk_data["aux_SFDU_7"]["outlier_flag"],
        "ul_stn_cal": trk_data["aux_SFDU_7"]["ul_stn_cal"],
        "dl_stn_cal": trk_data["aux_SFDU_7"]["dl_stn_cal"],
        "ul_freq": trk_data["aux_SFDU_7"]["ul_freq"],
        "year_SFDU7": trk_data["aux_SFDU_7"]["year"],
        "doy_SFDU7": trk_data["aux_SFDU_7"]["doy"],
        "sec_SFDU7": trk_data["aux_SFDU_7"]["sec"],
        # Delays auxiliary data
        "ul_zheight_corr": trk_data["aux_SFDU_2"]["ul_zheight_corr"],
        "ul_cal_freq": trk_data["aux_SFDU_2"]["ul_cal_freq"],
        "dl_zheight_corr": trk_data["aux_SFDU_3"]["dl_zheight_corr"],
        # Ramp table auxiliary (from SFDU_9)
        "ramp_sec": trk_data["aux_SFDU_9"]["sec"],
        "ramp_day": trk_data["aux_SFDU_9"]["doy"],
        "ramp_freq": trk_data["aux_SFDU_9"]["ramp_freq"],
        "ramp_rate": trk_data["aux_SFDU_9"]["ramp_rate"],
        "ramp_type": trk_data["aux_SFDU_9"]["ramp_type"],
    }
    return computed_meas_dict


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that object is constructed correctly.
    """
    name = "GS1 Sequential Ranging Model"
    sranging_sigma = scb.ArrayWUnits(3, None)
    SequentialRangingReal_GS1 = scb.SequentialRangingReal(
        name,
        instrument,
        sigma=sranging_sigma,
        computed_measurements_dict=computed_meas_dict,
    )

    assert SequentialRangingReal_GS1._name == name
    assert SequentialRangingReal_GS1._instrument == instrument
    assert SequentialRangingReal_GS1._sigma == sranging_sigma
