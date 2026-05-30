# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import src.scarabaeus as scb
import numpy as np  
import pytest
kg, km, sec = scb.Units.get_units(["kg", "km", "sec"])

#--------------------#
# region    Fixtures #
#--------------------#
@pytest.fixture
def computed_meas(trk_data_file):
    """
        Fixture to create a sample computed mass for testing.
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
    # Sequential-Ranging auxiliary data
    "RU": (
        (
            (
                np.array(trk_data["aux_SFDU_7"]["exc_scalar_den"])
                / np.array(trk_data["aux_SFDU_7"]["exc_scalar_num"])
            )
            / 16
        )
        ** -1
    ).tolist(),
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

@pytest.fixture
def state_dummy():
    state_dummy = [
    ("position", 3, "estimated", "dynamic", None, None),
    (
        "velocity",
        3,
        "estimated",
        "dynamic",
        None,
        None,
    ),
    (
        "range_bias_1",
        1,
        "estimated",
        "static",
        scb.GroundStation("DSS-14"),
        None,
    ),
]
    return state_dummy
# endregion Fixtures #
#--------------------#

#--------------#
# region Tests #
#--------------#
def test_initialization(computed_meas, state_dummy):
    """
        Verifies that object is constructed correctly.
    """
    name = "GS1 Doppler Model"
    GS1 = scb.GroundStation("DSS-14")
    doppler_sigma = scb.ArrayWUnits(1e-3, sec)

    DopplerReal_GS1 = scb.DopplerReal(name, GS1, sigma=doppler_sigma, computed_measurements_dict=computed_meas, state_definition=state_dummy)

    assert DopplerReal_GS1._name == name
    assert DopplerReal_GS1._instrument == GS1
    assert DopplerReal_GS1._sigma == doppler_sigma
    assert DopplerReal_GS1._computed_measurements_dict == computed_meas
    assert DopplerReal_GS1._state_definition == state_dummy
