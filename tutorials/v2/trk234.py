"""Auxiliary data for the real DSN measurement models (chapter 9).

TO BE UPSTREAMED. ``DopplerReal`` and ``SequentialRangingReal`` need auxiliary
data (count times, transponder ratios, ramp tables, calibrations and delays)
that is stored in the converted TRK-2-34 file but has no library loader yet.
This module builds those inputs; it belongs in Scarabaeus (e.g. as
``DopplerReal.from_trk234`` / ``SequentialRangingReal.from_trk234``).

The mapping and constants are taken from the original OSIRIS-REx real-data
tutorial (``tutorials/advanced_RealMSR_OSIRIS_REx_OD.ipynb``).
"""

import json
import os

import numpy as np
import scarabaeus as scb

# OSIRIS-REx transponder delay [RU] and its value in seconds, as used by the original tutorial.
SCFT_TRANSPD_DELAY_RU = 16529.719234434557
SCFT_TRANSPD_DELAY_S = SCFT_TRANSPD_DELAY_RU / ((221.0 / (749.0 * 2.0)) * 8445.767679 * 1e6)


def load(path):
    """Read a converted TRK-2-34 file (JSON)."""
    with open(path) as f:
        return json.load(f)


def station_name(trk, block="2W_doppler"):
    """DSN station of a tracking block, e.g. 'DSS-35'."""
    return f"DSS-{trk[block]['spice_id'][0]}"


def _ramps(trk):
    ramp = trk["aux_SFDU_9"]
    return {
        "ramp_sec": ramp["sec"],
        "ramp_day": ramp["doy"],
        "ramp_freq": ramp["ramp_freq"],
        "ramp_rate": ramp["ramp_rate"],
        "ramp_type": ramp["ramp_type"],
    }


def doppler_aux_data(trk):
    """Auxiliary data for ``DopplerReal(computed_measurements_dict=...)``."""
    sfdu2, sfdu3, sfdu7, sfdu16 = trk["aux_SFDU_2"], trk["aux_SFDU_3"], trk["aux_SFDU_7"], trk["aux_SFDU_16"]
    doppler = trk["2W_doppler"]
    zeros = (np.array(sfdu2["ul_zheight_corr"]) * 0).tolist()
    # Ranging-derived quantities are asynchronous with Doppler: use the unique (maximum) value.
    from_ranging = {key: np.max(np.unique(sfdu7[key]))
                    for key in ("ul_freq", "ul_stn_cal", "dl_stn_cal", "exc_scalar_num", "exc_scalar_den")}
    return {
        # Doppler count data (SFDU 16)
        "Tc": sfdu16["Tc"],
        "M2_num": sfdu16["M2_num"],
        "M2_den": sfdu16["M2_den"],
        "outlier_flag_doppler": sfdu16["outlier_flag"],
        "year_SFDU16": sfdu16["year"],
        "doy_SFDU16": sfdu16["doy"],
        "sec_SFDU16": sfdu16["sec"],
        # Delays
        "ul_zheight_corr": sfdu2["ul_zheight_corr"],
        "dl_zheight_corr": sfdu3["dl_zheight_corr"],
        "scft_transpd_delay": SCFT_TRANSPD_DELAY_RU,
        "scft_transpd_delay_s": SCFT_TRANSPD_DELAY_S,
        "transmit_time_tag_delay": doppler["transmit_time_tag_delay"],
        "rcv_time_tag_delay": doppler["rcv_time_tag_delay"],
        "array_delay": doppler["array_delay"],
        **{key: [z + value for z in zeros] for key, value in from_ranging.items()},
        "RU": 0.14753004005340453,
        **_ramps(trk),
    }


def sranging_aux_data(trk):
    """Auxiliary data for ``SequentialRangingReal(computed_measurements_dict=...)``."""
    sfdu2, sfdu3, sfdu7, sfdu16 = trk["aux_SFDU_2"], trk["aux_SFDU_3"], trk["aux_SFDU_7"], trk["aux_SFDU_16"]
    doppler = trk["2W_doppler"]
    return {
        # Sequential ranging data (SFDU 7)
        "RU": ((np.array(sfdu7["exc_scalar_den"]) / np.array(sfdu7["exc_scalar_num"]) / 16) ** -1).tolist(),
        "M": sfdu7["rng_modulo"],
        "M2_num": sfdu16["M2_num"],
        "M2_den": sfdu16["M2_den"],
        "rng_type": sfdu7["rng_type"],
        "outlier_flag_sranging": sfdu7["outlier_flag"],
        "year_SFDU7": sfdu7["year"],
        "doy_SFDU7": sfdu7["doy"],
        "sec_SFDU7": sfdu7["sec"],
        # Delays and calibrations
        "ul_freq": sfdu7["ul_freq"],
        "ul_zheight_corr": sfdu2["ul_zheight_corr"],
        "dl_zheight_corr": sfdu3["dl_zheight_corr"],
        "scft_transpd_delay": SCFT_TRANSPD_DELAY_RU,
        "scft_transpd_delay_s": SCFT_TRANSPD_DELAY_S,
        "transmit_time_tag_delay": doppler["transmit_time_tag_delay"],
        "rcv_time_tag_delay": doppler["rcv_time_tag_delay"],
        "array_delay": doppler["array_delay"],
        "ul_stn_cal": sfdu7["ul_stn_cal"],
        "dl_stn_cal": sfdu7["dl_stn_cal"],
        "exc_scalar_num": sfdu7["exc_scalar_num"],
        "exc_scalar_den": sfdu7["exc_scalar_den"],
        **_ramps(trk),
    }


def media_corrections(station, first_epoch, seasonal_file, nonseasonal_dir, name):
    """``MediaCorrections`` for a pass: seasonal troposphere plus, when available,
    the non-seasonal troposphere and ionosphere files covering the pass day."""
    corrections = scb.MediaCorrections(name=name, instrument=station, tropo_seasonal_file_path=seasonal_file)
    year, doy, _ = scb.SpiceManager.et2YDS([float(first_epoch.times.values)])
    tropo = scb.MediaCorrections.find_tro_metadata(nonseasonal_dir, year[0], doy[0])
    iono = scb.MediaCorrections.find_ion_metadata(nonseasonal_dir, year[0], doy[0])
    if tropo:
        corrections.tropo_non_seasonal_file_path = os.path.join(nonseasonal_dir, tropo["file"])
    if iono:
        corrections.iono_file_path = os.path.join(nonseasonal_dir, iono["file"])
    return corrections
