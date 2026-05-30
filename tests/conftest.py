# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Scarabaeus main testing suite configuration. Performs setup necessary
to run

Setup includes:
    - check for necessary testing data and generate it if missing
    - clear kernel pool and load SPICE metakernel for each module
"""

import src.scarabaeus as scb
from scarabaeus import ArrayWUnits as awu

import json, os
from pathlib import Path

import pytest
import numpy as np
import spiceypy as spice

J2000 = scb.Frame("J2000")
kg, km, sec, rad = scb.Units.get_units(["kg", "km", "sec", "rad"])


@pytest.fixture(scope="session", autouse=True)
def setup(tmp_path_factory):
    """
    Initialization fixture for all tests. Loads neccesary
    files and ensure a clear kernel pool.
    """
    # clear kernel pool
    spice.kclear()

    # generate meta kernel and load
    meta_kernel = meta_kernels(tmp_path_factory)
    spice.furnsh(str(meta_kernel))

    # load everything else
    spice.furnsh(str(earth_bplane_(tmp_path_factory)))
    json.load(open(nplate_config_json(tmp_path_factory)))
    json.load(open(trk_data_json(tmp_path_factory)))
    json.load(open(earth_config_json(tmp_path_factory)))
    json.load(open(media_corr_json(tmp_path_factory)))
    json.load(open(meas_data_json(tmp_path_factory)))

def meta_kernels(tmp_path_factory):
    """Generate metakernel used across suite."""
    # Use CWD-relative paths (same convention as data/kernels/locked/locked_generic.tm).
    # SPICE resolves relative paths from the process CWD when furnsh is called, not
    # from the metakernel file's directory, so the temp location doesn't matter.
    # Relative paths are always short (<80 chars), bypassing SPICE's per-string limit.
    config_path = tmp_path_factory.mktemp("data") / "locked_generic.tm"

    config_path.write_text(
        "KPL/MK\n"
        r"\begindata" + "\n"
        "\n"
        "KERNELS_TO_LOAD = (\n"
        "    'data/kernels/locked/ck/cas00084.tsc',\n"
        "    'data/kernels/locked/lsk/naif0012.tls',\n"
        "    'data/kernels/locked/spk/de430.bsp',\n"
        "    'data/kernels/scenario/jup310.bsp',\n"
        "    'data/kernels/scenario/mar097.bsp',\n"
        "    'data/kernels/scenario/sat360.bsp',\n"
        "    'data/kernels/locked/pck/pck00010.tpc',\n"
        "    'data/kernels/locked/spk/earthstns_fx_201023.bsp',\n"
        "    'data/kernels/locked/spk/earth_200101_990628_predict.bpc',\n"
        "    'data/kernels/locked/spk/earth_topo_201023.tf')\n"
        "\n"
        r"\begintext"
    )

    return config_path


def nplate_config_json(tmp_path_factory):
    """Generate JSON config used in N-plate tests."""
    data = {
        "names": ["plate 1", "plate 2"],
        "areas": [1e-06, 1e-06],
        "ref coeffs": [1.5, 1.1],
        "abs coeffs": [0.33, 0.33],
        "dref coeffs": [0.33, 0.33],
        "sref coeffs": [0.34, 0.34],
        "normal vectors": [[1, 0, 0], [-1, 0, 0]],
        "types": ["Fixed", "CK"],
        "ids": [None, -64027],
    }

    # place in temp directory
    config_path = tmp_path_factory.mktemp("data") / "nplate_config.json"
    config_path.write_text(json.dumps(data))

    # return file path when requested
    return config_path


@pytest.fixture(scope="session")
def nplate_config_file(tmp_path_factory):
    """Fetch N-plate config file."""
    file = nplate_config_json(tmp_path_factory)
    yield file

    # remove file upon testing completion
    os.remove(file)


def trk_data_json(tmp_path_factory):
    """Generate tracking config used in measurement tests."""
    data = {
        "name": "beno",
        "2W_doppler": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [134516, 180000, 225501],
            "meas": [-8445848561.5, -8445848522.9, -8445848484.3],
            "spice_id": [35, 35, 35],
            "outlier_flag": [0, 0, 0],
            "scft_transpd_delay": [0.0, 0.0, 0.0],
            "array_delay": [0.0, 0.0, 0.0],
            "rcv_time_tag_delay": [0.0, 0.0, 0.0],
            "transmit_time_tag_delay": [0.0, 0.0, 0.0],
        },
        "2W_sranging": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [134516, 180000, 225501],
            "meas": [-1, -1, -1],
            "spice_id": [35, 35, 35],
            "outlier_flag": [0, 0, 0],
        },
        "aux_SFDU_9": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [134516, 180000, 225501],
            "ramp_freq": [
                6184394368.0,
                6184394368.0,
                76184394368.0,
            ],
            "ramp_rate": [0.0, 0.0, 0.0],
            "ramp_type": [3, 0, 1],
        },
        "aux_SFDU_0": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [134516, 180000, 225501],
            "exc_scalar_num": [1, 1, 1],
            "exc_scalar_den": [16, 16, 16],
            "rng_modulo": [1, 1, 1],
            "rng_type": [0, 0, 0],
        },
        "aux_SFDU_7": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [59320.0, 59444.0, 59568.0],
            "rng_modulo": [1048576, 1048576, 1048576],
            "exc_scalar_num": [221, 221, 221],
            "exc_scalar_den": [23968, 23968, 23968],
            "ul_stn_cal": [0.0, 0.0, 0.0],
            "dl_stn_cal": [0.0, 0.0, 0.0],
            "ul_freq": [0.0, 0.0, 0.0],
            "rng_type": [1, 1, 1],
            "outlier_flag": [0, 0, 0],
        },
        "aux_SFDU_16": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [59320.0, 59444.0, 59568.0],
            "Tc": [60.0, 60.0, 60.0],
            "M2_num": [880, 880, 880],
            "M2_den": [749, 749, 749],
            "outlier_flag": [0, 0, 0],
        },
        "aux_SFDU_2": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [59320.0, 59444.0, 59568.0],
            "ul_zheight_corr": [0.0, 0.0, 0.0],
            "ul_cal_freq": [0.0, 0.0, 0.0],
        },
        "aux_SFDU_3": {
            "year": [2018, 2018, 2018],
            "doy": [102, 102, 102],
            "sec": [59320.0, 59444.0, 59568.0],
            "dl_zheight_corr": [0.0, 0.0, 0.0],
        },
    }

    # place in temp directory
    trk_path = tmp_path_factory.mktemp("data") / "trk_data.json"
    trk_path.write_text(json.dumps(data))

    # return file path when requested
    return trk_path


@pytest.fixture(scope="session")
def trk_data_file(tmp_path_factory):
    """Fetch tracking config file."""
    file = trk_data_json(tmp_path_factory)
    yield file

    # remove file upon testing completion
    os.remove(file)


def earth_config_json(tmp_path_factory):
    """
    Creates a temporary JSON config for SHG tests.
    """
    data = {
        "BodyName": "Earth",
        "ReferenceRadius": 6378.14,
        "GravitationalParameter": 398600.435436,
        "Normalized": False,
        "MaxDegree": 100,
        "Coefficients": [
            {"degree": 0, "order": 0, "Cnm": 9.9968e-01, "Snm": 0.0},
            {"degree": 1, "order": 0, "Cnm": 3.4757e-05, "Snm": 0.0},
        ],
    }

    # place in temp directory
    config_path = tmp_path_factory.mktemp("data") / "earth_config.json"
    config_path.write_text(json.dumps(data))

    # return file path when requested
    return config_path


@pytest.fixture(scope="session")
def earth_config_file(tmp_path_factory):
    """Fixture for Earth configuration file."""
    file = earth_config_json(tmp_path_factory)

    yield file
    # remove file upon testing completion
    os.remove(file)


def media_corr_json(tmp_path_factory):
    """
    Creates a temporary JSON config for media correction tests.
    """
    data = {
        "aux_name": "1972",
        "aux_tropo": {
            "C1": [
                31557600.0,
                31557600.0,
                31557600.0,
                31557600.0,
                31557600.0,
                31557600.0,
            ],
            "C2": [0.087, 2.0521, 0.1149, 2.1579, 0.1255, 2.1094],
        },
    }

    # place in temp directory
    config_path = tmp_path_factory.mktemp("data") / "media_corr_config.json"
    config_path.write_text(json.dumps(data))

    # return file path when requested
    return config_path


@pytest.fixture(scope="session")
def media_corr_file(tmp_path_factory):
    """Fixture for media correction configuration file."""

    file = media_corr_json(tmp_path_factory)
    yield file
    # remove file upon testing completion
    os.remove(file)


def earth_bplane_(tmp_path_factory):
    """
    Creates a temporary SPICE frame kernel defining the Earth B-plane frame for testing.
    """
    fk_content = r"""
        \begindata
        FRAME_EARTH_BPLANE   = 12345678
        FRAME_12345678_NAME   = 'EARTH_BPLANE'
        FRAME_12345678_CLASS   = 5
        FRAME_12345678_CLASS_ID   = 12345678
        FRAME_12345678_CENTER   = '3'
        FRAME_12345678_RELATIVE   = 'J2000'
        FRAME_12345678_DEF_STYLE   = 'PARAMETERIZED'
        FRAME_12345678_FAMILY   = 'TWO-VECTOR'

        FRAME_12345678_PRI_AXIS   = 'Z'
        FRAME_12345678_PRI_VECTOR_DEF   = 'OBSERVER_TARGET_VELOCITY'
        FRAME_12345678_PRI_OBSERVER   = '3'
        FRAME_12345678_PRI_TARGET   = '301'
        FRAME_12345678_PRI_ABCORR   = 'NONE'
        FRAME_12345678_PRI_FRAME   = 'J2000'

        FRAME_12345678_SEC_AXIS   = '-Y'
        FRAME_12345678_SEC_VECTOR_DEF   = 'CONSTANT'
        FRAME_12345678_SEC_SPEC   = 'RECTANGULAR'
        FRAME_12345678_SEC_VECTOR = (0, 0, 1)
        FRAME_12345678_SEC_FRAME   = 'J2000'
        FRAME_12345678_FREEZE_EPOCH   = @2029-APR-13/21:48:32.902099

        \begintext

        End of FK File.
        """

    fk_file = tmp_path_factory.mktemp("data") / "earth_bplane.tf"
    fk_file.write_text(fk_content)

    # return file path when requested
    return fk_file


@pytest.fixture(scope="session")
def earth_bplane_fk(tmp_path_factory):
    """Fixture for Earth B-plane frame kernel."""

    file = earth_bplane_(tmp_path_factory)
    yield file
    # remove file upon testing completion
    os.remove(file)


def meas_data_json(tmp_path_factory):
    """Creates a temporary directory for measurement data file."""

    data = {
        "name": "ideal_range",
        "meas_ideal": {
            "year": [2028, 2028, 2028],
            "doy": [39, 39, 39],
            "sec": [
                19868.24503994,
                19928.24503994,
                19988.245039821,
            ],
            "meas": [20000.0, 21000.0, 22000.0],
            "spice_id": 399014,
            "outlier_flag": [0, 0, 0],
        },
    }
    meas_path = tmp_path_factory.mktemp("data") / "meas_ideal.json"
    meas_path.write_text(json.dumps(data))

    # return file path when requested
    return meas_path


@pytest.fixture(scope="session")
def meas_data_file(tmp_path_factory):
    """Fixture for measurement data file."""
    file = meas_data_json(tmp_path_factory)
    yield file
    # remove file upon testing completion
    os.remove(file)
