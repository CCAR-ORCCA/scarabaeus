# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from pathlib import Path
import json, os, shutil

# Root data/ folder — lives inside the supplementary package so all tutorial
# reads, writes, and generated kernels (scenario BSPs, solution files, etc.)
# are co-located with the tutorial support code.
DATA_PATH = Path(__file__).parent / 'supp_data/'

# Full directory skeleton that mirrors the repo's data/ layout.
# load_data() creates every folder listed here on first run.
DATA_SKELETON = [
    'dynamic_setup/nplate_coefficients',
    'dynamic_setup/sph_coefficients',
    'dynamic_setup/thruster_coefficients',
    'kernels/locked/ck',
    'kernels/locked/lsk',
    'kernels/locked/pck',
    'kernels/locked/spk',
    'kernels/scenario',
    'kernels/validation',
    'measurements/media_correction',
    'measurements/optical',
    'measurements/radiometric',
]


def gen_mk() -> None:
    """ Generate the general tutorial metakernel at data/kernels/locked/locked_generic.tm. """
    with open(DATA_PATH / 'kernels/locked/locked_generic.tm', 'w') as f:
        f.write('KPL/MK\n'
                r'\begindata' + '\n'
                '\n'
                "PATH_VALUES = ( 'supplementary/supp_data/kernels/locked' )\n"
                "PATH_SYMBOLS = ( 'LOCKED' )\n"
                '\n'
                "KERNELS_TO_LOAD = (\n"
                f"    '$LOCKED/ck/cas00084.tsc',\n"
                f"    '$LOCKED/lsk/naif0012.tls',\n"
                f"    '$LOCKED/spk/de432s.bsp',\n"
                f"    '$LOCKED/pck/pck00010.tpc',\n"
                f"    '$LOCKED/pck/gm_de431.tpc',\n"
                f"    '$LOCKED/spk/earthstns_fx_201023.bsp',\n"
                f"    '$LOCKED/spk/earth_200101_990628_predict.bpc',\n"
                f"    '$LOCKED/spk/earth_topo_201023.tf',\n"
                f"    '$LOCKED/spk/mar099s.bsp',\n"
                f"    '$LOCKED/spk/jup348.bsp',\n"
                f"    '$LOCKED/spk/sat456.bsp',\n"
                f"    '$LOCKED/spk/ura184_part-3.bsp',\n"
                f"    '$LOCKED/spk/nep097.bsp')\n"
                '\n'
                r'\begintext')


def gen_orex_mk() -> None:
    """ Generate the OSIRIS-REx metakernel at data/kernels/scenario/orex_mk.tm. """
    scenario = os.path.relpath(DATA_PATH / 'kernels/scenario')
    lsk      = os.path.relpath(DATA_PATH / 'kernels/locked/lsk')
    with open(DATA_PATH / 'kernels/scenario/orex_mk.tm', 'w') as f:
        f.write('KPL/MK\n'
                r'\begindata' + '\n'
                '\n'
                "PATH_VALUES = ( 'supplementary/supp_data/kernels/scenario',\n"
                "                 'supplementary/supp_data/kernels/locked')\n"
                "PATH_SYMBOLS = ( 'SCENARIO', 'LOCKED' )\n"
                '\n'
                "KERNELS_TO_LOAD = (\n"
                f"    '$SCENARIO/orx_sc_rel_210816_210822_v02.bc',\n"
                f"    '$SCENARIO/orx_sa_rel_210816_210822_v02.bc',\n"
                f"    '$SCENARIO/ORX_SCLKSCET_00075.tsc',\n"
                f"    '$LOCKED/lsk/naif0012.tls',\n"
                f"    '$SCENARIO/orx_v14.tf')\n"
                '\n'
                r'\begintext')


def gen_orex_real_data_mk() -> None:
    """ Generate the OSIRIS-REx real-data metakernel at data/kernels/scenario/orex_real_data.tm. """
    scenario   = os.path.relpath(DATA_PATH / 'kernels/scenario')
    lsk        = os.path.relpath(DATA_PATH / 'kernels/locked/lsk')
    locked_pck = os.path.relpath(DATA_PATH / 'kernels/locked/pck')
    locked_spk = os.path.relpath(DATA_PATH / 'kernels/locked/spk')
    validation = os.path.relpath(DATA_PATH / 'kernels/validation')
    with open(DATA_PATH / 'kernels/scenario/orex_real_data.tm', 'w') as f:
        f.write('KPL/MK\n'
                r'\begindata' + '\n'
                '\n'
                "KERNELS_TO_LOAD = (\n"
                f"    '{scenario}/orx_210115_230917_230915_merged_v1.bsp',\n"
                f"    '{scenario}/orx_160909_171201_170830_od023_v1.bsp',\n"
                f"    '{scenario}/orx_170501_180710_171005_od027_v1.bsp',\n"
                f"    '{scenario}/orx_170923_180710_180125_od030_v1.bsp',\n"
                f"    '{scenario}/orx_170923_180710_180321_od031_v1.bsp',\n"
                f"    '{scenario}/orx_180301_181201_180921_od044_v1.bsp',\n"
                f"    '{scenario}/orx_180801_190302_181218_od077_v1.bsp',\n"
                f"    '{scenario}/orx_210101_210330_210310_od302_v1.bsp',\n"
                f"    '{scenario}/orx_201020_210524_210103_od297_v1.bsp',\n"
                f"    '{scenario}/orx_v14.tf',\n"
                f"    '{scenario}/bennu_v17.tpc',\n"
                f"    '{scenario}/earth_000101_250127_241031.bpc',\n"
                f"    '{lsk}/naif0012.tls',\n"
                f"    '{locked_pck}/pck00010.tpc',\n"
                f"    '{scenario}/jup310.bsp',\n"
                f"    '{scenario}/sat360.bsp',\n"
                f"    '{locked_spk}/de424.bsp',\n"
                f"    '{locked_spk}/earthstns_itrf93_201023.bsp',\n"
                f"    '{locked_spk}/earth_topo_201023.tf',\n"
                f"    '{validation}/earth_assoc_itrf93.tf')\n"
                '\n'
                r'\begintext')


def gen_earth_sph_config() -> None:
    """ Copy Earth_100.json from the repo data/ folder into supp_data/. """
    src = Path(__file__).parents[2] / 'data/dynamic_setup/sph_coefficients/Earth_100.json'
    dst = DATA_PATH / 'dynamic_setup/sph_coefficients/Earth_100.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def gen_example_meas_spec() -> None:
    """ Copy example.mission_maneuver_spec from the repo data/ folder into supp_data/. """
    src = Path(__file__).parents[2] / 'data/dynamic_setup/thruster_coefficients/example.mission_maneuver_spec'
    dst = DATA_PATH / 'kernels/scenario/example.mission_maneuver_spec'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def gen_orex_nplate_config() -> None:
    """ Generate the OSIRIS-REx N-plate config at data/dynamic_setup/nplate_coefficients/. """
    config_dict = {
        'names'          : ['plusX',   'minusX',   'ORX_SA_PY_IG',  'ORX_SA_NY_IG'],
        'areas'          : [1e-06,      1e-06,      1e-06,            1e-06         ],
        'ref coeffs'     : [1.5,        1.5,        1.1,              1.1           ],
        'abs coeffs'     : [0.33,       0.33,       0.33,             0.33          ],
        'dref coeffs'    : [0.33,       0.33,       0.33,             0.33          ],
        'sref coeffs'    : [0.34,       0.34,       0.34,             0.34          ],
        'normal vectors' : [[1, 0, 0], [-1, 0, 0], [1, 0, 0],        [1, 0, 0]     ],
        'types'          : ['Fixed',   'Fixed',    'CK',              'CK'          ],
        'ids'            : [None,       None,      -64017,            -64027        ],
    }
    out = DATA_PATH / 'dynamic_setup/nplate_coefficients/orex_nplate_tut_config.json'
    with open(out, 'w') as f:
        json.dump(config_dict, f, indent=4)
