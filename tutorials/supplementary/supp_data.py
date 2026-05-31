# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""
Supplementary Data

Provides functionality to assist with data management within Scarabaeus (SCB) tutorials.

Contains a function `load_data()` which acts as the main interface for any tutorial
data set. All files are stored under ``tutorials/supplementary/data/``, which is also
the working directory set at import time so that relative paths such as
``"data/kernels/..."`` resolve correctly from any tutorial:

    supplementary/data/
      kernels/
        locked/          ← general locked kernels (lsk, pck, spk, ck) + metakernel
        scenario/        ← mission-specific kernels, trajectory BSPs, solution files
      dynamic_setup/
        nplate_coefficients/
        sph_coefficients/
        thruster_coefficients/
      measurements/
        radiometric/
        optical/
        media_correction/

If a user has the populated data/ folder (e.g. from a full repo checkout), load_data()
finds everything already present. Otherwise it creates the folder structure and
downloads/generates any missing files.
"""

import os
import sys
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Callable, Optional, List

from supplementary import file_gen


@dataclass
class FileInfo:
    """File structure."""

    path: str
    comes_from: Literal["gen", "download", "existing"]
    generator: Optional[Callable] = None
    link: Optional[str] = None


class Data(Enum):
    """Data manager for SCB tutorials."""

    # general
    mk = FileInfo(
        path="kernels/locked/locked_generic.tm",
        comes_from="gen",
        generator=file_gen.gen_mk,
    )
    """ General SCB tutorials metakernel. """

    lsk = FileInfo(
        path="kernels/locked/lsk/naif0012.tls",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    )
    """ NAIF leapsecond kernel. """

    pck = FileInfo(
        path="kernels/locked/pck/pck00010.tpc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    )
    """ Planetary constants kernel. """

    gmk = FileInfo(
        path="kernels/locked/pck/gm_de431.tpc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de431.tpc",
    )
    """ GM planetary constants (DE431). """

    earth_pck = FileInfo(
        path="kernels/locked/spk/earth_200101_990628_predict.bpc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/a_old_versions/earth_200101_990628_predict.bpc",
    )
    """ Long-term Earth orientation predict kernel. """

    stns_bsp = FileInfo(
        path="kernels/locked/spk/earthstns_fx_201023.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/earthstns_fx_201023.bsp",
    )
    """ DSN ground-station positions/velocities. """

    earth_topo = FileInfo(
        path="kernels/locked/spk/earth_topo_201023.tf",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/EUROPACLIPPER/kernels/fk/earth_topo_201023.tf",
    )
    """ Earth topocentric frames kernel. """

    de432s = FileInfo(
        path="kernels/locked/spk/de432s.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp",
    )
    """ JPL planetary ephemeris DE432s. """

    mars_spk = FileInfo(
        path="kernels/locked/spk/mar099s.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar099s.bsp",
    )

    jupiter_spk = FileInfo(
        path="kernels/locked/spk/jup348.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup348.bsp",
    )

    saturn_spk = FileInfo(
        path="kernels/locked/spk/sat456.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat456.bsp",
    )

    uranus_spk = FileInfo(
        path="kernels/locked/spk/ura184_part-3.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/ura184_part-3.bsp",
    )

    neptune_spk = FileInfo(
        path="kernels/locked/spk/nep097.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/nep097.bsp",
    )

    Cassini_sclk = FileInfo(
        path="kernels/locked/ck/cas00084.tsc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/sclk/cas00084.tsc",
    )
    """ Cassini spacecraft clock coefficients. """

    de424 = FileInfo(
        path="kernels/locked/spk/de424.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/de424.bsp",
    )
    """ JPL planetary ephemeris DE424. """

    stns_itrf93 = FileInfo(
        path="kernels/locked/spk/earthstns_itrf93_201023.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/earthstns_itrf93_201023.bsp",
    )
    """ DSN ground-station positions/velocities (ITRF93 frame). """

    # provided files
    earth_sph_config = FileInfo(
        path="dynamic_setup/sph_coefficients/Earth_100.json",
        comes_from="gen",
        generator=file_gen.gen_earth_sph_config,
    )

    example_meas_spec = FileInfo(
        path="dynamic_setup/thruster_coefficients/example.mission_maneuver_spec",
        comes_from="existing",
    )

    """ Seasonal tropospheric coefficients. """
    tropo_seasonal = FileInfo(
        path="measurements/media_correction/2018_152_2018_182_tro.json",
        comes_from="existing",
    )

    """ Seasonal tropospheric coefficients. """
    tropo_seasonal_new = FileInfo(
        path="measurements/media_correction/1972_001_2048_001_tro_modified.json",
        comes_from="existing",
    )

    """Non-seasonal tropospheric"""
    tropo_nonseasonal = FileInfo(
        path="measurements/media_correction/2010_001_2010_032_tro.json",
        comes_from="existing",
    )

    """ Radiometric tracking file """
    trk_file = FileInfo(
        path="measurements/radiometric/orex_beno_2018_159_054533_2018_159_104001_35.json",
        comes_from="existing",
    )

    # OSIRIS-REx scenario kernels
    OREx_mk = FileInfo(
        path="kernels/scenario/orex_mk.tm",
        comes_from="gen",
        generator=file_gen.gen_orex_mk,
    )
    """ OSIRIS-REx tutorial metakernel (data/kernels/scenario/orex_mk.tm). """

    OREx_real_data_mk = FileInfo(
        path="kernels/scenario/orex_real_data.tm",
        comes_from="gen",
        generator=file_gen.gen_orex_real_data_mk,
    )
    """ OSIRIS-REx real-data metakernel with paths resolved to supplementary/data/. """

    OREx_ck_sc = FileInfo(
        path="kernels/scenario/orx_sc_rel_210816_210822_v02.bc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/ck/orx_sc_rel_210816_210822_v02.bc",
    )
    """ OSIRIS-REx spacecraft C-kernel. """

    OREx_ck_sa = FileInfo(
        path="kernels/scenario/orx_sa_rel_210816_210822_v02.bc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/ck/orx_sa_rel_210816_210822_v02.bc",
    )
    """ OSIRIS-REx solar-array C-kernel. """

    OREx_sclk = FileInfo(
        path="kernels/scenario/ORX_SCLKSCET_00075.tsc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/ORX/kernels/sclk/ORX_SCLKSCET.00075.tsc",
    )
    """ OSIRIS-REx spacecraft clock coefficients. """

    OREx_frames = FileInfo(
        path="kernels/scenario/orx_v14.tf",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/ORX/kernels/fk/orx_v14.tf",
    )
    """ OSIRIS-REx frames kernel. """

    OREx_nplate = FileInfo(
        path="dynamic_setup/nplate_coefficients/orex_nplate_tut_config.json",
        comes_from="gen",
        generator=file_gen.gen_orex_nplate_config,
    )
    """ OSIRIS-REx N-plate model configuration. """

    OREx_spk_merged = FileInfo(
        path="kernels/scenario/orx_210115_230917_230915_merged_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_210115_230917_230915_merged_v1.bsp",
    )
    """ OSIRIS-REx merged trajectory SPK (2021–2023). """

    OREx_spk_od023 = FileInfo(
        path="kernels/scenario/orx_160909_171201_170830_od023_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_160909_171201_170830_od023_v1.bsp",
    )
    """ OSIRIS-REx OD023 trajectory SPK. """

    OREx_spk_od027 = FileInfo(
        path="kernels/scenario/orx_170501_180710_171005_od027_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_170501_180710_171005_od027_v1.bsp",
    )
    """ OSIRIS-REx OD027 trajectory SPK. """

    OREx_spk_od030 = FileInfo(
        path="kernels/scenario/orx_170923_180710_180125_od030_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_170923_180710_180125_od030_v1.bsp",
    )
    """ OSIRIS-REx OD030 trajectory SPK. """

    OREx_spk_od031 = FileInfo(
        path="kernels/scenario/orx_170923_180710_180321_od031_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_170923_180710_180321_od031_v1.bsp",
    )
    """ OSIRIS-REx OD031 trajectory SPK. """

    OREx_spk_od044 = FileInfo(
        path="kernels/scenario/orx_180301_181201_180921_od044_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_180301_181201_180921_od044_v1.bsp",
    )
    """ OSIRIS-REx OD044 trajectory SPK. """

    OREx_spk_od077 = FileInfo(
        path="kernels/scenario/orx_180801_190302_181218_od077_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_180801_190302_181218_od077_v1.bsp",
    )
    """ OSIRIS-REx OD077 trajectory SPK. """

    OREx_spk_od297 = FileInfo(
        path="kernels/scenario/orx_201020_210524_210103_od297_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_201020_210524_210103_od297_v1.bsp",
    )
    """ OSIRIS-REx OD297 trajectory SPK. """

    OREx_spk_od302 = FileInfo(
        path="kernels/scenario/orx_210101_210330_210310_od302_v1.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/orx_210101_210330_210310_od302_v1.bsp",
    )
    """ OSIRIS-REx OD302 trajectory SPK. """

    OREx_bennu_pck = FileInfo(
        path="kernels/scenario/bennu_v17.tpc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/pck/bennu_v17.tpc",
    )
    """ Bennu shape/orientation constants (v17). """

    OREx_earth_bpc = FileInfo(
        path="kernels/scenario/earth_000101_250127_241031.bpc",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/pck/earth_000101_250127_241031.bpc",
    )
    """ Earth binary PCK (orientation coverage through 2025). """

    OREx_jup_spk = FileInfo(
        path="kernels/scenario/jup310.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/jup310.bsp",
    )
    """ Jupiter system satellite ephemeris (JUP310). """

    OREx_sat_spk = FileInfo(
        path="kernels/scenario/sat360.bsp",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/sat360.bsp",
    )
    """ Saturn system satellite ephemeris (SAT360). """

    # Validation kernels
    earth_assoc_itrf93 = FileInfo(
        path="kernels/validation/earth_assoc_itrf93.tf",
        comes_from="download",
        link="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_assoc_itrf93.tf",
    )
    """ Earth/ITRF93 topocentric station frames association kernel. """

    # make path simpler to get
    @property
    def path(self) -> str:
        """Absolute path to this data file."""
        return str(file_gen.DATA_PATH / self.value.path)


def load_data() -> type[Data]:
    """
    Ensure all tutorial data files are present under ``data/``.

    Files that already exist (e.g. from a full repo checkout) are left
    untouched. Missing files are downloaded from the internet or generated
    locally after prompting the user.

    Returns
    -------
    Data
        Enum class whose members expose ``.path`` to each data file.
    """
    data = Data

    # if file_gen.DATA_PATH.exists():
    #     # already have tutorial data folder -> check for missing files

    missing_folders = []
    missing_files: List[FileInfo] = []

    # check folders
    for folder in file_gen.DATA_SKELETON:
        if not (file_gen.DATA_PATH / folder).is_dir():
            missing_folders.append(folder)

    # check files
    for item in Data:
        file_path = file_gen.DATA_PATH / item.value.path
        if not file_path.exists():
            missing_files.append(item)

    if not missing_folders and not missing_files:
        #  have everything -> notify and return data
        print("SCB supplementary data up to date.")
        return data

    else:
        # missing one or more things -> ask if ok to download/gen
        print("SCB supplementary data out of date.")
        waiting = True
        while waiting:
            response = input("\nDownload and generate missing files? (y/n): ")
            match response.strip().lower():
                case "y":
                    n_downloads = sum(
                        1 for e in missing_files if e.value.comes_from == "download"
                    )
                    print("Received: (y), retrieving/generating missing files...")
                    if n_downloads >= 8:
                        print(
                            f"\n  Heads up: {n_downloads} kernels to download from NAIF. "
                            "This could take up to 20 minutes depending on your connection.\n"
                            "  Go grab a coffee!\n"
                        )
                    elif n_downloads >= 3:
                        print(
                            f"\n  Downloading {n_downloads} kernels from NAIF. "
                            "Sit tight, this may take a few minutes.\n"
                        )

                    # generate missing folders
                    for folder in missing_folders:
                        (file_gen.DATA_PATH / folder).mkdir(parents=True, exist_ok=True)

                    # then files
                    for entry in missing_files:
                        match entry.value.comes_from:
                            case "gen":
                                entry.value.generator()
                                print(f"  | Generated:  {entry.value.path}")
                            case "download":
                                try:
                                    urllib.request.urlretrieve(
                                        entry.value.link,
                                        file_gen.DATA_PATH / entry.value.path,
                                    )
                                except Exception:
                                    print(f"  | FAILED download: {entry.value.path}")
                                else:
                                    print(f"  | Downloaded: {entry.value.path}")
                            case "existing":
                                print(
                                    f"  | Missing provided file: {entry.value.path}. Pull from repository."
                                )
                    waiting = False

                    print("SCB supplementary data up to date.")
                    return data
                case "n":
                    print("Received: (n), exiting...")
                    sys.exit()
                case _:
                    print(f"Unknown response: ({response})")
