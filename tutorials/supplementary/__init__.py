# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# path to folder to place any tutorial results in
from pathlib import Path
RESULTS_PATH = str(Path(__file__).parent / 'tutorial_results')

# import tutorial stuff
from supplementary import file_gen
from supplementary.supp_data import load_data
from supplementary import supp_plotting as plotting