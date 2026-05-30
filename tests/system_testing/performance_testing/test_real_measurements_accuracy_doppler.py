# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import os
import runpy
import matplotlib

def test_real_measurements_accuracy_doppler():
    script_path = os.path.join("V&V", "RealMeasAccuracy_doppler.py")
    
    matplotlib.use('Agg')  # Use a non-GUI backend to prevent plots from rendering
    runpy.run_path(script_path)