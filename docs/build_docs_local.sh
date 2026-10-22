#!/bin/bash
# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
set -e

# Build the docs locally only
cd "$(dirname "$0")/.."

python3 docs/online_documentation/sphinx_files/release_notes/release_gen.py
python3 -m sphinx -b html docs/online_documentation/sphinx_files docs/online_documentation/_build

echo "Docs built at $(pwd)/docs/online_documentation/_build/index.html"
