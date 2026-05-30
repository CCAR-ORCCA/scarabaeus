#!/bin/bash
# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
set -e

# make sure all build dependencies are installed
pip install '.[dev]'

# build docs
python3 -m sphinx -b html docs/online_documentation/sphinx_files docs/online_documentation/_build

# NOTE: need access to the docs repo for this to work
# temp clone docs repo
git clone https://github.com/CCAR-ORCCA/scarabaeus-docs /tmp/scarabaeus-docs

# copy current build into docs
rsync -a --delete \
  --exclude='.git' \
  --exclude='README.md' \
  --exclude='.nojekyll' \
  docs/online_documentation/_build/ /tmp/scarabaeus-docs/

# commit to docs repo
cd /tmp/scarabaeus-docs
git add .
git commit -m "Update docs $(date '+%Y-%m-%d')"
git push

# remove temp docs
rm -rf /tmp/scarabaeus-docs