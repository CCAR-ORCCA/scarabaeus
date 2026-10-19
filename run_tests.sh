#!/bin/bash
# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC

# Runs the Rust unit tests and the Python test suite, and reports a combined result.

set -o pipefail

echo "==> Rust tests (cargo test)"
(cd src/scarabaeus_rust && cargo test --no-default-features)
rust_status=$?

echo ""
echo "==> Python tests (pytest)"
pytest
python_status=$?

echo ""
if [ $rust_status -eq 0 ] && [ $python_status -eq 0 ]; then
    echo "All tests passed."
    exit 0
else
    echo "FAILED: rust_status=$rust_status, python_status=$python_status"
    exit 1
fi
