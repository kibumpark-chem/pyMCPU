#!/usr/bin/env bash
# Clean editable install + link verification + test.
#
# Usage:  bash scripts/dev_install.sh
#
# Requirements:
#   - conda/mamba environment with: pybind11, cmake, ninja, numpy, mdtraj, etc.
#   - GCC >= 14 (on FASRC: module load gcc/14.2.0-fasrc01)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1/8  Load GCC if on FASRC ==="
if command -v module &>/dev/null; then
    module load gcc/14.2.0-fasrc01 2>/dev/null && echo "Loaded gcc/14.2.0-fasrc01" || true
fi

echo "=== 2/8  Clean stale CMake artifacts ==="
rm -rf CMakeCache.txt CMakeInit.txt CMakeFiles/ _deps/ _skbuild/ .cmake/ .skbuild-info.json

echo "=== 3/8  Clean stale .so ==="
rm -f pymcpu/mcpu_core.cpython-*.so
find . -name "*.so" -not -path "./.git/*" -exec rm -f {} \; 2>/dev/null || true

echo "=== 4/8  Verify toolchain ==="
python --version
cmake --version | head -1
g++ --version | head -1
python -c "import pybind11; print('pybind11 cmake dir:', pybind11.get_cmake_dir())"

echo "=== 5/8  Editable install ==="
pip install --no-build-isolation -e ".[dev]"

echo "=== 6/8  Import check ==="
python -c "import pymcpu.mcpu_core as m; print('mcpu_core:', m.__file__)"

echo "=== 7/8  Install verification ==="
python scripts/install_check.py

echo "=== 8/8  pytest ==="
python -m pytest -q || true

echo "=== Done ==="
