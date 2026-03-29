#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build-ci}"

echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Using build directory: ${BUILD_DIR}"

# Configure
cmake -S . -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v python)"

# Build C++ test executable
cmake --build "${BUILD_DIR}" --target temporal_negative_edge_sampler_test --parallel

# Run C++ test
"${BUILD_DIR}/temporal_negative_edge_sampler_test"

# Install Python package
python -m pip install .

# Run Python tests if they exist
if [ -d "py_tests" ]; then
    python -m pytest py_tests -v --maxfail=1
else
    echo "No py_tests directory found, skipping Python tests."
fi
