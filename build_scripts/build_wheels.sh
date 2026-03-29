#!/bin/bash
set -e
echo "Starting build process..."

# Directory to store repaired wheels
mkdir -p /project/wheelhouse

# Python versions to build for
PYTHON_VERSIONS=("python3.8" "python3.9" "python3.10" "python3.11")

# Build wheels for each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    if command -v $PYVER &> /dev/null; then
        echo "=========================================="
        echo "Building and repairing wheel for $PYVER"
        echo "=========================================="

        # Install build dependencies
        $PYVER -m pip install --upgrade pip
        $PYVER -m pip install -r /project/requirements.txt
        $PYVER -m pip install auditwheel

        cd /project

        # Find actual compilers
        export CC=$(which gcc)
        export CXX=$(which g++)
        echo "Using compilers:"
        echo "  CC: $CC"
        echo "  CXX: $CXX"

        # Set Python paths as environment variables
        export PYTHON_EXECUTABLE=$(which $PYVER)
        export PYTHON_INCLUDE_DIR=$($PYVER -c "from sysconfig import get_paths; print(get_paths()['include'])")

        # Find Python library
        PYTHON_VERSION=$($PYVER -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_LIBDIR=$($PYVER -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

        PYTHON_LIBRARY=""
        POSSIBLE_LOCATIONS=(
            "$PYTHON_LIBDIR/libpython${PYTHON_VERSION}.so"
            "$PYTHON_LIBDIR/libpython${PYTHON_VERSION}m.so"
            "/usr/lib64/libpython${PYTHON_VERSION}.so"
            "/usr/lib64/libpython${PYTHON_VERSION}m.so"
            "/usr/lib/libpython${PYTHON_VERSION}.so"
            "/usr/lib/libpython${PYTHON_VERSION}m.so"
            "/usr/local/lib/libpython${PYTHON_VERSION}.so"
            "/usr/local/lib/libpython${PYTHON_VERSION}.so.1.0"
            "/usr/local/lib/libpython${PYTHON_VERSION}m.so"
            "/usr/local/lib64/libpython${PYTHON_VERSION}.so"
            "/usr/local/lib64/libpython${PYTHON_VERSION}m.so"
        )

        for loc in "${POSSIBLE_LOCATIONS[@]}"; do
            if [ -f "$loc" ]; then
                PYTHON_LIBRARY="$loc"
                break
            fi
        done

        # If still not found, try to find any matching library
        if [ -z "$PYTHON_LIBRARY" ]; then
            echo "Searching for Python library in common directories..."
            PYTHON_LIBRARY=$(find /usr -name "libpython${PYTHON_VERSION}*.so*" | head -1)
        fi

        # Skip if not found
        if [ -z "$PYTHON_LIBRARY" ]; then
            echo "ERROR: Could not find Python library for ${PYVER}"
            continue
        fi

        export PYTHON_LIBRARY

        echo "Using Python paths:"
        echo "  PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE"
        echo "  PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR"
        echo "  PYTHON_LIBRARY: $PYTHON_LIBRARY"

        # Clean previous builds
        rm -rf build *.egg-info

        # Build wheel
        $PYVER setup.py bdist_wheel

        # Get Python version for wheel pattern
        PY_VER_SHORT=$(echo $PYTHON_VERSION | tr -d '.')

        # Find the wheel
        WHEEL_FILES=(/project/dist/*-cp${PY_VER_SHORT}-cp${PY_VER_SHORT}-linux*.whl)
        if [ ${#WHEEL_FILES[@]} -gt 0 ] && [ -f "${WHEEL_FILES[0]}" ]; then
            WHEEL="${WHEEL_FILES[0]}"
            echo "Built wheel: $WHEEL"

            # Repair the wheel
            # Rocky Linux 9 is based on RHEL 9, which corresponds to manylinux_2_34_x86_64
            echo "Repairing wheel..."
            $PYVER -m auditwheel repair "$WHEEL" --plat manylinux_2_34_x86_64 -w /project/wheelhouse/

            echo "Wheel repair complete"
        else
            echo "ERROR: No wheel was built for $PYVER"
        fi

        # Clean up build artifacts
        rm -rf build *.egg-info

        echo "Completed processing for $PYVER"
        echo "=========================================="
    else
        echo "Python version $PYVER not found, skipping..."
    fi
done

# Check if any wheels were built successfully
if [ "$(ls -A /project/wheelhouse/*.whl 2>/dev/null)" ]; then
    echo "All wheels built successfully!"
    echo "Wheels are available in /project/wheelhouse"
    ls -la /project/wheelhouse/*.whl
else
    echo "No wheels were built or repair process failed."
    echo "Check the build logs for errors."
    exit 1
fi
