cmake_minimum_required(VERSION 3.24)
project(temporal_negative_edge_sampler)

set(CMAKE_CXX_STANDARD 20)

# Use directly provided Python paths if available
if(DEFINED PYTHON_EXECUTABLE AND DEFINED PYTHON_INCLUDE_DIR AND DEFINED PYTHON_LIBRARY)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
    set(Python3_LIBRARIES ${PYTHON_LIBRARY})
    message(STATUS "Using provided Python paths:")
    message(STATUS "  Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
    message(STATUS "  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "  Python3_LIBRARIES: ${Python3_LIBRARIES}")
else()
    # Fall back to automatic detection
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Get pybind11 include path using the correct Python executable
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")

set(PYBIND11_PYTHON_VERSION ${DETECTED_PYTHON_VERSION} CACHE STRING "Python version used for pybind11")
message(STATUS "Setting pybind11 Python version to: ${PYBIND11_PYTHON_VERSION}")

find_package(pybind11 REQUIRED)
find_package(TBB REQUIRED)

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_negative_edge_sampler/src/NegativeEdgeSampler.cpp
)

option(BUILD_TESTS "Build C++ tests" ON)

add_subdirectory(temporal_negative_edge_sampler/py_interface)

if(BUILD_TESTS)
    add_subdirectory(temporal_negative_edge_sampler/tests)
endif()
