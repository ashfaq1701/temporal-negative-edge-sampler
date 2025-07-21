#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "NegativeEdgeSampler.h"

namespace py = pybind11;

std::pair<py::array_t<int>, py::array_t<int>> collect_all_negatives_by_timestamp_wrapper(
    const py::array_t<int>& sources_array,
    const py::array_t<int>& targets_array,
    const py::array_t<int64_t>& timestamps_array,
    bool is_directed,
    int num_negatives_per_positive,
    double historical_negative_percentage
) {
    // Get buffer info for input arrays
    py::buffer_info sources_buf = sources_array.request();
    py::buffer_info targets_buf = targets_array.request();
    py::buffer_info timestamps_buf = timestamps_array.request();

    // Validate input dimensions
    if (sources_buf.ndim != 1 || targets_buf.ndim != 1 || timestamps_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }

    if (sources_buf.size != targets_buf.size || sources_buf.size != timestamps_buf.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    // Convert numpy arrays to std::vector
    std::vector<int> sources(static_cast<int*>(sources_buf.ptr),
                            static_cast<int*>(sources_buf.ptr) + sources_buf.size);
    std::vector<int> targets(static_cast<int*>(targets_buf.ptr),
                            static_cast<int*>(targets_buf.ptr) + targets_buf.size);
    std::vector<int64_t> timestamps(static_cast<int64_t*>(timestamps_buf.ptr),
                                   static_cast<int64_t*>(timestamps_buf.ptr) + timestamps_buf.size);

    // Call the actual function
    auto [neg_sources, neg_targets] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, is_directed,
        num_negatives_per_positive,
        historical_negative_percentage
    );

    auto neg_sources_array = py::cast(neg_sources);
    auto neg_targets_array = py::cast(neg_targets);

    return std::make_pair(neg_sources_array, neg_targets_array);
}


PYBIND11_MODULE(temporal_negative_edge_sampler, m) {
    m.doc() = "Fast temporal negative edge sampler";

    m.def("collect_all_negatives_by_timestamp",
          &collect_all_negatives_by_timestamp_wrapper,
          "Sample negative edges for temporal graphs",
          py::arg("sources"),
          py::arg("targets"),
          py::arg("timestamps"),
          py::arg("is_directed"),
          py::arg("num_negatives_per_positive"),
          py::arg("historical_negative_percentage")=0.5);
}
