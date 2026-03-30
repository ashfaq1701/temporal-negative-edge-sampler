#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../src/NegativeEdgeSampler.h"

namespace py = pybind11;

PYBIND11_MODULE(temporal_negative_edge_sampler, m) {
    m.doc() = "Fast temporal negative edge sampler for streaming graphs";

    py::class_<NegativeEdgeSampler>(m, "NegativeEdgeSampler")
        .def(py::init<bool, int, double, unsigned int>(),
             R"(
             Initialize a temporal negative edge sampler.

             Args:
                 is_directed (bool): Whether the graph is directed.
                 num_negatives_per_positive (int): Number of negatives to sample per positive edge.
                 historical_negative_percentage (float): Fraction of negatives from historical neighbors (default 0.5).
                 seed (int): Random seed. 0 = use random_device (default 0).
             )",
             py::arg("is_directed"),
             py::arg("num_negatives_per_positive"),
             py::arg("historical_negative_percentage") = 0.5,
             py::arg("seed") = 0)

        .def("add_batch", [](NegativeEdgeSampler& self,
                             const py::array_t<int>& sources,
                             const py::array_t<int>& targets,
                             const py::array_t<int64_t>& timestamps) {
            auto src_buf = sources.request();
            auto tgt_buf = targets.request();
            auto ts_buf = timestamps.request();

            if (src_buf.ndim != 1 || tgt_buf.ndim != 1 || ts_buf.ndim != 1) {
                throw std::runtime_error("Input arrays must be 1-dimensional");
            }
            if (src_buf.size != tgt_buf.size || src_buf.size != ts_buf.size) {
                throw std::runtime_error("Input arrays must have the same size");
            }

            std::vector<int> src_vec(
                static_cast<int*>(src_buf.ptr),
                static_cast<int*>(src_buf.ptr) + src_buf.size);
            std::vector<int> tgt_vec(
                static_cast<int*>(tgt_buf.ptr),
                static_cast<int*>(tgt_buf.ptr) + tgt_buf.size);
            std::vector<int64_t> ts_vec(
                static_cast<int64_t*>(ts_buf.ptr),
                static_cast<int64_t*>(ts_buf.ptr) + ts_buf.size);

            self.add_batch(src_vec, tgt_vec, ts_vec);
        },
        R"(
        Ingest a batch of positive edges.

        Batches must arrive in temporal order. Call sample_negatives() after
        this to obtain negative edges for the batch.

        Args:
            sources (np.ndarray[int32]): Source node IDs.
            targets (np.ndarray[int32]): Target node IDs.
            timestamps (np.ndarray[int64]): Edge timestamps.
        )",
        py::arg("sources"),
        py::arg("targets"),
        py::arg("timestamps"))

        .def("sample_negatives", [](NegativeEdgeSampler& self) {
            auto result = self.sample_negatives();

            py::array_t<int> neg_src(result.sources.size(), result.sources.data());
            py::array_t<int> neg_tgt(result.targets.size(), result.targets.data());

            py::dict out;
            out["sources"] = neg_src;
            out["targets"] = neg_tgt;
            out["num_historical_actual"] = result.num_historical_actual;
            out["num_random_actual"] = result.num_random_actual;
            return out;
        },
        R"(
        Sample negative edges for the most recently added batch.

        Must be called after add_batch(). The batch edges are merged into
        the accumulated history after sampling.

        Returns:
            dict with keys:
                sources (np.ndarray[int32]): Negative source node IDs.
                targets (np.ndarray[int32]): Negative target node IDs.
                num_historical_actual (int): Actual historical negatives produced.
                num_random_actual (int): Actual random negatives produced.
        )")

        .def("get_node_count", &NegativeEdgeSampler::get_node_count,
             "Return the number of unique nodes seen so far.")
        .def("get_edge_count", &NegativeEdgeSampler::get_edge_count,
             "Return the total number of positive edges ingested so far.")
        .def("get_batch_count", &NegativeEdgeSampler::get_batch_count,
             "Return the number of batches processed so far.");
}
