pybind11_add_module(temporal_negative_edge_sampler
        negative_edge_sampler.cpp
        ${SRC_FILES}
)

target_compile_features(temporal_negative_edge_sampler PRIVATE cxx_std_20)

target_include_directories(temporal_negative_edge_sampler PRIVATE
        ${CMAKE_SOURCE_DIR}/temporal_negative_edge_sampler/src
)

target_link_libraries(temporal_negative_edge_sampler PRIVATE TBB::tbb)
