find_package(GTest CONFIG REQUIRED)

set(TEST_FILES
        test_negative_edge_sampler.cpp
        test_collect_all_negatives.cpp
)

add_executable(test_negative_edge_sampler ${TEST_FILES} ${SRC_FILES})

target_compile_features(test_negative_edge_sampler PRIVATE cxx_std_20)

target_include_directories(test_negative_edge_sampler PRIVATE
        ${CMAKE_SOURCE_DIR}/temporal_negative_edge_sampler/src
)

target_link_libraries(test_negative_edge_sampler PRIVATE
        GTest::gtest
        GTest::gtest_main
        TBB::tbb
)
