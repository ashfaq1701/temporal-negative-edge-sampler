#include <gtest/gtest.h>
#include "NegativeEdgeSampler.h"
#include <unordered_set>
#include <vector>
#include <algorithm>

class NegativeEdgeSamplerTest : public ::testing::Test {
protected:
    std::unordered_set<int> all_nodes{0, 1, 2, 3, 4};

    std::vector<int> batch_sources{0, 1, 2};
    std::vector<int> batch_targets{1, 2, 3};
    int64_t batch_timestamp = 100;
};

// Test output size matches expected dimensions
TEST_F(NegativeEdgeSamplerTest, OutputSizeMatchesExpected) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 3;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 0.5);

    EXPECT_EQ(neg_src.size(), batch_sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), batch_sources.size() * k);
}

// Test that negative sources match positive sources
TEST_F(NegativeEdgeSamplerTest, NegativeSourcesMatchPositiveSources) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 0.5);

    for (size_t i = 0; i < batch_sources.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            EXPECT_EQ(neg_src[i * k + j], batch_sources[i]);
        }
    }
}

// Test that negative targets are valid node IDs (or -1 sentinel)
TEST_F(NegativeEdgeSamplerTest, NegativeTargetsAreValidNodes) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 0.5);

    for (int tgt : neg_tgt) {
        EXPECT_TRUE(tgt == -1 || all_nodes.count(tgt) > 0);
    }
}

// Test directed mode produces correct output size
TEST_F(NegativeEdgeSamplerTest, DirectedModeOutputSize) {
    NegativeEdgeSampler sampler(all_nodes, true);
    int k = 2;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 0.5);

    EXPECT_EQ(neg_src.size(), batch_sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), batch_sources.size() * k);
}

// Test with single edge
TEST_F(NegativeEdgeSamplerTest, SingleEdgeBatch) {
    NegativeEdgeSampler sampler(all_nodes, false);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    int k = 3;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        src, tgt, 100, k, 0.5);

    EXPECT_EQ(neg_src.size(), static_cast<size_t>(k));
    EXPECT_EQ(neg_tgt.size(), static_cast<size_t>(k));

    for (int i = 0; i < k; ++i) {
        EXPECT_EQ(neg_src[i], 0);
    }
}

// Test full historical percentage (1.0)
TEST_F(NegativeEdgeSamplerTest, FullHistoricalPercentage) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 1.0);

    EXPECT_EQ(neg_src.size(), batch_sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), batch_sources.size() * k);
}

// Test full random percentage (0.0)
TEST_F(NegativeEdgeSamplerTest, FullRandomPercentage) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        batch_sources, batch_targets, batch_timestamp, k, 0.0);

    EXPECT_EQ(neg_src.size(), batch_sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), batch_sources.size() * k);
}

// Test that historical negatives come from previously seen edges
TEST_F(NegativeEdgeSamplerTest, HistoricalNegativesFromPreviousBatches) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    // First batch: establish history
    std::vector<int> src1{0, 1};
    std::vector<int> tgt1{1, 2};
    sampler.sample_negative_edges_per_batch(src1, tgt1, 100, k, 0.5);

    // Second batch: different edges, should have historical candidates
    std::vector<int> src2{0, 2};
    std::vector<int> tgt2{3, 4};
    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        src2, tgt2, 200, k, 1.0);

    EXPECT_EQ(neg_src.size(), src2.size() * k);
    EXPECT_EQ(neg_tgt.size(), src2.size() * k);

    // All negatives should be valid nodes or -1
    for (int tgt : neg_tgt) {
        EXPECT_TRUE(tgt == -1 || all_nodes.count(tgt) > 0);
    }
}

// Test requesting more negatives than available unique candidates
TEST_F(NegativeEdgeSamplerTest, MoreNegativesThanCandidates) {
    std::unordered_set<int> small_nodes{0, 1, 2};
    NegativeEdgeSampler sampler(small_nodes, false);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    int k = 10;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        src, tgt, 100, k, 0.5);

    // Should still return k entries per positive edge
    EXPECT_EQ(neg_src.size(), static_cast<size_t>(k));
    EXPECT_EQ(neg_tgt.size(), static_cast<size_t>(k));
}

// Test multiple sequential batches build state correctly
TEST_F(NegativeEdgeSamplerTest, MultipleBatchesAccumulateState) {
    NegativeEdgeSampler sampler(all_nodes, false);
    int k = 2;

    // Process 3 batches sequentially
    std::vector<int> src1{0}, tgt1{1};
    std::vector<int> src2{1}, tgt2{2};
    std::vector<int> src3{2}, tgt3{3};

    auto [ns1, nt1] = sampler.sample_negative_edges_per_batch(src1, tgt1, 100, k, 0.5);
    auto [ns2, nt2] = sampler.sample_negative_edges_per_batch(src2, tgt2, 200, k, 0.5);
    auto [ns3, nt3] = sampler.sample_negative_edges_per_batch(src3, tgt3, 300, k, 0.5);

    EXPECT_EQ(ns1.size(), static_cast<size_t>(k));
    EXPECT_EQ(ns2.size(), static_cast<size_t>(k));
    EXPECT_EQ(ns3.size(), static_cast<size_t>(k));
}

// Test with two-node graph (minimal case)
TEST_F(NegativeEdgeSamplerTest, TwoNodeGraph) {
    std::unordered_set<int> two_nodes{0, 1};
    NegativeEdgeSampler sampler(two_nodes, false);

    std::vector<int> src{0};
    std::vector<int> tgt{1};
    int k = 1;

    auto [neg_src, neg_tgt] = sampler.sample_negative_edges_per_batch(
        src, tgt, 100, k, 0.5);

    EXPECT_EQ(neg_src.size(), static_cast<size_t>(1));
    EXPECT_EQ(neg_tgt.size(), static_cast<size_t>(1));
}
