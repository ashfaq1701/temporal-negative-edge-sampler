#include <gtest/gtest.h>
#include "NegativeEdgeSampler.h"
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <set>

// ============================================================================
// Construction
// ============================================================================

TEST(NegativeEdgeSamplerTest, ConstructWithDefaults) {
    NegativeEdgeSampler sampler(false, 3);
    EXPECT_EQ(sampler.get_node_count(), 0u);
    EXPECT_EQ(sampler.get_edge_count(), 0u);
    EXPECT_EQ(sampler.get_batch_count(), 0u);
}

TEST(NegativeEdgeSamplerTest, ConstructWithAllParams) {
    NegativeEdgeSampler sampler(true, 5, 0.7, 42);
    EXPECT_EQ(sampler.get_node_count(), 0u);
}

// ============================================================================
// Output dimensions
// ============================================================================

TEST(NegativeEdgeSamplerTest, OutputSizeMatchesExpected) {
    NegativeEdgeSampler sampler(false, 3, 0.5, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 9u);  // 3 edges * 3 negatives
    EXPECT_EQ(result.targets.size(), 9u);
}

TEST(NegativeEdgeSamplerTest, NegativeSourcesMatchPositiveSources) {
    const int k = 2;
    NegativeEdgeSampler sampler(false, k, 0.5, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    for (size_t i = 0; i < src.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            EXPECT_EQ(result.sources[i * k + j], src[i]);
        }
    }
}

// ============================================================================
// First batch: all random (no history)
// ============================================================================

TEST(NegativeEdgeSamplerTest, FirstBatchAllRandom) {
    NegativeEdgeSampler sampler(false, 2, 0.5, 42);
    std::vector<int> src{0, 1};
    std::vector<int> tgt{1, 2};
    std::vector<int64_t> ts{100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.num_historical_actual, 0);
}

// ============================================================================
// Second batch has historical negatives
// ============================================================================

TEST(NegativeEdgeSamplerTest, SecondBatchHasHistorical) {
    NegativeEdgeSampler sampler(false, 2, 1.0, 42);

    // Batch 1: establish history
    std::vector<int> s1{0, 1};
    std::vector<int> t1{1, 2};
    std::vector<int64_t> ts1{100, 100};
    sampler.add_batch(s1, t1, ts1);
    sampler.sample_negatives();

    // Batch 2: different edges
    std::vector<int> s2{0, 2};
    std::vector<int> t2{3, 4};
    std::vector<int64_t> ts2{200, 200};
    sampler.add_batch(s2, t2, ts2);
    auto result = sampler.sample_negatives();

    // Node 0 had neighbor 1 in batch 1; node 2 had neighbors 1,3 (undirected) in batch 1.
    // Historical candidates should exist.
    EXPECT_GT(result.num_historical_actual, 0);
}

// ============================================================================
// Negative targets are valid
// ============================================================================

TEST(NegativeEdgeSamplerTest, NegativeTargetsAreValidNodes) {
    NegativeEdgeSampler sampler(false, 2, 0.5, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    std::set<int> valid_nodes{0, 1, 2, 3};
    for (int t : result.targets) {
        EXPECT_TRUE(t == -1 || valid_nodes.count(t) > 0)
            << "Invalid target: " << t;
    }
}

// ============================================================================
// Random negatives exclude neighbors and self
// ============================================================================

TEST(NegativeEdgeSamplerTest, RandomNegativesExcludeNeighborsAndSelf) {
    // 5 nodes, 1 edge: 0->1. Undirected, so 0 and 1 are mutual neighbors.
    // Random negatives for src=0 should not include 0 (self) or 1 (neighbor).
    NegativeEdgeSampler sampler(false, 10, 0.0, 42);
    std::vector<int> src{0, 2, 3};  // 5 nodes: 0,1,2,3,4
    std::vector<int> tgt{1, 4, 4};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    // Check negatives for src=0 (first 10 entries)
    for (int j = 0; j < 10; ++j) {
        int t = result.targets[j];
        if (t != -1) {
            EXPECT_NE(t, 0) << "Random negative should not be self";
            EXPECT_NE(t, 1) << "Random negative should not be current batch neighbor";
        }
    }
}

// ============================================================================
// Directed mode
// ============================================================================

TEST(NegativeEdgeSamplerTest, DirectedModeOutput) {
    NegativeEdgeSampler sampler(true, 2, 0.5, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 6u);
    EXPECT_EQ(result.targets.size(), 6u);
}

TEST(NegativeEdgeSamplerTest, DirectedDoesNotCreateReverseEdges) {
    // Directed: edge 0->1 should NOT make 1 a neighbor of 0 in reverse.
    // So for src=1, node 0 should be a valid random negative.
    NegativeEdgeSampler sampler(true, 5, 0.0, 42);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    std::vector<int64_t> ts{100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    // src=0's negatives should not include 1 (neighbor) or 0 (self).
    // Only node available is nothing (only 2 nodes, 0 excluded as self, 1 excluded as neighbor).
    // All should be -1 sentinels.
    for (int j = 0; j < 5; ++j) {
        EXPECT_EQ(result.targets[j], -1);
    }
}

// ============================================================================
// Single edge batch
// ============================================================================

TEST(NegativeEdgeSamplerTest, SingleEdgeBatch) {
    NegativeEdgeSampler sampler(false, 3, 0.5, 42);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    std::vector<int64_t> ts{100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 3u);
    EXPECT_EQ(result.targets.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(result.sources[i], 0);
    }
}

// ============================================================================
// Complete graph returns sentinels
// ============================================================================

TEST(NegativeEdgeSamplerTest, CompleteGraphReturnsSentinels) {
    // 3 nodes all connected: 0-1, 0-2, 1-2. Undirected.
    // Each node is connected to all others => no valid random negatives.
    NegativeEdgeSampler sampler(false, 2, 0.0, 42);
    std::vector<int> src{0, 0, 1};
    std::vector<int> tgt{1, 2, 2};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    // All targets should be -1 since every node is connected to all others.
    for (int t : result.targets) {
        EXPECT_EQ(t, -1);
    }
}

// ============================================================================
// Two-node graph
// ============================================================================

TEST(NegativeEdgeSamplerTest, TwoNodeGraph) {
    NegativeEdgeSampler sampler(false, 1, 0.5, 42);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    std::vector<int64_t> ts{100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 1u);
    EXPECT_EQ(result.targets.size(), 1u);
    // Only 2 nodes, both connected => sentinel
    EXPECT_EQ(result.targets[0], -1);
}

// ============================================================================
// Multiple batches accumulate state
// ============================================================================

TEST(NegativeEdgeSamplerTest, MultipleBatchesAccumulateState) {
    NegativeEdgeSampler sampler(false, 2, 0.5, 42);

    std::vector<int> s1{0}, t1{1};
    std::vector<int64_t> ts1{100};
    sampler.add_batch(s1, t1, ts1);
    sampler.sample_negatives();
    EXPECT_EQ(sampler.get_node_count(), 2u);
    EXPECT_EQ(sampler.get_edge_count(), 1u);
    EXPECT_EQ(sampler.get_batch_count(), 1u);

    std::vector<int> s2{1}, t2{2};
    std::vector<int64_t> ts2{200};
    sampler.add_batch(s2, t2, ts2);
    sampler.sample_negatives();
    EXPECT_EQ(sampler.get_node_count(), 3u);
    EXPECT_EQ(sampler.get_edge_count(), 2u);
    EXPECT_EQ(sampler.get_batch_count(), 2u);

    std::vector<int> s3{2}, t3{3};
    std::vector<int64_t> ts3{300};
    sampler.add_batch(s3, t3, ts3);
    sampler.sample_negatives();
    EXPECT_EQ(sampler.get_node_count(), 4u);
    EXPECT_EQ(sampler.get_edge_count(), 3u);
    EXPECT_EQ(sampler.get_batch_count(), 3u);
}

// ============================================================================
// Getters update after add_batch
// ============================================================================

TEST(NegativeEdgeSamplerTest, NodeCountAfterAddBatch) {
    NegativeEdgeSampler sampler(false, 2, 0.5, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    // Nodes visible after add_batch, before sample_negatives
    EXPECT_EQ(sampler.get_node_count(), 4u);
}

// ============================================================================
// More negatives than candidates (sampling with replacement)
// ============================================================================

TEST(NegativeEdgeSamplerTest, MoreNegativesThanCandidates) {
    // 3 nodes: 0, 1, 2. Edge 0-1. Random negatives for src=0: only candidate is 2.
    // Requesting 10 negatives should give 10 entries (sampling with replacement from {2}).
    NegativeEdgeSampler sampler(false, 10, 0.0, 42);
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    std::vector<int64_t> ts{100};

    // Need a third node visible. Add another edge in same batch.
    std::vector<int> src2{0, 2};
    std::vector<int> tgt2{1, 1};
    std::vector<int64_t> ts2{100, 100};

    sampler.add_batch(src2, tgt2, ts2);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 20u);  // 2 edges * 10 negatives
    EXPECT_EQ(result.targets.size(), 20u);
}

// ============================================================================
// Historical negatives are from history, not current batch
// ============================================================================

TEST(NegativeEdgeSamplerTest, HistoricalNegativesFromHistoryNotCurrentBatch) {
    NegativeEdgeSampler sampler(false, 3, 1.0, 42);

    // Batch 1: 0-1, 0-2
    std::vector<int> s1{0, 0};
    std::vector<int> t1{1, 2};
    std::vector<int64_t> ts1{100, 100};
    sampler.add_batch(s1, t1, ts1);
    sampler.sample_negatives();

    // Batch 2: 0-3. Historical for src=0 should be {1, 2} (from batch 1, not in batch 2).
    std::vector<int> s2{0};
    std::vector<int> t2{3};
    std::vector<int64_t> ts2{200};
    sampler.add_batch(s2, t2, ts2);
    auto result = sampler.sample_negatives();

    std::set<int> hist_candidates{1, 2};
    for (int j = 0; j < 3; ++j) {
        int t = result.targets[j];
        if (t != -1) {
            EXPECT_TRUE(hist_candidates.count(t) > 0)
                << "Historical negative " << t << " not from history";
        }
    }
    EXPECT_GT(result.num_historical_actual, 0);
}

// ============================================================================
// Full historical percentage
// ============================================================================

TEST(NegativeEdgeSamplerTest, FullHistoricalPercentage) {
    NegativeEdgeSampler sampler(false, 2, 1.0, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 6u);
    EXPECT_EQ(result.targets.size(), 6u);
}

// ============================================================================
// Full random percentage
// ============================================================================

TEST(NegativeEdgeSamplerTest, FullRandomPercentage) {
    NegativeEdgeSampler sampler(false, 2, 0.0, 42);
    std::vector<int> src{0, 1, 2};
    std::vector<int> tgt{1, 2, 3};
    std::vector<int64_t> ts{100, 100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 6u);
    EXPECT_EQ(result.targets.size(), 6u);
    // First batch, so all should be random regardless
    EXPECT_EQ(result.num_historical_actual, 0);
}

// ============================================================================
// Larger graph across multiple batches
// ============================================================================

TEST(NegativeEdgeSamplerTest, LargerGraph) {
    NegativeEdgeSampler sampler(false, 3, 0.5, 42);

    // 10 nodes, 20 edges across 5 timestamps
    for (int batch = 0; batch < 5; ++batch) {
        std::vector<int> src, tgt;
        std::vector<int64_t> ts;
        for (int i = 0; i < 4; ++i) {
            int idx = batch * 4 + i;
            src.push_back(idx % 10);
            tgt.push_back((idx + 1) % 10);
            ts.push_back(100 + batch * 100);
        }
        sampler.add_batch(src, tgt, ts);
        auto result = sampler.sample_negatives();

        EXPECT_EQ(result.sources.size(), static_cast<size_t>(4 * 3));
        EXPECT_EQ(result.targets.size(), static_cast<size_t>(4 * 3));
    }

    EXPECT_EQ(sampler.get_batch_count(), 5u);
    EXPECT_EQ(sampler.get_edge_count(), 20u);
    EXPECT_EQ(sampler.get_node_count(), 10u);
}

// ============================================================================
// Empty batch
// ============================================================================

TEST(NegativeEdgeSamplerTest, EmptyBatch) {
    NegativeEdgeSampler sampler(false, 2, 0.5, 42);
    std::vector<int> src, tgt;
    std::vector<int64_t> ts;

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    EXPECT_EQ(result.sources.size(), 0u);
    EXPECT_EQ(result.targets.size(), 0u);
    EXPECT_EQ(sampler.get_batch_count(), 1u);
}

// ============================================================================
// Deterministic with seed
// ============================================================================

TEST(NegativeEdgeSamplerTest, DeterministicWithSeed) {
    auto run = [](unsigned int seed) {
        NegativeEdgeSampler sampler(false, 3, 0.5, seed);
        std::vector<int> src{0, 1, 2, 3};
        std::vector<int> tgt{1, 2, 3, 4};
        std::vector<int64_t> ts{100, 100, 100, 100};
        sampler.add_batch(src, tgt, ts);
        return sampler.sample_negatives();
    };

    auto r1 = run(123);
    auto r2 = run(123);

    EXPECT_EQ(r1.targets, r2.targets);
}

// ============================================================================
// Skip-over correctness: non-contiguous node IDs
// ============================================================================

TEST(NegativeEdgeSamplerTest, NonContiguousNodeIds) {
    // Node IDs: 10, 20, 30, 40, 50
    NegativeEdgeSampler sampler(false, 5, 0.0, 42);
    std::vector<int> src{10, 30};
    std::vector<int> tgt{20, 40};
    std::vector<int64_t> ts{100, 100};

    sampler.add_batch(src, tgt, ts);
    auto result = sampler.sample_negatives();

    std::set<int> valid{10, 20, 30, 40};
    // Need a 5th node to have any candidate. With only 4 nodes visible and
    // undirected edges 10-20 and 30-40:
    // src=10 neighbors: {20}, self=10 => candidates: {30, 40}
    // src=30 neighbors: {40}, self=30 => candidates: {10, 20}
    for (int t : result.targets) {
        EXPECT_TRUE(t == -1 || valid.count(t) > 0)
            << "Invalid target: " << t;
    }
}
