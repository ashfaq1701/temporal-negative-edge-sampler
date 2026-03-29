#include <gtest/gtest.h>
#include "NegativeEdgeSampler.h"
#include <unordered_set>
#include <vector>
#include <set>

class CollectAllNegativesTest : public ::testing::Test {
protected:
    // 5 nodes, edges across 3 timestamps
    std::vector<int> sources{0, 1, 2, 0, 1, 3};
    std::vector<int> targets{1, 2, 3, 2, 3, 4};
    std::vector<int64_t> timestamps{100, 100, 100, 200, 200, 300};
    std::unordered_set<int> all_nodes{0, 1, 2, 3, 4};
};

// Test output arrays have correct length
TEST_F(CollectAllNegativesTest, OutputLengthCorrect) {
    int k = 3;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 0.5);

    size_t expected = sources.size() * k;
    EXPECT_EQ(neg_src.size(), expected);
    EXPECT_EQ(neg_tgt.size(), expected);
}

// Test all negative sources are valid nodes
TEST_F(CollectAllNegativesTest, NegativeSourcesAreValidNodes) {
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 0.5);

    for (int src : neg_src) {
        EXPECT_TRUE(src == -1 || all_nodes.count(src) > 0)
            << "Invalid source node: " << src;
    }
}

// Test all negative targets are valid nodes or -1 sentinel
TEST_F(CollectAllNegativesTest, NegativeTargetsAreValidNodesOrSentinel) {
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 0.5);

    for (int tgt : neg_tgt) {
        EXPECT_TRUE(tgt == -1 || all_nodes.count(tgt) > 0)
            << "Invalid target node: " << tgt;
    }
}

// Test directed mode
TEST_F(CollectAllNegativesTest, DirectedModeOutputLength) {
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, true, k, 0.5);

    EXPECT_EQ(neg_src.size(), sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), sources.size() * k);
}

// Test full historical (1.0)
TEST_F(CollectAllNegativesTest, FullHistorical) {
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 1.0);

    EXPECT_EQ(neg_src.size(), sources.size() * k);
}

// Test full random (0.0)
TEST_F(CollectAllNegativesTest, FullRandom) {
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 0.0);

    EXPECT_EQ(neg_src.size(), sources.size() * k);
}

// Test single timestamp
TEST_F(CollectAllNegativesTest, SingleTimestamp) {
    std::vector<int> src{0, 1};
    std::vector<int> tgt{1, 2};
    std::vector<int64_t> ts{100, 100};

    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        src, tgt, ts, false, k, 0.5);

    EXPECT_EQ(neg_src.size(), static_cast<size_t>(src.size() * k));
}

// Test many negatives per positive
TEST_F(CollectAllNegativesTest, ManyNegativesPerPositive) {
    int k = 10;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 0.5);

    EXPECT_EQ(neg_src.size(), sources.size() * k);
    EXPECT_EQ(neg_tgt.size(), sources.size() * k);
}

// Test with single edge
TEST_F(CollectAllNegativesTest, SingleEdge) {
    std::vector<int> src{0};
    std::vector<int> tgt{1};
    std::vector<int64_t> ts{100};

    int k = 3;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        src, tgt, ts, false, k, 0.5);

    EXPECT_EQ(neg_src.size(), static_cast<size_t>(k));
    EXPECT_EQ(neg_tgt.size(), static_cast<size_t>(k));

    // All neg sources should be the single source
    for (int s : neg_src) {
        EXPECT_EQ(s, 0);
    }
}

// Test edges arrive in different timestamp order (should still group correctly)
TEST_F(CollectAllNegativesTest, UnsortedTimestamps) {
    std::vector<int> src{0, 1, 2, 3};
    std::vector<int> tgt{1, 2, 3, 4};
    std::vector<int64_t> ts{300, 100, 200, 100};

    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        src, tgt, ts, false, k, 0.5);

    EXPECT_EQ(neg_src.size(), static_cast<size_t>(src.size() * k));
    EXPECT_EQ(neg_tgt.size(), static_cast<size_t>(src.size() * k));
}

// Test that negatives for later timestamps can use historical edges
TEST_F(CollectAllNegativesTest, LaterTimestampsHaveHistoricalCandidates) {
    // After processing timestamp 100 edges, timestamp 200 should have
    // historical candidates from the first batch
    int k = 2;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        sources, targets, timestamps, false, k, 1.0);

    // Focus on edges at timestamp 200 (indices 3,4 in original)
    // Their negatives start at index 3*k=6
    size_t offset = 3 * k;
    bool has_non_sentinel = false;
    for (size_t i = offset; i < offset + 2 * k; ++i) {
        if (neg_tgt[i] != -1) {
            has_non_sentinel = true;
            EXPECT_TRUE(all_nodes.count(neg_tgt[i]) > 0);
        }
    }
    EXPECT_TRUE(has_non_sentinel)
        << "Expected at least some non-sentinel negatives for later timestamps";
}

// Test with larger graph
TEST_F(CollectAllNegativesTest, LargerGraph) {
    std::vector<int> src, tgt;
    std::vector<int64_t> ts;

    // Create 10 nodes with 20 edges across 5 timestamps
    for (int i = 0; i < 20; ++i) {
        src.push_back(i % 10);
        tgt.push_back((i + 1) % 10);
        ts.push_back(100 + (i / 4) * 100);
    }

    int k = 3;
    auto [neg_src, neg_tgt] = collect_all_negatives_by_timestamp(
        src, tgt, ts, false, k, 0.5);

    EXPECT_EQ(neg_src.size(), src.size() * k);
    EXPECT_EQ(neg_tgt.size(), src.size() * k);

    // All targets should be valid
    std::unordered_set<int> nodes;
    for (int n : src) nodes.insert(n);
    for (int n : tgt) nodes.insert(n);

    for (int t : neg_tgt) {
        EXPECT_TRUE(t == -1 || nodes.count(t) > 0);
    }
}
