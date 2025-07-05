#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <unordered_set>
#include "NegativeEdgeSampler.h"

int main() {
    const std::vector<std::tuple<int, int, int64_t>> edges {
            {0, 1, 100}, {1, 2, 100}, {3, 0, 100},
            {0, 2, 101}, {1, 3, 101}, {2, 4, 101},
            {4, 0, 101}, {1, 4, 102}, {2, 3, 102},
            {3, 1, 102}, {4, 2, 102}, {0, 3, 103},
            {2, 0, 103}, {3, 4, 104}, {4, 1, 104},
            {0, 4, 104}, {1, 0, 105}, {2, 1, 105},
            {3, 2, 106}, {4, 3, 107}
    };

    // Collect all unique node IDs
    std::unordered_set<int> nodes;
    for (const auto& [src, dst, _] : edges) {
        nodes.insert(src);
        nodes.insert(dst);
    }

    // Group edges by timestamp
    std::map<int64_t, std::vector<std::pair<int, int>>> edge_groups;
    for (const auto& [src, dst, ts] : edges) {
        edge_groups[ts].emplace_back(src, dst);
    }

    // Initialize sampler
    NegativeEdgeSampler sampler(nodes, false);

    // Sample negatives per batch
    const int num_negatives = 4;
    const double hist_pct = 0.5;

    for (const auto& [ts, edge_list] : edge_groups) {
        std::vector<int> sources, targets;
        for (const auto& [src, dst] : edge_list) {
            sources.push_back(src);
            targets.push_back(dst);
        }

        auto [neg_sources, neg_targets] =
            sampler.sample_negative_edges_per_batch(sources, targets, num_negatives, hist_pct);

        std::cout << "\n=== Timestamp: " << ts << " ===" << std::endl;
        for (size_t i = 0; i < neg_sources.size(); ++i) {
            std::cout << "NegEdge(" << neg_sources[i] << " -> " << neg_targets[i] << ")\n";
        }
    }

    return 0;
}
