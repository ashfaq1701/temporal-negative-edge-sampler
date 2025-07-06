#include "NegativeEdgeSampler.h"

#include <iostream>
#include <map>
#include <ranges>
#include <tbb/parallel_for_each.h>

NegativeEdgeSampler::NegativeEdgeSampler(const std::unordered_set<int>& all_node_ids, const bool is_directed)
    : is_directed(is_directed)
{
    for (int n : all_node_ids) {
        all_nodes.insert(n);
    }
}

std::pair<std::vector<int>, std::vector<int>> NegativeEdgeSampler::sample_negative_edges_per_batch(
    const std::vector<int>& batch_sources,
    const std::vector<int>& batch_targets,
    const int64_t batch_timestamp,
    const int num_negatives_per_positive,
    const double historical_negative_percentage) {

    if (is_debug_enabled()) {
        std::cout << "Processing batch for timestamp " << batch_timestamp << std::endl;
    }

    const int hist_k = static_cast<int>(num_negatives_per_positive * historical_negative_percentage);
    const int rand_k = num_negatives_per_positive - hist_k;

    const size_t batch_size = batch_sources.size();

    // === Parallel insert current batch edges ===
    tbb::concurrent_unordered_set<std::pair<int, int>, PairHash> current_batch;
    tbb::parallel_for(static_cast<size_t>(0), batch_size, [&](const size_t i) {
        current_batch.insert({batch_sources[i], batch_targets[i]});
        if (!is_directed) {
            current_batch.insert({batch_targets[i], batch_sources[i]});
        }
    });

    tbb::concurrent_unordered_map<int, tbb::concurrent_unordered_set<int>> current_adj;
    tbb::parallel_for(static_cast<size_t>(0), batch_size, [&](const size_t i) {
        const int src = batch_sources[i];
        const int tgt = batch_targets[i];

        current_adj[src].insert(tgt);
        if (!is_directed) {
            current_adj[tgt].insert(src);
        }
    });

    // === Preallocate output arrays ===
    std::vector<int> neg_sources(batch_size * num_negatives_per_positive);
    std::vector<int> neg_targets(batch_size * num_negatives_per_positive);

    // === Parallel processing ===
    tbb::parallel_for(static_cast<size_t>(0), batch_size, [&](const size_t i) {
        thread_local std::mt19937 rng(std::random_device{}());
        const int src = batch_sources[i];
        std::vector<int> negs;

        // === Historical negatives ===
        if (hist_k > 0) {
            std::vector<int> hist_candidates;

            if (const auto hist_it = adj.find(src); hist_it != adj.end()) {
                const auto& current_targets = current_adj[src];

                for (int target : hist_it->second) {
                    if (!current_targets.contains(target)) {
                        hist_candidates.push_back(target);
                    }
                }
            }

            if (hist_candidates.empty()) {
                hist_candidates = get_random_candidates(src);
            }

            std::uniform_int_distribution<> dist(0, static_cast<int>(hist_candidates.size()) - 1);
            for (int j = 0; j < hist_k; ++j) {
                const int idx = dist(rng);
                negs.push_back(hist_candidates[idx]);
            }
        }

        // === Random negatives ===
        if (rand_k > 0) {
            const std::vector<int> rand_candidates = get_random_candidates(src);
            std::uniform_int_distribution<> dist(0, static_cast<int>(rand_candidates.size()) - 1);
            for (int j = 0; j < rand_k; ++j) {
                const int idx = dist(rng);
                negs.push_back(rand_candidates[idx]);
            }
        }

        // === Fill preallocated output ===
        const size_t base = i * num_negatives_per_positive;
        for (int j = 0; j < num_negatives_per_positive; ++j) {
            neg_sources[base + j] = src;
            neg_targets[base + j] = negs[j];
        }
    });

    update_state(current_adj);
    return {neg_sources, neg_targets};
}

std::vector<int> NegativeEdgeSampler::get_random_candidates(const int src) {
    // Use a dummy reference if src not in adj
    static const tbb::concurrent_unordered_set<int> dummy_set;
    const auto& neighbors = adj.contains(src) ? adj[src] : dummy_set;

    tbb::concurrent_vector<int> candidates;

    const std::vector<int> added_nodes_vec(added_nodes.begin(), added_nodes.end());
    const std::vector<int> all_nodes_vec(all_nodes.begin(), all_nodes.end());

    tbb::parallel_for(static_cast<size_t>(0), added_nodes_vec.size(), [&](const size_t i) {
        if (const int node = added_nodes_vec[i]; node != src && !neighbors.contains(node)) {
            candidates.push_back(node);
        }
    });

    // Fallback to all_nodes if no candidates
    if (candidates.empty()) {
        if (added_nodes.empty()) {
            std::cout << "First-ever batch — fallback used.\n";
        } else if (candidates.empty()) {
            std::cout << "src=" << src << " connected to all known nodes — fallback used.\n";
        }

        tbb::parallel_for(static_cast<size_t>(0), all_nodes_vec.size(), [&](const size_t i) {
            if (const int node = all_nodes_vec[i]; node != src) {
                candidates.push_back(node);
            }
        });
    }

    return {candidates.begin(), candidates.end()};
}

void NegativeEdgeSampler::update_state(const tbb::concurrent_unordered_map<int, tbb::concurrent_unordered_set<int>>& current_adj) {
    tbb::parallel_for_each(current_adj.begin(), current_adj.end(), [&](const auto& adj_entry) {
        const auto& [src, targets] = adj_entry;

        // Add the source node
        added_nodes.insert(src);

        // Process all targets for this source
        for (int dst : targets) {
            // Add target node
            added_nodes.insert(dst);

            // Update adjacency list
            adj[src].insert(dst);
        }
    });
}

std::pair<std::vector<int>, std::vector<int>> collect_all_negatives_by_timestamp(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<int64_t>& timestamps,
    const bool is_directed,
    const int num_negatives_per_positive,
    const double historical_negative_percentage
) {
    // 1. Extract all unique nodes
    std::unordered_set<int> all_nodes;
    for (int node : sources) all_nodes.insert(node);
    for (int node : targets) all_nodes.insert(node);

    // 2. Group edges by timestamp
    std::map<int64_t, std::vector<std::pair<int, int>>> ts_batches;
    for (size_t i = 0; i < sources.size(); ++i) {
        ts_batches[timestamps[i]].emplace_back(sources[i], targets[i]);
    }

    // 3. Initialize the sampler
    NegativeEdgeSampler sampler(all_nodes, is_directed);

    std::vector<int> neg_sources;
    std::vector<int> neg_targets;

    // 4. Process each timestamp batch
    for (const auto& [batch_timestamp, batch] : ts_batches) {
        std::vector<int> batch_srcs, batch_dsts;
        for (const auto&[src, tgt] : batch) {
            batch_srcs.push_back(src);
            batch_dsts.push_back(tgt);
        }

        auto [batch_negs_srcs, batch_negs_dsts] =
            sampler.sample_negative_edges_per_batch(
                batch_srcs,
                batch_dsts,
                batch_timestamp,
                num_negatives_per_positive,
                historical_negative_percentage);

        neg_sources.insert(neg_sources.end(), batch_negs_srcs.begin(), batch_negs_srcs.end());
        neg_targets.insert(neg_targets.end(), batch_negs_dsts.begin(), batch_negs_dsts.end());
    }

    return {neg_sources, neg_targets};
}
