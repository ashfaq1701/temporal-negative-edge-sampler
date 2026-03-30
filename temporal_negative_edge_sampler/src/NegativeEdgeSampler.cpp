#include "NegativeEdgeSampler.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <omp.h>

static constexpr size_t OMP_THRESHOLD = 64;

// ============================================================================
// Constructor
// ============================================================================

NegativeEdgeSampler::NegativeEdgeSampler(
    const bool is_directed,
    const int num_negatives_per_positive,
    const double historical_negative_percentage,
    const unsigned int seed)
    : is_directed_(is_directed),
      num_negatives_per_positive_(num_negatives_per_positive),
      historical_negative_percentage_(historical_negative_percentage),
      total_edges_(0),
      batch_count_(0)
{
    if (seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(seed);
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

int NegativeEdgeSampler::node_index(const int node_id) const {
    const auto it = node_to_index_.find(node_id);
    assert(it != node_to_index_.end() && "node_index: node_id not found");
    return it->second;
}

void NegativeEdgeSampler::merge_new_nodes(const std::vector<int>& new_nodes_sorted) {
    if (new_nodes_sorted.empty()) return;

    std::vector<int> merged;
    merged.reserve(all_nodes_sorted_.size() + new_nodes_sorted.size());
    std::set_union(
        all_nodes_sorted_.begin(), all_nodes_sorted_.end(),
        new_nodes_sorted.begin(), new_nodes_sorted.end(),
        std::back_inserter(merged));

    if (merged.size() == all_nodes_sorted_.size()) {
        return;
    }

    const size_t new_size = merged.size();
    std::vector<std::vector<int>> new_history(new_size);
    std::vector<std::vector<int>> new_batch(new_size);

    size_t old_idx = 0;
    for (size_t new_idx = 0; new_idx < new_size; ++new_idx) {
        if (old_idx < all_nodes_sorted_.size() && all_nodes_sorted_[old_idx] == merged[new_idx]) {
            new_history[new_idx] = std::move(history_neighbors_[old_idx]);
            ++old_idx;
        }
    }

    all_nodes_sorted_ = std::move(merged);
    history_neighbors_ = std::move(new_history);
    batch_neighbors_ = std::move(new_batch);

    node_to_index_.clear();
    node_to_index_.reserve(all_nodes_sorted_.size());
    for (size_t i = 0; i < all_nodes_sorted_.size(); ++i) {
        node_to_index_[all_nodes_sorted_[i]] = static_cast<int>(i);
    }
}

void NegativeEdgeSampler::build_batch_neighbors() {
    for (auto& bn : batch_neighbors_) {
        bn.clear();
    }

    const size_t batch_size = batch_sources_.size();
    for (size_t i = 0; i < batch_size; ++i) {
        const int src_idx = node_index(batch_sources_[i]);
        const int tgt_idx = node_index(batch_targets_[i]);

        batch_neighbors_[src_idx].push_back(tgt_idx);
        if (!is_directed_) {
            batch_neighbors_[tgt_idx].push_back(src_idx);
        }
    }

    const size_t n = batch_neighbors_.size();

    auto process = [&](size_t i) {
        if (!batch_neighbors_[i].empty()) {
            std::sort(batch_neighbors_[i].begin(), batch_neighbors_[i].end());
            batch_neighbors_[i].erase(
                std::unique(batch_neighbors_[i].begin(), batch_neighbors_[i].end()),
                batch_neighbors_[i].end());
        }
    };

    if (n >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) process(i);
    } else {
        for (size_t i = 0; i < n; ++i) process(i);
    }
}

void NegativeEdgeSampler::update_history() {
    const size_t n = all_nodes_sorted_.size();

    auto process = [&](size_t i) {
        if (batch_neighbors_[i].empty()) return;

        if (history_neighbors_[i].empty()) {
            history_neighbors_[i] = std::move(batch_neighbors_[i]);
        } else {
            std::vector<int> merged;
            merged.reserve(history_neighbors_[i].size() + batch_neighbors_[i].size());
            std::set_union(
                history_neighbors_[i].begin(), history_neighbors_[i].end(),
                batch_neighbors_[i].begin(), batch_neighbors_[i].end(),
                std::back_inserter(merged));
            history_neighbors_[i] = std::move(merged);
        }
    };

    if (n >= OMP_THRESHOLD) {
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; ++i) process(i);
    } else {
        for (size_t i = 0; i < n; ++i) process(i);
    }
}

// ============================================================================
// Skip-over algorithm
// ============================================================================

/*
 * Maps a "raw" index in the compressed valid range [0, valid_count)
 * to the corresponding index in the full node array [0, N),
 * skipping over excluded positions.
 *
 * Example:
 *   N = 10, excluded = {2, 5}
 *   valid indices map like:
 *     raw: 0 1 2 3 4 5 6 7
 *     real:0 1 3 4 6 7 8 9
 */
int NegativeEdgeSampler::skip_over(
    const int raw_index,
    const std::vector<int>& exclude_positions_sorted) {

    int adjusted = raw_index;
    size_t lo = 0;
    const size_t d = exclude_positions_sorted.size();

    while (lo < d) {
        auto it = std::upper_bound(
            exclude_positions_sorted.begin() + static_cast<long>(lo),
            exclude_positions_sorted.end(),
            adjusted);

        const size_t hi = static_cast<size_t>(it - exclude_positions_sorted.begin());
        const size_t skips = hi - lo;

        if (skips == 0) break;

        adjusted += static_cast<int>(skips);
        lo = hi;
    }

    return adjusted;
}

// ============================================================================
// Sampling methods
// ============================================================================

std::vector<int> NegativeEdgeSampler::sample_random_negatives(
    const int src_idx, const int count, std::mt19937& rng) const {

    if (count <= 0) return {};

    const int N = static_cast<int>(all_nodes_sorted_.size());

    const auto& hist = history_neighbors_[src_idx];
    const auto& batch = batch_neighbors_[src_idx];

    std::vector<int> combined_exclude;
    combined_exclude.reserve(hist.size() + batch.size() + 1);
    std::set_union(hist.begin(), hist.end(),
                   batch.begin(), batch.end(),
                   std::back_inserter(combined_exclude));

    auto self_pos = std::lower_bound(combined_exclude.begin(), combined_exclude.end(), src_idx);
    if (self_pos == combined_exclude.end() || *self_pos != src_idx) {
        combined_exclude.insert(self_pos, src_idx);
    }

    const int d = static_cast<int>(combined_exclude.size());
    const int valid_count = N - d;

    std::vector<int> result(count, -1);
    if (valid_count <= 0) return result;

    std::uniform_int_distribution<int> dist(0, valid_count - 1);

    for (int j = 0; j < count; ++j) {
        const int raw = dist(rng);
        const int sampled_idx = skip_over(raw, combined_exclude);
        result[j] = all_nodes_sorted_[sampled_idx];
    }

    return result;
}

std::vector<int> NegativeEdgeSampler::sample_historical_negatives(
    const int src_idx, const int count, std::mt19937& rng) const {

    if (count <= 0) return {};

    const auto& hist = history_neighbors_[src_idx];
    const auto& batch = batch_neighbors_[src_idx];

    std::vector<int> candidates;
    candidates.reserve(hist.size());
    std::set_difference(hist.begin(), hist.end(),
                        batch.begin(), batch.end(),
                        std::back_inserter(candidates));

    if (candidates.empty()) return {};

    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);

    std::vector<int> result(count);
    for (int j = 0; j < count; ++j) {
        result[j] = all_nodes_sorted_[candidates[dist(rng)]];
    }

    return result;
}

// ============================================================================
// Public API
// ============================================================================

void NegativeEdgeSampler::add_batch(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<int64_t>& /*timestamps*/) {

    batch_sources_ = sources;
    batch_targets_ = targets;

    std::vector<int> new_nodes;
    new_nodes.reserve(sources.size() + targets.size());
    new_nodes.insert(new_nodes.end(), sources.begin(), sources.end());
    new_nodes.insert(new_nodes.end(), targets.begin(), targets.end());
    std::sort(new_nodes.begin(), new_nodes.end());
    new_nodes.erase(std::unique(new_nodes.begin(), new_nodes.end()), new_nodes.end());

    merge_new_nodes(new_nodes);
    build_batch_neighbors();
}

NegativeSampleResult NegativeEdgeSampler::sample_negatives() {
    const size_t batch_size = batch_sources_.size();
    const size_t total_negatives = batch_size * num_negatives_per_positive_;

    NegativeSampleResult result;
    result.sources.resize(total_negatives);
    result.targets.resize(total_negatives, -1);
    result.num_historical_actual = 0;
    result.num_random_actual = 0;

    if (batch_size == 0) {
        update_history();
        ++batch_count_;
        return result;
    }

    int hist_k = static_cast<int>(num_negatives_per_positive_ * historical_negative_percentage_);
    int rand_k = num_negatives_per_positive_ - hist_k;

    if (batch_count_ == 0) {
        hist_k = 0;
        rand_k = num_negatives_per_positive_;
    }

    int total_hist = 0;
    int total_rand = 0;

    auto process = [&](size_t i, std::mt19937& local_rng) {
        const int src = batch_sources_[i];
        const int src_idx = node_index(src);
        const size_t base = i * num_negatives_per_positive_;

        for (int j = 0; j < num_negatives_per_positive_; ++j) {
            result.sources[base + j] = src;
        }

        int offset = 0;

        auto hist_samples = sample_historical_negatives(src_idx, hist_k, local_rng);
        const int hist_got = static_cast<int>(hist_samples.size());

        for (int j = 0; j < hist_got; ++j) {
            result.targets[base + offset++] = hist_samples[j];
        }

        const int extra_rand = hist_k - hist_got;
        const int actual_rand_k = rand_k + extra_rand;

        auto rand_samples = sample_random_negatives(src_idx, actual_rand_k, local_rng);
        for (int j = 0; j < static_cast<int>(rand_samples.size()); ++j) {
            if (rand_samples[j] != -1) {
                result.targets[base + offset] = rand_samples[j];
                ++total_rand;
            }
            ++offset;
        }

        total_hist += hist_got;
    };

    if (batch_size >= OMP_THRESHOLD) {
        const int max_threads = omp_get_max_threads();
        std::vector<std::mt19937> thread_rngs(max_threads);
        for (int t = 0; t < max_threads; ++t) {
            thread_rngs[t].seed(rng_());
        }

        #pragma omp parallel reduction(+:total_hist, total_rand)
        {
            auto& local_rng = thread_rngs[omp_get_thread_num()];

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < batch_size; ++i) {
                process(i, local_rng);
            }
        }
    } else {
        for (size_t i = 0; i < batch_size; ++i) {
            process(i, rng_);
        }
    }

    result.num_historical_actual = total_hist;
    result.num_random_actual = total_rand;

    update_history();
    total_edges_ += batch_sources_.size();
    ++batch_count_;

    return result;
}

// ============================================================================
// Getters
// ============================================================================

size_t NegativeEdgeSampler::get_node_count() const {
    return all_nodes_sorted_.size();
}

size_t NegativeEdgeSampler::get_edge_count() const {
    return total_edges_;
}

size_t NegativeEdgeSampler::get_batch_count() const {
    return batch_count_;
}
