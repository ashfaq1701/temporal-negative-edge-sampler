#ifndef NEGATIVEEDGESAMPLER_H
#define NEGATIVEEDGESAMPLER_H

#include <vector>
#include <cstdint>
#include <random>

// Result returned by sample_negatives()
struct NegativeSampleResult {
    std::vector<int> sources;          // length = batch_size * num_negatives_per_positive
    std::vector<int> targets;          // length = batch_size * num_negatives_per_positive
    int num_random_actual;             // actual count of random negatives produced (excluding -1 sentinels)
    int num_historical_actual;         // actual count of historical negatives produced
};

class NegativeEdgeSampler {
    // --- Configuration (immutable after construction) ---
    bool is_directed_;
    int num_negatives_per_positive_;
    double historical_negative_percentage_;

    // --- Accumulated state (grows with each add_batch) ---
    // Sorted, deduplicated list of all node IDs seen so far.
    std::vector<int> all_nodes_sorted_;

    // Per-node neighbor lists, indexed by position in all_nodes_sorted_.
    // Each inner vector is sorted and deduplicated, containing node IDs (not indices).
    // Stores cumulative adjacency from all previous batches (not including current batch).
    std::vector<std::vector<int>> history_neighbors_;

    // --- Current batch state (replaced each add_batch call) ---
    std::vector<int> batch_sources_;
    std::vector<int> batch_targets_;

    // Per-node neighbor lists for the current batch only.
    // Indexed by position in all_nodes_sorted_ (after merge).
    // Each inner vector is sorted and deduplicated.
    std::vector<std::vector<int>> batch_neighbors_;

    // --- Counters ---
    size_t total_edges_;
    size_t batch_count_;

    // --- RNG ---
    std::mt19937 rng_;

    // --- Internal helpers ---

    // Binary search in all_nodes_sorted_. Returns index if found, -1 otherwise.
    int node_index(int node_id) const;

    // Merge new sorted+deduped nodes into all_nodes_sorted_.
    // Extends history_neighbors_ and batch_neighbors_ to match new size,
    // copying existing history entries to their new positions.
    void merge_new_nodes(const std::vector<int>& new_nodes_sorted);

    // Populate batch_neighbors_ from batch_sources_ and batch_targets_.
    void build_batch_neighbors();

    // Merge current batch edges into history_neighbors_ after sampling.
    void update_history();

    // --- Sampling helpers ---

    // Sample count random non-neighbor nodes for node at src_idx using skip-over.
    // Excludes: history neighbors, batch neighbors, and self.
    // Returns vector of node IDs (may contain -1 sentinels if not enough candidates).
    std::vector<int> sample_random_negatives(int src_idx, int count, std::mt19937& rng) const;

    // Sample count historical negatives for node at src_idx.
    // Historical = in history_neighbors_ but NOT in batch_neighbors_.
    // Returns vector of node IDs (may be shorter than count if not enough candidates).
    std::vector<int> sample_historical_negatives(int src_idx, int count, std::mt19937& rng) const;

    // Map a virtual index (into the complement set) to an actual node,
    // skipping over sorted excluded positions in all_nodes_sorted_.
    int skip_over(int raw_index, const std::vector<int>& exclude_positions_sorted) const;

public:
    explicit NegativeEdgeSampler(
        bool is_directed,
        int num_negatives_per_positive,
        double historical_negative_percentage = 0.5,
        unsigned int seed = 0  // 0 = use random_device
    );

    // Ingest a batch of positive edges. Must be called before sample_negatives().
    // Batches must arrive in temporal order (caller's responsibility).
    void add_batch(
        const std::vector<int>& sources,
        const std::vector<int>& targets,
        const std::vector<int64_t>& timestamps
    );

    // Sample negative edges for the most recently added batch.
    // Must be called after add_batch(). After sampling, the batch edges
    // are merged into the accumulated history.
    NegativeSampleResult sample_negatives();

    // --- Getters ---
    size_t get_node_count() const;
    size_t get_edge_count() const;
    size_t get_batch_count() const;
};

#endif // NEGATIVEEDGESAMPLER_H
