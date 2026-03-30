import numpy as np
import pytest
from temporal_negative_edge_sampler import NegativeEdgeSampler


class TestConstruction:
    def test_basic_construction(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2)
        assert sampler.get_node_count() == 0
        assert sampler.get_edge_count() == 0
        assert sampler.get_batch_count() == 0

    def test_construction_with_all_params(self):
        sampler = NegativeEdgeSampler(
            is_directed=True,
            num_negatives_per_positive=5,
            historical_negative_percentage=0.7,
            seed=42,
        )
        assert sampler.get_node_count() == 0


class TestOutputDimensions:
    def test_output_size(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=3, seed=42)
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["sources"]) == 9  # 3 edges * 3 negatives
        assert len(result["targets"]) == 9

    def test_sources_match_positive_sources(self):
        k = 2
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=k, seed=42)
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        neg_src = result["sources"]
        for i in range(len(sources)):
            for j in range(k):
                assert neg_src[i * k + j] == sources[i]

    def test_returns_numpy_arrays(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2, seed=42)
        sources = np.array([0, 1], dtype=np.int32)
        targets = np.array([1, 2], dtype=np.int32)
        timestamps = np.array([100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert isinstance(result["sources"], np.ndarray)
        assert isinstance(result["targets"], np.ndarray)


class TestFirstBatch:
    def test_all_random_no_historical(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=2,
            historical_negative_percentage=0.5, seed=42,
        )
        sources = np.array([0, 1], dtype=np.int32)
        targets = np.array([1, 2], dtype=np.int32)
        timestamps = np.array([100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert result["num_historical_actual"] == 0


class TestHistoricalNegatives:
    def test_second_batch_has_historical(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=2,
            historical_negative_percentage=1.0, seed=42,
        )

        # Batch 1
        s1 = np.array([0, 1], dtype=np.int32)
        t1 = np.array([1, 2], dtype=np.int32)
        ts1 = np.array([100, 100], dtype=np.int64)
        sampler.add_batch(s1, t1, ts1)
        sampler.sample_negatives()

        # Batch 2
        s2 = np.array([0, 2], dtype=np.int32)
        t2 = np.array([3, 4], dtype=np.int32)
        ts2 = np.array([200, 200], dtype=np.int64)
        sampler.add_batch(s2, t2, ts2)
        result = sampler.sample_negatives()

        assert result["num_historical_actual"] > 0

    def test_historical_from_history_not_current_batch(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=3,
            historical_negative_percentage=1.0, seed=42,
        )

        # Batch 1: 0-1, 0-2
        s1 = np.array([0, 0], dtype=np.int32)
        t1 = np.array([1, 2], dtype=np.int32)
        ts1 = np.array([100, 100], dtype=np.int64)
        sampler.add_batch(s1, t1, ts1)
        sampler.sample_negatives()

        # Batch 2: 0-3. Historical for src=0 should be {1, 2}.
        s2 = np.array([0], dtype=np.int32)
        t2 = np.array([3], dtype=np.int32)
        ts2 = np.array([200], dtype=np.int64)
        sampler.add_batch(s2, t2, ts2)
        result = sampler.sample_negatives()

        hist_candidates = {1, 2}
        for t in result["targets"]:
            if t != -1:
                assert t in hist_candidates

    def test_full_historical_percentage(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=2,
            historical_negative_percentage=1.0, seed=42,
        )
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()
        assert len(result["sources"]) == 6

    def test_full_random_percentage(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=2,
            historical_negative_percentage=0.0, seed=42,
        )
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()
        assert len(result["sources"]) == 6
        assert result["num_historical_actual"] == 0


class TestDirected:
    def test_directed_output_size(self):
        sampler = NegativeEdgeSampler(is_directed=True, num_negatives_per_positive=2, seed=42)
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["sources"]) == 6
        assert len(result["targets"]) == 6


class TestEdgeCases:
    def test_complete_graph_sentinels(self):
        """All nodes connected to all others => all targets should be -1."""
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2, seed=42)
        sources = np.array([0, 0, 1], dtype=np.int32)
        targets = np.array([1, 2, 2], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        for t in result["targets"]:
            assert t == -1

    def test_two_node_graph(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=1, seed=42)
        sources = np.array([0], dtype=np.int32)
        targets = np.array([1], dtype=np.int32)
        timestamps = np.array([100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["targets"]) == 1
        assert result["targets"][0] == -1

    def test_single_edge(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=3, seed=42)
        sources = np.array([0], dtype=np.int32)
        targets = np.array([1], dtype=np.int32)
        timestamps = np.array([100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["sources"]) == 3
        for s in result["sources"]:
            assert s == 0

    def test_empty_batch(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2, seed=42)
        sources = np.array([], dtype=np.int32)
        targets = np.array([], dtype=np.int32)
        timestamps = np.array([], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["sources"]) == 0
        assert len(result["targets"]) == 0

    def test_many_negatives_per_positive(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=10, seed=42)
        sources = np.array([0, 1, 2, 3], dtype=np.int32)
        targets = np.array([1, 2, 3, 4], dtype=np.int32)
        timestamps = np.array([100, 100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        assert len(result["sources"]) == 40
        assert len(result["targets"]) == 40


class TestMultipleBatches:
    def test_accumulate_state(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2, seed=42)

        s1 = np.array([0], dtype=np.int32)
        t1 = np.array([1], dtype=np.int32)
        ts1 = np.array([100], dtype=np.int64)
        sampler.add_batch(s1, t1, ts1)
        sampler.sample_negatives()
        assert sampler.get_node_count() == 2
        assert sampler.get_edge_count() == 1
        assert sampler.get_batch_count() == 1

        s2 = np.array([1], dtype=np.int32)
        t2 = np.array([2], dtype=np.int32)
        ts2 = np.array([200], dtype=np.int64)
        sampler.add_batch(s2, t2, ts2)
        sampler.sample_negatives()
        assert sampler.get_node_count() == 3
        assert sampler.get_edge_count() == 2
        assert sampler.get_batch_count() == 2

    def test_larger_graph_multiple_batches(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=3, seed=42)

        for batch in range(5):
            src = np.array([(batch * 4 + i) % 10 for i in range(4)], dtype=np.int32)
            tgt = np.array([(batch * 4 + i + 1) % 10 for i in range(4)], dtype=np.int32)
            ts = np.full(4, 100 + batch * 100, dtype=np.int64)

            sampler.add_batch(src, tgt, ts)
            result = sampler.sample_negatives()
            assert len(result["sources"]) == 12

        assert sampler.get_batch_count() == 5
        assert sampler.get_edge_count() == 20
        assert sampler.get_node_count() == 10


class TestDeterminism:
    def test_same_seed_same_results(self):
        def run(seed):
            sampler = NegativeEdgeSampler(
                is_directed=False, num_negatives_per_positive=3, seed=seed,
            )
            sources = np.array([0, 1, 2, 3], dtype=np.int32)
            targets = np.array([1, 2, 3, 4], dtype=np.int32)
            timestamps = np.array([100, 100, 100, 100], dtype=np.int64)
            sampler.add_batch(sources, targets, timestamps)
            return sampler.sample_negatives()

        r1 = run(123)
        r2 = run(123)
        np.testing.assert_array_equal(r1["targets"], r2["targets"])


class TestValidTargets:
    def test_targets_are_valid_nodes_or_sentinel(self):
        sampler = NegativeEdgeSampler(is_directed=False, num_negatives_per_positive=2, seed=42)
        sources = np.array([0, 1, 2], dtype=np.int32)
        targets = np.array([1, 2, 3], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        valid_nodes = {0, 1, 2, 3, -1}
        for t in result["targets"]:
            assert int(t) in valid_nodes, f"Invalid target: {t}"

    def test_random_negatives_exclude_neighbors_and_self(self):
        sampler = NegativeEdgeSampler(
            is_directed=False, num_negatives_per_positive=10,
            historical_negative_percentage=0.0, seed=42,
        )
        sources = np.array([0, 2, 3], dtype=np.int32)
        targets = np.array([1, 4, 4], dtype=np.int32)
        timestamps = np.array([100, 100, 100], dtype=np.int64)

        sampler.add_batch(sources, targets, timestamps)
        result = sampler.sample_negatives()

        # Negatives for src=0 (first 10 entries): should not be 0 (self) or 1 (neighbor)
        for j in range(10):
            t = int(result["targets"][j])
            if t != -1:
                assert t != 0, "Should not be self"
                assert t != 1, "Should not be current batch neighbor"
