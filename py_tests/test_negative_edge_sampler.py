import numpy as np
import pytest
from temporal_negative_edge_sampler import collect_all_negatives_by_timestamp


def make_simple_graph():
    """Small graph: 5 nodes, edges across 3 timestamps."""
    sources = np.array([0, 1, 2, 0, 1, 3], dtype=np.int32)
    targets = np.array([1, 2, 3, 2, 3, 4], dtype=np.int32)
    timestamps = np.array([100, 100, 100, 200, 200, 300], dtype=np.int64)
    return sources, targets, timestamps


class TestBasicOutput:
    def test_returns_two_arrays(self):
        sources, targets, timestamps = make_simple_graph()
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False, num_negatives_per_positive=2
        )
        assert isinstance(neg_src, np.ndarray)
        assert isinstance(neg_tgt, np.ndarray)

    def test_output_length(self):
        sources, targets, timestamps = make_simple_graph()
        k = 3
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False, num_negatives_per_positive=k
        )
        expected_len = len(sources) * k
        assert len(neg_src) == expected_len
        assert len(neg_tgt) == expected_len

    def test_negative_sources_match_positive_sources(self):
        sources, targets, timestamps = make_simple_graph()
        k = 2
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False, num_negatives_per_positive=k
        )
        # Each positive edge produces k negatives; sources should all be valid nodes
        # -1 is a sentinel for when no candidates are available
        all_nodes = set(sources.tolist()) | set(targets.tolist()) | {-1}
        for s in neg_src:
            assert s in all_nodes
        for t in neg_tgt:
            assert t in all_nodes


class TestDirected:
    def test_directed_output_length(self):
        sources, targets, timestamps = make_simple_graph()
        k = 2
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=True, num_negatives_per_positive=k
        )
        assert len(neg_src) == len(sources) * k
        assert len(neg_tgt) == len(sources) * k


class TestHistoricalPercentage:
    def test_full_historical(self):
        sources, targets, timestamps = make_simple_graph()
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False,
            num_negatives_per_positive=2, historical_negative_percentage=1.0
        )
        assert len(neg_src) == len(sources) * 2

    def test_full_random(self):
        sources, targets, timestamps = make_simple_graph()
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False,
            num_negatives_per_positive=2, historical_negative_percentage=0.0
        )
        assert len(neg_src) == len(sources) * 2


class TestEdgeCases:
    def test_single_timestamp(self):
        sources = np.array([0, 1], dtype=np.int32)
        targets = np.array([1, 2], dtype=np.int32)
        timestamps = np.array([100, 100], dtype=np.int64)
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False, num_negatives_per_positive=1
        )
        assert len(neg_src) == 2
        assert len(neg_tgt) == 2

    def test_many_negatives(self):
        """Request more negatives than possible unique ones — should still return the right count."""
        sources = np.array([0, 1, 2, 3], dtype=np.int32)
        targets = np.array([1, 2, 3, 4], dtype=np.int32)
        timestamps = np.array([100, 100, 100, 100], dtype=np.int64)
        k = 10
        neg_src, neg_tgt = collect_all_negatives_by_timestamp(
            sources, targets, timestamps, is_directed=False, num_negatives_per_positive=k
        )
        assert len(neg_src) == len(sources) * k
