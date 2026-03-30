# Temporal Negative Edge Sampler

[![PyPI Latest Release](https://img.shields.io/pypi/v/temporal-negative-edge-sampler.svg)](https://pypi.org/project/temporal-negative-edge-sampler/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/temporal-negative-edge-sampler.svg)](https://pypi.org/project/temporal-negative-edge-sampler/)

## The Problem:

Poursafaei et al. in NeurIPS 2022 described that random negative edges make temporal link prediction task easy. This is because random negatives are often between unrelated node pairs that have never interacted, making them trivially easy for models to distinguish from true positive edges. So they introduced the concept of historical negatives—edges that existed in the past but are inactive at the current time—making the prediction task more realistic and challenging.

Huang et al. in NeurIPS 2023 created a benchmark based on this called TGB (Temporal Graph Benchmark) where in the validation and test sets they included both historical and random negatives.

- Historical negatives: Edges that existed in the past but are inactive at the current time between a pair of nodes.
- Random negatives: Edges that never existed in the past between a pair of nodes.

TGB generated 10-20 negative edges per positive edge where 50% is historical and 50% is random.

However, they only released pre-generated pickle files with negatives for their benchmark datasets, and did not publish code to generate such negatives for new datasets or for training-time augmentation.

Negative edges are generated using the following steps:

1. **Group all edges by timestamp**, sorted in chronological order.

2. **Iterate through each timestamp group** (i.e., a batch of edges sharing the same timestamp):
    - Add the current batch to a **streaming graph structure**, which keeps track of all previously seen edges (the historical graph).

3. For **each source node** in the current batch:
    - Sample `n × p` **historical negative edges**:
        - These are destination nodes that **have been connected to the source in the past**, but are **not connected to the source in the current timestamp**.
    - Sample `n × (1 - p)` **random negative edges**:
        - These are destination nodes that **have never been connected** to the source in the historical graph (including the current batch).

While the algorithm described above is correct and benchmark-aligned, it is computationally expensive if implemented naively in Python. A naive Python implementation will not scale to large datasets and may run indefinitely for even moderate-sized graphs.

**Temporal Negative Edge** Sampler is a high-performance library for generating negative edges in temporal link prediction tasks, supporting both historical and random negative sampling. It mirrors the evaluation strategy used in the Temporal Graph Benchmark (TGB), but extends it to training and custom datasets. The core implementation is written in parallelized C++ with Python bindings, making it scalable to large graphs with millions of edges.

## API

The Python package exposes a single class: `NegativeEdgeSampler`.

### `NegativeEdgeSampler`

#### **Constructor**
```python
from temporal_negative_edge_sampler import NegativeEdgeSampler

sampler = NegativeEdgeSampler(
    is_directed: bool,
    num_negatives_per_positive: int,
    historical_negative_percentage: float = 0.5,
    seed: int = 0,
)
```

#### **Constructor parameters**
- **`is_directed`** (`bool`): Whether edges are treated as directed.
- **`num_negatives_per_positive`** (`int`): Number of negative edges to sample per positive edge in each batch.
- **`historical_negative_percentage`** (`float`, optional): Target fraction of negatives sampled from historical neighbors (default: `0.5`).
- **`seed`** (`int`, optional): RNG seed. `0` uses `std::random_device` (default: `0`).

### Methods

#### `add_batch(sources, targets, timestamps) -> None`
Adds one timestamp-consistent batch of positive edges.

- `sources`: 1D `np.ndarray` of `int32`
- `targets`: 1D `np.ndarray` of `int32`
- `timestamps`: 1D `np.ndarray` of `int64`

All three arrays must be 1-dimensional and have the same length.

#### `sample_negatives() -> dict`
Samples negatives for the most recently added batch and then merges that batch into history.

Returns a dictionary with:
- `sources` (`np.ndarray[int32]`): repeated positive sources, length = `batch_size * num_negatives_per_positive`
- `targets` (`np.ndarray[int32]`): sampled negative targets (`-1` sentinel is used when not enough valid negatives exist)
- `num_historical_actual` (`int`): actual number of historical negatives produced
- `num_random_actual` (`int`): actual number of random negatives produced

#### Getters
- `get_node_count() -> int`: number of unique nodes seen so far.
- `get_edge_count() -> int`: number of positive edges ingested so far.
- `get_batch_count() -> int`: number of processed batches.

### Minimal example
```python
import numpy as np
from temporal_negative_edge_sampler import NegativeEdgeSampler

sampler = NegativeEdgeSampler(
    is_directed=False,
    num_negatives_per_positive=2,
    historical_negative_percentage=0.5,
    seed=42,
)

# Batch 1
sampler.add_batch(
    sources=np.array([0, 1], dtype=np.int32),
    targets=np.array([1, 2], dtype=np.int32),
    timestamps=np.array([100, 100], dtype=np.int64),
)
negatives = sampler.sample_negatives()

print(negatives["sources"])
print(negatives["targets"])
print(negatives["num_historical_actual"], negatives["num_random_actual"])
```

## References:

Poursafaei et al. (2022)  
**Towards Better Evaluation for Dynamic Link Prediction**  
Farimah Poursafaei, Shenyang Huang, Kris Pelrine, Reihaneh Rabbany  
In NeurIPS 2022 Datasets and Benchmarks Track.  
https://arxiv.org/pdf/2207.10128

Huang et al. (2023)  
**Temporal Graph Benchmark for Machine Learning on Temporal Graphs**  
Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, Reihaneh Rabbany  
In NeurIPS 2023 Datasets and Benchmarks Track.  
https://arxiv.org/pdf/2307.01026
