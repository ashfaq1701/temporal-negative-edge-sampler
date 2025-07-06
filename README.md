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

### `collect_all_negatives_by_timestamp`

Collects negative edges for temporal graph sampling, processing edges grouped by timestamp.

#### **Signature**
```python
collect_all_negatives_by_timestamp(
    sources: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    is_directed: bool,
    num_negatives_per_positive: int,
    historical_negative_percentage: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]
```

#### **Parameters**
- **`sources`** (`np.ndarray`): Array of source node IDs
- **`targets`** (`np.ndarray`): Array of target node IDs
- **`timestamps`** (`np.ndarray`): Array of timestamps for each edge
- **`is_directed`** (`bool`): Whether the graph is directed
- **`num_negatives_per_positive`** (`int`): Number of negative edges to sample per positive edge
- **`historical_negative_percentage`** (`float`, optional): Ratio of historical vs random negatives (default: 0.5)

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
