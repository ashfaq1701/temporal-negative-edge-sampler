# Temporal Negative Edge Sampler

[![PyPI Latest Release](https://img.shields.io/pypi/v/temporal-negative-edge-sampler.svg)](https://pypi.org/project/temporal-negative-edge-sampler/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/temporal-negative-edge-sampler.svg)](https://pypi.org/project/temporal-negative-edge-sampler/)

Poursafaei et al. in NeurIPS 2022 described that random negative edges make temporal link prediction task easy. This is because random negatives are often between unrelated node pairs that have never interacted, making them trivially easy for models to distinguish from true positive edges. So they introduced the concept of historical negatives—edges that existed in the past but are inactive at the current time—making the prediction task more realistic and challenging.

Huang et al. in NeurIPS 2023 created a benchmark based on this called TGB (Temporal Graph Benchmark) where in the validation and test sets they included both historical and random negatives.

- Historical negatives: Edges that existed in the past but are inactive at the current time between a pair of nodes.
- Random negatives: Edges that never existed in the past between a pair of nodes.

TGB generated 10-20 negative edges per positive edge where 50% is historical and 50% is random.

However, they only released pre-generated pickle files with negatives for their benchmark datasets, and did not publish code to generate such negatives for new datasets or for training-time augmentation.

Negative edges can be generated using the following generalized method,

- Group edges by timestamp, sorted by timestamp values.
- Take each timestamp group and add to a streaming graph.
- For every edge's source sample n * p negative edges where the destination appears in the historical (full) graph, but not in the current batch.
- For every edge's source sample n * (1.0 - p) negative edges which never happened in the historical graph - including the current batch.
