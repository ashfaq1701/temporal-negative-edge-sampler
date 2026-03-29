#!/bin/bash
set -e

sudo rm -rf build
sudo rm -rf temporal_negative_edge_sampler.egg-info/
sudo rm -rf dist
sudo rm -rf wheelhouse

docker build -t negative-edge-sampler-builder -f build_scripts/Dockerfile .
docker run --rm -v $(pwd):/project negative-edge-sampler-builder
