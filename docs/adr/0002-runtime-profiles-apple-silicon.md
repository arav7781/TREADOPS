# ADR 0002 - Runtime profiles on Apple Silicon and CPU-only Linux torch

Status: accepted (2026-10-22)

## Context
The spec's optional GPU profile assumes an NVIDIA RTX 4050. This build ran on an Apple M5 (10 cores, 16 GB)
with Docker via colima (2 vCPU, 5.7 GB). Containers there cannot see the GPU, and the default Linux torch
2.14.1 wheel pulls ~3-4 GB of CUDA 13 runtime libraries.

## Decision
- `make dev`: infrastructure in Compose; API, dispatcher and Celery workers run natively (solo pool, one
  process each). Training uses MPS when `TRAIN_DEVICE=mps`; inference serves on CPU as the spec recommends.
- `make up`: everything in containers on CPU. `pyproject.toml` routes Linux torch/torchvision to the PyTorch
  CPU index; macOS keeps the PyPI build. `uv.lock` pins both variants.
- The `gpu` Compose profile keeps a training worker definition for NVIDIA hosts; it is untested here.

## Consequences
Every training record states its device. The first baseline trained on MPS (26 epochs, 28.6 min under heavy
CPU contention from other workloads); the DVC re-run after an import-order change trained on CPU because
`.env` sets `TRAIN_DEVICE=cpu`. Neither number is a performance claim.
