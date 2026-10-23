# TreadOps documentation

Start with the [project README](../README.md). This index lists everything else.

## Architecture decisions

| ADR | Decision |
|---|---|
| [0001](adr/0001-object-store-seaweedfs.md) | SeaweedFS as the local S3-compatible object store |
| [0002](adr/0002-runtime-profiles-apple-silicon.md) | Runtime profiles on Apple Silicon and CPU-only Linux torch |
| [0003](adr/0003-duplicate-grouping-rules.md) | Reviewed duplicate grouping for leakage-safe splits |
| [0004](adr/0004-sanitized-ingest.md) | Store sanitised images only; one ingest path for training and serving |
| [0005](adr/0005-pinned-evidently-pandas.md) | pandas < 3 for the pinned Evidently adapter |
| [0006](adr/0006-copilot-architecture.md) | Guarded multimodal copilot on LangGraph with in-house LLMOps tracing |
| [0007](adr/0007-circularity-ledger.md) | Casing circularity as the product angle; verified-only impact ledger |
| [0008](adr/0008-gate-cohort.md) | Gate cohort and locked-test discipline |

## Operations

- [Runbooks](runbooks/README.md): symptoms and responses, restart drill, backup and restore
- [Data acquisition](runbooks/data-acquisition.md): fetching and verifying dataset D1

## Evidence

[Evidence index](evidence/README.md): saved results behind every number in the README (data audit, training and
gates, release drills, load and restart tests, drift, copilot evals). `make evidence` refreshes
[status.json](evidence/status.json).

## Copilot knowledge base

Bib retrieves and cites these documents:

| | Topic |
|---|---|
| [kb-001](kb/kb-001-triage-policy.md) | Triage policy and human clearance |
| [kb-002](kb/kb-002-visual-defect-cues.md) | Visual cues inspectors look for |
| [kb-003](kb/kb-003-retreading-circularity.md) | Retreading and casing circularity |
| [kb-004](kb/kb-004-photo-capture.md) | Photo capture guidance |
| [kb-005](kb/kb-005-model-and-data.md) | Model and data |
| [kb-006](kb/kb-006-monitoring.md) | Monitoring and drift |

## Licensing

- [LICENSE](../LICENSE): GNU AGPL-3.0 (the project is `AGPL-3.0-only`)
- [NOTICE.md](../NOTICE.md): copyright, third-party components, fonts, data attribution, trademarks
- [Licence inventories](licenses/README.md): every runtime dependency and its licence, generated from the lock files
- [DEPENDENCIES.md](../DEPENDENCIES.md): pinned ML-critical packages

## Project

[Contributing](../CONTRIBUTING.md) · [Code of conduct](../CODE_OF_CONDUCT.md) · [Security policy](../SECURITY.md) ·
[Support](../SUPPORT.md) · [Changelog](../CHANGELOG.md) · [Citation](../CITATION.cff) · [Agent rules](../AGENTS.md)
