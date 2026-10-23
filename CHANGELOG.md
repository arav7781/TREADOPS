# Changelog

All notable changes to TreadOps. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow `pyproject.toml`. Measured results are in [docs/evidence](docs/evidence/README.md).

## [0.1.0] - 2026-10-23 (not tagged)

First complete version of the blueprint: data to model to release to monitoring, with a guarded copilot and a web UI.

### Added

- **Data**: D1 download through the Mendeley publisher API (resumable ranged chunks, SHA-256 verified); one
  sanitising ingest path for training and serving; reviewed duplicate grouping (cosine, capture sequence, guarded
  pHash) and leakage-safe grouped splits.
- **Training and evaluation**: full-image YOLO26 classification on documented Ultralytics extension points; DVC
  pipeline (prepare → split → train → evaluate → export_parity → package); Wilson confidence intervals,
  temperature scaling and a triage policy; ONNX parity checks; occlusion attribution on the logit margin.
- **Release**: gate reports with hard gates, quality targets and a comparative bound; fail-closed bundles with
  file hashes; MLflow pyfunc registration and `candidate` / `challenger` / `champion` aliases; promotion with
  warm-up and expected-generation compare-and-swap; rollback; audit log.
- **Platform**: FastAPI `/v1` API with a typed error contract; PostgreSQL schema with Alembic; durable jobs with a
  transactional outbox, compare-and-set claims and leases; Celery workers for inference, release, snapshots,
  training, replay, monitoring and shadow evaluation; SeaweedFS object storage; scrypt + JWT sessions and signed
  media links; operator CLI.
- **Monitoring**: descriptor PSI, prediction shift and embedding drift per window; Evidently reports; alerts
  after consecutive breaches with a review workflow; deterministic replay scenarios (low light, blur, prior shift,
  burst).
- **Copilot (Bib)**: LangGraph agent with input and output guardrails, a YOLO classification tool, knowledge-base
  retrieval, LLMOps traces (spans, tokens, guardrail events, grounding, feedback) and an offline golden-set eval
  logged to MLflow.
- **Web**: Next.js 16 app behind a same-origin BFF: dashboard, inspections, review queue, circularity ledger and
  casing passport, datasets, training, models with evidence, monitoring, replay, LLMOps and audit pages.
- **Bib chat**: inspection happens inside the chat. Attach, drop or paste a photo; the agent's steps and each
  preprocessing stage play in an animated pipeline view, followed by the result card with an inline inspector
  review, the grounded answer and follow-up suggestions.
- **Quality**: unit, data/ML, integration (real Postgres/Redis/object store), Vitest and Playwright suites; CI for
  lint, types, contract diff, migrations, tests, web build and container images; load, restart and
  backup/restore drills.
- **Docs**: ADRs 0001–0008, runbooks, knowledge base, evidence index, licence notices and generated third-party
  inventories, contributing guide, security policy, code of conduct and citation file.

### Changed

- UI follows the michelin.in visual language (black header with dropdown navigation, navy panels, yellow calls to
  action, heavy italic headlines). The copilot is named Bib; there is no Michelin logo, artwork or proprietary font.
- `/inspect` and `/copilot` merged into one chat; `/copilot` redirects.
- Monitoring windows are defined by the completion count that triggered them.
- Replays resume after interruption; work stops as soon as a job lease is lost.

### Fixed

- Promotion created a job kind the worker did not know.
- `FOR UPDATE` on an outer join in reviews; ledger grouping on a JSON key; a bool→float cast in LLMOps overview.
- Upload bursts exhausted the database pool and timed out; excess uploads now get `429` before their bodies are read.
- Copilot photo turns were rolled back by the acceptance path.
- Certification rewrites removed negated phrases; grounding ignored figures inside strings.
- Pages kept a broken layout after leaving the chat (Next keeps the previous route alive in a hidden `<Activity>`).
- Horizontal overflow at phone widths; hydration mismatches in shell-shared queries.
