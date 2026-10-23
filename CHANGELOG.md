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
- **Vision second opinion** (`VISION_CHECK_MODEL`, optional): a vision LLM re-checks every tyre YOLO would
  auto-clear and sends it to an expert when it sees damage (or gives no usable answer). Stored per prediction with
  model, prompt version and hash; `scripts/eval_vision_check.py` measures it. Verdicts are now three-way: damaged,
  not damaged (clean auto-clear only) and uncertain (expert check).
- **Subject gate** (images that are not tyres): every image gets an unusual-image score (distance of the fixed
  encoder's embedding from the bundle's tyre reference subspace; threshold = p99 of training tyres). Unusual images
  are never auto-cleared and are shown to the vision model, which now also answers "is a tyre the subject?" (prompt
  v2). An image is rejected only when both agree: the inspection ends `rejected` with no prediction (the job
  succeeds), the chat and inspection page say "Not a tyre photo · no verdict", and an output guardrail
  (`verdict_on_non_tyre`) stops Bib from grading it. `scripts/eval_subject_gate.py` measures it.
- **Docs**: ADRs 0001–0008, runbooks, knowledge base, evidence index, licence notices and generated third-party
  inventories, contributing guide, security policy, code of conduct and citation file.

### Changed

- UI follows the michelin.in visual language (black header with dropdown navigation, navy panels, yellow calls to
  action, heavy italic headlines). The copilot is named Bib; there is no Michelin logo, artwork or proprietary font.
- `/inspect` and `/copilot` merged into one chat; `/copilot` redirects.
- Copilot answers render GitHub-flavoured markdown (headings, numbered lists, tables, rules, links); raw HTML and
  images are never rendered.
- The header is a single bar aligned to the content column; the environment/model/user utility row is gone.
- Monitoring windows are defined by the completion count that triggered them.
- Replays resume after interruption; work stops as soon as a job lease is lost.

### Fixed

- Promotion created a job kind the worker did not know.
- `FOR UPDATE` on an outer join in reviews; ledger grouping on a JSON key; a bool→float cast in LLMOps overview.
- Upload bursts exhausted the database pool and timed out; excess uploads now get `429` before their bodies are read.
- Copilot photo turns were rolled back by the acceptance path.
- Certification rewrites removed negated phrases; grounding ignored figures inside strings.
- A tyre with a fresh sidewall gash was auto-cleared as "not damaged" (the training data has no localized damage
  on new tyres); the vision second opinion now routes such tyres to an expert.
- Certification rewrites spliced "[requires inspector clearance]" into sentences ("certified recycling", negations
  far from the word); a real claim now replaces its whole sentence, and "certified inspector/recycler" is allowed.
- Camera "not damaged" counts included uncertain tyres.
- Follow-up questions lost the conversation's tyre ("the original photo is no longer in this thread"): each turn
  now gets the session's earlier inspections, Bib can show a stored inspection's heatmap in the chat
  (`show_attribution_heatmap`), and figures quoted from those inspections count as grounded.
- PII redaction turned the digits of inspection UUIDs into `[PHONE]`; IDs and hashes are now exempt, and card
  numbers are redacted whole instead of as a partial phone number.
- Grounding flagged negative tool values (e.g. min delta -0.504) quoted with their sign.
- A company logo uploaded to Bib was graded "Not damaged · Retread candidate" (p 0.024); it is now rejected as not a
  tyre.
- LLMOps sat inside a "Governance" dropdown; LLMOps and Audit are top-level links again.
- Pages kept a broken layout after leaving the chat (Next keeps the previous route alive in a hidden `<Activity>`).
- Horizontal overflow at phone widths; hydration mismatches in shell-shared queries.
