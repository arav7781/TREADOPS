# Changelog

All notable changes to TreadOps. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow `pyproject.toml`. Measured results are in [docs/evidence](docs/evidence/README.md).

## [Unreleased] - 2026-10-24

Closing the gaps against plant-grade casing inspection: more data, robustness, a model benchmark and damage
localisation.

### Added

- **Two more public data sources**: TyreNet (Mendeley, 1,693 photos) and the MSU tire sidewall set (Harvard
  Dataverse; 883 of its photos are copies of D1 and are collapsed onto D1, 144 remain). Each is fetched from its
  publisher and checksum-verified (`make fetch-data`), grouped against D1 without moving any D1 photo, and carved
  into roles on its own. Gate and test reports are broken down per source.
- **Robustness augmentation**: low light (gamma, 35-75 % brightness), Gaussian blur and JPEG recompression,
  off by default, enabled in the `*_robust` recipes.
- **Model benchmark** (`scripts/benchmark_models.py`): YOLO26 n/s/m robust, a YOLO26s control and ConvNeXt-Tiny
  (torchvision, pinned ImageNet weights, served through ONNX Runtime), trained on the same images and scored on the
  gate cohort per source and under low light and blur, with a selection rule fixed before any result.
- **Damage locator**: a YOLO26n detector for cracks, cuts, chunks, bulges and foreign objects, trained on two
  box-labelled Roboflow Universe datasets after removing copies of TreadOps evaluation photos. Its boxes are
  drawn on the classified image and in the result; a confident box stops an auto-clear (threshold chosen on the
  policy role's good tyres). DVC stages `damage_prepare`, `damage_train`, `damage_evaluate`; bundled only when its
  mAP gate passes; `scripts/eval_damage_locator.py` measures its effect end to end.
- Migration `0005`: `predictions.damage` (JSONB).
- **MLflow Tracing for LLMOps**: every Bib turn is an MLflow trace (experiment `treadops-copilot`) mirroring the
  LLMOps spans (agent, guardrail, planning, autologged OpenAI completions with token usage, tools with full
  inputs and outputs), tagged with session, user, prompt version and status; inspector feedback and golden-set
  case results are logged on the traces; the vision second opinion traces to `treadops-vision-check`. Spans are
  PII-redacted and stripped of images before export; export is asynchronous and best effort. The LLMOps page
  links each trace to MLflow. Migration `0006`: `llm_traces.mlflow_trace_id`.
- **ConvNeXt-Tiny as a second model family** (`tvmodel.py`, recipe `convnext_tiny_224_robust`): trained, evaluated,
  exported, parity-checked and packaged by the same pipeline stages; its bundles serve through ONNX Runtime
  (`serving.preferred_runtime`).
- **New serving model**: ConvNeXt-Tiny won the benchmark (the only candidate meeting both gate targets; YOLO26s/n
  robust missed recall/specificity; YOLO26m and the control unfinished when the benchmark was stopped) and is
  promoted as registry v4, generation 4. Locked test: recall 0.911, specificity 0.971, 4/282 defective tyres
  auto-cleared, low-light recall 0.901.
- **Depth stage (advisory)**: Depth Anything V2 Small (ONNX, Apache-2.0) as stage 3 of every inspection: relative
  depth, the nearest surface outlined and a framing check; never part of the verdict.
- **Line cameras, live**: inspection zone (ROI) per camera; annotated live frames about 5 per second (zone in the
  verdict's colour for the tyre in it, damage-locator boxes on every frame) on the Cameras page and in Bib's chat;
  connect cameras from Bib (inspectors and admins too); stop-and-go capture (`on_change`: once a tyre has stopped
  in the zone); demo belt video and an RTSP stream for it (`scripts/make_demo_line_video.py`,
  `python scripts/sim_cameras.py up`). Migration `0007`: `cameras.roi`, `cameras.overlay`.
- **Damage locator authority**: bundled even when its gate fails, as `display_only` (experimental boxes, no
  escalation); `escalate` only after it passes.
- **Vision check circuit breaker**: a TCP probe and a breaker (60 s, after two timeouts or connection errors in a
  row, or at once when the connection is refused) stop an unreachable provider from stalling inspections for ~50 s
  each (still fail-safe: no auto-clear without a usable answer).

### Changed

- Bib's system prompt v7 and its tool descriptions name "the deployed classifier" instead of YOLO (golden-set
  evaluation still to be run; the gateway was unreachable).
- Camera inspections use a 4×4 occlusion heatmap (16 copies instead of 64) to keep a camera's pace on CPU.
- Stages are ordered and numbered by their place in the pipeline (depth, stored as stage 9, is shown third).

### Fixed

- Bib's step timeline repeated the depth stage and never showed the quality stage (stage events were tracked by
  position in an index-ordered list, and depth, stored as 9, finishes before quality); React reported duplicate
  keys. Events are now tracked by stage index; the timeline numbers stages by pipeline step and shows each once,
  including in answers recorded before the fix.
- One slow vision answer (a timeout among normal 3-4 s answers) paused the second opinion for 5 minutes, sending
  every good tyre on a camera to an expert in the meantime (51 on 2026-10-25).
- The damage panel showed a green "No damage boxes" for the experimental (display-only) locator, which misses many
  cuts; it now says an empty result is no evidence of a sound tyre, and that its boxes never change a decision.
- Bib's PII redaction masked ISO dates and times ("2026-10-24 17:05") as phone numbers and missed numbers with
  a bracketed area code ("+1 (555) 123-4567").
- After Ultralytics was imported (it patches `PIL.Image.open` with a HEIF fallback), a non-image upload raised
  `ModuleNotFoundError` instead of the typed 415 rejection.
- The split took a group's stratification label from whichever member came last; it now comes from the group's
  home source, so an attached photo with a different label cannot re-stratify a primary group.

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
