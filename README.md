# TreadOps — casing circularity with YOLO26, governed MLOps and a guarded copilot

TreadOps turns a phone photo of a tyre into a **casing-intake decision** for a retreading operation:
*retread intake*, *expert inspection*, or *end-of-life material recovery*. An Ultralytics YOLO26 classifier
triages the photo, every preprocessing step is shown with its real timing, a human inspector gives final
clearance, and a sustainability ledger estimates the raw material and CO₂e kept in use, counting only
**inspector-verified** retreads. Around that sits a complete MLOps loop (DVC, MLflow, gated promotion,
rollback, drift monitoring, operational replay) and a multimodal **LangGraph copilot** with guardrails and
LLMOps tracing.

> Independent engineering prototype by Arav Saxena (23070126018) and Mahi Laddha (23070126069), SIT Pune —
> AI & Machine Learning. Inspired by public descriptions of Michelin's TreadVision AI casing inspection
> (March 2026). It uses public photographs and explicitly simulated operational metadata; **no Michelin
> production data is used** and nothing here describes a Michelin deployment.

## What you can do with it

| Role | Flow |
|---|---|
| Inspector | Drop a tyre photo on **Inspect** and watch the live pipeline: ingest (EXIF orientation, metadata stripped) → canonical image → 224×224 letterbox (padding tinted; no centre crop) → R/G/B tensor planes → image-quality screen → YOLO inference with calibrated probability → occlusion heatmap → annotated **classified image** and a recommended circular pathway. Record good / defective / unjudgeable plus a disposition in **Reviews**. |
| Anyone | Ask the **Copilot** about a photo ("Can this casing be retreaded?"). It classifies through the deployed YOLO model, explains the decision, cites the knowledge base and suggests next steps, without ever certifying a tyre as safe. |
| ML engineer | Create an eligible snapshot, train an allowlisted recipe, compare gate evidence, run shadow evaluation, run deterministic replays (low light, blur, prior shift, burst), and review drift alerts. |
| Admin | Promote or roll back with a reason and expected generation; audit everything. |

## Measured results (this repository, 2026-10-22/23)

All numbers below come from saved runs in `docs/evidence/`; the targets are the blueprint's proposed goals.

**Data (Mendeley D1, doi:10.17632/bn7ch8tvyp.1, CC BY 4.0).** The archive holds 1,856 image members: 28 are
byte-identical copies, leaving **1,828 unique photos** (1,000 defective, 828 good), all matching the
publisher's SHA-256. A visual review of duplicate candidates (ADR 0003) grouped re-photographed tyres into
**948 label-pure groups**, split 69.8 / 15.0 / 15.2 with zero shared hashes or groups across roles.

**Models** (same split; gates on the outer-validation "gate cohort" = development evidence):

| | YOLO26n-cls (v1) | YOLO26s-cls (v2) |
|---|---|---|
| Gate defect recall (target ≥ 0.95) | 0.967 (145/150) ✅ | 0.967 (145/150) ✅ |
| Gate specificity (target ≥ 0.90) | **0.896 (112/125) ❌** | 0.920 (115/125) ✅ |
| Hard gates (leakage, lineage, finite, class map, ONNX parity) | all passed | all passed |
| **Gate status** | **failed → cannot be promoted** | **passed → promoted (generation 1)** |
| Locked test recall / specificity | 0.940 (141/150) / 0.906 (115/127) | 0.953 (143/150) / 0.937 (119/127) |
| Defective test tyres routed to auto-clear queue | 6 | 3 |
| Low-light stress recall (brightness × 0.45) | 0.747 | 0.760 |
| ONNX vs native max abs probability error (120 images) | 6.7e-6 | 1.8e-6 |
| Model-only latency p95, CPU, batch 1 (Apple M5) | 15.0 ms | 12.6 ms |
| Trained on | CPU, 26 epochs (DVC pipeline) | MPS, 18 epochs (via the app's training job) |

The nano baseline missing its specificity target is shown as **failed** in the UI, the gate report and MLflow
(`validation_status=failed`); the API refuses to promote it. Low light remains the main weakness of both models,
which is exactly what the monitoring and replay tooling is for.

**Serving (real photo, end to end):** warm YOLO inference 7–13 ms; whole worker job incl. preprocessing,
occlusion attribution and stage images ≈ 0.3–0.6 s.

**Load (2 req/s for 5 min, Apple M5, single dev API process, CPU inference):** 600/600 accepted, acceptance
p50 53 ms / p95 123 ms / p99 176 ms (target p95 < 1 s ✅), 600/600 completed (target ≥ 99 % ✅). A burst of 180
simultaneous uploads first produced 150 client timeouts and no 429s ❌; after adding an in-flight upload cap it
returns 8 accepted and 172 immediate 429s with `Retry-After` (p95 248 ms) ✅.

**Drills:** restart drill (inference worker paused 30 s, Redis restarted under load): every accepted inspection
reached a terminal state, zero duplicate predictions, no stuck outbox events or orphaned leases ✅. Backup/restore
drill: Postgres restored into a scratch database with the active bundle and sampled inspection media all
resolvable ✅.

**Drift (seed-42 low-light replay, 800 events, simulated metadata):** brightness PSI rose 0.035 → 0.27 → 0.93 →
2.07 across consecutive windows; alerts opened only after two windows above 0.20 and duplicates merged. Mean
p(defective) fell 0.53 → 0.44 as images darkened, matching the stress-test recall drop.

**Tests:** 51 unit, 9 data/ML (on the real data and artifacts), 21 integration against real
PostgreSQL/Redis/object storage (API-01..04, JOB-01..04, REL-01..04, DB-01) — all passing.

## Architecture

```mermaid
flowchart TB
  B[Browser] --> W[Next.js App Router + BFF<br/>HttpOnly session, same-origin checks]
  W --> A[FastAPI /v1<br/>auth, contracts, release policy]
  A -->|txn: inspection + job + outbox| P[(PostgreSQL<br/>business truth, audit, LLMOps traces)]
  A -->|sanitized image| S[(S3-compatible store<br/>images, stages, bundles)]
  D[Outbox dispatcher<br/>+ lease sweeper + scheduler] --> R[(Redis)]
  P --> D
  R --> I[Inference worker<br/>pinned YOLO bundle, CPU]
  R --> O[Ops worker<br/>replay, monitoring, release, shadow]
  R --> T[Training worker<br/>single training-slot lease]
  I & O & T --> P & S
  T --> M[MLflow<br/>runs + registry aliases]
  V[DVC pipeline<br/>prepare → split → train → evaluate → export_parity → package] --> M
  A --> C[Copilot: LangGraph<br/>guard_input → plan ⇄ tools → guard_output]
  C -->|classify_tyre_image| A
```

Key properties: an inspection is accepted (202) only after the sanitized image and the inspection + job +
outbox rows are committed, and it is **pinned to the deployment generation at acceptance**, so a promotion can
never change a queued job. Celery delivery is at-least-once; SQL compare-and-set claims make every job
idempotent. Promotion stages the bundle by SHA, verifies every file hash, replays a golden batch, then switches
the pointer with an expected-generation compare-and-swap; MLflow's `champion` alias is reconciled afterwards by
a retried job.

## Quick start

Requirements: Docker (Compose v2 or `docker-compose`), Python 3.11 with `uv`, Node 22+.

```bash
make env            # .env with locally generated secrets (never committed)
make bootstrap      # uv sync --locked + npm ci
make infra migrate seed-demo
make fetch-data     # or: make prepare-data ARCHIVE=/path/to/bn7ch8tvyp-1.zip
make prepare-data register-data
make pipeline       # dvc repro: train → evaluate → export_parity → package (+ MLflow registration)
make import-bundle  # promote it from Models (admin) only if its gates pass
make dev            # API :8000, workers, web on http://localhost:3100
```

Demo users are `viewer@`, `inspector@`, `mlengineer@` and `admin@treadops.local`; the password is
`DEMO_PASSWORD` in `.env`. `make up` runs everything in containers instead (CPU profile).

### Copilot LLM

Set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` for any OpenAI-compatible endpoint (OpenAI, Azure OpenAI,
LiteLLM, vLLM, Ollama `/v1`). Without them, the copilot runs a deterministic, tool-grounded planner and labels its
traces `deterministic_fallback`. The system prompt is versioned in `configs/prompts/`; `make eval-copilot` runs
the golden set and logs results to MLflow.

## Sustainability ledger

Factors come from Michelin's published truck/bus retreading figures: one MICHELIN Remix retread needs about
50 kg less raw material than a new tyre, and Michelin's internal cradle-to-gate LCA puts the saving at 2.3 kg CO₂
per kg of raw material ([source](https://business.michelin.co.uk/help-advice/truck/tyre-basics-tyre-retreading)).
TreadOps multiplies these by **inspector-verified** retread dispositions only and labels every figure an
estimate. Model recommendations are reported separately.

## Repository map

```
apps/web/                 Next.js UI + BFF
services/api/             FastAPI app, SQL models, services, CLI
services/worker/          Celery runner, dispatcher, job handlers
packages/treadops_ml/     data, transforms, train, evaluate, parity, bundle, monitoring, replay
packages/treadops_agent/  LangGraph copilot, guardrails, tools, tracing, evals
configs/                  data, split, recipes, policy, drift, replay, sustainability, copilot, prompts
migrations/               Alembic
tests/                    unit, ml, integration, load (+ web tests in apps/web)
docs/                     adr/, runbooks/, kb/, evidence/
```

## Limitations

- D1 is phone photography of mixed passenger and commercial tyres, not plant or shearography imagery.
- Same-tyre groups were inferred from visual similarity and capture order; some leakage between visually
  different photos of the same tyre is still possible.
- Labels are image-level. The heatmap is an attribution aid, not a defect location; detection needs a separately
  annotated dataset (blueprint §38).
- Low-light robustness is weak (recall ~0.75 at 45 % brightness).
- Training ran on Apple Silicon (MPS) and CPU, not the RTX 4050 profile in the blueprint (ADR 0002).

## Licence and attribution

AGPL-3.0-only (Ultralytics YOLO is AGPL-3.0; see `DEPENDENCIES.md`). Dataset: Pathmanaban P. et al., "Digital
images of defective and good condition tyres", Mendeley Data V1, doi:10.17632/bn7ch8tvyp.1, CC BY 4.0.
