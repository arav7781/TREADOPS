# Runbooks

| Symptom | Response |
|---|---|
| No trained model | `/readyz` and the UI report `untrained`. Run `make train evaluate export-parity package import-bundle`, then promote from **Models** (admin). Never substitute ImageNet predictions. |
| Drift alert | Open **Monitoring**: check affected cameras/scenarios, descriptor PSI and sample images; collect verified labels in **Reviews**. Move the alert to investigating/resolved with a reason. Alerts never retrain or promote. |
| GPU / MPS out of memory | The training job fails with a typed error and releases the `training-slot` lease. Start a new run with a lower-batch recipe; do not kill the inference worker. |
| Queue growth | **Dashboard** shows oldest queued age and worker heartbeats. Check `scripts/dev.sh status` (or `docker compose ps`), broker connectivity and `retry_wait` jobs. Uploads return 429 when `INFERENCE_QUEUE_LIMIT` is reached; lower the replay rate. |
| Bad candidate / quality drop | Keep the current model. Read the candidate's gate report and stress results in **Models -> Evidence**; check leakage, labels, transforms and calibration before retraining. |
| Release failure | The promotion job is `failed` with a typed reason (`warmup_failed`, `generation_conflict`, `bundle_hash_mismatch`); the audit log records `release.*.failed` with `pointer: unchanged`. Fix the cause, then retry with the current expected generation. |
| Post-release regression | **Models -> Rollback** to a previously deployed, gate-passing version with a reason. Generation increments; earlier predictions keep their original lineage. |
| Artifact mismatch | Workers refuse bundles whose files do not match `manifest.json`; the error names the file and expected/actual SHA. Re-import the trusted bundle from the DVC cache (`dvc checkout artifacts/bundle && make import-bundle`). |
| Lost database/storage | Restore Postgres and the object store from the same backup point, `make migrate`, check `/readyz` reports the active bundle `ready`, and open a few inspection detail pages to confirm media links resolve. |
| Copilot answers look wrong | **LLMOps -> Traces**: inspect spans, tool results and guardrail events for the turn. Run `make eval-copilot` before changing the prompt; bump the prompt `version` on every edit. |

## Restart drill (spec §35)
1. `make test-load` in one terminal (2 req/s).
2. `scripts/dev.sh down inference` for 30 s, then `scripts/dev.sh up inference`; queue age rises, then drains.
3. `docker restart treadops-redis-1`; the dispatcher backs off and re-publishes pending outbox events.
4. Check: every accepted inspection reaches a terminal state, one prediction per inspection
   (`SELECT inspection_id, count(*) FROM predictions GROUP BY 1 HAVING count(*) > 1` returns nothing).

## Backup
`docker exec treadops-postgres-1 pg_dump -U treadops treadops > backup.sql` plus a copy of the `treadops`
bucket (`bundles/`, `images/`, `uploads/`, `canonical/`, `inspections/`) and `dvc push` for data and model bytes.
