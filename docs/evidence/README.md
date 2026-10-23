# Evidence index

Every claim in the top-level README maps to a file here. Files are produced by real runs; nothing is hand-edited.

| Phase | File | What it shows |
|---|---|---|
| Data | `data/download_receipt.json` | Publisher endpoint, retrieval time, archive SHA-256, publisher per-file hashes |
| Data | `../../data/manifests/source_manifest.json`, `quality_report.json`, `duplicate_review.csv`, `split_manifest.json` | Actual counts, hash checks, reviewed duplicate decisions, grouped split and leakage checks |
| Training | `phase2/train_nano_224.log` | First baseline training run (MPS) console output |
| Evaluation | `phase3/baseline_nano_224/` | Nano bundle: gate report (failed specificity), locked test, stress, parity, latency, training record |
| Evaluation | `test_access_log.jsonl` | Every locked-test access with purpose and checkpoint hash (one per candidate) |
| MLOps loop | `phase5/challenger_small_224/` | Small challenger trained through the app's training job: gate report (passed), locked test, stress, parity |
| Monitoring | `phase5/monitoring_low_light.json` | Low-light replay windows (PSI, embedding AUC, Evidently), alerts opened after two consecutive windows |
| Reliability | `phase6/load_report.json` | 2 req/s for 5 min: acceptance p50/p95/p99, completion within drain, overload 429s, hardware |
| LLMOps | `llmops/copilot_eval_*.json` | Golden-set copilot evaluation (guardrails, tool routing, grounding) with prompt version/SHA |
| Tests | `tests/*.xml` | JUnit results for unit, data/ML and integration suites |
| Summary | `status.json` | Aggregated by `make evidence` |
