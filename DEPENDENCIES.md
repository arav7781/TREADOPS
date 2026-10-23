# Dependencies

Python dependencies are locked in `uv.lock` (install with `uv sync --locked`); web dependencies in
`apps/web/package-lock.json` (`npm ci`). Exact pins for the ML-critical packages:

| Package | Version | Licence | Note |
|---|---|---|---|
| ultralytics | 8.4.183 | AGPL-3.0 (commercial licence available) | YOLO26 classification |
| torch / torchvision | 2.14.1 / 0.29.1 | BSD-3 | `+cpu` wheels on Linux (ADR 0002) |
| onnx / onnxruntime | 1.23.2 / 1.31.0 | Apache-2.0 / MIT | parity checks |
| mlflow | 3.17.2 | Apache-2.0 | tracking + registry (server image pins the same) |
| evidently | 0.7.23 | Apache-2.0 | requires pandas < 3 (ADR 0005) |
| langgraph | 1.2.16 | MIT | copilot state machine |
| openai | resolved in uv.lock | Apache-2.0 | OpenAI-compatible transport for `LLMClient` |
| dvc[s3] | 3.67.1 | Apache-2.0 | pipeline + S3 remote |

Pretrained weights are pinned by SHA-256 in `configs/weights.lock.yaml` and verified before every use.

## Licensing notes
- Ultralytics is AGPL-3.0; this project is therefore licensed AGPL-3.0-only. Review Ultralytics' commercial
  licence (https://www.ultralytics.com/license) before any proprietary distribution.
- Dataset D1: CC BY 4.0, Pathmanaban P. et al., Mendeley Data V1, doi:10.17632/bn7ch8tvyp.1. Attribution is shown
  in the UI and recorded in every snapshot.
- The full runtime inventories with every package's licence are generated into [docs/licenses/](docs/licenses/README.md)
  (`make licenses`); [NOTICE.md](NOTICE.md) covers copyleft components, fonts, data and trademarks.
