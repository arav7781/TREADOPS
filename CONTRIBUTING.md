# Contributing to TreadOps

Thanks for helping. TreadOps is an independent engineering prototype; issues and pull requests are welcome.
Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first, and report security problems privately as described
in [SECURITY.md](SECURITY.md).

## Ground rules

These come from [AGENTS.md](AGENTS.md) and apply to every change, human or tool-assisted:

1. **Never invent data, counts or metrics.** Every number in the UI, README or docs must come from a stored run or
   an API response. Measured results live in `docs/evidence/`; regenerate them with `make evidence`.
2. **Never train on calibration, policy or test roles**, or on anything descended from them. Replays are never
   eligible for training.
3. **Promote only through `POST /v1/deployments/promote`** (gates passed, warm-up, expected generation). Never
   write the deployment pointer directly.
4. **State lives in SQL.** Celery delivery is at-least-once, so every job handler must be idempotent.
5. **Keep secrets in `.env`** (created by `make env`). Never print or commit them.
6. **Don't edit DVC pipeline modules casually.** Files under `packages/treadops_ml/treadops_ml/` that `dvc.yaml`
   depends on invalidate `dvc.lock` when changed, even by an auto-formatter. If you change one, rerun the affected
   stages (`dvc repro`) and commit the new `dvc.lock`.

## Development setup

Requirements: Docker (Compose v2), Python 3.11 with [`uv`](https://docs.astral.sh/uv/), Node 22+.

```bash
make env                 # .env with locally generated secrets (never committed)
make bootstrap           # uv sync --locked + npm ci
make infra migrate seed-demo
make dev                 # API :8000, workers, web on http://localhost:3100
```

Getting data and a model (`make fetch-data`, `make pipeline`, `make import-bundle`) is covered in the
[README](README.md#quick-start) and [docs/runbooks/data-acquisition.md](docs/runbooks/data-acquisition.md).
`scripts/dev.sh status|restart <service>` manages the native processes.

The web app runs **Next.js 16**, which has breaking changes from earlier versions. Read
[apps/web/AGENTS.md](apps/web/AGENTS.md) and the guides in `apps/web/node_modules/next/dist/docs/` before changing
routing, caching or data fetching.

## Making a change

1. Fork the repository and branch from `main`.
2. Keep changes focused. Several small, logical commits are better than one large one.
3. Match the surrounding code: naming, comment density (comments explain *why*), error contracts and typing.
4. Add or update tests with the change.
5. Update docs when behaviour changes: README, an ADR in `docs/adr/` for architectural decisions, runbooks for
   operational ones.

### Checks to run before opening a pull request

| Change touches | Run |
|---|---|
| Anything | `make lint` (ruff, mypy, ESLint) and `make typecheck` |
| Python services or packages | `make test-unit test-ml`; `make test-integration` (needs `make infra`) |
| API request or response schemas | `make contracts`, then commit `packages/contracts/openapi.json` and `apps/web/src/lib/api-schema.ts`; CI runs `make contracts-check` |
| Database models | a new Alembic revision under `migrations/versions/`, then `make migrate` |
| Web UI | `npm --prefix apps/web run test` and `make test-e2e` against the running stack |
| Copilot prompt, tools or guardrails | `make eval-copilot`; bump the prompt `version` in `configs/prompts/` on every edit |
| ML pipeline or configs | `dvc repro` for the affected stages; commit `dvc.lock` and new evidence |
| `uv.lock` or `package-lock.json` | `make licenses` to refresh `docs/licenses/` |

Playwright uses a local Chrome by default. Set `PW_EXECUTABLE` to a Chromium or headless-shell binary, or
`PW_CHANNEL` to pick another channel.

### Commit messages

Write the subject in the imperative mood, at most about 72 characters ("Fix promote endpoint job kind"). Use the
body to explain *why*, and mention measured effects where they exist.

## Pull requests

Open the pull request against `main` and fill in the template. CI must be green (lint and types, contract diff,
migrations, unit, data and integration tests, web build, container images). For UI changes, include a screenshot.

## Licensing of contributions

TreadOps is licensed [AGPL-3.0-only](LICENSE). By submitting a contribution you agree that it is licensed under
the same terms, and that you have the right to submit it. Don't add code, data or assets whose licence is
incompatible with AGPL-3.0, and record any new third-party asset in [NOTICE.md](NOTICE.md).
