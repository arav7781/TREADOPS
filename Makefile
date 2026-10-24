# TreadOps command interface (spec §32). Every target exits non-zero on failure; evidence lands in docs/evidence.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY      := .venv/bin/python
UV      := uv
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose)
DC      := $(COMPOSE) --env-file .env -f infra/docker-compose.yml
ENVRUN  := set -a; [ -f .env ] && . ./.env; set +a; export YOLO_AUTOINSTALL=false PATH="$(CURDIR)/.venv/bin:$$PATH";
ARCHIVE ?= data/raw/archive/bn7ch8tvyp-1.zip
CONFIG  ?= configs/train.yaml
SCENARIO ?= low_light
SEED    ?= 42
COUNT   ?= 600
RATE    ?= 2
EVID    := docs/evidence

.PHONY: help env bootstrap infra up down dev dev-down migrate seed-demo fetch-data prepare-data split register-data \
        train-smoke train evaluate export-parity package import-bundle pipeline replay test test-unit test-ml \
        test-integration test-e2e test-load eval-copilot contracts contracts-check lint typecheck evidence licenses hooks clean-cache

help: ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-18s %s\n", $$1, $$2}'

env: ## render .env with locally generated secrets (never committed)
	python3 scripts/make_env.py

bootstrap: env ## install locked Python + JS dependencies
	UV_PYTHON=python3.11 $(UV) sync --locked
	npm --prefix apps/web ci

infra: ## start postgres, redis, object store, mlflow (localhost-only ports)
	$(DC) up -d --build postgres redis object-store mlflow
	$(ENVRUN) $(PY) -m treadops_api.cli ensure-bucket

up: ## Compose CPU application profile (all services in containers; web on $$WEB_PORT)
	$(DC) --profile app up -d --build

down: ## stop the Compose stack (volumes kept)
	$(DC) --profile app down

dev: infra migrate ## native dev: API, dispatcher and workers on the host + Next.js dev server
	scripts/dev.sh up

dev-down: ## stop native dev processes
	scripts/dev.sh down

migrate: ## alembic upgrade head (one-shot)
	$(ENVRUN) .venv/bin/alembic upgrade head

seed-demo: ## idempotent fixture users (password = DEMO_PASSWORD in .env)
	$(ENVRUN) $(PY) -m treadops_api.cli seed-demo

fetch-data: ## download D1 and the extra public sources through the publishers' APIs (resumable, checksum-verified)
	$(PY) -m treadops_ml.acquire --dest $(ARCHIVE)
	$(PY) -m treadops_ml.acquire --extra
	$(ENVRUN) $(PY) -m treadops_ml.acquire --damage

prepare-data: ## audit + canonicalise the archive, then split (ARCHIVE=/path/to/zip)
	$(ENVRUN) $(PY) -m treadops_ml.prepare --archive $(ARCHIVE)
	$(ENVRUN) $(PY) -m treadops_ml.split
	$(ENVRUN) .venv/bin/dvc commit -f prepare split

register-data: ## load the frozen catalogue into SQL/object storage + baseline snapshot
	$(ENVRUN) $(PY) -m treadops_api.cli register-dataset
	$(ENVRUN) $(PY) -m treadops_api.cli baseline-snapshot

train-smoke: ## real YOLO, 1 epoch on 10% of the dev cohort (CPU)
	@mkdir -p $(EVID)/phase2
	$(ENVRUN) $(PY) -m treadops_ml.train --recipe nano_smoke --out-dir artifacts/smoke --name nano_smoke --device cpu \
	  | tee $(EVID)/phase2/train_smoke.json

train: ## DVC train stage (CONFIG=configs/train.yaml; TRAIN_DEVICE=mps|0|cpu)
	$(ENVRUN) .venv/bin/dvc repro train

evaluate: ## calibration + policy + gate cohort + locked test (DVC)
	$(ENVRUN) .venv/bin/dvc repro evaluate

export-parity: ## ONNX export and native parity report (DVC)
	$(ENVRUN) .venv/bin/dvc repro export_parity

package: ## assemble, hash and register the bundle (DVC)
	$(ENVRUN) .venv/bin/dvc repro package

pipeline: ## full DVC pipeline then push artifacts to the DVC remote
	$(ENVRUN) .venv/bin/dvc repro && .venv/bin/dvc push

import-bundle: ## upload artifacts/bundle as a model version (promote it from the UI or API)
	$(ENVRUN) $(PY) -m treadops_api.cli import-bundle artifacts/bundle

replay: ## start a replay through the API (SCENARIO, SEED, COUNT, RATE)
	$(ENVRUN) $(PY) scripts/replay_cli.py --scenario $(SCENARIO) --seed $(SEED) --count $(COUNT) --rate $(RATE)

lint: ## python lint + types, web lint
	.venv/bin/ruff check packages services tests scripts
	.venv/bin/mypy --ignore-missing-imports services/api/treadops_api services/worker/treadops_worker packages/treadops_agent/treadops_agent packages/treadops_ml/treadops_ml/attribution.py packages/treadops_ml/treadops_ml/monitoring.py packages/treadops_ml/treadops_ml/replay.py packages/treadops_ml/treadops_ml/acquire.py packages/treadops_ml/treadops_ml/weights.py packages/treadops_ml/treadops_ml/mlflow_model.py
	npm --prefix apps/web run lint

typecheck: ## web typecheck
	npm --prefix apps/web run typecheck

contracts: ## regenerate OpenAPI + TS types
	$(PY) scripts/export_openapi.py
	npm --prefix apps/web run gen:api

contracts-check: ## fail on an unreviewed schema change
	$(PY) scripts/export_openapi.py --check

test-unit:
	$(ENVRUN) $(PY) -m pytest tests/unit -q --junitxml=$(EVID)/tests/unit.xml

test-ml:
	$(ENVRUN) $(PY) -m pytest tests/ml -q -rs --junitxml=$(EVID)/tests/ml.xml

test-integration: ## real postgres/redis/object store (make infra first)
	$(ENVRUN) $(PY) -m pytest tests/integration -q -rs --junitxml=$(EVID)/tests/integration.xml

test: lint contracts-check test-unit test-ml test-integration ## lint/types/unit/data/ML/integration suite
	npm --prefix apps/web run test

test-e2e: ## actual Next.js + API + worker (Playwright)
	npm --prefix apps/web run test:e2e

test-load: ## measured bounded-load scenario (2 req/s for 5 min + overload probe)
	$(ENVRUN) $(PY) tests/load/load_test.py --rate 2 --duration 300 --out $(EVID)/phase6/load_report.json

eval-copilot: ## offline copilot eval (guardrails, tool routing, grounding) -> docs/evidence/llmops
	$(ENVRUN) $(PY) -m treadops_agent.evals

evidence: ## aggregate links, checksums and results into docs/evidence/status.json
	$(ENVRUN) $(PY) scripts/evidence.py

hooks: ## install the pre-commit hook that keeps secrets out of .env.example
	printf '#!/bin/sh\nexec python3 scripts/check_env_example.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

licenses: ## regenerate the third-party licence inventories in docs/licenses/ from the lock files
	$(PY) scripts/license_report.py

clean-cache: ## remove local bundle cache (content-addressed; safe)
	rm -rf models/cache
