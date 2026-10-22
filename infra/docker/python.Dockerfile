# Application image for api / dispatcher / workers / migrate (CPU). Locked install from uv.lock.
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/tmp/ultralytics MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH=/app/.venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY packages packages
COPY services services
COPY configs configs
COPY migrations migrations
COPY alembic.ini dvc.yaml ./
COPY docs/kb docs/kb
COPY data/manifests data/manifests
RUN uv sync --locked --no-dev \
    && python -c "from treadops_ml.weights import pretrained; pretrained('yolo26n-cls.pt'); pretrained('yolo26s-cls.pt')"
RUN useradd --uid 10001 --create-home treadops && mkdir -p /app/models/cache /app/runs && chown -R treadops /app/models /app/runs
USER treadops
EXPOSE 8000
CMD ["uvicorn", "treadops_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
