#!/usr/bin/env bash
# Native dev runner: Compose provides postgres/redis/object-store/mlflow; app processes run on the host
# (Apple MPS for training, CPU for inference). Usage: scripts/dev.sh up|down|status [service...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/runs/dev"; mkdir -p "$RUN"
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export YOLO_AUTOINSTALL=false YOLO_VERBOSE=false OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin"
WEB_PORT="${WEB_PORT:-3100}"

cmd_for() {
  case "$1" in
    api) echo "$PY/uvicorn treadops_api.main:app --host 127.0.0.1 --port 8000" ;;
    dispatcher) echo "$PY/python -m treadops_worker.dispatcher" ;;
    inference) echo "$PY/celery -A treadops_worker.celery_app worker -Q inference -n inference@%h --pool=solo --loglevel INFO" ;;
    ops) echo "$PY/celery -A treadops_worker.celery_app worker -Q ops -n ops@%h --pool=threads --concurrency 2 --loglevel INFO" ;;
    training) echo "env TRAIN_DEVICE=${TRAIN_DEVICE_DEV:-mps} $PY/celery -A treadops_worker.celery_app worker -Q training -n training@%h --pool=solo --loglevel INFO" ;;
    cameras) echo "$PY/python -m treadops_worker.cameras" ;;
    web) echo "npm --prefix apps/web run dev -- --port $WEB_PORT" ;;
    *) echo "unknown service $1" >&2; return 1 ;;
  esac
}
ALL=(api dispatcher inference ops training cameras web)

start() {
  local s=$1 pidf="$RUN/$1.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "$s already running ($(cat "$pidf"))"; return; fi
  nohup bash -c "exec $(cmd_for "$s")" >>"$RUN/$s.log" 2>&1 &
  echo $! >"$pidf"; echo "started $s (pid $!, log runs/dev/$s.log)"
}
stop() {
  local s=$1 pidf="$RUN/$1.pid"
  if [ -f "$pidf" ]; then
    local pid; pid=$(cat "$pidf")
    pkill -TERM -P "$pid" 2>/dev/null || true; kill -TERM "$pid" 2>/dev/null || true
    rm -f "$pidf"; echo "stopped $s"
  fi
}
status() {
  for s in "${ALL[@]}"; do
    local pidf="$RUN/$s.pid"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "$s: running ($(cat "$pidf"))"; else echo "$s: stopped"; fi
  done
}
action=${1:-status}; shift || true
if [ $# -gt 0 ]; then targets=("$@"); else targets=("${ALL[@]}"); fi
case "$action" in
  up) for s in "${targets[@]}"; do start "$s"; done ;;
  down) for s in "${targets[@]}"; do stop "$s"; done ;;
  restart) for s in "${targets[@]}"; do stop "$s"; sleep 1; start "$s"; done ;;
  status) status ;;
  *) echo "usage: $0 up|down|restart|status [service...]"; exit 2 ;;
esac
