#!/usr/bin/env bash
# Backup/restore drill (spec §30): dump PostgreSQL, restore into a scratch database, and verify that the restored
# records still reference accessible artifacts (active bundle files + a sample of inspection media).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
set -a; . ./.env; set +a
OUT=docs/evidence/phase6; mkdir -p "$OUT" runs/backups
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DUMP="runs/backups/treadops-$STAMP.dump"
PG="docker exec -i treadops-postgres-1"

$PG pg_dump -U "$POSTGRES_USER" -d treadops -Fc > "$DUMP"
$PG psql -U "$POSTGRES_USER" -d postgres -qc "DROP DATABASE IF EXISTS treadops_restore" -c "CREATE DATABASE treadops_restore"
$PG pg_restore -U "$POSTGRES_USER" -d treadops_restore --no-owner < "$DUMP"

RESTORE_URL="${DATABASE_URL%/treadops}/treadops_restore"
DATABASE_URL="$RESTORE_URL" .venv/bin/python - "$DUMP" "$OUT/restore_drill.json" <<'EOF'
import hashlib, json, sys
from sqlalchemy import func, select, text
from treadops_api.db import session_scope
from treadops_api.models import Inspection, InspectionStage, ModelVersion
from treadops_api.services.inspections import active_deployment
from treadops_api.storage import get_storage

dump, out = sys.argv[1], sys.argv[2]
st = get_storage()
with session_scope() as db:
    ptr, dep = active_deployment(db)
    mv = db.get(ModelVersion, dep.model_version_id) if dep else None
    missing_bundle = [f for f in (mv.manifest["files"] if mv else {}) if not st.exists(f"bundles/{mv.bundle_sha256}/{f}")]
    sample = db.execute(select(Inspection).order_by(func.random()).limit(25)).scalars().all()
    missing_media = []
    for i in sample:
        keys = [i.image.object_key] + [s.object_key for s in db.execute(
            select(InspectionStage).where(InspectionStage.inspection_id == i.id)).scalars() if s.object_key]
        missing_media += [k for k in keys if not st.exists(k)]
    counts = {t: db.execute(text(f"select count(*) from {t}")).scalar() for t in
              ("inspections", "predictions", "review_labels", "deployments", "model_versions", "audit_events")}
report = {"dump": dump, "dump_sha256": hashlib.sha256(open(dump, "rb").read()).hexdigest(),
          "restored_database": "treadops_restore", "row_counts": counts,
          "active_generation": ptr.generation, "active_bundle_sha256": dep.bundle_sha256 if dep else None,
          "bundle_files_checked": len(mv.manifest["files"]) if mv else 0, "missing_bundle_files": missing_bundle,
          "sampled_inspections": len(sample), "missing_media_objects": missing_media,
          "status": "passed" if dep and not missing_bundle and not missing_media else "failed"}
json.dump(report, open(out, "w"), indent=2)
print(json.dumps({k: report[k] for k in ("status", "row_counts", "active_generation", "missing_bundle_files",
                                         "missing_media_objects")}, indent=2))
EOF
$PG psql -U "$POSTGRES_USER" -d postgres -qc "DROP DATABASE treadops_restore"
