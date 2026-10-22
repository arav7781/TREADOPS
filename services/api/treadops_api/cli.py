"""Operator commands (idempotent). Usage: python -m treadops_api.cli <command> [options]

  ensure-bucket            create the object-store bucket
  seed-demo                create the four fixture users (viewer, inspector, ml engineer, admin)
  register-dataset         load the frozen D1 catalogue into SQL and upload canonical images
  baseline-snapshot        create the baseline eligible snapshot from the registered catalogue
  import-bundle PATH       upload a packaged bundle and register it as a model version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import select

from .db import session_scope
from .models import User
from .security import hash_password
from .storage import get_storage

DEMO_USERS = [
    ("viewer@treadops.local", "Vera Viewer", ["viewer"]),
    ("inspector@treadops.local", "Ishan Inspector", ["viewer", "inspector"]),
    ("mlengineer@treadops.local", "Mira ML Engineer", ["viewer", "ml_engineer"]),
    ("admin@treadops.local", "Ada Admin", ["viewer", "inspector", "ml_engineer", "admin"]),
]


def seed_demo() -> dict:
    password = os.environ.get("DEMO_PASSWORD")
    if not password:
        raise SystemExit("DEMO_PASSWORD is not set (run `make env`).")
    created = []
    with session_scope() as db:
        for email, name, roles in DEMO_USERS:
            u = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if u is None:
                db.add(User(email=email, display_name=name, password_hash=hash_password(password), roles=roles,
                            is_fixture=True))
                created.append(email)
            else:
                u.roles, u.is_fixture = roles, True
    return {"created": created, "users": [e for e, _, _ in DEMO_USERS], "note": "fixture accounts (is_fixture=true)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="treadops_api.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure-bucket")
    sub.add_parser("seed-demo")
    rd = sub.add_parser("register-dataset")
    rd.add_argument("--no-upload", action="store_true")
    sub.add_parser("baseline-snapshot")
    ib = sub.add_parser("import-bundle")
    ib.add_argument("path", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "ensure-bucket":
        get_storage().ensure_bucket()
        out = {"bucket": get_storage().bucket, "status": "ok"}
    elif args.cmd == "seed-demo":
        out = seed_demo()
    elif args.cmd == "register-dataset":
        from .services.datasets import register_public_catalogue

        with session_scope() as db:
            out = register_public_catalogue(db, get_storage(), upload=not args.no_upload)
    elif args.cmd == "baseline-snapshot":
        from .services.datasets import build_snapshot

        with session_scope() as db:
            snap, created = build_snapshot(db, get_storage(), description="baseline: frozen D1 dev roles",
                                           include_verified_uploads=False)
            out = {"snapshot_id": str(snap.id), "created": created, "counts": snap.counts}
    elif args.cmd == "import-bundle":
        from .services.releases import import_bundle

        with session_scope() as db:
            mv = import_bundle(db, get_storage(), args.path)
            out = {"model_version_id": str(mv.id), "bundle_sha256": mv.bundle_sha256, "gate_status": mv.gate_status,
                   "stage": mv.stage, "registry_version": mv.registry_version}
    else:  # pragma: no cover
        ap.error("unknown command")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
