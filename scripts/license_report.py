"""Generate the third-party licence inventory under docs/licenses/ from the locked, installed dependencies.

Python: the runtime set from `uv export --no-dev` (uv.lock), licences read from installed package metadata.
Web: the production set from `npm query .prod` in apps/web, licences from each package.json.
Nothing is typed by hand; rerun after changing a lock file: `make licenses`.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/licenses"
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|EUPL|OSL)", re.I)
# Packages whose metadata misstates the licence; each value was checked against the LICENSE file the wheel ships.
OVERRIDES = {"grandalf": "GPL-2.0 OR EPL-1.0 (per bundled LICENSE; metadata lists GPLv2 only)"}
NORMALISE = {
    "Apache License 2.0": "Apache-2.0",
    "GNU Affero General Public License v3 or later (AGPLv3+)": "AGPL-3.0-or-later",
    "GPLv2 with linking exception": "GPL-2.0 with linking exception",
}

CLASSIFIER_SPDX = {
    "MIT License": "MIT", "BSD License": "BSD", "Apache Software License": "Apache-2.0",
    "Python Software Foundation License": "PSF-2.0", "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0", "GNU Affero General Public License v3": "AGPL-3.0",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0", "The Unlicense (Unlicense)": "Unlicense",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND", "Zope Public License": "ZPL-2.1",
}


def licence_from_file(name: str) -> str | None:
    """Last resort: recognise the licence from the LICENSE/COPYING file shipped in the wheel."""
    files = [f for f in metadata.distribution(name).files or [] if re.search(r"licen[cs]e|copying", str(f), re.I)]
    for f in files:
        head = (f.read_text() or "")[:600]
        if "Permission is hereby granted, free of charge" in head:
            return "MIT (per bundled LICENSE)"
        if "Apache License" in head and "Version 2.0" in head:
            return "Apache-2.0 (per bundled LICENSE)"
        if "Redistribution and use in source and binary forms" in head:
            return "BSD (per bundled LICENSE)"
    return None


def python_licence(meta: metadata.PackageMetadata) -> str:
    name = meta["Name"]
    if name.lower() in OVERRIDES:
        return OVERRIDES[name.lower()]
    found = _python_licence(meta)
    if found in ("UNKNOWN", "see project (full text in metadata)"):
        found = licence_from_file(name) or found
    return NORMALISE.get(found, found)


def _python_licence(meta: metadata.PackageMetadata) -> str:
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    named = [CLASSIFIER_SPDX.get(c, c) for c in classifiers if c != "OSI Approved"]
    if named:
        return " OR ".join(dict.fromkeys(named))
    text = (meta.get("License") or "").strip()
    if text and len(text) <= 60 and "\n" not in text:
        return text
    return "see project (full text in metadata)" if text else "UNKNOWN"


def python_rows() -> list[tuple[str, str, str, str]]:
    out = subprocess.run(["uv", "export", "--no-dev", "--no-hashes", "--frozen", "--format", "requirements-txt"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e")):
            continue
        spec, _, marker = line.partition(";")
        name, _, version = spec.strip().partition("==")
        try:
            licence = python_licence(metadata.metadata(name))
        except metadata.PackageNotFoundError:  # platform-specific wheel not installed on this machine
            licence = "not installed on this platform"
        rows.append((name, version, licence, marker.strip()))
    return sorted(rows, key=lambda r: r[0].lower())


def npm_rows() -> list[tuple[str, str, str, str]]:
    out = subprocess.run(["npm", "query", ".prod"], cwd=ROOT / "apps/web", capture_output=True, text=True,
                         check=True).stdout
    seen: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for p in json.loads(out):
        if p.get("name") == "web":
            continue
        lic = p.get("license")
        if isinstance(lic, dict):
            lic = lic.get("type")
        lic = lic or "UNKNOWN"
        platform = "/".join(filter(None, [",".join(p.get("os") or []), ",".join(p.get("cpu") or [])]))
        seen[(p["name"], p.get("version", ""))] = (p["name"], p.get("version", ""), lic, platform)
    return sorted(seen.values(), key=lambda r: r[0].lower())


def write(path: Path, title: str, source: str, rows: list[tuple[str, str, str, str]], extra_col: str) -> Counter:
    counts = Counter(r[2] for r in rows)
    lines = [f"# {title}", "", f"Generated by `scripts/license_report.py` from {source}. Do not edit by hand.", "",
             f"{len(rows)} packages. Licences by count:", ""]
    lines += [f"- {lic}: {n}" for lic, n in counts.most_common()]
    lines += ["", f"| Package | Version | Licence | {extra_col} |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {lic}{' ⚠️' if COPYLEFT.search(lic) else ''} | {x} |" for n, v, lic, x in rows]
    path.write_text("\n".join(lines) + "\n")
    return counts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    py = python_rows()
    js = npm_rows()
    write(OUT / "python-runtime.md", "Python runtime dependencies", "`uv.lock` (`uv export --no-dev`)", py,
          "Platform marker")
    write(OUT / "web-runtime.md", "Web runtime dependencies", "`apps/web/package-lock.json` (`npm query .prod`)", js,
          "Platform")
    flagged = [(n, v, lic, "Python") for n, v, lic, _ in py if COPYLEFT.search(lic)]
    flagged += [(n, v, lic, "Web") for n, v, lic, _ in js if COPYLEFT.search(lic)]
    print(json.dumps({"python": len(py), "web": len(js), "copyleft_or_weak_copyleft": flagged}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
