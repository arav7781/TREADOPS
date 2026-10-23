# Third-party licence inventories

Generated from the lock files by [`scripts/license_report.py`](../../scripts/license_report.py). Run
`make licenses` after changing `uv.lock` or `apps/web/package-lock.json`; don't edit the generated files by hand.

| File | Covers |
|---|---|
| [python-runtime.md](python-runtime.md) | Python runtime dependencies (`uv export --no-dev`), licences from installed package metadata |
| [web-runtime.md](web-runtime.md) | Web production dependencies (`npm query .prod`), licences from each `package.json` |

A ⚠️ marks copyleft or file-level copyleft licences (AGPL, GPL, LGPL, MPL, EPL). [NOTICE.md](../../NOTICE.md)
explains how TreadOps uses each one. Where a package's metadata misstates its licence, the script uses the LICENSE
file the package ships and says so in the table.

Development-only tools (linters, test runners, type checkers) are not listed because they are not part of the
running system.
