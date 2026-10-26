# Notices

**TreadOps**
Copyright (C) 2026 Arav Saxena and Mahi Laddha

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero
General Public License as published by the Free Software Foundation, version 3 only (`AGPL-3.0-only`).
It is distributed in the hope that it will be useful, but **without any warranty**; without even the implied
warranty of merchantability or fitness for a particular purpose. See [LICENSE](LICENSE) for the full text.

## Why AGPL-3.0

TreadOps trains and serves models with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is
licensed AGPL-3.0. The project therefore uses the same licence. AGPL section 13 matters for this kind of system: if
you run a **modified** TreadOps as a network service (the web UI and API are one), you must offer the users of that
service the Corresponding Source of your modified version. Ultralytics also sells a commercial licence
(<https://www.ultralytics.com/license>) for proprietary use; review it before any closed-source distribution.

## Third-party software

TreadOps uses its dependencies unmodified, through their published packages. The complete runtime inventories
are generated from the lock files by [`scripts/license_report.py`](scripts/license_report.py) (`make licenses`):

- [Python runtime dependencies](docs/licenses/python-runtime.md) (from `uv.lock`)
- [Web runtime dependencies](docs/licenses/web-runtime.md) (from `apps/web/package-lock.json`)

Each package's own licence text ships inside the installed distribution (`*.dist-info/` for Python,
`node_modules/<package>/` for the web app).

| Component | Licence | How TreadOps uses it |
|---|---|---|
| Ultralytics YOLO (`ultralytics`, `ultralytics-thop`, `ultralytics-platform`) | AGPL-3.0 | YOLO26 classification: training, export, serving |
| PyTorch, torchvision | BSD-style (the torch wheel declares a composite expression: BSD, Apache-2.0, MIT, BSL-1.0) | model runtime |
| ONNX, ONNX Runtime | Apache-2.0 / MIT | export and parity checks |
| MLflow | Apache-2.0 | tracking and model registry |
| DVC | Apache-2.0 | data and pipeline versioning |
| Evidently | Apache-2.0 | drift reports |
| LangGraph | MIT | copilot state machine |
| FastAPI, SQLAlchemy, Alembic, Celery | MIT / MIT / MIT / BSD-3-Clause | API, persistence, jobs |
| Next.js, React | MIT | web UI |
| lucide-react | ISC | icons |
| react-markdown, remark-gfm | MIT | rendering copilot answers (no raw HTML) |

Components under copyleft or file-level copyleft licences, all used unmodified:

| Component | Licence | Note |
|---|---|---|
| `psycopg`, `psycopg-binary` | LGPL-3.0-only | PostgreSQL driver, used as a library |
| `@img/sharp-libvips-*` | LGPL-3.0-or-later | platform binary behind Next.js image optimisation |
| `certifi`, `pathspec`, `shtab`, parts of `tqdm` and `orjson` | MPL-2.0 | file-level copyleft |
| `grandalf` | GPL-2.0 OR EPL-1.0 | DVC dependency (pipeline graphs); per its bundled LICENSE file |
| `pygit2` | GPL-2.0 with linking exception | DVC dependency |
| `asyncssh`, `dulwich` | EPL-2.0 OR GPL-2.0-or-later / Apache-2.0 OR GPL-2.0-or-later | DVC dependencies, dual-licensed |

## Fonts

The web app loads these through `next/font`, which self-hosts the files at build time:

- **Noto Sans**: SIL Open Font License 1.1
- **Exo 2**: SIL Open Font License 1.1
- **JetBrains Mono**: SIL Open Font License 1.1

## Data

The model is trained on **"Digital images of defective and good condition tyres"** by Pathmanaban P. et al.,
Mendeley Data V1, doi:[10.17632/bn7ch8tvyp.1](https://doi.org/10.17632/bn7ch8tvyp.1), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Two further phone-photo classification sources are added to training and evaluation, each kept in its own role
carve so the D1 split never changes ([`configs/data.yaml`](configs/data.yaml) `extra_sources`):

| Source | Licence | Attribution |
|---|---|---|
| TyreNet (1,698 photos, service stations and showrooms) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Mahendrada V. S., Parameswaran M., "TyreNet: A High-quality Annotated Dataset for Tyre Defect Classification with Deep Learning Models", Mendeley Data V1, doi:[10.17632/32b5vfj6tc.1](https://doi.org/10.17632/32b5vfj6tc.1) |
| Tire sidewall and tread textures (1,036 photos, cracked or normal) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Siegel J. E. et al., Michigan State University, "Oxidized and non-oxidized tire sidewall and tread images", Harvard Dataverse, doi:[10.7910/DVN/Z3ZYLI](https://doi.org/10.7910/DVN/Z3ZYLI) |

The damage locator (a YOLO26 detector, [`configs/damage.yaml`](configs/damage.yaml)) is trained on two
box-annotated Roboflow Universe exports, both [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
"tire_quality_dataset" v1 by *tirequalitydetection*
(<https://universe.roboflow.com/tirequalitydetection/tire_quality_dataset/dataset/1>) and "TireWear_v2" v11 by
*laundrystain* (<https://universe.roboflow.com/laundrystain/tirewear_v2/dataset/11>). Their classes are renamed
onto TreadOps' five (`Debries` becomes `foreign_object`); photos that duplicate a D1 evaluation image are dropped.

None of these images are stored in this Git repository. `make fetch-data` downloads them from the publishers and
verifies the published SHA-256 (or, for Harvard Dataverse, MD5) sums; download receipts are in
[`docs/evidence/data/`](docs/evidence/data/). Any copies in a DVC remote, and the derived images the app produces,
keep their source's licence and attribution. The derived images are sanitised (metadata stripped), resized, letterboxed,
attribution heatmaps and annotated classified images. The UI shows the attribution too, and every dataset
snapshot records it.

Pretrained weights are pinned by SHA-256 in [`configs/weights.lock.yaml`](configs/weights.lock.yaml):

| Weights | Licence | Used for |
|---|---|---|
| YOLO26 n/s/m classification and n/s detection (Ultralytics) | AGPL-3.0 | benchmark candidates, the damage locator, the drift / subject-gate encoder |
| ConvNeXt-Tiny, ImageNet-1k (torchvision, `convnext_tiny-983f1562.pth`) | BSD-3-Clause | the serving classifier (fine-tuned) |
| Depth Anything V2 Small (ONNX export by onnx-community; upstream depth-anything/Depth-Anything-V2-Small) | Apache-2.0 | the advisory depth stage (unmodified) |

Weights fine-tuned from Ultralytics checkpoints are distributed under AGPL-3.0, the fine-tuned ConvNeXt under the
project licence (its ImageNet weights stay BSD-3-Clause); no weights are committed to Git.

Operational metadata in replays (lines, cameras, scenarios) is **simulated** and labelled as such.

## Images in this repository

`docs/assets/treadops-banner.jpg` is the project banner supplied by the authors for this README. The TreadOps logo
and the icons derived from it (`apps/web/public/brand/`, `apps/web/src/app/icon.png`, `apple-icon.png`) are the
authors' own mark; the white variant for dark backgrounds is a recolouring of it.

## Demo trailer

The demo trailer (`docs/assets/TreadOps-Trailer-v9-1080p60.mp4`, stored with Git LFS and also attached to the
[`trailer-v9` release](https://github.com/arav7781/TREADOPS/releases/tag/trailer-v9)) and
its thumbnail `docs/assets/treadops-trailer-v9.jpg` are the authors' own edit of TreadOps screen recordings. They
include third-party material:

| Material | Licence | Attribution |
|---|---|---|
| Tyre photographs (the thumbnail shows one, in black and white) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | TyreNet (Mahendrada V. S., Parameswaran M.) and Pathmanaban P. et al., Mendeley Data, as listed under Data |
| Outro music, "Tempting Secrets" | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | "Tempting Secrets" Kevin MacLeod (incompetech.com), Licensed under Creative Commons: By Attribution 4.0 |
| Narration | ElevenLabs terms of service | Voice-over generated with ElevenLabs (elevenlabs.io) |

The rest of the music is an original composition. The trailer's own credits card repeats these attributions, and
its prompt-injection sequence is a reconstruction of the guardrail's real behaviour and reply text.

## Trademarks and affiliation

TreadOps is an independent engineering prototype. It is **not affiliated with, sponsored by or endorsed by
Michelin**. MICHELIN, the Michelin Man (Bibendum) and related names are trademarks of Compagnie Générale des
Établissements Michelin. TreadOps names its copilot "Bib" as an homage and draws its colour palette from Michelin's
public website. It uses no Michelin logo, mascot artwork, proprietary typeface or Michelin data. The project refers
to Michelin's published retreading figures only to estimate the circularity ledger, and cites the source.

Ultralytics and YOLO are trademarks of Ultralytics Inc. Next.js is a trademark of Vercel, Inc. All other names are
trademarks of their respective owners, used only to identify their products.
