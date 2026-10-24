"""Live-view annotation for line cameras, drawn on the frame the browser sees (about 5 per second).

- The inspection zone (the camera's ROI, or the whole frame) in the colour of the verdict for the tyre in it: green
  "NOT DAMAGED", red "DAMAGED", amber "EXPERT CHECK". The verdict comes from the normal inspection of a captured
  frame (about a second later, "INSPECTING..." until then) and is dropped as soon as the zone changes, so a tyre
  never wears the previous tyre's verdict.
- Damage-locator boxes (YOLO26) found in the zone on this very frame, so they follow the tyre. A locator that did
  not pass its gate is labelled experimental: its boxes are shown, never acted on.
- A heads-up line: camera, line, time, frame rate and the serving model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

COLOURS = {"not_damaged": (108, 164, 48), "damaged": (77, 72, 229), "uncertain": (36, 165, 245)}  # BGR
NO_RESULT = (235, 235, 235)
LABELS = {"not_damaged": "NOT DAMAGED", "damaged": "DAMAGED", "uncertain": "EXPERT CHECK"}
BOX = (0, 229, 252)  # TreadOps yellow
FONT = cv2.FONT_HERSHEY_SIMPLEX


def zone_px(shape: tuple[int, ...], roi: list[float] | None) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    r = roi or [0.0, 0.0, 1.0, 1.0]
    return int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h)


def _label(img: np.ndarray, text: str, x: int, y: int, bg: tuple[int, int, int], scale: float, fg=(255, 255, 255)) -> None:
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, FONT, scale, th)
    y0 = max(0, y - tht - base - 8)
    cv2.rectangle(img, (x, y0), (x + tw + 12, y0 + tht + base + 8), bg, -1)
    cv2.putText(img, text, (x + 6, y0 + tht + 4), FONT, scale, fg, th, cv2.LINE_AA)


def annotate(frame: np.ndarray, *, roi: list[float] | None, result: dict[str, Any] | None,
             boxes: list[dict[str, Any]] | None, experimental: bool, camera: str, line: str | None,
             fps: float | None, model: str | None, simulated: bool, inspecting: bool = False) -> np.ndarray:
    """A copy of `frame` (BGR) with the overlay. `boxes` are normalised to the zone; `result` is the latest
    verdict for this camera (None when there is none recent)."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.5, w / 1600)
    x0, y0, x1, y1 = zone_px(img.shape, roi)
    verdict = (result or {}).get("verdict")
    colour = COLOURS.get(str(verdict), NO_RESULT)
    thick = max(2, w // 320)
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, thick)
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):  # corner ticks
        L = max(18, (x1 - x0) // 12)
        cv2.line(img, (cx, cy), (cx + dx * L, cy), colour, thick * 2)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), colour, thick * 2)
    if result:
        text = f"{LABELS.get(str(verdict), 'RESULT')}  p {result.get('p_defective', 0):.2f}"
        _label(img, text, x0, y0, colour, scale * 1.1)
    else:
        _label(img, "INSPECTING..." if inspecting else "INSPECTION ZONE", x0, y0, (60, 60, 60), scale * 0.9)

    zw, zh = x1 - x0, y1 - y0
    for b in boxes or []:
        bx0, by0, bx1, by1 = (int(x0 + b["box"][0] * zw), int(y0 + b["box"][1] * zh),
                              int(x0 + b["box"][2] * zw), int(y0 + b["box"][3] * zh))
        cv2.rectangle(img, (bx0, by0), (bx1, by1), BOX, thick)
        tag = f"{b['class'].replace('_', ' ')} {b['confidence']:.2f}" + (" (exp.)" if experimental else "")
        _label(img, tag, bx0, max(by0, int(28 * scale)), BOX, scale * 0.75, fg=(30, 30, 30))

    stamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
    hud = f"LIVE  {camera}" + (f" · {line}" if line else "") + f"  {stamp}"
    cv2.circle(img, (int(18 * scale) + 4, int(22 * scale) + 4), max(4, int(7 * scale)), (60, 60, 230), -1)
    cv2.putText(img, hud, (int(34 * scale) + 4, int(30 * scale) + 4), FONT, scale * 0.8, (255, 255, 255),
                max(1, int(scale * 2)), cv2.LINE_AA)
    right = " · ".join(filter(None, [f"{fps:.0f} fps" if fps else None, model]))
    if right:
        (tw, _), _ = cv2.getTextSize(right, FONT, scale * 0.7, 1)
        cv2.putText(img, right, (w - tw - 14, int(30 * scale) + 4), FONT, scale * 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    if simulated:
        cv2.putText(img, "SIMULATED STREAM", (14, h - 14), FONT, scale * 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img
