"""Make a simulated inspection-line video for demos: real public tyre photos pass under a fixed camera on a belt.

There is no openly licensed footage of a retread plant, so this composes one: a moving conveyor, and each tyre photo
is indexed into the inspection zone, held there for 5 s, and moves on. Photos are a fixed-seed mix of good and defective
development-validation photos (TyreNet, D1, MSU: CC BY 4.0 / CC0; never the protected roles, because verified camera
inspections can become training data) plus box-labelled damage photos (Roboflow Universe, CC BY 4.0) and any extra
photos given. The frame says "simulated". `python scripts/sim_cameras.py up` publishes it as
rtsp://127.0.0.1:8554/line3-conveyor; register it as a simulated camera with the inspection zone printed at the end.

Usage: python scripts/make_demo_line_video.py [--out runs/demo/treadops_demo_line.mp4] [--extra photo.jpg ...]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 25
SIDE = 600                          # tyre photo size in the frame at rest
REST_X, REST_Y = (W - SIDE) // 2, (H - SIDE) // 2
MOVE_S, DWELL_S = 1.2, 5.0          # stop-and-go belt: index one pitch, then hold the tyre in the zone
PITCH = SIDE + 90                   # spacing between tyres on the belt
ZONE = [round((REST_X - 12) / W, 4), round((REST_Y - 12) / H, 4), round((REST_X + SIDE + 12) / W, 4),
        round((REST_Y + SIDE + 12) / H, 4)]
DAMAGE = ["rf-tw-d17b87c7837ed29c", "rf-tw-0614e2d514a9f845", "rf-tq-aba56c338b5609e6", "rf-tw-c84879f833fa388b"]


def square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (SIDE, SIDE), interpolation=cv2.INTER_AREA)


def rounded_mask(side: int, r: int = 28) -> np.ndarray:
    m = np.zeros((side, side), np.uint8)
    cv2.rectangle(m, (r, 0), (side - r, side), 255, -1)
    cv2.rectangle(m, (0, r), (side, side - r), 255, -1)
    for cx, cy in ((r, r), (side - r, r), (r, side - r), (side - r, side - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0


def photos(extra: list[str]) -> list[tuple[np.ndarray, str]]:
    rows = [json.loads(line) for line in (ROOT / "data/eval/dev_val.jsonl").read_text().splitlines()]
    rng = random.Random(7)
    good = rng.sample([r for r in rows if r["label"] == "good"], 6)
    bad = rng.sample([r for r in rows if r["label"] == "defective"], 4)
    out = [(cv2.imread(str(ROOT / r["path"])), f"{r['source_id']} {r['label']}") for r in good + bad]
    for stem in DAMAGE:
        hit = next(iter((ROOT / "data/damage/yolo/images").glob(f"*/{stem}.*")), None)
        if hit is not None:
            img = cv2.imread(str(hit))
            ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 12)  # drop the export's letterbox bars
            out.append((img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "Roboflow damage"))
    out += [(cv2.imread(p), "extra") for p in extra]
    order = [out[i] for i in (0, 10, 1, 6, 2, 11, 3, 7, 12, 4, 8, 13, 5, 9)] + out[14:]
    return [(square(img), tag) for img, tag in order if img is not None]


def background(belt: float) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = (38, 40, 44)
    belt_top, belt_bot = 40, H - 40
    bg[belt_top:belt_bot] = (26, 27, 29)
    offset = int(belt) % 64  # belt slats move with the tyres
    for x in range(-offset, W, 64):
        cv2.line(bg, (x, belt_top), (x, belt_bot), (34, 35, 38), 6)
    for y in (belt_top - 14, belt_bot + 4):  # side rails with bolts
        cv2.rectangle(bg, (0, y), (W, y + 10), (120, 124, 130), -1)
        for x in range(-(int(belt) % 120), W, 120):
            cv2.circle(bg, (x + 30, y + 5), 3, (70, 72, 76), -1)
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1.0 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return bg * vignette[:, :, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo/treadops_demo_line.mp4")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    tyres = photos(a.extra)
    mask = rounded_mask(SIDE)
    period = MOVE_S + DWELL_S
    total = period * len(tyres)
    loop = PITCH * len(tyres)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-preset", "medium", "-crf", "20", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    rng = np.random.default_rng(0)
    for i in range(int(total * FPS)):
        t = i / FPS
        cycle, local = divmod(t, period)
        u = min(1.0, local / MOVE_S)
        moving = u < 1.0
        belt = (cycle + (3 * u * u - 2 * u ** 3)) * PITCH  # smoothstep: start and stop gently
        frame = background(belt)
        for k, (img, _) in enumerate(tyres):
            d = (k * PITCH - belt) % loop  # wrap so the loop is seamless
            if d > loop / 2:
                d -= loop
            x = int(REST_X + d)
            if x <= -SIDE or x >= W:
                continue
            tile = cv2.GaussianBlur(img, (9, 1), 0) if moving else img  # a little motion blur in transit
            x0, x1 = max(0, x), min(W, x + SIDE)
            if x1 <= x0:
                continue
            sl = slice(x0 - x, x1 - x)
            shadow = frame[REST_Y + 10:REST_Y + SIDE + 10, x0 + 10:min(W, x1 + 10)]
            shadow *= 0.55
            region = frame[REST_Y:REST_Y + SIDE, x0:x1]
            m = mask[:, sl]
            region[:] = region * (1 - m) + tile[:, sl].astype(np.float32) * m
        frame += rng.normal(0, 2.2, frame.shape)  # sensor noise
        out8 = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.putText(out8, "Simulated line - public tyre photos (CC BY 4.0 / CC0)", (W - 470, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        assert ff.stdin is not None
        ff.stdin.write(out8.tobytes())
    assert ff.stdin is not None
    ff.stdin.close()
    ff.wait()
    print(json.dumps({"video": str(out), "seconds": round(total, 1), "tyres": [tag for _, tag in tyres],
                      "inspection_zone_roi": ZONE, "fps": FPS, "size": [W, H]}, indent=1))
    return ff.returncode


if __name__ == "__main__":
    raise SystemExit(main())
