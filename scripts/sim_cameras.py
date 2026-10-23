"""Simulated line cameras for demos and tests: real RTSP streams made from dataset tyre photos.

Builds one looping slideshow video per camera from validation images (never the locked test role), starts a
MediaMTX RTSP server in Docker and publishes each video with ffmpeg:

    rtsp://127.0.0.1:8554/line1-intake      rtsp://127.0.0.1:8554/line2-buffing

Register them in Cameras with "Simulated stream" ticked so their inspections are labelled simulated.
`clip` writes runs/sim-cameras/turning-tyre.mp4 (upload it in the Bib chat) and turning-tyre.mjpeg (Chromium's fake
camera for the live-camera journey): one tyre photo swept like a tyre turning in front of a camera.
Usage: python scripts/sim_cameras.py up|down|clip   (up needs Docker; up and clip need ffmpeg)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sim-cameras"
SERVER = "treadops-rtsp"
IMAGE = "bluenviron/mediamtx:1.9.3"
STREAMS = {"line1-intake": 0, "line2-buffing": 6}  # offset into the alternating defective/good image list
SECONDS_PER_TYRE, FPS, SIZE = 4, 10, (1280, 720)


def tyre_images(n: int = 12) -> list[Path]:
    base = ROOT / "data" / "yolo" / "dev" / "val"
    defective = sorted((base / "defective").glob("*"))[: n // 2]
    good = sorted((base / "good").glob("*"))[: n // 2]
    if not defective or not good:
        sys.exit("validation images not found under data/yolo/dev/val; run make prepare-data first")
    return [p for pair in zip(defective, good, strict=False) for p in pair]


def fill(img: np.ndarray) -> np.ndarray:
    """Scale to cover the 16:9 frame and centre-crop, like a camera framed on the tyre (no artificial bars,
    which the model never saw in training)."""
    w, h = SIZE
    scale = max(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (int(np.ceil(img.shape[1] * scale)), int(np.ceil(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    y, x = (resized.shape[0] - h) // 2, (resized.shape[1] - w) // 2
    return resized[y : y + h, x : x + w]


def build(name: str, images: list[Path]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{name}.avi"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for p in images:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        frame = fill(frame)
        for _ in range(SECONDS_PER_TYRE * FPS):
            writer.write(frame)
    writer.release()
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-g", str(FPS * 2), str(mp4)], check=True)
    raw.unlink()
    return mp4


def clip() -> Path:
    """One tyre 'turning': the tread of a validation photo scrolls past the camera through one full cycle in 8 s
    (wrap-around), so every sector is seen once and consecutive frames really differ."""
    OUT.mkdir(parents=True, exist_ok=True)
    img = fill(cv2.imread(str(tyre_images()[0])))
    w, h = 960, 540
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    raw = OUT / "turning-tyre.avi"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    steps = 8 * FPS
    for i in range(steps):
        writer.write(np.roll(img, int(h * i / steps), axis=0))
    writer.release()
    mp4 = OUT / "turning-tyre.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "mjpeg", "-q:v", "3",
                    str(OUT / "turning-tyre.mjpeg")], check=True)
    raw.unlink()
    print(mp4)
    return mp4


def up() -> None:
    images = tyre_images()
    subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", SERVER, "-p", "8554:8554", IMAGE], check=True,
                   capture_output=True)
    for name, offset in STREAMS.items():
        video = build(name, images[offset:] + images[:offset])
        log = (OUT / f"{name}.log").open("w")
        proc = subprocess.Popen(["ffmpeg", "-re", "-stream_loop", "-1", "-i", str(video), "-c", "copy", "-f", "rtsp",
                                 "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:8554/{name}"],
                                stdout=log, stderr=log, start_new_session=True)
        (OUT / f"{name}.pid").write_text(str(proc.pid))
        print(f"rtsp://127.0.0.1:8554/{name}  ({len(images)} tyres, {SECONDS_PER_TYRE} s each, looping)")


def down() -> None:
    for pid_file in OUT.glob("*.pid"):
        subprocess.run(["kill", pid_file.read_text().strip()], capture_output=True)
        pid_file.unlink()
    subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)
    print("simulated cameras stopped")


if __name__ == "__main__":
    {"up": up, "down": down, "clip": clip}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
