"""The single full-image transform contract shared by prepare, training, evaluation, serving and ONNX parity.

    raw bytes -> ingest: validate + decode (PIL) + EXIF transpose + RGB + re-encode without metadata ("sanitized")
              -> canonical JPEG (long side <= 1024)
              -> letterbox 224 (aspect preserved, pad 114) -> float32 NCHW in [0, 1]

The API persists only the sanitized image (raw uploads carry EXIF/GPS and are never stored). The prepare stage
runs the same ingest step on archive originals, so training and serving canonical bytes are produced by one path.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import sha256_bytes, sha256_json

# PIL's own bomb guard is replaced by an explicit pixel check before any full decode.
Image.MAX_IMAGE_PIXELS = None

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class PreprocessContract:
    version: str = "v1"
    sanitize_format: str = "JPEG"
    sanitize_quality: int = 95
    canonical_max_side: int = 1024
    canonical_resample: str = "LANCZOS"
    canonical_format: str = "JPEG"
    canonical_quality: int = 95
    canonical_subsampling: int = 0
    imgsz: int = 224
    pad_value: int = 114
    resize_interpolation: str = "cv2.INTER_LINEAR"
    rounding: str = "new=round(dim*r); top,left=round((S-new)/2-0.1)"
    channel_order: str = "RGB"
    layout: str = "NCHW"
    dtype: str = "float32"
    scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"], d["std"] = list(self.mean), list(self.std)
        return d

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessContract:
        d = dict(d)
        d["mean"], d["std"] = tuple(d.get("mean", (0, 0, 0))), tuple(d.get("std", (1, 1, 1)))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Limits:
    """Input validation limits. Deliberately not part of the transform contract hash."""

    max_bytes: int = 8 * 1024 * 1024
    max_decoded_pixels: int = 25_000_000
    min_side: int = 64


class ImageRejected(ValueError):
    """Typed, non-retryable rejection of an input image."""

    def __init__(self, code: str, message: str, http_status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class Ingested:
    sanitized: bytes                       # oriented RGB, re-encoded, no EXIF/XMP/ICC/GPS
    image: Image.Image                     # decoded sanitized image (RGB)
    info: dict[str, Any]


def _encode_jpeg(img: Image.Image, quality: int, subsampling: int = 0) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    return buf.getvalue()


def ingest(data: bytes, limits: Limits = Limits(), contract: PreprocessContract = PreprocessContract()) -> Ingested:
    """Validate size/format/pixels before a full decode, decode once, orient, and strip all metadata."""
    t0 = time.perf_counter()
    if not data:
        raise ImageRejected("empty_upload", "The uploaded file is empty.", 400)
    if len(data) > limits.max_bytes:
        raise ImageRejected("payload_too_large", f"Image exceeds {limits.max_bytes} bytes.", 413)
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        w, h = probe.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageRejected("unsupported_media", f"Not a decodable image ({exc.__class__.__name__}).", 415) from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected("unsupported_media", f"Image format {fmt} is not accepted.", 415)
    if w * h > limits.max_decoded_pixels:
        raise ImageRejected("too_many_pixels", f"Decoded image would have {w * h} pixels.", 413)
    if min(w, h) < limits.min_side:
        raise ImageRejected("image_too_small", f"Image side below {limits.min_side}px.", 422)
    try:
        probe.verify()  # structural check; the probe object is unusable afterwards
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ImageRejected("corrupt_image", f"Image failed to decode ({exc.__class__.__name__}).", 422) from exc

    exif_orientation = None
    had_exif = False
    with contextlib.suppress(Exception):
        exif = img.getexif()
        had_exif = len(exif) > 0
        exif_orientation = exif.get(0x0112)
    had_gps = False
    with contextlib.suppress(Exception):
        had_gps = bool(img.getexif().get_ifd(0x8825))
    mode = img.mode
    oriented = ImageOps.exif_transpose(img).convert("RGB")
    sanitized = _encode_jpeg(oriented, contract.sanitize_quality)
    image = Image.open(io.BytesIO(sanitized)).convert("RGB")
    info = {
        "format": fmt,
        "mode": mode,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "original_width": w,
        "original_height": h,
        "exif_orientation": exif_orientation,
        "had_exif": had_exif,
        "had_gps": had_gps,
        "width": image.width,
        "height": image.height,
        "sanitized_bytes": len(sanitized),
        "sanitized_sha256": sha256_bytes(sanitized),
        "ingest_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return Ingested(sanitized=sanitized, image=image, info=info)


def canonicalize(img: Image.Image, contract: PreprocessContract = PreprocessContract()) -> tuple[bytes, dict]:
    """Downscale (never upscale) to the canonical long side and re-encode without metadata."""
    w, h = img.size
    scale = min(1.0, contract.canonical_max_side / max(w, h))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(size, getattr(Image.Resampling, contract.canonical_resample))
    data = _encode_jpeg(img, contract.canonical_quality, contract.canonical_subsampling)
    return data, {"canonical_width": img.width, "canonical_height": img.height, "scale": scale}


def canonical_from_sanitized(sanitized: bytes, contract: PreprocessContract = PreprocessContract()):
    with Image.open(io.BytesIO(sanitized)) as im:
        return canonicalize(im.convert("RGB"), contract)


def load_canonical(data: bytes) -> np.ndarray:
    """Decode canonical bytes to contiguous RGB uint8 HWC (PIL decoder in every path)."""
    with Image.open(io.BytesIO(data)) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def letterbox(rgb: np.ndarray, contract: PreprocessContract = PreprocessContract()) -> tuple[np.ndarray, dict]:
    s = contract.imgsz
    ih, iw = rgb.shape[:2]
    r = min(s / ih, s / iw)
    h, w = round(ih * r), round(iw * r)
    top, left = round((s - h) / 2 - 0.1), round((s - w) / 2 - 0.1)
    out = np.full((s, s, 3), contract.pad_value, dtype=np.uint8)
    out[top : top + h, left : left + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, {"ratio": r, "resized_w": w, "resized_h": h, "top": top, "left": left, "size": s}


def to_tensor(lb: np.ndarray, contract: PreprocessContract = PreprocessContract()) -> np.ndarray:
    x = lb.astype(np.float32) * np.float32(contract.scale)
    mean = np.asarray(contract.mean, dtype=np.float32)
    std = np.asarray(contract.std, dtype=np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # 1x3xSxS


def tensor_from_canonical(data: bytes, contract: PreprocessContract = PreprocessContract()) -> np.ndarray:
    lb, _ = letterbox(load_canonical(data), contract)
    return to_tensor(lb, contract)


def canonical_from_raw(
    raw: bytes, contract: PreprocessContract = PreprocessContract(), limits: Limits = Limits()
) -> tuple[bytes, Ingested]:
    ing = ingest(raw, limits, contract)
    canonical, _ = canonical_from_sanitized(ing.sanitized, contract)
    return canonical, ing


def decode_to_tensor(
    image_bytes: bytes, contract: PreprocessContract = PreprocessContract(), limits: Limits = Limits()
) -> np.ndarray:
    """Full serving path from raw upload bytes (used by tests and the prediction adapter)."""
    canonical, _ = canonical_from_raw(image_bytes, contract, limits)
    return tensor_from_canonical(canonical, contract)


# ---------------------------------------------------------------------------------------------------------
# Visual stage trace for the UI: every intermediate the model path actually produced, with real timings.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Stage:
    index: int
    name: str
    title: str
    image: Image.Image | None
    params: dict[str, Any]
    duration_ms: float
    log: list[str]


def thumb(img: Image.Image, side: int = 640) -> Image.Image:
    img = img.copy()
    img.thumbnail((side, side), Image.Resampling.LANCZOS)
    return img


def _letterbox_display(lb: np.ndarray, meta: dict, scale: int = 2) -> Image.Image:
    """Letterboxed tensor input, enlarged, with the padding tinted so the preserved full frame is visible."""
    disp = lb.copy()
    mask = np.ones(disp.shape[:2], dtype=bool)
    mask[meta["top"] : meta["top"] + meta["resized_h"], meta["left"] : meta["left"] + meta["resized_w"]] = False
    tint = np.array([167, 139, 250], dtype=np.float32)  # lavender = padding, never model evidence
    disp[mask] = (0.55 * disp[mask] + 0.45 * tint).astype(np.uint8)
    im = Image.fromarray(disp)
    return im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)


def _tensor_planes(x: np.ndarray) -> Image.Image:
    """R, G, B planes of the NCHW tensor rendered side by side (grayscale per channel)."""
    planes = [(np.clip(x[0, c], 0, 1) * 255).astype(np.uint8) for c in range(3)]
    gap = np.full((x.shape[2], 6), 255, dtype=np.uint8)
    strip = np.concatenate([planes[0], gap, planes[1], gap, planes[2]], axis=1)
    rgb = np.stack([strip] * 3, axis=-1)
    w = strip.shape[1]
    bar = np.full((8, w, 3), 255, dtype=np.uint8)
    s = x.shape[3]
    for c, col in enumerate([(239, 68, 68), (34, 197, 94), (59, 130, 246)]):
        start = c * (s + 6)
        bar[:, start : start + s] = col
    im = Image.fromarray(np.concatenate([bar, rgb], axis=0))
    return im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST)


def run_stages(
    sanitized: bytes, ingest_info: dict[str, Any], contract: PreprocessContract = PreprocessContract()
) -> dict[str, Any]:
    """Re-run the serving preprocessing from the sanitized image, keeping each intermediate and its timing.

    Stage 1 reports the ingest step the API already performed (its timing is the API's measurement).
    """
    stages: list[Stage] = []
    with Image.open(io.BytesIO(sanitized)) as im:
        sanitized_img = im.convert("RGB")
    orient = ingest_info.get("exif_orientation")
    stages.append(
        Stage(
            1,
            "ingest",
            "Ingest: validate, orient, strip metadata",
            thumb(sanitized_img),
            {k: ingest_info.get(k) for k in ("format", "mode", "bytes", "original_width", "original_height",
                                             "exif_orientation", "had_exif", "had_gps", "width", "height")},
            float(ingest_info.get("ingest_ms", 0.0)),
            [
                f"format={ingest_info.get('format')} mode={ingest_info.get('mode')} bytes={ingest_info.get('bytes')}",
                f"stored {ingest_info.get('original_width')}x{ingest_info.get('original_height')}; structure verified"
                " before full decode (decompression-bomb guard)",
                f"EXIF orientation={orient if orient is not None else 'absent'}"
                + (" -> transposed" if orient not in (None, 1) else " -> no rotation needed")
                + f"; metadata removed (EXIF={'yes' if ingest_info.get('had_exif') else 'no'},"
                f" GPS={'yes' if ingest_info.get('had_gps') else 'no'})",
            ],
        )
    )

    t0 = time.perf_counter()
    canonical, cmeta = canonicalize(sanitized_img, contract)
    canonical_rgb = load_canonical(canonical)
    t1 = time.perf_counter()
    stages.append(
        Stage(
            2,
            "canonical",
            "Canonical derivative",
            thumb(Image.fromarray(canonical_rgb)),
            {**cmeta, "format": contract.canonical_format, "quality": contract.canonical_quality,
             "bytes": len(canonical)},
            (t1 - t0) * 1000,
            [
                f"long side <= {contract.canonical_max_side}px (scale {cmeta['scale']:.3f}, {contract.canonical_resample})",
                f"{cmeta['canonical_width']}x{cmeta['canonical_height']} {contract.canonical_format} "
                f"q{contract.canonical_quality} 4:4:4 -> {len(canonical)} bytes",
                "identical code path to the training set's canonical images",
            ],
        )
    )

    t2 = time.perf_counter()
    lb, lmeta = letterbox(canonical_rgb, contract)
    t3 = time.perf_counter()
    pad_frac = 1 - (lmeta["resized_w"] * lmeta["resized_h"]) / (contract.imgsz**2)
    stages.append(
        Stage(
            3,
            "letterbox",
            f"Letterbox {contract.imgsz}x{contract.imgsz}",
            _letterbox_display(lb, lmeta),
            {**lmeta, "pad_value": contract.pad_value, "pad_fraction": round(pad_frac, 4)},
            (t3 - t2) * 1000,
            [
                f"aspect preserved: {canonical_rgb.shape[1]}x{canonical_rgb.shape[0]} -> "
                f"{lmeta['resized_w']}x{lmeta['resized_h']} (r={lmeta['ratio']:.4f})",
                f"padding {pad_frac * 100:.1f}% with value {contract.pad_value} (tinted lavender in this view)",
                "no centre crop: tyre shoulders and beads stay in frame",
            ],
        )
    )

    t4 = time.perf_counter()
    x = to_tensor(lb, contract)
    t5 = time.perf_counter()
    stats = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "channel_mean": [round(float(x[0, c].mean()), 4) for c in range(3)],
        "channel_std": [round(float(x[0, c].std()), 4) for c in range(3)],
        "contiguous": bool(x.flags["C_CONTIGUOUS"]),
    }
    stages.append(
        Stage(
            4,
            "tensor",
            "Model tensor (R | G | B)",
            _tensor_planes(x),
            stats,
            (t5 - t4) * 1000,
            [
                f"tensor {stats['shape']} {stats['dtype']} range [{stats['min']:.3f}, {stats['max']:.3f}]",
                "channel means R/G/B = " + " / ".join(f"{m:.3f}" for m in stats["channel_mean"]),
                f"scale 1/255, mean {list(contract.mean)}, std {list(contract.std)} (contract {contract.version})",
            ],
        )
    )
    return {
        "canonical_bytes": canonical,
        "canonical_rgb": canonical_rgb,
        "letterboxed": lb,
        "letterbox_meta": lmeta,
        "tensor": x,
        "stages": stages,
    }
