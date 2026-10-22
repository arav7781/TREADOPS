"""The single full-image transform contract shared by prepare, training, evaluation, serving and ONNX parity.

    raw bytes -> validate + decode (PIL) -> EXIF transpose -> RGB -> canonical JPEG (long side <= 1024)
              -> decode canonical -> letterbox 224 (aspect preserved, pad 114) -> float32 NCHW in [0, 1]

Every model path starts from the *canonical bytes*: the prepare stage stores them for the public archive and the
inference worker stores them for uploads, so training and serving decode exactly the same JPEG.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import sha256_json

# PIL's own bomb guard is replaced by an explicit pixel check before any full decode.
Image.MAX_IMAGE_PIXELS = None

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class PreprocessContract:
    version: str = "v1"
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
        known = {f for f in cls.__dataclass_fields__}
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
class Decoded:
    image: Image.Image                    # oriented RGB
    raw: Image.Image                      # as stored in the file (no EXIF transpose), RGB
    info: dict[str, Any] = field(default_factory=dict)


def decode_validated(data: bytes, limits: Limits = Limits()) -> Decoded:
    """Validate size/format/pixels before a full decode, then decode once and apply EXIF orientation."""
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
    with contextlib.suppress(Exception):
        exif_orientation = img.getexif().get(0x0112)
    raw_rgb = img.convert("RGB")
    oriented = ImageOps.exif_transpose(img).convert("RGB")
    info = {
        "format": fmt,
        "mode": img.mode,
        "original_width": w,
        "original_height": h,
        "exif_orientation": exif_orientation,
        "oriented_width": oriented.width,
        "oriented_height": oriented.height,
        "bytes": len(data),
    }
    return Decoded(image=oriented, raw=raw_rgb, info=info)


def canonicalize(img: Image.Image, contract: PreprocessContract = PreprocessContract()) -> tuple[bytes, dict]:
    """Downscale (never upscale) to the canonical long side and re-encode without metadata."""
    w, h = img.size
    scale = min(1.0, contract.canonical_max_side / max(w, h))
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(size, getattr(Image.Resampling, contract.canonical_resample))
    buf = io.BytesIO()
    img.save(
        buf,
        format=contract.canonical_format,
        quality=contract.canonical_quality,
        subsampling=contract.canonical_subsampling,
        optimize=False,
    )
    return buf.getvalue(), {"canonical_width": img.width, "canonical_height": img.height, "scale": scale}


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


def decode_to_tensor(
    image_bytes: bytes, contract: PreprocessContract = PreprocessContract(), limits: Limits = Limits()
) -> np.ndarray:
    """Full serving path from raw upload bytes (used by tests and the prediction adapter)."""
    canonical, _ = canonicalize(decode_validated(image_bytes, limits).image, contract)
    return tensor_from_canonical(canonical, contract)


# ---------------------------------------------------------------------------------------------------------
# Visual stage trace for the UI: every intermediate the model path actually produced, with real timings.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Stage:
    index: int
    name: str
    title: str
    image: Image.Image
    params: dict[str, Any]
    duration_ms: float
    log: list[str]


def _thumb(img: Image.Image, side: int = 640) -> Image.Image:
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
    # colour cue bar per channel
    bar = np.full((8, w, 3), 255, dtype=np.uint8)
    s = x.shape[3]
    for c, col in enumerate([(239, 68, 68), (34, 197, 94), (59, 130, 246)]):
        start = c * (s + 6)
        bar[:, start : start + s] = col
    im = Image.fromarray(np.concatenate([bar, rgb], axis=0))
    return im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST)


def run_stages(
    image_bytes: bytes, contract: PreprocessContract = PreprocessContract(), limits: Limits = Limits()
) -> dict[str, Any]:
    """Run the serving preprocessing once, keeping each intermediate for the UI and lineage.

    Returns canonical bytes, the model tensor, descriptor inputs and a list of `Stage` records.
    Raises ImageRejected exactly like the serving path would.
    """
    stages: list[Stage] = []

    t0 = time.perf_counter()
    dec = decode_validated(image_bytes, limits)
    t1 = time.perf_counter()
    info = dec.info
    stages.append(
        Stage(
            1,
            "decode_validate",
            "Decode + validate",
            _thumb(dec.raw),
            info,
            (t1 - t0) * 1000,
            [
                f"format={info['format']} mode={info['mode']} bytes={info['bytes']}",
                f"stored size {info['original_width']}x{info['original_height']} "
                f"({info['original_width'] * info['original_height'] / 1e6:.1f} MP <= {limits.max_decoded_pixels / 1e6:.0f} MP)",
                "structure verified before full decode (decompression-bomb guard)",
            ],
        )
    )

    t2 = time.perf_counter()
    oriented = dec.image
    t3 = time.perf_counter()
    orient = info["exif_orientation"]
    stages.append(
        Stage(
            2,
            "exif_orient_rgb",
            "EXIF orientation + RGB",
            _thumb(oriented),
            {"exif_orientation": orient, "width": oriented.width, "height": oriented.height, "mode": "RGB"},
            (t3 - t2) * 1000 + 0.0,
            [
                f"EXIF orientation tag = {orient if orient is not None else 'absent'}"
                + (" -> transposed" if orient not in (None, 1) else " -> no rotation needed"),
                f"colour mode {info['mode']} -> RGB; metadata (EXIF/GPS) dropped from derivatives",
            ],
        )
    )

    t4 = time.perf_counter()
    canonical, cmeta = canonicalize(oriented, contract)
    canonical_rgb = load_canonical(canonical)
    t5 = time.perf_counter()
    stages.append(
        Stage(
            3,
            "canonical",
            "Canonical derivative",
            _thumb(Image.fromarray(canonical_rgb)),
            {**cmeta, "format": contract.canonical_format, "quality": contract.canonical_quality,
             "bytes": len(canonical)},
            (t5 - t4) * 1000,
            [
                f"long side <= {contract.canonical_max_side}px (scale {cmeta['scale']:.3f}, {contract.canonical_resample})",
                f"re-encoded {contract.canonical_format} q{contract.canonical_quality} 4:4:4 "
                f"-> {len(canonical)} bytes; same bytes the training set was built from",
            ],
        )
    )

    t6 = time.perf_counter()
    lb, lmeta = letterbox(canonical_rgb, contract)
    t7 = time.perf_counter()
    pad_frac = 1 - (lmeta["resized_w"] * lmeta["resized_h"]) / (contract.imgsz**2)
    stages.append(
        Stage(
            4,
            "letterbox",
            f"Letterbox {contract.imgsz}x{contract.imgsz}",
            _letterbox_display(lb, lmeta),
            {**lmeta, "pad_value": contract.pad_value, "pad_fraction": round(pad_frac, 4)},
            (t7 - t6) * 1000,
            [
                f"aspect preserved: {canonical_rgb.shape[1]}x{canonical_rgb.shape[0]} -> "
                f"{lmeta['resized_w']}x{lmeta['resized_h']} (r={lmeta['ratio']:.4f})",
                f"padding {pad_frac * 100:.1f}% with value {contract.pad_value} (tinted lavender in this view)",
                "no centre crop: tyre shoulders and beads stay in frame",
            ],
        )
    )

    t8 = time.perf_counter()
    x = to_tensor(lb, contract)
    t9 = time.perf_counter()
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
            5,
            "tensor",
            "Model tensor (R | G | B)",
            _tensor_planes(x),
            stats,
            (t9 - t8) * 1000,
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
        "decode_info": info,
        "stages": stages,
    }
