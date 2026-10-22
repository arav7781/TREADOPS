"""ML-02 transform geometry + API-01 input rejection, on tiny synthetic images."""

import io

import numpy as np
import pytest
from PIL import Image

from treadops_ml.preprocess import (
    ImageRejected,
    Limits,
    PreprocessContract,
    canonical_from_raw,
    ingest,
    letterbox,
    load_canonical,
    run_stages,
    tensor_from_canonical,
    to_tensor,
)


def _jpeg(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, **kw)
    return buf.getvalue()


def _border_image(w: int, h: int) -> np.ndarray:
    """Grey image with saturated markers on all four edges."""
    a = np.full((h, w, 3), 128, np.uint8)
    t = max(4, min(w, h) // 25)
    a[:t, :] = (255, 0, 0)        # top red
    a[-t:, :] = (0, 255, 0)       # bottom green
    a[:, :t] = (0, 0, 255)        # left blue
    a[:, -t:] = (255, 255, 0)     # right yellow
    return a


@pytest.mark.parametrize("imgsz", [224, 320])
@pytest.mark.parametrize("w,h", [(1200, 400), (400, 1200), (900, 900)])
def test_letterbox_keeps_all_four_edges_and_aspect(imgsz, w, h):
    c = PreprocessContract(imgsz=imgsz)
    lb, meta = letterbox(_border_image(w, h), c)
    assert lb.shape == (imgsz, imgsz, 3)
    top, left, rh, rw = meta["top"], meta["left"], meta["resized_h"], meta["resized_w"]
    # aspect ratio preserved within one pixel of rounding
    assert abs(rw / rh - w / h) < 0.02 * (w / h) + 1 / min(rh, rw)
    # each edge marker is still present inside the resized region (nothing cropped)
    assert lb[top + 1, left + rw // 2, 0] > 200 and lb[top + 1, left + rw // 2, 2] < 60          # red top
    assert lb[top + rh - 2, left + rw // 2, 1] > 200 and lb[top + rh - 2, left + rw // 2, 0] < 60  # green bottom
    assert lb[top + rh // 2, left + 1, 2] > 200                                                  # blue left
    assert lb[top + rh // 2, left + rw - 2, 0] > 200 and lb[top + rh // 2, left + rw - 2, 1] > 200  # yellow right
    # padding uses the contract value
    if top > 0:
        assert (lb[0, :, :] == c.pad_value).all()
    if left > 0:
        assert (lb[:, 0, :] == c.pad_value).all()


def test_tensor_layout_range_and_channel_order():
    a = np.zeros((50, 80, 3), np.uint8)
    a[..., 0] = 255  # pure red
    x = to_tensor(letterbox(a)[0])
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert x.min() >= 0.0 and x.max() <= 1.0
    centre = x[0, :, 112, 112]
    assert centre[0] == pytest.approx(1.0) and centre[1] == pytest.approx(0.0) and centre[2] == pytest.approx(0.0)


def test_exif_orientation_is_applied_and_metadata_stripped():
    img = Image.fromarray(_border_image(600, 300))  # landscape pixels
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display -> portrait
    raw = _jpeg(img, exif=exif.tobytes())
    ing = ingest(raw)
    assert ing.info["exif_orientation"] == 6 and ing.info["had_exif"]
    assert (ing.image.width, ing.image.height) == (300, 600)
    assert len(Image.open(io.BytesIO(ing.sanitized)).getexif()) == 0


def test_serving_and_training_paths_share_canonical_bytes():
    raw = _jpeg(Image.fromarray(_border_image(1500, 1000)))
    canonical, _ = canonical_from_raw(raw)
    ing = ingest(raw)
    staged = run_stages(ing.sanitized, ing.info)
    assert staged["canonical_bytes"] == canonical
    assert np.array_equal(staged["tensor"], tensor_from_canonical(canonical))
    assert max(load_canonical(canonical).shape[:2]) == 1024
    assert [s.name for s in staged["stages"]] == ["ingest", "canonical", "letterbox", "tensor"]


def test_canonical_never_upscales():
    raw = _jpeg(Image.fromarray(_border_image(300, 200)))
    canonical, _ = canonical_from_raw(raw)
    assert load_canonical(canonical).shape[:2] == (200, 300)


@pytest.mark.parametrize(
    "payload,code,status",
    [
        (b"", "empty_upload", 400),
        (b"not an image at all", "unsupported_media", 415),
        (_jpeg(Image.new("RGB", (32, 32))), "image_too_small", 422),
    ],
)
def test_rejections_are_typed(payload, code, status):
    with pytest.raises(ImageRejected) as e:
        ingest(payload)
    assert e.value.code == code and e.value.http_status == status


def test_oversized_payload_and_pixel_bomb_rejected_before_decode():
    raw = _jpeg(Image.new("RGB", (400, 400)))
    with pytest.raises(ImageRejected) as e:
        ingest(raw, Limits(max_bytes=100))
    assert e.value.code == "payload_too_large" and e.value.http_status == 413
    with pytest.raises(ImageRejected) as e:
        ingest(raw, Limits(max_decoded_pixels=1000))
    assert e.value.code == "too_many_pixels"


def test_truncated_jpeg_is_corrupt():
    raw = _jpeg(Image.fromarray(_border_image(400, 400)))
    with pytest.raises(ImageRejected) as e:
        ingest(raw[: len(raw) // 2])
    assert e.value.code in {"corrupt_image", "unsupported_media"}


def test_gif_is_unsupported():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="GIF")
    with pytest.raises(ImageRejected) as e:
        ingest(buf.getvalue())
    assert e.value.code == "unsupported_media" and e.value.http_status == 415


def test_contract_hash_is_stable_and_sensitive():
    assert PreprocessContract().sha256 == PreprocessContract.from_dict(PreprocessContract().to_dict()).sha256
    assert PreprocessContract(pad_value=0).sha256 != PreprocessContract().sha256
