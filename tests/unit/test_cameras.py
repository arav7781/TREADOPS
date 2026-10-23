"""Camera registry helpers and the connector's capture rule (no network, no database)."""

import numpy as np
import pytest

from treadops_api import cameras as svc
from treadops_api.errors import ApiError
from treadops_worker.cameras import frame_metrics, scene_changed, should_capture


def test_stream_urls_are_masked_and_round_trip_encrypted():
    url = "rtsp://admin:hunter2@10.0.0.5:554/stream1?subtype=0"
    assert svc.mask_url(url) == "rtsp://admin:•••@10.0.0.5:554/stream1?subtype=0"
    assert svc.mask_url("rtsp://10.0.0.5/s1") == "rtsp://10.0.0.5/s1"
    token = svc.encrypt_url(url)
    assert "hunter2" not in token and svc.decrypt_url(token) == url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://cam/x", "rtsp://", "javascript:alert(1)"])
def test_only_network_stream_urls_are_accepted(url, monkeypatch):
    monkeypatch.delenv("CAMERA_ALLOW_FILE_URLS", raising=False)
    with pytest.raises(ApiError) as e:
        svc.validate_url(url)
    assert e.value.code == "invalid_stream_url"


def test_names_become_stable_codes():
    assert svc.slug("Line 1 · Intake (east)") == "line-1-intake-east"


CAM = {"capture_mode": "on_change", "min_sharpness": 15.0, "change_threshold": 0.04}


def _frame(seed: int, flat: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if flat:
        return np.full((240, 320, 3), 128, np.uint8)
    return (rng.random((240, 320, 3)) * 255).astype(np.uint8)


def test_capture_rule_needs_sharp_well_exposed_changed_scenes():
    thumb, sharp, bright = frame_metrics(_frame(1))
    assert should_capture(CAM, sharp, bright, None, thumb) == (True, "captured")
    assert should_capture(CAM, sharp, bright, thumb, thumb)[1] == "scene unchanged since the last capture"
    assert should_capture({**CAM, "capture_mode": "interval"}, sharp, bright, thumb, thumb)[0]
    flat_thumb, flat_sharp, flat_bright = frame_metrics(_frame(0, flat=True))
    assert "too blurred" in should_capture(CAM, flat_sharp, flat_bright, None, flat_thumb)[1]
    dark = (_frame(2) // 40).astype(np.uint8)
    d_thumb, d_sharp, d_bright = frame_metrics(dark)
    assert "badly exposed" in should_capture({**CAM, "min_sharpness": 0}, d_sharp, d_bright, None, d_thumb)[1]


def test_scene_change_compares_downscaled_frames():
    a, _, _ = frame_metrics(_frame(3))
    b, _, _ = frame_metrics(_frame(4))
    assert scene_changed(None, a, 0.04) and not scene_changed(a, a, 0.04) and scene_changed(a, b, 0.04)
