"""Line-camera live view: inspection zone cropping and validation, the overlay, and the vision check's circuit
breaker (a dead provider must not stall a camera for ~50 s per frame, and a skipped check never auto-clears)."""

import numpy as np
import pytest

from treadops_api import cameras as svc
from treadops_api.errors import ApiError
from treadops_worker import overlay, vision_check


def test_roi_crop_and_validation():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert svc.crop_roi(frame, None) is frame
    assert svc.crop_roi(frame, [0.25, 0.1, 0.75, 0.9]).shape == (80, 100, 3)
    assert svc.validate_roi(None) is None and svc.validate_roi([0, 0, 1, 1]) == [0.0, 0.0, 1.0, 1.0]
    for bad in ([0.5, 0, 0.55, 1], [0, 0, 1], [0.6, 0, 0.2, 1], [-0.1, 0, 1, 1]):
        with pytest.raises(ApiError) as e:
            svc.validate_roi(bad)
        assert e.value.code == "invalid_roi"


def test_overlay_draws_the_zone_in_the_verdict_colour_and_the_boxes():
    frame = np.full((360, 640, 3), 40, np.uint8)
    roi = [0.25, 0.2, 0.75, 0.9]
    out = overlay.annotate(frame, roi=roi, result={"verdict": "damaged", "p_defective": 0.91},
                           boxes=[{"class": "cut", "confidence": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}], experimental=True,
                           camera="Line 1", line="L1", fps=25.0, model="convnext_tiny", simulated=True)
    assert out.shape == frame.shape and not np.array_equal(out, frame) and np.array_equal(frame, np.full_like(frame, 40))
    x0, y0, x1, y1 = overlay.zone_px(frame.shape, roi)
    assert tuple(out[(y0 + y1) // 2, x0]) == overlay.COLOURS["damaged"]  # zone edge in the verdict's colour
    zw, zh = x1 - x0, y1 - y0
    assert tuple(out[y0 + int(0.5 * zh), x0 + int(0.6 * zw)]) == overlay.BOX  # the box edge, mapped into the zone


def test_a_dead_vision_provider_is_skipped_for_a_minute_and_never_clears(monkeypatch):
    import openai

    calls = []

    class Dead:
        def __init__(self, **kw):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            calls.append(1)
            raise openai.APITimeoutError(request=None)  # type: ignore[arg-type]

    monkeypatch.setattr(openai, "OpenAI", Dead)
    monkeypatch.setattr(vision_check, "_reachable", lambda url: True)  # reachable, but the request times out
    monkeypatch.setattr(vision_check, "_breaker", {"open_until": 0.0, "reason": ""})
    rgb = np.zeros((64, 64, 3), np.uint8)
    first, second = vision_check.second_opinion(rgb), vision_check.second_opinion(rgb)
    assert len(calls) == 1 and "APITimeoutError" in first["error"] and second["error"].startswith("skipped")
    assert second["latency_ms"] == 0.0
    assert vision_check.decide("good_candidate", second)[0] == "manual_review"


def test_an_unreachable_provider_is_detected_by_a_quick_connect(monkeypatch):
    real_reachable = vision_check._reachable
    monkeypatch.setattr(vision_check, "_breaker", {"open_until": 0.0, "reason": ""})
    monkeypatch.setattr(vision_check, "_reachable", lambda url: False)
    out = vision_check.second_opinion(np.zeros((64, 64, 3), np.uint8))
    assert out["error"] == "provider unreachable (no connection)" and vision_check._breaker["open_until"] > 0
    assert vision_check.decide("good_candidate", out)[0] == "manual_review"
    assert not real_reachable("http://127.0.0.1:9", timeout=0.5)  # nothing listens on the discard port
