"""Multi-source data: extra sources never move a primary image, robustness augmentations do what they say, and the
Roboflow key never reaches an error message. All on synthetic inputs."""

import random

import httpx
import numpy as np
import pandas as pd
import pytest

from treadops_ml import acquire
from treadops_ml.split import ROLES, assign_roles
from treadops_ml.transforms import AugmentConfig, EvalTransform, TrainTransform

CFG = {"seed": 42, "outer": {"train": 0.70, "val": 0.15, "test": 0.15}, "dev_val_fraction": 0.15,
       "calibration_fraction": 0.50}


def _rows(source: str, n_groups: int, prefix: str, start: int = 0) -> list[dict]:
    rows = []
    for g in range(n_groups):
        for k in range(1 + g % 3):  # groups of 1-3 photos of one tyre
            i = start + g * 3 + k
            rows.append({"image_id": f"{prefix}-{i}", "group_id": f"{prefix}-g{g}", "source_id": source,
                         "label": "defective" if g % 2 else "good", "original_sha256": f"{prefix}{i:060d}",
                         "status": "ok", "group_label_conflict": False})
    return rows


def test_adding_a_source_never_moves_a_primary_image():
    d1 = pd.DataFrame(_rows("D1:v1", 300, "d1"))
    extra = _rows("X:v1", 200, "x")
    # three extra photos are re-shot primary tyres: they join a primary group and must follow its role. All are
    # labelled "good", so d1-g7 (defective) gets a disagreeing member: prepare quarantines those, but the split
    # must not depend on it
    attached = [{**_rows("X:v1", 1, "xa", 900 + i)[0], "group_id": f"d1-g{7 * i}"} for i in range(3)]
    alone, _ = assign_roles(d1, CFG, "D1:v1")
    both, report = assign_roles(pd.concat([d1, pd.DataFrame(extra + attached)]), CFG, "D1:v1")

    before = alone.set_index("image_id").role
    after = both[both.source_id == "D1:v1"].set_index("image_id").role
    assert after.reindex(before.index).equals(before)
    assert not any(report["leakage"].values())
    assert set(both[both.source_id == "X:v1"].role) == set(ROLES)  # the extra source gets its own full carve
    role_of_group = alone.groupby("group_id").role.first()
    for row in attached:
        assert both.set_index("image_id").role[row["image_id"]] == role_of_group[row["group_id"]]


def _checker(n: int = 96) -> np.ndarray:
    tile = (np.indices((n, n)).sum(axis=0) // 8) % 2
    return np.repeat((60 + 140 * tile).astype(np.uint8)[:, :, None], 3, axis=2)


def _only(**kw) -> AugmentConfig:
    return AugmentConfig(hflip=0, rotate_deg=0, brightness=0, contrast=0, saturation=0, **kw)


def test_robustness_augmentations_are_off_by_default():
    rgb = _checker()
    assert AugmentConfig().low_light == AugmentConfig().blur == AugmentConfig().jpeg == 0
    assert np.allclose(TrainTransform(aug=_only())(rgb), EvalTransform()(rgb))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_low_light_darkens_blur_softens_and_jpeg_changes_pixels(seed):
    random.seed(seed)
    rgb, base = _checker(), EvalTransform()(_checker())
    dark = TrainTransform(aug=_only(low_light=1.0))(rgb)
    assert dark.mean() < 0.8 * base.mean()
    soft = TrainTransform(aug=_only(blur=1.0))(rgb)
    assert soft.std() < base.std()
    assert not np.allclose(TrainTransform(aug=_only(jpeg=1.0))(rgb), base)


def test_roboflow_errors_never_carry_the_api_key(tmp_path, monkeypatch):
    secret = "rf-test-key-0123456789"
    monkeypatch.setenv("ROBOFLOW_API_KEY", secret)
    cfg = tmp_path / "damage.yaml"
    cfg.write_text(f"sources:\n  - id: 'RF-X:v1'\n    url: https://universe.roboflow.com/ws/proj/dataset/1\n"
                   f"    license: CC BY 4.0\n    archive: {tmp_path}/x.zip\n    archive_sha256: '{'0' * 64}'\n")
    real = httpx.Client
    monkeypatch.setattr(acquire.httpx, "Client", lambda **kw: real(
        transport=httpx.MockTransport(lambda req: httpx.Response(401, request=req)), **kw))
    with pytest.raises(RuntimeError) as err:
        acquire.fetch_damage_sources(str(cfg), str(tmp_path / "receipts"))
    assert "HTTP 401" in str(err.value) and secret not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__  # no chained httpx error with the URL
    assert not (tmp_path / "x.zip").exists()
