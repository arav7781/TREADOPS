"""Full-image Ultralytics classification trainer/validator built on the documented extension points.

`ClassificationTrainer.build_dataset` / `get_validator` and `ClassificationValidator.build_dataset` are
overridden so that training, validation and (via `preprocess`) serving all decode canonical bytes with PIL and
letterbox with the shared contract. Training augmentations never crop: rotation expands the canvas first.
"""

from __future__ import annotations

import math
import random
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

from .preprocess import PreprocessContract, letterbox, load_canonical, to_tensor


@dataclass(frozen=True)
class AugmentConfig:
    hflip: float = 0.5
    vflip: float = 0.0
    rotate_deg: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    # Robustness (probabilities; 0 = off, so older recipes train exactly as before): the conditions a line camera
    # or a phone in a workshop actually meets, and where the stress tests showed the model was weak.
    low_light: float = 0.0   # darken to 35-75 % with a gamma curve (dim bays, shadows under a vehicle)
    blur: float = 0.0        # Gaussian blur, sigma 0.5-2.0 (motion, focus)
    jpeg: float = 0.0        # recompress at quality 30-70 (messaging apps, cheap cameras)


class EvalTransform:
    """canonical RGB uint8 HWC -> letterboxed float32 CHW tensor in [0, 1]."""

    def __init__(self, contract: PreprocessContract | None = None):
        self.contract = contract or PreprocessContract()

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        lb, _ = letterbox(rgb, self.contract)
        return torch.from_numpy(to_tensor(lb, self.contract)[0])

    def __repr__(self) -> str:
        return f"EvalTransform(preprocess={self.contract.version}, sha={self.contract.sha256[:12]})"


def _rotate_expand(rgb: np.ndarray, deg: float, fill: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgb, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(fill, fill, fill))


class TrainTransform(EvalTransform):
    """Approved, non-destructive augmentations applied before the shared letterbox."""

    def __init__(self, contract: PreprocessContract | None = None, aug: AugmentConfig | None = None):
        super().__init__(contract)
        self.aug = aug or AugmentConfig()

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        a = self.aug
        if a.hflip and random.random() < a.hflip:
            rgb = rgb[:, ::-1]
        if a.vflip and random.random() < a.vflip:
            rgb = rgb[::-1]
        if a.rotate_deg:
            rgb = _rotate_expand(np.ascontiguousarray(rgb), random.uniform(-a.rotate_deg, a.rotate_deg),
                                 self.contract.pad_value)
        if a.blur and random.random() < a.blur:
            rgb = cv2.GaussianBlur(np.ascontiguousarray(rgb), (0, 0), random.uniform(0.5, 2.0))
        if a.jpeg and random.random() < a.jpeg:
            ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(rgb)[:, :, ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 70)])
            if ok:
                rgb = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
        x = np.ascontiguousarray(rgb).astype(np.float32)
        if a.low_light and random.random() < a.low_light:
            x = 255.0 * (x / 255.0) ** random.uniform(1.2, 1.8) * random.uniform(0.35, 0.75)
        if a.brightness:
            x *= random.uniform(1 - a.brightness, 1 + a.brightness)
        if a.contrast:
            mean = x.mean()
            x = (x - mean) * random.uniform(1 - a.contrast, 1 + a.contrast) + mean
        if a.saturation:
            gray = x.mean(axis=2, keepdims=True)
            x = (x - gray) * random.uniform(1 - a.saturation, 1 + a.saturation) + gray
        return super().__call__(np.clip(x, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return f"TrainTransform({asdict(self.aug)}, preprocess={self.contract.version})"


class FullImageDataset(ClassificationDataset):
    """ImageFolder dataset that decodes canonical bytes with PIL and applies the shared full-image transform."""

    augment_config: AugmentConfig = AugmentConfig()

    def __init__(self, root, args, augment=False, prefix="", names=None):
        super().__init__(root, args, augment=augment, prefix=prefix, names=names)
        contract = PreprocessContract(imgsz=int(args.imgsz))
        self.torch_transforms = TrainTransform(contract, self.augment_config) if augment else EvalTransform(contract)

    def __getitem__(self, i: int) -> dict:
        f, j, _fn, _im = self.samples[i]
        rgb = load_canonical(Path(f).read_bytes())
        return {"img": self.torch_transforms(rgb), "cls": j}


class FullImageValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return FullImageDataset(img_path, self.args, augment=False, prefix=self.args.split, names=self.names)


class FullImageTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return FullImageDataset(
            img_path,
            self.args,
            augment=mode == "train",
            prefix="train" if mode == "train" else self.args.split,
            names=self.data["names"],
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return FullImageValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
