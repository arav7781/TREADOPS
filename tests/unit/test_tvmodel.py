"""The torchvision model family: checkpoints are told apart from YOLO's, load safely, and ONNX matches torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from treadops_ml import tvmodel  # noqa: E402
from treadops_ml.common import repo_path  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = tvmodel.build("convnext_tiny")
    path = tmp_path_factory.mktemp("tv") / "best.pt"
    torch.save({"family": tvmodel.FAMILY, "arch": "convnext_tiny", "class_names": dict(enumerate(tvmodel.CLASSES)),
                "state_dict": model.inner.state_dict()}, path)
    return path


def test_a_torchvision_checkpoint_is_recognised_and_loads(checkpoint):
    assert tvmodel.is_checkpoint(checkpoint)
    model, names = tvmodel.load(checkpoint)
    assert names == {0: "defective", 1: "good"} and not model.training
    p = tvmodel.raw_probs(model, np.random.default_rng(0).random((3, 3, 224, 224), dtype=np.float32))
    assert p.shape == (3, 2) and np.allclose(p.sum(axis=1), 1.0)


def test_yolo_and_foreign_files_are_not_torchvision_checkpoints(tmp_path):
    junk = tmp_path / "x.pt"
    junk.write_bytes(b"not a checkpoint")
    assert not tvmodel.is_checkpoint(junk)
    yolo = repo_path("models/pretrained/yolo26n-cls.pt")
    if yolo.exists():  # needs Ultralytics' pickled classes, so weights_only loading must refuse it
        assert not tvmodel.is_checkpoint(yolo)


def test_onnx_export_matches_torch(checkpoint, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, _ = tvmodel.load(checkpoint)
    onnx = tvmodel.export_onnx(model, tmp_path / "model.onnx", 224)
    x = np.random.default_rng(1).random((2, 3, 224, 224), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert np.abs(got - tvmodel.raw_probs(model, x)).max() < 1e-4
