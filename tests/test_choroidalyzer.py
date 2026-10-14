"""Choroidalyzer: network layout, declared label space, and its output end to end.

No weights are downloaded: the network is checked against the released state dict's
shape, and inference runs on stubbed probability maps. The semantic -> instance step
itself is covered in test_semantic.py.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from models.choroidalyzer import Choroidalyzer, _ChoroidUNet


def _blob(probs: np.ndarray, y0: int, y1: int, x0: int, x1: int, value: float) -> None:
    probs[y0:y1, x0:x1] = value


def _request(tmp_path) -> InstanceSegmentationRequest:
    path = tmp_path / "bscan.png"
    cv2.imwrite(str(path), np.zeros((64, 96), dtype=np.uint8))
    return InstanceSegmentationRequest(
        image_url=str(path), user_id="tester", model_registry_key="choroidalyzer",
    )


@pytest.fixture
def probs() -> np.ndarray:
    """Fixed network output: a choroid band, two vessels inside it, a weak fovea."""
    probs = np.zeros((3, 64, 96), dtype=np.float32)
    _blob(probs[0], 20, 50, 0, 96, 0.9)
    _blob(probs[1], 25, 32, 10, 20, 0.8)
    _blob(probs[1], 35, 45, 50, 70, 0.7)
    _blob(probs[2], 5, 15, 40, 50, 0.3)
    return probs


@pytest.fixture
def model(monkeypatch, probs) -> Choroidalyzer:
    instance = Choroidalyzer()
    monkeypatch.setattr(instance, "_ensure_weights", lambda: None)
    monkeypatch.setattr(instance, "class_scores", lambda image: probs)
    return instance


def test_network_matches_released_weights_layout():
    state = _ChoroidUNet().state_dict()
    assert len(state) == 226
    trainable = sum(v.numel() for k, v in state.items() if "running" not in k and "num_batches" not in k)
    assert trainable == 788_283
    assert tuple(state["in_conv.conv.weight"].shape) == (8, 1, 3, 3)
    assert tuple(state["out_conv.1.weight"].shape) == (3, 8, 1, 1)


def test_scores_pad_any_scan_size(monkeypatch):
    instance = Choroidalyzer()
    instance._net = _ChoroidUNet().eval()
    monkeypatch.setattr(instance, "_device", lambda: torch.device("cpu"))
    probs = instance.class_scores(np.zeros((496, 200, 3), dtype=np.uint8))
    assert probs.shape == (3, 496, 200)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_label_space_is_declared():
    info = Choroidalyzer.model_info
    assert [(c.key, c.name, c.part_of) for c in info.classes] == [
        ("region", "Choroid region", None),
        ("vessel", "Choroidal vessels", "region"),
        ("fovea", "Fovea", None),
    ]
    assert info.tags["task_instance_segmentation"] == "true"
    (contract,) = info.input_contracts
    assert [p.key for p in contract.parameters] == [
        "region_threshold", "vessel_threshold", "fovea_threshold", "min_area",
    ]
    assert contract.parameters[2].default_value == 0.1  # upstream's fovea threshold


def test_vessels_arrive_inside_the_choroid_and_every_object_names_its_class(model, tmp_path):
    region, fovea = model.predict(None, _request(tmp_path), {"task": "instance-segmentation"})
    assert (region.model_class, fovea.model_class) == ("region", "fovea")
    assert [v.model_class for v in region.children] == ["vessel", "vessel"]
    assert fovea.confidence == pytest.approx(0.3)  # the heatmap's peak
    assert all(c.label_id is None for c in [region, fovea, *region.children])


def test_threshold_parameter_applies_per_class(model, tmp_path):
    (region, _) = model.predict(
        None, _request(tmp_path), {"task": "instance-segmentation", "vessel_threshold": 0.75},
    )
    assert len(region.children) == 1
