from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from models import yolo_polyp as yolo


def fake_model(monkeypatch, masks, scores):
    model = yolo.YoloPolyp()
    calls = []

    def predict(crop, **kwargs):
        calls.append((crop.copy(), kwargs))
        tile_masks = masks[len(calls) - 1] if isinstance(masks, list) else masks
        return [SimpleNamespace(masks=SimpleNamespace(data=torch.tensor(tile_masks)),
                                boxes=SimpleNamespace(conf=torch.tensor(scores)))]

    model._yolo = SimpleNamespace(predict=predict)
    model._configuration_identity = yolo._configuration()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    return model, calls


def test_native_masks_scores_roi_and_reference_settings(monkeypatch):
    masks = np.zeros((2, 640, 640), dtype=np.float32)
    masks[0, 30:50, 30:50] = 1
    masks[1, 100:120, 100:120] = 1
    model, calls = fake_model(monkeypatch, masks, [.8, .9])
    parent = np.zeros((640, 640), dtype=bool)
    parent[20:70, 20:70] = True
    roi = yolo.native_rle(parent, 0, 0, 640, 640)
    image = np.full((640, 640, 3), [10, 20, 30], dtype=np.uint8)
    request = SimpleNamespace(image=image, label=SimpleNamespace(id=781))
    result = model.segment_instances(request, {"parent_regions": [roi]})
    assert result["contours"][0]["label_id"] == 781
    assert result["contours"][0]["added_by"] == yolo.MODEL_KEY
    assert 0 < max(result["contours"][0]["x"]) < 1
    instances = result["prediction_metadata"]["instances"]
    assert len(instances) == 1
    assert instances[0]["score"] == pytest.approx(.8)
    assert np.array_equal(mask_utils.decode(instances[0]["rle"]), masks[0])
    assert calls[0][0][0, 0].tolist() == [10, 20, 30]  # Ultralytics receives BGR.
    assert calls[0][1] == dict(imgsz=640, conf=.001, max_det=1000, iou=.7,
                              retina_masks=True, device=yolo._configuration()[1],
                              half=False, augment=False, verbose=False, save=False)
    assert len(model.segment_instances(request)["contours"]) == 2
    assert model.segment_instances(request, {"threshold": .95})["contours"] == []


def test_native_tile_ownership_does_not_duplicate_overlap(monkeypatch):
    masks = np.zeros((1, 640, 640), dtype=np.float32)
    masks[0, 100:120, 560:600] = 1
    # The same global object appears at x=560 in the first crop and x=48 in the second.
    model, calls = fake_model(monkeypatch, [masks, np.roll(masks, -512, axis=2)], [.9])
    result = model.segment_instances(SimpleNamespace(
        image=np.zeros((640, 1152, 3), dtype=np.uint8), label=SimpleNamespace(id=43)))
    assert len(calls) == 2
    assert result["prediction_metadata"]["tiles_run"] == 2
    assert len(result["contours"]) == 1
    decoded = mask_utils.decode(result["prediction_metadata"]["instances"][0]["rle"])
    assert decoded[100:120, 560:600].all()
    assert decoded.sum() == 800


def test_missing_or_mismatched_artifacts_never_load_stock_weights(tmp_path, monkeypatch):
    path = tmp_path / "missing.pt"
    monkeypatch.setenv(yolo.CHECKPOINT_ENV, str(path))
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        yolo.YoloPolyp()._load_weights()
    path.write_bytes(b"wrong checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        yolo.YoloPolyp()._load_weights()


def test_invalid_threshold_and_geometry_are_rejected(monkeypatch):
    model, _ = fake_model(monkeypatch, np.zeros((0, 640, 640)), [])
    request = SimpleNamespace(image=np.zeros((640, 640, 3)), label=SimpleNamespace(id=43))
    with pytest.raises(ValueError, match="finite"):
        model.segment_instances(request, {"threshold": float("nan")})
    request.image = np.zeros((32, 32, 3))
    with pytest.raises(ValueError, match="smaller"):
        model.segment_instances(request)


def test_provider_declares_polyp_mapping_and_roi_contract():
    info = yolo.YoloPolyp.model_info
    assert info.registry_key == yolo.MODEL_KEY
    assert info.classes[0].key == "polyp"
    assert info.input_contracts[0].parameters[0].default_value == .25
    assert yolo.YoloPolyp._passthrough_input_params == ("parent_regions",)


def test_post_merge_suppression_removes_nested_and_overlapping_duplicates_without_deleting_adjacent(monkeypatch):
    # 4 masks in a 640x640 tile:
    # Mask 0: outer box at [50:100, 50:100], area = 2500, score 0.8
    # Mask 1: nested inner box at [60:85, 60:85], area = 625 (100% contained in mask 0), score 0.5 -> should be suppressed
    # Mask 2: overlapping box at [55:105, 50:100], area = 2500, IoU = 0.82 with mask 0, score 0.4 -> should be suppressed
    # Mask 3: separate adjacent box at [200:230, 200:230], score 0.75 -> separate polyp -> must be kept
    masks = np.zeros((4, 640, 640), dtype=np.float32)
    masks[0, 50:100, 50:100] = 1.0
    masks[1, 60:85, 60:85] = 1.0
    masks[2, 55:105, 50:100] = 1.0
    masks[3, 200:230, 200:230] = 1.0
    scores = [0.8, 0.5, 0.4, 0.75]

    model, _ = fake_model(monkeypatch, masks, scores)
    request = SimpleNamespace(image=np.zeros((640, 640, 3), dtype=np.uint8), label=SimpleNamespace(id=46))
    res = model.segment_instances(request, {"threshold": 0.25})

    meta = res["prediction_metadata"]
    assert meta["candidates_raw"] == 4
    assert meta["candidates_kept"] == 2
    assert len(res["contours"]) == 2

    # Kept instances should be mask 0 (score 0.8) and mask 3 (score 0.75)
    kept_scores = [inst["score"] for inst in meta["instances"]]
    assert kept_scores == pytest.approx([0.8, 0.75])

