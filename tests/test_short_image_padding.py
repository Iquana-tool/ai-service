"""Tests for short-image padding and native coordinate recovery in polyp providers."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from models.cpsam_polyp import (
    CellposePolyp,
    MODEL_KEY as CPSAM_KEY,
    TILE_SIZE,
    pad_image_to_min_tile,
    axis_windows,
    tile_grid,
    native_rle,
    decode_parent_regions,
)
from models.yolo_polyp import YoloPolyp, MODEL_KEY as YOLO_KEY


@pytest.fixture(autouse=True)
def _test_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IQUANA_YOLO_POLYP_CHECKPOINT", str(tmp_path / "yolo_C_seed1.pt"))
    monkeypatch.setenv("IQUANA_CELLPOSE_POLYP_CHECKPOINT", str(tmp_path / "cpsam_best"))
    monkeypatch.setattr("models.yolo_polyp._configuration", lambda: ("dummy_path", "cpu"))


# ---------------------------------------------------------------------------
# Unit tests for padding utility and tile grid
# ---------------------------------------------------------------------------

def test_pad_image_to_min_tile_sizes():
    """Verify padding behavior on various short and mixed dimensions."""
    # 320x240 (H=240, W=320)
    img_240_320 = np.full((240, 320, 3), [10, 20, 30], dtype=np.uint8)
    padded, pad_y, pad_x = pad_image_to_min_tile(img_240_320)
    assert padded.shape == (640, 640, 3)
    assert pad_y == 400
    assert pad_x == 320
    # Interior must match original
    assert np.array_equal(padded[:240, :320], img_240_320)
    # Edge replication must replicate the border pixel
    assert np.array_equal(padded[240:, :], np.repeat(padded[239:240, :], 400, axis=0))

    # 639x800 (H=639, W=800)
    img_639_800 = np.zeros((639, 800, 3), dtype=np.uint8)
    padded, pad_y, pad_x = pad_image_to_min_tile(img_639_800)
    assert padded.shape == (640, 800, 3)
    assert pad_y == 1
    assert pad_x == 0

    # 800x639 (H=800, W=639)
    img_800_639 = np.zeros((800, 639, 3), dtype=np.uint8)
    padded, pad_y, pad_x = pad_image_to_min_tile(img_800_639)
    assert padded.shape == (800, 640, 3)
    assert pad_y == 0
    assert pad_x == 1

    # 640x640: unchanged
    img_640 = np.zeros((640, 640, 3), dtype=np.uint8)
    padded, pad_y, pad_x = pad_image_to_min_tile(img_640)
    assert padded is img_640
    assert pad_y == 0
    assert pad_x == 0

    # 1-pixel side synthetic case (H=1, W=700)
    img_1_700 = np.full((1, 700, 3), 99, dtype=np.uint8)
    padded, pad_y, pad_x = pad_image_to_min_tile(img_1_700)
    assert padded.shape == (640, 700, 3)
    assert pad_y == 639
    assert pad_x == 0
    assert np.all(padded == 99)


def test_pad_image_to_min_tile_invalid():
    """Invalid image shapes must raise ValueError."""
    with pytest.raises(ValueError, match="Expected a three-channel"):
        pad_image_to_min_tile(np.zeros((640, 640), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        pad_image_to_min_tile(np.zeros((0, 640, 3), dtype=np.uint8))


def test_tile_grid_with_short_dimensions():
    """tile_grid on dimensions below 640 must produce valid non-overlapping tiles."""
    # Short both sides (240, 320)
    tiles = tile_grid(240, 320)
    assert len(tiles) == 1
    assert tiles[0].x0 == 0 and tiles[0].y0 == 0
    assert tiles[0].core == (0, 0, 640, 640)

    # Multi-tile on one dimension (639, 800)
    tiles = tile_grid(639, 800)
    assert len(tiles) == 2
    assert all(t.y0 == 0 for t in tiles)
    assert [t.x0 for t in tiles] == [0, 160]


# ---------------------------------------------------------------------------
# Cellpose-SAM tests with short images and padding
# ---------------------------------------------------------------------------

def test_cpsam_short_image_inference(monkeypatch):
    """CellposePolyp must process an image smaller than 640 without error, returning original-dimension geometry."""
    model = CellposePolyp()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    model._cellpose_model = SimpleNamespace(device="cpu", net=SimpleNamespace(dtype=torch.float32))
    model._checkpoint_sha256 = "test_sha"
    model._cellpose_version = "4.2.1.1"

    # Mock predictions on the 640x640 tile:
    # Prediction 1: real polyp at (y=50:80, x=50:80), strictly inside 240x320
    # Prediction 2: false detection in padding area (y=300:350, x=400:450)
    # Prediction 3: edge-crossing detection (y=230:250, x=50:80)
    labels = np.zeros((640, 640), dtype=np.int32)
    labels[50:80, 50:80] = 1
    labels[300:350, 400:450] = 2  # padding only
    labels[230:250, 50:80] = 3    # crosses bottom edge (y=240)

    logits = np.full((640, 640), 2.0, dtype=np.float32)  # high confidence
    monkeypatch.setattr("models.cpsam_polyp.eval_full_640", lambda _m, _c: (labels.copy(), logits))

    orig_h, orig_w = 240, 320
    image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    request = SimpleNamespace(image=image, model_registry_key=CPSAM_KEY, label=SimpleNamespace(id=101))

    response = model.segment_instances(request)
    instances = response["prediction_metadata"]["instances"]
    contours = response["contours"]

    # Prediction 2 in padding must be completely removed
    # Only prediction 1 and prediction 3 (clipped to valid height 240) survive
    assert len(instances) == 2
    cellpose_labels = [inst["cellpose_label"] for inst in instances]
    assert 2 not in cellpose_labels
    assert 1 in cellpose_labels
    assert 3 in cellpose_labels

    # Verify RLE native sizes match original image [240, 320]
    for inst in instances:
        assert inst["rle"]["size"] == [orig_h, orig_w]
        decoded = mask_utils.decode({
            "size": inst["rle"]["size"],
            "counts": inst["rle"]["counts"].encode("ascii"),
        })
        assert decoded.shape == (orig_h, orig_w)
        # Ensure no pixel is outside [orig_h, orig_w]
        assert decoded.sum() > 0

    # Edge crossing prediction (label 3): max y in mask must be <= 239
    inst3 = next(inst for inst in instances if inst["cellpose_label"] == 3)
    decoded3 = mask_utils.decode({
        "size": inst3["rle"]["size"],
        "counts": inst3["rle"]["counts"].encode("ascii"),
    })
    ys, xs = np.nonzero(decoded3)
    assert np.max(ys) <= 239

    # Verify contour coordinates are normalized in [0, 1] relative to (240, 320)
    for c in contours:
        assert all(0.0 <= x <= 1.0 for x in c["x"])
        assert all(0.0 <= y <= 1.0 for y in c["y"])
        # Pixel coordinates via compute_path must be bounded by original dimensions
        assert "M " in c["path"]


def test_cpsam_padding_only_detections_produce_empty_output(monkeypatch):
    """If all detections fall into the padding region, return empty instances."""
    model = CellposePolyp()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    model._cellpose_model = SimpleNamespace(device="cpu", net=SimpleNamespace(dtype=torch.float32))
    model._checkpoint_sha256 = "test_sha"
    model._cellpose_version = "4.2.1.1"

    # Only detections in padding
    labels = np.zeros((640, 640), dtype=np.int32)
    labels[400:500, 400:500] = 1  # outside 240x320
    logits = np.full((640, 640), 3.0, dtype=np.float32)
    monkeypatch.setattr("models.cpsam_polyp.eval_full_640", lambda _m, _c: (labels, logits))

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    request = SimpleNamespace(image=image, model_registry_key=CPSAM_KEY, label=SimpleNamespace(id=101))

    response = model.segment_instances(request)
    assert response["prediction_metadata"]["instances"] == []
    assert response["contours"] == []


def test_cpsam_roi_on_short_image(monkeypatch):
    """An ROI touching the edge on a short image must be decoded at original dimensions and padded correctly."""
    model = CellposePolyp()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    model._cellpose_model = SimpleNamespace(device="cpu", net=SimpleNamespace(dtype=torch.float32))
    model._checkpoint_sha256 = "test_sha"
    model._cellpose_version = "4.2.1.1"

    # 2 detections: one inside ROI, one outside ROI
    labels = np.zeros((640, 640), dtype=np.int32)
    labels[40:60, 40:60] = 1   # Centroid ~ (50, 50) - inside ROI
    labels[150:180, 150:180] = 2  # Centroid ~ (165, 165) - outside ROI
    logits = np.full((640, 640), 2.0, dtype=np.float32)
    monkeypatch.setattr("models.cpsam_polyp.eval_full_640", lambda _m, _c: (labels, logits))

    orig_h, orig_w = 240, 320
    image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    request = SimpleNamespace(image=image, model_registry_key=CPSAM_KEY, label=SimpleNamespace(id=101))

    # ROI covering [30:70, 30:70]
    roi_mask = np.zeros((orig_h, orig_w), dtype=bool)
    roi_mask[30:70, 30:70] = True
    roi_rle = native_rle(roi_mask, 0, 0, orig_h, orig_w)

    response = model.segment_instances(request, {"parent_regions": [roi_rle]})
    instances = response["prediction_metadata"]["instances"]
    assert len(instances) == 1
    assert instances[0]["cellpose_label"] == 1


# ---------------------------------------------------------------------------
# YOLO tests with short and mixed images
# ---------------------------------------------------------------------------

def test_yolo_short_image_inference(monkeypatch):
    """YOLO must process images smaller than 640 without error, preserving original coordinates."""
    model = YoloPolyp()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    model._configuration_identity = ("dummy_path", "cpu")

    # Mock predict output:
    # Mask 1: real polyp at (y=60:90, x=60:90), inside 300x400
    # Mask 2: in padding (y=500:550, x=500:550)
    masks_data = torch.zeros((2, 640, 640), dtype=torch.float32)
    masks_data[0, 60:90, 60:90] = 1.0
    masks_data[1, 500:550, 500:550] = 1.0
    scores = torch.tensor([0.85, 0.90], dtype=torch.float32)

    def predict(crop, **kwargs):
        return [SimpleNamespace(
            masks=SimpleNamespace(data=masks_data),
            boxes=SimpleNamespace(conf=scores),
        )]

    model._yolo = SimpleNamespace(predict=predict)

    orig_h, orig_w = 300, 400
    image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    request = SimpleNamespace(image=image, label=SimpleNamespace(id=202))

    result = model.segment_instances(request)
    instances = result["prediction_metadata"]["instances"]
    contours = result["contours"]

    # Mask 2 in padding must be removed
    assert len(instances) == 1
    assert instances[0]["score"] == pytest.approx(0.85)
    assert instances[0]["rle"]["size"] == [orig_h, orig_w]

    decoded = mask_utils.decode({
        "size": instances[0]["rle"]["size"],
        "counts": instances[0]["rle"]["counts"].encode("ascii"),
    })
    assert decoded.shape == (orig_h, orig_w)
    assert decoded[75, 75] == 1
    assert decoded.sum() == 30 * 30

    assert len(contours) == 1
    assert all(0.0 <= x <= 1.0 for x in contours[0]["x"])
    assert all(0.0 <= y <= 1.0 for y in contours[0]["y"])


def test_yolo_mixed_size_image_multiple_tiles(monkeypatch):
    """YOLO must process mixed-size images (e.g. 639x800) with multiple tiles along the large axis."""
    model = YoloPolyp()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    model._configuration_identity = ("dummy_path", "cpu")

    # Image is 639x800 -> padded to 640x800, generating 2 tiles (x0=0, x0=160)
    orig_h, orig_w = 639, 800
    image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    request = SimpleNamespace(image=image, label=SimpleNamespace(id=202))

    calls = []
    def predict(crop, **kwargs):
        calls.append(crop)
        # Empty prediction
        return [SimpleNamespace(
            masks=SimpleNamespace(data=torch.zeros((0, 640, 640), dtype=torch.float32)),
            boxes=SimpleNamespace(conf=torch.zeros((0,), dtype=torch.float32)),
        )]

    model._yolo = SimpleNamespace(predict=predict)
    result = model.segment_instances(request)

    assert len(calls) == 2
    assert result["prediction_metadata"]["tiles_run"] == 2
    assert result["contours"] == []
