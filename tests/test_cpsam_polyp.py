from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

import models.cpsam_polyp as cpsam
from models.base import validate_and_normalize_params


def test_provisional_provider_is_in_ready_catalog():
    info = cpsam.CellposePolyp.model_info
    assert info.status == "ready"
    assert info.tags["status"] == "ready"
    assert info.tags["maturity"] == "provisional"


def test_reference_tile_grid_and_half_open_centroid_ownership():
    tiles = cpsam.tile_grid(640, 1280)
    assert [(tile.x0, tile.core) for tile in tiles] == [
        (0, (0, 0, 576, 640)),
        (512, (576, 0, 896, 640)),
        (640, (896, 0, 1280, 640)),
    ]

    masks = np.zeros((2, 640, 640), dtype=bool)
    masks[0, 100, 575] = True
    masks[1, 100, 576] = True
    owned, x, _y = cpsam.centre_owned(masks, tiles[0])
    assert owned.tolist() == [True, False]
    assert x.tolist() == pytest.approx([575.0, 576.0])

    shifted = np.zeros((1, 640, 640), dtype=bool)
    shifted[0, 100, 64] = True  # native x=576 in the right tile
    right_owned, _, _ = cpsam.centre_owned(shifted, tiles[1])
    assert right_owned.tolist() == [True]


def test_roi_rle_round_trip_and_exact_tile_selection():
    roi = np.zeros((640, 1280), dtype=bool)
    roi[20:80, 30:100] = True
    encoded = cpsam.native_rle(roi[:, :640], 0, 0, 640, 1280)
    decoded = cpsam.decode_parent_regions([encoded], 640, 1280)
    assert np.array_equal(decoded, np.pad(roi[:, :640], ((0, 0), (0, 640))))
    tiles = cpsam.tile_grid(640, 1280)
    assert cpsam.select_tiles(tiles, decoded) == [tiles[0]]
    assert cpsam.decode_parent_regions([], 640, 1280) is None


def test_parent_region_requires_native_dimensions():
    with pytest.raises(ValueError, match="native size"):
        cpsam.decode_parent_regions([{"size": [10, 10], "counts": ""}], 640, 640)


def test_trained_position_grid_and_strict_checkpoint_loading(tmp_path):
    class TinyCPSAM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = 8
            self.encoder = torch.nn.Module()
            self.encoder.pos_embed = torch.nn.Parameter(torch.zeros(1, 32, 32, 2))
            self.head = torch.nn.Linear(2, 1)
            self.W2 = torch.nn.Parameter(torch.zeros(1))

    net = TinyCPSAM()
    cpsam.resize_cpsam_positional_embedding(net)
    assert tuple(net.encoder.pos_embed.shape) == (1, 80, 80, 2)

    expected = {key: value.clone() for key, value in net.state_dict().items()}
    checkpoint = tmp_path / "complete.pt"
    torch.save(expected, checkpoint)
    cpsam.strict_load_cpsam_checkpoint(net, checkpoint)

    missing_key = dict(expected)
    missing_key.pop("head.bias")
    incomplete = tmp_path / "incomplete.pt"
    torch.save(missing_key, incomplete)
    with pytest.raises(RuntimeError, match="strictly match"):
        cpsam.strict_load_cpsam_checkpoint(net, incomplete)


def test_eval_adapter_preserves_full_640_reference_arguments():
    cellpose_models = pytest.importorskip("cellpose.models")
    supported_eval_kwargs = set(inspect.signature(cellpose_models.CellposeModel.eval).parameters)
    supported_eval_kwargs -= {"self", "x"}

    class FakeCellpose:
        backbone = "sam_vitl"

        def eval(self, image, **kwargs):
            assert image.shape == (640, 640, 3)
            self.kwargs = kwargs
            labels = np.zeros((640, 640), dtype=np.int32)
            logits = np.zeros((640, 640), dtype=np.float32)
            return labels, [None, None, logits], None

    model = FakeCellpose()
    labels, logits = cpsam.eval_full_640(model, np.zeros((640, 640, 3), np.uint8))
    assert labels.shape == logits.shape == (640, 640)
    assert model.backbone == "sam_vitl"
    assert model.kwargs["bsize"] == 640
    assert model.kwargs["tile_overlap"] == pytest.approx(0.1)
    assert model.kwargs["batch_size"] == 1
    assert model.kwargs["channel_axis"] == 2
    assert model.kwargs["cellprob_threshold"] == 0.0
    assert model.kwargs["flow_threshold"] == 0.4
    assert model.kwargs["min_size"] == 15
    assert model.kwargs["diameter"] is None
    assert model.kwargs["augment"] is False
    assert "invert" not in model.kwargs
    assert set(model.kwargs) <= supported_eval_kwargs


def test_pinned_cpsam_has_uniform_bfloat16_after_full_module_cast():
    cellpose_vit = pytest.importorskip("cellpose.vit")
    net = cellpose_vit.CPSAM(dtype=torch.bfloat16)
    cpsam.resize_cpsam_positional_embedding(net)
    assert tuple(net.encoder.pos_embed.shape[1:3]) == (80, 80)

    cpsam._prepare_cpsam_for_inference(net, torch.device("cpu"), torch.bfloat16)

    assert all(parameter.dtype == torch.bfloat16 for parameter in net.parameters())
    assert all(buffer.dtype == torch.bfloat16 for buffer in net.buffers() if buffer.is_floating_point())
    assert net.encoder.patch_embed.proj.bias.dtype == torch.bfloat16


def test_provider_applies_roi_score_label_and_preserves_raw_masks(monkeypatch):
    model = cpsam.CellposePolyp()
    model._cellpose_model = SimpleNamespace(
        device="cpu", net=SimpleNamespace(dtype=torch.float32)
    )
    model._checkpoint_sha256 = cpsam.EXPECTED_CHECKPOINT_SHA256
    model._cellpose_version = cpsam.CELLPOSE_VERSION

    labels = np.zeros((640, 640), dtype=np.int32)
    labels[30:50, 30:50] = 3
    labels[100:120, 100:120] = 9
    logits = np.zeros((640, 640), dtype=np.float32)
    monkeypatch.setattr(cpsam, "eval_full_640", lambda _model, _image: (labels, logits))

    roi = np.zeros((640, 640), dtype=bool)
    roi[20:70, 20:70] = True
    encoded_roi = cpsam.native_rle(roi, 0, 0, 640, 640)
    request = SimpleNamespace(
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        model_registry_key=cpsam.MODEL_KEY,
        label=SimpleNamespace(id=781),
    )

    response = model.segment_instances(request, {"parent_regions": [encoded_roi], "threshold": 0.4})
    assert response["prediction_metadata"]["roi_mode"] == "parent_regions"
    instances = response["prediction_metadata"]["instances"]
    assert len(instances) == 1
    assert instances[0]["cellpose_label"] == 3
    assert instances[0]["score"] == pytest.approx(0.5)
    assert instances[0]["rle"]["size"] == [640, 640]
    decoded = mask_utils.decode(
        {"size": instances[0]["rle"]["size"], "counts": instances[0]["rle"]["counts"].encode("ascii")}
    )
    assert decoded[35, 35] == 1 and decoded[105, 105] == 0
    assert response["contours"]
    assert all(item["label_id"] == 781 for item in response["contours"])
    assert {item["source_instance_id"] for item in response["contours"]} == {
        instances[0]["source_instance_id"]
    }
    # Contours keep the shared normalized coordinate schema; exact native pixels live in the RLE.
    assert all(0.0 <= min(item["x"]) <= max(item["x"]) <= 1.0 for item in response["contours"])

    filtered = model.segment_instances(request, {"parent_regions": [encoded_roi], "threshold": 0.6})
    assert filtered["contours"] == []
    assert filtered["prediction_metadata"]["instances"] == []


def test_second_tile_contours_are_in_native_image_coordinates(monkeypatch):
    model = cpsam.CellposePolyp()
    model._cellpose_model = SimpleNamespace(
        device="cpu", net=SimpleNamespace(dtype=torch.float32)
    )
    model._checkpoint_sha256 = cpsam.EXPECTED_CHECKPOINT_SHA256
    model._cellpose_version = cpsam.CELLPOSE_VERSION

    labels = np.zeros((640, 640), dtype=np.int32)
    labels[200:220, 200:220] = 3
    logits = np.zeros((640, 640), dtype=np.float32)
    monkeypatch.setattr(cpsam, "eval_full_640", lambda _model, _image: (labels, logits))
    request = SimpleNamespace(
        image=np.zeros((640, 1280, 3), dtype=np.uint8),
        model_registry_key=cpsam.MODEL_KEY,
        label=SimpleNamespace(id=781),
    )

    response = model.segment_instances(request)
    instances = response["prediction_metadata"]["instances"]
    assert len(instances) == 2
    second = next(item for item in instances if ":t0001:" in item["source_instance_id"])
    second_contours = [
        item for item in response["contours"]
        if item["source_instance_id"] == second["source_instance_id"]
    ]
    assert second_contours
    assert min(min(item["x"]) for item in second_contours) == pytest.approx(712 / 1280)
    assert max(max(item["x"]) for item in second_contours) == pytest.approx(731 / 1280)
    assert min(min(item["y"]) for item in second_contours) == pytest.approx(200 / 640)
    assert max(max(item["y"]) for item in second_contours) == pytest.approx(219 / 640)
    assert second_contours[0]["path"] == "M 712 200 L 712 219 L 731 219 L 731 200 Z"

    native = mask_utils.decode(
        {"size": second["rle"]["size"], "counts": second["rle"]["counts"].encode("ascii")}
    )
    assert native[205, 712] == 1
    assert native[205, 200] == 0


def test_passthrough_parameters_are_explicit_and_legacy_provider_strictness_remains():
    from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
    from iquana_toolbox.schemas.training import HyperParameter

    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.05)],
    )
    assert validate_and_normalize_params(
        contract,
        {"threshold": 0.4, "parent_regions": []},
        passthrough_params=("parent_regions",),
    ) == {"parent_regions": [], "threshold": 0.4}
    with pytest.raises(ValueError, match="Unknown parameter 'parent_regions'"):
        validate_and_normalize_params(contract, {"parent_regions": []})


def test_checkpoint_path_and_digest_fail_without_falling_back_to_stock(monkeypatch, tmp_path):
    model = cpsam.CellposePolyp()
    monkeypatch.delenv(cpsam.CHECKPOINT_SHA256_ENV, raising=False)
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv(cpsam.CHECKPOINT_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match=cpsam.CHECKPOINT_ENV):
        model._load_weights()

    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the approved Cellpose-SAM weights")
    monkeypatch.setenv(cpsam.CHECKPOINT_ENV, str(wrong))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model._load_weights()
    assert model._cellpose_model is None

    configured_sha = "0" * 64
    monkeypatch.setenv(cpsam.CHECKPOINT_SHA256_ENV, configured_sha)
    with pytest.raises(RuntimeError, match=configured_sha):
        model._load_weights()


def test_checkpoint_sha_configuration_and_immutable_model_cache_identity(monkeypatch, tmp_path):
    monkeypatch.delenv(cpsam.CHECKPOINT_SHA256_ENV, raising=False)
    assert cpsam.configured_checkpoint_sha256() == cpsam.EXPECTED_CHECKPOINT_SHA256

    configured_sha = "A1" * 32
    monkeypatch.setenv(cpsam.CHECKPOINT_SHA256_ENV, configured_sha)
    assert cpsam.configured_checkpoint_sha256() == configured_sha.lower()
    for invalid_sha in ("a" * 63, "g" * 64, " "):
        monkeypatch.setenv(cpsam.CHECKPOINT_SHA256_ENV, invalid_sha)
        with pytest.raises(RuntimeError, match="exactly 64 hexadecimal characters"):
            cpsam.configured_checkpoint_sha256()

    checkpoint = tmp_path / "replacement.pt"
    checkpoint.write_bytes(b"configured replacement checkpoint")
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv(cpsam.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(cpsam.CHECKPOINT_SHA256_ENV, actual_sha.upper())
    monkeypatch.setenv(cpsam.DEVICE_ENV, "cpu")
    monkeypatch.setattr(
        cpsam,
        "_cellpose_eval_model",
        lambda path, device: (
            SimpleNamespace(device=device, net=SimpleNamespace(dtype=torch.float32)),
            cpsam.CELLPOSE_VERSION,
        ),
    )

    model = cpsam.CellposePolyp()
    model._load_weights()
    assert model._checkpoint_sha256 == actual_sha
    assert model._model_cache_identity == (str(checkpoint.resolve()), actual_sha, "cpu")

    monkeypatch.setattr(
        cpsam,
        "eval_full_640",
        lambda *_args: (
            np.zeros((640, 640), dtype=np.int32),
            np.zeros((640, 640), dtype=np.float32),
        ),
    )
    response = model.segment_instances(
        SimpleNamespace(
            image=np.zeros((640, 640, 3), dtype=np.uint8),
            model_registry_key=cpsam.MODEL_KEY,
            label=SimpleNamespace(id=781),
        )
    )
    assert response["prediction_metadata"]["checkpoint_sha256"] == actual_sha

    monkeypatch.setenv(cpsam.CHECKPOINT_SHA256_ENV, "b" * 64)
    with pytest.raises(RuntimeError, match="restart the worker"):
        model._load_weights()


@pytest.mark.asyncio
async def test_instance_seg_routes_keep_legacy_lists_and_forward_prediction_metadata(monkeypatch):
    from app.routes.instance_seg import inference, run_inference
    from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

    request = InstanceSegmentationRequest(
        image_url="http://example.com/image.png",
        user_id="test-user",
        model_registry_key=cpsam.MODEL_KEY,
    )
    legacy = [{"id": "old-contour"}]
    response_box = {"value": legacy}

    class FakeProvider:
        def predict(self, _request, _params):
            return response_box["value"]

    monkeypatch.setattr("app.routes.instance_seg.validate_model", lambda _request: None)
    monkeypatch.setattr(
        "app.routes.instance_seg.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: FakeProvider(),
    )
    assert await inference(request) == legacy
    assert (await run_inference(request))["result"] == legacy

    metadata = {"instances": [{"source_instance_id": "mask-1", "score": 0.9}]}
    response_box["value"] = {"contours": legacy, "prediction_metadata": metadata}
    assert await inference(request) == {"contours": legacy, "prediction_metadata": metadata}
    run_response = await run_inference(request)
    assert run_response["result"] == legacy
    assert run_response["prediction_metadata"] == metadata


def test_predict_recovers_passthrough_parameters_from_request_object():
    from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

    model = cpsam.CellposePolyp()
    model._ensure_weights = lambda: None
    captured_params = {}

    def fake_segment_instances(request, params=None):
        nonlocal captured_params
        captured_params = dict(params or {})
        return {"contours": [], "prediction_metadata": {}}

    model.segment_instances = fake_segment_instances

    req = InstanceSegmentationRequest(
        image_url="http://example.com/image.png",
        user_id="test",
        model_registry_key=cpsam.MODEL_KEY,
        parameters={"threshold": 0.05, "parent_regions": [{"test": "roi"}]},
    )
    # MLflow passes params with parent_regions sanitized away
    model.predict(None, req, params={"task": "instance-segmentation", "threshold": 0.05})
    assert "parent_regions" in captured_params
    assert captured_params["parent_regions"] == [{"test": "roi"}]
    assert captured_params["threshold"] == 0.05

