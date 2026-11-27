"""Hierarchy integration without gated weights, live storage, or model downloads."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.networking.http import services
from models.base import CapabilityModel, InstanceSegmentation, InstanceSuggestion
from app.hierarchy_inference import predict_concept


def binary(mask):
    return BinaryMask.from_numpy_array(mask)


@pytest.fixture
def scene(monkeypatch):
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(240, dtype=np.uint8)
    first, second = np.zeros(image.shape[:2], bool), np.zeros(image.shape[:2], bool)
    first[10:60, 10:80] = True
    second[65:115, 160:230] = True
    images = {"image": image, "external": np.full((30, 40, 3), 117, np.uint8)}
    monkeypatch.setattr(services, "get_image_from_url_cached", lambda url: images[url])
    return image, [first, second]


def request(task, scene, **fields):
    image, parents = scene
    classes = {"instance-segmentation": services.InstanceSegmentationRequest,
               "instance-suggestion": services.InstanceSuggestionRequest,
               "cross-image-suggestion": services.CrossImageSuggestionRequest}
    return classes[task](
        image_url="image", user_id="a", model_registry_key="strict-model",
        parameters={"hierarchy_regions": [{"parent_id": i + 101, "mask": binary(m).rle_mask}
                                           for i, m in enumerate(parents)]}, **fields,
    )


class StrictInstances(InstanceSegmentation, CapabilityModel):
    model_info = ModelInfo(registry_key="strict-model", name="Strict stub", description="Stub", usage_tip="Tests", input_contracts=[
        InputContract(task="instance-segmentation", conditioning=ConditioningSpec(kind="none", user_selectable_count=False), parameters=[]),
    ])

    def segment_instances(self, req, params):
        self.calls.append(req)
        # No service-only parameters may reach provider validation or its handler.
        assert params == {"task": "instance-segmentation"}
        assert req.parameters == params
        h, w = req.image.shape[:2]
        mask = np.zeros((h, w), bool)
        mask[10:20, 10:20] = True
        contour = Contour.from_binary_mask(mask, label_id=43, model_class="polyp", confidence=.87)
        payload = contour.model_dump()
        payload["source_instance_id"] = "same-local-id"
        return {"contours": [payload], "prediction_metadata": {
            "checkpoint_sha256": "verified-checkpoint", "tiles_run": 1,
            "instances": [{"source_instance_id": "same-local-id", "score": .87,
                           "rle": binary(mask).rle_mask, "cellpose_label": 7}],
        }}


def pyfunc(provider):
    provider.calls = []
    return SimpleNamespace(predict=lambda data, params: provider.predict(None, data, params))


def test_parent_crops_restore_geometry_and_native_rles_without_unknown_provider_params(scene):
    req = request("instance-segmentation", scene, label={
        "id": 43, "dataset_id": 1, "name": "Polyp", "value": 3})
    provider = StrictInstances()
    result = predict_concept(pyfunc(provider), req, {**req.parameters, "task": "instance-segmentation"})
    assert len(provider.calls) == 2
    assert all(r.image.shape[0] < 640 and r.image.shape[1] < 640 for r in provider.calls)
    assert {c["parent_id"] for c in result["contours"]} == {101, 102}
    for item, parent in zip(result["prediction_metadata"]["instances"], scene[1]):
        assert item["rle"]["size"] == [120, 240]
        assert not np.any(binary_from(item["rle"]) & ~parent)
        assert item["score"] == .87 and item["cellpose_label"] == 7
    assert len({c["source_instance_id"] for c in result["contours"]}) == 2
    assert result["prediction_metadata"]["checkpoint_sha256"] == "verified-checkpoint"
    assert result["prediction_metadata"]["tiles_run"] == 2
    assert {c["model_class"] for c in result["contours"]} == {"polyp"}
    assert all(c["label_id"] == 43 and c["confidence"] == .87 for c in result["contours"])
    left, right = result["contours"]
    assert max(left["x"]) < .5 < min(right["x"])
    assert "hierarchy_regions" in req.parameters  # the original request stays intact


def binary_from(rle):
    return BinaryMask(rle_mask=rle, height=rle["size"][0], width=rle["size"][1]).mask.astype(bool)


def test_suggestion_borrows_positive_and_negative_examples_across_parents(scene):
    positive, negative = np.zeros((120, 240), bool), np.zeros((120, 240), bool)
    positive[20:25, 20:25] = True
    negative[80:85, 180:185] = True
    req = request("instance-suggestion", scene, positive_exemplars=[binary(positive)],
                  negative_exemplars=[binary(negative)])
    calls = []

    def predict(data, params):
        local = data[0]
        calls.append(local)
        assert "hierarchy_regions" not in local.parameters
        assert "hierarchy_regions" not in params
        # Both reference prompts survive, even when on different parent objects.
        assert local.positive_exemplar_masks[0].sum() == 25
        assert local.negative_exemplar_masks[0].sum() == 25
        h, w = local.image.shape[:2]
        target = np.zeros((h, w), bool)
        target[10:20, 10:20] = True
        reference_echo = local.positive_exemplar_masks[0]
        return np.stack([target, reference_echo]), np.asarray([.9, .99])

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "instance-suggestion"})
    assert len(calls) == 2 and len(result["contours"]) == 2  # no reference echoes
    assert {c["parent_id"] for c in result["contours"]} == {101, 102}
    assert all(c["confidence"] == .9 for c in result["contours"])


def test_cross_image_keeps_external_exemplars_and_source_identity(scene):
    mask = np.zeros((30, 40), bool)
    mask[5:10, 15:20] = True
    req = request("cross-image-suggestion", scene, exemplars=[
        services.CrossImageExemplar(image_url="external", mask=binary(mask)),
    ])
    calls = []

    def predict(data, params):
        local = data[0]
        calls.append(local)
        assert local.exemplars[0].image_url == "external"
        assert np.array_equal(local.exemplars[0].exemplar_mask, mask)
        assert np.all(local.exemplars[0].image == 117)
        detected = np.zeros(local.image.shape[:2], bool)
        detected[10:20, 10:20] = True
        return np.asarray([detected]), np.asarray([.91])

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "cross-image-suggestion"})
    assert len(calls) == 2 and len(result["contours"]) == 2


def test_native_hierarchy_is_called_once_with_all_parents(scene):
    positive = scene[1][0].copy()
    req = request("instance-suggestion", scene, positive_exemplars=[binary(positive)])
    calls = []

    def predict(data, params):
        local = data[0]
        calls.append(local)
        assert len(local.parent_region_masks) == 2
        assert local.image.shape == scene[0].shape
        assert "hierarchy_regions" not in params and "hierarchy_regions" not in local.parameters
        masks = []
        for region in scene[1]:
            y, x = np.argwhere(region)[20]
            mask = np.zeros(region.shape, bool)
            mask[y:y + 5, x:x + 5] = True
            masks.append(mask)
        return np.stack(masks), np.asarray([.8, .9])

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "instance-suggestion"},
                             {"hierarchy_input": "parent_regions"})
    assert len(calls) == 1
    assert {c["parent_id"] for c in result["contours"]} == {101, 102}


@pytest.mark.parametrize("regions", [[], None, [{"parent_id": 101, "mask": {"size": [1, 1], "counts": "1"}}],
                                    [{"parent_id": "101", "mask": {}}]])
def test_invalid_hierarchy_never_falls_back_to_whole_image(scene, regions):
    req = request("instance-segmentation", scene)
    model = SimpleNamespace(predict=lambda *_: pytest.fail("invalid hierarchy dispatched"))
    with pytest.raises(ValueError):
        predict_concept(model, req, {"task": "instance-segmentation", "hierarchy_regions": regions})


def test_empty_parent_is_rejected(scene):
    req = request("instance-segmentation", scene)
    with pytest.raises(ValueError, match="must not be empty"):
        predict_concept(None, req, {"task": "instance-segmentation", "hierarchy_regions": [
            {"parent_id": 1, "mask": binary(np.zeros((120, 240), bool)).rle_mask},
        ]})


def test_whole_image_path_is_unchanged(scene):
    req = request("instance-segmentation", scene)
    req.parameters.clear()
    expected = object()
    def predict(data, params):
        assert data is req and params == {"task": "instance-segmentation"}
        return expected
    assert predict_concept(SimpleNamespace(predict=predict), req,
                           {"task": "instance-segmentation"}) is expected


def test_small_parent_crops_use_existing_cellpose_padding(scene, monkeypatch):
    import models.cpsam_polyp as cpsam
    model = cpsam.CellposePolyp()
    model._cellpose_model = SimpleNamespace(device="cpu", net=SimpleNamespace(dtype="float32"))
    model._checkpoint_sha256 = cpsam.EXPECTED_CHECKPOINT_SHA256
    model._cellpose_version = cpsam.CELLPOSE_VERSION
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)

    def evaluate(_model, image):
        assert image.shape == (640, 640, 3)
        mask = np.zeros((640, 640), np.int32)
        mask[10:20, 10:20] = 1
        mask[300:310, 300:310] = 2  # padded-only candidate must disappear
        return mask, np.zeros((640, 640), np.float32)

    monkeypatch.setattr(cpsam, "eval_full_640", evaluate)
    req = request("instance-segmentation", scene, label={
        "id": 43, "dataset_id": 1, "name": "Polyp", "value": 3})
    response = predict_concept(SimpleNamespace(predict=lambda data, params: model.predict(None, data, params)),
                               req, {**req.parameters, "task": "instance-segmentation"})
    assert len(response["contours"]) == 2
    assert all(i["rle"]["size"] == [120, 240] for i in response["prediction_metadata"]["instances"])


def test_routes_preserve_hierarchy_contours_metadata_and_translate_bad_regions(scene, monkeypatch):
    from fastapi import HTTPException
    from app.routes import cross_image, instance_seg, suggestion
    provider = StrictInstances()
    registry = SimpleNamespace(get_model_by_version=lambda *_: pyfunc(provider),
                               client=SimpleNamespace(get_registered_model=lambda *_: SimpleNamespace(tags={})))
    monkeypatch.setattr(instance_seg, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(instance_seg, "validate_model", lambda *_: None)
    req = request("instance-segmentation", scene)
    result = asyncio.run(instance_seg.run_inference(req))
    assert len(result["result"]) == 2 and result["prediction_metadata"]["hierarchy"]
    req.parameters["hierarchy_regions"] = []
    for module, route, task in [(instance_seg, instance_seg.run_inference, "instance-segmentation"),
                                (suggestion, suggestion.infer_instances, "instance-suggestion"),
                                (cross_image, cross_image.suggest_cross_image, "cross-image-suggestion")]:
        monkeypatch.setattr(module, "MODEL_REGISTRY", registry)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(route(req))
        assert exc.value.status_code == 422


def test_crop_detections_are_clipped_and_background_and_parent_masks_are_removed(scene):
    req = request("cross-image-suggestion", scene)
    req.parameters["hierarchy_regions"] = req.parameters["hierarchy_regions"][:1]

    def predict(data, params):
        h, w = data[0].image.shape[:2]
        masks = np.zeros((3, h, w), bool)
        masks[0, 1:8, 10:20] = True  # partly crosses the padded parent border
        masks[1, :2, :2] = True     # outside the parent
        masks[2, 3:53, 4:74] = True  # effectively resegments the parent
        return masks, np.asarray([.9, .8, .7])

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "cross-image-suggestion"})
    assert len(result["contours"]) == 1
    mask = binary_from(result["prediction_metadata"]["instances"][0]["rle"])
    assert mask.any() and not (mask & ~scene[1][0]).any()
    assert np.argwhere(mask)[:, 0].min() == 10


def test_sam3_prompts_use_correct_xy_axes_and_keep_negative_polarity(scene):
    import torch
    from unittest.mock import MagicMock
    from models.sam3 import SAM3
    positive, negative = np.zeros((120, 240), bool), np.zeros((120, 240), bool)
    positive[70:80, 150:160] = True
    negative[95:100, 180:190] = True
    req = request("instance-suggestion", scene, positive_exemplars=[binary(positive)],
                  negative_exemplars=[binary(negative)])
    model = SAM3()
    model.model, model.processor = MagicMock(), MagicMock()
    inputs = MagicMock()
    inputs.get.return_value = torch.tensor([[120, 240]])
    model.processor.return_value = inputs
    model.processor.post_process_instance_segmentation.return_value = [{
        "masks": torch.zeros((0, 120, 240)), "scores": torch.zeros(0),
    }]
    model.suggest_instances(req)
    args = model.processor.call_args.kwargs
    assert args["input_boxes"] == [[[150., 70., 160., 80.], [180., 95., 190., 100.]]]
    assert args["input_boxes_labels"].tolist() == [[1, 0]]


def test_native_sam3_declares_hierarchy_ownership():
    from models.sam3_hierarchical import SAM3Hierarchical
    assert SAM3Hierarchical.model_info.tags["hierarchy_input"] == "parent_regions"


def test_small_parent_crops_use_existing_yolo_padding(scene, monkeypatch, tmp_path):
    import torch
    from models import yolo_polyp as yolo
    monkeypatch.setenv(yolo.CHECKPOINT_ENV, str(tmp_path / "local-test.pt"))
    model = yolo.YoloPolyp()
    model._configuration_identity = yolo._configuration()
    monkeypatch.setattr(model, "_ensure_weights", lambda: None)
    calls = []

    def predict(image, **kwargs):
        assert image.shape == (640, 640, 3)
        calls.append(image.shape)
        masks = torch.zeros((2, 640, 640))
        masks[0, 10:20, 10:20] = 1
        masks[1, 300:310, 300:310] = 1  # padding-only prediction
        return [SimpleNamespace(masks=SimpleNamespace(data=masks),
                                boxes=SimpleNamespace(conf=torch.tensor([.9, .9])))]

    model._yolo = SimpleNamespace(predict=predict)
    req = request("instance-segmentation", scene, label={
        "id": 43, "dataset_id": 1, "name": "Polyp", "value": 3})
    response = predict_concept(SimpleNamespace(predict=lambda data, params: model.predict(None, data, params)),
                               req, {**req.parameters, "task": "instance-segmentation"})
    assert len(calls) == 2 and len(response["contours"]) == 2
    assert all(i["rle"]["size"] == [120, 240] for i in response["prediction_metadata"]["instances"])


@pytest.mark.parametrize("kind", ["none", "concept_text", "embeddings", "reference_images", "instances"])
def test_suggestion_without_spatial_examples_keeps_valid_contract_conditioning(scene, kind):
    if kind in {"none", "concept_text"}:
        spec = ConditioningSpec(kind=kind, user_selectable_count=False)
    elif kind == "embeddings":
        spec = ConditioningSpec(kind=kind, unit="vector", min_units=1,
                                embedding_kinds=["image_cls"])
    elif kind == "reference_images":
        spec = ConditioningSpec(kind=kind, unit="image", min_units=1)
    else:
        spec = ConditioningSpec(kind=kind, unit="instance", min_units=0)

    class StrictSuggestion(InstanceSuggestion, CapabilityModel):
        model_info = ModelInfo(
            registry_key="strict-model", name="Strict suggestion", description="Stub", usage_tip="Tests",
            input_contracts=[InputContract(task="instance-suggestion", conditioning=spec)],
        )

        def suggest_instances(self, req, params):
            self.calls.append(req)
            assert params == {"task": "instance-suggestion"}
            assert req.positive_exemplars == []
            assert req.concept.name == "A text concept"
            assert req.embeddings == {"image_cls": [1., 2.]}
            assert req.exemplar_embeddings == {"image_cls": [[3., 4.]]}
            assert req.exemplars[0].image_url == "external"
            assert np.all(req.exemplars[0].image == 117)
            mask = np.zeros(req.image.shape[:2], bool)
            mask[10:20, 10:20] = True
            return np.asarray([mask]), np.asarray([.9])

    req = request("instance-suggestion", scene, positive_exemplars=[],
                  concept={"id": 43, "dataset_id": 1, "value": 3, "name": "A text concept"},
                  embeddings={"image_cls": [1., 2.]}, exemplar_embeddings={"image_cls": [[3., 4.]]},
                  exemplars=[services.CrossImageExemplar(
                      image_url="external", mask=binary(np.ones((30, 40), bool)))])
    provider = StrictSuggestion()
    model = pyfunc(provider)
    whole = req.model_copy(update={"parameters": {}})
    normal = predict_concept(model, whole, {"task": "instance-suggestion"})
    hierarchical = predict_concept(model, req, {**req.parameters, "task": "instance-suggestion"})
    assert len(normal[0]) == 1 and len(hierarchical["contours"]) == 2
    assert len(provider.calls) == 3
    assert provider.calls[0].image.shape == scene[0].shape
    # No reference canvas for non-spatial conditioning: input remains the crop.
    assert all(c.image.shape[1] < 100 for c in provider.calls[1:])


def test_optional_suggestion_with_only_negative_examples_keeps_negative_context(scene):
    class OptionalSuggestion(InstanceSuggestion, CapabilityModel):
        model_info = ModelInfo(
            registry_key="strict-model", name="Optional", description="Stub", usage_tip="Tests",
            input_contracts=[InputContract(task="instance-suggestion", conditioning=ConditioningSpec(
                kind="instances", unit="instance", min_units=0))],
        )

        def suggest_instances(self, req, params):
            self.calls.append(req)
            assert req.positive_exemplars == []
            assert len(req.negative_exemplar_masks) == 1
            assert req.negative_exemplar_masks[0].sum() == 25
            mask = np.zeros(req.image.shape[:2], bool)
            mask[10:20, 10:20] = True
            return np.asarray([mask]), np.asarray([.9])

    negative = np.zeros(scene[0].shape[:2], bool)
    negative[80:85, 180:185] = True
    req = request("instance-suggestion", scene, positive_exemplars=[],
                  negative_exemplars=[binary(negative)])
    provider = OptionalSuggestion()
    model = pyfunc(provider)
    whole = req.model_copy(update={"parameters": {}})
    assert len(predict_concept(model, whole, {"task": "instance-suggestion"})[0]) == 1
    result = predict_concept(model, req, {**req.parameters, "task": "instance-suggestion"})
    assert len(result["contours"]) == 2 and len(provider.calls) == 3


def test_nested_provider_children_restore_coordinates_and_keep_each_model_class(scene):
    req = request("instance-segmentation", scene)
    req.parameters["hierarchy_regions"] = req.parameters["hierarchy_regions"][1:]
    child_crop_mask = None

    def predict(local, params):
        nonlocal child_crop_mask
        hw = local.image.shape[:2]
        outer = np.zeros(hw, bool)
        outer[8:36, 8:36] = True
        child_crop_mask = np.zeros(hw, bool)
        child_crop_mask[12:18, 12:18] = True
        child = Contour.from_binary_mask(child_crop_mask, model_class="child", label_id=44, confidence=.91)
        root = Contour.from_binary_mask(outer, model_class="outer", label_id=43, confidence=.89,
                                        children=[child])
        return [root]

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "instance-segmentation"})
    assert len(result["contours"]) == 2
    assert {c["model_class"] for c in result["contours"]} == {"outer", "child"}
    assert all(c["children"] == [] and c["parent_id"] == 102 for c in result["contours"])
    child = next(c for c in result["contours"] if c["model_class"] == "child")
    assert child["label_id"] == 44 and child["confidence"] == .91
    assert min(child["x"]) > .5 and min(child["y"]) > .5
    child_instance = next(i for i in result["prediction_metadata"]["instances"]
                          if i["source_instance_id"] == child["source_instance_id"])
    restored = binary_from(child_instance["rle"])
    assert restored.shape == (120, 240) and restored.sum() == child_crop_mask.sum()
    assert not (restored & ~scene[1][1]).any()
    assert restored[75, 170] and not restored[12, 12]


@pytest.mark.parametrize("negative_only", [False, True])
def test_hierarchy_keeps_provider_required_positive_validation(scene, negative_only):
    class RequiredSuggestion(InstanceSuggestion, CapabilityModel):
        model_info = ModelInfo(
            registry_key="strict-model", name="Required", description="Stub", usage_tip="Tests",
            input_contracts=[InputContract(task="instance-suggestion", conditioning=ConditioningSpec(
                kind="instances", unit="instance", min_units=1))],
        )

        def suggest_instances(self, req, params):
            pytest.fail("Missing positive conditioning must fail before provider execution")

    negative = np.zeros(scene[0].shape[:2], bool)
    negative[80:85, 180:185] = True
    req = request("instance-suggestion", scene, positive_exemplars=[],
                  negative_exemplars=[binary(negative)] if negative_only else [])
    with pytest.raises(ValueError, match="requires at least 1 instance conditioning"):
        predict_concept(pyfunc(RequiredSuggestion()), req,
                        {**req.parameters, "task": "instance-suggestion"})


def test_instance_segmentation_whole_image_hierarchy(scene):
    req = request("instance-segmentation", scene, label={
        "id": 43, "dataset_id": 1, "name": "Coral Surface", "value": 3})

    calls = []
    def predict(local, params):
        calls.append(local)
        assert local.image.shape == scene[0].shape
        h, w = local.image.shape[:2]
        d1 = np.zeros((h, w), bool)
        d1[15:45, 15:65] = True
        d2 = np.zeros((h, w), bool)
        d2[70:100, 170:210] = True
        c1 = Contour.from_binary_mask(d1, label_id=43, confidence=0.95)
        c2 = Contour.from_binary_mask(d2, label_id=43, confidence=0.92)
        return [c1, c2]

    result = predict_concept(SimpleNamespace(predict=predict), req,
                             {**req.parameters, "task": "instance-segmentation"},
                             {"hierarchy_input": "whole_image"})
    assert len(calls) == 1
    assert len(result["contours"]) == 2
    assert {c["parent_id"] for c in result["contours"]} == {101, 102}

