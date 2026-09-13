"""Focused unit tests for AI service input contracts and dispatch validation (Issue #27).

These tests exercise parameter validation, zero-capability detection, default normalization,
and handler forwarding for production models (SAM 3, Mask2Former, SAM 2) without loading
heavy model weights.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.networking.http.services import (
    BaseServiceRequest,
    CrossImageExemplar,
    CrossImageSuggestionRequest,
    EmbedRequest,
    InstanceSegmentationRequest,
    InstanceSuggestionRequest,
    PromptedSegmentationRequest,
)
from iquana_toolbox.schemas.prompts import BoxPrompt, Prompts
from iquana_toolbox.schemas.training import HyperParameter

from models.base import (
    CapabilityModel,
    CrossImageSuggestion,
    TaskCapability,
    register_task,
    validate_and_normalize_params,
    validate_request_conditioning,
)
from models.mask2former import Mask2Former
from models.mask2former_dataset import LabelMapping
from models.sam2 import SAM2Prompted
from models.sam3 import SAM3


# --------------------------------------------------------------------------- #
# Parameter validator tests
# --------------------------------------------------------------------------- #
def test_validate_and_normalize_params_none_contract():
    raw = {"task": "some-task", "foo": "bar", "num": 42}
    result = validate_and_normalize_params(None, raw)
    assert result == raw
    assert result is not raw  # returns copy


def test_validate_and_normalize_params_fills_defaults():
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.3, min_value=0.0, max_value=1.0),
            HyperParameter(key="enabled", label="Enabled", type="bool", default_value=True),
            HyperParameter(key="max_items", label="Max Items", type="int", default_value=10, min_value=1, max_value=100),
            HyperParameter(key="mode", label="Mode", type="str", default_value="fast", options=["fast", "accurate"]),
        ],
    )
    result = validate_and_normalize_params(contract, {})
    assert result == {
        "threshold": 0.3,
        "enabled": True,
        "max_items": 10,
        "mode": "fast",
    }


def test_validate_and_normalize_numeric_default_normalization():
    """A float parameter with integer default (e.g. 1) must normalize default to float (1.0)."""
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="rate", label="Rate", type="float", default_value=1, min_value=0, max_value=10),
            HyperParameter(key="zero_float", label="Zero Float", type="float", default_value=0),
        ],
    )
    # Default filling
    result = validate_and_normalize_params(contract, {})
    assert result["rate"] == 1.0
    assert isinstance(result["rate"], float)
    assert result["zero_float"] == 0.0
    assert isinstance(result["zero_float"], float)

    # Provided int value coerced to float
    result_custom = validate_and_normalize_params(contract, {"rate": 5})
    assert result_custom["rate"] == 5.0
    assert isinstance(result_custom["rate"], float)


def test_validate_and_normalize_params_preserves_task_routing():
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.3),
        ],
    )
    result = validate_and_normalize_params(contract, {"task": "dummy-task", "threshold": 0.7})
    assert result == {"task": "dummy-task", "threshold": 0.7}


def test_validate_and_normalize_params_rejects_unknown_param():
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.3),
        ],
    )
    with pytest.raises(ValueError, match="Unknown parameter 'bogus_param'"):
        validate_and_normalize_params(contract, {"bogus_param": 123})


def test_validate_and_normalize_params_type_checks():
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="flag", label="Flag", type="bool", default_value=True),
            HyperParameter(key="count", label="Count", type="int", default_value=5),
            HyperParameter(key="rate", label="Rate", type="float", default_value=0.1),
            HyperParameter(key="name", label="Name", type="str", default_value="abc"),
        ],
    )
    # Bool rejects int / str
    with pytest.raises(ValueError, match="must be a bool"):
        validate_and_normalize_params(contract, {"flag": 1})
    with pytest.raises(ValueError, match="must be a bool"):
        validate_and_normalize_params(contract, {"flag": "true"})

    # Int rejects bool / float / str
    with pytest.raises(ValueError, match="must be an int"):
        validate_and_normalize_params(contract, {"count": True})
    with pytest.raises(ValueError, match="must be an int"):
        validate_and_normalize_params(contract, {"count": 5.5})

    # Float accepts int (coerced) and float, rejects bool / str / inf / nan
    res = validate_and_normalize_params(contract, {"rate": 2})
    assert res["rate"] == 2.0 and isinstance(res["rate"], float)

    with pytest.raises(ValueError, match="must be a finite float"):
        validate_and_normalize_params(contract, {"rate": True})
    with pytest.raises(ValueError, match="must be a finite float"):
        validate_and_normalize_params(contract, {"rate": float("nan")})
    with pytest.raises(ValueError, match="must be a finite float"):
        validate_and_normalize_params(contract, {"rate": 10**1000})

    # Str rejects int
    with pytest.raises(ValueError, match="must be a str"):
        validate_and_normalize_params(contract, {"name": 123})


def test_validate_and_normalize_params_bounds_and_options():
    contract = InputContract(
        task="dummy-task",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.5, min_value=0.0, max_value=1.0),
            HyperParameter(key="mode", label="Mode", type="str", default_value="a", options=["a", "b"]),
        ],
    )
    # Below min
    with pytest.raises(ValueError, match="less than min_value"):
        validate_and_normalize_params(contract, {"threshold": -0.1})

    # Above max
    with pytest.raises(ValueError, match="greater than max_value"):
        validate_and_normalize_params(contract, {"threshold": 1.1})

    # Invalid option
    with pytest.raises(ValueError, match="not in allowed options"):
        validate_and_normalize_params(contract, {"mode": "c"})


# --------------------------------------------------------------------------- #
# CapabilityModel contract validation & dispatch tests (no weights)
# --------------------------------------------------------------------------- #
class DummyRequest(BaseServiceRequest):
    user_id: str = "test-user"
    model_registry_key: str = "dummy"


DUMMY_TASK = register_task("dummy-infer", DummyRequest, "run_dummy")


class DummyInfer(TaskCapability):
    TASK = DUMMY_TASK

    def run_dummy(self, request: DummyRequest, params: dict[str, Any]):
        return {"handled": True, "params": params}


def test_capability_model_unsupported_contract_task_raises_on_class_creation():
    """Declaring an input_contract for a task the model does not mix in must raise ValueError."""
    with pytest.raises(ValueError, match=r"declares input_contract for task 'unsupported-task'"):
        class BadModel(DummyInfer, CapabilityModel):
            model_info = ModelInfo(
                registry_key="bad-model",
                name="Bad Model",
                description="desc",
                usage_tip="tip",
                status="ready",
                input_contracts=[
                    InputContract(
                        task="unsupported-task",
                        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    ),
                ],
            )


def test_zero_capability_class_contract_mismatch_raises_on_class_creation():
    """A class with zero capability mixins that declares an input contract must fail validation."""
    with pytest.raises(ValueError, match=r"declares input_contract for task 'some-task'"):
        class ZeroCapabilityModel(CapabilityModel):
            model_info = ModelInfo(
                registry_key="zero-cap-model",
                name="Zero Cap Model",
                description="desc",
                usage_tip="tip",
                status="ready",
                input_contracts=[
                    InputContract(
                        task="some-task",
                        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    ),
                ],
            )


def test_zero_capability_class_contract_mismatch_raises_on_instance_init():
    """A class with zero capability mixins setting per-instance model_info contracts must fail validation."""
    class DynamicZeroCapabilityModel(CapabilityModel):
        def __init__(self):
            self.model_info = ModelInfo(
                registry_key="dyn-zero-cap",
                name="Dyn Zero Cap",
                description="desc",
                usage_tip="tip",
                status="ready",
                input_contracts=[
                    InputContract(
                        task="some-task",
                        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    ),
                ],
            )

    with pytest.raises(ValueError, match=r"declares input_contract for task 'some-task'"):
        DynamicZeroCapabilityModel()


def test_capability_model_unsupported_contract_task_raises_on_instance_init():
    """Instantiating a model whose per-instance model_info has an invalid contract must raise ValueError."""
    class DynamicBadModel(DummyInfer, CapabilityModel):
        def __init__(self):
            self.model_info = ModelInfo(
                registry_key="dyn-bad-model",
                name="Dyn Bad Model",
                description="desc",
                usage_tip="tip",
                status="ready",
                input_contracts=[
                    InputContract(
                        task="unsupported-task",
                        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    ),
                ],
            )

    with pytest.raises(ValueError, match=r"declares input_contract for task 'unsupported-task'"):
        DynamicBadModel()


def test_capability_model_predict_normalizes_parameters():
    """Predict fills defaults, validates constraints, and forwards normalized dict to handler."""
    class ValidDummyModel(DummyInfer, CapabilityModel):
        model_info = ModelInfo(
            registry_key="valid-dummy",
            name="Valid Dummy",
            description="desc",
            usage_tip="tip",
            status="ready",
            input_contracts=[
                InputContract(
                    task="dummy-infer",
                    conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    parameters=[
                        HyperParameter(
                            key="threshold", label="Threshold", type="float", default_value=0.3, min_value=0.0, max_value=1.0
                        ),
                        HyperParameter(
                            key="tag", label="Tag", type="str", default_value="default-tag"
                        ),
                    ],
                ),
            ],
        )

    model = ValidDummyModel()
    req = DummyRequest(image_url="http://example.com/img.png")

    # Omitted params -> defaults filled
    out = model.predict(None, req, {"task": "dummy-infer"})
    assert out["handled"] is True
    assert out["params"] == {"task": "dummy-infer", "threshold": 0.3, "tag": "default-tag"}

    # Provided params -> validated and forwarded
    out = model.predict(None, req, {"task": "dummy-infer", "threshold": 0.8, "tag": "custom"})
    assert out["params"] == {"task": "dummy-infer", "threshold": 0.8, "tag": "custom"}

    # Unknown param -> error before handler
    with pytest.raises(ValueError, match="Unknown parameter 'bad_key'"):
        model.predict(None, req, {"task": "dummy-infer", "bad_key": 123})

    # Out-of-bounds param -> error before handler
    with pytest.raises(ValueError, match="greater than max_value"):
        model.predict(None, req, {"task": "dummy-infer", "threshold": 1.5})


def test_conditioning_cardinality_rejects_required_empty_and_accepts_optional_empty(monkeypatch):
    """Conditioning cardinality is enforced before handlers, including min_units=0."""
    monkeypatch.setattr(SAM3, "_load_model", lambda self: None)
    sam3 = SAM3()
    handler = MagicMock()
    monkeypatch.setattr(SAM3, "suggest_cross_image", handler)
    empty_request = CrossImageSuggestionRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="sam3",
    )

    with pytest.raises(ValueError, match=r"requires at least 1 image"):
        sam3.predict(None, empty_request, {"task": "cross-image-suggestion"})
    handler.assert_not_called()

    class OptionalCrossImageModel(CrossImageSuggestion, CapabilityModel):
        model_info = ModelInfo(
            registry_key="optional-cross-image",
            name="Optional Cross Image",
            description="desc",
            usage_tip="tip",
            status="ready",
            input_contracts=[
                InputContract(
                    task="cross-image-suggestion",
                    conditioning=ConditioningSpec(
                        kind="reference_images",
                        unit="image",
                        min_units=0,
                        max_units=1,
                        user_selectable_count=False,
                    ),
                ),
            ],
        )

        def suggest_cross_image(self, request, params):
            return {"handled": True}

    optional_model = OptionalCrossImageModel()
    assert optional_model.predict(
        None, empty_request, {"task": "cross-image-suggestion"}
    ) == {"handled": True}


@pytest.mark.asyncio
async def test_cross_image_route_returns_422_for_invalid_conditioning(monkeypatch):
    from fastapi import HTTPException
    from app.routes.cross_image import suggest_cross_image

    model = MagicMock()
    model.predict.side_effect = ValueError(
        "Task 'cross-image-suggestion' requires at least 1 image conditioning unit(s), "
        "but request supplied 0."
    )
    monkeypatch.setattr(
        "app.routes.cross_image.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: model,
    )
    request = CrossImageSuggestionRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="sam3",
    )

    with pytest.raises(HTTPException) as exc_info:
        await suggest_cross_image(request)

    assert exc_info.value.status_code == 422
    assert "requires at least 1 image" in exc_info.value.detail


@pytest.mark.asyncio
async def test_instance_seg_routes_return_422_for_contract_validation_failure(monkeypatch):
    from fastapi import HTTPException
    from app.routes.instance_seg import inference, run_inference

    model = MagicMock()
    model.predict.side_effect = ValueError("Task 'instance-segmentation' requires threshold in [0.0, 1.0]")
    monkeypatch.setattr(
        "app.routes.instance_seg.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: model,
    )
    monkeypatch.setattr("app.routes.instance_seg.validate_model", lambda req: None)

    request = InstanceSegmentationRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="m2f",
    )

    with pytest.raises(HTTPException) as exc_info:
        await inference(request)
    assert exc_info.value.status_code == 422
    assert "requires threshold in [0.0, 1.0]" in exc_info.value.detail

    with pytest.raises(HTTPException) as exc_info_run:
        await run_inference(request)
    assert exc_info_run.value.status_code == 422
    assert "requires threshold in [0.0, 1.0]" in exc_info_run.value.detail


@pytest.mark.asyncio
async def test_prompted_seg_route_returns_422_for_contract_validation_failure(monkeypatch):
    from fastapi import HTTPException
    from app.routes.prompted import inference as prompted_inference

    model = MagicMock()
    model.predict.side_effect = ValueError("Unknown parameter 'invalid_param'")
    monkeypatch.setattr(
        "app.routes.prompted.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: model,
    )

    request = PromptedSegmentationRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="sam3",
        prompts=Prompts(points=[], boxes=[]),
    )

    with pytest.raises(HTTPException) as exc_info:
        await prompted_inference(request)
    assert exc_info.value.status_code == 422
    assert "Unknown parameter 'invalid_param'" in exc_info.value.detail


@pytest.mark.asyncio
async def test_instance_suggestion_route_returns_422_for_empty_exemplars(monkeypatch):
    from fastapi import HTTPException
    from app.routes.suggestion import infer_instances

    model = MagicMock()
    model.predict.side_effect = ValueError(
        "Task 'instance-suggestion' requires at least 1 instance conditioning unit(s), but request supplied 0."
    )
    monkeypatch.setattr(
        "app.routes.suggestion.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: model,
    )

    request = InstanceSuggestionRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="sam3",
        positive_exemplars=[],
    )

    with pytest.raises(HTTPException) as exc_info:
        await infer_instances(request)
    assert exc_info.value.status_code == 422
    assert "requires at least 1 instance" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embed_route_returns_422_for_contract_validation_failure(monkeypatch):
    from fastapi import HTTPException
    from app.routes.embed import inference as embed_inference

    model = MagicMock()
    model.predict.side_effect = ValueError("Task 'embed' invalid kind")
    monkeypatch.setattr(
        "app.routes.embed.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: model,
    )

    request = EmbedRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="dinov3",
    )

    with pytest.raises(HTTPException) as exc_info:
        await embed_inference(request)
    assert exc_info.value.status_code == 422
    assert "invalid kind" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embed_route_forwards_parameters_and_overrides_task(monkeypatch):
    from app.routes.embed import inference as embed_inference

    captured = {}

    class MockModel:
        def predict(self, data, params=None):
            captured["data"] = data
            captured["params"] = params
            return []

    monkeypatch.setattr(
        "app.routes.embed.MODEL_REGISTRY.get_model_by_version",
        lambda *_args: MockModel(),
    )
    request = EmbedRequest(
        image_url="http://example.com/img.png",
        user_id="test-user",
        model_registry_key="dinov3",
        parameters={"task": "hacked_task", "threshold": 0.4},
    )

    response = await embed_inference(request)

    assert response["success"] is True
    assert captured["data"] == [request]
    assert captured["params"] == {"task": "embed", "threshold": 0.4}


def test_conditioning_cardinality_counts_model_usable_units():
    instance_contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(
            kind="instances", unit="instance", min_units=1, user_selectable_count=False
        ),
    )
    ids_only = InstanceSegmentationRequest(
        image_url="http://example.com/target.png",
        user_id="test-user",
        model_registry_key="conditioned-model",
        contour_ids=[123],
    )
    with pytest.raises(ValueError, match=r"requires at least 1 instance"):
        validate_request_conditioning(instance_contract, ids_only)

    mask = BinaryMask.from_numpy_array(np.ones((4, 4), dtype=bool))
    duplicate_source = CrossImageSuggestionRequest(
        image_url="http://example.com/target.png",
        user_id="test-user",
        model_registry_key="reference-model",
        exemplars=[
            CrossImageExemplar(image_url="http://example.com/reference.png", mask=mask),
            CrossImageExemplar(image_url="http://example.com/reference.png", mask=mask),
        ],
    )
    reference_contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images", unit="image", min_units=2, user_selectable_count=False
        ),
    )
    with pytest.raises(ValueError, match=r"requires at least 2 image"):
        validate_request_conditioning(reference_contract, duplicate_source)


# --------------------------------------------------------------------------- #
# Production model declarations audit
# --------------------------------------------------------------------------- #
def test_sam3_declarations_audit():
    """SAM3 declarations must match its 3 tasks and handler parameter lookups."""
    info = SAM3.model_info
    assert len(info.input_contracts) == 3
    contracts = {c.task: c for c in info.input_contracts}

    assert "instance-suggestion" in contracts
    assert "cross-image-suggestion" in contracts
    assert "prompted-segmentation" in contracts

    # instance-suggestion
    sug_c = contracts["instance-suggestion"]
    assert sug_c.conditioning.kind == "instances"
    assert sug_c.conditioning.unit == "instance"
    assert sug_c.conditioning.min_units == 1
    assert sug_c.conditioning.max_units is None
    sug_keys = {p.key: p for p in sug_c.parameters}
    assert set(sug_keys.keys()) == {"threshold", "mask_threshold"}
    assert sug_keys["threshold"].default_value == 0.3
    assert sug_keys["mask_threshold"].default_value == 0.5

    # cross-image-suggestion
    cross_c = contracts["cross-image-suggestion"]
    assert cross_c.conditioning.kind == "reference_images"
    assert cross_c.conditioning.unit == "image"
    assert cross_c.conditioning.min_units == 1
    assert cross_c.conditioning.max_units == 1
    assert cross_c.conditioning.requires_complete_annotation is True
    cross_keys = {p.key: p for p in cross_c.parameters}
    assert set(cross_keys.keys()) == {"threshold", "mask_threshold", "min_target_frac"}
    assert cross_keys["threshold"].default_value == 0.3
    assert cross_keys["mask_threshold"].default_value == 0.5
    assert cross_keys["min_target_frac"].default_value == 0.5

    # prompted-segmentation
    prompt_c = contracts["prompted-segmentation"]
    assert prompt_c.conditioning.kind == "none"
    assert prompt_c.conditioning.unit is None
    assert prompt_c.conditioning.user_selectable_count is False
    prompt_keys = {p.key: p for p in prompt_c.parameters}
    assert set(prompt_keys.keys()) == {"threshold", "mask_threshold"}
    assert prompt_keys["threshold"].default_value == 0.3
    assert prompt_keys["mask_threshold"].default_value == 0.5


def test_mask2former_declarations_audit():
    """Mask2Former must declare 'threshold' (matching handler) and kind='none' with no unit."""
    info = Mask2Former.model_info
    assert len(info.input_contracts) == 1
    contract = info.input_contracts[0]
    assert contract.task == "instance-segmentation"
    assert contract.conditioning.kind == "none"
    assert contract.conditioning.unit is None
    assert contract.conditioning.user_selectable_count is False

    param_keys = {p.key: p for p in contract.parameters}
    assert "threshold" in param_keys
    assert "score_threshold" not in param_keys
    assert param_keys["threshold"].default_value == 0.5
    assert param_keys["threshold"].type == "float"


def test_sam2_declarations_audit(monkeypatch):
    """SAM 2 variants must declare kind='none' with no unit for prompted-segmentation."""
    # Mock _load_weights so we don't download/load Hugging Face weights in unit tests
    monkeypatch.setattr(SAM2Prompted, "_load_weights", lambda self: None)

    for variant_key in ["sam2-1-tiny", "sam2-1-small", "sam2-1-base-plus", "sam2-1-large"]:
        instance = SAM2Prompted(variant_key)
        info = instance.model_info
        assert len(info.input_contracts) == 1
        contract = info.input_contracts[0]
        assert contract.task == "prompted-segmentation"
        assert contract.conditioning.kind == "none"
        assert contract.conditioning.unit is None
        assert contract.conditioning.user_selectable_count is False
        assert contract.parameters == []


# --------------------------------------------------------------------------- #
# Production SAM 3 and Mask2Former handler parameter forwarding tests
# --------------------------------------------------------------------------- #
def test_sam3_handler_forwarding(monkeypatch):
    """Test that predict() on SAM3 validates and forwards normalized parameters to handlers."""
    monkeypatch.setattr(SAM3, "_load_model", lambda self: None)
    monkeypatch.setattr(
        "iquana_toolbox.schemas.networking.http.services.get_image_from_url_cached",
        lambda url: np.zeros((64, 64, 3), dtype=np.uint8),
    )

    sam3 = SAM3()
    sam3.processor = MagicMock()
    sam3.model = MagicMock()

    # 1. Instance suggestion: post-processor receives normalized threshold and mask_threshold
    mock_inputs = MagicMock()
    mock_inputs.get.return_value = torch.tensor([[64, 64]])
    sam3.processor.return_value = mock_inputs
    sam3.processor.post_process_instance_segmentation.return_value = [
        {"masks": torch.zeros((0, 64, 64)), "scores": torch.zeros((0,))}
    ]

    dummy_mask = BinaryMask.from_numpy_array(np.ones((10, 10), dtype=bool))
    req_sug = InstanceSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="sam3",
        positive_exemplars=[dummy_mask],
    )

    # Explicit params
    sam3.predict(None, req_sug, {"task": "instance-suggestion", "threshold": 0.25, "mask_threshold": 0.6})
    _, post_kwargs = sam3.processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.25
    assert post_kwargs["mask_threshold"] == 0.6

    # Omitted params -> filled with defaults (0.3, 0.5)
    sam3.predict(None, req_sug, {"task": "instance-suggestion"})
    _, post_kwargs = sam3.processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.3
    assert post_kwargs["mask_threshold"] == 0.5

    # Unknown param rejected before execution
    with pytest.raises(ValueError, match="Unknown parameter 'invalid_param'"):
        sam3.predict(None, req_sug, {"task": "instance-suggestion", "invalid_param": 1.0})

    # 2. Prompted segmentation: post-processor receives threshold and mask_threshold
    req_prompt = PromptedSegmentationRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="sam3",
        prompts=Prompts(box_prompt=BoxPrompt(min_x=0.1, min_y=0.1, max_x=0.9, max_y=0.9)),
    )

    sam3.predict(None, req_prompt, {"task": "prompted-segmentation", "threshold": 0.4, "mask_threshold": 0.7})
    _, post_kwargs = sam3.processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.4
    assert post_kwargs["mask_threshold"] == 0.7

    # 3. Cross-image suggestion: post-processor receives threshold, mask_threshold; extract_target_masks receives min_target_frac
    extract_mock = MagicMock(return_value=([np.zeros((64, 64), dtype=np.uint8)], [0]))
    monkeypatch.setattr("models.sam3.concat_ops.extract_target_masks", extract_mock)

    sam3.processor.post_process_instance_segmentation.return_value = [
        {"masks": torch.zeros((1, 64, 64)), "scores": torch.tensor([0.9])}
    ]

    req_cross = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="sam3",
        exemplars=[
            CrossImageExemplar(
                image_url="http://example.com/ex.png",
                mask=dummy_mask,
            )
        ],
    )

    # Explicit params
    sam3.predict(
        None, req_cross,
        {"task": "cross-image-suggestion", "threshold": 0.35, "mask_threshold": 0.45, "min_target_frac": 0.75}
    )
    _, post_kwargs = sam3.processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.35
    assert post_kwargs["mask_threshold"] == 0.45
    assert extract_mock.call_args[1]["min_target_frac"] == 0.75

    # Omitted params -> defaults filled (threshold=0.3, mask_threshold=0.5, min_target_frac=0.5)
    sam3.predict(None, req_cross, {"task": "cross-image-suggestion"})
    _, post_kwargs = sam3.processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.3
    assert post_kwargs["mask_threshold"] == 0.5
    assert extract_mock.call_args[1]["min_target_frac"] == 0.5


def test_sam3_multiple_exemplars_same_reference_image(monkeypatch):
    """Test that SAM 3 composites a single reference image once but generates prompt boxes for all its exemplar masks."""
    monkeypatch.setattr(SAM3, "_load_model", lambda self: None)
    monkeypatch.setattr(
        "iquana_toolbox.schemas.networking.http.services.get_image_from_url_cached",
        lambda url: np.zeros((64, 64, 3), dtype=np.uint8),
    )
    extract_mock = MagicMock(return_value=([np.zeros((64, 64), dtype=np.uint8)], [0]))
    monkeypatch.setattr("models.sam3.concat_ops.extract_target_masks", extract_mock)

    sam3 = SAM3()
    sam3.processor = MagicMock()
    sam3.model = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.get.return_value = torch.tensor([[128, 64]])
    sam3.processor.return_value = mock_inputs
    sam3.processor.post_process_instance_segmentation.return_value = [
        {"masks": torch.zeros((1, 64, 64)), "scores": torch.tensor([0.9])}
    ]

    mask1 = BinaryMask.from_numpy_array(np.ones((10, 10), dtype=bool))
    mask2 = BinaryMask.from_numpy_array(np.ones((15, 15), dtype=bool))

    # Two exemplars from the same reference image URL
    req_cross_multi = CrossImageSuggestionRequest(
        image_url="http://example.com/target.png",
        user_id="user1",
        model_registry_key="sam3",
        exemplars=[
            CrossImageExemplar(image_url="http://example.com/ref.png", mask=mask1),
            CrossImageExemplar(image_url="http://example.com/ref.png", mask=mask2),
        ],
    )

    sam3.predict(None, req_cross_multi, {"task": "cross-image-suggestion"})

    # Check processor call
    _, proc_kwargs = sam3.processor.call_args
    # There should be 2 input boxes corresponding to the 2 exemplar masks
    assert len(proc_kwargs["input_boxes"][0]) == 2
    # All boxes should have label 1 (positive prompt)
    assert proc_kwargs["input_boxes_labels"].tolist() == [[1, 1]]


def test_validate_request_conditioning_in_context_embeddings_min_units():
    """Validates that exemplar_embeddings vector count is validated against min_units."""
    from models.base import validate_request_conditioning

    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="embeddings",
            unit="vector",
            min_units=2,
            max_units=5,
            embedding_kinds=["region_mean"],
        ),
        parameters=[],
    )

    # 1 vector supplied -> should fail min_units=2
    req_1_vec = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        exemplar_embeddings={"region_mean": [[0.1, 0.2]]},
    )
    with pytest.raises(ValueError, match="requires at least 2 vector conditioning unit"):
        validate_request_conditioning(contract, req_1_vec)

    # Unrelated image vectors must not satisfy the required region_mean cardinality.
    req_1_region_2_image = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        exemplar_embeddings={
            "region_mean": [[0.1, 0.2]],
            "image_cls": [[0.3, 0.4], [0.5, 0.6]],
        },
    )
    with pytest.raises(ValueError, match="requires at least 2 vector conditioning unit"):
        validate_request_conditioning(contract, req_1_region_2_image)

    # 2 vectors supplied -> should succeed
    req_2_vec = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        exemplar_embeddings={"region_mean": [[0.1, 0.2], [0.3, 0.4]]},
    )
    validate_request_conditioning(contract, req_2_vec)


def test_validate_request_conditioning_requires_declared_embedding_kinds():
    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="embeddings",
            unit="vector",
            min_units=1,
            max_units=2,
            embedding_kinds=["region_mean"],
        ),
        parameters=[],
    )

    flat_image_only = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        embeddings={"image_cls": [0.1, 0.2]},
    )
    with pytest.raises(ValueError, match=r"requires embedding kind.*region_mean"):
        validate_request_conditioning(contract, flat_image_only)

    grouped_image_only = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        exemplar_embeddings={"image_cls": [[0.1, 0.2]]},
    )
    with pytest.raises(ValueError, match=r"requires embedding kind.*region_mean"):
        validate_request_conditioning(contract, grouped_image_only)

    flat_region = CrossImageSuggestionRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="emb-model",
        embeddings={"region_mean": [0.1, 0.2]},
    )
    validate_request_conditioning(contract, flat_region)


def test_mask2former_handler_forwarding(monkeypatch):
    """Test that predict() on Mask2Former validates and forwards normalized threshold to segment_instances."""
    monkeypatch.setattr(
        "iquana_toolbox.schemas.networking.http.services.get_image_from_url_cached",
        lambda url: np.zeros((64, 64, 3), dtype=np.uint8),
    )

    m2f = Mask2Former()
    m2f._label_mapping = LabelMapping(database_to_model={10: 0}, model_to_database={0: 10}, id2label={0: "cell"})
    m2f._load_weights = MagicMock()
    m2f._processor = MagicMock()
    m2f._model = MagicMock()

    m2f._processor.return_value = {}
    m2f._processor.post_process_instance_segmentation.return_value = [
        {"segmentation": torch.zeros((64, 64), dtype=torch.int32), "segments_info": []}
    ]

    req = InstanceSegmentationRequest(
        image_url="http://example.com/test.png",
        user_id="user1",
        model_registry_key="mask2former",
        label=Label(id=10, name="cell", dataset_id=1, value=1),
    )

    # 1. Explicit threshold is validated and forwarded to post-processor
    m2f.predict(None, req, {"task": "instance-segmentation", "threshold": 0.65})
    _, post_kwargs = m2f._processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.65

    # 2. Omitted threshold receives declared default 0.5
    m2f.predict(None, req, {"task": "instance-segmentation"})
    _, post_kwargs = m2f._processor.post_process_instance_segmentation.call_args
    assert post_kwargs["threshold"] == 0.5

    # 3. Passing legacy score_threshold raises ValueError before handler executes
    with pytest.raises(ValueError, match="Unknown parameter 'score_threshold'"):
        m2f.predict(None, req, {"task": "instance-segmentation", "score_threshold": 0.6})


def test_routes_protect_task_parameter(monkeypatch):
    """Test that all AI service routes protect their authoritative task against user parameters overrides."""
    import asyncio
    from app.routes.cross_image import suggest_cross_image
    from app.routes.instance_seg import inference, run_inference
    from app.routes.prompted import inference as prompted_inference
    from app.routes.suggestion import infer_instances

    captured_params = {}

    class MockModel:
        def predict(self, data, params=None):
            task = params.get("task") if params else None
            captured_params["task"] = task
            if task in ("cross-image-suggestion", "instance-suggestion"):
                return ([], [])
            return []

    monkeypatch.setattr("app.state.MODEL_REGISTRY.get_model_by_version", lambda key, ver: MockModel())
    monkeypatch.setattr("app.routes.instance_seg.validate_model", lambda req: None)

    # 1. cross-image route
    from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest, InstanceSuggestionRequest, PromptedSegmentationRequest
    from iquana_toolbox.schemas.prompts import Prompts

    req_cross = CrossImageSuggestionRequest(
        image_url="http://example.com/a.png",
        user_id="u1",
        model_registry_key="m",
        parameters={"task": "hacked_task", "threshold": 0.4},
    )
    asyncio.run(suggest_cross_image(req_cross))
    assert captured_params["task"] == "cross-image-suggestion"

    # 2. instance_seg route
    req_inst = InstanceSegmentationRequest(
        image_url="http://example.com/a.png",
        user_id="u1",
        model_registry_key="m",
        parameters={"task": "hacked_task", "threshold": 0.4},
    )
    asyncio.run(inference(req_inst))
    assert captured_params["task"] == "instance-segmentation"

    asyncio.run(run_inference(req_inst))
    assert captured_params["task"] == "instance-segmentation"

    # 3. prompted route
    req_prompted = PromptedSegmentationRequest(
        image_url="http://example.com/a.png",
        user_id="u1",
        model_registry_key="m",
        prompts=Prompts(points=[], boxes=[]),
        parameters={"task": "hacked_task"},
    )
    asyncio.run(prompted_inference(req_prompted))
    assert captured_params["task"] == "prompted-segmentation"

    # 4. suggestion route
    req_sugg = InstanceSuggestionRequest(
        image_url="http://example.com/a.png",
        user_id="u1",
        model_registry_key="m",
        positive_exemplars=[],
        parameters={"task": "hacked_task"},
    )
    asyncio.run(infer_instances(req_sugg))
    assert captured_params["task"] == "instance-suggestion"
