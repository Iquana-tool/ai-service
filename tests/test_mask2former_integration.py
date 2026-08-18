"""CPU integration coverage for the real Mask2Former optimization loop."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import iquana_toolbox.mlflow as toolbox_mlflow
import mlflow
import numpy as np
import pytest
import torch
from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    SwinConfig,
)

from models.mask2former import Mask2Former


@pytest.fixture
def cpu_torch_runtime(monkeypatch):
    """Keep the smoke model on CPU and restore the suite's thread setting."""
    previous_num_threads = torch.get_num_threads()
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_experiment_id = mlflow.tracking.fluent._active_experiment_id
    torch.set_num_threads(min(previous_num_threads, 2))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_tracking_uri)
    mlflow.tracking.fluent._active_experiment_id = previous_experiment_id
    torch.set_num_threads(previous_num_threads)


def _write_tiny_checkpoint(checkpoint_path) -> torch.Tensor:
    """Create an offline Mask2Former checkpoint small enough for a CPU test."""
    backbone_config = SwinConfig(
        image_size=64,
        patch_size=4,
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        window_size=4,
        drop_path_rate=0.0,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = Mask2FormerConfig(
        backbone_config=backbone_config,
        num_labels=2,
        feature_size=32,
        mask_feature_size=32,
        hidden_dim=32,
        encoder_feedforward_dim=64,
        encoder_layers=1,
        decoder_layers=2,
        num_attention_heads=4,
        dim_feedforward=64,
        num_queries=4,
        train_num_points=64,
        use_auxiliary_loss=True,
    )
    model = Mask2FormerForUniversalSegmentation(config)
    initial_classifier_weight = model.class_predictor.weight.detach().clone()
    model.save_pretrained(checkpoint_path)
    Mask2FormerImageProcessor(
        do_resize=False,
        ignore_index=255,
        do_reduce_labels=False,
    ).save_pretrained(checkpoint_path)
    return initial_classifier_weight


def _write_two_class_coco_fixture(fixture_path) -> tuple[str, str]:
    """Write two images whose COCO categories retain sparse database IDs."""
    images = []
    annotations = []
    cases = [
        (1, 7, (6, 6, 34, 34), (220, 20, 20)),
        (2, 42, (24, 20, 56, 54), (20, 220, 20)),
    ]
    for image_id, category_id, (left, top, right, bottom), color in cases:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[top:bottom, left:right] = color
        file_name = f"image-{image_id}.png"
        assert cv2.imwrite(str(fixture_path / file_name), image)
        images.append(
            {"id": image_id, "file_name": file_name, "width": 64, "height": 64}
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [
                    [left, top, right, top, right, bottom, left, bottom]
                ],
                "area": (right - left) * (bottom - top),
                "bbox": [left, top, right - left, bottom - top],
                "iscrowd": 0,
            }
        )

    annotation_path = fixture_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "categories": [
                    {"id": 7, "name": "cell"},
                    {"id": 42, "name": "core"},
                ],
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return str(annotation_path), str(fixture_path)


def test_real_mask2former_train_step_on_cpu(
    tmp_path, monkeypatch, cpu_torch_runtime
) -> None:
    """Train, publish, reload, and infer through the production CPU paths."""
    torch.manual_seed(7)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UV_OFFLINE", "1")

    checkpoint_path = tmp_path / "tiny-checkpoint"
    initial_classifier_weight = _write_tiny_checkpoint(checkpoint_path)
    annotation_path, image_folder = _write_two_class_coco_fixture(tmp_path)
    request = InstanceSegmentationTrainingRequest(
        dataset_id=11,
        image_folder_path=image_folder,
        model_registry_key="mask2former",
        user_id="integration-test",
        hyper_parameter={"epochs": 1, "learning_rate": 1e-3, "batch_size": 2},
        labels=[
            Label(id=42, dataset_id=11, name="core", value=2),
            Label(id=7, dataset_id=11, name="cell", value=1),
        ],
        annotation_file_url=annotation_path,
    )

    logged_metrics: list[tuple[dict[str, int | float], int | None]] = []
    monkeypatch.setattr("models.mask2former.mlflow.log_params", lambda _params: None)
    monkeypatch.setattr("models.mask2former.mlflow.set_tag", lambda _key, _value: None)
    monkeypatch.setattr(
        "models.mask2former.mlflow.log_metrics",
        lambda metrics, step=None: logged_metrics.append((metrics, step)),
    )
    progress_updates: list[dict[str, int | float]] = []

    model = Mask2Former(checkpoint=str(checkpoint_path))
    monkeypatch.setattr(model.model_info, "label_ids", model.model_info.label_ids)
    model.train(request, progress_callback=progress_updates.append)

    assert model._model is not None
    assert model._model.device.type == "cpu"
    assert model._has_fine_tuned_weights is True
    assert model._label_mapping is not None
    assert model._label_mapping.database_to_model == {7: 0, 42: 1}
    assert not torch.equal(
        initial_classifier_weight,
        model._model.class_predictor.weight.detach().cpu(),
    )

    assert len(logged_metrics) == 2
    loss_metrics, loss_step = logged_metrics[0]
    assert loss_step == 1
    assert math.isfinite(loss_metrics["loss"])
    assert loss_metrics["processed_batches"] == 1
    assert loss_metrics["processed_samples"] == 1
    validation_metrics, validation_step = logged_metrics[1]
    assert validation_step == 1
    assert math.isfinite(validation_metrics["val_mask_ap"])
    assert math.isfinite(validation_metrics["val_mask_ap50"])
    assert math.isfinite(validation_metrics["val_mask_ap75"])
    assert math.isfinite(validation_metrics["val_mask_iou_macro"])
    assert math.isfinite(validation_metrics["val_mask_f1_macro"])
    assert math.isfinite(validation_metrics["val_mask_precision_macro"])
    assert math.isfinite(validation_metrics["val_mask_recall_macro"])
    label_metric_keys = [
        key for key in validation_metrics if "_label_" in key
    ]
    assert label_metric_keys
    assert all(math.isfinite(validation_metrics[key]) for key in label_metric_keys)
    assert progress_updates == [
        {
            "epoch": 1,
            "loss": loss_metrics["loss"],
            "processed_batches": 1,
            "val_mask_ap": validation_metrics["val_mask_ap"],
            "val_mask_ap50": validation_metrics["val_mask_ap50"],
            "val_mask_ap75": validation_metrics["val_mask_ap75"],
            "val_mask_iou_macro": validation_metrics["val_mask_iou_macro"],
            "val_mask_f1_macro": validation_metrics["val_mask_f1_macro"],
            "val_mask_precision_macro": validation_metrics["val_mask_precision_macro"],
            "val_mask_recall_macro": validation_metrics["val_mask_recall_macro"],
        }
    ]

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        "mask2former-integration",
        artifact_location=(tmp_path / "mlflow-artifacts").as_uri(),
    )
    mlflow.set_experiment("mask2former-integration")

    worker_artifact_paths: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def make_observable_artifact_directory(*args, **kwargs):
        if kwargs.get("prefix") == "iquana_model_artifacts_":
            kwargs["dir"] = tmp_path
        artifact_path = real_mkdtemp(*args, **kwargs)
        if kwargs.get("prefix") == "iquana_model_artifacts_":
            worker_artifact_paths.append(artifact_path)
        return artifact_path

    monkeypatch.setattr(
        toolbox_mlflow.tempfile,
        "mkdtemp",
        make_observable_artifact_directory,
    )
    monkeypatch.setitem(
        model.model_info.tags, "dataset_id", str(request.dataset_id)
    )
    monkeypatch.setitem(model.model_info.tags, "user_id", str(request.user_id))
    registry = MLFlowModelRegistry(tracking_uri)
    registry.register_model(model)

    assert len(worker_artifact_paths) == 1
    assert not os.path.exists(worker_artifact_paths[0])
    shutil.rmtree(checkpoint_path)
    assert not checkpoint_path.exists()

    fresh_process_path = tmp_path / "fresh-process"
    fresh_process_path.mkdir()
    inference_image_path = tmp_path / "image-1.png"
    subprocess_code = """
import json
import os
import sys

from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.networking.http.services import (
    InstanceSegmentationRequest,
)

tracking_uri, image_path = sys.argv[1:]
registry = MLFlowModelRegistry(tracking_uri)
loaded = registry.get_model_by_version("mask2former", "latest")
request = InstanceSegmentationRequest(
    image_url=image_path,
    user_id="fresh-process",
    model_registry_key="mask2former",
)
contours = loaded.predict(
    request,
    {"task": "instance-segmentation", "threshold": 0.0},
)
python_model = loaded._model_impl.python_model
mapping = python_model._label_mapping
print("PHASE6B_RESULT=" + json.dumps({
    "artifact_path": str(python_model._artifact_path),
    "artifact_exists": os.path.isdir(python_model._artifact_path),
    "checkpoint_exists": os.path.exists(python_model.checkpoint),
    "database_to_model": mapping.database_to_model,
    "model_to_database": mapping.model_to_database,
    "has_fine_tuned_weights": python_model._has_fine_tuned_weights,
    "device": str(python_model._model.device),
    "inference_executed": True,
    "contour_count": len(contours),
    "labels": [contour.label_id for contour in contours],
}, sort_keys=True))
"""
    subprocess_environment = os.environ.copy()
    subprocess_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            subprocess_code,
            tracking_uri,
            str(inference_image_path),
        ],
        cwd=fresh_process_path,
        env=subprocess_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("PHASE6B_RESULT=")
    )
    result = json.loads(result_line.removeprefix("PHASE6B_RESULT="))

    assert result["inference_executed"] is True
    assert result["database_to_model"] == {"7": 0, "42": 1}
    assert result["model_to_database"] == {"0": 7, "1": 42}
    assert result["has_fine_tuned_weights"] is True
    assert result["device"] == "cpu"
    assert result["artifact_path"]
    assert result["artifact_exists"] is True
    assert result["checkpoint_exists"] is False
    assert len(result["labels"]) == result["contour_count"]
    assert set(result["labels"]) <= {7, 42}
