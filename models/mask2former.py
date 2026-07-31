"""Mask2Former instance segmentation with portable MLflow fine-tuning artifacts."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from iquana_service_core import register_model
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import HyperParameter, InstanceSegmentationModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from models.base import CapabilityModel, InstanceSegmentation
from models.mask2former_dataset import (
    INSTANCE_IGNORE_INDEX,
    CocoInstanceDataset,
    CocoTrainingDataError,
    LabelMapping,
)

logger = logging.getLogger(__name__)

_ARTIFACT_NAME = "mask2former"
_LABEL_MAPPING_FILENAME = "label_mapping.json"
_IGNORE_INDEX = INSTANCE_IGNORE_INDEX
_MIN_LEARNING_RATE = 1e-8
_MAX_LEARNING_RATE = 1.0


class TrainingCancelled(RuntimeError):
    """Raised when a training worker observes a cooperative cancellation request."""


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> tuple[int, float, int]:
    """Validate and normalize the Mask2Former hyperparameters supplied by a client."""
    try:
        epochs = int(hyperparameters.get("epochs", 5))
        learning_rate = float(hyperparameters.get("learning_rate", 1e-4))
        batch_size = int(hyperparameters.get("batch_size", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("epochs, learning_rate, and batch_size must be numeric.") from exc
    if not 1 <= epochs <= 50:
        raise ValueError("epochs must be between 1 and 50.")
    if not math.isfinite(learning_rate) or not _MIN_LEARNING_RATE <= learning_rate <= _MAX_LEARNING_RATE:
        raise ValueError("learning_rate must be finite and between 1e-8 and 1.0.")
    if not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be between 1 and 16.")
    return epochs, learning_rate, batch_size


@register_model
class Mask2Former(InstanceSegmentation, CapabilityModel):
    """Fine-tunable full-image instance segmentation model using Mask2Former."""

    model_info = InstanceSegmentationModelInfo(
        registry_key="mask2former",
        name="Mask2Former (Swin-Small)",
        description=(
            "Mask2Former (Masked-attention Mask Transformer) model fine-tuned for universal "
            "instance segmentation using a Swin-Small backbone."
        ),
        usage_tip="Performs full-image instance segmentation and supports COCO dataset fine-tuning.",
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
            "publisher": "facebook",
            "architecture": "Mask2Former",
        },
        status="ready",
        trainable=True,
        training_parameters=[
            HyperParameter(
                key="epochs", label="Epochs", default_value=5,
                description="Number of training epochs.", type="int", min_value=1, max_value=50, step=1,
            ),
            HyperParameter(
                key="learning_rate", label="Learning Rate", default_value=0.0001,
                description="Learning rate for fine-tuning.", type="float",
            ),
            HyperParameter(
                key="batch_size", label="Batch Size", default_value=2,
                description="Batch size during training.", type="int", min_value=1, max_value=16, step=1,
            ),
        ],
    )

    _unpicklable_attrs = ("_processor", "_model")

    def __init__(self, checkpoint: str = "facebook/mask2former-swin-small-coco-instance") -> None:
        """Initialize a model that lazily loads its base checkpoint or MLflow artifact."""
        super().__init__()
        self.checkpoint = checkpoint
        self._processor: Optional[Mask2FormerImageProcessor] = None
        self._model: Optional[Mask2FormerForUniversalSegmentation] = None
        self._artifact_path: Optional[Path] = None
        self._label_mapping: Optional[LabelMapping] = None
        self._has_fine_tuned_weights = False

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore omitted live-weight attributes after MLflow unpickles this model.

        ``BaseModel.__getstate__`` intentionally removes the processor and Torch
        module because Transformers objects are not safely cloudpickleable.  They
        must still exist as ``None`` placeholders so ``load_context`` can lazily
        reconstruct them in a fresh API or Celery process.
        """
        super().__setstate__(state)
        self._processor = None
        self._model = None
        self._artifact_path = getattr(self, "_artifact_path", None)
        self._label_mapping = getattr(self, "_label_mapping", None)
        self._has_fine_tuned_weights = getattr(self, "_has_fine_tuned_weights", False)

    def load_context(self, context: Any = None) -> None:
        """Restore packaged fine-tuned weights before falling back to Hugging Face."""
        artifacts = getattr(context, "artifacts", {}) if context is not None else {}
        artifact_path = artifacts.get(_ARTIFACT_NAME)
        if artifact_path:
            self._artifact_path = Path(artifact_path)
            mapping_path = self._artifact_path / _LABEL_MAPPING_FILENAME
            if not mapping_path.is_file():
                raise RuntimeError("Registered Mask2Former artifact is missing its label mapping.")
            with mapping_path.open(encoding="utf-8") as mapping_stream:
                self._label_mapping = LabelMapping.from_dict(json.load(mapping_stream))
            self.model_info.label_ids = sorted(self._label_mapping.database_to_model)
            self._has_fine_tuned_weights = True
        self._load_weights()

    def _device(self) -> torch.device:
        """Return the execution device used consistently for training and inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_weights(self) -> None:
        """Load packaged weights when present, otherwise the configured base checkpoint."""
        if getattr(self, "_model", None) is not None and getattr(self, "_processor", None) is not None:
            return
        source = str(self._artifact_path) if self._artifact_path is not None else self.checkpoint
        logger.info("Loading Mask2Former weights from '%s'.", source)
        self._processor = Mask2FormerImageProcessor.from_pretrained(
            source, ignore_index=_IGNORE_INDEX, do_reduce_labels=False
        )
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(source)
        self._model.to(self._device())
        self._model.eval()

    def _configure_trainable_head(self, label_mapping: LabelMapping) -> None:
        """Replace the COCO classifier with a head sized for selected dataset labels."""
        model_labels = {
            model_index: f"db_{database_id}: {label_mapping.id2label[model_index]}"
            for model_index, database_id in label_mapping.model_to_database.items()
        }
        source = str(self._artifact_path) if self._artifact_path is not None else self.checkpoint
        self._processor = Mask2FormerImageProcessor.from_pretrained(
            source, ignore_index=_IGNORE_INDEX, do_reduce_labels=False
        )
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
            source,
            num_labels=len(label_mapping.database_to_model),
            id2label=model_labels,
            label2id={label: model_index for model_index, label in model_labels.items()},
            ignore_mismatched_sizes=True,
        )
        self._model.to(self._device())
        self._label_mapping = label_mapping
        self.model_info.label_ids = sorted(label_mapping.database_to_model)

    @staticmethod
    def _move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        """Move processor tensors, including per-image label lists, to ``device``."""
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            elif isinstance(value, list):
                moved[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
            else:
                moved[key] = value
        return moved

    def segment_instances(
        self, request: InstanceSegmentationRequest, params: dict[str, Any] | None = None
    ) -> list[Contour]:
        """Run RGB instance inference and translate predicted classes to database labels."""
        if self._label_mapping is None:
            raise RuntimeError(
                "Mask2Former inference requires a persisted database label mapping."
            )
        self._load_weights()
        if self._model is None or self._processor is None:
            raise RuntimeError("Mask2Former model or processor failed to initialize.")
        params = params or {}
        try:
            threshold = float(params.get("threshold", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("Instance segmentation threshold must be numeric.") from exc
        if not math.isfinite(threshold):
            raise ValueError("Instance segmentation threshold must be finite.")
        threshold = min(max(threshold, 0.0), 1.0)

        image_bgr = request.image
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Instance segmentation expects a three-channel image.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        inputs = self._move_inputs_to_device(
            self._processor(images=image_rgb, return_tensors="pt"), self._device()
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        processed = self._processor.post_process_instance_segmentation(
            outputs, target_sizes=[(height, width)], threshold=threshold
        )[0]
        raw_segmentation = processed["segmentation"]
        segmentation = (
            raw_segmentation.detach().cpu().numpy()
            if isinstance(raw_segmentation, torch.Tensor)
            else np.asarray(raw_segmentation)
        )
        requested_label_id = int(request.label.id) if request.label is not None else None

        contours: list[Contour] = []
        for segment in processed["segments_info"]:
            score = float(segment.get("score", 1.0))
            try:
                raw_model_label_id = segment["label_id"]
                if (
                    isinstance(raw_model_label_id, bool)
                    or not isinstance(raw_model_label_id, (int, np.integer))
                ):
                    raise TypeError("Model class indices must be integers.")
                model_label_id = int(raw_model_label_id)
                database_label_id = self._label_mapping.model_to_database[model_label_id]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Mask2Former prediction referenced an unknown model class index."
                ) from exc
            if requested_label_id is not None and database_label_id != requested_label_id:
                continue
            if score < threshold:
                continue
            binary_mask = (segmentation == segment["id"]).astype(np.uint8)
            if not np.any(binary_mask):
                continue
            contour = Contour.from_binary_mask(
                binary_mask=binary_mask,
                only_return_biggest_contour=True,
                confidence=score,
                added_by=request.model_registry_key,
            )
            if contour is not None:
                contour.label_id = database_label_id
                contours.append(contour)
        return contours

    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs: Any) -> None:
        """Fine-tune a custom-head Mask2Former model on selected COCO label instances."""
        epochs, learning_rate, batch_size = validate_hyperparameters(request.hyper_parameter)
        label_mapping = LabelMapping.from_selected_labels(request.labels)
        dataset = CocoInstanceDataset(
            request.annotation_file_url, request.image_folder_path, label_mapping
        )
        self._configure_trainable_head(label_mapping)
        if self._model is None or self._processor is None:
            raise RuntimeError("Mask2Former model or processor failed to initialize.")

        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dataset_id": request.dataset_id,
            "selected_database_label_ids": json.dumps(sorted(label_mapping.database_to_model)),
        })
        mlflow.set_tag("label_ids", json.dumps(sorted(label_mapping.database_to_model)))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=list)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        is_cancelled: Callable[[], bool] = kwargs.get("is_cancelled", lambda: False)
        progress_callback: Callable[[dict[str, int | float]], None] = kwargs.get(
            "progress_callback", lambda _: None
        )
        device = self._device()
        processed_batches = 0
        self._model.train()
        try:
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                epoch_batches = 0
                for samples in dataloader:
                    if is_cancelled():
                        raise TrainingCancelled("Training cancellation was requested.")
                    inputs = self._processor(
                        images=[sample.image for sample in samples],
                        segmentation_maps=[sample.instance_mask for sample in samples],
                        instance_id_to_semantic_id=[
                            sample.instance_id_to_semantic_id for sample in samples
                        ],
                        return_tensors="pt",
                    )
                    inputs = self._move_inputs_to_device(inputs, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._model(**inputs).loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError("Mask2Former produced a non-finite training loss.")
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach().cpu())
                    epoch_batches += 1
                    processed_batches += 1
                if epoch_batches == 0:
                    raise RuntimeError("Mask2Former training processed zero batches in an epoch.")
                average_loss = epoch_loss / epoch_batches
                mlflow.log_metrics({
                    "loss": average_loss,
                    "epoch": epoch,
                    "processed_batches": processed_batches,
                    "processed_samples": min(epoch * len(dataset), epochs * len(dataset)),
                }, step=epoch)
                progress_callback({"epoch": epoch, "loss": average_loss, "processed_batches": processed_batches})
                logger.info("Mask2Former epoch %d/%d finished with loss %.6f.", epoch, epochs, average_loss)
            if processed_batches == 0:
                raise RuntimeError("Mask2Former training processed zero batches.")
            self._has_fine_tuned_weights = True
        finally:
            self._model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_artifacts(self, tmp_dir: str) -> dict[str, str] | None:
        """Package fine-tuned weights, processor configuration, and label mapping for MLflow."""
        if not self._has_fine_tuned_weights or self._model is None or self._processor is None:
            return None
        if self._label_mapping is None:
            raise RuntimeError("Fine-tuned Mask2Former weights have no label mapping to package.")
        artifact_dir = Path(tmp_dir) / _ARTIFACT_NAME
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(artifact_dir)
        self._processor.save_pretrained(artifact_dir)
        with (artifact_dir / _LABEL_MAPPING_FILENAME).open("w", encoding="utf-8") as mapping_stream:
            json.dump(self._label_mapping.to_dict(), mapping_stream, sort_keys=True)
        return {_ARTIFACT_NAME: str(artifact_dir)}
