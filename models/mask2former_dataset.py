"""COCO loading and label-space conversion for Mask2Former fine-tuning.

COCO category IDs originate in the IQUANA database and are therefore sparse.
Mask2Former expects contiguous semantic class indices, so this module owns the
lossless conversion between the two spaces and the validation of exported data.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from torch.utils.data import Dataset

INSTANCE_IGNORE_INDEX = 255


def split_dataset_indices(dataset_size: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Return deterministic, disjoint train and validation image indices."""
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative.")
    indices = list(range(dataset_size))
    if dataset_size < 2:
        return indices, []

    validation_size = min(dataset_size - 1, max(1, round(dataset_size * 0.2)))
    validation_indices = set(random.Random(seed).sample(indices, validation_size))
    train_indices = [index for index in indices if index not in validation_indices]
    return train_indices, sorted(validation_indices)


class CocoTrainingDataError(ValueError):
    """Raised when an exported COCO training dataset is incomplete or invalid."""


@dataclass(frozen=True)
class LabelMapping:
    """Convert sparse database label IDs to contiguous model class indices."""

    database_to_model: dict[int, int]
    model_to_database: dict[int, int]
    id2label: dict[int, str]

    @classmethod
    def from_selected_labels(cls, labels: Sequence[Any]) -> "LabelMapping":
        """Build a deterministic mapping from the selected IQUANA labels."""
        labels_by_id = {int(label.id): str(label.name) for label in labels}
        if not labels_by_id:
            raise CocoTrainingDataError("At least one label must be selected for training.")
        database_ids = sorted(labels_by_id)
        database_to_model = {
            database_id: model_index
            for model_index, database_id in enumerate(database_ids)
        }
        return cls(
            database_to_model=database_to_model,
            model_to_database={
                model_index: database_id
                for database_id, model_index in database_to_model.items()
            },
            id2label={
                model_index: labels_by_id[database_id]
                for database_id, model_index in database_to_model.items()
            },
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelMapping":
        """Restore a mapping persisted alongside a registered model artifact."""
        return cls(
            database_to_model={
                int(database_id): int(model_index)
                for database_id, model_index in payload["database_to_model"].items()
            },
            model_to_database={
                int(model_index): int(database_id)
                for model_index, database_id in payload["model_to_database"].items()
            },
            id2label={
                int(model_index): str(label_name)
                for model_index, label_name in payload["id2label"].items()
            },
        )

    def to_dict(self) -> dict[str, dict[str, int | str]]:
        """Return a JSON-safe representation suitable for an MLflow artifact."""
        return {
            "database_to_model": self.database_to_model,
            "model_to_database": self.model_to_database,
            "id2label": self.id2label,
        }


@dataclass(frozen=True)
class CocoSample:
    """One decoded RGB image and its instance-id training target."""

    image: np.ndarray
    instance_mask: np.ndarray
    instance_id_to_semantic_id: dict[int, int]


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Decode one COCO polygon or RLE segmentation into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        polygons = segmentation
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float32)
            if points.size < 6 or points.size % 2:
                continue
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)], 1)
        return mask
    if not isinstance(segmentation, dict):
        raise CocoTrainingDataError("COCO annotation has no polygon or RLE segmentation.")

    try:
        from pycocotools import mask as mask_utils

        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    except (TypeError, ValueError, ImportError) as exc:
        raise CocoTrainingDataError("COCO RLE segmentation could not be decoded.") from exc
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise CocoTrainingDataError("Decoded COCO RLE dimensions do not match its image.")
    return np.asarray(decoded, dtype=np.uint8)


class CocoInstanceDataset(Dataset[CocoSample]):
    """Validated selected-label COCO dataset for Mask2Former.

    Parent contours are excluded by the gateway's ``contour_selection='leaves'``
    export. If independent leaf annotations overlap, their stable COCO order is
    used: later annotations own overlapping pixels because Mask2Former receives a
    single instance-ID map rather than overlapping instance targets.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_folder: str | Path,
        label_mapping: LabelMapping,
    ) -> None:
        """Load and validate selected-label COCO metadata before training starts."""
        annotation_path = Path(annotation_file)
        if not annotation_path.is_file():
            raise CocoTrainingDataError("COCO annotation file does not exist.")
        try:
            with annotation_path.open(encoding="utf-8") as annotation_stream:
                coco = json.load(annotation_stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise CocoTrainingDataError("COCO annotation file is not valid JSON.") from exc
        if not isinstance(coco, dict):
            raise CocoTrainingDataError("COCO annotation data must be a JSON object.")

        selected_ids = set(label_mapping.database_to_model)
        categories = coco.get("categories")
        images = coco.get("images")
        annotations = coco.get("annotations")
        if not all(isinstance(value, list) for value in (categories, images, annotations)):
            raise CocoTrainingDataError("COCO data must contain images, annotations, and categories lists.")
        category_ids = {category.get("id") for category in categories if isinstance(category, dict)}
        missing_categories = selected_ids.difference(category_ids)
        if missing_categories:
            raise CocoTrainingDataError(
                f"Selected labels are absent from the COCO categories: {sorted(missing_categories)}."
            )

        self._image_folder = Path(image_folder)
        image_by_id = {
            image.get("id"): image
            for image in images
            if isinstance(image, dict) and image.get("id") is not None
        }
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if annotation.get("category_id") not in selected_ids:
                continue
            image_id = annotation.get("image_id")
            if image_id not in image_by_id:
                raise CocoTrainingDataError("COCO annotation references an unknown image.")
            annotations_by_image.setdefault(image_id, []).append(annotation)

        self._samples = [
            (image_by_id[image_id], image_annotations)
            for image_id, image_annotations in annotations_by_image.items()
            if image_annotations
        ]
        if not self._samples:
            raise CocoTrainingDataError("No training images remain after selected-label filtering.")

        for image, _ in self._samples:
            image_path = self._image_folder / str(image.get("file_name", ""))
            if not image_path.is_file():
                raise CocoTrainingDataError("A COCO training image is missing from the image folder.")
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise CocoTrainingDataError("A COCO training image could not be decoded.")
        self._label_mapping = label_mapping

    def __len__(self) -> int:
        """Return the number of images containing a selected-label instance."""
        return len(self._samples)

    def __getitem__(self, index: int) -> CocoSample:
        """Decode one image and construct its instance-id/semantic-class targets."""
        image_meta, annotations = self._samples[index]
        image_path = self._image_folder / str(image_meta["file_name"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise CocoTrainingDataError("A COCO training image could not be decoded.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        # Mask2Former expects background to be encoded as the processor's ignore
        # index, not as an instance with semantic class zero.  This excludes it
        # before the processor resolves instance IDs to trainable classes.
        instance_mask = np.full((height, width), INSTANCE_IGNORE_INDEX, dtype=np.int32)
        instance_id_to_semantic_id: dict[int, int] = {}

        for instance_id, annotation in enumerate(annotations, start=1):
            foreground = _decode_segmentation(annotation.get("segmentation"), height, width)
            if not np.any(foreground):
                continue
            instance_mask[foreground > 0] = instance_id
            instance_id_to_semantic_id[instance_id] = self._label_mapping.database_to_model[
                int(annotation["category_id"])
            ]

        if not instance_id_to_semantic_id:
            raise CocoTrainingDataError("A selected-label training image has no valid instance masks.")
        return CocoSample(
            image=image_rgb,
            instance_mask=instance_mask,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
        )
