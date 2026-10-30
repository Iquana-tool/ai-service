"""Verified YOLO11m-seg configuration C, seed 1, native-tile polyp provider."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import threading
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo, ModelClass
from iquana_toolbox.schemas.training import HyperParameter

from models.base import CapabilityModel, InstanceSegmentation
from models.registry import register_model
from models.cpsam_polyp import (
    _active_polyp_label_id, _contour_dicts, _device_from_name,
    _sha256, centre_owned, configured_checkpoint, decode_parent_regions, native_rle,
    parent_centroid_owned, runtime_status, select_tiles, tile_grid,
)

MODEL_KEY = "yolo_polyp_v3_c_seed1"
MODEL_NAME = "YOLO11m-seg · Polyp"
CHECKPOINT_ENV = "IQUANA_YOLO_POLYP_CHECKPOINT"
DEVICE_ENV = "IQUANA_YOLO_POLYP_DEVICE"
EXPECTED_CHECKPOINT_SHA256 = "7e9ac2eec159b4d227d1b8c6553de224a982f420f96b738362ef04bbfad08edf"
ULTRALYTICS_VERSION = "8.4.165"
DEFAULT_THRESHOLD = 0.25
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


POST_MERGE_IOU_THRESHOLD = 0.5
POST_MERGE_CONTAINMENT_THRESHOLD = 0.85


def suppress_duplicate_instances(
    candidates: list[dict],
    iou_threshold: float = POST_MERGE_IOU_THRESHOLD,
    containment_threshold: float = POST_MERGE_CONTAINMENT_THRESHOLD,
) -> list[dict]:
    """Suppress overlapping and nested duplicate masks after tile merging.

    Greedily keeps candidates ordered by descending confidence score.
    Suppresses a lower-scoring candidate if any higher-scoring kept candidate
    exhibits mask IoU >= iou_threshold OR containment >= containment_threshold,
    where containment is defined as intersection / min(area_i, area_j).
    """
    if len(candidates) <= 1:
        return candidates

    sorted_candidates = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)
    rles = [c["rle"] for c in sorted_candidates]
    bboxes = np.array([mask_utils.toBbox(r) for r in rles], dtype=np.float32)
    areas = np.array([float(mask_utils.area(r)) for r in rles], dtype=np.float32)

    boxes_xyxy = np.column_stack([
        bboxes[:, 0], bboxes[:, 1],
        bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1] + bboxes[:, 3]
    ])

    kept: list[dict] = []
    kept_indices: list[int] = []

    for idx, cand in enumerate(sorted_candidates):
        area = areas[idx]
        if area <= 0:
            continue

        if kept_indices:
            k_arr = np.array(kept_indices, dtype=np.int64)
            b = boxes_xyxy[idx]
            kb = boxes_xyxy[k_arr]

            ix0 = np.maximum(b[0], kb[:, 0])
            iy0 = np.maximum(b[1], kb[:, 1])
            ix1 = np.minimum(b[2], kb[:, 2])
            iy1 = np.minimum(b[3], kb[:, 3])

            inter_box = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
            candidate_indices = k_arr[inter_box > 0]
            suppressed = False

            for k_idx in candidate_indices:
                k_rle = rles[k_idx]
                k_area = areas[k_idx]

                iou = float(mask_utils.iou([rles[idx]], [k_rle], [0])[0, 0])
                if iou >= iou_threshold:
                    suppressed = True
                    break

                min_a = min(area, k_area)
                if min_a <= 0:
                    continue

                inter_mask = float(mask_utils.area(mask_utils.merge([rles[idx], k_rle], intersect=True)))
                containment = inter_mask / min_a
                if containment >= containment_threshold:
                    suppressed = True
                    break

            if suppressed:
                continue

        kept.append(cand)
        kept_indices.append(idx)

    return kept


def _configuration():
    checkpoint = configured_checkpoint(CHECKPOINT_ENV)
    if checkpoint is None:
        raise FileNotFoundError(f"Set {CHECKPOINT_ENV} to the YOLO C seed 1 checkpoint; there is no default path.")
    path = checkpoint.resolve()
    device = os.environ.get(DEVICE_ENV, "cuda:0").strip().lower()
    if device == "cuda":
        device = "cuda:0"
    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError(f"{DEVICE_ENV} must be cpu or cuda:N.")
    return str(path), device


@register_model
class YoloPolyp(InstanceSegmentation, CapabilityModel):
    model_info = InstanceSegmentationModelInfo(
        registry_key=MODEL_KEY, name=MODEL_NAME,
        description="Trained YOLO11m-seg C seed 1 polyp masks, using native 640/128 tiles.",
        usage_tip="Run on selected saved coral outlines or explicitly choose Whole image.",
        classes=[ModelClass(key="polyp", name="Polyp")],
        tags={"status": runtime_status(CHECKPOINT_ENV, "ultralytics"), "roi_input": "parent_regions",
              "maturity": "provisional", "pretrained": "true",
              "finetunable": "false", "architecture": "YOLO11m-seg", "model_version": "v3_C_seed1",
              "default_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256},
        status=runtime_status(CHECKPOINT_ENV, "ultralytics"), trainable=False,
        input_contracts=[InputContract(
            task="instance-segmentation",
            conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
            parameters=[HyperParameter(key="threshold", label="Confidence threshold", type="float",
                                       default_value=DEFAULT_THRESHOLD, min_value=0.0, max_value=1.0, step=0.01)],
        )],
    )
    _unpicklable_attrs = ("_yolo",)
    _passthrough_input_params = ("parent_regions",)

    def __init__(self):
        super().__init__()
        self._yolo = None
        self._configuration_identity = None

    def __setstate__(self, state):
        super().__setstate__(state)
        self._yolo = None
        self._configuration_identity = None

    def _load_weights(self):
        with _LOAD_LOCK:
            identity = _configuration()
            if self._yolo is not None:
                if identity != self._configuration_identity:
                    raise RuntimeError("YOLO checkpoint/device changed; restart the worker.")
                return
            path = Path(identity[0])
            if not path.is_file():
                raise FileNotFoundError(f"YOLO C seed 1 checkpoint missing: {path}. Set {CHECKPOINT_ENV}.")
            if _sha256(path) != EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError("YOLO C seed 1 checkpoint SHA-256 mismatch; stock weights are never substituted.")
            try:
                version = importlib.metadata.version("ultralytics")
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(f"Install the yolo-polyp extra (ultralytics=={ULTRALYTICS_VERSION}).") from exc
            if version != ULTRALYTICS_VERSION:
                raise RuntimeError(f"YOLO polyp requires ultralytics=={ULTRALYTICS_VERSION}; found {version}.")
            device = _device_from_name(identity[1])
            from ultralytics import YOLO
            model = YOLO(str(path), task="segment")
            if model.task != "segment" or dict(model.names) != {0: "polyp"}:
                raise RuntimeError("YOLO checkpoint must be a single-class polyp segmentation model.")
            model.to(device)
            self._yolo = model
            self._configuration_identity = identity

    def segment_instances(self, request, params=None):
        params = params or {}
        threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be finite and in [0, 1].")
        image = np.asarray(request.image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLO polyp requires a three-channel BGR image.")
        height, width = image.shape[:2]
        tiles_all = tile_grid(height, width)
        parent = decode_parent_regions(params.get("parent_regions"), height, width)
        tiles = select_tiles(tiles_all, parent)
        label_id = _active_polyp_label_id(request)
        self._ensure_weights()
        if self._configuration_identity != _configuration():
            raise RuntimeError("YOLO checkpoint/device changed; restart the worker.")
        raw_candidates = []
        with _INFERENCE_LOCK, torch.inference_mode():
            for tile in tiles:
                crop = image[tile.y0:tile.y0 + 640, tile.x0:tile.x0 + 640]
                result = self._yolo.predict(
                    crop, imgsz=640, conf=0.001, max_det=1000, iou=0.7,
                    retina_masks=True, device=self._configuration_identity[1],
                    half=False, augment=False, verbose=False, save=False,
                )[0]
                if result.masks is None:
                    continue
                masks = (result.masks.data > 0.5).cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                if len(scores) != len(masks):
                    raise RuntimeError("YOLO masks and confidence counts differ.")
                owned, cx, cy = centre_owned(masks, tile)
                if parent is not None:
                    owned &= parent_centroid_owned(parent, cx, cy)
                for index in np.flatnonzero(owned):
                    score = float(scores[index])
                    if not math.isfinite(score) or score < max(0.001, threshold):
                        continue
                    mask = masks[index]
                    rle = native_rle(mask, tile.x0, tile.y0, height, width)
                    raw_candidates.append({
                        "score": score,
                        "mask": mask,
                        "tile": tile,
                        "rle": rle,
                    })

        kept_candidates = suppress_duplicate_instances(raw_candidates)
        contours, instances = [], []
        for cand in kept_candidates:
            score = cand["score"]
            mask = cand["mask"]
            tile = cand["tile"]
            rle = cand["rle"]
            digest = hashlib.sha256(rle["counts"].encode("ascii")).hexdigest()[:16]
            source_id = f"{MODEL_KEY}:{height}x{width}:{digest}"
            contours.extend(_contour_dicts(
                mask, score=score, label_id=label_id, model_key=MODEL_KEY,
                source_instance_id=source_id, tile=tile,
                image_height=height, image_width=width,
            ))
            instances.append({"source_instance_id": source_id, "rle": rle, "score": score})

        return {"contours": contours, "prediction_metadata": {
            "model_key": MODEL_KEY, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "ultralytics_version": ULTRALYTICS_VERSION, "device": self._configuration_identity[1],
            "class_key": "polyp", "label_id": label_id, "tile_size": 640, "tile_overlap": 128,
            "candidate_score_floor": 0.001, "max_det_per_tile": 1000, "nms_iou": 0.7,
            "application_threshold": threshold, "roi_margin": 0,
            "roi_mode": "parent_regions" if parent is not None else "whole_image",
            "tiles_total": len(tiles_all), "tiles_run": len(tiles),
            "candidates_raw": len(raw_candidates), "candidates_kept": len(instances),
            "post_merge_suppression": {
                "iou_threshold": POST_MERGE_IOU_THRESHOLD,
                "containment_threshold": POST_MERGE_CONTAINMENT_THRESHOLD,
            },
            "instances": instances,
        }}
