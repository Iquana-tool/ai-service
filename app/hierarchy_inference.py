"""Task-independent parent crops for concept segmentation.

``hierarchy_regions`` belongs to the service, never to a provider's parameter
contract. Native hierarchy providers declare ``hierarchy_input=parent_regions``
and retain their own reference-sharing loop; other providers see one focused
parent at a time. Every returned geometry and RLE uses the original image size.
"""
from collections.abc import Mapping

import cv2
import numpy as np
from pydantic import PrivateAttr
from pycocotools import mask as mask_utils

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.networking.http.services import (
    CrossImageExemplar, CrossImageSuggestionRequest, InstanceSegmentationRequest, InstanceSuggestionRequest,
)
from models import concat_ops, hierarchy_ops


class _LocalImage:
    """Keep crop pixels local without inserting synthetic URLs into shared caches."""
    _local_image: np.ndarray = PrivateAttr()

    @property
    def image(self):
        return self._local_image


class _InstanceRequest(_LocalImage, InstanceSegmentationRequest):
    pass


class _SuggestionRequest(_LocalImage, InstanceSuggestionRequest):
    pass


class _CrossImageRequest(_LocalImage, CrossImageSuggestionRequest):
    pass


_REQUESTS = {
    "instance-segmentation": _InstanceRequest,
    "instance-suggestion": _SuggestionRequest,
    "cross-image-suggestion": _CrossImageRequest,
}
for _schema in _REQUESTS.values():
    _schema.model_rebuild(_types_namespace={"CrossImageExemplar": CrossImageExemplar})


def _request(request, task, image, params, **updates):
    data = request.model_dump()
    data.update(parameters=params, **updates)
    local = _REQUESTS[task].model_validate(data)
    local._local_image = image
    return local


def _decode(rle, hw):
    try:
        if not isinstance(rle, Mapping) or list(rle.get("size", [])) != list(hw):
            raise ValueError("RLE must have the original image dimensions.")
        decoded = mask_utils.decode(dict(rle))
        if decoded.shape != tuple(hw):
            raise ValueError("RLE must encode one two-dimensional mask.")
        return decoded.astype(bool)
    except Exception as exc:
        raise ValueError(f"Invalid hierarchy/prediction mask: {exc}") from exc


def _encode(mask):
    return BinaryMask.from_numpy_array(mask).rle_mask


def _regions(raw, hw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("Use hierarchy requires at least one saved parent region.")
    regions, seen = [], set()
    for item in raw:
        parent_id = item.get("parent_id") if isinstance(item, Mapping) else None
        if isinstance(parent_id, bool) or not isinstance(parent_id, int) or parent_id <= 0:
            raise ValueError("Each hierarchy region requires a saved parent_id.")
        if parent_id in seen:
            raise ValueError("Hierarchy parent_ids must be unique.")
        mask = _decode(item.get("mask"), hw)
        if not mask.any():
            raise ValueError("Hierarchy parent regions must not be empty.")
        seen.add(parent_id)
        regions.append((parent_id, mask))
    return regions


def _predict(model, request, params):
    data = request if params["task"] == "instance-segmentation" else [request]
    return model.predict(data, params)


def _suggestion_crop(request, crop, image):
    """Borrow every exemplar, including negatives on a different parent.

    The reference context is placed beside the focused target at native size.
    This keeps the instance-suggestion contract intact even for providers that
    lack a cross-image task. Reference-side detections are discarded later.
    """
    positives = [np.asarray(m).astype(bool) for m in request.positive_exemplar_masks]
    negatives = [np.asarray(m).astype(bool) for m in request.negative_exemplar_masks]
    masks = positives + negatives
    # Spatial exemplars are optional for some contracts (text, embeddings,
    # autonomous models, or instances with min_units=0). Leave their conditioning
    # intact and let the provider validate it instead of imposing SAM's needs.
    if not masks:
        return crop.image, [], []
    if any(m.shape != image.shape[:2] or not m.any() for m in masks):
        raise ValueError("Suggestion exemplars must be nonempty original-image masks.")
    reference = hierarchy_ops.focus_crop(
        image, np.logical_or.reduce(masks), padding=0.05, blur=0,
    )
    plan = concat_ops.plan_layout(crop.image.shape[:2], [reference.image.shape[:2]])
    canvas = concat_ops.composite_image(crop.image, [reference.image], plan)
    x, y, w, h = plan.exemplar_xywh[0]

    def shifted(mask):
        out = np.zeros(canvas.shape[:2], dtype=bool)
        out[y:y + h, x:x + w] = reference.to_crop(mask)
        return BinaryMask.from_numpy_array(out)

    return canvas, [shifted(m) for m in positives], [shifted(m) for m in negatives]


def _as_dict(contour):
    return contour.model_dump(mode="json") if hasattr(contour, "model_dump") else dict(contour)


def _contour_mask(contour, hw):
    h, w = hw
    points = np.column_stack((contour.get("x", []), contour.get("y", [])))
    mask = np.zeros(hw, dtype=np.uint8)
    if len(points) >= 3:
        points = np.rint(points * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _detections(prediction, hw):
    """Yield masks with the original provider's contour and instance provenance."""
    if isinstance(prediction, tuple):
        masks, scores = prediction
        if len(masks) != len(scores):
            raise ValueError("Provider masks and scores must have equal lengths.")
        return [(np.asarray(m).astype(bool), {"confidence": float(s)}, {"score": float(s)})
                for m, s in zip(masks, scores)], {}
    metadata = dict(prediction.get("prediction_metadata") or {}) if isinstance(prediction, Mapping) else {}
    contours = prediction["contours"] if isinstance(prediction, Mapping) else prediction
    instances = {i["source_instance_id"]: i for i in metadata.get("instances", [])}
    def flattened(items):
        for item in items:
            payload = _as_dict(item)
            children = payload.pop("children", []) or []
            payload["children"] = []
            yield payload
            yield from flattened(children)

    grouped = {}
    for index, payload in enumerate(flattened(contours)):
        source_id = payload.get("source_instance_id")
        key = source_id if source_id is not None else ("contour", index)
        if key not in grouped:
            instance = dict(instances.get(source_id, {}))
            mask = _decode(instance["rle"], hw) if "rle" in instance else np.zeros(hw, dtype=bool)
            grouped[key] = (mask, payload, instance)
        if "rle" not in grouped[key][2]:
            grouped[key][0][:] |= _contour_mask(payload, hw)
    return list(grouped.values()), metadata


def _serialize(mask, payload, instance, parent_id, index, request):
    source_id = payload.get("source_instance_id") or instance.get("source_instance_id") or f"i{index}"
    source_id = f"h{parent_id}:{source_id}"
    score = float(payload.get("confidence", instance.get("score", 1)))
    h, w = mask.shape
    result = []
    for contour in Contour.from_binary_mask(mask, only_return_biggest_contour=False):
        contour.compute_path(image_width=w, image_height=h)
        result.append({
            **payload, **{k: getattr(contour, k) for k in ("x", "y", "path")},
            "parent_id": parent_id, "source_instance_id": source_id,
            "children": [],
            "added_by": payload.get("added_by") or request.model_registry_key,
            "confidence": score,
        })
    return result, {**instance, "source_instance_id": source_id, "score": score,
                    "parent_id": parent_id, "rle": _encode(mask)}


def predict_concept(model, request, params, model_tags=None):
    """Run the unchanged provider path, or service-owned hierarchy execution."""
    if "hierarchy_regions" not in params:
        return _predict(model, request, params)
    clean = dict(params)
    raw = clean.pop("hierarchy_regions")
    # Legacy provider ROI parameters must never coexist with a service-owned loop.
    clean.pop("parent_regions", None)
    task = clean["task"]
    image = np.asarray(request.image)
    hw = image.shape[:2]
    regions = _regions(raw, hw)
    contours, instances, runs = [], [], []
    native = (model_tags or {}).get("hierarchy_input") in ("parent_regions", "whole_image") or (
        (model_tags or {}).get("architecture") == "Mask2Former"
    ) or ((model_tags or {}).get("base_model_registry_key") == "mask2former")
    if native:
        if task == "instance-suggestion":
            local = _request(request, task, image, clean,
                             parent_regions=[BinaryMask.from_numpy_array(m) for _, m in regions])
        elif task == "instance-segmentation":
            local = _request(request, task, image, clean)
        else:
            raise ValueError(f"Native whole-image hierarchy is not supported for task {task}.")
        detections, metadata = _detections(_predict(model, local, clean), hw)
        for i, (mask, payload, instance) in enumerate(detections):
            if mask.shape != hw:
                raise ValueError("Native hierarchy provider returned masks at the wrong size.")
            owner = hierarchy_ops.assign_to_regions([mask], [m for _, m in regions])[0]
            if owner is None:
                continue
            parent_id, region = regions[owner]
            result, info = _serialize(mask & region, payload, instance, parent_id, i, request)
            contours.extend(result)
            instances.append(info)
        runs.append({"parent_ids": [p for p, _ in regions], "prediction_metadata": {
            k: v for k, v in metadata.items() if k != "instances"}})
    else:
        max_parent_iou = clean.get("max_parent_iou", None if task == "instance-segmentation" else 0.8)
        min_inside_frac = clean.get("min_inside_frac", 0.5)
        for parent_id, region in regions:
            crop = hierarchy_ops.focus_crop(image, region)
            updates = {}
            target_image = crop.image
            if task == "instance-suggestion":
                target_image, positives, negatives = _suggestion_crop(request, crop, image)
                updates = dict(positive_exemplars=positives, negative_exemplars=negatives,
                               parent_regions=[])
            local = _request(request, task, target_image, clean, **updates)
            detections, metadata = _detections(_predict(model, local, clean), target_image.shape[:2])
            runs.append({"parent_id": parent_id, "crop_box": crop.box, "prediction_metadata": {
                k: v for k, v in metadata.items() if k != "instances"}})
            for i, (mask, payload, instance) in enumerate(detections):
                if mask.shape != target_image.shape[:2]:
                    raise ValueError("Provider returned a mask at the wrong crop size.")
                if task == "instance-suggestion":
                    masks, _ = concat_ops.extract_target_masks(
                        [mask], (0, 0, crop.image.shape[1], crop.image.shape[0]),
                    )
                    if not masks:
                        continue
                    mask = masks[0]
                full, _ = hierarchy_ops.paste_back(
                    [mask], crop, hw,
                    min_inside_frac=min_inside_frac,
                    max_parent_iou=max_parent_iou,
                )
                if not full:
                    continue
                result, info = _serialize(full[0], payload, instance, parent_id, i, request)
                contours.extend(result)
                instances.append(info)
    # Keep checkpoint/model details available to existing clients while retaining
    # per-parent diagnostics instead of pretending one crop was the whole run.
    metadata = dict(runs[0]["prediction_metadata"]) if runs else {}
    metadata.update(hierarchy=True, roi_mode="hierarchy", instances=instances, hierarchy_runs=runs,
                    coordinate_space="native_image_rle; normalized contours")
    for key in ("tiles_total", "tiles_run", "candidates_raw", "candidates_kept"):
        if key in metadata:
            metadata[key] = sum(r["prediction_metadata"].get(key, 0) for r in runs)
    if "candidates_kept" in metadata:
        metadata["candidates_kept"] = len(instances)
    return {"contours": contours, "prediction_metadata": metadata}
