"""Hierarchy-aware SAM 3 instance suggestion.

Plain SAM 3 suggestion runs once over the whole image. When the concept is a *child* label --
coral polyps on coral fragments, say -- that wastes most of SAM 3's fixed input resolution on
background: the fragments may cover a fifth of the frame and the polyps are tiny. This model
instead runs SAM 3 once **per parent object** (the existing contours of the concept's parent
label, sent as ``request.parent_regions``):

1. Crop the image to the parent's padded bbox and blur everything outside the parent mask, so
   resolution and attention go to the parent alone (:func:`models.hierarchy_ops.focus_crop`).
2. Prompt SAM 3 on that crop:

   * **Exemplar on this parent** -> standard SAM 3: the exemplar boxes (shifted into crop
     space) are the visual prompts.
   * **Exemplar on another parent** -> the cross-image concat workaround
     (:mod:`models.concat_ops`): the focused crop that holds the exemplars is pasted beside this
     crop and its exemplar boxes prompt across the seam; only target-side detections are kept.

3. Paste detections back to full-image coordinates, clipped to the parent. Detections that sit
   mostly on the blurred surroundings, or that are just the parent itself, are dropped.

With 4 fragments that is 4 forward passes. Without parent regions (root concept, or a backend
that doesn't send them) it degrades to one whole-image pass -- plain SAM 3 suggestion.
"""
from logging import getLogger
from typing import Any

import numpy as np
import torch
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.training import HyperParameter
from models.registry import register_model

from models import concat_ops, hierarchy_ops
from models.base import CapabilityModel, InstanceSuggestion
from paths import HF_ACCESS_TOKEN

logger = getLogger(__name__)

# An exemplar belongs to the parent holding at least this much of it.
_EXEMPLAR_IN_PARENT_FRAC = 0.5
# Context kept around exemplars that lie on no parent, when they must serve as a concat reference.
_ORPHAN_REFERENCE_PADDING = 1.0


@register_model
class SAM3Hierarchical(InstanceSuggestion, CapabilityModel):
    """SAM 3 instance suggestion that searches each parent object separately."""

    model_info = ModelInfo(
        registry_key="sam3_hierarchical",
        name="SAM 3 (hierarchy-aware)",
        description=(
            "SAM 3 instance suggestion that respects the label hierarchy: it runs once per "
            "existing object of the parent label, cropped to that object with everything "
            "outside it blurred, so small parts on large objects get SAM 3's full resolution."
        ),
        usage_tip=(
            "Annotate the parent objects first (e.g. all coral fragments), then give one or more "
            "examples of the part (e.g. a polyp). Each parent is searched separately; parents "
            "without an example borrow one from another parent. For root labels this behaves "
            "like plain SAM 3."
        ),
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "general",
            "publisher": "meta-ai",
        },
        status="ready",
        trainable=False,
        input_contracts=[
            InputContract(
                task="instance-suggestion",
                conditioning=ConditioningSpec(
                    kind="instances", unit="instance",
                    min_units=1, max_units=None,
                    user_selectable_count=False,
                ),
                parameters=[
                    HyperParameter(
                        key="threshold", label="Detection sensitivity", type="float",
                        default_value=0.3, min_value=0.0, max_value=1.0, step=0.05,
                        description="Lower values find more objects. "
                                    "Higher values keep only the clearest ones.",
                    ),
                    HyperParameter(
                        key="mask_threshold", label="Mask threshold", type="float",
                        default_value=0.5, min_value=0.0, max_value=1.0, step=0.05,
                        description="Lower values include more pixels. "
                                    "Higher values keep only the most certain pixels.",
                    ),
                    HyperParameter(
                        key="crop_padding", label="Crop margin", type="float",
                        default_value=0.05, min_value=0.0, max_value=0.5, step=0.05,
                        description="Extra space kept around each parent object, "
                                    "as a fraction of its size.",
                    ),
                    HyperParameter(
                        key="background_blur", label="Background blur", type="float",
                        default_value=0.03, min_value=0.0, max_value=0.2, step=0.01,
                        description="How strongly everything outside the parent object is blurred. "
                                    "0 turns blurring off.",
                    ),
                    HyperParameter(
                        key="min_target_frac", label="Target overlap", type="float",
                        default_value=0.5, min_value=0.0, max_value=1.0, step=0.05,
                        description="For parents without their own example: how much of a detected "
                                    "object must be on the searched parent rather than the borrowed "
                                    "example.",
                    ),
                ],
                notes="Searches inside the existing objects of the concept's parent label; "
                      "falls back to a whole-image pass for root labels.",
            ),
        ],
    )

    # Live HF objects can't be cloudpickled; rebuilt in ``load_context`` (see SAM3).
    _unpicklable_attrs = ("model", "processor")

    def __init__(self, threshold: float = 0.3, mask_threshold: float = 0.5, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._load_model()

    def _load_model(self):
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN).to(self.device)

    def load_context(self, context):
        self._load_model()

    # -- SAM 3 forward -------------------------------------------------------- #
    def _forward(
        self, image: np.ndarray, boxes: list[list[float]], labels: list[int], text: str,
        threshold: float, mask_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One SAM 3 pass with box visual prompts; returns ``(masks (N,H,W) bool, scores)``."""
        inputs = self.processor(
            images=[image],
            text=text,
            input_boxes=[boxes],
            input_boxes_labels=torch.tensor([labels], dtype=torch.int64),
            return_tensors="pt",
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        return results["masks"].cpu().numpy().astype(bool), results["scores"].cpu().numpy()

    @staticmethod
    def _prompts_on(crop: hierarchy_ops.FocusCrop, positives, negatives, x_off: int = 0, y_off: int = 0):
        """Box prompts for the exemplars clipped to ``crop``, shifted by a canvas offset."""
        boxes, labels = [], []
        for masks, label in ((positives, 1), (negatives, 0)):
            for mask in masks:
                box = hierarchy_ops.bbox_xyxy(crop.to_crop(mask), x_off, y_off)
                if box is not None:
                    boxes.append(box)
                    labels.append(label)
        return boxes, labels

    # -- capability handler: instance suggestion ------------------------------ #
    def suggest_instances(
        self, request, params: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        params = params or {}
        threshold = params.get("threshold", self.threshold)
        mask_threshold = params.get("mask_threshold", self.mask_threshold)
        padding = params.get("crop_padding", 0.05)
        blur = params.get("background_blur", 0.03)
        min_target_frac = params.get("min_target_frac", 0.5)

        image = request.image
        image_hw = image.shape[:2]
        text = request.concept.name if request.concept is not None else "visual"
        positives = [np.asarray(m).astype(bool) for m in request.positive_exemplar_masks]
        negatives = [np.asarray(m).astype(bool) for m in request.negative_exemplar_masks]
        parents = [
            m for m in (np.asarray(m).astype(bool) for m in getattr(request, "parent_region_masks", []))
            if m.any()
        ]

        if not parents:
            # Root concept (or no parent objects annotated yet): one whole-image pass.
            logger.info("No parent regions supplied; running whole-image SAM 3 suggestion.")
            whole = hierarchy_ops.FocusCrop(box=(0, image_hw[0], 0, image_hw[1]), image=image,
                                            parent=np.ones(image_hw, dtype=bool))
            boxes, labels = self._prompts_on(whole, positives, negatives)
            masks, scores = self._forward(image, boxes, labels, text, threshold, mask_threshold)
            return masks.astype(np.uint8), scores

        pos_parent = hierarchy_ops.assign_to_regions(positives, parents, _EXEMPLAR_IN_PARENT_FRAC)
        neg_parent = hierarchy_ops.assign_to_regions(negatives, parents, _EXEMPLAR_IN_PARENT_FRAC)
        crops = [hierarchy_ops.focus_crop(image, parent, padding, blur) for parent in parents]

        reference, ref_pos, ref_neg = self._reference(
            image, crops, positives, negatives, pos_parent, neg_parent
        )

        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []
        for p, crop in enumerate(crops):
            own_pos = [m for m, a in zip(positives, pos_parent) if a == p]
            own_neg = [m for m, a in zip(negatives, neg_parent) if a == p]

            if own_pos:
                # Exemplar on this parent: standard SAM 3 on the focused crop.
                boxes, labels = self._prompts_on(crop, own_pos, own_neg)
                masks, scores = self._forward(crop.image, boxes, labels, text, threshold, mask_threshold)
            else:
                # Exemplar elsewhere: paste the reference crop beside this one (concat workaround).
                plan = concat_ops.plan_layout(crop.image.shape[:2], [reference.image.shape[:2]])
                canvas = concat_ops.composite_image(crop.image, [reference.image], plan)
                x_off, y_off, _, _ = plan.exemplar_xywh[0]
                boxes, labels = self._prompts_on(reference, ref_pos, ref_neg, x_off, y_off)
                canvas_masks, canvas_scores = self._forward(
                    canvas, boxes, labels, text, threshold, mask_threshold
                )
                target_masks, kept = concat_ops.extract_target_masks(
                    canvas_masks, plan.target_xywh, min_target_frac=min_target_frac
                )
                masks, scores = target_masks, canvas_scores[kept]

            full_masks, kept = hierarchy_ops.paste_back(list(masks), crop, image_hw)
            all_masks.extend(full_masks)
            all_scores.extend(float(scores[i]) for i in kept)
            logger.debug("Parent %d (%s): %d suggestion(s).", p,
                         "standard" if own_pos else "concat", len(full_masks))

        if not all_masks:
            return np.zeros((0, *image_hw), dtype=np.uint8), np.zeros((0,))
        return np.stack(all_masks).astype(np.uint8), np.asarray(all_scores)

    @staticmethod
    def _reference(image, crops, positives, negatives, pos_parent, neg_parent):
        """The crop that lends its exemplars to parents without one of their own.

        The parent holding the most positive exemplars, pasted with all of them. SAM 3's concat
        trick only affords one reference tile (see ``SAM3.suggest_cross_image``). If no exemplar
        lies on any parent, a padded, unblurred crop around the exemplars stands in.
        """
        counts = [pos_parent.count(p) for p in range(len(crops))]
        best = int(np.argmax(counts))
        if counts[best] > 0:
            ref_pos = [m for m, a in zip(positives, pos_parent) if a == best]
            ref_neg = [m for m, a in zip(negatives, neg_parent) if a == best]
            return crops[best], ref_pos, ref_neg
        union = np.logical_or.reduce(positives)
        reference = hierarchy_ops.focus_crop(image, union, padding=_ORPHAN_REFERENCE_PADDING, blur=0.0)
        return reference, positives, []
