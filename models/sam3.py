"""Unified SAM 3 model: one class serving instance suggestion *and* prompted
segmentation.

This is the flagship of the model-centric merge. The former setup had SAM 3 in
two places: a hardened, working instance-suggestion implementation in
instance-discovery-service, and a broken prompted draft in prompted-seg-service
(it referenced ``prompts.noun_prompt`` / ``BoxPrompt.to_min_max_box()``, neither
of which exists, and never returned anything). Here there is a single SAM 3,
logged once, that advertises both tasks:

  * ``suggest_instances`` -- the hardened suggestion handler, carried over as-is.
  * ``segment_prompted``  -- a real, box-oriented prompted handler.

Because SAM 3's HF interface is box/text/mask-driven (it has no SAM 2-style point
decoder), the prompted handler localises with a single box derived from whatever
geometric prompt the user gave (box, else polygon/points/circle bounding box) and
returns the highest-scoring instance mask as the prompted object.
"""
from logging import getLogger
from typing import Any

import numpy as np
import torch
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.prompts import Prompts
from iquana_service_core import register_model

from models.base import CapabilityModel, InstanceSuggestion, PromptedSegmentation
from paths import HF_ACCESS_TOKEN

logger = getLogger(__name__)


@register_model
class SAM3(InstanceSuggestion, PromptedSegmentation, CapabilityModel):
    """SAM 3 (Meta) as a multi-task model: instance suggestion + prompted segmentation."""

    # InstanceSuggestion is listed first so the legacy single ``task`` tag (and the
    # ModelInfo subclass chosen on read) stays "instance-suggestion", matching how
    # SAM 3 was originally registered. Both tasks are advertised via ``task_*`` tags.
    model_info = ModelInfo(
        registry_key="sam3",
        name="SAM 3",
        description=(
            "SAM 3 is a unified foundation model for promptable segmentation in images "
            "and videos. It supports text and visual prompts including boxes and masks."
        ),
        usage_tip=(
            "Instance suggestion: provide one or more positive exemplar masks; an optional "
            "concept label guides text-prompted detection. Prompted segmentation: provide a "
            "box (or a polygon/points, whose bounding box is used) to segment one object. "
            "Tune `threshold` (detection sensitivity, default 0.3 -- lower finds more) and "
            "`mask_threshold` per request via params."
        ),
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
            "publisher": "meta-ai",
            "threshold": "0.3",
            # Prompted-surface descriptors (stored as tags because a single model
            # serves several tasks; a per-task ModelInfo is a Phase-1 toolbox item).
            "prompt_types_supported": "['box', 'polygon', 'point']",
            "refinement_supported": "false",
        },
        status="ready",
        trainable=False,
    )

    # Live HF objects can't be cloudpickled (transformers attaches ContextVar-backed
    # forward hooks). They are stripped from the pickle and rebuilt in ``load_context``.
    _unpicklable_attrs = ("model", "processor")

    # SAM 3 scores are sigmoid(class) * sigmoid(presence) -- a product of two
    # probabilities, so they sit low. The HF-calibrated default is 0.3; anything
    # near 0.5 over-filters and the model "finds almost nothing".
    def __init__(self, threshold: float = 0.3, mask_threshold: float = 0.5, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.threshold = threshold
        # Binarization point for each kept instance's mask. Note: 0.0 marks the whole
        # frame as foreground -- raise toward 0.5 for tight per-instance masks.
        self.mask_threshold = mask_threshold
        self._load_model()

    def _load_model(self):
        """Load the (pretrained) SAM 3 weights from the Hub. Reused on first init and
        when MLflow rebuilds the model in ``load_context`` after unpickling."""
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=HF_ACCESS_TOKEN).to(self.device)

    def load_context(self, context):
        self._load_model()

    # -- capability handler: instance suggestion ---------------------------- #
    def suggest_instances(
        self, request, params: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # Per-request overrides fall back to the values set at construction.
        params = params or {}
        threshold = params.get("threshold", self.threshold)
        mask_threshold = params.get("mask_threshold", self.mask_threshold)

        # Positive exemplar masks -> boxes with label 1; negative exemplars -> label 0.
        # SAM 3's geometry encoder uses these labels to push/pull the concept, so
        # dropping negatives discards a signal the caller explicitly provided.
        bboxes = request.get_bboxes(format="xyxy", relative_coordinates=False)
        labels = [1] * len(bboxes)
        for negative in request.negative_exemplars or []:
            bboxes.append(negative.get_as_bbox(relative_coords=False))
            labels.append(0)

        # Labels must be a LongTensor of shape (batch, num_boxes); an optional
        # concept label adds a text prompt, otherwise the "visual" token is used.
        bbox_labels = torch.tensor([labels], dtype=torch.int64)

        inputs = self.processor(
            images=[request.image],
            text=request.concept.name if request.concept is not None else "visual",
            input_boxes=[bboxes],
            input_boxes_labels=bbox_labels,
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

        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return masks, scores

    # -- capability handler: prompted segmentation -------------------------- #
    def segment_prompted(
        self, request, params: dict[str, Any] | None = None
    ) -> list[Contour]:
        """Segment one prompted object using a single localisation box."""
        params = params or {}
        threshold = params.get("threshold", self.threshold)
        mask_threshold = params.get("mask_threshold", self.mask_threshold)

        image = request.image
        box = self._prompt_to_box(request.prompts, image.shape)
        if box is None:
            logger.warning("SAM3 prompted request had no usable geometric prompt; returning nothing.")
            return []

        inputs = self.processor(
            images=[image],
            text="visual",
            input_boxes=[[box]],
            input_boxes_labels=torch.tensor([[1]], dtype=torch.int64),
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

        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if len(scores) == 0:
            return []

        # One prompted object: keep the highest-scoring instance the box elicited.
        best = int(np.argmax(scores))
        contour = Contour.from_binary_mask(
            binary_mask=masks[best].astype(np.uint8),
            only_return_biggest_contour=True,
            confidence=float(scores[best]),
            added_by=request.model_registry_key,
        )
        return [contour] if contour is not None else []

    @staticmethod
    def _prompt_to_box(prompts: Prompts, image_shape) -> list[float] | None:
        """Derive a single pixel-space ``[xmin, ymin, xmax, ymax]`` localisation box.

        Prefers an explicit box; otherwise falls back to the bounding box of a
        polygon, point set, or circle. Returns ``None`` if no geometric prompt was
        given. Coordinates in the request are normalised [0, 1]; SAM 3 expects pixels.
        """
        height, width = image_shape[0], image_shape[1]

        rel: tuple[float, float, float, float] | None = None
        if prompts.box_prompt:
            rel = tuple(prompts.box_prompt.xyxy)
        elif prompts.polygon_prompt and prompts.polygon_prompt.vertices:
            xs = [v[0] for v in prompts.polygon_prompt.vertices]
            ys = [v[1] for v in prompts.polygon_prompt.vertices]
            rel = (min(xs), min(ys), max(xs), max(ys))
        elif prompts.point_prompts:
            xs = [p.x for p in prompts.point_prompts]
            ys = [p.y for p in prompts.point_prompts]
            rel = (min(xs), min(ys), max(xs), max(ys))
        elif prompts.circle_prompt:
            c = prompts.circle_prompt
            rel = (c.center_x - c.radius, c.center_y - c.radius,
                   c.center_x + c.radius, c.center_y + c.radius)

        if rel is None:
            return None

        xmin, ymin, xmax, ymax = rel
        return [
            max(0.0, xmin) * width,
            max(0.0, ymin) * height,
            min(1.0, xmax) * width,
            min(1.0, ymax) * height,
        ]
