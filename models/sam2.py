"""SAM 2.1 prompted segmentation (four variants), on the capability interface.

Ported from the former prompted-seg-service. Changes from the original:
  * ``PromptedSegmentationModel`` -> ``PromptedSegmentation + CapabilityModel``;
    the old task-specific ``predict`` becomes the ``segment_prompted`` handler.
  * the hand-written ``__getstate__`` is replaced by declaring
    ``_unpicklable_attrs`` -- the toolbox ``BaseModel`` drops those on pickling and
    ``load_context`` rebuilds them, which is the same transformers-upgrade-proof
    behaviour with less code.
"""
from logging import getLogger
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize
from transformers import Sam2Model, Sam2Processor

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import PromptedSegmentationModelInfo
from iquana_toolbox.schemas.prompts import Prompts
from models.registry import register_model

from models.base import CapabilityModel, PromptedSegmentation
from paths import hf_token

logger = getLogger(__name__)

# Foreground/background logit magnitude for the dense mask prompt. SAM2's prompt
# encoder feeds the mask straight through conv layers (no thresholding/scaling),
# so a raw {0, 1} mask is a far weaker hint than the low-res *logits* it expects.
# We map background -> -L and foreground -> +L to approximate that signal.
_MASK_PROMPT_LOGIT = 50.0


# One entry per registered SAM 2.1 variant. The model is fully self-describing:
# each instance builds its own model_info from the entry keyed by registry_key.
_VARIANTS: dict[str, dict] = {
    "sam2-1-tiny": {
        "checkpoint": "facebook/sam2.1-hiera-tiny",
        "name": "SAM 2.1 Tiny",
        "description": (
            "Segment Anything Model 2.1 - Tiny variant. The smallest and fastest model "
            "with lowest memory footprint. Suitable for real-time inference but with "
            "reduced accuracy. Supports point and box prompts."
        ),
        "model_size": "tiny",
        "inference_speed": "fastest",
        "accuracy_level": "low",
        "requires_gpu": "false",
    },
    "sam2-1-small": {
        "checkpoint": "facebook/sam2.1-hiera-small",
        "name": "SAM 2.1 Small",
        "description": (
            "Segment Anything Model 2.1 - Small variant. Provides a good balance between "
            "inference speed and segmentation accuracy. Ideal for production use cases "
            "requiring reasonable performance. Supports point and box prompts."
        ),
        "model_size": "small",
        "inference_speed": "fast",
        "accuracy_level": "medium",
        "requires_gpu": "true",
    },
    "sam2-1-base-plus": {
        "checkpoint": "facebook/sam2.1-hiera-base-plus",
        "name": "SAM 2.1 Base+",
        "description": (
            "Segment Anything Model 2.1 - Base+ variant. Larger model with improved "
            "accuracy compared to the small variant. Good choice for accuracy-critical "
            "applications. Supports point and box prompts with refinement capabilities."
        ),
        "model_size": "base-plus",
        "inference_speed": "medium",
        "accuracy_level": "high",
        "requires_gpu": "true",
    },
    "sam2-1-large": {
        "checkpoint": "facebook/sam2.1-hiera-large",
        "name": "SAM 2.1 Large",
        "description": (
            "Segment Anything Model 2.1 - Large variant. The largest and most accurate "
            "SAM2 model. Best segmentation quality but requires more VRAM and slower "
            "inference. Recommended for offline and accuracy-critical workflows. Supports "
            "point and box prompts."
        ),
        "model_size": "large",
        "inference_speed": "slowest",
        "accuracy_level": "highest",
        "requires_gpu": "true",
    },
}


class SAM2Prompted(PromptedSegmentation, CapabilityModel):
    """Prompted segmentation backed by SAM 2.1 (HuggingFace Transformers).

    One class, four registered variants (tiny/small/base-plus/large). The variant
    is selected by ``registry_key``; the model derives its full ``model_info`` from
    the matching entry in :data:`_VARIANTS`.
    """

    # Live HF objects bake the installed transformers' module layout into the
    # cloudpickle artifact, so a later upgrade can break the unpickled object
    # (e.g. the ``num_pos_feats`` -> ``num_position_features`` rename). They are
    # dropped on pickling and rebuilt from the Hub in ``load_context``.
    _unpicklable_attrs = ("model", "processor")

    def __init__(self, registry_key: str, device: str = "auto"):
        cfg = _VARIANTS[registry_key]
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = cfg["checkpoint"]

        self.model_info = PromptedSegmentationModelInfo(
            registry_key=registry_key,
            name=cfg["name"],
            description=cfg["description"],
            usage_tip="Provide point, box and/or polygon prompts; supports iterative refinement using the "
                      "previous mask. Polygons are fed to SAM2 as a dense mask prompt.",
            tags={
                "status": "ready",
                "pretrained": "true",
                "finetunable": "false",
                "model_size": cfg["model_size"],
                "inference_speed": cfg["inference_speed"],
                "accuracy_level": cfg["accuracy_level"],
                "requires_gpu": cfg["requires_gpu"],
            },
            status="ready",
            trainable=False,
            prompt_types_supported=["point", "box", "polygon"],
            refinement_supported=True,
            input_contracts=[
                InputContract(
                    task="prompted-segmentation",
                    conditioning=ConditioningSpec(
                        kind="none",
                        user_selectable_count=False,
                    ),
                    # SAM2 has no user-exposed inference tunables beyond the
                    # geometric prompt itself.
                    parameters=[],
                ),
            ],
        )

        self._load_weights()

    def _load_weights(self) -> None:
        """(Re)build the HF processor + model from the checkpoint on ``self.device``."""
        # Read now rather than at import, so a token pushed in after the service
        # started is used by the next load instead of only after a restart.
        token = hf_token()
        self.processor = Sam2Processor.from_pretrained(self.checkpoint, token=token)
        self.model = Sam2Model.from_pretrained(self.checkpoint, token=token).to(self.device)

    def load_context(self, context: Any) -> None:
        """Runs once when MLflow loads the model; rebuild the HF objects fresh."""
        self._load_weights()

    # -- capability handler -------------------------------------------------- #
    def segment_prompted(
        self, request, params: dict[str, Any] | None = None
    ) -> list[Contour]:
        """Segment one prompted object; return it as a (single) candidate contour."""
        previous_mask = request.previous_mask.mask if request.previous_mask else None
        mask, score = self._segment(request.image, request.prompts, previous_mask)
        contour = Contour.from_binary_mask(
            binary_mask=mask,
            only_return_biggest_contour=True,  # one prompted object per request
            confidence=score,
            added_by=request.model_registry_key,
        )
        # An empty SAM2 mask yields no contour; skip it so it doesn't poison the
        # candidate list the backend validates downstream.
        return [contour] if contour is not None else []

    def _segment(self, image, prompts: Prompts, previous_mask=None) -> tuple[np.ndarray, float]:
        """Run SAM2 on a single image with point/box prompts; return (mask, score)."""
        # 1. Prepare prompts (coords are normalised in the request; scale to pixels).
        point_coords = None  # Image x Object x point x coords
        point_labels = None
        if prompts.point_prompts:
            point_coords = [[[[int(p.x * image.shape[1]), int(p.y * image.shape[0])] for p in prompts.point_prompts]]]
            point_labels = [[[p.label for p in prompts.point_prompts]]]

        box_coords = None
        if prompts.box_prompt:
            xmin, ymin, xmax, ymax = prompts.box_prompt.xyxy
            xmin = int(xmin * image.shape[1])
            ymin = int(ymin * image.shape[0])
            xmax = int(xmax * image.shape[1])
            ymax = int(ymax * image.shape[0])
            box_coords = [[[xmin, ymin, xmax, ymax]]]

        # A polygon/freehand prompt on its own is SAM2's weakest mode (dense mask
        # only), which often yields messy or empty masks. When no explicit box was
        # given, derive the polygon's bounding box as a reliable localisation prompt;
        # the polygon shape is still applied as the dense mask prompt below.
        if prompts.polygon_prompt and box_coords is None:
            xs = [v[0] for v in prompts.polygon_prompt.vertices]
            ys = [v[1] for v in prompts.polygon_prompt.vertices]
            box_coords = [[[
                int(min(xs) * image.shape[1]),
                int(min(ys) * image.shape[0]),
                int(max(xs) * image.shape[1]),
                int(max(ys) * image.shape[0]),
            ]]]

        # 2. Pre-process image + prompts (the processor handles resize/normalisation).
        inputs = self.processor(
            [image],
            input_points=point_coords,
            input_labels=point_labels,
            input_boxes=box_coords,
            return_tensors="pt",
        ).to(self.device)

        # Dense mask prompt. Two sources map onto SAM2's single dense input:
        #   * refinement -> the previous object mask, and
        #   * a polygon prompt -> rasterised to a mask (SAM2 has no native polygon
        #     encoder, so a polygon is fed as the dense mask prompt).
        # When both are present they are OR-combined.
        mask_prompt = None  # binary {0, 1} mask
        if previous_mask is not None:
            mask_prompt = previous_mask.astype(np.uint8)
        if prompts.polygon_prompt:
            # Match the polygon raster to the prior's grid when combining, else use
            # the model's 256x256 dense-prompt grid directly.
            height, width = mask_prompt.shape[:2] if mask_prompt is not None else (256, 256)
            polygon_mask = self._polygon_to_mask(prompts.polygon_prompt.vertices, height, width)
            mask_prompt = polygon_mask if mask_prompt is None else (
                np.logical_or(mask_prompt, polygon_mask).astype(np.uint8)
            )

        dense_mask = self._mask_to_logit_prompt(mask_prompt) if mask_prompt is not None else None

        # 3. Inference.
        with torch.no_grad():
            outputs = self.model(**inputs, input_masks=dense_mask, multimask_output=True)

        # 4. Post-process: upscale to original size and pick the best-scoring mask.
        batches = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
        scores = outputs.iou_scores.cpu().numpy().squeeze()
        best_index = int(np.argmax(scores))

        masks = batches[0].squeeze()
        final_mask = masks[best_index].numpy().astype(np.uint8) * 255
        return final_mask, float(scores[best_index])

    @staticmethod
    def _polygon_to_mask(vertices: list[list[float]], height: int, width: int) -> np.ndarray:
        """Rasterise a polygon (normalised vertices) into a filled binary mask."""
        points = np.array(
            [[int(round(x * width)), int(round(y * height))] for x, y in vertices],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], color=1)
        return mask

    def _mask_to_logit_prompt(self, mask: np.ndarray) -> torch.Tensor:
        """Turn a binary mask into SAM2's 256x256 dense logit mask prompt.

        The processor squashes the image to a square, so the mask prompt is
        squashed to the matching 256x256 grid (no aspect-ratio padding). Values
        are mapped from {0, 1} into logit space because the prompt encoder feeds
        the mask straight through conv layers without thresholding.
        """
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_t = resize(mask_t, [256, 256])  # bilinear; values land in [0, 1]
        logits = mask_t * (2.0 * _MASK_PROMPT_LOGIT) - _MASK_PROMPT_LOGIT
        return logits.to(self.device)


# --- Registered variants: each a zero-arg factory the catalog auto-discovers. ---
@register_model
def sam2_1_tiny() -> SAM2Prompted:
    return SAM2Prompted("sam2-1-tiny")


@register_model
def sam2_1_small() -> SAM2Prompted:
    return SAM2Prompted("sam2-1-small")


@register_model
def sam2_1_base_plus() -> SAM2Prompted:
    return SAM2Prompted("sam2-1-base-plus")


@register_model
def sam2_1_large() -> SAM2Prompted:
    return SAM2Prompted("sam2-1-large")
