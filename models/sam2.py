"""SAM 2.1 prompted segmentation and propagation (four variants), on the capability interface.

Propagation (#121) uses SAM 2's video model: the same checkpoint, loaded into
``Sam2VideoModel`` the first time a stack is propagated, so annotators who never
propagate never pay for it.

Ported from the former prompted-seg-service. Changes from the original:
  * ``PromptedSegmentationModel`` -> ``PromptedSegmentation + CapabilityModel``;
    the old task-specific ``predict`` becomes the ``segment_prompted`` handler.
  * the hand-written ``__getstate__`` is replaced by declaring
    ``_unpicklable_attrs`` -- the toolbox ``BaseModel`` drops those on pickling and
    the base's lazy loader rebuilds them on first use, which is the same
    transformers-upgrade-proof behaviour with less code.
"""
from logging import getLogger
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.training import HyperParameter
from iquana_toolbox.schemas.networking.http.services import PropagatedMask, PropagationObject
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import PromptedSegmentationModelInfo
from iquana_toolbox.schemas.prompts import Prompts
from models.registry import register_model

from models.base import CapabilityModel, PromptedSegmentation, Propagation
from paths import hf_token

logger = getLogger(__name__)

# Foreground/background logit magnitude for the dense mask prompt. SAM2's prompt
# encoder feeds the mask straight through conv layers (no thresholding/scaling),
# so a raw {0, 1} mask is a far weaker hint than the low-res *logits* it expects.
# We map background -> -L and foreground -> +L to approximate that signal.
_MASK_PROMPT_LOGIT = 50.0

#: Frames in a row below the presence threshold after which propagation in that
#: direction stops: the object has ended. One low frame alone is often noise
#: (a shadow, a blink in a video) and the object comes back on the next.
_FRAMES_TO_CALL_IT_GONE = 2


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


class SAM2Prompted(PromptedSegmentation, Propagation, CapabilityModel):
    """Prompted segmentation and propagation backed by SAM 2.1 (HuggingFace Transformers).

    The class keeps its name although it now does more than prompting: logged
    MLflow artifacts refer to it by name.

    One class, four registered variants (tiny/small/base-plus/large). The variant
    is selected by ``registry_key``; the model derives its full ``model_info`` from
    the matching entry in :data:`_VARIANTS`.
    """

    # Live HF objects bake the installed transformers' module layout into the
    # cloudpickle artifact, so a later upgrade can break the unpickled object
    # (e.g. the ``num_pos_feats`` -> ``num_position_features`` rename). They are
    # dropped on pickling and rebuilt from the Hub on first use. Declaring them
    # here is also what tells ``CapabilityModel._ensure_weights`` this model has
    # weights to load at all.
    _unpicklable_attrs = ("model", "processor", "video_model", "video_processor")
    # Prompted segmentation needs only the image model; the video model is built
    # by ``_ensure_video_weights`` when a stack is first propagated.
    _core_weights = ("model", "processor")

    #: The registered variants this class builds. A subclass serving other
    #: checkpoints (MedSAM2) swaps in its own table.
    VARIANTS: dict[str, dict] = _VARIANTS

    def __init__(self, registry_key: str, device: str = "auto"):
        cfg = self.VARIANTS[registry_key]
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = cfg["checkpoint"]

        self.model_info = PromptedSegmentationModelInfo(
            registry_key=registry_key,
            name=cfg["name"],
            description=cfg["description"],
            usage_tip=cfg.get(
                "usage_tip",
                "Provide point, box and/or polygon prompts; supports iterative refinement using the "
                "previous mask. Polygons are fed to SAM2 as a dense mask prompt.",
            ),
            tags={
                "status": "ready",
                "pretrained": "true",
                "finetunable": "false",
                "model_size": cfg["model_size"],
                "inference_speed": cfg["inference_speed"],
                "accuracy_level": cfg["accuracy_level"],
                "requires_gpu": cfg["requires_gpu"],
                **cfg.get("tags", {}),
            },
            status="ready",
            trainable=False,
            prompt_types_supported=["point", "box", "polygon"],
            refinement_supported=True,
            input_contracts=[
                InputContract(
                    task="propagation",
                    conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                    parameters=[
                        HyperParameter(
                            key="presence_threshold", label="Presence threshold", type="float",
                            default_value=0.5, min_value=0.0, max_value=1.0, step=0.05,
                            description="Below this, the object counts as absent from a slice; "
                                        "two such slices in a row end the propagation there.",
                        ),
                    ],
                ),
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

    def _load_weights(self) -> None:
        """(Re)build the HF processor + model from the checkpoint on ``self.device``."""
        # Read now rather than at import, so a token pushed in after the service
        # started is used by the next load instead of only after a restart.
        token = hf_token()
        self.processor = Sam2Processor.from_pretrained(self.checkpoint, token=token)
        self.model = Sam2Model.from_pretrained(self.checkpoint, token=token).to(self.device)

    def _ensure_video_weights(self) -> None:
        """Build the video model the first time a stack is propagated."""
        if getattr(self, "video_model", None) is not None and getattr(self, "video_processor", None) is not None:
            return
        token = hf_token()
        self.video_processor = Sam2VideoProcessor.from_pretrained(self.checkpoint, token=token)
        self.video_model = Sam2VideoModel.from_pretrained(self.checkpoint, token=token).to(self.device)

    # -- capability handlers ------------------------------------------------- #
    def propagate(self, request, params: dict[str, Any] | None = None) -> Iterator[PropagatedMask]:
        """Carry each object from the frames it is drawn on through the requested range.

        A generator: each frame is yielded as soon as the model is through with it,
        so a caller can stream results while the rest of the stack is still being
        worked on. Nothing runs until the first item is asked for.
        """
        self._ensure_video_weights()
        threshold = float((params or {}).get("presence_threshold", 0.5))
        # Image URLs are read with OpenCV (BGR); the video processor expects RGB.
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in request.frames]
        start, end = request.frame_range
        for obj in request.objects:
            yield from self._propagate_object(frames, obj, start, end, request.direction, threshold)

    def _propagate_object(
        self, frames: list[np.ndarray], obj: PropagationObject, start: int, end: int,
        direction: str, threshold: float,
    ) -> Iterator[PropagatedMask]:
        """Run one object through the stack in its own inference session.

        One session per object, because the processor tracks "objects with new
        inputs" per call and an object without inputs on the first conditioning
        frame cannot be run -- so objects drawn on different slices cannot share
        a session. The cost is re-encoding the frames per object, which at OCT
        sizes is seconds.

        Every drawn frame is a conditioning frame, and SAM 2 attends to all of them
        on every step. So one pass forward from the first drawn frame and one pass
        backward from that same frame visit each frame once, already informed by
        drawings further along -- no frame is produced twice. Only "backward" alone
        starts from the last drawn frame, so the frames below it are covered.
        """
        height, width = frames[0].shape[:2]
        drawn = sorted({frame_mask.frame_index for frame_mask in obj.frame_masks})
        drawn = [index for index in drawn if index < len(frames)]
        if not drawn:
            return

        # The session spans the range plus any drawn frame outside it, so every
        # drawing conditions the result. Indices inside the session are offset.
        first = min(start, drawn[0])
        last = max(end, drawn[-1])
        session = self.video_processor.init_video_session(
            video=frames[first:last + 1],
            inference_device=self.device,
            # A long stack at 1024 px is gigabytes as float32; only the frame being
            # processed needs to be on the GPU.
            video_storage_device="cpu",
        )
        masks_by_frame = {fm.frame_index: fm.mask.mask for fm in obj.frame_masks}

        with torch.no_grad():
            # Each drawn frame becomes a conditioning frame. The forward pass on it
            # writes it into the session's memory, which propagation requires.
            for index in drawn:
                self.video_processor.add_inputs_to_inference_session(
                    session, frame_idx=index - first, obj_ids=[obj.object_id],
                    input_masks=[masks_by_frame[index].astype(np.float32)],
                )
                self.video_model(inference_session=session, frame_idx=index - first)

            passes = []
            if direction in ("both", "forward"):
                passes.append((drawn[0], False))
            if direction == "both":
                passes.append((drawn[0], True))
            elif direction == "backward":
                passes.append((drawn[-1], True))

            for origin, reverse in passes:
                low_in_a_row = 0
                for output in self.video_model.propagate_in_video_iterator(
                    session, start_frame_idx=origin - first, reverse=reverse,
                ):
                    index = output.frame_idx + first
                    presence = 1.0
                    if output.object_score_logits is not None:
                        presence = float(torch.sigmoid(output.object_score_logits.float()).reshape(-1)[0])
                    low_in_a_row = low_in_a_row + 1 if presence < threshold else 0
                    if index not in masks_by_frame and start <= index <= end:
                        mask = None
                        if presence >= threshold:
                            mask = self.video_processor.post_process_masks(
                                [output.pred_masks], original_sizes=[[height, width]], binarize=True,
                            )[0][0, 0].cpu().numpy().astype(np.uint8)
                            if not mask.any():
                                mask = None
                        yield PropagatedMask(
                            object_id=obj.object_id,
                            frame_index=index,
                            mask=BinaryMask.from_numpy_array(mask, score=presence) if mask is not None else None,
                            presence=min(1.0, max(0.0, presence)),
                        )
                    if low_in_a_row >= _FRAMES_TO_CALL_IT_GONE:
                        break

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
