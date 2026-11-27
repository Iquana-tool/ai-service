"""M2C: SAM 3 that learns one concept embedding per label from a few masks.

Zhou et al., "Mask to Concept: Auto-Promptable SAM3 via Efficient Test-Time
Concept Embedding Search for Few-Shot Annotation", MICCAI 2026.
https://github.com/Huster-Hq/M2C (code under the SAM licence, on top of SAM 3).

M2C keeps every SAM 3 weight frozen. It learns an offset ``phi`` that is added
to the token embeddings of a text prompt ("A polyp in a colonoscopy image"),
trained with SAM 3's detection and mask losses on a few annotated images. The
tuned prompt then finds the concept on new images without any click or box.
The authors publish no weights: ``phi`` is specific to a dataset, so here it is
the result of a training run. One ``phi`` (32 x 1024 floats) is learnt per
selected label and travels in the model's pickle.

The port runs on transformers' SAM 3. ``phi`` is added through a forward hook on
the CLIP token embedding, which matches the authors' ``input_embeds`` path:
the position embeddings and the attention mask stay as the tokenizer made them.
Each step runs the frozen vision encoder once per image, under ``no_grad``, and
back-propagates only through the text encoder and the DETR/mask heads.

Differences from the authors' code, all for simplicity:

* The loss is a compact version of SAM 3's: one-to-one Hungarian matching, no
  one-to-many branch, no auxiliary decoder-layer terms. The term weights are the
  ones from their ``prompt.yaml``.
* No random-resize augmentation: the processor's fixed 1008 px resize is used.
* No expert library. The paper keeps one ``phi`` per annotation round and picks
  one per image by an uncertainty score; here a training run gives one ``phi``
  per label.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from logging import getLogger
from typing import Any, Callable

import cv2
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import HyperParameter, InstanceSegmentationModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from models.registry import register_model

from models.base import CapabilityModel, InstanceSegmentation
from models.mask2former import TrainingCancelled
from models.mask2former_dataset import (
    CocoInstanceDataset,
    CocoTrainingDataError,
    LabelMapping,
    _decode_segmentation,
)
from paths import hf_token

logger = getLogger(__name__)

#: ``{label}`` is replaced by the label name. ``phi`` can only change the tokens
#: of the prompt, so a sentence gives it more room than a bare name.
_DEFAULT_PROMPT_TEMPLATE = "A {label} in an image"
#: Side of SAM 3's ``pred_masks`` for its 1008 px input: 4x the 72 x 72 patch grid.
_MASK_SIZE = 288

# Matcher costs and loss weights from the authors' ``configs/kvasir-seg/prompt.yaml``.
_COST_CLASS, _COST_BBOX, _COST_GIOU = 2.0, 5.0, 2.0
_LOSS_WEIGHTS = {
    "ce": 20.0, "presence": 20.0, "bbox": 5.0, "giou": 2.0,
    "mask": 200.0, "dice": 10.0, "semantic": 20.0, "semantic_dice": 30.0,
}
_FOCAL_ALPHA, _FOCAL_GAMMA = 0.25, 2.0
_SEMANTIC_FOCAL_ALPHA = 0.6


def validate_hyperparameters(hyperparameters: dict[str, Any]) -> tuple[int, float, int, str]:
    """Validate and normalize the M2C hyperparameters supplied by a client."""
    try:
        epochs = int(hyperparameters.get("epochs", 30))
        learning_rate = float(hyperparameters.get("learning_rate", 5e-4))
        batch_size = int(hyperparameters.get("batch_size", 4))
    except (TypeError, ValueError) as exc:
        raise ValueError("epochs, learning_rate, and batch_size must be numeric.") from exc
    template = str(hyperparameters.get("prompt_template") or _DEFAULT_PROMPT_TEMPLATE).strip()
    if not 1 <= epochs <= 200:
        raise ValueError("epochs must be between 1 and 200.")
    if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-1:
        raise ValueError("learning_rate must be finite and between 1e-6 and 0.1.")
    if not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be between 1 and 32.")
    if "{label}" not in template:
        raise ValueError("prompt_template must contain '{label}'.")
    return epochs, learning_rate, batch_size, template


def _cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """``(N, H, W)`` binary masks -> ``(N, 4)`` normalized xyxy boxes."""
    height, width = masks.shape[-2:]
    boxes = []
    for mask in masks:
        ys, xs = torch.nonzero(mask, as_tuple=True)
        boxes.append(torch.stack([xs.min() / width, ys.min() / height,
                                  (xs.max() + 1) / width, (ys.max() + 1) / height]))
    return torch.stack(boxes).float()


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean soft-dice loss over the first dimension."""
    probs = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probs * targets).sum(1)
    denominator = probs.sum(1) + targets.sum(1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


@torch.no_grad()
def match(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of ``Q`` queries to ``N`` ground-truth boxes (SAM 3's binary matcher).

    ``pred_logits`` is ``(Q,)``, boxes are normalized xyxy. Returns query and
    target indices of equal length ``min(Q, N)``.
    """
    prob = pred_logits.float().sigmoid()
    positive = _FOCAL_ALPHA * (1 - prob) ** _FOCAL_GAMMA * -(prob + 1e-8).log()
    negative = (1 - _FOCAL_ALPHA) * prob ** _FOCAL_GAMMA * -(1 - prob + 1e-8).log()
    cost_class = (positive - negative)[:, None]
    cost_bbox = torch.cdist(_cxcywh(pred_boxes.float()), _cxcywh(gt_boxes), p=1)
    cost_giou = -generalized_box_iou(pred_boxes.float(), gt_boxes)
    cost = _COST_CLASS * cost_class + _COST_BBOX * cost_bbox + _COST_GIOU * cost_giou
    rows, cols = linear_sum_assignment(cost.cpu().numpy())
    return rows, cols


def m2c_loss(outputs: Any, gt_masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """SAM 3 detection + mask loss for one image and one concept.

    ``outputs`` is a ``Sam3ImageSegmentationOutput`` with batch size 1. ``gt_masks``
    is ``(N, H, W)`` binary at the resolution of ``outputs.pred_masks``; ``N`` may
    be 0, which makes every query a negative and the concept absent.
    Returns the weighted terms; their sum is the training loss.
    """
    logits = outputs.pred_logits[0].float()
    boxes = outputs.pred_boxes[0].float()
    masks = outputs.pred_masks[0].float()
    gt_masks = gt_masks.to(masks.device, torch.float32)
    num_gt = gt_masks.shape[0]
    terms: dict[str, torch.Tensor] = {}

    targets = torch.zeros_like(logits)
    if num_gt:
        gt_boxes = _masks_to_boxes(gt_masks).to(boxes.device)
        rows, cols = match(logits, boxes, gt_boxes)
        targets[rows] = 1.0
        matched_boxes, matched_gt_boxes = boxes[rows], gt_boxes[cols]
        terms["bbox"] = F.l1_loss(_cxcywh(matched_boxes), _cxcywh(matched_gt_boxes))
        terms["giou"] = (1 - torch.diag(generalized_box_iou(matched_boxes, matched_gt_boxes))).mean()
        matched_masks, matched_gt_masks = masks[rows], gt_masks[cols]
        terms["mask"] = sigmoid_focal_loss(
            matched_masks, matched_gt_masks, alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA, reduction="mean"
        )
        terms["dice"] = _dice_loss(matched_masks, matched_gt_masks)
    terms["ce"] = sigmoid_focal_loss(
        logits, targets, alpha=_FOCAL_ALPHA, gamma=_FOCAL_GAMMA, reduction="sum"
    ) / max(num_gt, 1)
    if outputs.presence_logits is not None:
        present = torch.full_like(outputs.presence_logits.float(), float(num_gt > 0))
        terms["presence"] = F.binary_cross_entropy_with_logits(outputs.presence_logits.float(), present)
    if outputs.semantic_seg is not None:
        semantic = outputs.semantic_seg[0].float()
        union = gt_masks.amax(0, keepdim=True) if num_gt else torch.zeros_like(semantic)
        union = F.interpolate(union[None], size=semantic.shape[-2:], mode="nearest")[0]
        terms["semantic"] = sigmoid_focal_loss(
            semantic, union, alpha=_SEMANTIC_FOCAL_ALPHA, gamma=_FOCAL_GAMMA, reduction="mean"
        )
        terms["semantic_dice"] = _dice_loss(semantic, union)
    return {key: _LOSS_WEIGHTS[key] * value for key, value in terms.items()}


class ConceptDataset(CocoInstanceDataset):
    """The Mask2Former COCO dataset, but with one full mask per instance.

    Mask2Former's single instance-id map lets a child outline (a polyp) cut a hole
    into its parent (the coral fragment). M2C learns each label on its own, so it
    keeps the masks apart. They are shrunk to the mask-head resolution at once:
    hundreds of full-size masks per photo would not fit in memory.
    """

    def __init__(self, *args, mask_size: int = _MASK_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_size = mask_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[int, torch.Tensor]]:
        """Return the RGB image and ``{database label id: (N, S, S) bool masks}``."""
        image_meta, annotations = self._samples[index]
        image_bgr = cv2.imread(str(self._image_folder / str(image_meta["file_name"])), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise CocoTrainingDataError("A COCO training image could not be decoded.")
        height, width = image_bgr.shape[:2]
        masks: dict[int, list[np.ndarray]] = {}
        for annotation in annotations:
            mask = _decode_segmentation(annotation.get("segmentation"), height, width)
            small = cv2.resize(mask.astype(np.float32), (self.mask_size, self.mask_size),
                               interpolation=cv2.INTER_AREA) > 0.5
            if small.any():  # an instance thinner than a mask pixel vanishes
                masks.setdefault(int(annotation["category_id"]), []).append(small)
        return (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                {label_id: torch.from_numpy(np.stack(found)) for label_id, found in masks.items()})


@register_model
class M2C(InstanceSegmentation, CapabilityModel):
    """SAM 3 with a learnt text prompt per label, for whole-image instance segmentation."""

    model_info = InstanceSegmentationModelInfo(
        registry_key="m2c-sam3",
        name="M2C (SAM 3)",
        description=(
            "M2C (Mask-to-Concept, MICCAI 2026) - SAM 3 that learns a concept embedding for each "
            "label from a few annotated images, with all SAM 3 weights frozen. The trained model "
            "finds every instance of the labels on new images, without clicks or boxes."
        ),
        usage_tip=(
            "Train it on a few fully annotated images (5 to 20 are enough). Training takes minutes, "
            "because only the text prompt is learnt. Describe the domain in the prompt template, "
            "for example 'A {label} in a colonoscopy image'. Before training, the model runs plain "
            "SAM 3 with the label name."
        ),
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "true",
            "domain": "general",
            "publisher": "huster-hq",
            "architecture": "SAM 3",
            "license": "sam",
            "source": "https://github.com/Huster-Hq/M2C",
        },
        status="ready",
        trainable=True,
        training_parameters=[
            HyperParameter(
                key="epochs", label="Epochs", default_value=30, type="int",
                min_value=1, max_value=200, step=1,
                description="Passes over the training images.",
            ),
            HyperParameter(
                key="learning_rate", label="Learning Rate", default_value=0.0005, type="float",
                description="Start learning rate of the concept embedding. It decays to 1% of it.",
            ),
            HyperParameter(
                key="batch_size", label="Batch Size", default_value=4, type="int",
                min_value=1, max_value=32, step=1,
                description="Images per optimizer step.",
            ),
            HyperParameter(
                key="prompt_template", label="Prompt template", default_value=_DEFAULT_PROMPT_TEMPLATE,
                type="str",
                description="Text prompt that training starts from. {label} becomes the label name.",
            ),
        ],
        input_contracts=[
            InputContract(
                task="instance-segmentation",
                conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
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
                ],
            ),
        ],
    )

    _unpicklable_attrs = ("model", "processor")

    def __init__(self, threshold: float = 0.3, mask_threshold: float = 0.5, device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        #: database label id -> {"name", "prompt", "phi"}; ``phi`` is a (32, 1024) float32 array.
        self.concepts: dict[int, dict[str, Any]] = {}

    def _load_weights(self):
        token = hf_token()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(self.device).eval()
        self.model.requires_grad_(False)

    # -- text prompt with the learnt offset --------------------------------- #
    @contextmanager
    def _token_offset(self, phi: torch.Tensor | None):
        """Add ``phi`` to the CLIP token embeddings while the block runs."""
        if phi is None:
            yield
            return
        embedding = self.model.text_encoder.text_model.embeddings.token_embedding
        handle = embedding.register_forward_hook(lambda _m, _i, out: out + phi.to(out.dtype))
        try:
            yield
        finally:
            handle.remove()

    def _encode_prompt(self, prompt: str, phi: torch.Tensor | None):
        """Return ``(text_embeds, attention_mask)`` for ``Sam3Model.forward``."""
        tokens = self.processor(text=prompt, return_tensors="pt").to(self.device)
        with self._token_offset(phi):
            text_embeds = self.model.get_text_features(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True
            )
        return text_embeds, tokens["attention_mask"]

    def _vision(self, image_rgb: np.ndarray):
        inputs = self.processor(images=[image_rgb], return_tensors="pt").to(self.device)
        with torch.no_grad(), self._autocast():
            vision_embeds = self.model.get_vision_features(pixel_values=inputs["pixel_values"])
        return vision_embeds, inputs["original_sizes"].tolist()

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda"))

    # -- capability handler: instance segmentation -------------------------- #
    def segment_instances(
        self, request: InstanceSegmentationRequest, params: dict[str, Any] | None = None
    ) -> list[Contour]:
        """Find every instance of each learnt concept (or of ``request.label`` before training)."""
        params = params or {}
        threshold = params.get("threshold", self.threshold)
        mask_threshold = params.get("mask_threshold", self.mask_threshold)

        if self.concepts:
            concepts = {
                label_id: concept for label_id, concept in self.concepts.items()
                if request.label is None or int(request.label.id) == label_id
            }
        elif request.label is not None:
            label_id = int(request.label.id)
            concepts = {label_id: {
                "name": request.label.name,
                "prompt": _DEFAULT_PROMPT_TEMPLATE.replace("{label}", request.label.name),
                "phi": None,
            }}
        else:
            raise ValueError("M2C needs a label until it is trained: it prompts SAM 3 with the label name.")

        image_rgb = cv2.cvtColor(request.image, cv2.COLOR_BGR2RGB)
        vision_embeds, original_sizes = self._vision(image_rgb)
        contours: list[Contour] = []
        for label_id, concept in concepts.items():
            phi = None if concept["phi"] is None else torch.from_numpy(concept["phi"]).to(self.device)[None]
            with torch.no_grad(), self._autocast():
                text_embeds, attention_mask = self._encode_prompt(concept["prompt"], phi)
                outputs = self.model(vision_embeds=vision_embeds, text_embeds=text_embeds,
                                     attention_mask=attention_mask)
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=original_sizes,
            )[0]
            for mask, score in zip(results["masks"].cpu().numpy(), results["scores"].float().cpu().numpy()):
                contour = Contour.from_binary_mask(
                    binary_mask=mask.astype(np.uint8),
                    only_return_biggest_contour=True,
                    confidence=float(score),
                    added_by=request.model_registry_key,
                )
                if contour is not None:
                    contour.label_id = label_id
                    contours.append(contour)
        return contours

    # -- training: concept embedding search --------------------------------- #
    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs: Any) -> None:
        """Learn one ``phi`` per selected label; SAM 3 itself stays frozen."""
        epochs, learning_rate, batch_size, template = validate_hyperparameters(request.hyper_parameter)
        label_mapping = LabelMapping.from_selected_labels(request.labels)
        dataset = ConceptDataset(request.annotation_file_url, request.image_folder_path, label_mapping)
        is_cancelled: Callable[[], bool] = kwargs.get("is_cancelled", lambda: False)
        progress_callback: Callable[[dict[str, int | float]], None] = kwargs.get(
            "progress_callback", lambda _: None
        )
        self._ensure_weights()

        prompts = {
            database_id: template.replace("{label}", label_mapping.id2label[model_index])
            for database_id, model_index in label_mapping.database_to_model.items()
        }
        phis = {database_id: torch.nn.Parameter(torch.zeros(1, 32, 1024, device=self.device))
                for database_id in prompts}
        optimizer = torch.optim.AdamW(phis.values(), lr=learning_rate, weight_decay=0.0)
        steps = epochs * math.ceil(len(dataset) / batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=learning_rate / 100)

        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "prompt_template": template,
            "dataset_id": request.dataset_id,
            "selected_database_label_ids": json.dumps(sorted(prompts)),
        })
        mlflow.set_tag("label_ids", json.dumps(sorted(prompts)))
        logger.info("M2C: learning %d concept(s) on %d image(s): %s", len(prompts), len(dataset), prompts)

        try:
            for epoch in range(1, epochs + 1):
                epoch_loss, order = 0.0, torch.randperm(len(dataset)).tolist()
                optimizer.zero_grad(set_to_none=True)
                for position, index in enumerate(order, start=1):
                    if is_cancelled():
                        raise TrainingCancelled("Training cancellation was requested.")
                    loss = self._image_loss(*dataset[index], prompts, phis) / batch_size
                    if not torch.isfinite(loss):
                        raise RuntimeError("M2C produced a non-finite training loss.")
                    loss.backward()
                    epoch_loss += float(loss.detach()) * batch_size
                    if position % batch_size == 0 or position == len(order):
                        torch.nn.utils.clip_grad_norm_(phis.values(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                average_loss = epoch_loss / len(order)
                phi_norm = max(float(phi.detach().norm()) for phi in phis.values())
                mlflow.log_metrics({"loss": average_loss, "epoch": epoch, "phi_norm": phi_norm}, step=epoch)
                progress_callback({"epoch": epoch, "loss": average_loss})
                logger.info("M2C epoch %d/%d: loss %.4f, |phi| %.3f", epoch, epochs, average_loss, phi_norm)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.concepts = {
            database_id: {
                "name": label_mapping.id2label[label_mapping.database_to_model[database_id]],
                "prompt": prompts[database_id],
                "phi": phis[database_id].detach()[0].float().cpu().numpy(),
            }
            for database_id in prompts
        }

    def _image_loss(self, image_rgb: np.ndarray, masks: dict[int, torch.Tensor], prompts: dict[int, str],
                    phis: dict[int, torch.nn.Parameter]) -> torch.Tensor:
        """Sum of the per-concept losses on one training image.

        The image has no instance of a label that ``masks`` leaves out; the export
        holds only fully annotated images, so that label is a negative here.
        """
        vision_embeds, _ = self._vision(image_rgb)
        loss = torch.zeros((), device=self.device)
        for database_id, prompt in prompts.items():
            with self._autocast():
                text_embeds, attention_mask = self._encode_prompt(prompt, phis[database_id])
                outputs = self.model(vision_embeds=vision_embeds, text_embeds=text_embeds,
                                     attention_mask=attention_mask)
            gt = masks.get(database_id, torch.zeros((0, _MASK_SIZE, _MASK_SIZE), dtype=torch.bool))
            if gt.shape[-2:] != outputs.pred_masks.shape[-2:]:
                raise RuntimeError(f"SAM 3 mask size {tuple(outputs.pred_masks.shape[-2:])} is not {_MASK_SIZE}.")
            loss = loss + sum(m2c_loss(outputs, gt).values())
        return loss
