"""UnSAMv2: SAM 2.1 with a granularity control (Yu, Darrell, Wang, 2025).

https://github.com/yujunwei04/UnSAMv2 -- code and weights Apache-2.0.
Paper: https://arxiv.org/abs/2511.13714

UnSAMv2 is SAM 2.1 Hiera-Small with one extra input: a granularity value in
[0.1, 1.0] that picks how coarse the mask for a prompt is (low = a small part,
high = the whole object). It was trained without human labels. Next to stock
SAM 2.1 it adds

  * a granularity *prompt token*: 128 fixed Fourier features of the value, a
    3-layer MLP to 256-d, plus a learned embedding; appended to the point/box
    tokens,
  * a granularity *output token* in the mask decoder with its own hypernetwork
    MLP; the decoder returns that token's mask only, and
  * rank-8 LoRA on the q/k/v/out projections of the decoder's two-way
    transformer, which :func:`convert_state_dict` folds into the plain weights.

The image encoder is stock Hiera-Small. Like MedSAM2 the weights are only
published in the original SAM 2 checkpoint format, so the checkpoint is renamed
onto the transformers classes (:mod:`models.medsam2`) and loaded strictly.

Images only for now: the granularity token was not carried into the video
memory path, so the propagation capability inherited from SAM 2 is not offered.
"""
import re
from logging import getLogger
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import Sam2Config, Sam2Model, Sam2Processor
from transformers.models.sam2.modeling_sam2 import (
    Sam2FeedForward,
    Sam2ImageSegmentationOutput,
    Sam2MaskDecoder,
    Sam2PromptEncoder,
)

from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter
from models.base import PROPAGATION
from models.medsam2 import _load_into, convert_state_dict as convert_sam2_state_dict
from models.registry import register_model
from models.sam2 import SAM2Prompted
from paths import hf_token

logger = getLogger(__name__)

_REPO = "yujunwei04/UnSAMv2"
#: Stock config and processor of the model UnSAMv2 was fine-tuned from.
_BASE = "facebook/sam2.1-hiera-small"

#: The demo's slider range and default (``app.py``: ``UNSAMV2_GRAN_MIN/MAX``, ``value=0.2``).
GRANULARITY_MIN, GRANULARITY_MAX, GRANULARITY_DEFAULT = 0.1, 1.0, 0.2

#: ``FourierEmbedder(hidden_dim=128, temperature=100)`` in the original.
_FOURIER_DIM = 128
_FOURIER_TEMPERATURE = 100.0
#: ``LoRAConfig(rank=8, alpha=16)``: the update is scaled by alpha / rank.
_LORA_SCALE = 16 / 8


_VARIANTS: dict[str, dict] = {
    "unsamv2": {
        "checkpoint": "unsamv2.pt",
        "name": "UnSAMv2",
        "description": (
            "UnSAMv2 - SAM 2.1 Small with a granularity control. Together with a point or "
            "box, a granularity value from 0.1 to 1.0 picks the size of the mask: low values "
            "give small parts, high values the whole object. Trained without human labels."
        ),
        "model_size": "small",
        "inference_speed": "fast",
        "accuracy_level": "medium",
    },
    "unsamv2-plus": {
        "checkpoint": "unsamv2_plus.pt",
        "name": "UnSAMv2+",
        "description": (
            "UnSAMv2+ - UnSAMv2 trained on more unlabeled images; the better of the two "
            "checkpoints in the authors' tests. SAM 2.1 Small with a granularity control: "
            "low values give small parts, high values the whole object."
        ),
        "model_size": "small",
        "inference_speed": "fast",
        "accuracy_level": "high",
    },
}
for _variant in _VARIANTS.values():
    _variant.update(
        usage_tip=(
            "Click or box the object, then move the granularity slider: down for a part, "
            "up for the whole object. Refine with more clicks at the same granularity."
        ),
        requires_gpu="false",
        tags={
            "domain": "generic",
            "publisher": "UC Berkeley",
            "license": "apache-2.0",
            "source": f"https://huggingface.co/{_REPO}",
        },
    )


# --------------------------------------------------------------------------- #
# The two heads, on top of the transformers SAM 2 classes
# --------------------------------------------------------------------------- #
class UnSAMv2PromptEncoder(Sam2PromptEncoder):
    """SAM 2's prompt encoder plus one sparse token for the granularity value."""

    def __init__(self, config):
        super().__init__(config)
        hidden = (_FOURIER_DIM + config.hidden_size) // 2
        # ``FourierGranularityMLP``: Linear/ReLU/Dropout twice, then a Linear (keys 0, 3, 6).
        self.granularity_mlp = nn.Sequential(
            nn.Linear(_FOURIER_DIM, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, config.hidden_size),
        )
        self.granularity_embedding = nn.Embedding(1, config.hidden_size)
        # Fixed, not learned: the original computes them outside any nn.Module.
        frequencies = _FOURIER_TEMPERATURE ** (torch.arange(_FOURIER_DIM // 2) / (_FOURIER_DIM // 2))
        self.register_buffer("granularity_frequencies", frequencies, persistent=False)

    def embed_granularity(self, granularity: torch.Tensor) -> torch.Tensor:
        """``(..., )`` granularity values -> ``(..., hidden)`` prompt tokens."""
        angles = granularity[..., None] * self.granularity_frequencies
        # Interleaved sin(f0 g), cos(f0 g), sin(f1 g), ... as in the original.
        features = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return self.granularity_mlp(features) + self.granularity_embedding.weight[0]

    def forward(self, input_points, input_labels, input_boxes, input_masks, granularity=None):
        sparse, dense = super().forward(input_points, input_labels, input_boxes, input_masks)
        if granularity is not None:
            # The token order does not matter to the two-way transformer, so the
            # token goes last; the original puts it between points and boxes.
            token = self.embed_granularity(granularity.to(sparse.dtype)).unsqueeze(2)
            sparse = torch.cat([sparse, token], dim=2)
        return sparse, dense


class UnSAMv2MaskDecoder(Sam2MaskDecoder):
    """SAM 2's mask decoder plus a granularity output token; returns only its mask.

    ``forward`` follows :meth:`Sam2MaskDecoder.forward` with the extra token,
    the way ``MaskDecoderGra.predict_masks`` extends SAM 2's original decoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.gra_token = nn.Embedding(1, self.hidden_size)
        self.gra_output_token_mlp = Sam2FeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        high_resolution_features: list[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight, self.gra_token.weight],
            dim=0,
        ).repeat(batch_size, point_batch_size, 1, 1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2).to(self.iou_token.weight.dtype)

        image_embeddings = (image_embeddings + dense_prompt_embeddings).repeat_interleave(point_batch_size, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)
        tokens, image_embeddings = self.transformer(
            point_embeddings=tokens,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=None,
            target_embedding=None,
        )
        iou_token_out = tokens[:, :, 1, :]
        # The SAM mask tokens, then the granularity token.
        mask_tokens_out = tokens[:, :, 2:(3 + self.num_mask_tokens), :]

        image_embeddings = image_embeddings.transpose(2, 3).view(
            batch_size * point_batch_size, num_channels, height, width
        )
        feat_s0, feat_s1 = high_resolution_features
        feat_s0 = feat_s0.repeat_interleave(point_batch_size, dim=0)
        feat_s1 = feat_s1.repeat_interleave(point_batch_size, dim=0)
        upscaled = self.activation(self.upscale_layer_norm(self.upscale_conv1(image_embeddings) + feat_s1))
        upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)

        # Only the granularity mask is returned, so only its hypernetwork runs.
        hyper_in = self.gra_output_token_mlp(mask_tokens_out[:, :, -1, :]).unsqueeze(2)
        _, num_channels, height, width = upscaled.shape
        upscaled = upscaled.view(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled).view(batch_size, point_batch_size, 1, height, width)

        # The granularity token has no IoU head of its own. The score is the best
        # IoU estimate of SAM's three multimask tokens, as the authors' demo reports
        # it (``MaskDecoderGra.forward`` with ``multimask_output=True``).
        iou_pred = self.iou_prediction_head(iou_token_out)[:, :, 1:].amax(dim=-1, keepdim=True)
        object_score_logits = self.pred_obj_score_head(tokens[:, :, 0, :])
        return masks, iou_pred, mask_tokens_out[:, :, -1:], object_score_logits


class UnSAMv2Model(Sam2Model):
    """:class:`Sam2Model` with the UnSAMv2 heads and a ``granularity`` input."""

    def __init__(self, config: Sam2Config):
        super().__init__(config)
        self.prompt_encoder = UnSAMv2PromptEncoder(config.prompt_encoder_config)
        self.mask_decoder = UnSAMv2MaskDecoder(config.mask_decoder_config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_points: torch.Tensor | None = None,
        input_labels: torch.Tensor | None = None,
        input_boxes: torch.Tensor | None = None,
        input_masks: torch.Tensor | None = None,
        granularity: torch.Tensor | float = GRANULARITY_DEFAULT,
        multimask_output: bool = False,
        **kwargs,
    ) -> Sam2ImageSegmentationOutput:
        """Segment one mask per prompt at ``granularity`` (a float, or ``(batch, point_batch)``).

        ``multimask_output`` is accepted for the SAM 2 call signature and ignored:
        UnSAMv2 always returns the one granularity mask.
        """
        batch_size = pixel_values.shape[0]
        feature_maps = self.get_image_features(pixel_values, return_dict=True).fpn_hidden_states
        feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding
        image_embeddings = [
            feat.permute(1, 2, 0).view(batch_size, -1, *size)
            for feat, size in zip(feature_maps, self.backbone_feature_sizes)
        ]
        positional = self.get_image_wide_positional_embeddings().repeat(batch_size, 1, 1, 1)

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[..., 0], dtype=torch.int)
        if input_points is None and input_boxes is None:
            input_points = image_embeddings[-1].new_zeros(batch_size, 1, 1, 2)
            input_labels = -torch.ones(batch_size, 1, 1, dtype=torch.int32, device=pixel_values.device)
        if input_masks is not None and input_masks.shape[-2:] != self.prompt_encoder.mask_input_size:
            input_masks = nn.functional.interpolate(
                input_masks.float(), size=self.prompt_encoder.mask_input_size,
                align_corners=False, mode="bilinear", antialias=True,
            ).to(input_masks.dtype)

        point_batch_size = (input_points if input_points is not None else input_boxes).shape[1]
        granularity = torch.as_tensor(granularity, dtype=positional.dtype, device=positional.device)
        granularity = granularity.expand(batch_size, point_batch_size)

        sparse, dense = self.prompt_encoder(input_points, input_labels, input_boxes, input_masks, granularity)
        masks, iou_scores, _, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=positional,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            high_resolution_features=image_embeddings[:-1],
        )
        return Sam2ImageSegmentationOutput(
            iou_scores=iou_scores,
            pred_masks=masks,
            object_score_logits=object_score_logits,
            image_embeddings=image_embeddings,
        )


# --------------------------------------------------------------------------- #
# Original UnSAMv2 checkpoint -> transformers state dict
# --------------------------------------------------------------------------- #
_LORA_LAYER = re.compile(r"(?P<layer>.+)\.lora_A$")
# Renames for the UnSAMv2 heads, applied after the SAM 2 renaming.
_HEAD_RENAMES = (
    ("prompt_encoder.fourier_embedder.mlp.mlp.", "prompt_encoder.granularity_mlp."),
    ("gra_output_token_mlp.layers.0.", "gra_output_token_mlp.proj_in."),
    ("gra_output_token_mlp.layers.1.", "gra_output_token_mlp.layers.0."),
    ("gra_output_token_mlp.layers.2.", "gra_output_token_mlp.proj_out."),
)


def merge_lora(original: dict[str, torch.Tensor], scale: float = _LORA_SCALE) -> dict[str, torch.Tensor]:
    """Fold each ``LoRALinear`` (``original_layer`` + ``lora_A``/``lora_B``) into a plain linear.

    ``LoRALinear`` computes ``W x + b + scale * B A x``, so the merged weight is
    ``W + scale * B @ A`` under the layer's own name.
    """
    merged = dict(original)
    layers = [m.group("layer") for key in original if (m := _LORA_LAYER.match(key))]
    for layer in layers:
        a, b = merged.pop(f"{layer}.lora_A"), merged.pop(f"{layer}.lora_B")
        weight = merged.pop(f"{layer}.original_layer.weight")
        merged[f"{layer}.weight"] = (weight.float() + scale * (b.float() @ a.float())).to(weight.dtype)
        merged[f"{layer}.bias"] = merged.pop(f"{layer}.original_layer.bias")
    return merged


def convert_state_dict(original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge the LoRA layers, then rename the checkpoint onto :class:`UnSAMv2Model`."""
    converted = convert_sam2_state_dict(merge_lora(original))
    for old, new in _HEAD_RENAMES:
        converted = {key.replace(old, new): value for key, value in converted.items()}
    return converted


# --------------------------------------------------------------------------- #
# The served model
# --------------------------------------------------------------------------- #
class UnSAMv2(SAM2Prompted):
    """Prompted segmentation with a granularity parameter; images only."""

    VARIANTS = _VARIANTS
    _unpicklable_attrs = ("model", "processor")
    _core_weights = ("model", "processor")

    @classmethod
    def supported_tasks(cls):
        # Not SAM 2's propagation: the granularity token has no video path yet.
        return [task for task in super().supported_tasks() if task is not PROPAGATION]

    def _input_contracts(self) -> list[InputContract]:
        return [
            InputContract(
                task="prompted-segmentation",
                conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
                parameters=[
                    HyperParameter(
                        key="granularity", label="Granularity", type="float",
                        default_value=GRANULARITY_DEFAULT,
                        min_value=GRANULARITY_MIN, max_value=GRANULARITY_MAX, step=0.05,
                        description="Size of the mask for the prompt: low values give small parts, "
                                    "high values the whole object.",
                    ),
                ],
            ),
        ]

    def _load_weights(self) -> None:
        token = hf_token()
        path = hf_hub_download(_REPO, self.checkpoint, token=token)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = UnSAMv2Model(Sam2Config.from_pretrained(_BASE, token=token))
        _load_into(model, convert_state_dict(checkpoint["model"]))
        self.model = model.eval().to(self.device)
        self.processor = Sam2Processor.from_pretrained(_BASE, token=token)

    def _model_kwargs(self, params: dict[str, Any] | None) -> dict[str, Any]:
        value = (params or {}).get("granularity")
        return {"granularity": GRANULARITY_DEFAULT if value is None else float(value)}


@register_model
def unsamv2() -> UnSAMv2:
    return UnSAMv2("unsamv2")


@register_model
def unsamv2_plus() -> UnSAMv2:
    return UnSAMv2("unsamv2-plus")
