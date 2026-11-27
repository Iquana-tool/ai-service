"""MedSAM2: SAM 2.1 fine-tuned on 3D medical images and videos (Ma et al., 2025).

https://github.com/bowang-lab/MedSAM2 -- code Apache-2.0, weights CC-BY-SA-4.0.

MedSAM2 is SAM 2.1 Hiera-Tiny trained at 512 px instead of 1024 px, so it is the
SAM 2 class with a different checkpoint and a smaller input. Everything else --
point/box/polygon prompting, refinement, propagation through a stack -- is
inherited from :class:`models.sam2.SAM2Prompted`. Propagation is where it should
shine: the model was trained on CT/MRI volumes and echo videos.

The weights are only published in the original SAM 2 checkpoint format, not as a
transformers model, so ``_load_weights`` builds the HF model from a config and
renames the checkpoint's keys onto it (:func:`convert_state_dict`). The renaming
is copied from transformers' own ``convert_sam2_video_to_hf.py``. Loading is
strict, so a key the renaming misses fails loudly instead of leaving a layer at
its random init.
"""
import re
from logging import getLogger

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    Sam2Config,
    Sam2HieraDetConfig,
    Sam2ImageProcessor,
    Sam2MaskDecoderConfig,
    Sam2Model,
    Sam2Processor,
    Sam2PromptEncoderConfig,
    Sam2VideoConfig,
    Sam2VideoMaskDecoderConfig,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam2VideoPromptEncoderConfig,
    Sam2VideoVideoProcessor,
    Sam2VisionConfig,
)

from models.registry import register_model
from models.sam2 import SAM2Prompted
from paths import hf_token

logger = getLogger(__name__)

_REPO = "wanglab/MedSAM2"

#: MedSAM2's ``sam2.1_hiera_t512.yaml``: Hiera-Tiny at 512 px. Every
#: size-dependent part of the HF config follows from it.
_IMAGE_SIZE = 512
#: Hiera downsamples by 4 and then halves three times; SAM 2 keeps the first three
#: levels (256/128/64 at 1024 px).
_FEATURE_SIZES = [[_IMAGE_SIZE // 4 // 2 ** i] * 2 for i in range(3)]
#: The lowest level, which memory attention rotates over (64 at 1024 px).
_ROPE_FEAT_SIZES = _FEATURE_SIZES[-1]


_VARIANTS: dict[str, dict] = {
    "medsam2": {
        "checkpoint": "MedSAM2_latest.pt",
        "name": "MedSAM2",
        "description": (
            "MedSAM2 - SAM 2.1 Tiny fine-tuned on large-scale 3D medical images and videos "
            "(CT, MRI, PET, ultrasound, endoscopy). Segments organs and lesions from a point "
            "or box and carries them through a volume or video. Trained at 512 px."
        ),
        "usage_tip": (
            "Box the structure on one slice and propagate it through the stack; correct a "
            "slice where it drifts and propagate again. Point, box and polygon prompts work "
            "on single images too."
        ),
        "model_size": "tiny",
        "inference_speed": "fastest",
        "accuracy_level": "medium",
        "requires_gpu": "false",
        "tags": {
            "domain": "medical",
            "publisher": "bowang-lab",
            "license": "cc-by-sa-4.0",
            "source": f"https://huggingface.co/{_REPO}",
        },
    },
}


def _vision_config() -> Sam2VisionConfig:
    return Sam2VisionConfig(
        backbone_config=Sam2HieraDetConfig(image_size=[_IMAGE_SIZE, _IMAGE_SIZE]),
        backbone_feature_sizes=_FEATURE_SIZES,
    )


def image_config() -> Sam2Config:
    return Sam2Config(
        vision_config=_vision_config(),
        prompt_encoder_config=Sam2PromptEncoderConfig(image_size=_IMAGE_SIZE),
        mask_decoder_config=Sam2MaskDecoderConfig(),
    )


def video_config() -> Sam2VideoConfig:
    return Sam2VideoConfig(
        vision_config=_vision_config(),
        prompt_encoder_config=Sam2VideoPromptEncoderConfig(image_size=_IMAGE_SIZE),
        mask_decoder_config=Sam2VideoMaskDecoderConfig(),
        image_size=_IMAGE_SIZE,
        memory_attention_rope_feat_sizes=_ROPE_FEAT_SIZES,
        # Both are SAM 2.1 features (off for 2.0 checkpoints).
        enable_temporal_pos_encoding_for_object_pointers=True,
        enable_occlusion_spatial_embedding=True,
    )


def _image_processor() -> Sam2ImageProcessor:
    # The dense mask prompt is 4x the image embedding: 128 px at 512 px input.
    return Sam2ImageProcessor(
        size={"height": _IMAGE_SIZE, "width": _IMAGE_SIZE},
        mask_size={"height": _IMAGE_SIZE // 4, "width": _IMAGE_SIZE // 4},
    )


# --------------------------------------------------------------------------- #
# Original SAM 2 checkpoint -> transformers state dict
# --------------------------------------------------------------------------- #
# From transformers' convert_sam2_video_to_hf.py (Apache-2.0). Applied in order;
# every match is replaced, so later entries see the result of earlier ones.
_KEY_RENAMES = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
}

# MLP layer lists become named layers: (pattern capturing the layer index as
# ``layer``, the text to replace, its replacement per layer index).
_MLPS = (
    (r"vision_encoder\.backbone\.blocks\.\d+\.mlp\.layers\.(?P<layer>\d+)",
     "layers.{}", ("proj_in", "proj_out")),
    # "mlp." is part of the match here: the key has a "layers.N" before it too.
    (r"mask_decoder\.transformer\.layers\.\d+\.mlp\.layers\.(?P<layer>\d+)",
     "mlp.layers.{}", ("mlp.proj_in", "mlp.proj_out")),
    (r"mask_decoder\.pred_obj_score_head\.layers\.(?P<layer>\d+)",
     "layers.{}", ("proj_in", "layers.0", "proj_out")),
    (r".*\.output_hypernetworks_mlps\.\d+\.layers\.(?P<layer>\d+)",
     "layers.{}", ("proj_in", "layers.0", "proj_out")),
    (r"object_pointer_proj\.layers\.(?P<layer>\d+)",
     "layers.{}", ("proj_in", "layers.0", "proj_out")),
)


def _rename(key: str) -> str:
    for old, new in _KEY_RENAMES.items():
        if old in key:
            key = key.replace(old, new)

    for pattern, old, new in _MLPS:
        match = re.match(pattern, key)
        if match:
            layer = int(match.group("layer"))
            key = key.replace(old.format(layer), new[layer])

    if re.match(r"vision_encoder\.neck\.convs\.(\d+)\.conv", key):
        key = key.replace(".conv.", ".")
    if re.match(r"memory_encoder\.o_proj.*", key):
        key = key.replace(".o_proj.", ".projection.")

    match = re.match(r"memory_encoder\.mask_downsampler\.encoder\.(\d+).*", key)
    if match:
        layer = int(match.group(1))
        if layer == 12:
            key = key.replace(f"encoder.{layer}", "final_conv")
        elif layer % 3 == 0:
            key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.conv")
        elif layer % 3 == 1:
            key = key.replace(f"encoder.{layer}", f"layers.{layer // 3}.layer_norm")
    return key


def convert_state_dict(original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename an original SAM 2 checkpoint's ``model`` dict onto ``Sam2VideoModel``.

    The image-only ``Sam2Model`` takes the same dict and ignores the memory keys.
    """
    converted = {_rename(key): value for key, value in original.items()}
    converted["shared_image_embedding.positional_embedding"] = converted[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]
    converted["prompt_encoder.point_embed.weight"] = torch.cat(
        [converted.pop(f"prompt_encoder.point_embed.{i}.weight") for i in range(4)], dim=0,
    )
    return converted


def _load_into(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load strictly, apart from keys the model declares it ignores (memory, for images)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    unexpected = [key for key in unexpected if not any(re.search(p, key) for p in ignored)]
    if missing or unexpected:
        raise RuntimeError(
            f"MedSAM2 checkpoint does not fit {type(model).__name__}: "
            f"missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)})"
        )


class MedSAM2(SAM2Prompted):
    """MedSAM2 on the SAM 2 class: same prompting and propagation, medical weights."""

    VARIANTS = _VARIANTS

    def _state_dict(self) -> dict[str, torch.Tensor]:
        path = hf_hub_download(_REPO, self.checkpoint, token=hf_token())
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return convert_state_dict(checkpoint["model"])

    def _load_weights(self) -> None:
        model = Sam2Model(image_config())
        _load_into(model, self._state_dict())
        self.model = model.eval().to(self.device)
        self.processor = Sam2Processor(image_processor=_image_processor())

    def _ensure_video_weights(self) -> None:
        if getattr(self, "video_model", None) is not None and getattr(self, "video_processor", None) is not None:
            return
        video_model = Sam2VideoModel(video_config())
        _load_into(video_model, self._state_dict())
        self.video_model = video_model.eval().to(self.device)
        self.video_processor = Sam2VideoProcessor(
            image_processor=_image_processor(),
            video_processor=Sam2VideoVideoProcessor(size={"height": _IMAGE_SIZE, "width": _IMAGE_SIZE}),
        )


@register_model
def medsam2() -> MedSAM2:
    return MedSAM2("medsam2")
