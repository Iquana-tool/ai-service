"""MedSAM3: SAM 3 fine-tuned on medical images and concepts (Liu et al., 2025).

https://github.com/Joey-S-Liu/MedSAM3 -- LoRA weights Apache-2.0, on top of
SAM 3 (gated, SAM licence).

MedSAM3 is SAM 3 plus a LoRA adapter, trained on ~660k medical images across CT,
MRI, X-ray, ultrasound, OCT, microscopy, endoscopy and more, each object tied to
one of 330 medical concept names. It serves SAM 3's suggestion tasks, not prompted
segmentation (see :class:`MedSAM3`), and works from the concept name sent with the
request ("polyp", "retinal fluid", ...), because that is what it was trained on.

The adapter is published as raw LoRA matrices for the *original* SAM 3 code, not
for transformers. ``_load_weights`` therefore loads the HF SAM 3 and folds each
``A @ B`` into the HF weight it adapts (:func:`merge_lora`), which needs the
original module names translated to transformers' (``_RENAMES``, a subset of
transformers' ``convert_sam3_to_hf.py``). Merged weights run at plain SAM 3 speed.
Every adapted layer has to be found with the right shape, so a renaming gap fails
the load instead of silently serving base SAM 3.
"""
import re
from logging import getLogger

import torch
from huggingface_hub import hf_hub_download
from transformers.models.sam3 import Sam3Model, Sam3Processor

from iquana_toolbox.schemas.model_info import ModelInfo
from models.registry import register_model

from models.base import PROMPTED_SEGMENTATION
from models.sam3 import SAM3
from paths import hf_token

logger = getLogger(__name__)

_LORA_REPO = "lal-Joey/MedSAM3_v1"
_LORA_FILE = "best_lora_weights.pt"
#: From the repo's ``configs/full_lora_config.yaml``, the config these weights
#: match (the rank is also readable off the matrices, alpha is not).
_LORA_ALPHA = 32

# Original SAM 3 module path -> transformers module path, applied in order.
_RENAMES = (
    (r"^backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.qkv$", r"vision_encoder.backbone.layers.\1.attention.qkv"),
    (r"^backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.proj$", r"vision_encoder.backbone.layers.\1.attention.o_proj"),
    (r"^backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.mlp\.", r"vision_encoder.backbone.layers.\1.mlp."),
    # CLIP keeps "out_proj" in transformers, so the text encoder skips the o_proj rename below.
    (r"^backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.attn\.", r"text_encoder.text_model.encoder.layers.\1.self_attn."),
    (r"^backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc$", r"text_encoder.text_model.encoder.layers.\1.mlp.fc1"),
    (r"^backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj$", r"text_encoder.text_model.encoder.layers.\1.mlp.fc2"),
    (r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.", r"geometry_encoder.layers.\1.cross_attn."),
    (r"^geometry_encoder\.encode\.(\d+)\.self_attn\.", r"geometry_encoder.layers.\1.self_attn."),
    (r"^geometry_encoder\.encode\.(\d+)\.linear([12])$", r"geometry_encoder.layers.\1.mlp.fc\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.", r"detr_encoder.layers.\1.cross_attn."),
    (r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.", r"detr_encoder.layers.\1.self_attn."),
    (r"^transformer\.encoder\.layers\.(\d+)\.linear([12])$", r"detr_encoder.layers.\1.mlp.fc\2"),
    (r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.", r"detr_decoder.layers.\1.self_attn."),
    (r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.", r"detr_decoder.layers.\1.text_cross_attn."),
    (r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.", r"detr_decoder.layers.\1.vision_cross_attn."),
    (r"^transformer\.decoder\.layers\.(\d+)\.linear([12])$", r"detr_decoder.layers.\1.mlp.fc\2"),
    (r"^dot_prod_scoring\.prompt_proj$", r"dot_product_scoring.text_proj"),
    (r"^dot_prod_scoring\.hs_proj$", r"dot_product_scoring.query_proj"),
    (r"^segmentation_head\.cross_attend_prompt\.", r"mask_decoder.prompt_cross_attn."),
    (r"^(?!text_encoder\.)(.+)\.out_proj$", r"\1.o_proj"),
)

#: Adapted layers transformers' SAM 3 doesn't have: it takes no point prompts.
_NOT_IN_TRANSFORMERS = re.compile(r"^geometry_encoder\.points_")


def _rename(module: str) -> str:
    for pattern, replacement in _RENAMES:
        module = re.sub(pattern, replacement, module)
    return module


def merge_lora(model: torch.nn.Module, lora: dict[str, torch.Tensor], alpha: float) -> int:
    """Fold LoRA matrices (original SAM 3 names) into a transformers SAM 3, in place.

    Each adapted linear computes ``W x + (x A B) * alpha / rank``, so its merged
    weight is ``W + (A B)^T * alpha / rank``. The vision trunk's fused ``qkv`` is
    three separate projections in transformers, so its update is split in three.

    Returns the number of layers merged; raises if an adapted layer has no
    counterpart of the right shape.
    """
    modules = sorted({key.rsplit(".lora.", 1)[0] for key in lora})
    merged = 0
    with torch.no_grad():
        for module in modules:
            if _NOT_IN_TRANSFORMERS.match(module):
                continue
            a, b = lora[f"{module}.lora.lora_A"], lora[f"{module}.lora.lora_B"]
            delta = (a.float() @ b.float()).T * (alpha / a.shape[1])
            target = _rename(module)
            if target.endswith(".qkv"):
                parts = zip((target[:-3] + p for p in ("q_proj", "k_proj", "v_proj")), delta.chunk(3, dim=0))
            else:
                parts = [(target, delta)]
            for name, update in parts:
                weight = model.get_parameter(f"{name}.weight")
                if weight.shape != update.shape:
                    raise RuntimeError(f"MedSAM3 LoRA for {module} is {tuple(update.shape)}, "
                                       f"but {name} is {tuple(weight.shape)}")
                weight.add_(update.to(weight.dtype))
            merged += 1
    return merged


@register_model
class MedSAM3(SAM3):
    """SAM 3 with the MedSAM3 adapter merged in: SAM 3's handlers, minus prompted segmentation.

    The adapter was trained on text prompts alone: its box- and point-projection
    LoRA matrices are exactly zero, i.e. they never received a gradient. Given a
    click or a box it returns its favourite structure in the image, wherever the
    prompt was, so it does not offer prompted segmentation. Its exemplar boxes in
    suggestion are likewise weak; the label name carries the request.
    """

    @classmethod
    def supported_tasks(cls):
        return [task for task in super().supported_tasks() if task is not PROMPTED_SEGMENTATION]

    model_info = ModelInfo(
        registry_key="medsam3",
        name="MedSAM3",
        description=(
            "MedSAM3 - SAM 3 fine-tuned on 660k medical images (CT, MRI, X-ray, ultrasound, "
            "OCT, microscopy, histopathology, endoscopy, ...) with 330 medical concepts. "
            "Finds every instance of a concept from examples and its name."
        ),
        usage_tip=(
            "For instance and cross-image suggestion. Name the label after the medical concept "
            "(e.g. 'polyp', 'liver tumor', 'choroid'): MedSAM3 was trained on such names and "
            "finds objects from the name more than from the examples. Without one it over-detects."
        ),
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "medical",
            "publisher": "medsam3",
            "license": "apache-2.0 (adapter), sam (base)",
            "source": f"https://huggingface.co/{_LORA_REPO}",
            # Explicit, because tags are only ever set, never removed, when a registered
            # model is re-synced: a "true" left from before would keep MedSAM3 listed.
            PROMPTED_SEGMENTATION.tag_key: "false",
        },
        status="ready",
        trainable=False,
        input_contracts=[
            c.model_copy(deep=True) for c in SAM3.model_info.input_contracts
            if c.task != PROMPTED_SEGMENTATION.name
        ],
    )

    def _load_weights(self):
        token = hf_token()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        model = Sam3Model.from_pretrained("facebook/sam3", token=token)
        path = hf_hub_download(_LORA_REPO, _LORA_FILE, token=token)
        merged = merge_lora(model, torch.load(path, map_location="cpu", weights_only=True), _LORA_ALPHA)
        logger.info("MedSAM3: merged LoRA into %d layers", merged)
        self.model = model.to(self.device)
