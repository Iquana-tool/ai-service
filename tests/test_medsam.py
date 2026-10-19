"""MedSAM2 and MedSAM3: checkpoint renaming, the LoRA merge, and what they register as.

Nothing is downloaded by default. ``test_real_checkpoints`` converts the real
checkpoints when ``IQUANA_TEST_MEDSAM_WEIGHTS`` is set (about 3.5 GB of downloads
on a cold cache, and SAM 3 is gated, so it needs a Hugging Face token).
"""
from __future__ import annotations

import os
import pickle

import pytest
import torch
from torch import nn

from models.medsam2 import MedSAM2, _rename as rename_sam2, image_config, video_config
from models.medsam3 import MedSAM3, _rename as rename_sam3, merge_lora
from models.registry import collected_models
from models.sam3 import SAM3


# -- MedSAM2 ----------------------------------------------------------------- #
@pytest.mark.parametrize("original, converted", [
    ("image_encoder.trunk.blocks.3.mlp.layers.0.weight", "vision_encoder.backbone.blocks.3.mlp.proj_in.weight"),
    ("sam_mask_decoder.transformer.layers.1.mlp.layers.1.bias", "mask_decoder.transformer.layers.1.mlp.proj_out.bias"),
    ("sam_mask_decoder.output_hypernetworks_mlps.2.layers.1.weight", "mask_decoder.output_hypernetworks_mlps.2.layers.0.weight"),
    ("sam_mask_decoder.pred_obj_score_head.layers.2.weight", "mask_decoder.pred_obj_score_head.proj_out.weight"),
    ("memory_encoder.mask_downsampler.encoder.12.weight", "memory_encoder.mask_downsampler.final_conv.weight"),
    ("memory_encoder.mask_downsampler.encoder.4.bias", "memory_encoder.mask_downsampler.layers.1.layer_norm.bias"),
    ("image_encoder.neck.convs.0.conv.weight", "vision_encoder.neck.convs.0.weight"),
    ("memory_encoder.out_proj.weight", "memory_encoder.projection.weight"),
])
def test_sam2_checkpoint_keys_are_renamed(original, converted):
    assert rename_sam2(original) == converted


def test_medsam2_runs_at_512():
    image, video = image_config(), video_config()
    assert image.vision_config.backbone_feature_sizes == [[128, 128], [64, 64], [32, 32]]
    assert image.prompt_encoder_config.image_size == 512
    assert video.image_size == 512
    assert video.memory_attention_rope_feat_sizes == [32, 32]


def test_medsam2_is_a_medical_sam2():
    model = MedSAM2("medsam2")
    info = model.model_info
    assert info.registry_key == "medsam2"
    assert info.tags["domain"] == "medical"
    assert set(info.tags["tasks"].split(",")) == {"prompted-segmentation", "propagation"}
    # No weights yet, so the instance pickles small and loads lazily.
    assert pickle.loads(pickle.dumps(model)).checkpoint == "MedSAM2_latest.pt"


# -- MedSAM3 ----------------------------------------------------------------- #
@pytest.mark.parametrize("original, converted", [
    ("backbone.vision_backbone.trunk.blocks.7.attn.qkv", "vision_encoder.backbone.layers.7.attention.qkv"),
    ("backbone.vision_backbone.trunk.blocks.7.attn.proj", "vision_encoder.backbone.layers.7.attention.o_proj"),
    ("backbone.vision_backbone.trunk.blocks.7.mlp.fc2", "vision_encoder.backbone.layers.7.mlp.fc2"),
    ("backbone.language_backbone.encoder.transformer.resblocks.3.attn.k_proj",
     "text_encoder.text_model.encoder.layers.3.self_attn.k_proj"),
    # CLIP keeps out_proj in transformers.
    ("backbone.language_backbone.encoder.transformer.resblocks.3.attn.out_proj",
     "text_encoder.text_model.encoder.layers.3.self_attn.out_proj"),
    ("backbone.language_backbone.encoder.transformer.resblocks.3.mlp.c_proj",
     "text_encoder.text_model.encoder.layers.3.mlp.fc2"),
    ("geometry_encoder.encode.2.cross_attn_image.out_proj", "geometry_encoder.layers.2.cross_attn.o_proj"),
    ("geometry_encoder.boxes_direct_project", "geometry_encoder.boxes_direct_project"),
    ("transformer.encoder.layers.0.cross_attn_image.v_proj", "detr_encoder.layers.0.cross_attn.v_proj"),
    ("transformer.decoder.layers.5.ca_text.out_proj", "detr_decoder.layers.5.text_cross_attn.o_proj"),
    ("transformer.decoder.layers.5.cross_attn.q_proj", "detr_decoder.layers.5.vision_cross_attn.q_proj"),
    ("transformer.decoder.layers.5.linear1", "detr_decoder.layers.5.mlp.fc1"),
    ("dot_prod_scoring.hs_proj", "dot_product_scoring.query_proj"),
    ("segmentation_head.cross_attend_prompt.out_proj", "mask_decoder.prompt_cross_attn.o_proj"),
])
def test_sam3_module_names_are_renamed(original, converted):
    assert rename_sam3(original) == converted


def _nested(**linears: nn.Linear) -> nn.Module:
    """A module tree with a linear at each dotted path (``__`` stands for ``.``)."""
    root = nn.Module()
    for path, linear in linears.items():
        *parents, leaf = path.split("__")
        node = root
        for name in parents:
            if not hasattr(node, name):
                node.add_module(name, nn.Module())
            node = getattr(node, name)
        node.add_module(leaf, linear)
    return root


def _lora(module: str, inputs: int, outputs: int, rank: int = 2) -> dict[str, torch.Tensor]:
    return {f"{module}.lora.lora_A": torch.randn(inputs, rank), f"{module}.lora.lora_B": torch.randn(rank, outputs)}


def test_lora_is_folded_into_the_renamed_weight():
    model = _nested(detr_encoder__layers__0__mlp__fc1=nn.Linear(4, 6))
    before = model.get_parameter("detr_encoder.layers.0.mlp.fc1.weight").clone()
    lora = _lora("transformer.encoder.layers.0.linear1", 4, 6)

    assert merge_lora(model, lora, alpha=4) == 1

    a, b = lora["transformer.encoder.layers.0.linear1.lora.lora_A"], lora["transformer.encoder.layers.0.linear1.lora.lora_B"]
    expected = before + (a @ b).T * (4 / 2)
    assert torch.allclose(model.get_parameter("detr_encoder.layers.0.mlp.fc1.weight"), expected)


def test_fused_qkv_update_is_split_across_q_k_v():
    prefix = "vision_encoder__backbone__layers__0__attention__"
    model = _nested(**{prefix + p: nn.Linear(4, 4) for p in ("q_proj", "k_proj", "v_proj")})
    before = {p: model.get_parameter(f"vision_encoder.backbone.layers.0.attention.{p}.weight").clone()
              for p in ("q_proj", "k_proj", "v_proj")}
    lora = _lora("backbone.vision_backbone.trunk.blocks.0.attn.qkv", 4, 12)

    merge_lora(model, lora, alpha=2)

    a, b = lora["backbone.vision_backbone.trunk.blocks.0.attn.qkv.lora.lora_A"], \
        lora["backbone.vision_backbone.trunk.blocks.0.attn.qkv.lora.lora_B"]
    delta = (a @ b).T
    for i, p in enumerate(("q_proj", "k_proj", "v_proj")):
        after = model.get_parameter(f"vision_encoder.backbone.layers.0.attention.{p}.weight")
        assert torch.allclose(after, before[p] + delta[4 * i:4 * (i + 1)])


def test_point_prompt_layers_are_skipped_and_mismatches_fail():
    model = _nested(dot_product_scoring__query_proj=nn.Linear(4, 4))
    skipped = _lora("geometry_encoder.points_direct_project", 4, 4)
    assert merge_lora(model, skipped, alpha=2) == 0

    with pytest.raises(RuntimeError, match="dot_product_scoring.query_proj"):
        merge_lora(model, _lora("dot_prod_scoring.hs_proj", 4, 8), alpha=2)
    with pytest.raises(AttributeError):
        merge_lora(model, _lora("dot_prod_scoring.prompt_proj", 4, 4), alpha=2)


def test_medsam3_serves_what_sam3_serves():
    assert MedSAM3 in collected_models()
    info = MedSAM3.model_info
    assert info.registry_key == "medsam3"
    assert info.tags["domain"] == "medical"
    assert info.tags["tasks"] == SAM3.model_info.tags["tasks"]
    assert [c.task for c in info.input_contracts] == [c.task for c in SAM3.model_info.input_contracts]
    # Its own contracts, so tuning one model's defaults never edits the other's.
    assert info.input_contracts[0] is not SAM3.model_info.input_contracts[0]
    assert pickle.loads(pickle.dumps(MedSAM3())).threshold == 0.3


# -- Real checkpoints (opt-in) ----------------------------------------------- #
@pytest.mark.skipif(not os.environ.get("IQUANA_TEST_MEDSAM_WEIGHTS"), reason="needs IQUANA_TEST_MEDSAM_WEIGHTS")
def test_real_checkpoints():
    """The SAM 2 renaming reproduces transformers' own conversion; both models load fully."""
    from huggingface_hub import hf_hub_download
    from transformers import Sam2Model, Sam2VideoModel

    from models.medsam2 import _load_into, convert_state_dict

    original = hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt")
    converted = convert_state_dict(torch.load(original, map_location="cpu", weights_only=True)["model"])
    for cls in (Sam2Model, Sam2VideoModel):
        reference = cls.from_pretrained("facebook/sam2.1-hiera-tiny")
        rebuilt = cls(reference.config)
        _load_into(rebuilt, converted)
        mine, theirs = rebuilt.state_dict(), reference.state_dict()
        assert all(torch.equal(mine[k], theirs[k]) for k in theirs)

    medsam2 = MedSAM2("medsam2")
    medsam2._ensure_weights()
    medsam2._ensure_video_weights()
    assert medsam2.model.prompt_encoder.mask_input_size == (128, 128)

    medsam3 = MedSAM3()
    medsam3._ensure_weights()  # raises if any adapted layer is missing
