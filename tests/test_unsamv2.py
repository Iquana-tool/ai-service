"""UnSAMv2: checkpoint conversion, the granularity token, and what it registers as.

Nothing is downloaded by default. ``test_real_checkpoints`` loads both real
checkpoints (about 1 GB on a cold cache) when ``IQUANA_TEST_UNSAMV2_WEIGHTS`` is
set. It checks that the granularity changes the mask size and, when
``fixtures/unsamv2/reference_unsamv2_plus.npz`` is present, that the masks match
the authors' own predictor.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Sam2PromptEncoderConfig

from models.sam2 import SAM2Prompted
from models.unsamv2 import (
    GRANULARITY_DEFAULT,
    UnSAMv2,
    UnSAMv2PromptEncoder,
    convert_state_dict,
    merge_lora,
)

FIXTURES = Path(__file__).parent / "fixtures" / "unsamv2"


# -- Conversion ---------------------------------------------------------------- #
def test_lora_is_folded_into_a_plain_linear():
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    weight, bias = torch.randn(3, 4), torch.randn(3)
    layer = "sam_mask_decoder.transformer.layers.0.self_attn.q_proj"
    merged = merge_lora({
        f"{layer}.lora_A": a, f"{layer}.lora_B": b,
        f"{layer}.original_layer.weight": weight, f"{layer}.original_layer.bias": bias,
    }, scale=2.0)

    assert set(merged) == {f"{layer}.weight", f"{layer}.bias"}
    # LoRALinear: W x + b + scale * B A x.
    x = torch.randn(4)
    expected = weight @ x + bias + 2.0 * (b @ (a @ x))
    assert torch.allclose(merged[f"{layer}.weight"] @ x + merged[f"{layer}.bias"], expected, atol=1e-5)


@pytest.mark.parametrize("original, converted", [
    ("sam_prompt_encoder.fourier_embedder.mlp.mlp.6.weight", "prompt_encoder.granularity_mlp.6.weight"),
    ("sam_prompt_encoder.granularity_embedding.weight", "prompt_encoder.granularity_embedding.weight"),
    ("sam_mask_decoder.gra_token.weight", "mask_decoder.gra_token.weight"),
    ("sam_mask_decoder.gra_output_token_mlp.layers.0.bias", "mask_decoder.gra_output_token_mlp.proj_in.bias"),
    ("sam_mask_decoder.gra_output_token_mlp.layers.1.weight", "mask_decoder.gra_output_token_mlp.layers.0.weight"),
    ("sam_mask_decoder.gra_output_token_mlp.layers.2.weight", "mask_decoder.gra_output_token_mlp.proj_out.weight"),
])
def test_granularity_heads_are_renamed(original, converted):
    # convert_state_dict needs the keys SAM 2's conversion always reads.
    stub = {"sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": torch.zeros(2, 128)}
    stub |= {f"sam_prompt_encoder.point_embeddings.{i}.weight": torch.zeros(1, 256) for i in range(4)}
    assert converted in convert_state_dict(stub | {original: torch.zeros(1)})


def test_merged_lora_is_renamed_like_sam2():
    layer = "sam_mask_decoder.transformer.final_attn_token_to_image.out_proj"
    stub = {"sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": torch.zeros(2, 128)}
    stub |= {f"sam_prompt_encoder.point_embeddings.{i}.weight": torch.zeros(1, 256) for i in range(4)}
    stub |= {f"{layer}.lora_A": torch.zeros(8, 128), f"{layer}.lora_B": torch.zeros(256, 8),
             f"{layer}.original_layer.weight": torch.zeros(256, 128), f"{layer}.original_layer.bias": torch.zeros(256)}
    converted = convert_state_dict(stub)
    assert "mask_decoder.transformer.final_attn_token_to_image.o_proj.weight" in converted
    assert not any("lora" in key or "original_layer" in key for key in converted)


# -- Granularity token ------------------------------------------------------------ #
def test_granularity_features_match_the_original_fourier_embedder():
    encoder = UnSAMv2PromptEncoder(Sam2PromptEncoderConfig())
    captured = {}
    encoder.granularity_mlp.register_forward_hook(lambda _m, args, _out: captured.update(features=args[0]))
    g = torch.tensor([[0.1, 0.7]])
    encoder.embed_granularity(g)

    # FourierEmbedder(hidden_dim=128, temperature=100): sin, cos per frequency, in order.
    freqs = 100 ** (torch.arange(64) / 64)
    expected = torch.cat([torch.cat([torch.sin(f * g[..., None]), torch.cos(f * g[..., None])], -1) for f in freqs], -1)
    assert torch.allclose(captured["features"], expected, atol=1e-6)


def test_granularity_token_is_appended_to_the_prompt_tokens():
    encoder = UnSAMv2PromptEncoder(Sam2PromptEncoderConfig()).eval()
    points, labels = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, dtype=torch.int)
    without, _ = encoder(points, labels, None, None)
    with_token, _ = encoder(points, labels, None, None, torch.tensor([[0.5]]))
    assert with_token.shape[2] == without.shape[2] + 1
    assert torch.equal(with_token[:, :, :-1], without)


# -- Registration ------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["unsamv2", "unsamv2-plus"])
def test_unsamv2_prompts_with_granularity_and_does_not_propagate(key):
    model = UnSAMv2(key)
    info = model.model_info
    assert info.tags["tasks"] == "prompted-segmentation"
    assert info.tags["license"] == "apache-2.0"
    (contract,) = info.input_contracts
    (param,) = contract.parameters
    assert (param.key, param.default_value, param.min_value, param.max_value) == ("granularity", 0.2, 0.1, 1.0)
    # No weights yet, so the instance pickles small and loads lazily.
    assert pickle.loads(pickle.dumps(model)).checkpoint == model.checkpoint


def test_granularity_reaches_the_model():
    model = UnSAMv2("unsamv2")
    assert model._model_kwargs({"granularity": 0.7}) == {"granularity": 0.7}
    assert model._model_kwargs({}) == {"granularity": GRANULARITY_DEFAULT}


def test_sam2_keeps_propagation_and_no_model_kwargs():
    model = SAM2Prompted("sam2-1-tiny")
    assert {c.task for c in model.model_info.input_contracts} == {"propagation", "prompted-segmentation"}
    assert model._model_kwargs({"anything": 1}) == {}


# -- Real checkpoints ------------------------------------------------------------- #
def _run(model: UnSAMv2, image: np.ndarray, prompt: dict, granularity: float) -> tuple[np.ndarray, float]:
    kwargs = {}
    if prompt.get("points"):
        kwargs |= {"input_points": [[prompt["points"]]], "input_labels": [[prompt["labels"]]]}
    if prompt.get("box"):
        kwargs |= {"input_boxes": [[prompt["box"]]]}
    inputs = model.processor([image], return_tensors="pt", **kwargs)
    with torch.no_grad():
        outputs = model.model(**inputs, granularity=granularity)
    mask = model.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][0, 0].numpy()
    return mask, float(outputs.iou_scores.reshape(-1)[0])


@pytest.mark.skipif(not os.environ.get("IQUANA_TEST_UNSAMV2_WEIGHTS"), reason="needs IQUANA_TEST_UNSAMV2_WEIGHTS")
def test_real_checkpoints():
    sys.path.insert(0, str(FIXTURES))
    from prompts import GRANULARITIES, IMAGES, PROMPTS

    for key in ("unsamv2", "unsamv2-plus"):
        model = UnSAMv2(key, device="cpu")
        model._load_weights()  # strict: raises on any key the conversion misses

        truck = np.array(Image.open(FIXTURES / "truck.jpg").convert("RGB"))
        click = PROMPTS["truck.jpg"][0]
        areas = [_run(model, truck, click, g)[0].sum() for g in (0.1, 0.5, 1.0)]
        assert areas[0] < areas[1] < areas[2], f"{key}: mask area does not grow with granularity: {areas}"

    reference = FIXTURES / "reference_unsamv2_plus.npz"
    if not reference.exists():
        pytest.skip("no reference masks from the authors' predictor")
    reference = np.load(reference)
    expected = iter(zip(reference["masks"], reference["scores"]))
    for name in IMAGES:
        image = np.array(Image.open(FIXTURES / name).convert("RGB"))
        for prompt in PROMPTS[name]:
            for g in GRANULARITIES:
                (ours, our_score), (theirs, their_score) = _run(model, image, prompt, g), next(expected)
                iou = (ours & theirs).sum() / max((ours | theirs).sum(), 1)
                assert iou >= 0.99, f"{name} {prompt} g={g}: IoU {iou:.4f} against the authors' predictor"
                assert our_score == pytest.approx(their_score, abs=0.01), f"{name} {prompt} g={g}: score"
