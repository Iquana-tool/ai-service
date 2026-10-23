"""Frozen image encoders loaded from Hugging Face, served on the ``embed`` task.

Each produces one L2-normalized whole-image vector (kind ``"image_cls"``) per image. The
backend's annotation queue uses them for its vector-based orderings (Diversity, Density),
so a dataset can be compared with the encoder that suits it: a general one (I-JEPA),
organisms (BioCLIP 2), biomedical images (BiomedCLIP), radiology (Curia) or retina/OCT
(RETFound). DINOv3, which also serves region vectors for cross-image retrieval, keeps its
own class in :mod:`models.dinov3_embedder`.

Adding an encoder takes only its Hugging Face id: append an :class:`EmbedderPreset` below,
or list it in ``IQUANA_EXTRA_EMBEDDERS`` (see ``paths.py``). Three loaders cover the
common checkpoint layouts:

* ``transformers`` -- ``AutoModel`` + ``AutoImageProcessor``; the vector is the CLS token
  or the mean of the patch tokens (for models without a CLS token, e.g. I-JEPA).
* ``open_clip`` -- CLIP-style repositories (``hf-hub:`` ids); the vector is the image
  embedding.
* ``timm`` -- a raw checkpoint file in the repository loaded into a named timm
  architecture; the vector is the model's pooled feature.

Gated repositories (Curia, RETFound) need the service's Hugging Face login and the
model's terms accepted on huggingface.co. Vectors from different encoders have different
lengths and are never comparable; every vector carries its encoder's id as ``model_id``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable

import numpy as np
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.networking.http.services import EmbedRequest, EmbeddingVector

from models.base import CapabilityModel, Embedding
from models.registry import register_model
from paths import EXTRA_EMBEDDERS, hf_token

logger = getLogger(__name__)

IMAGE_KIND = "image_cls"


@dataclass(frozen=True)
class EmbedderPreset:
    key: str
    name: str
    hf_id: str
    loader: str = "transformers"
    #: transformers only: "cls" token or "mean" of all tokens.
    pooling: str = "cls"
    #: transformers only: run model code shipped in the repository. Only for vetted repos.
    trust_remote_code: bool = False
    #: timm only: the weights file in the repository and the architecture it belongs to.
    checkpoint: str | None = None
    architecture: str | None = None
    domain: str = "general"
    license: str = "See the model card"


PRESETS: tuple[EmbedderPreset, ...] = (
    EmbedderPreset("ijepa", "I-JEPA ViT-H/14", "facebook/ijepa_vith14_1k",
                   pooling="mean", license="CC BY-NC 4.0"),
    EmbedderPreset("bioclip-2", "BioCLIP 2", "imageomics/bioclip-2",
                   loader="open_clip", domain="organisms"),
    EmbedderPreset("biomedclip", "BiomedCLIP",
                   "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                   loader="open_clip", domain="biomedical", license="MIT"),
    EmbedderPreset("curia", "Curia", "raidium/curia", trust_remote_code=True,
                   domain="radiology", license="Research-only RAIL-M"),
    EmbedderPreset("retfound-oct", "RETFound (OCT)", "YukunZhou/RETFound_mae_natureOCT",
                   loader="timm", checkpoint="RETFound_mae_natureOCT.pth",
                   architecture="vit_large_patch16_224", domain="retina/OCT",
                   license="CC BY-NC 4.0"),
)


def extra_presets(spec: str = EXTRA_EMBEDDERS) -> list[EmbedderPreset]:
    """Presets for ``IQUANA_EXTRA_EMBEDDERS``: ``hf_id[@loader]`` entries, comma-separated."""
    presets = []
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        hf_id, _, loader = entry.partition("@")
        key = "hf-" + re.sub(r"[^a-z0-9]+", "-", hf_id.lower()).strip("-")
        presets.append(EmbedderPreset(key, hf_id, hf_id, loader=loader or "transformers"))
    return presets


# -- loaders: preset -> (PIL image -> 1-D feature vector) ----------------------------- #
Encoder = Callable[[Any], np.ndarray]


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_transformers(preset: EmbedderPreset) -> Encoder:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    kwargs = {"token": hf_token(), "trust_remote_code": preset.trust_remote_code}
    processor = AutoImageProcessor.from_pretrained(preset.hf_id, **kwargs)
    device = _device()
    model = AutoModel.from_pretrained(preset.hf_id, **kwargs).eval().to(device)

    @torch.no_grad()
    def encode(image) -> np.ndarray:
        hidden = model(**processor(images=image, return_tensors="pt").to(device)).last_hidden_state
        pooled = hidden[:, 0] if preset.pooling == "cls" else hidden.mean(dim=1)
        return pooled[0].float().cpu().numpy()

    return encode


def _load_open_clip(preset: EmbedderPreset) -> Encoder:
    import open_clip
    import torch

    model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{preset.hf_id}")
    device = _device()
    model = model.eval().to(device)

    @torch.no_grad()
    def encode(image) -> np.ndarray:
        return model.encode_image(preprocess(image).unsqueeze(0).to(device))[0].float().cpu().numpy()

    return encode


def _load_timm(preset: EmbedderPreset) -> Encoder:
    import timm
    import torch
    from huggingface_hub import hf_hub_download

    model = timm.create_model(preset.architecture, pretrained=False, num_classes=0)
    state = torch.load(hf_hub_download(preset.hf_id, preset.checkpoint, token=hf_token()),
                       map_location="cpu", weights_only=False)
    # Research checkpoints usually wrap the weights ({"model": ...}); the classifier head
    # they may carry is not part of a feature extractor, hence strict=False.
    missing, _ = model.load_state_dict(state.get("model", state), strict=False)
    if missing:
        logger.warning("%s: %d backbone weights missing from the checkpoint", preset.key, len(missing))
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    device = _device()
    model = model.eval().to(device)

    @torch.no_grad()
    def encode(image) -> np.ndarray:
        return model(transform(image).unsqueeze(0).to(device))[0].float().cpu().numpy()

    return encode


LOADERS: dict[str, Callable[[EmbedderPreset], Encoder]] = {
    "transformers": _load_transformers,
    "open_clip": _load_open_clip,
    "timm": _load_timm,
}


class PretrainedEmbedder(Embedding, CapabilityModel):
    """A frozen encoder from :data:`PRESETS`; one generated subclass per preset."""

    preset: EmbedderPreset
    # The live encoder holds torch objects MLflow cannot pickle; it is rebuilt on first use.
    _unpicklable_attrs = ("encoder",)

    def _load_weights(self) -> None:
        self.encoder = LOADERS[self.preset.loader](self.preset)

    def embed(self, request: EmbedRequest, params: dict[str, Any] | None = None) -> list[EmbeddingVector]:
        """One whole-image vector; region vectors are DINOv3's job and are not produced."""
        if IMAGE_KIND not in request.image_kinds:
            return []
        from PIL import Image

        vector = self.encoder(Image.fromarray(request.image).convert("RGB")).astype(np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return [EmbeddingVector(kind=IMAGE_KIND, region_id=None, model_id=self.preset.hf_id,
                                dim=int(vector.shape[0]), vector=vector.tolist())]


def _model_info(preset: EmbedderPreset) -> ModelInfo:
    return ModelInfo(
        registry_key=preset.key,
        name=preset.name,
        description=f"Frozen {preset.name} image encoder ({preset.hf_id}), {preset.domain} "
                    "images. Produces one whole-image vector per image.",
        usage_tip="Request image_kinds=['image_cls']. Vectors are L2-normalized; compare them "
                  "only with vectors of the same model_id.",
        info_url=f"https://huggingface.co/{preset.hf_id}",
        tags={"status": "ready", "pretrained": "true", "finetunable": "false",
              "domain": preset.domain},
        status="ready",
        trainable=False,
        architecture=preset.architecture or preset.hf_id,
        license=preset.license,
    )


def _register(preset: EmbedderPreset) -> type:
    name = "Embedder_" + re.sub(r"\W", "_", preset.key)
    cls = type(name, (PretrainedEmbedder,),
               {"preset": preset, "model_info": _model_info(preset), "__module__": __name__})
    # A module-level name lets pickle find the generated class by reference.
    globals()[name] = cls
    return register_model(cls)


for _preset in (*PRESETS, *extra_presets()):
    _register(_preset)
