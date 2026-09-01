"""DINOv3 feature embedder -- the default model behind the ``embed`` capability.

Wraps the frozen DINOv3 backbone (:mod:`models.backbones.dinov3`) and turns an
image (and optional masked regions) into the named embedding kinds the retrieval store
expects:

  * ``"image_cls"``   -- the whole-image ``CLS`` token: a holistic scene/domain descriptor,
                          cheap to precompute once per image and used for coarse retrieval.
  * ``"region_mean"`` -- the L2-normalized mean of a masked object's foreground patch
                          features: the per-object *exemplar* vector (foveate.reid's ``mean``
                          mode), used for fine, object-level retrieval.

All vectors are 768-d (ViT-B/16) and L2-normalized. Every vector carries ``model_id`` (the
concrete HF backbone) so the store can version embeddings and never compare across backbones.

The numeric core lives in :mod:`models.embedding_ops` (pure NumPy); this class only runs the
backbone and shapes the request/response.
"""
from logging import getLogger
from typing import Any

import numpy as np

from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.networking.http.services import EmbedRequest, EmbeddingVector
from models.registry import register_model

from models import embedding_ops as ops
from models.backbones.dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Backbone
from models.base import CapabilityModel, Embedding
from paths import hf_token

logger = getLogger(__name__)


@register_model
class DINOv3Embedder(Embedding, CapabilityModel):
    """Frozen DINOv3 backbone serving the ``embed`` task."""

    #: Whole-image descriptor kinds this model can produce (region kind handled separately).
    IMAGE_KINDS = ("image_cls",)
    REGION_KIND = "region_mean"

    model_info = ModelInfo(
        registry_key="dinov3",
        name="DINOv3 Embedder",
        description=(
            "Frozen DINOv3 ViT-B/16 feature extractor. Produces whole-image (CLS) and "
            "masked-region (mean patch) embeddings for cross-image exemplar retrieval."
        ),
        usage_tip=(
            "Set `image_kinds=['image_cls']` for a whole-image descriptor and/or `regions=[...]` "
            "(RLE masks) for per-object 'region_mean' vectors. All outputs are 768-d and "
            "L2-normalized; compare with cosine similarity within one model_id."
        ),
        info_url="https://ai.meta.com/dinov3/",
        tags={
            "status": "ready",
            "pretrained": "true",
            "finetunable": "false",
            "domain": "general",
            "publisher": "meta-ai",
        },
        status="ready",
        trainable=False,
        architecture="DINOv3-ViT-B/16",
        # DINOv3 ships under Meta's non-commercial DINOv3 License -- fine for research; check
        # terms before commercial use.
        license="DINOv3 License (non-commercial)",
        input_resolution=[768, 768],
    )

    # The live backbone (torch + HF objects) can't be cloudpickled when MLflow logs the model;
    # it is stripped from the pickle and rebuilt in ``load_context`` from the stored config.
    _unpicklable_attrs = ("backbone",)

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_MODEL,
        image_size: int = 768,
        pad_frac: float = 0.1,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.image_size = image_size
        # Padding around a region's mask bbox before cropping, so the backbone sees a little
        # context (matches foveate's exemplar cropping).
        self.pad_frac = pad_frac
        self._device = None if device == "auto" else device
        self._load_backbone()

    def _load_backbone(self):
        """(Re)build the frozen backbone. Runs on first init and when MLflow reloads the model."""
        self.backbone = DINOv3Backbone(
            model_id=self.model_id,
            image_size=self.image_size,
            token=hf_token(),
            device=self._device,
        )

    def load_context(self, context):
        self._load_backbone()

    # -- capability handler: embed ------------------------------------------ #
    def embed(self, request: EmbedRequest, params: dict[str, Any] | None = None) -> list[EmbeddingVector]:
        """Compute the requested whole-image and per-region embeddings for one image."""
        vectors: list[EmbeddingVector] = []
        image = request.image

        wanted = [k for k in request.image_kinds if k in self.IMAGE_KINDS]
        unknown = set(request.image_kinds) - set(self.IMAGE_KINDS)
        if unknown:
            logger.warning("DINOv3Embedder ignoring unknown image kind(s): %s", sorted(unknown))

        if wanted:
            pixel_values = self.backbone.preprocess(image)
            _, cls = self.backbone.forward(pixel_values, return_cls=True)  # cls: (1, C)
            cls_np = cls[0].detach().cpu().numpy()
            if "image_cls" in wanted:
                vectors.append(self._vector("image_cls", None, ops.l2_normalize(cls_np)))

        for region in request.regions:
            mask = region.region_mask
            crop, crop_mask = ops.crop_to_mask(image, mask, self.pad_frac)
            pixel_values = self.backbone.preprocess(crop)
            grid = self.backbone.forward(pixel_values)          # (1, C, Hp, Wp)
            grid_chw = grid[0].detach().cpu().numpy()
            mask_grid = ops.resize_mask_to_grid(crop_mask, grid_chw.shape[1:])
            vectors.append(
                self._vector(self.REGION_KIND, region.region_id, ops.masked_mean(grid_chw, mask_grid))
            )

        return vectors

    def _vector(self, kind: str, region_id: int | None, vec: np.ndarray) -> EmbeddingVector:
        vec = np.asarray(vec, dtype=np.float64)
        return EmbeddingVector(
            kind=kind,
            region_id=region_id,
            model_id=self.model_id,
            dim=int(vec.shape[0]),
            vector=[float(x) for x in vec],
        )
