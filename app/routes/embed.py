"""Embedding surface (mounted under ``/embed``).

Precomputes DINOv3 (or any ``embed``-capable model) feature vectors for an image and/or its
masked regions. The backend calls this once per image/contour and persists the returned
vectors in the pgvector store; cross-image exemplar retrieval then runs entirely off those
stored embeddings, never re-embedding on the annotation hot path.
"""
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.networking.http.services import EmbedRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


@router.post("/inference", tags=["embed"])
async def inference(request: EmbedRequest):
    """Compute feature embeddings for an image and/or its masked regions.

    :param request: EmbedRequest with image_url, user_id, model_registry_key, the whole-image
        ``image_kinds`` to compute and any ``regions`` (RLE masks) to embed.
    :return: The computed embedding vectors; each carries its ``kind``, ``region_id``
        (None for whole-image kinds), ``model_id`` and the L2-normalized ``vector``.
    """
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    # model is an MLflow PyFuncModel; predict(data) forwards to the model's
    # predict(context, model_input=data, params) -> list[EmbeddingVector]. The explicit task
    # lets a multi-task model dispatch to its embed handler unambiguously.
    vectors = model.predict([request], {"task": "embed"})
    return {
        "success": True,
        "message": f"Computed {len(vectors)} embedding(s) for user {request.user_id}.",
        "result": vectors,
    }
