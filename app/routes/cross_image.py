"""Cross-image concept-suggestion surface (mounted under ``/cross-image-suggestion``).

Transfers a concept onto a target image from exemplars in *other* images (SAM 3's concat
workaround). The exemplars are typically the top hits the backend's retrieval strategy
selected. Output mirrors instance suggestion -- suggested instances on the target image, which
the backend accepts as contours.
"""
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@session_router.post("/run")
async def suggest_cross_image(request: CrossImageSuggestionRequest):
    """Suggest instances of a concept on the target image from cross-image exemplars."""
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "cross-image-suggestion"
    masks, scores = model.predict([request], params)
    result = []
    for mask, score in zip(masks, scores):
        try:
            result.append(Contour.from_binary_mask(mask, confidence=score))
        except Exception:
            logger.exception("Failed to build a contour from a cross-image suggestion; skipping it.")
    return {
        "success": True,
        "message": f"Suggested {len(result)} object(s) for user {request.user_id}",
        "result": result,
    }
