"""Instance-suggestion surface (mounted under ``/instance-suggestion``).

Ported from the former instance-discovery-service. The session router keeps its
``/annotation_session`` prefix, so under the mount the full path is
``/instance-suggestion/annotation_session/run``.
"""
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceSuggestionRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@session_router.post("/run")
async def infer_instances(request: InstanceSuggestionRequest):
    """Infer instances from seed instances."""
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    # model is an MLflow PyFuncModel; predict(data) forwards to the model's
    # predict(context, model_input=data, params), returning (masklets, scores).
    # The explicit task disambiguates suggestion from same-request-type tasks
    # (e.g. cross-image suggestion) on a multi-task model.
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "instance-suggestion"
    masklets, scores = model.predict([request], params)
    result = []
    for masklet, score in zip(masklets, scores):
        try:
            result.append(Contour.from_binary_mask(masklet, confidence=score))
        except Exception:
            logger.exception("Failed to build a contour from a suggested masklet; skipping it.")
    return {
        "success": True,
        "message": f"Detected {len(result)} objects for user {request.user_id}",
        "result": result,
    }
