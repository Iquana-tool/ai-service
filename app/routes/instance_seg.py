"""Full instance-segmentation surface (mounted under ``/instance-segmentation``).

Ported from the former instance-segmentation-service. Inference plus the
annotation-session ``/run`` variant that returns the gateway envelope.
"""
from fastapi import APIRouter
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from app.state import MODEL_REGISTRY
from util.validate_model import validate_model

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@router.post("/inference")
async def inference(request: InstanceSegmentationRequest) -> list[Contour]:
    """Load a model from the MLflow registry and run instance segmentation."""
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    return model.predict(request, {"task": "instance-segmentation"})


@session_router.post("/run")
async def run_inference(request: InstanceSegmentationRequest):
    """Run instance segmentation for an annotation session.

    Returns the detected instances wrapped in the ``{success, message, result}``
    envelope the annotation gateway expects from every session backend.
    """
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    contours = model.predict(request, {"task": "instance-segmentation"})
    return {
        "success": True,
        "message": f"Detected {len(contours)} instances for user {request.user_id}",
        "result": contours,
    }
