"""Full instance-segmentation surface (mounted under ``/instance-segmentation``).

Ported from the former instance-segmentation-service. Inference plus the
annotation-session ``/run`` variant that returns the gateway envelope.
"""
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, status
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from app.state import MODEL_REGISTRY
from util.validate_model import validate_model

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


def _prediction_parts(prediction: Any) -> tuple[Any, Any, bool]:
    """Unpack an optional metadata wrapper while preserving legacy list responses."""
    if isinstance(prediction, Mapping) and "contours" in prediction:
        contours = prediction["contours"]
        if not isinstance(contours, list):
            raise RuntimeError("Instance-segmentation provider returned non-list contours.")
        return contours, prediction.get("prediction_metadata"), True
    return prediction, None, False


@router.post("/inference")
async def inference(request: InstanceSegmentationRequest) -> Any:
    """Load a model from the MLflow registry and run instance segmentation."""
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "instance-segmentation"
    try:
        prediction = model.predict(request, params)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    contours, prediction_metadata, has_metadata = _prediction_parts(prediction)
    if has_metadata:
        return {"contours": contours, "prediction_metadata": prediction_metadata}
    return contours


@session_router.post("/run")
async def run_inference(request: InstanceSegmentationRequest):
    """Run instance segmentation for an annotation session.

    Returns the detected instances wrapped in the ``{success, message, result}``
    envelope the annotation gateway expects from every session backend.
    """
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "instance-segmentation"
    try:
        prediction = model.predict(request, params)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    contours, prediction_metadata, has_metadata = _prediction_parts(prediction)
    response = {
        "success": True,
        "message": f"Detected {len(contours)} instances for user {request.user_id}",
        "result": contours,
    }
    if has_metadata:
        response["prediction_metadata"] = prediction_metadata
    return response
