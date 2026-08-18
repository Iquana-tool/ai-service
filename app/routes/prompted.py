"""Prompted-segmentation surface (mounted under ``/prompted-segmentation``).

Ported verbatim from the former prompted-seg-service so the gateway contract is
unchanged; only the model registry is now the shared one.
"""
from logging import getLogger

from fastapi import APIRouter
from iquana_toolbox.schemas.networking.http.services import PromptedSegmentationRequest

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


@router.post("/inference", tags=["prompted-segmentation"])
async def inference(request: PromptedSegmentationRequest):
    """Segment an image using 2D prompts.

    :param request: PromptedSegmentationRequest with image_url, user_id,
        model_registry_key, prompts and an optional previous mask.
    :return: Segmentation result with the candidate contours.
    """
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    # model is an MLflow PyFuncModel; predict(data) forwards to the model's
    # predict(context, model_input=data, params), which returns a list[Contour].
    # Passing the task explicitly lets a multi-task model (e.g. SAM 3) dispatch to
    # its prompted handler unambiguously. Return all candidates so the backend can
    # pick the best one (e.g. discard a candidate that re-segments the parent).
    contours = model.predict([request], {"task": "prompted-segmentation"})
    return {
        "success": True,
        "message": f"Successfully performed prompted segmentation. Found {len(contours)} candidate(s).",
        "result": contours,
    }
