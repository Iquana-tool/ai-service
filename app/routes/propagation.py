"""Propagation surface (mounted under ``/propagation``).

Carries objects drawn on some frames of a stack -- an OCT volume's slices, a
video's frames -- to the others (#121). The model returns masks; this route turns
them into contours, as the other surfaces do, so the backend stores them like any
AI suggestion.
"""
from logging import getLogger

from fastapi import APIRouter, HTTPException, status
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import PropagationRequest
from starlette.concurrency import run_in_threadpool

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


@router.post("/inference", tags=["propagation"])
async def propagate(request: PropagationRequest):
    """Propagate objects through a stack.

    :return: ``result``: one entry per object and filled frame,
        ``{object_id, frame_index, presence, contour}``. ``contour`` is null where
        the model found the object absent; ``presence`` (0..1) says how sure it is
        the object is there, which is what flags a frame for a person to check.
    """
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "propagation"
    try:
        # A whole stack takes seconds; keep the event loop free meanwhile.
        propagated = await run_in_threadpool(model.predict, [request], params)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    result = []
    for item in propagated:
        contour = None
        if item.mask is not None:
            try:
                # One polygon per object and frame, as for prompted segmentation.
                contour = Contour.from_binary_mask(
                    item.mask.mask, only_return_biggest_contour=True,
                    confidence=item.presence, added_by=request.model_registry_key,
                )
            except Exception:
                logger.exception("Could not outline object %s on frame %s; reporting it as absent.",
                                 item.object_id, item.frame_index)
        result.append({
            "object_id": item.object_id,
            "frame_index": item.frame_index,
            "presence": item.presence,
            "contour": contour,
        })
    filled = sum(entry["contour"] is not None for entry in result)
    return {
        "success": True,
        "message": f"Propagated {len(request.objects)} object(s) onto {filled} frame(s).",
        "result": result,
    }
