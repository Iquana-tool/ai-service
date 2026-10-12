"""Propagation surface (mounted under ``/propagation``).

Carries objects drawn on some frames of a stack -- an OCT volume's slices, a
video's frames -- to the others (#121). The model returns masks; this route turns
them into contours, as the other surfaces do, so the backend stores them like any
AI suggestion.

Two ways to ask: ``/inference`` answers once the whole stack is done; ``/stream``
sends each frame as one line of JSON the moment the model is through with it, so
the reader watches the stack fill in and can stop it part-way.
"""
import json
from logging import getLogger

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import PropagatedMask, PropagationRequest
from starlette.concurrency import run_in_threadpool

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
router = APIRouter()


def _start(request: PropagationRequest):
    """Validate the request and return the model's (lazy) iterator of frames.

    ``predict`` checks the task and parameters before returning, so a bad request
    fails here, before any response has started, and gets a proper status code.
    """
    model = MODEL_REGISTRY.get_model_by_version(request.model_registry_key, "latest")
    params = dict(request.parameters) if getattr(request, "parameters", None) else {}
    params["task"] = "propagation"
    try:
        return model.predict([request], params)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


def _entry(item: PropagatedMask, model_key: str) -> dict:
    """One frame's result, with the mask outlined as a contour (null when absent)."""
    contour = None
    if item.mask is not None:
        try:
            # One polygon per object and frame, as for prompted segmentation.
            contour = Contour.from_binary_mask(
                item.mask.mask, only_return_biggest_contour=True,
                confidence=item.presence, added_by=model_key,
            )
        except Exception:
            logger.exception("Could not outline object %s on frame %s; reporting it as absent.",
                             item.object_id, item.frame_index)
    return {
        "object_id": item.object_id,
        "frame_index": item.frame_index,
        "presence": item.presence,
        "contour": contour,
    }


@router.post("/inference", tags=["propagation"])
async def propagate(request: PropagationRequest):
    """Propagate objects through a stack and answer when every frame is done.

    :return: ``result``: one entry per object and filled frame,
        ``{object_id, frame_index, presence, contour}``. ``contour`` is null where
        the model found the object absent; ``presence`` (0..1) says how sure it is
        the object is there, which is what flags a frame for a person to check.
    """
    frames = _start(request)
    # A whole stack takes seconds; keep the event loop free meanwhile.
    propagated = await run_in_threadpool(list, frames)
    result = [_entry(item, request.model_registry_key) for item in propagated]
    filled = sum(entry["contour"] is not None for entry in result)
    return {
        "success": True,
        "message": f"Propagated {len(request.objects)} object(s) onto {filled} frame(s).",
        "result": result,
    }


@router.post("/stream", tags=["propagation"])
async def propagate_stream(request: PropagationRequest):
    """Propagate objects through a stack, one NDJSON line per frame as it is done.

    Each line is an ``/inference`` result entry. A failure part-way ends the stream
    with ``{"error": "..."}``. Closing the connection stops the work after the
    frame in progress.
    """
    frames = _start(request)

    def lines():
        # A sync generator: Starlette runs it in a worker thread, one frame at a
        # time, so the model never blocks the event loop.
        try:
            for item in frames:
                entry = _entry(item, request.model_registry_key)
                if entry["contour"] is not None:
                    entry["contour"] = entry["contour"].model_dump(mode="json")
                yield json.dumps(entry) + "\n"
        except Exception as exc:  # noqa: BLE001 -- reported in-band; the status is already sent
            logger.exception("Propagation failed part-way.")
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
