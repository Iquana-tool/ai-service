"""Training surface (mounted under ``/instance-segmentation``).

Ported from the former instance-segmentation-service. Submits a Celery job and
returns its task id; the gateway polls progress via MLflow (the task tags its
run with ``celery_task_id``).
"""
import logging

from celery.result import AsyncResult
from fastapi import APIRouter
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from app.tasks import train_and_register_model
from util.validate_model import validate_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train")
async def start_training(request: InstanceSegmentationTrainingRequest):
    """Start a training job asynchronously. Delegates to Celery workers."""
    validate_model(request)
    task = train_and_register_model.delay(
        request_dict=request.model_dump(),  # serialize to dict for Celery/Redis
    )
    return {"task_id": task.id}


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a training job. Requires the worker to honour revocation."""
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return {"message": "Training cancelled"}


@router.get("/train/{task_id}")
async def get_training_task_state(task_id: str):
    """Return the Celery state used to reconcile a training run's MLflow state."""
    task = AsyncResult(task_id)
    return {"task_id": task_id, "state": task.state}
