"""Training surface (mounted under ``/instance-segmentation``).

Ported from the former instance-segmentation-service. Submits a Celery job and
returns its task id; the gateway polls progress via MLflow (the task tags its
run with ``celery_task_id``).
"""
import logging
import json
import time
from typing import Optional
from uuid import UUID, uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, status
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from app.tasks import train_and_register_model
from app.training_runs import (
    TASK_ID_TAG,
    TERMINAL_TRAINING_STATES,
    TRAINING_STATE_TAG,
    create_training_run,
    find_training_run,
    terminate_training_run,
    utc_now_iso,
)
from paths import TRAINING_START_TIMEOUT_SECONDS
from util.validate_model import validate_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train")
async def start_training(
    request: InstanceSegmentationTrainingRequest,
    model_run_name: Optional[str] = Query(default=None, max_length=120),
    dataset_name: Optional[str] = Query(default=None, max_length=200),
):
    """Start a training job asynchronously. Delegates to Celery workers.

    Args:
        request: Typed training configuration (labels, hyperparameters, etc.).
        model_run_name: Optional human-readable alias for this run stored as an
            MLflow tag.  Surfaced in the run-history UI as a display name.
        dataset_name: Optional snapshot of the dataset's human-readable name.
    """
    validate_model(request)
    task_id = str(uuid4())
    raw_model_run_name = model_run_name if isinstance(model_run_name, str) else None
    display_name = (raw_model_run_name or "").strip() or (
        f"train_{request.model_registry_key}_ds{request.dataset_id}"
    )
    raw_dataset_name = dataset_name if isinstance(dataset_name, str) else None
    training_dataset_name = (raw_dataset_name or "").strip() or None
    queued_at = time.time()
    start_deadline = queued_at + TRAINING_START_TIMEOUT_SECONDS
    run = create_training_run(
        run_name=display_name,
        tags={
            TASK_ID_TAG: task_id,
            TRAINING_STATE_TAG: "starting",
            "dataset_id": request.dataset_id,
            "user_id": request.user_id,
            "selected_label_ids": json.dumps(sorted(int(label.id) for label in request.labels)),
            "run_name": display_name,
            "queued_at": queued_at,
            "start_deadline": start_deadline,
            **({"dataset_name": training_dataset_name} if training_dataset_name else {}),
        },
    )
    try:
        train_and_register_model.apply_async(
            kwargs={
                "request_dict": request.model_dump(),
                "model_run_name": display_name,
                "training_dataset_name": training_dataset_name,
                "training_run_id": run.info.run_id,
            },
            task_id=task_id,
            expires=TRAINING_START_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error("Failed to dispatch training task %s: %s", task_id, exc, exc_info=True)
        terminate_training_run(
            run.info.run_id,
            state="failed",
            mlflow_status="FAILED",
            message="Training could not be queued.",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training could not be queued.",
        ) from exc
    return {"task_id": task_id}


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a queued/running task and reconcile its durable run state."""
    task_id = _canonical_task_id(task_id)
    run = find_training_run(task_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training task not found.")
    training_state = run.data.tags.get(TRAINING_STATE_TAG, "starting")
    if training_state in TERMINAL_TRAINING_STATES:
        return {"task_id": task_id, "state": _effective_state(training_state, "PENDING")}
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    terminate_training_run(
        run.info.run_id,
        state="cancelled",
        mlflow_status="KILLED",
        message="Training was cancelled.",
        tags={"cancelled_at": utc_now_iso()},
    )
    return {"task_id": task_id, "state": "REVOKED", "message": "Training was cancelled."}


@router.get("/train/{task_id}")
async def get_training_task_state(task_id: str):
    """Return effective Celery/MLflow state and reconcile an expired queued task."""
    task_id = _canonical_task_id(task_id)
    run = find_training_run(task_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training task not found.")
    task = AsyncResult(task_id)
    tags = run.data.tags
    training_state = tags.get(TRAINING_STATE_TAG, "starting")
    deadline = _float_tag(tags.get("start_deadline"))
    if (
        training_state == "starting"
        and task.state in {"PENDING", "REVOKED"}
        and deadline is not None
        and time.time() >= deadline
    ):
        task.revoke(terminate=False)
        training_state = "timed_out"
        terminate_training_run(
            run.info.run_id,
            state=training_state,
            mlflow_status="KILLED",
            message="Training did not start before the queue deadline.",
            tags={"timed_out_at": utc_now_iso()},
        )
        tags = find_training_run(task_id).data.tags

    response = {
        "task_id": task_id,
        "state": _effective_state(training_state, task.state),
        "training_state": training_state,
        "run_id": run.info.run_id,
    }
    for source_key, response_key in (
        ("status_message", "message"),
        ("queued_at", "queued_at"),
        ("start_deadline", "start_deadline"),
        ("started_at", "started_at"),
    ):
        if source_key in tags:
            response[response_key] = tags[source_key]
    return response


def _canonical_task_id(task_id: str) -> str:
    """Reject arbitrary filter input; every task started here uses a UUID."""
    try:
        return str(UUID(task_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training task not found.") from exc


def _float_tag(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _effective_state(training_state: str, celery_state: str) -> str:
    return {
        "completed": "SUCCESS",
        "failed": "FAILURE",
        "cancelled": "REVOKED",
        "timed_out": "TIMED_OUT",
        "running": "PROGRESS",
    }.get(training_state, celery_state)
