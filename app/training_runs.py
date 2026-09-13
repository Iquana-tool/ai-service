"""Small MLflow helpers for durable instance-training lifecycle state."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

from paths import MLFLOW_URL

TRAINING_EXPERIMENT = "instance-segmentation-training"
TASK_ID_TAG = "celery_task_id"
TRAINING_STATE_TAG = "training_state"

TERMINAL_TRAINING_STATES = frozenset({"completed", "failed", "cancelled", "timed_out"})


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for an MLflow tag."""
    return datetime.now(timezone.utc).isoformat()


def get_client() -> MlflowClient:
    """Return a client bound to the configured tracking server."""
    mlflow.set_tracking_uri(MLFLOW_URL)
    return MlflowClient(tracking_uri=MLFLOW_URL)


def get_or_create_experiment_id(client: MlflowClient) -> str:
    """Resolve the single experiment used for instance-training runs."""
    experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(TRAINING_EXPERIMENT)


def create_training_run(*, run_name: str, tags: dict[str, Any]) -> Run:
    """Create a discoverable training run without activating it in this process."""
    client = get_client()
    experiment_id = get_or_create_experiment_id(client)
    return client.create_run(
        experiment_id,
        start_time=int(time.time() * 1000),
        run_name=run_name,
        tags={key: str(value) for key, value in tags.items()},
    )


def find_training_run(task_id: str) -> Run | None:
    """Find the unique training run associated with a generated Celery task ID."""
    client = get_client()
    experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.{TASK_ID_TAG} = '{task_id}'",
        max_results=2,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        return None
    if len(runs) > 1:
        raise RuntimeError(f"Multiple training runs found for task {task_id}.")
    return runs[0]


def set_training_run_state(
    run_id: str,
    state: str,
    *,
    message: str | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    """Persist normalized lifecycle state and optional public metadata."""
    client = get_client()
    client.set_tag(run_id, TRAINING_STATE_TAG, state)
    if message is not None:
        client.set_tag(run_id, "status_message", message)
    for key, value in (tags or {}).items():
        client.set_tag(run_id, key, str(value))


def get_training_run_state(run_id: str) -> str:
    """Read the durable normalized state for a training run."""
    return get_client().get_run(run_id).data.tags.get(TRAINING_STATE_TAG, "starting")


def restore_training_run(run_id: str) -> None:
    """Restore MLflow's status after Celery schedules a transient retry."""
    get_client().update_run(run_id, status="RUNNING")


def terminate_training_run(
    run_id: str,
    *,
    state: str,
    mlflow_status: str,
    message: str | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    """Set final lifecycle tags and terminate the MLflow run once."""
    set_training_run_state(run_id, state, message=message, tags=tags)
    get_client().set_terminated(run_id, status=mlflow_status)
