import logging
import json
import re
from copy import deepcopy
from uuid import UUID

import mlflow
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from iquana_toolbox.ai.base_classes import InstanceSegmentationModel
from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from celery_app import app
from app.training_runs import (
    TERMINAL_TRAINING_STATES,
    get_training_run_state,
    restore_training_run,
    set_training_run_state,
    terminate_training_run,
    utc_now_iso,
)
from paths import MLFLOW_URL

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, RequestsConnectionError, RequestsTimeout)


class TrainingAlreadyTerminated(RuntimeError):
    """Raised when a cancelled or timed-out run reaches a worker checkpoint."""


@app.task(bind=True)
def train_and_register_model(
    self,
    request_dict: dict,
    model_run_name: str | None = None,
    training_dataset_name: str | None = None,
    training_run_id: str | None = None,
):
    """Generic training dispatcher. Loads the model from the registry and
    delegates all training logic to the model's own ``train()`` method.

    Args:
        request_dict: Serialised ``InstanceSegmentationTrainingRequest`` dict.
        model_run_name: Human-readable output model/run name.
        training_dataset_name: Human-readable dataset name captured at submission.
        training_run_id: MLflow run created before Celery dispatch.
    """
    if not training_run_id:
        raise ValueError("A pre-created MLflow training run is required.")
    try:
        _ensure_training_active(training_run_id)
        request = InstanceSegmentationTrainingRequest.model_validate(request_dict)
        mlflow.set_tracking_uri(MLFLOW_URL)
        with mlflow.start_run(run_id=training_run_id):
            set_training_run_state(
                training_run_id,
                "running",
                message="Training started.",
                tags={"started_at": utc_now_iso()},
            )
            self.update_state(state="PROGRESS", meta={"status": "training started"})

            registry: MLFlowModelRegistry = MLFlowModelRegistry(MLFLOW_URL)
            # Resolve the base model via the MLflow-supported latest version path.
            # The pyfunc wrapper owns the actual trainable Python model instance.
            pyfunc_model = registry.get_model_by_version(
                request.model_registry_key, "latest"
            )
            model: InstanceSegmentationModel = pyfunc_model._model_impl.python_model
            model.train(request)

        # Register the fine-tuned model. register_model opens its own MLflow run, so
        # this must happen after the training run above has closed (no nesting).
        # dataset_id / user_id must be set or the model does not get logged.
        _ensure_training_active(training_run_id)
        base_registry_key = request.model_registry_key
        output_registry_key = _trained_model_registry_key(
            base_registry_key, request.dataset_id, self.request.id
        )
        selected_labels = sorted(request.labels, key=lambda label: int(label.id))
        selected_label_ids = [int(label.id) for label in selected_labels]
        selected_label_names = [str(label.name) for label in selected_labels]
        model.model_info = deepcopy(model.model_info)
        model.model_info.registry_key = output_registry_key
        model.model_info.name = (model_run_name or "").strip() or (
            f"train_{base_registry_key}_ds{request.dataset_id}"
        )
        model.model_info.trainable = False
        model.model_info.label_ids = selected_label_ids
        model.model_info.tags.update({
            "dataset_id": str(request.dataset_id),
            "user_id": str(request.user_id),
            "training_task_id": str(self.request.id),
            "selected_label_ids": json.dumps(selected_label_ids),
            "trained_label_names": json.dumps(selected_label_names, ensure_ascii=False),
            "trained_on_dataset_id": str(request.dataset_id),
            "base_model_registry_key": base_registry_key,
            "segmentation_mode": "flat",
            "trainable": "false",
        })
        if training_dataset_name:
            model.model_info.tags["trained_on_dataset_name"] = training_dataset_name.strip()
        registry.register_model(model)
        terminate_training_run(
            training_run_id,
            state="completed",
            mlflow_status="FINISHED",
            message="Training completed.",
            tags={"output_model_registry_key": output_registry_key},
        )
        return {"status": "completed", "model_registry_key": output_registry_key}
    except TrainingAlreadyTerminated:
        logger.info("Training run %s was already terminated; skipping publication.", training_run_id)
        return {
            "status": get_training_run_state(training_run_id),
            "model_registry_key": None,
        }
    except _TRANSIENT_ERRORS as exc:
        logger.warning("Transient training failure: %s", exc, exc_info=True)
        if self.request.retries < _MAX_RETRIES:
            set_training_run_state(
                training_run_id,
                "running",
                message="Temporary infrastructure error; retrying training.",
            )
            restore_training_run(training_run_id)
            raise self.retry(exc=exc, countdown=60, max_retries=_MAX_RETRIES)
        terminate_training_run(
            training_run_id,
            state="failed",
            mlflow_status="FAILED",
            message="Training failed after repeated infrastructure errors.",
        )
        raise
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        terminate_training_run(
            training_run_id,
            state="failed",
            mlflow_status="FAILED",
            message=_public_failure_message(exc),
        )
        raise


def _trained_model_registry_key(base_key: str, dataset_id: int, task_id: str) -> str:
    """Build one deterministic MLflow registry key per training task."""
    normalized_base = re.sub(r"[^a-z0-9.-]+", "-", base_key.strip().lower()).strip("-.")
    if not normalized_base:
        normalized_base = "model"
    normalized_task_id = str(UUID(str(task_id)))
    return f"trained-{normalized_base}-ds{dataset_id}-{normalized_task_id}"


def _ensure_training_active(run_id: str) -> None:
    if get_training_run_state(run_id) in TERMINAL_TRAINING_STATES:
        raise TrainingAlreadyTerminated


def _public_failure_message(exc: Exception) -> str:
    """Return a bounded user-facing failure without losing the server traceback."""
    message = str(exc).strip()
    if isinstance(exc, (ValueError, RuntimeError)) and message:
        return message[:500]
    return "Training failed unexpectedly. Check the server logs for details."
