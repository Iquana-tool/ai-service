import logging

import mlflow
from iquana_toolbox.ai.base_classes import InstanceSegmentationModel
from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from celery_app import app
from paths import MLFLOW_URL

logger = logging.getLogger(__name__)

# Training runs are grouped under this MLflow experiment. The gateway resolves a
# Celery task id back to its run by searching this experiment for the
# ``celery_task_id`` tag, so progress can be polled straight from MLflow.
TRAINING_EXPERIMENT = "instance-segmentation-training"


@app.task(bind=True)
def train_and_register_model(self, request_dict: dict, model_run_name: str | None = None):
    """Generic training dispatcher. Loads the model from the registry and
    delegates all training logic to the model's own ``train()`` method.

    Args:
        request_dict: Serialised ``InstanceSegmentationTrainingRequest`` dict.
        model_run_name: Optional human-readable alias stored as an MLflow tag.
    """
    try:
        registry: MLFlowModelRegistry = MLFlowModelRegistry(MLFLOW_URL)

        # Reconstruct the typed request inside the worker.
        request = InstanceSegmentationTrainingRequest.model_validate(request_dict)
        # Resolve the base model via the ``/latest`` version path (MLflow-supported),
        # rather than an ``@latest`` alias, which registration never creates.
        # ``_model_impl`` is MLflow's pyfunc wrapper; ``.python_model`` is the actual
        # model instance (with load_context already applied).
        pyfunc_model = registry.get_model_by_version(request.model_registry_key, "latest")
        model: InstanceSegmentationModel = pyfunc_model._model_impl.python_model

        self.update_state(state="PROGRESS", meta={"status": "training started"})

        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment(TRAINING_EXPERIMENT)
        # A new run per task. We can't force the run id to equal the Celery task id
        # (MLflow assigns its own), so we tag the run with the task id; the gateway
        # finds it by that tag when polling progress.
        with mlflow.start_run(run_name=f"train_{request.model_registry_key}_ds{request.dataset_id}"):
            mlflow.set_tag("celery_task_id", self.request.id)
            mlflow.set_tag("dataset_id", str(request.dataset_id))
            mlflow.set_tag("user_id", str(request.user_id))
            if model_run_name:
                # Surface the user-supplied alias in the MLflow UI and run history.
                mlflow.set_tag("run_name", model_run_name)
            model.train(request)  # logs params + per-epoch loss/epoch to this active run

        # Register the fine-tuned model. register_model opens its own MLflow run, so
        # this must happen after the training run above has closed (no nesting).
        # dataset_id / user_id must be set or the model does not get logged.
        model.model_info.tags["dataset_id"] = request.dataset_id
        model.model_info.tags["user_id"] = request.user_id
        registry.register_model(model)

        return {"status": "completed"}
    except Exception as e:
        # exc_info + exc=e are both needed to see anything useful: without the traceback the
        # worker log shows a bare message, and without ``exc`` the retry drops the original
        # exception, so once the retries are used up the task fails with a contextless
        # MaxRetriesExceededError instead of the error that actually broke training.
        logger.error("Training failed: %s", e, exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)
