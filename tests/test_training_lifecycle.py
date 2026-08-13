"""Focused lifecycle and publication tests for flat instance training."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import mlflow
import pytest
from fastapi import HTTPException
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from requests.exceptions import Timeout as RequestsTimeout

import app.routes.training as training_routes
import app.tasks as training_tasks
import app.training_runs as training_runs


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch) -> str:
    """Use one isolated real MLflow store for lifecycle assertions."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(training_runs, "MLFLOW_URL", uri)
    monkeypatch.setattr(training_tasks, "MLFLOW_URL", uri)
    mlflow.end_run()
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.end_run()


@pytest.fixture
def training_request() -> InstanceSegmentationTrainingRequest:
    return InstanceSegmentationTrainingRequest(
        dataset_id=11,
        image_folder_path="/tmp/images",
        model_registry_key="mask2former",
        user_id="trainer",
        hyper_parameter={"epochs": 1, "learning_rate": 1e-3, "batch_size": 1},
        labels=[Label(id=7, dataset_id=11, name="cell", value=1)],
        annotation_file_url="/tmp/annotations.json",
    )


def _new_run(task_id: str, **extra_tags):
    return training_runs.create_training_run(
        run_name="test run",
        tags={
            training_runs.TASK_ID_TAG: task_id,
            training_runs.TRAINING_STATE_TAG: "starting",
            "queued_at": "1",
            "start_deadline": "9999999999",
            **extra_tags,
        },
    )


@pytest.mark.asyncio
async def test_submit_creates_run_before_dispatch(
    tracking_uri, training_request, monkeypatch
) -> None:
    monkeypatch.setattr(training_routes, "validate_model", lambda _request: None)
    observed = {}

    class DispatchProbe:
        def apply_async(self, *, kwargs, task_id, expires):
            run = training_runs.find_training_run(task_id)
            assert run is not None
            observed.update(
                task_id=task_id,
                run_id=run.info.run_id,
                kwargs=kwargs,
                expires=expires,
                tags=run.data.tags,
            )

    monkeypatch.setattr(training_routes, "train_and_register_model", DispatchProbe())
    response = await training_routes.start_training(
        training_request,
        "  coral model  ",
        " Cells dataset ",
    )

    assert response == {"task_id": observed["task_id"]}
    assert observed["kwargs"]["training_run_id"] == observed["run_id"]
    assert observed["kwargs"]["model_run_name"] == "coral model"
    assert observed["kwargs"]["training_dataset_name"] == "Cells dataset"
    assert observed["tags"]["training_state"] == "starting"
    assert observed["tags"]["run_name"] == "coral model"
    assert observed["tags"]["selected_label_ids"] == "[7]"
    assert observed["tags"]["dataset_name"] == "Cells dataset"
    assert observed["expires"] == training_routes.TRAINING_START_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_dispatch_failure_leaves_failed_discoverable_run(
    tracking_uri, training_request, monkeypatch
) -> None:
    monkeypatch.setattr(training_routes, "validate_model", lambda _request: None)

    class FailingDispatch:
        def apply_async(self, **_kwargs):
            raise ConnectionError("broker unavailable")

    monkeypatch.setattr(training_routes, "train_and_register_model", FailingDispatch())
    with pytest.raises(HTTPException) as error:
        await training_routes.start_training(training_request, None)
    assert error.value.status_code == 503

    client = training_runs.get_client()
    experiment = client.get_experiment_by_name(training_runs.TRAINING_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags["training_state"] == "failed"
    assert runs[0].data.tags["status_message"] == "Training could not be queued."
    assert runs[0].info.status == "FAILED"


class _FakeAsyncResult:
    def __init__(self, state: str = "PENDING") -> None:
        self.state = state
        self.revocations: list[dict] = []

    def revoke(self, **kwargs) -> None:
        self.revocations.append(kwargs)


@pytest.mark.asyncio
async def test_expired_pending_task_becomes_timed_out(tracking_uri, monkeypatch) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id, start_deadline="0")
    result = _FakeAsyncResult()
    monkeypatch.setattr(training_routes, "AsyncResult", lambda _task_id: result)

    response = await training_routes.get_training_task_state(task_id)

    assert response["state"] == "TIMED_OUT"
    assert response["training_state"] == "timed_out"
    assert result.revocations == [{"terminate": False}]
    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "KILLED"
    assert stored.data.tags["training_state"] == "timed_out"


@pytest.mark.asyncio
async def test_expired_revoked_task_becomes_timed_out_idempotently(tracking_uri, monkeypatch) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id, start_deadline="0")
    result = _FakeAsyncResult(state="REVOKED")
    monkeypatch.setattr(training_routes, "AsyncResult", lambda _task_id: result)

    first = await training_routes.get_training_task_state(task_id)
    second = await training_routes.get_training_task_state(task_id)

    assert first["state"] == "TIMED_OUT"
    assert first["training_state"] == "timed_out"
    assert second["state"] == "TIMED_OUT"
    assert second["training_state"] == "timed_out"
    assert result.revocations == [{"terminate": False}]
    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "KILLED"
    assert stored.data.tags["training_state"] == "timed_out"


@pytest.mark.asyncio
async def test_cancel_marks_run_killed_and_is_idempotent(tracking_uri, monkeypatch) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    result = _FakeAsyncResult(state="PROGRESS")
    monkeypatch.setattr(training_routes, "AsyncResult", lambda _task_id: result)

    first = await training_routes.cancel_training(task_id)
    second = await training_routes.cancel_training(task_id)

    assert first["state"] == "REVOKED"
    assert second["state"] == "REVOKED"
    assert result.revocations == [{"terminate": True}]
    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "KILLED"
    assert stored.data.tags["training_state"] == "cancelled"


@pytest.mark.asyncio
async def test_unknown_or_invalid_task_id_returns_404(tracking_uri) -> None:
    for task_id in ("not-a-uuid", str(uuid4())):
        with pytest.raises(HTTPException) as error:
            await training_routes.get_training_task_state(task_id)
        assert error.value.status_code == 404


class _FakeModel:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.active_run_ids: list[str] = []
        self.model_info = SimpleNamespace(
            registry_key="mask2former",
            name="Mask2Former",
            trainable=True,
            label_ids=[],
            tags={},
        )

    def train(self, _request) -> None:
        active_run = mlflow.active_run()
        assert active_run is not None
        self.active_run_ids.append(active_run.info.run_id)
        if self.failure is not None:
            raise self.failure


class _FakeRegistry:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.registered_infos = []

    def get_model_by_version(self, _key, _version):
        return SimpleNamespace(
            _model_impl=SimpleNamespace(python_model=self.model)
        )

    def register_model(self, model) -> None:
        self.registered_infos.append(deepcopy(model.model_info))


def _run_worker(
    monkeypatch,
    *,
    request: InstanceSegmentationTrainingRequest,
    run_id: str,
    task_id: str,
    registry: _FakeRegistry,
    retries: int = 0,
    training_dataset_name: str | None = None,
):
    monkeypatch.setattr(training_tasks, "MLFlowModelRegistry", lambda _uri: registry)
    monkeypatch.setattr(
        training_tasks.train_and_register_model,
        "update_state",
        lambda **_kwargs: None,
    )
    training_tasks.train_and_register_model.push_request(id=task_id, retries=retries)
    try:
        return training_tasks.train_and_register_model.run(
            request_dict=request.model_dump(),
            model_run_name="trained coral model",
            training_dataset_name=training_dataset_name,
            training_run_id=run_id,
        )
    finally:
        training_tasks.train_and_register_model.pop_request()


def test_worker_resumes_run_and_publishes_unique_flat_model(
    tracking_uri, training_request, monkeypatch
) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    model = _FakeModel()
    registry = _FakeRegistry(model)

    result = _run_worker(
        monkeypatch,
        request=training_request,
        run_id=run.info.run_id,
        task_id=task_id,
        registry=registry,
        training_dataset_name="Cells dataset",
    )

    expected_key = f"trained-mask2former-ds11-{task_id}"
    assert result == {"status": "completed", "model_registry_key": expected_key}
    assert model.active_run_ids == [run.info.run_id]
    assert len(registry.registered_infos) == 1
    info = registry.registered_infos[0]
    assert info.registry_key == expected_key
    assert info.name == "trained coral model"
    assert info.trainable is False
    assert info.label_ids == [7]
    assert info.tags == {
        "dataset_id": "11",
        "user_id": "trainer",
        "training_task_id": task_id,
        "selected_label_ids": "[7]",
        "trained_label_names": "[\"cell\"]",
        "trained_on_dataset_id": "11",
        "trained_on_dataset_name": "Cells dataset",
        "base_model_registry_key": "mask2former",
        "segmentation_mode": "flat",
        "trainable": "false",
    }
    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "FINISHED"
    assert stored.data.tags["training_state"] == "completed"
    assert stored.data.tags["output_model_registry_key"] == expected_key


def test_permanent_worker_failure_is_not_retried(
    tracking_uri, training_request, monkeypatch
) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    registry = _FakeRegistry(_FakeModel(ValueError("invalid training data")))
    monkeypatch.setattr(
        training_tasks.train_and_register_model,
        "retry",
        lambda **_kwargs: pytest.fail("permanent errors must not retry"),
    )

    with pytest.raises(ValueError, match="invalid training data"):
        _run_worker(
            monkeypatch,
            request=training_request,
            run_id=run.info.run_id,
            task_id=task_id,
            registry=registry,
        )

    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "FAILED"
    assert stored.data.tags["training_state"] == "failed"
    assert stored.data.tags["status_message"] == "invalid training data"


def test_cancelled_run_never_starts_or_publishes(
    tracking_uri, training_request, monkeypatch
) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    training_runs.terminate_training_run(
        run.info.run_id,
        state="cancelled",
        mlflow_status="KILLED",
        message="Training was cancelled.",
    )
    model = _FakeModel()
    registry = _FakeRegistry(model)

    result = _run_worker(
        monkeypatch,
        request=training_request,
        run_id=run.info.run_id,
        task_id=task_id,
        registry=registry,
    )

    assert result == {"status": "cancelled", "model_registry_key": None}
    assert model.active_run_ids == []
    assert registry.registered_infos == []


def test_cancelled_after_training_never_publishes(
    tracking_uri, training_request, monkeypatch
) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    model = _FakeModel()
    registry = _FakeRegistry(model)
    original_train = model.train

    def train_then_cancel(request) -> None:
        original_train(request)
        training_runs.terminate_training_run(
            run.info.run_id,
            state="cancelled",
            mlflow_status="KILLED",
            message="Training was cancelled.",
        )

    monkeypatch.setattr(model, "train", train_then_cancel)
    result = _run_worker(
        monkeypatch,
        request=training_request,
        run_id=run.info.run_id,
        task_id=task_id,
        registry=registry,
    )

    assert result == {"status": "cancelled", "model_registry_key": None}
    assert registry.registered_infos == []


def test_transient_worker_failure_uses_bounded_retry(
    tracking_uri, training_request, monkeypatch
) -> None:
    task_id = str(uuid4())
    run = _new_run(task_id)
    registry = _FakeRegistry(_FakeModel(RequestsTimeout("temporary timeout")))

    class RetryRequested(Exception):
        pass

    observed = {}

    def retry(**kwargs):
        observed.update(kwargs)
        raise RetryRequested

    monkeypatch.setattr(training_tasks.train_and_register_model, "retry", retry)
    with pytest.raises(RetryRequested):
        _run_worker(
            monkeypatch,
            request=training_request,
            run_id=run.info.run_id,
            task_id=task_id,
            registry=registry,
        )

    assert isinstance(observed["exc"], RequestsTimeout)
    assert observed["max_retries"] == 3
    stored = training_runs.get_client().get_run(run.info.run_id)
    assert stored.info.status == "RUNNING"
    assert stored.data.tags["training_state"] == "running"
    assert "retrying" in stored.data.tags["status_message"].lower()


def test_trained_registry_keys_are_task_unique() -> None:
    first = str(uuid4())
    second = str(uuid4())
    assert training_tasks._trained_model_registry_key("Mask2Former", 11, first) != (
        training_tasks._trained_model_registry_key("Mask2Former", 11, second)
    )
