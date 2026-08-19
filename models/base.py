"""Capability-based model interface for the unified AI service.

Why this exists
---------------
MLflow's ``pyfunc`` gives each logged model exactly one ``predict`` entry point.
The former toolbox design spent that entry point on a *single* task: each task
had its own ``BaseModel`` subclass whose ``predict`` had a task-specific
signature. A model that can do several tasks (SAM 3 does prompted segmentation
*and* instance suggestion) therefore had to be reimplemented once per service.

Here a model instead **composes the tasks it supports** as capability mixins and
implements one handler method per task. A single dispatching ``predict`` routes
an incoming request to the right handler. One class, several tasks, logged once.

Adding a model
--------------
    @register_model
    class MyModel(PromptedSegmentation, CapabilityModel):
        model_info = PromptedSegmentationModelInfo(registry_key="mymodel", ...)

        def load_context(self, context): ...
        def segment_prompted(self, request, params) -> list[Contour]: ...

Adding a *multi-task* model
---------------------------
    @register_model
    class SAM3(PromptedSegmentation, InstanceSuggestion, CapabilityModel):
        model_info = ModelInfo(registry_key="sam3", ...)

        def segment_prompted(self, request, params) -> list[Contour]: ...
        def suggest_instances(self, request, params): ...

Adding a *new task* (e.g. the ones on the roadmap)
--------------------------------------------------
    CROSS_IMAGE_SUGGESTION = register_task(
        "cross-image-suggestion", InstanceSuggestionRequest, "suggest_cross_image"
    )

    class CrossImageSuggestion(TaskCapability):
        TASK = CROSS_IMAGE_SUGGESTION
        def suggest_cross_image(self, request, params):
            raise NotImplementedError

Then mount the task in ``app.TASK_MOUNTS`` and add its route. Models opt in by
mixing in ``CrossImageSuggestion`` and implementing ``suggest_cross_image``.

The task tags a model advertises (``task``, ``tasks`` and one ``task_<name>``
boolean per task) are stamped automatically from the mixins at class-definition
time -- a model author never maintains them by hand.
"""
from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from iquana_toolbox.ai.base_classes import BaseModel
from iquana_toolbox.schemas.input_contract import InputContract, get_contract_for_task
from iquana_toolbox.schemas.networking.http.services import (
    BaseServiceRequest,
    CrossImageSuggestionRequest,
    EmbedRequest,
    InstanceSegmentationRequest,
    InstanceSuggestionRequest,
    PromptedSegmentationRequest,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """A task the tool understands.

    Attributes:
        name: Stable identifier. Doubles as the MLflow ``task`` tag value and the
            URL prefix the task surface is mounted under (e.g. ``instance-suggestion``).
        request_type: The request schema routed to this task's handler.
        handler: The method name a model implements to serve this task.
    """

    name: str
    request_type: type[BaseServiceRequest]
    handler: str

    @property
    def tag_key(self) -> str:
        """Filter-safe registered-model tag key for this task.

        MLflow tag filters treat ``.`` and ``-`` as syntax, so the boolean
        per-task tag uses an underscored key: ``instance-suggestion`` ->
        ``task_instance_suggestion``.
        """
        return "task_" + self.name.replace("-", "_")


_TASKS: dict[str, Task] = {}


def register_task(name: str, request_type: type[BaseServiceRequest], handler: str) -> Task:
    """Register (or return the existing) task by ``name``. Idempotent."""
    existing = _TASKS.get(name)
    if existing is not None:
        return existing
    task = Task(name=name, request_type=request_type, handler=handler)
    _TASKS[name] = task
    logger.debug("Registered task '%s' (handler=%s)", name, handler)
    return task


def get_task(name: str) -> Task:
    try:
        return _TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task '{name}'. Known tasks: {sorted(_TASKS)}")


def all_tasks() -> dict[str, Task]:
    return dict(_TASKS)


# --- Built-in tasks (the three current surfaces) --------------------------- #
PROMPTED_SEGMENTATION = register_task(
    "prompted-segmentation", PromptedSegmentationRequest, "segment_prompted"
)
INSTANCE_SUGGESTION = register_task(
    "instance-suggestion", InstanceSuggestionRequest, "suggest_instances"
)
INSTANCE_SEGMENTATION = register_task(
    "instance-segmentation", InstanceSegmentationRequest, "segment_instances"
)
EMBED = register_task("embed", EmbedRequest, "embed")
CROSS_IMAGE_SUGGESTION = register_task(
    "cross-image-suggestion", CrossImageSuggestionRequest, "suggest_cross_image"
)


# --------------------------------------------------------------------------- #
# Capability mixins
# --------------------------------------------------------------------------- #
class TaskCapability:
    """Base for capability mixins.

    A capability mixin binds a :class:`Task` (``TASK``) to the handler method a
    model must implement. Mixing it into a model both declares support for the
    task (so it is advertised and filterable) and provides the handler slot.
    """

    TASK: ClassVar[Task]


class PromptedSegmentation(TaskCapability):
    """2D prompted segmentation (points / boxes / previous mask)."""

    TASK = PROMPTED_SEGMENTATION

    def segment_prompted(self, request: PromptedSegmentationRequest, params: dict[str, Any]):
        raise NotImplementedError


class InstanceSuggestion(TaskCapability):
    """Suggest further instances from exemplar masks (few-shot / concept)."""

    TASK = INSTANCE_SUGGESTION

    def suggest_instances(self, request: InstanceSuggestionRequest, params: dict[str, Any]):
        raise NotImplementedError


class InstanceSegmentation(TaskCapability):
    """Full instance segmentation over the whole image."""

    TASK = INSTANCE_SEGMENTATION

    def segment_instances(self, request: InstanceSegmentationRequest, params: dict[str, Any]):
        raise NotImplementedError


class CrossImageSuggestion(TaskCapability):
    """Suggest instances of a concept on a target image, using exemplars from other images.

    The cross-image counterpart to :class:`InstanceSuggestion`: where that suggests from
    exemplar masks on the *same* image, this transfers a concept across images (e.g. SAM 3's
    concat workaround). Returns ``(masks, scores)`` on the target image, like suggestion.
    """

    TASK = CROSS_IMAGE_SUGGESTION

    def suggest_cross_image(self, request: CrossImageSuggestionRequest, params: dict[str, Any]):
        raise NotImplementedError


class Embedding(TaskCapability):
    """Precompute dense feature embeddings for an image and/or its masked regions.

    Backs cross-image exemplar retrieval: the returned vectors are persisted and later
    compared by cosine similarity to pick the exemplar to hand a cross-image segmenter.
    A model returns one :class:`EmbeddingVector` per requested whole-image kind and per
    region; unknown kinds are skipped rather than erroring.
    """

    TASK = EMBED

    def embed(self, request: EmbedRequest, params: dict[str, Any]):
        raise NotImplementedError


def validate_and_normalize_params(
    contract: InputContract | None, params: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate and normalize inference parameters against an :class:`InputContract`.

    - If *contract* is None, returns a shallow copy of *params*.
    - Rejects unknown parameter keys (ignoring the routing key ``'task'``).
    - Fills missing/None values with declared defaults.
    - Validates type (bool, int, float, str).
    - Validates options if declared.
    - Validates min_value and max_value bounds.
    - Coerces numbers to float for float parameters.
    """
    if contract is None:
        return dict(params or {})

    raw_params = dict(params or {})
    declared_by_key = {param.key: param for param in contract.parameters}

    # Reject unknown parameter keys (ignoring the routing key 'task')
    for key in raw_params:
        if key == "task":
            continue
        if key not in declared_by_key:
            raise ValueError(
                f"Unknown parameter '{key}' for task '{contract.task}'. "
                f"Declared parameters: {sorted(declared_by_key.keys())}"
            )

    normalized: dict[str, Any] = {}
    if "task" in raw_params:
        normalized["task"] = raw_params["task"]

    for key, spec in declared_by_key.items():
        if key in raw_params and raw_params[key] is not None:
            val = raw_params[key]
        else:
            val = spec.default_value

        spec_type = spec.type

        if spec_type == "bool":
            if not isinstance(val, bool):
                raise ValueError(
                    f"Parameter '{key}' must be a bool, got {type(val).__name__} ({val!r})"
                )
            normalized_val = bool(val)
        elif spec_type == "int":
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(
                    f"Parameter '{key}' must be an int, got {type(val).__name__} ({val!r})"
                )
            normalized_val = int(val)
        elif spec_type == "float":
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Parameter '{key}' must be a finite float, got {type(val).__name__} ({val!r})"
                )
            normalized_val = float(val)
        elif spec_type == "str":
            if not isinstance(val, str):
                raise ValueError(
                    f"Parameter '{key}' must be a str, got {type(val).__name__} ({val!r})"
                )
            normalized_val = str(val)
        else:
            raise ValueError(f"Unsupported parameter type '{spec_type}' for '{key}'")

        if spec.options is not None:
            if normalized_val not in spec.options:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val!r} is not in allowed options: {spec.options}"
                )

        if spec_type in {"int", "float"}:
            if spec.min_value is not None and normalized_val < spec.min_value:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val} is less than min_value {spec.min_value}"
                )
            if spec.max_value is not None and normalized_val > spec.max_value:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val} is greater than max_value {spec.max_value}"
                )

        normalized[key] = normalized_val

    return normalized


def _conditioning_unit_count(request: BaseServiceRequest, kind: str) -> int:
    """Count model-usable conditioning units for a request."""
    if kind == "instances":
        return len(getattr(request, "positive_exemplars", ()) or ())
    if kind == "reference_images":
        exemplars = getattr(request, "exemplars", ()) or ()
        return len({exemplar.image_url for exemplar in exemplars})
    if kind == "embeddings":
        return len(getattr(request, "embeddings", {}) or {})
    return 0


def validate_request_conditioning(
    contract: InputContract | None, request: BaseServiceRequest
) -> None:
    """Validate request conditioning against the selected task contract.

    Contracts that declare no cardinality (``none``/``concept_text``) are
    intentionally left alone. Missing request fields also count as empty, which
    keeps optional ``min_units=0`` contracts valid across request types.
    """
    if contract is None:
        return

    conditioning = contract.conditioning
    if conditioning.kind not in {"instances", "reference_images", "embeddings"}:
        return

    actual = _conditioning_unit_count(request, conditioning.kind)
    if actual < conditioning.min_units:
        raise ValueError(
            f"Task '{contract.task}' requires at least {conditioning.min_units} "
            f"{conditioning.unit} conditioning unit(s), but request supplied {actual}."
        )
    if conditioning.max_units is not None and actual > conditioning.max_units:
        raise ValueError(
            f"Task '{contract.task}' accepts at most {conditioning.max_units} "
            f"{conditioning.unit} conditioning unit(s), but request supplied {actual}."
        )


# --------------------------------------------------------------------------- #
# Dispatching base model
# --------------------------------------------------------------------------- #
class CapabilityModel(BaseModel):
    """Base for models that serve one or more tasks via capability mixins.

    Provides the single ``predict`` entry point MLflow calls, which dispatches to
    the handler for the request's task. Subclasses implement the handler(s) for
    the capabilities they mix in; they do not override ``predict``.
    """

    def __init_subclass__(cls, **kwargs):
        """Stamp task tags onto a concrete model's ``model_info`` from its mixins.

        The model author declares tasks purely by which capability mixins they
        inherit; the ``task`` / ``tasks`` / ``task_<name>`` tags used for
        registration and registry filtering are derived here so they can never
        drift from the actual capabilities. Two model-authoring styles are both
        supported with zero boilerplate:

        * ``model_info`` as a **class attribute** (e.g. SAM 3, Mask2Former):
          stamped immediately.
        * ``model_info`` **built in** ``__init__`` (e.g. the four SAM 2 variants
          share one class and set it per-instance): this class's ``__init__`` is
          wrapped so the tags are stamped right after construction -- before the
          registry ever reads ``model_info.tags``.
        """
        super().__init_subclass__(**kwargs)

        class_info = cls.__dict__.get("model_info")
        if class_info is not None:
            cls._stamp_task_tags(class_info)

        own_init = cls.__dict__.get("__init__")
        if own_init is not None:
            @functools.wraps(own_init)
            def _init_and_stamp(self, *args, _own_init=own_init, **kwargs):
                _own_init(self, *args, **kwargs)
                info = getattr(self, "model_info", None)
                if info is not None:
                    type(self)._stamp_task_tags(info)

            cls.__init__ = _init_and_stamp

    @classmethod
    def _stamp_task_tags(cls, info) -> None:
        """Write the task tags for this class's capabilities onto ``info.tags``."""
        tasks = cls.supported_tasks()
        task_names = {t.name for t in tasks}

        # Validate input contracts reference only tasks this model serves.
        contracts = getattr(info, "input_contracts", None) or []
        for contract in contracts:
            if contract.task not in task_names:
                raise ValueError(
                    f"{cls.__name__} declares input_contract for task '{contract.task}' "
                    f"but does not serve it (supported: {sorted(task_names)})."
                )

        if not tasks:
            return

        # Primary task keeps the legacy single ``task`` tag: it selects the
        # ModelInfo subclass in parse_tags_to_model_info and satisfies any
        # consumer still filtering on ``task``.
        info.tags.setdefault("task", tasks[0].name)
        info.tags["tasks"] = ",".join(t.name for t in tasks)
        for task in tasks:
            info.tags[task.tag_key] = "true"
        if getattr(info, "trainable", False):
            info.tags["trainable"] = "true"

    @classmethod
    def supported_tasks(cls) -> list[Task]:
        """The tasks this model serves, in mixin (MRO) order, de-duplicated."""
        found: dict[str, Task] = {}
        for klass in cls.__mro__:
            task = klass.__dict__.get("TASK")
            if isinstance(task, Task):
                found.setdefault(task.name, task)
        return list(found.values())

    def get_input_contract(self, task_name: str) -> InputContract | None:
        """Return the declared InputContract for *task_name*, or None."""
        info = getattr(self, "model_info", None)
        if info is None:
            return None
        contracts = getattr(info, "input_contracts", None) or []
        return get_contract_for_task(contracts, task_name)

    # -- MLflow entry point -------------------------------------------------- #
    def predict(self, context: Any, model_input, params: dict[str, Any] | None = None):
        params = dict(params or {})
        request = model_input[0] if isinstance(model_input, list) else model_input
        task = self._resolve_task(request, params)
        contract = self.get_input_contract(task.name)
        validate_request_conditioning(contract, request)
        normalized_params = validate_and_normalize_params(contract, params)
        handler = getattr(self, task.handler)
        return handler(request, normalized_params)

    def _resolve_task(self, request, params: dict[str, Any]) -> Task:
        """Pick the task to serve this request.

        Resolution order:
          1. Explicit ``params['task']`` -- set by the mounted route, which knows
             its task. Unambiguous, and the only way to disambiguate tasks that
             share a request type (e.g. suggestion vs cross-image suggestion).
          2. If the model serves exactly one task, use it.
          3. Otherwise, match by request type; error if 0 or >1 candidates.
        """
        supported = {t.name: t for t in self.supported_tasks()}

        requested = params.get("task")
        if requested:
            task = supported.get(requested)
            if task is None:
                raise ValueError(
                    f"{type(self).__name__} does not support task '{requested}'. "
                    f"Supported: {sorted(supported)}"
                )
            return task

        if len(supported) == 1:
            return next(iter(supported.values()))

        matches = [t for t in supported.values() if isinstance(request, t.request_type)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Cannot resolve task for {type(self).__name__} from request "
            f"{type(request).__name__}; pass params['task'] (candidates: "
            f"{[t.name for t in matches] or sorted(supported)})."
        )

    # ``train`` is optional; trainable models override it. Kept concrete so
    # non-trainable models need not implement anything.
    def train(self, request, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is not trainable.")
