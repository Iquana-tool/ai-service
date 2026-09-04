# IQUANA ai-service

The unified AI service behind [IQUANA](https://github.com/Iquana-tool/iquana-tool) —
**I**ntelligent **QU**antification, **AN**notation and **A**nalysis, a tool for AI-assisted
segmentation, annotation and quantification of scientific image datasets, built at
[DFKI](https://www.dfki.de/).

One FastAPI process hosts **every** model-backed task. It serves interactive inference over
HTTP and hands long-running work (training) to a Celery worker. Models are registered in
MLflow and discovered at startup. The [backend](https://github.com/Iquana-tool/backend) is
its only client; this service has no authentication of its own.

- **User documentation:** https://iquana-tool.github.io/docs/
- **Installing the whole tool:** do not clone this repo by hand — run the
  [installer](https://github.com/Iquana-tool/iquana-tool).
- **Issues:** all IQUANA bug reports and feature requests go to
  [iquana-tool/issues](https://github.com/Iquana-tool/iquana-tool/issues/new/choose).

---

## Task surfaces

The three former single-task services collided on paths (prompted segmentation and instance
segmentation both served `POST /inference`; instance suggestion and instance segmentation
both served `POST /annotation_session/run`). Each task is therefore mounted under its own URL
prefix, and each prefix carries the *full* shared surface — health plus model-registry
routes — so a client pointed at `http://host:8004/<task>` reaches every path it already
calls, unchanged.

| Prefix / task | What it does | Model |
|---|---|---|
| `/prompted-segmentation` | Turns a point, box, polygon or freedraw prompt into an outline | SAM 2, SAM 3 |
| `/instance-suggestion` | Suggests further instances resembling the annotated ones | SAM 3 |
| `/instance-segmentation` | Full-image instance segmentation, plus fine-tuning | Mask2Former |
| `/embed` | Whole-image (CLS) and region descriptors | DINOv3 |
| `/cross-image-suggestion` | Concept segmentation from exemplars retrieved across a dataset | SAM 3 |

`GET /health` and `POST /config` are also mounted at the root, for the launcher and the
backend's admin page respectively.

---

## The model interface

MLflow's `pyfunc` gives each logged model exactly one `predict` entry point. Rather than
spending it on a single task — which forced a multi-task model like SAM 3 to be reimplemented
once per service — a model here **composes the tasks it supports** as capability mixins and
implements one handler per task. A dispatching `predict` routes each request to the right
handler: one class, several tasks, logged once.

```python
@register_model
class SAM3(PromptedSegmentation, InstanceSuggestion, CapabilityModel):
    model_info = ModelInfo(registry_key="sam3", ...)

    def load_context(self, context): ...
    def segment_prompted(self, request, params) -> list[Contour]: ...
    def suggest_instances(self, request, params): ...
```

The task tags a model advertises (`task`, `tasks`, and one `task_<name>` boolean per task)
are stamped automatically from the mixins at class-definition time — a model author never
maintains them by hand. Per-task model listings filter on `task_<name>`, so a multi-task
model appears under every surface it serves rather than only its primary one.

### Registration

`@register_model` only *collects* a class. At startup `app.lifespan.build_lifespan`
instantiates each one and writes it to MLflow, storing the full `ModelInfo.model_dump()` as
the logged model's artifact metadata — the lossless source of truth the listing routes read
back. Registered-model *tags* carry only the filterable subset (task, status, ...).

Two ways a class reaches the catalog:

- **In-tree** — every submodule of `models/` is imported at startup, so dropping a file in
  the package is enough; no import has to be wired by hand.
- **Out-of-tree** — packages advertised via the `iquana.models` entry-point group are
  imported too. This is the seam for models that live outside this repository.

### Adding a task

Register it, define the capability mixin, mount it in `app.TASK_MOUNTS`, and add its route:

```python
CROSS_IMAGE_SUGGESTION = register_task(
    "cross-image-suggestion", InstanceSuggestionRequest, "suggest_cross_image"
)

class CrossImageSuggestion(TaskCapability):
    TASK = CROSS_IMAGE_SUGGESTION
    def suggest_cross_image(self, request, params):
        raise NotImplementedError
```

---

## Project structure

```
app/
├── __init__.py       # App factory — TASK_MOUNTS and the shared surface per prefix
├── lifespan.py       # Startup: discover models, register them in MLflow, HF login
├── state.py          # The shared MODEL_REGISTRY
├── routes/
│   ├── health.py     # Liveness (mounted at root and under every prefix)
│   ├── config.py     # Credentials pushed in from the backend admin page
│   ├── models.py     # Task-filtered model catalog + session routes
│   ├── prompted.py, suggestion.py, instance_seg.py, embed.py, cross_image.py
│   └── training.py   # Submits a Celery training job, polls it via MLflow
├── tasks.py          # Celery tasks (training)
└── training_runs.py  # Run bookkeeping and MLflow tags
models/
├── base.py           # Capability interface: tasks, mixins, dispatching predict
├── registry.py       # @register_model, in-tree discovery, plugin loading
├── sam2.py, sam3.py, mask2former.py, dinov3_embedder.py
├── backbones/        # DINOv3 backbone
├── dataloaders.py, mask2former_dataset.py
└── concat_ops.py, embedding_ops.py
celery_app.py         # Celery app; training routed to the ai.training queue
paths.py              # Environment-driven configuration
util/validate_model.py
tests/
```

---

## Setup

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)**; `pyproject.toml` and
`uv.lock` are the source of truth. Torch and torchvision come from the CUDA 13.0 index on
Linux and Windows, and from PyPI (CPU/MPS) on macOS.

```bash
uv sync
# create a .env with the variables below (there is no committed example)
uv run fastapi run main.py --port 8004
```

Training needs the Celery worker, on the `ai.training` queue:

```bash
uv run celery -A celery_app worker -Q ai.training,celery --loglevel=info
```

Use `--pool=solo` on Windows (the prefork pool does not work there) and on macOS. One worker
is enough in any case — a second concurrent worker on one GPU only causes VRAM contention.

> **uv ≥ 0.10 is required.** Older versions reject the PyTorch wheels this service needs.

Run the tests with `uv run pytest tests/ -q`.

---

## Configuration

Read from the environment via a project `.env` (see `paths.py`).

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_URL` | `http://localhost:5000` | Model registry and experiment tracking |
| `REDIS_URL` | `redis://localhost:6379` | Celery broker and result backend |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | CORS allow-list — the backend |
| `HF_ACCESS_TOKEN` | — | Hugging Face token for gated weights (SAM, DINOv3) |
| `AI_SERVICE_ADMIN_TOKEN` | — | Shared secret protecting `/config` (see below) |
| `TRAINING_START_TIMEOUT_SECONDS` | `900` | How long to wait for a submitted run to start |
| `LOG_DIR`, `DINO_REPO_DIR` | — | Paths |

`SERVICE_NAME` and `SERVICE_DESCRIPTION` override what appears in the OpenAPI docs.

### Configuration pushed in from the backend

Credentials this service needs but does not own are edited on the backend's admin page
(Datasets → Admin → Settings) and sent here over `PATCH /config`. Today that is the Hugging
Face access token, which decides whether gated model weights can be downloaded.

Pushed values are held **in memory only**: restarting this service drops them and it falls
back to its own `HF_ACCESS_TOKEN`. The settings page shows what this service is currently
holding, so the drift is visible, with a *Send to AI service* button to push it again.

`GET /config` reports whether a token is set and its last four characters. The value itself
is never returned.

The token is read through `paths.hf_token()` at the moment weights are fetched rather than
captured at import, so a freshly pushed token reaches models that were already imported.

### `AI_SERVICE_ADMIN_TOKEN`

Optional shared secret. When set, `/config` requires the same value in an `X-Admin-Token`
header, and the backend must be given the matching `AI_SERVICE_ADMIN_TOKEN`. When unset the
endpoint is open, matching the rest of this service, which has no authentication and is
expected to be reachable only from the backend.

**Set it on any deployment where this service's port is reachable from anywhere else** —
without it, whoever can reach the port can replace the token this service authenticates to
Hugging Face with.

---

## Related repositories

| Repo | Role |
|---|---|
| [iquana-tool](https://github.com/Iquana-tool/iquana-tool) | Installer, launcher and the issue tracker for all of IQUANA |
| [backend](https://github.com/Iquana-tool/backend) | REST + WebSocket API — this service's only client |
| [frontend-react](https://github.com/Iquana-tool/frontend-react) | The web UI |
| [iquana-toolbox](https://github.com/Iquana-tool/iquana-toolbox) | Shared schemas, the abstract model interface and the MLflow registry |
