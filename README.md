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
| `/instance-segmentation` | Full-image instance segmentation, plus fine-tuning | Mask2Former, Choroidalyzer (OCT) |
| `/embed` | Whole-image (CLS) and region descriptors | DINOv3 |
| `/cross-image-suggestion` | Concept segmentation from exemplars retrieved across a dataset | SAM 3 |

`GET /health` and `POST /config` are also mounted at the root, for the launcher and the
backend's admin page respectively.

### Concept segmentation with hierarchy

The backend resolves a label's saved parent instances and may send the reserved
`parameters.hierarchy_regions` list to any of the three concept surfaces
(`instance-segmentation`, `instance-suggestion`, `cross-image-suggestion`). Each entry
contains a saved `parent_id` and an original-image COCO RLE `mask`. The service consumes
this field before provider parameter validation, focuses one crop per parent, clips
predictions to that parent, and restores contours and RLEs to the original image size.
Missing hierarchy parameters preserve existing whole-image/legacy provider behavior;
an explicitly empty or invalid region list returns 422 rather than running image-wide.

Same-image positive and negative exemplars remain available across parents through a
reference context composited beside each target crop. Cross-image exemplars retain their
source images. Text, embeddings, and other contracts without spatial exemplars run directly
on the focused crop; the provider's declared conditioning requirements still apply.
Nested provider contours are flattened and each class's geometry is restored independently.
Providers that already own the parent loop advertise the tag
`hierarchy_input: parent_regions`; the service passes all parent masks to that
instance-suggestion provider once, preserving native reference-sharing behavior.

Returned contours include the saved `parent_id`. Their `source_instance_id` values are
prefixed per parent to disambiguate crop-local IDs; metadata retains exact full-image
instance RLEs and per-parent diagnostics. The wrapper does not load weights, change model
parameter contracts, or add provider-specific registry keys.

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
├── semantic.py       # Any semantic model as instance segmentation: classes + score maps in, objects out
├── registry.py       # @register_model, in-tree discovery, plugin loading
├── sam2.py, sam3.py, mask2former.py, dinov3_embedder.py, choroidalyzer.py
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

### Polyp model checkpoints from Hugging Face

Cellpose-SAM and YOLO polyp weights are hosted in the private
[`iquana/iquana-cellpose-sam-polyp`](https://huggingface.co/iquana/iquana-cellpose-sam-polyp)
and [`iquana/iquana-yolo-polyp`](https://huggingface.co/iquana/iquana-yolo-polyp)
repositories. The service loads local files; do not put HF URLs in the checkpoint
path variables. Run this setup once before starting the service:

```bash
# Use the deployment's compatible AI-service environment, with the
# cellpose-polyp and yolo-polyp extras installed.
# Authenticate with read access to both repositories: HF_ACCESS_TOKEN,
# HF_TOKEN, or an existing `hf auth login`. Keep tokens in deployment secrets.
python -m download_polyp_weights \
  --cache-dir /srv/iquana/model-cache \
  --env-file /srv/iquana/polyp-weights.env
source /srv/iquana/polyp-weights.env
# Start the AI service and its workers in this configured environment.
```

Choose writable deployment paths for the cache and output file. The command pins
exact HF commits, verifies both SHA-256 hashes, and writes only checkpoint paths
and the Cellpose digest to the shell file; it does not store the token. Existing
configuration is preserved if a download or integrity check fails. Repeated runs
reuse the Hub cache; `--local-files-only` verifies cached weights without network
access. Retain the cache on persistent storage.

For containers, mount that cache at the **same absolute path** used during setup,
and pass the generated variables into the AI-service and worker containers.
For systemd or another process manager, configure the equivalent environment
variables on the services; this shell file uses `export` statements and is not a
Docker `--env-file` or systemd `EnvironmentFile`. Do not commit the generated file.
An HF token pushed through the running service's admin API does not configure
this separate setup process: provide deployment credentials before running it.

Existing manually configured checkpoint paths still work. No download occurs
during model discovery or inference. Until the optional dependency and local file
are configured, the provider remains `not_ready`. Restart after setup, then map
model class `polyp` to the dataset label (for example, `Coral Polyp`) and select
the model in Segment settings. Website users need no individual HF credentials.

The pinned artifacts are `cpsam_best` and `yolo_polyp_finetuned.pt`.
See [Cellpose setup](docs/cellpose-polyp.md) and [YOLO setup](docs/yolo-polyp.md)
for device selection and inference behavior.

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
