"""Multi-task FastAPI app factory for the unified IQUANA AI service.

This host mounts *several* task surfaces on one app, each under its own URL
prefix, over shared building blocks (health router, model-registry routers,
lifespan).

Why prefixes are mandatory: the former services collide on paths (prompted-seg
and instance-seg both serve ``POST /inference``; instance-suggestion and
instance-seg both serve ``POST /annotation_session/run``). Mounting each task
under ``/{task}`` disambiguates them, and -- because the shared health/model
routes are mounted under each prefix too -- a gateway client pointed at
``http://host:port/{task}`` reaches every path it already calls unchanged.

Model registration is task-agnostic: ``build_lifespan(models_package="models")``
auto-discovers and registers *every* ``@register_model`` class once, into the
one shared registry. The per-task model routers merely *filter* that catalog by
tag, so a model that declares several tasks appears under several surfaces.
"""
import logging
from dataclasses import dataclass, field
from typing import Sequence

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.lifespan import build_lifespan
from app.routes.config import build_config_router
from app.routes.health import build_health_router
from app.routes.models import build_task_model_routers
from app.state import MODEL_REGISTRY
from app.routes.prompted import router as prompted_router
from app.routes.suggestion import (
    router as suggestion_router,
    session_router as suggestion_session_router,
)
from app.routes.instance_seg import (
    router as instance_seg_router,
    session_router as instance_seg_session_router,
)
from app.routes.embed import router as embed_router
from app.routes.cross_image import (
    router as cross_image_router,
    session_router as cross_image_session_router,
)
from app.routes.training import router as training_router
from paths import ALLOWED_ORIGINS, SERVICE_DESCRIPTION, SERVICE_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskMount:
    """One task surface mounted under a URL prefix.

    ``task`` is the registry tag used to filter models for this surface;
    ``prefix`` is the URL prefix (also what the gateway points its per-task
    client at); ``routers`` are the task-specific inference/training routers.
    """

    task: str
    prefix: str
    routers: Sequence[APIRouter] = field(default_factory=tuple)


# The three task surfaces. Prefix == task on purpose: it is self-documenting and
# lets each gateway client keep its exact relative paths.
TASK_MOUNTS: list[TaskMount] = [
    TaskMount(
        task="prompted-segmentation",
        prefix="/prompted-segmentation",
        routers=[prompted_router],
    ),
    TaskMount(
        task="instance-suggestion",
        prefix="/instance-suggestion",
        routers=[suggestion_router, suggestion_session_router],
    ),
    TaskMount(
        task="instance-segmentation",
        prefix="/instance-segmentation",
        routers=[instance_seg_router, instance_seg_session_router, training_router],
    ),
    TaskMount(
        task="embed",
        prefix="/embed",
        routers=[embed_router],
    ),
    TaskMount(
        task="cross-image-suggestion",
        prefix="/cross-image-suggestion",
        routers=[cross_image_router, cross_image_session_router],
    ),
]


def create_app() -> FastAPI:
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version="0.1.0",
        lifespan=build_lifespan(
            MODEL_REGISTRY,
            models_package="models",  # auto-discovers @register_model classes at startup
            hf_login=True,  # several models pull gated HF weights
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Service-level liveness at the root (used by the start script / monitoring).
    app.include_router(build_health_router())
    # Credentials pushed in from the backend's admin page. Root only: it
    # configures the process, not any one task surface.
    app.include_router(build_config_router())

    # Each task gets the full shared surface (health + model registry routes)
    # plus its own inference routers, all under its prefix.
    for mount in TASK_MOUNTS:
        app.include_router(build_health_router(), prefix=mount.prefix)
        model_router, model_session_router = build_task_model_routers(MODEL_REGISTRY, mount.task)
        app.include_router(model_router, prefix=mount.prefix)
        app.include_router(model_session_router, prefix=mount.prefix)
        for router in mount.routers:
            app.include_router(router, prefix=mount.prefix)
        logger.debug("Mounted task '%s' at '%s'", mount.task, mount.prefix)

    logger.info("Created unified AI service with %d task surfaces.", len(TASK_MOUNTS))
    return app
