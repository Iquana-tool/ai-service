"""Startup/shutdown lifecycle for the IQUANA AI service.

On startup: optionally log into HuggingFace (several models pull gated weights),
auto-discover every ``@register_model`` class in the ``models`` package plus any
``iquana.models`` entry-point plugins, and write them all into MLflow.
"""
import os
from contextlib import asynccontextmanager
from logging import getLogger

from fastapi import FastAPI
from iquana_toolbox.mlflow import MLFlowModelRegistry

from models.registry import collected_models, import_models_package, load_plugin_models

logger = getLogger(__name__)


def build_lifespan(
    registry: MLFlowModelRegistry,
    *,
    models_package: str = "models",
    hf_login: bool = False,
):
    """Build the FastAPI lifespan that registers this service's models on startup.

    Args:
        registry: Shared model registry.
        models_package: Dotted name of the package to auto-discover models in.
        hf_login: If True, log into HuggingFace using ``HF_ACCESS_TOKEN``
            (needed for the models that pull gated weights).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hf_login:
            _hf_login()
        import_models_package(models_package)
        load_plugin_models()
        classes = collected_models()
        logger.info("Registering %d auto-discovered model(s)...", len(classes))
        _register_into_mlflow(registry, classes)
        yield
        if hf_login:
            _hf_logout()
        logger.info("Service shutting down.")

    return lifespan


def _register_into_mlflow(registry: MLFlowModelRegistry, model_classes: list) -> None:
    """Register collected model classes into the MLflow registry.

    Registration is per-model and fault-isolated: if one model fails to
    instantiate or register (e.g. a gated weight download with no token, or a
    dependency that won't import), it is logged and skipped rather than taking
    the whole service down. A model that can't load simply doesn't appear in the
    registry; every other model still serves. This matters here because a single
    bad model would otherwise crash every task surface at once.
    """
    registered = 0
    for cls in model_classes:
        name = getattr(cls, "__name__", repr(cls))
        try:
            # ``cls`` may be a model class or a zero-arg factory function;
            # calling it yields the instance to register.
            registry.register_model(cls())
            registered += 1
        except Exception:
            logger.exception("Failed to register model '%s'; skipping it.", name)
    logger.info("Registered %d of %d model(s).", registered, len(model_classes))


def _hf_login() -> None:
    token = os.getenv("HF_ACCESS_TOKEN")
    if not token:
        logger.warning("HF_ACCESS_TOKEN not set; skipping HuggingFace login.")
        return
    try:
        from huggingface_hub import login, whoami

        login(token=token)
        logger.info("Logged into HuggingFace as: %s", whoami().get("name"))
    except Exception as e:
        logger.warning("HuggingFace login failed: %s", e)


def _hf_logout() -> None:
    try:
        from huggingface_hub import logout

        logout()
    except Exception as e:
        logger.debug("HuggingFace logout failed: %s", e)
