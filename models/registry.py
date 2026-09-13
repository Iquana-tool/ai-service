"""Model collection and discovery for the IQUANA AI service.

Add a model by dropping a class in this package and decorating it -- no
hand-maintained list of "models to register":

    from models.registry import register_model
    from iquana_toolbox.ai.base_classes import InstanceSegmentationModel

    @register_model
    class Mask2Former(InstanceSegmentationModel):
        ...

At startup ``app.lifespan.build_lifespan`` runs the decorators and writes
everything collected here into MLflow. Two ways a class reaches the catalog:

* **In-tree:** :func:`import_models_package` imports every submodule of
  ``models``, so dropping a file in the package is enough -- no import has to be
  wired by hand, and nothing relies on some unrelated module (a router, say)
  transitively pulling the model modules in.
* **Out-of-tree:** :func:`load_plugin_models` imports model packages advertised
  via the ``iquana.models`` entry-point group. This is the seam for
  user-installed model packages; it stays dormant until someone ships one. Such
  a package installs into *this* service's venv -- the env that logs a model has
  to be the env that serves it.

The catalog is process-global on purpose: the service runs one app per process,
and tests can reset it with :func:`clear_catalog`.

Keep this module import-light (stdlib only). The lifespan imports it before any
model module -- and therefore before torch -- is loaded.
"""
from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from logging import getLogger
from typing import TypeVar

logger = getLogger(__name__)

ENTRY_POINT_GROUP = "iquana.models"

_CATALOG: list[type] = []

_T = TypeVar("_T", bound=type)


def register_model(cls: _T | None = None):
    """Class decorator marking a model class for registration.

    Usable bare (``@register_model``) or called (``@register_model()``). The
    decorated class is only *collected* here; it is instantiated and written to
    MLflow later, at service startup.
    """

    def _add(target: _T) -> _T:
        if target not in _CATALOG:
            _CATALOG.append(target)
            logger.debug("Collected model class %s", getattr(target, "__name__", target))
        return target

    return _add if cls is None else _add(cls)


def collected_models() -> list[type]:
    """Return all model classes collected via :func:`register_model`."""
    return list(_CATALOG)


def clear_catalog() -> None:
    """Drop all collected classes. Intended for tests."""
    _CATALOG.clear()


def import_models_package(package: str) -> None:
    """Import every submodule of ``package`` so ``@register_model`` fires."""
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        # Plain module, not a package -- importing it already ran its decorators.
        return
    for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(name)
        logger.debug("Imported model module %s", name)


def load_plugin_models(group: str = ENTRY_POINT_GROUP) -> None:
    """Import models contributed by installed packages via entry points.

    A third-party package adds models by declaring, in its packaging metadata::

        [project.entry-points."iquana.models"]
        my_models = "my_pkg.models"

    Loading the entry point imports the target, triggering ``@register_model``.
    A failing plugin is logged and skipped rather than taking the service down.
    """
    for ep in entry_points(group=group):
        try:
            ep.load()
            logger.info("Loaded model plugin '%s'", ep.name)
        except Exception:
            logger.exception("Failed to load model plugin '%s'", ep.name)
