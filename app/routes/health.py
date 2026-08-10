"""Health endpoint for the IQUANA AI service.

Mounted at the service root (liveness for the start scripts and monitoring) and
again under every task prefix, so a gateway client pointed at
``http://host:port/{task}`` finds the same ``/health`` it already calls.

The device probe lives here rather than being injected by the app factory: this
service owns torch, so there is no dependency-light constraint to work around.
"""
from logging import getLogger

from fastapi import APIRouter

logger = getLogger(__name__)


def _device_info() -> dict:
    """Report the torch runtime models will actually run on."""
    import torch

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "mps (Apple Silicon)"
    else:
        device = "cpu"
    return {"device": device, "torch_version": torch.__version__}


def build_health_router() -> APIRouter:
    """Build the ``/health`` router."""
    router = APIRouter()

    @router.get("/health", tags=["health"])
    async def health_check():
        payload = {"status": "ok"}
        try:
            payload.update(_device_info())
        except Exception as e:  # never let diagnostics break the health check
            logger.warning("Device probe failed: %s", e)
            payload["extra_error"] = str(e)
        return payload

    return router
