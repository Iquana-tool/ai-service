"""Credentials pushed in from the backend's admin page.

This service holds no database and no account system, so it cannot own the
configuration an operator edits — the backend does, and sends the values this
process needs across to here. The only value in that set today is the Hugging
Face access token, which decides whether gated model weights can be downloaded.

The values land in ``os.environ``, which is where ``paths.hf_token()`` and the
Hugging Face libraries both look. They are held in memory only: restarting this
service drops them, and the backend's settings page shows the resulting drift so
an operator can re-push rather than silently running with an old token. Writing
them to a file instead would put a credential on disk in a second place, which is
exactly the duplication this endpoint exists to remove.

Access
------
Guarded by ``AI_SERVICE_ADMIN_TOKEN``. When that variable is set here, a request
must present the same value in ``X-Admin-Token``; when it is unset the endpoint
is open, matching the rest of this service, which has no authentication and is
expected to be reachable only from the backend. **Set it on any deployment where
this service's port is reachable from anywhere else** — without it, anyone who
can reach the port can replace the token this service authenticates to Hugging
Face with.
"""
from __future__ import annotations

import os
from logging import getLogger

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

logger = getLogger(__name__)

#: The variables the backend is allowed to set. An allowlist rather than a
#: passthrough: this endpoint must not be able to repoint MLFLOW_URL or set
#: PATH, which "write whatever you are sent into the environment" would permit.
WRITABLE = frozenset({"HF_ACCESS_TOKEN"})

_ADMIN_TOKEN_ENV = "AI_SERVICE_ADMIN_TOKEN"


class ConfigUpdate(BaseModel):
    """Environment values to apply, keyed by variable name."""

    values: dict[str, str | None] = Field(default_factory=dict)


def _authorise(supplied: str | None) -> None:
    """Refuse the write when a shared secret is configured and does not match."""
    expected = (os.getenv(_ADMIN_TOKEN_ENV) or "").strip()
    if not expected:
        return
    if (supplied or "").strip() != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid admin token.")


def _mask(value: str | None) -> str | None:
    """A hint at a secret: enough to tell two tokens apart, not enough to use one."""
    if not value:
        return None
    return f"…{value[-4:]}" if len(value) > 4 else "…"


def build_config_router() -> APIRouter:
    """Build the ``/config`` router."""
    router = APIRouter(tags=["config"])

    @router.get("/config")
    async def read_config(x_admin_token: str | None = Header(default=None)):
        """Report which credentials this service currently holds.

        Values are never returned — only whether each is set and its last four
        characters, which is what the backend's settings page needs to show that
        the two sides agree.
        """
        _authorise(x_admin_token)
        token = (os.environ.get("HF_ACCESS_TOKEN") or "").strip()
        return {
            "hf_token_set": bool(token),
            "hf_token_hint": _mask(token),
        }

    @router.patch("/config")
    async def update_config(body: ConfigUpdate,
                            x_admin_token: str | None = Header(default=None)):
        """Apply pushed credentials to this process."""
        _authorise(x_admin_token)

        unknown = sorted(set(body.values) - WRITABLE)
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not settable here: {', '.join(unknown)}.",
            )

        for name, value in body.values.items():
            cleaned = (value or "").strip()
            if cleaned:
                os.environ[name] = cleaned
            else:
                # An empty push means "no token" — anonymous downloads — so the
                # variable is removed rather than set to "", which would make
                # transformers send an empty Authorization header.
                os.environ.pop(name, None)

        if "HF_ACCESS_TOKEN" in body.values:
            # Setting the variable is not enough: most of huggingface_hub reads
            # the session established at login, which still holds whatever this
            # service booted with until it is re-established.
            from app.lifespan import refresh_hf_login
            refresh_hf_login()

        logger.info("Configuration updated from the backend: %s",
                    ", ".join(sorted(body.values)) or "nothing")
        return {"applied": sorted(body.values)}

    return router
