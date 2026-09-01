"""Environment-driven configuration for the unified AI service.

This is the union of the three former services' ``paths.py`` files. Values are
read from the environment (optionally via a project ``.env``) so the same code
runs locally and in deployment.
"""
import os
from os import getenv

from dotenv import load_dotenv

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

LOG_DIR = getenv("LOG_DIR", "logs")
EXAMPLES_DIR = os.path.join(ROOT, "examples")

# Service metadata (surfaced in the OpenAPI docs / health).
SERVICE_NAME = getenv("SERVICE_NAME", "IQUANA AI Service")
SERVICE_DESCRIPTION = getenv(
    "SERVICE_DESCRIPTION",
    "Unified AI service for prompted segmentation, instance suggestion and "
    "full instance segmentation.",
)

# Infrastructure. MLFLOW_URL is the canonical name; ML_FLOW_URL is accepted for
# compatibility with the former instance-segmentation-service config.
MLFLOW_URL = getenv("MLFLOW_URL", getenv("ML_FLOW_URL", "http://localhost:5000"))
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
TRAINING_START_TIMEOUT_SECONDS = int(getenv("TRAINING_START_TIMEOUT_SECONDS", "900"))

# HuggingFace. Empty/blank -> None so transformers does an anonymous request
# instead of sending an illegal "Authorization: Bearer " header.
#
# Read through hf_token() at the moment weights are fetched, not captured at
# import: the backend's admin page can push a new token into this process
# (see app/routes/config.py), and a module-level constant bound at import would
# leave every already-imported model holding the token the service started with.
#
# The two constants stay for compatibility with anything still importing them,
# and are the value as it stood at startup.
HF_ACCESS_TOKEN = getenv("HF_ACCESS_TOKEN") or None
HUGGINGFACE_TOKEN = HF_ACCESS_TOKEN


def hf_token() -> str | None:
    """The Hugging Face token to download weights with, or None for anonymous."""
    return (os.environ.get("HF_ACCESS_TOKEN") or "").strip() or None

# Weights (instance-suggestion / DINO encoders).
WEIGHTS = os.path.join(ROOT, "weights")
DINO_PATH = os.path.join(WEIGHTS, "dino")
DINO_REPO_DIR = getenv("DINO_REPO_DIR")
