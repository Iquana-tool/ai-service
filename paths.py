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
# instead of sending an illegal "Authorization: Bearer " header. Both names are
# exposed because the ported models import them under different aliases.
HF_ACCESS_TOKEN = getenv("HF_ACCESS_TOKEN") or None
HUGGINGFACE_TOKEN = HF_ACCESS_TOKEN

# Weights (instance-suggestion / DINO encoders).
WEIGHTS = os.path.join(ROOT, "weights")
DINO_PATH = os.path.join(WEIGHTS, "dino")
DINO_REPO_DIR = getenv("DINO_REPO_DIR")
