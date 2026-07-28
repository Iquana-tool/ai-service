from celery import Celery

from paths import REDIS_URL

# Single Celery app for the unified service. Long-running jobs (currently
# training) are submitted here and picked up by a worker:
#   uv run celery -A celery_app worker --loglevel=info
#
# ``include`` makes the worker import the task modules on startup so their
# @app.task functions are registered. Training is routed to its own queue so a
# dedicated worker can serve it without starving interactive inference.
celery_app = Celery(
    "iquana_ai_service",
    broker=f"{REDIS_URL}/0",
    backend=f"{REDIS_URL}/1",
    include=["app.tasks"],
)

celery_app.conf.task_routes = {
    "app.tasks.*": {"queue": "ai.training"},
}

# Task modules import the app as ``app`` (``from celery_app import app``).
app = celery_app
