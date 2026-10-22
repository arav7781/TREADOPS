"""Celery application. Delivery is at-least-once (acks_late); SQL claims make every task idempotent."""

from __future__ import annotations

import os

from celery import Celery

from treadops_api.settings import get_settings

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

app = Celery("treadops", broker=get_settings().redis_url, include=["treadops_worker.tasks"])
app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_default_queue="ops",
    broker_transport_options={"visibility_timeout": 4 * 3600},
    broker_connection_retry_on_startup=True,
    worker_hijack_root_logger=False,
    task_serializer="json",
    accept_content=["json"],
)
