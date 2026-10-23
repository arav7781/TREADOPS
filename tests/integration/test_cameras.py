"""Camera registry API and the connector's capture path, against real Postgres and object storage.

A local video file stands in for an RTSP stream (CAMERA_ALLOW_FILE_URLS=true), so CI needs no RTSP server; the
same OpenCV/FFmpeg reader handles both.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from treadops_api import db as dbmod
from treadops_api.models import AuditEvent, Camera, Inspection

from .conftest import admin_id, fixture_bundle, import_and_activate, jpeg, login, requires_bundle


@pytest.fixture()
def video(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_ALLOW_FILE_URLS", "true")
    path = tmp_path / "line.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(30):
        frame = cv2.imdecode(np.frombuffer(jpeg(70 + i // 10, size=(320, 240)), np.uint8), cv2.IMREAD_COLOR)
        writer.write(frame)
    writer.release()
    return f"file://{path}"


def test_camera_registry_masks_urls_enforces_roles_and_audits(client):
    ml = login(client, "ml")
    url = "rtsp://admin:hunter2@10.20.30.40:554/stream1"
    r = client.post("/v1/cameras", headers=ml, json={"name": "Line 9 intake", "line_id": "L9", "stream_url": url})
    assert r.status_code == 201, r.text
    cam = r.json()
    assert cam["stream_url"] == "rtsp://admin:•••@10.20.30.40:554/stream1" and cam["code"] == "line-9-intake"
    assert "hunter2" not in client.get("/v1/cameras", headers=login(client, "viewer")).text
    with dbmod.session_scope() as db:
        row = db.get(Camera, uuid.UUID(cam["id"]))
        assert "hunter2" not in row.stream_url_enc
        assert db.execute(select(AuditEvent).where(AuditEvent.action == "camera.create",
                                                   AuditEvent.resource_id == cam["id"])).scalar_one()
    assert client.post("/v1/cameras", headers=login(client, "inspector"),
                       json={"name": "x camera", "stream_url": url}).status_code == 403
    assert client.post("/v1/cameras", headers=ml, json={"name": "Line 9 intake", "stream_url": url}).status_code == 409
    p = client.patch(f"/v1/cameras/{cam['id']}", headers=ml, json={"enabled": False, "interval_s": 10})
    assert p.json()["state"] == "disabled" and p.json()["interval_s"] == 10
    assert client.post(f"/v1/cameras/{cam['id']}/inspect", headers=login(client, "inspector")).json()["error"]["code"] \
        == "camera_not_live"
    assert client.delete(f"/v1/cameras/{cam['id']}", headers=ml).status_code == 204


def test_probe_reads_a_frame_and_reports_unreachable_streams(client, video):
    ml = login(client, "ml")
    ok = client.post("/v1/cameras/probe", headers=ml, json={"stream_url": video}).json()
    assert ok["ok"] and ok["width"] == 320 and ok["height"] == 240 and ok["snapshot_url"].startswith("/v1/media/")
    bad = client.post("/v1/cameras/probe", headers=ml, json={"stream_url": "rtsp://127.0.0.1:9/nothing"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "camera_unreachable"


@requires_bundle
def test_connector_turns_a_captured_frame_into_a_labelled_inspection(client, video):
    from treadops_worker import cameras as connector

    import_and_activate(fixture_bundle(), admin_id())
    ml = login(client, "ml")
    cam = client.post("/v1/cameras", headers=ml, json={"name": "Sim line", "line_id": "S1", "stream_url": video,
                                                        "simulated": True, "interval_s": 1}).json()
    cfg = connector._load(uuid.UUID(cam["id"]))
    cap = connector.svc.open_stream(connector.svc.decrypt_url(cfg["url_enc"]))
    ok, frame = cap.read()
    cap.release()
    assert ok
    _thumb, sharp, _bright = connector.frame_metrics(frame)
    inspection_id = connector.submit(cfg, frame, sharp)
    assert inspection_id
    with dbmod.session_scope() as db:
        insp = db.get(Inspection, uuid.UUID(inspection_id))
        assert (insp.source, insp.metadata_origin, insp.camera_id, insp.line_id) == ("camera", "simulated", "sim-line", "S1")
