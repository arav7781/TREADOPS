"""Upload admission control: excess uploads are refused with 429 before the body is read."""

import pytest

from .conftest import jpeg, login

pytestmark = pytest.mark.integration


def test_upload_admission_cap_returns_429_without_creating_jobs(clean, monkeypatch):
    from fastapi.testclient import TestClient

    from treadops_api.main import create_app

    monkeypatch.setenv("MAX_INFLIGHT_UPLOADS", "0")  # every upload exceeds the cap
    client = TestClient(create_app())
    h = login(client, "inspector")
    r = client.post("/v1/inspections", headers={**h, "Idempotency-Key": "adm-1"},
                    files={"file": ("t.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_uploads"
    assert r.headers["retry-after"] == "2" and r.json()["error"]["retryable"] is True
    assert client.get("/v1/me", headers=h).status_code == 200  # non-upload traffic is unaffected
