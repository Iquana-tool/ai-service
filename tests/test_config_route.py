"""Checks for the credentials the backend pushes into this service.

The endpoint writes into the process environment, so the two things worth
defending are that it only writes what it is allowed to write, and that it
refuses the write when the deployment has configured a shared secret.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.config import build_config_router


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("AI_SERVICE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    # The route re-establishes the HuggingFace session on a token change; that
    # reaches the network, which a unit test has no business doing.
    monkeypatch.setattr("app.lifespan.refresh_hf_login", lambda: None)

    app = FastAPI()
    app.include_router(build_config_router())
    return TestClient(app)


def test_a_pushed_token_lands_in_the_environment(client, monkeypatch):
    import os

    response = client.patch("/config", json={"values": {"HF_ACCESS_TOKEN": "hf_abcdefgh"}})
    assert response.status_code == 200
    assert os.environ["HF_ACCESS_TOKEN"] == "hf_abcdefgh"

    # And is readable as state without being readable as a value.
    body = client.get("/config").json()
    assert body == {"hf_token_set": True, "hf_token_hint": "…efgh"}


def test_an_empty_push_removes_the_variable(client):
    """Rather than setting it to "", which makes transformers send an empty
    Authorization header instead of an anonymous request."""
    import os

    client.patch("/config", json={"values": {"HF_ACCESS_TOKEN": "hf_abcdefgh"}})
    client.patch("/config", json={"values": {"HF_ACCESS_TOKEN": ""}})

    assert "HF_ACCESS_TOKEN" not in os.environ
    assert client.get("/config").json()["hf_token_set"] is False


def test_only_allowlisted_variables_are_settable(client):
    response = client.patch("/config", json={"values": {"MLFLOW_URL": "http://evil"}})
    assert response.status_code == 400
    assert "MLFLOW_URL" in response.json()["detail"]


def test_the_shared_secret_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("AI_SERVICE_ADMIN_TOKEN", "s3cret")

    assert client.get("/config").status_code == 403
    assert client.patch("/config", json={"values": {}}).status_code == 403
    assert client.patch("/config", json={"values": {}},
                        headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.patch("/config", json={"values": {}},
                        headers={"X-Admin-Token": "s3cret"}).status_code == 200
