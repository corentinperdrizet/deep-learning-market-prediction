"""FastAPI route tests -- model_registry is monkeypatched so no real
checkpoint/network access happens here; the business logic itself is
covered by tst/test_model_registry.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_mod
from src.serving.model_registry import ModelNotFoundError, PredictionResult


@pytest.fixture
def client():
    return TestClient(api_mod.app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_models_lists_available(monkeypatch, client):
    monkeypatch.setattr(
        api_mod.model_registry,
        "list_available_models",
        lambda: [{"model_kind": "lstm", "ticker": "BTC-USD"}],
    )
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json() == [{"model_kind": "lstm", "ticker": "BTC-USD"}]


def test_predict_success(monkeypatch, client):
    fake_result = PredictionResult(
        ticker="BTC-USD",
        model_kind="lstm",
        as_of="2024-01-01 00:00:00+00:00",
        probability_up=0.62,
        signal=1,
        threshold=0.5,
    )
    monkeypatch.setattr(api_mod.model_registry, "predict_latest", lambda **kwargs: fake_result)

    resp = client.get("/predict/BTC-USD?model=lstm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "BTC-USD"
    assert body["probability_up"] == 0.62
    assert body["signal"] == 1


def test_predict_defaults_to_lstm(monkeypatch, client):
    captured = {}

    def fake_predict(**kwargs):
        captured.update(kwargs)
        return PredictionResult(
            ticker=kwargs["ticker"],
            model_kind=kwargs["model_kind"],
            as_of="x",
            probability_up=0.5,
            signal=0,
            threshold=0.5,
        )

    monkeypatch.setattr(api_mod.model_registry, "predict_latest", fake_predict)
    client.get("/predict/BTC-USD")
    assert captured["model_kind"] == "lstm"


def test_predict_unknown_model_kind_rejected_by_validation(client):
    resp = client.get("/predict/BTC-USD?model=xgboost")
    assert resp.status_code == 422  # FastAPI/pydantic enum validation, never reaches model_registry


def test_predict_missing_checkpoint_returns_404(monkeypatch, client):
    def raise_not_found(**kwargs):
        raise ModelNotFoundError("no checkpoint for you")

    monkeypatch.setattr(api_mod.model_registry, "predict_latest", raise_not_found)
    resp = client.get("/predict/DOES-NOT-EXIST?model=lstm")
    assert resp.status_code == 404
    assert "no checkpoint" in resp.json()["detail"]


def test_predict_bad_input_returns_400(monkeypatch, client):
    def raise_value_error(**kwargs):
        raise ValueError("not enough history")

    monkeypatch.setattr(api_mod.model_registry, "predict_latest", raise_value_error)
    resp = client.get("/predict/BTC-USD?model=lstm")
    assert resp.status_code == 400
    assert "not enough history" in resp.json()["detail"]
