"""Prevent separate entry points from exposing different sets of course routes."""

from fastapi.testclient import TestClient

from app.main import app
from main import app as compatibility_app


def test_both_entry_points_expose_all_assignments():
    assert compatibility_app is app
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        for path, methods in {
            "/": {"get"},
            "/generate": {"post"},
            "/embedding": {"get", "post"},
            "/predict": {"post"},
            "/cnn/health": {"get"},
            "/gan/health": {"get"},
            "/gan/generate": {"post"},
            "/gan/generate.png": {"get"},
        }.items():
            assert methods <= set(paths[path])
        assert client.get("/cnn/health").status_code == 200
        assert client.get("/gan/health").json()["epoch"] == 10
