"""API contract and functional verification tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /health returns 200 with service status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_events_indexed" in data
    assert data["total_events_indexed"] > 0


def test_list_companies_endpoint():
    """Verify GET /companies returns tracked CIKs."""
    response = client.get("/companies")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    ciks = [c["cik"] for c in data]
    assert "0000320193" in ciks  # Apple


def test_get_company_events():
    """Verify GET /companies/{cik}/events returns list of valid events."""
    response = client.get("/companies/0000320193/events?limit=5")
    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)
    assert len(events) <= 5
    if events:
        ev = events[0]
        assert ev["cik"] == "0000320193"
        assert "event_type" in ev
        assert "evidence" in ev


def test_get_event_by_id_found_and_not_found():
    """Verify GET /events/{id} returns 200 for valid ID and 404 for invalid ID."""
    from app.service import service
    assert len(service.events) > 0, "Service must have indexed events"
    target_id = service.events[0]["id"]

    # 1. Valid event
    res_valid = client.get(f"/events/{target_id}")
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["id"] == target_id

    # 2. Invalid event
    res_invalid = client.get("/events/non_existent_id_9999")
    assert res_invalid.status_code == 404


def test_get_company_changes():
    """Verify GET /companies/{cik}/changes returns list of guidance diffs."""
    response = client.get("/companies/0000320193/changes?from=2020-01-01&to=2025-12-31")
    assert response.status_code == 200
    changes = response.json()
    assert isinstance(changes, list)


def test_invalid_date_range_rejected():
    """Verify GET /companies/{cik}/changes returns 400 when from_date > to_date."""
    response = client.get("/companies/0000320193/changes?from=2025-01-01&to=2020-01-01")
    assert response.status_code == 400
    assert "Invalid date range" in response.json()["detail"]


def test_predict_event_valid_and_invalid():
    """Verify POST /predict/event handles inference and schema validation."""
    # Valid request
    payload = {
        "text": "For the upcoming quarter, Apple expects gross margin to be between 45.0 percent and 46.0 percent.",
    }
    response = client.post("/predict/event", json=payload)
    assert response.status_code == 200
    pred = response.json()
    assert pred["event_type"] in {"GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"}
    assert "confidence" in pred
    assert "arguments" in pred

    # Invalid request (text too short, violates min_length=5)
    invalid_payload = {"text": "hi"}
    res_invalid = client.post("/predict/event", json=invalid_payload)
    assert res_invalid.status_code == 422
