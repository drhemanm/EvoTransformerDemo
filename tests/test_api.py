"""Tests for the EvoCompliance API endpoints."""

import os
import pytest

# Set API key before importing app
os.environ["EVO_API_KEY"] = "test-api-key-12345"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-api-key-12345"}


# ===============================
# Health Endpoint
# ===============================

def test_health_no_auth():
    """Health endpoint should work without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# ===============================
# Auth
# ===============================

def test_analyse_no_auth():
    """Analyse should reject requests without API key."""
    response = client.post("/analyse", json={"text": "test", "task": "transaction"})
    assert response.status_code == 401


def test_analyse_wrong_auth():
    """Analyse should reject requests with wrong API key."""
    response = client.post(
        "/analyse",
        json={"text": "test", "task": "transaction"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


# ===============================
# Input Validation
# ===============================

def test_analyse_empty_text():
    """Should reject empty text."""
    response = client.post(
        "/analyse",
        json={"text": "", "task": "transaction"},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_analyse_invalid_task():
    """Should reject invalid task name."""
    response = client.post(
        "/analyse",
        json={"text": "some text", "task": "invalid_task"},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_feedback_invalid_task():
    """Should reject invalid task in feedback."""
    response = client.post(
        "/feedback",
        json={"text": "some text", "task": "invalid", "correct_label": "education"},
        headers=HEADERS,
    )
    assert response.status_code == 422


# ===============================
# Analyse Endpoint
# ===============================

def test_analyse_transaction():
    """Transaction classification should return expected fields."""
    response = client.post(
        "/analyse",
        json={"text": "Monthly salary payment from employer", "task": "transaction"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "transaction"
    assert "prediction_label" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert 0 <= data["confidence"] <= 1


def test_analyse_document():
    """Document classification should return expected fields."""
    response = client.post(
        "/analyse",
        json={"text": "Revenue increased 12 percent year over year", "task": "document"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "document"
    assert "prediction_label" in data
    assert "confidence" in data


def test_analyse_ner():
    """NER should return entities list."""
    response = client.post(
        "/analyse",
        json={"text": "Apple Inc paid John Smith in London", "task": "ner"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "ner"
    assert "entities" in data
    assert isinstance(data["entities"], list)
    for entity in data["entities"]:
        assert "token" in entity
        assert "label" in entity


# ===============================
# Feedback Endpoint
# ===============================

def test_feedback_valid():
    """Valid feedback should be accepted."""
    response = client.post(
        "/feedback",
        json={
            "text": "Monthly salary deposit from employer",
            "task": "transaction",
            "correct_label": "salary_income",
        },
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "stats" in data


def test_feedback_invalid_label():
    """Feedback with invalid label should be rejected."""
    response = client.post(
        "/feedback",
        json={
            "text": "some text",
            "task": "transaction",
            "correct_label": "nonexistent_label",
        },
        headers=HEADERS,
    )
    assert response.status_code == 400


# ===============================
# Learning Stats
# ===============================

def test_learning_stats():
    """Learning stats should return expected fields."""
    response = client.get("/learning-stats", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "feedback_buffered" in data
    assert "total_learned" in data
    assert "learning_rate" in data
    assert "batch_size" in data


# ===============================
# Analytics Endpoint
# ===============================

def test_analytics():
    """Analytics should return usage summaries."""
    response = client.get("/analytics", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "last_hour" in data
    assert "last_24h" in data
    assert "learning" in data


def test_analytics_no_auth():
    """Analytics should require authentication."""
    response = client.get("/analytics")
    assert response.status_code == 401
