"""
Tests for API endpoints.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

# Set test environment before importing app
import os
os.environ["API_KEY"] = "test_api_key"
os.environ["NVIDIA_API_KEY"] = "test_nvidia_key"

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def api_headers():
    """Standard API headers with auth."""
    return {
        "Content-Type": "application/json",
        "x-api-key": "test_api_key"
    }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns correctly."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Scam Honeypot API"
        assert data["status"] == "running"


class TestAuthMiddleware:
    """Test API key authentication."""
    
    def test_missing_api_key(self, client):
        """Test request without API key is rejected."""
        response = client.post(
            "/api/honeypot/interact",
            json={
                "sessionId": "test-123",
                "message": {
                    "sender": "scammer",
                    "text": "Hello",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        assert response.status_code == 422  # Missing required header
    
    def test_invalid_api_key(self, client):
        """Test invalid API key is rejected."""
        response = client.post(
            "/api/honeypot/interact",
            headers={"x-api-key": "wrong_key"},
            json={
                "sessionId": "test-123",
                "message": {
                    "sender": "scammer",
                    "text": "Hello",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        assert response.status_code == 401


class TestInteractEndpoint:
    """Test main interact endpoint."""
    
    def test_valid_request_structure(self, client, api_headers):
        """Test that valid request structure is accepted."""
        response = client.post(
            "/api/honeypot/interact",
            headers=api_headers,
            json={
                "sessionId": "test-session-1",
                "message": {
                    "sender": "scammer",
                    "text": "Your account will be blocked. Share OTP immediately.",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "conversationHistory": [],
                "metadata": {
                    "channel": "SMS",
                    "language": "English",
                    "locale": "IN"
                }
            }
        )
        
        # Should not return 4xx error for structure
        assert response.status_code in [200, 500, 503]  # 500/503 if LLM fails
    
    def test_scam_detection(self, client, api_headers):
        """Test that scam is detected in obvious scam message."""
        response = client.post(
            "/api/honeypot/interact",
            headers=api_headers,
            json={
                "sessionId": "test-scam-detect",
                "message": {
                    "sender": "scammer",
                    "text": "URGENT: Your bank account blocked. Share OTP NOW!",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "conversationHistory": []
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["scamDetected"] is True
    
    def test_response_format(self, client, api_headers):
        """Test response has correct format."""
        response = client.post(
            "/api/honeypot/interact",
            headers=api_headers,
            json={
                "sessionId": "test-format",
                "message": {
                    "sender": "scammer",
                    "text": "Hello, this is bank calling",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "conversationHistory": []
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "scamDetected" in data
            assert "agentResponse" in data
            assert "engagementMetrics" in data
            assert "extractedIntelligence" in data
            assert "agentNotes" in data
    
    def test_invalid_sender(self, client, api_headers):
        """Test that invalid sender is rejected."""
        response = client.post(
            "/api/honeypot/interact",
            headers=api_headers,
            json={
                "sessionId": "test-invalid",
                "message": {
                    "sender": "invalid_sender",
                    "text": "Hello",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_empty_message(self, client, api_headers):
        """Test that empty message is rejected."""
        response = client.post(
            "/api/honeypot/interact",
            headers=api_headers,
            json={
                "sessionId": "test-empty",
                "message": {
                    "sender": "scammer",
                    "text": "",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        assert response.status_code == 422


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_get_nonexistent_session(self, client, api_headers):
        """Test getting a session that doesn't exist."""
        response = client.get(
            "/api/session/nonexistent-session-id",
            headers=api_headers
        )
        assert response.status_code == 404
    
    def test_terminate_nonexistent_session(self, client, api_headers):
        """Test terminating a session that doesn't exist."""
        response = client.post(
            "/api/session/nonexistent-session-id/terminate",
            headers=api_headers
        )
        assert response.status_code == 404
