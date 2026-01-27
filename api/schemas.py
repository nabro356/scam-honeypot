"""
Pydantic schemas for API request and response validation.
Ensures strict type checking and validation for all API interactions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime


class MessageInput(BaseModel):
    """Individual message in the conversation."""
    sender: Literal["scammer", "user"]
    text: str = Field(..., min_length=1, max_length=5000)
    timestamp: datetime
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Message text cannot be empty")
        return v.strip()


class MetadataInput(BaseModel):
    """Optional metadata for the interaction."""
    channel: str = "SMS"
    language: str = "English"
    locale: str = "IN"


class InteractRequest(BaseModel):
    """Request body for the honeypot interact endpoint."""
    sessionId: str = Field(..., min_length=1, max_length=100)
    message: MessageInput
    conversationHistory: list[MessageInput] = []
    metadata: MetadataInput = Field(default_factory=MetadataInput)
    
    @field_validator("sessionId")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session ID is valid."""
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()


class EngagementMetrics(BaseModel):
    """Metrics about the engagement with the scammer."""
    engagementDurationSeconds: int = Field(ge=0)
    totalMessagesExchanged: int = Field(ge=0)


class ExtractedIntelligenceOutput(BaseModel):
    """Intelligence extracted from the conversation."""
    bankAccounts: list[str] = []
    upiIds: list[str] = []
    phishingLinks: list[str] = []
    phoneNumbers: list[str] = []
    suspiciousKeywords: list[str] = []


class InteractResponse(BaseModel):
    """Response body for the honeypot interact endpoint."""
    status: Literal["success", "error"]
    scamDetected: bool
    agentResponse: str
    engagementMetrics: EngagementMetrics
    extractedIntelligence: ExtractedIntelligenceOutput
    agentNotes: str


class ErrorResponse(BaseModel):
    """Error response format."""
    status: Literal["error"] = "error"
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime


class SessionInfoResponse(BaseModel):
    """Response for session info endpoint."""
    sessionId: str
    messageCount: int
    scamDetected: bool
    extractedIntelligence: ExtractedIntelligenceOutput
    isTerminated: bool
    createdAt: datetime
