"""
API routes for the Scam Honeypot system.
Handles all HTTP endpoints including the main interact endpoint.
"""

import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status

from api.auth import verify_api_key
from api.schemas import (
    InteractRequest,
    InteractResponse,
    ErrorResponse,
    HealthResponse,
    SessionInfoResponse,
    EngagementMetrics,
    ExtractedIntelligenceOutput
)
from core.scam_detector import get_scam_detector
from core.intelligence import get_intelligence_extractor
from core.session_manager import get_session_manager, Session
from core.agent_engine import get_agent_engine
from utils.callback import send_callback_async


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )


@router.post(
    "/api/honeypot/interact",
    response_model=InteractResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Honeypot"]
)
async def interact(
    request: InteractRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> InteractResponse:
    """
    Main endpoint for scammer interaction.
    
    Receives messages from suspected scammers, analyzes them,
    and generates believable victim responses while extracting intelligence.
    """
    try:
        # Get managers and engines
        session_manager = get_session_manager()
        scam_detector = get_scam_detector()
        intel_extractor = get_intelligence_extractor()
        agent_engine = get_agent_engine()
        
        # Get or create session
        session = await session_manager.get_or_create(request.sessionId)
        
        # Check if session is already terminated
        if session.is_terminated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session has been terminated"
            )
        
        # Get the scammer's message
        scammer_msg = request.message.text
        msg_timestamp = request.message.timestamp
        
        # Add scammer message to session
        session.add_message("scammer", scammer_msg, msg_timestamp)
        
        # Detect scam
        detection = scam_detector.detect(
            scammer_msg,
            history=[{"text": m["text"]} for m in session.messages]
        )
        
        # Update scam status
        if detection.is_scam and not session.scam_detected:
            session.scam_detected = True
            session.scam_confidence = detection.confidence
            session.add_note(f"Scam detected with confidence {detection.confidence:.2f}")
        
        # Extract intelligence from scammer message
        new_intel = intel_extractor.extract(scammer_msg)
        session.extracted_intelligence = session.extracted_intelligence.merge(new_intel)
        
        # Also extract from any URLs or details in history
        for msg in request.conversationHistory:
            if msg.sender == "scammer":
                hist_intel = intel_extractor.extract(msg.text)
                session.extracted_intelligence = session.extracted_intelligence.merge(hist_intel)
        
        # Generate agent response
        agent_response = await agent_engine.generate_response(
            session=session,
            scammer_message=scammer_msg,
            language=request.metadata.language
        )
        
        # Add agent response to session
        session.add_message("user", agent_response)
        
        # Generate agent notes
        agent_notes = agent_engine.generate_agent_notes(
            session,
            {
                "matched_patterns": detection.matched_patterns,
                "suspicious_keywords": detection.suspicious_keywords
            }
        )
        session.add_note(agent_notes)
        
        # Check if session should be terminated
        should_terminate = await session_manager.should_terminate(session)
        
        if should_terminate:
            session.is_terminated = True
            # Send callback in background
            background_tasks.add_task(send_callback_async, session)
        
        # Update session
        await session_manager.update(session)
        
        # Build response
        return InteractResponse(
            status="success",
            scamDetected=session.scam_detected,
            agentResponse=agent_response,
            engagementMetrics=EngagementMetrics(
                engagementDurationSeconds=session.get_engagement_duration_seconds(),
                totalMessagesExchanged=session.get_message_count()
            ),
            extractedIntelligence=ExtractedIntelligenceOutput(
                bankAccounts=session.extracted_intelligence.bank_accounts,
                upiIds=session.extracted_intelligence.upi_ids,
                phishingLinks=session.extracted_intelligence.phishing_links,
                phoneNumbers=session.extracted_intelligence.phone_numbers,
                suspiciousKeywords=session.extracted_intelligence.suspicious_keywords
            ),
            agentNotes=session.get_notes_summary()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing interaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/api/session/{session_id}",
    response_model=SessionInfoResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    tags=["Session"]
)
async def get_session_info(
    session_id: str,
    api_key: str = Depends(verify_api_key)
) -> SessionInfoResponse:
    """Get information about a session."""
    session_manager = get_session_manager()
    session = await session_manager.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return SessionInfoResponse(
        sessionId=session.session_id,
        messageCount=session.get_message_count(),
        scamDetected=session.scam_detected,
        extractedIntelligence=ExtractedIntelligenceOutput(
            bankAccounts=session.extracted_intelligence.bank_accounts,
            upiIds=session.extracted_intelligence.upi_ids,
            phishingLinks=session.extracted_intelligence.phishing_links,
            phoneNumbers=session.extracted_intelligence.phone_numbers,
            suspiciousKeywords=session.extracted_intelligence.suspicious_keywords
        ),
        isTerminated=session.is_terminated,
        createdAt=session.created_at
    )


@router.post(
    "/api/session/{session_id}/terminate",
    response_model=SessionInfoResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    tags=["Session"]
)
async def terminate_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> SessionInfoResponse:
    """Manually terminate a session and trigger GUVI callback."""
    session_manager = get_session_manager()
    session = await session_manager.terminate(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Send callback in background
    background_tasks.add_task(send_callback_async, session)
    
    return SessionInfoResponse(
        sessionId=session.session_id,
        messageCount=session.get_message_count(),
        scamDetected=session.scam_detected,
        extractedIntelligence=ExtractedIntelligenceOutput(
            bankAccounts=session.extracted_intelligence.bank_accounts,
            upiIds=session.extracted_intelligence.upi_ids,
            phishingLinks=session.extracted_intelligence.phishing_links,
            phoneNumbers=session.extracted_intelligence.phone_numbers,
            suspiciousKeywords=session.extracted_intelligence.suspicious_keywords
        ),
        isTerminated=session.is_terminated,
        createdAt=session.created_at
    )
