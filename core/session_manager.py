"""
Session management for multi-turn conversations.
Uses in-memory storage with TTL for fast access.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from core.intelligence import ExtractedIntelligence


@dataclass
class Session:
    """Represents a conversation session with a scammer."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages: list[dict] = field(default_factory=list)
    scam_detected: bool = False
    scam_confidence: float = 0.0
    extracted_intelligence: ExtractedIntelligence = field(default_factory=ExtractedIntelligence)
    persona_name: str = ""
    is_terminated: bool = False
    agent_notes: list[str] = field(default_factory=list)
    
    def add_message(self, sender: str, text: str, timestamp: datetime | None = None) -> None:
        """Add a message to the session history."""
        self.messages.append({
            "sender": sender,
            "text": text,
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        })
        self.last_activity = datetime.utcnow()
    
    def get_engagement_duration_seconds(self) -> int:
        """Calculate total engagement duration in seconds."""
        if not self.messages:
            return 0
        return int((self.last_activity - self.created_at).total_seconds())
    
    def get_message_count(self) -> int:
        """Get total number of messages exchanged."""
        return len(self.messages)
    
    def add_note(self, note: str) -> None:
        """Add an agent note."""
        if note not in self.agent_notes:
            self.agent_notes.append(note)
    
    def get_notes_summary(self) -> str:
        """Get all notes as a summary string."""
        if not self.agent_notes:
            return "No significant observations."
        return "; ".join(self.agent_notes)


class SessionManager:
    """
    Thread-safe in-memory session manager.
    Handles session creation, retrieval, and cleanup.
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_messages: int = 50):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_messages = max_messages
    
    async def get_or_create(self, session_id: str) -> Session:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session object
        """
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                # Check if session is still valid
                if not self._is_expired(session):
                    return session
                else:
                    # Session expired, create new one
                    del self._sessions[session_id]
            
            # Create new session
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            return session
    
    async def get(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and not self._is_expired(session):
                return session
            return None
    
    async def update(self, session: Session) -> None:
        """
        Update a session in the store.
        
        Args:
            session: Session to update
        """
        async with self._lock:
            self._sessions[session.session_id] = session
    
    async def terminate(self, session_id: str) -> Optional[Session]:
        """
        Terminate a session.
        
        Args:
            session_id: Session to terminate
            
        Returns:
            The terminated session
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_terminated = True
                return session
            return None
    
    async def should_terminate(self, session: Session) -> bool:
        """
        Check if session should be automatically terminated.
        
        Args:
            session: Session to check
            
        Returns:
            True if session should terminate
        """
        # Terminate if max messages reached
        if session.get_message_count() >= self._max_messages:
            session.add_note("Auto-terminated: max messages reached")
            return True
        
        # Terminate if session is too old
        if self._is_expired(session):
            session.add_note("Auto-terminated: session expired")
            return True
        
        return False
    
    def _is_expired(self, session: Session) -> bool:
        """Check if a session has expired."""
        return datetime.utcnow() - session.last_activity > self._ttl
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        async with self._lock:
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if self._is_expired(session)
            ]
            for sid in expired_ids:
                del self._sessions[sid]
            return len(expired_ids)
    
    async def get_active_count(self) -> int:
        """Get count of active sessions."""
        async with self._lock:
            return len([
                s for s in self._sessions.values()
                if not self._is_expired(s) and not s.is_terminated
            ])


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the session manager instance."""
    global _session_manager
    if _session_manager is None:
        from config import get_settings
        settings = get_settings()
        _session_manager = SessionManager(
            ttl_seconds=settings.session_ttl_seconds,
            max_messages=settings.max_messages_per_session
        )
    return _session_manager
