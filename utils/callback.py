"""
GUVI callback handler for sending final results.
Implements retry logic with exponential backoff.
"""

import asyncio
import logging
import httpx
from typing import Optional

from config import get_settings
from core.session_manager import Session


logger = logging.getLogger(__name__)


class GuviCallback:
    """
    Handler for sending final results to GUVI evaluation endpoint.
    Implements retry logic for reliability.
    """
    
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    TIMEOUT = 10.0  # seconds
    
    def __init__(self):
        settings = get_settings()
        self.callback_url = settings.guvi_callback_url
    
    async def send_final_result(self, session: Session) -> bool:
        """
        Send extracted intelligence to GUVI endpoint.
        
        Args:
            session: The completed session
            
        Returns:
            True if callback was successful
        """
        payload = self._build_payload(session)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                success = await self._send_request(payload)
                if success:
                    logger.info(f"GUVI callback successful for session {session.session_id}")
                    return True
            except Exception as e:
                logger.warning(f"GUVI callback attempt {attempt + 1} failed: {e}")
            
            # Exponential backoff
            if attempt < self.MAX_RETRIES - 1:
                delay = self.BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
        
        logger.error(f"GUVI callback failed after {self.MAX_RETRIES} attempts for session {session.session_id}")
        return False
    
    def _build_payload(self, session: Session) -> dict:
        """Build the callback payload."""
        intel = session.extracted_intelligence
        
        return {
            "sessionId": session.session_id,
            "scamDetected": session.scam_detected,
            "totalMessagesExchanged": session.get_message_count(),
            "extractedIntelligence": {
                "bankAccounts": intel.bank_accounts,
                "upiIds": intel.upi_ids,
                "phishingLinks": intel.phishing_links,
                "phoneNumbers": intel.phone_numbers,
                "suspiciousKeywords": intel.suspicious_keywords
            },
            "agentNotes": session.get_notes_summary()
        }
    
    async def _send_request(self, payload: dict) -> bool:
        """Send the HTTP request to GUVI."""
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            response = await client.post(
                self.callback_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"GUVI callback returned status {response.status_code}: {response.text}")
                return False


# Singleton instance
_callback: GuviCallback | None = None


def get_guvi_callback() -> GuviCallback:
    """Get or create the GUVI callback handler."""
    global _callback
    if _callback is None:
        _callback = GuviCallback()
    return _callback


async def send_callback_async(session: Session) -> None:
    """
    Send callback in background without blocking response.
    
    Args:
        session: The completed session
    """
    callback = get_guvi_callback()
    try:
        await callback.send_final_result(session)
    except Exception as e:
        logger.error(f"Background callback failed: {e}")
