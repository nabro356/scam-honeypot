"""
Agent engine that handles LLM-based conversation with scammers.
Uses NVIDIA AI Endpoints via LangChain for fast inference.
"""

import logging
from typing import Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import get_settings
from core.session_manager import Session
from personas.templates import Persona, get_random_persona, get_persona_for_language


logger = logging.getLogger(__name__)


class AgentEngine:
    """
    LLM-powered agent that engages scammers.
    Uses NVIDIA's Llama models for fast, reliable inference.
    """
    
    def __init__(self):
        settings = get_settings()
        
        # Initialize NVIDIA LLM client
        self.llm = ChatNVIDIA(
            model=settings.llm_model,
            api_key=settings.nvidia_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=0.7,
        )
        
        # Cache for personas
        self._persona_cache: dict[str, Persona] = {}
    
    def get_or_assign_persona(self, session: Session, language: str = "English") -> Persona:
        """
        Get existing persona or assign a new one to the session.
        
        Args:
            session: Current session
            language: Preferred language for persona selection
            
        Returns:
            Persona for this session
        """
        if session.persona_name:
            # Return cached persona
            if session.persona_name in self._persona_cache:
                return self._persona_cache[session.persona_name]
        
        # Assign new persona
        persona = get_persona_for_language(language)
        session.persona_name = persona.name
        self._persona_cache[persona.name] = persona
        
        return persona
    
    async def generate_response(
        self,
        session: Session,
        scammer_message: str,
        language: str = "English"
    ) -> str:
        """
        Generate a response to the scammer's message.
        
        Args:
            session: Current conversation session
            scammer_message: The scammer's latest message
            language: Conversation language
            
        Returns:
            Agent's response as the victim persona
        """
        try:
            # Get persona
            persona = self.get_or_assign_persona(session, language)
            
            # Build messages
            messages = self._build_messages(session, persona, scammer_message)
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            # Extract content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response
            response_text = self._clean_response(response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return a fallback response that keeps the conversation going
            return self._get_fallback_response(session)
    
    def _build_messages(
        self,
        session: Session,
        persona: Persona,
        current_message: str
    ) -> list:
        """Build the message list for the LLM."""
        messages = []
        
        # System prompt with persona
        messages.append(SystemMessage(content=persona.get_system_prompt()))
        
        # Add conversation history (last 10 messages for context)
        history = session.messages[-10:] if session.messages else []
        for msg in history:
            if msg["sender"] == "scammer":
                messages.append(HumanMessage(content=msg["text"]))
            else:
                messages.append(AIMessage(content=msg["text"]))
        
        # Add current scammer message
        messages.append(HumanMessage(content=current_message))
        
        return messages
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate the LLM response."""
        # Remove any meta-commentary
        response = response.strip()
        
        # Remove quotes if the whole response is quoted
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Ensure response isn't too long
        if len(response) > 500:
            # Truncate at sentence boundary
            sentences = response.split('. ')
            truncated = []
            length = 0
            for sentence in sentences:
                if length + len(sentence) < 400:
                    truncated.append(sentence)
                    length += len(sentence)
                else:
                    break
            response = '. '.join(truncated)
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _get_fallback_response(self, session: Session) -> str:
        """Get a fallback response when LLM fails."""
        fallback_responses = [
            "Sorry, network problem. Can you please repeat?",
            "One moment please, I didn't understand properly.",
            "My phone is giving some issue. What were you saying?",
            "Hello? Are you there? Please tell me again.",
            "Sorry sir, connection is bad. Please explain once more.",
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def generate_agent_notes(self, session: Session, detection_result: dict) -> str:
        """
        Generate notes about the scammer's behavior.
        
        Args:
            session: The session
            detection_result: Scam detection result
            
        Returns:
            Summary notes for the session
        """
        notes = []
        
        # Add detection-based notes
        if detection_result.get("matched_patterns"):
            patterns = detection_result["matched_patterns"][:5]
            # Convert tuples to strings (regex groups return tuples)
            pattern_strs = [str(p) if isinstance(p, str) else ' '.join(filter(None, p)) for p in patterns]
            notes.append(f"Detected patterns: {', '.join(pattern_strs)}")
        
        # Check for urgency tactics
        urgent_keywords = {"urgent", "immediately", "now", "today", "asap"}
        if any(kw in str(detection_result.get("suspicious_keywords", [])).lower() 
               for kw in urgent_keywords):
            notes.append("Used urgency tactics")
        
        # Check for authority impersonation
        authority_keywords = {"rbi", "bank", "government", "police", "income tax"}
        if any(kw in str(session.messages).lower() for kw in authority_keywords):
            notes.append("Impersonated authority figure")
        
        # Check for payment requests
        payment_keywords = {"pay", "transfer", "send", "upi", "account"}
        if any(kw in str(session.messages).lower() for kw in payment_keywords):
            notes.append("Requested payment/transfer")
        
        # Add intelligence summary
        intel = session.extracted_intelligence
        if intel.upi_ids:
            notes.append(f"Extracted {len(intel.upi_ids)} UPI ID(s)")
        if intel.bank_accounts:
            notes.append(f"Extracted {len(intel.bank_accounts)} bank account(s)")
        if intel.phishing_links:
            notes.append(f"Extracted {len(intel.phishing_links)} suspicious link(s)")
        
        return "; ".join(notes) if notes else "Scam engagement in progress"


# Singleton instance
_agent_engine: AgentEngine | None = None


def get_agent_engine() -> AgentEngine:
    """Get or create the agent engine instance."""
    global _agent_engine
    if _agent_engine is None:
        _agent_engine = AgentEngine()
    return _agent_engine
