"""
Scam detection module.
Uses pattern matching for fast, low-latency scam detection.
"""

import re
from dataclasses import dataclass
from utils.patterns import ALL_SCAM_PATTERNS, SUSPICIOUS_KEYWORD_PATTERN


@dataclass
class ScamDetectionResult:
    """Result of scam detection analysis."""
    is_scam: bool
    confidence: float
    matched_patterns: list[str]
    suspicious_keywords: list[str]


class ScamDetector:
    """
    Fast pattern-based scam detector.
    Uses weighted regex patterns for <5ms detection.
    """
    
    # Confidence threshold for scam classification
    SCAM_THRESHOLD = 0.4
    
    def __init__(self):
        self.patterns = ALL_SCAM_PATTERNS
    
    def detect(self, text: str, history: list[dict] | None = None) -> ScamDetectionResult:
        """
        Detect if a message is a scam.
        
        Args:
            text: The message text to analyze
            history: Optional conversation history for context
            
        Returns:
            ScamDetectionResult with detection details
        """
        # Combine current message with history for context
        full_text = text
        if history:
            history_text = " ".join(msg.get("text", "") for msg in history[-5:])
            full_text = f"{history_text} {text}"
        
        # Calculate scam score
        total_score = 0.0
        matched_patterns = []
        
        for pattern, weight in self.patterns:
            matches = pattern.findall(full_text)
            if matches:
                total_score += weight
                matched_patterns.extend(matches)
        
        # Cap confidence at 1.0
        confidence = min(total_score, 1.0)
        
        # Extract suspicious keywords
        suspicious_keywords = list(set(
            match.lower() for match in SUSPICIOUS_KEYWORD_PATTERN.findall(full_text)
        ))
        
        # Determine if it's a scam
        is_scam = confidence >= self.SCAM_THRESHOLD
        
        return ScamDetectionResult(
            is_scam=is_scam,
            confidence=confidence,
            matched_patterns=list(set(matched_patterns)),
            suspicious_keywords=suspicious_keywords
        )
    
    def get_threat_level(self, confidence: float) -> str:
        """Get human-readable threat level."""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        elif confidence >= 0.3:
            return "LOW"
        else:
            return "NONE"


# Singleton instance
_detector: ScamDetector | None = None


def get_scam_detector() -> ScamDetector:
    """Get or create the scam detector instance."""
    global _detector
    if _detector is None:
        _detector = ScamDetector()
    return _detector
