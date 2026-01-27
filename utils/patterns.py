"""
Regex patterns for detecting scams and extracting intelligence.
Optimized for Indian financial fraud patterns.
"""

import re
from typing import Pattern

# =============================================================================
# SCAM DETECTION PATTERNS
# =============================================================================

# Urgency and threat patterns
URGENCY_PATTERNS: list[tuple[Pattern, float]] = [
    (re.compile(r"\b(urgent|immediately|now|today|asap)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\b(block|suspend|freeze|deactivate|terminate)\b", re.IGNORECASE), 0.4),
    (re.compile(r"\b(expire|expir(ed|ing)|deadline)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\b(last\s+chance|final\s+(warning|notice))\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b(action\s+required|act\s+now)\b", re.IGNORECASE), 0.4),
]

# Financial fraud patterns
FINANCIAL_PATTERNS: list[tuple[Pattern, float]] = [
    (re.compile(r"\b(kyc|know\s+your\s+customer)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b(verify|verification|update)\s+(your\s+)?(account|details|kyc)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b(share|send|provide)\s+(your\s+)?(otp|pin|password|upi)\b", re.IGNORECASE), 0.7),
    (re.compile(r"\b(bank|account)\s+(block|suspend|freeze)\b", re.IGNORECASE), 0.6),
    (re.compile(r"\b(transfer|pay|send)\s+(\d+|money|amount|rs|inr)\b", re.IGNORECASE), 0.4),
    (re.compile(r"\b(processing|registration|verification)\s+fee\b", re.IGNORECASE), 0.6),
]

# Prize and lottery patterns
PRIZE_PATTERNS: list[tuple[Pattern, float]] = [
    (re.compile(r"\b(won|winner|prize|lottery|jackpot)\b", re.IGNORECASE), 0.6),
    (re.compile(r"\b(congratulations|congrats)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\b(claim|collect)\s+(your\s+)?(prize|reward|gift)\b", re.IGNORECASE), 0.6),
    (re.compile(r"\b(lucky|selected|chosen)\s+(winner|customer)\b", re.IGNORECASE), 0.5),
]

# Impersonation patterns
IMPERSONATION_PATTERNS: list[tuple[Pattern, float]] = [
    (re.compile(r"\b(rbi|reserve\s+bank|sbi|hdfc|icici|axis)\b", re.IGNORECASE), 0.4),
    (re.compile(r"\b(government|income\s+tax|it\s+dept)\b", re.IGNORECASE), 0.4),
    (re.compile(r"\b(customer\s+(care|support|service)|helpline)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\b(official|authorized|verified)\b", re.IGNORECASE), 0.2),
]

# Request patterns
REQUEST_PATTERNS: list[tuple[Pattern, float]] = [
    (re.compile(r"\b(click|tap|open)\s+(this|the|below)?\s*(link|url)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b(call|contact|reach)\s+(us|me|this\s+number)\b", re.IGNORECASE), 0.3),
    (re.compile(r"\b(download|install)\s+(this|the)?\s*(app|application)\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b(fill|complete)\s+(this|the)?\s*(form|details)\b", re.IGNORECASE), 0.3),
]

# All scam patterns combined
ALL_SCAM_PATTERNS: list[tuple[Pattern, float]] = (
    URGENCY_PATTERNS + 
    FINANCIAL_PATTERNS + 
    PRIZE_PATTERNS + 
    IMPERSONATION_PATTERNS + 
    REQUEST_PATTERNS
)

# =============================================================================
# INTELLIGENCE EXTRACTION PATTERNS
# =============================================================================

# UPI ID pattern: username@bankhandle
UPI_PATTERN = re.compile(
    r"\b([a-zA-Z0-9._-]+@[a-zA-Z]{2,})\b"
)

# Indian bank account number: 9-18 digits
BANK_ACCOUNT_PATTERN = re.compile(
    r"\b(\d{9,18})\b"
)

# IFSC code: 4 letters + 0 + 6 alphanumeric
IFSC_PATTERN = re.compile(
    r"\b([A-Z]{4}0[A-Z0-9]{6})\b"
)

# Indian mobile number: +91 or 0 prefix, starts with 6-9
PHONE_PATTERN = re.compile(
    r"(?:\+91[-\s]?|0)?([6-9]\d{9})\b"
)

# URL pattern
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
)

# Suspicious keywords for logging
SUSPICIOUS_KEYWORDS = [
    "urgent", "immediately", "block", "blocked", "suspend", "verify", "kyc",
    "otp", "pin", "password", "upi", "transfer", "pay", "won",
    "prize", "lottery", "claim", "click", "link", "download"
]

SUSPICIOUS_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(SUSPICIOUS_KEYWORDS) + r")\b",
    re.IGNORECASE
)
