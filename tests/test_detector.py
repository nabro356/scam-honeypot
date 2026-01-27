"""
Tests for scam detection module.
"""

import pytest
from core.scam_detector import ScamDetector, get_scam_detector


class TestScamDetector:
    """Test cases for ScamDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ScamDetector()
    
    def test_detect_basic_scam(self):
        """Test detection of obvious scam message."""
        result = self.detector.detect(
            "Your bank account will be blocked immediately. Share OTP now."
        )
        assert result.is_scam is True
        assert result.confidence >= 0.5
        assert len(result.suspicious_keywords) > 0
    
    def test_detect_legitimate_message(self):
        """Test that normal messages are not flagged."""
        result = self.detector.detect(
            "Hey, how are you? Want to meet for coffee tomorrow?"
        )
        assert result.is_scam is False
        assert result.confidence < 0.4
    
    def test_detect_urgency_patterns(self):
        """Test detection of urgency-based scams."""
        result = self.detector.detect(
            "URGENT: Act now or lose your account forever!"
        )
        assert result.is_scam is True
        assert "urgent" in [kw.lower() for kw in result.suspicious_keywords]
    
    def test_detect_kyc_scam(self):
        """Test detection of KYC-related scams."""
        result = self.detector.detect(
            "Dear customer, your KYC is expiring. Update immediately to avoid suspension."
        )
        assert result.is_scam is True
        assert result.confidence >= 0.4
    
    def test_detect_lottery_scam(self):
        """Test detection of lottery/prize scams."""
        result = self.detector.detect(
            "Congratulations! You have won Rs. 50 lakhs in our lottery. Claim now!"
        )
        assert result.is_scam is True
        assert result.confidence >= 0.4
    
    def test_detect_with_history(self):
        """Test detection with conversation history."""
        history = [
            {"text": "Hello, I am calling from your bank."},
            {"text": "There is a problem with your account."}
        ]
        result = self.detector.detect(
            "Share your OTP to verify your identity.",
            history=history
        )
        assert result.is_scam is True
        # History should increase confidence
        assert result.confidence >= 0.5
    
    def test_get_threat_level(self):
        """Test threat level classification."""
        assert self.detector.get_threat_level(0.9) == "HIGH"
        assert self.detector.get_threat_level(0.6) == "MEDIUM"
        assert self.detector.get_threat_level(0.35) == "LOW"
        assert self.detector.get_threat_level(0.1) == "NONE"
    
    def test_singleton_pattern(self):
        """Test that get_scam_detector returns same instance."""
        detector1 = get_scam_detector()
        detector2 = get_scam_detector()
        assert detector1 is detector2


class TestScamPatterns:
    """Test specific scam patterns."""
    
    def setup_method(self):
        self.detector = ScamDetector()
    
    def test_bank_impersonation(self):
        """Test bank impersonation detection."""
        messages = [
            "This is SBI customer care calling.",
            "HDFC Bank: Your account needs verification.",
            "RBI has flagged your account for suspicious activity."
        ]
        for msg in messages:
            result = self.detector.detect(msg)
            assert len(result.matched_patterns) > 0
    
    def test_otp_request(self):
        """Test OTP request detection."""
        result = self.detector.detect("Please share your OTP for verification.")
        assert result.is_scam is True
        assert result.confidence >= 0.6
    
    def test_upi_request(self):
        """Test UPI request detection."""
        result = self.detector.detect("Send Rs 1 to verify. Share UPI ID.")
        assert result.is_scam is True
