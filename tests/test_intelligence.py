"""
Tests for intelligence extraction module.
"""

import pytest
from core.intelligence import IntelligenceExtractor, ExtractedIntelligence, get_intelligence_extractor


class TestIntelligenceExtractor:
    """Test cases for IntelligenceExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = IntelligenceExtractor()
    
    def test_extract_upi_id(self):
        """Test UPI ID extraction."""
        text = "Please send money to my UPI: scammer@ybl or try fraud@paytm"
        result = self.extractor.extract(text)
        
        assert "scammer@ybl" in result.upi_ids
        assert "fraud@paytm" in result.upi_ids
    
    def test_extract_bank_account(self):
        """Test bank account number extraction."""
        text = "Transfer to account number 12345678901234"
        result = self.extractor.extract(text)
        
        assert "12345678901234" in result.bank_accounts
    
    def test_filter_short_numbers(self):
        """Test that short numbers are not extracted as bank accounts."""
        text = "Call me at 12345 or send 100 rupees"
        result = self.extractor.extract(text)
        
        # These should not be extracted as bank accounts (too short)
        assert "12345" not in result.bank_accounts
        assert "100" not in result.bank_accounts
    
    def test_extract_phone_number(self):
        """Test Indian phone number extraction."""
        text = "Call me at 9876543210 or +91 8765432109"
        result = self.extractor.extract(text)
        
        assert "+919876543210" in result.phone_numbers
        assert "+918765432109" in result.phone_numbers
    
    def test_extract_url(self):
        """Test URL extraction."""
        text = "Click this link: https://fake-bank.com/verify or http://scam.site.in/pay"
        result = self.extractor.extract(text)
        
        assert any("fake-bank.com" in url for url in result.phishing_links)
        assert any("scam.site.in" in url for url in result.phishing_links)
    
    def test_extract_www_url(self):
        """Test www URL extraction."""
        text = "Visit www.phishing-site.com for verification"
        result = self.extractor.extract(text)
        
        assert any("phishing-site.com" in url for url in result.phishing_links)
    
    def test_extract_ifsc_code(self):
        """Test IFSC code extraction."""
        text = "IFSC: SBIN0001234 Branch: Main"
        result = self.extractor.extract(text)
        
        assert "SBIN0001234" in result.ifsc_codes
    
    def test_extract_suspicious_keywords(self):
        """Test suspicious keyword extraction."""
        text = "URGENT: Verify your account immediately or it will be blocked"
        result = self.extractor.extract(text)
        
        assert "urgent" in result.suspicious_keywords
        assert "verify" in result.suspicious_keywords
        # Check for 'block' - the pattern may match 'block' or 'blocked'
        assert any("block" in kw for kw in result.suspicious_keywords)
    
    def test_filter_email_like_patterns(self):
        """Test that email addresses are not extracted as UPI IDs."""
        text = "Contact me at user@gmail.com or admin@yahoo.com"
        result = self.extractor.extract(text)
        
        # These should be filtered out as they look like emails
        assert "user@gmail" not in result.upi_ids
        assert "admin@yahoo" not in result.upi_ids
    
    def test_combined_extraction(self):
        """Test extraction of multiple entity types."""
        text = """
        Dear customer, your account 12345678901234 is blocked.
        Send Rs 100 to verify@ybl and call +91 9876543210.
        Or visit https://verify-bank.com for instant activation.
        """
        result = self.extractor.extract(text)
        
        assert len(result.bank_accounts) >= 1
        assert len(result.upi_ids) >= 1
        assert len(result.phone_numbers) >= 1
        assert len(result.phishing_links) >= 1


class TestExtractedIntelligence:
    """Test cases for ExtractedIntelligence dataclass."""
    
    def test_merge(self):
        """Test merging two intelligence objects."""
        intel1 = ExtractedIntelligence(
            upi_ids=["user1@ybl"],
            bank_accounts=["123456789012"]
        )
        intel2 = ExtractedIntelligence(
            upi_ids=["user2@paytm"],
            phone_numbers=["+919876543210"]
        )
        
        merged = intel1.merge(intel2)
        
        assert "user1@ybl" in merged.upi_ids
        assert "user2@paytm" in merged.upi_ids
        assert "123456789012" in merged.bank_accounts
        assert "+919876543210" in merged.phone_numbers
    
    def test_merge_deduplicates(self):
        """Test that merge removes duplicates."""
        intel1 = ExtractedIntelligence(upi_ids=["user@ybl"])
        intel2 = ExtractedIntelligence(upi_ids=["user@ybl", "other@paytm"])
        
        merged = intel1.merge(intel2)
        
        # Should only have 2 unique UPI IDs
        assert len(merged.upi_ids) == 2
    
    def test_is_empty(self):
        """Test is_empty method."""
        empty = ExtractedIntelligence()
        assert empty.is_empty() is True
        
        with_data = ExtractedIntelligence(upi_ids=["test@upi"])
        assert with_data.is_empty() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        intel = ExtractedIntelligence(
            bank_accounts=["123"],
            upi_ids=["test@upi"],
            phishing_links=["http://test.com"],
            phone_numbers=["+91123"],
            suspicious_keywords=["urgent"]
        )
        
        d = intel.to_dict()
        
        assert d["bankAccounts"] == ["123"]
        assert d["upiIds"] == ["test@upi"]
        assert d["phishingLinks"] == ["http://test.com"]
        assert d["phoneNumbers"] == ["+91123"]
        assert d["suspiciousKeywords"] == ["urgent"]
    
    def test_singleton_pattern(self):
        """Test singleton pattern for extractor."""
        ext1 = get_intelligence_extractor()
        ext2 = get_intelligence_extractor()
        assert ext1 is ext2
