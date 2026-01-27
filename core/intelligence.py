"""
Intelligence extraction module.
Extracts bank accounts, UPI IDs, phone numbers, and URLs from text.
"""

from dataclasses import dataclass, field
from utils.patterns import (
    UPI_PATTERN,
    BANK_ACCOUNT_PATTERN,
    IFSC_PATTERN,
    PHONE_PATTERN,
    URL_PATTERN,
    SUSPICIOUS_KEYWORD_PATTERN
)


@dataclass
class ExtractedIntelligence:
    """Container for all extracted intelligence."""
    bank_accounts: list[str] = field(default_factory=list)
    upi_ids: list[str] = field(default_factory=list)
    phishing_links: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    ifsc_codes: list[str] = field(default_factory=list)
    suspicious_keywords: list[str] = field(default_factory=list)
    
    def merge(self, other: "ExtractedIntelligence") -> "ExtractedIntelligence":
        """Merge another intelligence object into this one."""
        return ExtractedIntelligence(
            bank_accounts=list(set(self.bank_accounts + other.bank_accounts)),
            upi_ids=list(set(self.upi_ids + other.upi_ids)),
            phishing_links=list(set(self.phishing_links + other.phishing_links)),
            phone_numbers=list(set(self.phone_numbers + other.phone_numbers)),
            ifsc_codes=list(set(self.ifsc_codes + other.ifsc_codes)),
            suspicious_keywords=list(set(self.suspicious_keywords + other.suspicious_keywords))
        )
    
    def is_empty(self) -> bool:
        """Check if no intelligence has been extracted."""
        return not any([
            self.bank_accounts,
            self.upi_ids,
            self.phishing_links,
            self.phone_numbers,
            self.ifsc_codes
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "bankAccounts": self.bank_accounts,
            "upiIds": self.upi_ids,
            "phishingLinks": self.phishing_links,
            "phoneNumbers": self.phone_numbers,
            "suspiciousKeywords": self.suspicious_keywords
        }


class IntelligenceExtractor:
    """
    Extracts actionable intelligence from text.
    Uses regex patterns optimized for Indian financial data.
    """
    
    # Known legitimate UPI handles to filter out
    LEGITIMATE_UPI_HANDLES = {
        "paytm", "gpay", "phonepe", "ybl", "okhdfcbank", 
        "oksbi", "okicici", "okaxis", "apl", "upi"
    }
    
    def extract(self, text: str) -> ExtractedIntelligence:
        """
        Extract all intelligence from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ExtractedIntelligence with all found entities
        """
        return ExtractedIntelligence(
            bank_accounts=self._extract_bank_accounts(text),
            upi_ids=self._extract_upi_ids(text),
            phishing_links=self._extract_urls(text),
            phone_numbers=self._extract_phone_numbers(text),
            ifsc_codes=self._extract_ifsc_codes(text),
            suspicious_keywords=self._extract_keywords(text)
        )
    
    def _extract_upi_ids(self, text: str) -> list[str]:
        """Extract UPI IDs from text."""
        matches = UPI_PATTERN.findall(text)
        # Filter out email-like patterns and validate format
        upi_ids = []
        for match in matches:
            # UPI IDs typically have shorter bank handles
            parts = match.split("@")
            if len(parts) == 2:
                username, handle = parts
                # Filter out obvious email domains
                if handle.lower() not in ["gmail", "yahoo", "hotmail", "outlook", "email", "mail"]:
                    if len(handle) <= 15:  # UPI handles are typically short
                        upi_ids.append(match)
        return list(set(upi_ids))
    
    def _extract_bank_accounts(self, text: str) -> list[str]:
        """Extract bank account numbers from text."""
        matches = BANK_ACCOUNT_PATTERN.findall(text)
        # Filter out numbers that are likely not bank accounts
        accounts = []
        for match in matches:
            # Bank accounts are typically 11-16 digits
            if 11 <= len(match) <= 16:
                accounts.append(match)
        return list(set(accounts))
    
    def _extract_ifsc_codes(self, text: str) -> list[str]:
        """Extract IFSC codes from text."""
        matches = IFSC_PATTERN.findall(text.upper())
        return list(set(matches))
    
    def _extract_phone_numbers(self, text: str) -> list[str]:
        """Extract Indian phone numbers from text."""
        matches = PHONE_PATTERN.findall(text)
        # Format as +91XXXXXXXXXX
        phone_numbers = [f"+91{match}" for match in matches]
        return list(set(phone_numbers))
    
    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        matches = URL_PATTERN.findall(text)
        # Filter and clean URLs
        urls = []
        for url in matches:
            # Remove trailing punctuation
            url = url.rstrip(".,;:!?")
            # Add http if missing for www URLs
            if url.startswith("www."):
                url = "http://" + url
            urls.append(url)
        return list(set(urls))
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract suspicious keywords from text."""
        matches = SUSPICIOUS_KEYWORD_PATTERN.findall(text)
        return list(set(match.lower() for match in matches))


# Singleton instance
_extractor: IntelligenceExtractor | None = None


def get_intelligence_extractor() -> IntelligenceExtractor:
    """Get or create the intelligence extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = IntelligenceExtractor()
    return _extractor
