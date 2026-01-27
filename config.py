"""
Configuration management for Scam Honeypot API.
Uses pydantic-settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Authentication
    api_key: str = "dev_key_123"
    
    # NVIDIA LLM Configuration
    nvidia_api_key: str = ""
    llm_model: str = "meta/llama-3.1-8b-instruct"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256
    
    # GUVI Callback
    guvi_callback_url: str = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
    
    # Session Settings
    session_ttl_seconds: int = 3600
    max_messages_per_session: int = 50
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
