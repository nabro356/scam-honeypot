"""
API key authentication middleware.
Validates x-api-key header for all protected endpoints.
"""

from fastapi import Header, HTTPException, status
from config import get_settings


async def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """
    Dependency to verify API key from request headers.
    
    Args:
        x_api_key: The API key from the x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    settings = get_settings()
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return x_api_key
