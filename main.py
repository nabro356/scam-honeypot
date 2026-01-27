"""
Main FastAPI application entry point.
Scam Honeypot API for autonomous scam detection and engagement.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from api.routes import router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting Scam Honeypot API...")
    settings = get_settings()
    logger.info(f"Using LLM model: {settings.llm_model}")
    
    # Start background session cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Scam Honeypot API...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


async def periodic_cleanup():
    """Background task to clean up expired sessions."""
    from core.session_manager import get_session_manager
    
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            session_manager = get_session_manager()
            cleaned = await session_manager.cleanup_expired()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Scam Honeypot API",
    description="""
    AI-powered honeypot system for scam detection and intelligence extraction.
    
    ## Features
    - Real-time scam detection using pattern matching
    - Autonomous engagement with believable victim personas
    - Intelligence extraction (UPI IDs, bank accounts, phone numbers, URLs)
    - Multi-turn conversation handling
    - GUVI evaluation callback integration
    
    ## Authentication
    All endpoints require `x-api-key` header.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


# Include routers
app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Scam Honeypot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
