from urllib.request import Request

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from starlette.responses import JSONResponse

from api.models import ErrorResponse
from api.routes import router
from utils.logger import logger
from config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    logger.info("Starting Local LawBot API...")

    # Startup
    try:
        logger.info("API startup completed")
        yield
    finally:
        # Shutdown
        logger.info("Shutting down Local LawBot API...")


# Create FastAPI application
app = FastAPI(
    title="Local LawBot API",
    description="Retrieval-Augmented Legal Assistant using Gemini Pro and ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Local LawBot: Retrieval-Augmented Legal Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "api": "/api/v1"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    logger.info(f"Starting server on {config.api_host}:{config.api_port}")
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        log_level="info"
    )