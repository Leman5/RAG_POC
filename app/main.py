"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_vector_store
from app.models import HealthResponse
from app.routers import chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup: Initialize connections
    settings = get_settings()
    print(f"Starting RAG API with model: {settings.llm_model}")

    # Initialize vector store connection
    try:
        vector_store = get_vector_store()
        print("Vector store connection established")
    except Exception as e:
        print(f"Warning: Could not connect to vector store: {e}")

    yield

    # Shutdown: Cleanup resources
    print("Shutting down RAG API")


app = FastAPI(
    title="RAG POC API",
    description="Retrieval-Augmented Generation API with LangChain and PGVector",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Try to get vector store to check database connection
        get_vector_store()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(status="healthy", database=db_status)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG POC API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
