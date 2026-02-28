"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_vector_store
from app.models import HealthResponse
from app.routers import chat_router
from app.services.bm25_service import load_bm25_retriever, get_chunk_count
from app.services.parent_store import get_parent_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup: Initialize connections
    settings = get_settings()
    print(f"Starting RAG API with model: {settings.llm_model}")

    # Initialize ChromaDB vector store
    try:
        vector_store = get_vector_store()
        print(f"ChromaDB vector store initialized at: {settings.chroma_persist_dir}")
    except Exception as e:
        print(f"Warning: Could not connect to vector store: {e}")

    # Initialize BM25 retriever from JSON corpus
    try:
        bm25_retriever = load_bm25_retriever(
            settings.bm25_corpus_path,
            k=settings.retrieval_top_k,
        )
        app.state.bm25_retriever = bm25_retriever

        chunk_count = get_chunk_count(settings.bm25_corpus_path)
        parent_count = get_parent_count(settings.parents_path)
        print(f"BM25 retriever loaded: {chunk_count} chunks indexed")
        print(f"Parent document store: {parent_count} parents available")

        if bm25_retriever is None:
            print("Warning: No chunks in BM25 index. Run the ingestion pipeline first.")
    except Exception as e:
        app.state.bm25_retriever = None
        print(f"Warning: Could not load BM25 retriever: {e}")

    yield

    # Shutdown: Cleanup resources
    print("Shutting down RAG API")


app = FastAPI(
    title="RAG POC API",
    description="Retrieval-Augmented Generation API with LangChain and ChromaDB",
    version="2.0.0",
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
        # Try to get vector store to check ChromaDB connection
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
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }
