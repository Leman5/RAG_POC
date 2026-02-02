"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str = Field(..., min_length=1, description="User query to process")


class DocumentChunk(BaseModel):
    """Model representing a retrieved document chunk."""

    content: str = Field(..., description="Text content of the chunk")
    source: str = Field(..., description="Source file path")
    page: int | None = Field(None, description="Page number if available")
    score: float | None = Field(None, description="Similarity score")


class RouteDecision(BaseModel):
    """Model for query routing decision."""

    needs_retrieval: bool = Field(
        ..., description="Whether the query needs document retrieval"
    )
    reason: str = Field(..., description="Reasoning for the decision")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="Generated response")
    used_retrieval: bool = Field(
        ..., description="Whether document retrieval was used"
    )
    sources: list[DocumentChunk] = Field(
        default_factory=list, description="Retrieved source documents"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="healthy", description="Service health status")
    database: str = Field(default="unknown", description="Database connection status")
