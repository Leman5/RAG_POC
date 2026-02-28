"""Tests for app/models/schemas.py - Pydantic models."""

import pytest
from pydantic import ValidationError


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_chat_request(self):
        """Test ChatRequest accepts valid input."""
        from app.models.schemas import ChatRequest
        
        request = ChatRequest(query="What is machine learning?")
        assert request.query == "What is machine learning?"

    def test_empty_query_raises_error(self):
        """Test ChatRequest rejects empty query."""
        from app.models.schemas import ChatRequest
        
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="")
        
        assert "min_length" in str(exc_info.value).lower() or "String should have at least 1" in str(exc_info.value)

    def test_whitespace_only_query_is_valid(self):
        """Test ChatRequest accepts whitespace (validation is on length, not content)."""
        from app.models.schemas import ChatRequest
        
        # Whitespace counts as characters
        request = ChatRequest(query="   ")
        assert request.query == "   "

    def test_missing_query_raises_error(self):
        """Test ChatRequest rejects missing query."""
        from app.models.schemas import ChatRequest
        
        with pytest.raises(ValidationError):
            ChatRequest()


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_valid_document_chunk(self):
        """Test DocumentChunk with all fields."""
        from app.models.schemas import DocumentChunk
        
        chunk = DocumentChunk(
            content="Sample document content",
            source="document.pdf",
            page=5,
            score=0.95,
            category="tutorials",
            chunk_strategy="recursive",
        )
        
        assert chunk.content == "Sample document content"
        assert chunk.source == "document.pdf"
        assert chunk.page == 5
        assert chunk.score == 0.95
        assert chunk.category == "tutorials"
        assert chunk.chunk_strategy == "recursive"

    def test_document_chunk_minimal(self):
        """Test DocumentChunk with only required fields."""
        from app.models.schemas import DocumentChunk
        
        chunk = DocumentChunk(
            content="Minimal content",
            source="file.pdf",
        )
        
        assert chunk.content == "Minimal content"
        assert chunk.source == "file.pdf"
        assert chunk.page is None
        assert chunk.score is None
        assert chunk.category is None
        assert chunk.chunk_strategy is None

    def test_document_chunk_optional_fields_nullable(self):
        """Test that optional fields accept None."""
        from app.models.schemas import DocumentChunk
        
        chunk = DocumentChunk(
            content="Content",
            source="source.pdf",
            page=None,
            score=None,
            category=None,
            chunk_strategy=None,
        )
        
        assert chunk.page is None
        assert chunk.score is None


class TestRouteDecision:
    """Tests for RouteDecision model."""

    def test_valid_route_decision_retrieval_needed(self):
        """Test RouteDecision when retrieval is needed."""
        from app.models.schemas import RouteDecision
        
        decision = RouteDecision(
            needs_retrieval=True,
            reason="Query asks about document content",
        )
        
        assert decision.needs_retrieval is True
        assert decision.reason == "Query asks about document content"

    def test_valid_route_decision_no_retrieval(self):
        """Test RouteDecision when retrieval is not needed."""
        from app.models.schemas import RouteDecision
        
        decision = RouteDecision(
            needs_retrieval=False,
            reason="General greeting query",
        )
        
        assert decision.needs_retrieval is False
        assert decision.reason == "General greeting query"

    def test_missing_fields_raises_error(self):
        """Test RouteDecision requires both fields."""
        from app.models.schemas import RouteDecision
        
        with pytest.raises(ValidationError):
            RouteDecision(needs_retrieval=True)
        
        with pytest.raises(ValidationError):
            RouteDecision(reason="Some reason")


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_valid_chat_response_with_sources(self):
        """Test ChatResponse with retrieval and sources."""
        from app.models.schemas import ChatResponse, DocumentChunk
        
        sources = [
            DocumentChunk(content="Source 1", source="doc1.pdf"),
            DocumentChunk(content="Source 2", source="doc2.pdf"),
        ]
        
        response = ChatResponse(
            answer="Based on the documents...",
            used_retrieval=True,
            sources=sources,
        )
        
        assert response.answer == "Based on the documents..."
        assert response.used_retrieval is True
        assert len(response.sources) == 2

    def test_valid_chat_response_without_sources(self):
        """Test ChatResponse without retrieval."""
        from app.models.schemas import ChatResponse
        
        response = ChatResponse(
            answer="Hello! How can I help?",
            used_retrieval=False,
            sources=[],
        )
        
        assert response.answer == "Hello! How can I help?"
        assert response.used_retrieval is False
        assert len(response.sources) == 0

    def test_chat_response_default_sources(self):
        """Test ChatResponse uses empty list as default for sources."""
        from app.models.schemas import ChatResponse
        
        response = ChatResponse(
            answer="Answer text",
            used_retrieval=False,
        )
        
        assert response.sources == []


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health_response(self):
        """Test HealthResponse with all fields."""
        from app.models.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            database="connected",
        )
        
        assert response.status == "healthy"
        assert response.database == "connected"

    def test_health_response_defaults(self):
        """Test HealthResponse default values."""
        from app.models.schemas import HealthResponse
        
        response = HealthResponse()
        
        assert response.status == "healthy"
        assert response.database == "unknown"
