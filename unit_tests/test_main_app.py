"""Tests for app/main.py - FastAPI application."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(monkeypatch):
    """Create a test client with mocked dependencies."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("API_KEYS", "test-api-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./test_chroma")
    monkeypatch.setenv("BM25_CORPUS_PATH", "./test_bm25.json")
    monkeypatch.setenv("PARENTS_PATH", "./test_parents.json")
    
    # Clear all caches
    from app.config import get_settings
    from app.dependencies import get_embeddings, get_llm, get_router_llm, get_vector_store
    
    get_settings.cache_clear()
    get_embeddings.cache_clear()
    get_llm.cache_clear()
    get_router_llm.cache_clear()
    get_vector_store.cache_clear()
    
    # Mock dependencies to avoid actual connections
    with patch('app.dependencies.OpenAIEmbeddings'):
        with patch('app.dependencies.ChatOpenAI'):
            with patch('app.dependencies.Chroma'):
                with patch('app.main.get_vector_store'):
                    with patch('app.main.load_bm25_retriever', return_value=None):
                        with patch('app.main.get_chunk_count', return_value=0):
                            with patch('app.main.get_parent_count', return_value=0):
                                from app.main import app
                                yield TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_info(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert data["name"] == "RAG POC API"

    def test_root_includes_docs_url(self, test_client):
        """Test root endpoint includes docs URL."""
        response = test_client.get("/")
        
        data = response.json()
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_status(self, test_client):
        """Test health check returns status."""
        with patch('app.main.get_vector_store') as mock_vs:
            mock_vs.return_value = MagicMock()
            response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data

    def test_health_check_reports_db_connected(self, test_client):
        """Test health check reports database status."""
        with patch('app.main.get_vector_store') as mock_vs:
            mock_vs.return_value = MagicMock()
            response = test_client.get("/health")
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"

    def test_health_check_reports_db_disconnected(self, test_client):
        """Test health check handles database connection failure."""
        with patch('app.main.get_vector_store') as mock_vs:
            mock_vs.side_effect = Exception("Connection failed")
            response = test_client.get("/health")
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "disconnected"


class TestCorsMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_all_origins(self, test_client):
        """Test CORS allows any origin."""
        response = test_client.options(
            "/",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        
        # CORS should allow the request
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestApiRouterInclusion:
    """Tests for API router inclusion."""

    def test_chat_router_included(self, test_client):
        """Test that chat router is accessible."""
        # Attempt to access chat endpoint (should return 401 without API key)
        response = test_client.post(
            "/api/v1/chat",
            json={"query": "test"},
        )
        
        # 401 means endpoint exists but auth failed
        assert response.status_code == 401

    def test_chat_direct_router_included(self, test_client):
        """Test that chat/direct endpoint is accessible."""
        response = test_client.post(
            "/api/v1/chat/direct",
            json={"query": "test"},
        )
        
        assert response.status_code == 401

    def test_chat_rag_router_included(self, test_client):
        """Test that chat/rag endpoint is accessible."""
        response = test_client.post(
            "/api/v1/chat/rag",
            json={"query": "test"},
        )
        
        assert response.status_code == 401


class TestLifespan:
    """Tests for application lifespan events."""

    def test_lifespan_initializes_without_error(self, monkeypatch):
        """Test that lifespan context manager initializes correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key")
        
        from app.config import get_settings
        get_settings.cache_clear()
        
        with patch('app.main.get_vector_store') as mock_vs:
            with patch('app.main.load_bm25_retriever', return_value=None):
                with patch('app.main.get_chunk_count', return_value=0):
                    with patch('app.main.get_parent_count', return_value=0):
                        mock_vs.return_value = MagicMock()
                        
                        # Import should not raise
                        from app.main import app
                        assert app is not None


class TestOpenApiDocs:
    """Tests for OpenAPI documentation."""

    def test_openapi_docs_accessible(self, test_client):
        """Test that OpenAPI docs are accessible."""
        response = test_client.get("/docs")
        
        # Should return HTML (docs page) or redirect
        assert response.status_code in [200, 307]

    def test_openapi_schema_accessible(self, test_client):
        """Test that OpenAPI schema is accessible."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert data["info"]["title"] == "RAG POC API"
