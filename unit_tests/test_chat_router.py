"""Tests for app/routers/chat.py - Chat API endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def app_with_mocked_deps(monkeypatch):
    """Create FastAPI app with mocked dependencies."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("API_KEYS", "valid-api-key")
    
    # Clear caches
    from app.config import get_settings
    from app.dependencies import get_embeddings, get_llm, get_router_llm, get_vector_store
    
    get_settings.cache_clear()
    get_embeddings.cache_clear()
    get_llm.cache_clear()
    get_router_llm.cache_clear()
    get_vector_store.cache_clear()
    
    from app.routers.chat import router
    
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    
    return app


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_requires_api_key(self, app_with_mocked_deps):
        """Test that chat endpoint requires API key."""
        client = TestClient(app_with_mocked_deps)
        
        response = client.post(
            "/api/v1/chat",
            json={"query": "Hello"},
        )
        
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_rejects_invalid_api_key(self, app_with_mocked_deps):
        """Test that invalid API key is rejected."""
        client = TestClient(app_with_mocked_deps)
        
        response = client.post(
            "/api/v1/chat",
            json={"query": "Hello"},
            headers={"X-API-Key": "invalid-key"},
        )
        
        assert response.status_code == 401

    def test_chat_validates_empty_query(self, app_with_mocked_deps):
        """Test that empty query is rejected."""
        client = TestClient(app_with_mocked_deps)
        
        response = client.post(
            "/api/v1/chat",
            json={"query": ""},
            headers={"X-API-Key": "valid-api-key"},
        )
        
        assert response.status_code == 422


class TestChatDirectEndpoint:
    """Tests for the /chat/direct endpoint."""

    def test_chat_direct_requires_api_key(self, app_with_mocked_deps):
        """Test that direct chat requires API key."""
        client = TestClient(app_with_mocked_deps)
        
        response = client.post(
            "/api/v1/chat/direct",
            json={"query": "Hello"},
        )
        
        assert response.status_code == 401


class TestChatRagEndpoint:
    """Tests for the /chat/rag endpoint."""

    def test_chat_rag_requires_api_key(self, app_with_mocked_deps):
        """Test that RAG chat requires API key."""
        client = TestClient(app_with_mocked_deps)
        
        response = client.post(
            "/api/v1/chat/rag",
            json={"query": "Summarize the document"},
        )
        
        assert response.status_code == 401


class TestChatResponseStructure:
    """Tests for chat response structure."""

    def test_chat_response_model_fields(self):
        """Test ChatResponse has required fields."""
        from app.models.schemas import ChatResponse
        
        response = ChatResponse(
            answer="Test answer",
            used_retrieval=True,
            sources=[],
        )
        
        assert hasattr(response, 'answer')
        assert hasattr(response, 'used_retrieval')
        assert hasattr(response, 'sources')


class TestChatRouterIntegration:
    """Integration tests for chat router logic."""

    @pytest.mark.asyncio
    async def test_chat_flow_with_retrieval(self):
        """Test chat flow when retrieval is needed."""
        from app.routers.chat import chat
        from app.models.schemas import ChatRequest, RouteDecision
        
        mock_request = ChatRequest(query="What does the document say?")
        mock_settings = MagicMock()
        mock_settings.parents_path = "./parents.json"
        mock_settings.retrieval_top_k = 5
        
        mock_llm = MagicMock()
        mock_router_llm = MagicMock()
        mock_vector_store = MagicMock()
        mock_bm25 = None
        
        # Mock route_query to return needs_retrieval=True
        with patch('app.routers.chat.route_query') as mock_route:
            mock_route.return_value = RouteDecision(
                needs_retrieval=True,
                reason="Document query",
            )
            
            with patch('app.routers.chat.retrieve_documents') as mock_retrieve:
                mock_retrieve.return_value = []
                
                with patch('app.routers.chat.generate_response') as mock_generate:
                    mock_generate.return_value = "Generated answer"
                    
                    response = await chat(
                        request=mock_request,
                        api_key="valid-key",
                        settings=mock_settings,
                        llm=mock_llm,
                        router_llm=mock_router_llm,
                        vector_store=mock_vector_store,
                        bm25_retriever=mock_bm25,
                    )
        
        assert response.answer == "Generated answer"

    @pytest.mark.asyncio
    async def test_chat_flow_without_retrieval(self):
        """Test chat flow when retrieval is not needed."""
        from app.routers.chat import chat
        from app.models.schemas import ChatRequest, RouteDecision
        
        mock_request = ChatRequest(query="Hello!")
        mock_settings = MagicMock()
        
        mock_llm = MagicMock()
        mock_router_llm = MagicMock()
        mock_vector_store = MagicMock()
        mock_bm25 = None
        
        with patch('app.routers.chat.route_query') as mock_route:
            mock_route.return_value = RouteDecision(
                needs_retrieval=False,
                reason="Greeting",
            )
            
            with patch('app.routers.chat.generate_response') as mock_generate:
                mock_generate.return_value = "Hello there!"
                
                response = await chat(
                    request=mock_request,
                    api_key="valid-key",
                    settings=mock_settings,
                    llm=mock_llm,
                    router_llm=mock_router_llm,
                    vector_store=mock_vector_store,
                    bm25_retriever=mock_bm25,
                )
        
        assert response.answer == "Hello there!"
        assert response.used_retrieval is False
        assert response.sources == []
