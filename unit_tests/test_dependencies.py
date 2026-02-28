"""Tests for app/dependencies.py - Dependency injection."""

import pytest
from unittest.mock import MagicMock, patch


class TestGetEmbeddings:
    """Tests for the get_embeddings function."""

    def test_get_embeddings_returns_instance(self, monkeypatch):
        """Test that get_embeddings returns OpenAIEmbeddings instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.dependencies import get_embeddings
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        
        with patch('app.dependencies.OpenAIEmbeddings') as mock_embeddings:
            mock_instance = MagicMock()
            mock_embeddings.return_value = mock_instance
            
            result = get_embeddings()
            
            assert result == mock_instance
            mock_embeddings.assert_called_once()

    def test_get_embeddings_uses_correct_model(self, monkeypatch):
        """Test that embeddings use configured model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        from app.dependencies import get_embeddings
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        
        with patch('app.dependencies.OpenAIEmbeddings') as mock_embeddings:
            get_embeddings()
            
            call_kwargs = mock_embeddings.call_args.kwargs
            assert call_kwargs['model'] == "text-embedding-3-large"

    def test_get_embeddings_cached(self, monkeypatch):
        """Test that embeddings instance is cached."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.dependencies import get_embeddings
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        
        with patch('app.dependencies.OpenAIEmbeddings') as mock_embeddings:
            result1 = get_embeddings()
            result2 = get_embeddings()
            
            # Should only create once due to caching
            mock_embeddings.assert_called_once()
            assert result1 is result2


class TestGetLLM:
    """Tests for the get_llm function."""

    def test_get_llm_returns_instance(self, monkeypatch):
        """Test that get_llm returns ChatOpenAI instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.dependencies import get_llm
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_llm.cache_clear()
        
        with patch('app.dependencies.ChatOpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance
            
            result = get_llm()
            
            assert result == mock_instance

    def test_get_llm_uses_correct_config(self, monkeypatch):
        """Test LLM uses configured model and temperature."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        
        from app.dependencies import get_llm
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_llm.cache_clear()
        
        with patch('app.dependencies.ChatOpenAI') as mock_llm:
            get_llm()
            
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs['model'] == "gpt-4o"
            assert call_kwargs['temperature'] == 0.7


class TestGetRouterLLM:
    """Tests for the get_router_llm function."""

    def test_get_router_llm_returns_instance(self, monkeypatch):
        """Test that get_router_llm returns ChatOpenAI instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.dependencies import get_router_llm
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_router_llm.cache_clear()
        
        with patch('app.dependencies.ChatOpenAI') as mock_llm:
            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance
            
            result = get_router_llm()
            
            assert result == mock_instance

    def test_get_router_llm_uses_router_model(self, monkeypatch):
        """Test router LLM uses router_model config."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("ROUTER_MODEL", "gpt-3.5-turbo")
        
        from app.dependencies import get_router_llm
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_router_llm.cache_clear()
        
        with patch('app.dependencies.ChatOpenAI') as mock_llm:
            get_router_llm()
            
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs['model'] == "gpt-3.5-turbo"
            assert call_kwargs['temperature'] == 0


class TestGetVectorStore:
    """Tests for the get_vector_store function."""

    def test_get_vector_store_returns_instance(self, monkeypatch):
        """Test that get_vector_store returns Chroma instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "./test_chroma")
        
        from app.dependencies import get_vector_store, get_embeddings
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_vector_store.cache_clear()
        
        with patch('app.dependencies.OpenAIEmbeddings') as mock_embeddings:
            with patch('app.dependencies.Chroma') as mock_chroma:
                mock_embed_instance = MagicMock()
                mock_embeddings.return_value = mock_embed_instance
                mock_store_instance = MagicMock()
                mock_chroma.return_value = mock_store_instance
                
                result = get_vector_store()
                
                assert result == mock_store_instance
                mock_chroma.assert_called_once()

    def test_get_vector_store_uses_correct_config(self, monkeypatch):
        """Test vector store uses configured directory."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "./custom_chroma")
        
        from app.dependencies import get_vector_store, get_embeddings
        from app.config import get_settings
        
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_vector_store.cache_clear()
        
        with patch('app.dependencies.OpenAIEmbeddings'):
            with patch('app.dependencies.Chroma') as mock_chroma:
                get_vector_store()
                
                call_kwargs = mock_chroma.call_args.kwargs
                assert call_kwargs['persist_directory'] == "./custom_chroma"
                assert call_kwargs['collection_name'] == "documents"


class TestGetBm25Retriever:
    """Tests for the get_bm25_retriever function."""

    def test_get_bm25_retriever_from_app_state(self):
        """Test retrieving BM25 retriever from app state."""
        from app.dependencies import get_bm25_retriever
        
        mock_request = MagicMock()
        mock_retriever = MagicMock()
        mock_request.app.state.bm25_retriever = mock_retriever
        
        result = get_bm25_retriever(mock_request)
        
        assert result == mock_retriever

    def test_get_bm25_retriever_missing(self):
        """Test handling when BM25 retriever is not loaded."""
        from app.dependencies import get_bm25_retriever
        
        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])  # No bm25_retriever attribute
        
        result = get_bm25_retriever(mock_request)
        
        assert result is None
