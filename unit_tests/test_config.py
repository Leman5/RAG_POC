"""Tests for app/config.py - Application configuration."""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Tests for the Settings class."""

    def test_settings_loads_from_env(self, monkeypatch):
        """Test that Settings correctly loads from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("API_KEYS", "key1,key2,key3")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/custom/chroma")
        monkeypatch.setenv("BM25_CORPUS_PATH", "/custom/bm25.json")
        monkeypatch.setenv("PARENTS_PATH", "/custom/parents.json")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        
        # Clear the lru_cache to ensure fresh settings
        from app.config import get_settings, Settings
        get_settings.cache_clear()
        
        settings = Settings()
        
        assert settings.openai_api_key == "test-key-123"
        assert settings.chroma_persist_dir == "/custom/chroma"
        assert settings.bm25_corpus_path == "/custom/bm25.json"
        assert settings.parents_path == "/custom/parents.json"
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.llm_model == "gpt-4o"
        assert settings.retrieval_top_k == 10

    def test_api_keys_list_parsing(self, monkeypatch):
        """Test that api_keys_list correctly parses comma-separated keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "  key1  , key2,  key3  , key4")
        
        from app.config import Settings
        settings = Settings()
        
        keys = settings.api_keys_list
        assert keys == ["key1", "key2", "key3", "key4"]

    def test_api_keys_list_handles_empty_values(self, monkeypatch):
        """Test that api_keys_list filters out empty values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1,, ,key2,  ,key3")
        
        from app.config import Settings
        settings = Settings()
        
        keys = settings.api_keys_list
        assert keys == ["key1", "key2", "key3"]

    def test_default_values(self, monkeypatch):
        """Test that default values are set correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.config import Settings
        settings = Settings()
        
        assert settings.chroma_persist_dir == "./chroma_db"
        assert settings.bm25_corpus_path == "./data/bm25_corpus.json"
        assert settings.parents_path == "./data/parents.json"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.llm_model == "gpt-4o-mini"
        assert settings.router_model == "gpt-3.5-turbo"
        assert settings.vision_model == "gpt-4o-mini"
        assert settings.retrieval_top_k == 5
        assert settings.similarity_threshold == 0.7
        assert settings.vision_dpi == 150
        assert settings.chunk_size_recursive == 1000
        assert settings.chunk_overlap_recursive == 200
        assert settings.chunk_size_child == 400
        assert settings.chunk_overlap_child == 50
        assert settings.doc_level_max_chars == 2000

    def test_get_settings_returns_cached_instance(self, monkeypatch):
        """Test that get_settings returns a cached instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1")
        
        from app.config import get_settings
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_settings_missing_required_raises_error(self, monkeypatch):
        """Test that missing required fields raise validation error."""
        # Clear the API key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("API_KEYS", raising=False)
        
        from app.config import Settings
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            Settings()


class TestSettingsValidation:
    """Tests for Settings field validation."""

    def test_retrieval_top_k_accepts_positive_int(self, monkeypatch):
        """Test retrieval_top_k accepts positive integers."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
        
        from app.config import Settings
        settings = Settings()
        
        assert settings.retrieval_top_k == 15

    def test_similarity_threshold_accepts_float(self, monkeypatch):
        """Test similarity_threshold accepts float values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.85")
        
        from app.config import Settings
        settings = Settings()
        
        assert settings.similarity_threshold == 0.85

    def test_vision_dpi_accepts_custom_value(self, monkeypatch):
        """Test vision_dpi accepts custom values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1")
        monkeypatch.setenv("VISION_DPI", "300")
        
        from app.config import Settings
        settings = Settings()
        
        assert settings.vision_dpi == 300
