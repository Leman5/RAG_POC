"""Tests for app/middleware/auth.py - API key authentication."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException


class TestVerifyApiKey:
    """Tests for the verify_api_key function."""

    def test_valid_api_key(self, monkeypatch):
        """Test that valid API key passes verification."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "valid-key-1,valid-key-2")
        
        from app.config import get_settings, Settings
        get_settings.cache_clear()
        
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        result = verify_api_key("valid-key-1", settings)
        
        assert result == "valid-key-1"

    def test_valid_api_key_second_in_list(self, monkeypatch):
        """Test that any valid API key in the list passes."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key-1,key-2,key-3")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        result = verify_api_key("key-2", settings)
        
        assert result == "key-2"

    def test_missing_api_key_raises_401(self, monkeypatch):
        """Test that missing API key raises 401 error."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "valid-key")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None, settings)
        
        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    def test_empty_api_key_raises_401(self, monkeypatch):
        """Test that empty API key raises 401 error."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "valid-key")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("", settings)
        
        # Empty string is falsy in Python
        assert exc_info.value.status_code == 401

    def test_invalid_api_key_raises_401(self, monkeypatch):
        """Test that invalid API key raises 401 error."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "valid-key-1,valid-key-2")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("invalid-key", settings)
        
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    def test_similar_but_wrong_key_raises_401(self, monkeypatch):
        """Test that similar but incorrect key is rejected."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "my-secret-key")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("my-secret-key-extra", settings)
        
        assert exc_info.value.status_code == 401

    def test_key_with_whitespace_not_matched(self, monkeypatch):
        """Test that keys with whitespace are handled correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key1,key2")
        
        from app.config import Settings
        from app.middleware.auth import verify_api_key
        
        settings = Settings()
        
        # Key with leading/trailing whitespace should not match
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(" key1 ", settings)
        
        assert exc_info.value.status_code == 401


class TestApiKeyHeaderDefinition:
    """Tests for the API key header configuration."""

    def test_api_key_header_name(self):
        """Test that the API key header is correctly named."""
        from app.middleware.auth import api_key_header
        
        assert api_key_header.model.name == "X-API-Key"

    def test_api_key_header_auto_error_disabled(self):
        """Test that auto_error is disabled for manual handling."""
        from app.middleware.auth import api_key_header
        
        assert api_key_header.auto_error is False
