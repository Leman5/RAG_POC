"""Tests for app/services/query_router.py - Query routing service."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestRouteQuery:
    """Tests for the async route_query function."""

    @pytest.mark.asyncio
    async def test_route_query_needs_retrieval(self, mock_llm):
        """Test routing when query needs document retrieval."""
        from app.services.query_router import route_query
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = '{"needs_retrieval": true, "reason": "Query asks about documents"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        decision = await route_query("What does the policy say?", mock_llm)
        
        assert decision.needs_retrieval is True
        assert "documents" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_route_query_no_retrieval(self, mock_llm):
        """Test routing when query doesn't need retrieval."""
        from app.services.query_router import route_query
        
        mock_response = MagicMock()
        mock_response.content = '{"needs_retrieval": false, "reason": "General greeting"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        decision = await route_query("Hello, how are you?", mock_llm)
        
        assert decision.needs_retrieval is False
        assert "greeting" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_route_query_handles_json_in_codeblock(self, mock_llm):
        """Test parsing JSON wrapped in markdown code blocks."""
        from app.services.query_router import route_query
        
        mock_response = MagicMock()
        mock_response.content = '''```json
{"needs_retrieval": true, "reason": "Needs document lookup"}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        decision = await route_query("Query text", mock_llm)
        
        assert decision.needs_retrieval is True

    @pytest.mark.asyncio
    async def test_route_query_defaults_on_invalid_json(self, mock_llm):
        """Test defaults to retrieval when JSON parsing fails."""
        from app.services.query_router import route_query
        
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        decision = await route_query("Query", mock_llm)
        
        assert decision.needs_retrieval is True
        assert "defaulting" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_route_query_defaults_on_exception(self, mock_llm):
        """Test defaults to retrieval on LLM exception."""
        from app.services.query_router import route_query
        
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        decision = await route_query("Query", mock_llm)
        
        assert decision.needs_retrieval is True
        assert "error" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_route_query_handles_missing_fields(self, mock_llm):
        """Test handling of JSON with missing fields."""
        from app.services.query_router import route_query
        
        mock_response = MagicMock()
        mock_response.content = '{"needs_retrieval": false}'  # Missing 'reason'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        decision = await route_query("Query", mock_llm)
        
        assert decision.needs_retrieval is False
        assert decision.reason is not None


class TestRouteQuerySync:
    """Tests for the synchronous route_query_sync function."""

    def test_route_query_sync_needs_retrieval(self, mock_llm):
        """Test sync routing when query needs retrieval."""
        from app.services.query_router import route_query_sync
        
        mock_response = MagicMock()
        mock_response.content = '{"needs_retrieval": true, "reason": "Document query"}'
        mock_llm.invoke.return_value = mock_response
        
        decision = route_query_sync("Summarize the report", mock_llm)
        
        assert decision.needs_retrieval is True

    def test_route_query_sync_no_retrieval(self, mock_llm):
        """Test sync routing when query doesn't need retrieval."""
        from app.services.query_router import route_query_sync
        
        mock_response = MagicMock()
        mock_response.content = '{"needs_retrieval": false, "reason": "Math question"}'
        mock_llm.invoke.return_value = mock_response
        
        decision = route_query_sync("What is 2+2?", mock_llm)
        
        assert decision.needs_retrieval is False

    def test_route_query_sync_handles_codeblock(self, mock_llm):
        """Test sync parsing of code block response."""
        from app.services.query_router import route_query_sync
        
        mock_response = MagicMock()
        mock_response.content = '''```
{"needs_retrieval": true, "reason": "Test"}
```'''
        mock_llm.invoke.return_value = mock_response
        
        decision = route_query_sync("Query", mock_llm)
        
        assert decision.needs_retrieval is True

    def test_route_query_sync_defaults_on_error(self, mock_llm):
        """Test sync defaults on exception."""
        from app.services.query_router import route_query_sync
        
        mock_llm.invoke.side_effect = Exception("Connection failed")
        
        decision = route_query_sync("Query", mock_llm)
        
        assert decision.needs_retrieval is True
        assert "error" in decision.reason.lower()


class TestRouterSystemPrompt:
    """Tests for the router system prompt content."""

    def test_system_prompt_contains_examples(self):
        """Test that system prompt contains example queries."""
        from app.services.query_router import ROUTER_SYSTEM_PROMPT
        
        assert "needs_retrieval: true" in ROUTER_SYSTEM_PROMPT.lower() or "needs_retrieval\": true" in ROUTER_SYSTEM_PROMPT
        assert "needs_retrieval: false" in ROUTER_SYSTEM_PROMPT.lower() or "needs_retrieval\": false" in ROUTER_SYSTEM_PROMPT

    def test_system_prompt_specifies_json_format(self):
        """Test that system prompt specifies JSON output format."""
        from app.services.query_router import ROUTER_SYSTEM_PROMPT
        
        assert "json" in ROUTER_SYSTEM_PROMPT.lower()
        assert "needs_retrieval" in ROUTER_SYSTEM_PROMPT
        assert "reason" in ROUTER_SYSTEM_PROMPT

    def test_system_prompt_explains_rag(self):
        """Test that system prompt explains the RAG context."""
        from app.services.query_router import ROUTER_SYSTEM_PROMPT
        
        assert "retrieval" in ROUTER_SYSTEM_PROMPT.lower()
        assert "document" in ROUTER_SYSTEM_PROMPT.lower()
