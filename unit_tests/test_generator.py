"""Tests for app/services/generator.py - Response generation service."""

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestGenerateResponse:
    """Tests for the async generate_response function."""

    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, mock_llm):
        """Test response generation with document context."""
        from app.services.generator import generate_response
        from app.models.schemas import DocumentChunk
        
        mock_response = MagicMock()
        mock_response.content = "Based on the documents, the answer is..."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        chunks = [
            DocumentChunk(content="Relevant content", source="doc.pdf"),
        ]
        
        answer = await generate_response("What is X?", mock_llm, chunks)
        
        assert answer == "Based on the documents, the answer is..."
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_without_context(self, mock_llm):
        """Test direct response generation without context."""
        from app.services.generator import generate_response
        
        mock_response = MagicMock()
        mock_response.content = "Hello! I'm happy to help."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        answer = await generate_response("Hello!", mock_llm, None)
        
        assert answer == "Hello! I'm happy to help."

    @pytest.mark.asyncio
    async def test_generate_response_empty_chunks(self, mock_llm):
        """Test response with empty chunk list uses direct mode."""
        from app.services.generator import generate_response
        
        mock_response = MagicMock()
        mock_response.content = "Direct response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Empty list is falsy, should use direct mode
        answer = await generate_response("Query", mock_llm, [])
        
        assert answer == "Direct response"

    @pytest.mark.asyncio
    async def test_generate_response_handles_error(self, mock_llm):
        """Test error handling in response generation."""
        from app.services.generator import generate_response
        
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        answer = await generate_response("Query", mock_llm, None)
        
        assert "error" in answer.lower()


class TestGenerateResponseSync:
    """Tests for the synchronous generate_response_sync function."""

    def test_generate_response_sync_with_context(self, mock_llm):
        """Test sync response with context."""
        from app.services.generator import generate_response_sync
        from app.models.schemas import DocumentChunk
        
        mock_response = MagicMock()
        mock_response.content = "Context-based answer"
        mock_llm.invoke.return_value = mock_response
        
        chunks = [DocumentChunk(content="Context", source="doc.pdf")]
        
        answer = generate_response_sync("Query", mock_llm, chunks)
        
        assert answer == "Context-based answer"

    def test_generate_response_sync_direct(self, mock_llm):
        """Test sync direct response."""
        from app.services.generator import generate_response_sync
        
        mock_response = MagicMock()
        mock_response.content = "Direct answer"
        mock_llm.invoke.return_value = mock_response
        
        answer = generate_response_sync("Hi", mock_llm, None)
        
        assert answer == "Direct answer"

    def test_generate_response_sync_error_handling(self, mock_llm):
        """Test sync error handling."""
        from app.services.generator import generate_response_sync
        
        mock_llm.invoke.side_effect = Exception("Connection failed")
        
        answer = generate_response_sync("Query", mock_llm, None)
        
        assert "error" in answer.lower()


class TestGeneratorPrompts:
    """Tests for the generator prompt templates."""

    def test_rag_system_prompt_has_context_placeholder(self):
        """Test RAG prompt has context placeholder."""
        from app.services.generator import RAG_SYSTEM_PROMPT
        
        assert "{context}" in RAG_SYSTEM_PROMPT

    def test_rag_system_prompt_instructions(self):
        """Test RAG prompt contains instructions."""
        from app.services.generator import RAG_SYSTEM_PROMPT
        
        assert "document" in RAG_SYSTEM_PROMPT.lower()
        assert "context" in RAG_SYSTEM_PROMPT.lower()
        assert "answer" in RAG_SYSTEM_PROMPT.lower()

    def test_direct_system_prompt_no_context(self):
        """Test direct prompt doesn't expect context."""
        from app.services.generator import DIRECT_SYSTEM_PROMPT
        
        assert "{context}" not in DIRECT_SYSTEM_PROMPT
        assert "general knowledge" in DIRECT_SYSTEM_PROMPT.lower() or "conversational" in DIRECT_SYSTEM_PROMPT.lower()


class TestGenerateWithContext:
    """Tests for the _generate_with_context function."""

    @pytest.mark.asyncio
    async def test_generate_with_context_formats_chunks(self, mock_llm):
        """Test that chunks are formatted into context."""
        from app.services.generator import _generate_with_context
        from app.models.schemas import DocumentChunk
        
        mock_response = MagicMock()
        mock_response.content = "Answer"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        chunks = [
            DocumentChunk(content="Chunk 1", source="doc1.pdf", page=1),
            DocumentChunk(content="Chunk 2", source="doc2.pdf", page=2),
        ]
        
        await _generate_with_context("Query", mock_llm, chunks)
        
        # Verify the message was constructed with context
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_message = call_args[0]
        assert "Chunk 1" in system_message.content
        assert "Chunk 2" in system_message.content


class TestGenerateDirect:
    """Tests for the _generate_direct function."""

    @pytest.mark.asyncio
    async def test_generate_direct_uses_correct_prompt(self, mock_llm):
        """Test direct generation uses the correct system prompt."""
        from app.services.generator import _generate_direct, DIRECT_SYSTEM_PROMPT
        
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        await _generate_direct("Hello", mock_llm)
        
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_message = call_args[0]
        assert system_message.content == DIRECT_SYSTEM_PROMPT
