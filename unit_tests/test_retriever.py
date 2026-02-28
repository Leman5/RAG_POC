"""Tests for app/services/retriever.py - Hybrid retrieval service."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestBuildEnsembleRetriever:
    """Tests for the build_ensemble_retriever function."""

    def test_build_ensemble_with_bm25(self, mock_vector_store, mock_bm25_retriever):
        """Test building ensemble retriever with both BM25 and dense."""
        from app.services.retriever import build_ensemble_retriever
        
        # Setup mock
        mock_dense_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_dense_retriever
        
        ensemble = build_ensemble_retriever(
            vector_store=mock_vector_store,
            bm25_retriever=mock_bm25_retriever,
            k=5,
            bm25_weight=0.4,
            dense_weight=0.6,
        )
        
        assert ensemble is not None
        assert len(ensemble.retrievers) == 2
        assert ensemble.weights == [0.4, 0.6]

    def test_build_ensemble_without_bm25(self, mock_vector_store):
        """Test building ensemble with only dense retriever."""
        from app.services.retriever import build_ensemble_retriever
        
        mock_dense_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_dense_retriever
        
        ensemble = build_ensemble_retriever(
            vector_store=mock_vector_store,
            bm25_retriever=None,
            k=5,
        )
        
        assert ensemble is not None
        assert len(ensemble.retrievers) == 1
        assert ensemble.weights == [1.0]

    def test_build_ensemble_custom_k(self, mock_vector_store, mock_bm25_retriever):
        """Test that k parameter is passed to retrievers."""
        from app.services.retriever import build_ensemble_retriever
        
        mock_dense_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_dense_retriever
        
        build_ensemble_retriever(
            vector_store=mock_vector_store,
            bm25_retriever=mock_bm25_retriever,
            k=10,
        )
        
        # Verify as_retriever was called with correct k
        mock_vector_store.as_retriever.assert_called_once()
        call_kwargs = mock_vector_store.as_retriever.call_args.kwargs
        assert call_kwargs["search_kwargs"]["k"] == 10


class TestDocumentToChunk:
    """Tests for the _document_to_chunk function."""

    def test_document_to_chunk_basic(self):
        """Test basic conversion of Document to DocumentChunk."""
        from app.services.retriever import _document_to_chunk
        
        doc = Document(
            page_content="Test content",
            metadata={
                "source": "test.pdf",
                "page": 5,
                "category": "tutorials",
                "chunk_strategy": "recursive",
            },
        )
        
        chunk = _document_to_chunk(doc)
        
        assert chunk.content == "Test content"
        assert chunk.source == "test.pdf"
        assert chunk.page == 5
        assert chunk.category == "tutorials"
        assert chunk.chunk_strategy == "recursive"

    def test_document_to_chunk_with_score(self):
        """Test conversion with similarity score."""
        from app.services.retriever import _document_to_chunk
        
        doc = Document(
            page_content="Content",
            metadata={"source": "doc.pdf"},
        )
        
        chunk = _document_to_chunk(doc, score=0.87654321)
        
        assert chunk.score == 0.8765

    def test_document_to_chunk_missing_metadata(self):
        """Test handling of missing metadata fields."""
        from app.services.retriever import _document_to_chunk
        
        doc = Document(
            page_content="Content",
            metadata={},
        )
        
        chunk = _document_to_chunk(doc)
        
        assert chunk.source == "unknown"
        assert chunk.page is None
        assert chunk.category is None

    def test_document_to_chunk_page_type_conversion(self):
        """Test that page is converted to int."""
        from app.services.retriever import _document_to_chunk
        
        doc = Document(
            page_content="Content",
            metadata={"source": "doc.pdf", "page": "3"},
        )
        
        chunk = _document_to_chunk(doc)
        
        assert chunk.page == 3
        assert isinstance(chunk.page, int)

    def test_document_to_chunk_invalid_page(self):
        """Test handling of invalid page values."""
        from app.services.retriever import _document_to_chunk
        
        doc = Document(
            page_content="Content",
            metadata={"source": "doc.pdf", "page": "invalid"},
        )
        
        chunk = _document_to_chunk(doc)
        
        assert chunk.page is None


class TestResolveParents:
    """Tests for the _resolve_parents function."""

    def test_resolve_parents_replaces_child(self, temp_directory):
        """Test that child chunks are replaced with parents."""
        import os
        from app.services.retriever import _resolve_parents
        from app.services.parent_store import save_parents
        
        parents_path = os.path.join(temp_directory, "parents.json")
        
        # Save a parent
        parent_docs = [
            Document(
                page_content="Full parent content",
                metadata={"parent_id": "p-001", "is_parent": True},
            ),
        ]
        save_parents(parent_docs, parents_path)
        
        # Child document referencing the parent
        results = [
            Document(
                page_content="Child chunk",
                metadata={"parent_id": "p-001", "is_parent": False},
            ),
        ]
        
        resolved = _resolve_parents(results, parents_path)
        
        assert len(resolved) == 1
        assert resolved[0].page_content == "Full parent content"

    def test_resolve_parents_keeps_parent_chunks(self, temp_directory):
        """Test that parent chunks are kept as-is."""
        import os
        from app.services.retriever import _resolve_parents
        
        parents_path = os.path.join(temp_directory, "parents.json")
        
        results = [
            Document(
                page_content="I am a parent chunk",
                metadata={"is_parent": True},
            ),
        ]
        
        resolved = _resolve_parents(results, parents_path)
        
        assert len(resolved) == 1
        assert resolved[0].page_content == "I am a parent chunk"

    def test_resolve_parents_fallback_on_missing(self, temp_directory):
        """Test fallback to child when parent not found."""
        import os
        from app.services.retriever import _resolve_parents
        
        parents_path = os.path.join(temp_directory, "empty_parents.json")
        
        results = [
            Document(
                page_content="Orphan child",
                metadata={"parent_id": "nonexistent", "is_parent": False},
            ),
        ]
        
        resolved = _resolve_parents(results, parents_path)
        
        assert len(resolved) == 1
        assert resolved[0].page_content == "Orphan child"


class TestFormatContextForPrompt:
    """Tests for the format_context_for_prompt function."""

    def test_format_context_basic(self):
        """Test basic context formatting."""
        from app.services.retriever import format_context_for_prompt
        from app.models.schemas import DocumentChunk
        
        chunks = [
            DocumentChunk(
                content="First chunk content",
                source="doc1.pdf",
                page=1,
            ),
            DocumentChunk(
                content="Second chunk content",
                source="doc2.pdf",
                page=3,
            ),
        ]
        
        context = format_context_for_prompt(chunks)
        
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert "doc1.pdf" in context
        assert "Page 1" in context
        assert "First chunk content" in context

    def test_format_context_with_category(self):
        """Test context formatting includes category."""
        from app.services.retriever import format_context_for_prompt
        from app.models.schemas import DocumentChunk
        
        chunks = [
            DocumentChunk(
                content="Content",
                source="doc.pdf",
                category="tutorials",
            ),
        ]
        
        context = format_context_for_prompt(chunks)
        
        assert "Category: tutorials" in context

    def test_format_context_empty_list(self):
        """Test formatting empty chunk list."""
        from app.services.retriever import format_context_for_prompt
        
        context = format_context_for_prompt([])
        
        assert context == ""

    def test_format_context_no_page(self):
        """Test formatting when page is None."""
        from app.services.retriever import format_context_for_prompt
        from app.models.schemas import DocumentChunk
        
        chunks = [
            DocumentChunk(content="Content", source="doc.pdf", page=None),
        ]
        
        context = format_context_for_prompt(chunks)
        
        assert "Page" not in context
        assert "doc.pdf" in context


class TestRetrieveDocuments:
    """Tests for the retrieve_documents async function."""

    @pytest.mark.asyncio
    async def test_retrieve_documents_empty_when_no_retriever(
        self, mock_vector_store, monkeypatch
    ):
        """Test returns empty list when no documents found."""
        from app.services.retriever import retrieve_documents
        from app.config import Settings
        
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("API_KEYS", "key")
        
        settings = Settings()
        
        # Mock empty results
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        with patch('app.services.retriever.build_ensemble_retriever') as mock_build:
            mock_ensemble = MagicMock()
            mock_ensemble.invoke.return_value = []
            mock_build.return_value = mock_ensemble
            
            results = await retrieve_documents(
                query="test query",
                vector_store=mock_vector_store,
                settings=settings,
            )
        
        assert results == []
