"""Tests for app/services/bm25_service.py - BM25 index service."""

import os
import json
import pytest
from langchain_core.documents import Document


class TestSaveChunks:
    """Tests for the save_chunks function."""

    def test_save_chunks_to_corpus(self, temp_directory):
        """Test saving LangChain Documents to BM25 corpus."""
        from app.services.bm25_service import save_chunks, get_chunk_count
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [
            Document(
                page_content="Machine learning fundamentals",
                metadata={"source": "ml.pdf", "page": 1},
            ),
            Document(
                page_content="Deep learning architectures",
                metadata={"source": "dl.pdf", "page": 1},
            ),
        ]
        
        count = save_chunks(chunks, file_path)
        
        assert count == 2
        assert get_chunk_count(file_path) == 2

    def test_save_chunks_empty_list(self, temp_directory):
        """Test saving empty list returns 0."""
        from app.services.bm25_service import save_chunks
        
        file_path = os.path.join(temp_directory, "empty.json")
        count = save_chunks([], file_path)
        
        assert count == 0

    def test_save_chunks_preserves_metadata(self, temp_directory):
        """Test that chunk metadata is preserved."""
        from app.services.bm25_service import save_chunks
        from app.services.json_store import load_bm25_corpus
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [
            Document(
                page_content="Content here",
                metadata={
                    "source": "test.pdf",
                    "page": 5,
                    "chunk_strategy": "recursive",
                    "custom_field": "custom_value",
                },
            ),
        ]
        
        save_chunks(chunks, file_path)
        loaded = load_bm25_corpus(file_path)
        
        assert loaded[0]["metadata"]["source"] == "test.pdf"
        assert loaded[0]["metadata"]["page"] == 5
        assert loaded[0]["metadata"]["chunk_strategy"] == "recursive"
        assert loaded[0]["metadata"]["custom_field"] == "custom_value"


class TestLoadBm25Retriever:
    """Tests for the load_bm25_retriever function."""

    def test_load_bm25_retriever_success(self, temp_directory):
        """Test loading BM25 retriever from corpus file."""
        from app.services.bm25_service import save_chunks, load_bm25_retriever
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [
            Document(page_content="Machine learning guide", metadata={}),
            Document(page_content="Python programming tutorial", metadata={}),
            Document(page_content="Data science handbook", metadata={}),
        ]
        
        save_chunks(chunks, file_path)
        retriever = load_bm25_retriever(file_path, k=2)
        
        assert retriever is not None
        assert retriever.k == 2

    def test_load_bm25_retriever_empty_corpus(self, temp_directory):
        """Test loading from empty/non-existent corpus returns None."""
        from app.services.bm25_service import load_bm25_retriever
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        retriever = load_bm25_retriever(file_path)
        
        assert retriever is None

    def test_load_bm25_retriever_default_k(self, temp_directory):
        """Test that default k value is 5."""
        from app.services.bm25_service import save_chunks, load_bm25_retriever
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [Document(page_content=f"Doc {i}", metadata={}) for i in range(10)]
        
        save_chunks(chunks, file_path)
        retriever = load_bm25_retriever(file_path)
        
        assert retriever is not None
        assert retriever.k == 5

    def test_load_bm25_retriever_preserves_documents(self, temp_directory):
        """Test that loaded retriever has correct documents."""
        from app.services.bm25_service import save_chunks, load_bm25_retriever
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [
            Document(
                page_content="Unique content for testing",
                metadata={"id": "test-1"},
            ),
        ]
        
        save_chunks(chunks, file_path)
        retriever = load_bm25_retriever(file_path)
        
        # The retriever should have docs attribute
        assert hasattr(retriever, 'docs')
        assert len(retriever.docs) == 1


class TestClearBm25Store:
    """Tests for the clear_bm25_store function."""

    def test_clear_bm25_store(self, temp_directory):
        """Test clearing the BM25 corpus file."""
        from app.services.bm25_service import save_chunks, clear_bm25_store, get_chunk_count
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [Document(page_content="Test", metadata={})]
        
        save_chunks(chunks, file_path)
        assert get_chunk_count(file_path) == 1
        
        clear_bm25_store(file_path)
        assert get_chunk_count(file_path) == 0


class TestGetChunkCount:
    """Tests for the get_chunk_count function."""

    def test_get_chunk_count_with_data(self, temp_directory):
        """Test getting count from populated corpus."""
        from app.services.bm25_service import save_chunks, get_chunk_count
        
        file_path = os.path.join(temp_directory, "bm25.json")
        chunks = [Document(page_content=f"Chunk {i}", metadata={}) for i in range(7)]
        
        save_chunks(chunks, file_path)
        count = get_chunk_count(file_path)
        
        assert count == 7

    def test_get_chunk_count_empty(self, temp_directory):
        """Test getting count from non-existent file."""
        from app.services.bm25_service import get_chunk_count
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        count = get_chunk_count(file_path)
        
        assert count == 0
