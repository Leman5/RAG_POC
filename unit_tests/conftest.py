"""Pytest configuration and shared fixtures for RAG-POC tests."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("API_KEYS", "test-key-1,test-key-2")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./test_chroma_db")
    monkeypatch.setenv("BM25_CORPUS_PATH", "./test_data/bm25_corpus.json")
    monkeypatch.setenv("PARENTS_PATH", "./test_data/parents.json")


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_documents():
    """Create sample LangChain Document objects for testing."""
    return [
        Document(
            page_content="This is the first document about machine learning.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 1,
                "chunk_strategy": "recursive",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Neural networks are a type of machine learning model.",
            metadata={
                "source": "ml_guide.pdf",
                "page": 2,
                "chunk_strategy": "recursive",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content="This is a parent document with detailed information.",
            metadata={
                "source": "tutorial.pdf",
                "page": 1,
                "chunk_strategy": "parent_child",
                "parent_id": "parent-123",
                "is_parent": True,
            },
        ),
    ]


@pytest.fixture
def sample_bm25_corpus():
    """Sample BM25 corpus data for testing."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ai_guide.pdf", "page": 1},
        },
        {
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "ai_guide.pdf", "page": 2},
        },
    ]


@pytest.fixture
def sample_parents():
    """Sample parent documents data for testing."""
    return {
        "parent-001": {
            "content": "This is parent document one with comprehensive content.",
            "metadata": {
                "source": "doc1.pdf",
                "parent_id": "parent-001",
                "is_parent": True,
            },
        },
        "parent-002": {
            "content": "This is parent document two with different content.",
            "metadata": {
                "source": "doc2.pdf",
                "parent_id": "parent-002",
                "is_parent": True,
            },
        },
    }


@pytest.fixture
def mock_llm():
    """Create a mock ChatOpenAI instance."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock()
    mock.invoke = MagicMock()
    return mock


@pytest.fixture
def mock_vector_store():
    """Create a mock ChromaDB vector store."""
    mock = MagicMock()
    mock.as_retriever = MagicMock()
    mock.similarity_search_with_score = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_bm25_retriever():
    """Create a mock BM25 retriever."""
    mock = MagicMock()
    mock.get_relevant_documents = MagicMock(return_value=[])
    return mock


@pytest.fixture
def sample_pdf_text():
    """Sample text extracted from a PDF."""
    return {
        0: "Page 1: Introduction to Machine Learning\n\nMachine learning is a field of study...",
        1: "Page 2: Neural Networks\n\nNeural networks are computing systems...",
        2: "Page 3: Deep Learning\n\nDeep learning is a subset of machine learning...",
    }


@pytest.fixture
def sample_page_descriptions():
    """Sample vision API descriptions."""
    return {
        0: "Screenshot shows a diagram of the machine learning workflow.",
        1: "Screenshot displays a neural network architecture diagram.",
    }


@pytest.fixture
def bm25_corpus_file(temp_directory, sample_bm25_corpus):
    """Create a temporary BM25 corpus file."""
    file_path = os.path.join(temp_directory, "bm25_corpus.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_bm25_corpus, f)
    return file_path


@pytest.fixture
def parents_file(temp_directory, sample_parents):
    """Create a temporary parents file."""
    file_path = os.path.join(temp_directory, "parents.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_parents, f)
    return file_path
