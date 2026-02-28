"""JSON-backed BM25 keyword index service.

Persists chunk texts and metadata in a JSON file so the BM25 index
can be rebuilt on server restart. At runtime the index lives in memory
(via rank_bm25) for fast keyword search.
"""

import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.json_store import (
    save_bm25_corpus,
    load_bm25_corpus,
    clear_bm25_corpus,
    get_bm25_corpus_count,
)

logger = logging.getLogger(__name__)


def save_chunks(
    chunks: list[Document],
    file_path: str,
) -> int:
    """Persist chunks to the BM25 corpus JSON file.

    Args:
        chunks: List of LangChain Document objects to store.
        file_path: Path to the JSON file.

    Returns:
        Number of chunks saved.
    """
    if not chunks:
        return 0

    corpus_data = [
        {"content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]

    return save_bm25_corpus(corpus_data, file_path)


def load_bm25_retriever(
    file_path: str,
    k: int = 5,
) -> BM25Retriever | None:
    """Load all chunks from JSON and build an in-memory BM25Retriever.

    Args:
        file_path: Path to the JSON file.
        k: Number of results to return per query.

    Returns:
        BM25Retriever instance, or None if no chunks found.
    """
    corpus_data = load_bm25_corpus(file_path)

    if not corpus_data:
        logger.warning("No chunks found in BM25 corpus")
        return None

    documents = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in corpus_data
    ]

    retriever = BM25Retriever.from_documents(documents, k=k)
    logger.info(f"Loaded BM25 retriever with {len(documents)} chunks (k={k})")
    return retriever


def clear_bm25_store(file_path: str) -> None:
    """Delete the BM25 corpus JSON file.

    Args:
        file_path: Path to the JSON file.
    """
    clear_bm25_corpus(file_path)
    logger.info("Cleared BM25 corpus")


def get_chunk_count(file_path: str) -> int:
    """Return the number of chunks in the BM25 corpus.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Number of chunks.
    """
    return get_bm25_corpus_count(file_path)
