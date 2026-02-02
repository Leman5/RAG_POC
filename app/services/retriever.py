"""Vector similarity search service using PGVector."""

import logging
from typing import Optional

from langchain_postgres import PGVector
from langchain_core.documents import Document

from app.models.schemas import DocumentChunk
from app.config import Settings

logger = logging.getLogger(__name__)


async def retrieve_documents(
    query: str,
    vector_store: PGVector,
    settings: Settings,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[DocumentChunk]:
    """Retrieve relevant document chunks based on query similarity.

    Args:
        query: The user query to search for
        vector_store: PGVector store instance
        settings: Application settings
        top_k: Number of results to return (defaults to settings.retrieval_top_k)
        score_threshold: Minimum similarity score (defaults to settings.similarity_threshold)

    Returns:
        List of DocumentChunk objects with content and metadata
    """
    k = top_k or settings.retrieval_top_k
    threshold = score_threshold or settings.similarity_threshold

    try:
        # Perform similarity search with scores
        results_with_scores = await vector_store.asimilarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        # Filter by score threshold and convert to DocumentChunk
        chunks = []
        for doc, score in results_with_scores:
            # Skip results below threshold
            if score < threshold:
                logger.debug(f"Skipping chunk with score {score:.3f} (below threshold {threshold})")
                continue

            chunk = _document_to_chunk(doc, score)
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def retrieve_documents_sync(
    query: str,
    vector_store: PGVector,
    settings: Settings,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[DocumentChunk]:
    """Synchronous version of retrieve_documents.

    Args:
        query: The user query to search for
        vector_store: PGVector store instance
        settings: Application settings
        top_k: Number of results to return (defaults to settings.retrieval_top_k)
        score_threshold: Minimum similarity score (defaults to settings.similarity_threshold)

    Returns:
        List of DocumentChunk objects with content and metadata
    """
    k = top_k or settings.retrieval_top_k
    threshold = score_threshold or settings.similarity_threshold

    try:
        # Perform similarity search with scores
        results_with_scores = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        # Filter by score threshold and convert to DocumentChunk
        chunks = []
        for doc, score in results_with_scores:
            # Skip results below threshold
            if score < threshold:
                logger.debug(f"Skipping chunk with score {score:.3f} (below threshold {threshold})")
                continue

            chunk = _document_to_chunk(doc, score)
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def _document_to_chunk(doc: Document, score: float) -> DocumentChunk:
    """Convert a LangChain Document to a DocumentChunk.

    Args:
        doc: LangChain Document object
        score: Similarity score

    Returns:
        DocumentChunk with content and metadata
    """
    metadata = doc.metadata or {}

    # Extract source - could be stored in different metadata fields
    source = metadata.get("source", metadata.get("file_path", "unknown"))

    # Extract page number if available
    page = metadata.get("page", metadata.get("page_number"))
    if page is not None:
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = None

    return DocumentChunk(
        content=doc.page_content,
        source=str(source),
        page=page,
        score=round(score, 4),
    )


def format_context_for_prompt(chunks: list[DocumentChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt.

    Args:
        chunks: List of DocumentChunk objects

    Returns:
        Formatted context string
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"Source: {chunk.source}"
        if chunk.page is not None:
            source_info += f", Page {chunk.page}"

        context_parts.append(
            f"[Document {i}]\n{source_info}\n---\n{chunk.content}"
        )

    return "\n\n".join(context_parts)
