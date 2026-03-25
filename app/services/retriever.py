"""Hybrid retrieval service using Dense (ChromaDB) + Sparse (BM25).

Combines vector similarity search with BM25 keyword matching via
EnsembleRetriever, with parent-child lookup for hierarchical chunks.
"""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.models.schemas import DocumentChunk
from app.services.parent_store import get_parent
from app.config import Settings

logger = logging.getLogger(__name__)


def build_ensemble_retriever(
    vector_store: Chroma | None,
    bm25_retriever: BM25Retriever | None,
    k: int = 5,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> EnsembleRetriever | None:
    """Build an EnsembleRetriever combining dense and sparse search.

    Falls back gracefully if either retriever is unavailable:
    - If both available: combines dense + sparse
    - If only dense: uses dense-only
    - If only sparse: uses sparse-only
    - If neither: returns None

    Args:
        vector_store: ChromaDB store for dense retrieval (can be None).
        bm25_retriever: BM25Retriever for keyword search (can be None).
        k: Number of results per retriever.
        bm25_weight: Weight for BM25 results in RRF.
        dense_weight: Weight for dense results in RRF.

    Returns:
        EnsembleRetriever instance, or None if nothing is available.
    """
    retrievers = []
    weights = []

    # Add BM25 if available
    if bm25_retriever is not None:
        retrievers.append(bm25_retriever)
        weights.append(bm25_weight)

    # Add dense if available
    if vector_store is not None:
        dense_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        retrievers.append(dense_retriever)
        weights.append(dense_weight)

    if not retrievers:
        logger.warning("No retrievers available (BM25 and ChromaDB both missing)")
        return None

    # Normalize weights if only one retriever
    if len(retrievers) == 1:
        weights = [1.0]
    else:
        # Re-normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]

    return EnsembleRetriever(retrievers=retrievers, weights=weights)


async def retrieve_documents(
    query: str,
    vector_store: Chroma | None,
    settings: Settings,
    bm25_retriever: BM25Retriever | None = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    parents_path: Optional[str] = None,
) -> list[DocumentChunk]:
    """Retrieve relevant document chunks using hybrid search.

    Uses EnsembleRetriever (BM25 + Dense) and resolves parent-child
    relationships for hierarchical chunks. Gracefully falls back if
    either retriever is unavailable.

    Args:
        query: The user query to search for.
        vector_store: ChromaDB store instance (can be None).
        settings: Application settings.
        bm25_retriever: Optional BM25Retriever for keyword search.
        top_k: Number of results to return.
        score_threshold: Minimum similarity score (used for dense-only fallback).
        parents_path: Path to parents JSON file for parent document lookups.

    Returns:
        List of DocumentChunk objects with content and metadata.
    """
    k = top_k or settings.retrieval_top_k
    parent_file = parents_path or settings.parents_path

    try:
        print(f"[DEBUG retriever] retrieve_documents called for query: {query[:50]}...")
        print(f"[DEBUG retriever] k={k}, parent_file={parent_file}")
        
        # Build ensemble retriever
        ensemble = build_ensemble_retriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            k=k,
        )

        if ensemble is None:
            print("[DEBUG retriever] No retriever available")
            logger.warning("No retriever available")
            return []

        # Retrieve documents
        print("[DEBUG retriever] Invoking ensemble retriever...")
        results = ensemble.invoke(query)
        print(f"[DEBUG retriever] Got {len(results)} raw results")

        # Resolve parent-child relationships
        resolved_results = _resolve_parents(results, parent_file)
        print(f"[DEBUG retriever] Resolved to {len(resolved_results)} results")

        # Deduplicate (parents may be returned multiple times if
        # multiple children from the same parent matched)
        seen_contents: set[str] = set()
        unique_results: list[Document] = []
        for doc in resolved_results:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(doc)

        # Convert to DocumentChunk
        chunks = [_document_to_chunk(doc) for doc in unique_results[:k]]

        print(f"[DEBUG retriever] Returning {len(chunks)} chunks")
        logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        return chunks

    except Exception as e:
        print(f"[DEBUG retriever] EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        print(f"[DEBUG retriever] Traceback: {traceback.format_exc()}")
        logger.error(f"Error retrieving documents: {e}")
        return []


def _resolve_parents(
    results: list[Document],
    parents_path: str,
) -> list[Document]:
    """Replace child chunks with their parent documents for full context.

    Args:
        results: Raw retrieval results.
        parents_path: Path to parents JSON file for lookups.

    Returns:
        Results with children replaced by their parents.
    """
    resolved: list[Document] = []

    for doc in results:
        metadata = doc.metadata or {}
        parent_id = metadata.get("parent_id")
        is_parent = metadata.get("is_parent", True)

        # If this is a child chunk, fetch the parent
        if parent_id and not is_parent:
            parent = get_parent(parent_id, parents_path)
            if parent:
                resolved.append(parent)
            else:
                # Parent not found, use the child as-is
                logger.warning(f"Parent {parent_id} not found, using child chunk")
                resolved.append(doc)
        else:
            resolved.append(doc)

    return resolved


def _document_to_chunk(doc: Document, score: float | None = None) -> DocumentChunk:
    """Convert a LangChain Document to a DocumentChunk.

    Args:
        doc: LangChain Document object.
        score: Optional similarity score.

    Returns:
        DocumentChunk with content and metadata.
    """
    metadata = doc.metadata or {}

    source = metadata.get("source", metadata.get("source_filename", "unknown"))
    page = metadata.get("page", metadata.get("page_number"))
    category = metadata.get("category")
    chunk_strategy = metadata.get("chunk_strategy")

    if page is not None:
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = None

    return DocumentChunk(
        content=doc.page_content,
        source=str(source),
        page=page,
        score=round(score, 4) if score is not None else None,
        category=category,
        chunk_strategy=chunk_strategy,
    )


def format_context_for_prompt(chunks: list[DocumentChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt.

    Args:
        chunks: List of DocumentChunk objects.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"Source: {chunk.source}"
        if chunk.page is not None:
            source_info += f", Page {chunk.page}"
        if chunk.category:
            source_info += f" | Category: {chunk.category}"

        context_parts.append(
            f"[Document {i}]\n{source_info}\n---\n{chunk.content}"
        )

    return "\n\n".join(context_parts)
