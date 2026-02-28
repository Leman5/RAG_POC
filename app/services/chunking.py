"""Tiered document chunking service.

Classifies documents by content type and size, then applies the
appropriate chunking strategy:
  - Document-Level: short single-topic docs kept as one chunk
  - Recursive: medium tutorials split with overlap
  - Parent-Child: long multi-section / Q&A docs split hierarchically
"""

import logging
import re
import uuid
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document classifier
# ---------------------------------------------------------------------------

def classify_document(
    text: str,
    filename: str,
    max_doc_level_chars: int = 2000,
) -> Literal["document_level", "recursive", "parent_child"]:
    """Route a document to the appropriate chunking tier.

    Args:
        text: Full cleaned document text.
        filename: Source PDF filename (for logging).
        max_doc_level_chars: Maximum character count for document-level chunking.

    Returns:
        String identifying the chunking tier.
    """
    char_count = len(text)

    # Count Q&A markers
    qa_count = len(re.findall(r"(?:^|\n)\s*Q\s*:", text))

    # Count heading markers (Markdown-style from vision descriptions or structure)
    heading_count = len(re.findall(r"(?:^|\n)##\s+", text))

    # Count section-like breaks (triple newlines or [Screenshot Description] blocks)
    section_breaks = text.count("\n\n\n") + text.count("[Screenshot Description]")

    if qa_count >= 3:
        tier = "parent_child"
    elif char_count < max_doc_level_chars:
        tier = "document_level"
    elif char_count >= 5000 or heading_count >= 3 or section_breaks >= 5:
        tier = "parent_child"
    else:
        tier = "recursive"

    logger.info(
        f"Classified '{filename}': {tier} "
        f"(chars={char_count}, qa={qa_count}, headings={heading_count}, sections={section_breaks})"
    )
    return tier


# ---------------------------------------------------------------------------
# Tier 1: Document-level chunking
# ---------------------------------------------------------------------------

def chunk_document_level(text: str, metadata: dict) -> list[Document]:
    """Keep the entire document as a single chunk.

    Args:
        text: Full document text.
        metadata: Base metadata dict for the document.

    Returns:
        List containing a single Document.
    """
    doc_metadata = {
        **metadata,
        "chunk_strategy": "document_level",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    return [Document(page_content=text, metadata=doc_metadata)]


# ---------------------------------------------------------------------------
# Tier 2: Recursive character splitting
# ---------------------------------------------------------------------------

def chunk_recursive(
    text: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split document with RecursiveCharacterTextSplitter.

    Uses custom separators that respect screenshot description blocks
    and natural paragraph boundaries.

    Args:
        text: Full document text.
        metadata: Base metadata dict.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n[Screenshot Description]",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
        keep_separator=True,
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[metadata],
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "recursive"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    logger.info(f"Recursive split: {len(chunks)} chunks from '{metadata.get('source', 'unknown')}'")
    return chunks


# ---------------------------------------------------------------------------
# Tier 3: Parent-child hierarchical chunking
# ---------------------------------------------------------------------------

def chunk_parent_child(
    text: str,
    metadata: dict,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50,
) -> tuple[list[Document], list[Document]]:
    """Split document into parent sections and smaller child chunks.

    Parents are stored in a docstore (not embedded).
    Children are embedded and searched; matched children cause the
    full parent to be returned to the LLM.

    Args:
        text: Full document text.
        metadata: Base metadata dict.
        parent_chunk_size: Target size for parent sections.
        child_chunk_size: Target size for child chunks.
        child_chunk_overlap: Overlap between child chunks.

    Returns:
        Tuple of (parent_docs, child_docs).
    """
    # Split into parent sections
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
        separators=[
            "\nQ:",
            "\nQ: ",
            "\n## ",
            "\n### ",
            "\n\n[Screenshot Description]",
            "\n\n\n",
            "\n\n",
        ],
        keep_separator=True,
    )

    parent_texts = parent_splitter.split_text(text)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_docs: list[Document] = []
    child_docs: list[Document] = []

    for i, parent_text in enumerate(parent_texts):
        parent_id = str(uuid.uuid4())

        # Create parent document (stored in docstore, NOT embedded)
        parent_meta = {
            **metadata,
            "chunk_strategy": "parent_child",
            "parent_id": parent_id,
            "parent_index": i,
            "total_parents": len(parent_texts),
            "is_parent": True,
        }
        parent_docs.append(Document(page_content=parent_text, metadata=parent_meta))

        # Create child documents from this parent
        child_texts = child_splitter.split_text(parent_text)

        for j, child_text in enumerate(child_texts):
            child_meta = {
                **metadata,
                "chunk_strategy": "parent_child",
                "parent_id": parent_id,
                "child_index": j,
                "total_children_in_parent": len(child_texts),
                "is_parent": False,
            }
            child_docs.append(Document(page_content=child_text, metadata=child_meta))

    logger.info(
        f"Parent-child split: {len(parent_docs)} parents, {len(child_docs)} children "
        f"from '{metadata.get('source', 'unknown')}'"
    )
    return parent_docs, child_docs


# ---------------------------------------------------------------------------
# Master chunking function
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    metadata: dict,
    max_doc_level_chars: int = 2000,
    chunk_size_recursive: int = 1000,
    chunk_overlap_recursive: int = 200,
    chunk_size_child: int = 400,
    chunk_overlap_child: int = 50,
) -> tuple[list[Document], list[Document]]:
    """Classify and chunk a document using the appropriate tier.

    Args:
        text: Full cleaned document text.
        metadata: Base metadata dict (source, category, etc.).
        max_doc_level_chars: Threshold for document-level tier.
        chunk_size_recursive: Chunk size for recursive tier.
        chunk_overlap_recursive: Overlap for recursive tier.
        chunk_size_child: Child chunk size for parent-child tier.
        chunk_overlap_child: Child overlap for parent-child tier.

    Returns:
        Tuple of (chunks_to_embed, parent_docs_to_store).
        - chunks_to_embed: Documents to embed in ChromaDB and index in BM25.
        - parent_docs_to_store: Parent documents to store in docstore
          (empty list for document-level and recursive tiers).
    """
    filename = metadata.get("source", "unknown")
    tier = classify_document(text, filename, max_doc_level_chars)

    if tier == "document_level":
        chunks = chunk_document_level(text, metadata)
        return chunks, []

    elif tier == "recursive":
        chunks = chunk_recursive(
            text, metadata,
            chunk_size=chunk_size_recursive,
            chunk_overlap=chunk_overlap_recursive,
        )
        return chunks, []

    else:  # parent_child
        parent_docs, child_docs = chunk_parent_child(
            text, metadata,
            child_chunk_size=chunk_size_child,
            child_chunk_overlap=chunk_overlap_child,
        )
        return child_docs, parent_docs
