"""JSON-backed parent document store.

Stores parent chunks from the parent-child chunking tier. Parents are
NOT embedded -- they are retrieved by ID when a matched child chunk
needs its full context.
"""

import logging

from langchain_core.documents import Document

from app.services.json_store import (
    save_parents as json_save_parents,
    load_parents as json_load_parents,
    get_parent_by_id,
    clear_parents as json_clear_parents,
    get_parent_count as json_get_parent_count,
)

logger = logging.getLogger(__name__)


def save_parents(
    parents: list[Document],
    file_path: str,
) -> int:
    """Persist parent documents to the parents JSON file.

    Each parent Document must have a 'parent_id' key in its metadata.

    Args:
        parents: List of parent Document objects.
        file_path: Path to the JSON file.

    Returns:
        Number of parents saved.
    """
    if not parents:
        return 0

    # Load existing parents to merge
    existing = json_load_parents(file_path)

    for parent in parents:
        parent_id = parent.metadata.get("parent_id")
        if not parent_id:
            logger.warning("Parent document missing 'parent_id' in metadata, skipping")
            continue

        existing[parent_id] = {
            "content": parent.page_content,
            "metadata": parent.metadata,
        }

    json_save_parents(existing, file_path)
    logger.info(f"Saved {len(parents)} parent documents")
    return len(parents)


def get_parent(
    parent_id: str,
    file_path: str,
) -> Document | None:
    """Fetch a parent document by its ID.

    Args:
        parent_id: UUID string identifying the parent.
        file_path: Path to the parents JSON file.

    Returns:
        Document with parent content and metadata, or None if not found.
    """
    data = get_parent_by_id(parent_id, file_path)

    if not data:
        logger.warning(f"Parent document not found: {parent_id}")
        return None

    return Document(page_content=data["content"], metadata=data["metadata"])


def clear_parents(file_path: str) -> None:
    """Delete the parents JSON file.

    Args:
        file_path: Path to the JSON file.
    """
    json_clear_parents(file_path)
    logger.info("Cleared parent documents")


def get_parent_count(file_path: str) -> int:
    """Return the number of parent documents stored.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Number of parents.
    """
    return json_get_parent_count(file_path)
