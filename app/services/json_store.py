"""JSON-based persistence for BM25 corpus and parent documents.

File-based storage that replaces PostgreSQL tables, enabling the system
to run without Docker or external database dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_parent_dir(file_path: str) -> None:
    """Create parent directories if they don't exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_bm25_corpus(
    chunks: list[dict[str, Any]],
    file_path: str,
) -> int:
    """Save BM25 corpus (chunk content and metadata) to a JSON file.

    Args:
        chunks: List of dicts with 'content' and 'metadata' keys.
        file_path: Path to the JSON file.

    Returns:
        Number of chunks saved.
    """
    if not chunks:
        return 0

    _ensure_parent_dir(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(chunks)} chunks to {file_path}")
    return len(chunks)


def load_bm25_corpus(file_path: str) -> list[dict[str, Any]]:
    """Load BM25 corpus from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of dicts with 'content' and 'metadata' keys, or empty list if not found.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"BM25 corpus file not found: {file_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} chunks from {file_path}")
    return data


def clear_bm25_corpus(file_path: str) -> None:
    """Delete the BM25 corpus file.

    Args:
        file_path: Path to the JSON file.
    """
    path = Path(file_path)
    if path.exists():
        path.unlink()
        logger.info(f"Cleared BM25 corpus: {file_path}")


def get_bm25_corpus_count(file_path: str) -> int:
    """Return the number of chunks in the BM25 corpus.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Number of chunks.
    """
    path = Path(file_path)
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return len(data)


def save_parents(
    parents: dict[str, dict[str, Any]],
    file_path: str,
) -> int:
    """Save parent documents to a JSON file.

    Args:
        parents: Dict mapping parent_id to {'content': str, 'metadata': dict}.
        file_path: Path to the JSON file.

    Returns:
        Number of parents saved.
    """
    if not parents:
        return 0

    _ensure_parent_dir(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(parents, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(parents)} parent documents to {file_path}")
    return len(parents)


def load_parents(file_path: str) -> dict[str, dict[str, Any]]:
    """Load parent documents from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict mapping parent_id to {'content': str, 'metadata': dict}, or empty dict.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"Parents file not found: {file_path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} parent documents from {file_path}")
    return data


def get_parent_by_id(
    parent_id: str,
    file_path: str,
) -> dict[str, Any] | None:
    """Fetch a single parent document by ID.

    Args:
        parent_id: UUID string identifying the parent.
        file_path: Path to the parents JSON file.

    Returns:
        Dict with 'content' and 'metadata', or None if not found.
    """
    parents = load_parents(file_path)
    return parents.get(parent_id)


def clear_parents(file_path: str) -> None:
    """Delete the parents JSON file.

    Args:
        file_path: Path to the JSON file.
    """
    path = Path(file_path)
    if path.exists():
        path.unlink()
        logger.info(f"Cleared parents: {file_path}")


def get_parent_count(file_path: str) -> int:
    """Return the number of parent documents.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Number of parents.
    """
    path = Path(file_path)
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return len(data)
