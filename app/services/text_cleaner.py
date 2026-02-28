"""Text cleaning and merging service.

Strips page-boundary noise from extracted PDF text, merges with vision
descriptions, and injects document metadata headers.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Regex patterns for noise removal
PATTERNS_TO_REMOVE = [
    # Page markers like "-- 1 of 5 --"
    re.compile(r"--\s*\d+\s+of\s+\d+\s*--"),
    # Timestamp + title lines like "1/30/26, 11:03 PM    Document Title"
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s+.+"),
    # about:blank artifact lines like "about:blank    1/2"
    re.compile(r"about:blank\s+\d+/\d+"),
    # Repeated Zoho desk source URLs
    re.compile(r"https?://bottlecapps\.zohodesk\.com\S*"),
]


def clean_page_text(text: str) -> str:
    """Remove noise and artifacts from a single page's extracted text.

    Args:
        text: Raw extracted text from one PDF page.

    Returns:
        Cleaned text with noise removed.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip completely empty lines (will handle whitespace collapsing later)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Check if the entire line matches a noise pattern
        is_noise = False
        for pattern in PATTERNS_TO_REMOVE:
            if pattern.fullmatch(stripped):
                is_noise = True
                break

        if is_noise:
            continue

        # Remove inline URL occurrences (partial matches within a line)
        cleaned = PATTERNS_TO_REMOVE[3].sub("", stripped).strip()

        if cleaned:
            cleaned_lines.append(cleaned)

    # Collapse multiple consecutive blank lines into a single blank line
    result_lines: list[str] = []
    prev_blank = False
    for line in cleaned_lines:
        if not line:
            if not prev_blank:
                result_lines.append("")
            prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False

    return "\n".join(result_lines).strip()


def clean_and_merge(
    page_texts: dict[int, str],
    page_descriptions: dict[int, str],
    doc_title: str,
    folder_path: str,
) -> str:
    """Clean extracted text, merge with vision descriptions, and add metadata.

    Args:
        page_texts: Dictionary mapping page number to raw extracted text.
        page_descriptions: Dictionary mapping page number to vision descriptions.
        doc_title: Title derived from the PDF filename.
        folder_path: Category path derived from the folder structure.

    Returns:
        Full cleaned and merged document text with metadata header.
    """
    if not page_texts:
        return ""

    # Determine all page numbers
    all_pages = sorted(set(page_texts.keys()) | set(page_descriptions.keys()))

    page_parts: list[str] = []

    for page_num in all_pages:
        raw_text = page_texts.get(page_num, "")
        description = page_descriptions.get(page_num, "")

        # Clean the extracted text
        cleaned_text = clean_page_text(raw_text)

        parts: list[str] = []
        if cleaned_text:
            parts.append(cleaned_text)
        if description:
            parts.append(f"[Screenshot Description]: {description}")

        if parts:
            page_parts.append("\n\n".join(parts))

    # Join all pages with double newline
    body = "\n\n".join(page_parts)

    if not body.strip():
        logger.warning(f"No content extracted from document: {doc_title}")
        return ""

    # Prepend metadata header
    header = f"Document: {doc_title} | Category: {folder_path}"
    full_text = f"{header}\n\n{body}"

    logger.info(
        f"Cleaned and merged '{doc_title}': {len(full_text)} chars "
        f"from {len(page_texts)} text pages + {sum(1 for d in page_descriptions.values() if d)} descriptions"
    )

    return full_text


def derive_doc_title(pdf_path: str) -> str:
    """Derive a human-readable title from a PDF filename.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned title string.
    """
    from pathlib import Path

    name = Path(pdf_path).stem

    # Replace hyphens and underscores with spaces
    title = name.replace("-", " ").replace("_", " ")

    # Title-case the result
    title = title.title()

    return title


def derive_folder_path(pdf_path: str, documents_root: str) -> str:
    """Derive a category path from the PDF's position in the folder tree.

    Args:
        pdf_path: Full path to the PDF file.
        documents_root: Root documents directory path.

    Returns:
        Category string like "ONBOARDING > HELP CENTER".
    """
    from pathlib import Path

    pdf = Path(pdf_path)
    root = Path(documents_root)

    try:
        relative = pdf.parent.relative_to(root)
        parts = [p for p in relative.parts if p]
        if parts:
            return " > ".join(parts)
    except ValueError:
        pass

    return "General"
