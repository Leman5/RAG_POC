"""PDF text extraction and page rendering service.

Uses pdfplumber for text/table extraction and pypdfium2 for page rendering.
All libraries are permissively licensed (MIT/Apache-2.0/BSD).
"""

import io
import logging
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str | Path) -> dict[int, str]:
    """Extract text from each page of a PDF using pdfplumber.

    Handles tables by extracting them separately and formatting as readable text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary mapping page number (0-indexed) to extracted text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return {}

    page_texts: dict[int, str] = {}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                parts: list[str] = []

                # Extract tables first
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        formatted = _format_table(table)
                        if formatted:
                            parts.append(formatted)

                # Extract remaining text
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())

                page_text = "\n\n".join(parts) if parts else ""
                page_texts[page_num] = page_text

        logger.info(f"Extracted text from {len(page_texts)} pages of {pdf_path.name}")

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")

    return page_texts


def render_pages_as_images(
    pdf_path: str | Path,
    dpi: int = 150,
) -> dict[int, bytes]:
    """Render each page of a PDF as a PNG image using pypdfium2.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (higher = better quality but larger).
             PDFium uses scale factor: scale = dpi / 72.

    Returns:
        Dictionary mapping page number (0-indexed) to PNG image bytes.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return {}

    page_images: dict[int, bytes] = {}

    try:
        # Open PDF with pypdfium2
        pdf = pdfium.PdfDocument(str(pdf_path))
        
        # Calculate scale factor (pypdfium2 uses scale, not DPI directly)
        # Standard PDF is 72 DPI, so scale = target_dpi / 72
        scale = dpi / 72.0
        
        for page_num in range(len(pdf)):
            page = pdf.get_page(page_num)
            
            # Render page to bitmap, then convert to PIL Image
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            
            # Convert to PNG bytes
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            page_images[page_num] = buf.getvalue()
            
            page.close()
        
        pdf.close()
        
        logger.info(f"Rendered {len(page_images)} pages from {pdf_path.name} at {dpi} DPI")

    except Exception as e:
        logger.error(f"Error rendering pages from {pdf_path}: {e}")

    return page_images


def _format_table(table: list[list[str | None]]) -> str:
    """Format a pdfplumber table into readable text.

    Args:
        table: List of rows, each row is a list of cell values.

    Returns:
        Formatted table string, or empty string if table is empty.
    """
    if not table:
        return ""

    rows: list[str] = []
    for row in table:
        cells = [cell.strip() if cell else "" for cell in row]
        # Skip completely empty rows
        if not any(cells):
            continue
        rows.append(" | ".join(cells))

    if not rows:
        return ""

    return "\n".join(rows)
