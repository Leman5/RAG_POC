"""Tests for app/services/pdf_extractor.py - PDF extraction service."""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path


class TestExtractText:
    """Tests for the extract_text function."""

    def test_extract_text_file_not_found(self, temp_directory):
        """Test handling of non-existent PDF file."""
        from app.services.pdf_extractor import extract_text
        
        result = extract_text(os.path.join(temp_directory, "nonexistent.pdf"))
        
        assert result == {}

    @patch('app.services.pdf_extractor.pdfplumber')
    def test_extract_text_success(self, mock_pdfplumber, temp_directory):
        """Test successful text extraction."""
        from app.services.pdf_extractor import extract_text
        
        # Create a dummy PDF file
        pdf_path = os.path.join(temp_directory, "test.pdf")
        Path(pdf_path).touch()
        
        # Setup mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content here"
        mock_page.extract_tables.return_value = []
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        result = extract_text(pdf_path)
        
        assert 0 in result
        assert result[0] == "Page content here"

    @patch('app.services.pdf_extractor.pdfplumber')
    def test_extract_text_multiple_pages(self, mock_pdfplumber, temp_directory):
        """Test extraction from multiple pages."""
        from app.services.pdf_extractor import extract_text
        
        pdf_path = os.path.join(temp_directory, "multi.pdf")
        Path(pdf_path).touch()
        
        # Setup mock with multiple pages
        mock_pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"Page {i} content"
            mock_page.extract_tables.return_value = []
            mock_pages.append(mock_page)
        
        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        result = extract_text(pdf_path)
        
        assert len(result) == 3
        assert result[0] == "Page 0 content"
        assert result[1] == "Page 1 content"
        assert result[2] == "Page 2 content"

    @patch('app.services.pdf_extractor.pdfplumber')
    def test_extract_text_with_tables(self, mock_pdfplumber, temp_directory):
        """Test extraction includes table content."""
        from app.services.pdf_extractor import extract_text
        
        pdf_path = os.path.join(temp_directory, "table.pdf")
        Path(pdf_path).touch()
        
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Regular text"
        mock_page.extract_tables.return_value = [
            [["Header1", "Header2"], ["Cell1", "Cell2"]]
        ]
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        result = extract_text(pdf_path)
        
        assert "Header1 | Header2" in result[0]
        assert "Cell1 | Cell2" in result[0]

    @patch('app.services.pdf_extractor.pdfplumber')
    def test_extract_text_handles_exception(self, mock_pdfplumber, temp_directory):
        """Test graceful handling of extraction errors."""
        from app.services.pdf_extractor import extract_text
        
        pdf_path = os.path.join(temp_directory, "error.pdf")
        Path(pdf_path).touch()
        
        mock_pdfplumber.open.side_effect = Exception("PDF parsing error")
        
        result = extract_text(pdf_path)
        
        assert result == {}


class TestRenderPagesAsImages:
    """Tests for the render_pages_as_images function."""

    def test_render_pages_file_not_found(self, temp_directory):
        """Test handling of non-existent PDF file."""
        from app.services.pdf_extractor import render_pages_as_images
        
        result = render_pages_as_images(os.path.join(temp_directory, "nonexistent.pdf"))
        
        assert result == {}

    @patch('app.services.pdf_extractor.pdfium')
    def test_render_pages_success(self, mock_pdfium, temp_directory):
        """Test successful page rendering."""
        from app.services.pdf_extractor import render_pages_as_images
        from io import BytesIO
        from PIL import Image
        
        pdf_path = os.path.join(temp_directory, "render.pdf")
        Path(pdf_path).touch()
        
        # Create a mock bitmap that returns a PIL image
        mock_image = Image.new('RGB', (100, 100), color='white')
        
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = mock_image
        
        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap
        
        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=1)
        mock_pdf.get_page.return_value = mock_page
        
        mock_pdfium.PdfDocument.return_value = mock_pdf
        
        result = render_pages_as_images(pdf_path, dpi=150)
        
        assert 0 in result
        assert isinstance(result[0], bytes)

    @patch('app.services.pdf_extractor.pdfium')
    def test_render_pages_scale_calculation(self, mock_pdfium, temp_directory):
        """Test DPI to scale factor calculation."""
        from app.services.pdf_extractor import render_pages_as_images
        from PIL import Image
        
        pdf_path = os.path.join(temp_directory, "scale.pdf")
        Path(pdf_path).touch()
        
        mock_image = Image.new('RGB', (100, 100))
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = mock_image
        
        mock_page = MagicMock()
        mock_page.render.return_value = mock_bitmap
        
        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=1)
        mock_pdf.get_page.return_value = mock_page
        
        mock_pdfium.PdfDocument.return_value = mock_pdf
        
        # Test with DPI=144, should give scale=2.0
        render_pages_as_images(pdf_path, dpi=144)
        
        # Verify render was called with correct scale
        mock_page.render.assert_called_once()
        call_kwargs = mock_page.render.call_args.kwargs
        assert abs(call_kwargs['scale'] - 2.0) < 0.01

    @patch('app.services.pdf_extractor.pdfium')
    def test_render_pages_handles_exception(self, mock_pdfium, temp_directory):
        """Test graceful handling of rendering errors."""
        from app.services.pdf_extractor import render_pages_as_images
        
        pdf_path = os.path.join(temp_directory, "error.pdf")
        Path(pdf_path).touch()
        
        mock_pdfium.PdfDocument.side_effect = Exception("Rendering error")
        
        result = render_pages_as_images(pdf_path)
        
        assert result == {}


class TestFormatTable:
    """Tests for the _format_table helper function."""

    def test_format_table_basic(self):
        """Test basic table formatting."""
        from app.services.pdf_extractor import _format_table
        
        table = [
            ["Name", "Age"],
            ["Alice", "30"],
            ["Bob", "25"],
        ]
        
        result = _format_table(table)
        
        assert "Name | Age" in result
        assert "Alice | 30" in result
        assert "Bob | 25" in result

    def test_format_table_empty(self):
        """Test formatting empty table."""
        from app.services.pdf_extractor import _format_table
        
        result = _format_table([])
        
        assert result == ""

    def test_format_table_none_values(self):
        """Test formatting table with None values."""
        from app.services.pdf_extractor import _format_table
        
        table = [
            ["A", None, "C"],
            [None, "B", None],
        ]
        
        result = _format_table(table)
        
        assert "A |  | C" in result
        assert " | B | " in result

    def test_format_table_skips_empty_rows(self):
        """Test that completely empty rows are skipped."""
        from app.services.pdf_extractor import _format_table
        
        table = [
            ["Header"],
            [None],  # Empty row
            ["Data"],
        ]
        
        result = _format_table(table)
        lines = [l for l in result.split('\n') if l.strip()]
        
        assert len(lines) == 2  # Only Header and Data
