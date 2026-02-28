"""Tests for app/services/text_cleaner.py - Text cleaning and merging."""

import pytest


class TestCleanPageText:
    """Tests for the clean_page_text function."""

    def test_clean_page_text_removes_page_markers(self):
        """Test removal of page markers like '-- 1 of 5 --'."""
        from app.services.text_cleaner import clean_page_text
        
        text = """Some content here.
        -- 1 of 5 --
        More content.
        -- 2 of 5 --"""
        
        cleaned = clean_page_text(text)
        
        assert "-- 1 of 5 --" not in cleaned
        assert "-- 2 of 5 --" not in cleaned
        assert "Some content here" in cleaned
        assert "More content" in cleaned

    def test_clean_page_text_removes_timestamps(self):
        """Test removal of timestamp lines."""
        from app.services.text_cleaner import clean_page_text
        
        text = """1/30/26, 11:03 PM    Document Title
        Actual content here.
        12/25/25, 9:00 AM Another Title"""
        
        cleaned = clean_page_text(text)
        
        assert "1/30/26" not in cleaned
        assert "Actual content here" in cleaned

    def test_clean_page_text_removes_about_blank(self):
        """Test removal of about:blank artifacts."""
        from app.services.text_cleaner import clean_page_text
        
        text = """Content line one.
        about:blank    1/2
        Content line two."""
        
        cleaned = clean_page_text(text)
        
        assert "about:blank" not in cleaned
        assert "Content line one" in cleaned
        assert "Content line two" in cleaned

    def test_clean_page_text_removes_urls(self):
        """Test removal of specific URL patterns."""
        from app.services.text_cleaner import clean_page_text
        
        text = """Check out https://bottlecapps.zohodesk.com/portal/kb/article
        Regular content here.
        Another link: https://bottlecapps.zohodesk.com/something"""
        
        cleaned = clean_page_text(text)
        
        assert "zohodesk.com" not in cleaned
        assert "Regular content here" in cleaned

    def test_clean_page_text_collapses_blank_lines(self):
        """Test collapsing of multiple blank lines."""
        from app.services.text_cleaner import clean_page_text
        
        text = """Line one.


        
        
        Line two."""
        
        cleaned = clean_page_text(text)
        
        # Should not have multiple consecutive blank lines
        assert "\n\n\n" not in cleaned

    def test_clean_page_text_empty_input(self):
        """Test handling of empty input."""
        from app.services.text_cleaner import clean_page_text
        
        assert clean_page_text("") == ""
        assert clean_page_text(None) == ""

    def test_clean_page_text_preserves_content(self):
        """Test that actual content is preserved."""
        from app.services.text_cleaner import clean_page_text
        
        text = """Machine learning is a field of study.
        It uses algorithms to learn from data.
        Neural networks are a common approach."""
        
        cleaned = clean_page_text(text)
        
        assert "Machine learning" in cleaned
        assert "algorithms" in cleaned
        assert "Neural networks" in cleaned


class TestCleanAndMerge:
    """Tests for the clean_and_merge function."""

    def test_clean_and_merge_basic(self):
        """Test basic merging of page texts and descriptions."""
        from app.services.text_cleaner import clean_and_merge
        
        page_texts = {
            0: "Page one content.",
            1: "Page two content.",
        }
        page_descriptions = {
            0: "Screenshot of the dashboard.",
        }
        
        result = clean_and_merge(
            page_texts, page_descriptions,
            doc_title="Test Document",
            folder_path="tutorials",
        )
        
        assert "Test Document" in result
        assert "tutorials" in result
        assert "Page one content" in result
        assert "Page two content" in result
        assert "[Screenshot Description]" in result
        assert "dashboard" in result

    def test_clean_and_merge_adds_header(self):
        """Test that header with title and category is added."""
        from app.services.text_cleaner import clean_and_merge
        
        result = clean_and_merge(
            {0: "Content"},
            {},
            doc_title="My Document",
            folder_path="category/subcategory",
        )
        
        assert result.startswith("Document: My Document | Category: category/subcategory")

    def test_clean_and_merge_empty_input(self):
        """Test handling of empty page texts."""
        from app.services.text_cleaner import clean_and_merge
        
        result = clean_and_merge({}, {}, "Title", "Category")
        assert result == ""

    def test_clean_and_merge_cleans_text(self):
        """Test that page texts are cleaned before merging."""
        from app.services.text_cleaner import clean_and_merge
        
        page_texts = {
            0: "Content\n-- 1 of 3 --\nMore content",
        }
        
        result = clean_and_merge(page_texts, {}, "Doc", "Cat")
        
        assert "-- 1 of 3 --" not in result
        assert "Content" in result

    def test_clean_and_merge_handles_missing_pages(self):
        """Test handling when descriptions exist for pages without text."""
        from app.services.text_cleaner import clean_and_merge
        
        page_texts = {0: "Page 0 text"}
        page_descriptions = {
            0: "Description for page 0",
            1: "Description for page 1 (no text)",
        }
        
        result = clean_and_merge(page_texts, page_descriptions, "Doc", "Cat")
        
        # Both descriptions should be included
        assert "Description for page 0" in result
        assert "Description for page 1" in result


class TestDeriveDocTitle:
    """Tests for the derive_doc_title function."""

    def test_derive_doc_title_basic(self):
        """Test basic title derivation from filename."""
        from app.services.text_cleaner import derive_doc_title
        
        title = derive_doc_title("/path/to/my-document.pdf")
        assert title == "My Document"

    def test_derive_doc_title_underscores(self):
        """Test title derivation with underscores."""
        from app.services.text_cleaner import derive_doc_title
        
        title = derive_doc_title("getting_started_guide.pdf")
        assert title == "Getting Started Guide"

    def test_derive_doc_title_mixed(self):
        """Test title derivation with mixed separators."""
        from app.services.text_cleaner import derive_doc_title
        
        title = derive_doc_title("user-manual_v2.pdf")
        assert title == "User Manual V2"

    def test_derive_doc_title_windows_path(self):
        """Test title derivation from Windows path."""
        from app.services.text_cleaner import derive_doc_title
        
        title = derive_doc_title("C:\\Users\\docs\\my-file.pdf")
        assert title == "My File"


class TestDeriveFolderPath:
    """Tests for the derive_folder_path function."""

    def test_derive_folder_path_basic(self):
        """Test basic folder path derivation."""
        from app.services.text_cleaner import derive_folder_path
        
        path = derive_folder_path(
            "/docs/tutorials/beginner/file.pdf",
            "/docs",
        )
        
        assert path == "tutorials > beginner"

    def test_derive_folder_path_single_folder(self):
        """Test folder path with single folder."""
        from app.services.text_cleaner import derive_folder_path
        
        path = derive_folder_path(
            "/docs/guides/file.pdf",
            "/docs",
        )
        
        assert path == "guides"

    def test_derive_folder_path_root_level(self):
        """Test file directly in root returns 'General'."""
        from app.services.text_cleaner import derive_folder_path
        
        path = derive_folder_path(
            "/docs/file.pdf",
            "/docs",
        )
        
        assert path == "General"

    def test_derive_folder_path_non_relative(self):
        """Test non-relative path returns 'General'."""
        from app.services.text_cleaner import derive_folder_path
        
        path = derive_folder_path(
            "/other/path/file.pdf",
            "/docs",
        )
        
        assert path == "General"
