"""Tests for app/services/json_store.py - JSON persistence layer."""

import os
import json
import pytest
from pathlib import Path


class TestBm25CorpusPersistence:
    """Tests for BM25 corpus save/load operations."""

    def test_save_bm25_corpus(self, temp_directory):
        """Test saving BM25 corpus to JSON file."""
        from app.services.json_store import save_bm25_corpus, load_bm25_corpus
        
        file_path = os.path.join(temp_directory, "bm25_corpus.json")
        chunks = [
            {"content": "First chunk", "metadata": {"source": "doc1.pdf"}},
            {"content": "Second chunk", "metadata": {"source": "doc2.pdf"}},
        ]
        
        count = save_bm25_corpus(chunks, file_path)
        
        assert count == 2
        assert os.path.exists(file_path)
        
        # Verify file contents
        with open(file_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[0]["content"] == "First chunk"

    def test_save_bm25_corpus_empty_list(self, temp_directory):
        """Test saving empty list returns 0."""
        from app.services.json_store import save_bm25_corpus
        
        file_path = os.path.join(temp_directory, "bm25_empty.json")
        count = save_bm25_corpus([], file_path)
        
        assert count == 0
        assert not os.path.exists(file_path)

    def test_save_bm25_corpus_creates_parent_dirs(self, temp_directory):
        """Test that save creates parent directories if needed."""
        from app.services.json_store import save_bm25_corpus
        
        file_path = os.path.join(temp_directory, "nested", "dir", "bm25.json")
        chunks = [{"content": "Test", "metadata": {}}]
        
        save_bm25_corpus(chunks, file_path)
        
        assert os.path.exists(file_path)

    def test_load_bm25_corpus(self, temp_directory):
        """Test loading BM25 corpus from JSON file."""
        from app.services.json_store import save_bm25_corpus, load_bm25_corpus
        
        file_path = os.path.join(temp_directory, "bm25_corpus.json")
        original_chunks = [
            {"content": "Chunk A", "metadata": {"page": 1}},
            {"content": "Chunk B", "metadata": {"page": 2}},
        ]
        
        save_bm25_corpus(original_chunks, file_path)
        loaded_chunks = load_bm25_corpus(file_path)
        
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0]["content"] == "Chunk A"
        assert loaded_chunks[1]["metadata"]["page"] == 2

    def test_load_bm25_corpus_file_not_found(self, temp_directory):
        """Test loading from non-existent file returns empty list."""
        from app.services.json_store import load_bm25_corpus
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        result = load_bm25_corpus(file_path)
        
        assert result == []

    def test_clear_bm25_corpus(self, temp_directory):
        """Test clearing BM25 corpus file."""
        from app.services.json_store import save_bm25_corpus, clear_bm25_corpus
        
        file_path = os.path.join(temp_directory, "bm25_corpus.json")
        save_bm25_corpus([{"content": "Test", "metadata": {}}], file_path)
        assert os.path.exists(file_path)
        
        clear_bm25_corpus(file_path)
        assert not os.path.exists(file_path)

    def test_clear_bm25_corpus_nonexistent(self, temp_directory):
        """Test clearing non-existent file doesn't raise error."""
        from app.services.json_store import clear_bm25_corpus
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        # Should not raise
        clear_bm25_corpus(file_path)

    def test_get_bm25_corpus_count(self, temp_directory):
        """Test getting chunk count from BM25 corpus."""
        from app.services.json_store import save_bm25_corpus, get_bm25_corpus_count
        
        file_path = os.path.join(temp_directory, "bm25_corpus.json")
        chunks = [
            {"content": f"Chunk {i}", "metadata": {}}
            for i in range(5)
        ]
        
        save_bm25_corpus(chunks, file_path)
        count = get_bm25_corpus_count(file_path)
        
        assert count == 5

    def test_get_bm25_corpus_count_empty_file(self, temp_directory):
        """Test getting count from non-existent file returns 0."""
        from app.services.json_store import get_bm25_corpus_count
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        count = get_bm25_corpus_count(file_path)
        
        assert count == 0


class TestParentsPersistence:
    """Tests for parent document save/load operations."""

    def test_save_parents(self, temp_directory):
        """Test saving parent documents to JSON file."""
        from app.services.json_store import save_parents, load_parents
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = {
            "parent-001": {"content": "Parent 1", "metadata": {"id": "parent-001"}},
            "parent-002": {"content": "Parent 2", "metadata": {"id": "parent-002"}},
        }
        
        count = save_parents(parents, file_path)
        
        assert count == 2
        assert os.path.exists(file_path)

    def test_save_parents_empty_dict(self, temp_directory):
        """Test saving empty dict returns 0."""
        from app.services.json_store import save_parents
        
        file_path = os.path.join(temp_directory, "parents_empty.json")
        count = save_parents({}, file_path)
        
        assert count == 0
        assert not os.path.exists(file_path)

    def test_load_parents(self, temp_directory):
        """Test loading parent documents from JSON file."""
        from app.services.json_store import save_parents, load_parents
        
        file_path = os.path.join(temp_directory, "parents.json")
        original_parents = {
            "id-1": {"content": "Content 1", "metadata": {}},
            "id-2": {"content": "Content 2", "metadata": {}},
        }
        
        save_parents(original_parents, file_path)
        loaded_parents = load_parents(file_path)
        
        assert len(loaded_parents) == 2
        assert loaded_parents["id-1"]["content"] == "Content 1"

    def test_load_parents_file_not_found(self, temp_directory):
        """Test loading from non-existent file returns empty dict."""
        from app.services.json_store import load_parents
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        result = load_parents(file_path)
        
        assert result == {}

    def test_get_parent_by_id(self, temp_directory):
        """Test retrieving a parent by ID."""
        from app.services.json_store import save_parents, get_parent_by_id
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = {
            "parent-001": {"content": "Parent One", "metadata": {"type": "guide"}},
            "parent-002": {"content": "Parent Two", "metadata": {"type": "tutorial"}},
        }
        
        save_parents(parents, file_path)
        
        result = get_parent_by_id("parent-001", file_path)
        
        assert result is not None
        assert result["content"] == "Parent One"
        assert result["metadata"]["type"] == "guide"

    def test_get_parent_by_id_not_found(self, temp_directory):
        """Test retrieving non-existent parent returns None."""
        from app.services.json_store import save_parents, get_parent_by_id
        
        file_path = os.path.join(temp_directory, "parents.json")
        save_parents({"parent-001": {"content": "Test", "metadata": {}}}, file_path)
        
        result = get_parent_by_id("nonexistent-id", file_path)
        
        assert result is None

    def test_clear_parents(self, temp_directory):
        """Test clearing parents file."""
        from app.services.json_store import save_parents, clear_parents
        
        file_path = os.path.join(temp_directory, "parents.json")
        save_parents({"id": {"content": "Test", "metadata": {}}}, file_path)
        assert os.path.exists(file_path)
        
        clear_parents(file_path)
        assert not os.path.exists(file_path)

    def test_get_parent_count(self, temp_directory):
        """Test getting parent count."""
        from app.services.json_store import save_parents, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = {f"parent-{i}": {"content": f"P{i}", "metadata": {}} for i in range(3)}
        
        save_parents(parents, file_path)
        count = get_parent_count(file_path)
        
        assert count == 3


class TestUnicodeHandling:
    """Tests for Unicode character handling in JSON storage."""

    def test_save_and_load_unicode_content(self, temp_directory):
        """Test that Unicode content is preserved."""
        from app.services.json_store import save_bm25_corpus, load_bm25_corpus
        
        file_path = os.path.join(temp_directory, "unicode.json")
        chunks = [
            {"content": "日本語テキスト", "metadata": {"lang": "ja"}},
            {"content": "Émoji test 🎉", "metadata": {"lang": "emoji"}},
            {"content": "Κείμενο στα ελληνικά", "metadata": {"lang": "el"}},
        ]
        
        save_bm25_corpus(chunks, file_path)
        loaded = load_bm25_corpus(file_path)
        
        assert loaded[0]["content"] == "日本語テキスト"
        assert loaded[1]["content"] == "Émoji test 🎉"
        assert loaded[2]["content"] == "Κείμενο στα ελληνικά"
