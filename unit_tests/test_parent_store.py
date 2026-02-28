"""Tests for app/services/parent_store.py - Parent document storage."""

import os
import json
import pytest
from langchain_core.documents import Document


class TestSaveParents:
    """Tests for the save_parents function."""

    def test_save_parents_to_file(self, temp_directory):
        """Test saving parent documents to JSON file."""
        from app.services.parent_store import save_parents, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [
            Document(
                page_content="Parent document one content",
                metadata={"parent_id": "p-001", "source": "doc1.pdf"},
            ),
            Document(
                page_content="Parent document two content",
                metadata={"parent_id": "p-002", "source": "doc2.pdf"},
            ),
        ]
        
        count = save_parents(parents, file_path)
        
        assert count == 2
        assert get_parent_count(file_path) == 2

    def test_save_parents_empty_list(self, temp_directory):
        """Test saving empty list returns 0."""
        from app.services.parent_store import save_parents
        
        file_path = os.path.join(temp_directory, "parents.json")
        count = save_parents([], file_path)
        
        assert count == 0

    def test_save_parents_skips_missing_id(self, temp_directory, caplog):
        """Test that parents without parent_id are skipped."""
        from app.services.parent_store import save_parents, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [
            Document(
                page_content="Valid parent",
                metadata={"parent_id": "valid-id"},
            ),
            Document(
                page_content="Invalid parent - no ID",
                metadata={"source": "doc.pdf"},  # Missing parent_id
            ),
        ]
        
        count = save_parents(parents, file_path)
        
        # Only the valid parent should be saved
        assert count == 2  # Returns input count but only valid ones stored
        assert get_parent_count(file_path) == 1

    def test_save_parents_merges_with_existing(self, temp_directory):
        """Test that new parents are merged with existing ones."""
        from app.services.parent_store import save_parents, get_parent, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        
        # Save first batch
        batch1 = [
            Document(page_content="Parent A", metadata={"parent_id": "a"}),
        ]
        save_parents(batch1, file_path)
        
        # Save second batch
        batch2 = [
            Document(page_content="Parent B", metadata={"parent_id": "b"}),
        ]
        save_parents(batch2, file_path)
        
        # Both should exist
        assert get_parent_count(file_path) == 2
        assert get_parent("a", file_path) is not None
        assert get_parent("b", file_path) is not None


class TestGetParent:
    """Tests for the get_parent function."""

    def test_get_parent_by_id(self, temp_directory):
        """Test retrieving a parent document by ID."""
        from app.services.parent_store import save_parents, get_parent
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [
            Document(
                page_content="This is the parent content",
                metadata={"parent_id": "target-id", "page": 1},
            ),
        ]
        
        save_parents(parents, file_path)
        parent = get_parent("target-id", file_path)
        
        assert parent is not None
        assert isinstance(parent, Document)
        assert parent.page_content == "This is the parent content"
        assert parent.metadata["parent_id"] == "target-id"

    def test_get_parent_not_found(self, temp_directory):
        """Test retrieving non-existent parent returns None."""
        from app.services.parent_store import save_parents, get_parent
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [
            Document(page_content="Test", metadata={"parent_id": "existing"}),
        ]
        
        save_parents(parents, file_path)
        parent = get_parent("nonexistent", file_path)
        
        assert parent is None

    def test_get_parent_from_empty_store(self, temp_directory):
        """Test retrieving from empty store returns None."""
        from app.services.parent_store import get_parent
        
        file_path = os.path.join(temp_directory, "empty.json")
        parent = get_parent("any-id", file_path)
        
        assert parent is None


class TestClearParents:
    """Tests for the clear_parents function."""

    def test_clear_parents(self, temp_directory):
        """Test clearing the parents file."""
        from app.services.parent_store import save_parents, clear_parents, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [Document(page_content="Test", metadata={"parent_id": "id"})]
        
        save_parents(parents, file_path)
        assert get_parent_count(file_path) == 1
        
        clear_parents(file_path)
        assert get_parent_count(file_path) == 0


class TestGetParentCount:
    """Tests for the get_parent_count function."""

    def test_get_parent_count_with_data(self, temp_directory):
        """Test getting count from populated store."""
        from app.services.parent_store import save_parents, get_parent_count
        
        file_path = os.path.join(temp_directory, "parents.json")
        parents = [
            Document(page_content=f"P{i}", metadata={"parent_id": f"id-{i}"})
            for i in range(4)
        ]
        
        save_parents(parents, file_path)
        count = get_parent_count(file_path)
        
        assert count == 4

    def test_get_parent_count_empty(self, temp_directory):
        """Test getting count from non-existent file."""
        from app.services.parent_store import get_parent_count
        
        file_path = os.path.join(temp_directory, "nonexistent.json")
        count = get_parent_count(file_path)
        
        assert count == 0
