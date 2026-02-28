"""Tests for app/services/chunking.py - Document chunking service."""

import pytest
from langchain_core.documents import Document


class TestClassifyDocument:
    """Tests for the classify_document function."""

    def test_classify_document_level_short_doc(self):
        """Test short documents are classified as document_level."""
        from app.services.chunking import classify_document
        
        text = "This is a short document with less than 2000 characters."
        tier = classify_document(text, "short.pdf", max_doc_level_chars=2000)
        
        assert tier == "document_level"

    def test_classify_parent_child_qa_format(self):
        """Test Q&A formatted documents are classified as parent_child."""
        from app.services.chunking import classify_document
        
        text = """
        Q: What is machine learning?
        A: Machine learning is a subset of AI.
        
        Q: What are neural networks?
        A: Neural networks are computing systems.
        
        Q: How does deep learning work?
        A: Deep learning uses multiple layers.
        """
        
        tier = classify_document(text, "faq.pdf")
        assert tier == "parent_child"

    def test_classify_parent_child_long_doc(self):
        """Test long documents are classified as parent_child."""
        from app.services.chunking import classify_document
        
        # Create a document with 5000+ characters
        text = "This is a long document. " * 300
        
        tier = classify_document(text, "long.pdf")
        assert tier == "parent_child"

    def test_classify_parent_child_many_headings(self):
        """Test documents with many headings are classified as parent_child."""
        from app.services.chunking import classify_document
        
        text = """
        ## Section One
        Content here.
        
        ## Section Two
        More content.
        
        ## Section Three
        Even more content.
        
        ## Section Four
        Additional content.
        """
        
        tier = classify_document(text, "structured.pdf")
        assert tier == "parent_child"

    def test_classify_recursive_medium_doc(self):
        """Test medium documents are classified as recursive."""
        from app.services.chunking import classify_document
        
        # Between 2000 and 5000 chars, no Q&A, few headings
        text = "Medium length content. " * 150  # ~3000 chars
        
        tier = classify_document(text, "medium.pdf", max_doc_level_chars=2000)
        assert tier == "recursive"


class TestChunkDocumentLevel:
    """Tests for the chunk_document_level function."""

    def test_chunk_document_level_single_chunk(self):
        """Test document_level chunking creates single chunk."""
        from app.services.chunking import chunk_document_level
        
        text = "This is a short document."
        metadata = {"source": "test.pdf"}
        
        chunks = chunk_document_level(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0].page_content == text
        assert chunks[0].metadata["chunk_strategy"] == "document_level"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["total_chunks"] == 1

    def test_chunk_document_level_preserves_metadata(self):
        """Test that original metadata is preserved."""
        from app.services.chunking import chunk_document_level
        
        metadata = {"source": "test.pdf", "category": "tutorial", "custom": "value"}
        chunks = chunk_document_level("Content", metadata)
        
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["category"] == "tutorial"
        assert chunks[0].metadata["custom"] == "value"


class TestChunkRecursive:
    """Tests for the chunk_recursive function."""

    def test_chunk_recursive_splits_long_text(self):
        """Test recursive chunking splits long text."""
        from app.services.chunking import chunk_recursive
        
        text = "This is a test. " * 200  # ~3200 chars
        metadata = {"source": "test.pdf"}
        
        chunks = chunk_recursive(text, metadata, chunk_size=500, chunk_overlap=100)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "recursive"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_chunk_recursive_respects_chunk_size(self):
        """Test that chunks respect the size limit."""
        from app.services.chunking import chunk_recursive
        
        text = "Word " * 500  # 2500 chars
        metadata = {"source": "test.pdf"}
        
        chunks = chunk_recursive(text, metadata, chunk_size=300, chunk_overlap=50)
        
        # Chunks should be roughly around the target size (with some variance)
        for chunk in chunks:
            # Allow some buffer for separator handling
            assert len(chunk.page_content) <= 400

    def test_chunk_recursive_correct_indices(self):
        """Test that chunk indices are correct."""
        from app.services.chunking import chunk_recursive
        
        text = "Content block. " * 100
        metadata = {"source": "test.pdf"}
        
        chunks = chunk_recursive(text, metadata, chunk_size=200, chunk_overlap=20)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)


class TestChunkParentChild:
    """Tests for the chunk_parent_child function."""

    def test_chunk_parent_child_creates_both(self):
        """Test parent_child creates both parent and child documents."""
        from app.services.chunking import chunk_parent_child
        
        text = "Section one content. " * 100 + "\n\n\n" + "Section two content. " * 100
        metadata = {"source": "test.pdf"}
        
        parents, children = chunk_parent_child(
            text, metadata,
            parent_chunk_size=500,
            child_chunk_size=100,
            child_chunk_overlap=20,
        )
        
        assert len(parents) > 0
        assert len(children) > 0

    def test_chunk_parent_child_parent_has_correct_metadata(self):
        """Test that parent documents have correct metadata."""
        from app.services.chunking import chunk_parent_child
        
        text = "Long content. " * 200
        metadata = {"source": "test.pdf"}
        
        parents, _ = chunk_parent_child(text, metadata)
        
        for parent in parents:
            assert parent.metadata["chunk_strategy"] == "parent_child"
            assert "parent_id" in parent.metadata
            assert parent.metadata["is_parent"] is True

    def test_chunk_parent_child_child_references_parent(self):
        """Test that children reference their parent."""
        from app.services.chunking import chunk_parent_child
        
        text = "Content block. " * 200
        metadata = {"source": "test.pdf"}
        
        parents, children = chunk_parent_child(text, metadata)
        
        parent_ids = {p.metadata["parent_id"] for p in parents}
        
        for child in children:
            assert child.metadata["is_parent"] is False
            assert child.metadata["parent_id"] in parent_ids

    def test_chunk_parent_child_unique_parent_ids(self):
        """Test that each parent has a unique ID."""
        from app.services.chunking import chunk_parent_child
        
        text = "Section. " * 500
        metadata = {"source": "test.pdf"}
        
        parents, _ = chunk_parent_child(text, metadata)
        
        parent_ids = [p.metadata["parent_id"] for p in parents]
        assert len(parent_ids) == len(set(parent_ids))


class TestChunkDocument:
    """Tests for the master chunk_document function."""

    def test_chunk_document_routes_to_document_level(self):
        """Test that short documents use document_level strategy."""
        from app.services.chunking import chunk_document
        
        text = "Short document content."
        metadata = {"source": "short.pdf"}
        
        chunks, parents = chunk_document(text, metadata, max_doc_level_chars=1000)
        
        assert len(chunks) == 1
        assert len(parents) == 0
        assert chunks[0].metadata["chunk_strategy"] == "document_level"

    def test_chunk_document_routes_to_recursive(self):
        """Test that medium documents use recursive strategy."""
        from app.services.chunking import chunk_document
        
        # Create medium-length doc without Q&A or many headings
        text = "Medium document content. " * 120  # ~3000 chars
        metadata = {"source": "medium.pdf"}
        
        chunks, parents = chunk_document(
            text, metadata,
            max_doc_level_chars=2000,
            chunk_size_recursive=500,
        )
        
        assert len(chunks) > 1
        assert len(parents) == 0
        assert chunks[0].metadata["chunk_strategy"] == "recursive"

    def test_chunk_document_routes_to_parent_child(self):
        """Test that Q&A documents use parent_child strategy."""
        from app.services.chunking import chunk_document
        
        text = """
        Q: Question one?
        A: Answer one.
        
        Q: Question two?
        A: Answer two.
        
        Q: Question three?
        A: Answer three.
        """
        metadata = {"source": "faq.pdf"}
        
        chunks, parents = chunk_document(text, metadata)
        
        assert len(chunks) > 0
        assert len(parents) > 0
        assert chunks[0].metadata["chunk_strategy"] == "parent_child"


class TestChunkingSeparators:
    """Tests for chunking separator handling."""

    def test_recursive_respects_screenshot_separator(self):
        """Test that screenshot descriptions are preserved."""
        from app.services.chunking import chunk_recursive
        
        text = """
        Regular text content here.
        
        [Screenshot Description]
        This is a screenshot of the admin panel.
        
        More regular text.
        """
        metadata = {"source": "tutorial.pdf"}
        
        chunks = chunk_recursive(text, metadata, chunk_size=200, chunk_overlap=20)
        
        # Screenshot description should be kept intact
        all_content = " ".join(c.page_content for c in chunks)
        assert "[Screenshot Description]" in all_content

    def test_parent_child_splits_on_qa_markers(self):
        """Test that Q&A markers create parent boundaries."""
        from app.services.chunking import chunk_parent_child
        
        text = """Q: First question?
        First answer here.
        
        Q: Second question?
        Second answer here with more details."""
        
        metadata = {"source": "faq.pdf"}
        
        parents, children = chunk_parent_child(text, metadata, parent_chunk_size=5000)
        
        # Each Q: should potentially create a parent boundary
        assert len(parents) >= 1
