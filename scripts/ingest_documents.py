"""Document ingestion script for loading PDFs into PGVector."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


def load_pdfs(documents_path: str) -> list:
    """Load all PDF documents from the specified directory.

    Args:
        documents_path: Path to the directory containing PDF files

    Returns:
        List of loaded documents
    """
    path = Path(documents_path)

    if not path.exists():
        print(f"Error: Documents path '{documents_path}' does not exist.")
        return []

    # Check for PDF files
    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{documents_path}'")
        return []

    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    # Load all PDFs from the directory
    loader = DirectoryLoader(
        str(path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()
    print(f"\nLoaded {len(documents)} pages from PDF files.")
    return documents


def chunk_documents(documents: list, embeddings: OpenAIEmbeddings) -> list:
    """Split documents into semantic chunks.

    Args:
        documents: List of loaded documents
        embeddings: OpenAI embeddings instance for semantic chunking

    Returns:
        List of document chunks
    """
    if not documents:
        return []

    print("\nApplying semantic chunking...")

    # Use semantic chunker for better context preservation
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} semantic chunks.")

    # Display chunk statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    if chunk_lengths:
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        print(f"Chunk length statistics:")
        print(f"  - Average: {avg_length:.0f} characters")
        print(f"  - Min: {min_length} characters")
        print(f"  - Max: {max_length} characters")

    return chunks


def store_in_pgvector(chunks: list, embeddings: OpenAIEmbeddings, connection_string: str):
    """Store document chunks in PGVector.

    Args:
        chunks: List of document chunks
        embeddings: OpenAI embeddings instance
        connection_string: PostgreSQL connection string
    """
    if not chunks:
        print("No chunks to store.")
        return

    print("\nStoring chunks in PGVector...")
    print(f"This may take a while depending on the number of chunks...")

    # Create PGVector store and add documents
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="documents",
        connection=connection_string,
        use_jsonb=True,
        pre_delete_collection=False,  # Set to True to replace existing documents
    )

    print(f"Successfully stored {len(chunks)} chunks in PGVector!")
    return vector_store


def ingest_documents(clear_existing: bool = False):
    """Main ingestion function.

    Args:
        clear_existing: If True, clear existing documents before ingestion
    """
    settings = get_settings()

    print("=" * 60)
    print("Document Ingestion Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Documents path: {settings.documents_path}")
    print(f"  - Embedding model: {settings.embedding_model}")
    print(f"  - Database URL: {settings.database_url[:50]}...")
    print()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    # Load PDFs
    documents = load_pdfs(settings.documents_path)
    if not documents:
        print("\nNo documents to process. Exiting.")
        return

    # Chunk documents semantically
    chunks = chunk_documents(documents, embeddings)
    if not chunks:
        print("\nNo chunks created. Exiting.")
        return

    # Store in PGVector
    if clear_existing:
        print("\nClearing existing documents...")
        # We'll use pre_delete_collection=True in store_in_pgvector
        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="documents",
            connection=settings.database_url,
            use_jsonb=True,
            pre_delete_collection=True,
        )
        print(f"Replaced collection with {len(chunks)} new chunks!")
    else:
        store_in_pgvector(chunks, embeddings, settings.database_url)

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into PGVector for RAG"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before ingestion",
    )
    args = parser.parse_args()

    ingest_documents(clear_existing=args.clear)
