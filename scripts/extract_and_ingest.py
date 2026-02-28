"""Document extraction and ingestion pipeline.

Replaces the old ingest_documents.py with a complete pipeline:
  1. Find all PDFs recursively
  2. Extract text (pdfplumber) + render pages as images (pypdfium2)
  3. Describe screenshots with GPT-4o-mini vision
  4. Clean and merge text + descriptions
  5. Classify and chunk (document-level / recursive / parent-child)
  6. Store in ChromaDB, BM25 JSON corpus, and parent document store
"""

import argparse
import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.services.pdf_extractor import extract_text, render_pages_as_images
from app.services.vision_describer import describe_all_pages
from app.services.text_cleaner import (
    clean_and_merge,
    derive_doc_title,
    derive_folder_path,
)
from app.services.chunking import chunk_document
from app.services.bm25_service import (
    save_chunks,
    clear_bm25_store,
    get_chunk_count,
)
from app.services.parent_store import (
    save_parents,
    clear_parents,
    get_parent_count,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_all_pdfs(documents_path: str) -> list[Path]:
    """Recursively find all PDF files under the documents directory.

    Args:
        documents_path: Root documents directory.

    Returns:
        Sorted list of PDF file paths.
    """
    root = Path(documents_path)
    if not root.exists():
        logger.error(f"Documents path does not exist: {documents_path}")
        return []

    pdfs = sorted(root.glob("**/*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF files under {documents_path}")
    return pdfs


async def process_single_pdf(
    pdf_path: Path,
    documents_root: str,
    settings,
    skip_vision: bool = False,
) -> tuple[list, list]:
    """Process one PDF through the full extraction pipeline.

    Args:
        pdf_path: Path to the PDF.
        documents_root: Root documents directory (for category derivation).
        settings: Application settings.
        skip_vision: If True, skip vision descriptions.

    Returns:
        Tuple of (chunks_to_embed, parent_docs).
    """
    doc_title = derive_doc_title(str(pdf_path))
    folder_path = derive_folder_path(str(pdf_path), documents_root)

    logger.info(f"Processing: {pdf_path.name} ({doc_title})")

    # Step 1: Extract text via pdfplumber
    page_texts = extract_text(pdf_path)
    if not page_texts:
        logger.warning(f"No text extracted from {pdf_path.name}, skipping")
        return [], []

    # Step 2: Render pages as images and describe screenshots
    page_descriptions: dict[int, str] = {}
    if not skip_vision:
        page_images = render_pages_as_images(pdf_path, dpi=settings.vision_dpi)
        if page_images:
            page_descriptions = await describe_all_pages(
                page_images,
                doc_title=doc_title,
                api_key=settings.openai_api_key,
                model=settings.vision_model,
            )

    # Step 3: Clean and merge
    full_text = clean_and_merge(page_texts, page_descriptions, doc_title, folder_path)
    if not full_text:
        logger.warning(f"No content after cleaning {pdf_path.name}, skipping")
        return [], []

    # Step 4: Chunk
    base_metadata = {
        "source": str(pdf_path),
        "source_filename": pdf_path.name,
        "document_title": doc_title,
        "category": folder_path,
    }

    chunks_to_embed, parent_docs = chunk_document(
        text=full_text,
        metadata=base_metadata,
        max_doc_level_chars=settings.doc_level_max_chars,
        chunk_size_recursive=settings.chunk_size_recursive,
        chunk_overlap_recursive=settings.chunk_overlap_recursive,
        chunk_size_child=settings.chunk_size_child,
        chunk_overlap_child=settings.chunk_overlap_child,
    )

    return chunks_to_embed, parent_docs


async def run_ingestion(
    clear_existing: bool = False,
    skip_vision: bool = False,
    dry_run: bool = False,
):
    """Run the full ingestion pipeline.

    Args:
        clear_existing: If True, wipe all existing data first.
        skip_vision: If True, skip vision screenshot descriptions.
        dry_run: If True, process and chunk but do not store.
    """
    settings = get_settings()
    start_time = time.time()

    print("=" * 70)
    print("RAG Document Ingestion Pipeline")
    print("=" * 70)
    print(f"  Documents path:   {settings.documents_path}")
    print(f"  Vision model:     {settings.vision_model}")
    print(f"  Embedding model:  {settings.embedding_model}")
    print(f"  Skip vision:      {skip_vision}")
    print(f"  Dry run:          {dry_run}")
    print(f"  Clear existing:   {clear_existing}")
    print()

    # Find all PDFs
    pdfs = find_all_pdfs(settings.documents_path)
    if not pdfs:
        print("No PDF files found. Exiting.")
        return

    print(f"Found {len(pdfs)} PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf.relative_to(Path(settings.documents_path))}")
    print()

    # Clear existing data if requested
    if clear_existing and not dry_run:
        print("Clearing existing data...")
        clear_bm25_store(settings.bm25_corpus_path)
        clear_parents(settings.parents_path)
        # ChromaDB collection will be deleted and recreated
        chroma_dir = Path(settings.chroma_persist_dir)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            print(f"  Deleted ChromaDB directory: {chroma_dir}")
        print("Existing data cleared.\n")

    # Process all PDFs
    all_embed_chunks = []
    all_parent_docs = []
    stats = {
        "document_level": 0,
        "recursive": 0,
        "parent_child_children": 0,
        "parent_child_parents": 0,
        "vision_pages_described": 0,
        "total_pages": 0,
    }

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Processing: {pdf_path.name}")

        chunks, parents = await process_single_pdf(
            pdf_path, settings.documents_path, settings, skip_vision
        )

        if chunks:
            all_embed_chunks.extend(chunks)
            all_parent_docs.extend(parents)

            # Track stats
            for chunk in chunks:
                strategy = chunk.metadata.get("chunk_strategy", "unknown")
                if strategy == "document_level":
                    stats["document_level"] += 1
                elif strategy == "recursive":
                    stats["recursive"] += 1
                elif strategy == "parent_child":
                    stats["parent_child_children"] += 1

            stats["parent_child_parents"] += len(parents)

        print(f"  -> {len(chunks)} chunks, {len(parents)} parents")

    print()
    print("-" * 70)
    print(f"Total chunks to embed:  {len(all_embed_chunks)}")
    print(f"Total parent documents: {len(all_parent_docs)}")
    print(f"Breakdown:")
    print(f"  Document-level chunks:    {stats['document_level']}")
    print(f"  Recursive chunks:         {stats['recursive']}")
    print(f"  Parent-child children:    {stats['parent_child_children']}")
    print(f"  Parent-child parents:     {stats['parent_child_parents']}")
    print()

    if dry_run:
        print("DRY RUN -- no data stored. Exiting.")
        return

    if not all_embed_chunks:
        print("No chunks to store. Exiting.")
        return

    # Store in ChromaDB
    print("Storing chunks in ChromaDB...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    Chroma.from_documents(
        documents=all_embed_chunks,
        embedding=embeddings,
        collection_name="documents",
        persist_directory=settings.chroma_persist_dir,
    )
    print(f"  Stored {len(all_embed_chunks)} chunks in ChromaDB")

    # Store in BM25 JSON corpus
    print("Storing chunks in BM25 corpus...")
    save_chunks(all_embed_chunks, settings.bm25_corpus_path)
    print(f"  BM25 corpus now has {get_chunk_count(settings.bm25_corpus_path)} chunks")

    # Store parent documents
    if all_parent_docs:
        print("Storing parent documents...")
        save_parents(all_parent_docs, settings.parents_path)
        print(f"  Parent store now has {get_parent_count(settings.parents_path)} parents")

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"Ingestion complete in {elapsed:.1f} seconds!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and ingest PDF documents into the RAG system"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all existing data before ingestion",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision LLM screenshot descriptions (faster, for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and chunk documents but do not store in database",
    )
    args = parser.parse_args()

    asyncio.run(
        run_ingestion(
            clear_existing=args.clear,
            skip_vision=args.skip_vision,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
