# RAG POC - Retrieval-Augmented Generation System

A production-ready RAG system for Bottlecapps admin documentation, built with FastAPI, LangChain, ChromaDB, and hybrid retrieval (Dense + BM25).

## Architecture Overview

```
PDF Documents (52 files)
    │
    ├── pdfplumber (text + tables extraction)
    ├── pypdfium2 (page rendering)
    └── GPT-4o-mini Vision (screenshot descriptions)
            │
            ▼
    Text Cleaner (noise removal + merge)
            │
            ▼
    Tiered Chunking Classifier
    ├── Document-Level  (< 2K chars → 1 chunk)
    ├── Recursive Split  (2K-5K chars → overlapping chunks)
    └── Parent-Child     (5K+ chars / Q&A → hierarchical)
            │
            ▼
    ┌─────────────────────────────────┐
    │ File-based Storage              │
    │  ├── ChromaDB (dense embeddings)│
    │  ├── bm25_corpus.json (BM25)    │
    │  └── parents.json (parent docs) │
    └─────────────────────────────────┘
            │
            ▼
    Hybrid Retrieval (EnsembleRetriever)
    ├── BM25 Keyword Search  (weight: 0.4)
    └── Dense Vector Search  (weight: 0.6)
            │
    Parent-Child Resolution
            │
            ▼
    GPT-4o-mini (Response Generation)
```

## Prerequisites

- Python 3.11+
- OpenAI API key

No Docker required! All storage is file-based (ChromaDB + JSON files).

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual values:
#   - OPENAI_API_KEY (required)
#   - API_KEYS (at least one key for API authentication)
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest Documents

```bash
# Full ingestion with vision descriptions
python scripts/extract_and_ingest.py --clear

# Skip vision (faster, for testing)
python scripts/extract_and_ingest.py --clear --skip-vision

# Dry run (process but don't store)
python scripts/extract_and_ingest.py --dry-run
```

### 4. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint          | Description                                 |
|--------|-------------------|---------------------------------------------|
| POST   | `/api/v1/chat`     | Smart RAG (auto-routes query)               |
| POST   | `/api/v1/chat/rag` | Force retrieval for every query             |
| POST   | `/api/v1/chat/direct` | Direct LLM (no retrieval)                |
| GET    | `/health`          | Health check with DB status                 |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "How do I set up app banners?"}'
```

## Environment Variables

| Variable                  | Default                  | Description                          |
|---------------------------|--------------------------|--------------------------------------|
| `OPENAI_API_KEY`          | (required)               | OpenAI API key                       |
| `CHROMA_PERSIST_DIR`      | `./chroma_db`            | ChromaDB persistence directory       |
| `BM25_CORPUS_PATH`        | `./data/bm25_corpus.json`| BM25 corpus JSON file path           |
| `PARENTS_PATH`            | `./data/parents.json`    | Parent documents JSON file path      |
| `API_KEYS`                | (required)               | Comma-separated API keys             |
| `EMBEDDING_MODEL`         | `text-embedding-3-small` | OpenAI embedding model               |
| `LLM_MODEL`              | `gpt-4o-mini`            | LLM for response generation          |
| `ROUTER_MODEL`            | `gpt-3.5-turbo`          | LLM for query routing                |
| `VISION_MODEL`            | `gpt-4o-mini`            | Vision LLM for screenshots           |
| `VISION_DPI`              | `150`                    | DPI for PDF page rendering           |
| `CHUNK_SIZE_RECURSIVE`    | `1000`                   | Chunk size for recursive splitting   |
| `CHUNK_OVERLAP_RECURSIVE` | `200`                    | Overlap for recursive splitting      |
| `CHUNK_SIZE_CHILD`        | `400`                    | Child chunk size (parent-child)      |
| `CHUNK_OVERLAP_CHILD`     | `50`                     | Child overlap (parent-child)         |
| `DOC_LEVEL_MAX_CHARS`     | `2000`                   | Max chars for document-level tier    |
| `RETRIEVAL_TOP_K`         | `5`                      | Number of chunks to retrieve         |
| `SIMILARITY_THRESHOLD`    | `0.7`                    | Minimum similarity score             |

## Project Structure

```
RAG_POC/
├── app/
│   ├── config.py              # Settings from env vars
│   ├── dependencies.py        # FastAPI dependency injection
│   ├── main.py                # FastAPI app + lifespan
│   ├── middleware/
│   │   └── auth.py            # API key authentication
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic request/response models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── chat.py            # Chat endpoints
│   └── services/
│       ├── bm25_service.py    # JSON-backed BM25 index
│       ├── chunking.py        # Tiered chunking (doc-level/recursive/parent-child)
│       ├── generator.py       # LLM response generation
│       ├── json_store.py      # JSON persistence for BM25 + parents
│       ├── parent_store.py    # JSON-backed parent document store
│       ├── pdf_extractor.py   # pdfplumber text + pypdfium2 rendering
│       ├── query_router.py    # Query routing logic
│       ├── retriever.py       # Hybrid retrieval (BM25 + Dense)
│       ├── text_cleaner.py    # Noise removal + text merging
│       └── vision_describer.py # GPT-4o-mini vision descriptions
├── scripts/
│   └── extract_and_ingest.py  # Main ingestion pipeline
├── chroma_db/                 # ChromaDB persistence (created on ingestion)
├── data/                      # JSON storage (created on ingestion)
│   ├── bm25_corpus.json       # BM25 keyword index
│   └── parents.json           # Parent documents
├── documents/                 # PDF files (ONBOARDING/, SETUP/)
├── requirements.txt
├── .env.example
└── README.md
```

## Chunking Strategy Details

The system classifies each document and applies the optimal chunking strategy:

| Tier            | Trigger                          | Behavior                                |
|-----------------|----------------------------------|-----------------------------------------|
| Document-Level  | < 2,000 chars                    | Entire doc = 1 chunk                    |
| Recursive       | 2,000 - 5,000 chars              | Split with 1,000 char chunks, 200 overlap |
| Parent-Child    | > 5,000 chars OR 3+ Q&A markers | Parents (2K) stored separately; children (400) embedded |

## Retrieval Details

The hybrid retrieval system uses **Reciprocal Rank Fusion** to combine:

- **BM25** (weight 0.4): Keyword matching for exact terms, error codes, product names
- **Dense vectors** (weight 0.6): Semantic similarity for conceptual queries

For parent-child chunks, matched child chunks are swapped with their parent's full content before being passed to the LLM, ensuring complete context.
