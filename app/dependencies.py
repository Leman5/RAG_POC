"""Dependency injection for vector store and LLM clients."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings, get_settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance."""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    """Get cached LLM instance for response generation."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.7,
    )


@lru_cache
def get_router_llm() -> ChatOpenAI:
    """Get cached LLM instance for query routing (faster, cheaper model)."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.router_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )


@lru_cache
def get_vector_store() -> Chroma:
    """Get cached ChromaDB vector store instance."""
    settings = get_settings()
    embeddings = get_embeddings()
    return Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_bm25_retriever(request: Request) -> BM25Retriever | None:
    """Get the BM25 retriever from app state (loaded on startup).

    Returns None if no chunks have been ingested yet.
    """
    return getattr(request.app.state, "bm25_retriever", None)


# FastAPI dependency types
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[OpenAIEmbeddings, Depends(get_embeddings)]
LLMDep = Annotated[ChatOpenAI, Depends(get_llm)]
RouterLLMDep = Annotated[ChatOpenAI, Depends(get_router_llm)]
VectorStoreDep = Annotated[Chroma, Depends(get_vector_store)]
BM25RetrieverDep = Annotated[BM25Retriever | None, Depends(get_bm25_retriever)]