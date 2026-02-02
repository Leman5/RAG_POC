"""Dependency injection for database and LLM clients."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import Settings, get_settings


@lru_cache
def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance."""
    if settings is None:
        settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


@lru_cache
def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Get cached LLM instance for response generation."""
    if settings is None:
        settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.7,
    )


@lru_cache
def get_router_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Get cached LLM instance for query routing (faster, cheaper model)."""
    if settings is None:
        settings = get_settings()
    return ChatOpenAI(
        model=settings.router_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )


@lru_cache
def get_vector_store(settings: Settings | None = None) -> PGVector:
    """Get cached PGVector store instance."""
    if settings is None:
        settings = get_settings()
    embeddings = get_embeddings(settings)
    return PGVector(
        embeddings=embeddings,
        collection_name="documents",
        connection=settings.database_url,
        use_jsonb=True,
    )


# FastAPI dependency types
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[OpenAIEmbeddings, Depends(get_embeddings)]
LLMDep = Annotated[ChatOpenAI, Depends(get_llm)]
RouterLLMDep = Annotated[ChatOpenAI, Depends(get_router_llm)]
VectorStoreDep = Annotated[PGVector, Depends(get_vector_store)]
