"""Chat router with RAG endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.dependencies import get_bm25_retriever, get_llm, get_router_llm, get_vector_store
from app.middleware.auth import verify_api_key
from app.models.schemas import ChatRequest, ChatResponse, DocumentChunk
from app.services.query_router import route_query
from app.services.retriever import retrieve_documents
from app.services.generator import generate_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="RAG Chat Endpoint",
    description="Send a query to the RAG system. The system will determine if document retrieval is needed and generate an appropriate response.",
)
async def chat(
    request: ChatRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[ChatOpenAI, Depends(get_llm)],
    router_llm: Annotated[ChatOpenAI, Depends(get_router_llm)],
    vector_store: Annotated[Chroma, Depends(get_vector_store)],
    bm25_retriever: Annotated[BM25Retriever | None, Depends(get_bm25_retriever)],
) -> ChatResponse:
    """Process a chat query using the RAG system.

    The flow:
    1. Query router classifies if the query needs document retrieval
    2. If needed, retrieve relevant document chunks via hybrid search
    3. Generate response using LLM (with or without context)
    4. Return response with source information

    Args:
        request: Chat request with user query
        api_key: Validated API key from middleware
        settings: Application settings
        llm: LLM for response generation
        router_llm: LLM for query routing
        vector_store: ChromaDB store for retrieval
        bm25_retriever: BM25 retriever for keyword search

    Returns:
        ChatResponse with answer, retrieval status, and sources
    """
    query = request.query.strip()
    logger.info(f"Processing query: {query[:50]}...")

    try:
        # Step 1: Route the query
        route_decision = await route_query(query, router_llm)
        logger.info(f"Route decision: needs_retrieval={route_decision.needs_retrieval}, reason={route_decision.reason}")

        sources: list[DocumentChunk] = []
        context_chunks: list[DocumentChunk] | None = None

        # Step 2: Retrieve documents if needed
        if route_decision.needs_retrieval:
            sources = await retrieve_documents(
                query=query,
                vector_store=vector_store,
                settings=settings,
                bm25_retriever=bm25_retriever,
                parents_path=settings.parents_path,
            )
            if sources:
                context_chunks = sources
                logger.info(f"Retrieved {len(sources)} relevant chunks")
            else:
                logger.info("No relevant chunks found above threshold")

        # Step 3: Generate response
        answer = await generate_response(
            query=query,
            llm=llm,
            context_chunks=context_chunks,
        )

        return ChatResponse(
            answer=answer,
            used_retrieval=route_decision.needs_retrieval and bool(sources),
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


@router.post(
    "/chat/direct",
    response_model=ChatResponse,
    summary="Direct Chat (No Retrieval)",
    description="Send a query directly to the LLM without document retrieval.",
)
async def chat_direct(
    request: ChatRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    llm: Annotated[ChatOpenAI, Depends(get_llm)],
) -> ChatResponse:
    """Process a chat query directly without retrieval.

    Useful for general conversation or when you know retrieval isn't needed.

    Args:
        request: Chat request with user query
        api_key: Validated API key from middleware
        llm: LLM for response generation

    Returns:
        ChatResponse with answer (no sources)
    """
    query = request.query.strip()
    logger.info(f"Processing direct query: {query[:50]}...")

    try:
        answer = await generate_response(
            query=query,
            llm=llm,
            context_chunks=None,
        )

        return ChatResponse(
            answer=answer,
            used_retrieval=False,
            sources=[],
        )

    except Exception as e:
        logger.error(f"Error processing direct chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


@router.post(
    "/chat/rag",
    response_model=ChatResponse,
    summary="Force RAG (Always Retrieve)",
    description="Send a query and always perform document retrieval regardless of classification.",
)
async def chat_rag(
    request: ChatRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[ChatOpenAI, Depends(get_llm)],
    vector_store: Annotated[Chroma, Depends(get_vector_store)],
    bm25_retriever: Annotated[BM25Retriever | None, Depends(get_bm25_retriever)],
) -> ChatResponse:
    """Process a chat query with forced document retrieval.

    Skips the query routing step and always retrieves documents.

    Args:
        request: Chat request with user query
        api_key: Validated API key from middleware
        settings: Application settings
        llm: LLM for response generation
        vector_store: ChromaDB store for retrieval
        bm25_retriever: BM25 retriever for keyword search

    Returns:
        ChatResponse with answer and sources
    """
    query = request.query.strip()
    logger.info(f"Processing forced RAG query: {query[:50]}...")

    try:
        # Always retrieve
        sources = await retrieve_documents(
            query=query,
            vector_store=vector_store,
            settings=settings,
            bm25_retriever=bm25_retriever,
            parents_path=settings.parents_path,
        )

        context_chunks = sources if sources else None
        logger.info(f"Retrieved {len(sources)} chunks for forced RAG")

        # Generate response
        answer = await generate_response(
            query=query,
            llm=llm,
            context_chunks=context_chunks,
        )

        return ChatResponse(
            answer=answer,
            used_retrieval=bool(sources),
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Error processing RAG chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )
