"""Response generation service with context-aware and direct modes."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.schemas import DocumentChunk
from app.services.retriever import format_context_for_prompt

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
1. Use the provided document context to answer the user's question accurately.
2. If the context contains relevant information, synthesize it into a clear, comprehensive answer.
3. If the context doesn't contain enough information to fully answer the question, acknowledge what you found and indicate what's missing.
4. Always cite your sources when possible (e.g., "According to [Document 1]...").
5. Be concise but thorough in your responses.
6. If the documents contradict each other, acknowledge this and present both perspectives.

Context from documents:
{context}

Answer the user's question based on the above context."""

DIRECT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question directly and conversationally.

Note: This question doesn't require searching the document knowledge base, so respond based on your general knowledge and conversational abilities.

Keep your responses helpful, accurate, and concise."""


async def generate_response(
    query: str,
    llm: ChatOpenAI,
    context_chunks: list[DocumentChunk] | None = None,
) -> str:
    """Generate a response using the LLM.

    Uses context-aware generation if chunks are provided, otherwise uses direct mode.

    Args:
        query: The user's query
        llm: ChatOpenAI instance for generation
        context_chunks: Optional list of retrieved document chunks

    Returns:
        Generated response string
    """
    if context_chunks:
        return await _generate_with_context(query, llm, context_chunks)
    else:
        return await _generate_direct(query, llm)


async def _generate_with_context(
    query: str,
    llm: ChatOpenAI,
    chunks: list[DocumentChunk],
) -> str:
    """Generate a response using retrieved document context.

    Args:
        query: The user's query
        llm: ChatOpenAI instance
        chunks: Retrieved document chunks

    Returns:
        Generated response string
    """
    context = format_context_for_prompt(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    try:
        response = await llm.ainvoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating response with context: {e}")
        return f"I encountered an error while generating a response. Please try again."


async def _generate_direct(
    query: str,
    llm: ChatOpenAI,
) -> str:
    """Generate a direct response without document context.

    Args:
        query: The user's query
        llm: ChatOpenAI instance

    Returns:
        Generated response string
    """
    messages = [
        SystemMessage(content=DIRECT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        response = await llm.ainvoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating direct response: {e}")
        return f"I encountered an error while generating a response. Please try again."


def generate_response_sync(
    query: str,
    llm: ChatOpenAI,
    context_chunks: list[DocumentChunk] | None = None,
) -> str:
    """Synchronous version of generate_response.

    Args:
        query: The user's query
        llm: ChatOpenAI instance for generation
        context_chunks: Optional list of retrieved document chunks

    Returns:
        Generated response string
    """
    if context_chunks:
        return _generate_with_context_sync(query, llm, context_chunks)
    else:
        return _generate_direct_sync(query, llm)


def _generate_with_context_sync(
    query: str,
    llm: ChatOpenAI,
    chunks: list[DocumentChunk],
) -> str:
    """Synchronous version of context-aware generation."""
    context = format_context_for_prompt(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating response with context: {e}")
        return f"I encountered an error while generating a response. Please try again."


def _generate_direct_sync(
    query: str,
    llm: ChatOpenAI,
) -> str:
    """Synchronous version of direct generation."""
    messages = [
        SystemMessage(content=DIRECT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating direct response: {e}")
        return f"I encountered an error while generating a response. Please try again."
