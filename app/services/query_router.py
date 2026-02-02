"""LLM-based query intent classification service."""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.schemas import RouteDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a RAG (Retrieval-Augmented Generation) system.
Your job is to determine whether a user's query requires searching a document knowledge base.

The document knowledge base contains information from uploaded PDF documents.

Analyze the user's query and determine:
1. If the query is asking about specific information, facts, or content that would be found in documents, return needs_retrieval: true
2. If the query is a general conversation, greeting, or a question that doesn't require document lookup, return needs_retrieval: false

Examples of queries that NEED retrieval (needs_retrieval: true):
- "What does the policy say about refunds?"
- "Explain the process described in the manual"
- "What are the key findings in the report?"
- "Summarize the main points from the documents"
- "What is mentioned about X in the files?"

Examples of queries that DON'T need retrieval (needs_retrieval: false):
- "Hello, how are you?"
- "What is 2 + 2?"
- "Tell me a joke"
- "What's the weather like?" (unless you have weather documents)
- "Thanks for your help"
- General knowledge questions not specific to the documents

You MUST respond with ONLY a valid JSON object in this exact format:
{"needs_retrieval": true/false, "reason": "brief explanation"}

Do not include any other text, markdown formatting, or code blocks. Just the raw JSON object."""


async def route_query(query: str, router_llm: ChatOpenAI) -> RouteDecision:
    """Classify whether a query needs document retrieval.

    Uses an LLM to analyze the query intent and determine if it requires
    searching the document knowledge base.

    Args:
        query: The user's query to classify
        router_llm: The LLM instance for routing decisions

    Returns:
        RouteDecision with needs_retrieval flag and reasoning
    """
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Classify this query: {query}"),
    ]

    try:
        response = await router_llm.ainvoke(messages)
        response_text = response.content.strip()

        # Try to parse the JSON response
        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = [
                line for line in lines
                if not line.startswith("```")
            ]
            response_text = "\n".join(json_lines).strip()

        decision_data = json.loads(response_text)

        return RouteDecision(
            needs_retrieval=decision_data.get("needs_retrieval", True),
            reason=decision_data.get("reason", "Classification completed"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse router response: {e}. Defaulting to retrieval.")
        return RouteDecision(
            needs_retrieval=True,
            reason="Failed to classify query, defaulting to document search",
        )

    except Exception as e:
        logger.error(f"Error in query routing: {e}. Defaulting to retrieval.")
        return RouteDecision(
            needs_retrieval=True,
            reason=f"Routing error: {str(e)}, defaulting to document search",
        )


def route_query_sync(query: str, router_llm: ChatOpenAI) -> RouteDecision:
    """Synchronous version of route_query for non-async contexts.

    Args:
        query: The user's query to classify
        router_llm: The LLM instance for routing decisions

    Returns:
        RouteDecision with needs_retrieval flag and reasoning
    """
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Classify this query: {query}"),
    ]

    try:
        response = router_llm.invoke(messages)
        response_text = response.content.strip()

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = [
                line for line in lines
                if not line.startswith("```")
            ]
            response_text = "\n".join(json_lines).strip()

        decision_data = json.loads(response_text)

        return RouteDecision(
            needs_retrieval=decision_data.get("needs_retrieval", True),
            reason=decision_data.get("reason", "Classification completed"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse router response: {e}. Defaulting to retrieval.")
        return RouteDecision(
            needs_retrieval=True,
            reason="Failed to classify query, defaulting to document search",
        )

    except Exception as e:
        logger.error(f"Error in query routing: {e}. Defaulting to retrieval.")
        return RouteDecision(
            needs_retrieval=True,
            reason=f"Routing error: {str(e)}, defaulting to document search",
        )
