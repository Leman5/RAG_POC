"""In-memory session store for conversation history."""

import logging
import uuid

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGES = 10

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_or_create_history(session_id: str | None) -> tuple[str, InMemoryChatMessageHistory]:
    """Return an existing history or create a new one.

    If session_id is None, a new UUID is generated.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
        logger.info(f"Created new session: {session_id}")

    return session_id, _store[session_id]


def get_recent_messages(
    session_id: str,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
) -> list[BaseMessage]:
    """Return the last N messages for a session (empty list if unknown)."""
    history = _store.get(session_id)
    if not history:
        return []
    return history.messages[-max_messages:]


def add_exchange(
    session_id: str,
    user_msg: str,
    ai_msg: str,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
) -> None:
    """Append a user/AI exchange and trim the history to max_messages."""
    _, history = get_or_create_history(session_id)
    history.add_message(HumanMessage(content=user_msg))
    history.add_message(AIMessage(content=ai_msg))

    if len(history.messages) > max_messages:
        history.messages[:] = history.messages[-max_messages:]
