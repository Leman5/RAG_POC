"""Streamlit chat UI for RAG API."""

import os
import re
import time
import uuid
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

# Load .env from project root (RAG_POC/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_API_KEY = (
    os.getenv("API_KEYS", "").split(",")[0].strip()
    if os.getenv("API_KEYS")
    else ""
)


def call_chat_api(base_url: str, api_key: str, query: str, session_id: str, chat_history: list = None) -> dict | None:
    """POST to /api/v1/chat and return parsed JSON or None on failure."""
    url = f"{base_url.rstrip('/')}/api/v1/chat"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Build chat history without the current message
    history = []
    if chat_history:
        for msg in chat_history:
            if "role" in msg and "content" in msg:
                history.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "query": query,
        "session_id": session_id,
        "chat_history": history
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401:
            st.error("Invalid or missing API key.")
            return None
        if resp.status_code >= 500:
            detail = resp.json().get("detail", resp.text) if resp.text else "Unknown error"
            st.error(f"Server error: {detail}")
            return None
        st.error(f"Request failed: {resp.status_code} - {resp.text[:200]}")
        return None
    except httpx.ConnectError:
        st.error("Could not reach API. Is the server running?")
        return None
    except httpx.TimeoutException:
        st.error("Request timed out.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def _iter_text_deltas(text: str, *, tokens_per_update: int = 12, delay_s: float = 0.015):
    if not text:
        return

    tokens = re.findall(r"\s+|[^\s]+", text)
    buffer: list[str] = []
    for token in tokens:
        buffer.append(token)
        if len(buffer) < tokens_per_update:
            continue
        yield "".join(buffer)
        buffer.clear()
        time.sleep(delay_s)

    if buffer:
        yield "".join(buffer)


def main() -> None:
    st.set_page_config(page_title="Bottle Caps AI", page_icon="💬", layout="centered")
    st.title("Bottle-Caps AI Bot")

    # Sidebar config
    with st.sidebar:
        st.header("Settings")
        api_base_url = st.text_input(
            "API Base URL",
            value=DEFAULT_API_URL,
            help="Base URL of the RAG API (e.g. http://localhost:8000)",
        )
        api_key = st.text_input(
            "API Key",
            value=DEFAULT_API_KEY,
            type="password",
            help="X-API-Key for authentication",
        )

    # Session state for messages and session ID
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {src.get('source', 'Unknown')}**")
                        if src.get("page") is not None:
                            st.caption(f"Page {src['page']}")
                        if src.get("score") is not None:
                            st.caption(f"Score: {src['score']:.2f}")
                        st.text(src.get("content", "")[:300] + ("..." if len(src.get("content", "")) > 300 else ""))

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        if not api_key:
            st.error("Please set an API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_chat_api(
                    api_base_url,
                    api_key,
                    prompt,
                    st.session_state.session_id,
                    st.session_state.messages
                )

            if result:
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                used_retrieval = result.get("used_retrieval", False)
                if result.get("session_id"):
                    st.session_state.session_id = result["session_id"]

                rendered_answer = ""
                answer_placeholder = st.empty()
                for delta in _iter_text_deltas(answer):
                    rendered_answer += delta
                    answer_placeholder.markdown(rendered_answer)
                if not rendered_answer:
                    answer_placeholder.markdown(answer)
                    rendered_answer = answer

                if used_retrieval and sources:
                    with st.expander("Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. {src.get('source', 'Unknown')}**")
                            if src.get("page") is not None:
                                st.caption(f"Page {src['page']}")
                            if src.get("score") is not None:
                                st.caption(f"Score: {src['score']:.2f}")
                            content = src.get("content", "")
                            st.text(content[:300] + ("..." if len(content) > 300 else ""))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rendered_answer,
                        "sources": sources if used_retrieval else [],
                    }
                )


if __name__ == "__main__":
    main()
