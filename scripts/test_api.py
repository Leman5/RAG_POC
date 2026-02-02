"""Test script for the RAG API endpoints."""

import httpx
import argparse
import json


def test_health(base_url: str):
    """Test the health endpoint."""
    print("\n--- Testing Health Endpoint ---")
    response = httpx.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_root(base_url: str):
    """Test the root endpoint."""
    print("\n--- Testing Root Endpoint ---")
    response = httpx.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_chat_without_auth(base_url: str):
    """Test chat endpoint without API key (should fail)."""
    print("\n--- Testing Chat Without Auth (should fail) ---")
    response = httpx.post(
        f"{base_url}/api/v1/chat",
        json={"query": "Hello"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    return response.status_code == 401


def test_chat_with_invalid_key(base_url: str):
    """Test chat endpoint with invalid API key (should fail)."""
    print("\n--- Testing Chat With Invalid Key (should fail) ---")
    response = httpx.post(
        f"{base_url}/api/v1/chat",
        json={"query": "Hello"},
        headers={"X-API-Key": "invalid-key-12345"},
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    return response.status_code == 401


def test_chat_general_query(base_url: str, api_key: str):
    """Test chat with a general query (should NOT use retrieval)."""
    print("\n--- Testing General Query (no retrieval expected) ---")
    response = httpx.post(
        f"{base_url}/api/v1/chat",
        json={"query": "Hello! How are you doing today?"},
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Used Retrieval: {data['used_retrieval']}")
        print(f"Answer: {data['answer'][:200]}...")
        print(f"Sources: {len(data['sources'])} documents")
    else:
        print(f"Response: {response.text}")
    return response.status_code == 200


def test_chat_document_query(base_url: str, api_key: str, query: str):
    """Test chat with a document-related query (should use retrieval)."""
    print(f"\n--- Testing Document Query ---")
    print(f"Query: {query}")
    response = httpx.post(
        f"{base_url}/api/v1/chat",
        json={"query": query},
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Used Retrieval: {data['used_retrieval']}")
        print(f"Answer: {data['answer'][:300]}...")
        print(f"Sources: {len(data['sources'])} documents")
        if data['sources']:
            print("Source files:")
            for src in data['sources']:
                print(f"  - {src['source']} (page {src['page']}, score: {src['score']})")
    else:
        print(f"Response: {response.text}")
    return response.status_code == 200


def test_direct_endpoint(base_url: str, api_key: str):
    """Test the direct chat endpoint (no retrieval)."""
    print("\n--- Testing Direct Endpoint ---")
    response = httpx.post(
        f"{base_url}/api/v1/chat/direct",
        json={"query": "What is Python?"},
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Used Retrieval: {data['used_retrieval']}")
        print(f"Answer: {data['answer'][:200]}...")
    else:
        print(f"Response: {response.text}")
    return response.status_code == 200


def test_rag_endpoint(base_url: str, api_key: str, query: str):
    """Test the forced RAG endpoint (always retrieves)."""
    print("\n--- Testing Forced RAG Endpoint ---")
    response = httpx.post(
        f"{base_url}/api/v1/chat/rag",
        json={"query": query},
        headers={"X-API-Key": api_key},
        timeout=60.0,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Used Retrieval: {data['used_retrieval']}")
        print(f"Answer: {data['answer'][:200]}...")
        print(f"Sources: {len(data['sources'])} documents")
    else:
        print(f"Response: {response.text}")
    return response.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Test RAG API endpoints")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication",
    )
    parser.add_argument(
        "--query",
        default="What information is in the documents?",
        help="Query for document-related tests",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG API Test Suite")
    print("=" * 60)
    print(f"Base URL: {args.url}")
    print(f"API Key: {args.api_key[:8]}...")

    results = []

    # Basic endpoint tests
    results.append(("Health Check", test_health(args.url)))
    results.append(("Root Endpoint", test_root(args.url)))

    # Auth tests
    results.append(("Chat Without Auth", test_chat_without_auth(args.url)))
    results.append(("Chat Invalid Key", test_chat_with_invalid_key(args.url)))

    # Chat tests (require valid API key and running services)
    try:
        results.append(("General Query", test_chat_general_query(args.url, args.api_key)))
        results.append(("Document Query", test_chat_document_query(args.url, args.api_key, args.query)))
        results.append(("Direct Endpoint", test_direct_endpoint(args.url, args.api_key)))
        results.append(("Forced RAG", test_rag_endpoint(args.url, args.api_key, args.query)))
    except httpx.ConnectError:
        print("\nWarning: Could not connect to API. Make sure the server is running.")

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
