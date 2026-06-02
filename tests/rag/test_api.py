# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the RAG FastAPI service.

These exercise the endpoints through a real TestClient, but with the ``get_brain``
dependency overridden to a brain wired to fakes — so the service runs offline, with no
database and no model endpoint, exactly as Phase 2 requires. The brain itself is real
(retriever + generator + classifier), so these are end-to-end through the whole stack
below the HTTP layer.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from squire.rag import api
from squire.rag.brain import Brain
from tests.rag._support import RecordingVectorStore, make_embedder, make_llm, result


def _wire(brain: Brain) -> None:
    api.app.dependency_overrides[api.get_brain] = lambda: brain


def _brain(store: RecordingVectorStore, answer: str = "answer [1]") -> Brain:
    return Brain(embedder=make_embedder(), vector_store=store, llm=make_llm(answer))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_cited_answer(client: TestClient) -> None:
    """/query returns a grounded answer whose citations are the retrieved chunks."""
    store = RecordingVectorStore(
        [
            result(source_url="https://example.test/a"),
            result(source_url="https://example.test/b"),
        ]
    )
    _wire(_brain(store, answer="Do X then Y [1][2]."))

    response = client.post("/query", json={"question": "how do I do X?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["answer"] == "Do X then Y [1][2]."
    cited = [c["source_url"] for c in payload["citations"]]
    assert cited == ["https://example.test/a", "https://example.test/b"]


def test_query_citations_correspond_to_retrieved_chunks(client: TestClient) -> None:
    """Every citation the API returns is a chunk retrieval actually produced."""
    retrieved = ["https://example.test/x", "https://example.test/y"]
    store = RecordingVectorStore([result(source_url=u) for u in retrieved])
    _wire(_brain(store))

    payload = client.post("/query", json={"question": "q"}).json()

    assert {c["source_url"] for c in payload["citations"]} <= set(retrieved)
    assert [c["source_url"] for c in payload["citations"]] == retrieved


def test_query_source_type_filter_passes_through(client: TestClient) -> None:
    """A source_types filter in the request reaches the vector store."""
    store = RecordingVectorStore([result(source_url="u1")])
    _wire(_brain(store))

    client.post("/query", json={"question": "q", "source_types": ["docs", "pdf"]})

    assert store.searches[0]["source_types"] == ["docs", "pdf"]


def test_query_unsupported_says_so(client: TestClient) -> None:
    """When retrieval is empty the answer is a refusal with no citations."""
    _wire(_brain(RecordingVectorStore([])))

    payload = client.post("/query", json={"question": "obscure question"}).json()

    assert payload["grounded"] is False
    assert payload["citations"] == []
    assert "enough information" in payload["answer"].lower()


def test_draft_returns_cited_reply(client: TestClient) -> None:
    """/draft returns a grounded, cited suggested reply for an item."""
    store = RecordingVectorStore([result(source_url="https://example.test/doc")])
    _wire(_brain(store, answer="Try restarting [1]."))

    payload = client.post(
        "/draft", json={"title": "It crashes", "body": "on startup"}
    ).json()

    assert payload["grounded"] is True
    assert payload["answer"] == "Try restarting [1]."
    assert [c["source_url"] for c in payload["citations"]] == [
        "https://example.test/doc"
    ]


def test_classify_returns_classification(client: TestClient) -> None:
    """/classify returns a Classification via the brain's classifier."""
    _wire(_brain(RecordingVectorStore([])))

    payload = client.post(
        "/classify",
        json={
            "source_type": "github_issue",
            "source_id": "o/n#1",
            "source_url": "https://example.test/o/n/issues/1",
            "title": "Cannot log in",
            "body": "details",
            "content_hash": "h1",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    ).json()

    assert payload["suggested_title"] == "Cannot log in"
    assert payload["category"] == "uncategorized"
    assert payload["owner"] == "unassigned"


def test_classify_uses_injected_classifier(client: TestClient) -> None:
    """Swapping the brain's classifier changes /classify without touching the route."""
    from squire.common.schemas import Classification, Signal

    def custom(signal: Signal) -> Classification:
        return Classification(
            category="security",
            priority_key="p0",
            owner="security-team",
            suggested_title=signal.title,
        )

    store = RecordingVectorStore([])
    brain = Brain(
        embedder=make_embedder(),
        vector_store=store,
        llm=make_llm(),
        classifier=custom,
    )
    _wire(brain)

    payload = client.post(
        "/classify",
        json={
            "source_type": "forum_post",
            "source_id": "t/1",
            "source_url": "https://example.test/t/1",
            "title": "XSS report",
            "body": "...",
            "content_hash": "h2",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    ).json()

    assert payload["category"] == "security"
    assert payload["priority_key"] == "p0"
