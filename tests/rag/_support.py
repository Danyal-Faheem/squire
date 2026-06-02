# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared offline fakes for the RAG brain tests.

The brain talks only to three boundaries — the embedding client, the vector store,
and the LLM client — so these helpers fake exactly those. The embedding and LLM
clients are real instances with their transport swapped out (the idiom the Phase 0
client tests established), and the vector store is a recording fake that returns
canned results. Together they let every RAG test run with no network and no database.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from squire.common.config import Settings
from squire.common.embeddings import EmbeddingClient
from squire.common.llm import LLMClient
from squire.common.schemas import Document
from squire.common.vectorstore import SearchResult


def make_embedder(dim: int = 4, vector: list[float] | None = None) -> EmbeddingClient:
    """A real EmbeddingClient whose transport returns a fixed vector for any input."""
    fixed = vector if vector is not None else [0.1] * dim
    client = EmbeddingClient(Settings(embedding_dim=dim))

    def _create(*, model: str, input: list[str]) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(fixed)) for _ in input]
        )

    client._client = SimpleNamespace(  # type: ignore[assignment]
        embeddings=SimpleNamespace(create=_create)
    )
    return client


def make_llm(
    answer: str = "a grounded answer [1]", captured: list[Any] | None = None
) -> LLMClient:
    """A real LLMClient whose transport returns ``answer`` and records its calls.

    Pass ``captured`` to collect the keyword arguments of each call (notably the
    ``messages``), so a test can assert what prompt the generator constructed and
    whether the model was invoked at all.
    """
    client = LLMClient(Settings(generation_model="test-model"))

    def _create(**kwargs: Any) -> Any:
        if captured is not None:
            captured.append(kwargs)
        message = SimpleNamespace(content=answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    return client


class RecordingVectorStore:
    """A VectorStore that returns canned search results and records its queries."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.searches: list[dict[str, Any]] = []

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        return len(documents)

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        self.searches.append(
            {"embedding": embedding, "limit": limit, "source_types": source_types}
        )
        return self._results


def result(
    *,
    source_url: str,
    content: str = "some content",
    source_type: str = "docs",
    origin: str | None = "origin",
    score: float = 0.9,
) -> SearchResult:
    """Build a SearchResult for use as a canned retrieval hit."""
    return SearchResult(
        source_type=source_type,
        source_url=source_url,
        origin=origin,
        content=content,
        content_hash=f"hash-{source_url}",
        score=score,
    )
