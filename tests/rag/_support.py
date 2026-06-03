# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared offline fakes for the RAG brain tests.

The brain talks only to three boundaries — the embedding client, the vector store,
and the LLM client. The embedding and LLM client builders live in ``tests.fakes`` and
inject a typed fake transport through the constructor; re-exported here for the RAG
tests. The vector store is a recording fake that returns canned results. Together they
let every RAG test run with no network and no database.
"""

from __future__ import annotations

from typing import Any

from squire.common.schemas import Document
from squire.common.vectorstore import SearchResult
from tests.fakes import make_embedder, make_llm

__all__ = ["RecordingVectorStore", "make_embedder", "make_llm", "result"]


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
