# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for squire.common.vectorstore.PgVectorStore.

These use a fake SQLAlchemy session that records the statements and parameters, so we
assert on the store's contract (what it asks the database to do, how it applies the
source-type filter, how it maps rows back) without a real database. The data-layer
integration against a real Postgres lives in an @integration test added later.
"""

from __future__ import annotations

from typing import Any

from squire.common.schemas import Document
from squire.common.vectorstore import PgVectorStore


class _FakeResult:
    def __init__(self, rowcount: int = 0, rows: list[dict[str, Any]] | None = None):
        self.rowcount = rowcount
        self._rows = rows or []

    def mappings(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeSession:
    """Records executed statements and returns queued results in order."""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, statement: Any, params: dict[str, Any]) -> _FakeResult:
        self.calls.append((str(statement), params))
        return self._results.pop(0)


def _doc(content: str, content_hash: str) -> Document:
    return Document(
        source_type="github_code",
        source_url="https://example/x",
        origin="repo",
        content=content,
        content_hash=content_hash,
    )


def test_upsert_counts_written_rows() -> None:
    """upsert should sum rowcounts across inserts (ON CONFLICT yields 0)."""
    session = _FakeSession([_FakeResult(rowcount=1), _FakeResult(rowcount=0)])
    store = PgVectorStore(session)  # type: ignore[arg-type]
    written = store.upsert(
        [
            (_doc("a", "hash-a"), [0.0, 0.0, 0.0, 0.0]),
            (_doc("b", "hash-b"), [0.1, 0.1, 0.1, 0.1]),
        ]
    )
    assert written == 1
    assert len(session.calls) == 2
    # The insert must be idempotent on content_hash.
    assert "ON CONFLICT (content_hash) DO NOTHING" in session.calls[0][0]


def test_search_without_filter_has_no_where_clause() -> None:
    """An unfiltered search must not emit a source_type filter."""
    session = _FakeSession([_FakeResult(rows=[])])
    store = PgVectorStore(session)  # type: ignore[arg-type]
    store.search([0.0, 0.0, 0.0, 0.0], limit=3)
    statement, params = session.calls[0]
    assert "source_type = ANY" not in statement
    assert params["limit"] == 3


def test_search_with_filter_passes_source_types() -> None:
    """A filtered search must constrain to the requested source types."""
    session = _FakeSession([_FakeResult(rows=[])])
    store = PgVectorStore(session)  # type: ignore[arg-type]
    store.search(
        [0.0, 0.0, 0.0, 0.0],
        source_types=["github_issue", "forum_post"],
    )
    statement, params = session.calls[0]
    assert "source_type = ANY(:types)" in statement
    assert params["types"] == ["github_issue", "forum_post"]


def test_search_maps_rows_to_results() -> None:
    """Returned rows should map into SearchResult objects with their score."""
    row = {
        "source_type": "docs",
        "source_url": "https://docs/x",
        "origin": "docs",
        "content": "hello",
        "content_hash": "h1",
        "score": 0.875,
    }
    session = _FakeSession([_FakeResult(rows=[row])])
    store = PgVectorStore(session)  # type: ignore[arg-type]
    results = store.search([0.0, 0.0, 0.0, 0.0])
    assert len(results) == 1
    assert results[0].source_type == "docs"
    assert results[0].content == "hello"
    assert results[0].score == 0.875
