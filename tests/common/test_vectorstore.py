# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the pgvector vector store.

The store now queries through the typed ORM (including the similarity search via
pgvector's ORM distance expression), so its contract is verified against a real
Postgres rather than by matching hand-written SQL strings. These are ``@integration``
tests: they skip when no database is reachable. They cover the behaviours the rest of
the engine relies on — de-duplication on content hash, similarity ordering, the
source-type filter, and row mapping.
"""

from __future__ import annotations

import pytest

from squire.common.db import session_scope
from squire.common.schemas import Document
from squire.common.vectorstore import PgVectorStore

pytestmark = pytest.mark.integration

DIM = 1024


def _doc(
    content_hash: str, content: str = "content", source_type: str = "docs"
) -> Document:
    return Document(
        source_type=source_type,
        source_url=f"https://example.test/{content_hash}",
        origin="origin",
        content=content,
        content_hash=content_hash,
    )


def _vec(lead: float) -> list[float]:
    return [lead, *([0.0] * (DIM - 1))]


def test_upsert_counts_and_dedups_on_content_hash(postgres_ready: None) -> None:
    """upsert writes new chunks and skips any whose content_hash already exists."""
    with session_scope() as session:
        store = PgVectorStore(session)
        first = store.upsert([(_doc("a"), _vec(1.0)), (_doc("b"), _vec(0.5))])
    with session_scope() as session:
        second = PgVectorStore(session).upsert([(_doc("a"), _vec(1.0))])

    assert first == 2
    assert second == 0  # the duplicate content_hash is not written again


def test_search_orders_by_similarity_and_maps_rows(postgres_ready: None) -> None:
    """The closest chunk comes first, mapped into a SearchResult with its score."""
    with session_scope() as session:
        PgVectorStore(session).upsert(
            [
                (_doc("near", "the near one"), [1.0, 0.0, *([0.0] * (DIM - 2))]),
                (_doc("far", "the far one"), [0.0, 1.0, *([0.0] * (DIM - 2))]),
            ]
        )
    with session_scope() as session:
        results = PgVectorStore(session).search(
            [1.0, 0.0, *([0.0] * (DIM - 2))], limit=2
        )

    assert [r.content_hash for r in results] == ["near", "far"]
    assert results[0].content == "the near one"
    assert results[0].source_type == "docs"
    assert isinstance(results[0].score, float)
    assert results[0].score >= results[1].score


def test_search_without_filter_spans_source_types(postgres_ready: None) -> None:
    """An unfiltered search may return any source type."""
    with session_scope() as session:
        PgVectorStore(session).upsert(
            [
                (_doc("d", source_type="docs"), _vec(1.0)),
                (_doc("c", source_type="github_code"), _vec(1.0)),
            ]
        )
    with session_scope() as session:
        results = PgVectorStore(session).search(_vec(1.0), limit=10)

    assert {r.source_type for r in results} == {"docs", "github_code"}


def test_search_with_filter_restricts_source_type(postgres_ready: None) -> None:
    """A filtered search returns only the requested source types."""
    with session_scope() as session:
        PgVectorStore(session).upsert(
            [
                (_doc("d", source_type="docs"), _vec(1.0)),
                (_doc("c", source_type="github_code"), _vec(1.0)),
            ]
        )
    with session_scope() as session:
        results = PgVectorStore(session).search(
            _vec(1.0), limit=10, source_types=["docs"]
        )

    assert [r.source_type for r in results] == ["docs"]
