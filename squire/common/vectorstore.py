# SPDX-License-Identifier: AGPL-3.0-or-later
"""The swappable vector-store interface and its pgvector implementation.

This is one of Squire's deliberate seams (see CLAUDE.md, seam 5 in the seam map).
All similarity search and chunk persistence goes through :class:`VectorStore`
(a ``Protocol``). The default — and, for now, only — implementation is
:class:`PgVectorStore`, which keeps embeddings in the same PostgreSQL database as the
relational state. The interface exists so a dedicated vector database could be
slotted in later without touching anything downstream; that second implementation is
deliberately deferred until a real need for it appears.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from squire.common.schemas import Document


class SearchResult(Document):
    """A retrieved document chunk plus its similarity score (higher is closer)."""

    score: float


class VectorStore(Protocol):
    """The minimal contract every vector-store backend must satisfy."""

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        """Insert or update chunks with their embeddings; return the count written."""
        ...

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Return the closest chunks to ``embedding``, optionally filtered by type."""
        ...


class PgVectorStore:
    """A :class:`VectorStore` backed by PostgreSQL + pgvector.

    The session is injected so the store participates in the caller's transaction
    and so tests can supply a session bound to a throwaway database.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        """Insert chunks, skipping any whose ``content_hash`` already exists.

        Returns the number of rows actually written. De-duplication on
        ``content_hash`` is what keeps incremental re-ingestion from creating
        duplicate vectors.
        """
        written = 0
        for doc, embedding in documents:
            result = self._session.execute(
                text(
                    """
                    INSERT INTO chunks (
                        source_type, source_url, origin, content,
                        embedding, content_hash
                    )
                    VALUES (
                        :source_type, :source_url, :origin, :content,
                        :embedding, :content_hash
                    )
                    ON CONFLICT (content_hash) DO NOTHING
                    """
                ),
                {
                    "source_type": doc.source_type,
                    "source_url": doc.source_url,
                    "origin": doc.origin,
                    "content": doc.content,
                    "embedding": str(embedding),
                    "content_hash": doc.content_hash,
                },
            )
            # rowcount is defined on CursorResult, which is what a DML INSERT
            # returns at runtime; the static type is the broader Result.
            written += getattr(result, "rowcount", 0) or 0
        return written

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Return the ``limit`` closest chunks by cosine distance.

        When ``source_types`` is given, retrieval is restricted to those source
        types — this is how the various answer paths bias retrieval.
        """
        filter_sql = ""
        params: dict[str, object] = {
            "embedding": str(embedding),
            "limit": limit,
        }
        if source_types:
            filter_sql = "WHERE source_type = ANY(:types)"
            params["types"] = source_types

        rows = self._session.execute(
            text(
                f"""
                SELECT source_type, source_url, origin, content, content_hash,
                       1 - (embedding <=> :embedding) AS score
                FROM chunks
                {filter_sql}
                ORDER BY embedding <=> :embedding
                LIMIT :limit
                """
            ),
            params,
        ).mappings()

        return [
            SearchResult(
                source_type=row["source_type"],
                source_url=row["source_url"],
                origin=row["origin"],
                content=row["content"],
                content_hash=row["content_hash"],
                score=float(row["score"]),
            )
            for row in rows
        ]
