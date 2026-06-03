# SPDX-License-Identifier: AGPL-3.0-or-later
"""The swappable vector-store interface and its pgvector implementation.

This is one of Squire's deliberate seams (seam 4 in the seam map). All similarity
search and chunk persistence goes through :class:`VectorStore` (a ``Protocol``). The
default — and, for now, only — implementation is :class:`PgVectorStore`, which keeps
embeddings in the same PostgreSQL database as the relational state. The interface
exists so a dedicated vector database could be slotted in later without touching
anything downstream; that second implementation is deliberately deferred until a real
need for it appears.

The implementation queries through the typed ORM, including the similarity search:
pgvector's :class:`~pgvector.sqlalchemy.Vector` column exposes a ``cosine_distance``
expression usable in ``order_by``/``select``, so vectors stay in the ORM idiom rather
than dropping to raw SQL.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from squire.common.models import Chunk
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
    """A :class:`VectorStore` backed by PostgreSQL + pgvector, via the ORM.

    The session is injected so the store participates in the caller's transaction
    and so tests can supply a session bound to a throwaway database.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        """Insert chunks, skipping any whose ``content_hash`` already exists.

        Returns the number of rows actually written. De-duplication on
        ``content_hash`` (an ``ON CONFLICT DO NOTHING`` upsert) is what keeps
        incremental re-ingestion from creating duplicate vectors.
        """
        written = 0
        for doc, embedding in documents:
            statement = (
                pg_insert(Chunk)
                .values(
                    source_type=doc.source_type,
                    source_url=doc.source_url,
                    origin=doc.origin,
                    content=doc.content,
                    embedding=embedding,
                    content_hash=doc.content_hash,
                )
                .on_conflict_do_nothing(index_elements=[Chunk.content_hash])
                .returning(Chunk.id)
            )
            # RETURNING yields one row when the chunk was inserted and none when the
            # content_hash already existed, so the row count is exactly what was
            # written — more reliable than rowcount for ON CONFLICT statements.
            written += len(self._session.execute(statement).all())
        return written

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Return the ``limit`` closest chunks by cosine similarity.

        When ``source_types`` is given, retrieval is restricted to those source
        types — this is how the various answer paths bias retrieval.
        """
        distance = Chunk.embedding.cosine_distance(embedding)
        statement = (
            select(
                Chunk.source_type,
                Chunk.source_url,
                Chunk.origin,
                Chunk.content,
                Chunk.content_hash,
                (1 - distance).label("score"),
            )
            .order_by(distance)
            .limit(limit)
        )
        if source_types:
            statement = statement.where(Chunk.source_type.in_(source_types))

        rows = self._session.execute(statement).mappings()
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
