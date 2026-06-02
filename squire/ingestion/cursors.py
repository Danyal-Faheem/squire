# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-connector incremental bookkeeping.

Each connector instance has a watermark — an opaque :class:`~squire.connectors.base.
Cursor` — recording how far ingestion has progressed. The orchestrator reads it
before a live run and writes it after a successful run, so the next run resumes
rather than re-reading everything.

The watermark is stored in the ``connector_cursors`` table, keyed by
``(connector_key, instance_id)`` so that two instances of the same connector type
(say two repositories) keep independent progress. Access goes through a small
:class:`CursorStore` interface with a PostgreSQL-backed default, mirroring the
:class:`~squire.common.vectorstore.VectorStore` pattern: the interface keeps the
orchestrator's logic testable offline, while Postgres stays the fixed primary store.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from squire.connectors.base import Cursor


class CursorStore(Protocol):
    """Reads and writes connector watermarks."""

    def get(self, connector_key: str, instance_id: str) -> Cursor | None:
        """Return the stored watermark for an instance, or ``None`` if unset."""
        ...

    def set(self, connector_key: str, instance_id: str, cursor: Cursor) -> None:
        """Persist the watermark for an instance, overwriting any previous value."""
        ...


class PgCursorStore:
    """A :class:`CursorStore` backed by the ``connector_cursors`` table.

    The session is injected so the store participates in the caller's transaction,
    exactly like :class:`~squire.common.vectorstore.PgVectorStore`.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, connector_key: str, instance_id: str) -> Cursor | None:
        row = self._session.execute(
            text(
                """
                SELECT last_cursor
                FROM connector_cursors
                WHERE connector_key = :key AND instance_id = :instance
                """
            ),
            {"key": connector_key, "instance": instance_id},
        ).scalar_one_or_none()
        if row is None:
            return None
        return Cursor(value=row)

    def set(self, connector_key: str, instance_id: str, cursor: Cursor) -> None:
        self._session.execute(
            text(
                """
                INSERT INTO connector_cursors (connector_key, instance_id, last_cursor)
                VALUES (:key, :instance, :cursor)
                ON CONFLICT (connector_key, instance_id)
                DO UPDATE SET last_cursor = :cursor, updated_at = now()
                """
            ),
            {"key": connector_key, "instance": instance_id, "cursor": cursor.value},
        )
