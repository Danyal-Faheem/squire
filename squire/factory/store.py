# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistence for the original-signal → board-item mapping.

``tracked_items`` is the dedup ledger: one row per signal the factory has acted on,
keyed uniquely by ``(source_type, source_id)``. Because it is persisted, the factory
recognises a signal it has already handled across restarts and never creates a second
board item for it.

Access goes through a small :class:`TrackedItemStore` interface with a PostgreSQL
default, the same pattern as the vector and cursor stores: the interface keeps the
factory testable offline while Postgres remains the fixed primary store.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session


class TrackedItem(BaseModel):
    """A persisted mapping from an original signal to its board work item."""

    id: int
    source_type: str
    source_id: str
    source_url: str
    content_hash: str
    board_item_ref: str | None
    category: str | None
    priority: str | None
    assigned_owner: str | None
    status: str
    created_at: datetime


class NewTrackedItem(BaseModel):
    """The fields needed to record a freshly-created work item."""

    source_type: str
    source_id: str
    source_url: str
    content_hash: str
    board_item_ref: str
    category: str
    priority: str
    assigned_owner: str
    status: str = "created"


class TrackedItemStore(Protocol):
    """Reads and writes the original-signal → board-item mapping."""

    def get(self, source_type: str, source_id: str) -> TrackedItem | None:
        """Return the tracked item for a signal, or ``None`` if not yet seen."""
        ...

    def create(self, item: NewTrackedItem) -> int:
        """Insert a new tracked item and return its id."""
        ...

    def update_classification(
        self,
        item_id: int,
        *,
        content_hash: str,
        category: str,
        priority: str,
        assigned_owner: str,
    ) -> None:
        """Refresh a tracked item after its source changed."""
        ...


class PgTrackedItemStore:
    """A :class:`TrackedItemStore` backed by the ``tracked_items`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, source_type: str, source_id: str) -> TrackedItem | None:
        row = (
            self._session.execute(
                text(
                    """
                SELECT id, source_type, source_id, source_url, content_hash,
                       board_item_ref, category, priority, assigned_owner,
                       status, created_at
                FROM tracked_items
                WHERE source_type = :source_type AND source_id = :source_id
                """
                ),
                {"source_type": source_type, "source_id": source_id},
            )
            .mappings()
            .one_or_none()
        )
        return TrackedItem(**row) if row is not None else None

    def create(self, item: NewTrackedItem) -> int:
        new_id = self._session.execute(
            text(
                """
                INSERT INTO tracked_items (
                    source_type, source_id, source_url, content_hash,
                    board_item_ref, category, priority, assigned_owner, status
                )
                VALUES (
                    :source_type, :source_id, :source_url, :content_hash,
                    :board_item_ref, :category, :priority, :assigned_owner, :status
                )
                RETURNING id
                """
            ),
            item.model_dump(),
        ).scalar_one()
        return int(new_id)

    def update_classification(
        self,
        item_id: int,
        *,
        content_hash: str,
        category: str,
        priority: str,
        assigned_owner: str,
    ) -> None:
        self._session.execute(
            text(
                """
                UPDATE tracked_items
                SET content_hash = :content_hash, category = :category,
                    priority = :priority, assigned_owner = :assigned_owner
                WHERE id = :item_id
                """
            ),
            {
                "item_id": item_id,
                "content_hash": content_hash,
                "category": category,
                "priority": priority,
                "assigned_owner": assigned_owner,
            },
        )
