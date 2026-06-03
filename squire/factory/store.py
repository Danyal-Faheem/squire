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
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from squire.common import models


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
        row = self._session.execute(
            select(models.TrackedItem).where(
                models.TrackedItem.source_type == source_type,
                models.TrackedItem.source_id == source_id,
            )
        ).scalar_one_or_none()
        return (
            None
            if row is None
            else TrackedItem.model_validate(row, from_attributes=True)
        )

    def create(self, item: NewTrackedItem) -> int:
        row = models.TrackedItem(**item.model_dump())
        self._session.add(row)
        self._session.flush()  # populate the generated primary key
        return row.id

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
            update(models.TrackedItem)
            .where(models.TrackedItem.id == item_id)
            .values(
                content_hash=content_hash,
                category=category,
                priority=priority,
                assigned_owner=assigned_owner,
            )
        )
