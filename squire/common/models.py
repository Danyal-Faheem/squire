# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed SQLAlchemy ORM models for every table in the data model (Section 6).

All persistence in the engine goes through these models — there is no hand-written SQL
in any store implementation. They use the 2.0-style ``Mapped`` / ``mapped_column``
declarations, so columns are typed end-to-end, and the knowledge-base embedding uses
pgvector's ORM column type, which also exposes the distance expressions similarity
search needs (so vectors stay in the ORM idiom rather than dropping to raw SQL).

These models describe the schema the Alembic migrations create; they are the logical
shape of the tables, used for queries. Store *protocols* (the data-access seams) are
defined elsewhere and are unaffected by anything here — only the concrete stores use
these models.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from squire.common.config import get_settings

# The embedding column's width is fixed in the schema and must match EMBEDDING_DIM
# (see the initial migration). Read it once from configuration; swapping to a model of
# a different dimension requires a migration, per the embedding-seam caveat.
_EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    """Declarative base for all engine ORM models."""


class Chunk(Base):
    """A unit of indexed content plus its embedding — the knowledge base."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    origin: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIM))
    content_hash: Mapped[str] = mapped_column(Text, unique=True)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ConnectorCursor(Base):
    """A per-connector-instance incremental watermark."""

    __tablename__ = "connector_cursors"

    connector_key: Mapped[str] = mapped_column(Text, primary_key=True)
    instance_id: Mapped[str] = mapped_column(Text, primary_key=True)
    last_cursor: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TrackedItem(Base):
    """The mapping from an original signal to the board work item it produced."""

    __tablename__ = "tracked_items"
    __table_args__ = (UniqueConstraint("source_type", "source_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    board_item_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    priority: Mapped[str | None] = mapped_column(Text, nullable=True)
    assigned_owner: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, server_default="created")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentRequest(Base):
    """A queued request for the coding agent to open a pull request."""

    __tablename__ = "agent_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tracked_item_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("tracked_items.id"), nullable=True
    )
    requested_by: Mapped[str] = mapped_column(Text)
    surface: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="queued")
    pr_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CommandLog(Base):
    """An audit-log entry for one command received on any surface."""

    __tablename__ = "command_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tracked_item_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("tracked_items.id"), nullable=True
    )
    surface: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(Text)
    command: Mapped[str] = mapped_column(Text)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
