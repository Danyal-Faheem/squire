# SPDX-License-Identifier: AGPL-3.0-or-later
"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-01

Creates the generic core tables: the knowledge base (`chunks`, with a pgvector
embedding column), per-connector bookkeeping (`connector_cursors`), the original->
board mapping (`tracked_items`), agent PR requests, and a command audit log. No
column encodes a specific project.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The embedding dimension is fixed in the schema. It must match EMBEDDING_DIM in
# configuration (bge-m3 -> 1024). Swapping to an embedding model with a different
# dimension requires a new migration.
EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        f"""
        CREATE TABLE chunks (
            id            BIGSERIAL PRIMARY KEY,
            source_type   TEXT NOT NULL,
            source_url    TEXT NOT NULL,
            origin        TEXT,
            content       TEXT NOT NULL,
            embedding     vector({EMBEDDING_DIM}),
            content_hash  TEXT NOT NULL UNIQUE,
            indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX chunks_source_type_idx ON chunks (source_type)")
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.execute(
        """
        CREATE TABLE connector_cursors (
            connector_key TEXT NOT NULL,
            instance_id   TEXT NOT NULL,
            last_cursor   TEXT,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (connector_key, instance_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE tracked_items (
            id              BIGSERIAL PRIMARY KEY,
            source_type     TEXT NOT NULL,
            source_id       TEXT NOT NULL,
            source_url      TEXT NOT NULL,
            content_hash    TEXT NOT NULL,
            board_item_ref  TEXT,
            category        TEXT,
            priority        TEXT,
            assigned_owner  TEXT,
            status          TEXT NOT NULL DEFAULT 'created',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (source_type, source_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE agent_requests (
            id              BIGSERIAL PRIMARY KEY,
            tracked_item_id BIGINT REFERENCES tracked_items(id),
            requested_by    TEXT NOT NULL,
            surface         TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued',
            pr_url          TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE command_log (
            id              BIGSERIAL PRIMARY KEY,
            tracked_item_id BIGINT REFERENCES tracked_items(id),
            surface         TEXT NOT NULL,
            actor           TEXT NOT NULL,
            command         TEXT NOT NULL,
            payload         TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS command_log")
    op.execute("DROP TABLE IF EXISTS agent_requests")
    op.execute("DROP TABLE IF EXISTS tracked_items")
    op.execute("DROP TABLE IF EXISTS connector_cursors")
    op.execute("DROP TABLE IF EXISTS chunks")
