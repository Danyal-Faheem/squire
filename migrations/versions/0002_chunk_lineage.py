# SPDX-License-Identifier: AGPL-3.0-or-later
"""chunk lineage

Revision ID: 0002_chunk_lineage
Revises: 0001_initial
Create Date: 2026-06-04

Adds the chunk-lineage columns to `chunks`. A source Document is now split into
several chunks before embedding, so each row records which source Document it came
from (`parent_content_hash`), its ordinal position within that parent (`chunk_index`),
an optional section `title`, and opaque connector/chunker `metadata_json`. An index on
`parent_content_hash` lets all chunks of a source be found together.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_chunk_lineage"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN parent_content_hash TEXT")
    op.execute("ALTER TABLE chunks ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE chunks ADD COLUMN title TEXT")
    op.execute("ALTER TABLE chunks ADD COLUMN metadata_json JSONB")
    op.execute(
        "CREATE INDEX chunks_parent_content_hash_idx "
        "ON chunks (parent_content_hash)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_parent_content_hash_idx")
    op.execute("ALTER TABLE chunks DROP COLUMN metadata_json")
    op.execute("ALTER TABLE chunks DROP COLUMN title")
    op.execute("ALTER TABLE chunks DROP COLUMN chunk_index")
    op.execute("ALTER TABLE chunks DROP COLUMN parent_content_hash")
