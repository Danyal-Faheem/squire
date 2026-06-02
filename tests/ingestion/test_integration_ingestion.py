# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: ingestion against a real Postgres + pgvector.

These tests prove the literal Phase 1 acceptance criterion — "documents land in
``chunks`` with correct metadata; incremental re-runs add only new content" — against
the real database rather than a fake. They are marked ``integration`` and skip
automatically when no Postgres is reachable, so the offline unit suite and CI stay
green. The embedding endpoint is still faked: only the database is real, since that is
what these tests exist to exercise.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text

from squire.common.config import Settings
from squire.common.db import session_scope
from squire.common.deployment import load_connector_configs
from squire.common.embeddings import EmbeddingClient
from squire.common.vectorstore import PgVectorStore
from squire.connectors import registry
from squire.ingestion.cursors import PgCursorStore
from squire.ingestion.orchestrator import IngestionMode, IngestionOrchestrator

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 1024


def _embedder() -> EmbeddingClient:
    """A real client with a faked transport returning schema-width vectors."""
    client = EmbeddingClient(Settings(embedding_dim=EMBEDDING_DIM))

    def _create(*, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(embedding=[0.0] * EMBEDDING_DIM) for _ in input]
        return SimpleNamespace(data=data)

    client._client = SimpleNamespace(  # type: ignore[assignment]
        embeddings=SimpleNamespace(create=_create)
    )
    return client


def _fake_repo() -> SimpleNamespace:
    return SimpleNamespace(
        full_name="owner/name",
        get_issues=lambda **kwargs: [
            SimpleNamespace(
                number=1,
                title="A question",
                body="How does this work?",
                html_url="https://example.test/owner/name/issues/1",
                updated_at=datetime(2026, 1, 2, tzinfo=UTC),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                pull_request=None,
            )
        ],
    )


def _build_connectors(tmp_path: Path) -> list[Any]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("welcome to the project", encoding="utf-8")

    deployment = tmp_path / "dep"
    deployment.mkdir()
    (deployment / "deployment.yml").write_text(
        f"""
        connectors:
          - key: filesystem
            instance_id: docs
            settings:
              root: {docs_dir}
              source_type: docs
          - key: github
            instance_id: repo
            settings:
              repo: owner/name
              include_code: false
        """,
        encoding="utf-8",
    )

    connectors = registry.create_all(load_connector_configs(deployment))
    # Inject a fake repository so the github connector touches no network.
    connectors[1]._repo = _fake_repo()  # type: ignore[attr-defined]
    return connectors


def test_documents_land_in_chunks_with_metadata(
    postgres_ready: None, tmp_path: Path
) -> None:
    """A backfill writes chunks carrying each connector's source-type label."""
    connectors = _build_connectors(tmp_path)
    signals: list[str] = []

    with session_scope() as session:
        orchestrator = IngestionOrchestrator(
            embedder=_embedder(),
            vector_store=PgVectorStore(session),
            cursors=PgCursorStore(session),
            signal_handler=lambda s: signals.append(s.source_id),
        )
        result = orchestrator.run(connectors, mode=IngestionMode.BACKFILL)

    assert result.documents_indexed == 2
    assert signals == ["owner/name#1"]

    with session_scope() as session:
        rows = (
            session.execute(
                text("SELECT source_type, origin FROM chunks ORDER BY source_type")
            )
            .mappings()
            .all()
        )

    by_type = {row["source_type"]: row["origin"] for row in rows}
    assert by_type["docs"] == "intro.md"
    assert by_type["github_issue"] == "owner/name#1"


def test_cursors_persist_and_rerun_adds_nothing(
    postgres_ready: None, tmp_path: Path
) -> None:
    """Watermarks are stored, and a second live run re-ingests no documents."""
    connectors = _build_connectors(tmp_path)

    with session_scope() as session:
        orchestrator = IngestionOrchestrator(
            embedder=_embedder(),
            vector_store=PgVectorStore(session),
            cursors=PgCursorStore(session),
        )
        orchestrator.run(connectors, mode=IngestionMode.BACKFILL)

    with session_scope() as session:
        cursor_rows = (
            session.execute(
                text("SELECT connector_key, instance_id FROM connector_cursors")
            )
            .mappings()
            .all()
        )
    persisted = {(row["connector_key"], row["instance_id"]) for row in cursor_rows}
    assert ("filesystem", "docs") in persisted
    assert ("github", "repo") in persisted

    # The same connectors run again in live mode find nothing new to write: the
    # filesystem watermark skips the unchanged file, and the github documents
    # de-duplicate on content hash.
    with session_scope() as session:
        orchestrator = IngestionOrchestrator(
            embedder=_embedder(),
            vector_store=PgVectorStore(session),
            cursors=PgCursorStore(session),
        )
        second = orchestrator.run(connectors, mode=IngestionMode.LIVE)

    assert second.documents_indexed == 0
