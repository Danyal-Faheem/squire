# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: the live entrypoint turns a signal into a board item.

Phase 5 depends on the production path actually doing this, so this proves it through
the real ``run_ingest`` entrypoint rather than only in unit-composed pieces. A probe
connector emits one signal; with a deployment policy present, the composition root
wires the factory in as the signal handler, and under dry-run the factory composes the
board item and logs that it *would* create it — while writing nothing to the database.
The clients are injected as fakes; only the database is real. Skips without Postgres.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy import text

from squire.common.config import Settings
from squire.common.db import session_scope
from squire.common.schemas import Document, Signal
from squire.connectors import registry
from squire.connectors.base import ConnectorConfig, Cursor
from squire.ingestion import run_ingest
from squire.runtime import Runtime
from tests.fakes import make_embedder, make_llm

pytestmark = pytest.mark.integration

PROBE_KEY = "phase-hh-probe"
EMBEDDING_DIM = 1024


class _ProbeConnector:
    """A connector that emits exactly one signal, for driving the live path."""

    key: ClassVar[str] = PROBE_KEY

    def __init__(self, config: ConnectorConfig) -> None:
        self.instance_id = config.instance_id

    def healthcheck(self) -> None:
        return None

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        return iter(())

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        yield Signal(
            source_type="probe",
            source_id="probe#1",
            source_url="https://example.test/probe/1",
            title="A probe question",
            body="How does the probe work?",
            content_hash="probe-hash-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def current_cursor(self) -> Cursor | None:
        return None


@pytest.fixture
def registered_probe() -> Iterator[None]:
    saved = dict(registry._REGISTRY)
    registry.register(_ProbeConnector)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def _deployment(tmp_path: Path) -> Path:
    directory = tmp_path / "dep"
    directory.mkdir()
    (directory / "deployment.yml").write_text(
        f"connectors:\n  - key: {PROBE_KEY}\n    instance_id: probe\n",
        encoding="utf-8",
    )
    (directory / "taxonomy.yml").write_text(
        "categories:\n  - name: question\ndefault: question\n", encoding="utf-8"
    )
    (directory / "routing.yml").write_text(
        "routes: {}\ndefault:\n  owners: [maintainers]\n  repo: acme/widget\n",
        encoding="utf-8",
    )
    (directory / "priorities.yml").write_text(
        "levels:\n  - key: p2\n    label: 'priority: normal'\ndefault: p2\n",
        encoding="utf-8",
    )
    return directory


def test_live_entrypoint_turns_signal_into_board_item(
    postgres_ready: None,
    registered_probe: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """run_ingest drives the probe signal through the factory under dry-run."""
    deployment = _deployment(tmp_path)
    runtime = Runtime(
        Settings(embedding_dim=EMBEDDING_DIM, dry_run=True),
        deployment=deployment,
        llm=make_llm(),
        embedder=make_embedder(dim=EMBEDDING_DIM, vector=[0.0] * EMBEDDING_DIM),
    )

    with caplog.at_level(logging.INFO, logger="squire"):
        run_ingest.run([], runtime=runtime)

    # The factory ran the signal through to a (dry-run) board item — not merely
    # observed it — shown by the factory's dry-run action in the log.
    assert "would create board item" in caplog.text

    # Dry-run honesty end to end: nothing was actually written.
    with session_scope() as session:
        tracked = session.execute(
            text("SELECT count(*) FROM tracked_items")
        ).scalar_one()
        chunks = session.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    assert tracked == 0
    assert chunks == 0
