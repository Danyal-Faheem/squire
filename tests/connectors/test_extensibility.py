# SPDX-License-Identifier: AGPL-3.0-or-later
"""The extensibility proof: a brand-new source type as configuration alone.

This is the real test of the connector architecture (PLAN Section 1C / seam 1). It
defines a throwaway connector, registers it under a new key using only the public
``register`` decorator, lists that key in a deployment.yml, and drives it through the
ingestion orchestrator end-to-end. Crucially it touches *no* engine internals — not
the registry, not the orchestrator, not any existing connector — so it demonstrates
that adding a source type is "one new class plus a registration", the Open/Closed
Principle the design promises.

It deliberately lives as a test rather than a shipped connector; its only job is to
document and guard that contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from squire.common.deployment import load_connector_configs
from squire.common.schemas import Document, Signal
from squire.common.vectorstore import SearchResult
from squire.connectors import registry
from squire.connectors.base import ConnectorConfig, Cursor
from squire.ingestion.orchestrator import IngestionMode, IngestionOrchestrator
from tests.fakes import make_embedder

NULL_KEY = "null-extensibility-probe"


class NullConnector:
    """A minimal both-responsibilities connector built only for this proof."""

    key: ClassVar[str] = NULL_KEY

    def __init__(self, config: ConnectorConfig) -> None:
        self.instance_id = config.instance_id

    def healthcheck(self) -> None:
        return None

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        yield Document(
            source_type="null_doc",
            source_url="null://doc/1",
            origin="null-source",
            content="hello from a brand new connector",
            content_hash="null-doc-hash-1",
        )

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        yield Signal(
            source_type="null_signal",
            source_id="null#1",
            source_url="null://signal/1",
            title="A new signal",
            body="emitted by a connector the engine never heard of",
            content_hash="null-signal-hash-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def current_cursor(self) -> Cursor | None:
        return Cursor(value="null-watermark")


# --- offline fakes for the orchestrator's collaborators ----------------------------


class _FakeVectorStore:
    def __init__(self) -> None:
        self.rows: list[Document] = []

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        self.rows.extend(doc for doc, _ in documents)
        return len(documents)

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        return []


class _InMemoryCursorStore:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], Cursor] = {}

    def get(self, connector_key: str, instance_id: str) -> Cursor | None:
        return self.store.get((connector_key, instance_id))

    def set(self, connector_key: str, instance_id: str, cursor: Cursor) -> None:
        self.store[(connector_key, instance_id)] = cursor


@pytest.fixture
def registered_null_connector() -> Iterator[None]:
    """Register the null connector via the public API; restore the registry after."""
    saved = dict(registry._REGISTRY)
    registry.register(NullConnector)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_new_source_type_runs_from_config_alone(
    registered_null_connector: None, tmp_path: Path
) -> None:
    """A never-before-seen connector key in deployment.yml runs end-to-end.

    Nothing in the engine was edited to make this work: the orchestrator discovers
    the connector through the registry, indexes its document, and dispatches its
    signal, purely because the deployment activated its key.
    """
    deployment = tmp_path / "dep"
    deployment.mkdir()
    (deployment / "deployment.yml").write_text(
        f"""
        connectors:
          - key: {NULL_KEY}
            instance_id: probe
        """,
        encoding="utf-8",
    )

    connectors = registry.create_all(load_connector_configs(deployment))
    assert [c.key for c in connectors] == [NULL_KEY]

    store = _FakeVectorStore()
    cursors = _InMemoryCursorStore()
    signals: list[Signal] = []
    orchestrator = IngestionOrchestrator(
        embedder=make_embedder(),
        vector_store=store,
        cursors=cursors,
        signal_handler=signals.append,
    )

    result = orchestrator.run(connectors, mode=IngestionMode.BACKFILL)

    # The document was indexed with the connector's own source-type label...
    assert [doc.source_type for doc in store.rows] == ["null_doc"]
    # ...the signal was dispatched...
    assert [s.source_id for s in signals] == ["null#1"]
    # ...the counts add up...
    assert result.documents_indexed == 1
    assert result.signals_emitted == 1
    # ...and the watermark was persisted under the new key.
    assert cursors.get(NULL_KEY, "probe") == Cursor(value="null-watermark")
