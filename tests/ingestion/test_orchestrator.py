# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the ingestion orchestrator.

The orchestrator's collaborators are faked at their interfaces — a recording vector
store, an in-memory cursor store, an embedding client with a faked transport, and
connectors that yield canned content. That isolates the orchestration logic: mode
selection, cursor advancement, signal dispatch, and de-duplication on re-run. A final
end-to-end test drives the *real* filesystem and GitHub connectors from a deployment
config to confirm the whole Phase 1 path fits together.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from squire.common.config import Settings
from squire.common.deployment import load_connector_configs
from squire.common.embeddings import EmbeddingClient
from squire.common.schemas import Document, Signal
from squire.common.vectorstore import SearchResult
from squire.connectors import registry
from squire.connectors.base import Cursor
from squire.ingestion.cursors import CursorStore
from squire.ingestion.orchestrator import IngestionMode, IngestionOrchestrator

# --- fakes -------------------------------------------------------------------------


class FakeVectorStore:
    """A vector store that de-duplicates on content hash, like the real one."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.rows: list[Document] = []

    def upsert(self, documents: list[tuple[Document, list[float]]]) -> int:
        written = 0
        for doc, _embedding in documents:
            if doc.content_hash in self.seen:
                continue
            self.seen.add(doc.content_hash)
            self.rows.append(doc)
            written += 1
        return written

    def search(
        self,
        embedding: list[float],
        *,
        limit: int = 8,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        return []


class InMemoryCursorStore:
    """A CursorStore backed by a dict, for offline orchestrator tests."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], Cursor] = {}

    def get(self, connector_key: str, instance_id: str) -> Cursor | None:
        return self.store.get((connector_key, instance_id))

    def set(self, connector_key: str, instance_id: str, cursor: Cursor) -> None:
        self.store[(connector_key, instance_id)] = cursor


class FakeConnector:
    """A connector that yields canned content and records what it was asked for."""

    key: ClassVar[str] = "fake"

    def __init__(
        self,
        *,
        instance_id: str = "fake",
        documents: list[Document] | None = None,
        signals: list[Signal] | None = None,
        cursor: Cursor | None = None,
    ) -> None:
        self.instance_id = instance_id
        self._documents = documents or []
        self._signals = signals or []
        self._cursor = cursor
        self.since_seen: Cursor | None = None
        self.since_was_set = False

    def healthcheck(self) -> None:
        return None

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        self.since_seen = since
        self.since_was_set = True
        yield from self._documents

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        yield from self._signals

    def current_cursor(self) -> Cursor | None:
        return self._cursor


def _doc(content_hash: str, content: str = "body") -> Document:
    return Document(
        source_type="fake",
        source_url=f"https://example.test/{content_hash}",
        origin="origin",
        content=content,
        content_hash=content_hash,
    )


def _signal(source_id: str) -> Signal:
    return Signal(
        source_type="fake",
        source_id=source_id,
        source_url=f"https://example.test/{source_id}",
        title="t",
        body="b",
        content_hash=f"h-{source_id}",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _embedder(dim: int = 4) -> EmbeddingClient:
    """A real EmbeddingClient with a faked transport returning fixed vectors."""
    client = EmbeddingClient(Settings(embedding_dim=dim))

    def _create(*, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(embedding=[0.0] * dim) for _ in input]
        return SimpleNamespace(data=data)

    client._client = SimpleNamespace(  # type: ignore[assignment]
        embeddings=SimpleNamespace(create=_create)
    )
    return client


def _orchestrator(
    vector_store: FakeVectorStore,
    cursors: CursorStore,
    signal_handler: Any | None = None,
) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        embedder=_embedder(),
        vector_store=vector_store,
        cursors=cursors,
        signal_handler=signal_handler,
    )


# --- tests -------------------------------------------------------------------------


def test_backfill_indexes_documents_and_counts() -> None:
    """A backfill embeds and upserts every document, reporting the count."""
    store = FakeVectorStore()
    connector = FakeConnector(documents=[_doc("h1"), _doc("h2")])

    result = _orchestrator(store, InMemoryCursorStore()).run(
        [connector], mode=IngestionMode.BACKFILL
    )

    assert result.documents_indexed == 2
    assert len(store.rows) == 2


def test_signals_are_dispatched_to_handler() -> None:
    """Every emitted signal reaches the configured handler exactly once."""
    received: list[Signal] = []
    connector = FakeConnector(signals=[_signal("1"), _signal("2")])

    result = _orchestrator(
        FakeVectorStore(), InMemoryCursorStore(), received.append
    ).run([connector])

    assert [s.source_id for s in received] == ["1", "2"]
    assert result.signals_emitted == 2


def test_backfill_ignores_stored_cursor() -> None:
    """Backfill passes since=None even when a watermark exists."""
    cursors = InMemoryCursorStore()
    cursors.set("fake", "fake", Cursor(value="old"))
    connector = FakeConnector(documents=[_doc("h1")])

    _orchestrator(FakeVectorStore(), cursors).run(
        [connector], mode=IngestionMode.BACKFILL
    )

    assert connector.since_was_set
    assert connector.since_seen is None


def test_live_passes_stored_cursor_as_since() -> None:
    """Live mode hands the stored watermark back to the connector."""
    cursors = InMemoryCursorStore()
    cursors.set("fake", "fake", Cursor(value="2026-01-01T00:00:00+00:00"))
    connector = FakeConnector(documents=[_doc("h1")])

    _orchestrator(FakeVectorStore(), cursors).run([connector], mode=IngestionMode.LIVE)

    assert connector.since_seen == Cursor(value="2026-01-01T00:00:00+00:00")


def test_cursor_is_persisted_after_run() -> None:
    """A connector's reported watermark is stored for the next run."""
    cursors = InMemoryCursorStore()
    connector = FakeConnector(
        documents=[_doc("h1")], cursor=Cursor(value="new-watermark")
    )

    _orchestrator(FakeVectorStore(), cursors).run([connector])

    assert cursors.get("fake", "fake") == Cursor(value="new-watermark")


def test_none_cursor_leaves_stored_value_untouched() -> None:
    """A connector reporting no watermark does not clobber the stored one."""
    cursors = InMemoryCursorStore()
    cursors.set("fake", "fake", Cursor(value="keep"))
    connector = FakeConnector(documents=[_doc("h1")], cursor=None)

    _orchestrator(FakeVectorStore(), cursors).run([connector])

    assert cursors.get("fake", "fake") == Cursor(value="keep")


def test_rerun_adds_only_new_content() -> None:
    """Re-running over the same documents writes nothing the second time."""
    store = FakeVectorStore()
    orchestrator = _orchestrator(store, InMemoryCursorStore())

    first = orchestrator.run([FakeConnector(documents=[_doc("h1"), _doc("h2")])])
    second = orchestrator.run([FakeConnector(documents=[_doc("h1"), _doc("h2")])])

    assert first.documents_indexed == 2
    assert second.documents_indexed == 0
    assert len(store.rows) == 2


class ExplodingConnector:
    """A connector whose document stream fails partway through a run."""

    key: ClassVar[str] = "exploding"

    def __init__(self, *, instance_id: str = "boom") -> None:
        self.instance_id = instance_id

    def healthcheck(self) -> None:
        return None

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        yield _doc("partial")
        raise RuntimeError("API error mid-iteration")

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        return iter(())

    def current_cursor(self) -> Cursor | None:
        # If this were ever persisted after the failure, the next run would skip
        # content. The orchestrator must not reach this far on a failed run.
        return Cursor(value="advanced")


def test_failure_mid_iteration_does_not_advance_cursor() -> None:
    """A run that raises must leave the stored watermark untouched (resumable).

    Otherwise a transient source error would advance the cursor past content that
    was never successfully ingested, silently skipping it on the next run.
    """
    cursors = InMemoryCursorStore()
    cursors.set("exploding", "boom", Cursor(value="original"))
    orchestrator = _orchestrator(FakeVectorStore(), cursors)

    with pytest.raises(RuntimeError, match="API error mid-iteration"):
        orchestrator.run([ExplodingConnector()], mode=IngestionMode.LIVE)

    assert cursors.get("exploding", "boom") == Cursor(value="original")


def test_multiple_connectors_are_aggregated() -> None:
    """Results across connectors are summed."""
    store = FakeVectorStore()
    connectors = [
        FakeConnector(instance_id="a", documents=[_doc("a1")], signals=[_signal("s1")]),
        FakeConnector(instance_id="b", documents=[_doc("b1")]),
    ]

    result = _orchestrator(store, InMemoryCursorStore()).run(connectors)

    assert result.documents_indexed == 2
    assert result.signals_emitted == 1
    assert len(result.runs) == 2


def test_end_to_end_from_deployment_config(tmp_path: Path) -> None:
    """A deployment listing filesystem + github drives the whole pipeline.

    This is the Phase 1 acceptance test in miniature: configuration selects the
    connectors, documents land in the (fake) knowledge base with connector-owned
    metadata, the github connector emits a signal, and a second live run adds
    nothing new.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("welcome", encoding="utf-8")

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

    configs = load_connector_configs(deployment)
    connectors = registry.create_all(configs)

    # Inject a fake repository into the github connector so no network is touched.
    github = connectors[1]
    github._repo = SimpleNamespace(  # type: ignore[attr-defined]
        full_name="owner/name",
        get_issues=lambda **kwargs: [
            SimpleNamespace(
                number=1,
                title="Question",
                body="How?",
                html_url="https://example.test/owner/name/issues/1",
                updated_at=datetime(2026, 1, 2, tzinfo=UTC),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                pull_request=None,
            )
        ],
    )

    store = FakeVectorStore()
    cursors = InMemoryCursorStore()
    signals: list[Signal] = []
    orchestrator = _orchestrator(store, cursors, signals.append)

    result = orchestrator.run(connectors, mode=IngestionMode.BACKFILL)

    # Documents from both connectors landed with their connector-owned source types.
    source_types = {doc.source_type for doc in store.rows}
    assert source_types == {"docs", "github_issue"}
    # The github connector emitted exactly one signal.
    assert [s.source_id for s in signals] == ["owner/name#1"]
    assert result.signals_emitted == 1

    # A second live run re-reads nothing new: de-dup keeps the corpus stable.
    second = orchestrator.run(connectors, mode=IngestionMode.LIVE)
    assert second.documents_indexed == 0
