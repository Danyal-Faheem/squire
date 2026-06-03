# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ingestion orchestrator.

The orchestrator is deliberately source-agnostic: it iterates the activated
connectors and, for each, embeds the documents it yields into the knowledge base and
hands its signals to a handler. It knows nothing about GitHub, forums, or files —
only the connector contract.

Two modes:

* **Backfill** ignores any stored watermark and reads a source's full history. It is
  the right mode for first ingestion and for re-reading after a configuration change.
* **Live** reads only what changed since the stored watermark, which is what the
  scheduled updater runs continuously.

In both modes, the watermark a connector reports after a successful run is persisted,
so a backfill leaves the bookkeeping ready for live updates and an interrupted
backfill resumes connector by connector. Documents are de-duplicated on content hash
by the vector store, so even without watermarks a re-run adds only genuinely new
content; the watermark is what keeps a re-run from re-*fetching* it.

Signal handling is injected. Until the work-item factory exists it defaults to a
handler that simply logs, which keeps ingestion useful and observable on its own; a
later phase passes the factory in without the orchestrator changing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from squire.common.embeddings import EmbeddingClient
from squire.common.schemas import Chunk, Document, Signal
from squire.common.vectorstore import VectorStore
from squire.connectors.base import Connector, Cursor
from squire.ingestion.chunking import DocumentProcessor, LocalDocumentProcessor
from squire.ingestion.cursors import CursorStore

logger = logging.getLogger(__name__)

#: A signal handler consumes one newly-observed signal. The default logs it; a later
#: phase supplies the work-item factory in its place.
SignalHandler = Callable[[Signal], None]


class IngestionMode(StrEnum):
    """How much history to read on a run."""

    BACKFILL = "backfill"
    LIVE = "live"


def logging_signal_handler(signal: Signal) -> None:
    """The default signal handler: record that a signal was observed.

    It performs no outbound writes, so it is safe under any configuration. A later
    phase replaces it with the work-item factory.
    """
    logger.info(
        "observed signal %s %s (%s)",
        signal.source_type,
        signal.source_id,
        signal.source_url,
    )


@dataclass
class ConnectorRunResult:
    """The outcome of running one connector."""

    key: str
    instance_id: str
    chunks_indexed: int
    signals_emitted: int


@dataclass
class IngestionResult:
    """The outcome of an ingestion run across all connectors."""

    runs: list[ConnectorRunResult] = field(default_factory=list)

    @property
    def chunks_indexed(self) -> int:
        """Total new chunks written to the knowledge base across all connectors."""
        return sum(run.chunks_indexed for run in self.runs)

    @property
    def signals_emitted(self) -> int:
        """Total signals observed across all connectors."""
        return sum(run.signals_emitted for run in self.runs)


class IngestionOrchestrator:
    """Runs connectors, indexing their documents and dispatching their signals."""

    def __init__(
        self,
        *,
        embedder: EmbeddingClient,
        vector_store: VectorStore,
        cursors: CursorStore,
        processor: DocumentProcessor | None = None,
        signal_handler: SignalHandler | None = None,
        batch_size: int = 64,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._cursors = cursors
        self._processor = processor or LocalDocumentProcessor()
        self._signal_handler = signal_handler or logging_signal_handler
        self._batch_size = batch_size

    def run(
        self,
        connectors: Sequence[Connector],
        *,
        mode: IngestionMode = IngestionMode.BACKFILL,
    ) -> IngestionResult:
        """Run every connector in ``connectors`` and return the combined result."""
        result = IngestionResult()
        for connector in connectors:
            result.runs.append(self._run_one(connector, mode))
        return result

    def _run_one(self, connector: Connector, mode: IngestionMode) -> ConnectorRunResult:
        since = self._since(connector, mode)

        # Drain documents first and index them, then drain signals. Both iterators
        # must be exhausted before reading the cursor, since a connector's watermark
        # reflects everything it has yielded.
        chunks_indexed = self._index(connector.iter_documents(since))

        signals_emitted = 0
        for signal in connector.iter_signals(since):
            self._signal_handler(signal)
            signals_emitted += 1

        self._advance_cursor(connector)

        logger.info(
            "connector %s/%s indexed %d chunks, emitted %d signals",
            connector.key,
            connector.instance_id,
            chunks_indexed,
            signals_emitted,
        )
        return ConnectorRunResult(
            key=connector.key,
            instance_id=connector.instance_id,
            chunks_indexed=chunks_indexed,
            signals_emitted=signals_emitted,
        )

    def _since(self, connector: Connector, mode: IngestionMode) -> Cursor | None:
        if mode is IngestionMode.BACKFILL:
            return None
        return self._cursors.get(connector.key, connector.instance_id)

    def _advance_cursor(self, connector: Connector) -> None:
        cursor = connector.current_cursor()
        # A connector that reports no watermark (nothing new, or no notion of one)
        # leaves the stored value untouched rather than clobbering it with nothing.
        if cursor is not None:
            self._cursors.set(connector.key, connector.instance_id, cursor)

    def _index(self, documents: Iterable[Document]) -> int:
        """Chunk each source document, then embed and store the chunks in batches."""
        written = 0
        batch: list[Chunk] = []
        for document in documents:
            for chunk in self._processor.process(document):
                batch.append(chunk)
                if len(batch) >= self._batch_size:
                    written += self._embed_and_store(batch)
                    batch = []
        if batch:
            written += self._embed_and_store(batch)
        return written

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        embeddings = self._embedder.embed([chunk.content for chunk in chunks])
        pairs: list[tuple[Document, list[float]]] = [
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        return self._vector_store.upsert(pairs)
