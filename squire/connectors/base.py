# SPDX-License-Identifier: AGPL-3.0-or-later
"""The connector contract: the engine's normalization boundary for any source.

A *connector* is a pluggable adapter for one source. It is also the place where a
source's native shape (HTML, markdown, forum JSON, an issue tracker's API) is
translated into Squire's internal vocabulary so that nothing downstream ever has to
know where a piece of content came from. Everything inside the engine speaks
:class:`~squire.common.schemas.Document` and :class:`~squire.common.schemas.Signal`;
each connector is a translator at the border.

A connector has two responsibilities, either of which may be a no-op for a given
source:

* **Document ingestion** (:meth:`Connector.iter_documents`) — yield documents to
  embed into the knowledge base. Every source does this.
* **Signal emission** (:meth:`Connector.iter_signals`) — yield new, actionable items
  the work-item factory may turn into board tasks. Only sources that represent
  incoming work do this; static document sources yield nothing.

The two are deliberately separate methods because a source varies along two axes:
*where* it is and *what kind of thing* it provides. A repository provides both
knowledge and work; a docs site provides only knowledge; a forum provides both.

``Document`` and ``Signal`` are defined once in :mod:`squire.common.schemas` (they
are shared with the data layer) and are re-exported here so that the connector
contract reads as a single, self-contained unit.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Re-exported so the connector contract is self-contained; the canonical
# definitions live in the shared schema module and are used by the data layer too.
from squire.common.schemas import Document, Signal

__all__ = [
    "Connector",
    "ConnectorConfig",
    "Cursor",
    "Document",
    "Signal",
]


class Cursor(BaseModel):
    """An opaque, connector-defined incremental watermark.

    The engine treats ``value`` as opaque text: it persists it verbatim in the
    ``connector_cursors`` bookkeeping table and hands it back to the connector on the
    next run. Each connector alone decides what to encode in it — a timestamp, an
    object id, an encoded composite of several sub-watermarks — and how to interpret
    it. Keeping the cursor opaque means the bookkeeping is identical for every
    connector regardless of how its source paginates, which is what lets the
    orchestrator stay source-agnostic.
    """

    value: str


class ConnectorConfig(BaseModel):
    """The configuration for one activated connector instance.

    ``key`` selects the connector class from the registry. ``instance_id`` makes a
    single deployment able to run two instances of the same connector type (for
    example two GitHub repositories or two forums) without their bookkeeping
    colliding — it is the second half of the ``connector_cursors`` primary key.
    ``settings`` is the connector-specific block; each connector validates it into
    its own typed model, so the engine never needs to understand its shape.
    """

    key: str
    instance_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    """The structural contract every connector satisfies.

    A connector is constructed from a :class:`ConnectorConfig` and thereafter answers
    three questions: is it configured correctly (:meth:`healthcheck`), what documents
    does its source hold (:meth:`iter_documents`), and what new actionable items does
    it carry (:meth:`iter_signals`). After a run, :meth:`current_cursor` reports the
    watermark reflecting everything iterated, which the orchestrator persists so the
    next run can resume incrementally.
    """

    #: The registry key this connector type registers under (e.g. ``"github"``).
    key: ClassVar[str]
    #: Unique within a deployment; the second half of the cursor bookkeeping key.
    instance_id: str

    def __init__(self, config: ConnectorConfig) -> None:
        """Build the connector from its configuration block."""
        ...

    def healthcheck(self) -> None:
        """Validate configuration and access early, raising on any problem.

        Called before a run so misconfiguration (a missing directory, an
        unreachable repository, bad credentials) fails fast and loudly rather than
        midway through ingestion.
        """
        ...

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Yield documents to index, optionally only those newer than ``since``.

        ``since`` is the cursor persisted after the previous run, or ``None`` for a
        full backfill. Connectors should treat it as a best-effort lower bound;
        correctness does not depend on it because the knowledge base de-duplicates on
        content hash.
        """
        ...

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """Yield new actionable items, optionally only those newer than ``since``.

        Document-only sources return an empty iterator.
        """
        ...

    def current_cursor(self) -> Cursor | None:
        """Return the watermark reflecting everything iterated so far this run.

        Called by the orchestrator after both iterators have been drained, so the
        returned cursor must not advance past content the connector actually yielded.
        Returns ``None`` when the connector has no meaningful watermark (for example
        because nothing new was seen), in which case the stored cursor is left
        unchanged.
        """
        ...
