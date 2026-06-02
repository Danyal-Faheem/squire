# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ingestion: drive the activated connectors and keep the knowledge base fresh.

The orchestrator iterates a deployment's connectors in two modes — a full *backfill*
and an incremental *live* update — embedding their documents into the knowledge base
and handing their signals on for processing. Per-connector watermarks make the live
mode incremental and let a backfill resume connector by connector.
"""

from __future__ import annotations

from squire.ingestion.cursors import CursorStore, PgCursorStore
from squire.ingestion.orchestrator import (
    ConnectorRunResult,
    IngestionMode,
    IngestionOrchestrator,
    IngestionResult,
    SignalHandler,
    logging_signal_handler,
)

__all__ = [
    "ConnectorRunResult",
    "CursorStore",
    "IngestionMode",
    "IngestionOrchestrator",
    "IngestionResult",
    "PgCursorStore",
    "SignalHandler",
    "logging_signal_handler",
]
