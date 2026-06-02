# SPDX-License-Identifier: AGPL-3.0-or-later
"""The work-item factory: one signal in, one pre-digested board item out.

The factory ties the engine together for incoming work: it deduplicates a signal
against the tracked-items ledger, classifies it (via the Classifier seam), drafts a
grounded answer (via the RAG brain), and creates a single enriched item on the board
(via the Board seam) with an assigned owner and a priority label — all under the
global dry-run gate. It depends only on those interfaces, so each can be swapped
without changing the factory.
"""

from __future__ import annotations

from squire.factory.store import (
    NewTrackedItem,
    PgTrackedItemStore,
    TrackedItem,
    TrackedItemStore,
)
from squire.factory.work_items import (
    Drafter,
    FactoryOutcome,
    FactoryResult,
    WorkItemFactory,
)

__all__ = [
    "Drafter",
    "FactoryOutcome",
    "FactoryResult",
    "NewTrackedItem",
    "PgTrackedItemStore",
    "TrackedItem",
    "TrackedItemStore",
    "WorkItemFactory",
]
