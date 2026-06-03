# SPDX-License-Identifier: AGPL-3.0-or-later
"""The work-item factory.

For each new signal the factory turns one piece of incoming work into one pre-digested
board item: it deduplicates against the tracked-items ledger, classifies the signal,
composes a ``[Category]`` title and a rich body (a summary, the suggested answer drawn
from the knowledge base, the supporting sources, and a link back to the origin),
creates the item on the board with the routed owners and a priority label, and records
the mapping so the signal is never acted on twice.

Two invariants matter most. **Dedup:** exactly one item per signal — an unchanged
signal seen again is a no-op, recognised via the persisted ledger even across
restarts. **Dry-run honesty:** with dry-run on, the factory performs no outward write
and no ledger write; it composes everything and logs what it would do. The factory
depends only on the Classifier and Board interfaces and a drafting capability, so any
of them can be swapped without touching this logic.

The body template here is the engine default; Phase 5's prompts seam will let a
deployment override it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from squire.board.base import Board, NewWorkItem
from squire.classify.base import Classifier
from squire.common.deployment import Priorities, Route, Routing
from squire.common.schemas import Classification, Signal
from squire.factory.store import NewTrackedItem, TrackedItemStore
from squire.rag.schemas import GroundedAnswer

logger = logging.getLogger(__name__)


class Drafter(Protocol):
    """The drafting capability the factory needs (satisfied by the RAG brain)."""

    def draft(
        self,
        title: str,
        body: str = "",
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """Return a grounded, cited draft reply for an incoming item."""
        ...


class FactoryOutcome(StrEnum):
    """What the factory did with a signal."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    UPDATED = "updated"
    DRY_RUN = "dry_run"


@dataclass
class FactoryResult:
    """The outcome of processing one signal."""

    outcome: FactoryOutcome
    tracked_item_id: int | None = None
    board_item_ref: str | None = None
    title: str | None = None
    classification: Classification | None = None


class WorkItemFactory:
    """Turns signals into board work items, exactly once each."""

    def __init__(
        self,
        *,
        classifier: Classifier,
        drafter: Drafter,
        board: Board,
        store: TrackedItemStore,
        routing: Routing,
        priorities: Priorities,
        dry_run: bool = True,
    ) -> None:
        self._classifier = classifier
        self._drafter = drafter
        self._board = board
        self._store = store
        self._routing = routing
        self._priorities = priorities
        self._dry_run = dry_run

    def process(self, signal: Signal) -> FactoryResult:
        """Process one signal into (at most) one board item."""
        existing = self._store.get(signal.source_type, signal.source_id)
        if existing is not None and existing.content_hash == signal.content_hash:
            # Already handled and unchanged: a true no-op. No classification, no
            # drafting, no writes — this is what keeps re-runs from duplicating.
            logger.info(
                "signal %s/%s already tracked and unchanged; skipping",
                signal.source_type,
                signal.source_id,
            )
            return FactoryResult(
                outcome=FactoryOutcome.DUPLICATE,
                tracked_item_id=existing.id,
                board_item_ref=existing.board_item_ref,
            )

        classification = self._classifier.classify(signal)
        route = self._routing.resolve(classification.category)

        if existing is not None:
            return self._update(
                existing.id, existing.board_item_ref, classification, signal
            )
        return self._create(signal, classification, route)

    def _update(
        self,
        item_id: int,
        board_item_ref: str | None,
        classification: Classification,
        signal: Signal,
    ) -> FactoryResult:
        if self._dry_run:
            logger.info(
                "[dry-run] would update tracked item %d and note the change on %s",
                item_id,
                board_item_ref,
            )
            return FactoryResult(
                outcome=FactoryOutcome.DRY_RUN,
                tracked_item_id=item_id,
                board_item_ref=board_item_ref,
                classification=classification,
            )
        self._store.update_classification(
            item_id,
            content_hash=signal.content_hash,
            category=classification.category,
            priority=classification.priority_key,
            assigned_owner=classification.owner,
        )
        # Reflect the change on the board item itself, not only the ledger, so a
        # maintainer watching the item sees that its source moved.
        if board_item_ref is not None:
            self._board.comment(board_item_ref, _change_note(signal))
        return FactoryResult(
            outcome=FactoryOutcome.UPDATED,
            tracked_item_id=item_id,
            board_item_ref=board_item_ref,
            classification=classification,
        )

    def _create(
        self,
        signal: Signal,
        classification: Classification,
        route: Route,
    ) -> FactoryResult:
        draft = self._drafter.draft(classification.suggested_title, signal.body)
        title = f"[{classification.category}] {classification.suggested_title}"
        body = _compose_body(signal, draft)
        labels = [
            self._priorities.label_for(classification.priority_key),
            classification.category,
        ]
        new_item = NewWorkItem(
            repo=route.repo,
            title=title,
            body=body,
            assignees=list(route.owners),
            labels=labels,
        )

        if self._dry_run:
            logger.info(
                "[dry-run] would create board item %r in %s (assignees=%s, labels=%s)",
                title,
                route.repo,
                route.owners,
                labels,
            )
            return FactoryResult(
                outcome=FactoryOutcome.DRY_RUN,
                title=title,
                classification=classification,
            )

        board_item_ref = self._board.create_item(new_item)
        tracked_item_id = self._store.create(
            NewTrackedItem(
                source_type=signal.source_type,
                source_id=signal.source_id,
                source_url=signal.source_url,
                content_hash=signal.content_hash,
                board_item_ref=board_item_ref,
                category=classification.category,
                priority=classification.priority_key,
                assigned_owner=classification.owner,
            )
        )
        logger.info(
            "created board item %s for signal %s", board_item_ref, signal.source_id
        )
        return FactoryResult(
            outcome=FactoryOutcome.CREATED,
            tracked_item_id=tracked_item_id,
            board_item_ref=board_item_ref,
            title=title,
            classification=classification,
        )


def _change_note(signal: Signal) -> str:
    """A brief note posted on a work item when its source has changed."""
    return (
        "The source for this item changed and has been re-synced from "
        f"{signal.source_url}."
    )


def _compose_body(signal: Signal, draft: GroundedAnswer) -> str:
    """Compose the work-item body: summary, suggested answer, sources, origin link."""
    if draft.citations:
        background = "\n".join(
            f"- {c.source_url} (relevance {c.score:.2f})" for c in draft.citations
        )
    else:
        background = "_No supporting sources were found in the knowledge base._"

    return "\n".join(
        [
            "**Summary**",
            signal.title,
            "",
            "**Suggested answer**",
            draft.answer,
            "",
            "**Background**",
            background,
            "",
            "---",
            f"Source: {signal.source_url}",
        ]
    )
