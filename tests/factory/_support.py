# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared offline fakes for the work-item factory tests.

The factory depends on a classifier, a drafter, a board, and a tracked-item store,
each behind an interface. These fakes stand in for all four so the factory's logic —
dedup, body composition, dry-run gating, routing — is tested in isolation, with the
real persistence covered separately by an integration test.
"""

from __future__ import annotations

from datetime import UTC, datetime

from squire.board.base import NewWorkItem
from squire.common.deployment import Priorities, Priority, Route, Routing
from squire.common.schemas import Classification, Signal
from squire.factory.store import NewTrackedItem, TrackedItem
from squire.rag.schemas import Citation, GroundedAnswer


class StubClassifier:
    """Returns a fixed classification and counts how often it was called."""

    def __init__(self, classification: Classification) -> None:
        self._classification = classification
        self.calls = 0

    def classify(self, signal: Signal) -> Classification:
        self.calls += 1
        return self._classification


class FakeDrafter:
    """Returns a canned draft; ``grounded=False`` yields a no-sources refusal."""

    def __init__(
        self,
        answer: str = "Suggested reply [1].",
        citation_url: str = "https://src/1",
        grounded: bool = True,
    ) -> None:
        self._answer = answer
        self._citation_url = citation_url
        self._grounded = grounded
        self.calls: list[tuple[str, str]] = []

    def draft(
        self,
        title: str,
        body: str = "",
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        self.calls.append((title, body))
        if not self._grounded:
            return GroundedAnswer(answer=self._answer, grounded=False, citations=[])
        return GroundedAnswer(
            answer=self._answer,
            grounded=True,
            citations=[
                Citation(
                    index=1,
                    source_type="docs",
                    source_url=self._citation_url,
                    origin="o",
                    score=0.88,
                )
            ],
        )


class FakeBoard:
    """Records created items and comments; never touches a real backend."""

    def __init__(self) -> None:
        self.created: list[NewWorkItem] = []
        self.comments: list[tuple[str, str]] = []
        self._counter = 0

    def create_item(self, item: NewWorkItem) -> str:
        self._counter += 1
        self.created.append(item)
        return f"{item.repo}#{self._counter}"

    def comment(self, item_ref: str, body: str) -> None:
        self.comments.append((item_ref, body))


class InMemoryTrackedItemStore:
    """A dict-backed tracked-item store for offline factory tests."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], TrackedItem] = {}
        self._id = 0

    def get(self, source_type: str, source_id: str) -> TrackedItem | None:
        return self.items.get((source_type, source_id))

    def create(self, item: NewTrackedItem) -> int:
        self._id += 1
        record = TrackedItem(
            id=self._id,
            **item.model_dump(),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.items[(item.source_type, item.source_id)] = record
        return self._id

    def update_classification(
        self,
        item_id: int,
        *,
        content_hash: str,
        category: str,
        priority: str,
        assigned_owner: str,
    ) -> None:
        for record in self.items.values():
            if record.id == item_id:
                record.content_hash = content_hash
                record.category = category
                record.priority = priority
                record.assigned_owner = assigned_owner
                return


def routing() -> Routing:
    return Routing(
        routes={"bug": Route(owners=["team-core", "team-qa"], repo="acme/widget")},
        default=Route(owners=["maintainers"], repo="acme/fallback"),
    )


def priorities() -> Priorities:
    return Priorities(
        levels=[
            Priority(key="p0", label="priority: critical"),
            Priority(key="p2", label="priority: normal"),
        ],
        default="p2",
    )


def signal(content_hash: str = "h1", title: str = "Login fails") -> Signal:
    return Signal(
        source_type="github_issue",
        source_id="acme/widget#1",
        source_url="https://example.test/acme/widget/issues/1",
        title=title,
        body="It errors on submit.",
        content_hash=content_hash,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def classification(
    category: str = "bug", priority_key: str = "p0", owner: str = "team-core"
) -> Classification:
    return Classification(
        category=category,
        priority_key=priority_key,
        owner=owner,
        suggested_title="Login is broken on submit",
    )
