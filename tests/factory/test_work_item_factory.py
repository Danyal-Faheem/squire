# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the work-item factory.

These cover the factory's contract with all collaborators faked: the rich item it
composes, the dedup guarantee (one item per signal, even on reprocessing), the
update-on-change path, the dry-run gate (no writes reach the board or the ledger), and
the classifier seam (swapping the classifier changes classification with no change to
the factory).
"""

from __future__ import annotations

from pathlib import Path

from squire.factory.work_items import FactoryOutcome, WorkItemFactory
from squire.prompts import PromptLibrary
from tests.factory._support import (
    FakeBoard,
    FakeDrafter,
    InMemoryTrackedItemStore,
    StubClassifier,
    classification,
    priorities,
    routing,
    signal,
)


def _factory(
    *,
    classifier: StubClassifier | None = None,
    board: FakeBoard | None = None,
    store: InMemoryTrackedItemStore | None = None,
    drafter: FakeDrafter | None = None,
    dry_run: bool = False,
) -> WorkItemFactory:
    return WorkItemFactory(
        classifier=classifier or StubClassifier(classification()),
        drafter=drafter or FakeDrafter(),
        board=board or FakeBoard(),
        store=store or InMemoryTrackedItemStore(),
        routing=routing(),
        priorities=priorities(),
        dry_run=dry_run,
    )


def test_creates_one_enriched_item_with_all_fields() -> None:
    """A new signal yields one board item carrying every required field."""
    board = FakeBoard()
    store = InMemoryTrackedItemStore()
    factory = _factory(board=board, store=store)

    result = factory.process(signal())

    assert result.outcome is FactoryOutcome.CREATED
    # [Category] title.
    assert result.title == "[bug] Login is broken on submit"

    created = board.created[0]
    assert created.title == "[bug] Login is broken on submit"
    assert created.repo == "acme/widget"
    # Owner(s) from routing; priority label + category label.
    assert created.assignees == ["team-core", "team-qa"]
    assert created.labels == ["priority: critical", "bug"]
    # Body has summary, suggested answer, background (sources), and a link back.
    assert "**Summary**" in created.body and "Login fails" in created.body
    assert "Suggested reply [1]." in created.body
    assert "https://src/1" in created.body
    assert "Source: https://example.test/acme/widget/issues/1" in created.body

    # Recorded in the ledger with the board reference and classification.
    tracked = store.get("github_issue", "acme/widget#1")
    assert tracked is not None
    assert tracked.board_item_ref == result.board_item_ref == "acme/widget#1"
    assert tracked.category == "bug"
    assert tracked.priority == "p0"
    assert tracked.assigned_owner == "team-core"


def test_reprocessing_unchanged_signal_is_a_noop() -> None:
    """The same signal seen again creates no second item and re-runs nothing."""
    board = FakeBoard()
    store = InMemoryTrackedItemStore()
    classifier = StubClassifier(classification())
    factory = _factory(classifier=classifier, board=board, store=store)

    first = factory.process(signal(content_hash="h1"))
    second = factory.process(signal(content_hash="h1"))

    assert first.outcome is FactoryOutcome.CREATED
    assert second.outcome is FactoryOutcome.DUPLICATE
    assert len(board.created) == 1  # exactly one board item
    assert len(store.items) == 1
    assert classifier.calls == 1  # the no-op path does not even classify


def test_changed_signal_updates_without_duplicating() -> None:
    """A changed source updates the existing item rather than creating a new one."""
    board = FakeBoard()
    store = InMemoryTrackedItemStore()
    factory = _factory(board=board, store=store)

    factory.process(signal(content_hash="h1"))
    result = factory.process(signal(content_hash="h2", title="Login fails (still)"))

    assert result.outcome is FactoryOutcome.UPDATED
    assert len(board.created) == 1  # no duplicate board item
    tracked = store.get("github_issue", "acme/widget#1")
    assert tracked is not None
    assert tracked.content_hash == "h2"


def test_changed_signal_is_reflected_on_the_board_item() -> None:
    """A change posts a note on the board item, not only in the ledger."""
    board = FakeBoard()
    factory = _factory(board=board, store=InMemoryTrackedItemStore())

    created = factory.process(signal(content_hash="h1"))
    factory.process(signal(content_hash="h2", title="Login fails (still)"))

    assert board.comments  # the change was reflected on the item
    ref, note = board.comments[0]
    assert ref == created.board_item_ref
    assert "changed" in note.lower()


def test_dry_run_change_posts_no_comment() -> None:
    """Under dry-run, a detected change posts nothing to the board."""
    store = InMemoryTrackedItemStore()
    # Seed an existing tracked item so the next process is an update, not a create.
    _factory(store=store, dry_run=False).process(signal(content_hash="h1"))

    board = FakeBoard()
    dry = _factory(board=board, store=store, dry_run=True)
    result = dry.process(signal(content_hash="h2"))

    assert result.outcome is FactoryOutcome.DRY_RUN
    assert board.comments == []


def test_dry_run_performs_no_writes() -> None:
    """With dry-run on, nothing reaches the board and nothing is persisted."""
    board = FakeBoard()
    store = InMemoryTrackedItemStore()
    factory = _factory(board=board, store=store, dry_run=True)

    result = factory.process(signal())

    assert result.outcome is FactoryOutcome.DRY_RUN
    assert board.created == []  # no write reached the board layer
    assert store.items == {}  # nothing recorded in the ledger
    # It still composed the item it *would* have created.
    assert result.title == "[bug] Login is broken on submit"


def test_classifier_swap_changes_output_with_no_factory_change() -> None:
    """Injecting a different classifier changes the item; the factory is unchanged."""
    board = FakeBoard()
    # A stub that classifies everything as an unrouted category at p2.
    stub = StubClassifier(
        classification(category="question", priority_key="p2", owner="ignored")
    )
    factory = _factory(classifier=stub, board=board)

    result = factory.process(signal())

    assert result.title == "[question] Login is broken on submit"
    created = board.created[0]
    # 'question' is unrouted, so the default route is used.
    assert created.repo == "acme/fallback"
    assert created.assignees == ["maintainers"]
    assert created.labels == ["priority: normal", "question"]


def test_body_uses_an_overridden_template(tmp_path: Path) -> None:
    """A deployment's work-item body override changes what the factory creates."""
    overrides = tmp_path / "prompts"
    overrides.mkdir()
    (overrides / "work_item_body.md").write_text(
        "TICKET // $summary // $suggested_answer", encoding="utf-8"
    )
    board = FakeBoard()
    factory = WorkItemFactory(
        classifier=StubClassifier(classification()),
        drafter=FakeDrafter(answer="Try restarting."),
        board=board,
        store=InMemoryTrackedItemStore(),
        routing=routing(),
        priorities=priorities(),
        prompts=PromptLibrary(overrides),
        dry_run=False,
    )

    factory.process(signal())

    assert board.created[0].body == "TICKET // Login fails // Try restarting."


def test_unsupported_draft_still_creates_item_with_note() -> None:
    """When the draft finds no sources, the body says so instead of omitting it."""
    board = FakeBoard()
    factory = _factory(board=board, drafter=FakeDrafter(grounded=False))

    result = factory.process(signal())

    assert result.outcome is FactoryOutcome.CREATED
    assert "No supporting sources" in board.created[0].body
