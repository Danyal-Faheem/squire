# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the platform-agnostic messaging layer.

These use fake adapters (no Slack, no network) to cover the parts that are not
platform-specific: capability negotiation for approval (rich buttons when declared,
plain-text fallback otherwise), the owner-grouped digest, and — most importantly —
that the messaging command surface returns the *same* answers as the board surface
because both route through the same command core.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from squire.commands.board_surface import BoardSurface
from squire.commands.core import CommandCore
from squire.factory.store import TrackedItem
from squire.messaging import (
    PLAIN_TEXT_APPROVAL_HINT,
    MessagingSurface,
    build_digest,
    request_approval,
)

# --- fakes -------------------------------------------------------------------------


class FakeQuerier:
    def __init__(self, answer: str = "Restart it.") -> None:
        self._answer = answer

    def query(
        self,
        question: str,
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> Any:
        from squire.rag.schemas import Citation, GroundedAnswer

        return GroundedAnswer(
            answer=self._answer,
            grounded=True,
            citations=[
                Citation(
                    index=1, source_type="docs", source_url="https://src/1", score=0.9
                )
            ],
        )


class RecordingCommandStore:
    def log_command(self, **kwargs: Any) -> None:
        return None

    def queue_agent_request(self, **kwargs: Any) -> int:
        return 1


class BaseAdapter:
    """A messaging adapter with only the base capability (no rich interactivity)."""

    supports_rich_interactivity = False

    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []
        self.replies: list[tuple[str, str, str]] = []

    def mention(self, user: str) -> str:
        return f"@{user}"

    def post(self, channel: str, text: str) -> str:
        self.posts.append((channel, text))
        return "thread-1"

    def reply(self, channel: str, thread: str, text: str) -> None:
        self.replies.append((channel, thread, text))


class RichAdapter(BaseAdapter):
    """A messaging adapter that declares and implements rich approval."""

    supports_rich_interactivity = True

    def __init__(self) -> None:
        super().__init__()
        self.approvals: list[tuple[str, str, str]] = []

    def present_approval(self, channel: str, thread: str, prompt: str) -> None:
        self.approvals.append((channel, thread, prompt))


class _NoopBoard:
    def create_item(self, item: Any) -> str:
        return "unused#0"

    def comment(self, item_ref: str, body: str) -> None:
        self.commented = (item_ref, body)


def _core() -> CommandCore:
    return CommandCore(querier=FakeQuerier(), store=RecordingCommandStore())


def _tracked(owner: str, *, category: str = "bug", priority: str = "p2") -> TrackedItem:
    return TrackedItem(
        id=1,
        source_type="github_issue",
        source_id=f"o/n#{owner}",
        source_url=f"https://example.test/{owner}",
        content_hash="h",
        board_item_ref=f"acme/widget#{owner}",
        category=category,
        priority=priority,
        assigned_owner=owner,
        status="created",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --- capability negotiation --------------------------------------------------------


def test_plain_text_approval_fallback() -> None:
    """Without rich interactivity, approval is a plain-text reply that still works."""
    adapter = BaseAdapter()
    request_approval(adapter, "C1", "T1", "Open a draft PR?")
    assert len(adapter.replies) == 1
    _, _, text = adapter.replies[0]
    assert "Open a draft PR?" in text
    assert PLAIN_TEXT_APPROVAL_HINT in text


def test_rich_approval_used_when_available() -> None:
    """With rich interactivity declared, approval uses the interactive control."""
    adapter = RichAdapter()
    request_approval(adapter, "C1", "T1", "Open a draft PR?")
    assert adapter.approvals == [("C1", "T1", "Open a draft PR?")]
    assert adapter.replies == []  # no plain-text fallback


# --- digest ------------------------------------------------------------------------


def test_digest_groups_by_owner_and_tags_them() -> None:
    """The digest groups open items by owner and tags each via the adapter."""
    adapter = BaseAdapter()
    items = [_tracked("alice"), _tracked("bob"), _tracked("alice")]
    digest = build_digest(items, adapter)
    assert "@alice — 2 open:" in digest
    assert "@bob — 1 open:" in digest
    assert "acme/widget#alice" in digest


def test_empty_digest_is_friendly() -> None:
    assert "No open work items" in build_digest([], BaseAdapter())


# --- same command core as the board surface ----------------------------------------


def test_messaging_returns_same_answer_as_board() -> None:
    """A mention via messaging yields the same reply the board surface would."""
    core = _core()  # one core, shared by both surfaces
    board_adapter = _NoopBoard()
    board = BoardSurface(
        core=core,
        board=board_adapter,
        mention_handle="@squire",
        pr_keyword="open-pr",
    )
    messaging_adapter = BaseAdapter()
    messaging = MessagingSurface(
        core=core,
        adapter=messaging_adapter,
        mention_handle="@squire",
        pr_keyword="open-pr",
    )

    question = "@squire why does login fail?"
    board_reply = board.handle_mention(
        text=question, actor="m", board_item_ref="acme/widget#7"
    )
    messaging_reply = messaging.handle_mention(
        text=question, actor="m", channel="C1", thread="T1"
    )

    assert messaging_reply == board_reply
    # And the messaging reply was posted back in its thread.
    assert messaging_adapter.replies[0] == ("C1", "T1", messaging_reply)
