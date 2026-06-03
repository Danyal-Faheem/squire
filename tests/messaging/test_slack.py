# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Slack adapter and the Slack command surface, all offline.

The Slack Web client is faked and injected, so these cover the adapter's formatting and
posting, the rich-vs-plain approval negotiation through it, the owner-tagged digest in
Slack's mention syntax, and the command surface delivering the command core's reply
into a Slack thread.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from squire.commands.core import CommandCore
from squire.factory.store import TrackedItem
from squire.messaging import (
    MessagingSurface,
    SlackAdapter,
    build_digest,
    request_approval,
)


class FakeSlackClient:
    """Records chat_postMessage calls and returns a timestamp."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"ts": "1700000000.000100"}


class FakeQuerier:
    def query(self, question: str, **kwargs: Any) -> Any:
        from squire.rag.schemas import GroundedAnswer

        return GroundedAnswer(
            answer=f"Answer to: {question}", grounded=False, citations=[]
        )


class RecordingCommandStore:
    def log_command(self, **kwargs: Any) -> None:
        return None

    def queue_agent_request(self, **kwargs: Any) -> int:
        return 1


def _adapter(*, rich: bool = True) -> tuple[SlackAdapter, FakeSlackClient]:
    client = FakeSlackClient()
    return SlackAdapter(client=client, rich_interactivity=rich), client


def test_mention_uses_slack_syntax() -> None:
    adapter, _ = _adapter()
    assert adapter.mention("U123") == "<@U123>"


def test_post_returns_thread_ref_and_reply_threads_under_it() -> None:
    """post returns the message ts; reply posts under that thread_ts."""
    adapter, client = _adapter()

    ts = adapter.post("C1", "hello")
    adapter.reply("C1", ts, "a reply")

    assert ts == "1700000000.000100"
    assert client.calls[0] == {"channel": "C1", "text": "hello"}
    assert client.calls[1]["thread_ts"] == ts
    assert client.calls[1]["text"] == "a reply"


def test_rich_approval_posts_buttons() -> None:
    """With rich interactivity, approval is presented as Block Kit buttons."""
    adapter, client = _adapter(rich=True)
    request_approval(adapter, "C1", "T1", "Open a draft PR?")

    call = client.calls[0]
    assert call["thread_ts"] == "T1"
    action_ids = {
        element["action_id"]
        for block in call["blocks"]
        if block["type"] == "actions"
        for element in block["elements"]
    }
    assert action_ids == {"squire_approve", "squire_decline"}


def test_rich_disabled_falls_back_to_plain_text() -> None:
    """With rich interactivity off, approval is a plain-text reply (no blocks)."""
    adapter, client = _adapter(rich=False)
    request_approval(adapter, "C1", "T1", "Open a draft PR?")

    call = client.calls[0]
    assert "blocks" not in call
    assert "Open a draft PR?" in call["text"]
    assert "approve" in call["text"].lower()


def test_digest_tags_owners_with_slack_mentions() -> None:
    adapter, _ = _adapter()
    items = [
        TrackedItem(
            id=1,
            source_type="github_issue",
            source_id="o/n#1",
            source_url="https://example.test/1",
            content_hash="h",
            board_item_ref="acme/widget#1",
            category="bug",
            priority="p0",
            assigned_owner="U999",
            status="created",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    ]
    digest = build_digest(items, adapter)
    assert "<@U999> — 1 open:" in digest


def test_command_surface_replies_into_slack_thread() -> None:
    """A mention routed through the command core is replied into the Slack thread."""
    adapter, client = _adapter()
    core = CommandCore(querier=FakeQuerier(), store=RecordingCommandStore())
    surface = MessagingSurface(
        core=core, adapter=adapter, mention_handle="@squire", pr_keyword="open-pr"
    )

    reply = surface.handle_mention(
        text="@squire why is it broken?", actor="U1", channel="C1", thread="T1"
    )

    assert "why is it broken?" in reply
    assert client.calls[0]["channel"] == "C1"
    assert client.calls[0]["thread_ts"] == "T1"
    assert client.calls[0]["text"] == reply
