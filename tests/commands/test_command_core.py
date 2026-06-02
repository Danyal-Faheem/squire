# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the surface-agnostic command core and the board surface.

These cover parsing a raw mention into a command, answering question/context via the
brain (grounded, with sources), the open-pr gate (disabled by default; queues a
request only when enabled), the audit log, and the board surface turning a mention
into a reply posted back on the item. All collaborators are faked, so no network or
database is involved.
"""

from __future__ import annotations

from typing import Any

from squire.commands.board_surface import BoardSurface
from squire.commands.core import (
    Command,
    CommandCore,
    CommandKind,
    parse_command,
)
from squire.rag.schemas import Citation, GroundedAnswer


class FakeQuerier:
    """Returns a canned grounded answer and records the questions asked."""

    def __init__(
        self, answer: str = "Restart the service.", grounded: bool = True
    ) -> None:
        self._answer = answer
        self._grounded = grounded
        self.questions: list[str] = []

    def query(
        self,
        question: str,
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        self.questions.append(question)
        citations = (
            [
                Citation(
                    index=1, source_type="docs", source_url="https://src/1", score=0.9
                )
            ]
            if self._grounded
            else []
        )
        return GroundedAnswer(
            answer=self._answer, grounded=self._grounded, citations=citations
        )


class RecordingCommandStore:
    """Records audit entries and queued agent requests."""

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.agent_requests: list[dict[str, Any]] = []

    def log_command(
        self,
        *,
        tracked_item_id: int | None,
        surface: str,
        actor: str,
        command: str,
        payload: str,
    ) -> None:
        self.logged.append(
            {
                "tracked_item_id": tracked_item_id,
                "surface": surface,
                "actor": actor,
                "command": command,
                "payload": payload,
            }
        )

    def queue_agent_request(
        self, *, tracked_item_id: int | None, requested_by: str, surface: str
    ) -> int:
        self.agent_requests.append(
            {
                "tracked_item_id": tracked_item_id,
                "requested_by": requested_by,
                "surface": surface,
            }
        )
        return len(self.agent_requests)


class FakeBoard:
    def __init__(self) -> None:
        self.comments: list[tuple[str, str]] = []

    def create_item(self, item: Any) -> str:  # unused here
        return "unused#0"

    def comment(self, item_ref: str, body: str) -> None:
        self.comments.append((item_ref, body))


# --- parsing -----------------------------------------------------------------------


def test_parse_detects_each_command_kind() -> None:
    """Parsing recognises question, context, and open-pr, stripping the handle."""
    q = parse_command(
        "@squire how do I configure it?", mention_handle="@squire", pr_keyword="open-pr"
    )
    assert q == (CommandKind.QUESTION, "how do I configure it?")

    c = parse_command(
        "@squire context please", mention_handle="@squire", pr_keyword="open-pr"
    )
    assert c == (CommandKind.CONTEXT, "please")

    p = parse_command("@squire open-pr", mention_handle="@squire", pr_keyword="open-pr")
    assert p[0] is CommandKind.OPEN_PR


# --- core --------------------------------------------------------------------------


def _core(
    querier: FakeQuerier | None = None,
    store: RecordingCommandStore | None = None,
    *,
    autopr_enabled: bool = False,
) -> tuple[CommandCore, RecordingCommandStore]:
    store = store or RecordingCommandStore()
    core = CommandCore(
        querier=querier or FakeQuerier(),
        store=store,
        autopr_enabled=autopr_enabled,
    )
    return core, store


def test_question_returns_grounded_reply_with_sources() -> None:
    """A question is answered by the brain and the reply lists its sources."""
    querier = FakeQuerier(answer="Restart it.")
    core, store = _core(querier)

    result = core.handle(
        Command(
            kind=CommandKind.QUESTION,
            text="why broken?",
            actor="alice",
            surface="board",
        )
    )

    assert "Restart it." in result.reply
    assert "https://src/1" in result.reply
    assert querier.questions == ["why broken?"]
    assert store.logged[0]["command"] == "question"


def test_context_is_answered_too() -> None:
    """A context request is answered via the brain like a question."""
    querier = FakeQuerier(answer="Here is the background.")
    core, _ = _core(querier)
    result = core.handle(
        Command(
            kind=CommandKind.CONTEXT, text="this area", actor="bob", surface="board"
        )
    )
    assert "Here is the background." in result.reply


def test_open_pr_disabled_by_default_does_not_queue() -> None:
    """With the flag off, open-pr explains it is disabled and queues nothing."""
    core, store = _core(autopr_enabled=False)
    result = core.handle(
        Command(
            kind=CommandKind.OPEN_PR, text="open-pr", actor="alice", surface="board"
        )
    )
    assert "disabled" in result.reply.lower()
    assert store.agent_requests == []
    assert result.agent_request_id is None


def test_open_pr_enabled_queues_a_request() -> None:
    """With the flag on, open-pr queues a request (the agent itself is Phase 7)."""
    core, store = _core(autopr_enabled=True)
    result = core.handle(
        Command(
            kind=CommandKind.OPEN_PR,
            text="open-pr",
            actor="alice",
            surface="board",
            tracked_item_id=42,
        )
    )
    assert result.agent_request_id == 1
    assert store.agent_requests[0]["tracked_item_id"] == 42
    assert store.agent_requests[0]["requested_by"] == "alice"


def test_every_command_is_audited() -> None:
    """Each command is written to the audit log regardless of kind."""
    core, store = _core(autopr_enabled=True)
    core.handle(
        Command(kind=CommandKind.QUESTION, text="q", actor="a", surface="board")
    )
    core.handle(
        Command(kind=CommandKind.OPEN_PR, text="open-pr", actor="a", surface="board")
    )
    assert [entry["command"] for entry in store.logged] == ["question", "open-pr"]


# --- board surface -----------------------------------------------------------------


def test_board_surface_replies_on_the_item() -> None:
    """A mention on a work item yields a contextual reply posted back on it."""
    board = FakeBoard()
    core, _ = _core(FakeQuerier(answer="Check the config file."))
    surface = BoardSurface(
        core=core, board=board, mention_handle="@squire", pr_keyword="open-pr"
    )

    reply = surface.handle_mention(
        text="@squire why does login fail?",
        actor="maintainer",
        board_item_ref="acme/widget#7",
        tracked_item_id=7,
    )

    assert "Check the config file." in reply
    # The reply was posted back on the same board item.
    assert board.comments == [("acme/widget#7", reply)]


def test_board_surface_routes_open_pr_through_the_core() -> None:
    """An open-pr mention is gated by the core, not the surface."""
    board = FakeBoard()
    core, store = _core(autopr_enabled=False)
    surface = BoardSurface(
        core=core, board=board, mention_handle="@squire", pr_keyword="open-pr"
    )

    reply = surface.handle_mention(
        text="@squire open-pr please",
        actor="maintainer",
        board_item_ref="acme/widget#7",
    )

    assert "disabled" in reply.lower()
    assert store.agent_requests == []
