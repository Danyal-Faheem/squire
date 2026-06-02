# SPDX-License-Identifier: AGPL-3.0-or-later
"""The surface-agnostic command core.

Once a work item exists, maintainers drive Squire by talking to it — on the board or,
later, from a messaging platform. Each surface normalises an incoming message into the
same :class:`Command` and calls the same :class:`CommandCore`, so the behaviour is
identical wherever the command came from. There are three commands: ask a question,
ask for context, and open a PR.

Question and context are answered by the RAG brain, grounded and cited. Open-PR is the
one guarded action: it is gated by ``AUTOPR_ENABLED`` and, even when enabled, only
queues a request here — the coding agent that acts on it arrives in Phase 7. Every
command is written to the audit log regardless of outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from squire.commands.store import CommandStore
from squire.rag.schemas import GroundedAnswer

logger = logging.getLogger(__name__)


class CommandKind(StrEnum):
    """The three things a maintainer can ask of Squire on a work item."""

    QUESTION = "question"
    CONTEXT = "context"
    OPEN_PR = "open-pr"


@dataclass
class Command:
    """A normalised instruction, identical regardless of which surface produced it."""

    kind: CommandKind
    text: str
    actor: str
    surface: str
    tracked_item_id: int | None = None


@dataclass
class CommandResult:
    """The core's response: the reply to show, and whether a PR was queued."""

    reply: str
    agent_request_id: int | None = None


class Querier(Protocol):
    """The querying capability the command core needs (satisfied by the RAG brain)."""

    def query(
        self,
        question: str,
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """Return a grounded, cited answer to a question."""
        ...


def parse_command(
    raw: str, *, mention_handle: str, pr_keyword: str
) -> tuple[CommandKind, str]:
    """Normalise a raw mention into a command kind and its remaining text.

    The bot handle is stripped, then the kind is detected: the PR keyword anywhere
    means open-pr; a leading ``context`` means a context request; otherwise it is a
    question. Detection is case-insensitive and surface-independent.
    """
    text = raw.replace(mention_handle, " ").strip()
    lowered = text.lower()
    if pr_keyword.lower() in lowered.split():
        return CommandKind.OPEN_PR, text
    if lowered.startswith("context"):
        return CommandKind.CONTEXT, text[len("context") :].strip()
    return CommandKind.QUESTION, text


class CommandCore:
    """Handles normalised commands, shared by every surface."""

    def __init__(
        self,
        *,
        querier: Querier,
        store: CommandStore,
        autopr_enabled: bool = False,
    ) -> None:
        self._querier = querier
        self._store = store
        self._autopr_enabled = autopr_enabled

    def handle(self, command: Command) -> CommandResult:
        """Dispatch a command, logging it and returning the reply to show."""
        self._store.log_command(
            tracked_item_id=command.tracked_item_id,
            surface=command.surface,
            actor=command.actor,
            command=str(command.kind),
            payload=command.text,
        )
        if command.kind is CommandKind.OPEN_PR:
            return self._open_pr(command)
        return self._answer(command)

    def _answer(self, command: Command) -> CommandResult:
        answer = self._querier.query(command.text)
        return CommandResult(reply=_format_reply(answer))

    def _open_pr(self, command: Command) -> CommandResult:
        if not self._autopr_enabled:
            return CommandResult(
                reply=(
                    "Opening pull requests is disabled here "
                    "(set AUTOPR_ENABLED to enable it)."
                )
            )
        request_id = self._store.queue_agent_request(
            tracked_item_id=command.tracked_item_id,
            requested_by=command.actor,
            surface=command.surface,
        )
        logger.info("queued agent request %d for %s", request_id, command.actor)
        return CommandResult(
            reply=(
                "A draft pull request has been requested and queued; "
                "you'll get the link once it's ready."
            ),
            agent_request_id=request_id,
        )


def _format_reply(answer: GroundedAnswer) -> str:
    """Render a grounded answer as a reply, appending its sources when present."""
    if not answer.citations:
        return answer.answer
    sources = "\n".join(f"- {c.source_url}" for c in answer.citations)
    return f"{answer.answer}\n\nSources:\n{sources}"
