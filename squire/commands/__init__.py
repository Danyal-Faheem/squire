# SPDX-License-Identifier: AGPL-3.0-or-later
"""The command interface: how maintainers drive Squire on a work item.

A surface-agnostic core handles three commands — ask a question, ask for context, and
open a PR — and is shared by every surface. Each surface (the board here; messaging in
Phase 5) normalises an incoming message into the same :class:`~squire.commands.core.
Command` and calls the same :class:`~squire.commands.core.CommandCore`, so the answers
are identical wherever the message originated. Open-PR is gated by ``AUTOPR_ENABLED``
and only queues a request; the agent that acts on it arrives in Phase 7.
"""

from __future__ import annotations

from squire.commands.board_surface import BoardSurface
from squire.commands.core import (
    Command,
    CommandCore,
    CommandKind,
    CommandResult,
    Querier,
    parse_command,
)
from squire.commands.store import CommandStore, PgCommandStore

__all__ = [
    "BoardSurface",
    "Command",
    "CommandCore",
    "CommandKind",
    "CommandResult",
    "CommandStore",
    "PgCommandStore",
    "Querier",
    "parse_command",
]
