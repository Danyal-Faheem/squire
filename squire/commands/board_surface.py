# SPDX-License-Identifier: AGPL-3.0-or-later
"""The board command surface.

This is the board's half of the two-way command interface: it turns a mention on a
work item into a normalised :class:`~squire.commands.core.Command`, runs it through the
shared core, and posts the reply back on the same item. The messaging surface (Phase
5) will do exactly the same against its own platform, calling the identical core — the
only difference is where the message comes from and where the reply goes.

It is told which item a mention belongs to (its board reference) by the caller, since
resolving a platform event to an item is the delivery layer's job, not the core's.
"""

from __future__ import annotations

from squire.board.base import Board
from squire.commands.core import Command, CommandCore, parse_command


class BoardSurface:
    """Routes board mentions through the command core and replies in place."""

    SURFACE = "board"

    def __init__(
        self,
        *,
        core: CommandCore,
        board: Board,
        mention_handle: str,
        pr_keyword: str,
    ) -> None:
        self._core = core
        self._board = board
        self._mention_handle = mention_handle
        self._pr_keyword = pr_keyword

    def handle_mention(
        self,
        *,
        text: str,
        actor: str,
        board_item_ref: str,
        tracked_item_id: int | None = None,
    ) -> str:
        """Handle a mention on a work item and post the reply back on it.

        Returns the reply text (useful for logging and tests). The reply is posted
        through the board, which honours the dry-run gate, so under dry-run the reply
        is computed and returned but not actually posted.
        """
        kind, cleaned = parse_command(
            text, mention_handle=self._mention_handle, pr_keyword=self._pr_keyword
        )
        command = Command(
            kind=kind,
            text=cleaned,
            actor=actor,
            surface=self.SURFACE,
            tracked_item_id=tracked_item_id,
        )
        result = self._core.handle(command)
        self._board.comment(board_item_ref, result.reply)
        return result.reply
