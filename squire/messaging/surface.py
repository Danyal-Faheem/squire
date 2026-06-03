# SPDX-License-Identifier: AGPL-3.0-or-later
"""The messaging command surface.

This is the messaging half of the two-way command interface, and it deliberately
mirrors the board surface: it normalises a mention into the same
:class:`~squire.commands.core.Command`, runs it through the *same*
:class:`~squire.commands.core.CommandCore`, and replies in the same thread. There is
no second copy of the command logic — only the platform the message arrives on and
where the reply goes differ from the board surface.
"""

from __future__ import annotations

from squire.commands.core import Command, CommandCore, parse_command
from squire.messaging.base import MessagingAdapter


class MessagingSurface:
    """Routes messaging-platform mentions through the shared command core."""

    SURFACE = "messaging"

    def __init__(
        self,
        *,
        core: CommandCore,
        adapter: MessagingAdapter,
        mention_handle: str,
        pr_keyword: str,
    ) -> None:
        self._core = core
        self._adapter = adapter
        self._mention_handle = mention_handle
        self._pr_keyword = pr_keyword

    def handle_mention(
        self,
        *,
        text: str,
        actor: str,
        channel: str,
        thread: str,
        tracked_item_id: int | None = None,
    ) -> str:
        """Handle a mention and reply in its thread; returns the reply text."""
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
        self._adapter.reply(channel, thread, result.reply)
        return result.reply
