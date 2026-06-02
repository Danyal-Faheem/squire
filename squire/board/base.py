# SPDX-License-Identifier: AGPL-3.0-or-later
"""The board contract (seam 6).

A board is where the maintainers' work items live. The factory creates an item with a
title, a rich body, assignees, and labels; the command core comments on an existing
item to reply in place. Both refer to a created item by an opaque ``board_item_ref``
string, whose format each backend owns (the GitHub board uses ``owner/repo#number``),
so the rest of the engine never needs to know how a particular board identifies items.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class NewWorkItem(BaseModel):
    """A work item to create on a board.

    ``repo`` is the backend-specific target (for GitHub, ``owner/name``); ``assignees``
    and ``labels`` are plain strings the backend maps to its own users and labels.
    """

    repo: str
    title: str
    body: str
    assignees: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class Board(Protocol):
    """Creates and comments on work items."""

    def create_item(self, item: NewWorkItem) -> str:
        """Create ``item`` and return an opaque reference to it."""
        ...

    def comment(self, item_ref: str, body: str) -> None:
        """Post a comment on the item identified by ``item_ref``."""
        ...
