# SPDX-License-Identifier: AGPL-3.0-or-later
"""The messaging contract (seam 7) and capability negotiation.

A messaging surface is not a connector: a connector is a one-way inbound pipe, whereas
a messaging platform is a two-way conversation that must send, receive a message
normalised into a command, and reply in the same thread. So it has its own interface.

Every platform meets a *base* capability — produce a user mention, post a message,
reply in a thread — and may *optionally* declare rich interactivity (buttons/cards).
Approval rides on that: a platform that declares rich interactivity and implements
:class:`SupportsRichApproval` gets buttons; every other platform falls back to a plain
-text "approve" prompt. The fallback decision lives here in the core, not in each
adapter, so adding a new platform never has to reimplement it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

#: Appended to a plain-text approval prompt on platforms without rich interactivity.
PLAIN_TEXT_APPROVAL_HINT = "Reply `approve` to proceed."


@runtime_checkable
class MessagingAdapter(Protocol):
    """The base capability every messaging platform must provide."""

    @property
    def supports_rich_interactivity(self) -> bool:
        """Whether the platform can present buttons/cards (vs. plain text only)."""
        ...

    def mention(self, user: str) -> str:
        """Render a user as a platform mention (so a digest can tag owners)."""
        ...

    def post(self, channel: str, text: str) -> str:
        """Post a top-level message to a channel; return a thread reference."""
        ...

    def reply(self, channel: str, thread: str, text: str) -> None:
        """Post a reply in an existing thread."""
        ...


@runtime_checkable
class SupportsRichApproval(Protocol):
    """The optional rich-interactivity capability for approvals (buttons/cards)."""

    def present_approval(self, channel: str, thread: str, prompt: str) -> None:
        """Present an interactive approval (e.g. Approve/Decline buttons)."""
        ...


def request_approval(
    adapter: MessagingAdapter, channel: str, thread: str, prompt: str
) -> None:
    """Ask for approval, using rich controls when available and plain text otherwise.

    This is the capability negotiation: a platform that declares rich interactivity and
    implements :class:`SupportsRichApproval` is asked to present interactive controls;
    any other platform receives a plain-text prompt to which a maintainer replies
    "approve". Either way, approval works.
    """
    if adapter.supports_rich_interactivity and isinstance(
        adapter, SupportsRichApproval
    ):
        adapter.present_approval(channel, thread, prompt)
    else:
        adapter.reply(channel, thread, f"{prompt}\n\n{PLAIN_TEXT_APPROVAL_HINT}")
