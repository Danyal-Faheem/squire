# SPDX-License-Identifier: AGPL-3.0-or-later
"""The open-work-items digest.

A scheduled summary of open work items, grouped by owner with each owner tagged, so a
maintainer team sees at a glance what is waiting and on whom. The grouping is generic;
the *mention* syntax for tagging an owner is the platform's, so it comes from the
adapter — that is the one platform-specific piece. ``publish_digest`` fetches the open
items and posts the rendered digest; ``build_digest`` is the pure, testable core.
"""

from __future__ import annotations

from squire.factory.store import TrackedItem, TrackedItemStore
from squire.messaging.base import MessagingAdapter

#: Owner key used to group items that have no assigned owner.
_UNASSIGNED = "unassigned"


def build_digest(items: list[TrackedItem], adapter: MessagingAdapter) -> str:
    """Render open items grouped by owner, tagging each owner via the adapter."""
    if not items:
        return "No open work items. :tada:"

    by_owner: dict[str, list[TrackedItem]] = {}
    for item in items:
        by_owner.setdefault(item.assigned_owner or _UNASSIGNED, []).append(item)

    sections = ["*Open work items*"]
    for owner in sorted(by_owner):
        owned = by_owner[owner]
        sections.append(f"\n{adapter.mention(owner)} — {len(owned)} open:")
        for item in owned:
            sections.append(f"• {_line(item)}")
    return "\n".join(sections)


def publish_digest(
    adapter: MessagingAdapter, channel: str, store: TrackedItemStore
) -> str:
    """Build the digest from open items and post it to ``channel``."""
    text = build_digest(store.list_open(), adapter)
    adapter.post(channel, text)
    return text


def _line(item: TrackedItem) -> str:
    category = item.category or "uncategorized"
    priority = item.priority or "unprioritized"
    locator = item.board_item_ref or item.source_url
    return f"[{category}] {locator} ({priority})"
