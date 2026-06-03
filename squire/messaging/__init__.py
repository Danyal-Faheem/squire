# SPDX-License-Identifier: AGPL-3.0-or-later
"""Messaging: the two-way bot surface (seam 7).

The :class:`~squire.messaging.base.MessagingAdapter` interface defines the base
capability every platform meets (mention, post, threaded reply) plus an optional
rich-interactivity capability for approvals; :func:`~squire.messaging.base.
request_approval` negotiates between them. The :class:`~squire.messaging.surface.
MessagingSurface` routes mentions through the same command core the board surface uses,
and :mod:`~squire.messaging.digest` builds the scheduled owner-grouped digest. The
single shipped adapter is Slack (:class:`~squire.messaging.slack.SlackAdapter`); other
platforms (Teams, Discord, Matrix, Mattermost, Telegram) would slot in by implementing
the same interface and are intentionally not built.
"""

from __future__ import annotations

from squire.messaging.base import (
    PLAIN_TEXT_APPROVAL_HINT,
    MessagingAdapter,
    SupportsRichApproval,
    request_approval,
)
from squire.messaging.digest import build_digest, publish_digest
from squire.messaging.surface import MessagingSurface

__all__ = [
    "PLAIN_TEXT_APPROVAL_HINT",
    "MessagingAdapter",
    "MessagingSurface",
    "SupportsRichApproval",
    "build_digest",
    "publish_digest",
    "request_approval",
]
