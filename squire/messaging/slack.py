# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Slack messaging adapter — the single shipped messaging backend.

It meets the base capability (mention, post, threaded reply) over the Slack Web API
and declares rich interactivity, presenting approvals as Block Kit buttons. The
capability is configurable: turn it off and the core's approval negotiation falls back
to a plain-text prompt, which is how a platform without buttons would behave.

The Web client is an optional constructor parameter typed as a small Protocol, so the
adapter's formatting and posting are exercised offline against a fake. Receiving
mentions over Socket Mode is a live concern handled by :meth:`serve`, which routes each
mention through the shared command surface; it is wiring, not unit-tested logic.

Other platforms — Teams, Discord, Matrix, Mattermost, Telegram — would each be a new
adapter implementing the same interface; none is built here.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)


class SlackClient(Protocol):
    """The slice of the Slack Web client the adapter uses."""

    def chat_postMessage(self, **kwargs: Any) -> Any: ...


class SlackAdapter:
    """A :class:`~squire.messaging.base.MessagingAdapter` backed by the Slack API."""

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        rich_interactivity: bool = True,
        client: SlackClient | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._rich = rich_interactivity
        self._client = client

    @property
    def supports_rich_interactivity(self) -> bool:
        return self._rich

    def mention(self, user: str) -> str:
        """Render a Slack user mention."""
        return f"<@{user}>"

    def post(self, channel: str, text: str) -> str:
        """Post a top-level message; return its timestamp to thread replies under."""
        response = self._slack().chat_postMessage(channel=channel, text=text)
        return str(response["ts"])

    def reply(self, channel: str, thread: str, text: str) -> None:
        """Post a reply in an existing thread."""
        self._slack().chat_postMessage(channel=channel, thread_ts=thread, text=text)

    def present_approval(self, channel: str, thread: str, prompt: str) -> None:
        """Present an approval as Block Kit buttons in the thread."""
        self._slack().chat_postMessage(
            channel=channel,
            thread_ts=thread,
            text=prompt,
            blocks=_approval_blocks(prompt),
        )

    def _slack(self) -> SlackClient:
        if self._client is None:
            from slack_sdk import WebClient

            self._client = cast("SlackClient", WebClient(token=self._bot_token))
        return self._client


def _approval_blocks(prompt: str) -> list[dict[str, Any]]:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": prompt}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "value": "approve",
                    "action_id": "squire_approve",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Decline"},
                    "value": "decline",
                    "action_id": "squire_decline",
                },
            ],
        },
    ]
