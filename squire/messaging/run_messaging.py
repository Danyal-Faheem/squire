# SPDX-License-Identifier: AGPL-3.0-or-later
"""The messaging service entrypoint.

This wires the two messaging behaviours to their live triggers: a scheduled digest of
open work items (APScheduler on the configured cron) and a command surface that
listens for mentions over Slack Socket Mode and routes each through the shared command
core. The wiring is thin; the behaviour lives in the composition root, the command
core, and the digest builder, which are tested directly. ``publish_digest_now`` is
factored out so the digest's database-backed publishing path is testable on its own.
"""

from __future__ import annotations

import logging

from squire.common.config import get_settings
from squire.common.db import session_scope
from squire.factory.store import PgTrackedItemStore
from squire.messaging.base import MessagingAdapter
from squire.messaging.digest import publish_digest
from squire.runtime import Runtime

logger = logging.getLogger(__name__)


def publish_digest_now(runtime: Runtime, adapter: MessagingAdapter) -> str | None:
    """Publish the open-items digest to the configured channel, if one is set."""
    channel = get_settings().messaging_digest_channel
    if not channel:
        logger.warning("no MESSAGING_DIGEST_CHANNEL configured; skipping digest")
        return None
    with session_scope() as session:
        return publish_digest(adapter, channel, PgTrackedItemStore(session))


def serve(runtime: Runtime | None = None) -> None:  # pragma: no cover - live wiring
    """Run the live messaging service: schedule the digest and listen for mentions.

    This is operational glue around tested components, so it is not unit-tested: it
    schedules :func:`publish_digest_now` on ``MESSAGING_DIGEST_CRON`` via APScheduler
    and starts Slack Socket Mode, routing each ``app_mention`` through a per-event
    command surface built by the composition root.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    settings = get_settings()
    runtime = runtime or Runtime(settings)
    adapter = runtime.build_messaging_adapter()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: publish_digest_now(runtime, adapter),
        CronTrigger.from_crontab(settings.messaging_digest_cron),
    )
    scheduler.start()

    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def _on_mention(event: dict[str, object], say: object) -> None:
        with session_scope() as session:
            surface = runtime.build_messaging_surface(session, adapter)
            surface.handle_mention(
                text=str(event.get("text", "")),
                actor=str(event.get("user", "")),
                channel=str(event.get("channel", "")),
                thread=str(event.get("thread_ts") or event.get("ts", "")),
            )

    # slack_bolt's SocketModeHandler.start is untyped; the call is live wiring.
    SocketModeHandler(app, settings.slack_app_token).start()  # type: ignore[no-untyped-call]


def main() -> int:  # pragma: no cover - live wiring
    """Console entrypoint for the messaging service."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    serve()
    return 0


if __name__ == "__main__":  # pragma: no cover - live wiring
    raise SystemExit(main())
