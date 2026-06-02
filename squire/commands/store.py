# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistence for the command interface: the audit log and PR requests.

Two tables back the command core. ``command_log`` is an audit trail of every command
received on any surface — what was asked, by whom, where. ``agent_requests`` records a
request to open a pull request, which the coding agent (Phase 7) will later pick up.
Both are reached through a small :class:`CommandStore` interface with a PostgreSQL
default, the same pattern as the other stores.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session


class CommandStore(Protocol):
    """Records commands and queues pull-request requests."""

    def log_command(
        self,
        *,
        tracked_item_id: int | None,
        surface: str,
        actor: str,
        command: str,
        payload: str,
    ) -> None:
        """Append one command to the audit log."""
        ...

    def queue_agent_request(
        self,
        *,
        tracked_item_id: int | None,
        requested_by: str,
        surface: str,
    ) -> int:
        """Queue a pull-request request and return its id."""
        ...


class PgCommandStore:
    """A :class:`CommandStore` backed by ``command_log`` and ``agent_requests``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def log_command(
        self,
        *,
        tracked_item_id: int | None,
        surface: str,
        actor: str,
        command: str,
        payload: str,
    ) -> None:
        self._session.execute(
            text(
                """
                INSERT INTO command_log (
                    tracked_item_id, surface, actor, command, payload
                )
                VALUES (:tracked_item_id, :surface, :actor, :command, :payload)
                """
            ),
            {
                "tracked_item_id": tracked_item_id,
                "surface": surface,
                "actor": actor,
                "command": command,
                "payload": payload,
            },
        )

    def queue_agent_request(
        self,
        *,
        tracked_item_id: int | None,
        requested_by: str,
        surface: str,
    ) -> int:
        new_id = self._session.execute(
            text(
                """
                INSERT INTO agent_requests (
                    tracked_item_id, requested_by, surface, status
                )
                VALUES (:tracked_item_id, :requested_by, :surface, 'queued')
                RETURNING id
                """
            ),
            {
                "tracked_item_id": tracked_item_id,
                "requested_by": requested_by,
                "surface": surface,
            },
        ).scalar_one()
        return int(new_id)
