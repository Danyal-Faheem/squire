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

from sqlalchemy.orm import Session

from squire.common import models


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
        self._session.add(
            models.CommandLog(
                tracked_item_id=tracked_item_id,
                surface=surface,
                actor=actor,
                command=command,
                payload=payload,
            )
        )
        self._session.flush()

    def queue_agent_request(
        self,
        *,
        tracked_item_id: int | None,
        requested_by: str,
        surface: str,
    ) -> int:
        request = models.AgentRequest(
            tracked_item_id=tracked_item_id,
            requested_by=requested_by,
            surface=surface,
        )
        self._session.add(request)
        self._session.flush()  # populate the generated primary key
        return request.id
