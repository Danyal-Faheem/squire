# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: the command store against the real schema.

The command core is unit-tested with a fake store; this confirms the PostgreSQL store
actually matches the migration — that an audit entry and a queued agent request land in
their tables with the expected columns. Skips when no Postgres is reachable.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from squire.commands.store import PgCommandStore
from squire.common.db import session_scope

pytestmark = pytest.mark.integration


def test_log_command_and_queue_agent_request(postgres_ready: None) -> None:
    """Logging a command and queuing a request write rows to the real tables."""
    with session_scope() as session:
        store = PgCommandStore(session)
        store.log_command(
            tracked_item_id=None,
            surface="board",
            actor="alice",
            command="question",
            payload="why does it fail?",
        )
        request_id = store.queue_agent_request(
            tracked_item_id=None, requested_by="alice", surface="board"
        )

    assert request_id > 0

    with session_scope() as session:
        logged = (
            session.execute(
                text("SELECT surface, actor, command, payload FROM command_log")
            )
            .mappings()
            .all()
        )
        requests = (
            session.execute(
                text("SELECT requested_by, surface, status FROM agent_requests")
            )
            .mappings()
            .all()
        )

    assert {
        "surface": "board",
        "actor": "alice",
        "command": "question",
        "payload": "why does it fail?",
    } in [dict(row) for row in logged]
    assert dict(requests[0]) == {
        "requested_by": "alice",
        "surface": "board",
        "status": "queued",
    }
