# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for PgCursorStore.

The store queries through the typed ORM, so its contract is verified against a real
Postgres rather than by matching SQL strings. These are ``@integration`` tests that
skip when no database is reachable. They cover the watermark round-trip, the upsert on
conflict, and the independence of two instances of the same connector type.
"""

from __future__ import annotations

import pytest

from squire.common.db import session_scope
from squire.connectors.base import Cursor
from squire.ingestion.cursors import PgCursorStore

pytestmark = pytest.mark.integration


def test_get_returns_none_when_absent(postgres_ready: None) -> None:
    """An instance with no stored watermark reads back as None."""
    with session_scope() as session:
        assert PgCursorStore(session).get("github", "main") is None


def test_set_then_get_roundtrips(postgres_ready: None) -> None:
    """A watermark written in one session is read back in another."""
    with session_scope() as session:
        PgCursorStore(session).set(
            "github", "main", Cursor(value="2026-01-01T00:00:00")
        )
    with session_scope() as session:
        assert PgCursorStore(session).get("github", "main") == Cursor(
            value="2026-01-01T00:00:00"
        )


def test_set_overwrites_on_conflict(postgres_ready: None) -> None:
    """Setting the watermark again updates it in place rather than duplicating."""
    with session_scope() as session:
        store = PgCursorStore(session)
        store.set("github", "main", Cursor(value="v1"))
        store.set("github", "main", Cursor(value="v2"))
    with session_scope() as session:
        assert PgCursorStore(session).get("github", "main") == Cursor(value="v2")


def test_instances_keep_independent_watermarks(postgres_ready: None) -> None:
    """Two instances of one connector type track progress independently."""
    with session_scope() as session:
        store = PgCursorStore(session)
        store.set("github", "repo-a", Cursor(value="a"))
        store.set("github", "repo-b", Cursor(value="b"))
    with session_scope() as session:
        store = PgCursorStore(session)
        assert store.get("github", "repo-a") == Cursor(value="a")
        assert store.get("github", "repo-b") == Cursor(value="b")
