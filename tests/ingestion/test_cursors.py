# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for PgCursorStore.

A fake SQLAlchemy session records statements and parameters, so we assert on the
store's contract — what it asks the database to do and how it maps rows back — without
a real database, mirroring the PgVectorStore tests. The data-layer integration against
a real Postgres lives behind the @integration marker added in a later phase.
"""

from __future__ import annotations

from typing import Any

from squire.connectors.base import Cursor
from squire.ingestion.cursors import PgCursorStore


class _FakeScalarResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeSession:
    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, statement: Any, params: dict[str, Any]) -> Any:
        self.calls.append((str(statement), params))
        return self._results.pop(0)


def test_get_returns_none_when_absent() -> None:
    """An instance with no stored watermark reads back as None."""
    session = _FakeSession([_FakeScalarResult(None)])
    store = PgCursorStore(session)  # type: ignore[arg-type]
    assert store.get("github", "main") is None


def test_get_returns_cursor_when_present() -> None:
    """A stored watermark reads back wrapped in a Cursor."""
    session = _FakeSession([_FakeScalarResult("2026-01-01T00:00:00+00:00")])
    store = PgCursorStore(session)  # type: ignore[arg-type]
    cursor = store.get("github", "main")
    assert cursor == Cursor(value="2026-01-01T00:00:00+00:00")


def test_get_filters_by_key_and_instance() -> None:
    """The lookup is scoped to the connector key and instance id."""
    session = _FakeSession([_FakeScalarResult(None)])
    store = PgCursorStore(session)  # type: ignore[arg-type]
    store.get("github", "main")
    _, params = session.calls[0]
    assert params == {"key": "github", "instance": "main"}


def test_set_upserts_on_conflict() -> None:
    """Persisting a watermark is an idempotent upsert on the composite key."""
    session = _FakeSession([_FakeScalarResult(None)])
    store = PgCursorStore(session)  # type: ignore[arg-type]
    store.set("github", "main", Cursor(value="v2"))
    statement, params = session.calls[0]
    assert "INSERT INTO connector_cursors" in statement
    assert "ON CONFLICT (connector_key, instance_id)" in statement
    assert params == {"key": "github", "instance": "main", "cursor": "v2"}
