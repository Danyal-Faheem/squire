# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the connector registry.

The registry is the mechanism that makes a new source type "one class plus a
registration". These tests assert that contract: a class registers under its key, is
discoverable and instantiable by key, collisions and unknown keys fail loudly, and
the two built-in connectors are present.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pytest

from squire.connectors import registry
from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal


class _StubConnector:
    """A minimal connector used to exercise the registry in isolation."""

    key: ClassVar[str] = "stub-for-registry-tests"

    def __init__(self, config: ConnectorConfig) -> None:
        self.instance_id = config.instance_id
        self.config = config

    def healthcheck(self) -> None:
        return None

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        return iter(())

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        return iter(())

    def current_cursor(self) -> Cursor | None:
        return None


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot and restore the registry so a test's registrations don't leak."""
    saved = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_register_then_create_roundtrips(clean_registry: None) -> None:
    """A registered class is instantiable by key, carrying its config through."""
    registry.register(_StubConnector)
    config = ConnectorConfig(key=_StubConnector.key, instance_id="i1")
    connector = registry.create(config)
    assert isinstance(connector, _StubConnector)
    assert connector.instance_id == "i1"


def test_register_is_idempotent_identity(clean_registry: None) -> None:
    """register returns the class unchanged so it works as a decorator."""
    returned = registry.register(_StubConnector)
    assert returned is _StubConnector


def test_duplicate_key_raises(clean_registry: None) -> None:
    """Re-registering a key must fail rather than silently shadow."""
    registry.register(_StubConnector)
    with pytest.raises(registry.DuplicateConnectorError):
        registry.register(_StubConnector)


def test_unknown_key_raises() -> None:
    """Creating from an unregistered key is a clear error."""
    with pytest.raises(registry.UnknownConnectorError):
        registry.create(ConnectorConfig(key="does-not-exist", instance_id="x"))


def test_builtins_are_registered() -> None:
    """Importing the package registers the two Phase 1 connectors."""
    assert "filesystem" in registry.available()
    assert "github" in registry.available()


def test_create_all_preserves_order(clean_registry: None) -> None:
    """create_all instantiates each config in order."""
    registry.register(_StubConnector)
    configs = [
        ConnectorConfig(key=_StubConnector.key, instance_id="a"),
        ConnectorConfig(key=_StubConnector.key, instance_id="b"),
    ]
    connectors = registry.create_all(configs)
    assert [c.instance_id for c in connectors] == ["a", "b"]
