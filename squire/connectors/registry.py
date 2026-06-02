# SPDX-License-Identifier: AGPL-3.0-or-later
"""The in-process connector registry.

Connectors are discovered by a string key: each connector class registers itself
under a key (``"github"``, ``"filesystem"``, ...), and a deployment activates
connectors by listing those keys. Adding a new source type is therefore one new class
plus one registration — ingestion, the knowledge base, and the work-item factory
never change. This is the Open/Closed Principle applied to sources.

The registry is intentionally a simple module-level mapping. The contract is shaped
so it *could* later be exposed through Python entry points for third-party packages,
but that machinery is deliberately not built yet (YAGNI).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from squire.connectors.base import Connector, ConnectorConfig

# A registered connector class is, from the registry's point of view, a factory that
# turns a config into a connector. Storing the class as a callable (rather than as
# ``type[Connector]``) keeps mypy happy: a ``Protocol`` cannot be instantiated through
# its type, but a concrete connector class is a perfectly good ``Callable``.
ConnectorFactory = Callable[[ConnectorConfig], Connector]

_REGISTRY: dict[str, ConnectorFactory] = {}

C = TypeVar("C", bound=Connector)


class ConnectorRegistryError(Exception):
    """Base class for registry errors."""


class DuplicateConnectorError(ConnectorRegistryError):
    """Raised when two connector classes try to claim the same key."""


class UnknownConnectorError(ConnectorRegistryError):
    """Raised when a deployment activates a connector key that is not registered."""


def register(cls: type[C]) -> type[C]:
    """Register a connector class under its declared ``key``.

    Intended as a class decorator. The class must satisfy the :class:`Connector`
    contract (enforced structurally by the type bound) and declare a non-empty
    ``key``. Re-registering a key raises rather than silently shadowing, so a typo
    or a collision surfaces at import time.
    """
    key = cls.key
    if not key:
        raise ConnectorRegistryError(
            f"{cls.__name__} must declare a non-empty 'key' to be registered"
        )
    if key in _REGISTRY:
        existing = getattr(_REGISTRY[key], "__name__", repr(_REGISTRY[key]))
        raise DuplicateConnectorError(
            f"connector key {key!r} is already registered to {existing}"
        )
    _REGISTRY[key] = cls
    return cls


def get(key: str) -> ConnectorFactory:
    """Return the connector class registered under ``key``.

    Raises :class:`UnknownConnectorError` if the key is not registered.
    """
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise UnknownConnectorError(
            f"no connector registered under key {key!r}; available: {sorted(_REGISTRY)}"
        ) from exc


def create(config: ConnectorConfig) -> Connector:
    """Instantiate the connector named by ``config.key`` from ``config``."""
    return get(config.key)(config)


def available() -> list[str]:
    """Return the sorted list of registered connector keys."""
    return sorted(_REGISTRY)


def create_all(configs: Iterable[ConnectorConfig]) -> list[Connector]:
    """Instantiate every connector in ``configs``, preserving order."""
    return [create(config) for config in configs]
