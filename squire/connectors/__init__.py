# SPDX-License-Identifier: AGPL-3.0-or-later
"""The connector framework and its built-in connectors.

Importing this package registers every built-in connector under its key, so a
deployment can activate them by name. The connector *contract* lives in
:mod:`squire.connectors.base`; discovery lives in :mod:`squire.connectors.registry`.
"""

from __future__ import annotations

# Importing the built-in connector modules triggers their registration via the
# ``@register`` decorator. They are imported for that side effect.
from squire.connectors import filesystem as filesystem  # noqa: F401
from squire.connectors import github as github  # noqa: F401
from squire.connectors.base import Connector, ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import (
    ConnectorRegistryError,
    DuplicateConnectorError,
    UnknownConnectorError,
    available,
    create,
    create_all,
    get,
    register,
)

__all__ = [
    "Connector",
    "ConnectorConfig",
    "ConnectorRegistryError",
    "Cursor",
    "Document",
    "DuplicateConnectorError",
    "Signal",
    "UnknownConnectorError",
    "available",
    "create",
    "create_all",
    "get",
    "register",
]
