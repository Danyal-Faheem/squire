# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures for the RAG API tests.

A TestClient over the real FastAPI app, plus automatic teardown of any dependency
overrides a test installs, so the brain wiring can be swapped for fakes per test
without leaking into the next one.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from squire.rag import api


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a TestClient and clear any dependency overrides afterwards."""
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()
