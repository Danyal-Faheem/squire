# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures.

These fixtures let the Phase 0 unit tests run fully offline: no model endpoint, no
embedding endpoint, no database. Later phases add an opt-in Postgres fixture for
``@pytest.mark.integration`` tests.
"""

from __future__ import annotations

import pytest

from squire.common.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance built from explicit test values.

    Building Settings directly (rather than from the environment) keeps tests
    hermetic and independent of any local ``.env``.
    """
    return Settings(
        llm_base_url="http://test-llm/v1",
        embedding_base_url="http://test-embed/v1",
        embedding_dim=4,
        database_url="postgresql+psycopg://test:test@localhost:5432/test",
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure the cached global Settings never leaks between tests."""
    get_settings.cache_clear()
