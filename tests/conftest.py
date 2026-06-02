# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures.

These fixtures let the Phase 0 unit tests run fully offline: no model endpoint, no
embedding endpoint, no database. Later phases add an opt-in Postgres fixture for
``@pytest.mark.integration`` tests.
"""

from __future__ import annotations

from collections.abc import Iterator

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


@pytest.fixture
def postgres_ready() -> Iterator[None]:
    """Provide a migrated, empty database for ``@integration`` tests.

    The fixture skips the test outright if Postgres is unreachable, so the unit
    suite (and CI, which has no database) stays green while a developer with the
    Docker stack up gets real data-layer coverage. It applies migrations and
    truncates the core tables so each integration test starts from a clean slate.
    """
    from sqlalchemy import text

    from squire.common.db import make_engine, session_scope

    try:
        # Any failure to connect (no server, wrong creds) means "skip", not "fail".
        connection = make_engine().connect()
        connection.close()
    except Exception as exc:
        pytest.skip(f"Postgres not reachable for integration test: {exc}")

    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    with session_scope() as session:
        session.execute(
            text(
                "TRUNCATE chunks, connector_cursors, tracked_items, "
                "agent_requests, command_log RESTART IDENTITY CASCADE"
            )
        )
    yield
