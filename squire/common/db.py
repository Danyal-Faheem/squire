# SPDX-License-Identifier: AGPL-3.0-or-later
"""Database engine and session management.

This module owns the single SQLAlchemy engine and session factory for the engine.
PostgreSQL is Squire's fixed primary datastore and is intentionally not abstracted;
only the vector operations are placed behind an interface (see
:mod:`squire.common.vectorstore`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from squire.common.config import Settings, get_settings


def make_engine(settings: Settings | None = None):  # type: ignore[no-untyped-def]
    """Create a SQLAlchemy engine from settings.

    The return type is left to SQLAlchemy's inference; callers do not need to name
    it. ``pool_pre_ping`` guards against stale connections in long-running services
    such as the ingestion live updater.
    """
    settings = settings or get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def _session_factory() -> sessionmaker[Session]:
    """Return a lazily-initialized session factory bound to the engine."""
    global _engine, _SessionFactory
    if _SessionFactory is None:
        _engine = make_engine()
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope.

    Commits on success, rolls back on exception, and always closes the session.
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
