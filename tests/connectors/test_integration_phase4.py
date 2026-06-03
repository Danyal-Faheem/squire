# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: the Phase 4 connectors land their source types in chunks.

The unit tests prove each connector produces correctly-shaped documents and (for
discourse) signals; this proves those documents actually persist as chunks with their
own source-type label when run through the real orchestrator and pgvector store. The
source clients are injected as fakes, so only the database is real. Skips when no
Postgres is reachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from squire.common.db import session_scope
from squire.common.vectorstore import PgVectorStore
from squire.connectors.base import ConnectorConfig
from squire.connectors.discourse import DiscourseConnector
from squire.connectors.docs_site import DocsSiteConnector
from squire.connectors.pdf import PdfConnector
from squire.ingestion.cursors import PgCursorStore
from squire.ingestion.orchestrator import IngestionMode, IngestionOrchestrator
from tests.fakes import make_embedder

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 1024


class _JsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> Any:
        return None

    def json(self) -> Any:
        return self._payload

    @property
    def status_code(self) -> int:
        return 200

    @property
    def text(self) -> str:
        return ""


class _HtmlResponse:
    def __init__(self, html: str) -> None:
        self._html = html

    def raise_for_status(self) -> Any:
        return None

    def json(self) -> Any:
        return {}

    @property
    def status_code(self) -> int:
        return 200

    @property
    def text(self) -> str:
        return self._html


class _FakeJsonHttp:
    def __init__(self, routes: dict[str, dict[str, Any]]) -> None:
        self._routes = routes

    def get(self, url: str, **kwargs: Any) -> _JsonResponse:
        return _JsonResponse(self._routes[url])


class _FakeHtmlHttp:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def get(self, url: str, **kwargs: Any) -> _HtmlResponse:
        return _HtmlResponse(self._pages[url])


def _connectors(tmp_path: Path) -> list[Any]:
    discourse = DiscourseConnector(
        ConnectorConfig(
            key="discourse",
            instance_id="forum",
            settings={"base_url": "https://forum.example"},
        ),
        client=_FakeJsonHttp(
            {
                "/latest.json": {
                    "topic_list": {
                        "topics": [
                            {
                                "id": 1,
                                "slug": "q",
                                "title": "A question",
                                "tags": [],
                                "bumped_at": "2026-01-02T00:00:00Z",
                                "created_at": "2026-01-01T00:00:00Z",
                            }
                        ]
                    }
                },
                "/t/1.json": {
                    "post_stream": {
                        "posts": [{"post_number": 1, "cooked": "<p>Hi</p>"}]
                    }
                },
            }
        ),
    )
    docs = DocsSiteConnector(
        ConnectorConfig(
            key="docs_site",
            instance_id="docs",
            settings={"base_url": "https://docs.example/"},
        ),
        client=_FakeHtmlHttp(
            {"https://docs.example/": "<html><body><p>Docs page.</p></body></html>"}
        ),
    )
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    pdf = PdfConnector(
        ConnectorConfig(
            key="pdf", instance_id="manuals", settings={"paths": [str(pdf_path)]}
        ),
        extractor=lambda path: "# Manual\n\nExtracted text.",
    )
    return [discourse, docs, pdf]


def test_phase4_connectors_land_their_source_types(
    postgres_ready: None, tmp_path: Path
) -> None:
    """discourse, docs_site, and pdf each write chunks tagged with their source type."""
    signals: list[str] = []
    with session_scope() as session:
        orchestrator = IngestionOrchestrator(
            embedder=make_embedder(dim=EMBEDDING_DIM, vector=[0.0] * EMBEDDING_DIM),
            vector_store=PgVectorStore(session),
            cursors=PgCursorStore(session),
            signal_handler=lambda s: signals.append(s.source_type),
        )
        orchestrator.run(_connectors(tmp_path), mode=IngestionMode.BACKFILL)

    with session_scope() as session:
        rows = (
            session.execute(text("SELECT DISTINCT source_type FROM chunks"))
            .scalars()
            .all()
        )

    assert {"forum_post", "docs", "pdf"}.issubset(set(rows))
    # The discourse topic reached the signal path, like a github issue would.
    assert "forum_topic" in signals
