# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the docs-site connector.

The HTTP transport is faked and injected, so the crawl runs offline. The tests cover
the behaviours that matter for a crawler: it follows in-scope links and stays off
out-of-scope and off-host ones, honours the page limit, survives a page that fails to
fetch, extracts readable text (dropping scripts/markup), and emits no signals.
"""

from __future__ import annotations

from typing import Any

from squire.connectors.base import ConnectorConfig
from squire.connectors.docs_site import DocsSiteConnector

BASE = "https://docs.example/guide/"


class _FakeResponse:
    def __init__(self, text: str = "", status: int = 200) -> None:
        self._text = text
        self._status = status

    @property
    def status_code(self) -> int:
        return self._status

    def raise_for_status(self) -> Any:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")
        return None

    def json(self) -> Any:
        return {}

    @property
    def text(self) -> str:
        return self._text


class _FakeHttp:
    def __init__(self, pages: dict[str, str], errors: set[str] | None = None) -> None:
        self._pages = pages
        self._errors = errors or set()
        self.fetched: list[str] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.fetched.append(url)
        if url in self._errors:
            return _FakeResponse(status=500)
        if url not in self._pages:
            return _FakeResponse(status=404)
        return _FakeResponse(text=self._pages[url])


def _pages() -> dict[str, str]:
    return {
        BASE: (
            "<html><body><script>track()</script><h1>Guide</h1>"
            "<p>Welcome to the guide.</p>"
            '<a href="a">A</a><a href="b">B</a>'
            '<a href="/top">Top</a>'
            '<a href="https://other.example/x">External</a>'
            "</body></html>"
        ),
        "https://docs.example/guide/a": "<p>Page A content.</p>",
        "https://docs.example/guide/b": "<p>Page B content.</p>",
    }


def _connector(http: _FakeHttp, **settings: object) -> DocsSiteConnector:
    return DocsSiteConnector(
        ConnectorConfig(
            key="docs_site",
            instance_id="docs",
            settings={"base_url": BASE, **settings},
        ),
        client=http,
    )


def test_crawls_in_scope_pages_only() -> None:
    """The crawl follows in-scope links and ignores off-host and off-scope ones."""
    connector = _connector(_FakeHttp(_pages()))

    urls = {doc.source_url for doc in connector.iter_documents(None)}

    assert urls == {
        BASE,
        "https://docs.example/guide/a",
        "https://docs.example/guide/b",
    }


def test_extracts_clean_text() -> None:
    """Scripts and markup are dropped; readable text remains."""
    connector = _connector(_FakeHttp(_pages()))
    home = next(d for d in connector.iter_documents(None) if d.source_url == BASE)

    assert "Welcome to the guide." in home.content
    assert "track()" not in home.content
    assert home.source_type == "docs"


def test_emits_no_signals() -> None:
    """A docs site carries no actionable work."""
    connector = _connector(_FakeHttp(_pages()))
    assert list(connector.iter_signals(None)) == []
    assert connector.current_cursor() is None


def test_respects_max_pages() -> None:
    """The crawl stops at the configured page limit."""
    connector = _connector(_FakeHttp(_pages()), max_pages=2)
    assert len(list(connector.iter_documents(None))) == 2


def test_skips_pages_that_fail_to_fetch() -> None:
    """A page that errors is skipped without aborting the crawl."""
    http = _FakeHttp(_pages(), errors={"https://docs.example/guide/b"})
    connector = _connector(http)

    urls = {doc.source_url for doc in connector.iter_documents(None)}

    assert "https://docs.example/guide/b" not in urls
    assert "https://docs.example/guide/a" in urls  # crawl continued past the failure


def test_two_instances_stay_independent() -> None:
    """Two docs sites have distinct instance ids."""
    a = _connector(_FakeHttp(_pages()))
    b = DocsSiteConnector(
        ConnectorConfig(
            key="docs_site",
            instance_id="other-docs",
            settings={"base_url": "https://other.example/"},
        ),
        client=_FakeHttp({}),
    )
    assert a.instance_id != b.instance_id
