# SPDX-License-Identifier: AGPL-3.0-or-later
"""A connector that crawls a documentation website.

Starting from a configured base URL, it walks links breadth-first, staying within a
configurable scope (by default, the path the base URL sits under) and on the same
host, and turns each page's readable text into a document. It is document-only — a
docs site is knowledge, not incoming work — so it emits no signals.

The crawl is bounded by a page limit and is resilient: a page that fails to fetch is
skipped rather than aborting the run. Re-crawling is safe because the knowledge base
de-duplicates on content hash, so unchanged pages add nothing. The HTTP transport is
an optional constructor parameter, so the crawl is exercised offline against a fake.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Iterator
from typing import ClassVar, cast
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from squire.connectors._web import HttpClient, html_to_text
from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import register

logger = logging.getLogger(__name__)


class DocsSiteSettings(BaseModel):
    """Settings block for a :class:`DocsSiteConnector` instance."""

    #: URL to start crawling from. Required.
    base_url: str
    #: Path prefix the crawl stays within; defaults to the base URL's own path.
    scope: str | None = None
    #: Maximum number of pages to ingest in one run.
    max_pages: int = 100
    #: The source-type label stamped on every document this instance produces.
    source_type: str = "docs"


@register
class DocsSiteConnector:
    """Crawls a documentation site, ingesting each page as a document."""

    key: ClassVar[str] = "docs_site"

    def __init__(
        self, config: ConnectorConfig, *, client: HttpClient | None = None
    ) -> None:
        self.instance_id = config.instance_id
        self._settings = DocsSiteSettings.model_validate(config.settings)
        self._client = client
        parsed = urlparse(self._settings.base_url)
        self._host = parsed.netloc
        self._scope = self._settings.scope or (parsed.path or "/")

    def healthcheck(self) -> None:
        """Validate the base URL names a host and that the client can be built."""
        if not self._host:
            raise ValueError("docs_site base_url must be an absolute URL")
        self._http()

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Crawl from the base URL and yield one document per reachable page."""
        seen: set[str] = set()
        queue: deque[str] = deque([self._settings.base_url])
        yielded = 0
        while queue and yielded < self._settings.max_pages:
            url = urldefrag(queue.popleft())[0]
            if url in seen:
                continue
            seen.add(url)
            html = self._fetch(url)
            if html is None:
                continue
            text = html_to_text(html)
            if text:
                yield Document(
                    source_type=self._settings.source_type,
                    source_url=url,
                    origin=self._host,
                    content=text,
                    content_hash=_hash(url, text),
                )
                yielded += 1
            for link in self._links(url, html):
                if link not in seen and self._in_scope(link):
                    queue.append(link)

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """A documentation site carries no actionable work, so yield nothing."""
        return iter(())

    def current_cursor(self) -> Cursor | None:
        """Docs are re-crawled in full; content-hash de-duplication handles repeats."""
        return None

    def _fetch(self, url: str) -> str | None:
        try:
            response = self._http().get(url)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            # A crawl must survive a single bad page rather than abort the run.
            logger.warning("skipping %s: %s", url, exc)
            return None

    def _links(self, page_url: str, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"])
            links.append(urldefrag(urljoin(page_url, href))[0])
        return links

    def _in_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.netloc != self._host:
            return False
        path = parsed.path or "/"
        if self._scope in ("", "/"):
            return True
        return path == self._scope or path.startswith(self._scope.rstrip("/") + "/")

    def _http(self) -> HttpClient:
        if self._client is None:
            self._client = cast("HttpClient", httpx.Client(follow_redirects=True))
        return self._client


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
