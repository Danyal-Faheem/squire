# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared web-fetching helpers for the HTTP-based connectors.

The forum and docs-site connectors both read over HTTP and both need to turn HTML
into plain text. This module holds the small pieces they share: a narrow
:class:`HttpClient` protocol (so each connector's HTTP transport is an injectable
constructor parameter, per the testability rule) and an HTML-to-text helper. Neither
piece names any project or source — they are generic plumbing.
"""

from __future__ import annotations

from typing import Any, Protocol

from bs4 import BeautifulSoup


class HttpResponse(Protocol):
    """The slice of an HTTP response the connectors use."""

    @property
    def status_code(self) -> int: ...

    def raise_for_status(self) -> Any: ...

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...


class HttpClient(Protocol):
    """The slice of an HTTP client the connectors use (httpx.Client satisfies it)."""

    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...


def html_to_text(html: str) -> str:
    """Convert an HTML fragment or page to readable plain text.

    Scripts and styles are dropped and block structure becomes line breaks, so the
    result is reasonable input for embedding without carrying markup noise.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
