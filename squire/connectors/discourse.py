# SPDX-License-Identifier: AGPL-3.0-or-later
"""A connector for a Discourse forum.

Discourse instances vary along two axes — where they are and how you read them — so
both are configuration. The connector takes a required base URL and an *optional* API
key and username. With no credentials it polls the public JSON endpoints anonymously;
with credentials it attaches them to each request and reads a forum that requires
authentication. The same code handles both modes; only configuration differs, and the
credentials (when present) are sent per request so the access mode is observable.

It is a both-responsibilities, read-first source: each forum post becomes a document
and each topic becomes a signal the work-item factory can turn into a board item,
exactly as a GitHub issue does. It never writes back — there is no posting path here
at all — honouring the read-first rule regardless of the source-write flag.

The HTTP transport is an optional constructor parameter typed as a small Protocol, so
tests exercise both the anonymous and the authenticated path offline against a fake.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import httpx
from pydantic import BaseModel, Field

from squire.connectors._web import HttpClient, html_to_text
from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import register

#: Source-type labels this connector owns: a post is knowledge, a topic is work.
SOURCE_TYPE_POST = "forum_post"
SOURCE_TYPE_TOPIC = "forum_topic"


class DiscourseSettings(BaseModel):
    """Settings block for a :class:`DiscourseConnector` instance."""

    #: Forum base URL, e.g. ``https://forum.example``. Required.
    base_url: str
    #: Optional API key. Absent means anonymous, public-only reads.
    api_key: str | None = None
    #: Optional username the API key acts as (required by Discourse when a key is set).
    api_username: str | None = None
    #: Tags to watch; empty means the latest topics regardless of tag.
    tags: list[str] = Field(default_factory=list)
    #: Cap on topics considered per run.
    max_topics: int = 50


@register
class DiscourseConnector:
    """Ingests a Discourse forum's topics and posts; emits topics as signals."""

    key: ClassVar[str] = "discourse"

    def __init__(
        self, config: ConnectorConfig, *, client: HttpClient | None = None
    ) -> None:
        self.instance_id = config.instance_id
        self._settings = DiscourseSettings.model_validate(config.settings)
        self._base = self._settings.base_url.rstrip("/")
        self._client = client
        self._watermark: datetime | None = None
        # True when a run stopped at max_topics with more still available, so the
        # read covered only a bounded prefix of the forum and the cursor must not
        # advance (advancing would silently skip the unread older topics).
        self._truncated = False
        self._topic_cache: list[dict[str, Any]] | None = None
        self._topic_cache_key: str | None = None
        self._detail_cache: dict[int, dict[str, Any]] = {}

    # -- contract -----------------------------------------------------------------

    def healthcheck(self) -> None:
        """Validate configuration and that the HTTP client can be built."""
        if not self._base:
            raise ValueError("discourse base_url is required")
        if self._settings.api_key and not self._settings.api_username:
            raise ValueError("discourse api_username is required when api_key is set")
        self._http()

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Yield one document per post across the watched topics."""
        for topic in self._topics(since):
            detail = self._detail(int(topic["id"]))
            for post in detail.get("post_stream", {}).get("posts", []):
                yield self._post_document(topic, post)

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """Yield one signal per watched topic."""
        for topic in self._topics(since):
            yield self._topic_signal(topic)

    def current_cursor(self) -> Cursor | None:
        """Return the newest topic activity seen, encoded as the cursor.

        Returns ``None`` when the run was bounded by ``max_topics`` (it read only a
        prefix of a larger forum). Advancing the watermark then would claim a
        completeness the bounded read did not achieve and would skip the unread older
        topics on the next run, so the cursor is left where it was instead. Full
        pagination — which would let the cap advance honestly — is deferred.
        """
        if self._truncated or self._watermark is None:
            return None
        return Cursor(value=self._watermark.isoformat())

    # -- topics & posts -----------------------------------------------------------

    def _topics(self, since: Cursor | None) -> list[dict[str, Any]]:
        """Fetch the watched topics once per run, newest activity first."""
        floor = _parse_dt(since)
        key = since.value if since is not None else ""
        if self._topic_cache is not None and self._topic_cache_key == key:
            return self._topic_cache

        self._observe(floor)
        self._truncated = False
        paths = (
            [f"/tag/{tag}.json" for tag in self._settings.tags]
            if self._settings.tags
            else ["/latest.json"]
        )
        topics: list[dict[str, Any]] = []
        seen: set[int] = set()
        for path in paths:
            for topic in self._get(path).get("topic_list", {}).get("topics", []):
                topic_id = int(topic["id"])
                if topic_id in seen:
                    continue
                bumped = _parse_value(topic.get("bumped_at") or topic.get("created_at"))
                if floor is not None and bumped is not None and bumped <= floor:
                    continue
                if len(topics) >= self._settings.max_topics:
                    # A qualifying topic we cannot take: the read is bounded.
                    self._truncated = True
                    break
                seen.add(topic_id)
                self._observe(bumped)
                topics.append(topic)
            if self._truncated:
                break

        self._topic_cache = topics
        self._topic_cache_key = key
        return topics

    def _detail(self, topic_id: int) -> dict[str, Any]:
        if topic_id not in self._detail_cache:
            self._detail_cache[topic_id] = self._get(f"/t/{topic_id}.json")
        return self._detail_cache[topic_id]

    def _post_document(self, topic: dict[str, Any], post: dict[str, Any]) -> Document:
        url = f"{self._topic_url(topic)}/{post['post_number']}"
        content = html_to_text(post.get("cooked", "") or "")
        return Document(
            source_type=SOURCE_TYPE_POST,
            source_url=url,
            origin=str(topic.get("title", "")),
            content=content,
            content_hash=_hash(url, content),
        )

    def _topic_signal(self, topic: dict[str, Any]) -> Signal:
        detail = self._detail(int(topic["id"]))
        posts = detail.get("post_stream", {}).get("posts", [])
        body = html_to_text(posts[0].get("cooked", "") or "") if posts else ""
        url = self._topic_url(topic)
        title = str(topic.get("title", ""))
        return Signal(
            source_type=SOURCE_TYPE_TOPIC,
            source_id=url,
            source_url=url,
            title=title,
            body=body,
            content_hash=_hash(title, body),
            created_at=_ensure_aware(_parse_value(topic.get("created_at"))),
        )

    def _topic_url(self, topic: dict[str, Any]) -> str:
        return f"{self._base}/t/{topic.get('slug', 'topic')}/{int(topic['id'])}"

    # -- access -------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        response = self._http().get(path, headers=self._auth_headers())
        response.raise_for_status()
        data = response.json()
        return cast("dict[str, Any]", data)

    def _auth_headers(self) -> dict[str, str]:
        if self._settings.api_key and self._settings.api_username:
            return {
                "Api-Key": self._settings.api_key,
                "Api-Username": self._settings.api_username,
            }
        return {}

    def _http(self) -> HttpClient:
        if self._client is None:
            self._client = cast("HttpClient", httpx.Client(base_url=self._base))
        return self._client

    def _observe(self, when: datetime | None) -> None:
        if when is None:
            return
        if self._watermark is None or when > self._watermark:
            self._watermark = when


def _parse_value(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _parse_dt(since: Cursor | None) -> datetime | None:
    return None if since is None else _parse_value(since.value)


def _ensure_aware(value: datetime | None) -> datetime:
    """Treat naive timestamps as UTC; fall back to the epoch for missing values."""
    if value is None:
        return datetime(1970, 1, 1, tzinfo=UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
