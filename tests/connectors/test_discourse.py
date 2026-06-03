# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Discourse connector.

The HTTP transport is faked and injected through the constructor, so these run offline
and cover both access modes the connector must support: anonymous public polling (no
credentials sent) and authenticated reads (the API key and username attached to every
request). They also confirm documents and signals are normalised correctly, tag
filtering hits the tag endpoints, the cursor advances by activity, two instances stay
independent, and a discourse signal flows through the very same work-item factory a
github signal does.
"""

from __future__ import annotations

from typing import Any

import pytest

from squire.connectors.base import ConnectorConfig, Cursor
from squire.connectors.discourse import (
    SOURCE_TYPE_POST,
    SOURCE_TYPE_TOPIC,
    DiscourseConnector,
)
from squire.factory.work_items import FactoryOutcome, WorkItemFactory
from tests.factory._support import (
    FakeBoard,
    FakeDrafter,
    InMemoryTrackedItemStore,
    StubClassifier,
    classification,
    priorities,
    routing,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    @property
    def status_code(self) -> int:
        return 200

    def raise_for_status(self) -> Any:
        return None

    def json(self) -> Any:
        return self._payload

    @property
    def text(self) -> str:
        return ""


class _FakeHttp:
    """Routes paths to canned JSON and records each request's headers."""

    def __init__(self, routes: dict[str, dict[str, Any]]) -> None:
        self._routes = routes
        self.requests: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append((url, kwargs.get("headers") or {}))
        return _FakeResponse(self._routes[url])


def _topic(topic_id: int, *, slug: str, title: str, tags: list[str]) -> dict[str, Any]:
    return {
        "id": topic_id,
        "slug": slug,
        "title": title,
        "tags": tags,
        "bumped_at": "2026-01-02T00:00:00.000Z",
        "created_at": "2026-01-01T00:00:00.000Z",
    }


def _detail(*posts: str) -> dict[str, Any]:
    return {
        "post_stream": {
            "posts": [
                {"post_number": i, "cooked": f"<p>{text}</p>"}
                for i, text in enumerate(posts, start=1)
            ]
        }
    }


def _routes() -> dict[str, dict[str, Any]]:
    return {
        "/latest.json": {
            "topic_list": {
                "topics": [_topic(1, slug="how-to-x", title="How to X", tags=["help"])]
            }
        },
        "/tag/help.json": {
            "topic_list": {
                "topics": [_topic(1, slug="how-to-x", title="How to X", tags=["help"])]
            }
        },
        "/t/1.json": _detail("How do I do X?", "You do it like this."),
    }


def _connector(http: _FakeHttp, **settings: object) -> DiscourseConnector:
    return DiscourseConnector(
        ConnectorConfig(
            key="discourse",
            instance_id="forum",
            settings={"base_url": "https://forum.example", **settings},
        ),
        client=http,
    )


def test_anonymous_reads_documents_and_signals() -> None:
    """Without credentials the connector reads posts and topics and sends no auth."""
    http = _FakeHttp(_routes())
    connector = _connector(http)

    docs = list(connector.iter_documents(None))
    signals = list(connector.iter_signals(None))

    # One document per post, HTML stripped to text.
    assert [d.source_type for d in docs] == [SOURCE_TYPE_POST, SOURCE_TYPE_POST]
    assert docs[0].content == "How do I do X?"
    assert docs[0].source_url == "https://forum.example/t/how-to-x/1/1"

    # One signal per topic.
    assert len(signals) == 1
    assert signals[0].source_type == SOURCE_TYPE_TOPIC
    assert signals[0].title == "How to X"
    assert signals[0].body == "How do I do X?"
    assert signals[0].source_id == "https://forum.example/t/how-to-x/1"

    # No request carried credentials.
    assert all("Api-Key" not in headers for _, headers in http.requests)


def test_authenticated_sends_credentials_on_every_request() -> None:
    """With credentials the API key and username ride on every request."""
    http = _FakeHttp(_routes())
    connector = _connector(http, api_key="secret", api_username="bot")

    list(connector.iter_documents(None))

    assert http.requests  # something was fetched
    for _, headers in http.requests:
        assert headers["Api-Key"] == "secret"
        assert headers["Api-Username"] == "bot"


def test_tag_filter_uses_tag_endpoints() -> None:
    """Configured tags are read from their tag endpoints, not the latest feed."""
    http = _FakeHttp(_routes())
    connector = _connector(http, tags=["help"])

    list(connector.iter_signals(None))

    paths = [url for url, _ in http.requests]
    assert "/tag/help.json" in paths
    assert "/latest.json" not in paths


def test_cursor_advances_and_filters_incrementally() -> None:
    """The cursor tracks the newest activity, and old topics are skipped on re-run."""
    http = _FakeHttp(_routes())
    connector = _connector(http)
    list(connector.iter_signals(None))
    cursor = connector.current_cursor()
    assert cursor is not None

    # A fresh connector given that cursor sees nothing newer than it.
    fresh = _connector(_FakeHttp(_routes()))
    later = Cursor(value="2026-02-01T00:00:00+00:00")
    assert list(fresh.iter_signals(later)) == []


def test_capped_read_does_not_advance_cursor() -> None:
    """A run bounded by max_topics must not record a complete-backfill cursor.

    Otherwise the unread older topics would be skipped forever on the next run. The
    cap is honest: it advances the cursor only when it read everything available.
    """
    routes: dict[str, dict[str, Any]] = {
        "/latest.json": {
            "topic_list": {
                "topics": [
                    _topic(3, slug="c", title="C", tags=[]),
                    _topic(2, slug="b", title="B", tags=[]),
                    _topic(1, slug="a", title="A", tags=[]),
                ]
            }
        },
        "/t/3.json": _detail("newest topic"),
    }
    connector = DiscourseConnector(
        ConnectorConfig(
            key="discourse",
            instance_id="forum",
            settings={"base_url": "https://forum.example", "max_topics": 1},
        ),
        client=_FakeHttp(routes),
    )

    read = list(connector.iter_signals(None))

    assert [s.title for s in read] == ["C"]  # only the newest was read
    assert connector.current_cursor() is None  # ...so completion is not claimed


def test_complete_read_advances_cursor() -> None:
    """When the cap is not hit, the cursor advances normally for incremental polling."""
    connector = _connector(_FakeHttp(_routes()), max_topics=50)
    list(connector.iter_signals(None))
    assert connector.current_cursor() is not None


def test_signal_flows_through_the_same_factory() -> None:
    """A discourse signal is processed by the work-item factory just like github's."""
    connector = _connector(_FakeHttp(_routes()))
    signal = next(iter(connector.iter_signals(None)))

    board = FakeBoard()
    factory = WorkItemFactory(
        classifier=StubClassifier(classification()),
        drafter=FakeDrafter(),
        board=board,
        store=InMemoryTrackedItemStore(),
        routing=routing(),
        priorities=priorities(),
        dry_run=False,
    )

    result = factory.process(signal)

    assert result.outcome is FactoryOutcome.CREATED
    assert board.created[0].title.startswith("[bug]")
    assert board.created[0].body  # the topic became a real board item


def test_two_instances_stay_independent() -> None:
    """Two forums (distinct instance_id + base URL) yield distinct signal ids."""
    one = DiscourseConnector(
        ConnectorConfig(
            key="discourse",
            instance_id="forum-a",
            settings={"base_url": "https://a.example"},
        ),
        client=_FakeHttp(_routes()),
    )
    two = DiscourseConnector(
        ConnectorConfig(
            key="discourse",
            instance_id="forum-b",
            settings={"base_url": "https://b.example"},
        ),
        client=_FakeHttp(_routes()),
    )

    assert one.instance_id != two.instance_id
    id_a = next(iter(one.iter_signals(None))).source_id
    id_b = next(iter(two.iter_signals(None))).source_id
    assert id_a == "https://a.example/t/how-to-x/1"
    assert id_b == "https://b.example/t/how-to-x/1"


def test_healthcheck_requires_username_with_key() -> None:
    """An API key without a username is rejected up front, not mid-run."""
    connector = _connector(_FakeHttp(_routes()), api_key="secret")
    with pytest.raises(ValueError, match="api_username"):
        connector.healthcheck()
