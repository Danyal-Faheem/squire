# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration for messaging: the digest and the wired command surface.

These prove the parts that need the database and the composition root: the digest
reads real open items and posts an owner-grouped summary, and the runtime wires a
Slack command surface whose reply is delivered into the thread. The Slack client and
the model are faked; only the database is real. Skips without Postgres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from squire.common.config import Settings
from squire.common.db import session_scope
from squire.factory.store import NewTrackedItem, PgTrackedItemStore
from squire.messaging import SlackAdapter, publish_digest
from squire.runtime import Runtime
from tests.fakes import make_embedder, make_llm

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 1024


class FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"ts": "1700000000.000100"}


def _new(owner: str, number: int) -> NewTrackedItem:
    return NewTrackedItem(
        source_type="github_issue",
        source_id=f"acme/widget#{number}",
        source_url=f"https://example.test/{number}",
        content_hash=f"h{number}",
        board_item_ref=f"acme/widget#{number}",
        category="bug",
        priority="p1",
        assigned_owner=owner,
    )


def test_digest_posts_owner_grouped_summary(postgres_ready: None) -> None:
    """publish_digest reads real open items and posts them grouped by owner."""
    with session_scope() as session:
        store = PgTrackedItemStore(session)
        store.create(_new("U_alice", 1))
        store.create(_new("U_bob", 2))

    client = FakeSlackClient()
    adapter = SlackAdapter(client=client)
    with session_scope() as session:
        publish_digest(adapter, "C_DIGEST", PgTrackedItemStore(session))

    posted = client.calls[0]
    assert posted["channel"] == "C_DIGEST"
    assert "<@U_alice> — 1 open:" in posted["text"]
    assert "<@U_bob> — 1 open:" in posted["text"]


def test_runtime_wires_slack_command_surface(
    postgres_ready: None, tmp_path: Path
) -> None:
    """The composition root wires a Slack command surface that replies in-thread."""
    deployment = tmp_path / "dep"
    deployment.mkdir()
    (deployment / "deployment.yml").write_text("connectors: []\n", encoding="utf-8")

    runtime = Runtime(
        Settings(embedding_dim=EMBEDDING_DIM),
        deployment=deployment,
        llm=make_llm(content="Because of the config."),
        embedder=make_embedder(dim=EMBEDDING_DIM, vector=[0.0] * EMBEDDING_DIM),
    )
    client = FakeSlackClient()
    adapter = SlackAdapter(client=client)

    with session_scope() as session:
        surface = runtime.build_messaging_surface(session, adapter)
        reply = surface.handle_mention(
            text="@squire why does it fail?",
            actor="U1",
            channel="C1",
            thread="T1",
        )

    assert client.calls[0]["channel"] == "C1"
    assert client.calls[0]["thread_ts"] == "T1"
    assert client.calls[0]["text"] == reply
