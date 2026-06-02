# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: factory dedup against a real tracked-items ledger.

The unit tests prove dedup within one in-memory store; this proves it *persists*. Two
factories are built over separate database sessions — standing in for a restart — and
fed the same signal. The first creates a board item and records it; the second, with a
fresh store reading the committed ledger, recognises the signal and does nothing. This
is the guarantee that a restart cannot cause a duplicate item. Skips when no Postgres
is reachable.
"""

from __future__ import annotations

import pytest

from squire.common.db import session_scope
from squire.factory.store import PgTrackedItemStore
from squire.factory.work_items import FactoryOutcome, WorkItemFactory
from tests.factory._support import (
    FakeBoard,
    FakeDrafter,
    StubClassifier,
    classification,
    priorities,
    routing,
    signal,
)

pytestmark = pytest.mark.integration


def _factory(board: FakeBoard, store: PgTrackedItemStore) -> WorkItemFactory:
    return WorkItemFactory(
        classifier=StubClassifier(classification()),
        drafter=FakeDrafter(),
        board=board,
        store=store,
        routing=routing(),
        priorities=priorities(),
        dry_run=False,
    )


def test_dedup_persists_across_sessions(postgres_ready: None) -> None:
    """The same signal handled by two separate factories yields one tracked item."""
    first_board = FakeBoard()
    with session_scope() as session:
        result_one = _factory(first_board, PgTrackedItemStore(session)).process(
            signal()
        )

    assert result_one.outcome is FactoryOutcome.CREATED
    assert len(first_board.created) == 1

    # A fresh factory over a new session (as after a restart) sees the persisted row.
    second_board = FakeBoard()
    with session_scope() as session:
        result_two = _factory(second_board, PgTrackedItemStore(session)).process(
            signal()
        )

    assert result_two.outcome is FactoryOutcome.DUPLICATE
    assert second_board.created == []  # no duplicate board item

    with session_scope() as session:
        tracked = PgTrackedItemStore(session).get("github_issue", "acme/widget#1")
    assert tracked is not None
    assert tracked.board_item_ref == "acme/widget#1"
    assert tracked.category == "bug"
    assert tracked.priority == "p0"
