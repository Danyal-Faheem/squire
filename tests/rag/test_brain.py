# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the brain, which composes retrieval, generation, and classification.

These confirm the brain wires the pieces together (a query retrieves and then
generates a grounded answer), that the source-type filter flows through, and that
classification delegates to an injected function — so the real Phase 3 classifier can
replace the trivial default without the brain or its callers changing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from squire.common.schemas import Classification, Signal
from squire.rag.brain import Brain, ClassifierFn, default_classifier
from tests.rag._support import RecordingVectorStore, make_embedder, make_llm, result


def _signal(title: str = "A title", body: str = "A body") -> Signal:
    return Signal(
        source_type="github_issue",
        source_id="o/n#1",
        source_url="https://example.test/o/n/issues/1",
        title=title,
        body=body,
        content_hash="h1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _brain(
    store: RecordingVectorStore,
    *,
    answer: str = "grounded answer [1]",
    classifier: ClassifierFn = default_classifier,
) -> Brain:
    return Brain(
        embedder=make_embedder(),
        vector_store=store,
        llm=make_llm(answer),
        classifier=classifier,
    )


def test_query_retrieves_then_generates() -> None:
    """A query returns a grounded answer citing the retrieved chunk."""
    store = RecordingVectorStore([result(source_url="u1")])
    brain = _brain(store)

    answer = brain.query("how?", source_types=["docs"], top_k=5)

    assert answer.grounded is True
    assert [c.source_url for c in answer.citations] == ["u1"]
    assert store.searches[0]["source_types"] == ["docs"]
    assert store.searches[0]["limit"] == 5


def test_query_with_empty_index_refuses() -> None:
    """With nothing indexed, a query refuses rather than inventing an answer."""
    answer = _brain(RecordingVectorStore([])).query("how?")
    assert answer.grounded is False
    assert answer.citations == []


def test_draft_grounds_in_retrieved_chunks() -> None:
    """Drafting retrieves background for the item and cites it."""
    store = RecordingVectorStore([result(source_url="u1")])
    answer = _brain(store, answer="suggested [1]").draft("Title", "Body")
    assert answer.grounded is True
    assert [c.source_url for c in answer.citations] == ["u1"]


def test_default_classifier_returns_generic_placeholders() -> None:
    """The trivial default assigns generic, project-agnostic values."""
    classification = default_classifier(_signal(title="My issue"))
    assert classification.category == "uncategorized"
    assert classification.owner == "unassigned"
    assert classification.suggested_title == "My issue"


def test_classify_delegates_to_injected_classifier() -> None:
    """A custom classifier replaces the default without any other change."""

    def custom(signal: Signal) -> Classification:
        return Classification(
            category="bug",
            priority_key="p1",
            owner="team-core",
            suggested_title=f"[triaged] {signal.title}",
        )

    brain = _brain(RecordingVectorStore([]), classifier=custom)
    classification = brain.classify(_signal(title="Crash"))

    assert classification.category == "bug"
    assert classification.priority_key == "p1"
    assert classification.suggested_title == "[triaged] Crash"
