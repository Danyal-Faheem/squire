# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the retriever.

The retriever is the seam-facing half of the brain: it embeds a query and asks the
vector store for the nearest chunks, optionally filtered by source type. These tests
confirm it passes the query embedding, the limit, and the source-type filter through
faithfully, and that an empty question retrieves nothing without touching either
boundary.
"""

from __future__ import annotations

from squire.rag.retriever import Retriever
from tests.rag._support import RecordingVectorStore, make_embedder, result


def test_retrieve_passes_filter_and_limit() -> None:
    """The source-type filter and top_k reach the vector store unchanged."""
    store = RecordingVectorStore([result(source_url="u1")])
    retriever = Retriever(make_embedder(dim=4, vector=[0.5, 0.5, 0.5, 0.5]), store)

    hits = retriever.retrieve("how do I configure it?", source_types=["docs"], top_k=3)

    assert [hit.source_url for hit in hits] == ["u1"]
    assert len(store.searches) == 1
    call = store.searches[0]
    assert call["embedding"] == [0.5, 0.5, 0.5, 0.5]
    assert call["limit"] == 3
    assert call["source_types"] == ["docs"]


def test_retrieve_without_filter_passes_none() -> None:
    """With no filter, retrieval is unconstrained by source type."""
    store = RecordingVectorStore([])
    Retriever(make_embedder(), store).retrieve("question")
    assert store.searches[0]["source_types"] is None


def test_empty_question_retrieves_nothing() -> None:
    """A blank question short-circuits without querying the store."""
    store = RecordingVectorStore([result(source_url="u1")])
    retriever = Retriever(make_embedder(), store)
    assert retriever.retrieve("   ") == []
    assert store.searches == []
