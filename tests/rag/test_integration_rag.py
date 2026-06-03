# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: retrieval against a real Postgres + pgvector index.

This proves the Phase 2 acceptance criterion literally — "with a small indexed corpus,
/query returns a cited answer" — against the real vector index rather than a fake
store. A handful of chunks are embedded and written, then a query retrieves the
nearest one and the answer cites it. Only the database is real; the embedding and LLM
endpoints are faked, since the point here is the retrieval path. It skips when no
Postgres is reachable, like the Phase 1 integration tests.
"""

from __future__ import annotations

import pytest

from squire.common.db import session_scope
from squire.common.schemas import Document
from squire.common.vectorstore import PgVectorStore
from squire.rag.brain import Brain
from tests.rag._support import make_embedder, make_llm

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 1024


def _doc(content_hash: str, content: str, source_type: str = "docs") -> Document:
    return Document(
        source_type=source_type,
        source_url=f"https://example.test/{content_hash}",
        origin="handbook",
        content=content,
        content_hash=content_hash,
    )


def test_query_returns_cited_answer_from_real_index(postgres_ready: None) -> None:
    """A query against a real indexed corpus returns a grounded, cited answer."""
    # Embed three chunks with distinct vectors so retrieval has a clear winner.
    vectors = {
        "near": [1.0] + [0.0] * (EMBEDDING_DIM - 1),
        "mid": [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2),
        "far": [0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3),
    }
    docs = [
        (_doc("near", "To reset your token, run the rotate command."), vectors["near"]),
        (_doc("mid", "Unrelated note about colours."), vectors["mid"]),
        (_doc("far", "Another unrelated note."), vectors["far"]),
    ]

    with session_scope() as session:
        written = PgVectorStore(session).upsert(docs)
    assert written == 3

    # The query embedder returns the "near" vector, so retrieval should rank that
    # chunk first; the LLM is faked to echo a citation.
    embedder = make_embedder(dim=EMBEDDING_DIM, vector=vectors["near"])
    llm = make_llm(content="Run the rotate command [1].")

    with session_scope() as session:
        brain = Brain(embedder=embedder, vector_store=PgVectorStore(session), llm=llm)
        answer = brain.query("how do I reset my token?", source_types=["docs"], top_k=1)

    assert answer.grounded is True
    assert len(answer.citations) == 1
    assert answer.citations[0].source_url == "https://example.test/near"
    assert "[1]" in answer.answer


def test_source_type_filter_excludes_other_types(postgres_ready: None) -> None:
    """A query restricted to one source type does not retrieve from others.

    Two chunks share the same (identical) embedding, so without a filter both are
    equally relevant; with a ``docs`` filter only the docs chunk may come back. This
    proves real exclusion at the index, not merely that the filter was passed along.
    """
    shared_vector = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    docs = [
        (_doc("doc-chunk", "Documented behaviour.", source_type="docs"), shared_vector),
        (_doc("code-chunk", "Source code.", source_type="github_code"), shared_vector),
    ]
    with session_scope() as session:
        PgVectorStore(session).upsert(docs)

    embedder = make_embedder(dim=EMBEDDING_DIM, vector=shared_vector)
    llm = make_llm(content="Per the docs [1].")

    with session_scope() as session:
        brain = Brain(embedder=embedder, vector_store=PgVectorStore(session), llm=llm)
        docs_only = brain.query("question", source_types=["docs"], top_k=10)
        unfiltered = brain.query("question", source_types=None, top_k=10)

    # Restricted to docs: only the docs chunk is retrieved, never the code one.
    assert [c.source_type for c in docs_only.citations] == ["docs"]
    assert all(
        c.source_url != "https://example.test/code-chunk" for c in docs_only.citations
    )
    # Unfiltered: both source types are reachable, confirming the filter is what
    # excluded the code chunk above (not its absence from the index).
    assert {c.source_type for c in unfiltered.citations} == {"docs", "github_code"}
