# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RAG brain as a FastAPI service.

Three endpoints expose the brain's three jobs:

* ``POST /query``   — retrieval plus a grounded, cited answer.
* ``POST /draft``   — a grounded, cited reply drafted for a work item.
* ``POST /classify``— a classification for a signal (trivial default until Phase 3).

The brain is supplied per request through the :func:`get_brain` dependency, which
binds a vector store to a fresh database session and reuses the process-wide LLM and
embedding clients. Tests override that one dependency to inject a brain wired to
fakes, so the whole service runs offline without a database or any network call.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from squire.common.db import session_scope
from squire.common.embeddings import EmbeddingClient
from squire.common.llm import LLMClient
from squire.common.schemas import Classification, Signal
from squire.common.vectorstore import PgVectorStore
from squire.rag.brain import Brain
from squire.rag.schemas import GroundedAnswer

app = FastAPI(title="Squire RAG brain")


class QueryRequest(BaseModel):
    """A question for the knowledge base, optionally scoped to source types."""

    question: str
    source_types: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class DraftRequest(BaseModel):
    """An incoming item to draft a grounded reply for."""

    title: str
    body: str = ""
    source_types: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=50)


@lru_cache
def _clients() -> tuple[EmbeddingClient, LLMClient]:
    """Build the embedding and LLM clients once and reuse them across requests."""
    return EmbeddingClient(), LLMClient()


def get_brain() -> Iterator[Brain]:
    """Yield a brain bound to a request-scoped database session.

    Overridden in tests to supply a brain backed by fakes; in production it reuses
    the cached clients and opens a transactional session for the vector store.
    """
    embedder, llm = _clients()
    with session_scope() as session:
        yield Brain(embedder=embedder, vector_store=PgVectorStore(session), llm=llm)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest, brain: Brain = Depends(get_brain)) -> GroundedAnswer:
    """Return a grounded, cited answer to a question."""
    return brain.query(
        request.question, source_types=request.source_types, top_k=request.top_k
    )


@app.post("/draft")
def draft(request: DraftRequest, brain: Brain = Depends(get_brain)) -> GroundedAnswer:
    """Return a grounded, cited draft reply for an incoming item."""
    return brain.draft(
        request.title,
        request.body,
        source_types=request.source_types,
        top_k=request.top_k,
    )


@app.post("/classify")
def classify(signal: Signal, brain: Brain = Depends(get_brain)) -> Classification:
    """Classify a signal via the active classifier."""
    return brain.classify(signal)
