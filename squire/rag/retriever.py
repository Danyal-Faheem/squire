# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retrieval: turn a question into the most relevant chunks of the knowledge base.

Retrieval is deliberately thin and sits squarely on the engine's seams: it embeds the
question with the :class:`~squire.common.embeddings.EmbeddingClient` and asks the
:class:`~squire.common.vectorstore.VectorStore` for the nearest chunks, optionally
constrained to a set of source types. Because the source-type set is open — it is
whatever connectors populated the index — the filter is just a list of strings the
caller supplies; the retriever never hardcodes which types exist.
"""

from __future__ import annotations

from squire.common.embeddings import EmbeddingClient
from squire.common.vectorstore import SearchResult, VectorStore


class Retriever:
    """Embeds a query and fetches the closest chunks from the vector store."""

    def __init__(self, embedder: EmbeddingClient, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> list[SearchResult]:
        """Return the ``top_k`` chunks most similar to ``question``.

        When ``source_types`` is given, retrieval is restricted to those types, which
        is how a caller biases toward, say, documentation over issue chatter. An empty
        or whitespace-only question retrieves nothing.
        """
        if not question.strip():
            return []
        embedding = self._embedder.embed_one(question)
        return self._vector_store.search(
            embedding, limit=top_k, source_types=source_types
        )
