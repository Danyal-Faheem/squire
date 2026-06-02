# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 0 smoke check: embed a string, store it in pgvector, retrieve it back.

Run with: ``python -m scripts.smoke_vectorstore``. Requires a reachable Postgres
(with pgvector and migrations applied) and a reachable embedding endpoint.
"""

from __future__ import annotations

import hashlib

from squire.common.db import session_scope
from squire.common.embeddings import EmbeddingClient
from squire.common.schemas import Document
from squire.common.vectorstore import PgVectorStore


def main() -> None:
    text = "Squire prepares the ground so a maintainer's judgement is spent well."
    embedder = EmbeddingClient()
    vector = embedder.embed_one(text)

    doc = Document(
        source_type="file",
        source_url="local://smoke-test",
        origin="smoke-test",
        content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )

    with session_scope() as session:
        store = PgVectorStore(session)
        written = store.upsert([(doc, vector)])
        results = store.search(vector, limit=1)

    print(f"wrote {written} chunk(s); top result score={results[0].score:.4f}")
    print(f"retrieved content: {results[0].content!r}")


if __name__ == "__main__":
    main()
