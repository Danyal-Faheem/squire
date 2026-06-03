# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-layer integration: the composition root wires working components.

This proves the runtime-built orchestrator and brain actually function over a real
database: ingesting a filesystem document lands a chunk, and the runtime-built brain
retrieves it back with a grounded, cited answer. The clients are injected as fakes, so
only the database is real. Skips when no Postgres is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squire.common.config import Settings
from squire.common.db import session_scope
from squire.ingestion.orchestrator import IngestionMode
from squire.runtime import Runtime
from tests.fakes import make_embedder, make_llm

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 1024


def _runtime(deployment: Path) -> Runtime:
    vector = [1.0, *([0.0] * (EMBEDDING_DIM - 1))]
    return Runtime(
        Settings(embedding_dim=EMBEDDING_DIM),
        deployment=deployment,
        llm=make_llm(content="Per the docs [1]."),
        embedder=make_embedder(dim=EMBEDDING_DIM, vector=vector),
    )


def test_runtime_wires_ingestion_then_retrieval(
    postgres_ready: None, tmp_path: Path
) -> None:
    """A runtime-built orchestrator ingests; a runtime-built brain retrieves."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("To reset the token, run rotate.", encoding="utf-8")

    deployment = tmp_path / "dep"
    deployment.mkdir()
    (deployment / "deployment.yml").write_text(
        f"""
        connectors:
          - key: filesystem
            instance_id: docs
            settings:
              root: {docs}
              source_type: docs
        """,
        encoding="utf-8",
    )

    runtime = _runtime(deployment)
    connectors = runtime.build_connectors()

    with session_scope() as session:
        result = runtime.build_orchestrator(session).run(
            connectors, mode=IngestionMode.BACKFILL
        )
    assert result.chunks_indexed == 1

    with session_scope() as session:
        answer = runtime.build_brain(session).query("how do I reset the token?")

    assert answer.grounded is True
    assert answer.citations
    assert answer.citations[0].source_type == "docs"
