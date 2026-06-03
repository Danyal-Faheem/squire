# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the composition root.

These cover the wiring decisions the runtime owns without needing a database: which
connectors it builds from a deployment, and which classifier it activates — the LLM
classifier when the deployment provides a policy, and the trivial default until then,
so the engine runs either way. The clients are injected as fakes through the runtime's
constructor, so nothing here touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from squire.common import models
from squire.common.config import Settings
from squire.common.deployment import DeploymentConfigError
from squire.common.schemas import Signal
from squire.rag.brain import default_classifier
from squire.runtime import Runtime
from tests.fakes import make_embedder, make_llm


def _deployment(tmp_path: Path, *, with_policy: bool) -> Path:
    directory = tmp_path / "dep"
    directory.mkdir()
    (directory / "deployment.yml").write_text(
        """
        connectors:
          - key: filesystem
            instance_id: docs
            settings:
              root: .
        """,
        encoding="utf-8",
    )
    if with_policy:
        (directory / "taxonomy.yml").write_text(
            "categories:\n  - name: bug\n  - name: question\ndefault: question\n",
            encoding="utf-8",
        )
        (directory / "routing.yml").write_text(
            "routes:\n  bug:\n    owners: [core]\n    repo: o/n\n"
            "default:\n  owners: [maint]\n  repo: o/n\n",
            encoding="utf-8",
        )
        (directory / "priorities.yml").write_text(
            "levels:\n  - key: p0\n    label: 'priority: critical'\n"
            "  - key: p2\n    label: 'priority: normal'\ndefault: p2\n",
            encoding="utf-8",
        )
    return directory


def _runtime(deployment: Path, *, llm_reply: str = "{}") -> Runtime:
    return Runtime(
        Settings(embedding_dim=4),
        deployment=deployment,
        llm=make_llm(content=llm_reply),
        embedder=make_embedder(dim=4),
    )


def _signal() -> Signal:
    return Signal(
        source_type="github_issue",
        source_id="o/n#1",
        source_url="https://example.test/o/n/issues/1",
        title="Login broken",
        body="It fails",
        content_hash="h1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _bare_runtime(embedding_dim: int) -> Runtime:
    return Runtime(
        Settings(embedding_dim=embedding_dim),
        deployment="unused",
        llm=make_llm(),
        embedder=make_embedder(dim=4),
    )


def test_validate_embedding_dimension_accepts_match() -> None:
    """A configured dimension equal to the schema's passes validation silently."""
    _bare_runtime(models.EMBEDDING_DIM).validate_embedding_dimension()


def test_validate_embedding_dimension_rejects_mismatch() -> None:
    """A configured dimension that disagrees with the schema fails loudly."""
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM"):
        _bare_runtime(models.EMBEDDING_DIM + 1).validate_embedding_dimension()


def test_build_connectors_from_deployment(tmp_path: Path) -> None:
    """The runtime instantiates the deployment's activated connectors."""
    runtime = _runtime(_deployment(tmp_path, with_policy=False))
    connectors = runtime.build_connectors()
    assert [c.key for c in connectors] == ["filesystem"]


def test_classifier_falls_back_to_default_without_policy(tmp_path: Path) -> None:
    """With no policy files, the active classifier is the trivial default."""
    runtime = _runtime(_deployment(tmp_path, with_policy=False))
    assert runtime.classifier() is default_classifier


def test_classifier_is_llm_backed_with_policy(tmp_path: Path) -> None:
    """With a policy, the active classifier is the LLM-backed one bound to it."""
    runtime = _runtime(
        _deployment(tmp_path, with_policy=True),
        llm_reply='{"category": "bug", "priority": "p0", "title": "Login is broken"}',
    )
    classifier = runtime.classifier()

    assert classifier is not default_classifier
    result = classifier(_signal())
    assert result.category == "bug"
    assert result.priority_key == "p0"
    assert result.owner == "core"  # routed from the policy


def test_classifier_fails_loudly_on_malformed_policy(tmp_path: Path) -> None:
    """A present-but-invalid policy is a loud error, not a silent fallback."""
    directory = tmp_path / "dep"
    directory.mkdir()
    (directory / "deployment.yml").write_text("connectors: []\n", encoding="utf-8")
    (directory / "taxonomy.yml").write_text(
        "categories:\n  - name: bug\n", encoding="utf-8"
    )
    (directory / "priorities.yml").write_text(
        "levels:\n  - key: p0\n    label: x\n", encoding="utf-8"
    )
    # routing.yml is present but invalid: it omits the required default route.
    (directory / "routing.yml").write_text("routes: {}\n", encoding="utf-8")

    runtime = _runtime(directory)
    with pytest.raises(DeploymentConfigError):
        runtime.classifier()
