# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the ingestion CLI wiring.

The CLI is thin by design, so these tests cover the parts that have logic of their
own: how arguments map to a run mode, and that a run selecting no connectors returns
without touching the database. The full pipeline behaviour is covered by the
orchestrator tests; the database-backed path is exercised by integration tests later.
"""

from __future__ import annotations

from pathlib import Path

from squire.ingestion.orchestrator import IngestionMode
from squire.ingestion.run_ingest import build_arg_parser, run


def test_default_mode_is_backfill() -> None:
    """With no mode flag the CLI backfills."""
    args = build_arg_parser().parse_args([])
    assert args.mode is IngestionMode.BACKFILL


def test_live_flag_selects_live_mode() -> None:
    """--live switches to incremental mode."""
    args = build_arg_parser().parse_args(["--live"])
    assert args.mode is IngestionMode.LIVE


def test_connector_flag_is_repeatable() -> None:
    """--connector accumulates the keys to limit the run to."""
    args = build_arg_parser().parse_args(
        ["--connector", "github", "--connector", "pdf"]
    )
    assert args.connectors == ["github", "pdf"]


def test_run_returns_empty_when_no_connectors_match(tmp_path: Path) -> None:
    """Filtering to a key that is not activated short-circuits before any DB work."""
    deployment = tmp_path / "dep"
    deployment.mkdir()
    (deployment / "deployment.yml").write_text(
        """
        connectors:
          - key: filesystem
            settings:
              root: .
        """,
        encoding="utf-8",
    )

    result = run(["--deployment", str(deployment), "--connector", "github"])

    assert result.runs == []
    assert result.documents_indexed == 0
