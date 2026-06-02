# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the filesystem connector.

These run against real temporary directories, since the connector's whole job is to
read a directory tree faithfully. Modification times are set explicitly so the
incremental-cursor behaviour is deterministic rather than timing-dependent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from squire.connectors.base import ConnectorConfig, Cursor
from squire.connectors.filesystem import FilesystemConnector


def _connector(root: Path, **settings: object) -> FilesystemConnector:
    return FilesystemConnector(
        ConnectorConfig(
            key="filesystem",
            instance_id="fs",
            settings={"root": str(root), **settings},
        )
    )


def _write(path: Path, content: str, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_ingests_matching_files_with_metadata(tmp_path: Path) -> None:
    """Each matching file becomes one document with the expected metadata."""
    _write(tmp_path / "guide.md", "hello world")
    connector = _connector(tmp_path, source_type="docs")

    docs = list(connector.iter_documents(None))

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == "docs"
    assert doc.content == "hello world"
    assert doc.origin == "guide.md"
    assert doc.source_url.startswith("file://")
    assert doc.source_url.endswith("guide.md")
    assert doc.content_hash  # a non-empty digest


def test_extension_filter_excludes_other_files(tmp_path: Path) -> None:
    """Files outside the configured extensions are not ingested."""
    _write(tmp_path / "keep.md", "keep")
    _write(tmp_path / "skip.bin", "skip")
    connector = _connector(tmp_path, extensions=[".md"])

    origins = {doc.origin for doc in connector.iter_documents(None)}

    assert origins == {"keep.md"}


def test_non_recursive_ignores_subdirectories(tmp_path: Path) -> None:
    """With recursion off, nested files are not visited."""
    _write(tmp_path / "top.md", "top")
    _write(tmp_path / "nested" / "deep.md", "deep")
    connector = _connector(tmp_path, recursive=False)

    origins = {doc.origin for doc in connector.iter_documents(None)}

    assert origins == {"top.md"}


def test_emits_no_signals(tmp_path: Path) -> None:
    """A document-only source never yields signals."""
    _write(tmp_path / "a.md", "a")
    connector = _connector(tmp_path)
    assert list(connector.iter_signals(None)) == []


def test_healthcheck_rejects_missing_root(tmp_path: Path) -> None:
    """A non-existent root fails fast at healthcheck."""
    connector = _connector(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        connector.healthcheck()


def test_healthcheck_rejects_file_root(tmp_path: Path) -> None:
    """A root that is a file, not a directory, fails fast at healthcheck."""
    target = tmp_path / "afile.md"
    _write(target, "x")
    connector = _connector(target)
    with pytest.raises(NotADirectoryError):
        connector.healthcheck()


def test_cursor_tracks_newest_mtime(tmp_path: Path) -> None:
    """After a run the cursor encodes the newest modification time seen."""
    _write(tmp_path / "old.md", "old", mtime=1000.0)
    _write(tmp_path / "new.md", "new", mtime=2000.0)
    connector = _connector(tmp_path)

    list(connector.iter_documents(None))

    cursor = connector.current_cursor()
    assert cursor is not None
    assert float(cursor.value) == 2000.0


def test_incremental_run_skips_unchanged_files(tmp_path: Path) -> None:
    """Given a cursor, only files modified after it are re-ingested."""
    _write(tmp_path / "old.md", "old", mtime=1000.0)
    _write(tmp_path / "new.md", "new", mtime=2000.0)
    connector = _connector(tmp_path)

    docs = list(connector.iter_documents(Cursor(value="1000.0")))

    assert {doc.origin for doc in docs} == {"new.md"}
    # The watermark must not regress below the cursor it was given.
    cursor = connector.current_cursor()
    assert cursor is not None
    assert float(cursor.value) == 2000.0


def test_empty_directory_yields_no_cursor(tmp_path: Path) -> None:
    """A backfill that finds nothing reports no watermark to persist."""
    connector = _connector(tmp_path)
    assert list(connector.iter_documents(None)) == []
    assert connector.current_cursor() is None


def test_missing_required_setting_fails_fast() -> None:
    """Omitting the required root is rejected at construction, not mid-run."""
    with pytest.raises(ValidationError):
        FilesystemConnector(
            ConnectorConfig(key="filesystem", instance_id="fs", settings={})
        )
