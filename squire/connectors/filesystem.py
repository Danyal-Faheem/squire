# SPDX-License-Identifier: AGPL-3.0-or-later
"""A connector that ingests text files from a directory tree.

This is the simplest possible source: a directory of plain-text documents (markdown,
reStructuredText, and the like). It is *document-only* — it never emits signals,
because a directory of files represents knowledge, not incoming work. It exists both
as a useful connector in its own right and as the minimal worked example of the
document-ingestion half of the connector contract.

Access pattern, file selection, and the source-type label are all configuration, so
the same code can ingest a docs tree in one deployment and a changelog directory in
another without modification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import register


class FilesystemSettings(BaseModel):
    """Settings block for a :class:`FilesystemConnector` instance."""

    #: Directory to walk. Required; every other setting has a sensible default.
    root: str
    #: File extensions to ingest (compared case-insensitively, leading dot included).
    extensions: list[str] = Field(default=[".md", ".rst", ".txt"])
    #: Whether to descend into sub-directories.
    recursive: bool = True
    #: Encoding used to read files; undecodable files are skipped.
    encoding: str = "utf-8"
    #: The source-type label stamped on every document this instance produces. Left
    #: configurable because the same connector may ingest different kinds of files in
    #: different deployments.
    source_type: str = "file"


@register
class FilesystemConnector:
    """Ingests text files from a directory tree as documents."""

    key: ClassVar[str] = "filesystem"

    def __init__(self, config: ConnectorConfig) -> None:
        self.instance_id = config.instance_id
        self._settings = FilesystemSettings.model_validate(config.settings)
        self._root = Path(self._settings.root).expanduser()
        self._extensions = {ext.lower() for ext in self._settings.extensions}
        # The newest modification time observed during a run; serialized into the
        # cursor so the next run can skip files that have not changed.
        self._watermark: float | None = None

    def healthcheck(self) -> None:
        """Verify the configured root exists and is a directory."""
        if not self._root.exists():
            raise FileNotFoundError(f"filesystem root does not exist: {self._root}")
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"filesystem root is not a directory: {self._root}"
            )

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Yield one document per matching file modified after ``since``."""
        floor = _parse_mtime(since)
        self._observe(floor)
        for path in self._iter_files():
            mtime = path.stat().st_mtime
            if floor is not None and mtime <= floor:
                continue
            try:
                content = path.read_text(encoding=self._settings.encoding)
            except (UnicodeDecodeError, OSError):
                # A binary file that slipped through the extension filter, or a file
                # we cannot read, is silently skipped rather than aborting the run.
                continue
            self._observe(mtime)
            yield Document(
                source_type=self._settings.source_type,
                source_url=path.resolve().as_uri(),
                origin=self._relative(path),
                content=content,
                content_hash=_hash(content),
            )

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """A directory of files carries no actionable work, so yield nothing."""
        return iter(())

    def current_cursor(self) -> Cursor | None:
        """Return the newest modification time seen, encoded as the cursor."""
        if self._watermark is None:
            return None
        return Cursor(value=repr(self._watermark))

    def _iter_files(self) -> Iterator[Path]:
        pattern = "**/*" if self._settings.recursive else "*"
        for path in sorted(self._root.glob(pattern)):
            if path.is_file() and path.suffix.lower() in self._extensions:
                yield path

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self._root.resolve()))
        except ValueError:
            return str(path)

    def _observe(self, mtime: float | None) -> None:
        if mtime is None:
            return
        if self._watermark is None or mtime > self._watermark:
            self._watermark = mtime


def _parse_mtime(since: Cursor | None) -> float | None:
    if since is None:
        return None
    try:
        return float(since.value)
    except ValueError:
        return None


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
