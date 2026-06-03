# SPDX-License-Identifier: AGPL-3.0-or-later
"""A connector that extracts text from PDF files.

It ingests PDFs matched by configured paths or globs, document-only — PDFs are
knowledge, not incoming work. Extraction goes through a *structure-preserving*
extractor (pymupdf4llm by default) that yields Markdown with headings and tables
intact rather than a flat text dump: chunk quality at ingestion silently determines
retrieval quality forever after, so it is worth preserving structure at the source.

The extractor is an optional constructor parameter — a small callable seam — so tests
exercise the connector's file selection, document shaping, and incremental behaviour
offline without parsing a real PDF, while the default uses pymupdf4llm. A modification
time watermark lets incremental runs skip files that have not changed.
"""

from __future__ import annotations

import glob
import hashlib
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import register

logger = logging.getLogger(__name__)

#: Extracts a PDF at a path into Markdown text. The seam the connector depends on.
PdfExtractor = Callable[[str], str]


class PdfSettings(BaseModel):
    """Settings block for a :class:`PdfConnector` instance."""

    #: File paths or globs (``**`` supported) pointing at the PDFs to ingest.
    paths: list[str]
    #: The source-type label stamped on every document this instance produces.
    source_type: str = "pdf"


@register
class PdfConnector:
    """Ingests PDF files as structure-preserving Markdown documents."""

    key: ClassVar[str] = "pdf"

    def __init__(
        self, config: ConnectorConfig, *, extractor: PdfExtractor | None = None
    ) -> None:
        self.instance_id = config.instance_id
        self._settings = PdfSettings.model_validate(config.settings)
        self._extract = extractor or _default_extractor
        self._watermark: float | None = None

    def healthcheck(self) -> None:
        """Verify at least one path/glob is configured."""
        if not self._settings.paths:
            raise ValueError("pdf connector requires at least one path or glob")

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Yield one document per PDF modified after ``since``."""
        floor = _parse_mtime(since)
        self._observe(floor)
        for path in self._files():
            mtime = path.stat().st_mtime
            if floor is not None and mtime <= floor:
                continue
            try:
                markdown = self._extract(str(path))
            except Exception as exc:
                # A single unreadable PDF should not abort the whole run.
                logger.warning("skipping %s: %s", path, exc)
                continue
            self._observe(mtime)
            yield Document(
                source_type=self._settings.source_type,
                source_url=path.resolve().as_uri(),
                origin=path.name,
                content=markdown,
                content_hash=_hash(markdown),
            )

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """A PDF carries no actionable work, so yield nothing."""
        return iter(())

    def current_cursor(self) -> Cursor | None:
        """Return the newest modification time seen, encoded as the cursor."""
        if self._watermark is None:
            return None
        return Cursor(value=repr(self._watermark))

    def _files(self) -> list[Path]:
        found: dict[str, Path] = {}
        for pattern in self._settings.paths:
            for match in glob.glob(pattern, recursive=True):
                path = Path(match)
                if path.is_file() and path.suffix.lower() == ".pdf":
                    found[str(path.resolve())] = path
        return [found[key] for key in sorted(found)]

    def _observe(self, mtime: float | None) -> None:
        if mtime is None:
            return
        if self._watermark is None or mtime > self._watermark:
            self._watermark = mtime


def _default_extractor(path: str) -> str:
    # Imported lazily so importing the connectors package does not pull in the PDF
    # engine unless a deployment actually activates this connector.
    import pymupdf4llm

    return str(pymupdf4llm.to_markdown(path))


def _parse_mtime(since: Cursor | None) -> float | None:
    if since is None:
        return None
    try:
        return float(since.value)
    except ValueError:
        return None


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
