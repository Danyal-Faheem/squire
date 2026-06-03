# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the PDF connector.

The extractor is an injected seam, so the connector's file selection, document
shaping, and incremental behaviour are tested offline with a fake that returns canned
Markdown — no PDF parsing needed. A final test exercises the real default extractor
(pymupdf4llm) against a generated PDF to confirm structure-preserving extraction
actually works end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from squire.connectors.base import ConnectorConfig, Cursor
from squire.connectors.pdf import PdfConnector, PdfExtractor


def _connector(
    paths: list[str], *, extractor: PdfExtractor | None = None
) -> PdfConnector:
    return PdfConnector(
        ConnectorConfig(key="pdf", instance_id="pdf", settings={"paths": paths}),
        extractor=extractor,
    )


def _touch_pdf(path: Path, *, mtime: float | None = None) -> None:
    path.write_bytes(b"%PDF-1.4 not a real pdf")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_extracts_each_pdf_as_a_document(tmp_path: Path) -> None:
    """Each matched PDF becomes one document carrying the extracted markdown."""
    _touch_pdf(tmp_path / "a.pdf")
    _touch_pdf(tmp_path / "b.pdf")

    def fake(path: str) -> str:
        return f"# Heading\n\nContent of {Path(path).name}"

    connector = _connector([str(tmp_path / "*.pdf")], extractor=fake)
    docs = list(connector.iter_documents(None))

    assert {d.source_type for d in docs} == {"pdf"}
    assert {d.origin for d in docs} == {"a.pdf", "b.pdf"}
    by_origin = {d.origin: d for d in docs}
    assert "# Heading" in by_origin["a.pdf"].content
    assert "Content of a.pdf" in by_origin["a.pdf"].content
    assert by_origin["a.pdf"].source_url.startswith("file://")


def test_only_pdf_files_are_selected(tmp_path: Path) -> None:
    """Non-PDF files matched by a glob are ignored."""
    _touch_pdf(tmp_path / "report.pdf")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    connector = _connector([str(tmp_path / "*")], extractor=lambda p: "x")
    origins = {d.origin for d in connector.iter_documents(None)}

    assert origins == {"report.pdf"}


def test_emits_no_signals(tmp_path: Path) -> None:
    """A PDF carries no actionable work."""
    _touch_pdf(tmp_path / "a.pdf")
    connector = _connector([str(tmp_path / "*.pdf")], extractor=lambda p: "x")
    assert list(connector.iter_signals(None)) == []


def test_incremental_skips_unchanged_files(tmp_path: Path) -> None:
    """Given a cursor, only files modified after it are re-extracted."""
    _touch_pdf(tmp_path / "old.pdf", mtime=1000.0)
    _touch_pdf(tmp_path / "new.pdf", mtime=2000.0)

    connector = _connector([str(tmp_path / "*.pdf")], extractor=lambda p: "x")
    docs = list(connector.iter_documents(Cursor(value="1000.0")))

    assert {d.origin for d in docs} == {"new.pdf"}
    cursor = connector.current_cursor()
    assert cursor is not None
    assert float(cursor.value) == 2000.0


def test_extraction_failure_is_skipped(tmp_path: Path) -> None:
    """A PDF the extractor cannot read is skipped without aborting the run."""
    _touch_pdf(tmp_path / "good.pdf")
    _touch_pdf(tmp_path / "bad.pdf")

    def flaky(path: str) -> str:
        if Path(path).name == "bad.pdf":
            raise RuntimeError("corrupt PDF")
        return "ok"

    connector = _connector([str(tmp_path / "*.pdf")], extractor=flaky)
    origins = {d.origin for d in connector.iter_documents(None)}

    assert origins == {"good.pdf"}


def test_healthcheck_requires_paths() -> None:
    """A connector configured with no paths fails fast."""
    connector = _connector([])
    with pytest.raises(ValueError, match="path"):
        connector.healthcheck()


def test_real_extractor_preserves_text(tmp_path: Path) -> None:
    """The default pymupdf4llm extractor pulls real text out of a real PDF."""
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Installation Guide")
    page.insert_text((72, 96), "Run the setup command to begin.")
    pdf_path = tmp_path / "guide.pdf"
    document.save(str(pdf_path))
    document.close()

    connector = _connector([str(pdf_path)])  # default (real) extractor
    docs = list(connector.iter_documents(None))

    assert len(docs) == 1
    assert "Installation Guide" in docs[0].content
    assert "Run the setup command" in docs[0].content
