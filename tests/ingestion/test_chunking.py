# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the document processor (seam 4b).

These pin down the chunking contract: a short document stays one chunk; a long one is
split into several; an oversized single paragraph is hard-split so no chunk exceeds the
limit; and every chunk keeps its parent's source identity (source_type/url/origin) plus
lineage (parent hash, ordinal index) so a citation still resolves to the original
source. Each chunk gets a distinct content hash, so de-duplication works per chunk.
"""

from __future__ import annotations

from squire.common.schemas import Document
from squire.ingestion.chunking import LocalDocumentProcessor

PARENT = "parent-hash"


def _doc(content: str) -> Document:
    return Document(
        source_type="docs",
        source_url="https://example.test/page",
        origin="handbook",
        content=content,
        content_hash=PARENT,
    )


def test_short_document_is_a_single_chunk() -> None:
    chunks = LocalDocumentProcessor().process(_doc("a short paragraph"))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content == "a short paragraph"
    assert chunk.parent_content_hash == PARENT
    assert chunk.chunk_index == 0
    assert chunk.source_type == "docs"
    assert chunk.source_url == "https://example.test/page"
    assert chunk.origin == "handbook"


def test_long_document_splits_with_preserved_lineage() -> None:
    content = "\n\n".join(f"Paragraph {i}. " + "word " * 20 for i in range(12))
    chunks = LocalDocumentProcessor(max_chars=200, overlap=20).process(_doc(content))

    assert len(chunks) > 1
    assert all(c.parent_content_hash == PARENT for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source_url == "https://example.test/page" for c in chunks)
    assert all(c.source_type == "docs" for c in chunks)
    # Each chunk has a distinct content hash, so chunk-level de-dup works.
    assert len({c.content_hash for c in chunks}) == len(chunks)
    assert all(len(c.content) <= 200 for c in chunks)


def test_oversized_paragraph_is_hard_split() -> None:
    chunks = LocalDocumentProcessor(max_chars=100, overlap=20).process(_doc("y" * 350))
    assert len(chunks) > 1
    assert all(len(c.content) <= 100 for c in chunks)
    assert all(c.parent_content_hash == PARENT for c in chunks)


def test_title_is_extracted_from_a_heading() -> None:
    chunks = LocalDocumentProcessor().process(_doc("# Installation\n\nDo the thing."))
    assert chunks[0].title == "Installation"


def test_paragraph_without_heading_has_no_title() -> None:
    chunks = LocalDocumentProcessor().process(_doc("Just prose, no heading."))
    assert chunks[0].title is None


def test_empty_document_yields_no_chunks() -> None:
    assert LocalDocumentProcessor().process(_doc("   \n\n  ")) == []


def test_every_chunk_cites_the_original_source() -> None:
    """Citations resolve to the source: every chunk shares its parent's url + hash."""
    content = "\n\n".join("x" * 80 for _ in range(6))
    chunks = LocalDocumentProcessor(max_chars=120, overlap=10).process(_doc(content))
    assert len(chunks) > 1
    assert {c.source_url for c in chunks} == {"https://example.test/page"}
    assert {c.parent_content_hash for c in chunks} == {PARENT}
