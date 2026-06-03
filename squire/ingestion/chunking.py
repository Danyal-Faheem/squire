# SPDX-License-Identifier: AGPL-3.0-or-later
"""Document processing: splitting source documents into retrieval-sized chunks.

This is seam 4b, sitting between connectors and embedding. Connectors emit
*source-level* :class:`~squire.common.schemas.Document`s — a whole page, PDF, or
thread — but embedding one as a single vector makes that vector a blurry average of
everything in it, which wrecks retrieval. So a :class:`DocumentProcessor` splits each
document into :class:`~squire.common.schemas.Chunk`s with stable lineage back to their
source: same source_type/url/origin, plus the parent's content hash and the chunk's
position, so a citation still resolves to the original.

The engine ships one default — :class:`LocalDocumentProcessor`, a simple
structure-then-length splitter with no heavy dependency. The seam is the interface; a
LlamaIndex splitter could sit behind it later without changing anything downstream.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from squire.common.schemas import Chunk, Document

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


class DocumentProcessor(Protocol):
    """Splits a source document into chunks ready for embedding."""

    def process(self, document: Document) -> list[Chunk]:
        """Return the chunks of ``document``, in order; empty if it has no content."""
        ...


class LocalDocumentProcessor:
    """A local, dependency-free structure- and length-aware splitter.

    It first splits on blank lines (paragraphs/sections), then greedily packs
    paragraphs into chunks up to ``max_chars``; a single paragraph larger than the
    limit is hard-split into overlapping windows so no chunk exceeds the limit. The
    overlap carries a little context across hard boundaries.
    """

    def __init__(self, *, max_chars: int = 1500, overlap: int = 150) -> None:
        if overlap >= max_chars:
            raise ValueError("overlap must be smaller than max_chars")
        self._max_chars = max_chars
        self._overlap = overlap

    def process(self, document: Document) -> list[Chunk]:
        segments = self._split(document.content)
        return [
            Chunk(
                source_type=document.source_type,
                source_url=document.source_url,
                origin=document.origin,
                content=segment,
                content_hash=_chunk_hash(document.content_hash, index, segment),
                parent_content_hash=document.content_hash,
                chunk_index=index,
                title=_title(segment),
            )
            for index, segment in enumerate(segments)
        ]

    def _split(self, content: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > self._max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._hard_split(paragraph))
                continue
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) > self._max_chars:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        step = self._max_chars - self._overlap
        return [
            text[start : start + self._max_chars] for start in range(0, len(text), step)
        ]


def _title(segment: str) -> str | None:
    first_line = segment.splitlines()[0] if segment else ""
    match = _HEADING.match(first_line)
    return match.group(1) if match else None


def _chunk_hash(parent_content_hash: str, index: int, content: str) -> str:
    digest = hashlib.sha256()
    for part in (parent_content_hash, str(index), content):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
