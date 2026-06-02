# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared result models for the RAG brain.

These travel from the brain out through the FastAPI layer, so they double as the
service's response bodies. A :class:`Citation` is the evidence behind an answer — one
per retrieved chunk the answer was grounded in — and always carries the source locator
so a reader can follow every claim back to its origin.
"""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """A single piece of evidence: a retrieved chunk and how relevant it was.

    ``index`` is the number the chunk was given in the prompt, so an inline ``[n]``
    marker in the answer text maps back to a specific source. ``source_url`` is the
    locator the answer cites.
    """

    index: int
    source_type: str
    source_url: str
    origin: str | None = None
    score: float


class GroundedAnswer(BaseModel):
    """An answer plus the evidence it stands on.

    ``grounded`` is ``False`` when retrieval found nothing relevant; in that case the
    answer states that it cannot be supported rather than inventing detail, and
    ``citations`` is empty. When ``grounded`` is ``True`` the citations are exactly the
    chunks supplied to the model, so every citation corresponds to a real retrieval.
    """

    answer: str
    grounded: bool
    citations: list[Citation]
