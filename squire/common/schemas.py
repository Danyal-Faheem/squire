# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, typed domain models used across the engine.

These are deliberately framework-light pydantic models so they travel between the
ingestion, RAG, factory, and adapter layers without coupling those layers together.

Note that ``source_type`` is a plain string, not a closed enum. The set of source
types is *open* — it is whatever connectors are registered — so the engine must not
hardcode a fixed list. Connectors set this field to a value they own (for example a
GitHub connector might emit ``"github_code"`` and ``"github_issue"``).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Document(BaseModel):
    """A unit of indexed content plus the metadata retrieval filters on.

    Produced by a connector's ``iter_documents``. ``source_type`` is owned by the
    producing connector; ``source_url`` locates the content; ``origin`` names the
    repo, thread, or path it came from.
    """

    source_type: str
    source_url: str
    origin: str | None = None
    content: str
    content_hash: str


class Signal(BaseModel):
    """A new, potentially-actionable item observed by a connector.

    Produced by a connector's ``iter_signals``. The work-item factory may turn a
    signal into a board task. Document-only sources never emit signals.
    """

    source_type: str
    source_id: str
    source_url: str
    title: str
    body: str
    content_hash: str
    created_at: datetime


class Classification(BaseModel):
    """A classifier's verdict for a signal: where it goes and how urgent it is.

    Returned by the active :class:`Classifier` implementation. ``category`` and
    ``owner`` are drawn from the deployment's taxonomy and routing; ``priority_key``
    from its priority scheme.
    """

    category: str
    priority_key: str
    owner: str
    suggested_title: str
