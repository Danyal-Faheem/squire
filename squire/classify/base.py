# SPDX-License-Identifier: AGPL-3.0-or-later
"""The classifier contract (seam 5).

A classifier turns a :class:`~squire.common.schemas.Signal` into a
:class:`~squire.common.schemas.Classification` — the category it belongs to, how
urgent it is, who owns it, and a concise title. How that decision is made is policy:
the default asks an LLM, but a deployment could later supply a rule-based or hybrid
classifier by implementing this one method. The work-item factory depends only on this
interface, never on a concrete classifier.
"""

from __future__ import annotations

from typing import Protocol

from squire.common.schemas import Classification, Signal


class Classifier(Protocol):
    """Turns a signal into a classification."""

    def classify(self, signal: Signal) -> Classification:
        """Return the category, priority, owner, and suggested title for ``signal``."""
        ...
