# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification: deciding where a signal goes and how urgent it is.

"Classify and route" is a *policy*, not a fixed mechanism — one team wants an LLM to
infer everything, another wants rules, another wants CODEOWNERS-based routing. So this
is a seam: the :class:`~squire.classify.base.Classifier` interface takes a signal and
returns a :class:`~squire.common.schemas.Classification` (category, priority, owner,
and a suggested title), drawn from the deployment's taxonomy, routing, and priorities.

The engine ships the LLM-backed default, :class:`~squire.classify.llm.LLMClassifier`.
The interface is left open for rule-based or hybrid classifiers later; the work-item
factory depends on the interface, so swapping the classifier changes classification
with no change to the factory.
"""

from __future__ import annotations

from squire.classify.base import Classifier
from squire.classify.llm import LLMClassifier

__all__ = ["Classifier", "LLMClassifier"]
