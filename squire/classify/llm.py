# SPDX-License-Identifier: AGPL-3.0-or-later
"""The LLM-backed default classifier.

It asks the model to choose a category and a priority from the deployment's configured
sets and to propose a concise title, then derives the owner deterministically from the
routing policy. The model's freedom is bounded on both sides: the prompt offers only
the configured categories and priorities, and whatever the model returns is validated
back against those sets — anything unrecognised (or any unparseable response) falls
back to the configured defaults. So a misbehaving model degrades to a safe default
rather than inventing a category that nothing downstream can route.

The prompt is built entirely from configuration, so this module names no project,
category, or priority of its own.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from squire.common.deployment import DeploymentPolicy
from squire.common.llm import ChatMessage, LLMClient
from squire.common.schemas import Classification, Signal
from squire.prompts import PromptLibrary

logger = logging.getLogger(__name__)


class LLMClassifier:
    """A :class:`~squire.classify.base.Classifier` that asks an LLM to triage.

    The prompt wording comes from the :class:`~squire.prompts.PromptLibrary` (the
    ``classify_*`` templates), so a deployment can retune it; the bounding to the
    configured categories and priorities, and the safe-default fallback, are enforced
    here in code regardless.
    """

    def __init__(
        self,
        llm: LLMClient,
        policy: DeploymentPolicy,
        *,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self._llm = llm
        self._policy = policy
        self._prompts = prompts or PromptLibrary()

    def classify(self, signal: Signal) -> Classification:
        """Classify a signal, falling back to safe defaults on any uncertainty."""
        raw = self._llm.complete(self._messages(signal), temperature=0.0)
        proposed = _parse(raw)

        taxonomy = self._policy.taxonomy
        priorities = self._policy.priorities
        category = taxonomy.resolve(str(proposed.get("category", "")))
        priority = priorities.resolve(str(proposed.get("priority", "")))
        title = str(proposed.get("title") or "").strip() or signal.title
        owner = self._policy.routing.resolve(category).primary_owner

        return Classification(
            category=category,
            priority_key=priority,
            owner=owner,
            suggested_title=title,
        )

    def _messages(self, signal: Signal) -> list[ChatMessage]:
        return [
            {"role": "system", "content": self._prompts.render("classify_system.txt")},
            {"role": "user", "content": self._user_prompt(signal)},
        ]

    def _user_prompt(self, signal: Signal) -> str:
        categories = _bullets(
            f"{c.name}: {c.description}" if c.description else c.name
            for c in self._policy.taxonomy.categories
        )
        priorities = _bullets(
            f"{p.key}: {p.description}" if p.description else p.key
            for p in self._policy.priorities.levels
        )
        guidance = self._policy.priorities.guidance.strip()
        guidance_block = f"PRIORITY GUIDANCE: {guidance}\n\n" if guidance else ""
        item = signal.title if not signal.body else f"{signal.title}\n\n{signal.body}"
        return self._prompts.render(
            "classify_user.txt",
            categories=categories,
            priorities=priorities,
            guidance=guidance_block,
            item=item,
        )


def _bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _parse(raw: str) -> dict[str, object]:
    """Extract the JSON object from a model response, tolerating stray prose.

    Returns an empty mapping on any failure; the caller then uses defaults, so a
    malformed response never propagates as an exception or an invalid classification.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.warning("classifier response contained no JSON object")
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("classifier response was not valid JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}
