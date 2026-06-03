# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the LLM-backed classifier.

The LLM is faked at its boundary so these run offline and deterministically. The
tests focus on the contract that makes the classifier safe: it returns values from
the configured taxonomy/priorities, derives the owner from routing, and degrades to
defaults whenever the model proposes something unrecognised or unparseable — never
raising and never inventing a category nothing can route.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from squire.classify.llm import LLMClassifier
from squire.common.deployment import (
    Category,
    DeploymentPolicy,
    Priorities,
    Priority,
    Route,
    Routing,
    Taxonomy,
)
from squire.common.schemas import Signal
from tests.fakes import make_llm as _llm


def _policy() -> DeploymentPolicy:
    return DeploymentPolicy(
        taxonomy=Taxonomy(
            categories=[
                Category(name="bug", description="Something is broken"),
                Category(name="question", description="A usage question"),
            ],
            default="question",
        ),
        routing=Routing(
            routes={
                "bug": Route(owners=["team-core"], repo="acme/widget"),
                "question": Route(owners=["team-support"], repo="acme/widget"),
            },
            default=Route(owners=["maintainers"], repo="acme/widget"),
        ),
        priorities=Priorities(
            levels=[
                Priority(key="p0", label="priority: critical"),
                Priority(key="p1", label="priority: high"),
                Priority(key="p2", label="priority: normal"),
            ],
            default="p2",
        ),
    )


def _signal(title: str = "Login fails", body: str = "It errors out") -> Signal:
    return Signal(
        source_type="github_issue",
        source_id="acme/widget#1",
        source_url="https://example.test/acme/widget/issues/1",
        title=title,
        body=body,
        content_hash="h1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_valid_response_is_classified_and_routed() -> None:
    """A well-formed response yields the chosen category, priority, and routed owner."""
    reply = '{"category": "bug", "priority": "p0", "title": "Login is broken"}'
    classifier = LLMClassifier(_llm(reply), _policy())

    result = classifier.classify(_signal())

    assert result.category == "bug"
    assert result.priority_key == "p0"
    assert result.owner == "team-core"  # derived from routing, not the model
    assert result.suggested_title == "Login is broken"


def test_unknown_category_falls_back_to_default() -> None:
    """A category outside the taxonomy degrades to the configured default."""
    reply = '{"category": "nonsense", "priority": "p1", "title": "x"}'
    result = LLMClassifier(_llm(reply), _policy()).classify(_signal())
    assert result.category == "question"  # the taxonomy default
    assert result.owner == "team-support"  # routed from the fallback category


def test_unknown_priority_falls_back_to_default() -> None:
    """A priority outside the scheme degrades to the configured default."""
    reply = '{"category": "bug", "priority": "p9", "title": "x"}'
    result = LLMClassifier(_llm(reply), _policy()).classify(_signal())
    assert result.priority_key == "p2"


def test_malformed_json_uses_defaults_without_raising() -> None:
    """An unparseable response degrades to defaults rather than crashing."""
    result = LLMClassifier(_llm("sorry, I can't do that"), _policy()).classify(
        _signal(title="Original title")
    )
    assert result.category == "question"  # taxonomy default
    assert result.priority_key == "p2"  # priorities default
    assert result.suggested_title == "Original title"  # falls back to the signal title


def test_json_embedded_in_prose_is_extracted() -> None:
    """A JSON object wrapped in extra prose is still parsed."""
    reply = 'Sure! Here is the result:\n{"category": "bug", "priority": "p1", "title": "T"}\nHope that helps.'
    result = LLMClassifier(_llm(reply), _policy()).classify(_signal())
    assert result.category == "bug"
    assert result.priority_key == "p1"


def test_prompt_offers_configured_categories_and_priorities() -> None:
    """The prompt is built from config and offers the allowed choices and guidance."""
    captured: list[Any] = []
    policy = _policy()
    policy.priorities.guidance = "Security issues are p0."
    LLMClassifier(
        _llm('{"category":"bug","priority":"p0","title":"t"}', captured), policy
    ).classify(_signal())

    prompt = captured[0]["messages"][-1]["content"]
    assert "bug: Something is broken" in prompt
    assert "question: A usage question" in prompt
    assert "p0" in prompt and "p2" in prompt
    assert "Security issues are p0." in prompt
