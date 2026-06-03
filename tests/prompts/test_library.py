# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the prompt template library (seam 8).

These cover the seam's promise: the engine ships working defaults, a deployment
overrides any template by dropping a file of the same name in its ``prompts/``
directory, ``$placeholder`` substitution leaves literal braces (JSON examples) intact,
and an unknown template name is a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squire.prompts import PromptLibrary
from squire.prompts.library import PromptError


def test_default_templates_render() -> None:
    """The packaged defaults exist and substitute their placeholders."""
    library = PromptLibrary()
    body = library.render(
        "work_item_body.md",
        summary="Login broken",
        suggested_answer="Restart it.",
        background="- https://src/1",
        source="https://example.test/1",
    )
    assert "**Summary**" in body
    assert "Login broken" in body
    assert "Restart it." in body
    assert "Source: https://example.test/1" in body


def test_literal_braces_are_preserved() -> None:
    """The classify prompt's JSON example survives substitution untouched."""
    rendered = PromptLibrary().render("classify_system.txt")
    assert '{"category": "<name>"' in rendered


def test_deployment_override_takes_effect(tmp_path: Path) -> None:
    """A template in the deployment's prompts/ dir overrides the engine default."""
    overrides = tmp_path / "prompts"
    overrides.mkdir()
    (overrides / "work_item_body.md").write_text(
        "OVERRIDDEN: $summary // $suggested_answer", encoding="utf-8"
    )
    library = PromptLibrary(overrides)

    rendered = library.render(
        "work_item_body.md",
        summary="S",
        suggested_answer="A",
        background="b",
        source="u",
    )

    assert rendered == "OVERRIDDEN: S // A"
    # A template not overridden still falls back to the engine default.
    assert "$question" not in library.render(
        "query_user.txt", question="q", sources="s"
    )


def test_unknown_template_is_a_clear_error() -> None:
    """Rendering a template that exists in neither place fails clearly."""
    with pytest.raises(KeyError):
        PromptLibrary().render("does_not_exist.txt")


def test_typo_in_override_fails_loudly(tmp_path: Path) -> None:
    """A typo'd placeholder in an override raises, naming template + placeholder."""
    overrides = tmp_path / "prompts"
    overrides.mkdir()
    (overrides / "work_item_body.md").write_text(
        "Summary: $summary\nReply: $suggested_answr", encoding="utf-8"
    )
    library = PromptLibrary(overrides)

    with pytest.raises(PromptError) as info:
        library.render(
            "work_item_body.md",
            summary="S",
            suggested_answer="A",
            background="b",
            source="u",
        )
    message = str(info.value)
    assert "work_item_body.md" in message
    assert "suggested_answr" in message
