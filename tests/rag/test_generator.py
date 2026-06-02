# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for grounded, cited generation.

These pin down the guarantees that make the brain trustworthy: every citation
corresponds to a chunk that was actually retrieved; the prompt the model receives
contains those chunks' locators and content so it *can* cite them; and when nothing is
retrieved the generator refuses without ever calling the model, so it cannot invent an
answer from the model's own memory.
"""

from __future__ import annotations

from typing import Any

from squire.rag.generator import REFUSAL, Generator
from tests.rag._support import make_llm, result


def test_answer_cites_only_retrieved_chunks() -> None:
    """Citations are exactly the retrieved chunks, numbered in order."""
    chunks = [result(source_url="u1"), result(source_url="u2")]
    generator = Generator(make_llm("Configure via the file [1][2]."))

    answer = generator.answer("how?", chunks)

    assert answer.grounded is True
    assert answer.answer == "Configure via the file [1][2]."
    assert [(c.index, c.source_url) for c in answer.citations] == [(1, "u1"), (2, "u2")]
    # Every citation corresponds to an actually-retrieved chunk.
    retrieved = {c.source_url for c in chunks}
    assert all(c.source_url in retrieved for c in answer.citations)


def test_prompt_contains_locators_and_content() -> None:
    """The model is given each chunk's locator and content, so it can cite them."""
    captured: list[Any] = []
    chunks = [result(source_url="https://example.test/a", content="alpha detail")]
    Generator(make_llm("ok [1]", captured=captured)).answer("q", chunks)

    user_message = captured[0]["messages"][-1]["content"]
    assert "https://example.test/a" in user_message
    assert "alpha detail" in user_message
    assert "[1]" in user_message


def test_empty_retrieval_refuses_without_calling_model() -> None:
    """No chunks means a refusal and no LLM call at all."""
    captured: list[Any] = []
    generator = Generator(make_llm("should not be used", captured=captured))

    answer = generator.answer("anything", [])

    assert answer.grounded is False
    assert answer.answer == REFUSAL
    assert answer.citations == []
    assert captured == []  # the model was never invoked


def test_draft_uses_item_framing_and_cites() -> None:
    """Drafting frames the input as an item and still grounds in the chunks."""
    captured: list[Any] = []
    chunks = [result(source_url="u1", content="relevant background")]
    generator = Generator(make_llm("Suggested reply [1].", captured=captured))

    answer = generator.draft("Login broken", "Steps to reproduce...", chunks)

    assert answer.grounded is True
    assert [c.source_url for c in answer.citations] == ["u1"]
    user_message = captured[0]["messages"][-1]["content"]
    assert "ITEM:" in user_message
    assert "Login broken" in user_message
    assert "relevant background" in user_message


def test_draft_with_no_chunks_refuses() -> None:
    """A draft with no supporting chunks also refuses rather than guessing."""
    captured: list[Any] = []
    answer = Generator(make_llm(captured=captured)).draft("title", "body", [])
    assert answer.grounded is False
    assert answer.citations == []
    assert captured == []
