# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for squire.common.llm.

The chat transport is faked and injected through the constructor (per the testability
rule — no reaching in to reassign a private attribute), so these run offline and
assert on the exact request the LLMClient constructs and how it unpacks the response.
"""

from __future__ import annotations

from squire.common.config import Settings
from squire.common.llm import ChatMessage, LLMClient
from tests.fakes import FakeChatTransport


def _client(content: str | None = "pong") -> tuple[LLMClient, list[dict[str, object]]]:
    captured: list[dict[str, object]] = []
    transport = FakeChatTransport(content=content, captured=captured)
    return LLMClient(
        Settings(generation_model="test-model"), client=transport
    ), captured


def test_complete_returns_message_content() -> None:
    """complete() should return the text of the first choice's message."""
    client, _ = _client("pong")
    messages: list[ChatMessage] = [{"role": "user", "content": "ping"}]
    assert client.complete(messages) == "pong"


def test_complete_passes_model_and_params() -> None:
    """The configured model and the call's params must reach the API call."""
    client, captured = _client()
    messages: list[ChatMessage] = [{"role": "user", "content": "ping"}]
    client.complete(messages, temperature=0.7, max_tokens=42)
    assert captured[0]["model"] == "test-model"
    assert captured[0]["temperature"] == 0.7
    assert captured[0]["max_tokens"] == 42
    assert captured[0]["messages"] == messages


def test_complete_handles_null_content() -> None:
    """A None content from the API should become an empty string, not crash."""
    client, _ = _client(content=None)
    assert client.complete([{"role": "user", "content": "x"}]) == ""
