# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for squire.common.llm.

The OpenAI client is faked at its boundary so these run offline and assert on the
exact request the LLMClient constructs and how it unpacks the response.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from squire.common.config import Settings
from squire.common.llm import ChatMessage, LLMClient


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        message = SimpleNamespace(content="pong")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _build_client_with_fake() -> tuple[LLMClient, _FakeCompletions]:
    client = LLMClient(settings=Settings(generation_model="test-model"))
    fake = _FakeCompletions()
    # Deliberately swap in a fake transport so the test runs offline. The narrow
    # ignore documents that this assignment is intentional, not a type error.
    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=fake)
    )
    return client, fake


def test_complete_returns_message_content() -> None:
    """complete() should return the text of the first choice's message."""
    client, _ = _build_client_with_fake()
    messages: list[ChatMessage] = [{"role": "user", "content": "ping"}]
    assert client.complete(messages) == "pong"


def test_complete_passes_model_and_params() -> None:
    """The configured model and the call's params must reach the API call."""
    client, fake = _build_client_with_fake()
    messages: list[ChatMessage] = [{"role": "user", "content": "ping"}]
    client.complete(messages, temperature=0.7, max_tokens=42)
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["model"] == "test-model"
    assert fake.last_kwargs["temperature"] == 0.7
    assert fake.last_kwargs["max_tokens"] == 42
    assert fake.last_kwargs["messages"] == messages


def test_complete_handles_null_content() -> None:
    """A None content from the API should become an empty string, not crash."""
    client, fake = _build_client_with_fake()

    def _create_none(**kwargs: Any) -> Any:
        choice = SimpleNamespace(message=SimpleNamespace(content=None))
        return SimpleNamespace(choices=[choice])

    fake.create = _create_none  # type: ignore[method-assign]
    assert client.complete([{"role": "user", "content": "x"}]) == ""
