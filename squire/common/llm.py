# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single chat-completion abstraction for the whole project.

Every language-model call in Squire goes through :class:`LLMClient`. The client
speaks the OpenAI-compatible HTTP protocol, so the backend can be any service that
implements it — a self-hosted inference server or a cloud provider. Swapping the
backend is a matter of changing ``llm_base_url`` / ``llm_api_key`` /
``generation_model`` in configuration; no code changes anywhere downstream.

The underlying transport is an *optional constructor parameter* typed as a small
Protocol (:class:`ChatTransport`), defaulting to the real OpenAI-compatible client. Tests
inject a fake through the constructor rather than reaching in to reassign a private
attribute, so the seam is visible and typed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypedDict, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from squire.common.config import Settings, get_settings

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """A single message in a chat completion request."""

    role: Role
    content: str


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions: ...


class ChatTransport(Protocol):
    """The slice of an OpenAI-compatible client the LLM wrapper uses."""

    @property
    def chat(self) -> _Chat: ...


class LLMClient:
    """A thin, typed wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(
        self, settings: Settings | None = None, *, client: ChatTransport | None = None
    ) -> None:
        self._settings = settings or get_settings()
        # The real client is viewed through our narrow Protocol; cast at this one
        # boundary since the SDK's overloaded signatures don't structurally match.
        self._client: ChatTransport = client or cast(
            "ChatTransport",
            OpenAI(
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key,
            ),
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's reply text for a sequence of chat messages.

        Arguments:
            messages: The conversation so far, oldest first.
            temperature: Sampling temperature; low by default for consistency.
            max_tokens: Upper bound on the length of the generated reply.

        Returns:
            The text content of the model's first choice.
        """
        response = self._client.chat.completions.create(
            model=self._settings.generation_model,
            # Our ChatMessage TypedDict is shaped for the API; cast at the
            # boundary to the SDK's own message-param type.
            messages=cast("list[ChatCompletionMessageParam]", list(messages)),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""
