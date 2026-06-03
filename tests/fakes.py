# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, typed fakes injected through constructors.

The LLM and embedding clients wrap an OpenAI-compatible transport, accepted as an
optional constructor parameter typed as a small Protocol. These helpers build the real
wrapper around a typed fake transport, so tests inject the fake through the constructor
rather than reassigning a private attribute. The transport fakes are concrete typed
classes (not ``SimpleNamespace``) precisely so they satisfy those Protocols statically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from squire.common.config import Settings
from squire.common.embeddings import EmbeddingClient
from squire.common.llm import LLMClient


class _FakeChatCompletions:
    def __init__(self, content: str | None, captured: list[dict[str, Any]]) -> None:
        self._content = content
        self._captured = captured

    def create(self, **kwargs: Any) -> Any:
        self._captured.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class FakeChatTransport:
    """A typed fake satisfying the LLM client's ChatTransport protocol."""

    def __init__(
        self, content: str | None = "ok", captured: list[dict[str, Any]] | None = None
    ) -> None:
        self.captured: list[dict[str, Any]] = [] if captured is None else captured
        self.chat = _FakeChat(_FakeChatCompletions(content, self.captured))


class _FakeEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def create(self, **kwargs: Any) -> Any:
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(self._vector)) for _ in inputs]
        )


class FakeEmbeddingTransport:
    """A typed fake satisfying the embedding client's EmbeddingsTransport protocol."""

    def __init__(self, vector: list[float]) -> None:
        self.embeddings = _FakeEmbeddings(vector)


def make_llm(
    content: str | None = "a grounded answer [1]",
    captured: list[dict[str, Any]] | None = None,
) -> LLMClient:
    """Return a real LLMClient wired to a fake transport returning ``content``.

    Pass ``captured`` to collect each call's keyword arguments (notably ``messages``).
    """
    transport = FakeChatTransport(content=content, captured=captured)
    return LLMClient(Settings(generation_model="test-model"), client=transport)


def make_embedder(dim: int = 4, vector: list[float] | None = None) -> EmbeddingClient:
    """Return a real EmbeddingClient wired to a fake transport returning a fixed vector."""
    fixed = vector if vector is not None else [0.1] * dim
    return EmbeddingClient(
        Settings(embedding_dim=dim), client=FakeEmbeddingTransport(fixed)
    )
