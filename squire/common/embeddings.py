# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedding abstraction for the whole project.

Embeddings are served separately from the large generation model so they can run
independently — possibly on a different endpoint or provider. As with
:mod:`squire.common.llm`, the only assumption is the OpenAI-compatible protocol, so
swapping the backend is purely a configuration change.

Note on the embedding dimension: the vector column in the database is created with a
fixed dimension (see the initial migration and ``EMBEDDING_DIM``). You may swap the
embedding model freely among models of the *same* dimension; changing to a model with
a different dimension requires a new migration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from openai import OpenAI

from squire.common.config import Settings, get_settings


class _Embeddings(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class EmbeddingsTransport(Protocol):
    """The slice of an OpenAI-compatible client the embedding wrapper uses."""

    @property
    def embeddings(self) -> _Embeddings: ...


class EmbeddingClient:
    """A thin, typed wrapper over an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: EmbeddingsTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # The real client is viewed through our narrow Protocol; cast at this one
        # boundary since the SDK's overloaded signatures don't structurally match.
        self._client: EmbeddingsTransport = client or cast(
            "EmbeddingsTransport",
            OpenAI(
                base_url=self._settings.embedding_base_url,
                api_key=self._settings.embedding_api_key,
            ),
        )

    @property
    def dimension(self) -> int:
        """Return the configured embedding vector dimension."""
        return self._settings.embedding_dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Arguments:
            texts: The strings to embed.

        Returns:
            One embedding vector (list of floats) per input text, in order.
        """
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        return self.embed([text])[0]
