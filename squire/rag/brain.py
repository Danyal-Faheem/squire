# SPDX-License-Identifier: AGPL-3.0-or-later
"""The brain: composes retrieval and generation, and delegates classification.

``Brain`` is the single object the FastAPI layer talks to. ``query`` and ``draft``
retrieve relevant chunks and generate a grounded, cited answer from them. ``classify``
delegates to an injected classifier function.

The classifier is a real seam (``squire.classify``): the composition root passes the
LLM-backed classifier bound to a deployment's taxonomy and routing when one exists. The
brain depends only on the callable shape, so which classifier is active is decided
above it; when no classifier is supplied it uses the project-agnostic default below.
The ``/classify`` endpoint only calls ``brain.classify``, so swapping the classifier
never changes the endpoint.
"""

from __future__ import annotations

from collections.abc import Callable

from squire.common.embeddings import EmbeddingClient
from squire.common.llm import LLMClient
from squire.common.schemas import Classification, Signal
from squire.common.vectorstore import VectorStore
from squire.prompts import PromptLibrary
from squire.rag.generator import Generator
from squire.rag.retriever import Retriever
from squire.rag.schemas import GroundedAnswer

#: A classifier turns a signal into a classification. The ``squire.classify``
#: ``Classifier`` seam provides the real (LLM-backed) implementation; the brain depends
#: only on this callable shape, and an LLMClassifier's ``classify`` method satisfies it.
ClassifierFn = Callable[[Signal], Classification]


def default_classifier(signal: Signal) -> Classification:
    """The project-agnostic fallback classifier, used when no policy is configured.

    It assigns generic placeholders: with no taxonomy, routing, or priority scheme
    configured, the engine cannot do better. The composition root uses it only when a
    deployment has no policy yet; once a policy exists, the LLM-backed classifier from
    the ``Classifier`` seam takes its place.
    """
    return Classification(
        category="uncategorized",
        priority_key="unprioritized",
        owner="unassigned",
        suggested_title=signal.title,
    )


class Brain:
    """Retrieve-and-generate over the knowledge base, with pluggable classification."""

    def __init__(
        self,
        *,
        embedder: EmbeddingClient,
        vector_store: VectorStore,
        llm: LLMClient,
        classifier: ClassifierFn = default_classifier,
        prompts: PromptLibrary | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._retriever = Retriever(embedder, vector_store)
        self._generator = Generator(llm, prompts=prompts, max_tokens=max_tokens)
        self._classifier = classifier

    def query(
        self,
        question: str,
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """Answer a question, grounded in and citing retrieved chunks."""
        chunks = self._retriever.retrieve(
            question, source_types=source_types, top_k=top_k
        )
        return self._generator.answer(question, chunks)

    def draft(
        self,
        title: str,
        body: str = "",
        *,
        source_types: list[str] | None = None,
        top_k: int = 8,
    ) -> GroundedAnswer:
        """Draft a maintainer-facing reply to an item, grounded in retrieved chunks."""
        retrieval_query = title if not body else f"{title}\n\n{body}"
        chunks = self._retriever.retrieve(
            retrieval_query, source_types=source_types, top_k=top_k
        )
        return self._generator.draft(title, body, chunks)

    def classify(self, signal: Signal) -> Classification:
        """Classify a signal via the active classifier."""
        return self._classifier(signal)
