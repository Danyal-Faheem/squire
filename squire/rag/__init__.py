# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RAG brain: retrieve relevant chunks and generate grounded, cited answers.

The brain has three jobs — retrieve, generate, and (a thin delegation for now)
classify — and is exposed as a FastAPI service. It is built directly on the engine's
seams: query embedding goes through :class:`~squire.common.embeddings.EmbeddingClient`,
similarity search through the :class:`~squire.common.vectorstore.VectorStore`, and
generation through :class:`~squire.common.llm.LLMClient`. No retrieval or generation
framework stands in front of those seams, so the brain talks only to the
OpenAI-compatible client and stays trivially testable offline.
"""

from __future__ import annotations

from squire.rag.brain import Brain, ClassifierFn, default_classifier
from squire.rag.generator import Generator
from squire.rag.retriever import Retriever
from squire.rag.schemas import Citation, GroundedAnswer

__all__ = [
    "Brain",
    "Citation",
    "ClassifierFn",
    "Generator",
    "GroundedAnswer",
    "Retriever",
    "default_classifier",
]
