# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grounded, cited generation over retrieved chunks.

The generator never answers from the model's own memory. It presents the retrieved
chunks to the model as numbered sources and instructs it to answer using only those,
citing each by number, and to say plainly when the sources do not support an answer.
When retrieval comes back empty there is nothing to ground an answer in, so the
generator short-circuits to that refusal without calling the model at all — this is
the single most important guardrail against invented detail.

The prompt text here is the engine's sensible default. Phase 5 introduces the prompts
seam that lets a deployment override these templates; until then they live as module
constants, clearly marked, so nothing downstream depends on where they come from.
"""

from __future__ import annotations

from squire.common.llm import ChatMessage, LLMClient
from squire.common.vectorstore import SearchResult
from squire.rag.schemas import Citation, GroundedAnswer

# --- default prompt templates (moved behind the prompts seam in Phase 5) -----------

REFUSAL = (
    "I don't have enough information in the knowledge base to answer that. "
    "No relevant sources were found."
)

_QUERY_SYSTEM = (
    "You are a careful assistant answering questions about a software project. "
    "Answer using only the information in the numbered SOURCES provided. Cite the "
    "source number in square brackets, like [1], immediately after each claim it "
    "supports. If the sources do not contain enough information to answer, say so "
    "plainly and do not invent details or rely on outside knowledge."
)

_DRAFT_SYSTEM = (
    "You are assisting a project maintainer by drafting a reply to an incoming item. "
    "Using only the information in the numbered SOURCES, write a concise, helpful "
    "suggested response the maintainer could send. Cite the source number in square "
    "brackets, like [1], after each claim it supports. If the sources do not cover "
    "the item, say plainly that more information is needed rather than guessing."
)


class Generator:
    """Generates answers strictly grounded in the chunks it is given."""

    def __init__(self, llm: LLMClient, *, max_tokens: int = 1024) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def answer(self, question: str, chunks: list[SearchResult]) -> GroundedAnswer:
        """Answer a question using only ``chunks`` for evidence."""
        return self._generate(_QUERY_SYSTEM, _question_block(question), chunks)

    def draft(
        self, title: str, body: str, chunks: list[SearchResult]
    ) -> GroundedAnswer:
        """Draft a maintainer-facing reply to an item, grounded in ``chunks``."""
        return self._generate(_DRAFT_SYSTEM, _item_block(title, body), chunks)

    def _generate(
        self, system: str, task_block: str, chunks: list[SearchResult]
    ) -> GroundedAnswer:
        if not chunks:
            # Nothing retrieved: refuse rather than letting the model improvise.
            return GroundedAnswer(answer=REFUSAL, grounded=False, citations=[])

        citations = _citations(chunks)
        messages: list[ChatMessage] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{task_block}\n\n{_sources_block(chunks)}"},
        ]
        text = self._llm.complete(messages, max_tokens=self._max_tokens)
        return GroundedAnswer(answer=text, grounded=True, citations=citations)


def _citations(chunks: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            index=i,
            source_type=chunk.source_type,
            source_url=chunk.source_url,
            origin=chunk.origin,
            score=chunk.score,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]


def _sources_block(chunks: list[SearchResult]) -> str:
    lines = ["SOURCES:"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({chunk.source_url})\n{chunk.content}")
    return "\n\n".join(lines)


def _question_block(question: str) -> str:
    return f"QUESTION:\n{question}"


def _item_block(title: str, body: str) -> str:
    item = title if not body else f"{title}\n\n{body}"
    return f"ITEM:\n{item}"
