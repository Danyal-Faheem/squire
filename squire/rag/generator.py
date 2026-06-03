# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grounded, cited generation over retrieved chunks.

The generator never answers from the model's own memory. It presents the retrieved
chunks to the model as numbered sources and instructs it to answer using only those,
citing each by number, and to say plainly when the sources do not support an answer.
When retrieval comes back empty there is nothing to ground an answer in, so the
generator short-circuits to that refusal without calling the model at all — this is
the single most important guardrail against invented detail.

The prompt wording comes from the :class:`~squire.prompts.PromptLibrary`, so a
deployment can retune tone and format; the grounding discipline (cite the sources,
refuse without them) is enforced here in code regardless of the templates.
"""

from __future__ import annotations

from squire.common.llm import ChatMessage, LLMClient
from squire.common.vectorstore import SearchResult
from squire.prompts import PromptLibrary
from squire.rag.schemas import Citation, GroundedAnswer

#: Returned when retrieval finds nothing to ground an answer in. This is a fixed engine
#: behaviour, not a per-deployment template: the wording of the *prompts* is
#: configurable, but refusing without evidence is not negotiable.
REFUSAL = (
    "I don't have enough information in the knowledge base to answer that. "
    "No relevant sources were found."
)


class Generator:
    """Generates answers strictly grounded in the chunks it is given."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        prompts: PromptLibrary | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._llm = llm
        self._prompts = prompts or PromptLibrary()
        self._max_tokens = max_tokens

    def answer(self, question: str, chunks: list[SearchResult]) -> GroundedAnswer:
        """Answer a question using only ``chunks`` for evidence."""
        if not chunks:
            return GroundedAnswer(answer=REFUSAL, grounded=False, citations=[])
        user = self._prompts.render(
            "query_user.txt", question=question, sources=_sources(chunks)
        )
        return self._complete("query_system.txt", user, chunks)

    def draft(
        self, title: str, body: str, chunks: list[SearchResult]
    ) -> GroundedAnswer:
        """Draft a maintainer-facing reply to an item, grounded in ``chunks``."""
        if not chunks:
            return GroundedAnswer(answer=REFUSAL, grounded=False, citations=[])
        user = self._prompts.render(
            "draft_user.txt", item=_item_text(title, body), sources=_sources(chunks)
        )
        return self._complete("draft_system.txt", user, chunks)

    def _complete(
        self, system_template: str, user: str, chunks: list[SearchResult]
    ) -> GroundedAnswer:
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._prompts.render(system_template)},
            {"role": "user", "content": user},
        ]
        text = self._llm.complete(messages, max_tokens=self._max_tokens)
        return GroundedAnswer(answer=text, grounded=True, citations=_citations(chunks))


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


def _sources(chunks: list[SearchResult]) -> str:
    """Render the retrieved chunks as a numbered, locator-tagged block."""
    return "\n\n".join(
        f"[{i}] ({chunk.source_url})\n{chunk.content}"
        for i, chunk in enumerate(chunks, start=1)
    )


def _item_text(title: str, body: str) -> str:
    return title if not body else f"{title}\n\n{body}"
