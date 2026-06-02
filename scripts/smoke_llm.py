# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 0 smoke check: send one chat completion to the configured LLM endpoint.

Run with: ``python -m scripts.smoke_llm``. Requires a reachable OpenAI-compatible
LLM endpoint (set LLM_BASE_URL / LLM_API_KEY / GENERATION_MODEL).
"""

from __future__ import annotations

from squire.common.llm import ChatMessage, LLMClient


def main() -> None:
    client = LLMClient()
    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Reply with exactly the word: pong"},
    ]
    reply = client.complete(messages, max_tokens=8)
    print(f"LLM replied: {reply!r}")


if __name__ == "__main__":
    main()
