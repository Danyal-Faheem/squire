# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt and item-format templates (seam 8).

Ships default templates for the classify prompt, the answer/draft prompts, and the
work-item body, loaded through :class:`~squire.prompts.library.PromptLibrary` so a
deployment can override any of them from its own ``prompts/`` directory.
"""

from __future__ import annotations

from squire.prompts.library import PromptLibrary

__all__ = ["PromptLibrary"]
