# SPDX-License-Identifier: AGPL-3.0-or-later
"""The prompt and item-format template system (seam 8).

The language Squire "thinks in" — how it asks the model to classify, how it asks for a
grounded answer, and how it lays out a work item — differs by project: one wants terse
and citation-heavy, another warm and explanatory. So the templates are a seam. The
engine ships sensible defaults; a deployment overrides any of them by dropping a file
of the same name in its own ``prompts/`` directory.

Templates use ``string.Template`` ``$placeholder`` syntax rather than ``str.format``
``{}`` precisely because some prompts contain literal braces (a JSON example in the
classify prompt), which ``$`` substitution leaves untouched. Dynamic lists (sources,
categories, the work-item background) are rendered to strings by the caller and passed
in as placeholder values, so the templates stay flat and easy to override.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

_DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"


class PromptError(Exception):
    """Raised when a prompt template is malformed or references an unknown placeholder."""


class PromptLibrary:
    """Loads prompt templates, preferring a deployment's overrides over the defaults."""

    def __init__(self, overrides_dir: Path | None = None) -> None:
        self._overrides_dir = overrides_dir
        self._cache: dict[str, str] = {}

    def template(self, name: str) -> str:
        """Return the raw template text for ``name`` (override if present, else default)."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def render(self, name: str, /, **values: str) -> str:
        """Render template ``name``, substituting ``$placeholder`` values.

        Substitution is *strict*: a placeholder the caller did not provide — almost
        always a typo in a deployment override, like ``$suggested_answr`` — fails
        loudly here rather than silently leaking the literal ``$placeholder`` into the
        prompt or work item. The error names the template and the offending placeholder.
        """
        template = Template(self.template(name))
        try:
            return template.substitute(values)
        except KeyError as exc:
            raise PromptError(
                f"prompt template {name!r} uses unknown placeholder ${exc.args[0]} "
                f"(provided: {sorted(values)}); check the deployment override"
            ) from exc
        except ValueError as exc:
            raise PromptError(
                f"prompt template {name!r} has invalid placeholder syntax: {exc}"
            ) from exc

    def _load(self, name: str) -> str:
        if self._overrides_dir is not None:
            override = self._overrides_dir / name
            if override.is_file():
                return override.read_text(encoding="utf-8")
        default = _DEFAULTS_DIR / name
        if not default.is_file():
            raise KeyError(f"no prompt template named {name!r}")
        return default.read_text(encoding="utf-8")
