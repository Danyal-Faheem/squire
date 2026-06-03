# SPDX-License-Identifier: AGPL-3.0-or-later
"""The composition root.

:class:`Runtime` is the single place that reads configuration and constructs and wires
the engine's concrete implementations into its components. Entrypoints (the ingestion
CLI, the RAG service) ask the runtime for fully-wired components rather than building
collaborators themselves, so dependencies stay explicit and the engine stays free of
module-level global wiring.
"""

from __future__ import annotations

from squire.runtime.composition import Runtime

__all__ = ["Runtime"]
