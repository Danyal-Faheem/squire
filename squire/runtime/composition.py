# SPDX-License-Identifier: AGPL-3.0-or-later
"""The composition root: construct concrete implementations and wire them up.

As more seams come online, construction logic must not scatter across entrypoints and
dependencies must not hide in module-level globals. :class:`Runtime` is the single
place that reads the environment and the selected deployment, builds the concrete
implementations (the clients, the ORM-backed stores, the classifier, the board), and
wires them into the engine's components — which themselves depend only on interfaces.

The long-lived collaborators (settings, the LLM and embedding clients) are built once
per runtime; the per-transaction collaborators (the stores, and the brain/orchestrator
that hold them) are built against a caller-supplied session. The clients are injectable
so a runtime can be built for tests without touching the network.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from squire.classify.llm import LLMClassifier
from squire.common import models
from squire.common.config import Settings, get_settings
from squire.common.deployment import (
    MissingDeploymentConfigError,
    load_connector_configs,
    load_policy,
)
from squire.common.embeddings import EmbeddingClient
from squire.common.llm import LLMClient
from squire.common.vectorstore import PgVectorStore
from squire.connectors import registry
from squire.connectors.base import Connector
from squire.ingestion.cursors import PgCursorStore
from squire.ingestion.orchestrator import IngestionOrchestrator, SignalHandler
from squire.rag.brain import Brain, ClassifierFn, default_classifier

logger = logging.getLogger(__name__)


class Runtime:
    """Builds and wires the engine's components from configuration."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        deployment: str | Path | None = None,
        llm: LLMClient | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._deployment: str | Path = (
            deployment if deployment is not None else self._settings.squire_deployment
        )
        self._llm = llm or LLMClient(self._settings)
        self._embedder = embedder or EmbeddingClient(self._settings)

    def validate_embedding_dimension(self) -> None:
        """Fail loudly at startup if config disagrees with the schema's vector width.

        The embedding dimension is baked into the ``chunks`` table; a configured
        ``EMBEDDING_DIM`` that does not match it would write or query vectors the
        column rejects. Catching the mismatch here turns a confusing mid-run database
        error into a clear startup failure.
        """
        if self._settings.embedding_dim != models.EMBEDDING_DIM:
            raise RuntimeError(
                f"EMBEDDING_DIM is {self._settings.embedding_dim} but the schema's "
                f"vector column has dimension {models.EMBEDDING_DIM}; they must match "
                "(changing the embedding dimension requires a migration)."
            )

    def build_connectors(self) -> list[Connector]:
        """Instantiate the deployment's activated connectors via the registry."""
        return registry.create_all(load_connector_configs(self._deployment))

    def classifier(self) -> ClassifierFn:
        """Return the active classifier.

        When the deployment provides a policy (taxonomy, routing, priorities), this is
        the LLM-backed classifier bound to that policy. When the policy is genuinely
        *absent* — a legitimate early-phase state — it falls back to the trivial
        default so the engine still runs. A policy that is present but *malformed* is
        not caught: it propagates and fails loudly, because silently degrading to
        "everything uncategorized" on a typo is the worst kind of quiet config error.
        """
        try:
            policy = load_policy(self._deployment)
        except MissingDeploymentConfigError:
            logger.info("no deployment policy found; using the default classifier")
            return default_classifier
        return LLMClassifier(self._llm, policy).classify

    def build_brain(self, session: Session) -> Brain:
        """Wire a brain over a request-scoped session and the shared clients."""
        return Brain(
            embedder=self._embedder,
            vector_store=PgVectorStore(session),
            llm=self._llm,
            classifier=self.classifier(),
        )

    def build_orchestrator(
        self, session: Session, *, signal_handler: SignalHandler | None = None
    ) -> IngestionOrchestrator:
        """Wire an ingestion orchestrator over a session and the shared embedder."""
        return IngestionOrchestrator(
            embedder=self._embedder,
            vector_store=PgVectorStore(session),
            cursors=PgCursorStore(session),
            signal_handler=signal_handler,
        )
