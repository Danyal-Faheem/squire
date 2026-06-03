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

from squire.board.base import Board
from squire.board.github import GitHubBoard
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
from squire.common.schemas import Signal
from squire.common.vectorstore import PgVectorStore
from squire.connectors import registry
from squire.connectors.base import Connector
from squire.factory.store import PgTrackedItemStore
from squire.factory.work_items import WorkItemFactory
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

    def build_board(self) -> Board:
        """Build the configured board backend, honouring the dry-run flag.

        Live writes require GitHub credentials (the App-installation token flow is a
        deferred wiring detail); under dry-run — the default — the board performs no
        external write, so no token is needed.
        """
        return GitHubBoard(dry_run=self._settings.dry_run)

    def build_factory(self, session: Session) -> WorkItemFactory:
        """Wire the work-item factory over a session.

        Requires a deployment policy (it supplies the routing and priority scheme the
        factory needs); raises :class:`MissingDeploymentConfigError` when absent.
        """
        policy = load_policy(self._deployment)
        return WorkItemFactory(
            classifier=LLMClassifier(self._llm, policy),
            drafter=self.build_brain(session),
            board=self.build_board(),
            store=PgTrackedItemStore(session),
            routing=policy.routing,
            priorities=policy.priorities,
            dry_run=self._settings.dry_run,
        )

    def build_signal_handler(self, session: Session) -> SignalHandler | None:
        """Return a factory-backed signal handler, or ``None`` to observe only.

        This is the composition root's "turn signals into board items" decision: when
        a deployment policy exists, new signals are processed by the work-item factory
        into board items; without one — a legitimate early-phase state — it returns
        ``None`` so the orchestrator's default observe-only handler is used. A policy
        that is present but malformed propagates and fails loudly.
        """
        try:
            factory = self.build_factory(session)
        except MissingDeploymentConfigError:
            logger.info(
                "no deployment policy; observing signals without creating board items"
            )
            return None
        return _factory_signal_handler(factory)

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


def _factory_signal_handler(factory: WorkItemFactory) -> SignalHandler:
    """Adapt the factory into the orchestrator's signal-handler shape."""

    def handle(signal: Signal) -> None:
        result = factory.process(signal)
        logger.info(
            "factory outcome %s for signal %s (board item %s)",
            result.outcome,
            signal.source_id,
            result.board_item_ref,
        )

    return handle
