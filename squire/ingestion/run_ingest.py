# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line entrypoint that wires the orchestrator to real infrastructure.

This is the production wiring behind ``make ingest``: it loads a deployment's
connectors, builds the embedding client, the pgvector store, and the cursor store
against the configured database, and runs a backfill or a live update. The wiring is
kept thin on purpose — all the behaviour lives in the orchestrator and the
connectors, which are tested offline — so this module mostly translates command-line
arguments into those collaborators.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from squire.common.config import get_settings
from squire.common.db import session_scope
from squire.ingestion.orchestrator import IngestionMode, IngestionResult
from squire.runtime import Runtime

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="squire-ingest",
        description="Run Squire's connectors to populate the knowledge base.",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Deployment name or path (defaults to SQUIRE_DEPLOYMENT).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        dest="mode",
        action="store_const",
        const=IngestionMode.BACKFILL,
        help="Backfill every activated connector's full history (the default).",
    )
    mode.add_argument(
        "--live",
        dest="mode",
        action="store_const",
        const=IngestionMode.LIVE,
        help="Read only what changed since each connector's stored watermark.",
    )
    parser.set_defaults(mode=IngestionMode.BACKFILL)
    parser.add_argument(
        "--connector",
        action="append",
        dest="connectors",
        metavar="KEY",
        help="Limit the run to connector(s) with this key (repeatable).",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> IngestionResult:
    """Parse arguments, build collaborators, and run ingestion."""
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    deployment = args.deployment or settings.squire_deployment

    runtime = Runtime(settings, deployment=deployment)
    runtime.validate_embedding_dimension()
    connectors = runtime.build_connectors()
    if args.connectors:
        wanted = set(args.connectors)
        connectors = [c for c in connectors if c.key in wanted]
    if not connectors:
        logger.warning("no connectors selected for deployment %r", deployment)
        return IngestionResult()

    for connector in connectors:
        connector.healthcheck()

    with session_scope() as session:
        orchestrator = runtime.build_orchestrator(session)
        result = orchestrator.run(connectors, mode=args.mode)

    logger.info(
        "ingestion complete: %d documents indexed, %d signals emitted",
        result.documents_indexed,
        result.signals_emitted,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint: configure logging, run ingestion, return an exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
