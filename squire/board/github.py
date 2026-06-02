# SPDX-License-Identifier: AGPL-3.0-or-later
"""The GitHub board: work items as GitHub Issues.

This is the one shipped board backend. It creates issues (with assignees and labels)
and comments on them, identifying each item by an ``owner/repo#number`` reference.

It is also where the dry-run safety gate is enforced for the board. With dry-run on,
``create_item`` and ``comment`` log what they would do and return without touching the
GitHub client at all — so no outward write can escape during development. The API
client is built lazily from a token and is injectable, so the board is exercised
offline against a fake in tests.

(Adding the created issue to a GitHub Projects v2 board is a natural extension within
this same backend; it is intentionally deferred, as it is not needed to create,
assign, label, and comment on items, and requires project-specific configuration.)
"""

from __future__ import annotations

import logging
from typing import Any

from github import Auth, Github

from squire.board.base import NewWorkItem

logger = logging.getLogger(__name__)


class GitHubBoard:
    """A board backed by GitHub Issues."""

    def __init__(self, *, token: str | None = None, dry_run: bool = True) -> None:
        self._token = token
        self._dry_run = dry_run
        self._client: Any | None = None

    def create_item(self, item: NewWorkItem) -> str:
        """Create a GitHub issue for ``item`` and return its ``owner/repo#number`` ref."""
        if self._dry_run:
            logger.info(
                "[dry-run] would create issue in %s: %r (labels=%s, assignees=%s)",
                item.repo,
                item.title,
                item.labels,
                item.assignees,
            )
            return f"dry-run://{item.repo}"

        repo = self._github().get_repo(item.repo)
        issue = repo.create_issue(
            title=item.title,
            body=item.body,
            labels=item.labels,
            assignees=item.assignees,
        )
        ref = f"{item.repo}#{issue.number}"
        logger.info("created board item %s", ref)
        return ref

    def comment(self, item_ref: str, body: str) -> None:
        """Post a comment on the issue identified by ``item_ref``."""
        if self._dry_run:
            logger.info("[dry-run] would comment on %s: %r", item_ref, body)
            return

        repo_name, number = _parse_ref(item_ref)
        issue = self._github().get_repo(repo_name).get_issue(number)
        issue.create_comment(body)
        logger.info("commented on board item %s", item_ref)

    def _github(self) -> Any:
        if self._client is None:
            self._client = (
                Github(auth=Auth.Token(self._token)) if self._token else Github()
            )
        return self._client


def _parse_ref(item_ref: str) -> tuple[str, int]:
    """Split an ``owner/repo#number`` reference into its repository and issue number."""
    repo_name, _, number = item_ref.rpartition("#")
    if not repo_name or not number.isdigit():
        raise ValueError(f"not a GitHub board reference: {item_ref!r}")
    return repo_name, int(number)
