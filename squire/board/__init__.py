# SPDX-License-Identifier: AGPL-3.0-or-later
"""The board: where work items are created, assigned, labelled, and commented on.

The board is a seam (seam 6). The engine defines the :class:`~squire.board.base.Board`
interface and ships a single backend — :class:`~squire.board.github.GitHubBoard`,
which creates GitHub Issues — so a Jira or Linear board could slot in later by
implementing the same interface. The work-item factory and the command core depend
only on the interface.

Every outward write goes through the board, which is therefore where the global
dry-run safety gate lives: with ``DRY_RUN`` on, the board logs what it would do and
performs no real write.
"""

from __future__ import annotations

from squire.board.base import Board, NewWorkItem
from squire.board.github import GitHubBoard

__all__ = ["Board", "GitHubBoard", "NewWorkItem"]
