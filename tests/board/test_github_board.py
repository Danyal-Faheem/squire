# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the GitHub board.

PyGithub is faked at its boundary so these run offline. The most important assertions
here are about the dry-run safety gate: with dry-run on, no call may reach the GitHub
client. The rest confirm that, when live, the board creates an issue with the right
title/body/labels/assignees and comments on the issue named by a reference.
"""

from __future__ import annotations

from typing import Any

import pytest

from squire.board.base import NewWorkItem
from squire.board.github import GitHubBoard


class _FakeIssue:
    def __init__(self, number: int) -> None:
        self.number = number
        self.comments: list[str] = []

    def create_comment(self, body: str) -> None:
        self.comments.append(body)


class _FakeRepo:
    def __init__(self, full_name: str) -> None:
        self.full_name = full_name
        self.created: list[dict[str, Any]] = []
        self.issues: dict[int, _FakeIssue] = {}

    def create_issue(self, **kwargs: Any) -> _FakeIssue:
        self.created.append(kwargs)
        issue = _FakeIssue(number=101)
        self.issues[101] = issue
        return issue

    def get_issue(self, number: int) -> _FakeIssue:
        return self.issues.setdefault(number, _FakeIssue(number))


class _FakeGithub:
    def __init__(self) -> None:
        self.repos: dict[str, _FakeRepo] = {}
        self.get_repo_calls: list[str] = []

    def get_repo(self, full_name: str) -> _FakeRepo:
        self.get_repo_calls.append(full_name)
        return self.repos.setdefault(full_name, _FakeRepo(full_name))


def _item() -> NewWorkItem:
    return NewWorkItem(
        repo="acme/widget",
        title="[bug] Login fails",
        body="A grounded body with citations.",
        assignees=["team-core"],
        labels=["bug", "priority: critical"],
    )


# --- dry-run safety ----------------------------------------------------------------


def test_dry_run_create_makes_no_github_call() -> None:
    """With dry-run on, creating an item never touches the GitHub client."""
    board = GitHubBoard(dry_run=True)
    fake = _FakeGithub()
    board._client = fake

    ref = board.create_item(_item())

    assert fake.get_repo_calls == []  # the GitHub layer was never reached
    assert ref.startswith("dry-run")


def test_dry_run_comment_makes_no_github_call() -> None:
    """With dry-run on, commenting never touches the GitHub client."""
    board = GitHubBoard(dry_run=True)
    fake = _FakeGithub()
    board._client = fake

    board.comment("acme/widget#5", "a reply")

    assert fake.get_repo_calls == []


# --- live behaviour ----------------------------------------------------------------


def test_create_item_opens_issue_with_fields() -> None:
    """When live, the board creates an issue carrying the item's fields."""
    board = GitHubBoard(dry_run=False)
    fake = _FakeGithub()
    board._client = fake

    ref = board.create_item(_item())

    assert ref == "acme/widget#101"
    created = fake.repos["acme/widget"].created[0]
    assert created["title"] == "[bug] Login fails"
    assert created["labels"] == ["bug", "priority: critical"]
    assert created["assignees"] == ["team-core"]


def test_comment_posts_to_referenced_issue() -> None:
    """When live, a comment lands on the issue named by the reference."""
    board = GitHubBoard(dry_run=False)
    fake = _FakeGithub()
    board._client = fake

    board.comment("acme/widget#7", "here is some context")

    assert fake.repos["acme/widget"].issues[7].comments == ["here is some context"]


def test_bad_reference_is_rejected() -> None:
    """A reference that is not owner/repo#number is a clear error."""
    board = GitHubBoard(dry_run=False)
    board._client = _FakeGithub()
    with pytest.raises(ValueError, match="board reference"):
        board.comment("not-a-ref", "x")
