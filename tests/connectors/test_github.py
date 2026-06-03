# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the GitHub connector.

PyGithub is faked at its object boundary so these run offline. The fakes expose only
the slice of the API the connector actually touches — a repository with a code tree,
blobs, and an issue list — which doubles as documentation of that surface. The
connector's ``_repo`` / ``_client`` attributes are the injection seams, mirroring how
the existing LLM-client tests swap in a fake transport.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from squire.connectors.base import ConnectorConfig, Cursor
from squire.connectors.github import (
    SOURCE_TYPE_CODE,
    SOURCE_TYPE_ISSUE,
    SOURCE_TYPE_PULL,
    GithubConnector,
)


class _FakeTreeElement:
    def __init__(self, path: str, sha: str, type_: str = "blob") -> None:
        self.path = path
        self.sha = sha
        self.type = type_


class _FakeTree:
    def __init__(self, elements: list[_FakeTreeElement]) -> None:
        self.tree = elements


class _FakeBlob:
    def __init__(self, text: str, encoding: str = "base64") -> None:
        self.content = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.encoding = encoding


class _FakeIssue:
    def __init__(
        self,
        number: int,
        title: str,
        body: str,
        updated_at: datetime,
        created_at: datetime,
        is_pull: bool = False,
    ) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.updated_at = updated_at
        self.created_at = created_at
        self.html_url = f"https://example.test/repo/issues/{number}"
        self.pull_request = object() if is_pull else None


class _FakeRepo:
    def __init__(
        self,
        *,
        full_name: str = "owner/name",
        blobs: dict[str, str] | None = None,
        tree: list[_FakeTreeElement] | None = None,
        issues: list[_FakeIssue] | None = None,
    ) -> None:
        self.full_name = full_name
        self.html_url = f"https://example.test/{full_name}"
        self.default_branch = "main"
        self._blobs = blobs or {}
        self._tree = tree or []
        self._issues = issues or []
        self.get_issues_calls: list[dict[str, Any]] = []

    def get_branch(self, name: str) -> Any:
        commit = type("C", (), {"sha": f"sha-of-{name}"})()
        return type("B", (), {"commit": commit})()

    def get_git_tree(self, sha: str, recursive: bool = False) -> _FakeTree:
        return _FakeTree(self._tree)

    def get_git_blob(self, sha: str) -> _FakeBlob:
        return _FakeBlob(self._blobs[sha])

    def get_issues(self, **kwargs: Any) -> list[_FakeIssue]:
        self.get_issues_calls.append(kwargs)
        return self._issues


class _FakeGithub:
    """A GitHub client whose get_repo returns a pre-built fake repository."""

    def __init__(self, repo: _FakeRepo | None = None) -> None:
        self._repo = repo
        self.requested: list[str] = []

    def get_repo(self, full_name: str) -> _FakeRepo | None:
        self.requested.append(full_name)
        return self._repo


def _connector(repo: _FakeRepo, **settings: object) -> GithubConnector:
    return GithubConnector(
        ConnectorConfig(
            key="github",
            instance_id="gh",
            settings={"repo": "owner/name", **settings},
        ),
        client=_FakeGithub(repo),
    )


def _aware(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def test_ingests_code_documents() -> None:
    """Code blobs with watched extensions become github_code documents."""
    repo = _FakeRepo(
        tree=[
            _FakeTreeElement("src/app.py", "blob1"),
            _FakeTreeElement("README.md", "blob2"),
            _FakeTreeElement("logo.png", "blob3"),  # excluded by extension
            _FakeTreeElement("src", "tree1", type_="tree"),  # not a blob
        ],
        blobs={"blob1": "print('hi')", "blob2": "# Title"},
    )
    connector = _connector(repo, include_issues=False)

    docs = list(connector.iter_documents(None))

    by_origin = {doc.origin: doc for doc in docs}
    assert set(by_origin) == {"owner/name:src/app.py", "owner/name:README.md"}
    app = by_origin["owner/name:src/app.py"]
    assert app.source_type == SOURCE_TYPE_CODE
    assert app.content == "print('hi')"
    assert app.source_url == "https://example.test/owner/name/blob/main/src/app.py"


def test_skips_blobs_over_size_limit() -> None:
    """A blob larger than the configured limit is not ingested."""
    repo = _FakeRepo(
        tree=[_FakeTreeElement("big.py", "b1")],
        blobs={"b1": "x" * 50},
    )
    connector = _connector(repo, include_issues=False, max_code_bytes=10)
    assert list(connector.iter_documents(None)) == []


def test_ingests_issue_documents_and_signals() -> None:
    """Issues become both documents and signals with consistent identity."""
    repo = _FakeRepo(
        issues=[_FakeIssue(7, "Bug", "It broke", _aware(2), _aware(1))],
    )
    connector = _connector(repo, include_code=False)

    docs = list(connector.iter_documents(None))
    signals = list(connector.iter_signals(None))

    assert len(docs) == 1
    assert docs[0].source_type == SOURCE_TYPE_ISSUE
    assert docs[0].origin == "owner/name#7"
    assert "Bug" in docs[0].content and "It broke" in docs[0].content

    assert len(signals) == 1
    signal = signals[0]
    assert signal.source_type == SOURCE_TYPE_ISSUE
    assert signal.source_id == "owner/name#7"
    assert signal.title == "Bug"
    assert signal.created_at == _aware(1)


def test_pull_requests_get_their_own_source_type() -> None:
    """A pull request is tagged github_pull, distinct from a plain issue."""
    repo = _FakeRepo(
        issues=[
            _FakeIssue(9, "Add feature", "diff", _aware(3), _aware(2), is_pull=True)
        ],
    )
    connector = _connector(repo, include_code=False)

    signal = next(iter(connector.iter_signals(None)))
    assert signal.source_type == SOURCE_TYPE_PULL


def test_include_issues_false_emits_nothing_actionable() -> None:
    """With issues disabled there are no issue documents, signals, or cursor."""
    repo = _FakeRepo(issues=[_FakeIssue(1, "x", "y", _aware(2), _aware(1))])
    connector = _connector(repo, include_code=False, include_issues=False)

    assert list(connector.iter_documents(None)) == []
    assert list(connector.iter_signals(None)) == []
    assert connector.current_cursor() is None


def test_cursor_is_max_updated_at() -> None:
    """The cursor advances to the newest issue updated_at observed."""
    repo = _FakeRepo(
        issues=[
            _FakeIssue(1, "a", "a", _aware(2), _aware(1)),
            _FakeIssue(2, "b", "b", _aware(5), _aware(1)),
            _FakeIssue(3, "c", "c", _aware(3), _aware(1)),
        ],
    )
    connector = _connector(repo, include_code=False)

    list(connector.iter_documents(None))
    cursor = connector.current_cursor()

    assert cursor is not None
    assert datetime.fromisoformat(cursor.value) == _aware(5)


def test_incremental_passes_since_to_api() -> None:
    """A cursor is parsed and handed to the API as the since bound."""
    repo = _FakeRepo(issues=[_FakeIssue(1, "a", "a", _aware(9), _aware(1))])
    connector = _connector(repo, include_code=False)

    list(connector.iter_documents(Cursor(value=_aware(4).isoformat())))

    assert repo.get_issues_calls
    assert repo.get_issues_calls[0]["since"] == _aware(4)


def test_issues_fetched_once_across_both_iterators() -> None:
    """Both iterators in a run share a single issue fetch."""
    repo = _FakeRepo(issues=[_FakeIssue(1, "a", "a", _aware(2), _aware(1))])
    connector = _connector(repo, include_code=False)

    list(connector.iter_documents(None))
    list(connector.iter_signals(None))

    assert len(repo.get_issues_calls) == 1


def test_resolves_repo_through_injected_client() -> None:
    """Healthcheck resolves the configured repository through the injected client."""
    client = _FakeGithub(_FakeRepo())
    connector = GithubConnector(
        ConnectorConfig(
            key="github", instance_id="gh", settings={"repo": "owner/name"}
        ),
        client=client,
    )
    connector.healthcheck()

    assert client.requested == ["owner/name"]


def test_missing_required_setting_fails_fast() -> None:
    """Omitting the required repo is rejected at construction, not mid-run."""
    with pytest.raises(ValidationError):
        GithubConnector(ConnectorConfig(key="github", instance_id="gh", settings={}))


def test_healthcheck_propagates_access_error() -> None:
    """A failure to resolve the repository surfaces at healthcheck, before ingest."""

    class _BoomGithub:
        def get_repo(self, full_name: str) -> Any:
            raise RuntimeError("401 Bad credentials")

    connector = GithubConnector(
        ConnectorConfig(
            key="github", instance_id="gh", settings={"repo": "owner/name"}
        ),
        client=_BoomGithub(),
    )
    with pytest.raises(RuntimeError, match="Bad credentials"):
        connector.healthcheck()
