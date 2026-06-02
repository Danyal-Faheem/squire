# SPDX-License-Identifier: AGPL-3.0-or-later
"""A connector for a hosted Git repository's issues, pull requests, and code.

This is the worked example of a *both-responsibilities* source. A repository is at
once a body of knowledge (its code and the text of its issues and pull requests) and
a stream of incoming work (those same issues and pull requests). So this connector
emits documents from all three and emits signals from the issues and pull requests.

Access is configurable: with a token it reads as an authenticated user (higher rate
limits, private repositories); with no token it reads public repositories
anonymously. Whether to ingest code, whether to ingest issues, which issue state to
watch, and which branch to read are all settings, so the engine itself stays free of
any assumption about a particular repository.

The underlying API client is built lazily and can be injected, which is what lets the
whole connector be exercised offline against a fake.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar

from github import Auth, Github
from pydantic import BaseModel, Field

from squire.connectors.base import ConnectorConfig, Cursor, Document, Signal
from squire.connectors.registry import register

#: Source-type labels this connector owns. They name the *kind* of content, not any
#: particular project, so retrieval can bias toward or away from each.
SOURCE_TYPE_CODE = "github_code"
SOURCE_TYPE_ISSUE = "github_issue"
SOURCE_TYPE_PULL = "github_pull"


class GithubSettings(BaseModel):
    """Settings block for a :class:`GithubConnector` instance."""

    #: Repository in ``owner/name`` form. Required.
    repo: str
    #: Access token. Optional: absent means anonymous, public-only reads. Supply it
    #: from the environment via ``${VAR}`` interpolation rather than inline.
    token: str | None = None
    #: Whether to ingest the repository's file contents as documents.
    include_code: bool = True
    #: Whether to ingest issues and pull requests (as documents *and* signals).
    include_issues: bool = True
    #: Branch to read code from; ``None`` means the repository's default branch.
    branch: str | None = None
    #: Which issues/PRs to consider: ``"open"``, ``"closed"``, or ``"all"``.
    issue_state: str = "open"
    #: File extensions ingested as code (leading dot, compared case-insensitively).
    code_extensions: list[str] = Field(
        default=[
            ".py",
            ".md",
            ".rst",
            ".txt",
            ".js",
            ".ts",
            ".go",
            ".rs",
            ".java",
            ".rb",
            ".cfg",
            ".ini",
            ".toml",
            ".yaml",
            ".yml",
        ]
    )
    #: Skip code blobs larger than this many bytes (avoids embedding huge files).
    max_code_bytes: int = 1_000_000


@register
class GithubConnector:
    """Ingests a repository's code, issues, and pull requests."""

    key: ClassVar[str] = "github"

    def __init__(self, config: ConnectorConfig) -> None:
        self.instance_id = config.instance_id
        self._settings = GithubSettings.model_validate(config.settings)
        self._extensions = {ext.lower() for ext in self._settings.code_extensions}
        # Injectable seams: tests set these directly to run the connector offline.
        self._client: Any | None = None
        self._repo: Any | None = None
        # The newest issue/PR ``updated_at`` observed this run, serialized into the
        # cursor so the next run only fetches items changed since.
        self._watermark: datetime | None = None
        # Issues are needed by both iter_documents and iter_signals; fetch once.
        self._issue_cache: list[Any] | None = None
        self._issue_cache_key: str | None = None

    # -- contract -----------------------------------------------------------------

    def healthcheck(self) -> None:
        """Resolve the repository, surfacing auth or existence problems early."""
        self._repository()

    def iter_documents(self, since: Cursor | None) -> Iterator[Document]:
        """Yield code documents and issue/PR documents."""
        if self._settings.include_code:
            yield from self._iter_code_documents()
        if self._settings.include_issues:
            for issue in self._issues(since):
                yield self._issue_document(issue)

    def iter_signals(self, since: Cursor | None) -> Iterator[Signal]:
        """Yield one signal per issue and pull request."""
        if not self._settings.include_issues:
            return
        for issue in self._issues(since):
            yield self._issue_signal(issue)

    def current_cursor(self) -> Cursor | None:
        """Return the newest issue/PR ``updated_at`` seen, encoded as the cursor."""
        if self._watermark is None:
            return None
        return Cursor(value=self._watermark.isoformat())

    # -- code ---------------------------------------------------------------------

    def _iter_code_documents(self) -> Iterator[Document]:
        repo = self._repository()
        branch = self._settings.branch or repo.default_branch
        head_sha = repo.get_branch(branch).commit.sha
        tree = repo.get_git_tree(head_sha, recursive=True)
        for element in tree.tree:
            if element.type != "blob":
                continue
            if not _has_extension(element.path, self._extensions):
                continue
            content = self._read_blob(repo, element.sha)
            if content is None:
                continue
            origin = f"{repo.full_name}:{element.path}"
            yield Document(
                source_type=SOURCE_TYPE_CODE,
                source_url=f"{repo.html_url}/blob/{branch}/{element.path}",
                origin=origin,
                content=content,
                content_hash=_hash(origin, content),
            )

    def _read_blob(self, repo: Any, sha: str) -> str | None:
        blob = repo.get_git_blob(sha)
        raw = base64.b64decode(blob.content) if blob.encoding == "base64" else b""
        if len(raw) > self._settings.max_code_bytes:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    # -- issues & pull requests ---------------------------------------------------

    def _issues(self, since: Cursor | None) -> list[Any]:
        """Fetch issues/PRs once per run, caching for the second iterator."""
        floor = _parse_dt(since)
        key = since.value if since is not None else ""
        if self._issue_cache is not None and self._issue_cache_key == key:
            return self._issue_cache

        self._observe(floor)
        kwargs: dict[str, Any] = {"state": self._settings.issue_state}
        if floor is not None:
            kwargs["since"] = floor
        issues = list(self._repository().get_issues(**kwargs))
        for issue in issues:
            self._observe(_ensure_aware(issue.updated_at))

        self._issue_cache = issues
        self._issue_cache_key = key
        return issues

    def _issue_document(self, issue: Any) -> Document:
        origin = self._issue_origin(issue)
        content = _issue_content(issue)
        return Document(
            source_type=self._issue_source_type(issue),
            source_url=issue.html_url,
            origin=origin,
            content=content,
            content_hash=_hash(origin, content),
        )

    def _issue_signal(self, issue: Any) -> Signal:
        content = _issue_content(issue)
        return Signal(
            source_type=self._issue_source_type(issue),
            source_id=self._issue_origin(issue),
            source_url=issue.html_url,
            title=issue.title or "",
            body=issue.body or "",
            content_hash=_hash(content),
            created_at=_ensure_aware(issue.created_at),
        )

    def _issue_origin(self, issue: Any) -> str:
        return f"{self._repository().full_name}#{issue.number}"

    @staticmethod
    def _issue_source_type(issue: Any) -> str:
        return SOURCE_TYPE_PULL if issue.pull_request is not None else SOURCE_TYPE_ISSUE

    # -- access -------------------------------------------------------------------

    def _repository(self) -> Any:
        if self._repo is None:
            self._repo = self._github().get_repo(self._settings.repo)
        return self._repo

    def _github(self) -> Any:
        if self._client is None:
            token = self._settings.token
            self._client = Github(auth=Auth.Token(token)) if token else Github()
        return self._client

    def _observe(self, when: datetime | None) -> None:
        if when is None:
            return
        if self._watermark is None or when > self._watermark:
            self._watermark = when


def _issue_content(issue: Any) -> str:
    title = issue.title or ""
    body = issue.body or ""
    return f"{title}\n\n{body}".strip()


def _has_extension(path: str, extensions: set[str]) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(ext) for ext in extensions)


def _ensure_aware(value: datetime) -> datetime:
    """Treat naive timestamps as UTC so cursor comparisons never mix kinds."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _parse_dt(since: Cursor | None) -> datetime | None:
    if since is None:
        return None
    try:
        return _ensure_aware(datetime.fromisoformat(since.value))
    except ValueError:
        return None


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
