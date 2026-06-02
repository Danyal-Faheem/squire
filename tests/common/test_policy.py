# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the deployment policy loaders (taxonomy, routing, priorities).

The policy is the project-specific configuration the classifier and factory consume.
These tests cover loading the three files, the resolve/fallback behaviour the engine
relies on when a classifier proposes an unknown value, environment interpolation
(repos can be supplied from the environment), and clear errors on malformed config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squire.common.deployment import (
    DeploymentConfigError,
    Priorities,
    Routing,
    Taxonomy,
    load_policy,
)


def _write_policy(
    directory: Path,
    *,
    taxonomy: str | None = None,
    routing: str | None = None,
    priorities: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "taxonomy.yml").write_text(
        taxonomy
        or """
        categories:
          - name: bug
            description: Something is broken
          - name: question
            description: A usage question
        default: question
        """,
        encoding="utf-8",
    )
    (directory / "routing.yml").write_text(
        routing
        or """
        routes:
          bug:
            owners: [team-core, team-qa]
            repo: acme/widget
          question:
            owners: [team-support]
            repo: acme/widget
        default:
          owners: [maintainers]
          repo: acme/widget
        """,
        encoding="utf-8",
    )
    (directory / "priorities.yml").write_text(
        priorities
        or """
        levels:
          - key: p0
            label: "priority: critical"
          - key: p1
            label: "priority: high"
          - key: p2
            label: "priority: normal"
        default: p2
        guidance: Security issues are p0.
        """,
        encoding="utf-8",
    )
    return directory


def test_load_policy_parses_all_three(tmp_path: Path) -> None:
    """A deployment's three policy files load into typed models."""
    policy = load_policy(_write_policy(tmp_path / "dep"))

    assert isinstance(policy.taxonomy, Taxonomy)
    assert isinstance(policy.routing, Routing)
    assert isinstance(policy.priorities, Priorities)
    assert policy.taxonomy.names == ["bug", "question"]
    assert policy.priorities.keys == ["p0", "p1", "p2"]
    assert policy.priorities.guidance == "Security issues are p0."


def test_taxonomy_resolve_and_fallback(tmp_path: Path) -> None:
    """A known category resolves to itself; an unknown one to the default."""
    taxonomy = load_policy(_write_policy(tmp_path / "dep")).taxonomy
    assert taxonomy.resolve("bug") == "bug"
    assert taxonomy.resolve("nonsense") == "question"  # the declared default
    assert taxonomy.fallback == "question"


def test_taxonomy_fallback_defaults_to_first_when_unset(tmp_path: Path) -> None:
    """With no explicit default, the first category is the fallback."""
    taxonomy = load_policy(
        _write_policy(
            tmp_path / "dep",
            taxonomy="categories:\n  - name: alpha\n  - name: beta\n",
        )
    ).taxonomy
    assert taxonomy.fallback == "alpha"
    assert taxonomy.resolve("unknown") == "alpha"


def test_routing_resolves_category_and_default(tmp_path: Path) -> None:
    """Known categories use their route; unknown ones use the default route."""
    routing = load_policy(_write_policy(tmp_path / "dep")).routing
    assert routing.resolve("bug").owners == ["team-core", "team-qa"]
    assert routing.resolve("bug").primary_owner == "team-core"
    assert routing.resolve("bug").repo == "acme/widget"
    assert routing.resolve("unknown").owners == ["maintainers"]


def test_priorities_label_and_resolve(tmp_path: Path) -> None:
    """Priority keys resolve and map to their board labels."""
    priorities = load_policy(_write_policy(tmp_path / "dep")).priorities
    assert priorities.resolve("p0") == "p0"
    assert priorities.resolve("bogus") == "p2"  # the declared default
    assert priorities.label_for("p0") == "priority: critical"
    assert priorities.label_for("unknown-key") == "unknown-key"


def test_env_interpolation_in_policy(tmp_path: Path) -> None:
    """${VAR} placeholders in policy values are expanded from the environment."""
    routing = """
    routes: {}
    default:
      owners: [maintainers]
      repo: ${TARGET_REPO}
    """
    policy = load_policy(
        _write_policy(tmp_path / "dep", routing=routing),
        env={"TARGET_REPO": "acme/from-env"},
    )
    assert policy.routing.default.repo == "acme/from-env"


def test_missing_policy_file_raises(tmp_path: Path) -> None:
    """A deployment lacking a policy file is a clear error."""
    directory = tmp_path / "dep"
    directory.mkdir()
    (directory / "taxonomy.yml").write_text("categories: []\n", encoding="utf-8")
    with pytest.raises(DeploymentConfigError):
        load_policy(directory)


def test_invalid_policy_raises(tmp_path: Path) -> None:
    """A malformed policy file surfaces as a configuration error, not a crash."""
    # routing.yml without the required 'default' route.
    with pytest.raises(DeploymentConfigError):
        load_policy(
            _write_policy(tmp_path / "dep", routing="routes: {}\n"),
        )
