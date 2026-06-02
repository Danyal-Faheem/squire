# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the deployment configuration loader.

The loader is what makes "adopting Squire is configuration, not code" true for the
connector layer: a deployment.yml lists connectors, and the loader turns it into the
ConnectorConfig objects the registry instantiates. These tests cover parsing,
defaulting, environment interpolation (how secrets reach a connector without being
committed), and the failure modes that should surface as clear errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squire.common.deployment import DeploymentConfigError, load_connector_configs


def _write_deployment(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "deployment.yml").write_text(body, encoding="utf-8")
    return directory


def test_loads_connectors_with_settings(tmp_path: Path) -> None:
    """A well-formed deployment yields ConnectorConfig objects in order."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - key: filesystem
            instance_id: docs
            settings:
              root: ./docs
          - key: github
            settings:
              repo: owner/name
        """,
    )

    configs = load_connector_configs(directory)

    assert [c.key for c in configs] == ["filesystem", "github"]
    assert configs[0].instance_id == "docs"
    assert configs[0].settings["root"] == "./docs"


def test_instance_id_defaults_to_key(tmp_path: Path) -> None:
    """A connector that omits instance_id defaults it to the key."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - key: github
            settings:
              repo: owner/name
        """,
    )

    configs = load_connector_configs(directory)

    assert configs[0].instance_id == "github"


def test_env_interpolation_injects_secrets(tmp_path: Path) -> None:
    """${VAR} placeholders are expanded from the supplied environment."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - key: github
            settings:
              repo: owner/name
              token: ${MY_TOKEN}
        """,
    )

    configs = load_connector_configs(directory, env={"MY_TOKEN": "secret-123"})

    assert configs[0].settings["token"] == "secret-123"


def test_unset_env_var_expands_to_empty(tmp_path: Path) -> None:
    """An unset placeholder becomes empty, i.e. optional/anonymous access."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - key: github
            settings:
              repo: owner/name
              token: ${ABSENT}
        """,
    )

    configs = load_connector_configs(directory, env={})

    assert configs[0].settings["token"] == ""


def test_interpolation_recurses_into_nested_values(tmp_path: Path) -> None:
    """Placeholders inside nested mappings and lists are expanded too."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - key: filesystem
            settings:
              roots:
                - ${ROOT_A}
                - static
              nested:
                token: ${ROOT_B}
        """,
    )

    configs = load_connector_configs(directory, env={"ROOT_A": "a", "ROOT_B": "b"})

    settings = configs[0].settings
    assert settings["roots"] == ["a", "static"]
    assert settings["nested"]["token"] == "b"


def test_missing_directory_raises(tmp_path: Path) -> None:
    """Pointing at a directory that does not exist is a clear error."""
    with pytest.raises(DeploymentConfigError):
        load_connector_configs(tmp_path / "nope")


def test_missing_file_raises(tmp_path: Path) -> None:
    """A deployment directory without a deployment.yml is a clear error."""
    (tmp_path / "dep").mkdir()
    with pytest.raises(DeploymentConfigError):
        load_connector_configs(tmp_path / "dep")


def test_connector_without_key_raises(tmp_path: Path) -> None:
    """A connector block missing its key is rejected."""
    directory = _write_deployment(
        tmp_path / "dep",
        """
        connectors:
          - settings:
              root: ./docs
        """,
    )
    with pytest.raises(DeploymentConfigError):
        load_connector_configs(directory)


def test_empty_deployment_has_no_connectors(tmp_path: Path) -> None:
    """A deployment with no connectors section loads as an empty list."""
    directory = _write_deployment(tmp_path / "dep", "board: github\n")
    assert load_connector_configs(directory) == []
