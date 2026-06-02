# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loading the deployment tier of configuration.

Squire's configuration has two tiers. The *environment* tier (secrets and
infrastructure, identical across deployments) is handled by
:mod:`squire.common.config`. This module handles the *deployment* tier: the
project-specific configuration that lives in a directory under ``deployments/`` —
which connectors to activate and with what settings, plus, in later phases, the
taxonomy, routing, and priority scheme.

For now it reads the ``connectors`` section of a deployment's ``deployment.yml`` into
:class:`~squire.connectors.base.ConnectorConfig` objects. Settings strings may contain
``${VAR}`` placeholders, which are expanded from the environment at load time. That is
how a secret (an access token) reaches a connector without ever being written into a
committed configuration file: the file references the variable, and the value comes
from the environment. This module is the single place that reads the environment for
this purpose; the environment is injectable so the behaviour is testable.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from squire.connectors.base import ConnectorConfig

#: Default base directory holding deployment directories, relative to the working
#: directory (the repository root when running via the Make targets or the CLI).
DEFAULT_DEPLOYMENTS_DIR = Path("deployments")

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DeploymentConfigError(Exception):
    """Raised when a deployment's configuration is missing or malformed."""


def deployment_dir(name: str, *, base: Path | None = None) -> Path:
    """Return the directory for the named deployment under ``base``."""
    return (base or DEFAULT_DEPLOYMENTS_DIR) / name


def load_connector_configs(
    deployment: str | Path,
    *,
    base: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[ConnectorConfig]:
    """Load the activated connector configurations for a deployment.

    ``deployment`` may be a deployment name (resolved against ``base``) or a direct
    path to a deployment directory. ``env`` defaults to the process environment and
    is used to expand ``${VAR}`` placeholders in settings values.
    """
    directory = _resolve_dir(deployment, base)
    document = _read_yaml(directory / "deployment.yml")
    environment = os.environ if env is None else env

    raw_connectors = document.get("connectors", [])
    if not isinstance(raw_connectors, list):
        raise DeploymentConfigError(
            "'connectors' in deployment.yml must be a list of connector blocks"
        )

    configs: list[ConnectorConfig] = []
    for index, raw in enumerate(raw_connectors):
        configs.append(_parse_connector(raw, index, environment))
    return configs


def _resolve_dir(deployment: str | Path, base: Path | None) -> Path:
    path = Path(deployment)
    # A bare name (a single path component) is resolved against the deployments
    # base; anything with a separator or an absolute path is used as given, which
    # is what lets tests point at a throwaway directory.
    if path.is_absolute() or len(path.parts) > 1:
        directory = path
    else:
        directory = deployment_dir(str(deployment), base=base)
    if not directory.is_dir():
        raise DeploymentConfigError(f"deployment directory not found: {directory}")
    return directory


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DeploymentConfigError(f"missing configuration file: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise DeploymentConfigError(f"{path} must contain a top-level mapping")
    return loaded


def _parse_connector(raw: Any, index: int, env: Mapping[str, str]) -> ConnectorConfig:
    if not isinstance(raw, dict):
        raise DeploymentConfigError(
            f"connector #{index} must be a mapping, got {type(raw).__name__}"
        )
    key = raw.get("key")
    if not isinstance(key, str) or not key:
        raise DeploymentConfigError(f"connector #{index} is missing a string 'key'")
    # A single-instance connector may omit instance_id; it then defaults to the key.
    instance_id = raw.get("instance_id", key)
    if not isinstance(instance_id, str) or not instance_id:
        raise DeploymentConfigError(f"connector {key!r} has a non-string 'instance_id'")
    settings = raw.get("settings", {}) or {}
    if not isinstance(settings, dict):
        raise DeploymentConfigError(f"connector {key!r} 'settings' must be a mapping")
    return ConnectorConfig(
        key=key,
        instance_id=instance_id,
        settings=_expand(settings, env),
    )


def _expand(value: Any, env: Mapping[str, str]) -> Any:
    """Recursively expand ``${VAR}`` placeholders in strings using ``env``.

    An unset variable expands to the empty string, which is the right behaviour for
    optional credentials (an absent token simply yields anonymous access).
    """
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: env.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(item, env) for item in value]
    return value
