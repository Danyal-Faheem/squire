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
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from squire.connectors.base import ConnectorConfig

#: Default base directory holding deployment directories, relative to the working
#: directory (the repository root when running via the Make targets or the CLI).
DEFAULT_DEPLOYMENTS_DIR = Path("deployments")

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DeploymentConfigError(Exception):
    """Raised when a deployment's configuration is present but malformed."""


class MissingDeploymentConfigError(DeploymentConfigError):
    """Raised when a deployment directory or a config file is absent.

    Distinguished from the base error so callers can treat *absent* configuration
    (a legitimate early-phase state worth falling back on) differently from
    *malformed* configuration (a mistake that should fail loudly).
    """


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
        raise MissingDeploymentConfigError(
            f"deployment directory not found: {directory}"
        )
    return directory


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MissingDeploymentConfigError(f"missing configuration file: {path}")
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


# ---------------------------------------------------------------------------
# Policy: taxonomy, routing, and priorities.
#
# The factory and the classifier are driven by three project-specific config
# files — what categories exist, how a category routes to owners and a target
# repository, and what the priority scheme is. The engine ships only the models
# and loaders; the values themselves live in a deployment. None of the names here
# is project-specific.
# ---------------------------------------------------------------------------


class Category(BaseModel):
    """One category in the taxonomy. The description guides the classifier."""

    name: str
    description: str = ""


class Taxonomy(BaseModel):
    """The set of categories a signal may be sorted into."""

    categories: list[Category]
    #: Category to fall back to when a classifier proposes one outside the set;
    #: defaults to the first category if not given.
    default: str | None = None

    @property
    def names(self) -> list[str]:
        """The category names, in declared order."""
        return [c.name for c in self.categories]

    @property
    def fallback(self) -> str:
        """The category to use when a proposed one is not recognised."""
        if self.default and self.default in self.names:
            return self.default
        return self.names[0] if self.categories else "uncategorized"

    def resolve(self, name: str) -> str:
        """Return ``name`` if it is a known category, else the fallback."""
        return name if name in self.names else self.fallback


class Route(BaseModel):
    """Where a category's work goes: which owners, in which repository."""

    owners: list[str] = Field(default_factory=list)
    repo: str

    @property
    def primary_owner(self) -> str:
        """The first owner, or a generic placeholder when none are listed."""
        return self.owners[0] if self.owners else "unassigned"


class Routing(BaseModel):
    """Maps each category to a :class:`Route`, with a required default."""

    routes: dict[str, Route] = Field(default_factory=dict)
    default: Route

    def resolve(self, category: str) -> Route:
        """Return the route for ``category``, or the default route."""
        return self.routes.get(category, self.default)


class Priority(BaseModel):
    """One priority level: a key stored on the item and a label shown on it."""

    key: str
    label: str
    description: str = ""


class Priorities(BaseModel):
    """The priority scheme plus optional guidance for the classifier."""

    levels: list[Priority]
    #: Priority key to fall back to; defaults to the first level if not given.
    default: str | None = None
    #: Free-text guidance appended to the classify prompt (e.g. "security is p0").
    guidance: str = ""

    @property
    def keys(self) -> list[str]:
        """The priority keys, in declared order."""
        return [level.key for level in self.levels]

    @property
    def fallback(self) -> str:
        """The priority key to use when a proposed one is not recognised."""
        if self.default and self.default in self.keys:
            return self.default
        return self.keys[0] if self.levels else "unprioritized"

    def resolve(self, key: str) -> str:
        """Return ``key`` if it is a known priority, else the fallback."""
        return key if key in self.keys else self.fallback

    def label_for(self, key: str) -> str:
        """Return the board label for a priority key (the key itself if unknown)."""
        for level in self.levels:
            if level.key == key:
                return level.label
        return key


class DeploymentPolicy(BaseModel):
    """The bundle of project-specific policy the factory and classifier consume."""

    taxonomy: Taxonomy
    routing: Routing
    priorities: Priorities


def load_policy(
    deployment: str | Path,
    *,
    base: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> DeploymentPolicy:
    """Load taxonomy, routing, and priorities for a deployment."""
    directory = _resolve_dir(deployment, base)
    environment = os.environ if env is None else env
    taxonomy = _load_model(directory / "taxonomy.yml", Taxonomy, environment)
    routing = _load_model(directory / "routing.yml", Routing, environment)
    priorities = _load_model(directory / "priorities.yml", Priorities, environment)
    return DeploymentPolicy(taxonomy=taxonomy, routing=routing, priorities=priorities)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_model(path: Path, model: type[_ModelT], env: Mapping[str, str]) -> _ModelT:
    document = _read_yaml(path)
    try:
        return model.model_validate(_expand(document, env))
    except ValidationError as exc:
        raise DeploymentConfigError(f"invalid configuration in {path}: {exc}") from exc
