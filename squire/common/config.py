# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed application configuration, loaded from environment variables.

All runtime configuration for Squire flows through :class:`Settings`. Nothing
elsewhere in the codebase should read ``os.environ`` directly; import and call
:func:`get_settings` instead, so configuration is validated in one place and is
trivial to override in tests.

This module holds only environment-tier configuration (secrets and infrastructure
that are identical across deployments). Project-specific configuration — which
connectors to run, the category taxonomy, routing, priorities, prompt overrides —
lives in a deployment directory and is loaded separately.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings sourced from the environment / a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM / inference (any OpenAI-compatible endpoint) ----------------------
    # These point at whatever serves the model: a self-hosted server or a cloud
    # provider. The code only assumes the OpenAI-compatible protocol, never a
    # specific vendor.
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_api_key: str = Field(default="not-needed")
    generation_model: str = Field(default="Qwen/Qwen2.5-Coder-32B-Instruct")
    # Embeddings may come from a different endpoint/provider than generation, so
    # they carry their own base URL and key.
    embedding_base_url: str = Field(default="http://localhost:8001/v1")
    embedding_api_key: str = Field(default="not-needed")
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)

    # --- Database --------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://squire:squire@localhost:5432/squire"
    )

    # --- Deployment selection --------------------------------------------------
    # Which deployment directory under deployments/ to load project-specific
    # configuration from.
    squire_deployment: str = Field(default="example")

    # --- Board credentials (default board backend: GitHub) ---------------------
    github_app_id: str = Field(default="")
    github_private_key: str = Field(default="")
    github_webhook_secret: str = Field(default="")

    # --- Messaging credentials (default adapter: Slack) ------------------------
    slack_bot_token: str = Field(default="")
    slack_app_token: str = Field(default="")
    messaging_digest_channel: str = Field(default="")
    messaging_digest_cron: str = Field(default="0 9 * * MON")

    # --- Commands / agent ------------------------------------------------------
    agent_pr_keyword: str = Field(default="open-pr")
    bot_mention_handle: str = Field(default="@squire")

    # --- Feature flags (all default off / safe) --------------------------------
    autopr_enabled: bool = Field(default=False)
    source_write_enabled: bool = Field(default=False)
    dry_run: bool = Field(default=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Caching keeps configuration effectively singleton at runtime. Tests that need
    to override values can call ``get_settings.cache_clear()`` after patching the
    environment.
    """
    return Settings()
