# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for squire.common.config."""

from __future__ import annotations

from squire.common.config import Settings


def test_feature_flags_default_off() -> None:
    """The safety-critical flags must default to the safe (off / dry) state."""
    settings = Settings()
    assert settings.autopr_enabled is False
    assert settings.source_write_enabled is False
    assert settings.dry_run is True


def test_defaults_are_sane() -> None:
    """A few defaults the rest of the system relies on."""
    settings = Settings()
    assert settings.agent_pr_keyword == "open-pr"
    assert settings.embedding_dim == 1024
    assert settings.squire_deployment == "example"
    assert settings.bot_mention_handle == "@squire"


def test_overrides_apply() -> None:
    """Explicit values override the defaults."""
    settings = Settings(generation_model="custom-model", embedding_dim=4)
    assert settings.generation_model == "custom-model"
    assert settings.embedding_dim == 4
