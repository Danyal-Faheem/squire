# SPDX-License-Identifier: AGPL-3.0-or-later
# A single image that can run any Squire service; the compose `command` selects
# which one. Kept minimal for the scaffolding.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY . .
RUN pip install -e ".[dev]"
