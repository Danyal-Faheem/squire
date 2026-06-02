<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Squire

A configurable platform that unifies and AI-assists the maintainer workflow of an
open-source project. Squire ingests the things maintainers have to keep up with —
incoming issues, pull requests, and community-forum questions — and turns each new
item into a single, pre-digested task on a board you control: a categorized title, a
summary, relevant background pulled from a knowledge base of your project's own code
and docs, a suggested answer, an assigned owner, and a priority. Maintainers then
drive it conversationally from the board or from a chat platform.

The name fits the job. A squire attends a knight — carrying the tools, keeping the
gear in order, handling preparation so the knight acts decisively at the right
moment. Squire attends a *maintainer* the same way: it never does the maintaining, it
prepares the ground so a maintainer's judgement is spent only where it must be.

Squire is **project-agnostic**. The engine knows nothing about any particular
project; you point it at your sources, describe your categories and routing in a few
config files, and it becomes a tool tailored to your project. Adopting it for a new
project is a matter of configuration, not code.

## Features

- **One unified queue.** New issues, pull requests, and tagged forum questions become
  enriched work items on a single board, so maintainers stop hunting across systems.
- **Pluggable sources (connectors).** Ingest from GitHub repositories (code, docs,
  issues, PRs), Discourse forums, documentation websites, PDFs, and markdown/rst
  files out of the box. New source types are added as self-contained connectors.
- **A knowledge base over your project.** Everything ingested feeds a retrieval layer
  so the assistant answers and summarizes with context grounded in your own material.
- **Suggested answers and context.** Each work item includes a draft response and the
  background a maintainer needs, so they don't have to look everything up by hand.
- **Conversational control.** Mention the bot on a work item or in chat to ask
  questions, request more context, or trigger a draft pull request.
- **Scheduled digest.** A regular chat summary of open work items, grouped and tagged
  by responsible owner.

## How it works

Two layers. The **engine** (the `squire/` package) is the generic core — connectors,
the knowledge base, the work-item factory, the command interface, and the board and
messaging adapters — and contains nothing specific to any project. A **deployment**
is a small bundle of configuration that instantiates the engine for one project:
which connectors to run and with what settings, your category taxonomy, how
categories route to owners and repositories, your priority scheme, and optional
prompt overrides. The repository ships one reference deployment under `deployments/`
as a worked example.

```
your sources ─┐
(repos, forum, ├→ connectors → knowledge base (chunks + embeddings)
 docs, PDFs)  ─┘                work-item factory → your board
                                                      ↑ commands ↓
                                          maintainers (board / chat)
```

Squire is built around a small set of deliberate extension points (seams): the source
connectors and the prompt templates are built out; the LLM and embedding endpoints
swap by configuration; and the vector store, the board backend, the classifier, and
the messaging surface each sit behind an interface with a single default
implementation, so a new backend is a new class against an existing contract. A
single PostgreSQL database (with pgvector) holds both the relational state and the
embeddings; PostgreSQL itself is a fixed choice. Every language-model call targets an
**OpenAI-compatible endpoint**, so the inference backend — self-hosted or cloud — is a
configuration choice, not a code dependency.

## Quick start

Prerequisites: Python 3.11+, Docker, and a reachable OpenAI-compatible endpoint for
the LLM and embedding models.

```bash
cp .env.example .env     # fill in endpoints, tokens, and your board credentials
make install             # install with dev dependencies
make up                  # start Postgres (pgvector) via Docker Compose
make migrate             # apply database migrations
make ci                  # run lint, type checks, and tests
```

## Configuring it for your project

Infrastructure and secrets live in `.env` (model endpoints, database, credentials,
feature flags); every variable is documented inline there. Everything specific to
your project lives in a deployment directory under `deployments/<name>/`:

- `deployment.yml` — which connectors to activate and their settings, plus which
  board and messaging adapter to use.
- `taxonomy.yml` — the categories your work items are sorted into.
- `routing.yml` — how each category maps to owner(s) and a target repository.
- `priorities.yml` — your priority scheme and the guidance used to classify items.
- `prompts/` — optional overrides of the default prompt and item-format templates.

Copy the reference deployment, edit these files for your project, and select it with
`SQUIRE_DEPLOYMENT=<name>`. No engine code changes are needed.

### Adding a new source type

Sources are connectors registered under a string key. To support a source the engine
doesn't ship yet, implement the connector contract in one new class and register it;
nothing else in the system needs to change.

## Development

| Target            | Description                                   |
| ----------------- | --------------------------------------------- |
| `make format`     | Format and auto-fix lint issues               |
| `make test`       | Run lint, type checks, and unit tests         |
| `make ci`         | What CI runs on every push and pull request   |

Code is fully type-annotated and checked with `mypy`; formatting and linting use
`ruff`. Tests are treated as the contract: code is written to pass them.

## Safety posture

Squire is human-in-the-loop by default — the squire prepares, the maintainer decides.
Creating a work item, assigning an owner, and setting a priority are safe automated
actions on your own board. Anything that writes to an external source — posting to a
forum, opening a pull request — happens only on explicit instruction and sits behind
feature flags that default to off (`AUTOPR_ENABLED`, `SOURCE_WRITE_ENABLED`, and a
global `DRY_RUN` that defaults on in development). Document ingestion and signal
reading are always read-only.

## License

Licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See [LICENSE](./LICENSE).
