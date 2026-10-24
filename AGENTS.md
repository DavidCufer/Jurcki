# Working agreement for AI agents

This repo is built by an orchestrator agent that delegates the coding to
worker agents.

## Roles

* **Orchestrator (Opus):** plans the work, splits it into self-contained
  tasks and writes the shared contracts (`docs/ARCHITECTURE.md`,
  `backend/jurcki/config.py`). It reviews, tests and validates what the
  workers produce, and it is the only role that commits and pushes. It writes
  code itself only for small glue or for model logic where judgement matters
  most (e.g. `model/prior.py`, `model/evaluate.py`).
* **Workers (Sonnet 5.5, `model: "sonnet"`):** do the actual coding tasks, one
  area each, in parallel when the areas don't overlap. Workers never commit or
  push. Each one ends with a report: files changed, what was run, the
  validation numbers, blocked hosts and open doubts.

## Rules for worker tasks

* Every task prompt is self-contained: it names the repo path and the venv
  (`source backend/.venv/bin/activate`), the files to read first, the network
  limits, the exact outputs and the sanity checks to report.
* Tests live in `backend/tests/` and run without network (`pytest -q`).
* Large data goes under `data/raw`, `data/cache` or `data/interim`, which are
  git-ignored. Small derived data goes under `data/processed`, and
  app-facing output under `data/output`.
* Never route around the network policy: no mirrors, no proxies, no
  alternative hosts for blocked downloads. Report the blocked host instead.
* Don't invent data to fill a gap. If an input is missing, the code should
  fail clearly or skip, and the report should say so.
* Before committing, the orchestrator checks: the tests pass, the outputs
  are spot-checked against known geography or climate, and the diff has been
  read.
