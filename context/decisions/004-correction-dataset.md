# ADR-004 — SQLite as the correction-dataset store

**Status:** Accepted (2026-06-10)

## Context

The verifier emits routing failures. We want to:

- Inspect them by hand to debug bad routing decisions.
- Filter / sort them in the Streamlit dashboard.
- Export them as a corpus a future fine-tune could consume.
- Ship the whole thing in a Docker compose file with no extra dependencies.

Options considered:

1. **A vector store** (e.g. ChromaDB, Qdrant) — would let us run nearest-
   neighbour searches over divergent prompts. Heavy for what we need; the
   correction count is in the thousands, not millions.
2. **Postgres** — battle-tested for OLAP. Overkill for one file's worth
   of data; adds an operator burden and breaks "clone the repo, run it".
3. **JSONL files** — trivially portable. Can't do aggregate queries
   (`/v1/stats`) without re-reading the file each time.
4. **SQLite** — single file on disk, indexable, queryable, and committable.
   The dashboard reads it directly without going through the API.

## Decision

SQLite via `aiosqlite`, opened with `PRAGMA journal_mode = WAL` so the
dashboard can read while the API writes. The ledger lives in
`AUTOPILOT_SQLITE_PATH` (default `autopilot.db` in the working dir; the
Dockerfile mounts a `/data` volume).

Two tables — `routing_log` and `corrections` — both denormalised by
intent. The point is that you can `sqlite3 autopilot.db` and read both
tables stand-alone, no joins needed.

## Consequences

- **One-file portability** — copying the DB is enough to share a corpus.
- **No external dependency** — `aiosqlite` is in `dependencies`, not an
  extra. Docker compose has one service (Ollama) plus the app itself.
- **No vector search** — finding semantically similar failures means
  pulling the dataset and embedding client-side. Acceptable: the dashboard
  shows raw rows and the corpus is small.
- **WAL leaves sidecar files on disk** — `autopilot.db-wal` and
  `autopilot.db-shm` next to the main file. Documented in CLAUDE.md.
- **Schema versioning is intentional** — `CURRENT_SCHEMA_VERSION` + a
  `schema_version` table mean a binary refuses to run against a newer
  schema. Future migrations bump the constant and add a branch in
  `migrate()`.
