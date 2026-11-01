"""SQLite-backed ledger and correction dataset.

One connection per process: SQLite is single-writer, and ``aiosqlite`` serialises
inside the proxy thread. The ledger implements the verifier's ``CorrectionStore``
Protocol so it can be plugged in directly.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import structlog
from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ClassifierStage, ComplexityTier
from llm_cost_autopilot.classifier.correction_memory import DEFAULT_TENANT_ID
from llm_cost_autopilot.verifier.feedback import RoutingFailure

logger = structlog.get_logger(__name__)


CURRENT_SCHEMA_VERSION = 3
"""Schema this binary writes. v2 added tenants and the matched-correction id;
v3 folded the removed ``complex`` tier into ``moderate``."""


class RoutingLogEntry(BaseModel):
    """One row of the routing log, frozen so a recorded decision cannot be altered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tenant_id: str = DEFAULT_TENANT_ID
    prompt: str
    classified_tier: ComplexityTier
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    classifier_stage: ClassifierStage
    model_used: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    baseline_cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    matched_correction_request_id: str | None = None
    """Set when a past correction escalated this request (stage ``correction_memory``)."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExplainRecord(BaseModel):
    """Joined view for ``GET /v1/explain/{request_id}``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    routing: RoutingLogEntry
    correction: RoutingFailure | None
    """If present, the verifier disagreed with the routing decision."""


class LedgerStats(BaseModel):
    """Aggregates over the whole ledger, served by ``GET /v1/stats``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_requests: int = Field(ge=0)
    total_actual_cost_usd: float = Field(ge=0.0)
    total_baseline_cost_usd: float = Field(ge=0.0)
    savings_usd: float
    savings_pct: float
    routing_accuracy_pct: float
    correction_count: int = Field(ge=0)
    requests_by_tier: dict[ComplexityTier, int]


_SCHEMA_V1: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS routing_log (
        request_id            TEXT PRIMARY KEY,
        created_at            TEXT    NOT NULL,
        prompt                TEXT    NOT NULL,
        classified_tier       TEXT    NOT NULL,
        classifier_confidence REAL    NOT NULL,
        classifier_stage      TEXT    NOT NULL,
        model_used            TEXT    NOT NULL,
        prompt_tokens         INTEGER NOT NULL,
        completion_tokens     INTEGER NOT NULL,
        cost_usd              REAL    NOT NULL,
        baseline_cost_usd     REAL    NOT NULL,
        latency_ms            REAL    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corrections (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at            TEXT NOT NULL,
        request_id            TEXT NOT NULL,
        prompt                TEXT NOT NULL,
        cheap_model           TEXT NOT NULL,
        cheap_response        TEXT NOT NULL,
        cheap_tier            TEXT NOT NULL,
        classifier_confidence REAL NOT NULL,
        standard_model        TEXT NOT NULL,
        standard_response     TEXT NOT NULL,
        similarity_score      REAL NOT NULL,
        divergence_threshold  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corrections_request_id ON corrections(request_id)",
    "CREATE INDEX IF NOT EXISTS idx_routing_log_created_at ON routing_log(created_at)",
)
"""The original schema. Kept frozen: later changes go in ``_MIGRATIONS`` so old files upgrade."""

_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        "ALTER TABLE routing_log ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'",
        "ALTER TABLE routing_log ADD COLUMN matched_correction_request_id TEXT",
        "ALTER TABLE corrections ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'",
        "CREATE INDEX IF NOT EXISTS idx_corrections_created_at ON corrections(created_at)",
    ),
    (
        "UPDATE routing_log SET classified_tier = 'moderate' WHERE classified_tier = 'complex'",
        "UPDATE corrections SET cheap_tier = 'moderate' WHERE cheap_tier = 'complex'",
    ),
)
"""Ordered schema upgrades: entry ``i`` moves a ledger from version ``i + 1`` to ``i + 2``.

SQLite allows ``ADD COLUMN ... NOT NULL`` only with a non-NULL default; the
``'default'`` literal is ``DEFAULT_TENANT_ID`` and back-fills pre-tenant rows.
v3 rewrites ``complex`` rows because that tier no longer parses; their
stored costs keep the prices they were recorded with.
"""
if len(_MIGRATIONS) != CURRENT_SCHEMA_VERSION - 1:  # pragma: no cover
    raise RuntimeError("_MIGRATIONS must hold one entry per schema version above 1")


class SQLiteLedger:
    """Async SQLite ledger: routing log, corrections, and aggregate stats.

    Construct via the ``open_ledger`` async context manager so the schema is
    migrated and the connection closed. Tests pass ``":memory:"`` as the path.
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def schema_version(self) -> int:
        """Return the schema version recorded on disk.

        Exposed publicly so tests and ops can verify the migration ran
        without poking at the private connection.
        """
        async with self._connection.execute("SELECT version FROM schema_version LIMIT 1") as cur:
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError("schema_version table is empty: migrations did not run")
        return int(row["version"])

    async def migrate(self) -> None:
        """Create the v1 schema, then apply every newer migration in order.

        Idempotent: a database already at ``CURRENT_SCHEMA_VERSION`` is left
        untouched. A fresh database is created at v1 and upgraded like an old
        one, so both paths run the same statements.
        """
        for statement in _SCHEMA_V1:
            await self._connection.execute(statement)
        async with self._connection.execute("SELECT version FROM schema_version LIMIT 1") as cur:
            row = await cur.fetchone()
        if row is None:
            await self._connection.execute("INSERT INTO schema_version (version) VALUES (1)")
            version = 1
        else:
            version = int(row["version"])
        if version > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Ledger schema version {version} is newer than this "
                f"binary supports ({CURRENT_SCHEMA_VERSION}). Refusing to start."
            )
        for statements in _MIGRATIONS[version - 1 :]:
            for statement in statements:
                await self._connection.execute(statement)
            version += 1
            await self._connection.execute("UPDATE schema_version SET version = ?", (version,))
            logger.info("ledger.migrated", version=version)
        await self._connection.commit()

    async def record_routing(self, entry: RoutingLogEntry) -> None:
        """Insert one routing-log row. Idempotent on ``request_id`` collisions."""
        await self._connection.execute(
            """
            INSERT OR REPLACE INTO routing_log (
                request_id, tenant_id, created_at, prompt,
                classified_tier, classifier_confidence, classifier_stage,
                model_used, prompt_tokens, completion_tokens,
                cost_usd, baseline_cost_usd, latency_ms,
                matched_correction_request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.request_id,
                entry.tenant_id,
                entry.created_at.isoformat(),
                entry.prompt,
                entry.classified_tier.value,
                entry.classifier_confidence,
                entry.classifier_stage.value,
                entry.model_used,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.cost_usd,
                entry.baseline_cost_usd,
                entry.latency_ms,
                entry.matched_correction_request_id,
            ),
        )
        await self._connection.commit()

    async def record_failure(self, failure: RoutingFailure) -> None:
        """Implements ``CorrectionStore``: inserts one row into ``corrections``."""
        await self._connection.execute(
            """
            INSERT INTO corrections (
                created_at, request_id, tenant_id, prompt,
                cheap_model, cheap_response, cheap_tier, classifier_confidence,
                standard_model, standard_response,
                similarity_score, divergence_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failure.created_at.isoformat(),
                failure.request_id,
                failure.tenant_id,
                failure.prompt,
                failure.cheap_model,
                failure.cheap_response,
                failure.cheap_tier.value,
                failure.classifier_confidence,
                failure.standard_model,
                failure.standard_response,
                failure.similarity_score,
                failure.divergence_threshold,
            ),
        )
        await self._connection.commit()

    async def prune_older_than(self, cutoff: datetime) -> tuple[int, int]:
        """Delete routing-log and correction rows created before ``cutoff``.

        Both tables hold raw prompts, so this is the retention control.
        Returns ``(routing_rows, correction_rows)`` deleted.
        """
        cutoff_iso = cutoff.isoformat()
        async with self._connection.execute(
            "DELETE FROM routing_log WHERE created_at < ?", (cutoff_iso,)
        ) as cur:
            routing_rows = cur.rowcount
        async with self._connection.execute(
            "DELETE FROM corrections WHERE created_at < ?", (cutoff_iso,)
        ) as cur:
            correction_rows = cur.rowcount
        await self._connection.commit()
        return routing_rows, correction_rows

    async def get_explain(self, request_id: str) -> ExplainRecord | None:
        """Return the routing row and its latest correction, or None for an unknown id."""
        async with self._connection.execute(
            "SELECT * FROM routing_log WHERE request_id = ?",
            (request_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        routing = _row_to_routing(row)

        async with self._connection.execute(
            "SELECT * FROM corrections WHERE request_id = ? ORDER BY id DESC LIMIT 1",
            (request_id,),
        ) as cur:
            correction_row = await cur.fetchone()
        correction = _row_to_failure(correction_row) if correction_row is not None else None
        return ExplainRecord(routing=routing, correction=correction)

    async def list_failures(self, *, limit: int = 100) -> list[RoutingFailure]:
        """Return the ``limit`` most recent corrections, newest first."""
        if limit <= 0:
            raise ValueError("limit must be >= 1")
        async with self._connection.execute(
            "SELECT * FROM corrections ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_failure(r) for r in rows]

    async def list_failures_oldest_first(self) -> list[RoutingFailure]:
        """Every stored correction, oldest first, for rebuilding the correction memory.

        Unbounded on purpose: retention pruning bounds the table, and the
        memory applies its own per-tenant cap while loading.
        """
        async with self._connection.execute("SELECT * FROM corrections ORDER BY id ASC") as cur:
            rows = await cur.fetchall()
        return [_row_to_failure(r) for r in rows]

    async def get_stats(self, standard_ollama_model: str) -> LedgerStats:
        """Aggregate the ledger; ``standard_ollama_model`` defines what is verifiable.

        Routing accuracy is computed only over requests routed to a model
        other than the standard one: those are the only requests the verifier
        could have flagged. A request served by the standard model is counted
        as correct because it already got the reference answer.
        """
        async with self._connection.execute(
            """
            SELECT
                COUNT(*)                                  AS total_requests,
                COALESCE(SUM(cost_usd), 0.0)              AS actual,
                COALESCE(SUM(baseline_cost_usd), 0.0)     AS baseline,
                COALESCE(
                    SUM(CASE WHEN model_used != ? THEN 1 ELSE 0 END),
                    0
                )                                         AS verifiable
            FROM routing_log
            """,
            (standard_ollama_model,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("aggregate query returned no row")

        total = int(row["total_requests"])
        actual = float(row["actual"])
        baseline = float(row["baseline"])
        verifiable = int(row["verifiable"])

        async with self._connection.execute("SELECT COUNT(*) AS n FROM corrections") as cur:
            correction_count_row = await cur.fetchone()
        if correction_count_row is None:  # pragma: no cover
            raise RuntimeError("COUNT query returned no row")
        correction_count = int(correction_count_row["n"])

        async with self._connection.execute(
            "SELECT classified_tier, COUNT(*) AS n FROM routing_log GROUP BY classified_tier"
        ) as cur:
            tier_rows = await cur.fetchall()
        requests_by_tier: dict[ComplexityTier, int] = dict.fromkeys(ComplexityTier, 0)
        for tier_row in tier_rows:
            requests_by_tier[ComplexityTier(tier_row["classified_tier"])] = int(tier_row["n"])

        savings_usd = baseline - actual
        savings_pct = (savings_usd / baseline * 100.0) if baseline > 0 else 0.0
        # Accuracy: of the requests the verifier could have flagged, what
        # fraction were not flagged? Default to 100% when there's nothing to
        # judge, which avoids a divide-by-zero scare in an empty dashboard.
        accuracy_pct = (1.0 - correction_count / verifiable) * 100.0 if verifiable > 0 else 100.0

        return LedgerStats(
            total_requests=total,
            total_actual_cost_usd=actual,
            total_baseline_cost_usd=baseline,
            savings_usd=savings_usd,
            savings_pct=savings_pct,
            routing_accuracy_pct=max(0.0, min(100.0, accuracy_pct)),
            correction_count=correction_count,
            requests_by_tier=requests_by_tier,
        )


@asynccontextmanager
async def open_ledger(path: str | Path) -> AsyncGenerator[SQLiteLedger, None]:
    """Open a ledger connection, apply migrations, and yield it.

    Used as ``async with open_ledger(path) as ledger: ...``. The connection
    is closed when the context exits, including on exception.
    """
    async with aiosqlite.connect(path) as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        # WAL lets the Streamlit dashboard read while the API writes.
        await connection.execute("PRAGMA journal_mode = WAL")
        ledger = SQLiteLedger(connection)
        await ledger.migrate()
        yield ledger


def _row_to_routing(row: aiosqlite.Row) -> RoutingLogEntry:
    """Rebuild a ``RoutingLogEntry`` from a ``routing_log`` row."""
    return RoutingLogEntry(
        request_id=row["request_id"],
        tenant_id=row["tenant_id"],
        prompt=row["prompt"],
        classified_tier=ComplexityTier(row["classified_tier"]),
        classifier_confidence=float(row["classifier_confidence"]),
        classifier_stage=ClassifierStage(row["classifier_stage"]),
        model_used=row["model_used"],
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        cost_usd=float(row["cost_usd"]),
        baseline_cost_usd=float(row["baseline_cost_usd"]),
        latency_ms=float(row["latency_ms"]),
        matched_correction_request_id=row["matched_correction_request_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_failure(row: aiosqlite.Row) -> RoutingFailure:
    """Rebuild a ``RoutingFailure`` from a ``corrections`` row."""
    return RoutingFailure(
        request_id=row["request_id"],
        tenant_id=row["tenant_id"],
        prompt=row["prompt"],
        cheap_model=row["cheap_model"],
        cheap_response=row["cheap_response"],
        cheap_tier=ComplexityTier(row["cheap_tier"]),
        classifier_confidence=float(row["classifier_confidence"]),
        standard_model=row["standard_model"],
        standard_response=row["standard_response"],
        similarity_score=float(row["similarity_score"]),
        divergence_threshold=float(row["divergence_threshold"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ExplainRecord",
    "LedgerStats",
    "RoutingLogEntry",
    "SQLiteLedger",
    "open_ledger",
]
