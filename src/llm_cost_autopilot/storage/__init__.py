"""SQLite ledger: routing log and correction dataset.

Public surface:
    SQLiteLedger            async connection, writes, reads and aggregate stats.
    open_ledger             context manager that connects and migrates.
    RoutingLogEntry         frozen pydantic record of one routed request.
    ExplainRecord           joined view returned by ``/v1/explain/{id}``.
    LedgerStats             aggregate stats for ``/v1/stats``.
    CURRENT_SCHEMA_VERSION  schema version this build writes.
"""

from llm_cost_autopilot.storage.db import (
    CURRENT_SCHEMA_VERSION,
    ExplainRecord,
    LedgerStats,
    RoutingLogEntry,
    SQLiteLedger,
    open_ledger,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ExplainRecord",
    "LedgerStats",
    "RoutingLogEntry",
    "SQLiteLedger",
    "open_ledger",
]
