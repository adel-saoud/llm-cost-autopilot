"""Write the JSON Schema of the router API contract to docs/.

llm-gateway runs a contract test against this file, so it must match the
pydantic models. A test fails when it is stale; regenerate with:
    uv run python scripts/export_route_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_cost_autopilot.api.route import ROUTE_API_VERSION, ROUTE_PATH
from llm_cost_autopilot.api.schemas import RouteRequest, RouteResponse

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "route-api.schema.json"


def route_api_schema() -> dict[str, Any]:
    """Request and response schemas of the router API at ``ROUTE_API_VERSION``."""
    return {
        "api_version": ROUTE_API_VERSION,
        "endpoint": f"POST {ROUTE_PATH}",
        "request": RouteRequest.model_json_schema(),
        "response": RouteResponse.model_json_schema(),
    }


def main() -> None:
    """Overwrite the published schema file with the current models' schema."""
    SCHEMA_PATH.write_text(json.dumps(route_api_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
