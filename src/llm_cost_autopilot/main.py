"""Production ASGI entry point — ``uvicorn llm_cost_autopilot.main:app``."""

from __future__ import annotations

from llm_cost_autopilot.api import create_app

app = create_app()
