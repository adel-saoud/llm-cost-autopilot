# syntax=docker/dockerfile:1.7

# ── Builder ────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

# uv installs faster than pip and produces hermetic environments.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=python3.14

WORKDIR /app

# Copy lockfile + project metadata first so we can cache the dependency layer.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ml --no-dev


# ── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

# Run as an unprivileged user so a compromised model can't trash the host.
RUN useradd --create-home --uid 1000 autopilot

WORKDIR /app
COPY --from=builder --chown=autopilot:autopilot /app /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER autopilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "llm_cost_autopilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
