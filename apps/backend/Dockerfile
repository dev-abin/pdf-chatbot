# Use uv's official Python base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app

# Copy project metadata first for dependency caching
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
RUN uv sync --frozen --no-dev

# Copy source code and configs
COPY src ./src
COPY configs ./configs

# Final runtime image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

# Copy virtual environment + source from builder layer
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
