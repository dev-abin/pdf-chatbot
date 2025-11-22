FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app

# Copy metadata
COPY pyproject.toml uv.lock ./

# Install deps
RUN uv sync --frozen --no-dev

# Copy Streamlit code
COPY ui.py ./ui.py

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy env + app
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/ui.py ./ui.py

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
