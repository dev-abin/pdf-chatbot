FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY ./src ./src
COPY ./config ./config

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (entrypoint now inside src)
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
