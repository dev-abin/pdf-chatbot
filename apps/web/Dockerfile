FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ui.py ./ui.py
COPY core ./core
COPY rag_ui ./rag_ui
RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

