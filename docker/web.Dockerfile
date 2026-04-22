# Public-facing Streamlit web service.
# Reads CSVs produced by the parser container from a shared named volume.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: curl for the healthcheck only.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy only what the web service needs. Parser-specific code stays out.
COPY app/ ./app/
COPY src/ ./src/
COPY run.py ./

# Cache dir is mounted from a named volume at runtime; create mount points so
# Streamlit doesn't fail on first boot if the volume hasn't been populated.
RUN mkdir -p /app/data/processed /app/data/cache

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
