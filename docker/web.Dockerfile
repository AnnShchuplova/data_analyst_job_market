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
# Bake in the pre-trained models so the image is self-contained.
# The models/ dir is also mounted as a named volume so retraining via the
# web UI persists; Docker initialises an empty volume from the image layer.
COPY models/ ./models/

# Create mount points for runtime volumes.
RUN mkdir -p /app/finaldata /app/data/cache /app/models

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
