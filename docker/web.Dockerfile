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
# catboost's PyPI metadata incorrectly declares numpy<2.0, but catboost 1.2.5+
# works fine with numpy 2.x. Install catboost first without its dependency
# constraints, then install the rest normally.
RUN pip install --no-cache-dir catboost==1.2.10 --no-deps \
 && pip install --no-cache-dir -r requirements-web.txt

# Copy only what the web service needs. Parser-specific code stays out.
COPY app/ ./app/
COPY src/ ./src/
# Create mount points for runtime volumes.
# Model binaries (.pkl, .cbm, .joblib) are NOT baked into the image —
# they come from the bind-mounted ./models directory at runtime.
# Only the metrics JSONs are copied for reference.
RUN mkdir -p /app/finaldata /app/data/cache /app/models
COPY models/*.json ./models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
